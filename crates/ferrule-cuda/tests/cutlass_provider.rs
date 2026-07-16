#![cfg(feature = "cutlass")]

use cuda_core::{CudaContext, DeviceBuffer};
use ferrule_cuda::cutlass::{self, CutlassKernelId};
use ferrule_cuda::provider::{
    KernelOperation, KernelProviderId, LayerKernelRequirements, LinearBundleRequirement,
    WeightLayout,
};

#[test]
fn sm121_plan_selects_only_published_semantic_bundles() {
    let mut layer = LayerKernelRequirements::default();
    layer.add_linear_bundle(LinearBundleRequirement::new(
        KernelOperation::MlaQueryAKv,
        4096,
        [1024, 1024],
        WeightLayout::Fp8E4m3BlockScaled,
    ));
    layer.add_linear_bundle(LinearBundleRequirement::new(
        KernelOperation::MainCompressorProjection,
        4096,
        [1024, 1024],
        WeightLayout::Bf16RowMajor,
    ));
    layer.require_operation(KernelOperation::AttentionHcPre);
    layer.require_operation(KernelOperation::FeedForwardHcPre);
    layer.require_operation(KernelOperation::MlaOutput);
    layer.require_operation(KernelOperation::SharedFfn);
    layer.require_operation(KernelOperation::RoutedFp4Moe);
    let plan = ferrule_cuda::compile_cuda_model_plan(&[layer]).expect("compile SM121 plan");
    let launch = plan.layers[0]
        .operation(KernelOperation::MlaQueryAKv)
        .expect("SM121 QueryA+KV launch");
    assert_eq!(launch.kernel.provider, KernelProviderId::CutlassCubin);
    assert_eq!(launch.kernel.variant, 0);
    assert!(launch.is_capture_safe());

    let compressor = plan.layers[0]
        .operation(KernelOperation::MainCompressorProjection)
        .expect("SM121 BF16 compressor launch");
    assert_eq!(compressor.kernel.provider, KernelProviderId::CutlassCubin);
    assert_eq!(compressor.kernel.variant, 0);
    for operation in [
        KernelOperation::AttentionHcPre,
        KernelOperation::FeedForwardHcPre,
        KernelOperation::MlaOutput,
        KernelOperation::SharedFfn,
        KernelOperation::RoutedFp4Moe,
    ] {
        let launch = plan.layers[0]
            .operation(operation)
            .expect("required semantic launch");
        assert_eq!(launch.kernel.provider, KernelProviderId::CutlassCubin);
        assert_eq!(launch.kernel.operation, operation);
        assert_eq!(launch.kernel.variant, 0);
        assert!(launch.is_provider_managed());
    }
}

#[test]
fn sm121_plan_rejects_missing_required_bundle() {
    let mut layer = LayerKernelRequirements::default();
    layer.require_operation(KernelOperation::SparseAttention);
    let error = ferrule_cuda::compile_cuda_model_plan(&[layer])
        .expect_err("unbound operation must fail closed");
    assert!(
        error
            .to_string()
            .contains("no SM121 semantic provider binding")
    );
}

#[test]
fn sm121_manifest_contains_no_compatibility_provider() {
    let manifest = cutlass::discover_provider()
        .expect("SM121 provider")
        .manifest();
    assert_eq!(manifest.abi_version, cutlass::CUTLASS_ABI_VERSION);
    assert_eq!(manifest.cutlass_version, 461);
    assert_eq!(manifest.target_sm, 121);
    assert_eq!(manifest.kernel_count, 6);
    assert!(manifest.supports(CutlassKernelId::Fp8QueryAKvSm121));
    assert!(manifest.supports(CutlassKernelId::Bf16CompressorSm121));
    assert!(manifest.supports(CutlassKernelId::HcProducerSm121));
    assert!(manifest.supports(CutlassKernelId::SharedFfnSm121));
    assert!(manifest.supports(CutlassKernelId::StableFrameFp4MoeSm121));
    assert!(manifest.supports(CutlassKernelId::MlaOutputSm121));
}

#[test]
fn sm121_bf16_compressor_is_one_launch_and_numerically_exact() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const ROWS: usize = 2;
    const K: usize = 16;
    const N1: usize = 16;
    const N2: usize = 32;
    let activation =
        DeviceBuffer::from_host(&stream, &vec![1.0f32; ROWS * K]).expect("upload BF16 activation");
    let one_bf16 = 0x3f80u16.to_ne_bytes();
    let projection1_weight = DeviceBuffer::from_host(&stream, &one_bf16.repeat(N1 * K))
        .expect("upload projection1 weight");
    let projection2_weight = DeviceBuffer::from_host(&stream, &one_bf16.repeat(N2 * K))
        .expect("upload projection2 weight");
    let mut projection1_output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * N1).expect("projection1 output");
    let mut projection2_output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * N2).expect("projection2 output");

    cutlass::bf16_compressor(
        &stream,
        &activation,
        &projection1_weight,
        &projection2_weight,
        &mut projection1_output,
        &mut projection2_output,
        ROWS,
        N1,
        N2,
        K,
    )
    .expect("SM121 BF16 compressor launch");

    for value in projection1_output
        .to_host_vec(&stream)
        .expect("download projection1 output")
        .into_iter()
        .chain(
            projection2_output
                .to_host_vec(&stream)
                .expect("download projection2 output"),
        )
    {
        assert_eq!(value, 16.0);
    }
}

#[test]
fn sm121_fp8_query_a_kv_is_one_launch_and_numerically_exact() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const ROWS: usize = 2;
    const K: usize = 128;
    const N1: usize = 16;
    const N2: usize = 32;
    // E4M3 1.0 and UE8M0 1.0. Every dot product is exactly 128.
    let activation =
        DeviceBuffer::from_host(&stream, &vec![0x38u8; ROWS * K]).expect("upload FP8 activation");
    let activation_scales =
        DeviceBuffer::from_host(&stream, &vec![127u8; ROWS]).expect("upload activation scales");
    let query_a_weight =
        DeviceBuffer::from_host(&stream, &vec![0x38u8; N1 * K]).expect("upload QueryA weight");
    let query_a_scales = DeviceBuffer::from_host(&stream, &[127u8]).expect("upload QueryA scales");
    let kv_weight =
        DeviceBuffer::from_host(&stream, &vec![0x38u8; N2 * K]).expect("upload KV weight");
    let kv_scales = DeviceBuffer::from_host(&stream, &[127u8]).expect("upload KV scales");
    let mut query_a_output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * N1).expect("QueryA output");
    let mut kv_output = DeviceBuffer::<f32>::zeroed(&stream, ROWS * N2).expect("KV output");

    cutlass::fp8_query_a_kv(
        &stream,
        &activation,
        &activation_scales,
        &query_a_weight,
        &query_a_scales,
        &kv_weight,
        &kv_scales,
        &mut query_a_output,
        &mut kv_output,
        ROWS,
        N1,
        N2,
        K,
    )
    .expect("SM121 QueryA+KV launch");

    for value in query_a_output
        .to_host_vec(&stream)
        .expect("download QueryA output")
        .into_iter()
        .chain(kv_output.to_host_vec(&stream).expect("download KV output"))
    {
        assert_eq!(value, 128.0);
    }
}

#[test]
fn sm121_linear_semantic_entry_supports_grid_derived_row_range() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const ROWS: usize = 4097;
    const FP8_K: usize = 128;
    const BF16_K: usize = 16;
    const N1: usize = 16;
    const N2: usize = 32;

    let fp8_activation = DeviceBuffer::from_host(&stream, &vec![0x38u8; ROWS * FP8_K])
        .expect("upload FP8 prefill activation");
    let fp8_activation_scales = DeviceBuffer::from_host(&stream, &vec![127u8; ROWS])
        .expect("upload FP8 prefill activation scales");
    let query_a_weight = DeviceBuffer::from_host(&stream, &vec![0x38u8; N1 * FP8_K])
        .expect("upload QueryA prefill weight");
    let query_a_scales =
        DeviceBuffer::from_host(&stream, &[127u8]).expect("upload QueryA prefill scales");
    let kv_weight = DeviceBuffer::from_host(&stream, &vec![0x38u8; N2 * FP8_K])
        .expect("upload KV prefill weight");
    let kv_scales = DeviceBuffer::from_host(&stream, &[127u8]).expect("upload KV prefill scales");
    let mut query_a_output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * N1).expect("QueryA prefill output");
    let mut kv_output = DeviceBuffer::<f32>::zeroed(&stream, ROWS * N2).expect("KV prefill output");

    cutlass::fp8_query_a_kv(
        &stream,
        &fp8_activation,
        &fp8_activation_scales,
        &query_a_weight,
        &query_a_scales,
        &kv_weight,
        &kv_scales,
        &mut query_a_output,
        &mut kv_output,
        ROWS,
        N1,
        N2,
        FP8_K,
    )
    .expect("SM121 tiled FP8 QueryA+KV launch");
    for value in query_a_output
        .to_host_vec(&stream)
        .expect("download QueryA prefill output")
        .into_iter()
        .chain(
            kv_output
                .to_host_vec(&stream)
                .expect("download KV prefill output"),
        )
    {
        assert_eq!(value, 128.0);
    }

    let bf16_activation = DeviceBuffer::from_host(&stream, &vec![1.0f32; ROWS * BF16_K])
        .expect("upload BF16 prefill activation");
    let one_bf16 = 0x3f80u16.to_ne_bytes();
    let projection1_weight = DeviceBuffer::from_host(&stream, &one_bf16.repeat(N1 * BF16_K))
        .expect("upload BF16 prefill projection1 weight");
    let projection2_weight = DeviceBuffer::from_host(&stream, &one_bf16.repeat(N2 * BF16_K))
        .expect("upload BF16 prefill projection2 weight");
    let mut projection1_output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * N1).expect("BF16 prefill output1");
    let mut projection2_output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * N2).expect("BF16 prefill output2");

    cutlass::bf16_compressor(
        &stream,
        &bf16_activation,
        &projection1_weight,
        &projection2_weight,
        &mut projection1_output,
        &mut projection2_output,
        ROWS,
        N1,
        N2,
        BF16_K,
    )
    .expect("SM121 tiled BF16 compressor launch");
    for value in projection1_output
        .to_host_vec(&stream)
        .expect("download BF16 prefill output1")
        .into_iter()
        .chain(
            projection2_output
                .to_host_vec(&stream)
                .expect("download BF16 prefill output2"),
        )
    {
        assert_eq!(value, 16.0);
    }
}

#[test]
fn sm121_hc_producer_accepts_dynamic_m() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const HC: usize = 4;
    const HIDDEN: usize = 4096;
    const MIX: usize = 24;
    let function = DeviceBuffer::from_host(&stream, &vec![0.0f32; HC * HIDDEN * MIX])
        .expect("upload HC function");
    let hc_scale = DeviceBuffer::from_host(&stream, &[1.0f32; 3]).expect("upload HC scales");
    let hc_base = DeviceBuffer::from_host(&stream, &[0.0f32; MIX]).expect("upload HC base");
    let rms_weight =
        DeviceBuffer::from_host(&stream, &vec![1.0f32; HIDDEN]).expect("upload RMS weight");

    for rows in [1usize, 2, 4, 8, 17] {
        let state = DeviceBuffer::from_host(&stream, &vec![1.0f32; rows * HC * HIDDEN])
            .expect("upload HC state");
        let mut hidden = DeviceBuffer::<f32>::zeroed(&stream, rows * HIDDEN).expect("HC hidden");
        let mut normalized =
            DeviceBuffer::<f32>::zeroed(&stream, rows * HIDDEN).expect("HC normalized");
        let mut packed = DeviceBuffer::<u8>::zeroed(&stream, rows * HIDDEN).expect("HC packed");
        let mut scales =
            DeviceBuffer::<u8>::zeroed(&stream, rows * (HIDDEN / 128)).expect("HC FP8 scales");
        let mut pre = DeviceBuffer::<f32>::zeroed(&stream, rows * HC).expect("HC pre");
        let mut post = DeviceBuffer::<f32>::zeroed(&stream, rows * HC).expect("HC post");
        let mut comb = DeviceBuffer::<f32>::zeroed(&stream, rows * HC * HC).expect("HC comb");

        cutlass::hc_producer(
            &stream,
            &state,
            &function,
            &hc_scale,
            &hc_base,
            &rms_weight,
            &mut hidden,
            &mut normalized,
            &mut packed,
            &mut scales,
            &mut pre,
            &mut post,
            &mut comb,
            rows,
            HC,
            HIDDEN,
            1,
            1.0e-5,
            1.0e-5,
            1.0e-5,
        )
        .expect("dynamic-M HC producer");

        let normalized = normalized
            .to_host_vec(&stream)
            .expect("download normalized");
        assert!(normalized.iter().all(|value| value.is_finite()));
        assert!(normalized.iter().any(|value| *value != 0.0));
        assert!(
            packed
                .to_host_vec(&stream)
                .expect("download packed HC output")
                .iter()
                .any(|value| *value != 0)
        );
    }
}

#[test]
#[ignore = "manual GB10 latency checkpoint"]
fn sm121_hc_producer_formal_shape_latency() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const HC: usize = 4;
    const HIDDEN: usize = 4096;
    const MIX: usize = 24;
    const ITERATIONS: usize = 20;
    let function = DeviceBuffer::from_host(&stream, &vec![0.0f32; HC * HIDDEN * MIX])
        .expect("upload HC function");
    let hc_scale = DeviceBuffer::from_host(&stream, &[1.0f32; 3]).expect("upload HC scales");
    let hc_base = DeviceBuffer::from_host(&stream, &[0.0f32; MIX]).expect("upload HC base");
    let rms_weight =
        DeviceBuffer::from_host(&stream, &vec![1.0f32; HIDDEN]).expect("upload RMS weight");

    for rows in [1usize, 2, 4, 8, 17] {
        let state = DeviceBuffer::from_host(&stream, &vec![1.0f32; rows * HC * HIDDEN])
            .expect("upload HC state");
        let mut hidden = DeviceBuffer::<f32>::zeroed(&stream, rows * HIDDEN).expect("HC hidden");
        let mut normalized =
            DeviceBuffer::<f32>::zeroed(&stream, rows * HIDDEN).expect("HC normalized");
        let mut packed = DeviceBuffer::<u8>::zeroed(&stream, rows * HIDDEN).expect("HC packed");
        let mut scales =
            DeviceBuffer::<u8>::zeroed(&stream, rows * (HIDDEN / 128)).expect("HC FP8 scales");
        let mut pre = DeviceBuffer::<f32>::zeroed(&stream, rows * HC).expect("HC pre");
        let mut post = DeviceBuffer::<f32>::zeroed(&stream, rows * HC).expect("HC post");
        let mut comb = DeviceBuffer::<f32>::zeroed(&stream, rows * HC * HC).expect("HC comb");

        cutlass::hc_producer(
            &stream,
            &state,
            &function,
            &hc_scale,
            &hc_base,
            &rms_weight,
            &mut hidden,
            &mut normalized,
            &mut packed,
            &mut scales,
            &mut pre,
            &mut post,
            &mut comb,
            rows,
            HC,
            HIDDEN,
            20,
            1.0e-5,
            1.0e-5,
            1.0e-5,
        )
        .expect("warm HC producer");
        stream.synchronize().expect("warm HC sync");

        let started = std::time::Instant::now();
        for _ in 0..ITERATIONS {
            cutlass::hc_producer(
                &stream,
                &state,
                &function,
                &hc_scale,
                &hc_base,
                &rms_weight,
                &mut hidden,
                &mut normalized,
                &mut packed,
                &mut scales,
                &mut pre,
                &mut post,
                &mut comb,
                rows,
                HC,
                HIDDEN,
                20,
                1.0e-5,
                1.0e-5,
                1.0e-5,
            )
            .expect("HC producer launch");
        }
        stream.synchronize().expect("HC sync");
        let milliseconds = started.elapsed().as_secs_f64() * 1_000.0 / ITERATIONS as f64;
        println!("hc_producer formal rows={rows} latency_ms={milliseconds:.6}");
    }
}

#[test]
fn sm121_shared_ffn_accepts_dynamic_m_and_accumulates() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const INPUT: usize = 128;
    const INTERMEDIATE: usize = 128;
    const OUTPUT: usize = 16;
    let gate = DeviceBuffer::from_host(&stream, &vec![0u8; INTERMEDIATE * INPUT])
        .expect("upload zero gate");
    let up =
        DeviceBuffer::from_host(&stream, &vec![0u8; INTERMEDIATE * INPUT]).expect("upload zero up");
    let down = DeviceBuffer::from_host(&stream, &vec![0u8; OUTPUT * INTERMEDIATE])
        .expect("upload zero down");
    let gate_scales = DeviceBuffer::from_host(&stream, &[127u8]).expect("gate scales");
    let up_scales = DeviceBuffer::from_host(&stream, &[127u8]).expect("up scales");
    let down_scales = DeviceBuffer::from_host(&stream, &[127u8]).expect("down scales");

    for rows in [1usize, 2, 4, 8, 17] {
        let input = DeviceBuffer::from_host(&stream, &vec![0x38u8; rows * INPUT])
            .expect("upload shared FFN input");
        let input_scales = DeviceBuffer::from_host(&stream, &vec![127u8; rows])
            .expect("upload shared FFN input scales");
        let mut hidden_f32 = DeviceBuffer::<f32>::zeroed(&stream, rows * INTERMEDIATE)
            .expect("shared FFN hidden F32");
        let mut hidden =
            DeviceBuffer::<u8>::zeroed(&stream, rows * INTERMEDIATE).expect("shared FFN hidden");
        let mut hidden_scales =
            DeviceBuffer::<u8>::zeroed(&stream, rows).expect("shared FFN hidden scales");
        let mut overwrite = DeviceBuffer::from_host(&stream, &vec![9.0f32; rows * OUTPUT])
            .expect("upload overwrite output");
        cutlass::shared_ffn(
            &stream,
            &input,
            &input_scales,
            &gate,
            &gate_scales,
            &up,
            &up_scales,
            &down,
            &down_scales,
            &mut hidden_f32,
            &mut hidden,
            &mut hidden_scales,
            &mut overwrite,
            rows,
            INPUT,
            INTERMEDIATE,
            OUTPUT,
            (128, 128),
            (128, 128),
            (128, 128),
            1.0,
            0.0,
            false,
        )
        .expect("dynamic-M shared FFN overwrite");
        assert!(
            overwrite
                .to_host_vec(&stream)
                .expect("download overwrite")
                .iter()
                .all(|value| *value == 0.0)
        );

        let mut accumulate = DeviceBuffer::from_host(&stream, &vec![3.0f32; rows * OUTPUT])
            .expect("upload accumulator");
        cutlass::shared_ffn(
            &stream,
            &input,
            &input_scales,
            &gate,
            &gate_scales,
            &up,
            &up_scales,
            &down,
            &down_scales,
            &mut hidden_f32,
            &mut hidden,
            &mut hidden_scales,
            &mut accumulate,
            rows,
            INPUT,
            INTERMEDIATE,
            OUTPUT,
            (128, 128),
            (128, 128),
            (128, 128),
            1.0,
            0.0,
            true,
        )
        .expect("dynamic-M shared FFN accumulate");
        assert!(
            accumulate
                .to_host_vec(&stream)
                .expect("download accumulator")
                .iter()
                .all(|value| *value == 3.0)
        );
    }
}

#[test]
#[ignore = "manual GB10 latency checkpoint"]
fn sm121_shared_ffn_formal_shape_latency() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const INPUT: usize = 4096;
    const INTERMEDIATE: usize = 2048;
    const OUTPUT: usize = 4096;
    const ITERATIONS: usize = 5;
    let gate = DeviceBuffer::from_host(&stream, &vec![0x38u8; INTERMEDIATE * INPUT])
        .expect("upload formal gate");
    let up = DeviceBuffer::from_host(&stream, &vec![0x38u8; INTERMEDIATE * INPUT])
        .expect("upload formal up");
    let down = DeviceBuffer::from_host(&stream, &vec![0x38u8; OUTPUT * INTERMEDIATE])
        .expect("upload formal down");
    let gate_scales =
        DeviceBuffer::from_host(&stream, &vec![127u8; 16 * 32]).expect("upload formal gate scales");
    let up_scales =
        DeviceBuffer::from_host(&stream, &vec![127u8; 16 * 32]).expect("upload formal up scales");
    let down_scales =
        DeviceBuffer::from_host(&stream, &vec![127u8; 32 * 16]).expect("upload formal down scales");

    for rows in [1usize, 2, 4, 8, 17] {
        let input = DeviceBuffer::from_host(&stream, &vec![0x38u8; rows * INPUT])
            .expect("upload formal input");
        let input_scales = DeviceBuffer::from_host(&stream, &vec![127u8; rows * (INPUT / 128)])
            .expect("upload formal input scales");
        let mut hidden_f32 =
            DeviceBuffer::<f32>::zeroed(&stream, rows * INTERMEDIATE).expect("formal hidden F32");
        let mut hidden =
            DeviceBuffer::<u8>::zeroed(&stream, rows * INTERMEDIATE).expect("formal hidden");
        let mut hidden_scales = DeviceBuffer::<u8>::zeroed(&stream, rows * (INTERMEDIATE / 128))
            .expect("formal hidden scales");
        let mut output =
            DeviceBuffer::<f32>::zeroed(&stream, rows * OUTPUT).expect("formal output");
        cutlass::shared_ffn(
            &stream,
            &input,
            &input_scales,
            &gate,
            &gate_scales,
            &up,
            &up_scales,
            &down,
            &down_scales,
            &mut hidden_f32,
            &mut hidden,
            &mut hidden_scales,
            &mut output,
            rows,
            INPUT,
            INTERMEDIATE,
            OUTPUT,
            (128, 128),
            (128, 128),
            (128, 128),
            1.0,
            0.0,
            false,
        )
        .expect("warm formal shared FFN");
        stream.synchronize().expect("warm sync");

        let started = std::time::Instant::now();
        for _ in 0..ITERATIONS {
            cutlass::shared_ffn(
                &stream,
                &input,
                &input_scales,
                &gate,
                &gate_scales,
                &up,
                &up_scales,
                &down,
                &down_scales,
                &mut hidden_f32,
                &mut hidden,
                &mut hidden_scales,
                &mut output,
                rows,
                INPUT,
                INTERMEDIATE,
                OUTPUT,
                (128, 128),
                (128, 128),
                (128, 128),
                1.0,
                0.0,
                false,
            )
            .expect("formal shared FFN launch");
        }
        stream.synchronize().expect("formal sync");
        let milliseconds = started.elapsed().as_secs_f64() * 1_000.0 / ITERATIONS as f64;
        println!("shared_ffn formal rows={rows} latency_ms={milliseconds:.6}");
    }
}
