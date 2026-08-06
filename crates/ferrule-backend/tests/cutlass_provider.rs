#![cfg(feature = "cuda")]

use ferrule_backend::cuda::cutlass::{self, CutlassKernelId};
use ferrule_backend::cuda::provider::{
    ExecutionMode, KernelOperation, KernelProviderId, LayerKernelRequirements,
    LinearBundleRequirement, OperationRequirement, WeightLayout,
};
use ferrule_backend::cuda::runtime::{CudaContext, DeviceBuffer};

fn bf16_storage_word(value: f32) -> u16 {
    let bits = value.to_bits();
    let bias = 0x7fffu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(bias) >> 16) as u16
}

fn bf16_boundary(value: f32) -> f32 {
    f32::from_bits(u32::from(bf16_storage_word(value)) << 16)
}

fn native_kernel_available(kernel: CutlassKernelId) -> bool {
    cutlass::discover_provider()
        .expect("CUDA provider")
        .supports(kernel)
}

fn bf16_storage_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16_storage_word(*value).to_le_bytes())
        .collect()
}

#[test]
fn cuda_plan_selects_published_inference_operations() {
    let manifest = cutlass::discover_provider()
        .expect("CUDA provider")
        .manifest();
    let mut layer = LayerKernelRequirements::default();
    if manifest.supports(CutlassKernelId::Fp8QueryAKv) {
        layer.add_linear_bundle(LinearBundleRequirement::new(
            KernelOperation::MlaQueryAKv,
            ExecutionMode::Inference,
            4096,
            [1024, 1024],
            WeightLayout::Fp8E4m3BlockScaled,
        ));
    }
    if manifest.supports(CutlassKernelId::Fp8Projection) {
        layer.add_linear_bundle(LinearBundleRequirement::new(
            KernelOperation::MlaQueryB,
            ExecutionMode::Inference,
            1024,
            [32768],
            WeightLayout::Fp8E4m3BlockScaled,
        ));
    }
    layer.add_linear_bundle(LinearBundleRequirement::new(
        KernelOperation::MainCompressorProjection,
        ExecutionMode::Inference,
        4096,
        [1024, 1024],
        WeightLayout::Bf16RowMajor,
    ));
    let mut expected_operations = vec![KernelOperation::MainCompressorProjection];
    for (operation, kernel) in [
        (KernelOperation::MlaQueryAKv, CutlassKernelId::Fp8QueryAKv),
        (KernelOperation::MlaQueryB, CutlassKernelId::Fp8Projection),
        (
            KernelOperation::AttentionHcPre,
            CutlassKernelId::HyperConnectionProducer,
        ),
        (
            KernelOperation::FeedForwardHcPre,
            CutlassKernelId::HyperConnectionProducer,
        ),
        (KernelOperation::MlaOutput, CutlassKernelId::MlaOutput),
        (KernelOperation::SharedFfn, CutlassKernelId::SharedFfn),
        (
            KernelOperation::MainProjectNorm,
            CutlassKernelId::MainProjectNorm,
        ),
        (
            KernelOperation::HybridMlaAttention,
            CutlassKernelId::HybridMlaAttention,
        ),
        (KernelOperation::ProposalHead, CutlassKernelId::ProposalHead),
    ] {
        if manifest.supports(kernel) {
            if !matches!(
                operation,
                KernelOperation::MlaQueryAKv | KernelOperation::MlaQueryB
            ) {
                layer.require(OperationRequirement::new(
                    operation,
                    ExecutionMode::Inference,
                ));
            }
            expected_operations.push(operation);
        }
    }

    let plan = ferrule_backend::cuda::compile_cuda_model_plan(&[layer])
        .expect("compile CUDA inference plan");
    for operation in expected_operations {
        let launch = plan.layers[0]
            .operation(operation, ExecutionMode::Inference)
            .expect("required semantic launch");
        assert_eq!(launch.kernel.provider, KernelProviderId::CUDA_CUTLASS);
        assert_eq!(launch.kernel.operation, operation);
        assert_eq!(launch.kernel.mode, ExecutionMode::Inference);
        assert_eq!(launch.kernel.variant, 0);
        assert!(launch.is_provider_managed());
        assert!(launch.is_capture_safe());
    }
}

#[test]
fn cuda_plan_rejects_grouped_fp4_without_a_native_kernel() {
    let manifest = cutlass::discover_provider()
        .expect("CUDA provider")
        .manifest();
    if manifest.supports(CutlassKernelId::GroupedFp4Moe) {
        return;
    }

    let mut layer = LayerKernelRequirements::default();
    layer.require(OperationRequirement::new(
        KernelOperation::GroupedFp4Moe,
        ExecutionMode::Inference,
    ));
    let error = ferrule_backend::cuda::compile_cuda_model_plan(&[layer])
        .expect_err("grouped FP4 must fail closed without a native kernel");
    let message = error.to_string();
    assert!(message.contains("GroupedFp4Moe"), "{message}");
    assert!(message.contains("Inference"), "{message}");
}

#[test]
fn cuda_plan_rejects_unpublished_operation_mode() {
    let mut layer = LayerKernelRequirements::default();
    layer.require(OperationRequirement::new(
        KernelOperation::SparseAttention,
        ExecutionMode::Backward,
    ));
    let error = ferrule_backend::cuda::compile_cuda_model_plan(&[layer])
        .expect_err("unbound operation mode must fail closed");
    let message = error.to_string();
    assert!(message.contains("SparseAttention"), "{message}");
    assert!(message.contains("Backward"), "{message}");
}

#[test]
fn cuda_manifest_publishes_native_capabilities() {
    assert_eq!(CutlassKernelId::GroupedFp4Moe.name(), "grouped_fp4_moe");

    let manifest = cutlass::discover_provider()
        .expect("CUDA provider")
        .manifest();
    let target = ferrule_backend::cuda::CudaTarget::parse(ferrule_backend::cuda::COMPILED_TARGET)
        .expect("compiled CUDA target");
    let capabilities = target.capabilities();
    assert_eq!(
        manifest.supports(CutlassKernelId::Fp8QueryAKv),
        capabilities.fp8_mma_sync
    );
    assert!(manifest.supports(CutlassKernelId::Bf16Compressor));
    assert!(manifest.supports(CutlassKernelId::HyperConnectionProducer));
    assert_eq!(
        manifest.supports(CutlassKernelId::SharedFfn),
        capabilities.fp8_mma_sync
    );
    assert_eq!(
        manifest.supports(CutlassKernelId::GroupedFp4Moe),
        capabilities.sm103_block_scaled_fp4 && !cfg!(debug_assertions)
    );
    assert_eq!(
        manifest.supports(CutlassKernelId::MlaOutput),
        capabilities.fp8_mma_sync
    );
    assert_eq!(
        manifest.supports(CutlassKernelId::MainProjectNorm),
        capabilities.fp8_mma_sync
    );
    assert!(manifest.supports(CutlassKernelId::HybridMlaAttention));
    assert!(manifest.supports(CutlassKernelId::ProposalHead));
    assert_eq!(
        manifest.supports(CutlassKernelId::Fp8Projection),
        capabilities.fp8_mma_sync
    );
}

#[test]
fn cuda_mla_output_single_row_matches_cooperative_path_bitwise() {
    if !native_kernel_available(CutlassKernelId::MlaOutput) {
        return;
    }
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.new_stream().expect("create CUDA stream");

    const SINGLE_ROWS: usize = 1;
    const COOPERATIVE_ROWS: usize = 2;
    const GROUPS: usize = 2;
    const GROUP_INPUT: usize = 256;
    const CONTEXT: usize = GROUPS * GROUP_INPUT;
    const RANK: usize = 128;
    const LATENT: usize = GROUPS * RANK;
    const HIDDEN: usize = 192;

    let context_row = (0..CONTEXT)
        .map(|index| ((index % 8) + 1) as f32 * 0.125)
        .collect::<Vec<_>>();
    let mut cooperative_context_host = context_row.clone();
    cooperative_context_host.extend_from_slice(&context_row);
    let output_a_weight_host = (0..LATENT * GROUP_INPUT)
        .map(|index| if index % 3 == 0 { 0x30u8 } else { 0x38u8 })
        .collect::<Vec<_>>();
    let output_b_weight_host = (0..HIDDEN * LATENT)
        .map(|index| if index % 5 == 0 { 0x30u8 } else { 0x38u8 })
        .collect::<Vec<_>>();

    let single_context =
        DeviceBuffer::from_host(&stream, &context_row).expect("upload single-row MLA context");
    let cooperative_context = DeviceBuffer::from_host(&stream, &cooperative_context_host)
        .expect("upload cooperative MLA context");
    let output_a_weight = DeviceBuffer::from_host(&stream, &output_a_weight_host)
        .expect("upload MLA output-A weight");
    let output_a_scales = DeviceBuffer::from_host(
        &stream,
        &vec![127u8; LATENT.div_ceil(128) * (GROUP_INPUT / 128)],
    )
    .expect("upload MLA output-A scales");
    let output_b_weight = DeviceBuffer::from_host(&stream, &output_b_weight_host)
        .expect("upload MLA output-B weight");
    let output_b_scales =
        DeviceBuffer::from_host(&stream, &vec![127u8; HIDDEN.div_ceil(128) * (LATENT / 128)])
            .expect("upload MLA output-B scales");

    let mut single_latent =
        DeviceBuffer::<u16>::zeroed(&stream, SINGLE_ROWS * LATENT).expect("single-row BF16 latent");
    let mut single_latent_fp8 =
        DeviceBuffer::<u8>::zeroed(&stream, SINGLE_ROWS * LATENT).expect("single-row latent FP8");
    let mut single_latent_scales =
        DeviceBuffer::<u8>::zeroed(&stream, SINGLE_ROWS * (LATENT / 128))
            .expect("single-row latent scales");

    let mut single_output =
        DeviceBuffer::<f32>::zeroed(&stream, SINGLE_ROWS * HIDDEN).expect("single-row output");
    cutlass::mla_output(
        &stream,
        &single_context,
        &output_a_weight,
        &output_a_scales,
        &output_b_weight,
        &output_b_scales,
        &mut single_latent,
        &mut single_latent_fp8,
        &mut single_latent_scales,
        &mut single_output,
        SINGLE_ROWS,
        CONTEXT,
        GROUPS,
        GROUP_INPUT,
        RANK,
        LATENT,
        HIDDEN,
    )
    .expect("single-row MLA split launch");

    let mut cooperative_latent = DeviceBuffer::<u16>::zeroed(&stream, COOPERATIVE_ROWS * LATENT)
        .expect("cooperative BF16 latent");
    let mut cooperative_latent_fp8 = DeviceBuffer::<u8>::zeroed(&stream, COOPERATIVE_ROWS * LATENT)
        .expect("cooperative latent FP8");
    let mut cooperative_latent_scales =
        DeviceBuffer::<u8>::zeroed(&stream, COOPERATIVE_ROWS * (LATENT / 128))
            .expect("cooperative latent scales");

    let mut cooperative_output = DeviceBuffer::<f32>::zeroed(&stream, COOPERATIVE_ROWS * HIDDEN)
        .expect("cooperative output");
    cutlass::mla_output(
        &stream,
        &cooperative_context,
        &output_a_weight,
        &output_a_scales,
        &output_b_weight,
        &output_b_scales,
        &mut cooperative_latent,
        &mut cooperative_latent_fp8,
        &mut cooperative_latent_scales,
        &mut cooperative_output,
        COOPERATIVE_ROWS,
        CONTEXT,
        GROUPS,
        GROUP_INPUT,
        RANK,
        LATENT,
        HIDDEN,
    )
    .expect("cooperative MLA launch");

    let single_latent = single_latent
        .to_host_vec(&stream)
        .expect("download single-row latent");
    let cooperative_latent = cooperative_latent
        .to_host_vec(&stream)
        .expect("download cooperative latent");
    assert_eq!(single_latent, cooperative_latent[..LATENT]);
    assert_eq!(
        single_latent_fp8
            .to_host_vec(&stream)
            .expect("download single-row latent FP8"),
        cooperative_latent_fp8
            .to_host_vec(&stream)
            .expect("download cooperative latent FP8")[..LATENT]
    );
    assert_eq!(
        single_latent_scales
            .to_host_vec(&stream)
            .expect("download single-row latent scales"),
        cooperative_latent_scales
            .to_host_vec(&stream)
            .expect("download cooperative latent scales")[..LATENT / 128]
    );

    let single_output = single_output
        .to_host_vec(&stream)
        .expect("download single-row MLA output");
    let cooperative_output = cooperative_output
        .to_host_vec(&stream)
        .expect("download cooperative MLA output");
    let single_output_bits = single_output
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert!(single_output.iter().all(|value| value.is_finite()));
    assert!(single_output.iter().any(|value| *value != 0.0));
    assert_eq!(
        single_output_bits,
        cooperative_output[..HIDDEN]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        single_output_bits,
        cooperative_output[HIDDEN..]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn cuda_mla_output_b_accumulates_all_groups_before_bf16_rounding() {
    if !native_kernel_available(CutlassKernelId::MlaOutput) {
        return;
    }
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.new_stream().expect("create CUDA stream");

    const ROWS: usize = 1;
    const GROUPS: usize = 2;
    const GROUP_INPUT: usize = 128;
    const CONTEXT: usize = GROUPS * GROUP_INPUT;
    const RANK: usize = 128;
    const LATENT: usize = GROUPS * RANK;
    const HIDDEN: usize = 16;

    // Output-A produces an exact latent value of 1 in every channel. Output-B's
    // first K128 block contributes 1 + 1/256 and its second contributes -1.
    // Rounding each group first loses the residual; one final BF16 boundary keeps it.
    let mut context_host = vec![0.0f32; CONTEXT];
    for group in 0..GROUPS {
        context_host[group * GROUP_INPUT] = 1.0;
    }
    let mut output_a_weight_host = vec![0u8; LATENT * GROUP_INPUT];
    for channel in 0..LATENT {
        output_a_weight_host[channel * GROUP_INPUT] = 0x38; // E4M3 1.0
    }
    let output_a_scales_host = vec![127u8; LATENT / 128];

    let mut output_b_weight_host = vec![0u8; HIDDEN * LATENT];
    for channel in 0..HIDDEN {
        let base = channel * LATENT;
        output_b_weight_host[base..base + 128].fill(0x38); // E4M3 1.0
        output_b_weight_host[base + 127] = 0x3c; // E4M3 1.5
        output_b_weight_host[base + 128..base + 256].fill(0xb8); // E4M3 -1.0
    }
    let output_b_scales_host = vec![120u8; LATENT / 128]; // UE8M0 2^-7

    let context = DeviceBuffer::from_host(&stream, &context_host).expect("upload MLA context");
    let output_a_weight = DeviceBuffer::from_host(&stream, &output_a_weight_host)
        .expect("upload MLA output-A weight");
    let output_a_scales = DeviceBuffer::from_host(&stream, &output_a_scales_host)
        .expect("upload MLA output-A scales");
    let output_b_weight = DeviceBuffer::from_host(&stream, &output_b_weight_host)
        .expect("upload MLA output-B weight");
    let output_b_scales = DeviceBuffer::from_host(&stream, &output_b_scales_host)
        .expect("upload MLA output-B scales");
    let mut latent = DeviceBuffer::<u16>::zeroed(&stream, LATENT).expect("MLA BF16 latent");
    let mut latent_fp8 = DeviceBuffer::<u8>::zeroed(&stream, LATENT).expect("MLA latent FP8");
    let mut latent_scales =
        DeviceBuffer::<u8>::zeroed(&stream, LATENT / 128).expect("MLA latent scales");
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, HIDDEN).expect("MLA output");

    cutlass::mla_output(
        &stream,
        &context,
        &output_a_weight,
        &output_a_scales,
        &output_b_weight,
        &output_b_scales,
        &mut latent,
        &mut latent_fp8,
        &mut latent_scales,
        &mut output,
        ROWS,
        CONTEXT,
        GROUPS,
        GROUP_INPUT,
        RANK,
        LATENT,
        HIDDEN,
    )
    .expect("MLA accumulation-order launch");

    let latent = latent.to_host_vec(&stream).expect("download MLA latent");
    assert!(latent.iter().all(|&value| value == bf16_storage_word(1.0)));

    let expected = bf16_boundary((257.0f32 / 256.0) - 1.0);
    let rounded_per_group = bf16_boundary(bf16_boundary(257.0 / 256.0) + bf16_boundary(-1.0));
    assert_ne!(expected.to_bits(), rounded_per_group.to_bits());
    let actual = output.to_host_vec(&stream).expect("download MLA output");
    assert!(
        actual
            .iter()
            .all(|value| value.to_bits() == expected.to_bits()),
        "output-B must preserve the cross-group residual until its final BF16 boundary: actual={actual:?} expected={expected} old_group_rounded={rounded_per_group}"
    );
}

#[test]
fn cuda_proposal_head_keeps_markov_dependency_on_device() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const ROWS: usize = 5;
    const HC: usize = 2;
    const HIDDEN: usize = 16;
    const VOCAB: usize = 128;
    const RANK: usize = 16;
    const PARTIALS: usize = 64;

    let hc_state_host = (0..ROWS * HC * HIDDEN)
        .map(|index| ((index % 23) as f32 - 11.0) * 0.01)
        .collect::<Vec<_>>();
    let hc_function_host = vec![0.0f32; HC * HC * HIDDEN];
    let hc_scale_host = vec![1.0f32];
    let hc_base_host = vec![0.0f32; HC];
    let norm_host = vec![1.0f32; HIDDEN];
    let lm_head_host = (0..VOCAB * HIDDEN)
        .map(|index| ((index % HIDDEN) as f32 - 7.0) * 0.001)
        .collect::<Vec<_>>();
    let mut markov_w1_host = vec![0.0f32; VOCAB * RANK];
    for token in 0..VOCAB {
        markov_w1_host[token * RANK] = 1.0;
    }
    let mut markov_w2_host = vec![0.0f32; VOCAB * RANK];
    markov_w2_host[3 * RANK] = 8.0;
    let confidence_host = vec![0.01f32; HIDDEN + RANK];

    let hc_state = DeviceBuffer::from_host(&stream, &hc_state_host).expect("upload HC state");
    let hc_function =
        DeviceBuffer::from_host(&stream, &hc_function_host).expect("upload HC function");
    let hc_scale = DeviceBuffer::from_host(&stream, &hc_scale_host).expect("upload HC scale");
    let hc_base = DeviceBuffer::from_host(&stream, &hc_base_host).expect("upload HC base");
    let norm = DeviceBuffer::from_host(&stream, &norm_host).expect("upload final norm");
    let lm_head = DeviceBuffer::from_host(&stream, &bf16_storage_bytes(&lm_head_host))
        .expect("upload LM head");
    let markov_w1 = DeviceBuffer::from_host(&stream, &bf16_storage_bytes(&markov_w1_host))
        .expect("upload Markov W1");
    let markov_w2 = DeviceBuffer::from_host(&stream, &bf16_storage_bytes(&markov_w2_host))
        .expect("upload Markov W2");
    let confidence = DeviceBuffer::from_host(&stream, &bf16_storage_bytes(&confidence_host))
        .expect("upload confidence head");
    let mut hidden = DeviceBuffer::<f32>::zeroed(&stream, ROWS * HIDDEN).expect("hidden");
    let mut normalized = DeviceBuffer::<f32>::zeroed(&stream, ROWS * HIDDEN).expect("normalized");
    let mut base_logits = DeviceBuffer::<f32>::zeroed(&stream, ROWS * VOCAB).expect("base logits");
    let mut partial_values =
        DeviceBuffer::<f32>::zeroed(&stream, PARTIALS).expect("partial values");
    let mut partial_indices =
        DeviceBuffer::<i32>::zeroed(&stream, PARTIALS).expect("partial indices");
    let mut token_ids =
        DeviceBuffer::from_host(&stream, &[5i32, 0, 0, 0, 0, 0]).expect("token ids");
    let mut confidence_output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS).expect("confidence output");
    let mut status = DeviceBuffer::<i32>::zeroed(&stream, 1).expect("device status");

    cutlass::proposal_head(
        &stream,
        &hc_state,
        &hc_function,
        &hc_scale,
        &hc_base,
        &norm,
        &lm_head,
        &markov_w1,
        &markov_w2,
        &confidence,
        &mut hidden,
        &mut normalized,
        &mut base_logits,
        &mut partial_values,
        &mut partial_indices,
        &mut token_ids,
        &mut confidence_output,
        &mut status,
        cutlass::ProposalHeadLayout {
            rows: ROWS,
            hc: HC,
            hidden: HIDDEN,
            vocab: VOCAB,
            markov_rank: RANK,
            partial_capacity: PARTIALS,
            hc_eps: 1.0e-6,
            norm_eps: 1.0e-6,
        },
    )
    .expect("proposal proposal-head launch");

    assert_eq!(
        token_ids.to_host_vec(&stream).expect("download tokens"),
        [5, 3, 3, 3, 3, 3]
    );
    assert_eq!(status.to_host_vec(&stream).expect("download status"), [0]);
    assert!(
        confidence_output
            .to_host_vec(&stream)
            .expect("download confidence")
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn cuda_hc_mean_scatter_builds_proposal_target_taps_without_host_concat() {
    let ops = ferrule_backend::cuda::context::CudaArtifactOperatorContext::new()
        .expect("CUDA artifact operator context");
    const ROWS: usize = 2;
    const HC: usize = 4;
    const HIDDEN: usize = 128;
    const TAPS: usize = 3;
    const SLOT: usize = 1;
    let mut state = vec![0.0f32; ROWS * HC * HIDDEN];
    for row in 0..ROWS {
        for copy in 0..HC {
            for dim in 0..HIDDEN {
                state[row * HC * HIDDEN + copy * HIDDEN + dim] =
                    row as f32 * 10.0 + copy as f32 + dim as f32 / HIDDEN as f32;
            }
        }
    }
    let state = ops.upload_f32_buffer(&state).expect("upload HC state");
    let mut taps = ops
        .zero_f32_buffer(ROWS * TAPS * HIDDEN)
        .expect("proposal target-tap buffer");
    ops.hc_mean_scatter_from_device_into(&state, ROWS, HC, HIDDEN, SLOT, TAPS, &mut taps)
        .expect("HC mean-scatter");
    let taps = ops
        .download_f32_buffer(&taps)
        .expect("download proposal target taps");

    for row in 0..ROWS {
        for tap in 0..TAPS {
            for dim in 0..HIDDEN {
                let value = taps[row * TAPS * HIDDEN + tap * HIDDEN + dim];
                let expected = if tap == SLOT {
                    bf16_boundary(row as f32 * 10.0 + 1.5 + dim as f32 / HIDDEN as f32)
                } else {
                    0.0
                };
                assert_eq!(value, expected);
            }
        }
    }
}

#[test]
fn cuda_main_project_norm_preserves_bf16_boundary() {
    if !native_kernel_available(CutlassKernelId::MainProjectNorm) {
        return;
    }
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const ROWS: usize = 2;
    const INPUT: usize = 128;
    const OUTPUT: usize = 128;
    let input = DeviceBuffer::from_host(&stream, &vec![1.0f32; ROWS * INPUT])
        .expect("upload proposal target taps");
    let mut activation =
        DeviceBuffer::<u8>::zeroed(&stream, ROWS * INPUT).expect("proposal activation scratch");
    let mut activation_scales = DeviceBuffer::<u8>::zeroed(&stream, ROWS * (INPUT / 128))
        .expect("proposal activation-scale scratch");
    let weight = DeviceBuffer::from_host(&stream, &vec![0x38u8; OUTPUT * INPUT])
        .expect("upload proposal main projection");
    let weight_scales =
        DeviceBuffer::from_host(&stream, &[127u8]).expect("upload proposal main-projection scales");
    let norm_weight =
        DeviceBuffer::from_host(&stream, &vec![1.0f32; OUTPUT]).expect("upload proposal main norm");
    let mut inv_rms =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS).expect("proposal inverse-RMS scratch");
    let mut output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * OUTPUT).expect("proposal normalized output");

    cutlass::main_project_norm(
        &stream,
        &input,
        &mut activation,
        &mut activation_scales,
        &weight,
        &weight_scales,
        &norm_weight,
        &mut inv_rms,
        &mut output,
        ROWS,
        INPUT,
        OUTPUT,
        1.0e-6,
    )
    .expect("CUDA proposal main-project/norm launch");

    for value in output
        .to_host_vec(&stream)
        .expect("download proposal main-project/norm output")
    {
        assert_eq!(value, 1.0);
    }
}

#[test]
fn cuda_hybrid_mla_attention_matches_full_block_reference() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const SEQUENCE_TOKENS: usize = 18;
    const PAGE_TOKENS: usize = cutlass::HYBRID_MLA_ATTENTION_PAGE_TOKENS;
    const LAYER_COUNT: usize = 2;
    const LAYER_INDEX: usize = 1;
    const PHYSICAL_SLOTS: usize = 2;
    const ROWS: usize = cutlass::PROPOSAL_ROWS;
    const HEADS: usize = cutlass::HYBRID_MLA_ATTENTION_HEADS;
    const DIM: usize = cutlass::HYBRID_MLA_ATTENTION_HEAD_DIM;
    const CAPACITY: usize = cutlass::HYBRID_MLA_ATTENTION_TOKEN_CAPACITY;

    // Logical page zero is deliberately stored in physical slot one and page
    // one in slot zero. The tested layer is also nonzero.
    let block_slots_host = [1i32, 0i32];
    let slot_stride = LAYER_COUNT * PAGE_TOKENS * DIM;
    let layer_stride = PAGE_TOKENS * DIM;
    let mut context_rows = vec![0.0f32; SEQUENCE_TOKENS * DIM];
    let mut context_plane = vec![0.0f32; PHYSICAL_SLOTS * slot_stride];
    for token in 0..SEQUENCE_TOKENS {
        let slot = block_slots_host[token / PAGE_TOKENS] as usize;
        let plane_base =
            slot * slot_stride + LAYER_INDEX * layer_stride + (token % PAGE_TOKENS) * DIM;
        for dim in 0..DIM {
            let value = ((token * 29 + dim * 7) % 37) as f32 * 0.0025 - 18.0 * 0.0025;
            context_rows[token * DIM + dim] = value;
            context_plane[plane_base + dim] = value;
        }
    }

    let mut block_kv_host = vec![0.0f32; ROWS * DIM];
    for row in 0..ROWS {
        for dim in 0..DIM {
            block_kv_host[row * DIM + dim] =
                ((row * 31 + dim * 11) % 41) as f32 * 0.003 - 20.0 * 0.003;
        }
    }
    // A strong future-block feature makes an accidental causal mask observable.
    block_kv_host[(ROWS - 1) * DIM] += 8.0;

    let mut query_host = vec![0.0f32; ROWS * HEADS * DIM];
    for row in 0..ROWS {
        for head in 0..HEADS {
            for dim in 0..DIM {
                query_host[(row * HEADS + head) * DIM + dim] =
                    ((row * 13 + head * 17 + dim * 5) % 43) as f32 * 0.004 - 21.0 * 0.004;
            }
        }
    }
    for head in 0..HEADS {
        query_host[head * DIM] += 4.0;
    }
    let sink_host = (0..HEADS)
        .map(|head| -0.2 + head as f32 * 0.005)
        .collect::<Vec<_>>();
    let scale = (DIM as f32).powf(-0.5);

    fn reference(
        query: &[f32],
        context: &[f32],
        block: &[f32],
        sink: &[f32],
        block_rows_visible: impl Fn(usize) -> usize,
        scale: f32,
    ) -> Vec<f32> {
        const ROWS: usize = cutlass::PROPOSAL_ROWS;
        const HEADS: usize = cutlass::HYBRID_MLA_ATTENTION_HEADS;
        const DIM: usize = cutlass::HYBRID_MLA_ATTENTION_HEAD_DIM;
        let context_tokens = context.len() / DIM;
        let mut output = vec![0.0f32; ROWS * HEADS * DIM];
        for row in 0..ROWS {
            let visible_block = block_rows_visible(row);
            for head in 0..HEADS {
                let q_base = (row * HEADS + head) * DIM;
                let mut scores = Vec::with_capacity(context_tokens + visible_block);
                for token in 0..context_tokens + visible_block {
                    let values = if token < context_tokens {
                        &context[token * DIM..(token + 1) * DIM]
                    } else {
                        let block_row = token - context_tokens;
                        &block[block_row * DIM..(block_row + 1) * DIM]
                    };
                    let mut dot = 0.0f32;
                    for dim in 0..DIM {
                        dot += bf16_boundary(query[q_base + dim]) * bf16_boundary(values[dim]);
                    }
                    scores.push(dot * scale);
                }
                let mut running_maximum = f32::NEG_INFINITY;
                let mut denominator = 0.0f32;
                let mut accumulator = vec![0.0f32; DIM];
                for (tile_index, score_tile) in scores
                    .chunks(cutlass::HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILE)
                    .enumerate()
                {
                    let tile_maximum = score_tile.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let next_maximum = running_maximum.max(tile_maximum);
                    let rescale = if running_maximum == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (running_maximum - next_maximum).exp()
                    };
                    denominator *= rescale;
                    for value in &mut accumulator {
                        *value *= rescale;
                    }
                    let tile_base = tile_index * cutlass::HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILE;
                    for (tile_token, score) in score_tile.iter().copied().enumerate() {
                        let token = tile_base + tile_token;
                        let exponent = (score - next_maximum).exp();
                        denominator += exponent;
                        let weight = bf16_boundary(exponent);
                        let values = if token < context_tokens {
                            &context[token * DIM..(token + 1) * DIM]
                        } else {
                            let block_row = token - context_tokens;
                            &block[block_row * DIM..(block_row + 1) * DIM]
                        };
                        for dim in 0..DIM {
                            accumulator[dim] += weight * bf16_boundary(values[dim]);
                        }
                    }
                    running_maximum = next_maximum;
                }
                denominator += (sink[head] - running_maximum).exp();
                for dim in 0..DIM {
                    output[q_base + dim] = bf16_boundary(accumulator[dim] / denominator);
                }
            }
        }
        output
    }

    let expected = reference(
        &query_host,
        &context_rows,
        &block_kv_host,
        &sink_host,
        |_| ROWS,
        scale,
    );
    let causal = reference(
        &query_host,
        &context_rows,
        &block_kv_host,
        &sink_host,
        |row| row + 1,
        scale,
    );

    let query = DeviceBuffer::from_host(&stream, &query_host).expect("upload proposal Q");
    let expected_context_plane = context_plane.clone();
    let context_plane =
        DeviceBuffer::from_host(&stream, &context_plane).expect("upload paged proposal context");
    let block_kv =
        DeviceBuffer::from_host(&stream, &block_kv_host).expect("upload proposal block KV");
    let block_slots =
        DeviceBuffer::from_host(&stream, &block_slots_host).expect("upload proposal block slots");
    let sink = DeviceBuffer::from_host(&stream, &sink_host).expect("upload attention sink");
    let mut query_bf16 = DeviceBuffer::<u16>::zeroed(&stream, ROWS * HEADS * DIM)
        .expect("proposal query BF16 scratch");
    let mut gathered_kv_bf16 = DeviceBuffer::<u16>::zeroed(&stream, CAPACITY * DIM)
        .expect("proposal gathered KV BF16 scratch");
    let mut scores = DeviceBuffer::<f32>::zeroed(&stream, ROWS * HEADS * CAPACITY)
        .expect("proposal score scratch");
    let mut probabilities = DeviceBuffer::<u16>::zeroed(&stream, ROWS * HEADS * CAPACITY)
        .expect("proposal probability scratch");
    let mut online_rescales = DeviceBuffer::<f32>::zeroed(
        &stream,
        ROWS * HEADS * cutlass::HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILES,
    )
    .expect("proposal online-softmax rescale scratch");
    let mut denominators = DeviceBuffer::<f32>::zeroed(&stream, ROWS * HEADS)
        .expect("proposal softmax denominator scratch");
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, ROWS * HEADS * DIM)
        .expect("proposal attention output");
    let mut status = DeviceBuffer::<i32>::zeroed(&stream, 1).expect("proposal device status");

    cutlass::hybrid_mla_attention(
        &stream,
        &query,
        &context_plane,
        &block_kv,
        &block_slots,
        &sink,
        &mut query_bf16,
        &mut gathered_kv_bf16,
        &mut scores,
        &mut probabilities,
        &mut online_rescales,
        &mut denominators,
        &mut output,
        &mut status,
        cutlass::HybridMlaAttentionLayout {
            sequence_tokens: SEQUENCE_TOKENS,
            page_tokens: PAGE_TOKENS,
            elements_per_token: DIM,
            layer_index: LAYER_INDEX,
            layer_count: LAYER_COUNT,
            block_slot_offset: 0,
            block_slot_count: block_slots_host.len(),
            softmax_scale: scale,
        },
    )
    .expect("CUDA proposal hybrid-attention launch");

    assert_eq!(
        status
            .to_host_vec(&stream)
            .expect("download proposal status"),
        [0]
    );
    let actual = output
        .to_host_vec(&stream)
        .expect("download proposal hybrid-attention output");
    assert_eq!(
        context_plane
            .to_host_vec(&stream)
            .expect("download unchanged proposal context plane"),
        expected_context_plane,
        "ephemeral proposal-block attention modified committed paged KV"
    );
    assert!(
        actual.iter().all(|value| value.to_bits() & 0xffff == 0),
        "proposal hybrid attention output crossed its final BF16 boundary"
    );
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let max_bf16_ulp = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| bf16_ulp_distance(actual, expected))
        .max()
        .unwrap_or(0);
    assert!(
        max_bf16_ulp <= 1,
        "proposal hybrid attention differs from checkpoint-native BF16 reference: \
         max_abs={max_abs:e} max_bf16_ulp={max_bf16_ulp}"
    );
    assert!(
        (actual[0] - causal[0]).abs() > 0.1,
        "proposal query row zero appears causally masked: actual={} causal={}",
        actual[0],
        causal[0]
    );
}

const HYBRID_MLA_EXPLICIT_SELECTION_HEADS: usize = 64;
const HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM: usize = 512;

struct HybridMlaExplicitSelectionHostCase<'a> {
    label: &'static str,
    query: &'a [f32],
    first_plane: &'a [f32],
    second_plane: Option<&'a [f32]>,
    block_slots: Option<&'a [i32]>,
    block_offsets: Option<&'a [i32]>,
    sequence_kv_lens: Option<&'a [i32]>,
    second_sequence_kv_lens: Option<&'a [i32]>,
    row_sequence_ids: Option<&'a [i32]>,
    row_kv_lens: Option<&'a [i32]>,
    row_second_kv_lens: Option<&'a [i32]>,
    selected_indices: &'a [i32],
    selectors: Option<&'a [i32]>,
    attention_sink: &'a [f32],
    expected_gathered: &'a [f32],
    expected_valid: &'a [i32],
    layout: cutlass::HybridMlaExplicitSelectionLayout,
}

fn hybrid_mla_explicit_selection_f32_scalar_reference(
    query: &[f32],
    gathered: &[f32],
    valid: &[i32],
    attention_sink: &[f32],
    rows: usize,
    selected_width: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    let mut output = vec![
        0.0f32;
        rows * HYBRID_MLA_EXPLICIT_SELECTION_HEADS
            * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM
    ];
    for row in 0..rows {
        for (head, &sink_score) in attention_sink
            .iter()
            .take(HYBRID_MLA_EXPLICIT_SELECTION_HEADS)
            .enumerate()
        {
            let query_base = (row * HYBRID_MLA_EXPLICIT_SELECTION_HEADS + head)
                * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
            let mut maximum = sink_score;
            for selected in 0..selected_width {
                let selected_index = row * selected_width + selected;
                if valid[selected_index] == 0 {
                    continue;
                }
                let kv_base = selected_index * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
                let mut dot = 0.0f32;
                for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                    dot = query[query_base + dimension].mul_add(gathered[kv_base + dimension], dot);
                }
                maximum = maximum.max(dot * softmax_scale);
            }

            let mut denominator = (sink_score - maximum).exp();
            for selected in 0..selected_width {
                let selected_index = row * selected_width + selected;
                if valid[selected_index] == 0 {
                    continue;
                }
                let kv_base = selected_index * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
                let mut dot = 0.0f32;
                for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                    dot = query[query_base + dimension].mul_add(gathered[kv_base + dimension], dot);
                }
                let weight = dot.mul_add(softmax_scale, -maximum).exp();
                denominator += weight;
                for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                    output[query_base + dimension] = weight.mul_add(
                        gathered[kv_base + dimension],
                        output[query_base + dimension],
                    );
                }
            }
            for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                output[query_base + dimension] /= denominator;
            }
        }
    }
    output
}

fn hybrid_mla_explicit_selection_bf16_boundary_reference(
    query: &[f32],
    gathered: &[f32],
    valid: &[i32],
    attention_sink: &[f32],
    rows: usize,
    selected_width: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    let query = query.iter().copied().map(bf16_boundary).collect::<Vec<_>>();
    let gathered = gathered
        .iter()
        .copied()
        .map(bf16_boundary)
        .collect::<Vec<_>>();
    let mut output = vec![
        0.0f32;
        rows * HYBRID_MLA_EXPLICIT_SELECTION_HEADS
            * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM
    ];
    for row in 0..rows {
        for (head, &sink_score) in attention_sink
            .iter()
            .take(HYBRID_MLA_EXPLICIT_SELECTION_HEADS)
            .enumerate()
        {
            let query_base = (row * HYBRID_MLA_EXPLICIT_SELECTION_HEADS + head)
                * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
            let mut scores = vec![f32::NEG_INFINITY; selected_width];
            for (selected, score) in scores.iter_mut().enumerate() {
                let selected_index = row * selected_width + selected;
                if valid[selected_index] == 0 {
                    continue;
                }
                let kv_base = selected_index * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
                let mut dot = 0.0f32;
                for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                    dot += query[query_base + dimension] * gathered[kv_base + dimension];
                }
                *score = dot * softmax_scale;
            }
            let mut running_maximum = f32::NEG_INFINITY;
            let mut denominator = 0.0f32;
            for tile_base in (0..selected_width).step_by(64) {
                let tile_end = (tile_base + 64).min(selected_width);
                let tile_maximum = scores[tile_base..tile_end]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let next_maximum = running_maximum.max(tile_maximum);
                let rescale = if running_maximum == f32::NEG_INFINITY {
                    0.0
                } else {
                    (running_maximum - next_maximum).exp()
                };
                denominator *= rescale;
                for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                    output[query_base + dimension] *= rescale;
                }
                for (selected, &score) in scores[tile_base..tile_end].iter().enumerate() {
                    let selected = tile_base + selected;
                    let selected_index = row * selected_width + selected;
                    if valid[selected_index] == 0 {
                        continue;
                    }
                    let weight_f32 = (score - next_maximum).exp();
                    denominator += weight_f32;
                    let weight_bf16 = bf16_boundary(weight_f32);
                    let kv_base = selected_index * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
                    for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                        output[query_base + dimension] +=
                            weight_bf16 * gathered[kv_base + dimension];
                    }
                }
                running_maximum = next_maximum;
            }
            if running_maximum != f32::NEG_INFINITY {
                denominator += (sink_score - running_maximum).exp();
                for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                    output[query_base + dimension] =
                        bf16_boundary(output[query_base + dimension] / denominator);
                }
            }
        }
    }
    output
}

fn bf16_ordered_word(value: f32) -> u16 {
    let bits = bf16_storage_word(value);
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits | 0x8000
    }
}

fn bf16_ulp_distance(actual: f32, expected: f32) -> u16 {
    bf16_ordered_word(actual).abs_diff(bf16_ordered_word(expected))
}

fn max_abs_difference(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}

fn assert_hybrid_mla_explicit_selection_numerical_contract(
    label: &str,
    actual: &[f32],
    expected_f32: &[f32],
    expected_bf16: &[f32],
) {
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} produced non-finite output"
    );
    let max_abs_f32 = max_abs_difference(actual, expected_f32);
    let max_abs_bf16 = max_abs_difference(actual, expected_bf16);
    println!("{label}: max_abs_f32={max_abs_f32:e} max_abs_bf16={max_abs_bf16:e}");

    let scalar_oracle = cfg!(ferrule_cuda_test_oracle)
        && std::env::var("FERRULE_CUDA_HYBRID_MLA_EXPLICIT_SELECTION_TEST_ORACLE")
            .is_ok_and(|value| value == "1");
    let (expected, policy) = if scalar_oracle {
        (expected_f32, "test-only ordered F32")
    } else {
        (expected_bf16, "BF16 operands with FP32 accumulation")
    };
    let mismatch = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| {
            let error = (*actual - *expected).abs();
            error > 2.0e-5f32.max(expected.abs() * 2.0e-4)
        });
    assert!(
        mismatch.is_none(),
        "{label} violates the {policy} selected-attention contract: \
         max_abs_f32={max_abs_f32:e} max_abs_bf16={max_abs_bf16:e}; first mismatch={mismatch:?}"
    );
}

fn run_hybrid_mla_explicit_selection_case(case: HybridMlaExplicitSelectionHostCase<'_>) {
    assert_eq!(case.layout.heads, HYBRID_MLA_EXPLICIT_SELECTION_HEADS);
    assert_eq!(case.layout.head_dim, HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM);
    assert_eq!(
        case.query.len(),
        case.layout.rows
            * HYBRID_MLA_EXPLICIT_SELECTION_HEADS
            * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM
    );
    assert_eq!(
        case.expected_valid.len(),
        case.layout.rows * case.layout.selected_width
    );
    assert_eq!(
        case.expected_gathered.len(),
        case.expected_valid.len() * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM
    );

    let expected_f32 = hybrid_mla_explicit_selection_f32_scalar_reference(
        case.query,
        case.expected_gathered,
        case.expected_valid,
        case.attention_sink,
        case.layout.rows,
        case.layout.selected_width,
        case.layout.softmax_scale,
    );
    let expected_bf16 = hybrid_mla_explicit_selection_bf16_boundary_reference(
        case.query,
        case.expected_gathered,
        case.expected_valid,
        case.attention_sink,
        case.layout.rows,
        case.layout.selected_width,
        case.layout.softmax_scale,
    );

    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();
    let query = DeviceBuffer::from_host(&stream, case.query).expect("upload selected query");
    let first_plane =
        DeviceBuffer::from_host(&stream, case.first_plane).expect("upload selected first plane");
    let second_plane = case.second_plane.map(|values| {
        DeviceBuffer::from_host(&stream, values).expect("upload selected second plane")
    });
    let block_slots = case.block_slots.map(|values| {
        DeviceBuffer::from_host(&stream, values).expect("upload selected block slots")
    });
    let block_offsets = case.block_offsets.map(|values| {
        DeviceBuffer::from_host(&stream, values).expect("upload selected block offsets")
    });
    let sequence_kv_lens = case.sequence_kv_lens.map(|values| {
        DeviceBuffer::from_host(&stream, values).expect("upload selected sequence lengths")
    });
    let second_sequence_kv_lens = case.second_sequence_kv_lens.map(|values| {
        DeviceBuffer::from_host(&stream, values)
            .expect("upload selected second-plane sequence lengths")
    });
    let row_sequence_ids = case.row_sequence_ids.map(|values| {
        DeviceBuffer::from_host(&stream, values).expect("upload selected row sequence IDs")
    });
    let row_kv_lens = case.row_kv_lens.map(|values| {
        DeviceBuffer::from_host(&stream, values).expect("upload selected row KV lengths")
    });
    let row_second_kv_lens = case.row_second_kv_lens.map(|values| {
        DeviceBuffer::from_host(&stream, values)
            .expect("upload selected row second-plane KV lengths")
    });
    let selected_indices =
        DeviceBuffer::from_host(&stream, case.selected_indices).expect("upload selected indices");
    let selectors = case.selectors.map(|values| {
        DeviceBuffer::from_host(&stream, values).expect("upload selected plane selectors")
    });
    let attention_sink = DeviceBuffer::from_host(&stream, case.attention_sink)
        .expect("upload selected attention sink");
    let requirements = cutlass::hybrid_mla_explicit_selection_workspace_requirements(case.layout)
        .expect("query hybrid MLA explicit selection workspace");
    let workspace_bytes =
        usize::try_from(requirements.bytes).expect("workspace requirement fits usize");
    let mut workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_bytes)
        .expect("hybrid MLA explicit selection workspace");
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, case.query.len())
        .expect("hybrid MLA explicit selection output");
    #[cfg(ferrule_cuda_test_oracle)]
    let mut oracle_output = DeviceBuffer::<f32>::zeroed(&stream, case.query.len())
        .expect("selected attention oracle output");
    let mut status =
        DeviceBuffer::from_host(&stream, &[i32::MIN]).expect("selected attention status");

    let mut buffers = cutlass::HybridMlaExplicitSelectionBuffers {
        query: &query,
        #[cfg(ferrule_cuda_test_oracle)]
        oracle_output: &mut oracle_output,
        first_plane: &first_plane,
        second_plane: second_plane.as_ref(),
        block_slots: block_slots.as_ref(),
        block_offsets: block_offsets.as_ref(),
        sequence_kv_lens: sequence_kv_lens.as_ref(),
        second_sequence_kv_lens: second_sequence_kv_lens.as_ref(),
        row_sequence_ids: row_sequence_ids.as_ref(),
        row_kv_lens: row_kv_lens.as_ref(),
        row_second_kv_lens: row_second_kv_lens.as_ref(),
        selected_indices: &selected_indices,
        selectors: selectors.as_ref(),
        attention_sink: &attention_sink,
        workspace: &mut workspace,
        output: &mut output,
        status: &mut status,
    };
    cutlass::hybrid_mla_explicit_selection_launch(&stream, &mut buffers, case.layout)
        .expect("launch hybrid MLA explicit selection");

    assert_eq!(
        status.to_host_vec(&stream).expect("download device status"),
        [0],
        "{} status",
        case.label
    );

    let actual = output
        .to_host_vec(&stream)
        .expect("download hybrid MLA explicit selection output");
    assert_hybrid_mla_explicit_selection_numerical_contract(
        case.label,
        &actual,
        &expected_f32,
        &expected_bf16,
    );
}

fn explicit_selection_contiguous_gather(
    values: &[f32],
    selected_indices: &[i32],
    rows: usize,
    selected_width: usize,
) -> (Vec<f32>, Vec<i32>) {
    let kv_len = values.len() / HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
    let mut gathered = vec![0.0f32; rows * selected_width * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM];
    let mut valid = vec![0i32; rows * selected_width];
    for (selected_index, &logical) in selected_indices.iter().enumerate() {
        let Ok(token) = usize::try_from(logical) else {
            continue;
        };
        if token >= kv_len {
            continue;
        }
        valid[selected_index] = 1;
        let source = token * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
        let destination = selected_index * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
        gathered[destination..destination + HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM]
            .copy_from_slice(&values[source..source + HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM]);
    }
    (gathered, valid)
}

struct PagedGatherLayout {
    rows: usize,
    selected_width: usize,
    page_tokens: usize,
    first_elements_per_token: usize,
    second_elements_per_token: usize,
    layer_index: usize,
    layer_count: usize,
}

#[allow(clippy::too_many_arguments)]
fn explicit_selection_paged_gather(
    first_plane: &[f32],
    second_plane: Option<&[f32]>,
    block_slots: &[i32],
    block_offsets: &[i32],
    sequence_kv_lens: &[i32],
    second_sequence_kv_lens: Option<&[i32]>,
    row_sequence_ids: &[i32],
    row_kv_lens: &[i32],
    row_second_kv_lens: Option<&[i32]>,
    selected_indices: &[i32],
    selectors: Option<&[i32]>,
    layout: PagedGatherLayout,
) -> (Vec<f32>, Vec<i32>) {
    let mut gathered =
        vec![0.0f32; layout.rows * layout.selected_width * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM];
    let mut valid = vec![0i32; layout.rows * layout.selected_width];
    for row in 0..layout.rows {
        let sequence = usize::try_from(row_sequence_ids[row]).expect("valid row sequence ID");
        for selected in 0..layout.selected_width {
            let selected_index = row * layout.selected_width + selected;
            let Ok(logical) = usize::try_from(selected_indices[selected_index]) else {
                continue;
            };
            let selector = selectors.map_or(0, |values| values[selected_index]);
            let (plane, elements_per_token, sequence_len, visible) = match selector {
                0 => (
                    first_plane,
                    layout.first_elements_per_token,
                    usize::try_from(sequence_kv_lens[sequence]).expect("valid sequence len"),
                    usize::try_from(row_kv_lens[row]).expect("valid row KV len"),
                ),
                1 => (
                    second_plane.expect("selector one requires the second plane"),
                    layout.second_elements_per_token,
                    usize::try_from(
                        second_sequence_kv_lens
                            .expect("selector one requires second sequence lengths")[sequence],
                    )
                    .expect("valid second sequence len"),
                    usize::try_from(
                        row_second_kv_lens.expect("selector one requires second row lengths")[row],
                    )
                    .expect("valid second row KV len"),
                ),
                _ => continue,
            };
            if logical >= visible || logical >= sequence_len {
                continue;
            }
            let begin = usize::try_from(block_offsets[sequence]).expect("valid block begin");
            let end = usize::try_from(block_offsets[sequence + 1]).expect("valid block end");
            let block_entry = begin + logical / layout.page_tokens;
            assert!(block_entry < end, "complete page table for selected token");
            let slot = usize::try_from(block_slots[block_entry]).expect("valid physical slot");
            let layer_stride = layout.page_tokens * elements_per_token;
            let source = slot * layout.layer_count * layer_stride
                + layout.layer_index * layer_stride
                + logical % layout.page_tokens * elements_per_token;
            let destination = selected_index * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
            gathered[destination..destination + HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM]
                .copy_from_slice(&plane[source..source + HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM]);
            valid[selected_index] = 1;
        }
    }
    (gathered, valid)
}

fn explicit_selection_query(rows: usize) -> Vec<f32> {
    (0..rows * HYBRID_MLA_EXPLICIT_SELECTION_HEADS * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM)
        .map(|index| {
            let row = index
                / (HYBRID_MLA_EXPLICIT_SELECTION_HEADS * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM);
            let head = index / HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM
                % HYBRID_MLA_EXPLICIT_SELECTION_HEADS;
            let dimension = index % HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
            let centered = ((row * 19 + head * 7 + dimension * 11) % 31) as f32 - 15.0;
            centered * 0.0067 + row as f32 * 0.0013
        })
        .collect()
}

fn explicit_selection_attention_sink() -> Vec<f32> {
    (0..HYBRID_MLA_EXPLICIT_SELECTION_HEADS)
        .map(|head| -0.45 + head as f32 * 0.006)
        .collect()
}

fn hybrid_mla_explicit_selection_contiguous_stress_inputs(
    selected_width: usize,
) -> (Vec<f32>, Vec<f32>, Vec<i32>, Vec<f32>, Vec<f32>, Vec<i32>) {
    const ROWS: usize = 1;
    const KV_LEN: usize = 97;

    let query = explicit_selection_query(ROWS);
    let values = (0..KV_LEN * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM)
        .map(|index| {
            let token = index / HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
            let dimension = index % HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
            let centered = ((token * 43 + dimension * 17) % 67) as f32 - 33.0;
            centered * 0.0021 + token as f32 * 0.00037
        })
        .collect::<Vec<_>>();
    let selected_indices = (0..selected_width)
        .map(|selected| {
            if selected.is_multiple_of(19) {
                -1
            } else if selected.is_multiple_of(31) {
                KV_LEN as i32 + 7
            } else {
                ((selected * 37 + 11) % KV_LEN) as i32
            }
        })
        .collect::<Vec<_>>();
    let attention_sink = (0..HYBRID_MLA_EXPLICIT_SELECTION_HEADS)
        .map(|head| 3.75 + (head % 11) as f32 * 0.025)
        .collect::<Vec<_>>();
    let (gathered, valid) =
        explicit_selection_contiguous_gather(&values, &selected_indices, ROWS, selected_width);
    assert!(valid.iter().any(|&entry| entry == 0));
    assert!(valid.iter().any(|&entry| entry == 1));
    (
        query,
        values,
        selected_indices,
        attention_sink,
        gathered,
        valid,
    )
}

fn hybrid_mla_explicit_selection_policy_reference(
    query: &[f32],
    gathered: &[f32],
    valid: &[i32],
    attention_sink: &[f32],
    rows: usize,
    selected_width: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    let scalar_oracle = cfg!(ferrule_cuda_test_oracle)
        && std::env::var("FERRULE_CUDA_HYBRID_MLA_EXPLICIT_SELECTION_TEST_ORACLE")
            .is_ok_and(|value| value == "1");
    if scalar_oracle {
        hybrid_mla_explicit_selection_f32_scalar_reference(
            query,
            gathered,
            valid,
            attention_sink,
            rows,
            selected_width,
            softmax_scale,
        )
    } else {
        hybrid_mla_explicit_selection_bf16_boundary_reference(
            query,
            gathered,
            valid,
            attention_sink,
            rows,
            selected_width,
            softmax_scale,
        )
    }
}

fn assert_hybrid_mla_explicit_selection_policy_reference(
    label: &str,
    actual: &[f32],
    expected: &[f32],
) {
    assert_eq!(actual.len(), expected.len(), "{label} output length");
    let mut non_finite = 0usize;
    let mut mismatch_count = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut first_mismatch = None;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.is_finite() {
            non_finite += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((index, actual, expected));
            }
            continue;
        }
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
        max_abs = max_abs.max(absolute);
        max_rel = max_rel.max(relative);
        let is_bf16_boundary = actual.to_bits() & 0xffff == 0;
        if !is_bf16_boundary || bf16_ulp_distance(actual, expected) > 1 {
            mismatch_count += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((index, actual, expected));
            }
        }
    }
    println!(
        "{label}: values={} non_finite={} mismatches={} max_abs={max_abs:e} max_rel={max_rel:e}",
        actual.len(),
        non_finite,
        mismatch_count
    );
    assert!(
        non_finite == 0 && mismatch_count == 0,
        "{label} differs from the configured numerical-policy reference: non_finite={non_finite}/{} \
         mismatches={mismatch_count}/{} max_abs={max_abs:e} max_rel={max_rel:e} \
         first_mismatch={first_mismatch:?}; production output must be BF16 and within one \
         BF16 ULP of the scalar policy reference",
        actual.len(),
        actual.len()
    );
}

#[test]
fn cuda_hybrid_mla_explicit_selection_contiguous_production_widths_match_numerical_policy() {
    const ROWS: usize = 1;
    const KV_LEN: usize = 97;
    let softmax_scale = (HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM as f32)
        .sqrt()
        .recip();

    for selected_width in [128, 640] {
        let (query, values, selected_indices, attention_sink, gathered, expected_valid) =
            hybrid_mla_explicit_selection_contiguous_stress_inputs(selected_width);
        let expected = hybrid_mla_explicit_selection_policy_reference(
            &query,
            &gathered,
            &expected_valid,
            &attention_sink,
            ROWS,
            selected_width,
            softmax_scale,
        );

        let layout = cutlass::HybridMlaExplicitSelectionLayout {
            kind: cutlass::HybridMlaKvStorageKind::Contiguous,
            rows: ROWS,
            tokens_per_sequence: 0,
            kv_len: KV_LEN,
            heads: HYBRID_MLA_EXPLICIT_SELECTION_HEADS,
            head_dim: HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
            selected_width,
            page_tokens: 0,
            first_elements_per_token: 0,
            second_elements_per_token: 0,
            layer_index: 0,
            layer_count: 0,
            row_sequence_ids: false,
            row_kv_lens: false,
            softmax_scale,
        };
        let context = CudaContext::new(0).expect("CUDA context");
        context.bind_to_thread().expect("bind CUDA context");
        let stream = context.new_stream().expect("create CUDA stream");
        let query = DeviceBuffer::from_host(&stream, &query).expect("upload stress query");
        let values = DeviceBuffer::from_host(&stream, &values).expect("upload stress KV");
        let selected_indices = DeviceBuffer::from_host(&stream, &selected_indices)
            .expect("upload stress selected indices");
        let attention_sink = DeviceBuffer::from_host(&stream, &attention_sink)
            .expect("upload stress attention sink");
        let requirements = cutlass::hybrid_mla_explicit_selection_workspace_requirements(layout)
            .expect("query production-width explicit selection workspace");
        let mut workspace = DeviceBuffer::<u8>::zeroed(
            &stream,
            usize::try_from(requirements.bytes).expect("workspace bytes fit usize"),
        )
        .expect("production-width explicit selection workspace");
        let mut output = DeviceBuffer::<f32>::zeroed(
            &stream,
            HYBRID_MLA_EXPLICIT_SELECTION_HEADS * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
        )
        .expect("stress output");
        #[cfg(ferrule_cuda_test_oracle)]
        let mut oracle_output = DeviceBuffer::<f32>::zeroed(
            &stream,
            HYBRID_MLA_EXPLICIT_SELECTION_HEADS * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
        )
        .expect("stress oracle output");
        let mut status = DeviceBuffer::from_host(&stream, &[i32::MIN]).expect("stress status");
        let mut buffers = cutlass::HybridMlaExplicitSelectionBuffers {
            query: &query,
            #[cfg(ferrule_cuda_test_oracle)]
            oracle_output: &mut oracle_output,
            first_plane: &values,
            second_plane: None,
            block_slots: None,
            block_offsets: None,
            sequence_kv_lens: None,
            second_sequence_kv_lens: None,
            row_sequence_ids: None,
            row_kv_lens: None,
            row_second_kv_lens: None,
            selected_indices: &selected_indices,
            selectors: None,
            attention_sink: &attention_sink,
            workspace: &mut workspace,
            output: &mut output,
            status: &mut status,
        };
        cutlass::hybrid_mla_explicit_selection_launch(&stream, &mut buffers, layout)
            .expect("launch production-width hybrid MLA explicit selection attention");

        let label =
            format!("hybrid MLA explicit selection contiguous selected_width={selected_width}");
        assert_eq!(
            status.to_host_vec(&stream).expect("download stress status"),
            [0],
            "{label} status"
        );
        let actual = output.to_host_vec(&stream).expect("download stress output");
        assert_hybrid_mla_explicit_selection_policy_reference(&label, &actual, &expected);
    }
}

#[test]
#[ignore = "CUDA latency benchmark; run explicitly with --ignored --nocapture"]
fn cuda_hybrid_mla_explicit_selection_latency() {
    const ORACLE_ENVIRONMENT: [&str; 3] = [
        "FERRULE_CUDA_TEST_ORACLE",
        "FERRULE_CUDA_HYBRID_MLA_EXPLICIT_SELECTION_TEST_ORACLE",
        "FERRULE_CUDA_HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE",
    ];
    let configured_oracle_environment = ORACLE_ENVIRONMENT
        .iter()
        .copied()
        .filter(|name| std::env::var_os(name).is_some())
        .collect::<Vec<_>>();
    assert!(
        configured_oracle_environment.is_empty(),
        "production CUDA benchmark requires oracle environment variables to be unset; unset: {}",
        configured_oracle_environment.join(", ")
    );

    #[cfg(ferrule_cuda_test_oracle)]
    panic!(
        "production CUDA benchmark was compiled with ferrule_cuda_test_oracle; unset \
         FERRULE_CUDA_TEST_ORACLE and rebuild before benchmarking"
    );

    #[cfg(not(ferrule_cuda_test_oracle))]
    {
        const KV_LEN: usize = 97;
        const DEFAULT_WARMUP_ITERATIONS: usize = 20;
        const DEFAULT_TIMED_ITERATIONS: usize = 200;
        const LAUNCHES_PER_ITERATION: usize = 4;

        fn bench_values(name: &str, defaults: &[usize]) -> Vec<usize> {
            std::env::var(name).map_or_else(
                |_| defaults.to_vec(),
                |value| {
                    value
                        .split(',')
                        .map(|entry| {
                            entry.trim().parse::<usize>().unwrap_or_else(|error| {
                                panic!("invalid {name} entry `{entry}`: {error}")
                            })
                        })
                        .collect()
                },
            )
        }

        fn bench_count(name: &str, default: usize) -> usize {
            std::env::var(name).map_or(default, |value| {
                value
                    .parse::<usize>()
                    .unwrap_or_else(|error| panic!("invalid {name} value `{value}`: {error}"))
            })
        }

        let benchmark_rows = bench_values("FERRULE_CUDA_BENCH_ROWS", &[1, 2, 4, 8]);
        let benchmark_widths = bench_values("FERRULE_CUDA_BENCH_WIDTHS", &[128, 640]);
        let warmup_iterations = bench_count(
            "FERRULE_CUDA_BENCH_WARMUP_ITERATIONS",
            DEFAULT_WARMUP_ITERATIONS,
        );
        let timed_iterations = bench_count(
            "FERRULE_CUDA_BENCH_TIMED_ITERATIONS",
            DEFAULT_TIMED_ITERATIONS,
        );
        assert!(
            timed_iterations > 0,
            "timed benchmark iterations must be non-zero"
        );

        let softmax_scale = (HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM as f32)
            .sqrt()
            .recip();
        let context = CudaContext::new(0).expect("CUDA context");
        context.bind_to_thread().expect("bind CUDA context");
        let stream = context.new_stream().expect("create CUDA benchmark stream");

        for rows in benchmark_rows {
            for &selected_width in &benchmark_widths {
                let (
                    _,
                    values_host,
                    selected_pattern,
                    attention_sink_host,
                    gathered_pattern,
                    valid_pattern,
                ) = hybrid_mla_explicit_selection_contiguous_stress_inputs(selected_width);
                let query_host = explicit_selection_query(rows);
                let selected_indices_host = selected_pattern.repeat(rows);
                let gathered_host = gathered_pattern.repeat(rows);
                let valid_host = valid_pattern.repeat(rows);
                let reference_host = hybrid_mla_explicit_selection_policy_reference(
                    &query_host,
                    &gathered_host,
                    &valid_host,
                    &attention_sink_host,
                    rows,
                    selected_width,
                    softmax_scale,
                );
                let layout = cutlass::HybridMlaExplicitSelectionLayout {
                    kind: cutlass::HybridMlaKvStorageKind::Contiguous,
                    rows,
                    tokens_per_sequence: 0,
                    kv_len: KV_LEN,
                    heads: HYBRID_MLA_EXPLICIT_SELECTION_HEADS,
                    head_dim: HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
                    selected_width,
                    page_tokens: 0,
                    first_elements_per_token: 0,
                    second_elements_per_token: 0,
                    layer_index: 0,
                    layer_count: 0,
                    row_sequence_ids: false,
                    row_kv_lens: false,
                    softmax_scale,
                };

                let query =
                    DeviceBuffer::from_host(&stream, &query_host).expect("upload benchmark query");
                let values =
                    DeviceBuffer::from_host(&stream, &values_host).expect("upload benchmark KV");
                let selected_indices = DeviceBuffer::from_host(&stream, &selected_indices_host)
                    .expect("upload benchmark selected indices");
                let attention_sink = DeviceBuffer::from_host(&stream, &attention_sink_host)
                    .expect("upload benchmark attention sink");
                let requirements =
                    cutlass::hybrid_mla_explicit_selection_workspace_requirements(layout)
                        .expect("query benchmark explicit selection workspace");
                let mut workspace = DeviceBuffer::<u8>::zeroed(
                    &stream,
                    usize::try_from(requirements.bytes).expect("workspace bytes fit usize"),
                )
                .expect("allocate benchmark explicit selection workspace");
                let mut output = DeviceBuffer::<f32>::zeroed(
                    &stream,
                    rows * HYBRID_MLA_EXPLICIT_SELECTION_HEADS
                        * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
                )
                .expect("allocate benchmark output");
                let mut status = DeviceBuffer::from_host(&stream, &[i32::MIN])
                    .expect("allocate benchmark status");
                let start = context.new_event(true).expect("create CUDA start event");
                let end = context.new_event(true).expect("create CUDA end event");
                let mut buffers = cutlass::HybridMlaExplicitSelectionBuffers {
                    query: &query,
                    first_plane: &values,
                    second_plane: None,
                    block_slots: None,
                    block_offsets: None,
                    sequence_kv_lens: None,
                    second_sequence_kv_lens: None,
                    row_sequence_ids: None,
                    row_kv_lens: None,
                    row_second_kv_lens: None,
                    selected_indices: &selected_indices,
                    selectors: None,
                    attention_sink: &attention_sink,
                    workspace: &mut workspace,
                    output: &mut output,
                    status: &mut status,
                };

                for warmup in 0..warmup_iterations {
                    cutlass::hybrid_mla_explicit_selection_launch(&stream, &mut buffers, layout)
                        .unwrap_or_else(|error| {
                            panic!(
                                "warmup explicit-selection launch {warmup}/{warmup_iterations} \
                                 rows={rows} selected_width={selected_width}: {error}"
                            )
                        });
                }
                stream
                    .synchronize()
                    .expect("synchronize CUDA explicit-selection warmup");

                start.record(&stream).expect("record CUDA start event");
                for iteration in 0..timed_iterations {
                    cutlass::hybrid_mla_explicit_selection_launch(&stream, &mut buffers, layout)
                        .unwrap_or_else(|error| {
                            panic!(
                                "timed explicit-selection launch {iteration}/{timed_iterations} \
                                 rows={rows} selected_width={selected_width}: {error}"
                            )
                        });
                }
                end.record(&stream).expect("record CUDA end event");
                end.synchronize().expect("synchronize CUDA end event");
                let total_ms = f64::from(
                    start
                        .elapsed(&end)
                        .expect("measure CUDA explicit-selection events"),
                );
                assert!(total_ms > 0.0, "CUDA event interval must be positive");
                drop(buffers);

                let label = format!(
                    "hybrid MLA explicit selection latency rows={rows} \
                     selected_width={selected_width}"
                );
                assert_eq!(
                    status
                        .to_host_vec(&stream)
                        .expect("download benchmark status"),
                    [0],
                    "{label} status"
                );
                let actual = output
                    .to_host_vec(&stream)
                    .expect("download benchmark output");
                assert_hybrid_mla_explicit_selection_policy_reference(
                    &label,
                    &actual,
                    &reference_host,
                );

                let total_seconds = total_ms / 1_000.0;
                let avg_us = total_ms * 1_000.0 / timed_iterations as f64;
                let rows_per_s = (rows * timed_iterations) as f64 / total_seconds;
                let logical_fma_per_iteration = 2
                    * rows
                    * HYBRID_MLA_EXPLICIT_SELECTION_HEADS
                    * selected_width
                    * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
                let logical_fma_per_s =
                    logical_fma_per_iteration as f64 * timed_iterations as f64 / total_seconds;
                println!(
                    "kind=contiguous rows={rows} heads={} head_dim={} \
                     selected_width={selected_width} warmup_iterations={warmup_iterations} \
                     timed_iterations={timed_iterations} avg_us={avg_us:.3} \
                     rows/s={rows_per_s:.3} logical_fma/s={logical_fma_per_s:.6e} \
                     launches={LAUNCHES_PER_ITERATION}",
                    HYBRID_MLA_EXPLICIT_SELECTION_HEADS, HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
                );
            }
        }
    }
}

#[test]
fn cuda_hybrid_mla_explicit_selection_reuses_workspace_across_43_async_launches() {
    const ROWS: usize = 1;
    const KV_LEN: usize = 97;
    const MAX_SELECTED_WIDTH: usize = 640;
    const LAUNCHES: usize = 43;
    let softmax_scale = (HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM as f32)
        .sqrt()
        .recip();
    let (
        query_host,
        values_host,
        selected_indices_host,
        attention_sink_host,
        gathered,
        expected_valid,
    ) = hybrid_mla_explicit_selection_contiguous_stress_inputs(MAX_SELECTED_WIDTH);
    let expected = hybrid_mla_explicit_selection_policy_reference(
        &query_host,
        &gathered,
        &expected_valid,
        &attention_sink_host,
        ROWS,
        MAX_SELECTED_WIDTH,
        softmax_scale,
    );
    let maximum_layout = cutlass::HybridMlaExplicitSelectionLayout {
        kind: cutlass::HybridMlaKvStorageKind::Contiguous,
        rows: ROWS,
        tokens_per_sequence: 0,
        kv_len: KV_LEN,
        heads: HYBRID_MLA_EXPLICIT_SELECTION_HEADS,
        head_dim: HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
        selected_width: MAX_SELECTED_WIDTH,
        page_tokens: 0,
        first_elements_per_token: 0,
        second_elements_per_token: 0,
        layer_index: 0,
        layer_count: 0,
        row_sequence_ids: false,
        row_kv_lens: false,
        softmax_scale,
    };

    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.new_stream().expect("create CUDA stream");
    let query = DeviceBuffer::from_host(&stream, &query_host).expect("upload repeated query");
    let values = DeviceBuffer::from_host(&stream, &values_host).expect("upload repeated KV");
    let selected_indices = DeviceBuffer::from_host(&stream, &selected_indices_host)
        .expect("upload repeated selected indices");
    let attention_sink = DeviceBuffer::from_host(&stream, &attention_sink_host)
        .expect("upload repeated attention sink");
    let requirements =
        cutlass::hybrid_mla_explicit_selection_workspace_requirements(maximum_layout)
            .expect("query repeated explicit selection workspace");
    let mut workspace = DeviceBuffer::<u8>::zeroed(
        &stream,
        usize::try_from(requirements.bytes).expect("workspace bytes fit usize"),
    )
    .expect("repeated explicit selection workspace");
    let mut output = DeviceBuffer::<f32>::zeroed(
        &stream,
        HYBRID_MLA_EXPLICIT_SELECTION_HEADS * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
    )
    .expect("repeated output");
    #[cfg(ferrule_cuda_test_oracle)]
    let mut oracle_output = DeviceBuffer::<f32>::zeroed(
        &stream,
        HYBRID_MLA_EXPLICIT_SELECTION_HEADS * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
    )
    .expect("repeated oracle output");
    let mut status = DeviceBuffer::from_host(&stream, &[i32::MIN]).expect("repeated device status");

    for launch in 0..LAUNCHES {
        let selected_width = if launch.is_multiple_of(21) {
            MAX_SELECTED_WIDTH
        } else {
            128
        };
        let selected_indices_view = selected_indices
            .slice(0, selected_width)
            .expect("selected-index prefix view");
        let mut buffers = cutlass::HybridMlaExplicitSelectionBuffers {
            query: &query,
            #[cfg(ferrule_cuda_test_oracle)]
            oracle_output: &mut oracle_output,
            first_plane: &values,
            second_plane: None,
            block_slots: None,
            block_offsets: None,
            sequence_kv_lens: None,
            second_sequence_kv_lens: None,
            row_sequence_ids: None,
            row_kv_lens: None,
            row_second_kv_lens: None,
            selected_indices: &selected_indices_view,
            selectors: None,
            attention_sink: &attention_sink,
            workspace: &mut workspace,
            output: &mut output,
            status: &mut status,
        };
        cutlass::hybrid_mla_explicit_selection_launch(
            &stream,
            &mut buffers,
            cutlass::HybridMlaExplicitSelectionLayout {
                kind: cutlass::HybridMlaKvStorageKind::Contiguous,
                rows: ROWS,
                tokens_per_sequence: 0,
                kv_len: KV_LEN,
                heads: HYBRID_MLA_EXPLICIT_SELECTION_HEADS,
                head_dim: HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
                selected_width,
                page_tokens: 0,
                first_elements_per_token: 0,
                second_elements_per_token: 0,
                layer_index: 0,
                layer_count: 0,
                row_sequence_ids: false,
                row_kv_lens: false,
                softmax_scale,
            },
        )
        .unwrap_or_else(|error| {
            panic!("asynchronous hybrid MLA explicit selection launch {launch}: {error}")
        });
    }

    assert_eq!(
        status
            .to_host_vec(&stream)
            .expect("download repeated device status"),
        [0],
        "final status after {LAUNCHES} queued launches"
    );
    let actual = output
        .to_host_vec(&stream)
        .expect("download repeated output");
    assert_hybrid_mla_explicit_selection_policy_reference(
        "hybrid MLA explicit selection workspace reuse after 43 asynchronous launches",
        &actual,
        &expected,
    );
}

#[test]
fn cuda_hybrid_mla_explicit_selection_multi_row_paged_workspace_stress() {
    const PAGED_ROWS: usize = 5;
    const PAGED_WIDTH: usize = 128;
    const DUAL_ROWS: usize = 8;
    const DUAL_WIDTH: usize = 640;
    const PAGE_TOKENS: usize = 16;
    const FIRST_ELEMENTS_PER_TOKEN: usize = 528;
    const SECOND_ELEMENTS_PER_TOKEN: usize = 560;
    const LAYER_INDEX: usize = 2;
    const LAYER_COUNT: usize = 4;
    const PHYSICAL_SLOTS: usize = 12;
    const LAUNCHES: usize = 43;

    let block_slots = [7, 1, 9, 2, 10, 4, 6, 11, 0, 8];
    let block_offsets = [0, 3, 7, 10];
    let sequence_kv_lens = [35, 53, 47];
    let paged_row_sequence_ids = [2, 0, 1, 2, 0];
    let paged_row_kv_lens = [41, 17, 49, 13, 30];
    let dual_row_sequence_ids = [1, 2, 0, 1, 0, 2, 1, 2];
    let dual_row_kv_lens = [52, 33, 18, 27, 34, 46, 11, 29];

    fn selected_indices(
        row_sequence_ids: &[i32],
        row_kv_lens: &[i32],
        sequence_kv_lens: &[i32],
        selected_width: usize,
    ) -> Vec<i32> {
        let mut indices = Vec::with_capacity(row_sequence_ids.len() * selected_width);
        for (row, (&sequence, &visible)) in row_sequence_ids.iter().zip(row_kv_lens).enumerate() {
            let sequence_len = sequence_kv_lens[sequence as usize];
            let repeated = (row as i32 * 3 + 5) % visible.min(sequence_len);
            for selected in 0..selected_width {
                let index = match selected % 41 {
                    0 => -1,
                    1 => sequence_len + 9,
                    2 | 3 => repeated,
                    4 => visible,
                    5 => i32::MAX,
                    _ => ((selected * 29 + row * 11) % sequence_len as usize) as i32,
                };
                indices.push(index);
            }
        }
        indices
    }

    let paged_selected_indices = selected_indices(
        &paged_row_sequence_ids,
        &paged_row_kv_lens,
        &sequence_kv_lens,
        PAGED_WIDTH,
    );
    let dual_selected_indices = selected_indices(
        &dual_row_sequence_ids,
        &dual_row_kv_lens,
        &sequence_kv_lens,
        DUAL_WIDTH,
    );
    let dual_selectors = (0..DUAL_ROWS * DUAL_WIDTH)
        .map(|index| match index % 17 {
            0 => 2,
            1 => -1,
            2 | 5 | 8 | 11 | 14 => 1,
            _ => 0,
        })
        .collect::<Vec<_>>();

    let mut first_plane =
        vec![901.0f32; PHYSICAL_SLOTS * LAYER_COUNT * PAGE_TOKENS * FIRST_ELEMENTS_PER_TOKEN];
    let mut second_plane =
        vec![-809.0f32; PHYSICAL_SLOTS * LAYER_COUNT * PAGE_TOKENS * SECOND_ELEMENTS_PER_TOKEN];
    for (sequence, &sequence_len) in sequence_kv_lens.iter().enumerate() {
        for logical in 0..sequence_len as usize {
            let entry = block_offsets[sequence] as usize + logical / PAGE_TOKENS;
            let slot = block_slots[entry] as usize;
            let first_base = slot * LAYER_COUNT * PAGE_TOKENS * FIRST_ELEMENTS_PER_TOKEN
                + LAYER_INDEX * PAGE_TOKENS * FIRST_ELEMENTS_PER_TOKEN
                + logical % PAGE_TOKENS * FIRST_ELEMENTS_PER_TOKEN;
            let second_base = slot * LAYER_COUNT * PAGE_TOKENS * SECOND_ELEMENTS_PER_TOKEN
                + LAYER_INDEX * PAGE_TOKENS * SECOND_ELEMENTS_PER_TOKEN
                + logical % PAGE_TOKENS * SECOND_ELEMENTS_PER_TOKEN;
            for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                let first_centered =
                    ((sequence * 47 + logical * 23 + dimension * 7) % 71) as f32 - 35.0;
                first_plane[first_base + dimension] =
                    first_centered * 0.0023 + sequence as f32 * 0.013 + logical as f32 * 0.0007;
                let second_centered =
                    ((sequence * 31 + logical * 17 + dimension * 13) % 79) as f32 - 39.0;
                second_plane[second_base + dimension] =
                    second_centered * 0.0019 - sequence as f32 * 0.011 + logical as f32 * 0.0009;
            }
        }
    }

    let paged_gather_layout = PagedGatherLayout {
        rows: PAGED_ROWS,
        selected_width: PAGED_WIDTH,
        page_tokens: PAGE_TOKENS,
        first_elements_per_token: FIRST_ELEMENTS_PER_TOKEN,
        second_elements_per_token: 0,
        layer_index: LAYER_INDEX,
        layer_count: LAYER_COUNT,
    };
    let (paged_gathered, paged_valid) = explicit_selection_paged_gather(
        &first_plane,
        None,
        &block_slots,
        &block_offsets,
        &sequence_kv_lens,
        None,
        &paged_row_sequence_ids,
        &paged_row_kv_lens,
        None,
        &paged_selected_indices,
        None,
        paged_gather_layout,
    );
    let dual_gather_layout = PagedGatherLayout {
        rows: DUAL_ROWS,
        selected_width: DUAL_WIDTH,
        page_tokens: PAGE_TOKENS,
        first_elements_per_token: FIRST_ELEMENTS_PER_TOKEN,
        second_elements_per_token: SECOND_ELEMENTS_PER_TOKEN,
        layer_index: LAYER_INDEX,
        layer_count: LAYER_COUNT,
    };
    let (dual_gathered, dual_valid) = explicit_selection_paged_gather(
        &first_plane,
        Some(&second_plane),
        &block_slots,
        &block_offsets,
        &sequence_kv_lens,
        Some(&sequence_kv_lens),
        &dual_row_sequence_ids,
        &dual_row_kv_lens,
        Some(&dual_row_kv_lens),
        &dual_selected_indices,
        Some(&dual_selectors),
        dual_gather_layout,
    );
    assert!(paged_valid.iter().any(|&valid| valid == 0));
    assert!(paged_valid.iter().any(|&valid| valid == 1));
    assert!(dual_valid.iter().any(|&valid| valid == 0));
    assert!(dual_valid.iter().any(|&valid| valid == 1));
    assert!(dual_selectors.iter().any(|&selector| selector == 0));
    assert!(dual_selectors.iter().any(|&selector| selector == 1));

    let paged_query_host = explicit_selection_query(PAGED_ROWS);
    let dual_query_host = explicit_selection_query(DUAL_ROWS);
    let attention_sink_host = explicit_selection_attention_sink();
    let softmax_scale = (HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM as f32)
        .sqrt()
        .recip();
    let paged_expected = hybrid_mla_explicit_selection_policy_reference(
        &paged_query_host,
        &paged_gathered,
        &paged_valid,
        &attention_sink_host,
        PAGED_ROWS,
        PAGED_WIDTH,
        softmax_scale,
    );
    let dual_expected = hybrid_mla_explicit_selection_policy_reference(
        &dual_query_host,
        &dual_gathered,
        &dual_valid,
        &attention_sink_host,
        DUAL_ROWS,
        DUAL_WIDTH,
        softmax_scale,
    );

    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.new_stream().expect("create CUDA stream");
    let paged_query =
        DeviceBuffer::from_host(&stream, &paged_query_host).expect("upload paged stress query");
    let dual_query =
        DeviceBuffer::from_host(&stream, &dual_query_host).expect("upload dual-paged stress query");
    let first_plane =
        DeviceBuffer::from_host(&stream, &first_plane).expect("upload first paged stress plane");
    let second_plane =
        DeviceBuffer::from_host(&stream, &second_plane).expect("upload second paged stress plane");
    let block_slots =
        DeviceBuffer::from_host(&stream, &block_slots).expect("upload stress block slots");
    let block_offsets =
        DeviceBuffer::from_host(&stream, &block_offsets).expect("upload stress block offsets");
    let second_sequence_kv_lens = DeviceBuffer::from_host(&stream, &sequence_kv_lens)
        .expect("upload stress second-plane sequence lengths");
    let sequence_kv_lens = DeviceBuffer::from_host(&stream, &sequence_kv_lens)
        .expect("upload stress sequence lengths");
    let paged_row_sequence_ids = DeviceBuffer::from_host(&stream, &paged_row_sequence_ids)
        .expect("upload paged row sequence IDs");
    let paged_row_kv_lens =
        DeviceBuffer::from_host(&stream, &paged_row_kv_lens).expect("upload paged row visibility");
    let dual_row_sequence_ids = DeviceBuffer::from_host(&stream, &dual_row_sequence_ids)
        .expect("upload dual-paged row sequence IDs");
    let dual_row_second_kv_lens = DeviceBuffer::from_host(&stream, &dual_row_kv_lens)
        .expect("upload dual-paged second-plane row visibility");
    let dual_row_kv_lens = DeviceBuffer::from_host(&stream, &dual_row_kv_lens)
        .expect("upload dual-paged row visibility");
    let paged_selected_indices = DeviceBuffer::from_host(&stream, &paged_selected_indices)
        .expect("upload paged selected indices");
    let dual_selected_indices = DeviceBuffer::from_host(&stream, &dual_selected_indices)
        .expect("upload dual-paged selected indices");
    let dual_selectors =
        DeviceBuffer::from_host(&stream, &dual_selectors).expect("upload dual-paged selectors");
    let attention_sink = DeviceBuffer::from_host(&stream, &attention_sink_host)
        .expect("upload stress attention sink");

    let paged_layout = cutlass::HybridMlaExplicitSelectionLayout {
        kind: cutlass::HybridMlaKvStorageKind::Paged,
        rows: PAGED_ROWS,
        tokens_per_sequence: 0,
        kv_len: 0,
        heads: HYBRID_MLA_EXPLICIT_SELECTION_HEADS,
        head_dim: HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
        selected_width: PAGED_WIDTH,
        page_tokens: PAGE_TOKENS,
        first_elements_per_token: FIRST_ELEMENTS_PER_TOKEN,
        second_elements_per_token: 0,
        layer_index: LAYER_INDEX,
        layer_count: LAYER_COUNT,
        row_sequence_ids: true,
        row_kv_lens: true,
        softmax_scale,
    };
    let dual_layout = cutlass::HybridMlaExplicitSelectionLayout {
        kind: cutlass::HybridMlaKvStorageKind::DualPaged,
        rows: DUAL_ROWS,
        tokens_per_sequence: 0,
        kv_len: 0,
        heads: HYBRID_MLA_EXPLICIT_SELECTION_HEADS,
        head_dim: HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
        selected_width: DUAL_WIDTH,
        page_tokens: PAGE_TOKENS,
        first_elements_per_token: FIRST_ELEMENTS_PER_TOKEN,
        second_elements_per_token: SECOND_ELEMENTS_PER_TOKEN,
        layer_index: LAYER_INDEX,
        layer_count: LAYER_COUNT,
        row_sequence_ids: true,
        row_kv_lens: true,
        softmax_scale,
    };
    let paged_requirements =
        cutlass::hybrid_mla_explicit_selection_workspace_requirements(paged_layout)
            .expect("query paged explicit selection workspace");
    let dual_requirements =
        cutlass::hybrid_mla_explicit_selection_workspace_requirements(dual_layout)
            .expect("query dual-paged explicit selection workspace");
    let workspace_bytes = paged_requirements.bytes.max(dual_requirements.bytes);
    let mut workspace = DeviceBuffer::<u8>::zeroed(
        &stream,
        usize::try_from(workspace_bytes).expect("workspace bytes fit usize"),
    )
    .expect("allocate shared explicit selection workspace");
    let maximum_output =
        DUAL_ROWS * HYBRID_MLA_EXPLICIT_SELECTION_HEADS * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
    let output = DeviceBuffer::<f32>::zeroed(&stream, maximum_output)
        .expect("allocate shared output storage");
    #[cfg(ferrule_cuda_test_oracle)]
    let oracle_output = DeviceBuffer::<f32>::zeroed(&stream, maximum_output)
        .expect("allocate shared oracle output storage");
    let mut status = DeviceBuffer::from_host(&stream, &[i32::MIN]).expect("stress device status");
    let mut deterministic_bits: [Option<Vec<u32>>; 2] = [None, None];

    for launch in 0..LAUNCHES {
        let dual = launch % 2 == 1;
        let (
            shape_index,
            shape,
            query,
            second_plane,
            second_sequence_kv_lens,
            row_sequence_ids,
            row_kv_lens,
            row_second_kv_lens,
            selected_indices,
            selectors,
            expected_output,
            layout,
        ) = if dual {
            (
                1,
                "kind=dual-paged rows=8 heads=64 dim=512 width=640 page_tokens=16 layer=2/4 first_stride=528 second_stride=560",
                &dual_query,
                Some(&second_plane),
                Some(&second_sequence_kv_lens),
                &dual_row_sequence_ids,
                &dual_row_kv_lens,
                Some(&dual_row_second_kv_lens),
                &dual_selected_indices,
                Some(&dual_selectors),
                dual_expected.as_slice(),
                dual_layout,
            )
        } else {
            (
                0,
                "kind=paged rows=5 heads=64 dim=512 width=128 page_tokens=16 layer=2/4 first_stride=528 second_stride=0",
                &paged_query,
                None,
                None,
                &paged_row_sequence_ids,
                &paged_row_kv_lens,
                None,
                &paged_selected_indices,
                None,
                paged_expected.as_slice(),
                paged_layout,
            )
        };
        let label =
            format!("hybrid MLA explicit selection stress launch={launch}/{LAUNCHES} {shape}");
        let output_values = layout.rows
            * HYBRID_MLA_EXPLICIT_SELECTION_HEADS
            * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
        let mut output_view = output
            .slice(0, output_values)
            .unwrap_or_else(|error| panic!("{label}: output view: {error}"));
        #[cfg(ferrule_cuda_test_oracle)]
        let mut oracle_output_view = oracle_output
            .slice(0, output_values)
            .unwrap_or_else(|error| panic!("{label}: oracle output view: {error}"));
        let mut buffers = cutlass::HybridMlaExplicitSelectionBuffers {
            query,
            #[cfg(ferrule_cuda_test_oracle)]
            oracle_output: &mut oracle_output_view,
            first_plane: &first_plane,
            second_plane,
            block_slots: Some(&block_slots),
            block_offsets: Some(&block_offsets),
            sequence_kv_lens: Some(&sequence_kv_lens),
            second_sequence_kv_lens,
            row_sequence_ids: Some(row_sequence_ids),
            row_kv_lens: Some(row_kv_lens),
            row_second_kv_lens,
            selected_indices,
            selectors,
            attention_sink: &attention_sink,
            workspace: &mut workspace,
            output: &mut output_view,
            status: &mut status,
        };
        cutlass::hybrid_mla_explicit_selection_launch(&stream, &mut buffers, layout)
            .unwrap_or_else(|error| panic!("{label}: launch failed: {error}"));
        drop(buffers);

        assert_eq!(
            status
                .to_host_vec(&stream)
                .unwrap_or_else(|error| panic!("{label}: download status: {error}")),
            [0],
            "{label}: device status"
        );

        let actual = output_view
            .to_host_vec(&stream)
            .unwrap_or_else(|error| panic!("{label}: download output: {error}"));
        assert_hybrid_mla_explicit_selection_policy_reference(&label, &actual, expected_output);

        let actual_bits = actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        if let Some(previous_bits) = &deterministic_bits[shape_index] {
            assert_eq!(
                &actual_bits, previous_bits,
                "{label}: nondeterministic output relative to the first launch of this shape"
            );
        } else {
            deterministic_bits[shape_index] = Some(actual_bits);
        }
    }
}

#[test]
fn cuda_hybrid_mla_explicit_selection_contiguous_matches_f32_semantics() {
    const ROWS: usize = 1;
    const KV_LEN: usize = 5;
    const SELECTED_WIDTH: usize = 5;
    let query = explicit_selection_query(ROWS);
    let values = (0..KV_LEN * HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM)
        .map(|index| {
            let token = index / HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
            let dimension = index % HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM;
            let centered = ((token * 13 + dimension * 7) % 29) as f32 - 14.0;
            centered * 0.0043 + token as f32 * 0.0127
        })
        .collect::<Vec<_>>();
    let selected_indices = [3, -1, 0, KV_LEN as i32, 4];
    let (gathered, valid) =
        explicit_selection_contiguous_gather(&values, &selected_indices, ROWS, SELECTED_WIDTH);
    let attention_sink = explicit_selection_attention_sink();
    let softmax_scale = (HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM as f32)
        .sqrt()
        .recip();
    run_hybrid_mla_explicit_selection_case(HybridMlaExplicitSelectionHostCase {
        label: "hybrid MLA explicit selection contiguous",
        query: &query,
        first_plane: &values,
        second_plane: None,
        block_slots: None,
        block_offsets: None,
        sequence_kv_lens: None,
        second_sequence_kv_lens: None,
        row_sequence_ids: None,
        row_kv_lens: None,
        row_second_kv_lens: None,
        selected_indices: &selected_indices,
        selectors: None,
        attention_sink: &attention_sink,
        expected_gathered: &gathered,
        expected_valid: &valid,
        layout: cutlass::HybridMlaExplicitSelectionLayout {
            kind: cutlass::HybridMlaKvStorageKind::Contiguous,
            rows: ROWS,
            tokens_per_sequence: 0,
            kv_len: KV_LEN,
            heads: HYBRID_MLA_EXPLICIT_SELECTION_HEADS,
            head_dim: HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
            selected_width: SELECTED_WIDTH,
            page_tokens: 0,
            first_elements_per_token: 0,
            second_elements_per_token: 0,
            layer_index: 0,
            layer_count: 0,
            row_sequence_ids: false,
            row_kv_lens: false,
            softmax_scale,
        },
    });
}

#[test]
fn cuda_hybrid_mla_explicit_selection_paged_row_metadata_matches_f32_semantics() {
    const ROWS: usize = 2;
    const SELECTED_WIDTH: usize = 5;
    const PAGE_TOKENS: usize = 4;
    const ELEMENTS_PER_TOKEN: usize = 520;
    const LAYER_INDEX: usize = 1;
    const LAYER_COUNT: usize = 3;
    const PHYSICAL_SLOTS: usize = 4;
    let block_slots = [2, 0, 3, 1];
    let block_offsets = [0, 2, 4];
    let sequence_kv_lens = [7, 6];
    let row_sequence_ids = [1, 0];
    let row_kv_lens = [6, 5];
    let selected_indices = [3, 4, -1, 5, 6, 4, 0, 7, 3, 5];
    let mut plane = vec![901.0f32; PHYSICAL_SLOTS * LAYER_COUNT * PAGE_TOKENS * ELEMENTS_PER_TOKEN];
    for (sequence, &sequence_len) in sequence_kv_lens.iter().enumerate() {
        for logical in 0..sequence_len as usize {
            let entry = block_offsets[sequence] as usize + logical / PAGE_TOKENS;
            let slot = block_slots[entry] as usize;
            let base = slot * LAYER_COUNT * PAGE_TOKENS * ELEMENTS_PER_TOKEN
                + LAYER_INDEX * PAGE_TOKENS * ELEMENTS_PER_TOKEN
                + logical % PAGE_TOKENS * ELEMENTS_PER_TOKEN;
            for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                let centered = ((sequence * 41 + logical * 17 + dimension * 5) % 47) as f32 - 23.0;
                plane[base + dimension] =
                    centered * 0.0031 + sequence as f32 * 0.021 + logical as f32 * 0.004;
            }
        }
    }
    let gather_layout = PagedGatherLayout {
        rows: ROWS,
        selected_width: SELECTED_WIDTH,
        page_tokens: PAGE_TOKENS,
        first_elements_per_token: ELEMENTS_PER_TOKEN,
        second_elements_per_token: 0,
        layer_index: LAYER_INDEX,
        layer_count: LAYER_COUNT,
    };
    let (gathered, valid) = explicit_selection_paged_gather(
        &plane,
        None,
        &block_slots,
        &block_offsets,
        &sequence_kv_lens,
        None,
        &row_sequence_ids,
        &row_kv_lens,
        None,
        &selected_indices,
        None,
        gather_layout,
    );
    assert_eq!(valid, [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]);
    let query = explicit_selection_query(ROWS);
    let attention_sink = explicit_selection_attention_sink();
    let softmax_scale = (HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM as f32)
        .sqrt()
        .recip();
    run_hybrid_mla_explicit_selection_case(HybridMlaExplicitSelectionHostCase {
        label: "hybrid MLA explicit selection paged row metadata",
        query: &query,
        first_plane: &plane,
        second_plane: None,
        block_slots: Some(&block_slots),
        block_offsets: Some(&block_offsets),
        sequence_kv_lens: Some(&sequence_kv_lens),
        second_sequence_kv_lens: None,
        row_sequence_ids: Some(&row_sequence_ids),
        row_kv_lens: Some(&row_kv_lens),
        row_second_kv_lens: None,
        selected_indices: &selected_indices,
        selectors: None,
        attention_sink: &attention_sink,
        expected_gathered: &gathered,
        expected_valid: &valid,
        layout: cutlass::HybridMlaExplicitSelectionLayout {
            kind: cutlass::HybridMlaKvStorageKind::Paged,
            rows: ROWS,
            tokens_per_sequence: 0,
            kv_len: 0,
            heads: HYBRID_MLA_EXPLICIT_SELECTION_HEADS,
            head_dim: HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
            selected_width: SELECTED_WIDTH,
            page_tokens: PAGE_TOKENS,
            first_elements_per_token: ELEMENTS_PER_TOKEN,
            second_elements_per_token: 0,
            layer_index: LAYER_INDEX,
            layer_count: LAYER_COUNT,
            row_sequence_ids: true,
            row_kv_lens: true,
            softmax_scale,
        },
    });
}

#[test]
fn cuda_hybrid_mla_explicit_selection_dual_paged_uses_selector_stride() {
    const ROWS: usize = 2;
    const SELECTED_WIDTH: usize = 6;
    const PAGE_TOKENS: usize = 4;
    const FIRST_ELEMENTS_PER_TOKEN: usize = 528;
    const SECOND_ELEMENTS_PER_TOKEN: usize = 560;
    const LAYER_INDEX: usize = 2;
    const LAYER_COUNT: usize = 4;
    const PHYSICAL_SLOTS: usize = 4;
    let block_slots = [2, 0, 3, 1];
    let block_offsets = [0, 2, 4];
    let sequence_kv_lens = [6, 8];
    let second_sequence_kv_lens = [4, 5];
    let row_sequence_ids = [1, 0];
    let row_kv_lens = [7, 5];
    let row_second_kv_lens = [5, 3];
    let selected_indices = [3, 4, -1, 5, 2, 7, 4, 0, 3, 5, 1, -2];
    let selectors = [0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 2, 1];
    let mut first_plane =
        vec![701.0f32; PHYSICAL_SLOTS * LAYER_COUNT * PAGE_TOKENS * FIRST_ELEMENTS_PER_TOKEN];
    let mut second_plane =
        vec![-809.0f32; PHYSICAL_SLOTS * LAYER_COUNT * PAGE_TOKENS * SECOND_ELEMENTS_PER_TOKEN];
    for (sequence, &sequence_len) in sequence_kv_lens.iter().enumerate() {
        for logical in 0..sequence_len as usize {
            let entry = block_offsets[sequence] as usize + logical / PAGE_TOKENS;
            let slot = block_slots[entry] as usize;
            let first_base = slot * LAYER_COUNT * PAGE_TOKENS * FIRST_ELEMENTS_PER_TOKEN
                + LAYER_INDEX * PAGE_TOKENS * FIRST_ELEMENTS_PER_TOKEN
                + logical % PAGE_TOKENS * FIRST_ELEMENTS_PER_TOKEN;
            let second_base = slot * LAYER_COUNT * PAGE_TOKENS * SECOND_ELEMENTS_PER_TOKEN
                + LAYER_INDEX * PAGE_TOKENS * SECOND_ELEMENTS_PER_TOKEN
                + logical % PAGE_TOKENS * SECOND_ELEMENTS_PER_TOKEN;
            for dimension in 0..HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM {
                let first_centered =
                    ((sequence * 37 + logical * 13 + dimension * 7) % 43) as f32 - 21.0;
                first_plane[first_base + dimension] =
                    first_centered * 0.0029 + sequence as f32 * 0.013 + logical as f32 * 0.005;
                let second_centered =
                    ((sequence * 29 + logical * 19 + dimension * 11) % 53) as f32 - 26.0;
                second_plane[second_base + dimension] =
                    second_centered * 0.0037 - sequence as f32 * 0.017 + logical as f32 * 0.006;
            }
        }
    }
    let gather_layout = PagedGatherLayout {
        rows: ROWS,
        selected_width: SELECTED_WIDTH,
        page_tokens: PAGE_TOKENS,
        first_elements_per_token: FIRST_ELEMENTS_PER_TOKEN,
        second_elements_per_token: SECOND_ELEMENTS_PER_TOKEN,
        layer_index: LAYER_INDEX,
        layer_count: LAYER_COUNT,
    };
    let (gathered, valid) = explicit_selection_paged_gather(
        &first_plane,
        Some(&second_plane),
        &block_slots,
        &block_offsets,
        &sequence_kv_lens,
        Some(&second_sequence_kv_lens),
        &row_sequence_ids,
        &row_kv_lens,
        Some(&row_second_kv_lens),
        &selected_indices,
        Some(&selectors),
        gather_layout,
    );
    assert_eq!(valid, [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
    let query = explicit_selection_query(ROWS);
    let attention_sink = explicit_selection_attention_sink();
    let softmax_scale = (HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM as f32)
        .sqrt()
        .recip();
    run_hybrid_mla_explicit_selection_case(HybridMlaExplicitSelectionHostCase {
        label: "hybrid MLA explicit selection dual-paged selector stride",
        query: &query,
        first_plane: &first_plane,
        second_plane: Some(&second_plane),
        block_slots: Some(&block_slots),
        block_offsets: Some(&block_offsets),
        sequence_kv_lens: Some(&sequence_kv_lens),
        second_sequence_kv_lens: Some(&second_sequence_kv_lens),
        row_sequence_ids: Some(&row_sequence_ids),
        row_kv_lens: Some(&row_kv_lens),
        row_second_kv_lens: Some(&row_second_kv_lens),
        selected_indices: &selected_indices,
        selectors: Some(&selectors),
        attention_sink: &attention_sink,
        expected_gathered: &gathered,
        expected_valid: &valid,
        layout: cutlass::HybridMlaExplicitSelectionLayout {
            kind: cutlass::HybridMlaKvStorageKind::DualPaged,
            rows: ROWS,
            tokens_per_sequence: 0,
            kv_len: 0,
            heads: HYBRID_MLA_EXPLICIT_SELECTION_HEADS,
            head_dim: HYBRID_MLA_EXPLICIT_SELECTION_HEAD_DIM,
            selected_width: SELECTED_WIDTH,
            page_tokens: PAGE_TOKENS,
            first_elements_per_token: FIRST_ELEMENTS_PER_TOKEN,
            second_elements_per_token: SECOND_ELEMENTS_PER_TOKEN,
            layer_index: LAYER_INDEX,
            layer_count: LAYER_COUNT,
            row_sequence_ids: true,
            row_kv_lens: true,
            softmax_scale,
        },
    });
}

#[test]
#[ignore = "manual CUDA proposal hybrid-attention latency checkpoint"]
fn cuda_hybrid_mla_attention_formal_shape_latency() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const SEQUENCE_TOKENS: usize = cutlass::HYBRID_MLA_ATTENTION_WINDOW;
    const PAGE_TOKENS: usize = cutlass::HYBRID_MLA_ATTENTION_PAGE_TOKENS;
    const ROWS: usize = cutlass::PROPOSAL_ROWS;
    const HEADS: usize = cutlass::HYBRID_MLA_ATTENTION_HEADS;
    const DIM: usize = cutlass::HYBRID_MLA_ATTENTION_HEAD_DIM;
    const CAPACITY: usize = cutlass::HYBRID_MLA_ATTENTION_TOKEN_CAPACITY;
    const ITERATIONS: usize = 100;

    let query = DeviceBuffer::from_host(&stream, &vec![0.03125f32; ROWS * HEADS * DIM])
        .expect("upload proposal benchmark Q");
    let context_plane = DeviceBuffer::from_host(&stream, &vec![0.015625f32; SEQUENCE_TOKENS * DIM])
        .expect("upload proposal benchmark context");
    let block_kv = DeviceBuffer::from_host(&stream, &vec![0.0625f32; ROWS * DIM])
        .expect("upload proposal benchmark block KV");
    let block_slots_host = (0..SEQUENCE_TOKENS / PAGE_TOKENS)
        .map(|slot| slot as i32)
        .collect::<Vec<_>>();
    let block_slots = DeviceBuffer::from_host(&stream, &block_slots_host)
        .expect("upload proposal benchmark slots");
    let sink = DeviceBuffer::from_host(&stream, &vec![0.0f32; HEADS])
        .expect("upload proposal benchmark sink");
    let mut query_bf16 = DeviceBuffer::<u16>::zeroed(&stream, ROWS * HEADS * DIM)
        .expect("proposal benchmark query BF16");
    let mut gathered_kv_bf16 = DeviceBuffer::<u16>::zeroed(&stream, CAPACITY * DIM)
        .expect("proposal benchmark gathered KV BF16");
    let mut scores = DeviceBuffer::<f32>::zeroed(&stream, ROWS * HEADS * CAPACITY)
        .expect("proposal benchmark scores");
    let mut probabilities = DeviceBuffer::<u16>::zeroed(&stream, ROWS * HEADS * CAPACITY)
        .expect("proposal benchmark probabilities");
    let mut online_rescales = DeviceBuffer::<f32>::zeroed(
        &stream,
        ROWS * HEADS * cutlass::HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILES,
    )
    .expect("proposal benchmark online-softmax rescales");
    let mut denominators = DeviceBuffer::<f32>::zeroed(&stream, ROWS * HEADS)
        .expect("proposal benchmark softmax denominators");
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, ROWS * HEADS * DIM)
        .expect("proposal benchmark output");
    let mut status = DeviceBuffer::<i32>::zeroed(&stream, 1).expect("proposal benchmark status");
    let layout = cutlass::HybridMlaAttentionLayout {
        sequence_tokens: SEQUENCE_TOKENS,
        page_tokens: PAGE_TOKENS,
        elements_per_token: DIM,
        layer_index: 0,
        layer_count: 1,
        block_slot_offset: 0,
        block_slot_count: block_slots_host.len(),
        softmax_scale: (DIM as f32).powf(-0.5),
    };

    for _ in 0..5 {
        cutlass::hybrid_mla_attention(
            &stream,
            &query,
            &context_plane,
            &block_kv,
            &block_slots,
            &sink,
            &mut query_bf16,
            &mut gathered_kv_bf16,
            &mut scores,
            &mut probabilities,
            &mut online_rescales,
            &mut denominators,
            &mut output,
            &mut status,
            layout,
        )
        .expect("warm proposal hybrid attention");
    }
    stream.synchronize().expect("warm proposal synchronization");

    let started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        cutlass::hybrid_mla_attention(
            &stream,
            &query,
            &context_plane,
            &block_kv,
            &block_slots,
            &sink,
            &mut query_bf16,
            &mut gathered_kv_bf16,
            &mut scores,
            &mut probabilities,
            &mut online_rescales,
            &mut denominators,
            &mut output,
            &mut status,
            layout,
        )
        .expect("timed proposal hybrid attention");
    }
    stream
        .synchronize()
        .expect("timed proposal synchronization");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;
    let per_launch_ms = elapsed_ms / ITERATIONS as f64;
    assert_eq!(
        status
            .to_host_vec(&stream)
            .expect("download proposal status"),
        [0]
    );
    println!(
        "hybrid_mla_attention rows={ROWS} heads={HEADS} dim={DIM} context={SEQUENCE_TOKENS} iterations={ITERATIONS} total_ms={elapsed_ms:.4} per_launch_ms={per_launch_ms:.6}"
    );
}

#[test]
fn cuda_bf16_compressor_is_one_launch_and_numerically_exact() {
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
    .expect("CUDA BF16 compressor launch");

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
fn cuda_fp8_query_a_kv_is_one_launch_and_numerically_exact() {
    let fp8_available = native_kernel_available(CutlassKernelId::Fp8QueryAKv);
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

    let result = cutlass::fp8_query_a_kv(
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
    );
    if !fp8_available {
        let message = result
            .expect_err("FP8 MMA must fail closed on this target")
            .to_string();
        assert!(message.contains("unsupported capability (4)"), "{message}");
        return;
    }
    result.expect("CUDA QueryA+KV launch");

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
fn cuda_linear_semantic_entry_supports_grid_derived_row_range() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const ROWS: usize = 4097;
    const FP8_K: usize = 128;
    const BF16_K: usize = 16;
    const N1: usize = 16;
    const N2: usize = 32;

    if native_kernel_available(CutlassKernelId::Fp8QueryAKv) {
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
        let kv_scales =
            DeviceBuffer::from_host(&stream, &[127u8]).expect("upload KV prefill scales");
        let mut query_a_output =
            DeviceBuffer::<f32>::zeroed(&stream, ROWS * N1).expect("QueryA prefill output");
        let mut kv_output =
            DeviceBuffer::<f32>::zeroed(&stream, ROWS * N2).expect("KV prefill output");

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
        .expect("CUDA tiled FP8 QueryA+KV launch");
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
    .expect("CUDA tiled BF16 compressor launch");
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
fn cuda_hc_single_row_tile_matches_tiled_path_bitwise() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const HC: usize = 4;
    const HIDDEN: usize = 4096;
    const MIX: usize = 24;
    const SCALES: usize = HIDDEN / 128;
    let state_row = (0..HC * HIDDEN)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.01)
        .collect::<Vec<_>>();
    let mut tiled_state_host = state_row.clone();
    tiled_state_host.extend_from_slice(&state_row);
    let function_host = (0..HC * HIDDEN * MIX)
        .map(|index| ((index % 13) as f32 - 6.0) * 0.0001)
        .collect::<Vec<_>>();
    let rms_weight_host = (0..HIDDEN)
        .map(|index| 1.0 + (index % 5) as f32 * 0.01)
        .collect::<Vec<_>>();

    let single_state =
        DeviceBuffer::from_host(&stream, &state_row).expect("upload single-row HC state");
    let tiled_state =
        DeviceBuffer::from_host(&stream, &tiled_state_host).expect("upload tiled HC state");
    let function = DeviceBuffer::from_host(&stream, &function_host).expect("upload HC function");
    let hc_scale =
        DeviceBuffer::from_host(&stream, &[0.7f32, -0.3, 0.2]).expect("upload HC scales");
    let hc_base_host = (0..MIX)
        .map(|index| (index as f32 - 12.0) * 0.001)
        .collect::<Vec<_>>();
    let hc_base = DeviceBuffer::from_host(&stream, &hc_base_host).expect("upload HC base");
    let rms_weight = DeviceBuffer::from_host(&stream, &rms_weight_host).expect("upload RMS weight");

    let run = |state: &DeviceBuffer<f32>, rows: usize| {
        let mut mix = DeviceBuffer::<f32>::zeroed(&stream, rows * MIX).expect("HC mix");
        let mut workspace =
            DeviceBuffer::<f32>::zeroed(&stream, rows * MIX * 64 + 1).expect("HC workspace");
        let mut hidden = DeviceBuffer::<f32>::zeroed(&stream, rows * HIDDEN).expect("HC hidden");
        let mut normalized =
            DeviceBuffer::<f32>::zeroed(&stream, rows * HIDDEN).expect("HC normalized");
        let mut packed = DeviceBuffer::<u8>::zeroed(&stream, rows * HIDDEN).expect("HC packed");
        let mut scales = DeviceBuffer::<u8>::zeroed(&stream, rows * SCALES).expect("HC scales");
        let mut pre = DeviceBuffer::<f32>::zeroed(&stream, rows * HC).expect("HC pre");
        let mut post = DeviceBuffer::<f32>::zeroed(&stream, rows * HC).expect("HC post");
        let mut comb = DeviceBuffer::<f32>::zeroed(&stream, rows * HC * HC).expect("HC comb");
        cutlass::hc_producer(
            &stream,
            state,
            &function,
            &hc_scale,
            &hc_base,
            &rms_weight,
            &mut mix,
            &mut workspace,
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
            1.0e-6,
            1.0e-6,
            1.0e-6,
        )
        .expect("HC producer launch");
        (
            hidden.to_host_vec(&stream).expect("download HC hidden"),
            normalized
                .to_host_vec(&stream)
                .expect("download HC normalized"),
            packed.to_host_vec(&stream).expect("download HC packed"),
            scales.to_host_vec(&stream).expect("download HC scales"),
            pre.to_host_vec(&stream).expect("download HC pre"),
            post.to_host_vec(&stream).expect("download HC post"),
            comb.to_host_vec(&stream).expect("download HC comb"),
        )
    };
    let single = run(&single_state, 1);
    let tiled = run(&tiled_state, 2);

    let assert_f32_rows = |label: &str, single: &[f32], tiled: &[f32], width: usize| {
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(single), bits(&tiled[..width]), "{label} single/tiled");
        assert_eq!(
            bits(&tiled[..width]),
            bits(&tiled[width..]),
            "{label} duplicate tiled rows"
        );
    };
    assert_f32_rows("hidden", &single.0, &tiled.0, HIDDEN);
    assert_f32_rows("normalized", &single.1, &tiled.1, HIDDEN);
    assert_eq!(single.2, tiled.2[..HIDDEN]);
    assert_eq!(tiled.2[..HIDDEN], tiled.2[HIDDEN..]);
    assert_eq!(single.3, tiled.3[..SCALES]);
    assert_eq!(tiled.3[..SCALES], tiled.3[SCALES..]);
    assert_f32_rows("pre", &single.4, &tiled.4, HC);
    assert_f32_rows("post", &single.5, &tiled.5, HC);
    assert_f32_rows("comb", &single.6, &tiled.6, HC * HC);
}

#[test]
fn cuda_hc_producer_accepts_dynamic_m() {
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
        let mut mix = DeviceBuffer::<f32>::zeroed(&stream, rows * MIX).expect("HC mix");
        let mut workspace =
            DeviceBuffer::<f32>::zeroed(&stream, rows * MIX * 64 + 1).expect("HC workspace");
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
            &mut mix,
            &mut workspace,
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
#[ignore = "manual CUDA latency checkpoint"]
fn cuda_hc_producer_formal_shape_latency() {
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
        let mut mix = DeviceBuffer::<f32>::zeroed(&stream, rows * MIX).expect("HC mix");
        let mut workspace =
            DeviceBuffer::<f32>::zeroed(&stream, rows * MIX * 64 + 1).expect("HC workspace");
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
            &mut mix,
            &mut workspace,
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
                &mut mix,
                &mut workspace,
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
fn cuda_shared_ffn_accepts_dynamic_m_and_accumulates() {
    if !native_kernel_available(CutlassKernelId::SharedFfn) {
        return;
    }
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
#[ignore = "manual CUDA latency checkpoint"]
fn cuda_shared_ffn_formal_shape_latency() {
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
