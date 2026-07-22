#![cfg(feature = "cutlass")]

use cuda_core::{CudaContext, DeviceBuffer};
use ferrule_cuda::cutlass::{self, CutlassKernelId};
use ferrule_cuda::provider::{
    KernelOperation, KernelProviderId, LayerKernelRequirements, LinearBundleRequirement,
    WeightLayout,
};

fn bf16_boundary(value: f32) -> f32 {
    let bits = value.to_bits();
    let bias = 0x7fffu32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
}

fn bf16_storage_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let bits = value.to_bits();
            let bias = 0x7fffu32 + ((bits >> 16) & 1);
            ((bits.wrapping_add(bias) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

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
        KernelOperation::MlaQueryB,
        1024,
        [32768],
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
    layer.require_operation(KernelOperation::DsparkMainProjectNorm);
    layer.require_operation(KernelOperation::DsparkHybridMlaAttention);
    layer.require_operation(KernelOperation::DsparkProposalHead);
    let plan = ferrule_cuda::compile_cuda_model_plan(&[layer]).expect("compile SM121 plan");
    let launch = plan.layers[0]
        .operation(KernelOperation::MlaQueryAKv)
        .expect("SM121 QueryA+KV launch");
    assert_eq!(launch.kernel.provider, KernelProviderId::CutlassCubin);
    assert_eq!(launch.kernel.variant, 0);
    assert!(launch.is_capture_safe());

    let query_b = plan.layers[0]
        .operation(KernelOperation::MlaQueryB)
        .expect("SM121 QueryB launch");
    assert_eq!(query_b.kernel.provider, KernelProviderId::CutlassCubin);
    assert_eq!(query_b.kernel.variant, 0);
    assert!(query_b.is_capture_safe());

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
        KernelOperation::DsparkMainProjectNorm,
        KernelOperation::DsparkHybridMlaAttention,
        KernelOperation::DsparkProposalHead,
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
    assert_eq!(manifest.kernel_count, 10);
    assert!(manifest.supports(CutlassKernelId::Fp8QueryAKvSm121));
    assert!(manifest.supports(CutlassKernelId::Bf16CompressorSm121));
    assert!(manifest.supports(CutlassKernelId::HcProducerSm121));
    assert!(manifest.supports(CutlassKernelId::SharedFfnSm121));
    assert!(manifest.supports(CutlassKernelId::StableFrameFp4MoeSm121));
    assert!(manifest.supports(CutlassKernelId::MlaOutputSm121));
    assert!(manifest.supports(CutlassKernelId::DsparkMainProjectNormSm121));
    assert!(manifest.supports(CutlassKernelId::DsparkHybridMlaAttentionSm121));
    assert!(manifest.supports(CutlassKernelId::DsparkProposalHeadSm121));
    assert!(manifest.supports(CutlassKernelId::Fp8ProjectionSm121));
}

#[test]
fn sm121_mla_output_single_row_matches_cooperative_path_bitwise() {
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
        DeviceBuffer::<f32>::zeroed(&stream, SINGLE_ROWS * LATENT).expect("single-row latent");
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

    let mut cooperative_latent = DeviceBuffer::<f32>::zeroed(&stream, COOPERATIVE_ROWS * LATENT)
        .expect("cooperative latent");
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
    assert_eq!(
        single_latent
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        cooperative_latent[..LATENT]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
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
fn sm121_dspark_proposal_head_keeps_markov_dependency_on_device() {
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

    cutlass::dspark_proposal_head(
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
        cutlass::DsparkProposalHeadLayout {
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
    .expect("DSpark proposal-head launch");

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
fn cuda_hc_mean_scatter_builds_dspark_target_taps_without_host_concat() {
    let ops = ferrule_cuda::context::CudaArtifactOperatorContext::new()
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
        .expect("DSpark target-tap buffer");
    ops.hc_mean_scatter_from_device_into(&state, ROWS, HC, HIDDEN, SLOT, TAPS, &mut taps)
        .expect("HC mean-scatter");
    let taps = ops
        .download_f32_buffer(&taps)
        .expect("download DSpark target taps");

    for row in 0..ROWS {
        for tap in 0..TAPS {
            for dim in 0..HIDDEN {
                let value = taps[row * TAPS * HIDDEN + tap * HIDDEN + dim];
                let expected = if tap == SLOT {
                    row as f32 * 10.0 + 1.5 + dim as f32 / HIDDEN as f32
                } else {
                    0.0
                };
                assert_eq!(value, expected);
            }
        }
    }
}

#[test]
fn sm121_dspark_main_project_norm_preserves_bf16_boundary() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const ROWS: usize = 2;
    const INPUT: usize = 128;
    const OUTPUT: usize = 128;
    let input = DeviceBuffer::from_host(&stream, &vec![1.0f32; ROWS * INPUT])
        .expect("upload DSpark target taps");
    let mut activation =
        DeviceBuffer::<u8>::zeroed(&stream, ROWS * INPUT).expect("DSpark activation scratch");
    let mut activation_scales = DeviceBuffer::<u8>::zeroed(&stream, ROWS * (INPUT / 128))
        .expect("DSpark activation-scale scratch");
    let weight = DeviceBuffer::from_host(&stream, &vec![0x38u8; OUTPUT * INPUT])
        .expect("upload DSpark main projection");
    let weight_scales =
        DeviceBuffer::from_host(&stream, &[127u8]).expect("upload DSpark main-projection scales");
    let norm_weight =
        DeviceBuffer::from_host(&stream, &vec![1.0f32; OUTPUT]).expect("upload DSpark main norm");
    let mut inv_rms =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS).expect("DSpark inverse-RMS scratch");
    let mut output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * OUTPUT).expect("DSpark normalized output");

    cutlass::dspark_main_project_norm(
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
    .expect("SM121 DSpark main-project/norm launch");

    for value in output
        .to_host_vec(&stream)
        .expect("download DSpark main-project/norm output")
    {
        assert_eq!(value, 1.0);
    }
}

#[test]
fn sm121_dspark_hybrid_attention_matches_full_block_reference() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const SEQUENCE_TOKENS: usize = 18;
    const PAGE_TOKENS: usize = cutlass::DSPARK_ATTENTION_PAGE_TOKENS;
    const LAYER_COUNT: usize = 2;
    const LAYER_INDEX: usize = 1;
    const PHYSICAL_SLOTS: usize = 2;
    const ROWS: usize = cutlass::DSPARK_PROPOSAL_ROWS;
    const HEADS: usize = cutlass::DSPARK_ATTENTION_HEADS;
    const DIM: usize = cutlass::DSPARK_ATTENTION_HEAD_DIM;
    const CAPACITY: usize = cutlass::DSPARK_ATTENTION_TOKEN_CAPACITY;

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
        const ROWS: usize = cutlass::DSPARK_PROPOSAL_ROWS;
        const HEADS: usize = cutlass::DSPARK_ATTENTION_HEADS;
        const DIM: usize = cutlass::DSPARK_ATTENTION_HEAD_DIM;
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
                let maximum = scores.iter().copied().fold(sink[head], f32::max);
                let denominator = (sink[head] - maximum).exp()
                    + scores
                        .iter()
                        .map(|score| (*score - maximum).exp())
                        .sum::<f32>();
                for (token, score) in scores.into_iter().enumerate() {
                    let probability = bf16_boundary((score - maximum).exp() / denominator);
                    let values = if token < context_tokens {
                        &context[token * DIM..(token + 1) * DIM]
                    } else {
                        let block_row = token - context_tokens;
                        &block[block_row * DIM..(block_row + 1) * DIM]
                    };
                    for dim in 0..DIM {
                        output[q_base + dim] += probability * bf16_boundary(values[dim]);
                    }
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

    let query = DeviceBuffer::from_host(&stream, &query_host).expect("upload DSpark Q");
    let expected_context_plane = context_plane.clone();
    let context_plane =
        DeviceBuffer::from_host(&stream, &context_plane).expect("upload paged DSpark context");
    let block_kv =
        DeviceBuffer::from_host(&stream, &block_kv_host).expect("upload DSpark block KV");
    let block_slots =
        DeviceBuffer::from_host(&stream, &block_slots_host).expect("upload DSpark block slots");
    let sink = DeviceBuffer::from_host(&stream, &sink_host).expect("upload attention sink");
    let mut query_bf16 = DeviceBuffer::<u16>::zeroed(&stream, ROWS * HEADS * DIM)
        .expect("DSpark query BF16 scratch");
    let mut gathered_kv_bf16 = DeviceBuffer::<u16>::zeroed(&stream, CAPACITY * DIM)
        .expect("DSpark gathered KV BF16 scratch");
    let mut scores = DeviceBuffer::<f32>::zeroed(&stream, ROWS * HEADS * CAPACITY)
        .expect("DSpark score scratch");
    let mut probabilities = DeviceBuffer::<u16>::zeroed(&stream, ROWS * HEADS * CAPACITY)
        .expect("DSpark probability scratch");
    let mut output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * HEADS * DIM).expect("DSpark attention output");
    let mut status = DeviceBuffer::<i32>::zeroed(&stream, 1).expect("DSpark device status");

    cutlass::dspark_hybrid_mla_attention(
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
        &mut output,
        &mut status,
        cutlass::DsparkHybridMlaAttentionLayout {
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
    .expect("SM121 DSpark hybrid-attention launch");

    assert_eq!(
        status.to_host_vec(&stream).expect("download DSpark status"),
        [0]
    );
    let actual = output
        .to_host_vec(&stream)
        .expect("download DSpark hybrid-attention output");
    assert_eq!(
        context_plane
            .to_host_vec(&stream)
            .expect("download unchanged DSpark context plane"),
        expected_context_plane,
        "ephemeral proposal-block attention modified committed paged KV"
    );
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 3.0e-3,
        "DSpark hybrid attention differs from full-block reference: max_abs={max_abs:e}"
    );
    assert!(
        (actual[0] - causal[0]).abs() > 0.1,
        "DSpark query row zero appears causally masked: actual={} causal={}",
        actual[0],
        causal[0]
    );
}

#[test]
#[ignore = "manual GB10 DSpark hybrid-attention latency checkpoint"]
fn sm121_dspark_hybrid_attention_formal_shape_latency() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    const SEQUENCE_TOKENS: usize = cutlass::DSPARK_ATTENTION_WINDOW;
    const PAGE_TOKENS: usize = cutlass::DSPARK_ATTENTION_PAGE_TOKENS;
    const ROWS: usize = cutlass::DSPARK_PROPOSAL_ROWS;
    const HEADS: usize = cutlass::DSPARK_ATTENTION_HEADS;
    const DIM: usize = cutlass::DSPARK_ATTENTION_HEAD_DIM;
    const CAPACITY: usize = cutlass::DSPARK_ATTENTION_TOKEN_CAPACITY;
    const ITERATIONS: usize = 100;

    let query = DeviceBuffer::from_host(&stream, &vec![0.03125f32; ROWS * HEADS * DIM])
        .expect("upload DSpark benchmark Q");
    let context_plane = DeviceBuffer::from_host(&stream, &vec![0.015625f32; SEQUENCE_TOKENS * DIM])
        .expect("upload DSpark benchmark context");
    let block_kv = DeviceBuffer::from_host(&stream, &vec![0.0625f32; ROWS * DIM])
        .expect("upload DSpark benchmark block KV");
    let block_slots_host = (0..SEQUENCE_TOKENS / PAGE_TOKENS)
        .map(|slot| slot as i32)
        .collect::<Vec<_>>();
    let block_slots =
        DeviceBuffer::from_host(&stream, &block_slots_host).expect("upload DSpark benchmark slots");
    let sink = DeviceBuffer::from_host(&stream, &vec![0.0f32; HEADS])
        .expect("upload DSpark benchmark sink");
    let mut query_bf16 = DeviceBuffer::<u16>::zeroed(&stream, ROWS * HEADS * DIM)
        .expect("DSpark benchmark query BF16");
    let mut gathered_kv_bf16 = DeviceBuffer::<u16>::zeroed(&stream, CAPACITY * DIM)
        .expect("DSpark benchmark gathered KV BF16");
    let mut scores = DeviceBuffer::<f32>::zeroed(&stream, ROWS * HEADS * CAPACITY)
        .expect("DSpark benchmark scores");
    let mut probabilities = DeviceBuffer::<u16>::zeroed(&stream, ROWS * HEADS * CAPACITY)
        .expect("DSpark benchmark probabilities");
    let mut output =
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * HEADS * DIM).expect("DSpark benchmark output");
    let mut status = DeviceBuffer::<i32>::zeroed(&stream, 1).expect("DSpark benchmark status");
    let layout = cutlass::DsparkHybridMlaAttentionLayout {
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
        cutlass::dspark_hybrid_mla_attention(
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
            &mut output,
            &mut status,
            layout,
        )
        .expect("warm DSpark hybrid attention");
    }
    stream.synchronize().expect("warm DSpark synchronization");

    let started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        cutlass::dspark_hybrid_mla_attention(
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
            &mut output,
            &mut status,
            layout,
        )
        .expect("timed DSpark hybrid attention");
    }
    stream.synchronize().expect("timed DSpark synchronization");
    let elapsed_ms = started.elapsed().as_secs_f64() * 1.0e3;
    let per_launch_ms = elapsed_ms / ITERATIONS as f64;
    assert_eq!(
        status.to_host_vec(&stream).expect("download DSpark status"),
        [0]
    );
    println!(
        "dspark_hybrid_mla_attention rows={ROWS} heads={HEADS} dim={DIM} context={SEQUENCE_TOKENS} iterations={ITERATIONS} total_ms={elapsed_ms:.4} per_launch_ms={per_launch_ms:.6}"
    );
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
fn sm121_hc_single_row_tile_matches_tiled_path_bitwise() {
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
