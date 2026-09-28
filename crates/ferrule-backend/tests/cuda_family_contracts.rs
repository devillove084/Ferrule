//! PR21 family ownership and compatibility contracts; no CUDA needed for sources.
const PROVIDER: &str = include_str!("../src/cuda/providers/cutlass.rs");
const LEGACY_PROVIDER: &str = include_str!("../src/cuda/providers/cutlass_legacy.rs");
const F32_PROVIDER: &str = include_str!("../src/cuda/providers/cutlass_f32.rs");
const CONTEXT: &str = include_str!("../src/cuda/context.rs");
const F32: &str = include_str!("../src/cuda/operators/f32.rs");
const PROPOSAL: &str = include_str!("../src/cuda/operators/proposal.rs");
const HYBRID: &str = include_str!("../src/cuda/operators/attention/hybrid.rs");
const GROUPED: &str = include_str!("../src/cuda/operators/moe/grouped_fp4.rs");

#[test]
fn family_contracts_are_definitions_not_provider_type_aliases() {
    for (source, names) in [
        (F32, &["F32GemmLayout", "F32GemmError"][..]),
        (
            PROPOSAL,
            &["ProposalHeadLayout", "CudaProposalHeadWorkspace"][..],
        ),
        (
            HYBRID,
            &[
                "HybridMlaAttentionLayout",
                "HybridMlaExplicitSelectionLayout",
                "HybridMlaKvStorageKind",
                "HybridMlaExplicitSelectionBuffers",
                "CudaHybridMlaAttentionWorkspace",
                "CudaHybridMlaExplicitSelectionWorkspace",
            ][..],
        ),
        (
            GROUPED,
            &[
                "GroupedFp4MoeLayout",
                "GroupedFp4MoeBuffers",
                "CudaMoeBatchedWorkspace",
                "CudaExpertGroupRoutePlan",
                "CudaExpertGroupRoutePlanHost",
            ][..],
        ),
    ] {
        for name in names {
            let declarations = [
                format!("pub struct {name} {{"),
                format!("pub struct {name}<"),
                format!("pub enum {name} {{"),
            ];
            assert!(
                declarations.iter().any(|d| source.contains(d)),
                "missing semantic definition {name}"
            );
            for old in [PROVIDER, F32_PROVIDER, CONTEXT] {
                assert!(
                    !declarations.iter().any(|d| old.contains(d)),
                    "duplicate provider/context definition {name}"
                );
            }
        }
    }
    assert!(
        F32_PROVIDER.contains("use crate::cuda::operators::linear::{F32GemmError, F32GemmLayout}")
    );
    assert!(PROVIDER.contains("pub use crate::cuda::operators::proposal::{"));
    assert!(PROVIDER.contains("pub use crate::cuda::operators::moe::grouped_fp4::{"));
    assert!(PROVIDER.contains("pub use crate::cuda::operators::attention::hybrid::{"));
}

#[test]
fn production_workspace_preparation_stays_behind_family_contracts() {
    assert!(!CONTEXT.contains("providers::cutlass"));
    assert!(!CONTEXT.contains("CutlassKernelId"));
    let contracts = include_str!("../src/cuda/operators/contracts.rs");
    assert!(!contracts.contains("providers::cutlass"));
    assert!(contracts.contains("operators::legacy"));
    for forbidden in [
        "mxfp4_sfb_storage_bytes(",
        "prepare_mxfp4_sfb(",
        "grouped_fp4_moe_workspace_size(",
        "HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILES",
    ] {
        assert!(
            !CONTEXT.contains(forbidden),
            "context still consumes {forbidden}"
        );
    }
    assert!(CONTEXT.contains("grouped_fp4_moe_workspace_requirements(GroupedFp4MoeLayout"));
    assert!(CONTEXT.contains("CudaHybridMlaAttentionWorkspace::storage_lengths()?"));
    assert!(CONTEXT.contains("ExpertScaleShape {"));
    assert!(GROUPED.contains("pub mod legacy"));
    assert!(GROUPED.contains("pub(crate) struct ExpertScaleShape"));
    // Native POD lowering is still private and is never defined in the family.
    for name in [
        "CutlassProposalHeadArgs",
        "FerruleCutlassGroupedFp4MoeArgs",
        "FerruleCutlassHybridMlaExplicitSelectionArgs",
    ] {
        assert!(PROVIDER.contains(&format!("pub(crate) struct {name}")));
        for family in [PROPOSAL, HYBRID, GROUPED] {
            assert!(!family.contains(name));
        }
    }
}

#[test]
fn inherent_precision_const_bridge_is_explicitly_temporary() {
    assert!(LEGACY_PROVIDER.contains("Temporary source-compatibility bridge"));
    assert!(LEGACY_PROVIDER.contains("pub const fn kernel(self) -> CutlassKernelId"));
    assert!(
        LEGACY_PROVIDER
            .contains("legacy provider bridge; submit a semantic numeric FP8 plan instead")
    );
    assert!(LEGACY_PROVIDER.contains("pub trait NumericFp8PrecisionExt"));
}

#[cfg(feature = "cuda")]
#[test]
fn family_types_preserve_all_legacy_identities() {
    use ferrule_backend::cuda::{context, operators, providers::cutlass};
    use operators::{attention::hybrid, linear, moe::grouped_fp4, proposal};
    use std::any::{TypeId, type_name};
    macro_rules! identity {
        ($new:ty, $old:ty, $owner:expr) => {
            assert_eq!(TypeId::of::<$new>(), TypeId::of::<$old>());
            assert!(
                type_name::<$new>().contains($owner),
                "{}",
                type_name::<$new>()
            );
        };
    }
    identity!(
        linear::F32GemmLayout,
        cutlass::F32GemmLayout,
        "operators::linear::f32"
    );
    identity!(
        linear::F32GemmError,
        cutlass::F32GemmError,
        "operators::linear::f32"
    );
    identity!(
        proposal::ProposalHeadLayout,
        cutlass::ProposalHeadLayout,
        "operators::proposal"
    );
    identity!(
        proposal::ProposalHeadLayout,
        linear::ProposalHeadLayout,
        "operators::proposal"
    );
    identity!(
        proposal::CudaProposalHeadWorkspace,
        context::CudaProposalHeadWorkspace,
        "operators::proposal"
    );
    identity!(
        hybrid::HybridMlaAttentionLayout,
        cutlass::HybridMlaAttentionLayout,
        "operators::attention::hybrid"
    );
    identity!(
        hybrid::HybridMlaExplicitSelectionLayout,
        cutlass::HybridMlaExplicitSelectionLayout,
        "operators::attention::hybrid"
    );
    identity!(
        hybrid::HybridMlaKvStorageKind,
        cutlass::HybridMlaKvStorageKind,
        "operators::attention::hybrid"
    );
    identity!(
        hybrid::HybridMlaExplicitSelectionBuffers<'static>,
        cutlass::HybridMlaExplicitSelectionBuffers<'static>,
        "operators::attention::hybrid"
    );
    identity!(
        hybrid::CudaHybridMlaAttentionWorkspace,
        context::CudaHybridMlaAttentionWorkspace,
        "operators::attention::hybrid"
    );
    identity!(
        hybrid::CudaHybridMlaExplicitSelectionWorkspace,
        context::CudaHybridMlaExplicitSelectionWorkspace,
        "operators::attention::hybrid"
    );
    identity!(
        grouped_fp4::GroupedFp4MoeLayout,
        cutlass::GroupedFp4MoeLayout,
        "operators::moe::grouped_fp4"
    );
    identity!(
        grouped_fp4::GroupedFp4MoeBuffers<'static>,
        cutlass::GroupedFp4MoeBuffers<'static>,
        "operators::moe::grouped_fp4"
    );
    identity!(
        grouped_fp4::CudaMoeBatchedWorkspace,
        context::CudaMoeBatchedWorkspace,
        "operators::moe::grouped_fp4"
    );
    identity!(
        grouped_fp4::CudaExpertGroupRoutePlan,
        context::CudaExpertGroupRoutePlan,
        "operators::moe::grouped_fp4"
    );
    identity!(
        grouped_fp4::CudaExpertGroupRoutePlanHost,
        context::CudaExpertGroupRoutePlanHost,
        "operators::moe::grouped_fp4"
    );
    assert_eq!(hybrid::HybridMlaKvStorageKind::Contiguous as u32, 1);
    assert_eq!(hybrid::HybridMlaKvStorageKind::Paged as u32, 2);
    assert_eq!(hybrid::HybridMlaKvStorageKind::DualPaged as u32, 3);
}

#[cfg(feature = "cuda")]
#[test]
fn semantic_workspace_queries_keep_legacy_results() {
    use ferrule_backend::cuda::{
        operators::{attention::hybrid, moe::grouped_fp4},
        providers::cutlass,
    };
    let attention = hybrid::HybridMlaExplicitSelectionLayout {
        kind: hybrid::HybridMlaKvStorageKind::Contiguous,
        rows: 1,
        tokens_per_sequence: 1,
        kv_len: 4,
        heads: 64,
        head_dim: 512,
        selected_width: 4,
        page_tokens: 0,
        first_elements_per_token: 0,
        second_elements_per_token: 0,
        layer_index: 0,
        layer_count: 0,
        row_sequence_ids: false,
        row_kv_lens: false,
        softmax_scale: 0.125,
    };
    assert_eq!(
        hybrid::workspace_requirements(attention).unwrap(),
        cutlass::hybrid_mla_explicit_selection_workspace_requirements(attention).unwrap()
    );
    let grouped = grouped_fp4::GroupedFp4MoeLayout {
        active_group_count: 1,
        small_group_count: 1,
        slot_capacity: 1,
        max_group_rows: 1,
        total_routed_rows: 1,
        num_tokens: 1,
        num_routes: 1,
        input_size: 128,
        intermediate_size: 128,
        hidden_size: 64,
        swiglu_limit: 0.0,
    };
    let requirements = grouped_fp4::workspace_requirements(grouped).unwrap();
    assert_eq!(
        requirements.bytes as usize,
        cutlass::grouped_fp4_moe_workspace_size(grouped).unwrap()
    );
    assert_eq!(requirements.alignment, 256);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires SM86 CUDA; grouped FP4 capability must fail closed before allocation"]
fn sm86_grouped_owner_rejects_unsupported_without_allocation() {
    use ferrule_backend::cuda::{
        operators::linear::CudaOperators,
        providers::{COMPILED_TARGET, cutlass},
    };
    assert_eq!(
        COMPILED_TARGET, "sm_86",
        "run this targeted contract only on SM86"
    );
    assert!(
        !cutlass::discover_provider()
            .unwrap()
            .supports(cutlass::CutlassKernelId::GroupedFp4Moe)
    );
    let op = CudaOperators::new_on_device(0).unwrap();
    op.reset_counters();
    assert!(op.expert_group_route_plan(1, 1, 1, 128, 128, 64).is_err());
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().compute_kernel_launches, 0);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA GPU; family-owned workspace construction on the existing owner"]
fn owner_workspace_construction_preserves_status_and_capture_rejection() {
    use ferrule_backend::cuda::operators::{attention::hybrid, linear::CudaOperators, proposal};
    let op = CudaOperators::new_on_device(0).unwrap();
    let proposal: proposal::CudaProposalHeadWorkspace =
        op.proposal_head_workspace(5, 8, 16, 10).unwrap();
    assert_eq!(proposal.token_ids().len(), 6);
    assert_eq!(proposal.confidence().len(), 5);
    assert_eq!(op.download_i32_buffer(proposal.status()).unwrap(), [0]);
    let attention: hybrid::CudaHybridMlaAttentionWorkspace =
        op.hybrid_mla_attention_workspace().unwrap();
    assert_eq!(op.download_i32_buffer(attention.status()).unwrap(), [0]);
    let layout = hybrid::HybridMlaExplicitSelectionLayout {
        kind: hybrid::HybridMlaKvStorageKind::Contiguous,
        rows: 1,
        tokens_per_sequence: 1,
        kv_len: 4,
        heads: 64,
        head_dim: 512,
        selected_width: 4,
        page_tokens: 0,
        first_elements_per_token: 0,
        second_elements_per_token: 0,
        layer_index: 0,
        layer_count: 0,
        row_sequence_ids: false,
        row_kv_lens: false,
        softmax_scale: 0.125,
    };
    let explicit: hybrid::CudaHybridMlaExplicitSelectionWorkspace =
        op.hybrid_mla_explicit_selection_workspace(layout).unwrap();
    assert!(!explicit.status().is_empty());
    op.reset_counters();
    op.enable_capture_safe();
    assert!(op.hybrid_mla_attention_workspace().is_err());
    assert!(op.hybrid_mla_explicit_selection_workspace(layout).is_err());
    assert!(op.proposal_head_workspace(5, 8, 16, 10).is_err());
    op.disable_capture_safe();
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().compute_kernel_launches, 0);
    assert_eq!(op.counters().stream_wide_syncs, 0);
}

#[test]
fn family_impls_are_outside_aggregate_context() {
    for (source, methods) in [
        (
            include_str!("../src/cuda/operators/impls/projection.rs"),
            &[
                "artifact_fp8_projection_rows_from_device_into_with_scratch",
                "artifact_fp8_query_a_kv_into",
                "artifact_mla_output_into",
                "prepare_fp8_activation_from_device",
                "prepared_fp8_activation_from_storage",
            ][..],
        ),
        (
            include_str!("../src/cuda/operators/impls/hc.rs"),
            &["hc_pre_rmsnorm_fp8_into"][..],
        ),
        (
            include_str!("../src/cuda/operators/impls/compressor.rs"),
            &["artifact_bf16_compressor_into"][..],
        ),
        (
            include_str!("../src/cuda/operators/impls/proposal.rs"),
            &[
                "artifact_proposal_head_into",
                "artifact_main_project_norm_into",
            ][..],
        ),
        (
            include_str!("../src/cuda/operators/impls/attention.rs"),
            &["hybrid_mla_attention_into"][..],
        ),
        (
            include_str!("../src/cuda/operators/impls/moe.rs"),
            &["artifact_shared_ffn_into"][..],
        ),
    ] {
        assert!(source.contains("impl CudaOperators"));
        assert!(source.contains("self.submit_operator("));
        assert!(!source.contains("self.stream"));
        assert!(!source.contains("self.record_kernel_launch"));
        assert!(!source.contains("operators::legacy"));
        assert!(!source.contains("struct CudaOperators"));
        assert!(!source.contains("new_stream("));
        for method in methods {
            let declaration = format!("pub fn {method}");
            assert!(
                !CONTEXT.contains(&declaration),
                "{method} remains in context"
            );
            assert!(
                source.contains(&declaration),
                "{method} is missing from its family"
            );
        }
    }
    for owner_method in [
        "record_kernel_launch",
        "uninitialized_device_buffer",
        "record_compute_event",
        "proposal_head_workspace",
        "begin_proposal_head_result_download",
    ] {
        assert!(
            CONTEXT.contains(&format!("fn {owner_method}")),
            "owner coordination moved: {owner_method}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn legacy_aggregate_paths_remain_source_compatible() {
    use ferrule_backend::cuda::operators::{self, legacy};
    let _: fn(usize, usize) -> ferrule_common::Result<usize> = legacy::mxfp4_sfb_storage_bytes;
    let _: fn(usize, usize) -> ferrule_common::Result<usize> = operators::mxfp4_sfb_storage_bytes;
    let _ = legacy::bf16_compressor;
    let _ = operators::bf16_compressor;
    let _ = legacy::hc_producer;
    let _ = legacy::shared_ffn;
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "CUDA SM86+; relocated compressor owner method, preflight and capture-safe submission"]
fn relocated_compressor_owner_preserves_preflight_counts_and_output() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    let op = CudaOperators::new_on_device(0).unwrap();
    let first = op
        .upload_bf16_linear(&0x3f80u16.to_le_bytes().repeat(16 * 16), 16, 16)
        .unwrap();
    let second = op
        .upload_bf16_linear(&0x3f80u16.to_le_bytes().repeat(32 * 16), 32, 16)
        .unwrap();
    let input = op.upload_f32_buffer(&[1.0; 32]).unwrap();
    let mut out1 = op.zero_f32_buffer(32).unwrap();
    let mut out2 = op.zero_f32_buffer(64).unwrap();
    op.reset_counters();
    op.enable_capture_safe();
    assert!(
        op.artifact_bf16_compressor_into(&first, &second, &input, 1, &mut out1, &mut out2)
            .is_err()
    );
    assert_eq!(op.counters().compute_kernel_launches, 0);
    op.artifact_bf16_compressor_into(&first, &second, &input, 2, &mut out1, &mut out2)
        .unwrap();
    op.disable_capture_safe();
    assert_eq!(op.counters().compute_kernel_launches, 1);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().stream_wide_syncs, 0);
    op.record_compute_event().unwrap().synchronize().unwrap();
    assert_eq!(op.download_f32_buffer(&out1).unwrap(), [16.0; 32]);
    assert_eq!(op.download_f32_buffer(&out2).unwrap(), [16.0; 64]);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "CUDA SM86+; relocated HC producer and prepared activation owner methods"]
fn relocated_hc_owner_keeps_single_launch_and_prepared_activation() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    let op = CudaOperators::new_on_device(0).unwrap();
    let (hc, hidden, mix) = (4, 4096, 24);
    let state = op.zero_f32_buffer(hc * hidden).unwrap();
    let function = op.zero_f32_buffer(hc * hidden * mix).unwrap();
    let scales = op.upload_f32_buffer(&[1.0; 3]).unwrap();
    let base = op.zero_f32_buffer(mix).unwrap();
    let norm = op.upload_f32_buffer(&vec![1.0; hidden]).unwrap();
    let mut mix_output = op.zero_f32_buffer(mix).unwrap();
    let mut workspace = op.zero_f32_buffer(mix * 64 + 1).unwrap();
    let mut hidden_output = op.zero_f32_buffer(hidden).unwrap();
    let mut normalized = op.zero_f32_buffer(hidden).unwrap();
    let mut pre = op.zero_f32_buffer(hc).unwrap();
    let mut post = op.zero_f32_buffer(hc).unwrap();
    let mut comb = op.zero_f32_buffer(hc * hc).unwrap();
    let mut packed = op.fp8_activation_pack(1, hidden).unwrap();
    op.reset_counters();
    op.enable_capture_safe();
    op.hc_pre_rmsnorm_fp8_into(
        &state,
        &function,
        &scales,
        &base,
        &norm,
        &mut mix_output,
        &mut workspace,
        1,
        hc,
        hidden,
        20,
        1e-6,
        1e-6,
        1e-6,
        &mut hidden_output,
        &mut normalized,
        &mut pre,
        &mut post,
        &mut comb,
        &mut packed,
    )
    .unwrap();
    assert!(
        op.prepared_fp8_activation_from_storage(&packed, 1, hidden)
            .is_ok()
    );
    assert!(
        op.prepared_fp8_activation_from_storage(&packed, 2, hidden)
            .is_err()
    );
    op.disable_capture_safe();
    assert_eq!(op.counters().compute_kernel_launches, 1);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().stream_wide_syncs, 0);
    op.record_compute_event().unwrap().synchronize().unwrap();
    assert!(
        op.download_f32_buffer(&hidden_output)
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
    assert!(
        op.download_f32_buffer(&normalized)
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "CUDA SM86+; relocated proposal owner method and existing host-result coordination"]
fn relocated_proposal_owner_keeps_markov_and_status_contract() {
    use ferrule_backend::cuda::operators::{linear::CudaOperators, proposal::ProposalHeadLayout};
    let op = CudaOperators::new_on_device(0).unwrap();
    let (rows, hc, hidden, vocab, rank) = (5, 2, 16, 128, 16);
    let state = op.zero_f32_buffer(rows * hc * hidden).unwrap();
    let function = op.zero_f32_buffer(hc * hc * hidden).unwrap();
    let scale = op.upload_f32_buffer(&[1.0]).unwrap();
    let base = op.zero_f32_buffer(hc).unwrap();
    let norm = op.upload_f32_buffer(&vec![1.0; hidden]).unwrap();
    let lm = op
        .upload_bf16_linear(&vec![0; vocab * hidden * 2], vocab, hidden)
        .unwrap();
    let mut w1 = vec![0u16; vocab * rank];
    for token in 0..vocab {
        w1[token * rank] = 0x3f80;
    }
    let mut w2 = vec![0u16; vocab * rank];
    w2[3 * rank] = 0x4100;
    let bytes = |v: &[u16]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>();
    let w1 = op.upload_bf16_linear(&bytes(&w1), vocab, rank).unwrap();
    let w2 = op.upload_bf16_linear(&bytes(&w2), vocab, rank).unwrap();
    let confidence = op
        .upload_bf16_linear(&vec![0; (hidden + rank) * 2], 1, hidden + rank)
        .unwrap();
    let layout = ProposalHeadLayout {
        rows,
        hc,
        hidden,
        vocab,
        markov_rank: rank,
        partial_capacity: 64,
        hc_eps: 1e-6,
        norm_eps: 1e-6,
    };
    let mut workspace = op.proposal_head_workspace(rows, hidden, vocab, 64).unwrap();
    op.reset_counters();
    op.artifact_proposal_head_into(
        &state,
        &function,
        &scale,
        &base,
        &norm,
        &lm,
        &w1,
        &w2,
        &confidence,
        5,
        layout,
        &mut workspace,
    )
    .unwrap();
    assert_eq!(op.counters().compute_kernel_launches, 3);
    op.record_compute_event().unwrap().synchronize().unwrap();
    assert_eq!(op.download_i32_buffer(workspace.status()).unwrap(), [0]);
    assert_eq!(
        op.download_i32_buffer(workspace.token_ids()).unwrap(),
        [5, 3, 3, 3, 3, 3]
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "CUDA SM86+; relocated Hybrid owner method with opaque family scratch"]
fn relocated_hybrid_owner_keeps_status_reset_and_launch_counts() {
    use ferrule_backend::cuda::operators::{
        attention::hybrid::HybridMlaAttentionLayout, linear::CudaOperators,
    };
    let op = CudaOperators::new_on_device(0).unwrap();
    let query = op.zero_f32_buffer(5 * 64 * 512).unwrap();
    let context = op.zero_f32_buffer(16 * 512).unwrap();
    let block = op.zero_f32_buffer(5 * 512).unwrap();
    let slots = op.upload_i32_buffer(&[0]).unwrap();
    let sink = op.zero_f32_buffer(64).unwrap();
    let mut output = op.zero_f32_buffer(5 * 64 * 512).unwrap();
    let mut workspace = op.hybrid_mla_attention_workspace().unwrap();
    let layout = HybridMlaAttentionLayout {
        sequence_tokens: 4,
        page_tokens: 16,
        elements_per_token: 512,
        layer_index: 0,
        layer_count: 1,
        block_slot_offset: 0,
        block_slot_count: 1,
        softmax_scale: 1.0 / 512.0f32.sqrt(),
    };
    op.reset_counters();
    op.enable_capture_safe();
    op.hybrid_mla_attention_into(
        &query,
        &context,
        &block,
        &slots,
        &sink,
        layout,
        &mut output,
        &mut workspace,
    )
    .unwrap();
    op.disable_capture_safe();
    assert_eq!(op.counters().compute_kernel_launches, 2);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().stream_wide_syncs, 0);
    op.record_compute_event().unwrap().synchronize().unwrap();
    assert_eq!(op.download_i32_buffer(workspace.status()).unwrap(), [0]);
    assert!(
        op.download_f32_buffer(&output)
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
}

fn production_boundary(source: &str) -> Result<(), String> {
    let code: String = source
        .lines()
        .map(|line| line.split("//").next().unwrap())
        .flat_map(str::chars)
        .filter(|ch| !ch.is_whitespace())
        .collect();
    for forbidden in [
        "operators::legacy",
        "cutlass::legacy",
        "CutlassKernelId",
        "NumericFp8PrecisionExt",
        ".kernel(",
        "cutlass::*",
        "cutlass::bf16_compressor",
        "cutlass::fp8_projection",
        "cutlass::fp8_query_a_kv",
        "cutlass::hc_producer",
        "cutlass::main_project_norm",
        "cutlass::mla_output",
        "cutlass::shared_ffn",
    ] {
        if code.contains(forbidden) {
            return Err(format!(
                "production family crosses legacy boundary: {forbidden}"
            ));
        }
    }
    // Also catch renamed imports from a grouped use declaration.
    for imports in code.split("cutlass::{").skip(1) {
        let imports = imports.split('}').next().unwrap();
        for import in imports.split(',') {
            for name in [
                "bf16_compressor",
                "fp8_projection",
                "fp8_query_a_kv",
                "hc_producer",
                "main_project_norm",
                "mla_output",
                "shared_ffn",
            ] {
                if import == name || import.starts_with(&format!("{name}as")) {
                    return Err(format!("production family imports legacy {name}"));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn owner_resource_boundary_has_positive_and_negative_fixtures() {
    for legal in [
        "crate::cuda::providers::cutlass::submit_bf16_compressor(stream, input)",
        "use crate::cuda::operators::linear::NumericFp8Precision;",
        "// operators::legacy::bf16_compressor is forbidden in production",
    ] {
        production_boundary(legal).unwrap();
    }
    for illegal in [
        "crate::cuda::providers::cutlass::bf16_compressor(stream, input)",
        "use crate :: cuda :: providers :: cutlass :: { bf16_compressor as launch };",
        "use crate::cuda::operators::legacy as provider;",
        "use cutlass::{submit_mla_output}; use cutlass::{mla_output as launch};",
        "precision . kernel ( )",
        "use crate::cuda::providers::cutlass::*;",
    ] {
        assert!(production_boundary(illegal).is_err(), "accepted {illegal}");
    }
}

#[test]
fn owner_families_do_not_call_compatibility_launches_or_own_resources() {
    let families = [
        (
            "attention",
            include_str!("../src/cuda/operators/impls/attention.rs"),
        ),
        (
            "compressor",
            include_str!("../src/cuda/operators/impls/compressor.rs"),
        ),
        ("hc", include_str!("../src/cuda/operators/impls/hc.rs")),
        ("moe", include_str!("../src/cuda/operators/impls/moe.rs")),
        (
            "projection",
            include_str!("../src/cuda/operators/impls/projection.rs"),
        ),
        (
            "proposal",
            include_str!("../src/cuda/operators/impls/proposal.rs"),
        ),
        ("linear", include_str!("../src/cuda/operators/linear.rs")),
    ];
    let actual: std::collections::BTreeSet<_> = std::fs::read_dir(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cuda/operators/impls"),
    )
    .unwrap()
    .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
    .collect();
    let expected: std::collections::BTreeSet<_> = families
        .iter()
        .filter(|(name, _)| *name != "linear")
        .map(|(name, _)| format!("{name}.rs"))
        .chain(std::iter::once("mod.rs".to_owned()))
        .collect();
    assert_eq!(
        actual, expected,
        "new family implementations must join this boundary inventory"
    );
    for (name, source) in families {
        let source = source.split("#[cfg(test)]").next().unwrap();
        production_boundary(source).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(source.contains("self.submit_operator("), "{name}");
        for forbidden in [
            "self.stream",
            "self.counters",
            "self._ctx",
            "new_stream(",
            "Arc<CudaStream>",
            "CudaOpCounterCells",
            "workspace.poison(",
            ".buffer",
            "use super::",
            "pack_fp8_rows_from_f32_preallocated",
            "zero_i32_buffer_in_place",
        ] {
            assert!(!source.contains(forbidden), "{name}: {forbidden}");
        }
    }
    let contracts = include_str!("../src/cuda/operators/contracts.rs");
    let owner = contracts
        .split("pub(crate) trait OperatorOwner {")
        .nth(1)
        .unwrap()
        .split("/// Provider-neutral workspace size")
        .next()
        .unwrap();
    assert_eq!(owner.matches("fn ").count(), 1);
    assert!(owner.contains("FnOnce(&CudaStream) -> Result<T>"));
    assert!(!owner.contains("DeviceBuffer"));
    assert!(!owner.contains("Arc<"));
    // Numeric execution now compiles outside context-child privacy; resource
    // creation remains on the original context owner.
    assert!(
        !include_str!("../src/cuda/operators/numeric_fp8.rs")
            .contains("fn numeric_fp8_linear_into")
    );
    assert!(CONTEXT.contains("let result = submit(&self.stream)?;"));
    assert!(CONTEXT.contains("self.record_kernel_launches(launches);"));
    for name in [
        "bf16_compressor",
        "fp8_projection",
        "fp8_query_a_kv",
        "hc_producer",
        "main_project_norm",
        "mla_output",
        "shared_ffn",
    ] {
        assert!(PROVIDER.contains(&format!("pub(crate) fn submit_{name}(")));
        assert!(!PROVIDER.contains(&format!("pub fn {name}(")));
        assert!(LEGACY_PROVIDER.contains(&format!("pub fn {name}(")));
        assert!(LEGACY_PROVIDER.contains(&format!("super::submit_{name}(")));
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "actual CUDA sm80+; semantic owner mismatch, capture cleanup and replay for both numeric profiles"]
fn owner_numeric_rejects_foreign_resources_and_recovers_capture() {
    use ferrule_backend::cuda::operators::linear::{
        CudaOperators, NumericFp8Layout, NumericFp8LinearPlan, NumericFp8Precision,
        NumericFp8ScaleType,
    };
    let op = CudaOperators::new_on_device(0).expect("required CUDA owner");
    let other = CudaOperators::new_on_device(0).expect("required distinct same-ordinal owner");
    let layout = NumericFp8Layout {
        n: 3,
        k: 8,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let artifact = op
        .upload_numeric_fp8_linear(layout, &[0x38; 24], &1.0f32.to_le_bytes())
        .unwrap();
    let foreign_artifact = other
        .upload_numeric_fp8_linear(layout, &[0x38; 24], &1.0f32.to_le_bytes())
        .unwrap();
    let input = op.upload_f32_buffer(&[1.0; 8]).unwrap();
    let foreign_input = other.upload_f32_buffer(&[1.0; 8]).unwrap();
    for precision in [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::F32Tf32x3,
    ] {
        let plan = NumericFp8LinearPlan::new(layout, 1, 64, precision).unwrap();
        let mut scratch = op.numeric_fp8_linear_workspace(plan).unwrap();
        let mut foreign_scratch = other.numeric_fp8_linear_workspace(plan).unwrap();
        let mut output = op.upload_f32_buffer(&[73.0; 3]).unwrap();
        let mut foreign_output = other.upload_f32_buffer(&[73.0; 3]).unwrap();
        op.reset_counters();
        for (weight, activation) in [(&foreign_artifact, &input), (&artifact, &foreign_input)] {
            assert!(
                op.numeric_fp8_linear_into(
                    weight,
                    activation,
                    &mut output,
                    &mut scratch,
                    plan,
                    8,
                    3
                )
                .is_err()
            );
        }
        assert!(
            op.numeric_fp8_linear_into(
                &artifact,
                &input,
                &mut foreign_output,
                &mut scratch,
                plan,
                8,
                3
            )
            .is_err()
        );
        assert!(
            op.numeric_fp8_linear_into(
                &artifact,
                &input,
                &mut output,
                &mut foreign_scratch,
                plan,
                8,
                3
            )
            .is_err()
        );
        assert!(!scratch.is_poisoned());
        assert!(!foreign_scratch.is_poisoned());
        assert_eq!(op.counters().compute_kernel_launches, 0);
        assert_eq!(op.download_f32_buffer(&output).unwrap(), [73.0; 3]);
        assert_eq!(
            other.download_f32_buffer(&foreign_output).unwrap(),
            [73.0; 3]
        );
        op.reset_counters();
        // Native capture really begins; host preflight failure must still end
        // capture once and restore the assertion scope without poisoning scratch.
        let error = op
            .capture_decode_graph(|| {
                assert!(op.is_capture_safe());
                op.numeric_fp8_linear_into(
                    &artifact,
                    &foreign_input,
                    &mut output,
                    &mut scratch,
                    plan,
                    8,
                    3,
                )
            })
            .err()
            .expect("foreign input must fail during capture");
        assert!(!error.to_string().is_empty());
        assert!(!op.is_capture_safe());
        assert!(!scratch.is_poisoned());
        assert_eq!(op.counters().compute_kernel_launches, 0);
        let graph = op
            .capture_decode_graph(|| {
                op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut scratch, plan, 8, 3)
            })
            .unwrap();
        assert_eq!(
            op.counters().compute_kernel_launches,
            plan.kernel_launches() as u64
        );
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(op.counters().stream_wide_syncs, 0);
        assert!(other.launch_graph(&graph).is_err());
        assert!(other.upload_graph(&graph).is_err());
        for _ in 0..2 {
            op.launch_graph(&graph).unwrap();
            op.record_compute_event().unwrap().synchronize().unwrap();
            assert_eq!(op.download_f32_buffer(&output).unwrap(), [8.0; 3]);
        }
        assert!(!scratch.is_poisoned());
    }
}

#[test]
fn family_module_privacy_is_separate_from_context_resources() {
    let modules = include_str!("../src/cuda/mod.rs");
    let impls = include_str!("../src/cuda/operators/impls/mod.rs");
    let resources = include_str!("../src/cuda/operators/resources.rs");
    assert!(modules.contains("mod impls;"));
    assert!(!CONTEXT.contains("operators/impls/"));
    for family in [
        "attention",
        "compressor",
        "hc",
        "moe",
        "projection",
        "proposal",
    ] {
        assert!(impls.contains(&format!("mod {family};")));
        assert!(!CONTEXT.contains(&format!("mod {family}_impl;")));
    }
    assert!(CONTEXT.contains("operators/resources.rs"));
    assert!(CONTEXT.contains("mod operator_resources;"));
    assert!(!CONTEXT.contains("pub(crate) mod operator_resources"));
    for field in [
        "weight: DeviceBuffer<u8>",
        "x_packed: DeviceBuffer<u8>",
        "x_scales: DeviceBuffer<u8>",
        "cloned: CudaF32Buffer",
        "buffer: DeviceBuffer<T>",
    ] {
        assert!(CONTEXT.contains(field), "missing owner field {field}");
        assert!(
            !CONTEXT.contains(&format!("pub(crate) {field}")),
            "widened {field}"
        );
    }
    for declaration in [
        "struct ArtifactLinearRef<'a>",
        "struct Fp8StorageMut<'a>",
        "struct PreparedFp8ActivationRef<'a>",
    ] {
        assert!(resources.contains(declaration));
    }
    for forbidden in [
        "Arc<",
        "RefCell<",
        "Cell<",
        "CudaOpCounterCells",
        "clone()",
        "DeviceBuffer::",
        "pub(crate) cloned:",
        "unsafe",
        "new_stream",
        "synchronize(",
    ] {
        assert!(
            !resources.contains(forbidden),
            "resource adapter must not own/submit {forbidden}"
        );
    }
    assert!(resources.contains("weight: &'a DeviceBuffer<u8>"));
    assert!(resources.contains("x_packed: &'a mut DeviceBuffer<u8>"));
    assert!(resources.contains("scale: Option<&'a DeviceBuffer<u8>>"));
    assert!(resources.contains("self.pack_fp8_rows_from_f32_preallocated("));
    assert!(resources.contains("self.zero_i32_buffer_in_place(status)"));
}
