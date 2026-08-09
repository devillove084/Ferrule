//! Private symbols for the CUTLASS CUDA provider.

use crate::cuda::providers::cutlass::{
    CutlassBf16CompressorArgs, CutlassFp8QueryAKvArgs, CutlassHcProducerArgs,
    CutlassHybridMlaAttentionArgs, CutlassMainProjectNormArgs, CutlassMlaOutputArgs,
    CutlassProposalHeadArgs, CutlassProviderManifest, CutlassSharedFfnArgs,
    FerruleCutlassGroupedFp4MoeArgs, FerruleCutlassHybridMlaExplicitSelectionArgs,
    PrepareMxfp4SfbArgs,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct CutlassWorkspaceRequirements {
    pub(crate) bytes: u64,
    pub(crate) alignment: u32,
    pub(crate) reserved: u32,
}

unsafe extern "C" {
    pub fn ferrule_cutlass_provider_manifest() -> CutlassProviderManifest;
    pub fn ferrule_cutlass_bf16_compressor_can_implement(
        args: *const CutlassBf16CompressorArgs,
    ) -> i32;
    pub fn ferrule_cutlass_bf16_compressor_launch(args: *const CutlassBf16CompressorArgs) -> i32;
    pub fn ferrule_cutlass_fp8_query_a_kv_can_implement(args: *const CutlassFp8QueryAKvArgs)
    -> i32;
    pub fn ferrule_cutlass_fp8_query_a_kv_launch(args: *const CutlassFp8QueryAKvArgs) -> i32;
    pub fn ferrule_cutlass_fp8_projection_can_implement(args: *const CutlassFp8QueryAKvArgs)
    -> i32;
    pub fn ferrule_cutlass_fp8_projection_launch(args: *const CutlassFp8QueryAKvArgs) -> i32;
    pub fn ferrule_cutlass_main_project_norm_can_implement(
        args: *const CutlassMainProjectNormArgs,
    ) -> i32;
    pub fn ferrule_cutlass_main_project_norm_launch(args: *const CutlassMainProjectNormArgs)
    -> i32;
    pub fn ferrule_cutlass_hybrid_mla_attention_can_implement(
        args: *const CutlassHybridMlaAttentionArgs,
    ) -> i32;
    pub fn ferrule_cutlass_hybrid_mla_attention_launch(
        args: *const CutlassHybridMlaAttentionArgs,
    ) -> i32;
    pub fn ferrule_cutlass_hybrid_mla_explicit_selection_workspace_requirements(
        args: *const FerruleCutlassHybridMlaExplicitSelectionArgs,
        requirements: *mut CutlassWorkspaceRequirements,
    ) -> i32;
    pub fn ferrule_cutlass_hybrid_mla_explicit_selection_can_implement(
        args: *const FerruleCutlassHybridMlaExplicitSelectionArgs,
    ) -> i32;
    pub fn ferrule_cutlass_hybrid_mla_explicit_selection_launch(
        args: *const FerruleCutlassHybridMlaExplicitSelectionArgs,
    ) -> i32;
    #[cfg(ferrule_cuda_test_oracle)]
    pub fn ferrule_cutlass_test_hybrid_mla_explicit_selection_scalar_launch(
        args: *const FerruleCutlassHybridMlaExplicitSelectionArgs,
    ) -> i32;
    #[cfg(ferrule_cuda_test_oracle)]
    pub fn ferrule_cutlass_test_hybrid_mla_explicit_selection_compare_launch(
        args: *const FerruleCutlassHybridMlaExplicitSelectionArgs,
        oracle_output_f32: u64,
        compare_result_i32: u64,
    ) -> i32;
    pub fn ferrule_cutlass_proposal_head_can_implement(args: *const CutlassProposalHeadArgs)
    -> i32;
    pub fn ferrule_cutlass_proposal_head_launch(args: *const CutlassProposalHeadArgs) -> i32;
    pub fn ferrule_cutlass_hc_producer_can_implement(args: *const CutlassHcProducerArgs) -> i32;
    pub fn ferrule_cutlass_hc_producer_launch(args: *const CutlassHcProducerArgs) -> i32;
    pub fn ferrule_cutlass_shared_ffn_can_implement(args: *const CutlassSharedFfnArgs) -> i32;
    pub fn ferrule_cutlass_shared_ffn_launch(args: *const CutlassSharedFfnArgs) -> i32;
    pub fn ferrule_cutlass_mla_output_can_implement(args: *const CutlassMlaOutputArgs) -> i32;
    pub fn ferrule_cutlass_mla_output_launch(args: *const CutlassMlaOutputArgs) -> i32;
    pub fn ferrule_cutlass_grouped_fp4_moe_workspace_size(
        args: *const FerruleCutlassGroupedFp4MoeArgs,
    ) -> u64;
    pub fn ferrule_cutlass_grouped_fp4_moe_can_implement(
        args: *const FerruleCutlassGroupedFp4MoeArgs,
    ) -> i32;
    pub fn ferrule_cutlass_grouped_fp4_moe_launch(
        args: *const FerruleCutlassGroupedFp4MoeArgs,
    ) -> i32;
    pub fn ferrule_cutlass_mxfp4_sfb_storage_bytes(n: u32, k: u32) -> u64;
    pub fn ferrule_cutlass_prepare_mxfp4_sfb(args: *const PrepareMxfp4SfbArgs) -> i32;
}
