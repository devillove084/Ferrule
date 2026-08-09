//! Provider-neutral attention operators and layouts.

use ferrule_common::Result;

pub use crate::cuda::context::{
    CombinedRingWindowLens, CudaBf16Buffer, CudaF32Buffer, CudaHybridMlaAttentionWorkspace,
    CudaHybridMlaExplicitSelectionWorkspace, CudaI32Buffer, CudaI32HostDownload, CudaI32HostMirror,
    CudaOperators, CudaProposalHeadWorkspace, cuda_sparse_attention_sink_f32,
};
pub use crate::cuda::operators::contracts::{
    PAGED_BF16_TRANSFORMER_ADDRESS_ERROR, PAGED_BF16_TRANSFORMER_METADATA_ERROR,
    PagedBf16CausalGqaLayout, PagedBf16PlaneLayout, validate_paged_bf16_transformer_status,
};
pub use crate::cuda::providers::cutlass::{
    HYBRID_MLA_ATTENTION_HEAD_DIM, HYBRID_MLA_ATTENTION_HEADS,
    HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILE, HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILES,
    HYBRID_MLA_ATTENTION_PAGE_TOKENS, HYBRID_MLA_ATTENTION_TOKEN_CAPACITY,
    HYBRID_MLA_ATTENTION_WINDOW, HYBRID_MLA_EXPLICIT_SELECTION_MAXIMUM_WIDTH,
    HybridMlaAttentionLayout, HybridMlaExplicitSelectionLayout, HybridMlaKvStorageKind,
};

pub mod selection;
pub mod sparse;

/// Device buffers for one packed/ragged paged-attention transaction.
pub struct PagedBf16GqaBuffers<'a> {
    pub query: &'a CudaF32Buffer,
    pub append_key: &'a CudaBf16Buffer,
    pub append_value: &'a CudaBf16Buffer,
    pub key_cache: &'a mut CudaBf16Buffer,
    pub value_cache: &'a mut CudaBf16Buffer,
    pub block_slots: &'a CudaI32Buffer,
    pub block_offsets: &'a CudaI32Buffer,
    pub row_sequence_ids: &'a CudaI32Buffer,
    pub row_positions: &'a CudaI32Buffer,
    pub row_kv_lens: &'a CudaI32Buffer,
    pub output: &'a mut CudaF32Buffer,
    pub status: &'a mut CudaI32Buffer,
}

impl CudaOperators {
    /// Append BF16 K/V rows and execute causal grouped-query attention.
    pub fn append_and_attend_paged_bf16(
        &self,
        buffers: PagedBf16GqaBuffers<'_>,
        layout: PagedBf16CausalGqaLayout,
    ) -> Result<()> {
        let PagedBf16GqaBuffers {
            query,
            append_key,
            append_value,
            key_cache,
            value_cache,
            block_slots,
            block_offsets,
            row_sequence_ids,
            row_positions,
            row_kv_lens,
            output,
            status,
        } = buffers;
        self.paged_bf16_append_causal_gqa_from_device_into(
            query,
            append_key,
            append_value,
            key_cache,
            value_cache,
            block_slots,
            block_offsets,
            row_sequence_ids,
            row_positions,
            row_kv_lens,
            output,
            status,
            layout,
        )?;
        self.check_paged_bf16_transformer_status(status)
    }
}
