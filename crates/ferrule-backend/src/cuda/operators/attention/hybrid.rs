//! Semantic hybrid MLA layouts and caller-owned launch buffers.

use crate::cuda::runtime::DeviceBuffer;

pub const HYBRID_MLA_ATTENTION_HEADS: usize = 64;
pub const HYBRID_MLA_ATTENTION_HEAD_DIM: usize = 512;
pub const HYBRID_MLA_ATTENTION_WINDOW: usize = 128;
pub const HYBRID_MLA_ATTENTION_PAGE_TOKENS: usize = 16;
pub const HYBRID_MLA_ATTENTION_TOKEN_CAPACITY: usize =
    HYBRID_MLA_ATTENTION_WINDOW + crate::cuda::operators::proposal::PROPOSAL_ROWS;
/// Legacy scratch-tiling constant; owner callers use the opaque workspace.
pub const HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILE: usize = 64;
/// Legacy scratch-tiling constant; not a provider-neutral planning promise.
pub const HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILES: usize =
    HYBRID_MLA_ATTENTION_TOKEN_CAPACITY.div_ceil(HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILE);
pub const HYBRID_MLA_EXPLICIT_SELECTION_MAXIMUM_WIDTH: usize = 640;
#[cfg(ferrule_cuda_test_oracle)]
pub const HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE_RESULT_WORDS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridMlaAttentionLayout {
    pub sequence_tokens: usize,
    pub page_tokens: usize,
    pub elements_per_token: usize,
    pub layer_index: usize,
    pub layer_count: usize,
    pub block_slot_offset: usize,
    pub block_slot_count: usize,
    pub softmax_scale: f32,
}

/// KV storage topology for hybrid MLA with explicit selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum HybridMlaKvStorageKind {
    Contiguous = 1,
    Paged = 2,
    DualPaged = 3,
}

/// Dimensions and paging metadata for hybrid MLA with explicit selection.
/// Fields unused by a storage topology must be zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridMlaExplicitSelectionLayout {
    pub kind: HybridMlaKvStorageKind,
    pub rows: usize,
    pub tokens_per_sequence: usize,
    pub kv_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub selected_width: usize,
    pub page_tokens: usize,
    pub first_elements_per_token: usize,
    pub second_elements_per_token: usize,
    pub layer_index: usize,
    pub layer_count: usize,
    pub row_sequence_ids: bool,
    pub row_kv_lens: bool,
    pub softmax_scale: f32,
}

/// Low-level compatibility buffer view. New owner callers use the opaque
/// `CudaHybridMlaExplicitSelectionWorkspace` rather than supplying raw scratch.
/// Caller-owned inputs, outputs, metadata, and opaque provider workspace for
/// hybrid MLA with explicit selection. Paged metadata is present only for paged
/// topologies; the second plane and selectors are present only for `DualPaged`.
pub struct HybridMlaExplicitSelectionBuffers<'a> {
    pub query: &'a DeviceBuffer<f32>,
    #[cfg(ferrule_cuda_test_oracle)]
    pub oracle_output: &'a mut DeviceBuffer<f32>,
    pub first_plane: &'a DeviceBuffer<f32>,
    pub second_plane: Option<&'a DeviceBuffer<f32>>,
    pub block_slots: Option<&'a DeviceBuffer<i32>>,
    pub block_offsets: Option<&'a DeviceBuffer<i32>>,
    pub sequence_kv_lens: Option<&'a DeviceBuffer<i32>>,
    pub second_sequence_kv_lens: Option<&'a DeviceBuffer<i32>>,
    pub row_sequence_ids: Option<&'a DeviceBuffer<i32>>,
    pub row_kv_lens: Option<&'a DeviceBuffer<i32>>,
    pub row_second_kv_lens: Option<&'a DeviceBuffer<i32>>,
    pub selected_indices: &'a DeviceBuffer<i32>,
    pub selectors: Option<&'a DeviceBuffer<i32>>,
    pub attention_sink: &'a DeviceBuffer<f32>,
    pub workspace: &'a mut DeviceBuffer<u8>,
    pub output: &'a mut DeviceBuffer<f32>,
    pub status: &'a mut DeviceBuffer<i32>,
}

use crate::cuda::context::{CudaF32Buffer, CudaI32Buffer};
use crate::cuda::operators::OperatorWorkspaceRequirements;
use ferrule_common::{Error, Result};

/// Query opaque scratch; native layout remains private to the selected provider.
pub fn workspace_requirements(
    layout: HybridMlaExplicitSelectionLayout,
) -> Result<OperatorWorkspaceRequirements> {
    crate::cuda::providers::cutlass::hybrid_mla_explicit_selection_workspace_requirements(layout)
}

/// Low-level stream/buffer entry points retained for existing callers. The
/// family owns the contracts; native ABI lowering/submission stays private.
pub use crate::cuda::providers::cutlass::{
    hybrid_mla_attention, hybrid_mla_explicit_selection_can_implement,
    hybrid_mla_explicit_selection_launch,
};

/// Graph-stable scratch for one checkpoint-native proposal hybrid-attention launch.
///
/// The five-row query and block KV remain caller-owned stage values. This workspace
/// owns only the BF16 boundaries, score/probability matrices, output, and device
/// status needed by the semantic CUTLASS bundle.
/// Backend-private storage remains inaccessible through semantic and legacy paths.
/// ```compile_fail,E0616
/// use ferrule_backend::cuda::operators::attention::hybrid::CudaHybridMlaAttentionWorkspace;
/// fn access(workspace: &CudaHybridMlaAttentionWorkspace) { let _ = &workspace.scores; }
/// ```
pub struct CudaHybridMlaAttentionWorkspace {
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS query scratch alive")
    )]
    pub(crate) query_bf16: DeviceBuffer<u16>,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS KV scratch alive")
    )]
    pub(crate) gathered_kv_bf16: DeviceBuffer<u16>,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS score scratch alive")
    )]
    pub(crate) scores: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS probability scratch alive")
    )]
    pub(crate) probabilities_bf16: DeviceBuffer<u16>,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(
            dead_code,
            reason = "keeps native CUTLASS online-softmax rescale scratch alive"
        )
    )]
    pub(crate) online_rescales: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(
            dead_code,
            reason = "keeps native CUTLASS softmax denominator scratch alive"
        )
    )]
    pub(crate) denominators: CudaF32Buffer,
    pub(crate) status: CudaI32Buffer,
}

impl CudaHybridMlaAttentionWorkspace {
    pub(crate) fn storage_lengths() -> Result<[usize; 5]> {
        let output_values = crate::cuda::operators::proposal::PROPOSAL_ROWS
            .checked_mul(HYBRID_MLA_ATTENTION_HEADS)
            .and_then(|value| value.checked_mul(HYBRID_MLA_ATTENTION_HEAD_DIM))
            .ok_or_else(|| Error::Internal {
                message: "proposal attention output size overflow".into(),
            })?;
        let score_values = crate::cuda::operators::proposal::PROPOSAL_ROWS
            .checked_mul(HYBRID_MLA_ATTENTION_HEADS)
            .and_then(|value| value.checked_mul(HYBRID_MLA_ATTENTION_TOKEN_CAPACITY))
            .ok_or_else(|| Error::Internal {
                message: "proposal attention score size overflow".into(),
            })?;
        let gathered_values = HYBRID_MLA_ATTENTION_TOKEN_CAPACITY
            .checked_mul(HYBRID_MLA_ATTENTION_HEAD_DIM)
            .ok_or_else(|| Error::Internal {
                message: "proposal gathered KV size overflow".into(),
            })?;
        let pair_values = crate::cuda::operators::proposal::PROPOSAL_ROWS
            .checked_mul(HYBRID_MLA_ATTENTION_HEADS)
            .ok_or_else(|| Error::Internal {
                message: "proposal attention row/head size overflow".into(),
            })?;
        let rescale_values = pair_values
            .checked_mul(HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILES)
            .ok_or_else(|| Error::Internal {
                message: "proposal attention online-softmax size overflow".into(),
            })?;
        Ok([
            output_values,
            score_values,
            gathered_values,
            pair_values,
            rescale_values,
        ])
    }

    pub fn status(&self) -> &CudaI32Buffer {
        &self.status
    }
}

/// Backend-private storage remains inaccessible through semantic and legacy paths.
/// ```compile_fail,E0616
/// use ferrule_backend::cuda::context::CudaHybridMlaExplicitSelectionWorkspace;
/// fn access(workspace: &CudaHybridMlaExplicitSelectionWorkspace) { let _ = &workspace.storage; }
/// ```
pub struct CudaHybridMlaExplicitSelectionWorkspace {
    pub(crate) storage: DeviceBuffer<u8>,
    pub(crate) status: CudaI32Buffer,
    pub(crate) capacity_bytes: usize,
    pub(crate) alignment: usize,
    pub(crate) allocated_layout: HybridMlaExplicitSelectionLayout,
    #[cfg(ferrule_cuda_test_oracle)]
    pub(crate) oracle_output: DeviceBuffer<f32>,
}

impl CudaHybridMlaExplicitSelectionWorkspace {
    pub fn status(&self) -> &CudaI32Buffer {
        &self.status
    }

    pub(crate) fn supports(&self, layout: HybridMlaExplicitSelectionLayout) -> Result<bool> {
        let requirements = workspace_requirements(layout)?;
        let required_bytes = usize::try_from(requirements.bytes).map_err(|_| Error::Internal {
            message: format!(
                "hybrid MLA explicit selection workspace requirement exceeds usize: {}",
                requirements.bytes
            ),
        })?;
        let required_alignment = requirements.alignment as usize;
        Ok(required_bytes <= self.capacity_bytes
            && required_alignment <= self.alignment
            && self
                .storage
                .cu_deviceptr()
                .is_multiple_of(requirements.alignment.into()))
    }
}
