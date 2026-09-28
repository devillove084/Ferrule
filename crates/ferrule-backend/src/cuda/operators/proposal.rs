//! Semantic checkpoint-native proposal-head contracts.

/// Number of proposal rows emitted by the checkpoint-native proposal head.
pub const PROPOSAL_ROWS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProposalHeadLayout {
    pub rows: usize,
    pub hc: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub markov_rank: usize,
    pub partial_capacity: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
}

use crate::cuda::context::{CudaF32Buffer, CudaI32Buffer, CudaI32HostMirror};

/// Low-level compatibility submission; new code uses
/// `CudaOperators::artifact_proposal_head_into` with this family's workspace.
pub use crate::cuda::providers::cutlass::proposal_head;

/// Graph-stable outputs and reduction scratch for the checkpoint-native proposal
/// HC/LM/Markov/confidence semantic bundle.
/// Backend-private storage remains inaccessible through semantic and legacy paths.
/// ```compile_fail,E0616
/// use ferrule_backend::cuda::operators::proposal::CudaProposalHeadWorkspace;
/// fn access(workspace: &CudaProposalHeadWorkspace) { let _ = &workspace.hidden; }
/// ```
pub struct CudaProposalHeadWorkspace {
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS hidden scratch alive")
    )]
    pub(crate) hidden: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS normalization scratch alive")
    )]
    pub(crate) normalized: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS logits scratch alive")
    )]
    pub(crate) base_logits: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS reduction values alive")
    )]
    pub(crate) partial_values: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS reduction indices alive")
    )]
    pub(crate) partial_indices: CudaI32Buffer,
    pub(crate) token_ids: CudaI32HostMirror,
    pub(crate) confidence: CudaF32Buffer,
    pub(crate) status: CudaI32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps the CUTLASS result mirror alive")
    )]
    pub(crate) result: CudaI32HostMirror,
}

impl CudaProposalHeadWorkspace {
    pub fn token_ids(&self) -> &CudaI32Buffer {
        self.token_ids.device()
    }

    pub fn confidence(&self) -> &CudaF32Buffer {
        &self.confidence
    }

    pub fn status(&self) -> &CudaI32Buffer {
        &self.status
    }
}
