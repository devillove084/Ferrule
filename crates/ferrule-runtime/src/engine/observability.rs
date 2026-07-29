//! Generic observability for the resident runtime driver.

use crate::speculation::SpeculativeMetrics;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidentTopKDriverStats {
    pub actions: usize,
    pub prefill_chunks: usize,
    pub prefill_tokens: usize,
    pub decode_steps: usize,
    pub emitted_tokens: usize,
    pub staged_tokens: usize,
    pub finished_sequences: usize,
    pub hard_resource_high_water: Vec<(crate::scheduling::ResourceKind, u64)>,
    pub speculative: SpeculativeMetrics,
}

#[derive(Debug, Default)]
pub(crate) struct ResidentDriverObservability {
    pub(crate) stats: ResidentTopKDriverStats,
}

impl ResidentDriverObservability {
    pub(crate) fn stats(&self) -> &ResidentTopKDriverStats {
        &self.stats
    }
}
