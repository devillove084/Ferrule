//! Model-neutral expert-I/O prediction values shared by model and runtime.

/// Execution phase for one expert-I/O scheduling candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertIoPhase {
    Prefill,
    Decode,
}

/// Predicted incremental expert cost for adding one candidate to the current
/// batch. Byte counts describe the union relative to already admitted work.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertIoEstimate {
    pub resident_union_bytes: u64,
    pub incremental_unique_bytes: u64,
    pub predicted_cold_bytes: u64,
    pub inflight_reusable_bytes: u64,
    pub inflight_reads: usize,
    pub pinned_slab_bytes: u64,
    pub upload_slots: usize,
    pub rejected_prefetch_risk: u32,
    pub confidence: f32,
    pub earliest_ready_in_us: u64,
    pub latency_debt_us: u64,
}

impl Default for ExpertIoEstimate {
    fn default() -> Self {
        Self {
            resident_union_bytes: 0,
            incremental_unique_bytes: 0,
            predicted_cold_bytes: 0,
            inflight_reusable_bytes: 0,
            inflight_reads: 0,
            pinned_slab_bytes: 0,
            upload_slots: 0,
            rejected_prefetch_risk: 0,
            confidence: 1.0,
            earliest_ready_in_us: 0,
            latency_debt_us: 0,
        }
    }
}
