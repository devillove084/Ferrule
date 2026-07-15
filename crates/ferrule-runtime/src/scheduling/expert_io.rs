//! Model-neutral expert-I/O admission contracts.
//!
//! The scheduler owns budgets and fairness; model-specific route/cache prediction
//! plugs in through [`ExpertIoAdvisor`]. Dense models and deployments without an
//! oracle use [`ZeroExpertIoAdvisor`], preserving ordinary token-budget behavior.

use std::collections::HashMap;

pub use ferrule_common::expert_io::{ExpertIoEstimate, ExpertIoPhase};
use ferrule_common::{Error, Result};
use ferrule_model::ExpertIoModelRunner;

use super::session::SessionId;

/// A request candidate presented to a model-specific expert-I/O oracle.
#[derive(Debug, Clone, Copy)]
pub struct ExpertIoCandidate<'a> {
    pub session_id: SessionId,
    pub phase: ExpertIoPhase,
    pub token_ids: &'a [u32],
}

/// Per-iteration expert-I/O limits. `unbounded` is the compatibility policy;
/// setting a numeric limit to zero means only zero-cost candidates are admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertIoBudget {
    pub max_incremental_expert_bytes: u64,
    pub max_inflight_expert_reads: usize,
    pub pinned_slab_budget_bytes: u64,
    pub upload_slot_budget: usize,
    pub io_deadline_us: u64,
    pub max_rejected_prefetch_risk: u32,
    pub max_latency_debt_us: u64,
    /// Permit one oldest candidate to exceed expert budgets when an otherwise
    /// non-empty runnable set would make no progress.
    pub allow_singleton_overflow: bool,
}

impl ExpertIoBudget {
    pub const fn unbounded() -> Self {
        Self {
            max_incremental_expert_bytes: u64::MAX,
            max_inflight_expert_reads: usize::MAX,
            pinned_slab_budget_bytes: u64::MAX,
            upload_slot_budget: usize::MAX,
            io_deadline_us: u64::MAX,
            max_rejected_prefetch_risk: u32::MAX,
            max_latency_debt_us: u64::MAX,
            allow_singleton_overflow: true,
        }
    }
}

impl Default for ExpertIoBudget {
    fn default() -> Self {
        Self::unbounded()
    }
}

/// Model-specific route/cache cost oracle.
pub trait ExpertIoAdvisor {
    /// Candidate-local state committed only after scheduler admission.
    type Admission;

    /// Compile-time switch used to erase all oracle/trace work from the default
    /// scheduler monomorphization.
    const ENABLED: bool = true;

    fn begin_batch(&mut self);

    fn estimate(
        &mut self,
        candidate: ExpertIoCandidate<'_>,
    ) -> Result<(ExpertIoEstimate, Self::Admission)>;

    fn admit(&mut self, admission: Self::Admission);
}

pub(crate) struct ModelExpertIoAdvisor<'a, R>
where
    R: ExpertIoModelRunner,
{
    runner: &'a R,
    states: &'a HashMap<SessionId, R::SequenceState>,
    batch: Option<R::ExpertIoBatchState>,
}

impl<'a, R> ModelExpertIoAdvisor<'a, R>
where
    R: ExpertIoModelRunner,
{
    pub(crate) fn new(runner: &'a R, states: &'a HashMap<SessionId, R::SequenceState>) -> Self {
        Self {
            runner,
            states,
            batch: None,
        }
    }
}

impl<R> ExpertIoAdvisor for ModelExpertIoAdvisor<'_, R>
where
    R: ExpertIoModelRunner,
{
    type Admission = R::ExpertIoAdmission;

    fn begin_batch(&mut self) {
        self.batch = Some(self.runner.begin_expert_io_batch());
    }

    fn estimate(
        &mut self,
        candidate: ExpertIoCandidate<'_>,
    ) -> Result<(ExpertIoEstimate, Self::Admission)> {
        let state = self.states.get(&candidate.session_id).ok_or_else(|| {
            Error::Internal(format!(
                "expert-I/O candidate {:?} has no model sequence state",
                candidate.session_id
            ))
        })?;
        let batch = self.batch.as_mut().ok_or_else(|| {
            Error::Internal("expert-I/O batch was not initialized before estimation".into())
        })?;
        self.runner
            .estimate_expert_io(batch, state, candidate.phase, candidate.token_ids)
    }

    fn admit(&mut self, admission: Self::Admission) {
        if let Some(batch) = self.batch.as_mut() {
            self.runner.admit_expert_io(batch, admission);
        }
    }
}

/// Dense/no-oracle compatibility implementation.
#[derive(Debug, Default)]
pub struct ZeroExpertIoAdvisor;

impl ExpertIoAdvisor for ZeroExpertIoAdvisor {
    type Admission = ();

    const ENABLED: bool = false;

    fn begin_batch(&mut self) {}

    fn estimate(
        &mut self,
        _candidate: ExpertIoCandidate<'_>,
    ) -> Result<(ExpertIoEstimate, Self::Admission)> {
        Ok((ExpertIoEstimate::default(), ()))
    }

    fn admit(&mut self, _admission: Self::Admission) {}
}

/// Logical queue selected by expert-I/O admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertIoQueueClass {
    ResidentReady,
    IoAdmissible,
    MissBlocked,
    PrefillReady,
}

/// First budget that rejected a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertIoRejection {
    IncrementalBytes,
    InflightReads,
    PinnedSlab,
    UploadSlots,
    IoDeadline,
    RejectedPrefetchRisk,
    LatencyDebt,
    InvalidConfidence,
}

/// Reproducible scheduling decision for one inspected candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertIoDecisionTrace {
    pub session_id: SessionId,
    pub phase: ExpertIoPhase,
    pub queue: ExpertIoQueueClass,
    pub admitted: bool,
    pub forced_progress: bool,
    pub estimate: ExpertIoEstimate,
    pub rejection: Option<ExpertIoRejection>,
}

#[derive(Debug, Default)]
pub(super) struct ExpertIoBatchUsage {
    incremental_expert_bytes: u64,
    inflight_reads: usize,
    pinned_slab_bytes: u64,
    upload_slots: usize,
    rejected_prefetch_risk: u32,
    latency_debt_us: u64,
}

impl ExpertIoBatchUsage {
    pub(super) fn inspect(
        &self,
        budget: ExpertIoBudget,
        estimate: ExpertIoEstimate,
    ) -> Option<ExpertIoRejection> {
        if !estimate.confidence.is_finite() || !(0.0..=1.0).contains(&estimate.confidence) {
            return Some(ExpertIoRejection::InvalidConfidence);
        }
        if self
            .incremental_expert_bytes
            .saturating_add(estimate.incremental_unique_bytes)
            > budget.max_incremental_expert_bytes
        {
            return Some(ExpertIoRejection::IncrementalBytes);
        }
        if self.inflight_reads.saturating_add(estimate.inflight_reads)
            > budget.max_inflight_expert_reads
        {
            return Some(ExpertIoRejection::InflightReads);
        }
        if self
            .pinned_slab_bytes
            .saturating_add(estimate.pinned_slab_bytes)
            > budget.pinned_slab_budget_bytes
        {
            return Some(ExpertIoRejection::PinnedSlab);
        }
        if self.upload_slots.saturating_add(estimate.upload_slots) > budget.upload_slot_budget {
            return Some(ExpertIoRejection::UploadSlots);
        }
        if estimate.earliest_ready_in_us > budget.io_deadline_us {
            return Some(ExpertIoRejection::IoDeadline);
        }
        if self
            .rejected_prefetch_risk
            .saturating_add(estimate.rejected_prefetch_risk)
            > budget.max_rejected_prefetch_risk
        {
            return Some(ExpertIoRejection::RejectedPrefetchRisk);
        }
        if self
            .latency_debt_us
            .saturating_add(estimate.latency_debt_us)
            > budget.max_latency_debt_us
        {
            return Some(ExpertIoRejection::LatencyDebt);
        }
        None
    }

    pub(super) fn admit(&mut self, estimate: ExpertIoEstimate) {
        self.incremental_expert_bytes = self
            .incremental_expert_bytes
            .saturating_add(estimate.incremental_unique_bytes);
        self.inflight_reads = self.inflight_reads.saturating_add(estimate.inflight_reads);
        self.pinned_slab_bytes = self
            .pinned_slab_bytes
            .saturating_add(estimate.pinned_slab_bytes);
        self.upload_slots = self.upload_slots.saturating_add(estimate.upload_slots);
        self.rejected_prefetch_risk = self
            .rejected_prefetch_risk
            .saturating_add(estimate.rejected_prefetch_risk);
        self.latency_debt_us = self
            .latency_debt_us
            .saturating_add(estimate.latency_debt_us);
    }
}

pub(super) fn classify_admitted(
    phase: ExpertIoPhase,
    estimate: ExpertIoEstimate,
) -> ExpertIoQueueClass {
    if phase == ExpertIoPhase::Prefill {
        ExpertIoQueueClass::PrefillReady
    } else if estimate.incremental_unique_bytes == 0 && estimate.predicted_cold_bytes == 0 {
        ExpertIoQueueClass::ResidentReady
    } else {
        ExpertIoQueueClass::IoAdmissible
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_advisor_preserves_unbounded_admission() {
        let (estimate, ()) = ZeroExpertIoAdvisor
            .estimate(ExpertIoCandidate {
                session_id: SessionId(1),
                phase: ExpertIoPhase::Decode,
                token_ids: &[7],
            })
            .unwrap();
        assert_eq!(
            ExpertIoBatchUsage::default().inspect(ExpertIoBudget::unbounded(), estimate),
            None
        );
        assert_eq!(
            classify_admitted(ExpertIoPhase::Decode, estimate),
            ExpertIoQueueClass::ResidentReady
        );
    }

    #[test]
    fn byte_budget_blocks_incremental_miss() {
        let estimate = ExpertIoEstimate {
            incremental_unique_bytes: 65,
            predicted_cold_bytes: 65,
            ..Default::default()
        };
        let budget = ExpertIoBudget {
            max_incremental_expert_bytes: 64,
            ..ExpertIoBudget::unbounded()
        };
        assert_eq!(
            ExpertIoBatchUsage::default().inspect(budget, estimate),
            Some(ExpertIoRejection::IncrementalBytes)
        );
    }
}
