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
/// These are predictive admission limits, not physical allocator guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertIoBudget {
    pub max_incremental_expert_bytes: u64,
    pub max_inflight_expert_reads: usize,
    pub pinned_slab_budget_bytes: u64,
    pub upload_slot_budget: usize,
    pub max_storage_read_bytes: u64,
    pub max_read_ops: usize,
    pub max_pageable_host_bytes: u64,
    pub max_pinned_host_bytes: u64,
    pub max_h2d_bytes: u64,
    pub max_device_install_bytes: u64,
    pub io_deadline_us: u64,
    pub max_rejected_prefetch_risk: u32,
    pub max_latency_debt_us: u64,
    /// Maximum times one session may be rejected by predictive expert-I/O
    /// limits before it becomes eligible for forced progress. `usize::MAX`
    /// disables age-based overflow.
    pub max_expert_io_deferrals: usize,
    /// Permit at most one oldest eligible candidate per batch to exceed the
    /// predictive expert-I/O limits. Hard token and batch limits still apply.
    pub allow_singleton_overflow: bool,
}

impl ExpertIoBudget {
    pub const fn unbounded() -> Self {
        Self {
            max_incremental_expert_bytes: u64::MAX,
            max_inflight_expert_reads: usize::MAX,
            pinned_slab_budget_bytes: u64::MAX,
            upload_slot_budget: usize::MAX,
            max_storage_read_bytes: u64::MAX,
            max_read_ops: usize::MAX,
            max_pageable_host_bytes: u64::MAX,
            max_pinned_host_bytes: u64::MAX,
            max_h2d_bytes: u64::MAX,
            max_device_install_bytes: u64::MAX,
            io_deadline_us: u64::MAX,
            max_rejected_prefetch_risk: u32::MAX,
            max_latency_debt_us: u64::MAX,
            max_expert_io_deferrals: usize::MAX,
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
    StorageReadBytes,
    ReadOps,
    PageableHostBytes,
    PinnedHostBytes,
    H2dBytes,
    UploadSlots,
    DeviceInstallBytes,
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
    storage_read_bytes: u64,
    read_ops: usize,
    pageable_host_bytes: u64,
    pinned_host_bytes: u64,
    h2d_bytes: u64,
    upload_slots: usize,
    device_install_bytes: u64,
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
            .storage_read_bytes
            .saturating_add(estimate.storage_read_bytes)
            > budget.max_storage_read_bytes
        {
            return Some(ExpertIoRejection::StorageReadBytes);
        }
        if self.read_ops.saturating_add(estimate.read_ops) > budget.max_read_ops {
            return Some(ExpertIoRejection::ReadOps);
        }
        if self
            .pageable_host_bytes
            .saturating_add(estimate.pageable_host_bytes)
            > budget.max_pageable_host_bytes
        {
            return Some(ExpertIoRejection::PageableHostBytes);
        }
        if self
            .pinned_host_bytes
            .saturating_add(estimate.pinned_host_bytes)
            > budget.max_pinned_host_bytes
        {
            return Some(ExpertIoRejection::PinnedHostBytes);
        }
        if self.h2d_bytes.saturating_add(estimate.h2d_bytes) > budget.max_h2d_bytes {
            return Some(ExpertIoRejection::H2dBytes);
        }
        if self.upload_slots.saturating_add(estimate.upload_slots) > budget.upload_slot_budget {
            return Some(ExpertIoRejection::UploadSlots);
        }
        if self
            .device_install_bytes
            .saturating_add(estimate.device_install_bytes)
            > budget.max_device_install_bytes
        {
            return Some(ExpertIoRejection::DeviceInstallBytes);
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
        self.storage_read_bytes = self
            .storage_read_bytes
            .saturating_add(estimate.storage_read_bytes);
        self.read_ops = self.read_ops.saturating_add(estimate.read_ops);
        self.pageable_host_bytes = self
            .pageable_host_bytes
            .saturating_add(estimate.pageable_host_bytes);
        self.pinned_host_bytes = self
            .pinned_host_bytes
            .saturating_add(estimate.pinned_host_bytes);
        self.h2d_bytes = self.h2d_bytes.saturating_add(estimate.h2d_bytes);
        self.upload_slots = self.upload_slots.saturating_add(estimate.upload_slots);
        self.device_install_bytes = self
            .device_install_bytes
            .saturating_add(estimate.device_install_bytes);
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
    } else if estimate.incremental_unique_bytes == 0
        && estimate.predicted_cold_bytes == 0
        && estimate.storage_read_bytes == 0
        && estimate.read_ops == 0
        && estimate.pageable_host_bytes == 0
        && estimate.pinned_host_bytes == 0
        && estimate.h2d_bytes == 0
        && estimate.upload_slots == 0
        && estimate.device_install_bytes == 0
    {
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
    fn every_additive_budget_is_inclusive_and_rejects_plus_one() {
        macro_rules! assert_dimension_boundary {
            ($estimate_field:ident, $budget_field:ident, $rejection:expr) => {{
                let mut usage = ExpertIoBatchUsage::default();
                usage.admit(ExpertIoEstimate {
                    $estimate_field: 3,
                    ..Default::default()
                });
                let budget = ExpertIoBudget {
                    $budget_field: 7,
                    ..ExpertIoBudget::unbounded()
                };
                assert_eq!(
                    usage.inspect(
                        budget,
                        ExpertIoEstimate {
                            $estimate_field: 4,
                            ..Default::default()
                        }
                    ),
                    None,
                    "{} must admit its exact limit",
                    stringify!($estimate_field)
                );
                assert_eq!(
                    usage.inspect(
                        budget,
                        ExpertIoEstimate {
                            $estimate_field: 5,
                            ..Default::default()
                        }
                    ),
                    Some($rejection),
                    "{} must reject limit + 1",
                    stringify!($estimate_field)
                );
            }};
        }

        assert_dimension_boundary!(
            incremental_unique_bytes,
            max_incremental_expert_bytes,
            ExpertIoRejection::IncrementalBytes
        );
        assert_dimension_boundary!(
            inflight_reads,
            max_inflight_expert_reads,
            ExpertIoRejection::InflightReads
        );
        assert_dimension_boundary!(
            pinned_slab_bytes,
            pinned_slab_budget_bytes,
            ExpertIoRejection::PinnedSlab
        );
        assert_dimension_boundary!(
            storage_read_bytes,
            max_storage_read_bytes,
            ExpertIoRejection::StorageReadBytes
        );
        assert_dimension_boundary!(read_ops, max_read_ops, ExpertIoRejection::ReadOps);
        assert_dimension_boundary!(
            pageable_host_bytes,
            max_pageable_host_bytes,
            ExpertIoRejection::PageableHostBytes
        );
        assert_dimension_boundary!(
            pinned_host_bytes,
            max_pinned_host_bytes,
            ExpertIoRejection::PinnedHostBytes
        );
        assert_dimension_boundary!(h2d_bytes, max_h2d_bytes, ExpertIoRejection::H2dBytes);
        assert_dimension_boundary!(
            upload_slots,
            upload_slot_budget,
            ExpertIoRejection::UploadSlots
        );
        assert_dimension_boundary!(
            device_install_bytes,
            max_device_install_bytes,
            ExpertIoRejection::DeviceInstallBytes
        );
        assert_dimension_boundary!(
            rejected_prefetch_risk,
            max_rejected_prefetch_risk,
            ExpertIoRejection::RejectedPrefetchRisk
        );
        assert_dimension_boundary!(
            latency_debt_us,
            max_latency_debt_us,
            ExpertIoRejection::LatencyDebt
        );
    }

    #[test]
    fn deadline_budget_is_inclusive_and_rejects_plus_one() {
        let budget = ExpertIoBudget {
            io_deadline_us: 7,
            ..ExpertIoBudget::unbounded()
        };
        let usage = ExpertIoBatchUsage::default();
        assert_eq!(
            usage.inspect(
                budget,
                ExpertIoEstimate {
                    earliest_ready_in_us: 7,
                    ..Default::default()
                }
            ),
            None
        );
        assert_eq!(
            usage.inspect(
                budget,
                ExpertIoEstimate {
                    earliest_ready_in_us: 8,
                    ..Default::default()
                }
            ),
            Some(ExpertIoRejection::IoDeadline)
        );
    }

    #[test]
    fn additive_usage_saturates_every_resource_dimension() {
        let mut usage = ExpertIoBatchUsage::default();
        usage.admit(ExpertIoEstimate {
            incremental_unique_bytes: u64::MAX,
            inflight_reads: usize::MAX,
            pinned_slab_bytes: u64::MAX,
            storage_read_bytes: u64::MAX,
            read_ops: usize::MAX,
            pageable_host_bytes: u64::MAX,
            pinned_host_bytes: u64::MAX,
            h2d_bytes: u64::MAX,
            upload_slots: usize::MAX,
            device_install_bytes: u64::MAX,
            rejected_prefetch_risk: u32::MAX,
            latency_debt_us: u64::MAX,
            ..Default::default()
        });
        usage.admit(ExpertIoEstimate {
            incremental_unique_bytes: 1,
            inflight_reads: 1,
            pinned_slab_bytes: 1,
            storage_read_bytes: 1,
            read_ops: 1,
            pageable_host_bytes: 1,
            pinned_host_bytes: 1,
            h2d_bytes: 1,
            upload_slots: 1,
            device_install_bytes: 1,
            rejected_prefetch_risk: 1,
            latency_debt_us: 1,
            ..Default::default()
        });

        assert_eq!(usage.incremental_expert_bytes, u64::MAX);
        assert_eq!(usage.inflight_reads, usize::MAX);
        assert_eq!(usage.pinned_slab_bytes, u64::MAX);
        assert_eq!(usage.storage_read_bytes, u64::MAX);
        assert_eq!(usage.read_ops, usize::MAX);
        assert_eq!(usage.pageable_host_bytes, u64::MAX);
        assert_eq!(usage.pinned_host_bytes, u64::MAX);
        assert_eq!(usage.h2d_bytes, u64::MAX);
        assert_eq!(usage.upload_slots, usize::MAX);
        assert_eq!(usage.device_install_bytes, u64::MAX);
        assert_eq!(usage.rejected_prefetch_risk, u32::MAX);
        assert_eq!(usage.latency_debt_us, u64::MAX);
    }
}
