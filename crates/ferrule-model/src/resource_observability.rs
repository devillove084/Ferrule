//! Model-neutral, owner-local resource evidence, never resource authority.
//!
//! A scope names one observation owner and its exact generation. It must not be
//! reused for a replacement owner. Neither a snapshot nor this ledger owns any
//! resources, synchronizes work, releases credit, or proves physical closure.
//! Unsupported measurements remain `Unobserved`; zero is an actual observation.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::io_protocol::QuiescenceEvidence;
use serde::{Serialize, Serializer};

mod physical;
pub use physical::{
    AllocatorObservation, ArenaObservation, DefaultStateObservation, EvidenceCoverage,
    ExpertCacheObservation, PhysicalResourceObservation, PhysicalResourceObservationMismatch,
    ResourceQuiescence, ResourceRelease, ResourceResidency, ResourceSummaryError,
    TransferObservation, WorkspaceObservation, resource_summary_json,
};

/// Identity in the observation owner's namespace, not a backend resource handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct ResourceObservationScope {
    pub owner: NonZeroU64,
    pub generation: NonZeroU64,
}

static NEXT_OBSERVATION_OWNER: AtomicU64 = AtomicU64::new(1);

impl ResourceObservationScope {
    pub fn fresh(generation: NonZeroU64) -> Self {
        let owner = NEXT_OBSERVATION_OWNER
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("resource observation identity space exhausted");
        Self {
            owner: NonZeroU64::new(owner).expect("nonzero observation owner"),
            generation,
        }
    }
}

/// Evidence quality is independent of the value (including zero or false).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(tag = "state", content = "value", rename_all = "snake_case")]
pub enum ResourceEvidence<T> {
    #[default]
    Unobserved,
    /// An owner read a gauge or recorded a lifecycle event, without a proof.
    Observed(T),
    /// The original owner explicitly supplied authoritative evidence.
    /// This remains a report, not a transferable cleanup or release token.
    Verified(T),
    /// Custody/evidence was lost or ambiguous. Sticky within this scope.
    Unknown,
}

impl<T> ResourceEvidence<T> {
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Observed(value) | Self::Verified(value) => Some(value),
            Self::Unobserved | Self::Unknown => None,
        }
    }
}

impl<T: Clone> ResourceEvidence<T> {
    fn observe(&mut self, sample: &Self) {
        if !matches!(self, Self::Unknown) && !matches!(sample, Self::Unobserved) {
            *self = sample.clone();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceClosure {
    Open,
    Closed,
}

/// Gauges and lifecycle evidence for exactly the named scope, not device totals.
/// Counts do not establish a fence, quiescence, or closure, even when all are zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceObservationSnapshot {
    pub scope: ResourceObservationScope,
    /// Residency charged by this owner; not allocator residency or free memory.
    pub resident_bytes: ResourceEvidence<u64>,
    /// Physically submitted operations whose completion remains pending.
    pub pending_operations: ResourceEvidence<usize>,
    pub scratch_bytes: ResourceEvidence<u64>,
    /// Whether this scope retains unknown-completion quarantine.
    pub quarantine: ResourceEvidence<bool>,
    /// Number of live credit grants, not a sum across incompatible credit units.
    pub active_credit_grants: ResourceEvidence<usize>,
    pub pending_retirements: ResourceEvidence<usize>,
    /// Cumulative completed retirements in the owner's operation namespace.
    pub retired_operations: ResourceEvidence<u64>,
    #[serde(serialize_with = "serialize_quiescence")]
    pub fence: ResourceEvidence<QuiescenceEvidence>,
    pub closure: ResourceEvidence<ResourceClosure>,
}

impl ResourceObservationSnapshot {
    pub fn unobserved(scope: ResourceObservationScope) -> Self {
        Self {
            scope,
            resident_bytes: ResourceEvidence::Unobserved,
            pending_operations: ResourceEvidence::Unobserved,
            scratch_bytes: ResourceEvidence::Unobserved,
            quarantine: ResourceEvidence::Unobserved,
            active_credit_grants: ResourceEvidence::Unobserved,
            pending_retirements: ResourceEvidence::Unobserved,
            retired_operations: ResourceEvidence::Unobserved,
            fence: ResourceEvidence::Unobserved,
            closure: ResourceEvidence::Unobserved,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("resource observation scope mismatch: expected {expected:?}, received {received:?}")]
pub struct ResourceObservationScopeMismatch {
    pub expected: ResourceObservationScope,
    pub received: ResourceObservationScope,
}

/// A fixed-size evidence ledger to keep with the existing resource owner.
/// There is deliberately no reset, implicit generation advance, or release API.
#[derive(Debug)]
pub struct ResourceObservabilityLedger {
    snapshot: ResourceObservationSnapshot,
}

impl ResourceObservabilityLedger {
    pub fn new(scope: ResourceObservationScope) -> Self {
        Self {
            snapshot: ResourceObservationSnapshot::unobserved(scope),
        }
    }

    /// Pure read: no polling, synchronization, release, or evidence promotion.
    pub fn snapshot(&self) -> ResourceObservationSnapshot {
        self.snapshot.clone()
    }

    /// Combine a read-only live sample with retained lifecycle evidence.
    /// Unknown is absorbing for each field; omitted measurements do not erase
    /// earlier evidence. This method never mutates the ledger. The source must
    /// retain live Unknown evidence itself, or record it before projection.
    pub fn snapshot_with_live(
        &self,
        live: &ResourceObservationSnapshot,
    ) -> Result<ResourceObservationSnapshot, ResourceObservationScopeMismatch> {
        let mut combined = Self {
            snapshot: self.snapshot.clone(),
        };
        combined.observe(live)?;
        Ok(combined.snapshot)
    }

    /// Record evidence supplied by the original owner. Scope mismatch leaves
    /// the ledger unchanged. Unknown is absorbing for each field; omitted
    /// measurements do not erase earlier evidence. Success only records a sample.
    pub fn observe(
        &mut self,
        sample: &ResourceObservationSnapshot,
    ) -> Result<(), ResourceObservationScopeMismatch> {
        if sample.scope != self.snapshot.scope {
            return Err(ResourceObservationScopeMismatch {
                expected: self.snapshot.scope,
                received: sample.scope,
            });
        }
        self.snapshot.resident_bytes.observe(&sample.resident_bytes);
        self.snapshot
            .pending_operations
            .observe(&sample.pending_operations);
        self.snapshot.scratch_bytes.observe(&sample.scratch_bytes);
        self.snapshot.quarantine.observe(&sample.quarantine);
        self.snapshot
            .active_credit_grants
            .observe(&sample.active_credit_grants);
        self.snapshot
            .pending_retirements
            .observe(&sample.pending_retirements);
        self.snapshot
            .retired_operations
            .observe(&sample.retired_operations);
        self.snapshot.fence.observe(&sample.fence);
        self.snapshot.closure.observe(&sample.closure);
        if matches!(self.snapshot.fence, ResourceEvidence::Unknown)
            || matches!(
                self.snapshot.fence,
                ResourceEvidence::Observed(QuiescenceEvidence::Unknown)
                    | ResourceEvidence::Verified(QuiescenceEvidence::Unknown)
            )
        {
            self.record_quiescence(QuiescenceEvidence::Unknown);
        }
        Ok(())
    }

    /// Record the custody protocol's status, not an authenticated fence receipt.
    /// Verified evidence must be explicitly supplied by the original protocol
    /// owner through observe. Neither status nor a fence closes/releases resources.
    pub fn record_quiescence(&mut self, evidence: QuiescenceEvidence) {
        let sample = match evidence {
            QuiescenceEvidence::Pending => ResourceEvidence::Observed(evidence),
            QuiescenceEvidence::Quiescent => ResourceEvidence::Observed(evidence),
            QuiescenceEvidence::Unknown => ResourceEvidence::Unknown,
        };
        self.snapshot.fence.observe(&sample);
        if matches!(self.snapshot.fence, ResourceEvidence::Unknown) {
            // Lost custody does not by itself establish a physical quarantine.
            // Preserve explicit positive evidence, otherwise report uncertainty.
            if !matches!(
                self.snapshot.quarantine,
                ResourceEvidence::Observed(true) | ResourceEvidence::Verified(true)
            ) {
                self.snapshot.quarantine.observe(&ResourceEvidence::Unknown);
            }
            self.snapshot.closure.observe(&ResourceEvidence::Unknown);
        }
    }

    /// Observe successful explicit close only after the original owner's close
    /// API succeeds. Success is not physical release evidence, even if all gauges
    /// are zero. Does not infer a fence or clear earlier Unknown evidence.
    pub fn record_closed(&mut self) {
        self.snapshot
            .closure
            .observe(&ResourceEvidence::Observed(ResourceClosure::Closed));
    }
}

fn serialize_quiescence<S: Serializer>(
    evidence: &ResourceEvidence<QuiescenceEvidence>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let value = |q: &QuiescenceEvidence| match q {
        QuiescenceEvidence::Pending => "pending",
        QuiescenceEvidence::Unknown => "unknown",
        QuiescenceEvidence::Quiescent => "quiescent",
    };
    let serialized = match evidence {
        ResourceEvidence::Unobserved => ResourceEvidence::Unobserved,
        ResourceEvidence::Unknown => ResourceEvidence::Unknown,
        ResourceEvidence::Observed(q) => ResourceEvidence::Observed(value(q)),
        ResourceEvidence::Verified(q) => ResourceEvidence::Verified(value(q)),
    };
    serialized.serialize(serializer)
}

#[cfg(test)]
#[path = "../tests/resource_observability/mod.rs"]
mod tests;
