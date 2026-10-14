//! Read-only physical-owner projections. Payloads, reservations and allocator
//! grants have different units/scopes and must not be summed across these views.

use super::{ResourceClosure, ResourceEvidence, ResourceObservationScope};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceResidency {
    /// The live owner still retains its image/state/workspace resources.
    Retained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceRelease {
    /// Only an explicit final-owner release protocol may supply this evidence.
    Released,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceQuiescence {
    Pending,
    Quiescent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DefaultStateObservation {
    pub topology_id: u64,
    /// Zero is a valid sequence execution generation, unlike image generations.
    pub generation: u64,
    pub position: usize,
    pub poisoned: bool,
    /// Subset of live_model_state_bytes, never an additional charge.
    pub resident_bytes: ResourceEvidence<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExpertCacheObservation {
    pub resident_entries: usize,
    pub resident_bytes: usize,
    pub pending_upload_bytes: usize,
    pub scratch_bytes: usize,
    /// resident + pending + scratch, not allocator grants or device-wide usage.
    pub charged_bytes: usize,
    /// Subset of charged_bytes, NOT an additional charge.
    pub unknown_quarantine_bytes: usize,
    pub peak_bytes: usize,
    pub quarantined: bool,
    pub uploads: u64,
    pub hits: u64,
    pub evictions: u64,
}

impl From<crate::transformer::ExpertCacheStats> for ExpertCacheObservation {
    fn from(s: crate::transformer::ExpertCacheStats) -> Self {
        Self {
            resident_entries: s.resident_experts,
            resident_bytes: s.resident_bytes,
            pending_upload_bytes: s.pending_upload_bytes,
            scratch_bytes: s.scratch_bytes,
            charged_bytes: s.charged_bytes(),
            unknown_quarantine_bytes: s.unknown_quarantine_bytes,
            peak_bytes: s.peak_bytes,
            quarantined: s.quarantined,
            uploads: s.uploads,
            hits: s.hits,
            evictions: s.evictions,
        }
    }
}

/// Logical workspace identity is (image_generation, kind). It is not an
/// allocator handle. Creations distinguish replacements of reusable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct WorkspaceObservation {
    pub image_generation: u64,
    pub kind: &'static str,
    /// Numeric reservation is INCLUDED in expert_cache.scratch_bytes.
    pub reserved_bytes: usize,
    pub allocated_bytes: usize,
    pub creations: u64,
    pub reuses: u64,
    pub submissions: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TransferObservation {
    pub tx_slots: usize,
    pub rx_slots: usize,
    pub tx_in_use: usize,
    pub rx_in_use: usize,
    pub tx_high_water: usize,
    pub rx_high_water: usize,
    pub pinned_allocations: usize,
    pub pinned_bytes: usize,
    /// Aliases held by DMA, not new device allocation charges.
    pub device_holds_in_use: usize,
    pub device_holds_high_water: usize,
    pub submissions: u64,
    pub failed_submissions: u64,
    pub completions: u64,
    pub cancel_intents: u64,
    pub canceled_completions: u64,
    pub quarantined: bool,
}

/// Context-wide allocator gauges, not model-only ownership. Reads do not poll
/// retirement fences. Zero counts never establish final-owner release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AllocatorObservation {
    pub live_requested_bytes: usize,
    pub live_granted_bytes: usize,
    pub reserved_bytes: usize,
    pub peak_live_granted_bytes: usize,
    pub pending_retirement_bytes: usize,
    pub pending_retirement_blocks: usize,
    pub pending_retirement_batches: usize,
    pub pending_unfenced_bytes: usize,
    pub retired_blocks: u64,
    pub retirement_created_fences: u64,
    pub driver_allocations: u64,
    pub driver_allocation_bytes: u64,
    pub driver_frees: u64,
    pub driver_free_bytes: u64,
    /// Existing allocator protocol observations, not inferred from gauges.
    pub shutdown_quiescent: bool,
    pub shutdown_retained: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ArenaObservation {
    pub hits: u64,
    pub misses: u64,
    pub installs: u64,
}

/// Model-neutral TP1 physical snapshot. Default means unsupported, never zero.
/// Each partition carries its own evidence; absent getters remain Unobserved.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PhysicalResourceObservation {
    pub scope: ResourceEvidence<ResourceObservationScope>,
    pub prepared_plan_id: ResourceEvidence<u64>,
    pub image_generation: ResourceEvidence<u64>,
    pub device_ordinal: ResourceEvidence<usize>,
    pub default_state: ResourceEvidence<DefaultStateObservation>,
    /// Parameter bindings; cache resident bytes are a SUBSET, not added again.
    pub resident_parameter_bytes: ResourceEvidence<usize>,
    /// All state copies charged to the physical owner, including default state.
    pub live_model_state_bytes: ResourceEvidence<usize>,
    pub expert_cache: ResourceEvidence<ExpertCacheObservation>,
    pub numeric_workspace: ResourceEvidence<WorkspaceObservation>,
    /// Image-owned chunk workspace; excluded from numeric/cache scratch.
    pub recurrent_chunk_workspace_bytes: ResourceEvidence<usize>,
    /// Other provider/operation scratch has no complete TP1 getter.
    pub other_workspace_bytes: ResourceEvidence<usize>,
    pub arenas: ResourceEvidence<ArenaObservation>,
    pub boundary_transfer: ResourceEvidence<TransferObservation>,
    pub allocator: ResourceEvidence<AllocatorObservation>,
    /// Existing owner fence completions; a cumulative counter, not a new proof.
    pub owner_fence_completions: ResourceEvidence<u64>,
    pub quarantine: ResourceEvidence<bool>,
    /// Forward-image owner's protocol evidence. Other owners/allocators have
    /// their own coverage; this is not a universal final release certificate.
    pub quiescence: ResourceEvidence<ResourceQuiescence>,
    pub residency: ResourceEvidence<ResourceResidency>,
    /// Explicit close event; not final-owner resource release.
    pub close: ResourceEvidence<ResourceClosure>,
    pub release: ResourceEvidence<ResourceRelease>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PhysicalResourceObservationMismatch {
    #[error(transparent)]
    Scope(#[from] super::ResourceObservationScopeMismatch),
    #[error("physical resource {field} mismatch: expected {expected}, received {received}")]
    Identity {
        field: &'static str,
        expected: u64,
        received: u64,
    },
}

impl PhysicalResourceObservation {
    /// Merge gauges and retained evidence for the same physical image. A read
    /// never commits the result; owners retain it at existing lifecycle hooks.
    pub fn snapshot_with_live(
        &self,
        live: &Self,
    ) -> Result<Self, PhysicalResourceObservationMismatch> {
        // Once identity is ambiguous, no later image can refresh these retained
        // measurements. Construct a new owner/scope instead of resetting it.
        if matches!(self.scope, ResourceEvidence::Unknown)
            || matches!(self.prepared_plan_id, ResourceEvidence::Unknown)
            || matches!(self.image_generation, ResourceEvidence::Unknown)
        {
            return Ok(self.clone());
        }
        if matches!(live.scope, ResourceEvidence::Unknown)
            || matches!(live.prepared_plan_id, ResourceEvidence::Unknown)
            || matches!(live.image_generation, ResourceEvidence::Unknown)
        {
            let mut retained = self.clone();
            retained.scope = ResourceEvidence::Unknown;
            retained.prepared_plan_id = ResourceEvidence::Unknown;
            retained.image_generation = ResourceEvidence::Unknown;
            retained.mark_unknown();
            return Ok(retained);
        }
        if let (Some(expected), Some(received)) = (self.scope.value(), live.scope.value())
            && expected != received
        {
            return Err(super::ResourceObservationScopeMismatch {
                expected: *expected,
                received: *received,
            }
            .into());
        }
        for (field, retained, sample) in [
            (
                "prepared_plan_id",
                &self.prepared_plan_id,
                &live.prepared_plan_id,
            ),
            (
                "image_generation",
                &self.image_generation,
                &live.image_generation,
            ),
        ] {
            if let (Some(expected), Some(received)) = (retained.value(), sample.value())
                && expected != received
            {
                return Err(PhysicalResourceObservationMismatch::Identity {
                    field,
                    expected: *expected,
                    received: *received,
                });
            }
        }
        let mut combined = self.clone();
        macro_rules! fields {
            ($($field:ident),* $(,)?) => { $(combined.$field.observe(&live.$field);)* };
        }
        fields!(
            scope,
            prepared_plan_id,
            image_generation,
            device_ordinal,
            default_state,
            resident_parameter_bytes,
            live_model_state_bytes,
            expert_cache,
            numeric_workspace,
            recurrent_chunk_workspace_bytes,
            other_workspace_bytes,
            arenas,
            boundary_transfer,
            allocator,
            owner_fence_completions,
            quarantine,
            quiescence,
            residency,
            close,
            release
        );
        if let Some(previous) = self.default_state.value()
            && let ResourceEvidence::Observed(state) | ResourceEvidence::Verified(state) =
                &mut combined.default_state
        {
            let mut resident_bytes = previous.resident_bytes;
            resident_bytes.observe(&state.resident_bytes);
            state.resident_bytes = resident_bytes;
        }
        if combined.quarantine.value() == Some(&true)
            || matches!(combined.quiescence, ResourceEvidence::Unknown)
        {
            combined.mark_unknown();
        }
        Ok(combined)
    }

    /// A mismatched image cannot refresh a retained owner's measurements.
    pub fn snapshot_or_unknown(&self, live: &Self) -> Self {
        self.snapshot_with_live(live).unwrap_or_else(|_| {
            let mut retained = self.clone();
            retained.mark_unknown();
            retained.scope = ResourceEvidence::Unknown;
            retained.prepared_plan_id = ResourceEvidence::Unknown;
            retained.image_generation = ResourceEvidence::Unknown;
            retained
        })
    }

    /// Failed close supplies no release proof. Preserve an independently
    /// authenticated lower-owner quiescence proof if one was actually supplied.
    pub fn record_close(&mut self, succeeded: bool) {
        if succeeded {
            self.close
                .observe(&ResourceEvidence::Observed(ResourceClosure::Closed));
        } else {
            self.close.observe(&ResourceEvidence::Unknown);
            if !matches!(
                self.quiescence,
                ResourceEvidence::Verified(ResourceQuiescence::Quiescent)
            ) {
                self.quiescence.observe(&ResourceEvidence::Unknown);
            }
            self.release.observe(&ResourceEvidence::Unknown);
        }
    }

    pub fn mark_unknown(&mut self) {
        self.quiescence = ResourceEvidence::Unknown;
        self.close = ResourceEvidence::Unknown;
        self.release = ResourceEvidence::Unknown;
    }

    pub fn coverage(&self) -> EvidenceCoverage {
        let mut coverage = EvidenceCoverage::default();
        macro_rules! fields {
            ($($field:ident),* $(,)?) => { $(coverage.record(stringify!($field), &self.$field);)* };
        }
        fields!(
            scope,
            prepared_plan_id,
            image_generation,
            device_ordinal,
            default_state,
            resident_parameter_bytes,
            live_model_state_bytes,
            expert_cache,
            numeric_workspace,
            recurrent_chunk_workspace_bytes,
            other_workspace_bytes,
            arenas,
            boundary_transfer,
            allocator,
            owner_fence_completions,
            quarantine,
            quiescence,
            residency,
            close,
            release
        );
        let state_bytes = match self.default_state {
            ResourceEvidence::Observed(state) | ResourceEvidence::Verified(state) => {
                state.resident_bytes
            }
            ResourceEvidence::Unknown => ResourceEvidence::Unknown,
            ResourceEvidence::Unobserved => ResourceEvidence::Unobserved,
        };
        coverage.record("default_state.resident_bytes", &state_bytes);
        coverage
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct EvidenceCoverage {
    pub unobserved: Vec<&'static str>,
    pub observed: Vec<&'static str>,
    pub verified: Vec<&'static str>,
    pub unknown: Vec<&'static str>,
}

impl EvidenceCoverage {
    pub fn record<T>(&mut self, name: &'static str, evidence: &ResourceEvidence<T>) {
        match evidence {
            ResourceEvidence::Unobserved => &mut self.unobserved,
            ResourceEvidence::Observed(_) => &mut self.observed,
            ResourceEvidence::Verified(_) => &mut self.verified,
            ResourceEvidence::Unknown => &mut self.unknown,
        }
        .push(name);
    }
}

#[derive(Debug, thiserror::Error)]
#[error("resource summary serialization failed: {0}")]
pub struct ResourceSummaryError(#[from] serde_json::Error);

pub fn resource_summary_json(summary: &impl Serialize) -> Result<String, ResourceSummaryError> {
    Ok(serde_json::to_string(summary)?)
}
