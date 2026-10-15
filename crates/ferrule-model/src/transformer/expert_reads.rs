//! Host-only staging for the existing synchronous routed-expert owner.
//! One next read, unchanged compressed representation and synchronous forward.
//! Host intervals are observable; real GPU overlap needs a correlated device trace.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use serde::Serialize;

use ferrule_common::numeric_fp8::NumericFp8PayloadLiveness;

use ferrule_common::{Error, Result, SourceGeneration};

use crate::checkpoint::read_pipeline::{
    CheckpointReadPipeline, ReadCompletion, ReadJoinHandoff, ReadOutcome, ReadPipelineError,
    ReadPipelineLimits, ReadPipelineOptions, ReadReceipt, ReadTicket,
};
use crate::checkpoint::{
    CheckpointReadCounters, CheckpointReadPlan, CheckpointSourceFileIdentity,
    CheckpointTensorReader, NumericFp8Read,
};
use crate::nn::ParameterResidency;

use super::{BoundParameter, ExpertMetadata, PreparedSwiGlu};

/// Scheduling only: both modes use the same worker, slots, byte and stack limits.
/// Neither mode changes the synchronous forward or its CUDA completion proofs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertReadSchedule {
    #[default]
    Blocking,
    /// One next bucket's host read may overlap current GPU execution. GPU overlap
    /// must be measured; submission timestamps do not establish that it occurred.
    Lookahead,
}

/// Explicit opt-in; an unconfigured materializer keeps its existing demand path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExpertReadPipelineOptions {
    pub schedule: ExpertReadSchedule,
    /// Combined packed buffers, prepared compressed artifacts, metadata and stack.
    pub max_host_bytes: u64,
    pub max_metadata_bytes: u64,
    /// Requested stack reservation; OS rounding, TLS and allocator overhead are
    /// separate admission headroom, not an RSS guarantee.
    pub worker_stack_bytes: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ExpertReadPipelineError {
    #[error("expert read pipeline: {0}")]
    Invalid(&'static str),
}

fn invalid(message: &'static str) -> Error {
    Error::ModelSource {
        source: Box::new(ExpertReadPipelineError::Invalid(message)),
    }
}

fn pipeline_error(source: ReadPipelineError) -> Error {
    Error::ModelSource {
        source: Box::new(source),
    }
}

#[derive(Debug, Default)]
pub(crate) struct ExpertReadAdmissionState {
    active: bool,
    quarantined: bool,
    owner_closed: bool,
    payloads: Vec<NumericFp8PayloadLiveness>,
    join: Option<ReadJoinHandoff>,
    worker: Option<CheckpointReadPipeline>,
    identity: Option<ExpertReadOwnerIdentity>,
}

impl ExpertReadAdmissionState {
    pub fn close_owner(state: &Mutex<Self>) {
        let (worker, join) = {
            let mut state = state.lock().unwrap_or_else(|p| p.into_inner());
            state.owner_closed = true;
            (state.worker.take(), state.join.take())
        };
        // The materializer, not the last temporary observer Arc, owns teardown.
        // No admission mutex may span callbacks or the actual off-worker join.
        drop(worker);
        if let Some(join) = join {
            let _ = join.join();
        }
    }
}

#[derive(Debug)]
struct ExpertReadOwnerIdentity {
    image: u64,
    options: ExpertReadPipelineOptions,
    reader: CheckpointTensorReader,
    // The provider checks selected bindings against the prepared image before
    // admission. Remember each encountered shard without refreshing its snapshot.
    sources: BTreeMap<std::path::PathBuf, CheckpointSourceFileIdentity>,
}

#[derive(Debug, Clone)]
pub(crate) struct ExpertReadAdmission(Arc<ReadAdmissionInner>);

#[derive(Debug)]
struct ReadAdmissionInner {
    state: Arc<Mutex<ExpertReadAdmissionState>>,
}

impl ExpertReadAdmission {
    pub fn take(state: &Arc<Mutex<ExpertReadAdmissionState>>) -> Result<Self> {
        let join = {
            let mut state = state.lock().map_err(|poisoned| {
                poisoned.into_inner().quarantined = true;
                invalid("expert read owner ledger is poisoned")
            })?;
            if state.active || state.quarantined || state.owner_closed {
                return Err(invalid(
                    "another expert read sequence is live or quarantined",
                ));
            }
            state.active = true;
            state.join.clone()
        };
        let admission = Self(Arc::new(ReadAdmissionInner {
            state: Arc::clone(state),
        }));
        // No admission lock is held across join callbacks. A worker-side caller
        // gets Pending, never a self-join or a premature stack/payload refund.
        if let Some(join) = join {
            let shutdown = join.join();
            if shutdown.custody_unknown || shutdown.join_failed || !shutdown.unknown.is_empty() {
                admission.quarantine();
                return Err(invalid("expert read worker custody is unknown"));
            }
            if shutdown.join_pending {
                return Err(invalid("expert read worker join is still pending"));
            }
            state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .join = None;
        }
        {
            let mut state = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.payloads.retain(NumericFp8PayloadLiveness::is_live);
            if !state.payloads.is_empty() {
                return Err(invalid("prepared host payload aliases are still live"));
            }
        }
        Ok(admission)
    }

    fn checkout(
        &self,
        options: ExpertReadPipelineOptions,
        image: u64,
        reader: &CheckpointTensorReader,
        plans: &BTreeMap<(usize, usize), Arc<ExpertReadPlan>>,
    ) -> Result<Option<CheckpointReadPipeline>> {
        if image == 0 {
            return Err(invalid("image generation must be nonzero"));
        }
        let mut state = self.0.state.lock().map_err(|poisoned| {
            poisoned.into_inner().quarantined = true;
            invalid("expert read owner ledger is poisoned")
        })?;
        let identity = state
            .identity
            .get_or_insert_with(|| ExpertReadOwnerIdentity {
                image,
                options,
                reader: reader.clone(),
                sources: BTreeMap::new(),
            });
        if identity.image != image
            || identity.options != options
            || !identity.reader.shares_owner_with(reader)
        {
            return Err(invalid(
                "expert read owner image/reader/limits/options changed",
            ));
        }
        let mut sources = identity.sources.clone();
        for source in plans.values().flat_map(|plan| plan.packed.source_files()) {
            if let Some(previous) =
                sources.insert(source.catalog_path().to_path_buf(), source.clone())
                && previous != *source
            {
                return Err(invalid("expert read owner source snapshot changed"));
            }
        }
        let metadata = plans
            .values()
            .map(|plan| plan.metadata_bytes)
            .chain(sources.values().map(source_metadata_reservation))
            .try_fold(4096u64, u64::checked_add)
            .ok_or_else(|| invalid("owner metadata overflows"))?;
        if metadata > options.max_metadata_bytes {
            return Err(invalid("owner plus selected metadata exceeds host bound"));
        }
        identity.sources = sources;
        Ok(state.worker.take())
    }

    fn park(&self, pipeline: CheckpointReadPipeline) {
        let mut state = self.0.state.lock().unwrap_or_else(|poisoned| {
            let mut state = poisoned.into_inner();
            state.quarantined = true;
            state
        });
        let displaced = if state.owner_closed {
            Some(pipeline)
        } else {
            let displaced = state.worker.replace(pipeline);
            state.quarantined |= displaced.is_some();
            displaced
        };
        drop(state);
        // Even invariant-failure cleanup must never run a join/waker under the
        // admission mutex.
        drop(displaced);
    }

    fn quarantine(&self) {
        self.0
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .quarantined = true;
    }

    fn retain_payloads(&self, payloads: &[NumericFp8PayloadLiveness]) {
        self.0
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .payloads = payloads
            .iter()
            .filter(|payload| payload.is_live())
            .cloned()
            .collect();
    }

    fn retain_join(&self, join: ReadJoinHandoff) {
        self.0
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .join = Some(join);
    }
}

impl Drop for ReadAdmissionInner {
    fn drop(&mut self) {
        let join = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.active = false;
            if state.owner_closed {
                state.join.take()
            } else {
                None
            }
        };
        if let Some(join) = join {
            let _ = join.join();
        }
    }
}

#[derive(Debug)]
struct FailedRead {
    completion: ReadCompletion,
    // A caller can retain the typed error after the sequence joins. Its packed
    // charge and this materializer admission must survive together.
    _admission: Option<ExpertReadAdmission>,
}

impl std::fmt::Display for FailedRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.completion.outcome() {
            ReadOutcome::Failed(error) => write!(f, "expert checkpoint read failed: {error}"),
            _ => f.write_str("expert checkpoint read was cancelled"),
        }
    }
}

impl std::error::Error for FailedRead {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.completion.outcome() {
            ReadOutcome::Failed(error) => Some(error),
            _ => None,
        }
    }
}

fn source_metadata_reservation(source: &CheckpointSourceFileIdentity) -> u64 {
    // Includes the transactional map copy while admitting new shards.
    512 + 4 * source.catalog_path().as_os_str().len() as u64
        + 2 * source.canonical_path().as_os_str().len() as u64
}

impl ExpertReadPipelineOptions {
    pub fn validate(self) -> Result<()> {
        if self.max_host_bytes > isize::MAX as u64 {
            return Err(invalid("host staging budget exceeds address space"));
        }
        if self.max_metadata_bytes == 0
            || self.worker_stack_bytes == 0
            || self.worker_stack_bytes > isize::MAX as usize
        {
            return Err(invalid("metadata and worker stack bounds must be nonzero"));
        }
        let overhead = self
            .max_metadata_bytes
            .checked_add(self.worker_stack_bytes as u64)
            .ok_or_else(|| invalid("host overhead overflows"))?;
        if self.max_host_bytes <= overhead {
            return Err(invalid(
                "host bound must leave room beyond metadata and stack",
            ));
        }
        Ok(())
    }

    /// Source/shape-only constructor admission. No payload reads, worker or
    /// CUDA owner are created; the selected scope rechecks the same plan bounds.
    pub(crate) fn validate_resources(
        self,
        resources: &super::BoundDecoderResources,
        max_batch_tokens: usize,
        max_parameter_bytes: u64,
    ) -> Result<()> {
        use super::FeedForward;
        use crate::support::TensorRole;
        let payload_bound = self.payload_bound()?;
        let reader = CheckpointTensorReader::new(max_parameter_bytes);
        let mut bindings = BTreeMap::<(usize, usize), Vec<&BoundParameter>>::new();
        let mut sources = BTreeMap::new();
        for parameter in resources.state_dict().parameters() {
            if let ParameterResidency::Expert { layer, expert } = *parameter.residency() {
                bindings.entry((layer, expert)).or_default().push(parameter);
                for part in std::iter::once(parameter.weight()).chain(parameter.scale()) {
                    let source = part.source_identity();
                    if let Some(previous) = sources.insert(source.catalog_path(), source)
                        && previous != source
                    {
                        return Err(invalid("conflicting owner source snapshots"));
                    }
                }
            }
        }
        let owner_metadata = sources
            .values()
            .map(|source| source_metadata_reservation(source))
            .try_fold(4096u64, u64::checked_add)
            .ok_or_else(|| invalid("owner metadata overflows"))?;
        let mut routed = false;
        for layer in resources.spec().layers() {
            let FeedForward::Moe(moe) = layer.feed_forward() else {
                continue;
            };
            routed = true;
            let selected = max_batch_tokens
                .checked_mul(moe.router_spec().experts_per_token())
                .ok_or_else(|| invalid("route capacity overflows"))?
                .min(moe.router_spec().num_experts());
            let mut metadata = Vec::new();
            for expert in 0..moe.router_spec().num_experts() {
                let parameters = bindings
                    .remove(&(layer.index(), expert))
                    .ok_or_else(|| invalid("incomplete expert source catalog"))?;
                if parameters.len() != 3 {
                    return Err(invalid("requires exactly gate/up/down expert projections"));
                }
                let parameters = [
                    TensorRole::RoutedExpertGate,
                    TensorRole::RoutedExpertUp,
                    TensorRole::RoutedExpertDown,
                ]
                .map(|role| {
                    parameters
                        .iter()
                        .find(|p| p.role() == &role)
                        .map(|p| (*p).clone())
                        .ok_or_else(|| invalid("missing expert projection role"))
                });
                let [gate, up, down] = parameters;
                let expert_metadata =
                    ExpertMetadata::new(layer.index(), expert, [gate?, up?, down?], None)?;
                let plan = ExpertReadPlan::new(1, &expert_metadata, &reader)?;
                if plan
                    .packed
                    .storage_bytes()
                    .checked_mul(2)
                    .is_none_or(|n| n > payload_bound)
                {
                    return Err(invalid("two whole expert payloads exceed host budget"));
                }
                metadata.push(plan.metadata_bytes);
            }
            metadata.sort_unstable_by(|a, b| b.cmp(a));
            let worst =
                metadata
                    .into_iter()
                    .take(selected)
                    .try_fold(owner_metadata, |total, bytes| {
                        total
                            .checked_add(bytes)
                            .ok_or_else(|| invalid("selected metadata overflows"))
                    })?;
            if worst > self.max_metadata_bytes {
                return Err(invalid("selected metadata exceeds host budget"));
            }
        }
        if !routed || !bindings.is_empty() {
            return Err(invalid(
                "source must contain only canonical routed numeric experts",
            ));
        }
        Ok(())
    }

    fn payload_bound(self) -> Result<u64> {
        self.validate()?;
        Ok(self.max_host_bytes - self.max_metadata_bytes - self.worker_stack_bytes as u64)
    }

    fn limits(self) -> Result<ReadPipelineLimits> {
        Ok(ReadPipelineLimits {
            max_host_bytes: self.payload_bound()? / 2,
            max_read_slots: 2,
        })
    }
}

/// Host-only owner-lifetime counts, never GPU completion or certified RSS.
/// Completed reads include verified, cancelled and failed terminal results;
/// Unknown has no proved completion. Byte receipts prove whole packed reads,
/// not partial syscalls. Reader intervals sample the shared checkpoint reader
/// from submission through take, so concurrent unrelated reader work can be
/// included; they are not certified per-ticket syscall attribution.
/// All durations are host monotonic nanoseconds. Peaks are lifetime gauges.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ExpertReadPipelineStats {
    pub sequences_started: u64,
    pub sequences_finished: u64,
    pub workers_created: u64,
    pub worker_reuses: u64,
    pub workers_joined: u64,
    pub join_pending_sequences: u64,
    pub join_failed_sequences: u64,
    pub submitted_reads: u64,
    pub speculative_reads: u64,
    pub rejected_submissions: u64,
    pub demanded_reads: u64,
    pub completed_reads: u64,
    pub verified_reads: u64,
    pub cancelled_reads: u64,
    pub failed_reads: u64,
    pub unknown_reads: u64,
    /// Accepted reads whose final receipt could not be taken on owner teardown.
    pub unobserved_terminal_reads: u64,
    pub worker_started_reads: u64,
    pub io_started_reads: u64,
    pub prefetched_reads_taken: u64,
    pub resident_skips: u64,
    pub demand_failures: u64,
    pub materialization_failures: u64,
    pub wasted_reads: u64,
    pub wasted_read_bytes: u64,
    pub unproved_wasted_reads: u64,
    pub proved_read_bytes: u64,
    pub reader_interval_calls: u64,
    pub reader_interval_bytes: u64,
    pub wasted_reader_interval_calls: u64,
    pub wasted_reader_interval_bytes: u64,
    pub unobserved_reader_intervals: u64,
    pub demand_wait_calls: u64,
    pub demand_wait_ns: u64,
    pub io_duration_ns: u64,
    pub unknown_bytes: u64,
    pub peak_packed_bytes: u64,
    pub peak_artifact_bytes: u64,
    pub peak_metadata_bytes: u64,
    pub peak_requested_stack_bytes: u64,
    pub peak_payload_bytes: u64,
    pub peak_host_charged_bytes: u64,
}

macro_rules! statistics {
    (counters: [$($counter:ident),*], peaks: [$($peak:ident),*]) => {
        impl ExpertReadPipelineStats {
            fn with_sequence(self, sequence: Self) -> Self {
                Self {
                    $($counter: self.$counter.saturating_add(sequence.$counter),)*
                    $($peak: self.$peak.max(sequence.$peak),)*
                }
            }
            /// Counter reset means missing delta, never an invented zero. Peak
            /// fields remain owner-lifetime values, not invocation deltas.
            pub fn checked_delta(self, before: Self) -> Option<Self> {
                Some(Self {
                    $($counter: self.$counter.checked_sub(before.$counter)?,)*
                    $($peak: self.$peak,)*
                })
            }
        }
    }
}
statistics!(
    counters: [sequences_started, sequences_finished, workers_created, worker_reuses, workers_joined, join_pending_sequences,
        join_failed_sequences, submitted_reads, speculative_reads, rejected_submissions,
        demanded_reads, completed_reads, verified_reads, cancelled_reads, failed_reads,
        unknown_reads, unobserved_terminal_reads, worker_started_reads, io_started_reads, prefetched_reads_taken,
        resident_skips, demand_failures, materialization_failures, wasted_reads,
        wasted_read_bytes, unproved_wasted_reads, proved_read_bytes, reader_interval_calls,
        reader_interval_bytes, wasted_reader_interval_calls, wasted_reader_interval_bytes,
        unobserved_reader_intervals, demand_wait_calls, demand_wait_ns, io_duration_ns, unknown_bytes],
    peaks: [peak_packed_bytes, peak_artifact_bytes, peak_metadata_bytes,
        peak_requested_stack_bytes, peak_payload_bytes, peak_host_charged_bytes]
);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ExpertReadHostCharge {
    pub packed_bytes: u64,
    pub artifact_bytes: u64,
    pub metadata_bytes: u64,
    pub requested_stack_bytes: u64,
    pub payload_bytes: u64,
    pub host_charged_bytes: u64,
    pub unknown_packed_bytes: u64,
    pub unknown_stack_bytes: u64,
    pub reserved_read_slots: usize,
    pub unknown_read_slots: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExpertReadIdentity {
    pub image_generation: u64,
    pub layer: usize,
    pub expert: usize,
    pub operation: u64,
}

/// Offsets use this read scope's start as origin, not wall time or CUPTI time.
/// Missing worker/IO timestamps remain null; no zero-duration proof is inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExpertReadHostReceipt {
    pub identity: ExpertReadIdentity,
    pub outcome: &'static str,
    pub requested_bytes: u64,
    pub io_bytes: Option<u64>,
    pub cancellation_requested: bool,
    pub submitted_ns: u64,
    pub worker_started_ns: Option<u64>,
    pub io_started_ns: Option<u64>,
    pub io_finished_ns: Option<u64>,
    pub source_verified_ns: Option<u64>,
    pub completed_ns: Option<u64>,
    pub taken_ns: Option<u64>,
}

/// Copied diagnostic observation. Reads neither poll/drain/join the worker nor
/// synchronize CUDA or refund quota. Host charge is a conservative logical
/// reservation sampled from existing custody, not physical RSS. Last identities
/// and receipt are bounded to one scope; this is not a full operation journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExpertReadPipelineObservation {
    pub enabled: bool,
    pub requested_mode: Option<ExpertReadSchedule>,
    pub effective_mode: Option<ExpertReadSchedule>,
    /// A returned receipt proves positive whole-read bytes after host IO began.
    /// False includes missing/partial/Unknown evidence; it does not prove no IO.
    pub worker_read_observed: bool,
    pub status: &'static str,
    pub limits: Option<ExpertReadPipelineOptions>,
    pub current_charge: ExpertReadHostCharge,
    pub stats: ExpertReadPipelineStats,
    pub worker_alive: bool,
    pub worker_join_pending: bool,
    pub worker_unknown: bool,
    pub current_read: Option<ExpertReadIdentity>,
    pub pending_read: Option<ExpertReadIdentity>,
    pub last_receipt: Option<ExpertReadHostReceipt>,
    pub timing_origin: &'static str,
    pub reader_counter_scope: &'static str,
    pub charge_basis: &'static str,
    pub stats_scope: &'static str,
}
impl Default for ExpertReadPipelineObservation {
    fn default() -> Self {
        Self {
            enabled: false,
            requested_mode: None,
            effective_mode: None,
            worker_read_observed: false,
            status: "disabled",
            limits: None,
            current_charge: ExpertReadHostCharge::default(),
            stats: ExpertReadPipelineStats::default(),
            worker_alive: false,
            worker_join_pending: false,
            worker_unknown: false,
            current_read: None,
            pending_read: None,
            last_receipt: None,
            timing_origin: "expert_read_sequence_start_host_monotonic_ns_not_cuda",
            reader_counter_scope: "shared_checkpoint_reader_submit_through_take_not_ticket_certified",
            charge_basis: "logical_host_reservation_not_physical_rss",
            stats_scope: "owner_lifetime_owner_observed_transitions_peaks_not_invocation_deltas",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ExpertReadObservationState {
    pub observation: ExpertReadPipelineObservation,
    join: Option<ReadJoinHandoff>,
    // Observe the existing custody ledger rather than introducing additional
    // weak references that could retain the payload allocations themselves.
    admission_state: Option<Weak<Mutex<ExpertReadAdmissionState>>>,
    artifact_bytes: u64,
    prior_worker_joins: u64,
    scope_open: bool,
}
impl ExpertReadObservationState {
    pub fn snapshot(&self) -> ExpertReadPipelineObservation {
        let mut observed = self.observation;
        if let (Some(join), Some(options)) = (&self.join, observed.limits) {
            let sample = join.observe_budget();
            let worker = join.observe_worker();
            let budget = sample.budget;
            observed.worker_alive = worker.alive;
            observed.worker_join_pending = worker.join_pending;
            observed.worker_unknown = worker.poisoned
                || worker.join_failed
                || worker.custody_unknown
                || sample.ledger_poisoned
                || budget.unknown_slots != 0
                || budget.worker_stack.unknown_bytes != 0;
            observed.stats.workers_joined = self.prior_worker_joins + u64::from(worker.joined);
            let (artifacts_live, owner_poisoned, owner_quarantined) = self
                .admission_state
                .as_ref()
                .and_then(Weak::upgrade)
                .map(|state| {
                    let (state, poisoned) = match state.lock() {
                        Ok(state) => (state, false),
                        Err(error) => (error.into_inner(), true),
                    };
                    // Weak witnesses themselves retain backing allocations;
                    // only operational admission drops the retired witnesses.
                    (!state.payloads.is_empty(), poisoned, state.quarantined)
                })
                .unwrap_or_default();
            let artifact_bytes = if self.scope_open || artifacts_live || owner_poisoned {
                self.artifact_bytes
            } else {
                0
            };
            let retained = self.scope_open
                || artifacts_live
                || budget.reserved_bytes != 0
                || budget.worker_stack.reserved_bytes != 0;
            observed.current_charge = ExpertReadHostCharge {
                packed_bytes: budget.reserved_bytes,
                artifact_bytes,
                metadata_bytes: if retained {
                    options.max_metadata_bytes
                } else {
                    0
                },
                requested_stack_bytes: budget.worker_stack.reserved_bytes as u64,
                payload_bytes: budget.reserved_bytes.saturating_add(artifact_bytes),
                unknown_packed_bytes: budget.unknown_bytes,
                unknown_stack_bytes: budget.worker_stack.unknown_bytes as u64,
                reserved_read_slots: budget.reserved_slots,
                unknown_read_slots: budget.unknown_slots,
                ..Default::default()
            };
            let charge = &mut observed.current_charge;
            charge.host_charged_bytes = charge
                .payload_bytes
                .saturating_add(charge.metadata_bytes)
                .saturating_add(charge.requested_stack_bytes);
            if !self.scope_open && worker.joined {
                observed.status = if retained {
                    "closed_retained"
                } else {
                    "closed"
                };
            }
            if sample.ledger_poisoned || worker.poisoned || owner_poisoned {
                observed.worker_unknown = true;
                observed.current_charge.unknown_packed_bytes = observed.current_charge.packed_bytes;
                observed.current_charge.unknown_stack_bytes =
                    observed.current_charge.requested_stack_bytes;
                observed.status = "unknown_ledger_poisoned";
            } else if owner_quarantined
                || worker.custody_unknown
                || budget.unknown_slots != 0
                || budget.worker_stack.unknown_bytes != 0
            {
                observed.worker_unknown = true;
                observed.status = "quarantined";
            }
            if !self.scope_open && budget.reserved_slots == 0 && !sample.ledger_poisoned {
                observed.pending_read = None;
            }
        }
        observed.worker_read_observed =
            observed.stats.io_started_reads != 0 && observed.stats.proved_read_bytes != 0;
        observed
    }
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[derive(Debug)]
pub(crate) struct ExpertReadPlan {
    pub image: u64,
    pub layer: usize,
    pub expert: usize,
    pub parameters: [BoundParameter; 3],
    pub projections: [NumericFp8Read; 3],
    pub packed: CheckpointReadPlan,
    pub ranges: [(Range<usize>, Range<usize>); 3],
    metadata_bytes: u64,
}

impl ExpertReadPlan {
    fn new(image: u64, metadata: &ExpertMetadata, reader: &CheckpointTensorReader) -> Result<Self> {
        if image == 0 {
            return Err(invalid("image generation must be nonzero"));
        }
        let ParameterResidency::Expert { layer, expert } = *metadata.parameters()[0].residency()
        else {
            return Err(invalid("expert residency required"));
        };
        let parameters = metadata.parameters().clone();
        let mut reads = Vec::with_capacity(3);
        let mut sources = BTreeMap::new();
        let mut cursor = 0usize;
        let mut ranges = Vec::with_capacity(3);
        let mut metadata_bytes = 4096u64;
        for parameter in &parameters {
            if parameter.id() != parameter.canonical_id()
                || parameter.residency() != &(ParameterResidency::Expert { layer, expert })
            {
                return Err(invalid("noncanonical or foreign expert projection"));
            }
            let source = parameter.numeric_fp8_source_metadata(
                parameter
                    .numeric_fp8_encoding()
                    .ok_or_else(|| invalid("numeric FP8 pair required"))?,
            )?;
            if source.expert_count().is_some() {
                return Err(invalid("individual full-matrix expert bindings required"));
            }
            let [rows, columns] = source.matrix_shape();
            let read = source.plan_tile_metadata(reader, None, 0..rows, 0..columns)?;
            let weight_end = cursor
                .checked_add(read.weight_read().read_plan().storage_bytes() as usize)
                .ok_or_else(|| invalid("packed weight range overflows"))?;
            let scale_end = weight_end
                .checked_add(read.scale_read().read_plan().storage_bytes() as usize)
                .ok_or_else(|| invalid("packed scale range overflows"))?;
            ranges.push((cursor..weight_end, weight_end..scale_end));
            cursor = scale_end;
            for source in read.read_plan().source_files() {
                if let Some(previous) =
                    sources.insert(source.catalog_path().to_path_buf(), source.clone())
                    && previous != *source
                {
                    return Err(invalid("conflicting catalog snapshots"));
                }
            }
            // Conservative allowance for retained binding/plan/ticket/session
            // clones and map nodes, including variable path and tensor names.
            let mut variable = 0u64;
            for slice in [source.weight(), source.scale()] {
                variable = variable
                    .checked_add((slice.path.as_os_str().len() + slice.name.len()) as u64)
                    .ok_or_else(|| invalid("metadata size overflows"))?;
            }
            for extent in read.read_plan().extents() {
                variable = variable
                    .checked_add(extent.path().as_os_str().len() as u64 + 256)
                    .ok_or_else(|| invalid("extent metadata size overflows"))?;
            }
            metadata_bytes = metadata_bytes
                .checked_add(
                    variable
                        .checked_mul(16)
                        .and_then(|n| n.checked_add(4096))
                        .ok_or_else(|| invalid("metadata size overflows"))?,
                )
                .ok_or_else(|| invalid("metadata size overflows"))?;
            reads.push(read);
        }
        let packed = CheckpointReadPlan::new(
            reads
                .iter()
                .flat_map(|read| read.read_plan().extents().iter().cloned()),
            Arc::from(sources.into_values().collect::<Vec<_>>()),
        )?;
        if packed.storage_bytes() > reader.max_tensor_bytes() {
            return Err(invalid(
                "whole expert exceeds the shared checkpoint reader batch bound",
            ));
        }
        Ok(Self {
            image,
            layer,
            expert,
            parameters,
            projections: reads.try_into().expect("three projection reads"),
            packed,
            ranges: ranges.try_into().expect("three packed ranges"),
            metadata_bytes,
        })
    }

    pub fn validate_binding(&self, index: usize, binding: &BoundParameter) -> Result<()> {
        let expected = self
            .parameters
            .get(index)
            .ok_or_else(|| invalid("projection index out of range"))?;
        if !expected.shares_storage_with(binding)
            || expected.id() != binding.id()
            || expected.canonical_id() != binding.canonical_id()
            || expected.role() != binding.role()
            || expected.residency() != binding.residency()
            || expected.spec().shape() != binding.spec().shape()
        {
            return Err(invalid(
                "prefetched projection belongs to another bound image",
            ));
        }
        Ok(())
    }

    fn validate_completion(&self, ticket: &ReadTicket, completion: &ReadCompletion) -> Result<()> {
        if completion.ticket() != ticket
            || ticket.source_generation() != SourceGeneration::new(self.image)
            || ticket.read_plan() != &self.packed
            || completion.receipt().operation != ticket.operation()
            || completion.receipt().source_generation != ticket.source_generation()
        {
            return Err(invalid(
                "prefetched completion image/operation/plan mismatch",
            ));
        }
        if let ReadOutcome::SourceVerified(bytes) = completion.outcome() {
            if bytes.len() as u64 != self.packed.storage_bytes() {
                return Err(invalid("prefetched byte length mismatch"));
            }
            self.packed
                .validate_source_identity()
                .map_err(|_| invalid("stale prefetched source"))?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct PendingRead {
    key: (usize, usize),
    ticket: std::result::Result<ReadTicket, ReadPipelineError>,
    speculative: bool,
    reader_before: CheckpointReadCounters,
}

#[derive(Debug)]
pub(crate) struct CurrentExpertRead {
    pub plan: Arc<ExpertReadPlan>,
    pub completion: ReadCompletion,
    pub ticket: ReadTicket,
}

/// No CUDA handles, cache leases or runtime continuations enter the worker.
#[derive(Debug)]
pub(crate) struct ExpertReadSequence {
    options: ExpertReadPipelineOptions,
    plans: BTreeMap<(usize, usize), Arc<ExpertReadPlan>>,
    pipeline: Option<CheckpointReadPipeline>,
    join: ReadJoinHandoff,
    reader: CheckpointTensorReader,
    started: Instant,
    pending: Option<PendingRead>,
    current: Option<CurrentExpertRead>,
    artifacts: Vec<NumericFp8PayloadLiveness>,
    artifact_bytes: u64,
    deferred: Option<((usize, usize), Error)>,
    stats: ExpertReadPipelineStats,
    stats_base: ExpertReadPipelineStats,
    last_receipt: Option<ExpertReadHostReceipt>,
    last_submitted: Option<ExpertReadIdentity>,
    pub admission: Option<ExpertReadAdmission>,
    pub stats_sink: Option<Arc<Mutex<ExpertReadObservationState>>>,
}

impl ExpertReadSequence {
    #[cfg(test)]
    pub(crate) fn observation_test_handoff(&self) -> ReadJoinHandoff {
        self.join.clone()
    }

    pub fn for_owner(
        options: ExpertReadPipelineOptions,
        image: u64,
        selected: &[&ExpertMetadata],
        reader: CheckpointTensorReader,
        admission: ExpertReadAdmission,
    ) -> Result<Self> {
        let started = Instant::now();
        let plans = Self::select(options, image, selected, &reader)?;
        let parked = admission.checkout(options, image, &reader, &plans)?;
        let reused = parked.is_some();
        let pipeline = if let Some(pipeline) = parked {
            let ready = pipeline.ready_for_sequence();
            if let Err(error) = ready {
                admission.quarantine();
                admission.park(pipeline);
                return Err(pipeline_error(error));
            }
            // Retaining a thread/reader is never a retained freshness proof.
            // Sessions remain one-shot and recheck open handles pre AND post IO.
            if plans
                .values()
                .any(|plan| plan.packed.validate_source_identity().is_err())
            {
                admission.park(pipeline);
                return Err(invalid("stale source before expert worker reuse"));
            }
            pipeline
        } else {
            CheckpointReadPipeline::with_options(
                reader.clone(),
                options.limits()?,
                ReadPipelineOptions {
                    worker_stack_bytes: options.worker_stack_bytes,
                    max_worker_stack_bytes: options.worker_stack_bytes,
                },
            )
            .map_err(pipeline_error)?
        };
        let mut sequence = Self::from_pipeline(options, plans, pipeline, reader, started);
        sequence.stats.workers_created = u64::from(!reused);
        sequence.stats.worker_reuses = u64::from(reused);
        sequence.admission = Some(admission);
        Ok(sequence)
    }

    fn pipeline(&self) -> &CheckpointReadPipeline {
        self.pipeline
            .as_ref()
            .expect("sequence owns worker until teardown")
    }

    fn pipeline_mut(&mut self) -> &mut CheckpointReadPipeline {
        self.pipeline
            .as_mut()
            .expect("sequence owns worker until teardown")
    }

    fn select(
        options: ExpertReadPipelineOptions,
        image: u64,
        selected: &[&ExpertMetadata],
        reader: &CheckpointTensorReader,
    ) -> Result<BTreeMap<(usize, usize), Arc<ExpertReadPlan>>> {
        let payload_bound = options.payload_bound()?;
        let mut plans = BTreeMap::new();
        let mut metadata_bytes = 4096u64;
        for metadata in selected {
            let plan = ExpertReadPlan::new(image, metadata, reader)?;
            if plan
                .packed
                .storage_bytes()
                .checked_mul(2)
                .is_none_or(|n| n > payload_bound)
            {
                return Err(invalid("two exact expert payloads exceed host bound"));
            }
            metadata_bytes = metadata_bytes
                .checked_add(plan.metadata_bytes)
                .ok_or_else(|| invalid("selected metadata overflows"))?;
            if metadata_bytes > options.max_metadata_bytes {
                return Err(invalid("selected metadata exceeds host bound"));
            }
            if plans
                .insert((plan.layer, plan.expert), Arc::new(plan))
                .is_some()
            {
                return Err(invalid("duplicate selected expert"));
            }
        }
        Ok(plans)
    }

    fn from_pipeline(
        options: ExpertReadPipelineOptions,
        plans: BTreeMap<(usize, usize), Arc<ExpertReadPlan>>,
        pipeline: CheckpointReadPipeline,
        reader: CheckpointTensorReader,
        started: Instant,
    ) -> Self {
        Self {
            options,
            plans,
            join: pipeline.join_handoff(),
            pipeline: Some(pipeline),
            reader,
            started,
            pending: None,
            current: None,
            artifacts: Vec::new(),
            artifact_bytes: 0,
            deferred: None,
            stats: ExpertReadPipelineStats {
                sequences_started: 1,
                workers_created: 1,
                ..Default::default()
            },
            stats_base: ExpertReadPipelineStats::default(),
            last_receipt: None,
            last_submitted: None,
            admission: None,
            stats_sink: None,
        }
    }

    pub fn attach_observation(&mut self, sink: Arc<Mutex<ExpertReadObservationState>>) {
        let previous = sink.lock().unwrap_or_else(|p| p.into_inner()).clone();
        self.stats_base = previous.snapshot().stats;
        sink.lock()
            .unwrap_or_else(|p| p.into_inner())
            .prior_worker_joins = self.stats_base.workers_joined;
        self.stats_sink = Some(sink);
        self.publish("active");
        // Count requested metadata/stack even if every selected expert is a hit.
        let _ = self.observe_peak();
    }

    fn identity(&self, key: (usize, usize), ticket: &ReadTicket) -> ExpertReadIdentity {
        ExpertReadIdentity {
            image_generation: ticket.source_generation().get(),
            layer: key.0,
            expert: key.1,
            operation: ticket.operation().get(),
        }
    }

    fn publish(&self, status: &'static str) {
        let Some(sink) = &self.stats_sink else { return };
        let mut sink = sink.lock().unwrap_or_else(|p| p.into_inner());
        sink.observation.effective_mode = Some(self.options.schedule);
        sink.observation.status = status;
        sink.observation.stats = self.stats_base.with_sequence(self.stats);
        sink.observation.current_read = self
            .current
            .as_ref()
            .filter(|_| status == "active")
            .map(|read| self.identity((read.plan.layer, read.plan.expert), &read.ticket));
        sink.observation.pending_read = self
            .pending
            .as_ref()
            .filter(|_| status == "active" || status == "join_pending_at_scope_end")
            .and_then(|pending| {
                pending
                    .ticket
                    .as_ref()
                    .ok()
                    .map(|ticket| self.identity(pending.key, ticket))
            });
        sink.observation.last_receipt = self.last_receipt;
        sink.join = Some(self.join.clone());
        sink.admission_state = self
            .admission
            .as_ref()
            .map(|admission| Arc::downgrade(&admission.0.state));
        sink.artifact_bytes = self.artifact_bytes;
        sink.scope_open = status == "active";
    }

    fn record_completion(
        &mut self,
        pending: &PendingRead,
        completion: &ReadCompletion,
        wasted: bool,
    ) {
        let receipt = completion.receipt();
        self.stats.completed_reads += 1;
        self.stats.worker_started_reads += u64::from(receipt.worker_started_at.is_some());
        self.stats.io_started_reads += u64::from(receipt.io_started_at.is_some());
        let outcome = match completion.outcome() {
            ReadOutcome::SourceVerified(_) => {
                self.stats.verified_reads += 1;
                "source_verified"
            }
            ReadOutcome::Cancelled(_) => {
                self.stats.cancelled_reads += 1;
                "cancelled"
            }
            ReadOutcome::Failed(_) => {
                self.stats.failed_reads += 1;
                "failed"
            }
        };
        self.record_receipt(pending.key, receipt, outcome);
        if let Some(bytes) = receipt.io_bytes {
            self.stats.proved_read_bytes += bytes;
        }
        if let (Some(start), Some(end)) = (receipt.io_started_at, receipt.io_finished_at) {
            self.stats.io_duration_ns = self
                .stats
                .io_duration_ns
                .saturating_add(nanos(end.saturating_duration_since(start)));
        }
        let after = self.reader.read_counters();
        match (
            after
                .read_calls
                .checked_sub(pending.reader_before.read_calls),
            after
                .bytes_read
                .checked_sub(pending.reader_before.bytes_read),
        ) {
            (Some(calls), Some(bytes)) => {
                self.stats.reader_interval_calls =
                    self.stats.reader_interval_calls.saturating_add(calls);
                self.stats.reader_interval_bytes =
                    self.stats.reader_interval_bytes.saturating_add(bytes);
                if wasted {
                    self.stats.wasted_reader_interval_calls = self
                        .stats
                        .wasted_reader_interval_calls
                        .saturating_add(calls);
                    self.stats.wasted_reader_interval_bytes = self
                        .stats
                        .wasted_reader_interval_bytes
                        .saturating_add(bytes);
                }
            }
            _ => self.stats.unobserved_reader_intervals += 1,
        }
        if wasted {
            self.stats.wasted_reads += 1;
            if let Some(bytes) = receipt.io_bytes {
                self.stats.wasted_read_bytes += bytes;
            } else if receipt.io_started_at.is_some() {
                self.stats.unproved_wasted_reads += 1;
            }
        }
    }

    fn record_receipt(
        &mut self,
        key: (usize, usize),
        receipt: &ReadReceipt,
        outcome: &'static str,
    ) {
        let offset = |instant: Instant| nanos(instant.saturating_duration_since(self.started));
        self.last_receipt = Some(ExpertReadHostReceipt {
            identity: ExpertReadIdentity {
                image_generation: receipt.source_generation.get(),
                layer: key.0,
                expert: key.1,
                operation: receipt.operation.get(),
            },
            outcome,
            requested_bytes: receipt.requested_bytes,
            io_bytes: receipt.io_bytes,
            cancellation_requested: receipt.cancellation_requested,
            submitted_ns: offset(receipt.submitted_at),
            worker_started_ns: receipt.worker_started_at.map(offset),
            io_started_ns: receipt.io_started_at.map(offset),
            io_finished_ns: receipt.io_finished_at.map(offset),
            source_verified_ns: receipt.source_verified_at.map(offset),
            completed_ns: receipt.completed_at.map(offset),
            taken_ns: receipt.taken_at.map(offset),
        });
    }

    pub fn record_materialization_failure(&mut self) {
        self.stats.materialization_failures += 1;
        self.publish("active");
    }

    fn observe_peak(&mut self) -> Result<()> {
        let budget = self.pipeline().budget();
        self.stats.peak_packed_bytes = self.stats.peak_packed_bytes.max(budget.reserved_bytes);
        self.stats.peak_artifact_bytes = self.stats.peak_artifact_bytes.max(self.artifact_bytes);
        self.stats.peak_metadata_bytes = self
            .stats
            .peak_metadata_bytes
            .max(self.options.max_metadata_bytes);
        self.stats.peak_requested_stack_bytes = self
            .stats
            .peak_requested_stack_bytes
            .max(budget.worker_stack.reserved_bytes as u64);
        let live = budget
            .reserved_bytes
            .checked_add(self.artifact_bytes)
            .ok_or_else(|| invalid("live host payload overflows"))?;
        if live > self.options.payload_bound()? {
            return Err(invalid(
                "live packed plus prepared payload exceeds host bound",
            ));
        }
        self.stats.peak_payload_bytes = self.stats.peak_payload_bytes.max(live);
        let charged = live
            .checked_add(self.options.max_metadata_bytes)
            .and_then(|bytes| bytes.checked_add(self.options.worker_stack_bytes as u64))
            .ok_or_else(|| invalid("host charge overflows"))?;
        self.stats.peak_host_charged_bytes = self.stats.peak_host_charged_bytes.max(charged);
        self.publish("active");
        Ok(())
    }

    fn release_current(&mut self) -> Result<()> {
        if self
            .artifacts
            .iter()
            .any(NumericFp8PayloadLiveness::is_live)
        {
            return Err(invalid("current prepared host payload is still live"));
        }
        self.artifacts.clear();
        self.artifact_bytes = 0;
        self.current = None;
        Ok(())
    }

    fn submit(&mut self, key: (usize, usize), speculative: bool) -> Result<()> {
        if self.pending.is_some() {
            return Err(invalid("only the immediately next expert may be queued"));
        }
        let plan = self
            .plans
            .get(&key)
            .ok_or_else(|| invalid("expert was not selected before compute"))?;
        let reader_before = self.reader.read_counters();
        let image = plan.image;
        let packed = plan.packed.clone();
        let ticket = self
            .pipeline_mut()
            .submit(SourceGeneration::new(image), packed);
        if let Ok(ticket) = &ticket {
            self.last_submitted = Some(self.identity(key, ticket));
            self.stats.submitted_reads += 1;
            self.stats.speculative_reads += u64::from(speculative);
        } else {
            self.stats.rejected_submissions += 1;
        }
        self.pending = Some(PendingRead {
            key,
            ticket,
            speculative,
            reader_before,
        });
        self.observe_peak()
    }

    fn discard_pending(&mut self) -> Result<()> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };
        if let Ok(ticket) = &pending.ticket {
            self.pipeline_mut().cancel(ticket).map_err(pipeline_error)?;
            self.pipeline().wait(ticket).map_err(pipeline_error)?;
            let completion = self.pipeline_mut().take(ticket).map_err(pipeline_error)?;
            self.record_completion(&pending, &completion, true);
            drop(completion);
        }
        self.publish("active");
        Ok(())
    }

    /// Speculation never publishes a forward error. Submission and freshness
    /// failures are kept until this exact expert is demanded.
    pub fn lookahead(
        &mut self,
        current: (usize, usize),
        next: Option<(usize, usize)>,
        resident: bool,
    ) {
        let result = self.release_current().and_then(|()| {
            if self
                .pending
                .as_ref()
                .is_some_and(|pending| pending.key == current)
            {
                self.discard_pending()?;
            }
            self.stats.resident_skips += u64::from(resident);
            if self.options.schedule == ExpertReadSchedule::Blocking || resident {
                return Ok(());
            }
            if let Some(key) = next {
                self.submit(key, true)?;
            }
            Ok(())
        });
        if let Err(error) = result
            && let Some(key) = next
        {
            self.deferred = Some((key, error));
        }
        self.publish("active");
    }

    pub fn demand(&mut self, layer: usize, expert: usize) -> Result<&CurrentExpertRead> {
        self.stats.demanded_reads += 1;
        let result = self.demand_inner(layer, expert);
        if result.is_err() {
            self.stats.demand_failures += 1;
        }
        self.publish("active");
        result?;
        Ok(self.current.as_ref().expect("current demand"))
    }

    fn demand_inner(&mut self, layer: usize, expert: usize) -> Result<()> {
        let _span = tracing::trace_span!(target: "ferrule_performance", "expert_read_demand_host",
            layer, expert, schedule = ?self.options.schedule)
        .entered();
        self.release_current()?;
        let key = (layer, expert);
        if let Some((failed_key, error)) = self.deferred.take()
            && failed_key == key
        {
            return Err(error);
        }
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.key != key)
        {
            self.discard_pending()?;
        }
        if self.pending.is_none() {
            self.submit(key, false)?;
        }
        let pending = self.pending.take().expect("submitted demand");
        let ticket = match pending.ticket {
            Ok(ref ticket) => ticket.clone(),
            Err(error) => return Err(pipeline_error(error)),
        };
        let wait_started = Instant::now();
        let wait = self.pipeline().wait(&ticket);
        self.stats.demand_wait_calls += 1;
        self.stats.demand_wait_ns = self
            .stats
            .demand_wait_ns
            .saturating_add(nanos(wait_started.elapsed()));
        wait.map_err(pipeline_error)?;
        let completion = self.pipeline_mut().take(&ticket).map_err(pipeline_error)?;
        self.record_completion(&pending, &completion, false);
        tracing::trace!(target: "ferrule_performance", layer, expert, receipt = ?completion.receipt(),
            "expert read receipt; no GPU execution timing");
        let plan = Arc::clone(self.plans.get(&key).expect("selected demand"));
        plan.validate_completion(&ticket, &completion)?;
        if !matches!(completion.outcome(), ReadOutcome::SourceVerified(_)) {
            return Err(Error::ModelSource {
                source: Box::new(FailedRead {
                    completion,
                    _admission: self.admission.clone(),
                }),
            });
        }
        // Reserve the final immutable artifacts before checkpoint allocates
        // them, while the original packed completion is still charged.
        self.stats.prefetched_reads_taken += u64::from(pending.speculative);
        self.artifact_bytes = plan.packed.storage_bytes();
        self.current = Some(CurrentExpertRead {
            plan,
            completion,
            ticket: ticket.clone(),
        });
        self.observe_peak()?;
        Ok(())
    }

    pub fn track_artifacts(&mut self, expert: &PreparedSwiGlu) -> Result<()> {
        let current = self
            .current
            .as_ref()
            .ok_or_else(|| invalid("no current read custody"))?;
        for (index, linear) in [expert.gate(), expert.up(), expert.down()]
            .into_iter()
            .enumerate()
        {
            current
                .plan
                .validate_binding(index, linear.parameter().binding())?;
            if linear
                .numeric_fp8()
                .is_none_or(|artifact| artifact.provenance() != &current.plan.projections[index])
            {
                return Err(invalid("prepared artifact differs from the selected read"));
            }
        }
        self.artifacts = [expert.gate(), expert.up(), expert.down()]
            .into_iter()
            .map(|linear| {
                linear
                    .numeric_fp8()
                    .expect("validated numeric artifact")
                    .validated_payload()
                    .allocation_liveness()
            })
            .collect();
        self.observe_peak()
    }
}

impl Drop for ExpertReadSequence {
    fn drop(&mut self) {
        if self.admission.is_some() && !self.pipeline().on_worker() {
            let drained = self.discard_pending();
            self.current = None;
            if drained.is_ok() && self.pipeline().finish_sequence().is_ok() {
                if let Some(admission) = &self.admission {
                    admission.retain_payloads(&self.artifacts);
                }
                self.stats.sequences_finished += 1;
                self.stats.unobserved_terminal_reads = self
                    .stats
                    .submitted_reads
                    .saturating_sub(self.stats.completed_reads);
                self.publish("idle_retained");
                self.admission
                    .as_ref()
                    .expect("owner admission")
                    .park(self.pipeline.take().expect("owned worker"));
                return;
            }
        }
        // Joining proves only host-read custody, never CUDA consumer completion.
        let mut shutdown = self.pipeline_mut().shutdown();
        if shutdown.join_pending {
            shutdown = self.join.join();
        }
        if shutdown.join_pending {
            if let Some(admission) = &self.admission {
                admission.retain_join(self.join.clone());
            }
        } else {
            if let Err(error) = self.discard_pending() {
                tracing::warn!(%error, "expert read cleanup retained unproved host custody");
            }
            drop(self.pipeline_mut().drain());
        }
        if let Some(admission) = &self.admission
            && !shutdown.join_pending
        {
            admission.quarantine();
        }
        self.stats.unknown_bytes = self.pipeline().budget().unknown_bytes;
        if let Some(admission) = &self.admission {
            admission.retain_payloads(&self.artifacts);
        }
        let host_payload_live = self
            .artifacts
            .iter()
            .any(NumericFp8PayloadLiveness::is_live);
        if shutdown.custody_unknown || shutdown.join_failed || !shutdown.unknown.is_empty() {
            if let Some(admission) = &self.admission {
                // Keep this materializer's admission closed. A later successful
                // host drain cannot prove an earlier unknown operation.
                admission.quarantine();
            }
            tracing::warn!(
                ?shutdown,
                host_payload_live,
                "expert read worker requires quarantine"
            );
        }
        self.stats.sequences_finished += 1;
        self.stats.join_pending_sequences += u64::from(shutdown.join_pending);
        self.stats.join_failed_sequences += u64::from(shutdown.join_failed);
        self.stats.unknown_reads += shutdown.unknown.len() as u64;
        self.stats.unobserved_terminal_reads = self
            .stats
            .submitted_reads
            .saturating_sub(self.stats.completed_reads)
            .saturating_sub(self.stats.unknown_reads);
        if let Some(unknown) = shutdown.unknown.last()
            && let Some(identity) = self.last_submitted.filter(|identity| {
                identity.operation == unknown.ticket.operation().get()
                    && identity.image_generation == unknown.ticket.source_generation().get()
            })
        {
            self.record_receipt(
                (identity.layer, identity.expert),
                &unknown.receipt,
                "unknown",
            );
        }
        // The normal field teardown still owns these resources; the read-only
        // snapshot samples the retained handoff/aliases after this Drop returns.
        self.publish(
            if shutdown.custody_unknown || shutdown.join_failed || !shutdown.unknown.is_empty() {
                "quarantined"
            } else if shutdown.join_pending {
                "join_pending_at_scope_end"
            } else {
                "closed"
            },
        );
        tracing::trace!(target: "ferrule_performance", schedule = ?self.options.schedule,
            stats = ?self.stats, "host-only expert read sequence; GPU overlap unverified");
    }
}

#[cfg(test)]
#[path = "../../tests/expert_read_pipeline/mod.rs"]
pub(crate) mod tests;
