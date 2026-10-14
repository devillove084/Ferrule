//! Host-only staging for the existing synchronous routed-expert owner.
//! One next read, unchanged compressed representation and synchronous forward.
//! Host intervals are observable; real GPU overlap needs a correlated device trace.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use ferrule_common::numeric_fp8::NumericFp8PayloadLiveness;

use ferrule_common::{Error, Result, SourceGeneration};

use crate::checkpoint::read_pipeline::{
    CheckpointReadPipeline, ReadCompletion, ReadJoinHandoff, ReadOutcome, ReadPipelineError,
    ReadPipelineLimits, ReadPipelineOptions, ReadTicket,
};
use crate::checkpoint::{CheckpointReadPlan, CheckpointTensorReader, NumericFp8Read};
use crate::nn::ParameterResidency;

use super::{BoundParameter, ExpertMetadata, PreparedSwiGlu};

/// Scheduling only: both modes use the same worker, slots, byte and stack limits.
/// Neither mode changes the synchronous forward or its CUDA completion proofs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExpertReadSchedule {
    #[default]
    Blocking,
    /// One next bucket's host read may overlap current GPU execution. GPU overlap
    /// must be measured; submission timestamps do not establish that it occurred.
    Lookahead,
}

/// Explicit opt-in; an unconfigured materializer keeps its existing demand path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    payloads: Vec<NumericFp8PayloadLiveness>,
    join: Option<ReadJoinHandoff>,
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
            let mut state = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if state.active || state.quarantined {
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
            if shutdown.join_failed || !shutdown.unknown.is_empty() {
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
            .payloads = payloads.to_vec();
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
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .active = false;
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
        for parameter in resources.state_dict().parameters() {
            if let ParameterResidency::Expert { layer, expert } = *parameter.residency() {
                bindings.entry((layer, expert)).or_default().push(parameter);
            }
        }
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
            let worst = metadata
                .into_iter()
                .take(selected)
                .try_fold(4096u64, |total, bytes| {
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

/// Counts host work only; these counters are not evidence of GPU execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertReadPipelineStats {
    pub submitted_reads: u64,
    pub speculative_reads: u64,
    pub resident_skips: u64,
    pub wasted_reads: u64,
    /// Proved whole-read bytes only; failed/partial reads can have unproved bytes.
    pub wasted_read_bytes: u64,
    pub unproved_wasted_reads: u64,
    pub peak_payload_bytes: u64,
    pub unknown_bytes: u64,
    /// Includes metadata grant and requested stack, not allocator/OS overhead.
    pub peak_host_charged_bytes: u64,
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
    pipeline: CheckpointReadPipeline,
    join: ReadJoinHandoff,
    pending: Option<PendingRead>,
    current: Option<CurrentExpertRead>,
    artifacts: Vec<NumericFp8PayloadLiveness>,
    artifact_bytes: u64,
    deferred: Option<((usize, usize), Error)>,
    stats: ExpertReadPipelineStats,
    pub admission: Option<ExpertReadAdmission>,
    pub stats_sink: Option<Arc<Mutex<ExpertReadPipelineStats>>>,
}

impl ExpertReadSequence {
    pub fn new(
        options: ExpertReadPipelineOptions,
        image: u64,
        selected: &[&ExpertMetadata],
        reader: CheckpointTensorReader,
    ) -> Result<Self> {
        let plans = Self::select(options, image, selected, &reader)?;
        let pipeline = CheckpointReadPipeline::with_options(
            reader,
            options.limits()?,
            ReadPipelineOptions {
                worker_stack_bytes: options.worker_stack_bytes,
                max_worker_stack_bytes: options.worker_stack_bytes,
            },
        )
        .map_err(pipeline_error)?;
        Ok(Self::from_pipeline(options, plans, pipeline))
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
    ) -> Self {
        Self {
            options,
            plans,
            join: pipeline.join_handoff(),
            pipeline,
            pending: None,
            current: None,
            artifacts: Vec::new(),
            artifact_bytes: 0,
            deferred: None,
            stats: ExpertReadPipelineStats::default(),
            admission: None,
            stats_sink: None,
        }
    }

    fn observe_peak(&mut self) -> Result<()> {
        let live = self
            .pipeline
            .budget()
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
        let ticket = self
            .pipeline
            .submit(SourceGeneration::new(plan.image), plan.packed.clone());
        if ticket.is_ok() {
            self.stats.submitted_reads += 1;
            self.stats.speculative_reads += u64::from(speculative);
        }
        self.pending = Some(PendingRead { key, ticket });
        self.observe_peak()
    }

    fn discard_pending(&mut self) -> Result<()> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };
        if let Ok(ticket) = pending.ticket {
            self.pipeline.cancel(&ticket).map_err(pipeline_error)?;
            self.pipeline.wait(&ticket).map_err(pipeline_error)?;
            let completion = self.pipeline.take(&ticket).map_err(pipeline_error)?;
            self.stats.wasted_reads += 1;
            if let Some(bytes) = completion.receipt().io_bytes {
                self.stats.wasted_read_bytes += bytes;
            } else if completion.receipt().io_started_at.is_some() {
                self.stats.unproved_wasted_reads += 1;
            }
            drop(completion);
        }
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
            if self.options.schedule == ExpertReadSchedule::Blocking {
                return Ok(());
            }
            if resident {
                self.stats.resident_skips += 1;
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
    }

    pub fn demand(&mut self, layer: usize, expert: usize) -> Result<&CurrentExpertRead> {
        let _span = tracing::debug_span!(target: "ferrule_performance", "expert_read_demand_host",
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
        let ticket = pending.ticket.map_err(pipeline_error)?;
        self.pipeline.wait(&ticket).map_err(pipeline_error)?;
        let completion = self.pipeline.take(&ticket).map_err(pipeline_error)?;
        tracing::debug!(target: "ferrule_performance", layer, expert, receipt = ?completion.receipt(),
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
        self.artifact_bytes = plan.packed.storage_bytes();
        self.current = Some(CurrentExpertRead {
            plan,
            completion,
            ticket,
        });
        self.observe_peak()?;
        Ok(self.current.as_ref().expect("current demand"))
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
        // Joining proves only host-read custody, never CUDA consumer completion.
        let mut shutdown = self.pipeline.shutdown();
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
            drop(self.pipeline.drain());
        }
        self.stats.unknown_bytes = self.pipeline.budget().unknown_bytes;
        if let Some(admission) = &self.admission {
            admission.retain_payloads(&self.artifacts);
        }
        let host_payload_live = self
            .artifacts
            .iter()
            .any(NumericFp8PayloadLiveness::is_live);
        if shutdown.join_failed || !shutdown.unknown.is_empty() {
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
        if let Some(sink) = &self.stats_sink {
            let mut total = sink.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            total.submitted_reads = total
                .submitted_reads
                .saturating_add(self.stats.submitted_reads);
            total.speculative_reads = total
                .speculative_reads
                .saturating_add(self.stats.speculative_reads);
            total.resident_skips = total
                .resident_skips
                .saturating_add(self.stats.resident_skips);
            total.wasted_reads = total.wasted_reads.saturating_add(self.stats.wasted_reads);
            total.wasted_read_bytes = total
                .wasted_read_bytes
                .saturating_add(self.stats.wasted_read_bytes);
            total.unproved_wasted_reads = total
                .unproved_wasted_reads
                .saturating_add(self.stats.unproved_wasted_reads);
            total.peak_payload_bytes = total.peak_payload_bytes.max(self.stats.peak_payload_bytes);
            total.unknown_bytes = total.unknown_bytes.saturating_add(self.stats.unknown_bytes);
            total.peak_host_charged_bytes = total
                .peak_host_charged_bytes
                .max(self.stats.peak_host_charged_bytes);
        }
        tracing::debug!(target: "ferrule_performance", schedule = ?self.options.schedule,
            stats = ?self.stats, "host-only expert read sequence; GPU overlap unverified");
    }
}

#[cfg(test)]
#[path = "../../tests/expert_read_pipeline/mod.rs"]
pub(crate) mod tests;
