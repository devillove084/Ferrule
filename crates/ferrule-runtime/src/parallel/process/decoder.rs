//! Process adapter for the production pipeline dispatcher. The parent owns only
//! host metadata; one child owns the segment, sequences and physical KV journal.
//! Loss is terminal for the transport: neither rollback nor poll is replayed to
//! a replacement owner, and OS reaping is never a physical retirement receipt.

use crate::parallel::pipeline::PhysicalKvLayout;
use ferrule_model::ModelExecutionBackend;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::topology::ExpertSourceScope;
use ferrule_common::{Error, ParallelRankId};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::moe::ExpertId;
use ferrule_model::transformer::{BoundDecoderResources, LayerSegmentPlan};
use serde::{Deserialize, Serialize};

use super::decoder_expert::ProcessExperts;
use super::decoder_stage::prepare as prepare_stage;
use super::*;
use crate::parallel::data::ReplicaWorker;
use crate::parallel::expert::ExpertGroup;
use crate::parallel::pipeline::completion_boundary::{
    self, Boundary, CompletionClass, CompletionContext,
};
use crate::parallel::pipeline::{
    BoxedPipelineStageWorker, PipelineCommand, PipelineConfig, PipelineRank, PipelineReply,
    PipelineStage, PipelineStageBoot, PipelineStageDescription, PipelineStageWorker,
    PipelineTransport,
};
use ferrule_model::transformer::expert_parallel::{
    ExpertDispatchContext, ExpertDispatchLimits, ExpertPlacement, ExpertPlacementEntry,
    ExpertPlacementEpoch, ExpertRouteMetadata,
};

pub use super::decoder_endpoint::{DecoderBindings, DecoderEndpoint};
pub use super::decoder_expert::{
    ExpertBoot, ExpertCommand, ExpertProcessStats, ExpertReply, ExpertRouteTokenFrame,
    ExpertTokenFrame,
};
pub use super::decoder_hybrid::{HybridBootFrame, HybridCudaBudgetFrame, HybridLayerFrame};
pub use super::decoder_wire::{DecoderCommand, DecoderProcessStats, DecoderReply};
pub(super) type Result<T> = ferrule_common::Result<T>;
pub(super) fn error(message: impl std::fmt::Display) -> Error {
    Error::Execution {
        message: format!("process decoder: {message}"),
    }
}

// Hybrid schemas and physical capacity require coordinated parent/child restart.
// Older Boots cannot establish the hybrid state/geometry contract.
pub const DECODER_WIRE_VERSION: u16 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderDevice {
    Cpu,
    Cuda { ordinal: usize },
}
impl DecoderDevice {
    /// Metadata conversion; does not require or initialize the selected backend.
    pub const fn backend(self) -> ferrule_model::ModelExecutionBackend {
        match self {
            Self::Cpu => ModelExecutionBackend::Cpu,
            Self::Cuda { .. } => ModelExecutionBackend::Cuda,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderPrecision {
    F32,
    Bf16Compatibility,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderRecipeKind {
    Qwen3Dense,
    Qwen3Moe,
    Qwen35DenseHybrid,
    Synthetic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentFrame {
    pub total_layers: usize,
    pub layers: std::ops::Range<usize>,
    pub embedding: bool,
    pub output: bool,
}
impl SegmentFrame {
    pub fn encode(plan: &LayerSegmentPlan) -> Self {
        Self {
            total_layers: plan.total_layers(),
            layers: plan.layers(),
            embedding: plan.owns_embedding(),
            output: plan.owns_output(),
        }
    }
    pub fn decode(&self) -> Result<LayerSegmentPlan> {
        LayerSegmentPlan::new(
            self.total_layers,
            self.layers.clone(),
            self.embedding,
            self.output,
        )
        .map_err(error)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvConfigFrame {
    pub page_size: usize,
    /// Logical page capacity.
    pub max_pages: usize,
    /// Required physical slots; never inferred for old peers.
    pub physical_pages: usize,
    pub max_positions: usize,
    pub max_batch_tokens: usize,
    pub session_capacity: usize,
    pub max_parameter_bytes: u64,
    pub max_ack_polls: usize,
}
impl KvConfigFrame {
    /// Standard CPU encoder: physical slots equal logical pages.
    /// Hybrid callers use `encode_for_hybrid`; standard CUDA uses `encode_for_device`.
    pub fn encode(c: PipelineConfig) -> Self {
        Self {
            page_size: c.page_size,
            max_pages: c.max_pages,
            physical_pages: c.max_pages,
            max_positions: c.max_positions,
            max_batch_tokens: c.max_batch_tokens,
            session_capacity: c.session_capacity,
            max_parameter_bytes: c.max_parameter_bytes,
            max_ack_polls: c.max_ack_polls,
        }
    }
    /// Encode the explicit backend capacity without initializing that backend.
    pub fn encode_for_device(c: PipelineConfig, device: DecoderDevice) -> Result<Self> {
        let mut frame = Self::encode(c);
        frame.physical_pages = c.physical_pages(PhysicalKvLayout::standard(device.backend()))?;
        Ok(frame)
    }

    /// Hybrid CPU and CUDA both retain shadow/COW slots in the working journal.
    pub fn encode_for_hybrid(c: PipelineConfig, device: DecoderDevice) -> Result<Self> {
        let mut frame = Self::encode(c);
        frame.physical_pages = c.physical_pages(PhysicalKvLayout::hybrid(device.backend()))?;
        Ok(frame)
    }

    fn decode(
        &self,
        precision: DecoderPrecision,
        layout: PhysicalKvLayout,
    ) -> Result<PipelineConfig> {
        let c = PipelineConfig {
            page_size: self.page_size,
            max_pages: self.max_pages,
            max_positions: self.max_positions,
            max_batch_tokens: self.max_batch_tokens,
            session_capacity: self.session_capacity,
            max_parameter_bytes: self.max_parameter_bytes,
            max_ack_polls: self.max_ack_polls,
            precision: match precision {
                DecoderPrecision::F32 => ExecutionPrecisionPolicy::f32(),
                DecoderPrecision::Bf16Compatibility => {
                    ExecutionPrecisionPolicy::bf16_compatibility()
                }
            },
        };
        c.validate()?;
        if self.physical_pages != c.physical_pages(layout)? {
            return Err(error(
                "decoder physical capacity differs from backend L/2L policy",
            ));
        }
        Ok(c)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertPlacementFrame {
    pub source_scope: ExpertSourceScope,
    pub source: u32,
    pub members: Vec<u32>,
    /// Legacy fixed routes: `(layer, logical expert, owner)`.
    pub entries: Vec<(usize, usize, u32)>,
    /// Explicit route placement. Its presence is the wire discriminator for
    /// physical slots, replicas, and a placement epoch; it is never inferred
    /// from `entries` or from the logical primary owner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_entries: Option<Vec<ExpertPlacementEntryFrame>>,
    /// Zero is reserved for the legacy fixed placement. Explicit route
    /// entries require a nonzero epoch, so they cannot silently enter legacy.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub placement_epoch: u64,
    pub max_tokens: usize,
    pub max_bytes: usize,
    /// One device per independent expert owner, in `members` order.
    #[serde(default)]
    pub devices: Vec<DecoderDevice>,
    /// Trusted finite timeout for the nested EP owners.
    #[serde(default = "default_expert_timeout_ms")]
    pub timeout_ms: u64,
}

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertPlacementEntryFrame {
    pub layer: usize,
    pub expert: usize,
    pub physical_slot: usize,
    pub replica: usize,
    pub owner: u32,
}

/// Route identity carried by the explicit process expert command. It contains
/// only model routing metadata: no pointer, event, weight, residency, or
/// transaction handle crosses the process boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertRouteFrame {
    pub placement_epoch: u64,
    pub layer: usize,
    pub expert: usize,
    pub physical_slot: usize,
    pub replica: usize,
    pub owner: u32,
}

impl ExpertRouteFrame {
    pub fn encode(metadata: ExpertRouteMetadata) -> Self {
        Self {
            placement_epoch: metadata.placement_epoch.get(),
            layer: metadata.logical_expert.layer,
            expert: metadata.logical_expert.expert,
            physical_slot: metadata.physical_slot,
            replica: metadata.replica,
            owner: metadata.owner_rank.get(),
        }
    }

    pub fn metadata(self) -> ExpertRouteMetadata {
        ExpertRouteMetadata::new(
            ExpertId::new(self.layer, self.expert),
            self.physical_slot,
            self.replica,
            ParallelRankId::new(self.owner),
            ExpertPlacementEpoch::new(self.placement_epoch),
        )
    }
}

/// Validate route metadata at the receiving owner, before model execution.
/// Error text deliberately includes the route coordinates and epoch so a
/// rejected stale/future/wrong-owner message is observable without exposing
/// any storage or device handle.
pub(crate) fn validate_expert_route(
    placement: &ExpertPlacement,
    expected_owner: ParallelRankId,
    context: ExpertDispatchContext,
    token_expert: ExpertId,
    frame: ExpertRouteFrame,
) -> Result<ExpertRouteMetadata> {
    let metadata = frame.metadata();
    let route = format!(
        "transaction={} source={} logical={}:{} slot={} replica={} owner={} epoch={}",
        context.transaction.get(),
        context.source_rank.get(),
        metadata.logical_expert.layer,
        metadata.logical_expert.expert,
        metadata.physical_slot,
        metadata.replica,
        metadata.owner_rank.get(),
        metadata.placement_epoch.get()
    );
    if metadata.placement_epoch != placement.epoch() {
        let reason = if metadata.placement_epoch < placement.epoch() {
            "stale"
        } else {
            "future"
        };
        return Err(error(format!(
            "expert route validation: {reason} placement epoch in {route}, current={}",
            placement.epoch().get()
        )));
    }
    if metadata.logical_expert != token_expert || metadata.logical_expert.layer != context.layer {
        return Err(error(format!(
            "expert route validation: logical expert mismatch in {route}, token={}:{}",
            token_expert.layer, token_expert.expert
        )));
    }
    let Some(entry) = placement.entry(metadata.logical_expert, metadata.physical_slot) else {
        return Err(error(format!(
            "expert route validation: wrong physical slot in {route}"
        )));
    };
    if entry.replica != metadata.replica {
        return Err(error(format!(
            "expert route validation: wrong replica in {route}, registered={}",
            entry.replica
        )));
    }
    if entry.owner_rank != metadata.owner_rank || metadata.owner_rank != expected_owner {
        return Err(error(format!(
            "expert route validation: wrong owner in {route}, registered={}, receiver={}",
            entry.owner_rank.get(),
            expected_owner.get()
        )));
    }
    Ok(metadata)
}
fn default_expert_timeout_ms() -> u64 {
    30_000
}

impl ExpertPlacementFrame {
    pub fn decode(&self, layers: std::ops::Range<usize>) -> Result<ExpertGroup> {
        let explicit_entries = self.route_entries.as_deref();
        if self.members.is_empty()
            || self.members.len() > PROCESS_REAPER_CAPACITY
            || (self.devices.len() != self.members.len() && !self.devices.is_empty())
            || self.timeout_ms == 0
            || match self.source_scope {
                ExpertSourceScope::Member => !self.members.contains(&self.source),
                ExpertSourceScope::ExternalStage => self.members.contains(&self.source),
                ExpertSourceScope::ExternalTensorSource => true,
            }
            || self
                .members
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.members.len()
            || self.max_tokens == 0
            || self.max_bytes == 0
            || (explicit_entries.is_none() && self.placement_epoch != 0)
            || (explicit_entries.is_some() && self.placement_epoch == 0)
            || (explicit_entries.is_some() && !self.entries.is_empty())
            || self
                .entries
                .iter()
                .any(|(layer, _, owner)| !layers.contains(layer) || !self.members.contains(owner))
            || explicit_entries.is_some_and(|entries| {
                entries.iter().any(|entry| {
                    !layers.contains(&entry.layer) || !self.members.contains(&entry.owner)
                })
            })
        {
            return Err(error(
                "invalid expert placement or missing explicit versioned route entries",
            ));
        }
        let placement = match explicit_entries {
            Some(entries) => ExpertPlacement::from_entries(
                ExpertPlacementEpoch::new(self.placement_epoch),
                entries.iter().map(|entry| ExpertPlacementEntry {
                    logical_expert: ExpertId::new(entry.layer, entry.expert),
                    physical_slot: entry.physical_slot,
                    replica: entry.replica,
                    owner_rank: ParallelRankId::new(entry.owner),
                }),
                self.members.iter().copied().map(ParallelRankId::new),
            )?,
            None => ExpertPlacement::new(
                self.entries
                    .iter()
                    .map(|&(l, e, o)| (l, e, ParallelRankId::new(o))),
            )?,
        };
        Ok(ExpertGroup {
            source_rank: ParallelRankId::new(self.source),
            members: self
                .members
                .iter()
                .copied()
                .map(ParallelRankId::new)
                .collect(),
            layers,
            placement,
            limits: ExpertDispatchLimits {
                max_tokens: self.max_tokens,
                max_bytes: self.max_bytes,
            },
        })
    }
}

/// Versioned Boot contains checkpoint location and execution policy only.
/// The checkpoint is opened/materialized by the child, never by this transport.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderBoot {
    pub version: u16,
    pub rank: u32,
    pub checkpoint: PathBuf,
    pub recipe: DecoderRecipeKind,
    pub segment: SegmentFrame,
    pub precision: DecoderPrecision,
    pub device: DecoderDevice,
    pub kv: KvConfigFrame,
    pub experts: Option<ExpertPlacementFrame>,
    pub hybrid: Option<HybridBootFrame>,
}
impl DecoderBoot {
    pub fn stage_boot(&self) -> Result<PipelineStageBoot> {
        if self.version != DECODER_WIRE_VERSION
            || self.checkpoint.as_os_str().is_empty()
            || matches!(self.device, DecoderDevice::Cuda { ordinal } if i32::try_from(ordinal).is_err())
            || (matches!(self.device, DecoderDevice::Cuda { .. }) || self.experts.is_some())
                && self.precision != DecoderPrecision::F32
        {
            return Err(error("unsupported decoder boot version/device/precision"));
        }
        // Wire tag/schema association is protocol validation, not family binding.
        if (self.recipe == DecoderRecipeKind::Qwen35DenseHybrid) != self.hybrid.is_some() {
            return Err(error("hybrid recipe/schema mismatch"));
        }
        if let Some(hybrid) = &self.hybrid {
            hybrid.validate(self)?;
        }
        let rank = ParallelRankId::new(self.rank);
        let boot = PipelineStageBoot {
            rank: PipelineRank {
                local: rank,
                global: rank,
            },
            plan: self.segment.decode()?,
            config: self.kv.decode(
                self.precision,
                match &self.hybrid {
                    Some(_) => PhysicalKvLayout::hybrid(self.device.backend()),
                    None => PhysicalKvLayout::standard(self.device.backend()),
                },
            )?,
            physical_pages: self.kv.physical_pages,
            program_spec: Vec::new(),
        };
        boot.validate()?;
        if let Some(experts) = &self.experts {
            let group = experts.decode(boot.plan.layers())?;
            if experts.source_scope != ExpertSourceScope::ExternalStage
                || experts.source != self.rank
            {
                return Err(error(
                    "process expert source must be the current external PP stage",
                ));
            }
            if group.members.contains(&rank) {
                return Err(error("expert owner overlaps pipeline rank"));
            }
            if experts.devices.len() != experts.members.len()
                || experts
                    .devices
                    .iter()
                    .any(|device| match (self.device, device) {
                        (DecoderDevice::Cpu, DecoderDevice::Cpu) => false,
                        (DecoderDevice::Cuda { .. }, DecoderDevice::Cuda { ordinal }) => {
                            i32::try_from(*ordinal).is_err()
                        }
                        _ => true,
                    })
            {
                return Err(error(
                    "expert devices must explicitly match the stage backend; no CPU fallback",
                ));
            }
        }
        Ok(boot)
    }
    pub fn description(
        &self,
        hidden: usize,
        vocabulary: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<PipelineStageDescription> {
        let boot = self.stage_boot()?;
        if let Some(hybrid) = &self.hybrid
            && (hidden, vocabulary, kv_heads, head_dim) != hybrid.geometry()?
        {
            return Err(error("hybrid description geometry differs from Boot"));
        }
        let description = PipelineStageDescription {
            plan: boot.plan,
            config: boot.config,
            physical_pages: boot.physical_pages,
            hidden,
            vocabulary,
            kv_heads,
            head_dim,
            expert_group: self
                .experts
                .as_ref()
                .map(|g| g.decode(self.segment.layers.clone()))
                .transpose()?,
        };
        description.validate()?;
        if let Some(experts) = &self.experts {
            description
                .validate_expert_source(experts.source_scope, ParallelRankId::new(self.rank))?;
        }
        Ok(description)
    }
    /// Validate an application-bound image without selecting a model family or
    /// opening a checkpoint. Numeric expert source identity has its own boot contract.
    pub fn validate_resources(&self, resources: &BoundDecoderResources) -> Result<()> {
        self.stage_boot()?;
        if resources.spec().layers().len() != self.segment.total_layers {
            return Err(error("checkpoint/segment layer count mismatch"));
        }
        if let Some(hybrid) = &self.hybrid {
            hybrid.validate_resources(self, resources)?;
        }
        if let Some(experts) = &self.experts {
            experts
                .decode(self.segment.layers.clone())?
                .validate_resources(resources)?;
        }
        Ok(())
    }
}

/// Factory seam for a device-local worker. `factory` runs after child Boot, not
/// during parent launch. Its errors must preserve unknown retained resources.
pub(super) struct Retained<T>(Option<Box<T>>);
impl<T> std::ops::Deref for Retained<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0.as_deref().expect("retained value")
    }
}
impl<T> std::ops::DerefMut for Retained<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.as_deref_mut().expect("retained value")
    }
}
impl<T> Retained<T> {
    pub(super) fn new(value: T) -> Self {
        Self(Some(Box::new(value)))
    }
    pub(super) fn release(mut self) -> T {
        *self.0.take().expect("retained value")
    }
}
impl<T> Drop for Retained<T> {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            std::mem::forget(value);
        }
    }
}

pub struct DecoderChild {
    identity: ProcessIdentity,
    boot: DecoderBoot,
    worker: BoxedPipelineStageWorker,
    experts: Option<super::decoder_stage::Experts>,
}
impl DecoderChild {
    /// Production child factory. `launch` re-executes the same hidden worker;
    /// each EP child has private pipes and inherits this PP isolation group.
    pub fn initialize_builtin(
        boot: ProcessBoot,
        launch: ProcessLaunch,
        bind: impl FnOnce(&DecoderBoot) -> Result<BoundDecoderResources>,
    ) -> std::result::Result<Self, ProcessHandlerError> {
        let identity = boot.identity;
        let limits = boot.limits;
        let mut experts = None;
        let mut child = Self::initialize(boot, |config| {
            let resources = bind(config)?;
            config.validate_resources(&resources)?;
            if config.experts.is_some() {
                experts = Some(std::rc::Rc::new(std::cell::RefCell::new(
                    ProcessExperts::spawn(config, identity, launch, limits)?,
                )));
            }
            prepare_stage(config, &resources, experts.clone())
        })?;
        child.experts = experts;
        Ok(child)
    }

    pub fn initialize(
        boot: ProcessBoot,
        factory: impl FnOnce(&DecoderBoot) -> Result<BoxedPipelineStageWorker>,
    ) -> std::result::Result<Self, ProcessHandlerError> {
        let config: DecoderBoot = serde_json::from_value(boot.config)
            .map_err(|e| fenced(ProcessFailureKind::Startup, e))?;
        config
            .stage_boot()
            .map_err(|e| fenced(ProcessFailureKind::Startup, e))?;
        config
            .validate_frame_limits(boot.limits)
            .map_err(|e| fenced(ProcessFailureKind::Startup, e))?;
        if boot.identity.rank.get() != config.rank {
            return Err(fenced(
                ProcessFailureKind::Startup,
                "Boot owner/rank mismatch",
            ));
        }
        let mut worker =
            factory(&config).map_err(|e| handler_failure(ProcessFailureKind::Startup, e))?;
        let d = worker.boot_description();
        let expected = config.description(d.hidden, d.vocabulary, d.kv_heads, d.head_dim);
        if !expected.is_ok_and(|expected| *d == expected)
            || worker.has_hybrid_state() != config.hybrid.is_some()
        {
            let cleanup = worker.shutdown();
            if let Err(e) = cleanup {
                std::mem::forget(worker);
                return Err(ProcessHandlerError::unknown(
                    ProcessFailureKind::Startup,
                    e.to_string(),
                ));
            }
            return Err(fenced(
                ProcessFailureKind::Startup,
                "factory description differs from Boot",
            ));
        }
        tracing::debug!(
            rank = config.rank,
            hybrid = config.hybrid.is_some(),
            physical_pages = config.kv.physical_pages,
            "validated process decoder owner"
        );
        Ok(Self {
            identity: boot.identity,
            boot: config,
            worker,
            experts: None,
        })
    }
    /// Bind an already validated checkpoint on the child owner. This seam also
    /// lets CPU-reference fixtures use tiny resources without weakening Qwen profiles.
    pub fn hybrid_factory(
        boot: &DecoderBoot,
        resources: &BoundDecoderResources,
    ) -> Result<BoxedPipelineStageWorker> {
        super::decoder_stage::prepare_hybrid(boot, resources)
    }
    pub fn cpu_factory(
        boot: &DecoderBoot,
        resources: &BoundDecoderResources,
    ) -> Result<BoxedPipelineStageWorker> {
        if boot.device != DecoderDevice::Cpu {
            return Err(error(
                "CUDA Boot requires a CUDA child factory; no CPU fallback",
            ));
        }
        if boot.hybrid.is_some() {
            return prepare_stage(boot, resources, None);
        }
        let stage_boot = boot.stage_boot()?;
        boot.validate_resources(resources)?;
        let stage = if let Some(placement) = &boot.experts {
            let group = placement.decode(stage_boot.plan.layers())?;
            let expert_resources = std::sync::Arc::new(resources.clone());
            let max = stage_boot.config.max_parameter_bytes;
            let experts =
                crate::parallel::expert::ExpertParallelExecutor::new(group, move |rank, group| {
                    crate::parallel::expert::ExpertRankWorker::prepare_cpu(
                        &expert_resources,
                        rank,
                        group,
                        max,
                    )
                })?;
            PipelineStage::prepare_cpu_with_experts(
                resources,
                stage_boot.plan.clone(),
                stage_boot.config,
                experts,
            )?
        } else {
            PipelineStage::prepare_cpu(resources, stage_boot.plan.clone(), stage_boot.config)?
        };
        Ok(PipelineStageWorker::new(&stage_boot, stage)?.boxed())
    }
}
fn fenced(kind: ProcessFailureKind, e: impl std::fmt::Display) -> ProcessHandlerError {
    ProcessHandlerError::fenced(kind, e.to_string())
}
fn handler_failure(kind: ProcessFailureKind, e: Error) -> ProcessHandlerError {
    if completion_boundary::is_unknown(&e) {
        std::mem::forget(e);
        return ProcessHandlerError::unknown(kind, "decoder quiescence unknown");
    }
    fenced(kind, e)
}
impl ProcessChildHandler for DecoderChild {
    type Command = DecoderCommand;
    type Output = DecoderReply;
    fn execute(
        &mut self,
        request: ProcessRequest<DecoderCommand>,
    ) -> std::result::Result<DecoderReply, ProcessHandlerError> {
        if request.identity.owner != self.identity
            || request.session != request.command.session()
            || request.command.key().is_some_and(|k| {
                k.rank != self.boot.rank || k.transaction != request.identity.transaction.get()
            })
        {
            return Err(fenced(
                ProcessFailureKind::CommandDecode,
                "decoder envelope/key mismatch",
            ));
        }
        let command = request
            .command
            .decode(self.worker.boot_description())
            .map_err(|e| fenced(ProcessFailureKind::CommandDecode, e))?;
        if matches!(&command, PipelineCommand::Execute { cancellation, .. } if cancellation.load(Ordering::Acquire))
        {
            return Err(fenced(
                ProcessFailureKind::Handler,
                "cancelled before child dispatch",
            ));
        }
        let reply = self.worker.dispatch(command).map_err(|e| {
            // The generic dispatcher cannot distinguish a failed physical KV
            // fence from an ordinary CUDA backend error. Fail closed; never
            // label unproven custody Fenced just because the host call returned.
            if matches!(self.boot.device, DecoderDevice::Cuda { .. }) {
                std::mem::forget(e);
                ProcessHandlerError::unknown(
                    ProcessFailureKind::Handler,
                    "CUDA dispatcher failed; physical quiescence unproven",
                )
            } else {
                handler_failure(ProcessFailureKind::Handler, e)
            }
        })?;
        let mut reply = DecoderReply::encode(reply, &self.boot).map_err(|e| {
            ProcessHandlerError::unknown(ProcessFailureKind::OutputEncode, e.to_string())
        })?;
        if let DecoderReply::Stats(stats) = &mut reply
            && let Some(experts) = &self.experts
        {
            stats.experts = experts.borrow_mut().process_stats().map_err(|e| {
                ProcessHandlerError::unknown(ProcessFailureKind::Handler, e.to_string())
            })?;
        }
        Ok(reply)
    }
    fn shutdown(&mut self) -> std::result::Result<(), ProcessHandlerError> {
        // The dispatcher refuses shutdown with active physical custody. Its
        // typed panic is caught by child_serve and retains the whole handler.
        self.worker
            .shutdown()
            .map_err(|e| ProcessHandlerError::unknown(ProcessFailureKind::Shutdown, e.to_string()))
    }
}

type Owner = ProcessRankOwner<DecoderBoot, DecoderCommand, DecoderReply>;
struct Rank {
    owner: Owner,
    boot: DecoderBoot,
    description: PipelineStageDescription,
    // Expected wire reply shape, not physical custody or a second transaction.
    prepared: Option<(super::decoder_wire::KeyFrame, usize)>,
}

pub struct ProcessPipelineTransport {
    ranks: BTreeMap<u32, Rank>,
    unavailable: bool,
}
impl ProcessPipelineTransport {
    pub fn spawn(
        launch: ProcessLaunch,
        boots: Vec<(ProcessIdentity, DecoderBoot)>,
        options: ProcessOwnerConfig,
    ) -> Result<Self> {
        if boots.iter().any(|(_, boot)| boot.hybrid.is_some()) {
            return Err(error(
                "hybrid process requires explicit compact layout factory",
            ));
        }
        Self::spawn_hybrid_admitted(launch, boots, options)
    }

    /// Only the compact-layout executor factory admits hybrid Boots. The public
    /// generic transport path must not silently build a standard logical KV map.
    pub(crate) fn spawn_hybrid_admitted(
        launch: ProcessLaunch,
        boots: Vec<(ProcessIdentity, DecoderBoot)>,
        options: ProcessOwnerConfig,
    ) -> Result<Self> {
        if boots.is_empty() || boots.len() > PROCESS_REAPER_CAPACITY {
            return Err(error("invalid decoder process count"));
        }
        let mut ids = std::collections::BTreeSet::new();
        for (identity, boot) in &boots {
            boot.stage_boot()?;
            boot.validate_frame_limits(options.frame_limits)?;
            if identity.rank.get() != boot.rank || !ids.insert(boot.rank) {
                return Err(error("duplicate/mismatched decoder owner"));
            }
        }
        for (_, boot) in &boots {
            if let Some(group) = &boot.experts {
                for &member in &group.members {
                    if !ids.insert(member) {
                        return Err(error("PP/EP owner IDs must be globally disjoint"));
                    }
                }
            }
        }
        if ids.len() > PROCESS_REAPER_CAPACITY {
            return Err(error("PP/EP group exceeds process capacity"));
        }
        let mut transport = Self {
            ranks: BTreeMap::new(),
            unavailable: false,
        };
        let startup = (|| {
            for (identity, boot) in boots {
                let context = CompletionContext {
                    rank: PipelineRank {
                        local: identity.rank,
                        global: identity.rank,
                    },
                    stage: "startup",
                    transaction: None,
                };
                let mut owner =
                    Owner::spawn(launch.clone(), identity, &boot, options).map_err(|failure| {
                        // Preserve the typed process evidence; owner Drop still
                        // transfers unreaped children to the existing reaper.
                        context.failure(
                            CompletionClass::Failed,
                            true,
                            Some(Box::new(failure.source)),
                        )
                    })?;
                let reply = owner
                    .execute(
                        ExecutionTransactionId::new(1)?,
                        0,
                        &DecoderCommand::Describe,
                    )
                    .map_err(|source| {
                        context.failure(CompletionClass::Failed, true, Some(Box::new(source)))
                    })?;
                let DecoderReply::Description {
                    boot: actual,
                    hidden,
                    vocabulary,
                    kv_heads,
                    head_dim,
                } = reply
                else {
                    return Err(error("missing child decoder description"));
                };
                if actual != boot {
                    return Err(error("child Boot mismatch"));
                }
                let description = boot.description(hidden, vocabulary, kv_heads, head_dim)?;
                transport.ranks.insert(
                    boot.rank,
                    Rank {
                        owner,
                        boot,
                        description,
                        prepared: None,
                    },
                );
            }
            Ok(())
        })();
        if let Err(source) = startup {
            let cleanup = transport.shutdown();
            return Err(Error::with_cleanup(
                "process pipeline startup",
                source,
                cleanup,
            ));
        }
        Ok(transport)
    }
    /// Retain a clone for PID-based statistics after passing the transport to
    /// PipelineParallelExecutor. No ThreadId is reconstructed from wire bytes.
    pub fn shared(self) -> SharedProcessPipelineTransport {
        SharedProcessPipelineTransport(std::rc::Rc::new(std::cell::RefCell::new(self)))
    }

    pub fn process_stats(&mut self, rank: u32) -> Result<DecoderProcessStats> {
        if self.unavailable {
            return Err(completion_boundary::quarantined());
        }
        let entry = self
            .ranks
            .get_mut(&rank)
            .ok_or_else(|| error("unknown rank"))?;
        let context = CompletionContext {
            rank: PipelineRank {
                local: ParallelRankId::new(rank),
                global: ParallelRankId::new(rank),
            },
            stage: "stats",
            transaction: Some(ExecutionTransactionId::new(1)?),
        };
        match entry
            .owner
            .execute(ExecutionTransactionId::new(1)?, 0, &DecoderCommand::Stats)
        {
            Ok(DecoderReply::Stats(stats))
                if stats.rank == rank && Some(stats.pid) == entry.owner.pid() =>
            {
                Ok(stats)
            }
            Err(e) if e.is_command_rejection() => {
                Err(context.failure(CompletionClass::Failed, false, Some(Box::new(e))))
            }
            Err(e) => Err(self.lost(context, e)),
            Ok(_) => Err(self.lost(context, error("invalid process statistics"))),
        }
    }
    fn lost(
        &mut self,
        context: CompletionContext,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Error {
        self.quarantine();
        context.failure(CompletionClass::Unknown, true, Some(Box::new(source)))
    }
}
impl PipelineTransport for ProcessPipelineTransport {
    fn call_observed(
        &mut self,
        rank: PipelineRank,
        command: PipelineCommand,
        poll: &mut dyn FnMut(),
    ) -> Result<PipelineReply> {
        if self.unavailable {
            return Err(completion_boundary::quarantined());
        }
        if rank.local != rank.global {
            return Err(error("rank mapping mismatch"));
        }
        if matches!(command, PipelineCommand::Stats) {
            return Err(error(
                "ThreadId statistics are not portable; use SharedProcessPipelineTransport::process_stats",
            ));
        }
        // ProcessRankOwner observes cancellation during finite pipe waits, but
        // admitted child work still drains before rollback. No flag address or
        // claimed interrupt crosses IPC.
        poll();
        let wire = DecoderCommand::encode(&command)?;
        if wire.key().is_some_and(|k| k.rank != rank.global.get()) {
            return Err(error("command rank mismatch"));
        }
        let tx = ExecutionTransactionId::new(wire.key().map_or(1, |k| k.transaction))?;
        let mut context = command.completion_context(rank);
        context.transaction = Some(tx);
        let boundary = Boundary::new("pipeline.process", context);
        let result = (|| {
            let entry = self
                .ranks
                .get_mut(&rank.global.get())
                .ok_or_else(|| error("unknown pipeline process rank"))?;
            if let DecoderCommand::Execute { key, .. } = &wire
                && entry
                    .prepared
                    .as_ref()
                    .is_none_or(|(expected, _)| expected != key)
            {
                return Err(error("Execute has no matching preparation projection"));
            }
            let reply = entry
                .owner
                .execute_observed(tx, wire.session(), &wire, poll);
            // Exact byte/phase timing remains in IPC, without re-encoding payloads.
            let reply = reply.map_err(|source| {
                let rejected = source.is_command_rejection();
                context.failure(
                    if rejected {
                        CompletionClass::Failed
                    } else {
                        CompletionClass::Unknown
                    },
                    !rejected,
                    Some(Box::new(source)),
                )
            });
            poll();
            let reply = match reply {
                Ok(reply) => reply,
                Err(e) if !completion_boundary::needs_quarantine(&e) => return Err(e),
                Err(e) => return Err(self.lost(context, e)),
            };
            let decoded = reply.decode(&wire, &entry.description, &entry.boot);
            let reply = match decoded {
                Ok(r) => r,
                Err(e) => return Err(self.lost(context, e)),
            };
            if let PipelineReply::Executed { output } = &reply {
                let rows = entry.prepared.as_ref().expect("checked preparation").1;
                if let Err(e) = entry.description.validate_output(output, rows) {
                    return Err(self.lost(context, e));
                }
            }
            if let DecoderCommand::Prepare { key, batch, .. } = &wire {
                entry.prepared = Some((key.clone(), batch.tokens.len()));
            }
            if matches!(
                wire,
                DecoderCommand::Finish { .. } | DecoderCommand::Rollback { .. }
            ) && matches!(
                reply,
                PipelineReply::Ack(ferrule_model::decoder::KvEndProgress::Complete)
            ) {
                entry.prepared = None;
            }
            if let PipelineCommand::Execute { cancellation, .. } = command
                && cancellation.load(Ordering::Acquire)
            {
                return Err(error("decoder cancelled after draining child execution"));
            }
            Ok(reply)
        })();
        // Transport receipt is not semantic success: decode, output validation,
        // ACK bookkeeping and cancellation must settle before the single event.
        boundary.finish(&result);
        result
    }
    fn outstanding(&self) -> usize {
        0
    }
    fn shutdown(&mut self) -> Result<()> {
        if self.unavailable {
            return Err(completion_boundary::quarantined());
        }
        let mut failures = Vec::new();
        for entry in self.ranks.values_mut() {
            if let Err(e) = entry.owner.shutdown() {
                let rank = ParallelRankId::new(entry.boot.rank);
                failures.push(
                    CompletionContext {
                        rank: PipelineRank {
                            local: rank,
                            global: rank,
                        },
                        stage: "shutdown",
                        transaction: None,
                    }
                    .failure(CompletionClass::Unknown, true, Some(Box::new(e))),
                );
            }
        }
        if !failures.is_empty() {
            self.quarantine();
        }
        Error::failures("process decoder shutdown", failures)
    }
    fn quarantine(&mut self) {
        self.unavailable = true;
        // Drop hands unreaped children to the pre-reserved bounded reaper. It
        // never blocks waiting on GPU work, and never manufactures an ACK.
        self.ranks.clear();
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/parallel/process_completion.rs"]
mod completion_tests;

#[derive(Clone)]
pub struct SharedProcessPipelineTransport(
    std::rc::Rc<std::cell::RefCell<ProcessPipelineTransport>>,
);
impl SharedProcessPipelineTransport {
    pub fn process_stats(&self, rank: u32) -> Result<DecoderProcessStats> {
        self.0.borrow_mut().process_stats(rank)
    }
    pub fn is_quarantined(&self) -> bool {
        self.0.borrow().unavailable
    }
}
impl PipelineTransport for SharedProcessPipelineTransport {
    fn call_observed(
        &mut self,
        rank: PipelineRank,
        command: PipelineCommand,
        poll: &mut dyn FnMut(),
    ) -> Result<PipelineReply> {
        self.0.borrow_mut().call_observed(rank, command, poll)
    }
    fn outstanding(&self) -> usize {
        self.0.borrow().outstanding()
    }
    fn shutdown(&mut self) -> Result<()> {
        self.0.borrow_mut().shutdown()
    }
    fn quarantine(&mut self) {
        self.0.borrow_mut().quarantine();
    }
}
