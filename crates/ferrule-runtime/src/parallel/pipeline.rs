//! Device- and transport-neutral pipeline execution for whole decoder layers.
//!
//! This is a bounded, synchronous `forward` entry point, not a serving engine or
//! a test-owned runner. Each pipeline rank owns one persistent segment, sequence
//! state, and KV backend on its owner thread or process. Only
//! host-packed commands and host activations cross that boundary. The parent is
//! the sole logical `KvPageManager` owner and the sole `DistributedTransaction`
//! coordinator.
//!
//! Physical cohort custody uses the model's existing `PreparedKvCommit`. Its
//! borrowed owner API is bridged by a private wire-key adapter: the non-cloneable
//! physical transaction remains in the rank owner, while the parent only holds a
//! sealed key and the already validated packed projection. Install is dispatched
//! once; pending ACKs are polled and unknown custody quarantines the executor.
//! No logical page is reused before every owner retirement ACK and the parent's
//! `KvPageManager::confirm_page_retirement`.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionSequence, ExecutionTransactionId, ForwardMode,
    ForwardPhase, KvElementType, KvPageId, LogitsRequest, StateSlot,
};
use ferrule_common::{
    MeshCoordinate, ParallelExecutionScopes, ParallelRankId, ValidatedParallelTopology,
};
use ferrule_model::decoder::{
    CpuPagedKvBackend, CpuPagedKvPool, DecoderKvBackend, DecoderKvCapacity, DecoderKvCommitBackend,
    DecoderKvPrepare, DenseLogits, GenericDecoderOptions, GenericDecoderSequenceState,
    KvCommitBinding, KvCommitOwner, KvEndProgress, PackedDecoderBatch, PagedKvBackend,
    PreparedKvCommit, StandardGqaPlanes,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{
    Attention, BoundDecoderResources, HostRows, LayerSegmentPlan, SegmentInput, SegmentOutput,
    StandardDecoderSegment,
};

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaPipelineStageProgram;

pub mod owner;
pub mod stage;
#[cfg(feature = "cuda")]
pub mod tensor;
pub mod tensor_scope;
#[cfg(test)]
mod tensor_tests;
pub mod transport;
#[cfg(feature = "cuda")]
pub use tensor::StandardCudaTensorConfig;

use ferrule_model::{ModelFamily, WeightSource};
pub use owner::{PipelineOwnerStats, PipelineStageWorker};
pub use stage::{
    CpuPipelineStageProgram, PipelineExecutionContext, PipelineExpertContext, PipelineStageProgram,
};
pub use transport::{
    DataPoolPipelineTransport, PipelineCommand, PipelineCommandKey, PipelinePreparedProjection,
    PipelineReply, PipelineTransport,
};
use transport::{
    DataPoolPipelineTransport as ThreadTransport, PipelineCommand as Command,
    PipelineCommandKey as WireKey, PipelineReply as Reply,
};

use super::data::{
    DataParallelConfig, DataParallelExecutor, PanicQuiescence, ReplicaWorker, WorkRequest,
};
use super::expert::{ExpertGroup, ExpertParallelExecutor};
use crate::{
    DistributedTransaction, FinalizeOutcome, KvPageManager, KvReservationCommit, SessionId,
};

type Result<T> = ferrule_common::Result<T>;

fn error(message: impl std::fmt::Display) -> ferrule_common::Error {
    ferrule_common::Error::Execution {
        message: message.to_string(),
    }
}

fn coordinator<T>(result: std::result::Result<T, crate::DistributedTransactionError>) -> Result<T> {
    result.map_err(|source| error(format!("pipeline coordinator: {source:?}")))
}

/// Resource and queue limits shared by device programs and owner transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineConfig {
    pub page_size: usize,
    pub max_pages: usize,
    pub max_positions: usize,
    pub max_batch_tokens: usize,
    pub session_capacity: usize,
    pub max_parameter_bytes: u64,
    pub precision: ExecutionPrecisionPolicy,
    /// A zero value is rejected. CPU normally completes on the first poll; this
    /// remains bounded for adapters which expose pending install fences.
    pub max_ack_polls: usize,
}

impl PipelineConfig {
    pub fn validate(self) -> Result<()> {
        if [
            self.page_size,
            self.max_pages,
            self.max_positions,
            self.max_batch_tokens,
            self.session_capacity,
            self.max_ack_polls,
        ]
        .contains(&0)
            || self.max_parameter_bytes == 0
            || self.max_batch_tokens > self.max_positions
            || self.max_positions > u32::MAX as usize
            || self.session_capacity > u32::MAX as usize
            || self
                .max_pages
                .checked_mul(self.page_size)
                .is_none_or(|elements| elements > u32::MAX as usize)
        {
            return Err(error("invalid or overflowing pipeline capacity"));
        }
        if self.precision != ExecutionPrecisionPolicy::f32()
            && self.precision != ExecutionPrecisionPolicy::bf16_compatibility()
        {
            return Err(error(
                "pipeline supports only F32 or BF16 compatibility precision",
            ));
        }
        Ok(())
    }

    pub fn dtype(self) -> KvElementType {
        if self.precision == ExecutionPrecisionPolicy::f32() {
            KvElementType::F32
        } else {
            KvElementType::Bf16
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineRank {
    pub local: ParallelRankId,
    pub global: ParallelRankId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStageDescription {
    pub plan: LayerSegmentPlan,
    pub config: PipelineConfig,
    pub hidden: usize,
    pub vocabulary: usize,
    /// Physical heads per owner (global heads / TP for a tensor stage).
    pub kv_heads: usize,
    pub head_dim: usize,
    pub expert_group: Option<ExpertGroup>,
}

type Description = PipelineStageDescription;

/// One persistent complete-layer program and its owner-local associated KV backend.
/// `prepare_cpu` is intended to run inside the `DataParallelExecutor` factory.
pub struct PipelineStage<B = CpuPagedKvBackend, P = CpuPipelineStageProgram> {
    program: P,
    backend: B,
    description: PipelineStageDescription,
    capabilities: ExecutionCapabilities,
}

impl PipelineStage<CpuPagedKvBackend, CpuPipelineStageProgram> {
    /// Optional factory injection, constructed on the PP owner just like the
    /// segment. EP owners retain only their placement's prepared weights; this
    /// segment retains lazy bindings but never calls its local routed provider.
    /// Explicit expert IDs are not common topology/KV participant ranks.
    pub fn prepare_cpu_with_experts(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        config: PipelineConfig,
        mut experts: ExpertParallelExecutor,
    ) -> Result<Self> {
        let prepared = (|| {
            if config.precision != ExecutionPrecisionPolicy::f32() {
                return Err(error("pipeline expert injection requires F32"));
            }
            if experts.group().layers != plan.layers() {
                return Err(error("expert group must cover exactly the segment layers"));
            }
            experts.group().validate_resources(resources)?;
            Self::prepare_cpu(resources, plan, config)
        })();
        let mut stage = match prepared {
            Ok(stage) => stage,
            Err(source) => {
                return Err(ferrule_common::Error::with_cleanup(
                    "pipeline expert stage startup",
                    source,
                    experts.shutdown(),
                ));
            }
        };
        stage.description.expert_group = Some(experts.group().clone());
        stage.program = stage.program.with_experts(experts);
        Ok(stage)
    }

    pub fn prepare_cpu(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        config: PipelineConfig,
    ) -> Result<Self> {
        config.validate()?;
        let mut geometry = None;
        for layer in resources.spec().layers() {
            let Attention::Gqa(attention) = layer.attention() else {
                return Err(error("CPU pipeline currently requires standard GQA layers"));
            };
            let current = (attention.num_kv_heads(), attention.head_dim());
            if geometry.is_some_and(|previous| previous != current) {
                return Err(error("pipeline requires uniform GQA KV geometry"));
            }
            geometry = Some(current);
        }
        let (kv_heads, head_dim) = geometry.ok_or_else(|| error("decoder has no layers"))?;
        let planes = StandardGqaPlanes::new(
            plan.layer_count(),
            kv_heads,
            head_dim,
            config.page_size,
            config.max_positions,
            config.dtype(),
        )?;
        let capabilities = GenericDecoderOptions::standard_cpu(
            resources.spec(),
            ModelFamily::Unknown("pipeline".into()),
            WeightSource::Safetensors,
            config.page_size,
            config.max_positions,
            config.max_batch_tokens,
            1,
            config.max_parameter_bytes,
            config.precision,
        )?
        .capabilities();
        let description = Description {
            plan: plan.clone(),
            config,
            hidden: resources.spec().hidden_size(),
            vocabulary: resources.spec().vocab_size(),
            kv_heads,
            head_dim,
            expert_group: None,
        };
        description.validate()?;
        let segment = StandardDecoderSegment::prepare(
            resources,
            plan,
            config.precision,
            config.max_positions,
            config.max_parameter_bytes,
        )
        .map_err(stage::segment_error)?;
        let backend =
            PagedKvBackend::new(CpuPagedKvPool::from_strategy(&planes, config.max_pages)?);
        Self::new(
            CpuPipelineStageProgram::new(segment),
            backend,
            description,
            capabilities,
        )
    }
}

impl<B, P> PipelineStage<B, P> {
    /// Adapt or instrument the physical backend without moving model execution
    /// out of the owner or copying the forward implementation.
    pub fn map_backend<C>(self, adapt: impl FnOnce(B) -> C) -> PipelineStage<C, P> {
        PipelineStage {
            program: self.program,
            backend: adapt(self.backend),
            description: self.description,
            capabilities: self.capabilities,
        }
    }

    /// Decorate or erase a program inside its owner. Backend/view compatibility
    /// is checked by the worker constructor, not by moving device objects.
    pub fn map_program<Q>(self, adapt: impl FnOnce(P) -> Q) -> PipelineStage<B, Q> {
        PipelineStage {
            program: adapt(self.program),
            backend: self.backend,
            description: self.description,
            capabilities: self.capabilities,
        }
    }

    pub fn boot_description(&self) -> &PipelineStageDescription {
        &self.description
    }
}

impl PipelineStageDescription {
    /// Required serial standard-GQA host boundary. Device programs may advertise
    /// larger capabilities, but every stage must implement at least this subset.
    pub fn execution_capabilities(&self) -> Result<ExecutionCapabilities> {
        self.validate()?;
        Ok(ExecutionCapabilities {
            max_batch_tokens: self.config.max_batch_tokens,
            max_sequences: 1,
            max_prefill_query_tokens_per_sequence: self.config.max_batch_tokens,
            max_decode_query_tokens_per_sequence: 1,
            max_top_k: None,
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: false,
            full_logits_width: std::num::NonZeroU32::new(self.vocabulary as u32),
            kv_binding_mode: ferrule_common::execution::KvBindingMode::Paged,
            logits_row_policy: ferrule_common::execution::LogitsRowPolicy::Any,
        })
    }

    /// Validate a decoded host payload before physical enter or device upload.
    pub fn validate_input(&self, input: &SegmentInput, row_count: usize) -> Result<()> {
        if row_count == 0 || row_count > self.config.max_batch_tokens {
            return Err(error("invalid pipeline input row count"));
        }
        match input {
            SegmentInput::Tokens if self.plan.owns_embedding() => Ok(()),
            SegmentInput::Hidden { next_layer, rows }
                if !self.plan.owns_embedding() && *next_layer == self.plan.layers().start =>
            {
                self.validate_rows(rows, row_count)
            }
            _ => Err(error("invalid pipeline segment input endpoint")),
        }
    }

    /// Validate host-staged output independently of the device implementation.
    pub fn validate_output(&self, output: &SegmentOutput, row_count: usize) -> Result<()> {
        if row_count == 0 || row_count > self.config.max_batch_tokens {
            return Err(error("invalid pipeline output row count"));
        }
        match output {
            SegmentOutput::Hidden { next_layer, rows }
                if !self.plan.owns_output() && *next_layer == self.plan.layers().end =>
            {
                self.validate_rows(rows, row_count)
            }
            SegmentOutput::Logits(logits)
                if self.plan.owns_output()
                    && logits.rows() == row_count
                    && logits.width() == self.vocabulary =>
            {
                Ok(())
            }
            _ => Err(error(
                "invalid pipeline segment output endpoint or logits shape",
            )),
        }
    }

    fn validate_rows(&self, rows: &HostRows, row_count: usize) -> Result<()> {
        let dtype = if self.config.dtype() == KvElementType::F32 {
            ferrule_model::transformer::RowsDType::F32
        } else {
            ferrule_model::transformer::RowsDType::Bf16
        };
        if rows.shape().rows() != row_count
            || rows.shape().width() != self.hidden
            || rows.dtype() != dtype
        {
            return Err(error("invalid pipeline hidden shape or precision"));
        }
        Ok(())
    }

    /// Validate host geometry before admitting a stage or allocating wire payloads.
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if [self.hidden, self.vocabulary, self.kv_heads, self.head_dim].contains(&0)
            || self.vocabulary > u32::MAX as usize
        {
            return Err(error("invalid pipeline decoder geometry"));
        }
        for width in [self.hidden, self.vocabulary] {
            width
                .checked_mul(self.config.max_batch_tokens)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| error("pipeline host payload bound overflows usize"))?;
        }
        StandardGqaPlanes::new(
            self.plan.layer_count(),
            self.kv_heads,
            self.head_dim,
            self.config.page_size,
            self.config.max_positions,
            self.config.dtype(),
        )?;
        if let Some(group) = &self.expert_group {
            if group.layers != self.plan.layers()
                || group.members.is_empty()
                || !group.members.contains(&group.source_rank)
                || group.members.iter().collect::<BTreeSet<_>>().len() != group.members.len()
                || group.limits.max_tokens == 0
                || group.limits.max_bytes == 0
            {
                return Err(error("invalid pipeline expert group"));
            }
        }
        Ok(())
    }
}

/// Owned startup data. `program_spec` is opaque to runtime: a factory/IPC codec
/// chooses its version and size bound. It must describe resources, never contain
/// parent-initialized CUDA objects, device pointers or private KV tokens.
#[derive(Debug, Clone)]
pub struct PipelineStageBoot {
    pub rank: PipelineRank,
    pub plan: LayerSegmentPlan,
    pub config: PipelineConfig,
    pub program_spec: Vec<u8>,
}

impl PipelineStageBoot {
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if self.rank.local != self.rank.global {
            return Err(error(
                "serial pipeline requires matching local/global ranks",
            ));
        }
        Ok(())
    }
}

impl<B, P> PipelineStage<B, P>
where
    B: DecoderKvCommitBackend<SequenceState = GenericDecoderSequenceState>,
    P: PipelineStageProgram<KvView = B::KvView>,
{
    /// Assemble an owner-local device program and its associated KV backend.
    /// On rejection BOTH program and backend shutdown are attempted.
    pub fn new(
        program: P,
        backend: B,
        description: PipelineStageDescription,
        capabilities: ExecutionCapabilities,
    ) -> Result<Self> {
        let mut stage = Self {
            program,
            backend,
            description,
            capabilities,
        };
        if let Err(source) = stage.validate() {
            return Err(ferrule_common::Error::with_cleanup(
                "pipeline stage construction",
                source,
                stage.shutdown(),
            ));
        }
        Ok(stage)
    }

    fn validate(&self) -> Result<()> {
        self.description.validate()?;
        let config = self.description.config;
        let capabilities = self.capabilities;
        if self.program.plan() != &self.description.plan
            || capabilities.max_batch_tokens < config.max_batch_tokens
            || capabilities.max_sequences == 0
            || capabilities.max_prefill_query_tokens_per_sequence < config.max_batch_tokens
            || capabilities.max_decode_query_tokens_per_sequence == 0
            || !capabilities.supports_prefill
            || !capabilities.supports_decode
            || capabilities
                .full_logits_width
                .map(|width| width.get() as usize)
                != Some(self.description.vocabulary)
            || capabilities.kv_binding_mode != ferrule_common::execution::KvBindingMode::Paged
            || capabilities.logits_row_policy != ferrule_common::execution::LogitsRowPolicy::Any
        {
            return Err(error("pipeline stage plan/capabilities mismatch"));
        }
        let capacity = self.backend.capacity();
        if capacity.physical_pages != config.max_pages
            || capacity.free_pages != config.max_pages
            || capacity.active_transactions != 0
            || capacity.resident_pages != 0
            || capacity.preempted_pages != 0
        {
            return Err(error(
                "pipeline stage needs an empty backend with the configured capacity",
            ));
        }
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        let program = self.program.shutdown();
        let backend = self.backend.shutdown();
        ferrule_common::Error::failures(
            "pipeline stage shutdown",
            program.err().into_iter().chain(backend.err()).collect(),
        )
    }
}

/// Type erasure occurs INSIDE the factory; all variants retain the same owner
/// journal. Construct with `PipelineStageWorker::boxed`, never from a second
/// transaction runner. The boxed value itself is not Send.
pub struct BoxedPipelineStageWorker(Box<dyn StageWorkerDispatch>);

trait StageWorkerDispatch:
    ReplicaWorker<Command, Output = Reply, Error = ferrule_common::Error>
{
    fn dispatch(&mut self, command: Command) -> Result<Reply>;
    fn boot_description(&self) -> &PipelineStageDescription;
    fn has_active_custody(&self) -> bool;
}

impl BoxedPipelineStageWorker {
    pub fn dispatch(&mut self, command: PipelineCommand) -> Result<PipelineReply> {
        self.0.dispatch(command)
    }
    pub fn boot_description(&self) -> &PipelineStageDescription {
        self.0.boot_description()
    }
    pub fn has_active_custody(&self) -> bool {
        self.0.has_active_custody()
    }
}

impl ReplicaWorker<Command> for BoxedPipelineStageWorker {
    type Output = Reply;
    type Error = ferrule_common::Error;
    fn execute(&mut self, request: WorkRequest<Command>) -> Result<Reply> {
        self.0.execute(request)
    }
    fn panic_quiescence(&mut self) -> PanicQuiescence {
        self.0.panic_quiescence()
    }
    fn shutdown(&mut self) -> Result<()> {
        self.0.shutdown()
    }
}

fn validate_pipeline(
    topology: &ValidatedParallelTopology,
    plans: &[LayerSegmentPlan],
    config: PipelineConfig,
) -> Result<()> {
    validate_pipeline_layout(topology, plans, config, false)
}

fn validate_pipeline_layout(
    topology: &ValidatedParallelTopology,
    plans: &[LayerSegmentPlan],
    config: PipelineConfig,
    thread_tensor: bool,
) -> Result<()> {
    config.validate()?;
    if topology.plan().data_parallel != 1 {
        return Err(error("pipeline requires DP=1"));
    }
    if !thread_tensor && topology.plan().tensor_parallel != 1 {
        return Err(error(
            "process/transport TP unsupported; use the thread standard tensor constructor",
        ));
    }
    if topology.plan().sequence_parallel != 1 || topology.plan().context_parallel != 1 {
        return Err(error("pipeline requires SP=CP=1"));
    }
    if thread_tensor
        && (!matches!(topology.plan().tensor_parallel, 1 | 2 | 4)
            || config.precision != ExecutionPrecisionPolicy::f32())
    {
        return Err(error("thread tensor pipeline requires TP1/2/4 and F32"));
    }
    if topology.plan().expert_parallel != 1 && topology.plan().tensor_parallel != 1 {
        return Err(error(
            "EP×TP unsupported; expert dispatch has no TP owner mapping",
        ));
    }
    if topology.plan().expert_parallel != 1 {
        return Err(error(
            "common EP topology is metadata-only; configure explicit stage ExpertGroups instead",
        ));
    }
    if topology.plan().pipeline_parallel != plans.len() {
        return Err(error("segment count differs from topology PP degree"));
    }
    LayerSegmentPlan::validate_pipeline(plans).map_err(stage::segment_error)
}

fn owner_failure(
    failure: super::data::OwnerFailure<ferrule_common::Error>,
) -> ferrule_common::Error {
    use super::data::OwnerFailureKind;
    let source = match failure.kind {
        OwnerFailureKind::Initialization(source) | OwnerFailureKind::Shutdown(source) => source,
        other => error(format!("pipeline owner failure: {other:?}")),
    };
    ferrule_common::Error::context(format!("pipeline owner {:?}", failure.rank), source)
}

fn pool_build_error(
    failure: super::data::BuildError<ferrule_common::Error>,
) -> ferrule_common::Error {
    match failure {
        super::data::BuildError::Config(source) => {
            error(format!("pipeline pool config: {source:?}"))
        }
        super::data::BuildError::Owners(failures) => errors(
            "pipeline construction",
            failures.into_iter().map(owner_failure).collect(),
        ),
    }
}

fn pool_shutdown_error(
    failure: super::data::ShutdownError<ferrule_common::Error>,
) -> ferrule_common::Error {
    errors(
        "pipeline shutdown",
        failure.failures.into_iter().map(owner_failure).collect(),
    )
}

fn errors(operation: &str, mut failures: Vec<ferrule_common::Error>) -> ferrule_common::Error {
    if failures.len() == 1 {
        ferrule_common::Error::context(operation, failures.remove(0))
    } else {
        ferrule_common::Error::FailureBatch {
            operation: operation.into(),
            failures,
        }
    }
}

fn ack(reply: Reply) -> Result<KvEndProgress> {
    match reply {
        Reply::Ack(progress) if progress != KvEndProgress::ConsumedRejected => Ok(progress),
        Reply::Ack(KvEndProgress::ConsumedRejected) => {
            Err(error("consumed rejection is not an owner ACK"))
        }
        _ => Err(error("unexpected pipeline owner reply")),
    }
}

fn complete(reply: Reply) -> Result<()> {
    if ack(reply)? != KvEndProgress::Complete {
        return Err(error("owner receipt is pending"));
    }
    Ok(())
}

// Only RemoteToken, not its addressing key, is the non-cloneable cohort input.
struct RemoteToken {
    key: WireKey,
}

#[derive(Default)]
struct Receipts {
    published: Cell<bool>,
    finished: Cell<bool>,
}

struct RemotePhysicalOwner {
    transport: Rc<RefCell<Box<dyn PipelineTransport>>>,
    batch: PackedDecoderBatch,
    receipts: Rc<Receipts>,
}

impl RemotePhysicalOwner {
    fn call(&self, rank: PipelineRank, command: Command) -> Result<Reply> {
        self.transport.borrow_mut().call(rank, command)
    }
}

impl DecoderKvBackend for RemotePhysicalOwner {
    type SequenceState = ();
    type Transaction = RemoteToken;
    type KvView = ();

    fn configure_capacity(&mut self, _: usize) -> Result<()> {
        Err(error(
            "remote pipeline backend cannot be configured by model code",
        ))
    }
    fn prepare(&mut self, _: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        Err(error("remote pipeline backend does not duplicate prepare"))
    }
    fn enter(
        &mut self,
        _: &mut Self::Transaction,
        _: &PackedDecoderBatch,
        _: &mut [Self::SequenceState],
    ) -> Result<()> {
        Err(error("remote pipeline backend does not duplicate enter"))
    }
    fn active_view(&mut self, _: &mut Self::Transaction) -> Result<Self::KvView> {
        Err(error("remote pipeline backend has no activation view"))
    }
    fn leave(&mut self, _: &mut Self::Transaction) -> Result<()> {
        Err(error("remote pipeline backend does not duplicate leave"))
    }
    fn commit(&mut self, _: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        Err(error("remote pipeline backend uses PreparedKvCommit"))
    }
    fn rollback(&mut self, _: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        Err(error("remote pipeline backend uses PreparedKvCommit"))
    }
    fn release(&mut self, _: &[KvPageId]) -> Result<()> {
        Err(error("remote pipeline backend release is owner-local"))
    }
    fn preempt(&mut self, _: &[KvPageId]) -> Result<()> {
        Err(error("remote pipeline backend preempt is owner-local"))
    }
    fn restore(&mut self, _: &[KvPageId]) -> Result<()> {
        Err(error("remote pipeline backend restore is owner-local"))
    }
    fn capacity(&self) -> DecoderKvCapacity {
        panic!("remote capacity must be queried from its physical owner")
    }
    fn page_status(&self, _: KvPageId) -> ferrule_model::decoder::DecoderKvPageStatus {
        panic!("remote status must be queried from its physical owner")
    }
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

impl DecoderKvCommitBackend for RemotePhysicalOwner {
    fn preflight_commit_ready(
        &self,
        token: &RemoteToken,
        binding: &KvCommitBinding,
        rank: ParallelRankId,
        _: &[()],
    ) -> Result<()> {
        if &token.key.binding != binding || token.key.rank.global != rank {
            return Err(error("remote KV binding/rank mismatch"));
        }
        complete(self.call(token.key.rank, Command::Ready(token.key.clone()))?)
    }

    fn commit_batch(&self, _: &RemoteToken) -> Result<&PackedDecoderBatch> {
        Ok(&self.batch)
    }

    fn install_commit(
        &mut self,
        token: &mut RemoteToken,
        generation: u64,
    ) -> Result<KvEndProgress> {
        ack(self.call(
            token.key.rank,
            Command::Install {
                key: token.key.clone(),
                generation,
            },
        )?)
    }

    fn poll_install_ack(
        &mut self,
        token: &mut RemoteToken,
        generation: u64,
    ) -> Result<KvEndProgress> {
        ack(self.call(
            token.key.rank,
            Command::PollInstall {
                key: token.key.clone(),
                generation,
            },
        )?)
    }

    fn abort_prepared(
        &mut self,
        token: &mut RemoteToken,
        generation: u64,
    ) -> Result<KvEndProgress> {
        ack(self.call(
            token.key.rank,
            Command::Abort {
                key: token.key.clone(),
                generation,
            },
        )?)
    }

    fn publish_committed(&mut self, _: &RemoteToken) {
        assert!(
            self.receipts.published.get(),
            "publication needs a real owner receipt"
        );
    }

    fn preflight_retirement(&self, token: &RemoteToken, pages: &[KvPageId]) -> Result<()> {
        complete(self.call(
            token.key.rank,
            Command::CheckRetirement {
                key: token.key.clone(),
                pages: pages.to_vec(),
            },
        )?)
    }

    fn retire_prepared(
        &mut self,
        token: &mut RemoteToken,
        generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress> {
        ack(self.call(
            token.key.rank,
            Command::Retire {
                key: token.key.clone(),
                generation,
                pages: pages.to_vec(),
            },
        )?)
    }

    fn finish_prepared(&mut self, _: RemoteToken) {
        assert!(
            self.receipts.finished.get(),
            "finish needs a real owner receipt"
        );
    }
}

#[derive(Debug)]
pub struct PipelineOutput {
    pub transaction: ExecutionTransactionId,
    pub logits: DenseLogits,
}

/// Read-only progress, including the publication gate. No provisional logits are
/// exposed. Observers may request cancellation but do not drive KV or commands.
#[derive(Debug, Clone)]
pub struct PipelineProgress {
    pub transaction: ExecutionTransactionId,
    pub state: crate::TransactionState,
    pub pending_ranks: Vec<ParallelRankId>,
    pub committed_tokens: usize,
    pub allocated_pages: usize,
    pub publications: usize,
}

/// One serial pipeline, independently selecting owner transport and device program.
/// Generic constructors retain DP=TP=1. `new_standard_cuda_tensor` opts into
/// sealed concurrent thread PP×TP, with one KV manager and one coordinator.
/// Process TP and EP×TP are not supported.
pub struct PipelineParallelExecutor {
    transport: Rc<RefCell<Box<dyn PipelineTransport>>>,
    coordinator: DistributedTransaction,
    scopes: ParallelExecutionScopes,
    page_manager: KvPageManager,
    config: PipelineConfig,
    descriptions: Vec<Description>,
    tensor_controls: Vec<super::tensor::decoder_collective::DecoderTensorCollectiveControl>,
    sessions: BTreeMap<SessionId, StateSlot>,
    next_transaction: u64,
    // Unresolved custody only. is_quarantined also includes terminal TP failure.
    quarantined: bool,
    closed: bool,
    unresolved_retirement: Option<crate::cache::KvRetirement>,
}

impl PipelineParallelExecutor {
    /// Backwards-compatible CPU constructor, including explicit `<F, B>` callers.
    /// Only the factory is Send; resources and backend are created on the owner.
    pub fn new<F, B>(
        topology: ValidatedParallelTopology,
        plans: Vec<LayerSegmentPlan>,
        config: PipelineConfig,
        factory: F,
    ) -> Result<Self>
    where
        F: FnOnce(PipelineRank, LayerSegmentPlan) -> Result<PipelineStage<B>>
            + Clone
            + Send
            + 'static,
        B: DecoderKvCommitBackend<
                SequenceState = GenericDecoderSequenceState,
                KvView = ferrule_model::decoder::CpuKvView,
            > + 'static,
    {
        Self::new_with_program(topology, plans, config, factory)
    }

    /// Typed CPU/GPU program factory, invoked only inside each persistent owner.
    /// `B`, `P`, their views, and all device objects need NOT implement Send.
    pub fn new_with_program<F, B, P>(
        topology: ValidatedParallelTopology,
        plans: Vec<LayerSegmentPlan>,
        config: PipelineConfig,
        factory: F,
    ) -> Result<Self>
    where
        F: FnOnce(PipelineRank, LayerSegmentPlan) -> Result<PipelineStage<B, P>>
            + Clone
            + Send
            + 'static,
        B: DecoderKvCommitBackend<SequenceState = GenericDecoderSequenceState> + 'static,
        P: PipelineStageProgram<KvView = B::KvView>,
    {
        let boots = plans
            .into_iter()
            .enumerate()
            .map(|(index, plan)| PipelineStageBoot {
                rank: Self::rank(index),
                plan,
                config,
                program_spec: Vec::new(),
            })
            .collect();
        Self::new_with_factory(topology, boots, move |boot| {
            let stage = factory(boot.rank, boot.plan.clone())?;
            Ok(PipelineStageWorker::new(&boot, stage)?.boxed())
        })
    }

    /// Dynamic owner-local factory. A parent captures only owned plans/spec bytes
    /// or checkpoint paths; select CPU/CUDA and initialize the device in `factory`.
    /// The same boot can be encoded by a process launcher without parent cuInit.
    /// A failing factory must clean up resources it has not returned to the worker.
    pub fn new_with_factory<F>(
        topology: ValidatedParallelTopology,
        boots: Vec<PipelineStageBoot>,
        factory: F,
    ) -> Result<Self>
    where
        F: FnOnce(PipelineStageBoot) -> Result<BoxedPipelineStageWorker> + Clone + Send + 'static,
    {
        let first = boots
            .first()
            .ok_or_else(|| error("pipeline has no stage boot"))?;
        let config = first.config;
        let plans = boots
            .iter()
            .map(|boot| boot.plan.clone())
            .collect::<Vec<_>>();
        validate_pipeline(&topology, &plans, config)?;
        for (index, boot) in boots.iter().enumerate() {
            boot.validate()?;
            if boot.rank != Self::rank(index) || boot.config != config {
                return Err(error("stage boot rank or config mismatch"));
            }
        }
        let degree = boots.len();
        let boots = Arc::new(boots);
        let pool = DataParallelExecutor::new(
            DataParallelConfig {
                replicas: degree,
                max_outstanding_per_replica: 1,
                session_capacity: degree,
            },
            move |local| factory(boots[local.get() as usize].clone()),
        )
        .map_err(pool_build_error)?;
        Self::new_with_transport(topology, plans, config, ThreadTransport::new(pool))
    }

    /// Generic thread owner factory. TP>1 requires `new_standard_cuda_tensor`
    /// so arbitrary programs cannot bypass sealed collective validation.
    ///
    /// `plans` has one entry per PP stage. With TP=1 the factory runs once per
    /// stage and receives its mesh coordinate; all resources remain owner-local.
    pub fn new_thread_tensor_with_program<F, B, P>(
        topology: ValidatedParallelTopology,
        plans: Vec<LayerSegmentPlan>,
        config: PipelineConfig,
        factory: F,
    ) -> Result<Self>
    where
        F: FnOnce(PipelineRank, MeshCoordinate, LayerSegmentPlan) -> Result<PipelineStage<B, P>>
            + Clone
            + Send
            + 'static,
        B: DecoderKvCommitBackend<SequenceState = GenericDecoderSequenceState> + 'static,
        P: PipelineStageProgram<KvView = B::KvView>,
    {
        validate_pipeline_layout(&topology, &plans, config, true)?;
        if topology.plan().tensor_parallel > 1 {
            return Err(error(
                "generic tensor program requires TP=1; use new_standard_cuda_tensor for sealed TP",
            ));
        }
        Self::new_thread_tensor_with_program_inner(topology, plans, config, factory)
    }

    // Only the standard constructor and internal fault-injection tests may admit
    // TP programs; they install both scoped programs and their cohort controls.
    fn new_thread_tensor_with_program_inner<F, B, P>(
        topology: ValidatedParallelTopology,
        plans: Vec<LayerSegmentPlan>,
        config: PipelineConfig,
        factory: F,
    ) -> Result<Self>
    where
        F: FnOnce(PipelineRank, MeshCoordinate, LayerSegmentPlan) -> Result<PipelineStage<B, P>>
            + Clone
            + Send
            + 'static,
        B: DecoderKvCommitBackend<SequenceState = GenericDecoderSequenceState> + 'static,
        P: PipelineStageProgram<KvView = B::KvView>,
    {
        validate_pipeline_layout(&topology, &plans, config, true)?;
        let owner_topology = topology.clone();
        let owner_plans = Arc::new(plans.clone());
        let degree = topology.world_size() as usize;
        let pool = DataParallelExecutor::new(
            DataParallelConfig {
                replicas: degree,
                max_outstanding_per_replica: 1,
                session_capacity: degree,
            },
            move |local| {
                let coordinate = owner_topology
                    .coordinate_of(local)
                    .map_err(|e| error(format!("tensor owner coordinate: {e:?}")))?;
                let boot = PipelineStageBoot {
                    rank: Self::rank(local.get() as usize),
                    plan: owner_plans[coordinate.stage() as usize].clone(),
                    config,
                    program_spec: Vec::new(),
                };
                let stage = factory(boot.rank, coordinate, boot.plan.clone())?;
                Ok(PipelineStageWorker::new(&boot, stage)?.boxed())
            },
        )
        .map_err(pool_build_error)?;
        Self::new_with_transport_inner(topology, plans, config, ThreadTransport::new(pool), true)
    }

    /// Inject thread/process transport without a second owner or KV state machine.
    /// This executor alone constructs/owns the logical page manager. Serving should
    /// own this pipeline directly, not wrap it in another KV-owning driver.
    /// ALL startup failures attempt transport shutdown, including invalid topology
    /// and descriptions. Transport shutdown must attempt every initialized owner.
    pub fn new_with_transport<T: PipelineTransport + 'static>(
        topology: ValidatedParallelTopology,
        plans: Vec<LayerSegmentPlan>,
        config: PipelineConfig,
        transport: T,
    ) -> Result<Self> {
        Self::new_with_transport_inner(topology, plans, config, transport, false)
    }

    fn new_with_transport_inner<T: PipelineTransport + 'static>(
        topology: ValidatedParallelTopology,
        plans: Vec<LayerSegmentPlan>,
        config: PipelineConfig,
        mut transport: T,
        thread_tensor: bool,
    ) -> Result<Self> {
        let startup = (|| {
            validate_pipeline_layout(&topology, &plans, config, thread_tensor)?;
            let degree = topology.world_size() as usize;
            let tp = topology.plan().tensor_parallel;
            let mut descriptions: Vec<Description> = Vec::with_capacity(degree);
            let mut scopes = topology
                .execution_scopes(0)
                .map_err(|e| error(format!("pipeline execution scope: {e:?}")))?;
            for index in 0..degree {
                let plan = &plans[index / tp];
                let Reply::Description(description) =
                    transport.call(Self::rank(index), Command::Describe)?
                else {
                    return Err(error("missing owner description"));
                };
                description.validate()?;
                if description.config != config || &description.plan != plan {
                    return Err(error("owner factory returned a different plan or capacity"));
                }
                if let Some(first) = descriptions.first()
                    && (
                        description.hidden,
                        description.vocabulary,
                        description.kv_heads,
                        description.head_dim,
                    ) != (
                        first.hidden,
                        first.vocabulary,
                        first.kv_heads,
                        first.head_dim,
                    )
                {
                    return Err(error("pipeline owners disagree on decoder geometry"));
                }
                if let Some(group) = &description.expert_group {
                    if tp > 1 {
                        return Err(error("EP×TP is unsupported; no local expert fallback"));
                    }
                    scopes
                        .attach_expert_dispatch_members(group.dispatch_members(
                            &topology,
                            0,
                            index as u32,
                        )?)
                        .map_err(|e| error(format!("pipeline expert scope: {e:?}")))?;
                }
                descriptions.push(description);
            }
            let first = &descriptions[0];
            let schema = StandardGqaPlanes::new(
                plans[0].total_layers(),
                first
                    .kv_heads
                    .checked_mul(tp)
                    .ok_or_else(|| error("global KV head count overflow"))?,
                first.head_dim,
                config.page_size,
                config.max_positions,
                config.dtype(),
            )?;
            Ok((descriptions, schema, scopes))
        })();
        let (descriptions, schema, scopes) = match startup {
            Ok(startup) => startup,
            Err(source) => {
                return Err(ferrule_common::Error::with_cleanup(
                    "pipeline startup",
                    source,
                    transport.shutdown(),
                ));
            }
        };
        let degree = topology.world_size() as usize;
        Ok(Self {
            transport: Rc::new(RefCell::new(Box::new(transport))),
            coordinator: DistributedTransaction::new_with_limits(
                topology,
                degree,
                crate::distributed::DistributedTransactionLimits {
                    max_transactions: 1,
                    max_operations: degree,
                },
            ),
            scopes,
            page_manager: KvPageManager::new(Box::new(schema), config.max_pages),
            config,
            descriptions,
            tensor_controls: Vec::new(),
            sessions: BTreeMap::new(),
            next_transaction: 1,
            quarantined: false,
            closed: false,
            unresolved_retirement: None,
        })
    }

    fn rank(index: usize) -> PipelineRank {
        let rank = ParallelRankId::new(index as u32);
        PipelineRank {
            local: rank,
            global: rank,
        }
    }
    fn command(&self, rank: PipelineRank, command: Command) -> Result<Reply> {
        self.transport.borrow_mut().call(rank, command)
    }
    fn available(&self) -> Result<()> {
        self.custody_available()?;
        if self.is_quarantined() {
            return Err(error(
                "tensor collective lifetime failed; pipeline terminally unavailable",
            ));
        }
        Ok(())
    }
    // A poisoned collective forbids new work, but is not evidence of unknown
    // device custody. Proven rollback still permits session cleanup and shutdown.
    fn custody_available(&self) -> Result<()> {
        if self.closed || self.quarantined {
            return Err(error("pipeline closed or quarantined; custody retained"));
        }
        Ok(())
    }
    pub fn config(&self) -> PipelineConfig {
        self.config
    }
    pub fn execution_scopes(&self) -> &ParallelExecutionScopes {
        &self.scopes
    }
    pub fn stage_descriptions(&self) -> &[PipelineStageDescription] {
        &self.descriptions
    }
    pub fn create_session(&mut self, session: SessionId) -> Result<StateSlot> {
        self.available()?;
        self.ensure_session(session)
    }
    pub fn coordinator(&self) -> &DistributedTransaction {
        &self.coordinator
    }
    pub fn page_manager(&self) -> &KvPageManager {
        &self.page_manager
    }
    /// Unknown custody or a permanently failed TP lifetime blocks new work.
    /// Successful rollback/shutdown does not reset collective failure; there is
    /// no implicit epoch replacement or replay.
    pub fn is_quarantined(&self) -> bool {
        self.quarantined
            || self
                .tensor_controls
                .iter()
                .any(|control| control.is_failed())
    }
    pub fn outstanding(&self) -> usize {
        self.transport.borrow().outstanding()
    }
    pub fn session_slot(&self, session: SessionId) -> Option<StateSlot> {
        self.sessions.get(&session).copied()
    }
    pub fn owner_stats(&self) -> Result<Vec<PipelineOwnerStats>> {
        (0..self.descriptions.len())
            .map(
                |index| match self.command(Self::rank(index), Command::Stats)? {
                    Reply::Stats(stats) => Ok(stats),
                    _ => Err(error("missing owner statistics")),
                },
            )
            .collect()
    }

    fn new_slot(&self, session: SessionId) -> Result<StateSlot> {
        if self.sessions.contains_key(&session)
            || self.sessions.len() >= self.config.session_capacity
        {
            return Err(error("pipeline session capacity or identity conflict"));
        }
        (0..self.config.session_capacity)
            .map(|index| StateSlot::new(index as u32))
            .find(|slot| !self.sessions.values().any(|used| used == slot))
            .ok_or_else(|| error("pipeline slot capacity exhausted"))
    }

    fn create_owners(
        &self,
        session: SessionId,
        source: Option<SessionId>,
        generation: u64,
    ) -> Result<()> {
        for index in 0..self.descriptions.len() {
            match self.command(
                Self::rank(index),
                match source {
                    Some(source) => Command::Fork {
                        source,
                        target: session,
                    },
                    None => Command::Create { session },
                },
            )? {
                Reply::Generation(actual) if actual == generation => {}
                _ => return Err(error("owner sequence generation mismatch")),
            }
        }
        Ok(())
    }

    fn ensure_session(&mut self, session: SessionId) -> Result<StateSlot> {
        if let Some(&slot) = self.sessions.get(&session) {
            return Ok(slot);
        }
        let slot = self.new_slot(session)?;
        self.quarantined = true;
        self.create_owners(session, None, 0)?;
        self.page_manager.alloc_sequence(slot, 0).map_err(error)?;
        self.sessions.insert(session, slot);
        self.quarantined = false;
        Ok(slot)
    }

    /// Exact-prefix fork: no physical page copy. The next partial-tail append is
    /// COW, allocated once by the parent and projected to every local KV plane.
    pub fn fork_session(&mut self, source: SessionId, target: SessionId) -> Result<()> {
        self.available()?;
        let source_slot = *self
            .sessions
            .get(&source)
            .ok_or_else(|| error("unknown fork source"))?;
        let target_slot = self.new_slot(target)?;
        let generation = self
            .page_manager
            .sequence_generation(source_slot)
            .map_err(error)?
            .checked_add(1)
            .ok_or_else(|| error("sequence generation exhausted"))?;
        let position = self
            .page_manager
            .block_table(source_slot)
            .expect("live source")
            .committed_tokens();
        let prepared = self
            .page_manager
            .prepare_fork_sequence_exact(source_slot, target_slot, generation, position)
            .map_err(error)?;
        self.quarantined = true;
        self.create_owners(target, Some(source), generation)?;
        self.page_manager
            .publish_fork_sequence_exact(prepared)
            .map_err(error)?;
        self.sessions.insert(target, target_slot);
        self.quarantined = false;
        Ok(())
    }

    pub fn release_session(&mut self, session: SessionId) -> Result<()> {
        self.custody_available()?;
        let slot = *self
            .sessions
            .get(&session)
            .ok_or_else(|| error("unknown released session"))?;
        self.quarantined = true;
        let retirement = self.page_manager.free_sequence_pages(slot).map_err(error)?;
        let pages = retirement.pages().to_vec();
        self.unresolved_retirement = Some(retirement);
        for index in 0..self.descriptions.len() {
            complete(self.command(
                Self::rank(index),
                Command::Release {
                    session,
                    pages: pages.clone(),
                },
            )?)?;
        }
        let retirement = self
            .unresolved_retirement
            .take()
            .expect("retained retirement");
        self.confirm_retirement(retirement)?;
        self.sessions.remove(&session);
        self.quarantined = false;
        Ok(())
    }

    fn confirm_retirement(&mut self, retirement: crate::cache::KvRetirement) -> Result<()> {
        if let Err(failure) = self.page_manager.confirm_page_retirement(retirement) {
            let (source, retirement) = failure.into_parts();
            self.unresolved_retirement = Some(retirement);
            return Err(error(source));
        }
        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.custody_available()?;
        let sessions = self.sessions.keys().copied().collect::<Vec<_>>();
        for session in sessions {
            self.release_session(session)?;
        }
        self.closed = true;
        self.transport.borrow_mut().shutdown()
    }
}

// Ownership variants only, not a second transaction/decision registry.
enum LogicalRollback {
    Reservations(Vec<crate::cache::KvReservation>),
    Prepared(crate::cache::PreparedKvCommit),
}

impl PipelineParallelExecutor {
    fn report(
        &self,
        id: ExecutionTransactionId,
        observe: &mut impl FnMut(PipelineProgress),
    ) -> Result<()> {
        let stats = self.page_manager.stats();
        observe(PipelineProgress {
            transaction: id,
            state: coordinator(self.coordinator.state(id))?,
            pending_ranks: coordinator(self.coordinator.pending_ranks(id))?,
            committed_tokens: stats.committed_tokens,
            allocated_pages: stats.allocated_pages,
            publications: self.coordinator.publication_count(),
        });
        Ok(())
    }

    fn abort_before_ready(
        &mut self,
        id: ExecutionTransactionId,
        keys: &[WireKey],
        logical: Option<LogicalRollback>,
    ) -> Result<()> {
        // Even an unexpected cleanup error must not drop/consume logical custody.
        let mut logical = logical.map(std::mem::ManuallyDrop::new);
        coordinator(self.coordinator.abort_decision(id))?;
        let mut failure = None;
        for key in keys {
            let result = poll_end(self.config.max_ack_polls, || {
                ack(self.command(key.rank, Command::Rollback(key.clone()))?)
            });
            match result {
                Ok(()) => coordinator(self.coordinator.finalize(
                    id,
                    key.rank.global,
                    FinalizeOutcome::Success,
                ))?,
                Err(source) => {
                    failure.get_or_insert(source);
                }
            }
        }

        if let Some(source) = failure {
            return Err(source);
        }

        if let Some(logical) = logical.take().map(std::mem::ManuallyDrop::into_inner) {
            let retirement = match logical {
                LogicalRollback::Reservations(reservations) => {
                    match self.page_manager.abort_reservations(reservations) {
                        Ok(retirement) => retirement,
                        Err(failure) => {
                            let (source, reservations) = failure.into_parts();
                            std::mem::forget(reservations);
                            return Err(error(source));
                        }
                    }
                }
                LogicalRollback::Prepared(prepared) => {
                    self.page_manager.abort_prepared_commit(prepared)
                }
            };
            // Rollback ACKs above prove all newly allocated/COW destinations are
            // physically gone. Shared sources were never released.
            self.confirm_retirement(retirement)?;
        }
        coordinator(self.coordinator.cancel(id))?;
        coordinator(self.coordinator.retire(id))?;
        self.quarantined = false;
        Ok(())
    }

    fn cleanup_error(
        &mut self,
        id: ExecutionTransactionId,
        keys: &[WireKey],
        logical: Option<LogicalRollback>,
        source: ferrule_common::Error,
    ) -> ferrule_common::Error {
        match self.abort_before_ready(id, keys, logical) {
            Ok(()) => source,
            Err(cleanup) => ferrule_common::Error::with_cleanup(
                "cleanup unresolved, pipeline quarantined",
                source,
                Err(cleanup),
            ),
        }
    }

    /// One complete call. Only committed full logits are returned. Prefill chunks
    /// may extend an existing session; decode requires exactly one token and a
    /// nonempty committed prefix. Forward calls and PP stages are serial; TP
    /// owners within a stage execute concurrently.
    pub fn forward(
        &mut self,
        session: SessionId,
        tokens: &[u32],
        phase: ForwardPhase,
    ) -> Result<PipelineOutput> {
        self.forward_cancellable(session, tokens, phase, &AtomicBool::new(false))
    }

    /// Cancellation is honored before the global decision. After Commit, it
    /// cannot reverse physical installation; all install/retirement ACKs must drain.
    pub fn forward_cancellable(
        &mut self,
        session: SessionId,
        tokens: &[u32],
        phase: ForwardPhase,
        cancellation: &AtomicBool,
    ) -> Result<PipelineOutput> {
        self.forward_observed(session, tokens, phase, cancellation, |_| {})
    }

    pub fn forward_observed(
        &mut self,
        session: SessionId,
        tokens: &[u32],
        phase: ForwardPhase,
        cancellation: &AtomicBool,
        mut observe: impl FnMut(PipelineProgress),
    ) -> Result<PipelineOutput> {
        self.available()?;
        if tokens.is_empty()
            || tokens.len() > self.config.max_batch_tokens
            || tokens
                .iter()
                .any(|&token| token as usize >= self.descriptions[0].vocabulary)
            || (phase == ForwardPhase::Decode && tokens.len() != 1)
        {
            return Err(error("invalid pipeline token, phase or batch capacity"));
        }
        if phase == ForwardPhase::Decode && !self.sessions.contains_key(&session) {
            return Err(error("decode needs a committed session prefix"));
        }
        if cancellation.load(Ordering::Acquire) {
            return Err(error("pipeline cancelled before admission"));
        }
        let slot = self.ensure_session(session)?;
        let start = self
            .page_manager
            .block_table(slot)
            .expect("allocated sequence")
            .committed_tokens();
        let end = start
            .checked_add(tokens.len())
            .filter(|&end| end <= self.config.max_positions)
            .ok_or_else(|| error("pipeline position capacity exceeded"))?;
        if phase == ForwardPhase::Decode && start == 0 {
            return Err(error("decode needs a nonempty prefix"));
        }
        let generation = self.page_manager.sequence_generation(slot).map_err(error)?;
        let id = ExecutionTransactionId::new(self.next_transaction)?;
        self.next_transaction = self
            .next_transaction
            .checked_add(1)
            .ok_or_else(|| error("transaction identity exhausted"))?;
        let binding = KvCommitBinding::new(
            id,
            self.coordinator.topology().topology_id(),
            self.scopes.kv_participants().as_participants().clone(),
            id.get(),
        )?;
        let keys = (0..self.descriptions.len())
            .map(|index| WireKey {
                binding: binding.clone(),
                rank: Self::rank(index),
                session,
            })
            .collect::<Vec<_>>();
        coordinator(
            self.coordinator
                .begin_scoped(id, self.scopes.kv_participants().as_participants().clone()),
        )?;
        // This custody flag is cleared only by an acknowledged terminal path.
        // Collective failure independently blocks reuse even after safe rollback.
        self.quarantined = true;
        coordinator(self.coordinator.reserve(id, keys.len()))?;
        for key in &keys {
            coordinator(self.coordinator.prepare(id, key.rank.global))?;
        }
        let mut reservation = match self.page_manager.reserve(slot, generation, tokens.len()) {
            Ok(reservation) => reservation,
            Err(source) => return Err(self.cleanup_error(id, &keys, None, error(source))),
        };
        let inputs = (|| {
            self.page_manager
                .bind_reservation_execution(&mut reservation, StateSlot::new(0), generation)
                .map_err(error)?;
            let view = self
                .page_manager
                .reservation_view(&reservation)
                .map_err(error)?;
            let bindings = self
                .page_manager
                .reservation_bindings(&view)
                .map_err(error)?;
            let mode = match phase {
                ForwardPhase::Prefill => ForwardMode::Prefill,
                ForwardPhase::Decode => ForwardMode::Decode,
            };
            let batch = ExecutionBatch::new(
                mode,
                tokens.to_vec(),
                (start..end).map(|p| p as u32).collect(),
                bindings.write_slots.into_iter().map(Some).collect(),
                vec![LogitsRequest::Full; tokens.len()],
                vec![ExecutionSequence::new(
                    StateSlot::new(0),
                    phase,
                    0..tokens.len() as u32,
                    start as u32,
                    end as u32,
                    0..bindings.block_ids.len() as u32,
                )],
                bindings.block_ids,
            );
            Ok((batch, view))
        })();
        let (batch, view) = match inputs {
            Ok(inputs) => inputs,
            Err(source) => {
                return Err(self.cleanup_error(
                    id,
                    &keys,
                    Some(LogicalRollback::Reservations(vec![reservation])),
                    source,
                ));
            }
        };
        let logical = match self
            .page_manager
            .prepare_commit(vec![KvReservationCommit::new(reservation, tokens.len())])
        {
            Ok(logical) => logical,
            Err(failure) => {
                let (source, commits) = failure.into_parts();
                return Err(self.cleanup_error(
                    id,
                    &keys,
                    Some(LogicalRollback::Reservations(
                        commits
                            .into_iter()
                            .map(|commit| commit.reservation)
                            .collect(),
                    )),
                    error(source),
                ));
            }
        };
        let mut remotes = Vec::with_capacity(keys.len());
        let mut slots = Vec::with_capacity(keys.len());
        let receipts = keys
            .iter()
            .map(|_| Rc::new(Receipts::default()))
            .collect::<Vec<_>>();
        let computed = (|| {
            let mut input = SegmentInput::Tokens;
            let mut logits = None;
            let tp = self.coordinator.topology().plan().tensor_parallel;
            let pp = self.coordinator.topology().plan().pipeline_parallel;
            for stage in 0..pp {
                if cancellation.load(Ordering::Acquire) {
                    return Err(error("pipeline cancelled during compute"));
                }
                let cancelled = Arc::new(AtomicBool::new(false));
                let stage_keys = &keys[stage * tp..(stage + 1) * tp];
                // Prepare everyone before admitting any collective execution.
                for (local, key) in stage_keys.iter().enumerate() {
                    let index = stage * tp + local;
                    let Reply::Prepared { projection } = self.command(
                        key.rank,
                        Command::Prepare {
                            key: key.clone(),
                            batch: Box::new(batch.clone()),
                            reservation: view.clone(),
                        },
                    )?
                    else {
                        return Err(error("missing stage preparation"));
                    };
                    let packed =
                        projection.into_commit_batch(&batch, &view, &self.descriptions[index])?;
                    remotes.push(RemotePhysicalOwner {
                        transport: Rc::clone(&self.transport),
                        batch: packed,
                        receipts: Rc::clone(&receipts[index]),
                    });
                    slots.push(Some(RemoteToken { key: key.clone() }));
                }
                if let Some(control) = self.tensor_controls.get(stage) {
                    control.begin_sealed(
                        tensor_scope::TensorBatchIdentity::from_packed(
                            &binding,
                            &remotes[stage * tp].batch,
                            &[session.0],
                        )?,
                        Arc::clone(&cancelled),
                    )?;
                }
                let commands = stage_keys
                    .iter()
                    .map(|key| {
                        let input = match &input {
                            SegmentInput::Tokens => SegmentInput::Tokens,
                            SegmentInput::Hidden { next_layer, rows } => SegmentInput::Hidden {
                                next_layer: *next_layer,
                                rows: rows.clone(),
                            },
                        };
                        (
                            key.rank,
                            Command::Execute {
                                key: key.clone(),
                                input,
                                cancellation: Arc::clone(&cancelled),
                            },
                        )
                    })
                    .collect();
                let replies = self
                    .transport
                    .borrow_mut()
                    .call_batch_observed_with_failure(
                        commands,
                        &mut || {
                            if cancellation.load(Ordering::Acquire)
                                || cancelled.load(Ordering::Acquire)
                                || !matches!(
                                    self.coordinator.state(id),
                                    Ok(crate::TransactionState::Preparing)
                                )
                            {
                                cancelled.store(true, Ordering::Release);
                                if let Some(control) = self.tensor_controls.get(stage) {
                                    let _ = control.abort(id, "pipeline cancellation");
                                }
                            }
                        },
                        &mut || {
                            cancelled.store(true, Ordering::Release);
                            if let Some(control) = self.tensor_controls.get(stage) {
                                let _ = control.abort(id, "pipeline owner failure");
                            }
                        },
                    )?;
                if replies.len() != tp {
                    return Err(error("incomplete tensor stage output cohort"));
                }
                let mut leader = None;
                for reply in replies {
                    let Reply::Executed { output } = reply else {
                        return Err(error("missing segment output"));
                    };
                    self.descriptions[stage * tp].validate_output(&output, tokens.len())?;
                    if let Some(first) = &leader {
                        if !identical_output(first, &output) {
                            return Err(error(
                                "TP boundary outputs are not identical replicated values",
                            ));
                        }
                    } else {
                        leader = Some(output);
                    }
                }
                if let Some(control) = self.tensor_controls.get(stage) {
                    control.finish(id)?;
                }
                input = match leader.expect("nonempty tensor stage") {
                    SegmentOutput::Hidden { next_layer, rows } if stage + 1 < pp => {
                        let shape = rows.shape();
                        let dtype = rows.dtype();
                        SegmentInput::Hidden {
                            next_layer,
                            rows: HostRows::new(shape, dtype, None, rows.into_values())?,
                        }
                    }
                    SegmentOutput::Logits(output) if stage + 1 == pp => {
                        logits = Some(output);
                        SegmentInput::Tokens
                    }
                    _ => return Err(error("segment output endpoint mismatch")),
                };
            }
            logits.ok_or_else(|| error("pipeline output is absent"))
        })();
        let logits = match computed {
            Ok(logits) => logits,
            Err(source) => {
                for control in &self.tensor_controls {
                    control.fail_execution(id);
                }
                return Err(self.cleanup_error(
                    id,
                    &keys,
                    Some(LogicalRollback::Prepared(logical)),
                    source,
                ));
            }
        };
        let owners = remotes
            .iter_mut()
            .zip(slots.iter_mut())
            .zip(&keys)
            .map(|((backend, slot), key)| KvCommitOwner::new(key.rank.global, backend, slot, &[]))
            .collect();
        let mut prepared =
            match PreparedKvCommit::prepare_commit_ready(binding.clone(), logical, owners) {
                Ok(prepared) => prepared,
                Err(failure) => {
                    let (source, logical, owners) = failure.into_parts();
                    drop(owners);
                    return Err(self.cleanup_error(
                        id,
                        &keys,
                        Some(LogicalRollback::Prepared(logical)),
                        source,
                    ));
                }
            };
        for key in &keys {
            coordinator(self.coordinator.prepare_vote(
                id,
                key.rank.global,
                FinalizeOutcome::Success,
            ))?;
        }
        self.report(id, &mut observe)?;
        if cancellation.load(Ordering::Acquire) {
            coordinator(self.coordinator.abort_decision(id))?;
            for key in &keys {
                poll_end(self.config.max_ack_polls, || {
                    prepared
                        .abort_prepared(&binding, key.rank.global)
                        .map(|ack| ack.progress())
                })?;
                coordinator(self.coordinator.finalize(
                    id,
                    key.rank.global,
                    FinalizeOutcome::Success,
                ))?;
            }
            self.finish_receipts(&keys, &receipts)?;
            let retirement = prepared.abort_logical(&binding, |logical| {
                self.page_manager.abort_prepared_commit(logical)
            })?;
            self.confirm_retirement(retirement)?;
            coordinator(self.coordinator.cancel(id))?;
            coordinator(self.coordinator.retire(id))?;
            self.quarantined = false;
            return Err(error("pipeline cancelled before decision"));
        }
        let decision = coordinator(self.coordinator.commit_decision(id))?;
        if decision != crate::Decision::Commit {
            return Err(error("all-ready pipeline did not commit"));
        }
        self.report(id, &mut observe)?;
        for key in &keys {
            // Even Err after dispatch is poll-only. A revalidation error before
            // dispatch cannot be guessed safe either: the immutable decision and
            // its reservation remain quarantined if no owner ACK can be obtained.
            let installed = prepared.install_commit(&binding, key.rank.global);
            if !matches!(installed, Ok(ref ack) if ack.progress() == KvEndProgress::Complete) {
                self.report(id, &mut observe)?;
                poll_end(self.config.max_ack_polls, || {
                    let progress = prepared
                        .poll_install_ack(&binding, key.rank.global)
                        .map(|ack| ack.progress());
                    self.report(id, &mut observe)?;
                    progress
                })?;
            }
            coordinator(
                self.coordinator
                    .finalize(id, key.rank.global, FinalizeOutcome::Success),
            )?;
            self.report(id, &mut observe)?;
        }
        // The model callbacks are infallible, the transport is not. Collect all
        // physical publication receipts before invoking that infallible boundary.
        for (key, receipts) in keys.iter().zip(&receipts) {
            complete(self.command(key.rank, Command::Publish(key.clone()))?)?;
            receipts.published.set(true);
        }
        let mut retirement = prepared.publish_logical(&binding, |logical| {
            let retirement = self.page_manager.publish_commit(logical);
            let pages = retirement.pages().to_vec();
            (retirement, pages)
        })?;
        coordinator(self.coordinator.publish(id))?;
        self.report(id, &mut observe)?;
        for key in &keys {
            poll_end(self.config.max_ack_polls, || {
                retirement
                    .retire_owner(&binding, key.rank.global)
                    .map(|ack| ack.progress())
            })?;
        }
        self.finish_receipts(&keys, &receipts)?;
        let retirement = retirement.finish_retirement(&binding)?;
        self.confirm_retirement(retirement)?;
        coordinator(self.coordinator.retire(id))?;
        self.quarantined = false;
        Ok(PipelineOutput {
            transaction: id,
            logits,
        })
    }

    fn finish_receipts(&self, keys: &[WireKey], receipts: &[Rc<Receipts>]) -> Result<()> {
        for (key, receipts) in keys.iter().zip(receipts) {
            complete(self.command(key.rank, Command::Finish(key.clone()))?)?;
            receipts.finished.set(true);
        }
        Ok(())
    }
}

fn poll_end(limit: usize, mut poll: impl FnMut() -> Result<KvEndProgress>) -> Result<()> {
    let mut last_error = None;
    for _ in 0..limit {
        match poll() {
            Ok(KvEndProgress::Complete) => return Ok(()),
            Ok(KvEndProgress::ConsumedRejected) => {
                return Err(error("consumed rejection is not a cleanup ACK"));
            }
            Ok(KvEndProgress::Pending) => {}
            Err(source) => {
                last_error = Some(source);
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Err(last_error.unwrap_or_else(|| error("owner ACK poll budget exhausted; custody retained")))
}

impl Drop for PipelineParallelExecutor {
    fn drop(&mut self) {
        if !self.quarantined && !self.closed {
            let _ = self.shutdown();
        }
        if self.quarantined {
            // No potentially unbounded join of an unknown GPU worker on Drop.
            self.transport.borrow_mut().quarantine();
        }
    }
}

fn identical_output(left: &SegmentOutput, right: &SegmentOutput) -> bool {
    let equal = |left: &[f32], right: &[f32]| {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits())
    };
    match (left, right) {
        (
            SegmentOutput::Hidden {
                next_layer: a,
                rows: x,
            },
            SegmentOutput::Hidden {
                next_layer: b,
                rows: y,
            },
        ) => {
            a == b
                && x.shape() == y.shape()
                && x.dtype() == y.dtype()
                && equal(x.values(), y.values())
        }
        (SegmentOutput::Logits(x), SegmentOutput::Logits(y)) => {
            x.rows() == y.rows() && x.width() == y.width() && equal(x.values(), y.values())
        }
        _ => false,
    }
}
