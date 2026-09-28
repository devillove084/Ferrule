//! Persistent CPU/CUDA expert owners, with activation/result-only transport.
//!
//! Explicit groups are independent of common's PP/DP/TP mesh. `ExpertRank::slot`
//! addresses a private DP pool; `owner` and `ExpertGroup::source_rank` are global
//! identities, NOT pool slots or device ordinals. The source is the caller and
//! may be a root KV rank outside the expert owners. Pipeline integration requires
//! disjoint owner IDs across groups and from KV ranks.
//!
//! There is no transaction allocator, publication, retry or KV state here. The
//! caller supplies the existing transaction and its liveness check on every call.
//! Private monotonic command serials only admit work to DataParallelExecutor.
//! Cancellation drains admitted work before returning, never publishes a partial
//! reply, and cannot preempt a synchronous CPU kernel. Shutdown/Drop join owners
//! using DP's existing lifecycle; unknown quiescence propagates as its typed fatal
//! signal so an enclosing PP owner cannot fabricate a rollback ACK.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{
    Error, ExpertDispatchMembers, ParallelRankId, Result, ValidatedParallelTopology,
};
use ferrule_model::TensorRole;
use ferrule_model::moe::ExpertId;
use ferrule_model::transformer::expert_parallel::{
    CpuReferenceExpertWorker, ExpertDispatchContext, ExpertDispatchLimits,
    ExpertParallelRoutedExecutor, ExpertPlacement, ExpertResult, ExpertResultExecutor,
    ExpertTokenBucket, ExpertWorker, RoutedSwiGluExecutor, RoutedSwiGluRequest,
};
use ferrule_model::transformer::{
    BoundDecoderResources, ExpertAvailability, ExpertProvider, FeedForward, OperatorProgress,
    PreparedLinear, PreparedSwiGlu, Rows, StateDictMaterializer,
};

use super::data::{
    BuildError, CompletionOutcome, DataParallelConfig, DataParallelExecutor, OwnerFailureKind,
    PanicQuiescence, ReplicaWorker, WorkRequest,
};
use crate::SessionId;

/// The already-prewarmed image authority, shared by every CUDA owner.
#[cfg(feature = "cuda")]
pub type ArcHostExpertCache = Arc<ferrule_model::transformer::host_experts::HostExpertCache>;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct NumericExpertConfig {
    pub max_parameter_bytes: u64,
    /// All owned compressed weights + workspace + worst admitted activation
    /// scratch. Physical-card/root/allocator admission remains the factory's job.
    pub max_device_bytes: usize,
    pub workspace_bytes: usize,
    pub dispatch_timeout: Duration,
}

fn error(message: impl std::fmt::Display) -> Error {
    Error::Execution {
        message: format!("expert runtime: {message}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRank {
    pub slot: ParallelRankId,
    pub owner: ParallelRankId,
}

/// Explicit per-segment EP membership and dispatch bounds. Member order is the
/// model dispatch order and creates exactly one worker per member. The source
/// is the explicitly attached caller: it may be a member (legacy PP/EP) or an
/// external root, but is never implicitly added to the workers or placement.
/// Placement coordinates and layer range are global decoder coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertGroup {
    pub source_rank: ParallelRankId,
    pub members: Vec<ParallelRankId>,
    pub layers: Range<usize>,
    pub placement: ExpertPlacement,
    pub limits: ExpertDispatchLimits,
}

impl ExpertGroup {
    /// Bind this existing dispatch group to mesh metadata without changing its
    /// slot order, placement, or activation-only execution/transaction protocol.
    /// This legacy PP mesh attachment retains its member-source and disjoint-KV
    /// contract. An external-root hybrid attachment binds the source directly
    /// through this group instead; it does not extend the PP mesh membership.
    pub fn dispatch_members(
        &self,
        topology: &ValidatedParallelTopology,
        replica: u32,
        stage: u32,
    ) -> Result<ExpertDispatchMembers> {
        self.validate()?;
        ExpertDispatchMembers::new(
            topology,
            replica,
            stage,
            self.source_rank,
            self.members.iter().copied(),
        )
        .map_err(|e| error(format!("expert attachment scope: {e:?}")))
    }

    fn validate(&self) -> Result<()> {
        if self.members.is_empty()
            || self.members.len() > u32::MAX as usize
            || self.layers.is_empty()
            || self.limits.max_tokens == 0
            || self.limits.max_bytes == 0
            || self.members.iter().collect::<BTreeSet<_>>().len() != self.members.len()
        {
            return Err(error(
                "invalid expert owners, layer range or dispatch limits",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_resources(&self, resources: &BoundDecoderResources) -> Result<()> {
        self.validate()?;
        if self.layers.end > resources.spec().layers().len() {
            return Err(error("group layers outside decoder"));
        }
        let mut routed = false;
        for layer in self.layers.clone() {
            if let FeedForward::Moe(moe) = resources.spec().layers()[layer].feed_forward() {
                routed = true;
                for expert in 0..moe.router_spec().num_experts() {
                    if self
                        .placement
                        .owner(layer, expert)
                        .is_none_or(|owner| !self.members.contains(&owner))
                    {
                        return Err(error(format!(
                            "missing or non-member owner for {layer}:{expert}"
                        )));
                    }
                }
            }
        }
        if !routed {
            return Err(error("expert group contains no routed layer"));
        }
        Ok(())
    }

    fn validate_context(&self, context: ExpertDispatchContext) -> Result<()> {
        if context.source_rank != self.source_rank || !self.layers.contains(&context.layer) {
            return Err(error("unknown expert source or layer"));
        }
        Ok(())
    }

    fn validate_bucket(
        &self,
        context: ExpertDispatchContext,
        bucket: &ExpertTokenBucket,
    ) -> Result<()> {
        self.validate_context(context)?;
        if !self.members.contains(&bucket.owner_rank) {
            return Err(error("unknown expert owner"));
        }
        if bucket.tokens.len() > self.limits.max_tokens {
            return Err(error("routed token limit exceeded"));
        }
        let mut bytes = 0usize;
        for token in &bucket.tokens {
            if token.transaction != context.transaction
                || token.source_rank != context.source_rank
                || token.expert.layer != context.layer
            {
                return Err(error("unknown token transaction, source or layer"));
            }
            if self
                .placement
                .owner(token.expert.layer, token.expert.expert)
                != Some(bucket.owner_rank)
            {
                return Err(error("unknown or non-local expert"));
            }
            bytes = token
                .payload
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .and_then(|size| bytes.checked_add(size))
                .ok_or_else(|| error("activation byte count overflow"))?;
            if bytes > self.limits.max_bytes {
                return Err(error("dispatch byte limit exceeded"));
            }
        }
        Ok(())
    }
}

/// A strictly local provider. No checkpoint directory/materializer or unowned
/// bindings survive preparation. Prepared weight handles never enter a reply.
struct LocalExperts(BTreeMap<ExpertId, Arc<PreparedSwiGlu>>);

impl ExpertProvider for LocalExperts {
    fn expert(&mut self, layer: usize, expert: usize) -> Result<ExpertAvailability> {
        self.0
            .get(&ExpertId::new(layer, expert))
            .cloned()
            .map(ExpertAvailability::Ready)
            .ok_or_else(|| error(format!("provider does not own expert {layer}:{expert}")))
    }
}

/// Construct only inside the owner factory. The factory borrows resources,
/// never transfers them. CUDA owners retain only assigned resident experts.
pub struct ExpertRankWorker {
    rank: ExpertRank,
    group: ExpertGroup,
    experts: LocalExperts,
    stats: ExpertOwnerStats,
    #[cfg(feature = "cuda")]
    cuda: Option<ferrule_model::transformer::CudaStandardDecoderOperators>,
    #[cfg(feature = "cuda")]
    resident_failed: bool,
}

#[derive(Debug, Clone)]
pub struct ExpertOwnerStats {
    pub rank: ExpertRank,
    pub thread: std::thread::ThreadId,
    pub owned_experts: Vec<ExpertId>,
    pub calls: usize,
    pub tokens: usize,
    pub last_context: Option<ExpertDispatchContext>,
    #[cfg(feature = "cuda")]
    pub cuda_cache: Option<ferrule_model::transformer::ExpertCacheStats>,
}

impl ExpertRankWorker {
    pub fn prepare_cpu(
        resources: &BoundDecoderResources,
        rank: ExpertRank,
        group: &ExpertGroup,
        max_parameter_bytes: u64,
    ) -> Result<Self> {
        Self::prepare_parameters(resources, rank, group, max_parameter_bytes)
    }

    fn prepare_parameters(
        resources: &BoundDecoderResources,
        rank: ExpertRank,
        group: &ExpertGroup,
        max_parameter_bytes: u64,
    ) -> Result<Self> {
        let materializer = StateDictMaterializer::new(max_parameter_bytes)?;
        Self::prepare_with_materializer(resources, rank, group, &materializer)
    }

    fn prepare_with_materializer(
        resources: &BoundDecoderResources,
        rank: ExpertRank,
        group: &ExpertGroup,
        materializer: &StateDictMaterializer,
    ) -> Result<Self> {
        group.validate_resources(resources)?;
        if group.members.get(rank.slot.get() as usize) != Some(&rank.owner) {
            return Err(error("expert slot/owner mismatch"));
        }
        let mut experts = BTreeMap::new();
        for layer in group.layers.clone() {
            let FeedForward::Moe(moe) = resources.spec().layers()[layer].feed_forward() else {
                continue;
            };
            for expert in 0..moe.router_spec().num_experts() {
                if group.placement.owner(layer, expert) != Some(rank.owner) {
                    continue;
                }
                let linear = |role: TensorRole, shape: [usize; 2]| {
                    let binding =
                        resources
                            .experts()
                            .require_shape(layer, expert, role.clone(), &shape)?;
                    PreparedLinear::from_parameter(
                        materializer.expert_parameter(layer, expert, binding)?,
                        role,
                    )
                };
                let prepared = PreparedSwiGlu::new(
                    linear(
                        TensorRole::RoutedExpertGate,
                        moe.expert().gate().weight_shape(),
                    )?,
                    linear(TensorRole::RoutedExpertUp, moe.expert().up().weight_shape())?,
                    linear(
                        TensorRole::RoutedExpertDown,
                        moe.expert().down().weight_shape(),
                    )?,
                    moe.expert().activation_limit(),
                )?;
                experts.insert(ExpertId::new(layer, expert), Arc::new(prepared));
            }
        }
        let stats = ExpertOwnerStats {
            rank,
            thread: std::thread::current().id(),
            owned_experts: experts.keys().copied().collect(),
            calls: 0,
            tokens: 0,
            last_context: None,
            #[cfg(feature = "cuda")]
            cuda_cache: None,
        };
        Ok(Self {
            rank,
            group: group.clone(),
            experts: LocalExperts(experts),
            stats,
            #[cfg(feature = "cuda")]
            cuda: None,
            #[cfg(feature = "cuda")]
            resident_failed: false,
        })
    }

    /// Called on the expert owner with its own operators. Parameter preparation
    /// is shared with CPU, but all activation computation is explicitly CUDA.
    #[cfg(feature = "cuda")]
    pub fn prepare_cuda(
        resources: &BoundDecoderResources,
        rank: ExpertRank,
        group: &ExpertGroup,
        max_parameter_bytes: u64,
        ops: std::rc::Rc<ferrule_backend::cuda::operators::linear::CudaOperators>,
    ) -> Result<Self> {
        let mut worker = Self::prepare_parameters(resources, rank, group, max_parameter_bytes)?;
        let parameters = worker
            .experts
            .0
            .values()
            .flat_map(|expert| {
                [expert.gate(), expert.up(), expert.down()]
                    .map(|linear| linear.parameter().binding().clone())
            })
            .collect::<Vec<_>>();
        let mut operators = ferrule_model::transformer::CudaStandardDecoderOperators::new(
            ops,
            ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
            &parameters,
        )?;
        for expert in worker.experts.0.values() {
            let prepared = operators.prepare_expert(expert);
            if operators.needs_quarantine() {
                std::mem::forget(operators);
                std::mem::forget(prepared);
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
            prepared?;
        }
        worker.cuda = Some(operators);
        Ok(worker)
    }

    /// Eager bounded all-owned residency. Never prewarms or reads the full
    /// checkpoint here; prepared parameters are Arc hits in the shared image.
    #[cfg(feature = "cuda")]
    pub fn prepare_cuda_numeric_f32(
        resources: &BoundDecoderResources,
        rank: ExpertRank,
        group: &ExpertGroup,
        host: &ArcHostExpertCache,
        config: NumericExpertConfig,
        ops: std::rc::Rc<ferrule_backend::cuda::operators::linear::CudaOperators>,
    ) -> Result<Self> {
        use ferrule_model::transformer::{
            CudaStandardDecoderOperators, ExpertCacheLimits, ExpertCachePolicy, NumericFp8Precision,
        };
        host.preflight()?;
        let materializer = StateDictMaterializer::new(config.max_parameter_bytes)?;
        materializer.attach_host_experts(host)?;
        let mut worker = Self::prepare_with_materializer(resources, rank, group, &materializer)?;
        let mut payload = 0usize;
        let mut scratch = 0usize;
        let mut parameters = Vec::new();
        for expert in worker.experts.0.values() {
            for linear in [expert.gate(), expert.up(), expert.down()] {
                let artifact = linear
                    .numeric_fp8()
                    .ok_or_else(|| error("numeric EP requires compressed FP8 experts"))?;
                payload = usize::try_from(artifact.storage_bytes())
                    .ok()
                    .and_then(|n| payload.checked_add(n))
                    .ok_or_else(|| error("resident expert payload overflow"))?;
                parameters.push(linear.parameter().binding().clone());
            }
            let bytes = expert
                .gate()
                .out_features()
                .checked_mul(3)
                .and_then(|n| n.checked_add(expert.input_width()))
                .and_then(|n| n.checked_add(expert.output_width()))
                .and_then(|n| n.checked_mul(group.limits.max_tokens))
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| error("expert scratch overflow"))?;
            scratch = scratch.max(bytes);
        }
        let required = payload
            .checked_add(config.workspace_bytes)
            .and_then(|n| n.checked_add(scratch))
            .ok_or_else(|| error("resident expert budget overflow"))?;
        if config.workspace_bytes == 0 || required > config.max_device_bytes {
            return Err(error(format!(
                "all-owned residency requires {required} bytes, budget {}",
                config.max_device_bytes
            )));
        }
        let count = worker.experts.0.len();
        let mut operators = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            ops,
            &parameters,
            ExpertCachePolicy::Bounded(ExpertCacheLimits {
                // Exactly the declared owned set; not the historical 1024 cap.
                max_experts: count.max(1),
                max_bytes: config.max_device_bytes,
            }),
            config.workspace_bytes,
            NumericFp8Precision::F32Tf32x3,
        )?;
        for expert in worker.experts.0.values() {
            let prepared = operators.prepare_expert(expert);
            if operators.needs_quarantine() {
                std::mem::forget(operators);
                std::mem::forget(worker);
                std::mem::forget(prepared);
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
            prepared?;
        }
        let stats = operators
            .expert_cache_stats()
            .expect("bounded numeric profile");
        if stats.resident_experts != count
            || stats.resident_bytes != payload
            || stats.evictions != 0
            || stats.pending_upload_bytes != 0
            || stats.quarantined
        {
            return Err(error(
                "incomplete all-owned CUDA residency at Ready barrier",
            ));
        }
        worker.cuda = Some(operators);
        Ok(worker)
    }

    fn compute(
        &mut self,
        tokens: &[ferrule_model::transformer::ExpertToken],
    ) -> Result<Vec<ExpertResult>> {
        #[cfg(feature = "cuda")]
        if let Some(operators) = self.cuda.as_mut() {
            if self.resident_failed {
                return Err(error(
                    "resident expert owner failed; reload/retry is disabled",
                ));
            }
            let result = ferrule_model::transformer::CudaExpertWorker::new(
                self.rank.owner,
                &self.group.placement,
                &mut self.experts,
                operators,
            )
            .compute(tokens);
            if operators.needs_quarantine() {
                std::mem::forget(result);
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
            if operators.numeric_fp8_precision().is_some() {
                // A failed numeric operation may evict its local lease. This
                // all-owned profile must never reload weights during a retry.
                let stats = operators
                    .expert_cache_stats()
                    .expect("bounded resident owner");
                self.resident_failed = result.is_err()
                    || stats.evictions != 0
                    || stats.resident_experts != self.experts.0.len()
                    || stats.uploads != self.experts.0.len() as u64;
                if self.resident_failed && result.is_ok() {
                    return Err(error("resident expert set changed after Ready"));
                }
            }
            return result;
        }
        CpuReferenceExpertWorker::new(self.rank.owner, &self.group.placement, &mut self.experts)
            .compute(tokens)
    }
}

#[derive(Debug)]
enum Command {
    Stats,
    Compute {
        context: ExpertDispatchContext,
        bucket: ExpertTokenBucket,
    },
}

#[derive(Debug)]
enum Reply {
    Stats(ExpertOwnerStats),
    Results(Vec<ExpertResult>),
}

impl ReplicaWorker<Command> for ExpertRankWorker {
    type Output = Reply;
    type Error = Error;

    fn execute(&mut self, request: WorkRequest<Command>) -> Result<Reply> {
        if request.rank != self.rank.slot {
            return Err(error("incorrect expert pool slot"));
        }
        match request.input {
            Command::Stats => {
                #[cfg(feature = "cuda")]
                {
                    self.stats.cuda_cache =
                        self.cuda.as_ref().and_then(|ops| ops.expert_cache_stats());
                }
                Ok(Reply::Stats(self.stats.clone()))
            }
            Command::Compute { context, bucket } => {
                self.group.validate_bucket(context, &bucket)?;
                if bucket.owner_rank != self.rank.owner {
                    return Err(error("incorrect expert owner"));
                }
                if request.cancellation.is_requested() {
                    return Err(error("cancelled expert work"));
                }
                let results = self.compute(&bucket.tokens)?;
                self.stats.calls = self.stats.calls.saturating_add(1);
                self.stats.tokens = self.stats.tokens.saturating_add(bucket.tokens.len());
                self.stats.last_context = Some(context);
                if request.cancellation.is_requested() {
                    return Err(error("cancelled expert result"));
                }
                Ok(Reply::Results(results))
            }
        }
    }

    fn panic_quiescence(&mut self) -> PanicQuiescence {
        #[cfg(feature = "cuda")]
        if let Some(operators) = self.cuda.as_mut() {
            if operators.quiesce().is_err() || operators.needs_quarantine() {
                return PanicQuiescence::Unknown;
            }
        }
        PanicQuiescence::Quiescent
    }

    fn shutdown(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Some(operators) = self.cuda.as_mut() {
            let result = operators.quiesce();
            if operators.needs_quarantine() {
                std::mem::forget(result);
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
            result?;
        }
        Ok(())
    }
}

/// One bounded persistent pool per explicit group. At most one command is
/// outstanding per owner; all owners are admitted before collection. Routes
/// are bounded by group size. Payloads are bounded before admission, replies by the
/// worker's shape-preserving SwiGLU. No test-supplied worker loop is necessary.
pub struct ExpertRankWorkers {
    group: Arc<ExpertGroup>,
    pool: Option<DataParallelExecutor<Command, Reply, Error>>,
    next_serial: u64,
    closed: bool,
    unknown: bool,
    timeout: Duration,
    tickets: BTreeMap<ExecutionTransactionId, ExpertTicket>,
}

struct ExpertTicket {
    slot: ParallelRankId,
    route: SessionId,
    outer: Option<ExecutionTransactionId>,
    index: usize,
}

impl Drop for ExpertRankWorkers {
    fn drop(&mut self) {
        let Some(mut pool) = self.pool.take() else {
            return;
        };
        if !self.unknown {
            pool.begin_shutdown();
            let deadline = Instant::now() + self.timeout;
            while !pool.shutdown_ready() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_micros(50));
            }
            self.unknown = !pool.shutdown_ready();
        }
        if self.unknown {
            // Never block Drop indefinitely or release unresolved ticket custody.
            std::mem::forget(pool);
            std::mem::forget(std::mem::take(&mut self.tickets));
        }
    }
}

impl ExpertRankWorkers {
    /// Factory runs on each DP owner; only paths/metadata should be captured.
    pub fn new<F>(group: ExpertGroup, factory: F) -> Result<Self>
    where
        F: FnOnce(ExpertRank, &ExpertGroup) -> Result<ExpertRankWorker> + Clone + Send + 'static,
    {
        group.validate()?;
        let group = Arc::new(group);
        let owners = Arc::clone(&group);
        let pool = DataParallelExecutor::new(
            DataParallelConfig {
                replicas: group.members.len(),
                max_outstanding_per_replica: 1,
                session_capacity: group.members.len(),
            },
            move |slot| {
                let rank = ExpertRank {
                    slot,
                    owner: owners.members[slot.get() as usize],
                };
                let worker = factory(rank, &owners)?;
                if worker.rank != rank
                    || worker.group != *owners
                    || worker.stats.thread != std::thread::current().id()
                {
                    return Err(error(
                        "factory returned a different group/owner or non-local worker",
                    ));
                }
                Ok(worker)
            },
        )
        .map_err(|e| {
            if let BuildError::Owners(failures) = &e {
                if failures
                    .iter()
                    .any(|f| matches!(f.kind, OwnerFailureKind::QuiescenceUnknown))
                {
                    std::panic::panic_any(PanicQuiescence::Unknown);
                }
            }
            error(format!("owner construction: {e:?}"))
        })?;
        Ok(Self {
            group,
            pool: Some(pool),
            next_serial: 1,
            closed: false,
            unknown: false,
            timeout: Duration::from_secs(30),
            tickets: BTreeMap::new(),
        })
    }

    pub fn group(&self) -> &ExpertGroup {
        &self.group
    }
    pub fn outstanding(&self) -> usize {
        self.tickets.len()
    }

    /// Bounds an entire dispatch, including cancellation drain. Expiry loses
    /// proof permanently; a later host reply cannot retroactively acknowledge it.
    pub fn set_timeout(&mut self, timeout: Duration) -> Result<()> {
        if timeout.is_zero() || Instant::now().checked_add(timeout).is_none() {
            return Err(error("invalid expert dispatch timeout"));
        }
        self.timeout = timeout;
        Ok(())
    }

    fn pool(&mut self) -> &mut DataParallelExecutor<Command, Reply, Error> {
        self.pool.as_mut().expect("live expert transport")
    }

    fn lose_proof(&mut self) -> ! {
        self.unknown = true;
        self.closed = true;
        self.cancel_all();
        self.pool().begin_shutdown();
        std::panic::panic_any(PanicQuiescence::Unknown);
    }

    /// Cancellation intent for the enclosing transaction, never a private DP
    /// serial. This does not retire tickets or claim a device acknowledgement.
    pub fn cancel_transaction(&mut self, transaction: ExecutionTransactionId) {
        for (serial, ticket) in &self.tickets {
            if ticket.outer == Some(transaction) {
                let _ = self.pool.as_mut().unwrap().cancel(*serial);
            }
        }
    }

    fn cancel_all(&mut self) {
        for serial in self.tickets.keys() {
            let _ = self.pool.as_mut().unwrap().cancel(*serial);
        }
    }

    fn admit(&mut self, owner: ParallelRankId, command: Command, index: usize) -> Result<()> {
        let index_owner = self
            .group
            .members
            .iter()
            .position(|&m| m == owner)
            .ok_or_else(|| error("unknown expert owner"))?;
        let slot = ParallelRankId::new(index_owner as u32);
        let route = SessionId(index_owner as u64 + 1);
        let serial = ExecutionTransactionId::new(self.next_serial)?;
        self.next_serial = self
            .next_serial
            .checked_add(1)
            .ok_or_else(|| error("command serial exhausted"))?;
        let outer = match &command {
            Command::Compute { context, .. } => Some(context.transaction),
            Command::Stats => None,
        };
        self.pool()
            .try_submit_to(route, slot, serial, command)
            .map_err(|e| error(format!("owner admission: {:?}", e.kind)))?;
        self.tickets.insert(
            serial,
            ExpertTicket {
                slot,
                route,
                outer,
                index,
            },
        );
        Ok(())
    }

    fn receive(&mut self) -> Option<(usize, Result<Reply>)> {
        let completion = self.pool().try_recv()?;
        let Some(ticket) = self.tickets.get(&completion.transaction) else {
            self.lose_proof();
        };
        if completion.rank != ticket.slot || completion.session != ticket.route {
            self.lose_proof();
        }
        if matches!(completion.outcome, CompletionOutcome::QuiescenceUnknown) {
            self.lose_proof();
        }
        let ticket = self.tickets.remove(&completion.transaction).unwrap();
        let reply = match completion.outcome {
            CompletionOutcome::Success(reply) => Ok(reply),
            CompletionOutcome::Failed(source) => Err(source),
            CompletionOutcome::Cancelled => Err(error("cancelled expert work")),
            CompletionOutcome::QuiescenceUnknown => self.lose_proof(),
            other => {
                self.closed = true;
                Err(error(format!("expert owner unavailable: {other:?}")))
            }
        };
        Some((ticket.index, reply))
    }

    fn call_many(
        &mut self,
        commands: Vec<(ParallelRankId, Command)>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<Reply>> {
        if self.closed || !self.tickets.is_empty() {
            return Err(error("expert workers closed or busy"));
        }
        check()?;
        let deadline = Instant::now() + self.timeout;
        let mut replies = (0..commands.len()).map(|_| None).collect::<Vec<_>>();
        let mut rejected = None;
        // No waits between admissions: every legal owner can execute at once.
        for (index, (owner, command)) in commands.into_iter().enumerate() {
            let admission = check().and_then(|()| {
                if Instant::now() >= deadline {
                    Err(error("expert admission deadline expired"))
                } else {
                    self.admit(owner, command, index)
                }
            });
            if let Err(source) = admission {
                rejected = Some(source);
                self.cancel_all();
                break;
            }
        }
        while !self.tickets.is_empty() {
            if rejected.is_none()
                && let Err(source) = check()
            {
                rejected = Some(source);
                self.cancel_all();
            }
            if let Some((index, result)) = self.receive() {
                match result {
                    Ok(reply) => replies[index] = Some(reply),
                    Err(source) => {
                        if rejected.is_none() {
                            rejected = Some(source);
                        }
                        self.cancel_all();
                    }
                }
            } else {
                if Instant::now() >= deadline {
                    self.lose_proof();
                }
                std::thread::sleep(Duration::from_micros(50));
            }
        }
        if let Some(source) = rejected {
            return Err(source);
        }
        check()?;
        replies
            .into_iter()
            .map(|reply| reply.ok_or_else(|| error("missing expert reply")))
            .collect()
    }

    fn call(
        &mut self,
        owner: ParallelRankId,
        command: Command,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Reply> {
        self.call_many(vec![(owner, command)], check)
            .map(|mut replies| replies.remove(0))
    }

    /// Validate the entire outer plan before admitting its first nonempty owner.
    pub fn execute_batch(
        &mut self,
        context: ExpertDispatchContext,
        buckets: Vec<ExpertTokenBucket>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<Vec<(ParallelRankId, Vec<ExpertResult>)>> {
        if self.closed {
            return Err(error("expert workers closed or unavailable"));
        }
        check_active(context.transaction)?;
        self.group.validate_context(context)?;
        let mut seen = BTreeSet::new();
        let mut identities = BTreeSet::new();
        let mut tokens = 0usize;
        let mut bytes = 0usize;
        for bucket in &buckets {
            self.group.validate_bucket(context, bucket)?;
            if !seen.insert(bucket.owner_rank) {
                return Err(error("duplicate batch owner"));
            }
            tokens = tokens
                .checked_add(bucket.tokens.len())
                .ok_or_else(|| error("token overflow"))?;
            for token in &bucket.tokens {
                if !identities.insert((token.source_row, token.route_slot)) {
                    return Err(error("duplicate dispatch route identity"));
                }
                bytes = token
                    .payload
                    .len()
                    .checked_mul(4)
                    .and_then(|n| bytes.checked_add(n))
                    .ok_or_else(|| error("activation byte overflow"))?;
            }
        }
        if tokens > self.group.limits.max_tokens || bytes > self.group.limits.max_bytes {
            return Err(error("outer dispatch exceeds limits"));
        }
        let owners = buckets.iter().map(|b| b.owner_rank).collect::<Vec<_>>();
        let mut nonempty = Vec::new();
        let mut output = owners
            .into_iter()
            .map(|owner| (owner, Vec::new()))
            .collect::<Vec<_>>();
        let mut commands = Vec::new();
        for (index, bucket) in buckets.into_iter().enumerate() {
            if !bucket.tokens.is_empty() {
                nonempty.push(index);
                commands.push((bucket.owner_rank, Command::Compute { context, bucket }));
            }
        }
        let replies = self.call_many(commands, &mut || check_active(context.transaction))?;
        for (index, reply) in nonempty.into_iter().zip(replies) {
            let Reply::Results(results) = reply else {
                return Err(error("unexpected expert reply"));
            };
            output[index].1 = results;
        }
        Ok(output)
    }

    /// Low-level activation/result call with caller authority. The model routed
    /// executor owns route-set validation/combine; this boundary checks context,
    /// ownership and admission size. Even empty buckets must name a live caller
    /// transaction and a known group member, but never invoke a provider.
    pub fn execute_bucket(
        &mut self,
        context: ExpertDispatchContext,
        bucket: ExpertTokenBucket,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<Vec<ExpertResult>> {
        if self.closed {
            return Err(error("expert workers closed or unavailable"));
        }
        check_active(context.transaction)?;
        self.group.validate_bucket(context, &bucket)?;
        if bucket.tokens.is_empty() {
            return Ok(Vec::new());
        }
        match self.call(
            bucket.owner_rank,
            Command::Compute { context, bucket },
            &mut || check_active(context.transaction),
        )? {
            Reply::Results(results) => Ok(results),
            _ => Err(error("unexpected expert reply")),
        }
    }

    pub fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        let members = self.group.members.clone();
        members
            .into_iter()
            .map(
                |owner| match self.call(owner, Command::Stats, &mut || Ok(()))? {
                    Reply::Stats(stats) => Ok(stats),
                    _ => Err(error("missing expert statistics")),
                },
            )
            .collect()
    }

    pub fn shutdown(&mut self) -> Result<()> {
        self.closed = true;
        if self.unknown {
            self.lose_proof();
        }
        self.pool().begin_shutdown();
        let deadline = Instant::now() + self.timeout;
        while !self.pool().shutdown_ready() {
            if Instant::now() >= deadline {
                self.lose_proof();
            }
            std::thread::sleep(Duration::from_micros(50));
        }
        let result = self.pool().shutdown();
        if let Err(failure) = &result {
            if failure
                .failures
                .iter()
                .any(|f| matches!(f.kind, OwnerFailureKind::QuiescenceUnknown))
            {
                self.lose_proof();
            }
        }
        result.map_err(|e| error(format!("owner shutdown: {e:?}")))
    }
}

/// Production shared-layer injection: only dispatch transport is runtime-owned;
/// all router ordering, result validation and weighting remain in model.
pub struct ExpertParallelExecutor {
    workers: ExpertRankWorkers,
    #[cfg(feature = "cuda")]
    cuda: bool,
}

impl ExpertParallelExecutor {
    pub fn new<F>(group: ExpertGroup, factory: F) -> Result<Self>
    where
        F: FnOnce(ExpertRank, &ExpertGroup) -> Result<ExpertRankWorker> + Clone + Send + 'static,
    {
        Ok(Self {
            workers: ExpertRankWorkers::new(group, factory)?,
            #[cfg(feature = "cuda")]
            cuda: false,
        })
    }
    /// Reject a CPU worker at initialization, rather than allowing an implicit
    /// host fallback behind the activation/result seam.
    #[cfg(feature = "cuda")]
    pub fn new_cuda<F>(group: ExpertGroup, factory: F) -> Result<Self>
    where
        F: FnOnce(ExpertRank, &ExpertGroup) -> Result<ExpertRankWorker> + Clone + Send + 'static,
    {
        let workers = ExpertRankWorkers::new(group, move |rank, group| {
            let worker = factory(rank, group)?;
            if worker.cuda.is_none() {
                return Err(error(
                    "CUDA expert executor requires CUDA owners; CPU fallback is disabled",
                ));
            }
            Ok(worker)
        })?;
        Ok(Self {
            workers,
            cuda: true,
        })
    }

    /// Resources/cache cross threads only as immutable Send metadata/host
    /// payloads. Each CUDA context is constructed inside its persistent owner.
    /// Returning proves every owner's eager uploads completed (DP Ready barrier).
    #[cfg(feature = "cuda")]
    pub fn new_numeric_f32(
        group: ExpertGroup,
        resources: Arc<BoundDecoderResources>,
        host: ArcHostExpertCache,
        devices: Vec<usize>,
        config: NumericExpertConfig,
    ) -> Result<Self> {
        group.validate_resources(&resources)?;
        if devices.len() != group.members.len()
            || config.dispatch_timeout.is_zero()
            || Instant::now()
                .checked_add(config.dispatch_timeout)
                .is_none()
        {
            return Err(error(
                "numeric EP requires explicit devices and finite timeout",
            ));
        }
        let mut executor = Self::new_cuda(group, move |rank, group| {
            let ops = std::rc::Rc::new(
                ferrule_backend::cuda::operators::linear::CudaOperators::new_on_device(
                    devices[rank.slot.get() as usize],
                )?,
            );
            ExpertRankWorker::prepare_cuda_numeric_f32(&resources, rank, group, &host, config, ops)
        })?;
        executor.workers.set_timeout(config.dispatch_timeout)?;
        Ok(executor)
    }

    /// Root GPU combine only: no second model weight directory, context or
    /// mutable borrow of the hybrid module's operators.
    #[cfg(feature = "cuda")]
    pub fn into_hybrid_cuda(
        self,
        ops: std::rc::Rc<ferrule_backend::cuda::operators::linear::CudaOperators>,
    ) -> Result<HybridCudaExpertAdapter> {
        if !self.cuda {
            return Err(error("hybrid attachment requires CUDA owners"));
        }
        let combine = ferrule_model::transformer::CudaStandardDecoderOperators::new(
            ops,
            ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
            &[],
        )?;
        Ok(HybridCudaExpertAdapter {
            executor: self,
            combine,
            shutdown_started: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn is_cuda(&self) -> bool {
        self.cuda
    }

    #[cfg(feature = "cuda")]
    pub fn routed_swiglu_cuda(
        &mut self,
        operators: &mut ferrule_model::transformer::CudaStandardDecoderOperators,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        if !self.cuda {
            return Err(error("CUDA routed execution requires CUDA expert owners"));
        }
        let group = Arc::clone(&self.workers.group);
        group.validate_context(request.context)?;
        let check = RefCell::new(check_active);
        let mut results = ActiveResults {
            workers: &mut self.workers,
            context: request.context,
            check: &check,
        };
        let result = ferrule_model::transformer::CudaExpertParallelRoutedExecutor::new(
            operators,
            group.members.clone(),
            &group.placement,
            group.limits,
            &mut results,
        )
        .routed_swiglu(request, &mut |id| check.borrow_mut()(id));
        if operators.needs_quarantine() {
            std::mem::forget(result);
            std::panic::panic_any(PanicQuiescence::Unknown);
        }
        result
    }

    pub fn group(&self) -> &ExpertGroup {
        self.workers.group()
    }
    pub fn outstanding(&self) -> usize {
        self.workers.outstanding()
    }
    pub fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        self.workers.owner_stats()
    }
    pub fn shutdown(&mut self) -> Result<()> {
        self.workers.shutdown()
    }
}

impl RoutedSwiGluExecutor for ExpertParallelExecutor {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        #[cfg(feature = "cuda")]
        if self.cuda {
            return Err(error("CUDA experts require the GPU result-combine seam"));
        }
        let group = Arc::clone(&self.workers.group);
        group.validate_context(request.context)?;
        // These borrows are sequential: model checks and transport polling both
        // consult the SAME caller authority, without an EP transaction registry.
        let check = RefCell::new(check_active);
        let mut results = ActiveResults {
            workers: &mut self.workers,
            context: request.context,
            check: &check,
        };
        ExpertParallelRoutedExecutor::new(
            group.members.clone(),
            &group.placement,
            group.limits,
            &mut results,
        )
        .routed_swiglu(request, &mut |id| check.borrow_mut()(id))
    }
}

struct ActiveResults<'a, 'b> {
    workers: &'a mut ExpertRankWorkers,
    context: ExpertDispatchContext,
    check: &'a RefCell<&'b mut dyn FnMut(ExecutionTransactionId) -> Result<()>>,
}

impl ExpertResultExecutor for ActiveResults<'_, '_> {
    fn supports_batch(&self) -> bool {
        true
    }

    fn execute_batch(
        &mut self,
        buckets: Vec<ExpertTokenBucket>,
    ) -> Result<Vec<(ParallelRankId, Vec<ExpertResult>)>> {
        self.workers
            .execute_batch(self.context, buckets, &mut |id| self.check.borrow_mut()(id))
    }

    fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
        self.workers
            .execute_bucket(self.context, bucket, &mut |id| self.check.borrow_mut()(id))
    }
}

/// Owner-local adapter around the same EP lifecycle/tickets, not another pool
/// or logical transaction manager. Unknown is a permanent custody state.
#[cfg(feature = "cuda")]
pub struct HybridCudaExpertAdapter {
    executor: ExpertParallelExecutor,
    combine: ferrule_model::transformer::CudaStandardDecoderOperators,
    shutdown_started: Option<Instant>,
}

#[cfg(feature = "cuda")]
impl HybridCudaExpertAdapter {
    /// Query the actual owners after drain completes, without changing expert
    /// weights or compute counters. Uses the same bounded transport tickets;
    /// never fabricates startup counters for a closed/unknown owner.
    pub fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        use ferrule_model::decoder::{HybridCudaExpertProgress, HybridCudaRoutedExecutor};
        if self.drain()? != HybridCudaExpertProgress::Complete || self.shutdown_started.is_some() {
            return Err(error(
                "owner statistics require drained, open expert owners",
            ));
        }
        self.executor.owner_stats()
    }
}

#[cfg(feature = "cuda")]
impl RoutedSwiGluExecutor for HybridCudaExpertAdapter {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        if self.shutdown_started.is_some() || self.executor.workers.unknown {
            return Err(error("hybrid expert attachment closed or unknown"));
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.executor
                .routed_swiglu_cuda(&mut self.combine, request, check_active)
        })) {
            Ok(result) => result,
            Err(payload) => {
                self.executor.workers.unknown = true;
                self.executor.workers.closed = true;
                self.executor.workers.cancel_all();
                self.executor.workers.pool().begin_shutdown();
                std::mem::forget(payload);
                Err(error("expert completion unknown; retaining owner custody"))
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl ferrule_model::decoder::HybridCudaRoutedExecutor for HybridCudaExpertAdapter {
    fn drain(&mut self) -> Result<ferrule_model::decoder::HybridCudaExpertProgress> {
        use ferrule_model::decoder::HybridCudaExpertProgress as Progress;
        if self.executor.workers.unknown || self.combine.needs_quarantine() {
            self.executor.workers.unknown = true;
            return Ok(Progress::Unknown);
        }
        // Synchronous successful/failed dispatch drains its admitted tickets.
        // The only escaped in-flight path is caught above and permanently unknown.
        Ok(if self.executor.outstanding() == 0 {
            Progress::Complete
        } else {
            Progress::Pending
        })
    }

    fn shutdown(&mut self) -> Result<ferrule_model::decoder::HybridCudaExpertProgress> {
        use ferrule_model::decoder::HybridCudaExpertProgress as Progress;
        if self.drain()? != Progress::Complete {
            return self.drain();
        }
        let workers = &mut self.executor.workers;
        let started = *self.shutdown_started.get_or_insert_with(Instant::now);
        workers.closed = true;
        workers.pool().begin_shutdown();
        if !workers.pool().shutdown_ready() {
            if started.elapsed() >= workers.timeout {
                workers.unknown = true;
                return Ok(Progress::Unknown);
            }
            return Ok(Progress::Pending);
        }
        if let Err(failure) = workers.pool().shutdown() {
            // A shutdown failure may hide device custody. Never turn a consumed
            // one-shot DP error into Complete on the next adapter poll.
            workers.unknown = true;
            tracing::error!(?failure, "expert owner shutdown unknown");
            return Ok(Progress::Unknown);
        }
        Ok(Progress::Complete)
    }

    fn on_error(&mut self, transaction: ExecutionTransactionId) {
        self.executor.workers.cancel_transaction(transaction);
    }
}

#[cfg(all(test, unix))]
#[path = "expert_tests.rs"]
mod tests;
