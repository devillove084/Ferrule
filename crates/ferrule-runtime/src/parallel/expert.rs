//! Persistent CPU/CUDA expert owners, with activation/result-only transport.
//!
//! Explicit groups are independent of common's PP/DP/TP mesh. `ExpertRank::slot`
//! addresses a private DP pool; `owner` and `ExpertGroup::source_rank` are global
//! expert identities, NOT PP slots, KV participants or device ordinals. Pipeline
//! integration requires disjoint owner IDs across groups and from KV ranks.
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
use std::time::Duration;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, ParallelRankId, Result};
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
    CompletionOutcome, DataParallelConfig, DataParallelExecutor, PanicQuiescence, ReplicaWorker,
    WorkRequest,
};
use crate::SessionId;

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
/// model dispatch order. The source must be a member (it may own no experts).
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
    fn validate(&self) -> Result<()> {
        if self.members.is_empty()
            || self.members.len() > u32::MAX as usize
            || self.layers.is_empty()
            || self.limits.max_tokens == 0
            || self.limits.max_bytes == 0
            || !self.members.contains(&self.source_rank)
            || self.members.iter().collect::<BTreeSet<_>>().len() != self.members.len()
        {
            return Err(error("invalid group, source membership or dispatch limits"));
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

    fn validate_bucket(
        &self,
        context: ExpertDispatchContext,
        bucket: &ExpertTokenBucket,
    ) -> Result<()> {
        if context.source_rank != self.source_rank || !self.layers.contains(&context.layer) {
            return Err(error("unknown expert source or layer"));
        }
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
}

#[derive(Debug, Clone)]
pub struct ExpertOwnerStats {
    pub rank: ExpertRank,
    pub thread: std::thread::ThreadId,
    pub owned_experts: Vec<ExpertId>,
    pub calls: usize,
    pub tokens: usize,
    pub last_context: Option<ExpertDispatchContext>,
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
        group.validate_resources(resources)?;
        if group.members.get(rank.slot.get() as usize) != Some(&rank.owner) {
            return Err(error("expert slot/owner mismatch"));
        }
        let materializer = StateDictMaterializer::new(max_parameter_bytes)?;
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
        };
        Ok(Self {
            rank,
            group: group.clone(),
            experts: LocalExperts(experts),
            stats,
            #[cfg(feature = "cuda")]
            cuda: None,
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

    fn compute(
        &mut self,
        tokens: &[ferrule_model::transformer::ExpertToken],
    ) -> Result<Vec<ExpertResult>> {
        #[cfg(feature = "cuda")]
        if let Some(operators) = self.cuda.as_mut() {
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
            Command::Stats => Ok(Reply::Stats(self.stats.clone())),
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
/// outstanding globally (and one credit per owner); routes are bounded by group
/// size. Payloads are bounded before admission, replies by the concrete CPU
/// worker's shape-preserving SwiGLU. No test-supplied worker loop is necessary.
pub struct ExpertRankWorkers {
    group: Arc<ExpertGroup>,
    pool: DataParallelExecutor<Command, Reply, Error>,
    next_serial: u64,
    closed: bool,
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
        .map_err(|e| error(format!("owner construction: {e:?}")))?;
        Ok(Self {
            group,
            pool,
            next_serial: 1,
            closed: false,
        })
    }

    pub fn group(&self) -> &ExpertGroup {
        &self.group
    }
    pub fn outstanding(&self) -> usize {
        self.pool.outstanding()
    }

    fn call(
        &mut self,
        owner: ParallelRankId,
        command: Command,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Reply> {
        if self.closed {
            return Err(error("expert workers closed or unavailable"));
        }
        check()?;
        let index = self
            .group
            .members
            .iter()
            .position(|&member| member == owner)
            .ok_or_else(|| error("unknown expert owner"))?;
        let slot = ParallelRankId::new(index as u32);
        let route = SessionId(index as u64 + 1);
        let serial = ExecutionTransactionId::new(self.next_serial)?;
        self.next_serial = self
            .next_serial
            .checked_add(1)
            .ok_or_else(|| error("command serial exhausted"))?;
        self.pool
            .try_submit_to(route, slot, serial, command)
            .map_err(|e| error(format!("owner admission: {:?}", e.kind)))?;
        let mut rejected = None;
        let completion = loop {
            if rejected.is_none()
                && let Err(source) = check()
            {
                rejected = Some(source);
                let _ = self.pool.cancel(serial);
            }
            if let Some(completion) = self.pool.try_recv() {
                break completion;
            }
            std::thread::sleep(Duration::from_micros(50));
        };
        if completion.transaction != serial
            || completion.rank != slot
            || completion.session != route
        {
            self.closed = true;
            std::panic::panic_any(PanicQuiescence::Unknown);
        }
        let reply = match completion.outcome {
            CompletionOutcome::Success(reply) => Ok(reply),
            CompletionOutcome::Failed(source) => Err(source),
            CompletionOutcome::Cancelled => Err(error("cancelled expert work")),
            CompletionOutcome::QuiescenceUnknown => {
                self.closed = true;
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
            other => {
                self.closed = true;
                Err(error(format!("expert owner unavailable: {other:?}")))
            }
        };
        if let Some(source) = rejected {
            return Err(source);
        }
        check()?;
        reply
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
        self.pool
            .shutdown()
            .map_err(|e| error(format!("owner shutdown: {e:?}")))
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
        if request.context.source_rank != group.source_rank
            || !group.layers.contains(&request.context.layer)
        {
            return Err(error("unknown expert source or layer"));
        }
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
        if request.context.source_rank != group.source_rank
            || !group.layers.contains(&request.context.layer)
        {
            return Err(error("unknown expert source or layer"));
        }
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
    fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
        self.workers
            .execute_bucket(self.context, bucket, &mut |id| self.check.borrow_mut()(id))
    }
}
