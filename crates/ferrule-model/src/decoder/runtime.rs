use super::{
    DecoderCancelProgress, DecoderKvBackend, DecoderLogits, DecoderWait,
    GenericDecoderObservabilitySnapshot, PackedDecoderBatch,
};
use crate::execution::{
    ResolvedStage, SequenceStateCore, SequenceTopologyId, StageMaterializationRequest,
    WorkspaceClaim,
};
use crate::materialization::{
    MaterializationPlacement, MaterializationProvider, MaterializationRequest,
    MaterializationResolver, resolve_stage_resources,
};
use crate::runner::{
    ModelCompletionReactor, NativeProposal, NativeProposalSource, SequenceStateReleaseError,
};
use ferrule_common::execution::{ExecutionBatch, ExecutionTransactionId};
use ferrule_common::expert_residency::{ExpertResidencyControl, ExpertResidencyRequirements};
use ferrule_common::{
    CompletionHub, ContinuationId, DependencySet, Error, ResidencyLeaseSet, Result,
};
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
/// Public identity/core projection required by the model-neutral transaction shell.
///
/// Model sequence attachments do not need to be `Clone`. Their fallible working-copy,
/// fork, reset, and release behavior is supplied separately by
/// [`DecoderSequenceLifecycle`].
pub trait DecoderSequence {
    fn topology_id(&self) -> SequenceTopologyId;
    fn core(&self) -> &SequenceStateCore;
    fn core_mut(&mut self) -> &mut SequenceStateCore;
}
/// Context supplied when a committed sequence is checked out into a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderSequenceCheckout {
    transaction: ExecutionTransactionId,
    sequence_index: usize,
    source_index: usize,
}
impl DecoderSequenceCheckout {
    pub const fn new(
        transaction: ExecutionTransactionId,
        sequence_index: usize,
        source_index: usize,
    ) -> Self {
        Self {
            transaction,
            sequence_index,
            source_index,
        }
    }
    pub const fn transaction(self) -> ExecutionTransactionId {
        self.transaction
    }
    pub const fn sequence_index(self) -> usize {
        self.sequence_index
    }
    pub const fn source_index(self) -> usize {
        self.source_index
    }
}
/// Complete fallible lifecycle for model-owned sequence state.
///
/// The transaction shell never clones sequence state. A model may deep-copy,
/// checkpoint, borrow from a pool, or otherwise construct a working topology in
/// `checkout`. Failed release returns the exact original state to its caller.
pub trait DecoderSequenceLifecycle<S> {
    fn create(&mut self) -> Result<S>;
    fn checkout(&mut self, request: DecoderSequenceCheckout, source: &S) -> Result<S>;
    fn logical_fork(&mut self, source: &S, expected_position: usize) -> Result<S>;
    fn reset(&mut self, state: &mut S) -> Result<()>;
    fn try_release(&mut self, state: S) -> std::result::Result<(), SequenceStateReleaseError<S>>;
}
#[derive(Debug, Clone)]
pub struct DecoderContinuationAllocator {
    next: Arc<AtomicU64>,
}
impl Default for DecoderContinuationAllocator {
    fn default() -> Self {
        Self::new()
    }
}
impl DecoderContinuationAllocator {
    pub fn new() -> Self {
        Self {
            next: Arc::new(AtomicU64::new(1)),
        }
    }
    pub fn allocate(&self) -> Result<ContinuationId> {
        let value = self
            .next
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| Error::Execution {
                message: "decoder continuation ID space is exhausted".into(),
            })?;
        Ok(ContinuationId::new(value))
    }
}
#[derive(Debug, Clone)]
pub struct DecoderControlPlane {
    completion_hub: CompletionHub,
    continuations: DecoderContinuationAllocator,
}
impl DecoderControlPlane {
    pub fn new(completion_hub: CompletionHub) -> Self {
        Self {
            completion_hub,
            continuations: DecoderContinuationAllocator::new(),
        }
    }
    pub const fn completion_hub(&self) -> &CompletionHub {
        &self.completion_hub
    }
    pub const fn continuations(&self) -> &DecoderContinuationAllocator {
        &self.continuations
    }
}
/// Generic-decoder ownership of the runtime-installed materialization resolver.
///
/// Every transaction context shares this one resolver cell. Model resource managers
/// own providers and placement metadata, but never resolver storage or forwarding
/// adapters.
#[derive(Clone, Default)]
pub struct DecoderResolverHandle {
    resolver: Arc<Mutex<Option<Box<dyn MaterializationResolver>>>>,
    placement: Arc<OnceLock<MaterializationPlacement>>,
}
impl std::fmt::Debug for DecoderResolverHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecoderResolverHandle")
            .field("installed", &self.installed())
            .field("placement", &self.placement.get())
            .finish()
    }
}
impl DecoderResolverHandle {
    pub fn installed(&self) -> bool {
        self.resolver
            .lock()
            .is_ok_and(|resolver| resolver.is_some())
    }
    pub fn install(
        &self,
        resolver: Box<dyn MaterializationResolver>,
        expected: Option<MaterializationPlacement>,
    ) -> Result<()> {
        let placement = resolver.placement();
        if expected.is_some_and(|expected| expected != placement) {
            return Err(Error::Execution {
                message: "decoder materialization resolver placement mismatch".into(),
            });
        }
        let mut installed = self.resolver.lock().map_err(|_| Error::Internal {
            message: "decoder materialization resolver lock is poisoned".into(),
        })?;
        if installed.is_some() {
            return Err(Error::Execution {
                message: "decoder materialization resolver is already installed".into(),
            });
        }
        if self
            .placement
            .get()
            .is_some_and(|installed| *installed != placement)
        {
            return Err(Error::Execution {
                message: "decoder materialization resolver placement changed".into(),
            });
        }
        let _ = self.placement.set(placement);
        *installed = Some(resolver);
        Ok(())
    }
    pub fn clear(&self) -> Result<()> {
        self.resolver
            .lock()
            .map_err(|_| Error::Internal {
                message: "decoder materialization resolver lock is poisoned".into(),
            })?
            .take();
        Ok(())
    }
    pub fn resolve_stage(
        &self,
        resources: Vec<StageMaterializationRequest>,
        workspace: WorkspaceClaim,
    ) -> Result<ResolvedStage> {
        let mut installed = self.resolver.lock().map_err(|_| Error::Internal {
            message: "decoder materialization resolver lock is poisoned".into(),
        })?;
        let resolver = installed.as_deref_mut().ok_or_else(|| Error::Execution {
            message: "decoder operation has no resource resolver".into(),
        })?;
        resolve_stage_resources(resources, workspace, resolver)
    }
}
impl MaterializationResolver for DecoderResolverHandle {
    fn placement(&self) -> MaterializationPlacement {
        *self
            .placement
            .get()
            .expect("installed decoder resolver retains its placement")
    }
    fn resolve(
        &mut self,
        request: MaterializationRequest,
    ) -> ferrule_common::MaterializationResolveResult<ferrule_common::MaterializationKey> {
        let Ok(mut installed) = self.resolver.lock() else {
            return Err(ferrule_common::MaterializationResolveError::Provider {
                purpose: ferrule_common::MaterializationPurpose::Execution,
                source: ferrule_common::FailureReason::ContractViolation {
                    message: "decoder materialization resolver lock is poisoned".into(),
                },
            });
        };
        installed.as_deref_mut().map_or_else(
            || {
                Err(
                    ferrule_common::MaterializationResolveError::ProviderUnavailable {
                        purpose: ferrule_common::MaterializationPurpose::Execution,
                        model: request.model(),
                        backend: request.backend(),
                        device: request.device(),
                    },
                )
            },
            |resolver| resolver.resolve(request),
        )
    }
}
/// Per-operation forward/proposal context. Resolver and residency-lease custody
/// stay in the generic decoder; model code receives only narrow resolution and
/// immutable current-resume access.
pub struct DecoderTransactionContext {
    transaction: ExecutionTransactionId,
    continuation_id: Option<ContinuationId>,
    control: DecoderControlPlane,
    resolver: DecoderResolverHandle,
    leases: Vec<ResidencyLeaseSet>,
    has_current_resume_lease: bool,
}
impl std::fmt::Debug for DecoderTransactionContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecoderTransactionContext")
            .field("transaction", &self.transaction)
            .field("continuation_id", &self.continuation_id)
            .field("resolver_available", &self.resolver.installed())
            .field("leases", &self.leases.len())
            .field("has_current_resume_lease", &self.has_current_resume_lease)
            .finish()
    }
}
impl DecoderTransactionContext {
    pub fn new(
        transaction: ExecutionTransactionId,
        control: DecoderControlPlane,
        leases: Vec<ResidencyLeaseSet>,
    ) -> Self {
        Self::with_services(
            transaction,
            control,
            leases,
            false,
            None,
            DecoderResolverHandle::default(),
        )
    }
    pub(crate) fn with_services(
        transaction: ExecutionTransactionId,
        control: DecoderControlPlane,
        leases: Vec<ResidencyLeaseSet>,
        has_current_resume_lease: bool,
        continuation_id: Option<ContinuationId>,
        resolver: DecoderResolverHandle,
    ) -> Self {
        debug_assert!(!has_current_resume_lease || !leases.is_empty());
        Self {
            transaction,
            continuation_id,
            control,
            resolver,
            leases,
            has_current_resume_lease,
        }
    }
    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.transaction
    }
    pub const fn completion_hub(&self) -> &CompletionHub {
        self.control.completion_hub()
    }
    pub fn continuation_id(&self) -> Result<ContinuationId> {
        self.continuation_id.ok_or_else(|| Error::Internal {
            message: "decoder operation has no preallocated continuation ID".into(),
        })
    }
    pub fn allocate_continuation(&self) -> Result<ContinuationId> {
        match self.continuation_id {
            Some(continuation) => Ok(continuation),
            None => self.control.continuations().allocate(),
        }
    }
    pub fn materialization_placement(&self) -> Result<MaterializationPlacement> {
        self.resolver
            .placement
            .get()
            .copied()
            .ok_or_else(|| Error::Execution {
                message: "decoder operation has no resource resolver".into(),
            })
    }
    pub fn resolve_stage(
        &self,
        resources: Vec<StageMaterializationRequest>,
        workspace: WorkspaceClaim,
    ) -> Result<ResolvedStage> {
        self.resolver.resolve_stage(resources, workspace)
    }
    /// Resolves a materialized stage, or creates the generic completion edge used
    /// while physical backend progress has no resource dependency yet.
    pub fn resolve_stage_wait(
        &self,
        resources: Vec<StageMaterializationRequest>,
        workspace: WorkspaceClaim,
    ) -> Result<DecoderWait> {
        if resources.is_empty() {
            let continuation = self.continuation_id()?;
            let dependency = ferrule_common::LogicalDependency::operation_retired(
                ferrule_common::OperationId::new(continuation.get()),
            )?;
            return Ok(DecoderWait::Dependencies(DependencySet::new([dependency])?));
        }
        self.resolve_stage(resources, workspace)
            .map(DecoderWait::ResolvedStage)
    }
    /// Borrows the lease delivered for this exact resume edge. Historical leases
    /// remain generic-decoder custody and are not exposed as resumable input.
    pub fn resume_lease(&self) -> Result<&ResidencyLeaseSet> {
        if !self.has_current_resume_lease {
            return Err(Error::Execution {
                message: "decoder operation has no current resume lease".into(),
            });
        }
        self.leases.last().ok_or_else(|| Error::Internal {
            message: "decoder current resume lease custody is missing".into(),
        })
    }
    pub(crate) fn into_leases(self) -> Vec<ResidencyLeaseSet> {
        self.leases
    }
}
/// Custody retained after forward completion until backend publish or abort ends.
pub trait DecoderTerminalGuard: Debug {}
/// Empty guard used by synchronous standard executors.
#[derive(Debug, Default)]
pub struct NoTerminalGuard;
impl DecoderTerminalGuard for NoTerminalGuard {}
/// Initial progress from the forward-execution boundary.
#[derive(Debug)]
pub enum DecoderForwardProgress<C, G> {
    Waiting {
        continuation: C,
        wait: DecoderWait,
    },
    Complete {
        logits: DecoderLogits,
        terminal_guard: G,
    },
    FailedActive {
        continuation: C,
        error: Error,
    },
    FailedQuiescent(Error),
}
/// Resume progress while continuation custody stays in the transaction shell.
#[derive(Debug)]
pub enum DecoderForwardResumeProgress<G> {
    Waiting(DecoderWait),
    Complete {
        logits: DecoderLogits,
        terminal_guard: G,
    },
    FailedActive(Error),
    FailedQuiescent(Error),
}
/// Typed asynchronous forward executor consumed by the one decoder transaction shell.
pub trait DecoderForwardExecutor<S, KvView> {
    type Continuation;
    type TerminalGuard: DecoderTerminalGuard;
    fn start_forward(
        &mut self,
        context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [S],
        kv: &mut KvView,
    ) -> Result<DecoderForwardProgress<Self::Continuation, Self::TerminalGuard>>;
    fn resume_forward(
        &mut self,
        context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [S],
        kv: &mut KvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderForwardResumeProgress<Self::TerminalGuard>>;
    fn cancel_forward(
        &mut self,
        context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [S],
        kv: &mut KvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderCancelProgress>;
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}
/// Optional model-native proposal attachment consumed by `ResidentModelRunner`.
pub trait DecoderProposalExecutor<S, K>
where
    K: DecoderKvBackend<SequenceState = S>,
{
    type Continuation;
    fn source(&self) -> Result<Option<NativeProposalSource>>;
    fn start(
        &mut self,
        context: &mut DecoderTransactionContext,
        state: &mut S,
        kv: &mut K::KvView,
        anchor_token_id: u32,
    ) -> Result<DecoderProposalProgress<Self::Continuation>>;
    fn resume(
        &mut self,
        context: &mut DecoderTransactionContext,
        state: &mut S,
        kv: &mut K::KvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderProposalResumeProgress>;
    fn cancel(
        &mut self,
        context: &mut DecoderTransactionContext,
        state: &mut S,
        kv: &mut K::KvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderCancelProgress>;
    /// Atomically restores model working states and the backend checkpoint to the
    /// same retained prefix. An error must occur before either side is mutated.
    fn retain_provisional(
        &mut self,
        context: &mut DecoderProvisionalRetainContext<'_, S, K>,
    ) -> Result<()>;
}
#[derive(Debug)]
pub enum DecoderProposalProgress<C> {
    Waiting { continuation: C, wait: DecoderWait },
    Complete(NativeProposal),
    FailedActive { continuation: C, error: Error },
    FailedQuiescent(Error),
}
#[derive(Debug)]
pub enum DecoderProposalResumeProgress {
    Waiting(DecoderWait),
    Complete(NativeProposal),
    FailedActive(Error),
    FailedQuiescent(Error),
}
/// Borrowed atomic retain boundary joining model state and backend checkpoint.
pub struct DecoderProvisionalRetainContext<'a, S, K>
where
    K: DecoderKvBackend<SequenceState = S>,
{
    transaction: ExecutionTransactionId,
    sources: &'a [S],
    working_states: &'a mut [S],
    batch: &'a PackedDecoderBatch,
    executed_rows: &'a [usize],
    retained_rows: &'a [usize],
    backend: &'a mut K,
    kv_transaction: &'a mut K::Transaction,
}
impl<'a, S, K> DecoderProvisionalRetainContext<'a, S, K>
where
    K: DecoderKvBackend<SequenceState = S>,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        transaction: ExecutionTransactionId,
        sources: &'a [S],
        working_states: &'a mut [S],
        batch: &'a PackedDecoderBatch,
        executed_rows: &'a [usize],
        retained_rows: &'a [usize],
        backend: &'a mut K,
        kv_transaction: &'a mut K::Transaction,
    ) -> Self {
        Self {
            transaction,
            sources,
            working_states,
            batch,
            executed_rows,
            retained_rows,
            backend,
            kv_transaction,
        }
    }
    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.transaction
    }
    pub const fn sources(&self) -> &[S] {
        self.sources
    }
    pub fn working_states(&mut self) -> &mut [S] {
        self.working_states
    }
    pub const fn batch(&self) -> &PackedDecoderBatch {
        self.batch
    }
    pub const fn executed_rows(&self) -> &[usize] {
        self.executed_rows
    }
    pub const fn retained_rows(&self) -> &[usize] {
        self.retained_rows
    }
    pub fn backend_parts(&mut self) -> (&mut K, &mut K::Transaction) {
        (self.backend, self.kv_transaction)
    }
    pub fn with_retain_parts<R>(
        &mut self,
        operation: impl FnOnce(&[S], &mut [S], &[usize], &[usize], &mut K, &mut K::Transaction) -> R,
    ) -> R {
        operation(
            self.sources,
            self.working_states,
            self.executed_rows,
            self.retained_rows,
            self.backend,
            self.kv_transaction,
        )
    }
}
/// Explicit no-proposal implementation for ordinary decoders.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoProposal;
impl<S, K> DecoderProposalExecutor<S, K> for NoProposal
where
    K: DecoderKvBackend<SequenceState = S>,
{
    type Continuation = std::convert::Infallible;
    fn source(&self) -> Result<Option<NativeProposalSource>> {
        Ok(None)
    }
    fn start(
        &mut self,
        _context: &mut DecoderTransactionContext,
        _state: &mut S,
        _kv: &mut K::KvView,
        _anchor_token_id: u32,
    ) -> Result<DecoderProposalProgress<Self::Continuation>> {
        Err(Error::Execution {
            message: "decoder has no proposal executor".into(),
        })
    }
    fn resume(
        &mut self,
        _context: &mut DecoderTransactionContext,
        _state: &mut S,
        _kv: &mut K::KvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderProposalResumeProgress> {
        match *continuation {}
    }
    fn cancel(
        &mut self,
        _context: &mut DecoderTransactionContext,
        _state: &mut S,
        _kv: &mut K::KvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderCancelProgress> {
        match *continuation {}
    }
    fn retain_provisional(
        &mut self,
        _context: &mut DecoderProvisionalRetainContext<'_, S, K>,
    ) -> Result<()> {
        Err(Error::Execution {
            message: "decoder has no provisional proposal executor".into(),
        })
    }
}
/// Optional forwarding surface for decoder resource and residency ownership.
pub trait DecoderResourceManager {
    fn expert_residency_requirements(&self) -> Option<ExpertResidencyRequirements> {
        None
    }
    fn expert_residency_control_installed(&self) -> bool {
        false
    }
    fn install_expert_residency_control(
        &mut self,
        _control: Box<dyn ExpertResidencyControl>,
    ) -> Result<()> {
        if self.expert_residency_requirements().is_some() {
            return Err(Error::Execution {
                message: "decoder resource manager does not accept residency control".into(),
            });
        }
        Ok(())
    }
    fn take_provider(&mut self) -> Option<Box<dyn MaterializationProvider>> {
        None
    }
    fn take_warmup_requests(&mut self) -> Result<Vec<MaterializationRequest>> {
        Ok(Vec::new())
    }
    fn transaction_prefetch_requests(
        &self,
        _transaction: ExecutionTransactionId,
        _batch: &ExecutionBatch,
    ) -> Result<Vec<MaterializationRequest>> {
        Ok(Vec::new())
    }
    fn materialization_placement(&self) -> Option<MaterializationPlacement> {
        None
    }
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
    fn observer_snapshot(&self) -> DecoderResourceSnapshot {
        DecoderResourceSnapshot::default()
    }
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecoderResourceSnapshot {
    pub resolver_installed: bool,
    pub shutdown: bool,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoResourceManager;
impl DecoderResourceManager for NoResourceManager {}
/// Produces the typed snapshot exposed by `ResidentModelRunner`.
pub trait DecoderObserver<K> {
    type Snapshot: Clone;
    fn snapshot(
        &self,
        core: &GenericDecoderObservabilitySnapshot,
        backend: &K,
        resources: &DecoderResourceSnapshot,
    ) -> Self::Snapshot;
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StandardDecoderObserver;
impl<K> DecoderObserver<K> for StandardDecoderObserver
where
    K: DecoderKvBackend,
{
    type Snapshot = GenericDecoderObservabilitySnapshot;
    fn snapshot(
        &self,
        core: &GenericDecoderObservabilitySnapshot,
        backend: &K,
        _resources: &DecoderResourceSnapshot,
    ) -> Self::Snapshot {
        let mut snapshot = core.clone();
        let capacity = backend.capacity();
        snapshot.physical_kv_pages = capacity.physical_pages;
        snapshot.resident_kv_pages = capacity.resident_pages;
        snapshot.preempted_kv_pages = capacity.preempted_pages;
        snapshot.free_kv_pages = capacity.free_pages;
        snapshot
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedDecoderSnapshot<A, B> {
    pub primary: A,
    pub secondary: B,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ComposedDecoderObserver<A, B> {
    pub primary: A,
    pub secondary: B,
}
impl<K, A, B> DecoderObserver<K> for ComposedDecoderObserver<A, B>
where
    A: DecoderObserver<K>,
    B: DecoderObserver<K>,
{
    type Snapshot = ComposedDecoderSnapshot<A::Snapshot, B::Snapshot>;
    fn snapshot(
        &self,
        core: &GenericDecoderObservabilitySnapshot,
        backend: &K,
        resources: &DecoderResourceSnapshot,
    ) -> Self::Snapshot {
        ComposedDecoderSnapshot {
            primary: self.primary.snapshot(core, backend, resources),
            secondary: self.secondary.snapshot(core, backend, resources),
        }
    }
}
/// One composition spec naming every model-specific decoder attachment.
pub trait DecoderComposition: Sized {
    type Resources;
    type SequenceState: DecoderSequence;
    type KvBackend: DecoderKvBackend<SequenceState = Self::SequenceState>;
    type ForwardExecutor: DecoderForwardExecutor<
            Self::SequenceState,
            <Self::KvBackend as DecoderKvBackend>::KvView,
            Continuation = Self::Continuation,
            TerminalGuard = Self::TerminalGuard,
        >;
    type Proposal: DecoderProposalExecutor<
            Self::SequenceState,
            Self::KvBackend,
            Continuation = Self::ProposalContinuation,
        >;
    type SequenceLifecycle: DecoderSequenceLifecycle<Self::SequenceState>;
    type ResourceManager: DecoderResourceManager;
    type Observer: DecoderObserver<Self::KvBackend, Snapshot = Self::Snapshot>;
    type Snapshot: Clone;
    type Continuation;
    type ProposalContinuation;
    type TerminalGuard: DecoderTerminalGuard;
}
/// Complete construction payload for a custom decoder composition.
pub struct DecoderComponents<D>
where
    D: DecoderComposition,
{
    pub resources: D::Resources,
    pub model_info: crate::runner::ModelInfo,
    pub tokenizer: crate::tokenizer::TokenizerHandle,
    pub capabilities: ferrule_common::execution::ExecutionCapabilities,
    pub page_size: usize,
    pub prepared_plan_id: u64,
    pub forward_executor: D::ForwardExecutor,
    pub backend: D::KvBackend,
    pub default_state: D::SequenceState,
    pub sequence_lifecycle: D::SequenceLifecycle,
    pub proposal: D::Proposal,
    pub resource_manager: D::ResourceManager,
    pub observer: D::Observer,
    pub completion_hub: CompletionHub,
    pub completion_reactors: Vec<ModelCompletionReactor>,
}
