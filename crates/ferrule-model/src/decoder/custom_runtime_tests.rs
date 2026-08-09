use super::*;
use crate::execution::{
    ResourceAccess, ResourceRetention, SequenceStateCore, SequenceTopologyId,
    StageMaterializationRequest, StageResourceUse, WorkspaceClaim,
};
use crate::materialization::{
    MaterializationPlacement, MaterializationPreparation, MaterializationProvider,
    MaterializationRequest, MaterializationResident, MaterializationResolver,
    PhysicalMaterializationOperationReservation, PhysicalMaterializationTopology, ResourceSource,
};
use crate::runner::{
    ModelInfo, ModelRunner, MultiSessionBatchProgress, MultiSessionRunner, NativeProposal,
    NativeProposalProgress, NativeProposalSource, ResidentModelRunner, SequenceStateReleaseError,
    TransactionEndIntent, TransactionEndProgress,
};
use crate::spec::{AttentionKind, ModelFamily, WeightSource};
use crate::tokenizer::TokenizerHandle;
use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionSequence, ExecutionTransactionId, ForwardMode,
    ForwardPhase, KvBindingMode, KvBlockId, KvPageId, KvReservationView, KvWriteSlot,
    LogitsRequest, LogitsRowPolicy, StateSlot,
};
use ferrule_common::materialization_io::{
    MaterializationResourceLimits, MaterializationResourcePlan,
};
use ferrule_common::{
    BackendId, CancellationReason, CompletionEvent, CompletionHub, ContentHash, ContinuationId,
    DestinationGeneration, DestinationSlotId, DeviceId, DispatchFenceContract, Error,
    FailureReason, FenceId, LoadStage, MappingEpoch, MaterializationKey, MaterializationPurpose,
    MaterializationResolveResult, MaterializedResourceId, MaterializedResourceKind,
    ModelInstanceId, OperationId, PayloadEncodingId, ResidencyBinding, ResidencyLeaseSet, Result,
    SourceGeneration, SourceIdentityHash, ValidatedResidencyBinding,
};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex, MutexGuard};
const PAGE_SIZE: usize = 4;
const PHYSICAL_PAGES: usize = 8;
const PREPARED_PLAN_ID: u64 = 700;
#[derive(Debug, Default)]
struct CustomTrace {
    checkout_requests: Vec<(u64, usize, usize)>,
    released_topologies: Vec<u64>,
    reset_count: usize,
    typed_kv_views: usize,
    forward_starts: usize,
    forward_resumes: usize,
    forward_cancels: usize,
    forward_shutdown: bool,
    terminal_guard_drop_lease_sizes: Vec<usize>,
    proposal_starts: usize,
    proposal_resumes: usize,
    proposal_cancels: usize,
    proposal_kv_views: usize,
    proposal_ids: Vec<ContinuationId>,
    proposal_resume_lease_sizes: Vec<usize>,
    retained_working_rows: Vec<Vec<u32>>,
    provider_prepares: usize,
    resolver_calls: usize,
    warmup_requests_taken: usize,
    prefetches: Vec<(u64, usize)>,
    backend_shutdown: bool,
    resource_shutdown: bool,
    resource_observer_snapshots: usize,
}
type SharedTrace = Arc<Mutex<CustomTrace>>;
struct NeverControl;
impl ferrule_common::expert_residency::ExpertResidencyControl for NeverControl {
    fn requirements(&self) -> ferrule_common::expert_residency::ExpertResidencyRequirements {
        unreachable!()
    }
    fn binding(
        &self,
        _key: ferrule_common::expert_residency::ExpertKey,
    ) -> Result<Option<ferrule_common::expert_residency::ExpertSlotBinding>> {
        unreachable!()
    }
    fn acquire_selected(
        &mut self,
        _key: ferrule_common::expert_residency::ExpertKey,
    ) -> Result<Option<ferrule_common::expert_residency::ExpertResidencyGrant>> {
        unreachable!()
    }
    fn release(&mut self, _lease: ferrule_common::expert_residency::ExpertLease) -> Result<()> {
        unreachable!()
    }
    fn prepare_install(
        &mut self,
        _intent: ferrule_common::expert_residency::ExpertInstallIntent,
    ) -> Result<ferrule_common::expert_residency::ExpertInstallPrepareOutcome> {
        unreachable!()
    }
    fn promote_install(
        &mut self,
        _prepared: ferrule_common::expert_residency::PreparedExpertInstall,
    ) -> Result<ferrule_common::expert_residency::PreparedExpertInstall> {
        unreachable!()
    }
    fn activate_install(
        &mut self,
        _prepared: ferrule_common::expert_residency::PreparedExpertInstall,
    ) -> Result<ferrule_common::expert_residency::ExpertInstallActivationOutcome> {
        unreachable!()
    }
    fn publish_install(
        &mut self,
        _prepared: ferrule_common::expert_residency::PreparedExpertInstall,
    ) -> Result<ferrule_common::expert_residency::ExpertResidencyGrant> {
        unreachable!()
    }
    fn cancel_install(
        &mut self,
        _prepared: ferrule_common::expert_residency::PreparedExpertInstall,
    ) -> Result<()> {
        unreachable!()
    }
    fn stats(&self) -> ferrule_common::expert_residency::ExpertResidencyStats {
        unreachable!()
    }
}
fn trace(trace: &SharedTrace) -> MutexGuard<'_, CustomTrace> {
    trace.lock().expect("custom runtime trace mutex poisoned")
}
#[derive(Debug, PartialEq, Eq)]
struct CustomSequenceState {
    core: SequenceStateCore,
    topology_id: SequenceTopologyId,
    rows: Vec<u32>,
    release_blocked: bool,
}
impl CustomSequenceState {
    fn new() -> Self {
        Self {
            core: SequenceStateCore::new(),
            topology_id: SequenceTopologyId::take(),
            rows: Vec::new(),
            release_blocked: false,
        }
    }
    fn source_at(position: usize, rows: Vec<u32>) -> Self {
        Self {
            core: SequenceStateCore::with_position(position),
            topology_id: SequenceTopologyId::take(),
            rows,
            release_blocked: false,
        }
    }
}
impl DecoderSequence for CustomSequenceState {
    fn topology_id(&self) -> SequenceTopologyId {
        self.topology_id
    }
    fn core(&self) -> &SequenceStateCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut SequenceStateCore {
        &mut self.core
    }
}
struct CustomSequenceLifecycle {
    trace: SharedTrace,
}
impl DecoderSequenceLifecycle<CustomSequenceState> for CustomSequenceLifecycle {
    fn create(&mut self) -> Result<CustomSequenceState> {
        Ok(CustomSequenceState::new())
    }
    fn checkout(
        &mut self,
        request: DecoderSequenceCheckout,
        source: &CustomSequenceState,
    ) -> Result<CustomSequenceState> {
        source.core.begin_step()?;
        trace(&self.trace).checkout_requests.push((
            request.transaction().get(),
            request.sequence_index(),
            request.source_index(),
        ));
        Ok(CustomSequenceState {
            core: source.core.clone(),
            topology_id: source.topology_id,
            rows: source.rows.to_vec(),
            release_blocked: source.release_blocked,
        })
    }
    fn logical_fork(
        &mut self,
        source: &CustomSequenceState,
        expected_position: usize,
    ) -> Result<CustomSequenceState> {
        if source.core.position() != expected_position {
            return Err(execution_error(format!(
                "custom sequence fork expected position {expected_position}, got {}",
                source.core.position()
            )));
        }
        Ok(CustomSequenceState {
            core: source.core.forked()?,
            topology_id: SequenceTopologyId::take(),
            rows: source.rows.to_vec(),
            release_blocked: source.release_blocked,
        })
    }
    fn reset(&mut self, state: &mut CustomSequenceState) -> Result<()> {
        state.core.reset();
        state.rows.clear();
        trace(&self.trace).reset_count += 1;
        Ok(())
    }
    fn try_release(
        &mut self,
        state: CustomSequenceState,
    ) -> std::result::Result<(), SequenceStateReleaseError<CustomSequenceState>> {
        if state.release_blocked {
            return Err(SequenceStateReleaseError::new(
                execution_error("custom sequence release preflight is blocked"),
                state,
            ));
        }
        trace(&self.trace)
            .released_topologies
            .push(state.topology_id.get());
        Ok(())
    }
}
#[derive(Debug)]
struct CustomKvTransaction {
    id: ExecutionTransactionId,
    entered: bool,
    new_pages: Vec<KvPageId>,
    executed_rows: Vec<usize>,
    retained_rows: Option<Vec<usize>>,
}
#[derive(Debug)]
struct CustomKvView {
    transaction: ExecutionTransactionId,
    trace: SharedTrace,
}
impl CustomKvView {
    fn validate(
        &mut self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) -> Result<()> {
        if self.transaction != transaction || batch.is_empty() {
            return Err(execution_error("typed custom KV view identity mismatch"));
        }
        trace(&self.trace).typed_kv_views += 1;
        Ok(())
    }
}
struct CustomKvBackend {
    trace: SharedTrace,
    physical_pages: usize,
    pages: HashMap<KvPageId, DecoderKvPageStatus>,
    active: HashSet<ExecutionTransactionId>,
    pending_commits: HashSet<ExecutionTransactionId>,
    pending_rollbacks: HashSet<ExecutionTransactionId>,
    provisional_checkpoints: HashMap<ExecutionTransactionId, Vec<usize>>,
    committed_checkpoints: HashMap<ExecutionTransactionId, Vec<usize>>,
    fail_next_release: bool,
    fail_next_proposal_view: bool,
    shutdown: bool,
}
impl CustomKvBackend {
    fn new(trace: SharedTrace) -> Self {
        Self {
            trace,
            physical_pages: PHYSICAL_PAGES,
            pages: HashMap::new(),
            active: HashSet::new(),
            pending_commits: HashSet::new(),
            pending_rollbacks: HashSet::new(),
            provisional_checkpoints: HashMap::new(),
            committed_checkpoints: HashMap::new(),
            fail_next_release: false,
            fail_next_proposal_view: false,
            shutdown: false,
        }
    }
    fn make_commit_pending_once(&mut self, transaction: ExecutionTransactionId) {
        self.pending_commits.insert(transaction);
    }
    fn make_rollback_pending_once(&mut self, transaction: ExecutionTransactionId) {
        self.pending_rollbacks.insert(transaction);
    }
    fn fail_next_release(&mut self) {
        self.fail_next_release = true;
    }
    fn fail_next_proposal_view(&mut self) {
        self.fail_next_proposal_view = true;
    }
    fn validate_checkpoint(
        &self,
        transaction: &CustomKvTransaction,
        retained_rows: &[usize],
    ) -> Result<()> {
        if !self.active.contains(&transaction.id)
            || transaction.entered
            || retained_rows.len() != transaction.executed_rows.len()
            || retained_rows
                .iter()
                .zip(&transaction.executed_rows)
                .any(|(&retained, &executed)| retained == 0 || retained > executed)
        {
            return Err(execution_error(
                "custom backend cannot retain the requested provisional checkpoint",
            ));
        }
        Ok(())
    }
    fn apply_checkpoint(&mut self, transaction: &mut CustomKvTransaction, retained_rows: &[usize]) {
        transaction.retained_rows = Some(retained_rows.to_vec());
        self.provisional_checkpoints
            .insert(transaction.id, retained_rows.to_vec());
    }
    fn finish_transaction(&mut self, transaction: CustomKvTransaction, commit: bool) {
        let removed = self.active.remove(&transaction.id);
        debug_assert!(removed);
        if commit {
            for page in transaction.new_pages {
                self.pages.insert(page, DecoderKvPageStatus::Resident);
            }
            let checkpoint = transaction
                .retained_rows
                .unwrap_or(transaction.executed_rows);
            self.committed_checkpoints
                .insert(transaction.id, checkpoint);
        }
        self.provisional_checkpoints.remove(&transaction.id);
    }
}
impl DecoderKvBackend for CustomKvBackend {
    type SequenceState = CustomSequenceState;
    type Transaction = CustomKvTransaction;
    type KvView = CustomKvView;
    fn configure_capacity(&mut self, max_pages: usize) -> Result<()> {
        if max_pages == 0 || !self.active.is_empty() || !self.pages.is_empty() {
            return Err(execution_error(
                "custom KV capacity cannot be changed in its current state",
            ));
        }
        self.physical_pages = max_pages;
        Ok(())
    }
    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        if self.shutdown || request.capacity.physical_pages != self.physical_pages {
            return Err(execution_error("custom KV backend is not available"));
        }
        if !self.active.insert(request.transaction) {
            return Err(execution_error("custom KV transaction is already active"));
        }
        if request
            .new_pages
            .iter()
            .any(|page| self.page_status(*page) != DecoderKvPageStatus::Vacant)
            || request
                .page_statuses
                .iter()
                .any(|snapshot| snapshot.status != self.page_status(snapshot.page))
        {
            self.active.remove(&request.transaction);
            return Err(execution_error("custom KV prepare page snapshot is stale"));
        }
        Ok(CustomKvTransaction {
            id: request.transaction,
            entered: false,
            new_pages: request.new_pages.to_vec(),
            executed_rows: request
                .sequences
                .iter()
                .map(|sequence| sequence.query_len)
                .collect(),
            retained_rows: None,
        })
    }
    fn enter(
        &mut self,
        transaction: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        states: &mut [Self::SequenceState],
    ) -> Result<()> {
        if transaction.entered
            || !self.active.contains(&transaction.id)
            || batch.sequences().len() != states.len()
        {
            return Err(execution_error("custom KV transaction cannot enter"));
        }
        transaction.entered = true;
        Ok(())
    }
    fn active_view(&mut self, transaction: &mut Self::Transaction) -> Result<Self::KvView> {
        if !transaction.entered || !self.active.contains(&transaction.id) {
            return Err(execution_error("custom KV transaction is not entered"));
        }
        Ok(CustomKvView {
            transaction: transaction.id,
            trace: Arc::clone(&self.trace),
        })
    }
    fn proposal_view(
        &mut self,
        transaction: ExecutionTransactionId,
        _state: &mut Self::SequenceState,
    ) -> Result<Self::KvView> {
        if std::mem::take(&mut self.fail_next_proposal_view) {
            return Err(execution_error("injected custom proposal KV view failure"));
        }
        Ok(CustomKvView {
            transaction,
            trace: Arc::clone(&self.trace),
        })
    }
    fn leave(&mut self, transaction: &mut Self::Transaction) -> Result<()> {
        if !transaction.entered {
            return Err(execution_error("custom KV transaction already left"));
        }
        transaction.entered = false;
        Ok(())
    }
    fn commit(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        let kv_transaction = transaction
            .as_ref()
            .ok_or_else(|| execution_error("custom KV commit lost its transaction"))?;
        if kv_transaction.entered {
            return Err(execution_error("custom KV commit is still entered"));
        }
        if self.pending_commits.remove(&kv_transaction.id) {
            return Ok(KvEndProgress::Pending);
        }
        let kv_transaction = transaction
            .take()
            .expect("custom KV commit transaction validated above");
        self.finish_transaction(kv_transaction, true);
        Ok(KvEndProgress::Complete)
    }
    fn rollback(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        let kv_transaction = transaction
            .as_ref()
            .ok_or_else(|| execution_error("custom KV rollback lost its transaction"))?;
        if kv_transaction.entered {
            return Err(execution_error("custom KV rollback is still entered"));
        }
        if self.pending_rollbacks.remove(&kv_transaction.id) {
            return Ok(KvEndProgress::Pending);
        }
        let kv_transaction = transaction
            .take()
            .expect("custom KV rollback transaction validated above");
        self.finish_transaction(kv_transaction, false);
        Ok(KvEndProgress::Complete)
    }
    fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        if self.fail_next_release {
            self.fail_next_release = false;
            return Err(execution_error("injected custom KV release failure"));
        }
        if pages
            .iter()
            .any(|page| self.page_status(*page) == DecoderKvPageStatus::Vacant)
        {
            return Err(execution_error("custom KV release does not own every page"));
        }
        for page in pages {
            self.pages.remove(page);
        }
        Ok(())
    }
    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        if pages
            .iter()
            .any(|page| self.page_status(*page) != DecoderKvPageStatus::Resident)
        {
            return Err(execution_error("custom KV preempt requires resident pages"));
        }
        for page in pages {
            self.pages.insert(*page, DecoderKvPageStatus::Preempted);
        }
        Ok(())
    }
    fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        if pages
            .iter()
            .any(|page| self.page_status(*page) != DecoderKvPageStatus::Preempted)
        {
            return Err(execution_error(
                "custom KV restore requires preempted pages",
            ));
        }
        for page in pages {
            self.pages.insert(*page, DecoderKvPageStatus::Resident);
        }
        Ok(())
    }
    fn capacity(&self) -> DecoderKvCapacity {
        let resident_pages = self
            .pages
            .values()
            .filter(|status| **status == DecoderKvPageStatus::Resident)
            .count();
        let preempted_pages = self
            .pages
            .values()
            .filter(|status| **status == DecoderKvPageStatus::Preempted)
            .count();
        DecoderKvCapacity {
            physical_pages: self.physical_pages,
            resident_pages,
            preempted_pages,
            active_transactions: self.active.len(),
            free_pages: self.physical_pages.saturating_sub(resident_pages),
        }
    }
    fn page_status(&self, page: KvPageId) -> DecoderKvPageStatus {
        self.pages
            .get(&page)
            .copied()
            .unwrap_or(DecoderKvPageStatus::Vacant)
    }
    fn shutdown(&mut self) -> Result<()> {
        if !self.active.is_empty() {
            return Err(execution_error(
                "custom KV backend still owns active transactions",
            ));
        }
        self.shutdown = true;
        trace(&self.trace).backend_shutdown = true;
        Ok(())
    }
}
#[derive(Debug)]
struct CustomForwardContinuation {
    transaction: ExecutionTransactionId,
    id: ContinuationId,
    dependency: MaterializationKey,
    cancel_attempts: usize,
}
#[derive(Debug)]
struct CustomTerminalGuard {
    lease_size: usize,
    trace: SharedTrace,
}
impl DecoderTerminalGuard for CustomTerminalGuard {}
impl Drop for CustomTerminalGuard {
    fn drop(&mut self) {
        trace(&self.trace)
            .terminal_guard_drop_lease_sizes
            .push(self.lease_size);
    }
}
struct CustomForwardExecutor {
    request: MaterializationRequest,
    dependency: MaterializationKey,
    trace: SharedTrace,
}
impl CustomForwardExecutor {
    fn logits(batch: &PackedDecoderBatch) -> DecoderLogits {
        DecoderLogits::Dense(
            DenseLogits::from_rows(
                (0..batch.len())
                    .map(|row| vec![row as f32, 3.0, 2.0, 1.0])
                    .collect(),
            )
            .expect("custom forward logits are rectangular"),
        )
    }
}
impl DecoderForwardExecutor<CustomSequenceState, CustomKvView> for CustomForwardExecutor {
    type Continuation = CustomForwardContinuation;
    type TerminalGuard = CustomTerminalGuard;
    fn start_forward(
        &mut self,
        context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [CustomSequenceState],
        kv: &mut CustomKvView,
    ) -> Result<DecoderForwardProgress<Self::Continuation, Self::TerminalGuard>> {
        kv.validate(context.transaction(), batch)?;
        if states.len() != batch.sequences().len() {
            return Err(execution_error("custom forward state cohort mismatch"));
        }
        for (state, sequence) in states.iter_mut().zip(batch.sequences()) {
            state
                .rows
                .extend_from_slice(&batch.token_ids()[sequence.query()]);
        }
        let use_ = StageResourceUse::new(
            self.request.resource(),
            ResourceAccess::Read,
            ResourceRetention::ThroughTransaction,
        );
        let stage = context.resolve_stage(
            vec![StageMaterializationRequest::new(use_, self.request)?],
            WorkspaceClaim::NONE,
        )?;
        let id = context.continuation_id()?;
        trace(&self.trace).forward_starts += 1;
        Ok(DecoderForwardProgress::Waiting {
            continuation: CustomForwardContinuation {
                transaction: context.transaction(),
                id,
                dependency: self.dependency,
                cancel_attempts: 0,
            },
            wait: DecoderWait::ResolvedStage(stage),
        })
    }
    fn resume_forward(
        &mut self,
        context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        _states: &mut [CustomSequenceState],
        kv: &mut CustomKvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderForwardResumeProgress<Self::TerminalGuard>> {
        kv.validate(context.transaction(), batch)?;
        if continuation.transaction != context.transaction()
            || continuation.id != context.continuation_id()?
            || continuation.dependency != self.dependency
            || context
                .resume_lease()?
                .binding_for(self.dependency)
                .is_none()
        {
            return Err(execution_error("custom forward resume custody mismatch"));
        }
        let lease = context.resume_lease()?;
        if lease.is_empty() {
            return Err(execution_error(
                "custom forward terminal guard requires a non-empty lease",
            ));
        }
        trace(&self.trace).forward_resumes += 1;
        Ok(DecoderForwardResumeProgress::Complete {
            logits: Self::logits(batch),
            terminal_guard: CustomTerminalGuard {
                lease_size: lease.len(),
                trace: Arc::clone(&self.trace),
            },
        })
    }
    fn cancel_forward(
        &mut self,
        context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        _states: &mut [CustomSequenceState],
        kv: &mut CustomKvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderCancelProgress> {
        kv.validate(context.transaction(), batch)?;
        if continuation.transaction != context.transaction()
            || continuation.id != context.continuation_id()?
        {
            return Err(execution_error(
                "custom forward cancel transaction mismatch",
            ));
        }
        continuation.cancel_attempts += 1;
        trace(&self.trace).forward_cancels += 1;
        if continuation.cancel_attempts == 1 {
            Ok(DecoderCancelProgress::Waiting)
        } else {
            Ok(DecoderCancelProgress::Complete)
        }
    }
    fn shutdown(&mut self) -> Result<()> {
        trace(&self.trace).forward_shutdown = true;
        Ok(())
    }
}
#[derive(Debug)]
struct CustomProposalContinuation {
    transaction: ExecutionTransactionId,
    topology_id: SequenceTopologyId,
    anchor_token_id: u32,
    cancel_attempts: usize,
}
struct CustomProposal {
    request: MaterializationRequest,
    dependency: MaterializationKey,
    trace: SharedTrace,
}
impl DecoderProposalExecutor<CustomSequenceState, CustomKvBackend> for CustomProposal {
    type Continuation = CustomProposalContinuation;
    fn source(&self) -> Result<Option<NativeProposalSource>> {
        Ok(Some(NativeProposalSource {
            implementation: "mock-custom-runtime",
            prepared_plan_id: PREPARED_PLAN_ID,
            native_width: 2,
        }))
    }
    fn start(
        &mut self,
        context: &mut DecoderTransactionContext,
        state: &mut CustomSequenceState,
        kv: &mut CustomKvView,
        anchor_token_id: u32,
    ) -> Result<DecoderProposalProgress<Self::Continuation>> {
        state.core.begin_step()?;
        if kv.transaction != context.transaction() {
            return Err(execution_error("custom proposal KV view mismatch"));
        }
        let continuation_id = context.continuation_id()?;
        let mut trace = trace(&self.trace);
        trace.proposal_starts += 1;
        trace.proposal_kv_views += 1;
        trace.proposal_ids.push(continuation_id);
        drop(trace);
        let use_ = StageResourceUse::new(
            self.request.resource(),
            ResourceAccess::Read,
            ResourceRetention::ThroughTransaction,
        );
        let stage = context.resolve_stage(
            vec![StageMaterializationRequest::new(use_, self.request)?],
            WorkspaceClaim::NONE,
        )?;
        Ok(DecoderProposalProgress::Waiting {
            continuation: CustomProposalContinuation {
                transaction: context.transaction(),
                topology_id: state.topology_id,
                anchor_token_id,
                cancel_attempts: 0,
            },
            wait: DecoderWait::ResolvedStage(stage),
        })
    }
    fn resume(
        &mut self,
        context: &mut DecoderTransactionContext,
        state: &mut CustomSequenceState,
        kv: &mut CustomKvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderProposalResumeProgress> {
        if kv.transaction != context.transaction()
            || continuation.transaction != context.transaction()
            || continuation.topology_id != state.topology_id
            || context
                .resume_lease()?
                .binding_for(self.dependency)
                .is_none()
        {
            return Err(execution_error("custom proposal resume custody mismatch"));
        }
        let lease = context.resume_lease()?;
        trace(&self.trace).proposal_resumes += 1;
        trace(&self.trace).proposal_kv_views += 1;
        trace(&self.trace)
            .proposal_resume_lease_sizes
            .push(lease.len());
        let first = continuation
            .anchor_token_id
            .checked_add(1)
            .ok_or_else(|| execution_error("custom proposal token overflow"))?;
        let second = first
            .checked_add(1)
            .ok_or_else(|| execution_error("custom proposal token overflow"))?;
        Ok(DecoderProposalResumeProgress::Complete(NativeProposal {
            token_ids: vec![first, second],
            confidence_logits: vec![0.9, 0.8],
        }))
    }
    fn cancel(
        &mut self,
        context: &mut DecoderTransactionContext,
        state: &mut CustomSequenceState,
        kv: &mut CustomKvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderCancelProgress> {
        if kv.transaction != context.transaction()
            || continuation.transaction != context.transaction()
            || continuation.topology_id != state.topology_id
        {
            return Err(execution_error("custom proposal cancel custody mismatch"));
        }
        continuation.cancel_attempts += 1;
        trace(&self.trace).proposal_cancels += 1;
        trace(&self.trace).proposal_kv_views += 1;
        if continuation.cancel_attempts == 1 {
            Ok(DecoderCancelProgress::Waiting)
        } else {
            Ok(DecoderCancelProgress::Complete)
        }
    }
    fn retain_provisional(
        &mut self,
        context: &mut DecoderProvisionalRetainContext<'_, CustomSequenceState, CustomKvBackend>,
    ) -> Result<()> {
        let executed_rows = context.executed_rows().to_vec();
        let retained_rows = context.retained_rows().to_vec();
        let source_rows = context
            .sources()
            .iter()
            .map(|source| source.rows.to_vec())
            .collect::<Vec<_>>();
        if source_rows.len() != retained_rows.len() {
            return Err(execution_error("custom provisional source shape mismatch"));
        }
        {
            let (backend, transaction) = context.backend_parts();
            backend.validate_checkpoint(transaction, &retained_rows)?;
        }
        {
            let working_states = context.working_states();
            for (((working, source), &executed), &retained) in working_states
                .iter_mut()
                .zip(&source_rows)
                .zip(&executed_rows)
                .zip(&retained_rows)
            {
                let expected = source
                    .len()
                    .checked_add(executed)
                    .ok_or_else(|| execution_error("custom provisional row count overflow"))?;
                if working.rows.len() != expected || !working.rows.starts_with(source) {
                    return Err(execution_error(
                        "custom provisional working rows do not extend their source",
                    ));
                }
                working.rows.truncate(source.len() + retained);
            }
            trace(&self.trace).retained_working_rows = working_states
                .iter()
                .map(|working| working.rows.to_vec())
                .collect();
        }
        let (backend, transaction) = context.backend_parts();
        backend.apply_checkpoint(transaction, &retained_rows);
        Ok(())
    }
}
#[derive(Debug)]
struct CustomMaterializationProvider {
    placement: MaterializationPlacement,
    trace: SharedTrace,
    prepared: HashMap<MaterializationKey, MaterializationResident>,
}
impl CustomMaterializationProvider {
    fn new(placement: MaterializationPlacement, trace: SharedTrace) -> Self {
        Self {
            placement,
            trace,
            prepared: HashMap::new(),
        }
    }
    fn resident(
        &self,
        request: MaterializationRequest,
    ) -> std::result::Result<MaterializationResident, FailureReason> {
        if request.model() != self.placement.model()
            || request.backend() != self.placement.backend()
            || request.device() != self.placement.device()
        {
            return Err(provider_contract_error("request placement mismatch"));
        }
        let key = request
            .materialization_key(DestinationGeneration::new(9))
            .map_err(|error| provider_contract_error(error.to_string()))?;
        let binding = ResidencyBinding::new(
            key.model(),
            key.resource(),
            key.backend(),
            key.device(),
            DestinationSlotId::new(7),
            key.destination_generation(),
        );
        MaterializationResident::new(key, binding)
            .map_err(|source| FailureReason::Protocol { source })
    }
    fn unsupported_transfer() -> FailureReason {
        provider_contract_error("resident-only custom provider has no transfer operation")
    }
}
impl MaterializationProvider for CustomMaterializationProvider {
    fn placement(&self) -> MaterializationPlacement {
        self.placement
    }
    fn resource_topology(&self) -> Result<PhysicalMaterializationTopology> {
        PhysicalMaterializationTopology::new(MaterializationResourceLimits::default(), 0, 0)
    }
    fn prepare(
        &mut self,
        request: MaterializationRequest,
        _purpose: MaterializationPurpose,
    ) -> std::result::Result<MaterializationPreparation, FailureReason> {
        let resident = self.resident(request)?;
        self.prepared.insert(resident.key(), resident);
        trace(&self.trace).provider_prepares += 1;
        Ok(MaterializationPreparation::Resident(resident))
    }
    fn prepared(
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<MaterializationPreparation, FailureReason> {
        self.prepared
            .get(&key)
            .copied()
            .map(MaterializationPreparation::Resident)
            .ok_or_else(|| provider_contract_error("unknown prepared custom resource"))
    }
    fn promote_to_execution(
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<MaterializationPreparation, FailureReason> {
        self.prepared(key)
    }
    fn materialization_plan(
        &self,
        key: MaterializationKey,
    ) -> std::result::Result<MaterializationResourcePlan, FailureReason> {
        if !self.prepared.contains_key(&key) {
            return Err(provider_contract_error("unknown custom resource plan"));
        }
        MaterializationResourcePlan::uniform_payload(16)
            .map_err(|source| FailureReason::Resources { source })
    }
    fn discard_preparation(
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<(), FailureReason> {
        self.prepared
            .remove(&key)
            .map(|_| ())
            .ok_or_else(|| provider_contract_error("unknown custom preparation discard"))
    }
    fn release_execution_lease(
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<(), FailureReason> {
        self.prepared
            .remove(&key)
            .map(|_| ())
            .ok_or_else(|| provider_contract_error("unknown custom execution lease"))
    }
    fn reserve(
        &mut self,
        _operation: OperationId,
        _key: MaterializationKey,
        _plan: MaterializationResourcePlan,
    ) -> std::result::Result<PhysicalMaterializationOperationReservation, FailureReason> {
        Err(Self::unsupported_transfer())
    }
    fn submit_read(
        &mut self,
        _operation: OperationId,
        _key: MaterializationKey,
        _reservation: &PhysicalMaterializationOperationReservation,
        _plan: MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason> {
        Err(Self::unsupported_transfer())
    }
    fn submit_upload(
        &mut self,
        _operation: OperationId,
        _key: MaterializationKey,
        _reservation: &PhysicalMaterializationOperationReservation,
        _plan: MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason> {
        Err(Self::unsupported_transfer())
    }
    fn poll_install(
        &mut self,
        _operation: OperationId,
        _key: MaterializationKey,
        _reservation: &PhysicalMaterializationOperationReservation,
        _plan: MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason> {
        Err(Self::unsupported_transfer())
    }
    fn cancel(
        &mut self,
        _operation: OperationId,
        _key: MaterializationKey,
        _stage: LoadStage,
        _reason: CancellationReason,
    ) -> std::result::Result<(), FailureReason> {
        Err(Self::unsupported_transfer())
    }
    fn next_completion(&mut self) -> Option<CompletionEvent> {
        None
    }
}
struct CustomMaterializationResolver {
    placement: MaterializationPlacement,
    trace: SharedTrace,
}
impl MaterializationResolver for CustomMaterializationResolver {
    fn placement(&self) -> MaterializationPlacement {
        self.placement
    }
    fn resolve(
        &mut self,
        request: MaterializationRequest,
    ) -> MaterializationResolveResult<MaterializationKey> {
        assert_eq!(request.model(), self.placement.model());
        assert_eq!(request.backend(), self.placement.backend());
        assert_eq!(request.device(), self.placement.device());
        trace(&self.trace).resolver_calls += 1;
        Ok(request
            .materialization_key(DestinationGeneration::new(9))
            .expect("custom resolver request must be valid"))
    }
}
struct CustomResourceManager {
    placement: MaterializationPlacement,
    trace: SharedTrace,
    provider: Option<CustomMaterializationProvider>,
    warmup_requests: Vec<MaterializationRequest>,
    prefetch_request: MaterializationRequest,
    shutdown: bool,
}
impl CustomResourceManager {
    fn new(
        placement: MaterializationPlacement,
        request: MaterializationRequest,
        trace: SharedTrace,
    ) -> Self {
        Self {
            placement,
            provider: Some(CustomMaterializationProvider::new(
                placement,
                Arc::clone(&trace),
            )),
            trace,
            warmup_requests: vec![request],
            prefetch_request: request,
            shutdown: false,
        }
    }
}
impl DecoderResourceManager for CustomResourceManager {
    fn take_provider(&mut self) -> Option<Box<dyn MaterializationProvider>> {
        self.provider
            .take()
            .map(|provider| Box::new(provider) as Box<dyn MaterializationProvider>)
    }
    fn take_warmup_requests(&mut self) -> Result<Vec<MaterializationRequest>> {
        let requests = std::mem::take(&mut self.warmup_requests);
        trace(&self.trace).warmup_requests_taken += requests.len();
        Ok(requests)
    }
    fn transaction_prefetch_requests(
        &self,
        transaction: ExecutionTransactionId,
        batch: &ExecutionBatch,
    ) -> Result<Vec<MaterializationRequest>> {
        if batch.is_empty() {
            return Err(execution_error(
                "custom materialization prefetch requires a non-empty batch",
            ));
        }
        trace(&self.trace)
            .prefetches
            .push((transaction.get(), batch.len()));
        Ok(vec![self.prefetch_request])
    }
    fn materialization_placement(&self) -> Option<MaterializationPlacement> {
        Some(self.placement)
    }
    fn shutdown(&mut self) -> Result<()> {
        self.shutdown = true;
        trace(&self.trace).resource_shutdown = true;
        Ok(())
    }
    fn observer_snapshot(&self) -> DecoderResourceSnapshot {
        trace(&self.trace).resource_observer_snapshots += 1;
        DecoderResourceSnapshot {
            resolver_installed: false,
            shutdown: self.shutdown,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct CustomObserverSnapshot {
    typed_kv_views: usize,
    provisional_checkpoints: usize,
    committed_checkpoints: usize,
    backend_shutdown: bool,
    resources: DecoderResourceSnapshot,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CustomObserver;
impl DecoderObserver<CustomKvBackend> for CustomObserver {
    type Snapshot = CustomObserverSnapshot;
    fn snapshot(
        &self,
        _core: &GenericDecoderObservabilitySnapshot,
        backend: &CustomKvBackend,
        resources: &DecoderResourceSnapshot,
    ) -> Self::Snapshot {
        CustomObserverSnapshot {
            typed_kv_views: trace(&backend.trace).typed_kv_views,
            provisional_checkpoints: backend.provisional_checkpoints.len(),
            committed_checkpoints: backend.committed_checkpoints.len(),
            backend_shutdown: backend.shutdown,
            resources: resources.clone(),
        }
    }
}
#[derive(Debug)]
struct CustomResources {
    resource_contract: &'static str,
}
struct MockCustomRuntime;
impl DecoderComposition for MockCustomRuntime {
    type Resources = CustomResources;
    type SequenceState = CustomSequenceState;
    type KvBackend = CustomKvBackend;
    type ForwardExecutor = CustomForwardExecutor;
    type Proposal = CustomProposal;
    type SequenceLifecycle = CustomSequenceLifecycle;
    type ResourceManager = CustomResourceManager;
    type Observer = ComposedDecoderObserver<StandardDecoderObserver, CustomObserver>;
    type Snapshot =
        ComposedDecoderSnapshot<GenericDecoderObservabilitySnapshot, CustomObserverSnapshot>;
    type Continuation = CustomForwardContinuation;
    type ProposalContinuation = CustomProposalContinuation;
    type TerminalGuard = CustomTerminalGuard;
}
type CustomRunner = GenericDecoderRunner<MockCustomRuntime>;
struct CustomRuntimeFixture {
    runner: CustomRunner,
    trace: SharedTrace,
    completion_hub: CompletionHub,
    request: MaterializationRequest,
    dependency: MaterializationKey,
}
fn custom_runtime() -> CustomRuntimeFixture {
    let trace = Arc::new(Mutex::new(CustomTrace::default()));
    let placement =
        MaterializationPlacement::new(ModelInstanceId::new(1), BackendId::new(3), DeviceId::new(0))
            .unwrap();
    let request = materialization_request(placement);
    let dependency = request
        .materialization_key(DestinationGeneration::new(9))
        .unwrap();
    let completion_hub = CompletionHub::new();
    let resource_manager = CustomResourceManager::new(placement, request, Arc::clone(&trace));
    let mut runner = GenericDecoderRunner::from_runtime(DecoderComponents {
        resources: CustomResources {
            resource_contract: "custom-runtime-v1",
        },
        model_info: ModelInfo {
            family: ModelFamily::Unknown("mock-custom".into()),
            architecture: Some("mock-custom-decoder".into()),
            attention: AttentionKind::DenseMha,
            weight_source: WeightSource::Safetensors,
            hidden_size: 4,
            num_layers: 2,
            num_experts: 0,
            num_experts_per_tok: 0,
            vocab_size: 4,
            backend: "mock-custom",
        },
        tokenizer: tokenizer(),
        capabilities: capabilities(),
        page_size: PAGE_SIZE,
        prepared_plan_id: PREPARED_PLAN_ID,
        forward_executor: CustomForwardExecutor {
            request,
            dependency,
            trace: Arc::clone(&trace),
        },
        backend: CustomKvBackend::new(Arc::clone(&trace)),
        default_state: CustomSequenceState::new(),
        sequence_lifecycle: CustomSequenceLifecycle {
            trace: Arc::clone(&trace),
        },
        proposal: CustomProposal {
            request,
            dependency,
            trace: Arc::clone(&trace),
        },
        resource_manager,
        observer: ComposedDecoderObserver {
            primary: StandardDecoderObserver,
            secondary: CustomObserver,
        },
        completion_hub: completion_hub.clone(),
        completion_reactors: Vec::new(),
    })
    .unwrap();
    MultiSessionRunner::install_materialization_resolver(
        &mut runner,
        Box::new(CustomMaterializationResolver {
            placement,
            trace: Arc::clone(&trace),
        }),
    )
    .unwrap();
    CustomRuntimeFixture {
        runner,
        trace,
        completion_hub,
        request,
        dependency,
    }
}
fn capabilities() -> ExecutionCapabilities {
    ExecutionCapabilities {
        max_batch_tokens: 8,
        max_sequences: 4,
        max_prefill_query_tokens_per_sequence: 8,
        max_decode_query_tokens_per_sequence: 2,
        max_top_k: NonZeroU32::new(4),
        supports_prefill: true,
        supports_decode: true,
        supports_mixed: true,
        full_logits_width: NonZeroU32::new(4),
        kv_binding_mode: KvBindingMode::Paged,
        logits_row_policy: LogitsRowPolicy::Any,
    }
}
fn tokenizer() -> TokenizerHandle {
    let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
    let tokens = ["a", "b", "c", "d"]
        .into_iter()
        .map(|token| tokenizers::AddedToken::from(token, false));
    assert_eq!(tokenizer.add_tokens(tokens).unwrap(), 4);
    TokenizerHandle::from_parts(tokenizer, Some(3))
}
fn materialization_request(placement: MaterializationPlacement) -> MaterializationRequest {
    let source = ResourceSource::new(
        SourceIdentityHash::new([2; 32]),
        ContentHash::new([3; 32]),
        PayloadEncodingId::new(4),
        SourceGeneration::new(5),
    )
    .unwrap();
    MaterializationRequest::for_placement(
        placement,
        source,
        MaterializedResourceId::new(MaterializedResourceKind::Parameter, 0, 1),
    )
    .unwrap()
}
fn transaction(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).unwrap()
}
fn packed_input(
    state: &CustomSequenceState,
    page: KvPageId,
    tokens: &[u32],
) -> (ExecutionBatch, Vec<KvReservationView>) {
    let context_len = state.core.position();
    let sequence_len = context_len + tokens.len();
    assert_eq!(context_len, 0, "custom fixture starts on a fresh page");
    assert!(sequence_len <= PAGE_SIZE);
    let positions = (context_len..sequence_len)
        .map(|position| u32::try_from(position).unwrap())
        .collect::<Vec<_>>();
    let write_slots = (context_len..sequence_len)
        .map(|position| {
            let page = usize::try_from(page.0).unwrap();
            Some(KvWriteSlot::try_from(page * PAGE_SIZE + position).unwrap())
        })
        .collect::<Vec<_>>();
    let mut logits = vec![LogitsRequest::None; tokens.len()];
    *logits.last_mut().expect("custom fixture has tokens") =
        LogitsRequest::TopK(NonZeroU32::new(2).unwrap());
    let token_count = u32::try_from(tokens.len()).unwrap();
    let batch = ExecutionBatch::new(
        ForwardMode::Prefill,
        tokens.to_vec(),
        positions,
        write_slots,
        logits,
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            ForwardPhase::Prefill,
            0..token_count,
            u32::try_from(context_len).unwrap(),
            u32::try_from(sequence_len).unwrap(),
            0..1,
        )],
        vec![KvBlockId::new(page.0)],
    );
    let reservations = vec![KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: context_len..sequence_len,
        newly_allocated: vec![page],
        generation: state.core.generation(),
        execution_generation: state.core.generation(),
        cow_replacement: None,
    }];
    (batch, reservations)
}
fn lease(key: MaterializationKey) -> ResidencyLeaseSet {
    ResidencyLeaseSet::new(
        [key],
        [ValidatedResidencyBinding::new(
            key,
            ResidencyBinding::new(
                key.model(),
                key.resource(),
                key.backend(),
                key.device(),
                DestinationSlotId::new(7),
                key.destination_generation(),
            ),
        )
        .unwrap()],
        MappingEpoch::new(1),
        DispatchFenceContract::new(
            OperationId::new(91),
            FenceId::new(92),
            key.backend(),
            key.device(),
        ),
    )
    .unwrap()
}
fn begin_and_resume(
    runner: &mut CustomRunner,
    states: &mut [CustomSequenceState],
    transaction: ExecutionTransactionId,
    batch: &ExecutionBatch,
    dependency: MaterializationKey,
) {
    let pending = match MultiSessionRunner::execute_multi_session_batch_progress(
        runner,
        transaction,
        states,
        batch,
    )
    .unwrap()
    {
        MultiSessionBatchProgress::Waiting(pending) => pending,
        MultiSessionBatchProgress::Complete(_) => panic!("custom forward unexpectedly completed"),
    };
    assert_eq!(pending.transaction(), transaction);
    assert_eq!(pending.dependencies().len(), 1);
    let output = match MultiSessionRunner::resume_multi_session_batch(
        runner,
        transaction,
        states,
        batch,
        pending.continuation(),
        lease(dependency),
    )
    .unwrap()
    {
        MultiSessionBatchProgress::Complete(output) => output,
        MultiSessionBatchProgress::Waiting(_) => panic!("custom forward unexpectedly waited twice"),
    };
    output
        .validate_with_capabilities(batch, &capabilities())
        .unwrap();
}
#[test]
fn custom_runtime_publish_retains_terminal_and_release_ownership() {
    let CustomRuntimeFixture {
        mut runner,
        trace: shared_trace,
        completion_hub,
        dependency,
        ..
    } = custom_runtime();
    assert_eq!(runner.resources().resource_contract, "custom-runtime-v1");
    assert_eq!(ModelRunner::model_info(&runner).backend, "mock-custom");
    let encoded = ModelRunner::encode(&runner, "a").unwrap();
    assert!(!encoded.is_empty());
    ModelRunner::decode(&runner, &encoded).unwrap();
    let mut states = vec![MultiSessionRunner::create_sequence_state(&mut runner).unwrap()];
    let topology = states[0].topology_id;
    let page = KvPageId(10);
    let (batch, reservations) = packed_input(&states[0], page, &[11, 12]);
    let active = transaction(1);
    runner.backend_mut().make_commit_pending_once(active);
    MultiSessionRunner::prepare_multi_session_batch(
        &mut runner,
        active,
        &mut states,
        &batch,
        &reservations,
    )
    .unwrap();
    assert_eq!(trace(&shared_trace).checkout_requests, [(1, 0, 0)]);
    let placement = runner.resource_manager().placement;
    let resolver_error = MultiSessionRunner::install_materialization_resolver(
        &mut runner,
        Box::new(CustomMaterializationResolver {
            placement,
            trace: Arc::clone(&shared_trace),
        }),
    )
    .unwrap_err();
    assert!(
        resolver_error
            .to_string()
            .contains("transactions are active")
    );
    let control_error =
        MultiSessionRunner::install_expert_residency_control(&mut runner, Box::new(NeverControl))
            .unwrap_err();
    assert!(
        control_error
            .to_string()
            .contains("transactions are active")
    );
    begin_and_resume(&mut runner, &mut states, active, &batch, dependency);
    assert_eq!(states[0].core.position(), 0);
    assert!(
        trace(&shared_trace)
            .terminal_guard_drop_lease_sizes
            .is_empty()
    );
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            active,
            &mut states,
            TransactionEndIntent::Publish,
        )
        .unwrap(),
        TransactionEndProgress::Pending
    );
    assert_eq!(states[0].core.position(), 0);
    assert!(
        trace(&shared_trace)
            .terminal_guard_drop_lease_sizes
            .is_empty()
    );
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            active,
            &mut states,
            TransactionEndIntent::Publish,
        )
        .unwrap(),
        TransactionEndProgress::Complete
    );
    assert_eq!(states[0].core.position(), 2);
    assert_eq!(states[0].rows, [11, 12]);
    assert_eq!(states[0].topology_id, topology);
    assert_eq!(trace(&shared_trace).terminal_guard_drop_lease_sizes, [1]);
    assert_eq!(
        runner.backend().committed_checkpoints.get(&active),
        Some(&vec![2])
    );
    let snapshot: ComposedDecoderSnapshot<
        GenericDecoderObservabilitySnapshot,
        CustomObserverSnapshot,
    > = ResidentModelRunner::observability_snapshot(&runner);
    assert_eq!(snapshot.primary.commits, 1);
    assert_eq!(snapshot.secondary.typed_kv_views, 2);
    assert_eq!(snapshot.secondary.committed_checkpoints, 1);
    assert_eq!(trace(&shared_trace).resource_observer_snapshots, 1);
    runner.backend_mut().fail_next_release();
    let error = MultiSessionRunner::release_kv_pages(&mut runner, &[page]).unwrap_err();
    assert!(error.to_string().contains("injected custom KV release"));
    assert_eq!(
        runner.backend().page_status(page),
        DecoderKvPageStatus::Resident
    );
    MultiSessionRunner::release_kv_pages(&mut runner, &[page]).unwrap();
    assert_eq!(
        runner.backend().page_status(page),
        DecoderKvPageStatus::Vacant
    );
    states[0].release_blocked = true;
    let error = MultiSessionRunner::try_release_sequence_state(&mut runner, states.pop().unwrap())
        .unwrap_err();
    assert!(error.source_error().to_string().contains("preflight"));
    assert_eq!(error.state().topology_id, topology);
    assert_eq!(error.state().rows, [11, 12]);
    let (_, mut retained_state) = error.into_parts();
    retained_state.release_blocked = false;
    MultiSessionRunner::try_release_sequence_state(&mut runner, retained_state).unwrap();
    assert_eq!(trace(&shared_trace).released_topologies, [topology.get()]);
    runner.shutdown().unwrap();
    assert!(completion_hub.is_closed());
    let snapshot = ResidentModelRunner::observability_snapshot(&runner);
    assert!(snapshot.primary.shutdown);
    assert!(snapshot.secondary.backend_shutdown);
    assert!(snapshot.secondary.resources.shutdown);
    assert!(trace(&shared_trace).forward_shutdown);
    assert!(trace(&shared_trace).resource_shutdown);
}
#[test]
fn custom_runtime_abort_and_cancel_quiesce_before_rollback() {
    let CustomRuntimeFixture {
        mut runner,
        trace: shared_trace,
        dependency,
        ..
    } = custom_runtime();
    let mut states = vec![MultiSessionRunner::create_sequence_state(&mut runner).unwrap()];
    let executed = transaction(2);
    let executed_page = KvPageId(20);
    let (batch, reservations) = packed_input(&states[0], executed_page, &[21, 22]);
    runner.backend_mut().make_rollback_pending_once(executed);
    MultiSessionRunner::prepare_multi_session_batch(
        &mut runner,
        executed,
        &mut states,
        &batch,
        &reservations,
    )
    .unwrap();
    begin_and_resume(&mut runner, &mut states, executed, &batch, dependency);
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            executed,
            &mut states,
            TransactionEndIntent::Abort,
        )
        .unwrap(),
        TransactionEndProgress::Pending
    );
    assert!(
        trace(&shared_trace)
            .terminal_guard_drop_lease_sizes
            .is_empty()
    );
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            executed,
            &mut states,
            TransactionEndIntent::Abort,
        )
        .unwrap(),
        TransactionEndProgress::Complete
    );
    assert_eq!(states[0].core.position(), 0);
    assert!(states[0].rows.is_empty());
    assert_eq!(
        runner.backend().page_status(executed_page),
        DecoderKvPageStatus::Vacant
    );
    assert_eq!(trace(&shared_trace).terminal_guard_drop_lease_sizes, [1]);
    let cancelled = transaction(3);
    let cancelled_page = KvPageId(21);
    let (batch, reservations) = packed_input(&states[0], cancelled_page, &[31, 32]);
    MultiSessionRunner::prepare_multi_session_batch(
        &mut runner,
        cancelled,
        &mut states,
        &batch,
        &reservations,
    )
    .unwrap();
    assert!(matches!(
        MultiSessionRunner::execute_multi_session_batch_progress(
            &mut runner,
            cancelled,
            &mut states,
            &batch,
        )
        .unwrap(),
        MultiSessionBatchProgress::Waiting(_)
    ));
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            cancelled,
            &mut states,
            TransactionEndIntent::Abort,
        )
        .unwrap(),
        TransactionEndProgress::Pending
    );
    assert_eq!(runner.backend().capacity().active_transactions, 1);
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            cancelled,
            &mut states,
            TransactionEndIntent::Abort,
        )
        .unwrap(),
        TransactionEndProgress::Complete
    );
    assert_eq!(trace(&shared_trace).forward_cancels, 2);
    assert_eq!(states[0].core.position(), 0);
    assert!(states[0].rows.is_empty());
    assert_eq!(
        runner.backend().page_status(cancelled_page),
        DecoderKvPageStatus::Vacant
    );
    MultiSessionRunner::try_release_sequence_state(&mut runner, states.pop().unwrap()).unwrap();
    runner.shutdown().unwrap();
}
#[test]
fn custom_runtime_retain_updates_working_rows_and_backend_checkpoint() {
    let CustomRuntimeFixture {
        mut runner,
        trace: shared_trace,
        dependency,
        ..
    } = custom_runtime();
    let source = CustomSequenceState::source_at(0, Vec::new());
    let mut branches = vec![MultiSessionRunner::create_sequence_state(&mut runner).unwrap()];
    let active = transaction(4);
    let page = KvPageId(30);
    let (batch, reservations) = packed_input(&branches[0], page, &[41, 42]);
    MultiSessionRunner::prepare_multi_session_batch(
        &mut runner,
        active,
        &mut branches,
        &batch,
        &reservations,
    )
    .unwrap();
    begin_and_resume(&mut runner, &mut branches, active, &batch, dependency);
    MultiSessionRunner::retain_provisional_prefixes(
        &mut runner,
        active,
        std::slice::from_ref(&source),
        &mut branches,
        &[2],
        &[1],
    )
    .unwrap();
    assert_eq!(branches[0].core.position(), 0);
    assert!(branches[0].rows.is_empty());
    assert_eq!(trace(&shared_trace).retained_working_rows, [vec![41]]);
    assert_eq!(
        runner.backend().provisional_checkpoints.get(&active),
        Some(&vec![1])
    );
    let snapshot = ResidentModelRunner::observability_snapshot(&runner);
    assert_eq!(snapshot.secondary.provisional_checkpoints, 1);
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            active,
            &mut branches,
            TransactionEndIntent::Publish,
        )
        .unwrap(),
        TransactionEndProgress::Complete
    );
    assert_eq!(branches[0].core.position(), 1);
    assert_eq!(branches[0].rows, [41]);
    assert_eq!(
        runner.backend().committed_checkpoints.get(&active),
        Some(&vec![1])
    );
    assert!(runner.backend().provisional_checkpoints.is_empty());
    assert_eq!(trace(&shared_trace).terminal_guard_drop_lease_sizes, [1]);
    MultiSessionRunner::release_kv_pages(&mut runner, &[page]).unwrap();
    MultiSessionRunner::try_release_sequence_state(&mut runner, branches.pop().unwrap()).unwrap();
    runner.shutdown().unwrap();
}
#[test]
fn custom_runtime_forwards_proposal_materialization_observer_and_shutdown() {
    let CustomRuntimeFixture {
        mut runner,
        trace: shared_trace,
        completion_hub,
        request,
        dependency,
    } = custom_runtime();
    let source = ResidentModelRunner::native_proposal_source(&runner)
        .unwrap()
        .expect("custom runtime exposes a proposal source");
    assert_eq!(source.implementation, "mock-custom-runtime");
    let proposal_transaction = transaction(5);
    let pending =
        match ResidentModelRunner::begin_native_proposal(&mut runner, proposal_transaction, 50)
            .unwrap()
        {
            NativeProposalProgress::Waiting(pending) => pending,
            NativeProposalProgress::Complete(_) => panic!("custom proposal unexpectedly completed"),
        };
    assert_eq!(pending.resources().len(), 1);
    assert_eq!(pending.resources()[0].key(), dependency);
    assert_eq!(trace(&shared_trace).proposal_kv_views, 1);
    assert_eq!(trace(&shared_trace).proposal_ids, [pending.continuation()]);
    let placement = runner.resource_manager().placement;
    let resolver_error = MultiSessionRunner::install_materialization_resolver(
        &mut runner,
        Box::new(CustomMaterializationResolver {
            placement,
            trace: Arc::clone(&shared_trace),
        }),
    )
    .unwrap_err();
    assert!(
        resolver_error
            .to_string()
            .contains("transactions are active")
    );
    assert!(ModelRunner::reset_session(&mut runner).is_err());
    let mut explicit_state = MultiSessionRunner::create_sequence_state(&mut runner).unwrap();
    assert!(MultiSessionRunner::with_sequence_state(
        &mut runner,
        &mut explicit_state,
        |_runner| Ok(()),
    )
    .is_err());
    let proposal = match ResidentModelRunner::resume_native_proposal(
        &mut runner,
        proposal_transaction,
        pending.continuation(),
        lease(dependency),
    )
    .unwrap()
    {
        NativeProposalProgress::Complete(proposal) => proposal,
        NativeProposalProgress::Waiting(_) => panic!("custom proposal unexpectedly waited twice"),
    };
    assert_eq!(proposal.token_ids, [51, 52]);
    assert_eq!(trace(&shared_trace).proposal_resume_lease_sizes, [1]);
    assert_eq!(trace(&shared_trace).proposal_kv_views, 2);
    ModelRunner::reset_session(&mut runner).unwrap();
    let cancelled_proposal = transaction(6);
    assert!(matches!(
        ResidentModelRunner::begin_native_proposal(&mut runner, cancelled_proposal, 60).unwrap(),
        NativeProposalProgress::Waiting(_)
    ));
    assert!(runner.shutdown().is_err());
    let mut no_states = Vec::new();
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            cancelled_proposal,
            &mut no_states,
            TransactionEndIntent::Abort,
        )
        .unwrap(),
        TransactionEndProgress::Pending
    );
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            cancelled_proposal,
            &mut no_states,
            TransactionEndIntent::Abort,
        )
        .unwrap(),
        TransactionEndProgress::Complete
    );
    let explicit_proposal = transaction(8);
    assert!(matches!(
        MultiSessionRunner::with_sequence_state(&mut runner, &mut explicit_state, |runner| {
            ResidentModelRunner::begin_native_proposal(runner, explicit_proposal, 80)
        },)
        .unwrap(),
        NativeProposalProgress::Waiting(_)
    ));
    let (blocked_batch, blocked_reservations) = packed_input(&explicit_state, KvPageId(41), &[81]);
    let error = MultiSessionRunner::prepare_multi_session_batch(
        &mut runner,
        explicit_proposal,
        std::slice::from_mut(&mut explicit_state),
        &blocked_batch,
        &blocked_reservations,
    )
    .unwrap_err();
    assert!(error.to_string().contains("owned by proposal continuation"));
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            explicit_proposal,
            std::slice::from_mut(&mut explicit_state),
            TransactionEndIntent::Abort,
        )
        .unwrap(),
        TransactionEndProgress::Pending
    );
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            explicit_proposal,
            std::slice::from_mut(&mut explicit_state),
            TransactionEndIntent::Abort,
        )
        .unwrap(),
        TransactionEndProgress::Complete
    );
    assert_eq!(trace(&shared_trace).proposal_starts, 3);
    assert_eq!(trace(&shared_trace).proposal_resumes, 1);
    assert_eq!(trace(&shared_trace).proposal_cancels, 4);
    let warmup = MultiSessionRunner::take_warmup_requests(&mut runner).unwrap();
    assert_eq!(warmup, [request]);
    assert!(
        MultiSessionRunner::take_warmup_requests(&mut runner)
            .unwrap()
            .is_empty()
    );
    let mut state = MultiSessionRunner::create_sequence_state(&mut runner).unwrap();
    let (batch, _) = packed_input(&state, KvPageId(40), &[71]);
    let prefetch =
        MultiSessionRunner::transaction_prefetch_requests(&runner, transaction(7), &batch).unwrap();
    assert_eq!(prefetch, [request]);
    let mut provider = MultiSessionRunner::take_materialization_provider(&mut runner)
        .expect("custom provider transfers exactly once");
    assert!(MultiSessionRunner::take_materialization_provider(&mut runner).is_none());
    assert_eq!(provider.placement().model(), request.model());
    let topology = provider.resource_topology().unwrap();
    assert!(topology.stage_limits().capacity.is_empty());
    let preparation = provider
        .prepare(request, MaterializationPurpose::Prefetch)
        .unwrap();
    assert_eq!(preparation.key(), dependency);
    assert_eq!(provider.prepared(dependency).unwrap(), preparation);
    assert_eq!(
        provider.promote_to_execution(dependency).unwrap(),
        preparation
    );
    assert_eq!(
        provider
            .materialization_plan(dependency)
            .unwrap()
            .resident_bytes,
        16
    );
    provider.discard_preparation(dependency).unwrap();
    let preparation = provider
        .prepare(request, MaterializationPurpose::Execution)
        .unwrap();
    provider.release_execution_lease(preparation.key()).unwrap();
    assert!(MultiSessionRunner::materialization_resolver_installed(
        &runner
    ));
    let resolved = MultiSessionRunner::materialization_resolver(&mut runner)
        .unwrap()
        .resolve(request)
        .unwrap();
    assert_eq!(resolved, dependency);
    let snapshot: ComposedDecoderSnapshot<
        GenericDecoderObservabilitySnapshot,
        CustomObserverSnapshot,
    > = ResidentModelRunner::observability_snapshot(&runner);
    assert_eq!(snapshot.primary.active_proposals, 0);
    assert_eq!(snapshot.secondary.typed_kv_views, 0);
    assert!(snapshot.secondary.resources.resolver_installed);
    assert!(!snapshot.secondary.resources.shutdown);
    assert_eq!(trace(&shared_trace).provider_prepares, 2);
    assert_eq!(trace(&shared_trace).resolver_calls, 4);
    assert_eq!(trace(&shared_trace).warmup_requests_taken, 1);
    assert_eq!(trace(&shared_trace).prefetches, [(7, 1)]);
    MultiSessionRunner::reset_sequence_state(&mut runner, &mut state).unwrap();
    MultiSessionRunner::try_release_sequence_state(&mut runner, state).unwrap();
    MultiSessionRunner::try_release_sequence_state(&mut runner, explicit_state).unwrap();
    runner.shutdown().unwrap();
    assert!(completion_hub.is_closed());
    assert!(trace(&shared_trace).backend_shutdown);
    assert!(trace(&shared_trace).forward_shutdown);
    assert!(trace(&shared_trace).resource_shutdown);
    let snapshot = ResidentModelRunner::observability_snapshot(&runner);
    assert!(snapshot.secondary.resources.shutdown);
    assert!(!snapshot.secondary.resources.resolver_installed);
}
#[test]
fn generic_decoder_is_the_only_resolver_owner() {
    let CustomRuntimeFixture {
        mut runner,
        trace,
        request,
        dependency,
        ..
    } = custom_runtime();
    assert!(MultiSessionRunner::materialization_resolver_installed(
        &runner
    ));
    assert!(
        !DecoderResourceManager::observer_snapshot(runner.resource_manager()).resolver_installed
    );

    let placement = runner.resource_manager().placement;
    let error = MultiSessionRunner::install_materialization_resolver(
        &mut runner,
        Box::new(CustomMaterializationResolver {
            placement,
            trace: Arc::clone(&trace),
        }),
    )
    .unwrap_err();
    assert!(error.to_string().contains("already installed"));
    assert_eq!(
        MultiSessionRunner::materialization_resolver(&mut runner)
            .unwrap()
            .resolve(request)
            .unwrap(),
        dependency
    );
}

#[test]
fn proposal_resume_failure_preserves_generic_lease_custody() {
    let CustomRuntimeFixture {
        mut runner,
        trace: shared_trace,
        dependency,
        ..
    } = custom_runtime();
    let transaction = transaction(81);
    let pending =
        match ResidentModelRunner::begin_native_proposal(&mut runner, transaction, 90).unwrap() {
            NativeProposalProgress::Waiting(pending) => pending,
            NativeProposalProgress::Complete(_) => panic!("custom proposal unexpectedly completed"),
        };
    let continuation = pending.continuation();
    let wrong_key = MaterializationRequest::new(
        dependency.model(),
        materialization_request(runner.resource_manager().placement).source(),
        dependency.resource(),
        dependency.backend(),
        dependency.device(),
    )
    .unwrap()
    .materialization_key(DestinationGeneration::new(10))
    .unwrap();
    let error = ResidentModelRunner::resume_native_proposal(
        &mut runner,
        transaction,
        continuation,
        lease(wrong_key),
    )
    .unwrap_err();
    assert!(error.to_string().contains("does not exactly satisfy"));
    assert_eq!(
        runner
            .proposal_held_resume_lease_count(continuation)
            .unwrap(),
        0
    );

    runner.backend_mut().fail_next_proposal_view();
    let error = ResidentModelRunner::resume_native_proposal(
        &mut runner,
        transaction,
        continuation,
        lease(dependency),
    )
    .unwrap_err();
    assert!(error.to_string().contains("proposal KV view failure"));
    assert_eq!(
        runner
            .proposal_held_resume_lease_count(continuation)
            .unwrap(),
        1
    );
    assert_eq!(trace(&shared_trace).proposal_resumes, 0);

    let mut states = Vec::new();
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            transaction,
            &mut states,
            TransactionEndIntent::Abort,
        )
        .unwrap(),
        TransactionEndProgress::Pending
    );
    assert_eq!(
        MultiSessionRunner::end_transaction(
            &mut runner,
            transaction,
            &mut states,
            TransactionEndIntent::Abort,
        )
        .unwrap(),
        TransactionEndProgress::Complete
    );
    assert!(
        runner
            .proposal_held_resume_lease_count(continuation)
            .is_err()
    );
}

fn provider_contract_error(message: impl Into<String>) -> FailureReason {
    FailureReason::ContractViolation {
        message: message.into(),
    }
}
fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: message.into(),
    }
}
