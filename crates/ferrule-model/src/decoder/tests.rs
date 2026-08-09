use super::*;
use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionSequence, ExecutionTransactionId, ForwardMode,
    ForwardPhase, KvBindingMode, KvBlockId, KvCowReplacement, KvPageId, KvReservationView,
    KvWriteSlot, LogitsRequest, LogitsRowPolicy, StateSlot,
};
use ferrule_common::{
    BackendId, ContentHash, ContinuationId, DependencySet, DestinationGeneration,
    DestinationSlotId, DeviceId, DispatchFenceContract, Error, FenceId, LogicalDependency,
    MappingEpoch, MaterializationKey, MaterializedResourceId, MaterializedResourceKind,
    ModelInstanceId, OperationId, PayloadEncodingId, ResidencyBinding, ResidencyLeaseSet, Result,
    SourceGeneration, SourceIdentityHash, ValidatedResidencyBinding,
};
use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;
type State = DecoderSequenceState<u32, u32>;
type Registry = PackedTransactionRegistry<MockRuntime>;
struct MockRuntime;
impl DecoderSequenceAttachment for u32 {
    type Release = ();
    fn preflight_release(&self) -> Result<Self::Release> {
        Ok(())
    }
    fn release(self, _release: Self::Release) {}
}
impl DecoderComposition for MockRuntime {
    type Resources = ();
    type SequenceState = State;
    type KvBackend = MockKvBackend;
    type ForwardExecutor = MockForwardExecutor;
    type Proposal = NoProposal;
    type SequenceLifecycle = StandardSequenceLifecycle<u32, u32>;
    type ResourceManager = NoResourceManager;
    type Observer = StandardDecoderObserver;
    type Snapshot = GenericDecoderObservabilitySnapshot;
    type Continuation = u32;
    type ProposalContinuation = std::convert::Infallible;
    type TerminalGuard = NoTerminalGuard;
}
struct Fixture {
    states: Vec<State>,
    batch: ExecutionBatch,
    reservations: Vec<KvReservationView>,
    pages: BTreeMap<KvPageId, DecoderKvPageStatus>,
}
impl Fixture {
    fn lower(&self) -> Result<PackedDecoderBatch> {
        PackedDecoderBatch::lower(
            &self.batch,
            &self.reservations,
            &self.states,
            &capabilities(),
            4,
            &|page| {
                self.pages
                    .get(&page)
                    .copied()
                    .unwrap_or(DecoderKvPageStatus::Vacant)
            },
        )
    }
}
fn capabilities() -> ExecutionCapabilities {
    ExecutionCapabilities {
        max_batch_tokens: 32,
        max_sequences: 8,
        max_prefill_query_tokens_per_sequence: 16,
        max_decode_query_tokens_per_sequence: 4,
        max_top_k: NonZeroU32::new(4),
        supports_prefill: true,
        supports_decode: true,
        supports_mixed: true,
        full_logits_width: NonZeroU32::new(4),
        kv_binding_mode: KvBindingMode::Paged,
        logits_row_policy: LogitsRowPolicy::Any,
    }
}
fn transaction(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).unwrap()
}
fn mixed_fixture() -> Fixture {
    let states = vec![
        State::with_position(4, 10, 100),
        State::new(20, 200),
        State::new(30, 300),
    ];
    let batch = ExecutionBatch::new(
        ForwardMode::Mixed,
        vec![11, 12, 13, 21],
        vec![0, 1, 2, 4],
        vec![
            Some(KvWriteSlot::new(40)),
            Some(KvWriteSlot::new(41)),
            Some(KvWriteSlot::new(42)),
            Some(KvWriteSlot::new(84)),
        ],
        vec![
            LogitsRequest::None,
            LogitsRequest::TopK(NonZeroU32::new(2).unwrap()),
            LogitsRequest::None,
            LogitsRequest::Full,
        ],
        vec![
            ExecutionSequence::new(StateSlot::new(2), ForwardPhase::Prefill, 0..3, 0, 3, 0..1),
            ExecutionSequence::new(StateSlot::new(0), ForwardPhase::Decode, 3..4, 4, 5, 1..3),
        ],
        vec![KvBlockId::new(10), KvBlockId::new(20), KvBlockId::new(21)],
    );
    let reservations = vec![
        KvReservationView {
            state_slot: StateSlot::new(7),
            execution_state_slot: StateSlot::new(2),
            positions: 0..3,
            newly_allocated: vec![KvPageId(10)],
            generation: 11,
            execution_generation: states[2].core().generation(),
            cow_replacement: None,
        },
        KvReservationView {
            state_slot: StateSlot::new(8),
            execution_state_slot: StateSlot::new(0),
            positions: 4..5,
            newly_allocated: vec![KvPageId(21)],
            generation: 12,
            execution_generation: states[0].core().generation(),
            cow_replacement: None,
        },
    ];
    let pages = BTreeMap::from([(KvPageId(20), DecoderKvPageStatus::Resident)]);
    Fixture {
        states,
        batch,
        reservations,
        pages,
    }
}
fn batch_with_different_first_token(batch: &ExecutionBatch) -> ExecutionBatch {
    let mut token_ids = batch.token_ids().to_vec();
    token_ids[0] ^= 1;
    ExecutionBatch::new(
        batch.mode(),
        token_ids,
        batch.positions().to_vec(),
        batch.kv_write_slots().to_vec(),
        batch.logits().to_vec(),
        batch.sequences().to_vec(),
        batch.kv_block_ids().to_vec(),
    )
    .with_intent(batch.intent())
}
fn single_sequence_fixture(
    states: Vec<State>,
    execution_state_slot: u32,
    page_state_slot: u32,
    context_len: u32,
    block_ids: Vec<u32>,
    new_pages: Vec<KvPageId>,
    cow_replacement: Option<KvCowReplacement>,
    resident_pages: &[KvPageId],
) -> Fixture {
    let position = context_len;
    let sequence_len = context_len + 1;
    let page = block_ids[(position as usize) / 4];
    let write_slot = page * 4 + position % 4;
    let block_count = u32::try_from(block_ids.len()).unwrap();
    let state_index = usize::try_from(execution_state_slot).unwrap();
    Fixture {
        batch: ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![17],
            vec![position],
            vec![Some(KvWriteSlot::new(write_slot))],
            vec![LogitsRequest::TopK(NonZeroU32::new(1).unwrap())],
            vec![ExecutionSequence::new(
                StateSlot::new(execution_state_slot),
                ForwardPhase::Prefill,
                0..1,
                context_len,
                sequence_len,
                0..block_count,
            )],
            block_ids.into_iter().map(KvBlockId::new).collect(),
        ),
        reservations: vec![KvReservationView {
            state_slot: StateSlot::new(page_state_slot),
            execution_state_slot: StateSlot::new(execution_state_slot),
            positions: context_len as usize..sequence_len as usize,
            newly_allocated: new_pages,
            generation: 90 + u64::from(page_state_slot),
            execution_generation: states[state_index].core().generation(),
            cow_replacement,
        }],
        states,
        pages: resident_pages
            .iter()
            .copied()
            .map(|page| (page, DecoderKvPageStatus::Resident))
            .collect(),
    }
}
#[test]
fn lowers_ragged_mixed_batch_with_exact_kv_and_logits_plan() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    assert_eq!(packed.mode(), ForwardMode::Mixed);
    assert_eq!(packed.row_to_sequence(), [0, 0, 0, 1]);
    assert_eq!(packed.sequences()[0].state_index(), 2);
    assert_eq!(packed.sequences()[0].query(), 0..3);
    assert_eq!(packed.sequences()[1].state_index(), 0);
    assert_eq!(
        packed.sequences()[1].block_table(),
        [KvPageId(20), KvPageId(21)]
    );
    assert_eq!(packed.new_pages(), [KvPageId(10), KvPageId(21)]);
    assert!(packed.writable_pages().is_empty());
    assert!(packed.cow_replacements().is_empty());
    assert_eq!(
        packed.protected_pages(),
        [KvPageId(10), KvPageId(20), KvPageId(21)]
    );
    assert_eq!(
        packed
            .logits_plan()
            .rows()
            .iter()
            .map(|row| row.input_row())
            .collect::<Vec<_>>(),
        [1, 3]
    );
    assert_eq!(packed.logits_plan().full_logits_width(), Some(4));
}
#[test]
fn lowering_rejects_each_reservation_identity_and_physical_mismatch() {
    let mut fixture = mixed_fixture();
    fixture.reservations[0].execution_generation += 1;
    assert!(fixture.lower().unwrap_err().to_string().contains("stale"));
    let mut fixture = mixed_fixture();
    fixture.reservations[0].execution_state_slot = StateSlot::new(0);
    assert!(
        fixture
            .lower()
            .unwrap_err()
            .to_string()
            .contains("execution state slot")
    );
    let mut fixture = mixed_fixture();
    fixture.reservations[1].state_slot = fixture.reservations[0].state_slot;
    assert!(
        fixture
            .lower()
            .unwrap_err()
            .to_string()
            .contains("page-manager state slot")
    );
    let mut fixture = mixed_fixture();
    fixture.reservations[0].positions = 1..3;
    assert!(fixture.lower().unwrap_err().to_string().contains("covers"));
    let mut fixture = mixed_fixture();
    fixture.batch = ExecutionBatch::new(
        ForwardMode::Mixed,
        fixture.batch.token_ids().to_vec(),
        fixture.batch.positions().to_vec(),
        vec![
            Some(KvWriteSlot::new(99)),
            Some(KvWriteSlot::new(41)),
            Some(KvWriteSlot::new(42)),
            Some(KvWriteSlot::new(84)),
        ],
        fixture.batch.logits().to_vec(),
        fixture.batch.sequences().to_vec(),
        fixture.batch.kv_block_ids().to_vec(),
    );
    assert!(
        fixture
            .lower()
            .unwrap_err()
            .to_string()
            .contains("write slot")
    );
    let mut fixture = mixed_fixture();
    fixture.reservations[1].newly_allocated = vec![KvPageId(22)];
    assert!(
        fixture
            .lower()
            .unwrap_err()
            .to_string()
            .contains("new pages")
    );
    let mut fixture = mixed_fixture();
    fixture
        .pages
        .insert(KvPageId(20), DecoderKvPageStatus::Preempted);
    assert!(
        fixture
            .lower()
            .unwrap_err()
            .to_string()
            .contains("Preempted")
    );
}
#[test]
fn prepare_rejects_state_reset_or_topology_replacement_after_lowering() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let mut reset_states = fixture.states.clone();
    reset_states[2].core_mut().reset();
    let mut backend = MockKvBackend::default();
    let mut registry = Registry::new("mock");
    let error = registry
        .prepare(transaction(80), packed, &reset_states, &mut backend)
        .unwrap_err();
    assert!(error.to_string().contains("stale"));
    assert!(backend.prepared.is_empty());
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let mut replaced_states = fixture.states;
    replaced_states[2] = State::new(30, 300);
    let error = registry
        .prepare(transaction(81), packed, &replaced_states, &mut backend)
        .unwrap_err();
    assert!(error.to_string().contains("changed topology identity"));
    assert!(backend.prepared.is_empty());
}
#[test]
fn lowering_classifies_writable_and_cow_pages_and_protects_sources() {
    let writable = single_sequence_fixture(
        vec![State::with_position(2, 1, 1)],
        0,
        4,
        2,
        vec![20],
        vec![],
        None,
        &[KvPageId(20)],
    );
    let packed = writable.lower().unwrap();
    assert_eq!(packed.writable_pages(), [KvPageId(20)]);
    assert_eq!(packed.protected_pages(), [KvPageId(20)]);
    let cow = KvCowReplacement {
        logical_page: 0,
        source: KvPageId(30),
        replacement: KvPageId(31),
    };
    let cow_fixture = single_sequence_fixture(
        vec![State::with_position(2, 1, 1)],
        0,
        5,
        2,
        vec![31],
        vec![],
        Some(cow),
        &[KvPageId(30)],
    );
    let packed = cow_fixture.lower().unwrap();
    assert!(packed.writable_pages().is_empty());
    assert_eq!(packed.cow_replacements(), [cow]);
    assert_eq!(packed.protected_pages(), [KvPageId(30), KvPageId(31)]);
    let mut malformed = cow_fixture;
    malformed.reservations[0]
        .cow_replacement
        .as_mut()
        .unwrap()
        .logical_page = 1;
    assert!(
        malformed
            .lower()
            .unwrap_err()
            .to_string()
            .contains("invalid COW")
    );
}
#[derive(Debug)]
struct MockKvTransaction {
    entered: bool,
}
#[derive(Debug, Clone, Copy)]
enum MockKvFailure {
    RetainToken,
    ConsumeToken,
}
#[derive(Default)]
struct MockKvBackend {
    events: Vec<&'static str>,
    commit_progress: VecDeque<KvEndProgress>,
    rollback_progress: VecDeque<KvEndProgress>,
    commit_failures: VecDeque<MockKvFailure>,
    rollback_failures: VecDeque<MockKvFailure>,
    prepared: Vec<ExecutionTransactionId>,
}
impl MockKvBackend {
    fn fail(
        operation: &'static str,
        failure: MockKvFailure,
        transaction: &mut Option<MockKvTransaction>,
    ) -> Error {
        if matches!(failure, MockKvFailure::ConsumeToken) {
            transaction.take();
        }
        Error::Execution {
            message: format!("injected {operation} failure"),
        }
    }
    fn terminal(
        progress: KvEndProgress,
        transaction: &mut Option<MockKvTransaction>,
    ) -> KvEndProgress {
        if matches!(
            progress,
            KvEndProgress::Complete | KvEndProgress::ConsumedRejected
        ) {
            transaction.take();
        }
        progress
    }
}
impl DecoderKvBackend for MockKvBackend {
    type SequenceState = State;
    type Transaction = MockKvTransaction;
    type KvView = ();
    fn configure_capacity(&mut self, _max_pages: usize) -> Result<()> {
        Ok(())
    }
    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        assert!(
            request
                .new_pages
                .iter()
                .all(|page| request.protected_pages.contains(page))
        );
        self.events.push("prepare");
        self.prepared.push(request.transaction);
        Ok(MockKvTransaction { entered: false })
    }
    fn enter(
        &mut self,
        transaction: &mut Self::Transaction,
        _batch: &PackedDecoderBatch,
        _states: &mut [Self::SequenceState],
    ) -> Result<()> {
        assert!(!transaction.entered);
        transaction.entered = true;
        self.events.push("enter");
        Ok(())
    }
    fn active_view(&mut self, transaction: &mut Self::Transaction) -> Result<Self::KvView> {
        assert!(transaction.entered);
        Ok(())
    }
    fn leave(&mut self, transaction: &mut Self::Transaction) -> Result<()> {
        assert!(transaction.entered);
        transaction.entered = false;
        self.events.push("leave");
        Ok(())
    }
    fn commit(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        self.events.push("commit");
        if let Some(failure) = self.commit_failures.pop_front() {
            return Err(Self::fail("commit", failure, transaction));
        }
        let progress = self
            .commit_progress
            .pop_front()
            .unwrap_or(KvEndProgress::Complete);
        Ok(Self::terminal(progress, transaction))
    }
    fn rollback(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        self.events.push("rollback");
        if let Some(failure) = self.rollback_failures.pop_front() {
            return Err(Self::fail("rollback", failure, transaction));
        }
        let progress = self
            .rollback_progress
            .pop_front()
            .unwrap_or(KvEndProgress::Complete);
        Ok(Self::terminal(progress, transaction))
    }
    fn release(&mut self, _pages: &[KvPageId]) -> Result<()> {
        self.events.push("release");
        Ok(())
    }
    fn preempt(&mut self, _pages: &[KvPageId]) -> Result<()> {
        self.events.push("preempt");
        Ok(())
    }
    fn restore(&mut self, _pages: &[KvPageId]) -> Result<()> {
        self.events.push("restore");
        Ok(())
    }
    fn capacity(&self) -> DecoderKvCapacity {
        DecoderKvCapacity::default()
    }
    fn page_status(&self, _page: KvPageId) -> DecoderKvPageStatus {
        DecoderKvPageStatus::Vacant
    }
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}
#[derive(Debug, Clone, Copy)]
enum ForwardScript {
    Sync,
    Async,
    ResumeWaiting,
    ResumeError,
    FailedActive,
    FailedQuiescent,
}
struct MockForwardExecutor {
    script: ForwardScript,
    cancel_wait_once: bool,
    cancel_calls: usize,
}
impl MockForwardExecutor {
    fn new(script: ForwardScript) -> Self {
        Self {
            script,
            cancel_wait_once: false,
            cancel_calls: 0,
        }
    }
    fn with_cancel_wait(mut self) -> Self {
        self.cancel_wait_once = true;
        self
    }
    fn mutate(states: &mut [State]) {
        for state in states {
            *state.attachment_mut() += 1;
            *state.kv_state_mut() += 10;
        }
    }
    fn logits(batch: &PackedDecoderBatch) -> DecoderLogits {
        DecoderLogits::Dense(
            DenseLogits::from_rows(
                (0..batch.len())
                    .map(|row| vec![row as f32, 2.0, 2.0, -1.0])
                    .collect(),
            )
            .unwrap(),
        )
    }
    fn dependencies() -> DependencySet {
        DependencySet::new([LogicalDependency::resource_resident(materialization_key(1)).unwrap()])
            .unwrap()
    }
}
impl DecoderForwardExecutor<State, ()> for MockForwardExecutor {
    type Continuation = u32;
    type TerminalGuard = NoTerminalGuard;
    fn start_forward(
        &mut self,
        _context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [State],
        _kv: &mut (),
    ) -> Result<DecoderForwardProgress<Self::Continuation, Self::TerminalGuard>> {
        Self::mutate(states);
        match self.script {
            ForwardScript::Sync => Ok(DecoderForwardProgress::Complete {
                logits: Self::logits(batch),
                terminal_guard: NoTerminalGuard,
            }),
            ForwardScript::Async | ForwardScript::ResumeWaiting | ForwardScript::ResumeError => {
                Ok(DecoderForwardProgress::Waiting {
                    continuation: 9,
                    wait: DecoderWait::Dependencies(Self::dependencies()),
                })
            }
            ForwardScript::FailedActive => Ok(DecoderForwardProgress::FailedActive {
                continuation: 9,
                error: Error::Execution {
                    message: "injected active forward failure".into(),
                },
            }),
            ForwardScript::FailedQuiescent => {
                Ok(DecoderForwardProgress::FailedQuiescent(Error::Execution {
                    message: "injected quiescent forward failure".into(),
                }))
            }
        }
    }
    fn resume_forward(
        &mut self,
        _context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [State],
        _kv: &mut (),
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderForwardResumeProgress<Self::TerminalGuard>> {
        assert_eq!(*continuation, 9);
        Self::mutate(states);
        match self.script {
            ForwardScript::Async => Ok(DecoderForwardResumeProgress::Complete {
                logits: Self::logits(batch),
                terminal_guard: NoTerminalGuard,
            }),
            ForwardScript::ResumeWaiting => Ok(DecoderForwardResumeProgress::Waiting(
                DecoderWait::Dependencies(Self::dependencies()),
            )),
            ForwardScript::ResumeError => Err(Error::Execution {
                message: "injected resume failure".into(),
            }),
            ForwardScript::Sync | ForwardScript::FailedActive | ForwardScript::FailedQuiescent => {
                Err(Error::Internal {
                    message: "unexpected mock resume".into(),
                })
            }
        }
    }
    fn cancel_forward(
        &mut self,
        _context: &mut DecoderTransactionContext,
        _batch: &PackedDecoderBatch,
        _states: &mut [State],
        _kv: &mut (),
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderCancelProgress> {
        assert_eq!(*continuation, 9);
        self.cancel_calls += 1;
        if self.cancel_wait_once && self.cancel_calls == 1 {
            Ok(DecoderCancelProgress::Waiting)
        } else {
            Ok(DecoderCancelProgress::Complete)
        }
    }
}
fn materialization_key(index: u32) -> MaterializationKey {
    MaterializationKey::new(
        ModelInstanceId::new(1),
        SourceIdentityHash::new([2; 32]),
        ContentHash::new([3; 32]),
        MaterializedResourceId::new(MaterializedResourceKind::RoutedExpert, 0, index),
        PayloadEncodingId::new(4),
        BackendId::new(3),
        DeviceId::new(0),
        SourceGeneration::new(5),
        DestinationGeneration::new(6),
    )
    .unwrap()
}
fn leases(keys: &[MaterializationKey]) -> ResidencyLeaseSet {
    ResidencyLeaseSet::new(
        keys.iter().copied(),
        keys.iter().copied().enumerate().map(|(index, key)| {
            ValidatedResidencyBinding::new(
                key,
                ResidencyBinding::new(
                    key.model(),
                    key.resource(),
                    key.backend(),
                    key.device(),
                    DestinationSlotId::new(index as u32 + 1),
                    key.destination_generation(),
                ),
            )
            .unwrap()
        }),
        MappingEpoch::new(1),
        DispatchFenceContract::new(
            OperationId::new(91),
            FenceId::new(92),
            BackendId::new(3),
            DeviceId::new(0),
        ),
    )
    .unwrap()
}
fn empty_leases() -> ResidencyLeaseSet {
    leases(&[])
}
fn continuation_leases() -> ResidencyLeaseSet {
    leases(&[materialization_key(1)])
}
#[test]
fn synchronous_standard_forward_publishes_working_copies_exactly_once() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let mut states = fixture.states;
    let original_topologies = [states[2].topology_id(), states[0].topology_id()];
    let mut backend = MockKvBackend::default();
    backend.commit_progress = VecDeque::from([KvEndProgress::Pending, KvEndProgress::Complete]);
    let mut executor = MockForwardExecutor::new(ForwardScript::Sync);
    let mut registry = Registry::new("mock");
    let active = transaction(1);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    assert!(
        registry
            .ensure_sequence_available(&states[2], "reset")
            .is_err()
    );
    assert!(
        registry
            .ensure_pages_available(&[KvPageId(20)], "release")
            .is_err()
    );
    let progress = registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap();
    let DecoderTransactionProgress::Complete(output) = progress else {
        panic!("synchronous forward unexpectedly waited");
    };
    output
        .validate_with_capabilities(&fixture.batch, &capabilities())
        .unwrap();
    assert_eq!(states[2].core().position(), 0);
    assert_eq!(states[0].core().position(), 4);
    assert_eq!(states[2].attachment(), &30);
    assert_eq!(
        registry.publish(active, &mut states, &mut backend).unwrap(),
        KvEndProgress::Pending
    );
    assert_eq!(states[2].core().position(), 0);
    assert!(registry.contains(active));
    assert_eq!(
        registry.publish(active, &mut states, &mut backend).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(states[2].core().position(), 3);
    assert_eq!(states[0].core().position(), 5);
    assert_eq!(states[2].attachment(), &31);
    assert_eq!(states[0].kv_state(), &110);
    assert_eq!(states[2].topology_id(), original_topologies[0]);
    assert_eq!(states[0].topology_id(), original_topologies[1]);
    assert!(registry.is_empty());
    registry
        .ensure_pages_available(&[KvPageId(20)], "release")
        .unwrap();
}
#[test]
fn execute_and_resume_require_the_prepared_batch_identity() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let mismatched = batch_with_different_first_token(&fixture.batch);
    let mut states = fixture.states;
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::Async);
    let mut registry = Registry::new("mock");
    let active = transaction(101);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    let error = registry
        .execute_batch(active, &states, &mismatched, &mut backend, &mut executor)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("differs from its prepared batch")
    );
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::Prepared
    );
    assert_eq!(backend.events, ["prepare"]);
    let waiting = registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap();
    let DecoderTransactionProgress::Waiting { continuation, .. } = waiting else {
        panic!("asynchronous forward unexpectedly completed");
    };
    let error = registry
        .resume_batch(
            active,
            &states,
            &mismatched,
            continuation,
            continuation_leases(),
            &mut backend,
            &mut executor,
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("differs from its prepared batch")
    );
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::Waiting
    );
    assert_eq!(registry.held_resume_lease_count(active).unwrap(), 0);
    assert!(matches!(
        registry
            .resume_batch(
                active,
                &states,
                &fixture.batch,
                continuation,
                continuation_leases(),
                &mut backend,
                &mut executor,
            )
            .unwrap(),
        DecoderTransactionProgress::Complete(_)
    ));
    assert_eq!(
        registry.publish(active, &mut states, &mut backend).unwrap(),
        KvEndProgress::Complete
    );
}
#[test]
fn async_streaming_forward_retains_dependency_and_lease_custody() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let mut states = fixture.states;
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::Async);
    let mut registry = Registry::new("mock");
    let active = transaction(2);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    let waiting = registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap();
    let DecoderTransactionProgress::Waiting { continuation, wait } = waiting else {
        panic!("asynchronous forward unexpectedly completed");
    };
    assert_eq!(continuation, ContinuationId::new(1));
    assert_eq!(
        match &wait {
            DecoderWait::Dependencies(dependencies) => dependencies.len(),
            DecoderWait::ResolvedStage(stage) => stage
                .dependencies()
                .expect("waiting resolved stage has dependencies")
                .len(),
        },
        1
    );
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::Waiting
    );
    let wrong = registry
        .resume_batch(
            active,
            &states,
            &fixture.batch,
            ContinuationId::new(99),
            empty_leases(),
            &mut backend,
            &mut executor,
        )
        .unwrap_err();
    assert!(wrong.to_string().contains("continuation mismatch"));
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::Waiting
    );
    let incomplete = registry
        .resume_batch(
            active,
            &states,
            &fixture.batch,
            continuation,
            empty_leases(),
            &mut backend,
            &mut executor,
        )
        .unwrap_err();
    assert!(incomplete.to_string().contains("does not exactly satisfy"));
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::Waiting
    );
    assert_eq!(registry.held_resume_lease_count(active).unwrap(), 0);
    let progress = registry
        .resume_batch(
            active,
            &states,
            &fixture.batch,
            continuation,
            continuation_leases(),
            &mut backend,
            &mut executor,
        )
        .unwrap();
    assert!(matches!(progress, DecoderTransactionProgress::Complete(_)));
    assert_eq!(registry.held_resume_lease_count(active).unwrap(), 1);
    assert_eq!(states[2].attachment(), &30);
    assert_eq!(
        registry.publish(active, &mut states, &mut backend).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(states[2].attachment(), &32);
    assert!(registry.is_empty());
}
#[test]
fn missing_kv_after_state_take_remains_cancelable_and_rollbackable() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let states = fixture.states;
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::Async);
    let mut registry = Registry::new("mock");
    let active = transaction(21);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap();

    let kv_transaction = registry.take_kv_transaction(active).unwrap();
    let error = registry
        .resume_batch(
            active,
            &states,
            &fixture.batch,
            ContinuationId::new(1),
            continuation_leases(),
            &mut backend,
            &mut executor,
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("lost its KV transaction while waiting")
    );
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::Waiting
    );
    assert_eq!(registry.held_resume_lease_count(active).unwrap(), 1);

    registry
        .restore_kv_transaction(active, kv_transaction)
        .unwrap();
    assert_eq!(
        registry.abort(active, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(executor.cancel_calls, 1);
    assert!(backend.events.contains(&"leave"));
    assert!(backend.events.contains(&"rollback"));
    assert!(registry.is_empty());
}
#[test]
fn wait_rebuild_failure_remains_cancelable_and_rollbackable() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let states = fixture.states;
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::ResumeWaiting);
    let mut registry = Registry::new("mock");
    let active = transaction(22);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap();
    registry.fail_next_wait_rebuild(active).unwrap();

    let error = registry
        .resume_batch(
            active,
            &states,
            &fixture.batch,
            ContinuationId::new(1),
            continuation_leases(),
            &mut backend,
            &mut executor,
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected decoder wait rebuild failure")
    );
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::FailedActive
    );
    assert_eq!(registry.held_resume_lease_count(active).unwrap(), 1);

    assert_eq!(
        registry.abort(active, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(executor.cancel_calls, 1);
    assert!(backend.events.contains(&"leave"));
    assert!(backend.events.contains(&"rollback"));
    assert!(registry.is_empty());
}
#[test]
fn abort_discards_executed_work_and_can_progress_asynchronously() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let states = fixture.states;
    let mut backend = MockKvBackend::default();
    backend.rollback_progress = VecDeque::from([KvEndProgress::Pending, KvEndProgress::Complete]);
    let mut executor = MockForwardExecutor::new(ForwardScript::Sync);
    let mut registry = Registry::new("mock");
    let active = transaction(3);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap();
    assert_eq!(
        registry.abort(active, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Pending
    );
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::Aborting
    );
    assert_eq!(states[2].core().position(), 0);
    assert_eq!(
        registry.abort(active, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(states[2].attachment(), &30);
    assert!(registry.is_empty());
}
#[test]
fn retained_commit_error_can_be_retried_without_losing_custody() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let mut states = fixture.states;
    let mut backend = MockKvBackend::default();
    backend.commit_failures = VecDeque::from([MockKvFailure::RetainToken]);
    let mut executor = MockForwardExecutor::new(ForwardScript::Sync);
    let mut registry = Registry::new("mock");
    let active = transaction(31);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap();
    let error = registry
        .publish(active, &mut states, &mut backend)
        .unwrap_err();
    assert!(error.to_string().contains("injected commit failure"));
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::Publishing
    );
    assert_eq!(states[2].core().position(), 0);
    assert!(
        registry
            .ensure_pages_available(&[KvPageId(10)], "release")
            .is_err()
    );
    assert_eq!(
        registry.publish(active, &mut states, &mut backend).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(states[2].core().position(), 3);
    assert!(registry.is_empty());
}
#[test]
fn consumed_commit_error_releases_custody_without_publication() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let mut states = fixture.states;
    let mut backend = MockKvBackend::default();
    backend.commit_failures = VecDeque::from([MockKvFailure::ConsumeToken]);
    let mut executor = MockForwardExecutor::new(ForwardScript::Sync);
    let mut registry = Registry::new("mock");
    let active = transaction(32);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap();
    let error = registry
        .publish(active, &mut states, &mut backend)
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("consumed its KV transaction"));
    assert!(message.contains("injected commit failure"));
    assert_eq!(states[2].core().position(), 0);
    assert_eq!(states[2].attachment(), &30);
    assert!(!registry.contains(active));
    registry
        .ensure_sequence_available(&states[2], "reset")
        .unwrap();
    registry
        .ensure_pages_available(&[KvPageId(10)], "release")
        .unwrap();
}
#[test]
fn retained_rollback_error_can_be_retried_without_losing_custody() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let states = fixture.states;
    let mut backend = MockKvBackend::default();
    backend.rollback_failures = VecDeque::from([MockKvFailure::RetainToken]);
    let mut executor = MockForwardExecutor::new(ForwardScript::Sync);
    let mut registry = Registry::new("mock");
    let active = transaction(33);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    let error = registry
        .abort(active, &mut backend, &mut executor)
        .unwrap_err();
    assert!(error.to_string().contains("injected rollback failure"));
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::Aborting
    );
    assert!(
        registry
            .ensure_pages_available(&[KvPageId(10)], "release")
            .is_err()
    );
    assert_eq!(
        registry.abort(active, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Complete
    );
    assert!(registry.is_empty());
}
#[test]
fn consumed_rollback_error_releases_all_registry_custody() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let states = fixture.states;
    let mut backend = MockKvBackend::default();
    backend.rollback_failures = VecDeque::from([MockKvFailure::ConsumeToken]);
    let mut executor = MockForwardExecutor::new(ForwardScript::Sync);
    let mut registry = Registry::new("mock");
    let active = transaction(34);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    let error = registry
        .abort(active, &mut backend, &mut executor)
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("consumed its KV transaction"));
    assert!(message.contains("injected rollback failure"));
    assert!(registry.is_empty());
    registry
        .ensure_sequence_available(&states[2], "reset")
        .unwrap();
    registry
        .ensure_pages_available(&[KvPageId(10)], "release")
        .unwrap();
}
#[test]
fn consumed_commit_rejection_releases_all_registry_custody_without_publication() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let mut states = fixture.states;
    let mut backend = MockKvBackend::default();
    backend.commit_progress = VecDeque::from([KvEndProgress::ConsumedRejected]);
    let mut executor = MockForwardExecutor::new(ForwardScript::Sync);
    let mut registry = Registry::new("mock");
    let active = transaction(4);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap();
    assert_eq!(
        registry.publish(active, &mut states, &mut backend).unwrap(),
        KvEndProgress::ConsumedRejected
    );
    assert_eq!(states[2].core().position(), 0);
    assert_eq!(states[2].attachment(), &30);
    assert!(!registry.contains(active));
    registry
        .ensure_sequence_available(&states[2], "reset")
        .unwrap();
    registry
        .ensure_pages_available(&[KvPageId(20)], "release")
        .unwrap();
}
#[test]
fn failed_active_forward_is_cancelled_to_quiescence_before_rollback() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let states = fixture.states;
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::FailedActive).with_cancel_wait();
    let mut registry = Registry::new("mock");
    let active = transaction(5);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    let error = registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap_err();
    assert!(error.to_string().contains("active forward failure"));
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::FailedActive
    );
    assert_eq!(
        registry.abort(active, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Pending
    );
    assert_eq!(executor.cancel_calls, 1);
    assert!(!backend.events.contains(&"leave"));
    assert!(registry.contains(active));
    assert_eq!(
        registry.abort(active, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(executor.cancel_calls, 2);
    assert!(backend.events.contains(&"leave"));
    assert!(backend.events.contains(&"rollback"));
    assert!(registry.is_empty());
}
#[test]
fn failed_quiescent_forward_skips_cancel_and_rolls_back_directly() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let states = fixture.states;
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::FailedQuiescent);
    let mut registry = Registry::new("mock");
    let active = transaction(51);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    let error = registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap_err();
    assert!(error.to_string().contains("quiescent forward failure"));
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::FailedQuiescent
    );
    assert_eq!(executor.cancel_calls, 0);
    assert!(backend.events.contains(&"leave"));
    assert_eq!(
        registry.abort(active, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(executor.cancel_calls, 0);
}
#[test]
fn resume_error_retains_active_continuation_for_abort_recovery() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let states = fixture.states;
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::ResumeError);
    let mut registry = Registry::new("mock");
    let active = transaction(6);
    registry
        .prepare(active, packed, &states, &mut backend)
        .unwrap();
    registry
        .execute_batch(active, &states, &fixture.batch, &mut backend, &mut executor)
        .unwrap();
    let error = registry
        .resume_batch(
            active,
            &states,
            &fixture.batch,
            ContinuationId::new(1),
            continuation_leases(),
            &mut backend,
            &mut executor,
        )
        .unwrap_err();
    assert!(error.to_string().contains("resume failure"));
    assert_eq!(
        registry.phase(active).unwrap(),
        DecoderTransactionPhase::FailedActive
    );
    assert_eq!(registry.held_resume_lease_count(active).unwrap(), 1);
    assert_eq!(
        registry.abort(active, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(executor.cancel_calls, 1);
}
#[test]
fn registry_rejects_writer_writer_conflicts_across_distinct_topologies() {
    let states = vec![State::with_position(4, 1, 1), State::with_position(4, 2, 2)];
    let first = single_sequence_fixture(
        states.clone(),
        0,
        20,
        4,
        vec![50, 60],
        vec![KvPageId(60)],
        None,
        &[KvPageId(50)],
    );
    let second = single_sequence_fixture(
        states.clone(),
        1,
        21,
        4,
        vec![51, 60],
        vec![KvPageId(60)],
        None,
        &[KvPageId(51)],
    );
    assert_ne!(states[0].topology_id(), states[1].topology_id());
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::Sync);
    let mut registry = Registry::new("mock");
    let first_id = transaction(40);
    registry
        .prepare(first_id, first.lower().unwrap(), &states, &mut backend)
        .unwrap();
    let error = registry
        .prepare(
            transaction(41),
            second.lower().unwrap(),
            &states,
            &mut backend,
        )
        .unwrap_err();
    assert!(error.to_string().contains("already mutates it"));
    assert_eq!(backend.prepared, [first_id]);
    let error = registry
        .ensure_pages_available(&[KvPageId(60)], "release")
        .unwrap_err();
    assert!(error.to_string().contains("write custody"));
    registry
        .abort(first_id, &mut backend, &mut executor)
        .unwrap();
    registry
        .ensure_pages_available(&[KvPageId(60)], "release")
        .unwrap();
}
#[test]
fn registry_allows_shared_prefix_reads_but_excludes_sequence_and_page_writers() {
    let states = vec![
        State::with_position(4, 1, 1),
        State::with_position(4, 2, 2),
        State::with_position(2, 3, 3),
    ];
    let first = single_sequence_fixture(
        states.clone(),
        0,
        10,
        4,
        vec![50, 60],
        vec![KvPageId(60)],
        None,
        &[KvPageId(50)],
    );
    let second = single_sequence_fixture(
        states.clone(),
        1,
        11,
        4,
        vec![50, 61],
        vec![KvPageId(61)],
        None,
        &[KvPageId(50)],
    );
    let writer = single_sequence_fixture(
        states.clone(),
        2,
        12,
        2,
        vec![50],
        vec![],
        None,
        &[KvPageId(50)],
    );
    let repeated_sequence = single_sequence_fixture(
        states.clone(),
        0,
        13,
        4,
        vec![50, 62],
        vec![KvPageId(62)],
        None,
        &[KvPageId(50)],
    );
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::Sync);
    let mut registry = Registry::new("mock");
    let first_id = transaction(10);
    let second_id = transaction(11);
    registry
        .prepare(first_id, first.lower().unwrap(), &states, &mut backend)
        .unwrap();
    registry
        .prepare(second_id, second.lower().unwrap(), &states, &mut backend)
        .unwrap();
    assert_eq!(registry.len(), 2);
    assert!(
        registry
            .prepare(
                transaction(12),
                repeated_sequence.lower().unwrap(),
                &states,
                &mut backend,
            )
            .unwrap_err()
            .to_string()
            .contains("already owned")
    );
    assert!(
        registry
            .prepare(
                transaction(13),
                writer.lower().unwrap(),
                &states,
                &mut backend,
            )
            .unwrap_err()
            .to_string()
            .contains("cannot mutate KV page 50")
    );
    assert!(
        registry
            .ensure_pages_available(&[KvPageId(50)], "release")
            .is_err()
    );
    registry
        .abort(first_id, &mut backend, &mut executor)
        .unwrap();
    assert!(
        registry
            .ensure_pages_available(&[KvPageId(50)], "release")
            .is_err()
    );
    registry
        .abort(second_id, &mut backend, &mut executor)
        .unwrap();
    registry
        .ensure_pages_available(&[KvPageId(50)], "release")
        .unwrap();
}
#[test]
fn no_proposal_executor_is_explicitly_target_only() {
    let mut executor = NoProposal;
    let mut state = State::new(0, 0);
    let control = DecoderControlPlane::new(ferrule_common::CompletionHub::new());
    let mut context = DecoderTransactionContext::new(transaction(1), control, Vec::new());
    assert!(
        <NoProposal as DecoderProposalExecutor<State, MockKvBackend>>::source(&executor)
            .unwrap()
            .is_none()
    );
    assert!(
        <NoProposal as DecoderProposalExecutor<State, MockKvBackend>>::start(
            &mut executor,
            &mut context,
            &mut state,
            &mut (),
            1,
        )
        .is_err()
    );
}
mod synthetic_end_to_end {
    use super::*;
    use super::{
        CpuPagedKvBackend, CpuPagedKvPool, GenericDecoderOptions, GenericDecoderRunner,
        GenericDecoderSequenceState, PagedKvBackend, StandardGqaPlanes,
    };
    use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
    use crate::execution::ExecutionPrecisionPolicy;
    use crate::runner::{
        ModelRunner, MultiSessionBatchProgress, MultiSessionRunner, ResidentModelRunner,
        TransactionEndIntent, TransactionEndProgress,
    };
    use crate::spec::{ModelFamily, WeightSource};
    use crate::transformer::{DecoderLoadOptions, DecoderRecipe, SyntheticDecoderRecipe};
    use crate::{TensorRole, TokenizerHandle};
    use ferrule_common::execution::{ExecutionOutput, KvElementType, LogitsOutput};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    const PAGE_SIZE: usize = 2;
    struct SyntheticFile {
        directory: PathBuf,
        path: PathBuf,
    }
    impl SyntheticFile {
        fn new() -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let directory = std::env::temp_dir().join(format!(
                "ferrule-generic-decoder-e2e-{}-{id}",
                std::process::id()
            ));
            std::fs::create_dir_all(&directory).unwrap();
            let path = directory.join("weights.bin");
            Self { directory, path }
        }
    }
    impl Drop for SyntheticFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }
    #[derive(Clone)]
    struct RuntimeSequence {
        state_index: usize,
        phase: ForwardPhase,
        tokens: Vec<u32>,
        pages: Vec<KvPageId>,
        newly_allocated: Vec<KvPageId>,
        cow_replacement: Option<KvCowReplacement>,
        logits: Vec<LogitsRequest>,
    }
    #[test]
    fn recipe_bound_standard_cpu_runner_executes_all_protocol_modes() {
        let config = synthetic_config();
        let recipe = SyntheticDecoderRecipe::new();
        let files = SyntheticFile::new();
        let resources = bind_resources(&recipe, &config, &files);
        assert_eq!(resources.spec().architecture(), "synthetic-decoder");
        assert_eq!(resources.spec().layers().len(), 2);
        assert!(resources.state_dict().validate_source_identities());
        let planes = StandardGqaPlanes::new(2, 1, 2, PAGE_SIZE, 16, KvElementType::Bf16).unwrap();
        let pool = CpuPagedKvPool::from_strategy(&planes, 16).unwrap();
        let backend: CpuPagedKvBackend = PagedKvBackend::new(pool);
        let options = GenericDecoderOptions::standard_cpu(
            resources.spec(),
            ModelFamily::Unknown("synthetic".into()),
            WeightSource::Safetensors,
            PAGE_SIZE,
            16,
            16,
            4,
            4096,
            ExecutionPrecisionPolicy::bf16_compatibility(),
        )
        .unwrap();
        let mut runner =
            GenericDecoderRunner::new(resources, tokenizer(), backend, options).unwrap();
        let info = ModelRunner::model_info(&runner);
        assert_eq!(info.num_layers, 2);
        assert_eq!(info.num_experts, 2);
        assert_eq!(ModelRunner::bound_layer_count(&runner), Some(2));
        let encoded = ModelRunner::encode(&runner, "a").unwrap();
        assert!(!encoded.is_empty());
        ModelRunner::decode(&runner, &encoded).unwrap();
        assert!(MultiSessionRunner::multi_session_capabilities(&runner).supports_mixed);
        assert!(
            ResidentModelRunner::native_proposal_source(&runner)
                .unwrap()
                .is_none()
        );
        assert!(ResidentModelRunner::take_completion_reactors(&mut runner).is_empty());
        let completion_hub = ResidentModelRunner::completion_hub(&runner);
        let mut states = vec![MultiSessionRunner::create_sequence_state(&mut runner).unwrap()];
        let (prefill, reservations) = runtime_batch(
            &states,
            ForwardMode::Prefill,
            vec![RuntimeSequence {
                state_index: 0,
                phase: ForwardPhase::Prefill,
                tokens: vec![1, 2],
                pages: vec![KvPageId(10)],
                newly_allocated: vec![KvPageId(10)],
                cow_replacement: None,
                logits: vec![LogitsRequest::None, top_k(3)],
            }],
        );
        let prefill_output = execute(&mut runner, &mut states, tx(1), &prefill, &reservations);
        assert_eq!(states[0].core().position(), 0);
        publish(&mut runner, &mut states, tx(1));
        assert_eq!(states[0].core().position(), 2);
        assert_top_k(&prefill_output, 3);
        assert!(
            MultiSessionRunner::end_transaction(
                &mut runner,
                tx(1),
                &mut states,
                TransactionEndIntent::Publish,
            )
            .is_err()
        );
        let (decode, reservations) = runtime_batch(
            &states,
            ForwardMode::Decode,
            vec![RuntimeSequence {
                state_index: 0,
                phase: ForwardPhase::Decode,
                tokens: vec![3],
                pages: vec![KvPageId(10), KvPageId(11)],
                newly_allocated: vec![KvPageId(11)],
                cow_replacement: None,
                logits: vec![LogitsRequest::Full],
            }],
        );
        let decode_output = execute(&mut runner, &mut states, tx(2), &decode, &reservations);
        publish(&mut runner, &mut states, tx(2));
        assert_eq!(states[0].core().position(), 3);
        assert_eq!(decode_output.logits.len(), 1);
        assert!(matches!(
            decode_output.logits[0].logits,
            LogitsOutput::Full(_)
        ));
        let fork =
            MultiSessionRunner::fork_sequence_state_from(&mut runner, &states[0], 3).unwrap();
        assert_ne!(fork.topology_id(), states[0].topology_id());
        states.push(fork);
        let cow = KvCowReplacement {
            logical_page: 1,
            source: KvPageId(11),
            replacement: KvPageId(12),
        };
        let (fork_decode, reservations) = runtime_batch(
            &states,
            ForwardMode::Decode,
            vec![RuntimeSequence {
                state_index: 1,
                phase: ForwardPhase::Decode,
                tokens: vec![4],
                pages: vec![KvPageId(10), KvPageId(12)],
                newly_allocated: vec![],
                cow_replacement: Some(cow),
                logits: vec![top_k(2)],
            }],
        );
        let fork_output = execute(&mut runner, &mut states, tx(3), &fork_decode, &reservations);
        publish(&mut runner, &mut states, tx(3));
        assert_eq!(states[0].core().position(), 3);
        assert_eq!(states[1].core().position(), 4);
        assert_top_k(&fork_output, 2);
        let sibling =
            MultiSessionRunner::fork_sequence_state_from(&mut runner, &states[0], 3).unwrap();
        states.push(sibling);
        let (writer, writer_reservations) = runtime_batch(
            &states,
            ForwardMode::Decode,
            vec![RuntimeSequence {
                state_index: 0,
                phase: ForwardPhase::Decode,
                tokens: vec![5],
                pages: vec![KvPageId(10), KvPageId(11)],
                newly_allocated: vec![],
                cow_replacement: None,
                logits: vec![top_k(1)],
            }],
        );
        MultiSessionRunner::prepare_multi_session_batch(
            &mut runner,
            tx(4),
            &mut states,
            &writer,
            &writer_reservations,
        )
        .unwrap();
        assert!(MultiSessionRunner::release_kv_pages(&mut runner, &[KvPageId(11)]).is_err());
        let (conflicting_writer, conflicting_reservations) = runtime_batch(
            &states,
            ForwardMode::Decode,
            vec![RuntimeSequence {
                state_index: 2,
                phase: ForwardPhase::Decode,
                tokens: vec![6],
                pages: vec![KvPageId(10), KvPageId(11)],
                newly_allocated: vec![],
                cow_replacement: None,
                logits: vec![top_k(1)],
            }],
        );
        let conflict = MultiSessionRunner::prepare_multi_session_batch(
            &mut runner,
            tx(5),
            &mut states,
            &conflicting_writer,
            &conflicting_reservations,
        )
        .unwrap_err();
        assert!(conflict.to_string().contains("already mutates"));
        let mismatched = different_first_token(&writer);
        assert!(
            MultiSessionRunner::execute_multi_session_batch_progress(
                &mut runner,
                tx(4),
                &mut states,
                &mismatched,
            )
            .unwrap_err()
            .to_string()
            .contains("differs from its prepared batch")
        );
        assert!(matches!(
            MultiSessionRunner::execute_multi_session_batch_progress(
                &mut runner,
                tx(4),
                &mut states,
                &writer,
            )
            .unwrap(),
            MultiSessionBatchProgress::Complete(_)
        ));
        abort(&mut runner, &mut states, tx(4));
        assert_eq!(states[0].core().position(), 3);
        states.push(MultiSessionRunner::create_sequence_state(&mut runner).unwrap());
        let (mixed, reservations) = runtime_batch(
            &states,
            ForwardMode::Mixed,
            vec![
                RuntimeSequence {
                    state_index: 0,
                    phase: ForwardPhase::Decode,
                    tokens: vec![6],
                    pages: vec![KvPageId(10), KvPageId(11)],
                    newly_allocated: vec![],
                    cow_replacement: None,
                    logits: vec![top_k(2)],
                },
                RuntimeSequence {
                    state_index: 3,
                    phase: ForwardPhase::Prefill,
                    tokens: vec![7, 1],
                    pages: vec![KvPageId(13)],
                    newly_allocated: vec![KvPageId(13)],
                    cow_replacement: None,
                    logits: vec![LogitsRequest::None, LogitsRequest::Full],
                },
            ],
        );
        let mixed_output = execute(&mut runner, &mut states, tx(6), &mixed, &reservations);
        assert_eq!(mixed_output.logits.len(), 2);
        abort(&mut runner, &mut states, tx(6));
        assert_eq!(states[0].core().position(), 3);
        assert_eq!(states[3].core().position(), 0);
        assert_eq!(
            DecoderKvBackend::page_status(runner.backend(), KvPageId(13)),
            DecoderKvPageStatus::Vacant
        );
        let snapshot = ResidentModelRunner::observability_snapshot(&runner);
        assert_eq!(snapshot.bound_layers, 2);
        assert_eq!(snapshot.commits, 3);
        assert_eq!(snapshot.aborts, 2);
        runner.shutdown().unwrap();
        assert!(completion_hub.is_closed());
    }
    fn synthetic_config() -> serde_json::Value {
        serde_json::json!({
            "vocab_size": 8,
            "hidden_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "intermediate_size": 4,
            "num_experts": 2,
            "experts_per_token": 1,
            "max_position_embeddings": 16,
            "rms_norm_eps": 0.00001,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        })
    }
    fn bind_resources(
        recipe: &SyntheticDecoderRecipe,
        config: &serde_json::Value,
        files: &SyntheticFile,
    ) -> crate::transformer::BoundDecoderResources {
        let recipe_output = recipe.build(config).unwrap();
        let mut payload = Vec::new();
        let mut slices = Vec::new();
        for parameter in recipe_output
            .schema()
            .parameters()
            .iter()
            .filter(|parameter| parameter.alias_of().is_none())
        {
            let values = tensor_values(parameter.path().as_str(), parameter.shape());
            let offset = u64::try_from(payload.len()).unwrap();
            payload.extend(values.iter().flat_map(|value| value.to_le_bytes()));
            let bytes = u64::try_from(values.len() * std::mem::size_of::<f32>()).unwrap();
            let external_name = SyntheticDecoderRecipe::external_name(parameter.path());
            assert!(
                recipe_output
                    .name_mapper()
                    .map(
                        &external_name,
                        crate::transformer::ExternalTensorMeta {
                            dtype: "F32",
                            shape: parameter.shape(),
                            bytes,
                        },
                    )
                    .unwrap()
                    .is_some()
            );
            slices.push(CheckpointTensorSlice {
                name: external_name,
                role: TensorRole::Unknown,
                path: files.path.clone(),
                offset,
                bytes,
                dtype: CheckpointDType::F32,
                shape: parameter.shape().to_vec(),
            });
        }
        std::fs::write(&files.path, payload).unwrap();
        DecoderLoadOptions::new(recipe, config)
            .bind_slices(slices)
            .unwrap()
    }
    fn tensor_values(path: &str, shape: &[usize]) -> Vec<f32> {
        let elements = shape.iter().product::<usize>();
        if shape.len() == 1 {
            return vec![1.0; elements];
        }
        let rows = shape[0];
        let columns = shape[1];
        let mut values = vec![0.0; elements];
        for row in 0..rows {
            for column in 0..columns {
                values[row * columns + column] = if path == "token_embedding.weight" {
                    (row + 1) as f32 * (column + 1) as f32 * 0.025
                } else if path == "output.weight" {
                    (row + 1) as f32 * (column + 1) as f32 * 0.015
                } else if path.ends_with("router.weight") {
                    if row == 0 { 0.05 } else { -0.05 }
                } else if row % columns == column {
                    0.2
                } else {
                    0.01
                };
            }
        }
        values
    }
    fn tokenizer() -> TokenizerHandle {
        let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
            .into_iter()
            .map(|token| tokenizers::AddedToken::from(token, false));
        assert_eq!(tokenizer.add_tokens(tokens).unwrap(), 8);
        TokenizerHandle::from_parts(tokenizer, Some(7))
    }
    fn runtime_batch(
        states: &[GenericDecoderSequenceState],
        mode: ForwardMode,
        sequences: Vec<RuntimeSequence>,
    ) -> (ExecutionBatch, Vec<KvReservationView>) {
        let mut token_ids = Vec::new();
        let mut positions = Vec::new();
        let mut write_slots = Vec::new();
        let mut logits = Vec::new();
        let mut execution_sequences = Vec::new();
        let mut block_ids = Vec::new();
        let mut reservations = Vec::new();
        for sequence in sequences {
            let state = &states[sequence.state_index];
            let context_len = state.core().position();
            let sequence_len = context_len + sequence.tokens.len();
            assert_eq!(sequence.pages.len(), sequence_len.div_ceil(PAGE_SIZE));
            assert_eq!(sequence.logits.len(), sequence.tokens.len());
            let query_start = u32::try_from(token_ids.len()).unwrap();
            let block_start = u32::try_from(block_ids.len()).unwrap();
            block_ids.extend(sequence.pages.iter().map(|page| KvBlockId::new(page.0)));
            for (offset, token) in sequence.tokens.into_iter().enumerate() {
                let position = context_len + offset;
                let page = sequence.pages[position / PAGE_SIZE];
                let write = usize::try_from(page.0).unwrap() * PAGE_SIZE + position % PAGE_SIZE;
                token_ids.push(token);
                positions.push(u32::try_from(position).unwrap());
                write_slots.push(Some(KvWriteSlot::try_from(write).unwrap()));
            }
            logits.extend(sequence.logits);
            let query_end = u32::try_from(token_ids.len()).unwrap();
            let block_end = u32::try_from(block_ids.len()).unwrap();
            let state_slot = StateSlot::try_from(sequence.state_index).unwrap();
            execution_sequences.push(ExecutionSequence::new(
                state_slot,
                sequence.phase,
                query_start..query_end,
                u32::try_from(context_len).unwrap(),
                u32::try_from(sequence_len).unwrap(),
                block_start..block_end,
            ));
            reservations.push(KvReservationView {
                state_slot,
                execution_state_slot: state_slot,
                positions: context_len..sequence_len,
                newly_allocated: sequence.newly_allocated,
                generation: state.core().generation(),
                execution_generation: state.core().generation(),
                cow_replacement: sequence.cow_replacement,
            });
        }
        (
            ExecutionBatch::new(
                mode,
                token_ids,
                positions,
                write_slots,
                logits,
                execution_sequences,
                block_ids,
            ),
            reservations,
        )
    }
    fn execute(
        runner: &mut GenericDecoderRunner<CpuPagedKvBackend>,
        states: &mut [GenericDecoderSequenceState],
        transaction: ExecutionTransactionId,
        batch: &ExecutionBatch,
        reservations: &[KvReservationView],
    ) -> ExecutionOutput {
        MultiSessionRunner::prepare_multi_session_batch(
            runner,
            transaction,
            states,
            batch,
            reservations,
        )
        .unwrap();
        match MultiSessionRunner::execute_multi_session_batch_progress(
            runner,
            transaction,
            states,
            batch,
        )
        .unwrap()
        {
            MultiSessionBatchProgress::Complete(output) => output,
            MultiSessionBatchProgress::Waiting(_) => {
                panic!("standard CPU decoder unexpectedly suspended")
            }
        }
    }
    fn publish(
        runner: &mut GenericDecoderRunner<CpuPagedKvBackend>,
        states: &mut [GenericDecoderSequenceState],
        transaction: ExecutionTransactionId,
    ) {
        assert_eq!(
            MultiSessionRunner::end_transaction(
                runner,
                transaction,
                states,
                TransactionEndIntent::Publish,
            )
            .unwrap(),
            TransactionEndProgress::Complete
        );
    }
    fn abort(
        runner: &mut GenericDecoderRunner<CpuPagedKvBackend>,
        states: &mut [GenericDecoderSequenceState],
        transaction: ExecutionTransactionId,
    ) {
        assert_eq!(
            MultiSessionRunner::end_transaction(
                runner,
                transaction,
                states,
                TransactionEndIntent::Abort,
            )
            .unwrap(),
            TransactionEndProgress::Complete
        );
    }
    fn different_first_token(batch: &ExecutionBatch) -> ExecutionBatch {
        let mut token_ids = batch.token_ids().to_vec();
        token_ids[0] ^= 1;
        ExecutionBatch::new(
            batch.mode(),
            token_ids,
            batch.positions().to_vec(),
            batch.kv_write_slots().to_vec(),
            batch.logits().to_vec(),
            batch.sequences().to_vec(),
            batch.kv_block_ids().to_vec(),
        )
    }
    fn assert_top_k(output: &ExecutionOutput, expected: usize) {
        let LogitsOutput::TopK(candidates) = &output.logits[0].logits else {
            panic!("expected top-k output")
        };
        assert_eq!(candidates.len(), expected);
        for pair in candidates.windows(2) {
            assert!(
                pair[0].logit > pair[1].logit
                    || (pair[0].logit == pair[1].logit && pair[0].token_id < pair[1].token_id)
            );
        }
    }
    fn top_k(width: u32) -> LogitsRequest {
        LogitsRequest::TopK(NonZeroU32::new(width).unwrap())
    }
    fn tx(value: u64) -> ExecutionTransactionId {
        ExecutionTransactionId::new(value).unwrap()
    }
}
