//! CPU-only simulation of submission failures; the real ledger is not mocked.
use super::*;
use ferrule_common::execution::{ExecutionTransactionId, KvCowReplacement};

#[derive(Clone, Copy)]
enum Failure {
    None,
    Validation,
    Cleaned,
    Unknown,
    TypedUnknown,
}
struct Pool {
    cpu: CpuPagedKvPool,
    failure: Failure,
    retained: Option<CpuPagedKvTransaction>,
    submissions: usize,
}
impl PhysicalKvPool for Pool {
    type SequenceState = DecoderSequenceState<(), ()>;
    type Transaction = CpuPagedKvTransaction;
    type KvView = CpuKvView;
    fn configured_capacity(&self) -> usize {
        self.cpu.configured_capacity()
    }
    fn configure_capacity(&mut self, n: usize) -> Result<()> {
        self.cpu.configure_capacity(n)
    }
    fn prepare_failure_retains_custody(&self) -> bool {
        self.retained.is_some() && !matches!(self.failure, Failure::TypedUnknown)
    }
    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        if matches!(self.failure, Failure::Validation) {
            return Err(kv_error("injected validation failure"));
        }
        self.submissions += 1;
        let mut tx = Some(self.cpu.prepare(request)?);
        match self.failure {
            Failure::None => Ok(tx.take().unwrap()),
            Failure::Unknown | Failure::TypedUnknown => {
                self.retained = tx;
                let source = kv_error("injected submission/fence failure");
                Err(if matches!(self.failure, Failure::TypedUnknown) {
                    KvPrepareQuiescenceUnknown::wrap(request.transaction, source)
                } else {
                    source
                })
            }
            Failure::Cleaned => {
                assert_eq!(self.cpu.abort(&mut tx)?, KvEndProgress::Complete);
                Err(kv_error("injected failure after proven cleanup"))
            }
            Failure::Validation => unreachable!(),
        }
    }
    fn enter(
        &mut self,
        tx: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        states: &mut [Self::SequenceState],
    ) -> Result<()> {
        self.cpu.enter(tx, batch, states)
    }
    fn active_view(
        &mut self,
        tx: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
    ) -> Result<Self::KvView> {
        self.cpu.active_view(tx, batch)
    }
    fn leave(&mut self, tx: &mut Self::Transaction) -> Result<()> {
        self.cpu.leave(tx)
    }
    fn commit(&mut self, tx: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        self.cpu.commit(tx)
    }
    fn abort(&mut self, tx: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        self.cpu.abort(tx)
    }
    fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.cpu.release(pages)
    }
    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.cpu.preempt(pages)
    }
    fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.cpu.restore(pages)
    }
    fn shutdown(&mut self) -> Result<()> {
        self.cpu.shutdown()
    }
}
fn backend() -> PagedKvBackend<Pool> {
    let schema = StandardGqaPlanes::new(1, 1, 1, 4, 16, KvElementType::F32).unwrap();
    PagedKvBackend::new(Pool {
        cpu: CpuPagedKvPool::from_strategy(&schema, 4).unwrap(),
        failure: Failure::None,
        retained: None,
        submissions: 0,
    })
}
fn request<'a>(n: u64, pages: &'a [KvPageId], capacity: DecoderKvCapacity) -> DecoderKvPrepare<'a> {
    DecoderKvPrepare {
        transaction: ExecutionTransactionId::new(n).unwrap(),
        sequences: &[],
        new_pages: pages,
        writable_pages: &[],
        cow_replacements: &[],
        protected_pages: &[],
        capacity,
        page_statuses: &[],
    }
}

#[test]
fn unknown_prepare_keeps_original_ledger_cow_source_and_provisional_pins() {
    let mut backend = backend();
    let source = KvPageId(1);
    let mut seed = Some(
        backend
            .prepare(request(1, &[source], backend.capacity()))
            .unwrap(),
    );
    assert_eq!(backend.commit(&mut seed).unwrap(), KvEndProgress::Complete);
    let mut clone = backend.pool().cpu.clone();
    let cows = [KvCowReplacement {
        logical_page: 0,
        source,
        replacement: KvPageId(2),
    }];
    backend.pool_mut().failure = Failure::Unknown;
    let mut prepare = request(2, &[KvPageId(3)], backend.capacity());
    prepare.cow_replacements = &cows;
    let id = prepare.transaction;
    let err = backend.prepare(prepare).unwrap_err();
    assert_eq!(
        KvPrepareQuiescenceUnknown::from_error(&err)
            .unwrap()
            .transaction(),
        id
    );
    let record = backend.ownership().unwrap().transaction(id).unwrap();
    assert!(record.payload().is_none()); // No fabricated physical handle.
    assert_eq!(
        record.protected_pages,
        [source, KvPageId(2), KvPageId(3)].into()
    );
    assert_eq!(record.cow_replacements.as_ref(), &cows);
    let nonce = record.handle_nonce;
    let capacity = backend.capacity();
    assert_eq!(capacity.active_transactions, 1);
    assert_eq!(capacity.free_pages, 1);
    assert_eq!(backend.pool().cpu.active_transaction_count(), 1);
    drop(err);
    assert!(backend.release(&[source]).is_err());
    assert!(backend.preempt(&[source]).is_err());
    assert!(backend.restore(&[source]).is_err());
    assert!(clone.release(&[source]).is_err());
    assert!(clone.shutdown().is_err());
    assert!(backend.shutdown().is_err());
    assert!(backend.configure_capacity(4).is_err());
    assert!(backend.rollback(&mut None).is_err());
    for n in [2, 3] {
        let err = backend
            .prepare(request(n, &[KvPageId(2)], backend.capacity()))
            .unwrap_err();
        assert!(KvPrepareQuiescenceUnknown::from_error(&err).is_some());
    }
    assert_eq!(backend.pool().submissions, 2); // Seed + failed submission; no retry dispatch.
    assert_eq!(backend.capacity(), capacity);
    assert_eq!(
        backend
            .ownership()
            .unwrap()
            .transaction(id)
            .unwrap()
            .handle_nonce,
        nonce
    );
}

#[test]
fn typed_unknown_without_pool_flag_blocks_retries_from_the_retained_ledger() {
    let mut backend = backend();
    backend.pool_mut().failure = Failure::TypedUnknown;
    let id = ExecutionTransactionId::new(1).unwrap();
    let err = backend
        .prepare(request(1, &[KvPageId(1)], backend.capacity()))
        .unwrap_err();
    assert_eq!(
        KvPrepareQuiescenceUnknown::from_error(&err)
            .unwrap()
            .transaction(),
        id
    );
    assert!(!backend.pool().prepare_failure_retains_custody());
    let nonce = backend
        .ownership()
        .unwrap()
        .transaction(id)
        .unwrap()
        .handle_nonce;
    let capacity = backend.capacity();
    drop(err);
    for n in [1, 2] {
        let err = backend
            .prepare(request(n, &[KvPageId(2)], capacity))
            .unwrap_err();
        assert!(KvPrepareQuiescenceUnknown::from_error(&err).is_some());
    }
    assert_eq!(backend.pool().submissions, 1);
    assert_eq!(backend.capacity(), capacity);
    let retained = backend.ownership().unwrap().transaction(id).unwrap();
    assert_eq!(retained.handle_nonce, nonce);
    assert!(retained.payload().is_none());
    assert_eq!(retained.protected_pages, [KvPageId(1)].into());
    assert!(backend.shutdown().is_err());
}

#[test]
fn validation_and_proven_cleanup_errors_leave_no_ledger_and_can_retry() {
    for failure in [Failure::Validation, Failure::Cleaned] {
        let mut backend = backend();
        let before = backend.capacity();
        backend.pool_mut().failure = failure;
        let err = backend
            .prepare(request(1, &[KvPageId(1)], before))
            .unwrap_err();
        assert!(KvPrepareQuiescenceUnknown::from_error(&err).is_none());
        assert!(backend.ownership().unwrap().is_quiescent());
        assert_eq!(backend.capacity(), before);
        assert_eq!(backend.pool().cpu.free_slot_count(), 4);
        assert_eq!(backend.pool().cpu.active_transaction_count(), 0);
        backend.pool_mut().failure = Failure::None;
        let mut tx = Some(backend.prepare(request(1, &[KvPageId(1)], before)).unwrap());
        assert_eq!(backend.rollback(&mut tx).unwrap(), KvEndProgress::Complete);
        backend.shutdown().unwrap();
    }
}

#[test]
fn unknown_type_survives_error_context_cleanup_and_batches() {
    let id = ExecutionTransactionId::new(17).unwrap();
    let unknown = KvPrepareQuiescenceUnknown::wrap(id, kv_error("device failure"));
    let context = ferrule_common::Error::context("prepare", unknown);
    let cleanup =
        ferrule_common::Error::with_cleanup("cleanup", kv_error("validation"), Err(context));
    let batch =
        ferrule_common::Error::failures("failures", vec![kv_error("other"), cleanup]).unwrap_err();
    assert_eq!(
        KvPrepareQuiescenceUnknown::from_error(&batch)
            .unwrap()
            .transaction(),
        id
    );
}
