//! Exercise the production scope and transaction shell without physical CUDA.
use super::*;
use std::cell::Cell;

// This validator can only exercise invocation policy. It implements no backend
// authority and cannot manufacture an append ticket or resource capability.
struct LiveOwner {
    owner: Cell<u64>,
    generation: Cell<u64>,
}
struct BorrowedInvocation<'a> {
    live: &'a LiveOwner,
    transaction: ExecutionTransactionId,
    generation: u64,
}
impl DecoderInvocation for BorrowedInvocation<'_> {
    fn validate_batch(
        &self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) -> Result<()> {
        if transaction != self.transaction
            || batch.sequences()[0].page_generation() != self.generation
        {
            return Err(failure("foreign transaction or generation"));
        }
        self.validate_layer(0)
    }
    fn layer_count(&self) -> usize {
        2
    }
    fn validate_layer(&self, layer: usize) -> Result<()> {
        if layer >= 2 || self.live.owner.get() != 1 || self.live.generation.get() != self.generation
        {
            return Err(failure("stale live owner or generation"));
        }
        Ok(())
    }
}
fn failure(message: &str) -> Error {
    Error::Execution {
        message: message.into(),
    }
}
fn owner(batch: &PackedDecoderBatch) -> LiveOwner {
    LiveOwner {
        owner: Cell::new(1),
        generation: Cell::new(batch.sequences()[0].page_generation()),
    }
}
fn borrow(live: &LiveOwner) -> BorrowedInvocation<'_> {
    BorrowedInvocation {
        live,
        transaction: transaction(1),
        generation: live.generation.get(),
    }
}
fn retire(scope: &DecoderInvocationScope<'_>, layer: usize) {
    scope.begin_layer(layer).unwrap();
    scope.retire_layer(layer).unwrap();
}

#[test]
fn invocation_checks_foreign_transaction_generation_and_live_owner_before_enter() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let live = owner(&packed);
    let invocation = borrow(&live);
    let scope = DecoderInvocationScope::new(&invocation).unwrap();
    assert!(scope.begin(transaction(2), &packed).is_err());
    let foreign = BorrowedInvocation {
        generation: invocation.generation + 1,
        ..borrow(&live)
    };
    assert!(
        DecoderInvocationScope::new(&foreign)
            .unwrap()
            .begin(transaction(1), &packed)
            .is_err()
    );
    live.owner.set(2);
    assert!(scope.begin(transaction(1), &packed).is_err());
    live.owner.set(1);
    live.generation.set(invocation.generation + 1);
    assert!(scope.begin(transaction(1), &packed).is_err());
    live.generation.set(invocation.generation);
    scope.begin(transaction(1), &packed).unwrap();
}

#[test]
fn invocation_revalidates_live_generation_at_start_retirement_and_finish() {
    let packed = mixed_fixture().lower().unwrap();
    let live = owner(&packed);
    let invocation = borrow(&live);
    let scope = DecoderInvocationScope::new(&invocation).unwrap();
    scope.begin(transaction(1), &packed).unwrap();
    live.generation.set(invocation.generation + 1);
    assert!(scope.begin_layer(0).is_err());
    live.generation.set(invocation.generation);
    scope.begin_layer(0).unwrap();
    live.generation.set(invocation.generation + 1);
    assert!(scope.retire_layer(0).is_err());
    assert!(scope.begin_layer(1).is_err());
    live.generation.set(invocation.generation);
    scope.retire_layer(0).unwrap();
    retire(&scope, 1);
    live.generation.set(invocation.generation + 1);
    assert!(scope.finish().is_err());
    live.generation.set(invocation.generation);
    scope.finish().unwrap();
    live.owner.set(2);
    assert!(scope.finish().is_err());
}

#[test]
fn invocation_rejects_order_replay_unretired_output_and_incomplete_finish() {
    let packed = mixed_fixture().lower().unwrap();
    let live = owner(&packed);
    let invocation = borrow(&live);
    let scope = DecoderInvocationScope::new(&invocation).unwrap();
    assert!(scope.begin_layer(0).is_err());
    assert!(scope.finish().is_err());
    scope.begin(transaction(1), &packed).unwrap();
    assert!(scope.begin(transaction(1), &packed).is_err());
    assert!(scope.begin_layer(1).is_err());
    assert!(scope.retire_layer(0).is_err());
    scope.begin_layer(0).unwrap();
    assert!(scope.begin_layer(0).is_err());
    assert!(scope.begin_layer(1).is_err());
    assert!(scope.retire_layer(1).is_err());
    assert!(scope.finish().is_err());
    scope.retire_layer(0).unwrap();
    assert!(scope.begin_layer(0).is_err());
    assert!(scope.retire_layer(0).is_err());
    assert!(scope.finish().is_err());
    retire(&scope, 1);
    scope.finish().unwrap();
    assert!(scope.begin_layer(2).is_err());
    assert!(scope.begin(transaction(1), &packed).is_err());
}

#[test]
fn invocation_waiting_and_unknown_keep_failed_active_custody_until_abort_ack() {
    for script in [ForwardScript::Async, ForwardScript::FailedActive] {
        let fixture = mixed_fixture();
        let packed = fixture.lower().unwrap();
        let live = owner(&packed);
        let invocation = borrow(&live);
        let scope = DecoderInvocationScope::new(&invocation).unwrap();
        let mut backend = MockKvBackend::default();
        let mut executor = MockForwardExecutor::new(script).with_cancel_wait();
        let mut registry = Registry::new("scope");
        let id = transaction(1);
        registry
            .prepare(id, packed, &fixture.states, &mut backend)
            .unwrap();
        let error = registry
            .execute_batch_scoped(
                id,
                (&fixture.states, &fixture.batch),
                &mut backend,
                &mut executor,
                Some(&scope),
            )
            .unwrap_err();
        if matches!(script, ForwardScript::Async) {
            assert!(error.to_string().contains("abort required"));
        } else {
            assert!(
                error
                    .to_string()
                    .contains("injected active forward failure")
            );
        }
        assert_eq!(
            registry.phase(id).unwrap(),
            DecoderTransactionPhase::FailedActive
        );
        assert!(!backend.events.contains(&"leave"));
        assert!(!backend.events.contains(&"rollback"));
        assert!(registry.prepare_publication(id, &fixture.states).is_err());
        assert!(
            registry
                .resume_batch(
                    id,
                    &fixture.states,
                    &fixture.batch,
                    ContinuationId::new(1),
                    continuation_leases(),
                    &mut backend,
                    &mut executor
                )
                .is_err()
        );
        assert_eq!(
            registry.abort(id, &mut backend, &mut executor).unwrap(),
            KvEndProgress::Pending
        );
        assert_eq!(
            registry.phase(id).unwrap(),
            DecoderTransactionPhase::FailedActive
        );
        assert!(!backend.events.contains(&"leave"));
        assert_eq!(
            registry.abort(id, &mut backend, &mut executor).unwrap(),
            KvEndProgress::Complete
        );
        assert_eq!(executor.cancel_calls, 2);
        assert!(backend.events.contains(&"leave"));
        assert!(backend.events.contains(&"rollback"));
    }
}

#[test]
fn invocation_incomplete_complete_cannot_publish_and_preserves_abort_path() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let live = owner(&packed);
    let invocation = borrow(&live);
    let scope = DecoderInvocationScope::new(&invocation).unwrap();
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::Sync);
    let mut registry = Registry::new("scope");
    let id = transaction(1);
    registry
        .prepare(id, packed, &fixture.states, &mut backend)
        .unwrap();
    let error = registry
        .execute_batch_scoped(
            id,
            (&fixture.states, &fixture.batch),
            &mut backend,
            &mut executor,
            Some(&scope),
        )
        .unwrap_err();
    assert!(error.to_string().contains("did not retire"));
    assert_eq!(
        registry.phase(id).unwrap(),
        DecoderTransactionPhase::FailedQuiescent
    );
    assert!(registry.prepare_publication(id, &fixture.states).is_err());
    assert_eq!(
        registry.abort(id, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(executor.cancel_calls, 0);
    assert!(!backend.events.contains(&"commit"));
}

#[test]
fn invocation_stale_admission_never_enters_kv_or_loses_prepared_custody() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let live = owner(&packed);
    let invocation = borrow(&live);
    let scope = DecoderInvocationScope::new(&invocation).unwrap();
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::Async);
    let mut registry = Registry::new("scope");
    let id = transaction(1);
    registry
        .prepare(id, packed, &fixture.states, &mut backend)
        .unwrap();
    live.generation.set(invocation.generation + 1);
    assert!(
        registry
            .execute_batch_scoped(
                id,
                (&fixture.states, &fixture.batch),
                &mut backend,
                &mut executor,
                Some(&scope)
            )
            .is_err()
    );
    assert_eq!(
        registry.phase(id).unwrap(),
        DecoderTransactionPhase::Prepared
    );
    assert!(!backend.events.contains(&"enter"));
    assert_eq!(
        registry.abort(id, &mut backend, &mut executor).unwrap(),
        KvEndProgress::Complete
    );
}

#[test]
fn invocation_complete_publishes_once_through_the_same_transaction_shell() {
    let fixture = mixed_fixture();
    let packed = fixture.lower().unwrap();
    let live = owner(&packed);
    let invocation = borrow(&live);
    let scope = DecoderInvocationScope::new(&invocation).unwrap();
    let mut states = fixture.states;
    let mut backend = MockKvBackend::default();
    let mut executor = MockForwardExecutor::new(ForwardScript::ScopedSync);
    let mut registry = Registry::new("scope");
    let id = transaction(1);
    registry.prepare(id, packed, &states, &mut backend).unwrap();
    assert!(matches!(
        registry
            .execute_batch_scoped(
                id,
                (&states, &fixture.batch),
                &mut backend,
                &mut executor,
                Some(&scope)
            )
            .unwrap(),
        DecoderTransactionProgress::Complete(_)
    ));
    assert_eq!(
        registry.phase(id).unwrap(),
        DecoderTransactionPhase::Executed
    );
    assert!(
        registry
            .execute_batch_scoped(
                id,
                (&states, &fixture.batch),
                &mut backend,
                &mut executor,
                Some(&scope)
            )
            .is_err()
    );
    assert_eq!(
        registry.publish(id, &mut states, &mut backend).unwrap(),
        KvEndProgress::Complete
    );
    assert!(registry.is_empty());
    assert_eq!(backend.events.iter().filter(|e| **e == "commit").count(), 1);
    assert_eq!(states[0].core().position(), 5);
    assert_eq!(states[2].core().position(), 3);
}
