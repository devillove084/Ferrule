//! Exercises the existing driver and mock runner, not a second execution harness.
use super::*;
use crate::engine::runner_executor::HostKvDispatch;
use crate::engine::{HostKvDecodeConfig, KvExecutionMode, KvExecutionPolicy};
use ferrule_common::execution::{ExecutionBatch, ExecutionSequence, ForwardMode};

type Driver = ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool>;
fn policy() -> KvExecutionPolicy {
    KvExecutionPolicy::HostKvDecode(HostKvDecodeConfig::default())
}
fn dispatch() -> HostKvDispatch<MockTopKRunner> {
    HostKvDispatch {
        validate: |_, config, pages| {
            assert!(pages > 0);
            config.validate()
        },
        prepare: |executor, prefix, states, batch| {
            prefix.validate().unwrap();
            assert!(prefix.witness().source().committed_tokens > 0);
            assert_eq!(batch.mode(), ForwardMode::Decode);
            assert_eq!(batch.len(), 1);
            executor.prepare_batch_with_kv(
                prefix.witness().transaction(),
                states,
                batch,
                std::slice::from_ref(prefix.witness().reservation()),
            )
        },
        execute: |executor, prefix, states, batch, _, _| {
            prefix.validate().unwrap();
            assert_eq!(
                batch.positions()[0] as usize,
                prefix.witness().source().committed_tokens
            );
            executor.execute_prepared_batch(prefix.witness().transaction(), states, batch)
        },
    }
}
fn driver() -> Driver {
    let mut driver = prefix_cache_driver(vec![vec![TokenLogit::new(7, 1.)]; 8], 0, 16);
    driver.executor.install_host_kv_dispatch(dispatch());
    driver.configure_kv_execution(policy()).unwrap();
    driver
}
fn warm(driver: &mut Driver) {
    let mut input = request(1, &[1, 2, 3], 3, vec![]);
    input.session_id = Some(SessionId(1));
    input.ignore_eos = true;
    driver.submit(input);
    driver.step(&mut |_| Ok(())).unwrap();
    assert_eq!(driver.kv_execution_stats().resident_warming, 1);
    assert_eq!(driver.kv_execution_stats().host_kv_decode, 0);
    assert_eq!(
        driver
            .page_manager()
            .unwrap()
            .block_table(StateSlot::new(0))
            .unwrap()
            .committed_tokens(),
        3
    );
}
#[test]
fn host_kv_driver_warms_then_borrows_the_live_reservation_for_every_decode() {
    let mut driver = driver();
    let owner = driver.page_manager().unwrap().owner_identity();
    warm(&mut driver);
    driver.drive_ready_test_work(|_| Ok(())).unwrap();
    assert!(driver.kv_execution_stats().host_kv_decode >= 2);
    assert_eq!(driver.page_manager().unwrap().owner_identity(), owner);
    assert!(driver.resident_transactions.is_empty());
    assert!(driver.continuations.is_empty());
    assert_eq!(driver.executor.runner().rolled_back_batches, 0);
    assert!(
        driver
            .configure_kv_execution(KvExecutionPolicy::Resident)
            .is_err()
    );
    driver.shutdown(&mut |_| Ok(()), 32).unwrap();
    assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
}
#[test]
fn host_kv_driver_rejects_uninstalled_dispatch_and_multisequence_at_startup() {
    let mut driver = prefix_cache_driver(vec![], 0, 16);
    assert!(driver.configure_kv_execution(policy()).is_err());
    assert_eq!(driver.executor.runner().prepared_batches, 0);
    driver
        .configure_kv_execution(KvExecutionPolicy::Resident)
        .unwrap();
    let mut multi = concurrent_transaction_driver(MockTopKRunner::new(vec![]));
    multi.executor.install_host_kv_dispatch(dispatch());
    assert!(multi.configure_kv_execution(policy()).is_err());
    assert_eq!(multi.executor.runner().prepared_batches, 0);
}
#[test]
fn host_kv_driver_prepare_unknown_retains_original_reservation_until_abort_ack() {
    let mut driver = driver();
    warm(&mut driver);
    let mut route = dispatch();
    route.prepare = |_, prefix, _, _| {
        prefix.validate().unwrap();
        Err(ferrule_model::decoder::KvPrepareQuiescenceUnknown::wrap(
            prefix.witness().transaction(),
            ModelError::Execution {
                message: "injected unknown".into(),
            },
        )
        .into())
    };
    driver.executor.install_host_kv_dispatch(route);
    let error = driver.step(&mut |_| Ok(())).unwrap_err();
    assert!(
        error
            .diagnostic_chain()
            .any(|e| e.is::<ferrule_model::decoder::KvPrepareQuiescenceUnknown>())
    );
    assert_eq!(driver.resident_transactions.len(), 1);
    let pending = driver.resident_transactions.values().next().unwrap();
    assert_eq!(pending.execution_mode, KvExecutionMode::HostKvDecode);
    let Some(PendingResidentKv::Reserved(reservations)) = &pending.kv else {
        panic!("lost original reservation")
    };
    driver
        .page_manager()
        .unwrap()
        .derive_append_prefix(&reservations[0], pending.transaction, 0)
        .unwrap()
        .validate()
        .unwrap();
    assert!(matches!(
        pending.phase,
        ResidentTransactionPhase::Aborting { .. }
    ));
    assert!(driver.continuations.is_empty());
    assert_eq!(driver.kv_execution_stats().host_kv_decode, 0);
}
#[test]
fn host_kv_driver_waiting_is_aborted_without_registering_a_continuation() {
    let mut driver = driver();
    warm(&mut driver);
    let mut route = dispatch();
    route.execute = |_, prefix, _, _, _, _| {
        prefix.validate().unwrap();
        let dependencies =
            DependencySet::new([LogicalDependency::operation_retired(OperationId::new(900))?])?;
        Ok(MultiSessionBatchProgress::Waiting(
            PendingModelProgress::new(
                prefix.witness().transaction(),
                ContinuationId::new(900),
                dependencies,
            )?,
        ))
    };
    driver.executor.install_host_kv_dispatch(route);
    let error = driver.step(&mut |_| Ok(())).unwrap_err();
    assert!(error.to_string().contains("Waiting"));
    assert!(driver.continuations.is_empty());
    assert_eq!(driver.kv_execution_stats().host_kv_decode, 0);
    assert_eq!(driver.executor.runner().rolled_back_batches, 1);
    assert!(driver.resident_transactions.is_empty());
}
#[test]
fn host_kv_driver_execution_failure_keeps_custody_when_abort_fails() {
    let mut driver = driver();
    warm(&mut driver);
    let mut route = dispatch();
    route.execute = |_, prefix, _, _, _, _| {
        prefix.validate().unwrap();
        Err(Error::EngineUnavailable)
    };
    driver.executor.install_host_kv_dispatch(route);
    driver.executor.runner_mut().rollback_failures_remaining = 1;
    assert!(driver.step(&mut |_| Ok(())).is_err());
    assert_eq!(driver.resident_transactions.len(), 1);
    let pending = driver.resident_transactions.values().next().unwrap();
    let Some(PendingResidentKv::Reserved(reservations)) = &pending.kv else {
        panic!("lost custody")
    };
    driver
        .page_manager()
        .unwrap()
        .derive_append_prefix(&reservations[0], pending.transaction, 0)
        .unwrap()
        .validate()
        .unwrap();
    assert!(driver.continuations.is_empty());
    assert_eq!(driver.kv_execution_stats().host_kv_decode, 0);
}
#[test]
fn host_kv_policy_rejects_mixed_multiquery_empty_prefix_and_speculation() {
    let batch = |mode, phase, rows, position| {
        ExecutionBatch::new(
            mode,
            vec![1; rows],
            vec![position; rows],
            vec![None; rows],
            vec![],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                phase,
                0..rows as u32,
                position,
                position + rows as u32,
                0..0,
            )],
            vec![],
        )
    };
    assert_eq!(
        policy()
            .select(&batch(ForwardMode::Prefill, ForwardPhase::Prefill, 3, 0))
            .unwrap(),
        KvExecutionMode::ResidentWarming
    );
    assert_eq!(
        policy()
            .select(&batch(ForwardMode::Decode, ForwardPhase::Decode, 1, 3))
            .unwrap(),
        KvExecutionMode::HostKvDecode
    );
    for b in [
        batch(ForwardMode::Mixed, ForwardPhase::Decode, 1, 3),
        batch(ForwardMode::Decode, ForwardPhase::Decode, 2, 3),
        batch(ForwardMode::Decode, ForwardPhase::Decode, 1, 0),
        batch(ForwardMode::Decode, ForwardPhase::Decode, 1, 3)
            .with_intent(ExecutionIntent::ProvisionalVerification),
    ] {
        assert!(policy().select(&b).is_err());
        assert_eq!(
            KvExecutionPolicy::Resident.select(&b).unwrap(),
            KvExecutionMode::Resident
        );
    }
}
