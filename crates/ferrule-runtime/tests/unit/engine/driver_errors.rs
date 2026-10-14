use super::*;

fn failing_driver(
    rollback: usize,
    release: usize,
) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
    let runner = MockTopKRunner::new(Vec::new())
        .with_committed_resumable_batch(1, vec![TokenLogit::new(7, 1.0)])
        .with_resume_errors(1)
        .with_rollback_failures(rollback)
        .with_kv_release_failures(release);
    let mut driver = driver_from_runner(runner)
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
    let mut submitted = request(1, &[1], 2, Vec::new());
    submitted.session_id = Some(SessionId(1));
    driver.try_submit(submitted).unwrap();
    assert!(matches!(
        driver.step(&mut |_| Ok(())).unwrap(),
        ResidentDriverStep::WaitingForModelProgress(_)
    ));
    driver
}

#[test]
fn execution_primary_survives_kv_release_failure() {
    let mut driver = failing_driver(0, 1);
    let error = driver
        .step(&mut |_| panic!("failed execution cannot emit"))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("simulated resumable batch failure"),
        "primary lost: {error:?}"
    );
    assert!(
        error
            .to_string()
            .contains("simulated KV page release failure")
    );
    assert_eq!(driver.executor().runner().rolled_back_batches, 1);
    assert_eq!(
        driver
            .load_registry()
            .resources()
            .in_use(ResourceKind::KvPage),
        1
    );
    assert_eq!(driver.page_manager().unwrap().stats().retiring_pages, 1);
    assert!(driver.drain_finished().is_empty());
    let retried = driver
        .step(&mut |_| panic!("retry cannot emit"))
        .unwrap_err();
    assert!(
        retried
            .to_string()
            .contains("simulated resumable batch failure")
    );
    assert!(
        retried
            .to_string()
            .contains("simulated KV page release failure"),
        "secondary lost on retry: {retried:?}"
    );
    assert_eq!(driver.executor().runner().rolled_back_batches, 1);
    assert!(driver.kv.grants_empty());
    assert!(driver.resident_transactions.is_empty());
    assert_eq!(driver.drain_failed().len(), 1);
}

#[test]
fn execution_primary_survives_backend_rollback_failure() {
    let mut driver = failing_driver(1, 0);
    let error = driver
        .step(&mut |_| panic!("failed execution cannot emit"))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("simulated resumable batch failure"),
        "primary lost: {error:?}"
    );
    assert!(
        error
            .to_string()
            .contains("simulated backend rollback failure")
    );
    assert!(driver.executor().runner().released_kv_pages.is_empty());
    assert_eq!(
        driver
            .load_registry()
            .resources()
            .in_use(ResourceKind::KvPage),
        1
    );
    let retried = driver
        .step(&mut |_| panic!("retry cannot emit"))
        .unwrap_err();
    assert!(
        retried
            .to_string()
            .contains("simulated resumable batch failure")
    );
    assert!(
        retried
            .to_string()
            .contains("simulated backend rollback failure")
    );
    assert!(driver.kv.grants_empty());
    assert_eq!(driver.executor().runner().rolled_back_batches, 2);
}

#[test]
fn unknown_page_is_secondary_and_never_a_release_ack() {
    let mut driver = failing_driver(0, 0);
    driver.executor.runner_mut().kv_release_error = Some(ModelError::Execution {
        message: "cannot release an unknown paged KV page".into(),
    });
    let error = driver
        .step(&mut |_| panic!("failed execution cannot emit"))
        .unwrap_err();
    assert!(
        error
            .primary()
            .to_string()
            .contains("simulated resumable batch failure")
    );
    assert_eq!(error.cleanup_errors().len(), 1);
    assert!(
        error.cleanup_errors()[0]
            .to_string()
            .contains("unknown paged KV page")
    );
    assert_eq!(driver.page_manager().unwrap().stats().retiring_pages, 1);
    assert_eq!(driver.kv.grant_count(), 1);
    assert!(driver.executor().runner().released_kv_pages.is_empty());
    assert!(driver.drain_finished().is_empty());
    let retry = driver
        .step(&mut |_| panic!("failed execution cannot emit"))
        .unwrap_err();
    assert_eq!(retry.cleanup_errors().len(), 1);
    assert_eq!(driver.executor().runner().rolled_back_batches, 1);
    assert_eq!(driver.executor().runner().released_kv_pages.len(), 1);
}

#[test]
fn successive_rollback_and_release_failures_retain_all_secondary_errors() {
    let mut driver = failing_driver(1, 1);
    let first = driver.step(&mut |_| Ok(())).unwrap_err();
    assert_eq!(first.cleanup_errors().len(), 1);
    let second = driver.step(&mut |_| Ok(())).unwrap_err();
    let secondary = second.cleanup_errors();
    assert_eq!(secondary.len(), 2);
    assert!(
        secondary[0]
            .to_string()
            .contains("backend rollback failure")
    );
    assert!(secondary[1].to_string().contains("KV page release failure"));
    assert!(
        second
            .primary()
            .to_string()
            .contains("resumable batch failure")
    );
    assert_eq!(
        first.cleanup_errors().len(),
        1,
        "published snapshot is immutable"
    );
    let last = driver.step(&mut |_| Ok(())).unwrap_err();
    assert_eq!(last.cleanup_errors().len(), 2);
    assert!(driver.kv.grants_empty());
    assert_eq!(driver.executor().runner().rolled_back_batches, 2);
}

#[test]
fn typed_unknown_prepare_retains_pages_until_backend_abort_ack() {
    use ferrule_model::decoder::KvPrepareQuiescenceUnknown;
    let transaction = ExecutionTransactionId::new(1).unwrap();
    let mut runner = MockTopKRunner::new(Vec::new()).with_rollback_failures(1);
    runner.paged = true;
    runner.prepare_error = Some(KvPrepareQuiescenceUnknown::wrap(
        transaction,
        ModelError::Execution {
            message: "injected prepare execution failure".into(),
        },
    ));
    let mut driver = driver_from_runner(runner)
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
    driver.try_submit(request(1, &[1], 0, Vec::new())).unwrap();
    let error = driver
        .step(&mut |_| panic!("unknown prepare cannot emit"))
        .unwrap_err();
    assert!(
        error
            .diagnostic_chain()
            .any(|cause| cause.is::<KvPrepareQuiescenceUnknown>())
    );
    assert_eq!(driver.executor().runner().rolled_back_batches, 0);
    assert!(driver.executor().runner().released_kv_pages.is_empty());
    assert_eq!(driver.kv.grant_count(), 1);
    assert!(matches!(
        driver.resident_transactions.values().next().unwrap().phase,
        ResidentTransactionPhase::Aborting { .. }
    ));
    let rollback = driver
        .step(&mut |_| panic!("unknown prepare cannot emit"))
        .unwrap_err();
    assert!(
        rollback
            .diagnostic_chain()
            .any(|cause| cause.is::<KvPrepareQuiescenceUnknown>())
    );
    assert_eq!(rollback.cleanup_errors().len(), 1);
    assert!(driver.executor().runner().released_kv_pages.is_empty());
    assert_eq!(driver.kv.grant_count(), 1);
    let acknowledged = driver
        .step(&mut |_| panic!("failed prepare cannot emit"))
        .unwrap_err();
    assert!(
        acknowledged
            .primary()
            .to_string()
            .contains("injected prepare execution failure")
    );
    assert_eq!(acknowledged.cleanup_errors().len(), 1);
    assert_eq!(driver.executor().runner().rolled_back_batches, 2);
    assert!(driver.kv.grants_empty());
    assert!(driver.resident_transactions.is_empty());
    assert!(driver.drain_finished().is_empty());
}

#[test]
fn rejected_prepare_keeps_execution_primary_when_release_reports_unknown_page() {
    let mut runner = MockTopKRunner::new(Vec::new());
    runner.paged = true;
    runner.prepare_error = Some(ModelError::Execution {
        message: "original execution failure during prepare".into(),
    });
    runner.kv_release_error = Some(ModelError::Execution {
        message: "cannot release an unknown paged KV page".into(),
    });
    let mut driver = driver_from_runner(runner)
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
    driver.try_submit(request(1, &[1], 0, Vec::new())).unwrap();
    let error = driver
        .step(&mut |_| panic!("failed prepare cannot emit"))
        .unwrap_err();
    assert!(
        error
            .primary()
            .to_string()
            .contains("original execution failure during prepare")
    );
    assert_eq!(error.cleanup_errors().len(), 1);
    assert!(
        error.cleanup_errors()[0]
            .to_string()
            .contains("unknown paged KV page")
    );
    assert_eq!(driver.kv.grant_count(), 1);
    assert_eq!(driver.page_manager().unwrap().stats().retiring_pages, 1);
    assert_eq!(driver.executor().runner().rolled_back_batches, 0);
    let acknowledged = driver.step(&mut |_| Ok(())).unwrap_err();
    assert_eq!(acknowledged.cleanup_errors().len(), 1);
    assert!(driver.kv.grants_empty());
    assert_eq!(driver.executor().runner().rolled_back_batches, 0);
    assert!(driver.drain_finished().is_empty());
}
