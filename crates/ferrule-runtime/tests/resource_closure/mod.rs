//! Exercise real driver/registry/shutdown boundaries with existing CPU fixtures.

mod live_forwarding;

use super::*;
use ferrule_common::{CompletionExpectation, ProviderFault, QuiescenceEvidence};
use ferrule_model::resource_observability::{ResourceClosure, ResourceEvidence};

#[test]
fn idle_observation_is_read_only_and_not_quiescence_or_closure() {
    let driver = driver_with_outputs(Vec::new());
    assert!(driver.scheduler().is_idle());
    let before = driver.resource_observability_snapshot();
    assert_eq!(before.runtime.resident_bytes, ResourceEvidence::Observed(0));
    assert_eq!(
        before.runtime.pending_operations,
        ResourceEvidence::Observed(0)
    );
    assert_eq!(
        before.runtime.active_credit_grants,
        ResourceEvidence::Observed(0)
    );
    assert_eq!(
        before.runtime.retired_operations,
        ResourceEvidence::Observed(0)
    );
    assert_eq!(before.runtime.scratch_bytes, ResourceEvidence::Unobserved);
    assert_eq!(
        before.runtime.pending_retirements,
        ResourceEvidence::Unobserved
    );
    assert_eq!(before.runtime.fence, ResourceEvidence::Unobserved);
    assert_eq!(
        before.runtime.closure,
        ResourceEvidence::Observed(ResourceClosure::Open)
    );
    assert_eq!(before.runtime.quarantine, ResourceEvidence::Unobserved);
    assert!(!driver.physical_shutdown_complete());
    for _ in 0..3 {
        assert_eq!(driver.resource_observability_snapshot(), before);
    }
    assert!(driver.executor.runner().physical_owned);
    assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
}

#[test]
fn live_credit_observation_preserves_broker_units_and_does_not_release() {
    let mut driver = driver_with_outputs(Vec::new());
    let mut grant = driver
        .load_registry
        .acquire_hard_resources(
            123,
            crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode),
            [PhysicalResourceClaim::new(ResourceKind::Arena, 1)],
        )
        .unwrap();
    let before = driver.resource_observability_snapshot();
    let arena = before
        .credits
        .iter()
        .find(|credit| credit.kind == ResourceKind::Arena)
        .unwrap();
    assert_eq!(arena.unit, crate::scheduling::ResourceUnit::Slots);
    assert_eq!(arena.in_use, 1);
    assert_eq!(
        before.runtime.active_credit_grants,
        ResourceEvidence::Observed(1)
    );
    assert_eq!(driver.resource_observability_snapshot(), before);
    assert!(!grant.is_released());
    assert_eq!(
        driver.load_registry.resources().in_use(ResourceKind::Arena),
        1
    );
    driver
        .load_registry
        .release_hard_resources(&mut grant)
        .unwrap();
    let after = driver.resource_observability_snapshot();
    assert_eq!(after.runtime.scope, before.runtime.scope);
    assert_eq!(
        after.runtime.active_credit_grants,
        ResourceEvidence::Observed(0)
    );
    assert_eq!(after.runtime.closure, before.runtime.closure);
    assert_eq!(after.runtime.fence, ResourceEvidence::Unobserved);
}

#[test]
fn logical_shutdown_is_not_physical_closure() {
    let mut driver = driver_with_outputs(Vec::new());
    assert!(matches!(
        driver.shutdown_progress(&mut |_| Ok(())).unwrap(),
        ResidentShutdownProgress::Complete(_)
    ));
    let snapshot = driver.resource_observability_snapshot();
    assert_eq!(
        snapshot.runtime.closure,
        ResourceEvidence::Observed(ResourceClosure::Open)
    );
    assert_eq!(snapshot.runtime.fence, ResourceEvidence::Unobserved);
    assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
    assert!(driver.executor.runner().physical_owned);
    assert!(!driver.physical_shutdown_complete());
}

#[test]
fn explicit_close_reports_closure_without_inventing_a_fence() {
    let mut driver = driver_with_outputs(Vec::new());
    let before = driver.resource_observability_snapshot();
    assert!(matches!(
        driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
        ResidentShutdownProgress::Complete(_)
    ));
    let after = driver.resource_observability_snapshot();
    assert_eq!(after.runtime.scope, before.runtime.scope);
    assert_eq!(
        after.runtime.closure,
        ResourceEvidence::Observed(ResourceClosure::Closed)
    );
    assert_eq!(after.runtime.fence, ResourceEvidence::Unobserved);
    assert_eq!(after.runtime.scratch_bytes, ResourceEvidence::Unobserved);
    assert!(driver.physical_shutdown_complete());
    assert_eq!(driver.executor.runner().physical_shutdown_calls, 1);
    assert!(matches!(
        driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
        ResidentShutdownProgress::Complete(_)
    ));
    assert_eq!(driver.resource_observability_snapshot(), after);
    assert_eq!(driver.executor.runner().physical_shutdown_calls, 1);
}

#[test]
fn failed_close_retains_unknown_after_later_successful_close() {
    let mut driver = driver_with_outputs(Vec::new());
    driver.executor.runner_mut().physical_shutdown_failures = 1;
    let scope = driver.resource_observability_snapshot().runtime.scope;
    let error = driver
        .shutdown_and_close_progress(&mut |_| Ok(()))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected physical shutdown failure")
    );
    assert!(driver.executor.runner().physical_owned);
    assert!(!driver.physical_shutdown_complete());
    assert!(matches!(
        driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
        ResidentShutdownProgress::Complete(_)
    ));
    for _ in 0..3 {
        let snapshot = driver.resource_observability_snapshot();
        assert_eq!(snapshot.runtime.scope, scope);
        assert_eq!(snapshot.runtime.fence, ResourceEvidence::Unknown);
        assert_eq!(snapshot.runtime.closure, ResourceEvidence::Unknown);
        assert_eq!(snapshot.runtime.quarantine, ResourceEvidence::Unknown);
    }
    // Lifecycle success is preserved; the observational history is conservative.
    assert!(driver.physical_shutdown_complete());
    assert!(!driver.executor.runner().physical_owned);
    assert_eq!(driver.executor.runner().physical_shutdown_calls, 2);
}

#[test]
fn physical_unknown_retries_never_clear_evidence_or_release_the_owner() {
    let mut driver = driver_with_outputs(Vec::new());
    driver.executor.runner_mut().physical_shutdown_failures = 1;
    driver.executor.runner_mut().physical_shutdown_unknown = true;
    for _ in 0..3 {
        assert!(driver.shutdown_and_close_progress(&mut |_| Ok(())).is_err());
        let before = driver.resource_observability_snapshot();
        assert_eq!(before.runtime.fence, ResourceEvidence::Unknown);
        assert_eq!(before.runtime.closure, ResourceEvidence::Unknown);
        assert_eq!(before.runtime.quarantine, ResourceEvidence::Unknown);
        assert_eq!(driver.resource_observability_snapshot(), before);
        assert!(driver.executor.runner().physical_owned);
        assert!(!driver.physical_shutdown_complete());
    }
}

#[test]
fn registry_global_unknown_is_not_hidden_by_zero_work_and_credit_gauges() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let mut driver = driver_from_runner(
        MockTopKRunner::new(Vec::new()).with_materialization_provider(Box::new(physical)),
    );
    let fault = inject_global_unknown(&mut driver, &handle);
    for _ in 0..2 {
        let before = driver.resource_observability_snapshot();
        assert_eq!(
            before.runtime.pending_operations,
            ResourceEvidence::Observed(0)
        );
        assert_eq!(
            before.runtime.active_credit_grants,
            ResourceEvidence::Observed(0)
        );
        assert_eq!(before.runtime.fence, ResourceEvidence::Unknown);
        assert_eq!(before.runtime.closure, ResourceEvidence::Unknown);
        assert_eq!(before.runtime.quarantine, ResourceEvidence::Unknown);
        assert_eq!(driver.resource_observability_snapshot(), before);
        assert_eq!(
            driver.load_registry.provider_faults(),
            std::slice::from_ref(&fault)
        );
        assert_eq!(
            driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
            ResidentShutdownProgress::Pending
        );
        assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
        assert_eq!(
            driver.observability.resources.snapshot().fence,
            ResourceEvidence::Unknown
        );
        let summary = driver.resource_shutdown_summary().unwrap();
        assert_eq!(
            summary.outcome,
            crate::engine::ResourceShutdownOutcome::Pending
        );
        assert_eq!(summary.runtime.scope, before.runtime.scope);
        assert_eq!(summary.runtime.fence, ResourceEvidence::Unknown);
        assert_eq!(summary.physical, Default::default());
        assert!(summary.coverage.physical.unobserved.contains(&"release"));
        assert!(summary.coverage.runtime.unknown.contains(&"fence"));
    }
}

fn driver_at_read() -> (
    ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool>,
    crate::io::testing::MockPhysicalHandle,
    CompletionExpectation,
) {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.defer_cancellation_completion();
    let runner = MockTopKRunner::new(Vec::new())
        .with_resumable_wait_scripts([2])
        .with_materialization_request(materialization_request(3))
        .with_materialization_provider(Box::new(physical));
    let mut driver = concurrent_transaction_driver(runner);
    driver.submit(request(1, &[1], 2, Vec::new()));
    for _ in 0..2 {
        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
    }
    let scope = handle
        .commands()
        .iter()
        .find_map(|command| match command {
            MockPhysicalCommand::SubmitRead(operation, key, plan) => Some(
                CompletionExpectation::new(
                    *operation,
                    *key,
                    LoadStage::ReadSubmitted,
                    plan.completion_bytes(LoadStage::ReadSubmitted).unwrap(),
                )
                .unwrap(),
            ),
            _ => None,
        })
        .expect("a submitted read must own physical credit");
    assert_eq!(
        driver
            .load_registry
            .resources()
            .in_use(ResourceKind::ReadSlot),
        1
    );
    (driver, handle, scope)
}

fn inject_fault(
    driver: &mut ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool>,
    handle: &crate::io::testing::MockPhysicalHandle,
    fault: &ProviderFault,
) {
    handle.push_fault(fault.clone());
    assert_eq!(driver.load_registry.collect_provider_completions(1), 1);
    assert!(matches!(
        driver.load_registry.process_one_completion(),
        Err(crate::io::RegistryError::Provider { .. })
    ));
}

#[test]
fn scoped_unknown_then_exact_proof_does_not_invent_retained_quarantine() {
    let (mut driver, handle, scope) = driver_at_read();
    let owner = driver.resource_observability_snapshot().runtime.scope;
    let fault = ProviderFault {
        scope: Some(scope),
        failure: FailureReason::StorageUnavailable,
        quiescence: QuiescenceEvidence::Unknown,
    };
    inject_fault(&mut driver, &handle, &fault);
    let retained = driver.observability.resources.snapshot();
    let before = driver.resource_observability_snapshot();
    assert_eq!(
        before.runtime.pending_operations,
        ResourceEvidence::Observed(1)
    );
    assert_eq!(before.runtime.fence, ResourceEvidence::Unknown);
    assert_eq!(before.runtime.closure, ResourceEvidence::Unknown);
    assert_eq!(before.runtime.quarantine, ResourceEvidence::Unknown);
    assert_eq!(driver.resource_observability_snapshot(), before);
    assert_eq!(driver.observability.resources.snapshot(), retained);
    assert_eq!(
        driver
            .load_registry
            .resources()
            .in_use(ResourceKind::ReadSlot),
        1
    );
    assert!(driver.load_registry.retirement(scope.operation).is_none());
    assert_eq!(
        driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
        ResidentShutdownProgress::Pending
    );
    assert_eq!(
        driver
            .load_registry
            .resources()
            .in_use(ResourceKind::ReadSlot),
        1
    );
    handle.push_outcome(
        scope.operation,
        scope.key,
        scope.stage,
        CompletionOutcome::Cancelled(CancellationReason::OwnerShutdown),
    );
    assert!(matches!(
        driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
        ResidentShutdownProgress::Complete(_)
    ));
    assert_eq!(driver.load_registry.active_operations(), 0);
    assert_eq!(driver.load_registry.resources().active_grants(), 0);
    assert!(driver.load_registry.retirement(scope.operation).is_some());
    assert!(!driver.load_registry.has_pending_owner_work());
    assert_eq!(driver.load_registry.provider_faults(), &[fault]);
    let retained = driver.observability.resources.snapshot();
    let commands = handle.commands();
    for _ in 0..3 {
        let after = driver.resource_observability_snapshot();
        assert_eq!(after.runtime.scope, owner);
        assert_eq!(
            after.runtime.pending_operations,
            ResourceEvidence::Observed(0)
        );
        assert_eq!(
            after.runtime.active_credit_grants,
            ResourceEvidence::Observed(0)
        );
        assert_eq!(
            after.runtime.retired_operations,
            ResourceEvidence::Observed(1)
        );
        assert_eq!(after.runtime.fence, ResourceEvidence::Unknown);
        assert_eq!(after.runtime.closure, ResourceEvidence::Unknown);
        assert_eq!(after.runtime.quarantine, ResourceEvidence::Unknown);
        assert_eq!(driver.observability.resources.snapshot(), retained);
        assert_eq!(handle.commands(), commands);
    }
    assert!(driver.physical_shutdown_complete());
    assert!(!driver.executor.runner().physical_owned);
    let summary = driver.resource_shutdown_summary().unwrap();
    assert_eq!(summary.runtime.quarantine, ResourceEvidence::Unknown);
    assert!(summary.coverage.runtime.unknown.contains(&"quarantine"));
    assert!(!summary.coverage.runtime.observed.contains(&"quarantine"));
}

#[test]
fn mismatched_quiescent_fault_cannot_hide_registry_custody_uncertainty() {
    for mismatch in 0..3 {
        let (mut driver, handle, scope) = driver_at_read();
        let owner = driver.resource_observability_snapshot().runtime.scope;
        let mut mismatched = scope;
        match mismatch {
            0 => mismatched.operation = OperationId::new(u64::MAX),
            1 => mismatched.stage = LoadStage::UploadSubmitted,
            2 => mismatched.bytes += 1,
            _ => unreachable!(),
        }
        let fault = ProviderFault {
            scope: Some(mismatched),
            failure: FailureReason::StorageUnavailable,
            quiescence: QuiescenceEvidence::Quiescent,
        };
        inject_fault(&mut driver, &handle, &fault);
        assert!(driver.load_registry.has_pending_owner_work());
        assert_eq!(
            driver.load_registry.provider_faults(),
            std::slice::from_ref(&fault)
        );
        let retained = driver.observability.resources.snapshot();
        let before = driver.resource_observability_snapshot();
        assert_eq!(before.runtime.fence, ResourceEvidence::Unknown);
        assert_eq!(before.runtime.closure, ResourceEvidence::Unknown);
        assert_eq!(before.runtime.quarantine, ResourceEvidence::Unknown);
        assert_eq!(driver.resource_observability_snapshot(), before);
        assert_eq!(driver.observability.resources.snapshot(), retained);
        assert_eq!(
            driver
                .load_registry
                .resources()
                .in_use(ResourceKind::ReadSlot),
            1
        );
        assert!(driver.load_registry.retirement(scope.operation).is_none());
        assert_eq!(
            driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
            ResidentShutdownProgress::Pending
        );
        handle.push_outcome(
            scope.operation,
            scope.key,
            scope.stage,
            CompletionOutcome::Cancelled(CancellationReason::OwnerShutdown),
        );
        assert_eq!(
            driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
            ResidentShutdownProgress::Pending
        );
        assert_eq!(driver.load_registry.active_operations(), 0);
        assert_eq!(driver.load_registry.resources().active_grants(), 0);
        assert!(driver.load_registry.retirement(scope.operation).is_some());
        assert!(driver.load_registry.has_pending_owner_work());
        let retained = driver.observability.resources.snapshot();
        let commands = handle.commands();
        for _ in 0..3 {
            let after = driver.resource_observability_snapshot();
            assert_eq!(after.runtime.scope, owner);
            assert_eq!(
                after.runtime.pending_operations,
                ResourceEvidence::Observed(0)
            );
            assert_eq!(
                after.runtime.active_credit_grants,
                ResourceEvidence::Observed(0)
            );
            assert_eq!(after.runtime.fence, ResourceEvidence::Unknown);
            assert_eq!(after.runtime.closure, ResourceEvidence::Unknown);
            assert_eq!(after.runtime.quarantine, ResourceEvidence::Unknown);
            assert_eq!(driver.observability.resources.snapshot(), retained);
            assert_eq!(handle.commands(), commands);
        }
        assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
        assert!(driver.executor.runner().physical_owned);
        assert!(!driver.physical_shutdown_complete());
        let summary = driver.resource_shutdown_summary().unwrap();
        assert_eq!(summary.runtime.fence, ResourceEvidence::Unknown);
        assert!(summary.coverage.runtime.unknown.contains(&"fence"));
    }
}

#[test]
fn registry_fault_preserves_an_explicit_retained_quarantine_observation() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let mut driver = driver_from_runner(
        MockTopKRunner::new(Vec::new()).with_materialization_provider(Box::new(physical)),
    );
    let mut retained = driver.observability.resources.snapshot();
    retained.quarantine = ResourceEvidence::Observed(true);
    driver.observability.resources.observe(&retained).unwrap();
    inject_global_unknown(&mut driver, &handle);
    let snapshot = driver.resource_observability_snapshot();
    assert_eq!(
        snapshot.runtime.quarantine,
        ResourceEvidence::Observed(true)
    );
    assert_eq!(snapshot.runtime.fence, ResourceEvidence::Unknown);
    assert_eq!(snapshot.runtime.closure, ResourceEvidence::Unknown);
    assert_eq!(driver.observability.resources.snapshot(), retained);
}

#[test]
fn boxed_inference_engine_forwards_resource_evidence_and_scope_survives_move() {
    use crate::engine::{InferenceEngine, ResidentInferenceEngine};
    let driver = driver_with_outputs(Vec::new());
    let before = driver.resource_observability_snapshot();
    let engine = ResidentInferenceEngine::new(driver);
    let boxed: Box<dyn InferenceEngine> = Box::new(engine);
    assert_eq!(boxed.resource_observability_snapshot(), Some(before));
}

#[test]
fn replacement_driver_gets_a_distinct_observation_owner() {
    let first = driver_with_outputs(Vec::new()).resource_observability_snapshot();
    let second = driver_with_outputs(Vec::new()).resource_observability_snapshot();
    assert_ne!(first.runtime.scope.owner, second.runtime.scope.owner);
    assert_eq!(first.runtime.scope.generation, std::num::NonZeroU64::MIN);
    assert_eq!(second.runtime.scope.generation, std::num::NonZeroU64::MIN);
}
