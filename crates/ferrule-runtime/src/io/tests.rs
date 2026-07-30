use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::expert_io::{ExpertIoResourceDemand, ExpertIoResourceLimits};
use ferrule_common::io_protocol::{
    ArtifactFormat, BackendId, CancellationReason, CompletionEvent, CompletionGeneration,
    CompletionOutcome, CompletionTimestamp, ContentHash, ContinuationId, DependencySetEpoch,
    DestinationGeneration, DeviceId, ExpertId, FailureReason, LayerId, LoadKey, LoadStage,
    ModelInstanceId, OperationId, RequestGeneration, SourceGeneration, SourceIdentityHash,
    StaleReason, WaiterId,
};

use super::*;
use crate::scheduling::{
    HardResourceBroker, HardResourceClaim, HardResourceError, HardResourceLimit, ResourceClass,
    ResourceKind,
};

const BYTES: u64 = 4096;

fn key(seed: u8, destination: u64) -> LoadKey {
    key_with_source_generation(seed, 1, destination)
}

fn key_with_source_generation(seed: u8, source_generation: u64, destination: u64) -> LoadKey {
    LoadKey::new(
        ModelInstanceId::new(1),
        SourceIdentityHash::new([seed.max(1); 32]),
        ContentHash::new([seed.saturating_add(1).max(1); 32]),
        LayerId::new(seed as u32),
        ExpertId::new(seed as u32),
        ArtifactFormat::new(1),
        BackendId::new(1),
        DeviceId::new(1),
        SourceGeneration::new(source_generation),
        DestinationGeneration::new(destination),
    )
    .unwrap()
}

fn waiter(id: u64) -> WaiterId {
    waiter_for(id, id)
}

fn waiter_for(id: u64, continuation: u64) -> WaiterId {
    WaiterId::new(
        ExecutionTransactionId::new(id).unwrap(),
        RequestGeneration::new(1),
        DependencySetEpoch::new(1),
        ContinuationId::new(continuation),
    )
    .unwrap()
}

fn request(key: LoadKey) -> LoadRequest {
    LoadRequest::new(key, BYTES, ResourceClass::Decode)
}

fn auto_registry() -> LoadRegistry<FakeBackend> {
    LoadRegistry::with_testing_resources(FakeBackend::new()).unwrap()
}

fn manual_registry() -> LoadRegistry<FakeBackend> {
    LoadRegistry::with_testing_resources(FakeBackend::manual()).unwrap()
}

fn attach_one(
    registry: &mut LoadRegistry<FakeBackend>,
    waiter_id: WaiterId,
    load_key: LoadKey,
) -> OperationId {
    registry
        .attach_waiter(waiter_id, [request(load_key)], 10)
        .unwrap()
        .created[0]
}

fn event(
    operation: OperationId,
    load_key: LoadKey,
    stage: LoadStage,
    outcome: CompletionOutcome,
    bytes: u64,
    timestamp: u64,
) -> CompletionEvent {
    CompletionEvent::new(
        operation,
        load_key,
        stage,
        outcome,
        bytes,
        CompletionGeneration::for_key(load_key),
        CompletionTimestamp::from_nanos(timestamp),
    )
}

fn success_event(
    operation: OperationId,
    load_key: LoadKey,
    stage: LoadStage,
    timestamp: u64,
) -> CompletionEvent {
    event(
        operation,
        load_key,
        stage,
        CompletionOutcome::Succeeded,
        BYTES,
        timestamp,
    )
}

fn manual_at_read() -> (LoadRegistry<FakeBackend>, LoadKey, OperationId) {
    let mut registry = manual_registry();
    let load_key = key(1, 1);
    let operation = attach_one(&mut registry, waiter(1), load_key);
    assert!(registry.schedule_one(19).unwrap());
    assert!(registry.schedule_one(20).unwrap());
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
    (registry, load_key, operation)
}

fn manual_at_upload() -> (LoadRegistry<FakeBackend>, LoadKey, OperationId) {
    let (mut registry, load_key, operation) = manual_at_read();
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        30,
    ));
    registry.process_one_completion().unwrap();
    registry.schedule_one(40).unwrap();
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::UploadSubmitted
    );
    (registry, load_key, operation)
}

fn manual_at_install() -> (LoadRegistry<FakeBackend>, LoadKey, OperationId) {
    let (mut registry, load_key, operation) = manual_at_upload();
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::UploadSubmitted,
        50,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::Installing
    );
    registry.schedule_one(60).unwrap();
    (registry, load_key, operation)
}

fn hard_broker_with(changed: ResourceKind, capacity: u64, reserve: u64) -> HardResourceBroker {
    HardResourceBroker::new(ResourceKind::ALL.map(|kind| {
        if kind == changed {
            HardResourceLimit::new(kind, capacity, reserve)
        } else {
            let default_capacity = match kind {
                ResourceKind::PinnedSlab
                | ResourceKind::ReadBytes
                | ResourceKind::UploadBytes
                | ResourceKind::ExpertFrame => 1 << 20,
                _ => 128,
            };
            HardResourceLimit::new(kind, default_capacity, 0)
        }
    }))
    .unwrap()
}

fn staged_broker(
    sqe: u64,
    pinned_operations: u64,
    upload_slots: u64,
    load_operations: u64,
) -> HardResourceBroker {
    HardResourceBroker::new(ResourceKind::ALL.map(|kind| {
        let capacity = match kind {
            ResourceKind::Sqe => sqe,
            ResourceKind::PinnedSlab => BYTES * pinned_operations,
            ResourceKind::ReadBytes => BYTES * sqe,
            ResourceKind::UploadSlot => upload_slots,
            ResourceKind::UploadBytes => BYTES * upload_slots,
            ResourceKind::ExpertFrame => BYTES * load_operations,
            ResourceKind::Lease | ResourceKind::LoadOperation => load_operations,
            ResourceKind::Arena
            | ResourceKind::KvPage
            | ResourceKind::Continuation
            | ResourceKind::Waiter
            | ResourceKind::ReadyCohort => 128,
        };
        HardResourceLimit::new(kind, capacity, 0)
    }))
    .unwrap()
}

#[test]
fn full_success_reaches_resident_and_targeted_ready() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    let operation = attach_one(&mut registry, waiter(1), load_key);
    assert!(registry.drive(100, 32).unwrap() >= 6);
    assert!(registry.residency_binding(load_key).is_some());
    assert_eq!(
        registry.pop_ready(101).unwrap(),
        Some(ContinuationId::new(1))
    );
    assert!(registry.retirement(operation).is_some());
}

#[test]
fn drive_one_reports_the_key_affected_by_each_transition() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    attach_one(&mut registry, waiter(1), load_key);

    let mut progressed = 0;
    loop {
        match registry.drive_one(100).unwrap() {
            RegistryDriveStep::Idle => break,
            RegistryDriveStep::Progressed { key } => {
                assert_eq!(key, Some(load_key));
                progressed += 1;
            }
        }
    }

    assert!(progressed >= 6);
    assert!(registry.residency_binding(load_key).is_some());
}

#[test]
fn initial_stage_is_reserved_without_physical_claims_or_reservation() {
    let mut registry = manual_registry();
    let operation = attach_one(&mut registry, waiter(1), key(1, 1));
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::Reserved
    );
    assert!(registry.backend().commands().is_empty());
    assert_eq!(registry.resources().in_use(ResourceKind::Sqe), 0);
    assert_eq!(registry.resources().in_use(ResourceKind::PinnedSlab), 0);
    assert_eq!(registry.resources().in_use(ResourceKind::LoadOperation), 0);
}

#[test]
fn read_completion_reaches_host_ready() {
    let (mut registry, load_key, operation) = manual_at_read();
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::HostReady
    );
}

#[test]
fn upload_completion_reaches_installing() {
    let (mut registry, load_key, operation) = manual_at_upload();
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::UploadSubmitted,
        50,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::Installing
    );
}

#[test]
fn successful_operation_retires_exactly_once() {
    let mut registry = auto_registry();
    let operation = attach_one(&mut registry, waiter(1), key(1, 1));
    registry.drive(100, 32).unwrap();
    assert_eq!(registry.stats().retirements, 1);
    assert!(registry.operation(operation).is_none());
    assert_eq!(registry.retirement(operation).unwrap().operation, operation);
}

#[test]
fn success_stage_history_is_complete_and_exactly_once() {
    let mut registry = auto_registry();
    let operation = attach_one(&mut registry, waiter(1), key(1, 1));
    registry.drive(100, 32).unwrap();
    assert_eq!(
        registry.stage_history(operation).unwrap(),
        &[
            LoadStage::Reserved,
            LoadStage::ReadSubmitted,
            LoadStage::HostReady,
            LoadStage::UploadSubmitted,
            LoadStage::Installing,
            LoadStage::Resident,
            LoadStage::Retired,
        ]
    );
}

#[test]
fn published_binding_has_exact_destination_generation() {
    let mut registry = auto_registry();
    let load_key = key(1, 9);
    attach_one(&mut registry, waiter(1), load_key);
    registry.drive(100, 32).unwrap();
    assert_eq!(
        registry.residency_binding(load_key).unwrap().generation,
        DestinationGeneration::new(9)
    );
}

#[test]
fn operation_to_key_is_durable_after_retirement() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    let operation = attach_one(&mut registry, waiter(1), load_key);
    registry.drive(100, 32).unwrap();
    assert_eq!(registry.key_for_operation(operation), Some(load_key));
}

#[test]
fn single_flight_n_waiters_issues_one_physical_read() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    let first = attach_one(&mut registry, waiter(1), load_key);
    for id in 2..=16 {
        let report = registry
            .attach_waiter(waiter(id), [request(load_key)], id)
            .unwrap();
        assert_eq!(report.joined, vec![first]);
    }
    registry.drive(100, 64).unwrap();
    assert_eq!(registry.backend().physical_reads(), 1);
    assert_eq!(registry.stats().single_flight_joins, 15);
}

#[test]
fn different_destination_generation_never_joins() {
    let mut registry = manual_registry();
    let first = registry
        .attach_waiter(waiter(1), [request(key(1, 1))], 1)
        .unwrap();
    let second = registry
        .attach_waiter(waiter(2), [request(key(1, 2))], 2)
        .unwrap();
    assert_ne!(first.created[0], second.created[0]);
    assert_eq!(registry.active_operations(), 2);
}

#[test]
fn different_source_generation_never_joins() {
    let mut registry = manual_registry();
    let first_key = key_with_source_generation(1, 1, 1);
    let second_key = key_with_source_generation(1, 2, 1);
    let first = attach_one(&mut registry, waiter(1), first_key);
    let second = attach_one(&mut registry, waiter(2), second_key);
    assert_ne!(first, second);
}

#[test]
fn joined_byte_expectation_must_match() {
    let mut registry = manual_registry();
    let load_key = key(1, 1);
    attach_one(&mut registry, waiter(1), load_key);
    let error = registry
        .attach_waiter(
            waiter(2),
            [LoadRequest::new(load_key, BYTES * 2, ResourceClass::Decode)],
            2,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        RegistryError::ByteExpectationMismatch { .. }
    ));
}

#[test]
fn duplicate_dependency_is_rejected_before_side_effects() {
    let mut registry = manual_registry();
    let load_key = key(1, 1);
    let error = registry
        .attach_waiter(waiter(1), [request(load_key), request(load_key)], 1)
        .unwrap_err();
    assert!(matches!(error, RegistryError::DuplicateDependency(_)));
    assert_eq!(registry.active_operations(), 0);
}

#[test]
fn zero_byte_load_is_rejected() {
    let mut registry = manual_registry();
    let load_key = key(1, 1);
    let error = registry
        .attach_waiter(
            waiter(1),
            [LoadRequest::new(load_key, 0, ResourceClass::Decode)],
            1,
        )
        .unwrap_err();
    assert!(matches!(error, RegistryError::ZeroByteLoad(_)));
}

#[test]
fn transition_larger_than_configured_max_is_rejected() {
    let mut registry = manual_registry();
    let load_key = key(1, 1);
    let maximum = FairQueueConfig::default().max_transition_cost;
    let error = registry
        .attach_waiter(
            waiter(1),
            [LoadRequest::new(
                load_key,
                maximum + 1,
                ResourceClass::Decode,
            )],
            1,
        )
        .unwrap_err();
    assert!(matches!(error, RegistryError::Fairness(_)));
}

#[test]
fn cancel_one_shared_waiter_keeps_demand_load() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    let operation = attach_one(&mut registry, waiter(1), load_key);
    registry
        .attach_waiter(waiter(2), [request(load_key)], 2)
        .unwrap();
    registry.schedule_one(10).unwrap();
    registry
        .detach_waiter(waiter(1), CancellationReason::ExternalRequest, 20)
        .unwrap();
    assert_eq!(registry.waiters().waiter_count(operation), 1);
    assert_eq!(registry.backend().cancellations(), 0);
}

#[test]
fn cancel_last_reserved_waiter_retires_immediately() {
    let mut registry = auto_registry();
    let operation = attach_one(&mut registry, waiter(1), key(1, 1));
    registry
        .detach_waiter(waiter(1), CancellationReason::ExternalRequest, 20)
        .unwrap();
    assert!(registry.operation(operation).is_none());
    assert!(registry.retirement(operation).is_some());
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn cancel_last_submitted_waiter_waits_for_completion() {
    let mut registry = auto_registry();
    let operation = attach_one(&mut registry, waiter(1), key(1, 1));
    registry.schedule_one(19).unwrap();
    registry.schedule_one(20).unwrap();
    registry
        .detach_waiter(waiter(1), CancellationReason::ExternalRequest, 21)
        .unwrap();
    assert!(registry.operation(operation).is_some());
    assert_eq!(registry.backend().cancellations(), 1);
    registry.collect_backend_completions(8);
    registry.process_one_completion().unwrap();
    assert!(registry.operation(operation).is_none());
}

#[test]
fn queued_read_success_before_last_waiter_cancel_retries_cleanup_at_host_ready() {
    let mut registry = auto_registry();
    let operation = attach_one(&mut registry, waiter(1), key(1, 1));
    registry.schedule_one(19).unwrap();
    registry.schedule_one(20).unwrap();

    assert_eq!(registry.collect_backend_completions(1), 1);
    registry
        .detach_waiter(waiter(1), CancellationReason::ExternalRequest, 21)
        .unwrap();
    assert!(
        registry
            .operation(operation)
            .unwrap()
            .cancellation_requested()
    );
    assert_eq!(registry.backend().cancellations(), 1);

    registry.process_one_completion().unwrap();
    assert!(registry.operation(operation).is_none());
    assert_eq!(registry.backend().cancellations(), 2);
    assert_eq!(registry.resources().active_grants(), 0);

    assert_eq!(registry.collect_backend_completions(1), 1);
    assert!(matches!(
        registry.process_one_completion().unwrap(),
        CompletionDisposition::Rejected(CompletionRejectionReason::RetiredOperation)
    ));
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn queued_upload_success_before_last_waiter_cancel_retries_cleanup_at_installing() {
    let mut registry = auto_registry();
    let operation = attach_one(&mut registry, waiter(1), key(1, 1));
    registry.schedule_one(19).unwrap();
    registry.schedule_one(20).unwrap();
    assert_eq!(registry.collect_backend_completions(1), 1);
    registry.process_one_completion().unwrap();
    registry.schedule_one(21).unwrap();

    assert_eq!(registry.collect_backend_completions(1), 1);
    registry
        .detach_waiter(waiter(1), CancellationReason::OwnerShutdown, 22)
        .unwrap();
    assert!(
        registry
            .operation(operation)
            .unwrap()
            .cancellation_requested()
    );
    assert_eq!(registry.backend().cancellations(), 1);

    registry.process_one_completion().unwrap();
    assert!(registry.operation(operation).is_none());
    assert_eq!(registry.backend().cancellations(), 2);
    assert_eq!(registry.resources().active_grants(), 0);

    assert_eq!(registry.collect_backend_completions(1), 1);
    assert!(matches!(
        registry.process_one_completion().unwrap(),
        CompletionDisposition::Rejected(CompletionRejectionReason::RetiredOperation)
    ));
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn draining_stage_history_is_exactly_once() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    let operation = attach_one(&mut registry, waiter(1), load_key);
    registry.schedule_one(9).unwrap();
    registry.schedule_one(10).unwrap();
    registry
        .detach_waiter(waiter(1), CancellationReason::ExternalRequest, 20)
        .unwrap();
    registry.collect_backend_completions(1);
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.stage_history(operation).unwrap(),
        &[
            LoadStage::Reserved,
            LoadStage::ReadSubmitted,
            LoadStage::Draining,
            LoadStage::Retired,
        ]
    );
}

#[test]
fn cancel_all_shared_waiters_requests_one_physical_cancel() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    attach_one(&mut registry, waiter(1), load_key);
    registry
        .attach_waiter(waiter(2), [request(load_key)], 2)
        .unwrap();
    registry
        .attach_waiter(waiter(3), [request(load_key)], 3)
        .unwrap();
    registry.schedule_one(10).unwrap();
    for id in 1..=3 {
        registry
            .detach_waiter(waiter(id), CancellationReason::ExternalRequest, 20 + id)
            .unwrap();
    }
    assert_eq!(registry.backend().cancellations(), 1);
}

#[test]
fn duplicate_detach_is_detected() {
    let mut registry = auto_registry();
    attach_one(&mut registry, waiter(1), key(1, 1));
    registry
        .detach_waiter(waiter(1), CancellationReason::ExternalRequest, 20)
        .unwrap();
    assert!(
        registry
            .detach_waiter(waiter(1), CancellationReason::ExternalRequest, 21)
            .is_err()
    );
}

#[test]
fn demand_load_is_not_cancelled_while_any_waiter_remains() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    attach_one(&mut registry, waiter(1), load_key);
    for id in 2..=8 {
        registry
            .attach_waiter(waiter(id), [request(load_key)], id)
            .unwrap();
    }
    registry.schedule_one(10).unwrap();
    for id in 1..8 {
        registry
            .detach_waiter(waiter(id), CancellationReason::ExternalRequest, 20 + id)
            .unwrap();
    }
    assert_eq!(registry.backend().cancellations(), 0);
}

#[test]
fn read_failure_fails_waiter_and_retires() {
    let (mut registry, load_key, operation) = manual_at_read();
    registry.enqueue_completion(event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Failed(FailureReason::StorageUnavailable),
        0,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert!(registry.operation(operation).is_none());
    assert!(matches!(
        registry.pop_failed().unwrap().failure,
        ContinuationFailure::Failed(FailureReason::StorageUnavailable)
    ));
}

#[test]
fn failure_stage_history_is_exactly_once() {
    let (mut registry, load_key, operation) = manual_at_read();
    registry.enqueue_completion(event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Failed(FailureReason::StorageUnavailable),
        0,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.stage_history(operation).unwrap(),
        &[
            LoadStage::Reserved,
            LoadStage::ReadSubmitted,
            LoadStage::Failed,
            LoadStage::Retired,
        ]
    );
}

#[test]
fn upload_failure_fails_waiter_and_retires() {
    let (mut registry, load_key, operation) = manual_at_upload();
    registry.enqueue_completion(event(
        operation,
        load_key,
        LoadStage::UploadSubmitted,
        CompletionOutcome::Failed(FailureReason::UploadRejected),
        0,
        50,
    ));
    registry.process_one_completion().unwrap();
    assert!(registry.operation(operation).is_none());
    assert!(registry.pop_failed().is_some());
}

#[test]
fn install_failure_fails_waiter_and_retires() {
    let (mut registry, load_key, operation) = manual_at_install();
    registry.enqueue_completion(event(
        operation,
        load_key,
        LoadStage::Installing,
        CompletionOutcome::Failed(FailureReason::InstallationRejected),
        0,
        70,
    ));
    registry.process_one_completion().unwrap();
    assert!(registry.operation(operation).is_none());
    assert!(registry.pop_failed().is_some());
}

#[test]
fn stale_read_never_publishes() {
    let (mut registry, load_key, operation) = manual_at_read();
    let stale = StaleReason::DestinationReused;
    registry.enqueue_completion(event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Stale(stale.clone()),
        0,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert!(registry.residency_binding(load_key).is_none());
    assert!(matches!(
        registry.pop_failed().unwrap().failure,
        ContinuationFailure::Stale(reason) if reason == stale
    ));
}

#[test]
fn stale_stage_history_is_exactly_once() {
    let (mut registry, load_key, operation) = manual_at_read();
    registry.enqueue_completion(event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Stale(StaleReason::DestinationReused),
        0,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.stage_history(operation).unwrap(),
        &[
            LoadStage::Reserved,
            LoadStage::ReadSubmitted,
            LoadStage::Stale,
            LoadStage::Retired,
        ]
    );
}

#[test]
fn stale_upload_never_publishes() {
    let (mut registry, load_key, operation) = manual_at_upload();
    registry.enqueue_completion(event(
        operation,
        load_key,
        LoadStage::UploadSubmitted,
        CompletionOutcome::Stale(StaleReason::SupersededOperation),
        0,
        50,
    ));
    registry.process_one_completion().unwrap();
    assert!(registry.residency_binding(load_key).is_none());
}

#[test]
fn stale_install_never_publishes() {
    let (mut registry, load_key, operation) = manual_at_install();
    registry.enqueue_completion(event(
        operation,
        load_key,
        LoadStage::Installing,
        CompletionOutcome::Stale(StaleReason::DestinationReused),
        0,
        70,
    ));
    registry.process_one_completion().unwrap();
    assert!(registry.residency_binding(load_key).is_none());
}

#[test]
fn wrong_generation_completion_is_fail_closed() {
    let (mut registry, load_key, operation) = manual_at_read();
    let mut completion = success_event(operation, load_key, LoadStage::ReadSubmitted, 30);
    completion.generation =
        CompletionGeneration::new(SourceGeneration::new(1), DestinationGeneration::new(999));
    registry.enqueue_completion(completion);
    assert!(matches!(
        registry.process_one_completion().unwrap(),
        CompletionDisposition::Rejected(_)
    ));
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
    assert!(registry.resources().in_use(ResourceKind::Sqe) > 0);
}

#[test]
fn wrong_operation_completion_is_fail_closed() {
    let (mut registry, load_key, operation) = manual_at_read();
    registry.enqueue_completion(success_event(
        OperationId::new(999),
        load_key,
        LoadStage::ReadSubmitted,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
}

#[test]
fn wrong_key_completion_is_fail_closed() {
    let (mut registry, _load_key, operation) = manual_at_read();
    registry.enqueue_completion(success_event(
        operation,
        key(2, 1),
        LoadStage::ReadSubmitted,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
}

#[test]
fn wrong_stage_completion_is_fail_closed() {
    let (mut registry, load_key, operation) = manual_at_read();
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::UploadSubmitted,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
}

#[test]
fn short_success_completion_is_rejected() {
    let (mut registry, load_key, operation) = manual_at_read();
    registry.enqueue_completion(event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Succeeded,
        BYTES - 1,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
}

#[test]
fn duplicate_completion_is_diagnosed_without_double_release() {
    let (mut registry, load_key, operation) = manual_at_read();
    let completion = success_event(operation, load_key, LoadStage::ReadSubmitted, 30);
    registry.enqueue_completion(completion.clone());
    registry.enqueue_completion(completion);
    registry.process_one_completion().unwrap();
    let read_bytes = registry.resources().in_use(ResourceKind::ReadBytes);
    assert!(matches!(
        registry.process_one_completion().unwrap(),
        CompletionDisposition::Rejected(_)
    ));
    assert_eq!(
        registry.resources().in_use(ResourceKind::ReadBytes),
        read_bytes
    );
}

#[test]
fn out_of_order_completion_does_not_advance_owner_stage() {
    let (mut registry, load_key, operation) = manual_at_read();
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::Installing,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
}

#[test]
fn unknown_completion_is_retained_as_rejection() {
    let mut registry = manual_registry();
    let load_key = key(1, 1);
    registry.enqueue_completion(success_event(
        OperationId::new(99),
        load_key,
        LoadStage::ReadSubmitted,
        1,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(registry.rejected_completions().len(), 1);
}

#[test]
fn lost_completion_keeps_submitted_credit_and_never_publishes() {
    let mut backend = FakeBackend::new();
    backend.lose_next(LoadStage::ReadSubmitted);
    let mut registry = LoadRegistry::with_testing_resources(backend).unwrap();
    let load_key = key(1, 1);
    let operation = attach_one(&mut registry, waiter(1), load_key);
    registry.schedule_one(19).unwrap();
    registry.schedule_one(20).unwrap();
    assert_eq!(registry.collect_backend_completions(8), 0);
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
    assert_eq!(registry.resources().in_use(ResourceKind::Sqe), 1);
    assert!(registry.residency_binding(load_key).is_none());
}

#[test]
fn completion_fifo_rejects_bad_then_applies_good() {
    let (mut registry, load_key, operation) = manual_at_read();
    let mut bad = success_event(operation, load_key, LoadStage::ReadSubmitted, 30);
    bad.generation.destination = DestinationGeneration::new(99);
    registry.enqueue_completion(bad);
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        31,
    ));
    assert!(matches!(
        registry.process_one_completion().unwrap(),
        CompletionDisposition::Rejected(_)
    ));
    assert!(matches!(
        registry.process_one_completion().unwrap(),
        CompletionDisposition::Applied { .. }
    ));
}

#[test]
fn every_illegal_load_stage_transition_is_rejected() {
    for from in LoadStage::ALL {
        for to in LoadStage::ALL {
            if !from.can_transition_to(to) {
                assert!(from.validate_transition(to).is_err(), "{from:?} -> {to:?}");
            }
        }
    }
}

#[test]
fn pinned_slab_exhaustion_requeues_without_failing_attach() {
    let resources = hard_broker_with(ResourceKind::PinnedSlab, BYTES, 0);
    let mut registry =
        LoadRegistry::new(FakeBackend::manual(), resources, FairQueueConfig::default()).unwrap();
    let first = attach_one(&mut registry, waiter(1), key(1, 1));
    let second = attach_one(&mut registry, waiter(2), key(2, 1));

    assert!(registry.schedule_one(10).unwrap());
    assert!(registry.schedule_one(11).unwrap());
    assert!(!registry.schedule_one(12).unwrap());

    assert_eq!(
        registry.operation(first).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
    assert_eq!(
        registry.operation(second).unwrap().stage(),
        LoadStage::Reserved
    );
    assert_eq!(registry.resources().in_use(ResourceKind::PinnedSlab), BYTES);
    assert_eq!(registry.backend().physical_reads(), 1);
    assert!(registry.pop_failed().is_none());
}

#[test]
fn demand_reserve_requeues_prefetch_and_allows_decode_progress() {
    let mut resources = staged_broker(2, 2, 2, 2);
    resources
        .reconfigure_limit(ResourceKind::PinnedSlab, BYTES * 2, BYTES)
        .unwrap();
    let mut base = resources
        .acquire(
            99,
            ResourceClass::Prefill,
            [HardResourceClaim::new(ResourceKind::PinnedSlab, BYTES)],
        )
        .unwrap();
    let mut registry =
        LoadRegistry::new(FakeBackend::manual(), resources, FairQueueConfig::default()).unwrap();
    let prefetch = registry
        .attach_waiter(
            waiter(1),
            [LoadRequest::new(key(1, 1), BYTES, ResourceClass::Prefetch)],
            1,
        )
        .unwrap()
        .created[0];
    let decode = attach_one(&mut registry, waiter(2), key(2, 1));

    assert!(registry.schedule_one(20).unwrap());
    assert!(registry.schedule_one(21).unwrap());
    assert_eq!(
        registry.operation(prefetch).unwrap().stage(),
        LoadStage::Reserved
    );
    assert_eq!(
        registry.operation(decode).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
    assert_eq!(registry.backend().physical_reads(), 1);
    assert!(registry.pop_failed().is_none());

    registry.release_hard_resources(&mut base).unwrap();
    registry.begin_shutdown(30).unwrap();
    registry.enqueue_completion(event(
        decode,
        registry.key_for_operation(decode).unwrap(),
        LoadStage::ReadSubmitted,
        CompletionOutcome::Cancelled(CancellationReason::OwnerShutdown),
        0,
        31,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn upload_backpressure_allows_reads_to_continue_to_pinned_capacity() {
    let resources = staged_broker(2, 4, 1, 4);
    let mut registry =
        LoadRegistry::new(FakeBackend::manual(), resources, FairQueueConfig::default()).unwrap();
    let operations = (1..=4)
        .map(|seed| attach_one(&mut registry, waiter(seed), key(seed as u8, 1)))
        .collect::<Vec<_>>();

    while registry.schedule_one(20).unwrap() {}
    assert_eq!(registry.backend().physical_reads(), 2);
    for (index, operation) in operations[..2].iter().enumerate() {
        let load_key = registry.key_for_operation(*operation).unwrap();
        registry.enqueue_completion(success_event(
            *operation,
            load_key,
            LoadStage::ReadSubmitted,
            30 + index as u64,
        ));
        registry.process_one_completion().unwrap();
    }

    while registry.schedule_one(40).unwrap() {}

    assert_eq!(registry.backend().physical_uploads(), 1);
    assert_eq!(registry.backend().physical_reads(), 4);
    assert_eq!(
        operations
            .iter()
            .filter(|operation| registry.operation(**operation).unwrap().stage()
                == LoadStage::ReadSubmitted)
            .count(),
        2
    );
    assert_eq!(
        registry.resources().in_use(ResourceKind::PinnedSlab),
        BYTES * 4
    );
    assert!(registry.pop_failed().is_none());

    registry.begin_shutdown(50).unwrap();
    for operation in [operations[0], operations[2], operations[3]] {
        let stage = if operation == operations[0] {
            LoadStage::UploadSubmitted
        } else {
            LoadStage::ReadSubmitted
        };
        registry.enqueue_completion(event(
            operation,
            registry.key_for_operation(operation).unwrap(),
            stage,
            CompletionOutcome::Cancelled(CancellationReason::OwnerShutdown),
            0,
            51,
        ));
        registry.process_one_completion().unwrap();
    }
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn single_operation_larger_than_physical_capacity_is_permanent_attach_error() {
    let resources = hard_broker_with(ResourceKind::PinnedSlab, BYTES - 1, 0);
    let mut registry =
        LoadRegistry::new(FakeBackend::manual(), resources, FairQueueConfig::default()).unwrap();
    let error = registry
        .attach_waiter(waiter(1), [request(key(1, 1))], 1)
        .unwrap_err();
    assert!(matches!(
        error,
        RegistryError::Resources(HardResourceError::ExceedsCapacity {
            kind: ResourceKind::PinnedSlab,
            requested: BYTES,
            capacity,
        }) if capacity == BYTES - 1
    ));
    assert_eq!(registry.active_operations(), 0);
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn cancelling_unsubmitted_load_releases_credit_for_next_load() {
    let resources = hard_broker_with(ResourceKind::ReadBytes, BYTES, 0);
    let mut registry =
        LoadRegistry::new(FakeBackend::manual(), resources, FairQueueConfig::default()).unwrap();
    attach_one(&mut registry, waiter(1), key(1, 1));
    registry
        .detach_waiter(waiter(1), CancellationReason::ExternalRequest, 2)
        .unwrap();
    assert_eq!(registry.resources().in_use(ResourceKind::ReadBytes), 0);
    assert!(
        registry
            .attach_waiter(waiter(2), [request(key(2, 1))], 3)
            .is_ok()
    );
}

#[test]
fn submitted_hard_grant_cannot_be_revoked() {
    let mut broker = HardResourceBroker::testing_default();
    let mut grant = broker
        .acquire(
            1,
            ResourceClass::Decode,
            [HardResourceClaim::new(ResourceKind::Sqe, 1)],
        )
        .unwrap();
    broker
        .mark_submitted(&mut grant, &[ResourceKind::Sqe])
        .unwrap();
    assert!(matches!(
        broker.release_all_held(&mut grant),
        Err(HardResourceError::SubmittedClaimCannotBeRevoked { .. })
    ));
    assert_eq!(broker.in_use(ResourceKind::Sqe), 1);
}

#[test]
fn hard_grant_return_requires_submitted_identity() {
    let mut broker = HardResourceBroker::testing_default();
    let mut grant = broker
        .acquire(
            1,
            ResourceClass::Decode,
            [HardResourceClaim::new(ResourceKind::Sqe, 1)],
        )
        .unwrap();
    assert!(matches!(
        broker.mark_returned(&mut grant, &[ResourceKind::Sqe]),
        Err(HardResourceError::NotSubmitted { .. })
    ));
}

#[test]
fn hard_catalog_contains_every_required_kind() {
    let broker = HardResourceBroker::testing_default();
    let kinds: Vec<_> = broker.snapshots().map(|snapshot| snapshot.kind).collect();
    assert_eq!(kinds.len(), ResourceKind::ALL.len());
    for kind in ResourceKind::ALL {
        assert!(kinds.contains(&kind));
    }
}

#[test]
fn decode_can_use_hard_reserve() {
    let mut broker = hard_broker_with(ResourceKind::Continuation, 2, 1);
    let _base = broker
        .acquire(
            1,
            ResourceClass::Prefill,
            [HardResourceClaim::new(ResourceKind::Continuation, 1)],
        )
        .unwrap();
    assert!(
        broker
            .acquire(
                2,
                ResourceClass::Decode,
                [HardResourceClaim::new(ResourceKind::Continuation, 1)],
            )
            .is_ok()
    );
}

#[test]
fn prefill_cannot_consume_decode_verification_reserve() {
    let mut broker = hard_broker_with(ResourceKind::Continuation, 2, 1);
    let _base = broker
        .acquire(
            1,
            ResourceClass::Prefill,
            [HardResourceClaim::new(ResourceKind::Continuation, 1)],
        )
        .unwrap();
    assert!(matches!(
        broker.acquire(
            2,
            ResourceClass::Prefill,
            [HardResourceClaim::new(ResourceKind::Continuation, 1)],
        ),
        Err(HardResourceError::DemandReserve { .. })
    ));
}

#[test]
fn multi_claim_hard_admission_is_atomic() {
    let mut broker = hard_broker_with(ResourceKind::Sqe, 1, 0);
    let error = broker
        .acquire(
            1,
            ResourceClass::Decode,
            [
                HardResourceClaim::new(ResourceKind::Sqe, 2),
                HardResourceClaim::new(ResourceKind::PinnedSlab, 1),
            ],
        )
        .unwrap_err();
    assert!(matches!(error, HardResourceError::ExceedsCapacity { .. }));
    assert_eq!(broker.in_use(ResourceKind::PinnedSlab), 0);
}

#[test]
fn hard_claims_release_at_exact_stage_boundaries() {
    let (mut registry, load_key, operation) = manual_at_read();
    assert_eq!(registry.resources().in_use(ResourceKind::ReadBytes), BYTES);
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(registry.resources().in_use(ResourceKind::ReadBytes), 0);
    assert_eq!(registry.resources().in_use(ResourceKind::PinnedSlab), BYTES);
    registry.schedule_one(40).unwrap();
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::UploadSubmitted,
        50,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(registry.resources().in_use(ResourceKind::PinnedSlab), 0);
    assert_eq!(registry.resources().in_use(ResourceKind::UploadBytes), 0);
    assert_eq!(
        registry.resources().in_use(ResourceKind::ExpertFrame),
        BYTES
    );
}

fn production_fairness_limits(
    storage_read_bytes: u64,
    pinned_host_bytes: u64,
    h2d_bytes: u64,
    device_install_bytes: u64,
) -> ExpertIoResourceLimits {
    ExpertIoResourceLimits {
        capacity: ExpertIoResourceDemand {
            read_slots: 1,
            storage_read_bytes,
            pinned_host_bytes,
            upload_slots: 1,
            h2d_bytes,
            install_slots: 1,
            device_install_bytes,
        },
        demand_reserve: ExpertIoResourceDemand::default(),
    }
    .validate()
    .unwrap()
}

#[test]
fn production_fairness_accepts_large_expert_and_rejects_capacity_plus_one() {
    const EXPERT_BYTES: u64 = 13_369_344;

    let config = FairQueueConfig::for_production(production_fairness_limits(
        EXPERT_BYTES,
        EXPERT_BYTES,
        EXPERT_BYTES,
        EXPERT_BYTES,
    ))
    .unwrap();
    assert_eq!(config.max_transition_cost, EXPERT_BYTES);

    let mut queue = FairQueue::new(config).unwrap();
    queue
        .push('e', ResourceClass::Decode, EXPERT_BYTES, 0)
        .unwrap();
    assert!(matches!(
        queue.push('x', ResourceClass::Decode, EXPERT_BYTES + 1, 0),
        Err(FairQueueError::TransitionTooLarge { cost, maximum })
            if cost == EXPERT_BYTES + 1 && maximum == EXPERT_BYTES
    ));
}

#[test]
fn production_fairness_uses_largest_physical_stage_byte_capacity() {
    for (limits, expected) in [
        (production_fairness_limits(17, 5, 7, 11), 17),
        (production_fairness_limits(3, 19, 7, 11), 19),
        (production_fairness_limits(3, 5, 23, 11), 23),
        (production_fairness_limits(3, 5, 7, 29), 29),
    ] {
        let config = FairQueueConfig::for_production(limits).unwrap();
        assert_eq!(config.max_transition_cost, expected);
    }
}

#[test]
fn production_fairness_rejects_zero_and_out_of_signed_range_capacity() {
    assert!(matches!(
        FairQueueConfig::for_production(ExpertIoResourceLimits::default().validate().unwrap()),
        Err(FairQueueError::ZeroMaxTransitionCost)
    ));

    let signed_max = i64::MAX as u64;
    let config = FairQueueConfig::for_production(production_fairness_limits(
        signed_max, signed_max, signed_max, signed_max,
    ))
    .unwrap();
    for value in [
        config.prefetch_quantum,
        config.prefill_quantum,
        config.verification_quantum,
        config.decode_quantum,
        config.max_surplus,
        config.debt_limit,
        config.max_transition_cost,
    ] {
        assert!(value <= signed_max);
    }
    let mut queue = FairQueue::new(config).unwrap();
    queue
        .push('x', ResourceClass::Decode, signed_max, 0)
        .unwrap();
    assert_eq!(queue.pop_next(0, |_| true), Some('x'));

    assert!(matches!(
        FairQueueConfig::for_production(production_fairness_limits(signed_max + 1, 1, 1, 1,)),
        Err(FairQueueError::CostOutOfRange)
    ));
}

#[test]
fn production_fairness_keeps_prefill_progress_bounded_under_decode_pressure() {
    const EXPERT_BYTES: u64 = 13_369_344;

    let config = FairQueueConfig::for_production(production_fairness_limits(
        EXPERT_BYTES,
        EXPERT_BYTES,
        EXPERT_BYTES,
        EXPERT_BYTES,
    ))
    .unwrap();
    assert_eq!(config.verification_quantum, EXPERT_BYTES);
    assert_eq!(config.decode_quantum, EXPERT_BYTES);
    assert!(config.prefill_quantum > config.prefetch_quantum);
    assert!(config.prefill_quantum < config.decode_quantum);

    let mut queue = FairQueue::new(config).unwrap();
    queue
        .push('p', ResourceClass::Prefill, EXPERT_BYTES, 0)
        .unwrap();
    let mut selected_at = None;
    for tick in 0..=config.starvation_ticks {
        queue
            .push('d', ResourceClass::Decode, EXPERT_BYTES, tick)
            .unwrap();
        if queue.pop_next(tick, |_| true) == Some('p') {
            selected_at = Some(tick);
            break;
        }
    }
    assert!(selected_at.is_some_and(|tick| tick <= config.starvation_ticks));
}

fn fairness_config() -> FairQueueConfig {
    FairQueueConfig {
        prefetch_quantum: 1,
        prefill_quantum: 1,
        verification_quantum: 1,
        decode_quantum: 1,
        max_surplus: 16,
        debt_limit: 8,
        starvation_ticks: 2,
        max_transition_cost: 8,
    }
}

#[test]
fn fair_queue_prefers_decode_on_equal_age() {
    let mut queue = FairQueue::new(fairness_config()).unwrap();
    queue.push('p', ResourceClass::Prefill, 1, 0).unwrap();
    queue.push('d', ResourceClass::Decode, 1, 0).unwrap();
    assert_eq!(queue.pop_next(0, |_| true), Some('d'));
}

#[test]
fn aged_prefill_forces_progress_under_decode_pressure() {
    let mut config = fairness_config();
    config.decode_quantum = 8;
    let mut queue = FairQueue::new(config).unwrap();
    queue.push('p', ResourceClass::Prefill, 4, 0).unwrap();
    for tick in 0..2 {
        queue.push('d', ResourceClass::Decode, 1, tick).unwrap();
        assert_eq!(queue.pop_next(tick, |_| true), Some('d'));
    }
    queue.push('d', ResourceClass::Decode, 1, 2).unwrap();
    assert_eq!(queue.pop_next(2, |_| true), Some('p'));
}

#[test]
fn prefetch_has_no_aging_forced_progress() {
    let mut queue = FairQueue::new(fairness_config()).unwrap();
    queue.push('x', ResourceClass::Prefetch, 4, 0).unwrap();
    assert_eq!(queue.pop_next(100, |_| true), None);
}

#[test]
fn fairness_never_bypasses_hard_feasibility() {
    let mut queue = FairQueue::new(fairness_config()).unwrap();
    queue.push('x', ResourceClass::Decode, 1, 0).unwrap();
    assert_eq!(queue.pop_next(100, |_| false), None);
}

#[test]
fn fairness_skips_infeasible_head_for_feasible_work() {
    let mut queue = FairQueue::new(fairness_config()).unwrap();
    queue.push('x', ResourceClass::Decode, 1, 0).unwrap();
    queue.push('y', ResourceClass::Prefill, 1, 0).unwrap();
    assert_eq!(queue.pop_next(0, |item| *item == 'y'), Some('y'));
}

#[test]
fn fair_queue_rejects_cost_over_max_transition() {
    let mut queue = FairQueue::new(fairness_config()).unwrap();
    assert!(matches!(
        queue.push('x', ResourceClass::Decode, 9, 0),
        Err(FairQueueError::TransitionTooLarge { .. })
    ));
}

#[test]
fn forced_progress_debt_stays_bounded() {
    let mut config = fairness_config();
    config.prefill_quantum = 1;
    config.starvation_ticks = 0;
    config.debt_limit = 3;
    let mut queue = FairQueue::new(config).unwrap();
    queue.push('p', ResourceClass::Prefill, 4, 0).unwrap();
    assert_eq!(queue.pop_next(0, |_| true), Some('p'));
    assert_eq!(queue.deficit(ResourceClass::Prefill), -3);
}

#[test]
fn fair_queue_rejects_zero_quantum_configuration() {
    let mut config = fairness_config();
    config.decode_quantum = 0;
    assert!(matches!(
        FairQueue::<u8>::new(config),
        Err(FairQueueError::ZeroQuantum)
    ));
}

#[test]
fn waiter_index_updates_both_directions() {
    let mut index = WaiterIndex::new();
    let waiter_id = waiter(1);
    index
        .register(waiter_id, [OperationId::new(1), OperationId::new(2)])
        .unwrap();
    assert_eq!(index.waiter_count(OperationId::new(1)), 1);
    assert_eq!(index.loads_for(waiter_id).unwrap().len(), 2);
    index.detach_waiter(waiter_id).unwrap();
    assert_eq!(index.waiter_count(OperationId::new(1)), 0);
}

#[test]
fn continuation_unresolved_set_requires_all_dependencies() {
    let mut index = WaiterIndex::new();
    let continuation = ContinuationId::new(1);
    index
        .register(waiter(1), [OperationId::new(1), OperationId::new(2)])
        .unwrap();
    index.satisfy_operation(OperationId::new(1));
    assert_eq!(index.unresolved_for(continuation).unwrap().len(), 1);
    assert_eq!(index.pop_ready(), None);
    index.satisfy_operation(OperationId::new(2));
    assert_eq!(index.pop_ready(), Some(continuation));
}

#[test]
fn targeted_ready_does_not_wake_unrelated_continuation() {
    let mut index = WaiterIndex::new();
    index.register(waiter(1), [OperationId::new(1)]).unwrap();
    index.register(waiter(2), [OperationId::new(2)]).unwrap();
    index.satisfy_operation(OperationId::new(1));
    assert_eq!(index.pop_ready(), Some(ContinuationId::new(1)));
    assert_eq!(index.pop_ready(), None);
    assert_eq!(index.waiter_count(OperationId::new(2)), 1);
}

#[test]
fn fake_backend_reverse_completion_wakes_only_targeted_continuation() {
    let mut registry = manual_registry();
    let first_key = key(1, 1);
    let second_key = key(2, 1);
    let first = attach_one(&mut registry, waiter(1), first_key);
    let second = attach_one(&mut registry, waiter(2), second_key);
    for now in 20..24 {
        assert!(registry.schedule_one(now).unwrap());
    }

    registry.enqueue_completion(success_event(
        second,
        second_key,
        LoadStage::ReadSubmitted,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert!(registry.schedule_one(31).unwrap());
    registry.enqueue_completion(success_event(
        second,
        second_key,
        LoadStage::UploadSubmitted,
        32,
    ));
    registry.process_one_completion().unwrap();
    assert!(registry.schedule_one(33).unwrap());
    registry.enqueue_completion(success_event(second, second_key, LoadStage::Installing, 34));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.pop_ready(35).unwrap(),
        Some(ContinuationId::new(2))
    );
    assert_eq!(registry.pop_ready(35).unwrap(), None);
    assert_eq!(
        registry.operation(first).unwrap().stage(),
        LoadStage::ReadSubmitted
    );

    registry.enqueue_completion(success_event(
        first,
        first_key,
        LoadStage::ReadSubmitted,
        40,
    ));
    registry.process_one_completion().unwrap();
    assert!(registry.schedule_one(41).unwrap());
    registry.enqueue_completion(success_event(
        first,
        first_key,
        LoadStage::UploadSubmitted,
        42,
    ));
    registry.process_one_completion().unwrap();
    assert!(registry.schedule_one(43).unwrap());
    registry.enqueue_completion(success_event(first, first_key, LoadStage::Installing, 44));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.pop_ready(45).unwrap(),
        Some(ContinuationId::new(1))
    );
}

#[test]
fn duplicate_waiter_identity_is_rejected_after_completion() {
    let mut index = WaiterIndex::new();
    index.register(waiter(1), [OperationId::new(1)]).unwrap();
    index.satisfy_operation(OperationId::new(1));
    assert!(matches!(
        index.register(waiter(1), [OperationId::new(2)]),
        Err(WaiterIndexError::DuplicateWaiter(_))
    ));
}

#[test]
fn failure_is_targeted_and_unrelated_load_remains() {
    let mut registry = manual_registry();
    let first_key = key(1, 1);
    let second_key = key(2, 1);
    let first = attach_one(&mut registry, waiter(1), first_key);
    let second = attach_one(&mut registry, waiter(2), second_key);
    for now in 20..24 {
        registry.schedule_one(now).unwrap();
    }
    registry.enqueue_completion(event(
        first,
        first_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Failed(FailureReason::StorageUnavailable),
        0,
        30,
    ));
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.pop_failed().unwrap().continuation,
        ContinuationId::new(1)
    );
    assert!(registry.operation(second).is_some());
}

#[test]
fn shared_wait_is_not_multiplied_by_waiter_count() {
    let mut ledger = CriticalPathLedger::new();
    ledger
        .record_shared_wait(OperationId::new(1), CohortId::new(1), 0, 10)
        .unwrap();
    ledger
        .record_shared_wait(OperationId::new(1), CohortId::new(2), 0, 10)
        .unwrap();
    let snapshot = ledger
        .snapshot_output(
            OutputTokenId::new(1),
            1,
            [OperationId::new(1)],
            [CohortId::new(1), CohortId::new(2)],
            10,
        )
        .unwrap();
    assert_eq!(snapshot.wait_ns, 10);
    assert_eq!(snapshot.uncovered_wait_ns, 10);
}

#[test]
fn runnable_overlap_reduces_only_uncovered_wait() {
    let mut ledger = CriticalPathLedger::new();
    ledger
        .record_shared_wait(OperationId::new(1), CohortId::new(1), 0, 20)
        .unwrap();
    ledger.record_runnable_work(5, 15).unwrap();
    let snapshot = ledger
        .snapshot_output(
            OutputTokenId::new(1),
            1,
            [OperationId::new(1)],
            [CohortId::new(1)],
            20,
        )
        .unwrap();
    assert_eq!(snapshot.wait_ns, 20);
    assert_eq!(snapshot.covered_wait_ns, 10);
    assert_eq!(snapshot.uncovered_wait_ns, 10);
}

#[test]
fn overlapping_different_waits_are_unioned_on_critical_path() {
    let mut ledger = CriticalPathLedger::new();
    ledger
        .record_shared_wait(OperationId::new(1), CohortId::new(1), 0, 10)
        .unwrap();
    ledger
        .record_shared_wait(OperationId::new(2), CohortId::new(1), 5, 15)
        .unwrap();
    let snapshot = ledger
        .snapshot_output(
            OutputTokenId::new(1),
            1,
            [OperationId::new(1), OperationId::new(2)],
            [CohortId::new(1)],
            15,
        )
        .unwrap();
    assert_eq!(snapshot.wait_ns, 15);
}

#[test]
fn output_token_snapshot_records_external_commit_count_and_is_exactly_once() {
    let mut ledger = CriticalPathLedger::new();
    let snapshot = ledger
        .snapshot_output(OutputTokenId::new(1), 3, [], [], 10)
        .unwrap();
    assert_eq!(snapshot.externally_committed_tokens, 3);
    assert!(matches!(
        ledger.snapshot_output(OutputTokenId::new(1), 3, [], [], 11),
        Err(LedgerError::DuplicateOutputToken(_))
    ));
}

#[test]
fn ledger_reports_per_operation_read_upload_publish() {
    let mut ledger = CriticalPathLedger::new();
    let operation = OperationId::new(1);
    ledger
        .record_operation_phase(operation, CriticalPhase::Read, 0, 4)
        .unwrap();
    ledger
        .record_operation_phase(operation, CriticalPhase::Upload, 4, 7)
        .unwrap();
    ledger
        .record_operation_phase(operation, CriticalPhase::Publish, 7, 9)
        .unwrap();
    let snapshot = ledger
        .snapshot_output(OutputTokenId::new(1), 1, [operation], [], 9)
        .unwrap();
    let phases = snapshot.operation_phases.get(&operation).unwrap();
    assert_eq!(
        (phases.read_ns, phases.upload_ns, phases.publish_ns),
        (4, 3, 2)
    );
}

#[test]
fn ledger_reports_per_cohort_wait_and_resume() {
    let mut ledger = CriticalPathLedger::new();
    let cohort = CohortId::new(1);
    ledger
        .record_shared_wait(OperationId::new(1), cohort, 0, 5)
        .unwrap();
    ledger
        .record_cohort_phase(cohort, CriticalPhase::Resume, 5, 8)
        .unwrap();
    let snapshot = ledger
        .snapshot_output(OutputTokenId::new(1), 1, [OperationId::new(1)], [cohort], 8)
        .unwrap();
    let phases = snapshot.cohort_phases.get(&cohort).unwrap();
    assert_eq!((phases.wait_ns, phases.resume_ns), (5, 3));
}

#[test]
fn reversed_ledger_span_is_rejected() {
    let mut ledger = CriticalPathLedger::new();
    assert!(matches!(
        ledger.record_runnable_work(2, 1),
        Err(LedgerError::ReversedSpan { .. })
    ));
}

#[test]
fn ready_cohort_credit_exhaustion_delays_resume_without_false_wake() {
    let resources = hard_broker_with(ResourceKind::ReadyCohort, 0, 0);
    let mut registry =
        LoadRegistry::new(FakeBackend::new(), resources, FairQueueConfig::default()).unwrap();
    attach_one(&mut registry, waiter(1), key(1, 1));
    registry.drive(100, 32).unwrap();
    assert_eq!(registry.pop_ready(101).unwrap(), None);
    assert_eq!(registry.waiters().ready_len(), 1);
}

#[test]
fn shutdown_drains_submitted_operation_without_leak() {
    let mut registry = auto_registry();
    attach_one(&mut registry, waiter(1), key(1, 1));
    registry.schedule_one(19).unwrap();
    registry.schedule_one(20).unwrap();
    let report = registry.shutdown(30, 16).unwrap();
    assert!(report.drained);
    assert_eq!(report.active_grants, 0);
}

#[test]
fn shutdown_drains_mixed_queued_and_submitted_without_grant_leak() {
    let resources = staged_broker(1, 1, 1, 3);
    let mut registry =
        LoadRegistry::new(FakeBackend::new(), resources, FairQueueConfig::default()).unwrap();
    for seed in 1..=3 {
        attach_one(&mut registry, waiter(seed), key(seed as u8, 1));
    }
    assert!(registry.schedule_one(10).unwrap());
    assert!(registry.schedule_one(11).unwrap());
    assert!(!registry.schedule_one(12).unwrap());

    let report = registry.shutdown(20, 16).unwrap();
    assert!(report.drained);
    assert_eq!(report.active_grants, 0);
    assert_eq!(registry.resources().active_grants(), 0);
    assert_eq!(
        registry
            .backend()
            .commands()
            .iter()
            .filter(|command| matches!(command, FakeCommand::Reserve(_)))
            .count(),
        1
    );
    assert_eq!(registry.backend().cancellations(), 1);
}

#[test]
fn shutdown_releases_residency_ready_and_continuation_grants() {
    let mut registry = auto_registry();
    attach_one(&mut registry, waiter(1), key(1, 1));
    registry.drive(100, 32).unwrap();
    let report = registry.shutdown(101, 16).unwrap();
    assert!(report.drained);
    assert_eq!(report.active_grants, 0);
}

#[test]
fn shutdown_with_lost_manual_completion_is_explicit_and_retryable_without_leak() {
    let (mut registry, load_key, operation) = manual_at_read();
    let error = registry.shutdown(30, 4).unwrap_err();
    assert!(matches!(
        error,
        RegistryError::LostCompletion {
            operation: lost_operation,
            stage: LoadStage::ReadSubmitted,
            pending_operations: 1,
            active_grants,
        } if lost_operation == operation && active_grants > 0
    ));
    assert!(registry.resources().active_grants() > 0);

    registry.enqueue_completion(event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Cancelled(CancellationReason::OwnerShutdown),
        0,
        31,
    ));
    let report = registry.shutdown(32, 4).unwrap();
    assert!(report.drained);
    assert_eq!(report.active_grants, 0);
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn resident_attach_is_immediately_ready_without_second_read() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    attach_one(&mut registry, waiter(1), load_key);
    registry.drive(100, 32).unwrap();
    registry.pop_ready(101).unwrap();
    let report = registry
        .attach_waiter(waiter(2), [request(load_key)], 102)
        .unwrap();
    assert!(report.continuation_ready);
    assert_eq!(report.already_resident, 1);
    assert_eq!(registry.backend().physical_reads(), 1);
}

#[test]
fn eviction_releases_transferred_frame_without_holding_execution_lease() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    attach_one(&mut registry, waiter(1), load_key);
    registry.drive(100, 32).unwrap();
    assert_eq!(
        registry.resources().in_use(ResourceKind::ExpertFrame),
        BYTES
    );
    assert_eq!(registry.resources().in_use(ResourceKind::Lease), 0);
    assert!(registry.evict(load_key).unwrap());
    assert_eq!(registry.resources().in_use(ResourceKind::ExpertFrame), 0);
    assert_eq!(registry.resources().in_use(ResourceKind::Lease), 0);
}

#[test]
fn execution_lease_is_held_only_from_prepare_through_finish_resume() {
    let mut registry = auto_registry();
    let load_key = key(1, 1);
    let continuation = ContinuationId::new(1);
    attach_one(&mut registry, waiter(1), load_key);
    registry.drive(100, 32).unwrap();
    assert_eq!(registry.pop_ready(101).unwrap(), Some(continuation));
    assert_eq!(registry.resources().in_use(ResourceKind::Lease), 0);

    let dependencies = ferrule_common::io_protocol::DependencySet::new([
        ferrule_common::io_protocol::LogicalDependency::expert_resident(load_key).unwrap(),
    ])
    .unwrap();
    let mut resume = registry
        .prepare_resume(continuation, &dependencies)
        .unwrap();
    assert_eq!(registry.resources().in_use(ResourceKind::Lease), 1);
    assert_eq!(registry.resources().in_use(ResourceKind::Arena), 1);
    let _leases = resume.take().unwrap();
    registry
        .finish_resume(continuation, resume, 102, 103)
        .unwrap();
    assert_eq!(registry.resources().in_use(ResourceKind::Lease), 0);
    assert_eq!(registry.resources().in_use(ResourceKind::Arena), 0);
}

#[test]
fn read_command_rejection_fails_without_submitted_leak() {
    let mut backend = FakeBackend::manual();
    backend.reject_next(LoadStage::ReadSubmitted, FailureReason::ReadRejected);
    let mut registry = LoadRegistry::with_testing_resources(backend).unwrap();
    let operation = attach_one(&mut registry, waiter(1), key(1, 1));
    registry.schedule_one(19).unwrap();
    registry.schedule_one(20).unwrap();
    registry.process_one_completion().unwrap();
    assert!(registry.operation(operation).is_none());
    assert_eq!(registry.resources().in_use(ResourceKind::Sqe), 0);
    assert!(registry.pop_failed().is_some());
}

#[test]
fn upload_command_rejection_fails_and_releases_host_artifact() {
    let (mut registry, _load_key, operation) = manual_at_read();
    let load_key = registry.key_for_operation(operation).unwrap();
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::ReadSubmitted,
        30,
    ));
    registry.process_one_completion().unwrap();
    registry
        .backend_mut()
        .reject_next(LoadStage::UploadSubmitted, FailureReason::UploadRejected);
    registry.schedule_one(40).unwrap();
    registry.process_one_completion().unwrap();
    assert!(registry.operation(operation).is_none());
    assert_eq!(registry.resources().in_use(ResourceKind::PinnedSlab), 0);
}

#[test]
fn install_command_rejection_fails_and_releases_destination() {
    let (mut registry, load_key, operation) = manual_at_upload();
    registry.enqueue_completion(success_event(
        operation,
        load_key,
        LoadStage::UploadSubmitted,
        50,
    ));
    registry.process_one_completion().unwrap();
    registry
        .backend_mut()
        .reject_next(LoadStage::Installing, FailureReason::InstallationRejected);
    registry.schedule_one(60).unwrap();
    registry.process_one_completion().unwrap();
    assert!(registry.operation(operation).is_none());
    assert_eq!(registry.resources().in_use(ResourceKind::ExpertFrame), 0);
}

#[test]
fn reserve_failure_rolls_back_all_hard_claims() {
    let mut backend = FakeBackend::manual();
    backend.fail_next_reserve(FailureReason::DeviceUnavailable);
    let mut registry = LoadRegistry::with_testing_resources(backend).unwrap();
    let report = registry
        .attach_waiter(waiter(1), [request(key(1, 1))], 1)
        .unwrap();
    assert_eq!(report.created.len(), 1);
    assert!(registry.schedule_one(2).unwrap());
    assert!(matches!(
        registry.pop_failed().unwrap().failure,
        ContinuationFailure::Failed(FailureReason::DeviceUnavailable)
    ));
    assert_eq!(registry.resources().active_grants(), 0);
    assert_eq!(registry.active_operations(), 0);
}

#[test]
fn retry_after_failure_gets_new_operation_identity() {
    let (mut registry, load_key, first) = manual_at_read();
    registry.enqueue_completion(event(
        first,
        load_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Failed(FailureReason::StorageUnavailable),
        0,
        30,
    ));
    registry.process_one_completion().unwrap();
    registry.pop_failed();
    let second = attach_one(&mut registry, waiter(2), load_key);
    assert_ne!(first, second);
}

#[test]
fn duplicate_old_completion_cannot_affect_retry() {
    let (mut registry, load_key, first) = manual_at_read();
    let failure = event(
        first,
        load_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Failed(FailureReason::StorageUnavailable),
        0,
        30,
    );
    registry.enqueue_completion(failure.clone());
    registry.process_one_completion().unwrap();
    let second = attach_one(&mut registry, waiter(2), load_key);
    registry.schedule_one(39).unwrap();
    registry.schedule_one(40).unwrap();
    registry.enqueue_completion(failure);
    assert!(matches!(
        registry.process_one_completion().unwrap(),
        CompletionDisposition::Rejected(CompletionRejectionReason::RetiredOperation)
    ));
    assert_eq!(
        registry.operation(second).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
}

#[test]
fn fake_backend_script_can_inject_wrong_generation() {
    let mut backend = FakeBackend::new();
    let mut spec = FakeCompletionSpec::success();
    spec.generation = Some(CompletionGeneration::new(
        SourceGeneration::new(1),
        DestinationGeneration::new(77),
    ));
    backend.script_next(LoadStage::ReadSubmitted, spec);
    let mut registry = LoadRegistry::with_testing_resources(backend).unwrap();
    let operation = attach_one(&mut registry, waiter(1), key(1, 1));
    registry.schedule_one(19).unwrap();
    registry.schedule_one(20).unwrap();
    registry.collect_backend_completions(1);
    registry.process_one_completion().unwrap();
    assert_eq!(
        registry.operation(operation).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
}

#[test]
fn registry_rejects_new_attach_after_shutdown_begins() {
    let mut registry = auto_registry();
    registry.begin_shutdown(1).unwrap();
    assert!(matches!(
        registry.attach_waiter(waiter(1), [request(key(1, 1))], 2),
        Err(RegistryError::RegistryShuttingDown)
    ));
}
