use std::num::NonZeroU64;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::io_protocol::{
    CancellationReason, CompletionEvent, CompletionGeneration, CompletionOutcome,
    CompletionTimestamp, ContinuationId, DestinationGeneration, DestinationSlotId, FailureReason,
    FenceId, LoadStage, MaterializationKey, OperationId, RegistrationId, SlabId, StaleReason,
};
use ferrule_common::materialization_io::{
    MaterializationResourceLimits, MaterializationResourcePlan, MaterializationResourceRequirements,
};
use ferrule_model::{MaterializationPreparation, MaterializationPurpose, MaterializationResolver};

use ferrule_runtime::io::testing::*;
use ferrule_runtime::io::{
    CompletionDisposition, ContinuationFailure, FairQueueConfig, LoadRegistry,
    RuntimeMaterializationProvider, RuntimeMaterializationResolver, SharedMaterializationProvider,
};
use ferrule_runtime::scheduling::{
    ExecutionPhase, PhysicalResourceBroker, PhysicalResourceLimit, ResourceDemand, ResourceKind,
};
#[test]
fn physical_bridge_reservation_keeps_exact_key() {
    let (mut backend, _, key) = resolved_provider();
    let reservation = backend
        .reserve(OperationId::new(7), key, uniform_plan())
        .unwrap();
    assert_eq!(
        reservation.binding().generation,
        key.destination_generation()
    );
}

#[test]
fn physical_bridge_reservation_keeps_exact_binding() {
    let (mut backend, handle, key) = resolved_provider();
    let reservation = backend
        .reserve(OperationId::new(7), key, uniform_plan())
        .unwrap();
    assert_eq!(reservation.binding(), handle.binding(key));
}

#[test]
fn physical_bridge_reservation_keeps_exact_slab_descriptor() {
    let (mut backend, _, key) = resolved_provider();
    let operation = OperationId::new(7);
    let reservation = backend.reserve(operation, key, uniform_plan()).unwrap();
    let descriptor = reservation.slabs()[0].descriptor();
    assert_eq!(descriptor.operation(), operation);
    assert_eq!(descriptor.slab(), SlabId::new(7 + SLAB_ID_OFFSET));
    assert_eq!(
        descriptor.registration(),
        RegistrationId::new(7 + REGISTRATION_ID_OFFSET)
    );
    assert_eq!(descriptor.address().get(), 0x10000);
}

#[test]
fn physical_bridge_reservation_keeps_exact_upload_fence() {
    let (mut backend, _, key) = resolved_provider();
    let operation = OperationId::new(7);
    let reservation = backend.reserve(operation, key, uniform_plan()).unwrap();
    assert_eq!(reservation.upload_fence().operation, operation);
    assert_eq!(
        reservation.upload_fence().fence,
        FenceId::new(7 + FENCE_ID_OFFSET)
    );
}

#[test]
fn physical_bridge_rejects_reservation_for_different_key() {
    let (mut backend, handle, canonical_key) = resolved_provider();
    handle.override_reservation_key(key(2));
    assert!(matches!(
        backend.reserve(OperationId::new(7), canonical_key, uniform_plan()),
        Err(FailureReason::ContractViolation { message: _ })
    ));
}

#[test]
fn physical_bridge_rejects_reservation_for_different_operation() {
    let (mut backend, handle, key) = resolved_provider();
    handle.override_reservation_operation(OperationId::new(99));
    assert!(matches!(
        backend.reserve(OperationId::new(7), key, uniform_plan()),
        Err(FailureReason::ContractViolation { message: _ })
    ));
}

#[test]
fn physical_bridge_rejects_binding_changed_after_resolve() {
    let (mut backend, handle, key) = resolved_provider();
    handle.override_reservation_slot(DestinationSlotId::new(SLOT + 1));
    assert!(matches!(
        backend.reserve(OperationId::new(7), key, uniform_plan()),
        Err(FailureReason::ContractViolation { message: _ })
    ));
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::Reserved, _)
        )),
        1
    );
}

#[test]
fn physical_bridge_forwards_materialization_plan() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let backend = SharedMaterializationProvider::new(Box::new(physical));
    backend
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    assert_eq!(
        backend.materialization_plan(key(1)).unwrap(),
        uniform_plan()
    );
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::MaterializationPlan(_)
        )),
        1
    );
}

#[test]
fn physical_bridge_forwards_read_upload_install_commands() {
    let (mut registry, handle) = registry(true);
    attach(&mut registry, waiter(1, 1), key(1), 1);
    registry.drive(100, 32).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        1
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitUpload(..))),
        1
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::PollInstall(..))),
        1
    );
}

#[test]
fn foreground_drains_submitted_warmup_without_reserving_another() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparations = [
        provider
            .prepare(request(1), MaterializationPurpose::Prefetch)
            .unwrap(),
        provider
            .prepare(request(2), MaterializationPurpose::Prefetch)
            .unwrap(),
    ];
    let keys = preparations.map(|preparation| preparation.key());
    let fairness = FairQueueConfig {
        model_warmup_quantum: BYTES,
        ..FairQueueConfig::default()
    };
    let mut registry =
        LoadRegistry::new(provider, bounded_physical_resources(1, 1, 1, 1), fairness).unwrap();
    let report = registry
        .prefetch(
            ferrule_runtime::io::PrefetchOwner::ModelWarmup,
            preparations.map(|preparation| {
                stage_request(preparation, uniform_plan(), ResourceDemand::ModelWarmup)
            }),
            1,
        )
        .unwrap();

    registry.drive(2, 16).unwrap();

    let submitted = report
        .created
        .iter()
        .copied()
        .find(|operation| {
            registry.operation(*operation).unwrap().stage() == LoadStage::ReadSubmitted
        })
        .expect("one warmup operation must enter the physical pipeline");
    let queued = report
        .created
        .iter()
        .copied()
        .find(|operation| registry.operation(*operation).unwrap().stage() == LoadStage::Reserved)
        .expect("the next warmup operation must remain unreserved");
    let submitted_key = registry.operation(submitted).unwrap().key();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Reserve(..))),
        1
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        1
    );

    handle.push_outcome(
        submitted,
        submitted_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Succeeded,
    );
    handle.script_outcome(LoadStage::UploadSubmitted, CompletionOutcome::Succeeded);
    handle.script_outcome(LoadStage::Installing, CompletionOutcome::Succeeded);
    registry.drive_foreground(3, 32).unwrap();

    assert!(registry.residency_binding(submitted_key).is_some());
    assert_eq!(
        registry.operation(queued).unwrap().stage(),
        LoadStage::Reserved
    );
    assert_eq!(
        keys.iter()
            .filter(|key| registry.residency_binding(**key).is_some())
            .count(),
        1
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Reserve(..))),
        1,
        "foreground drive must not reserve the next warmup operation"
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitUpload(..))),
        1
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::PollInstall(..))),
        1
    );
}

#[test]
fn physical_bridge_completion_enters_registry_unchanged() {
    let (mut registry, handle, key, operation) = manual_at_read();
    let mut completion = CompletionEvent::new(
        operation,
        key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Succeeded,
        BYTES,
        CompletionGeneration::for_key(key),
        CompletionTimestamp::from_nanos(88),
    );
    completion.generation.destination = DestinationGeneration::new(999);
    handle.push_completion(completion.clone());
    assert!(matches!(
        apply_physical(&mut registry, 4),
        CompletionDisposition::Rejected(_)
    ));
    assert_eq!(registry.rejected_completions()[0].event, completion);
}

#[test]
fn resolver_uses_provider_generation_without_logical_cache() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let mut resolver = RuntimeMaterializationResolver::new(handle.placement(), Some(provider));
    let first = resolver.resolve(request(1)).unwrap();
    let second = resolver.resolve(request(1)).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.destination_generation(),
        DestinationGeneration::new(DESTINATION_GENERATION)
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Prepare(_))),
        2
    );
    assert_eq!(resolver.stats().resolves, 2);
}

#[test]
fn registry_adopts_resident_preparation_without_a_read() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let load = stage_request(
        preparation,
        uniform_plan(),
        ResourceDemand::required(ExecutionPhase::Prefill),
    );
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    let report = registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [load],
            1,
        )
        .unwrap();
    assert_eq!(report.already_resident, 1);
    assert!(report.created.is_empty());
    assert_eq!(registry.residency_binding(key), Some(handle.binding(key)));
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        0
    );
    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 2)
        .unwrap();
    registry.shutdown(3, 0).unwrap();
}

#[test]
fn registry_releases_provider_lease_after_last_continuation() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let first = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = first.key();
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                first,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();
    let second = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    registry
        .attach_waiter(
            waiter(2, 2),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                second,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            2,
        )
        .unwrap();

    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 3)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );
    registry
        .detach_continuation(ContinuationId::new(2), CancellationReason::Superseded, 4)
        .unwrap();
    registry.drive(5, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(5, 0).unwrap();
}

fn consume_resident_resume(
    registry: &mut LoadRegistry<SharedMaterializationProvider>,
    continuation: ContinuationId,
    key: MaterializationKey,
    disposition: ferrule_runtime::io::ResumeDisposition,
    now_ns: u64,
) {
    assert_eq!(registry.pop_ready(now_ns).unwrap(), Some(continuation));
    let dependencies =
        ferrule_common::DependencySet::new([ferrule_common::LogicalDependency::resource_resident(
            key,
        )
        .unwrap()])
        .unwrap();
    let mut resume = registry
        .prepare_resume(continuation, &dependencies)
        .unwrap();
    let leases = resume.take().unwrap();
    assert_eq!(leases.len(), 1);
    registry
        .finish_resume(&mut resume, disposition, now_ns + 1, now_ns + 2)
        .unwrap();
}

#[test]
fn still_active_stage_keeps_provider_lease_until_quiescent_detach() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();

    consume_resident_resume(
        &mut registry,
        ContinuationId::new(1),
        key,
        ferrule_runtime::io::ResumeDisposition::StillActive,
        2,
    );
    registry.drive(5, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );
    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 6)
        .unwrap();
    registry.drive(7, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(8, 0).unwrap();
}

#[test]
fn transaction_custody_survives_resume_and_releases_on_commit_or_rollback() {
    for (seed, outcome, terminal_ns) in [
        (
            1,
            ferrule_runtime::io::TransactionCustodyOutcome::Committed {
                started_ns: 10,
                finished_ns: 11,
            },
            11,
        ),
        (
            2,
            ferrule_runtime::io::TransactionCustodyOutcome::RolledBack,
            6,
        ),
    ] {
        let (physical, handle) = MockPhysicalProvider::manual();
        handle.set_resident(true);
        let provider = SharedMaterializationProvider::new(Box::new(physical));
        let preparation = provider
            .prepare(request(seed), MaterializationPurpose::Execution)
            .unwrap();
        let key = preparation.key();
        let mut registry =
            LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
        registry
            .attach_waiter(
                waiter(seed as u64, seed as u64),
                ResourceDemand::required(ExecutionPhase::Prefill),
                [retained_request(
                    preparation,
                    uniform_plan(),
                    ResourceDemand::required(ExecutionPhase::Prefill),
                    ferrule_model::ResourceRetention::ThroughTransaction,
                )],
                1,
            )
            .unwrap();
        consume_resident_resume(
            &mut registry,
            ContinuationId::new(seed as u64),
            key,
            ferrule_runtime::io::ResumeDisposition::Consumed,
            2,
        );
        registry.drive(5, 1).unwrap();
        assert_eq!(
            handle.command_count(|command| matches!(
                command,
                MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
            )),
            0
        );

        let transaction = ExecutionTransactionId::new(seed as u64).unwrap();
        registry
            .finish_transaction_custody(transaction, outcome, terminal_ns)
            .unwrap();
        assert!(matches!(
            registry.finish_transaction_custody(transaction, outcome, terminal_ns),
            Err(ferrule_runtime::io::RegistryError::TransactionCustodyAlreadyFinished { transaction: candidate })
                if candidate == transaction
        ));
        registry.drive(6, 1).unwrap();
        assert_eq!(
            handle.command_count(|command| matches!(
                command,
                MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
            )),
            1
        );
        registry.shutdown(7, 0).unwrap();
    }
}

#[test]
fn persistent_custody_survives_transaction_terminal_until_explicit_retirement() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [retained_request(
                preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
                ferrule_model::ResourceRetention::Persistent,
            )],
            1,
        )
        .unwrap();
    consume_resident_resume(
        &mut registry,
        ContinuationId::new(1),
        key,
        ferrule_runtime::io::ResumeDisposition::Consumed,
        2,
    );
    registry
        .finish_transaction_custody(
            ExecutionTransactionId::new(1).unwrap(),
            ferrule_runtime::io::TransactionCustodyOutcome::Committed {
                started_ns: 5,
                finished_ns: 6,
            },
            6,
        )
        .unwrap();
    registry.drive(7, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );

    registry.retire_persistent_custody(key).unwrap();
    assert!(matches!(
        registry.retire_persistent_custody(key),
        Err(ferrule_runtime::io::RegistryError::PersistentCustodyNotFound { key: candidate }) if *candidate == key
    ));
    registry.drive(8, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(9, 0).unwrap();
}

#[test]
fn shared_key_releases_only_after_last_transaction_owner() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let first = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = first.key();
    let second = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    for (preparation, id) in [(first, 1), (second, 2)] {
        registry
            .attach_waiter(
                waiter(id, id),
                ResourceDemand::required(ExecutionPhase::Prefill),
                [retained_request(
                    preparation,
                    uniform_plan(),
                    ResourceDemand::required(ExecutionPhase::Prefill),
                    ferrule_model::ResourceRetention::ThroughTransaction,
                )],
                id,
            )
            .unwrap();
        consume_resident_resume(
            &mut registry,
            ContinuationId::new(id),
            key,
            ferrule_runtime::io::ResumeDisposition::Consumed,
            id + 2,
        );
    }
    registry
        .finish_transaction_custody(
            ExecutionTransactionId::new(1).unwrap(),
            ferrule_runtime::io::TransactionCustodyOutcome::Committed {
                started_ns: 10,
                finished_ns: 11,
            },
            11,
        )
        .unwrap();
    registry.drive(12, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );
    registry
        .finish_transaction_custody(
            ExecutionTransactionId::new(2).unwrap(),
            ferrule_runtime::io::TransactionCustodyOutcome::RolledBack,
            13,
        )
        .unwrap();
    registry.drive(13, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(14, 0).unwrap();
}

#[test]
fn new_attach_cancels_pending_provider_lease_release() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let first = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = first.key();
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                first,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();
    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 2)
        .unwrap();

    let second = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    registry
        .attach_waiter(
            waiter(2, 2),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                second,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            3,
        )
        .unwrap();
    registry.drive(4, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );
    registry
        .detach_continuation(ContinuationId::new(2), CancellationReason::Superseded, 5)
        .unwrap();
    registry.drive(6, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(7, 0).unwrap();
}

#[test]
fn registry_retries_failed_provider_lease_release_without_replay() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(9, 9),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();
    handle.fail_next_release(FailureReason::DeviceUnavailable);
    registry
        .detach_continuation(ContinuationId::new(9), CancellationReason::Superseded, 2)
        .unwrap();
    assert!(registry.drive(3, 1).is_err());
    assert_eq!(registry.residency_binding(key), Some(handle.binding(key)));
    registry.drive(4, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        2
    );
    registry.shutdown(4, 0).unwrap();
}

#[test]
fn queued_install_success_after_last_detach_releases_lease_once() {
    let (mut registry, handle) = registry(false);
    let key = key(1);
    let operation = attach(&mut registry, waiter(1, 1), key, 1);
    registry.schedule_one(2).unwrap();
    handle.script_outcome(LoadStage::ReadSubmitted, CompletionOutcome::Succeeded);
    registry.schedule_one(3).unwrap();
    apply_physical(&mut registry, 1);
    handle.script_outcome(LoadStage::UploadSubmitted, CompletionOutcome::Succeeded);
    registry.schedule_one(4).unwrap();
    apply_physical(&mut registry, 1);
    handle.script_outcome(LoadStage::Installing, CompletionOutcome::Succeeded);
    registry.schedule_one(5).unwrap();
    assert_eq!(registry.collect_provider_completions(1), 1);

    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 6)
        .unwrap();
    registry.process_one_completion_at(7).unwrap();
    assert!(registry.operation(operation).is_none());
    assert_eq!(registry.residency_binding(key), Some(handle.binding(key)));
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );
    registry.drive(8, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.drive(9, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(10, 0).unwrap();
}

#[test]
fn replacement_commit_swaps_exact_slot_generation_and_bytes() {
    let (physical, handle) = MockPhysicalProvider::automatic();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let old_preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let old_key = old_preparation.key();
    let slot = old_preparation.binding().slot;
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                old_preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();
    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 2)
        .unwrap();
    registry.drive(3, 1).unwrap();

    handle.set_resident(false);
    handle.configure_next_preparation(DESTINATION_GENERATION + 1, slot, Some(old_key));
    let new_preparation = provider
        .prepare(request(2), MaterializationPurpose::Execution)
        .unwrap();
    let new_key = new_preparation.key();
    registry
        .attach_waiter(
            waiter(2, 2),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                new_preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            4,
        )
        .unwrap();
    registry.drive(100, 32).unwrap();

    assert!(registry.residency_binding(old_key).is_none());
    assert_eq!(
        registry.residency_binding(new_key),
        Some(handle.binding(new_key))
    );
    assert_eq!(registry.resident_entries(), 1);
    assert_eq!(
        registry.resources().in_use(ResourceKind::ResidentBytes),
        BYTES
    );
    assert_eq!(
        registry.pop_ready(101).unwrap(),
        Some(ContinuationId::new(2))
    );
    registry
        .detach_continuation(ContinuationId::new(2), CancellationReason::Superseded, 102)
        .unwrap();
    registry.shutdown(103, 0).unwrap();
}

#[test]
fn replacement_waits_for_old_logical_owner_without_losing_operation() {
    let (physical, handle) = MockPhysicalProvider::automatic();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let old_preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let old_key = old_preparation.key();
    let slot = old_preparation.binding().slot;
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                old_preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();

    handle.set_resident(false);
    handle.configure_next_preparation(DESTINATION_GENERATION + 1, slot, Some(old_key));
    let new_preparation = provider
        .prepare(request(2), MaterializationPurpose::Execution)
        .unwrap();
    let new_key = new_preparation.key();
    let operation = registry
        .attach_waiter(
            waiter(2, 2),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                new_preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            2,
        )
        .unwrap()
        .created[0];
    registry.drive(100, 32).unwrap();

    assert_eq!(
        registry.residency_binding(old_key),
        Some(handle.binding(old_key))
    );
    assert!(registry.residency_binding(new_key).is_none());
    assert_eq!(
        registry
            .operation(operation)
            .map(|operation| operation.stage()),
        Some(LoadStage::Resident)
    );
    assert!(registry.retirement(operation).is_none());
    assert_eq!(
        registry.pop_ready(101).unwrap(),
        Some(ContinuationId::new(1))
    );

    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 102)
        .unwrap();
    registry.drive(103, 8).unwrap();
    assert!(registry.residency_binding(old_key).is_none());
    assert_eq!(
        registry.residency_binding(new_key),
        Some(handle.binding(new_key))
    );
    assert!(registry.operation(operation).is_none());
    assert!(registry.retirement(operation).is_some());
    registry
        .detach_continuation(ContinuationId::new(2), CancellationReason::Superseded, 104)
        .unwrap();
    registry.shutdown(105, 0).unwrap();
}

#[test]
fn shutdown_reclaims_replacement_published_during_fixed_point() {
    let (physical, handle) = MockPhysicalProvider::automatic();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let old_preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let old_key = old_preparation.key();
    let slot = old_preparation.binding().slot;
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                old_preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();

    handle.set_resident(false);
    handle.configure_next_preparation(DESTINATION_GENERATION + 1, slot, Some(old_key));
    let new_preparation = provider
        .prepare(request(2), MaterializationPurpose::Execution)
        .unwrap();
    let new_key = new_preparation.key();
    let operation = registry
        .attach_waiter(
            waiter(2, 2),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                new_preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            2,
        )
        .unwrap()
        .created[0];
    registry.drive(100, 32).unwrap();
    assert_eq!(
        registry.operation(operation).map(|active| active.stage()),
        Some(LoadStage::Resident)
    );
    assert!(registry.residency_binding(old_key).is_some());
    assert!(registry.residency_binding(new_key).is_none());

    let report = registry.shutdown(101, 0).unwrap();

    assert!(report.drained);
    assert_eq!(report.active_grants, 0);
    assert_eq!(registry.resident_entries(), 0);
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn replacement_cannot_evict_key_from_another_slot() {
    let (physical, handle) = MockPhysicalProvider::automatic();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let old_preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let old_key = old_preparation.key();
    let wrong_slot = DestinationSlotId::new(old_preparation.binding().slot.get() + 1);
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                old_preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();

    handle.set_resident(false);
    handle.configure_next_preparation(DESTINATION_GENERATION + 1, wrong_slot, Some(old_key));
    let new_preparation = provider
        .prepare(request(2), MaterializationPurpose::Execution)
        .unwrap();
    let new_key = new_preparation.key();
    registry
        .attach_waiter(
            waiter(2, 2),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                new_preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            2,
        )
        .unwrap();
    assert!(matches!(
        registry.drive(100, 32),
        Err(ferrule_runtime::io::RegistryError::PublishedResidencyConflict { key: _ })
    ));
    assert_eq!(
        registry.residency_binding(old_key),
        Some(handle.binding(old_key))
    );
    assert!(registry.residency_binding(new_key).is_none());
    assert_eq!(registry.resident_entries(), 1);
}

#[test]
fn pre_reserve_cancellation_discards_provider_preparation() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(11, 11),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();
    registry
        .detach_continuation(ContinuationId::new(11), CancellationReason::Superseded, 2)
        .unwrap();
    assert_eq!(registry.active_operations(), 0);
    assert!(registry.provider().preparation(key).is_err());
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::DiscardPreparation(candidate) if *candidate == key
        )),
        1
    );
}

#[test]
fn resolver_surfaces_provider_prepare_failure() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.fail_next_resolve(FailureReason::StorageUnavailable);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let mut resolver = RuntimeMaterializationResolver::new(handle.placement(), Some(provider));
    assert!(resolver.resolve(request(1)).is_err());
}

#[test]
fn physical_bridge_registry_surfaces_reserve_failure_without_credit_leak() {
    let (mut registry, handle) = registry(false);
    handle.fail_next_reserve(FailureReason::DeviceUnavailable);
    let report = registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [load_request(
                &registry,
                key(1),
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();
    assert_eq!(report.created.len(), 1);
    assert!(registry.schedule_one(2).unwrap());
    assert!(matches!(
        registry.pop_failed().unwrap().failure,
        ContinuationFailure::Failed(FailureReason::DeviceUnavailable)
    ));
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn physical_bridge_execution_promotes_prefetch_exactly_once_and_can_release_back() {
    let (mut registry, handle, key) = prefetch_registry(true);
    let binding = registry.provider().preparation(key).unwrap().binding();
    let prefetch = registry
        .prefetch(
            ferrule_runtime::io::PrefetchOwner::external(NonZeroU64::new(1).unwrap()),
            [stage_request(
                registry.provider().preparation(key).unwrap(),
                uniform_plan(),
                ResourceDemand::ModelWarmup,
            )],
            1,
        )
        .unwrap();
    let operation = prefetch.created[0];
    let joined = registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                registry.provider().preparation(key).unwrap(),
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            2,
        )
        .unwrap();

    assert_eq!(joined.joined, [operation]);
    assert_eq!(registry.operation(operation).unwrap().key(), key);
    assert_eq!(
        registry.provider().preparation(key).unwrap().binding(),
        binding
    );
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::PromoteToExecution(candidate) if *candidate == key
        )),
        1
    );

    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 3)
        .unwrap();
    assert!(registry.operation_has_prefetch_owner(operation));
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    assert_eq!(
        registry.operation(operation).unwrap().demand(),
        ResourceDemand::ModelWarmup
    );
    assert_eq!(
        registry.provider().preparation(key).unwrap().binding(),
        binding
    );
}

#[test]
fn failed_multi_key_promotion_restores_prefetch_and_reclaims_new_work() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let mut preparations = [
        provider
            .prepare(request(1), MaterializationPurpose::Prefetch)
            .unwrap(),
        provider
            .prepare(request(2), MaterializationPurpose::Prefetch)
            .unwrap(),
    ];
    preparations.sort_unstable_by_key(|preparation| preparation.key());
    let prefetch_key = preparations[0].key();
    let failing_key = preparations[1].key();
    let owner = ferrule_runtime::io::PrefetchOwner::external(NonZeroU64::new(41).unwrap());
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    let prefetched = registry
        .prefetch(
            owner,
            [stage_request(
                preparations[0],
                uniform_plan(),
                ResourceDemand::ModelWarmup,
            )],
            1,
        )
        .unwrap();
    let prefetch_operation = prefetched.created[0];
    let baseline_grants = registry.resources().active_grants();
    handle.fail_next_promotion(failing_key, FailureReason::DeviceUnavailable);

    let execution_waiter = waiter(51, 51);
    let error = registry
        .attach_waiter(
            execution_waiter,
            ResourceDemand::required(ExecutionPhase::Prefill),
            preparations.map(|preparation| {
                stage_request(
                    preparation,
                    uniform_plan(),
                    ResourceDemand::required(ExecutionPhase::Prefill),
                )
            }),
            2,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        ferrule_runtime::io::RegistryError::Provider {
            source: FailureReason::DeviceUnavailable
        }
    ));
    assert_eq!(registry.active_operations(), 1);
    assert_eq!(registry.active_prefetches(), 1);
    assert_eq!(
        registry.operation_for_key(prefetch_key),
        Some(prefetch_operation)
    );
    assert_eq!(registry.operation_for_key(failing_key), None);
    assert!(registry.operation_has_prefetch_owner(prefetch_operation));
    assert_eq!(
        registry.operation(prefetch_operation).unwrap().demand(),
        ResourceDemand::ModelWarmup
    );
    assert_eq!(registry.resources().active_grants(), baseline_grants);
    assert_eq!(registry.waiters().active_waiters().count(), 0);
    assert!(registry.provider().preparation(failing_key).is_err());
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == prefetch_key
        )),
        1
    );
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::DiscardPreparation(candidate) if *candidate == failing_key
        )),
        1
    );

    let retry = registry
        .attach_waiter(
            execution_waiter,
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                registry.provider().preparation(prefetch_key).unwrap(),
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            3,
        )
        .unwrap();
    assert_eq!(retry.joined, [prefetch_operation]);
    registry
        .detach_waiter(execution_waiter, CancellationReason::ExternalRequest, 4)
        .unwrap();
    assert!(registry.operation_has_prefetch_owner(prefetch_operation));
}

#[test]
fn physical_bridge_single_flight_issues_one_physical_read() {
    let (mut registry, handle) = registry(true);
    let key = key(1);
    let operation = attach(&mut registry, waiter(1, 1), key, 1);
    let joined_request = load_request(
        &registry,
        key,
        uniform_plan(),
        ResourceDemand::required(ExecutionPhase::Prefill),
    );
    let joined = registry
        .attach_waiter(
            waiter(2, 2),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [joined_request],
            2,
        )
        .unwrap();
    assert_eq!(joined.joined, vec![operation]);
    registry.drive(100, 32).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        1
    );
}

#[test]
fn physical_bridge_reverse_completion_wakes_only_target() {
    let (mut registry, handle) = registry(false);
    let first_key = key(1);
    let second_key = key(2);
    let first = attach(&mut registry, waiter(1, 11), first_key, 1);
    let second = attach(&mut registry, waiter(2, 22), second_key, 2);
    for now in 3..7 {
        assert!(registry.schedule_one(now).unwrap());
    }
    handle.push_outcome(
        second,
        second_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    assert!(registry.schedule_one(5).unwrap());
    handle.push_outcome(
        second,
        second_key,
        LoadStage::UploadSubmitted,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    assert!(registry.schedule_one(6).unwrap());
    handle.push_outcome(
        second,
        second_key,
        LoadStage::Installing,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    assert_eq!(
        registry.pop_ready(7).unwrap(),
        Some(ContinuationId::new(22))
    );
    assert_eq!(registry.pop_ready(7).unwrap(), None);
    assert_eq!(
        registry.operation(first).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
}

#[test]
fn physical_bridge_cancel_one_waiter_retains_shared_operation() {
    let (mut registry, handle) = registry(false);
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    let joined_request = load_request(
        &registry,
        key,
        uniform_plan(),
        ResourceDemand::required(ExecutionPhase::Prefill),
    );
    registry
        .attach_waiter(
            waiter(2, 2),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [joined_request],
            2,
        )
        .unwrap();
    registry.schedule_one(3).unwrap();
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 4)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Cancel(..))),
        0
    );
}

#[test]
fn physical_bridge_cancel_last_queued_skips_physical_cancel() {
    let (mut registry, handle) = registry(false);
    attach(&mut registry, waiter(1, 1), key(1), 1);
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 2)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Reserve(..))),
        0
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Cancel(..))),
        0
    );
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn physical_bridge_cancel_last_reserved_calls_physical_cancel() {
    let (mut registry, handle) = registry(false);
    attach(&mut registry, waiter(1, 1), key(1), 1);
    assert!(registry.schedule_one(2).unwrap());
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 3)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::Reserved, _)
        )),
        1
    );
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn physical_bridge_cancel_last_submitted_drains_completion() {
    let (mut registry, handle, _, operation) = manual_at_read();
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 3)
        .unwrap();
    assert!(registry.operation(operation).is_some());
    apply_physical(&mut registry, 4);
    assert!(registry.operation(operation).is_none());
    assert_eq!(registry.resources().active_grants(), 0);
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::ReadSubmitted, _)
        )),
        1
    );
}

#[test]
fn physical_bridge_cancel_host_ready_calls_physical_cancel() {
    let (mut registry, handle, key, operation) = manual_at_read();
    handle.push_outcome(
        operation,
        key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 4)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::HostReady, _)
        )),
        1
    );
    assert!(registry.operation(operation).is_none());
}

#[test]
fn physical_bridge_cancel_install_ready_calls_physical_cancel() {
    let (mut registry, handle, key, operation) = manual_at_read();
    handle.push_outcome(
        operation,
        key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    registry.schedule_one(4).unwrap();
    handle.push_outcome(
        operation,
        key,
        LoadStage::UploadSubmitted,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 5)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::Installing, _)
        )),
        1
    );
    assert!(registry.operation(operation).is_none());
}

#[test]
fn physical_bridge_read_failure_fails_all_waiters() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::ReadSubmitted,
        CompletionOutcome::Failed(FailureReason::StorageUnavailable),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    let joined_request = load_request(
        &registry,
        key,
        uniform_plan(),
        ResourceDemand::required(ExecutionPhase::Prefill),
    );
    registry
        .attach_waiter(
            waiter(2, 2),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [joined_request],
            2,
        )
        .unwrap();
    registry.drive(100, 16).unwrap();
    let failures = [
        registry.pop_failed().unwrap(),
        registry.pop_failed().unwrap(),
    ];
    assert!(failures.iter().all(|failure| matches!(
        failure.failure,
        ContinuationFailure::Failed(FailureReason::StorageUnavailable)
    )));
}

#[test]
fn physical_bridge_upload_failure_never_publishes() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::UploadSubmitted,
        CompletionOutcome::Failed(FailureReason::UploadRejected),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 16).unwrap();
    assert!(registry.residency_binding(key).is_none());
    assert!(matches!(
        registry.pop_failed().unwrap().failure,
        ContinuationFailure::Failed(FailureReason::UploadRejected)
    ));
}

#[test]
fn physical_bridge_install_failure_never_publishes() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::Installing,
        CompletionOutcome::Failed(FailureReason::InstallationRejected),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 16).unwrap();
    assert!(registry.residency_binding(key).is_none());
    assert!(registry.pop_failed().is_some());
}

#[test]
fn physical_bridge_read_stale_never_publishes() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::ReadSubmitted,
        CompletionOutcome::Stale(StaleReason::SourceIdentityChanged),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 16).unwrap();
    assert!(registry.residency_binding(key).is_none());
    assert!(matches!(
        registry.pop_failed().unwrap().failure,
        ContinuationFailure::Stale(StaleReason::SourceIdentityChanged)
    ));
}

#[test]
fn physical_bridge_upload_stale_never_publishes() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::UploadSubmitted,
        CompletionOutcome::Stale(StaleReason::DestinationReused),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 16).unwrap();
    assert!(registry.residency_binding(key).is_none());
}

#[test]
fn physical_bridge_install_stale_never_publishes() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::Installing,
        CompletionOutcome::Stale(StaleReason::SupersededOperation),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 16).unwrap();
    assert!(registry.residency_binding(key).is_none());
}

#[test]
fn registry_publication_uses_provider_binding_without_reconciliation() {
    let (physical, handle) = MockPhysicalProvider::automatic();
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                preparation,
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();
    registry.drive(100, 32).unwrap();
    assert_eq!(registry.residency_binding(key), Some(handle.binding(key)));
}

#[test]
fn physical_bridge_shutdown_cancels_and_drains_submitted_work() {
    let (mut registry, handle, _, _) = manual_at_read();
    let report = registry.shutdown(10, 8).unwrap();
    assert!(report.drained);
    assert_eq!(report.active_grants, 0);
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::ReadSubmitted, _)
        )),
        1
    );
}

#[test]
fn forty_dependencies_attach_and_complete_with_qd_two() {
    const DEPENDENCIES: u8 = 40;
    const QD: u64 = 2;

    let (physical, handle) = MockPhysicalProvider::manual();
    let backend = SharedMaterializationProvider::new(Box::new(physical));
    for seed in 1..=DEPENDENCIES {
        backend
            .prepare(request(seed), MaterializationPurpose::Execution)
            .unwrap();
    }
    let mut resources = bounded_physical_resources(u64::from(DEPENDENCIES), QD, QD, QD);
    resources
        .reconfigure_limit(ResourceKind::LoadOperation, QD, 0)
        .unwrap();
    let mut registry = LoadRegistry::new(backend, resources, FairQueueConfig::default()).unwrap();
    let requests = (1..=DEPENDENCIES)
        .map(|seed| {
            load_request(
                &registry,
                key(seed),
                uniform_plan(),
                ResourceDemand::required(ExecutionPhase::Prefill),
            )
        })
        .collect::<Vec<_>>();

    let report = registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            requests,
            1,
        )
        .unwrap();
    assert_eq!(report.created.len(), usize::from(DEPENDENCIES));
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Reserve(..))),
        0
    );
    assert_eq!(registry.resources().in_use(ResourceKind::ReadSlot), 0);
    assert_eq!(registry.resources().in_use(ResourceKind::LoadOperation), 0);

    registry.drive(10, 128).unwrap();
    let submitted = report
        .created
        .iter()
        .copied()
        .filter(|operation| {
            registry.operation(*operation).unwrap().stage() == LoadStage::ReadSubmitted
        })
        .collect::<Vec<_>>();
    assert_eq!(submitted.len(), QD as usize);
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        QD as usize
    );

    for _ in QD..u64::from(DEPENDENCIES) {
        handle.script_outcome(LoadStage::ReadSubmitted, CompletionOutcome::Succeeded);
    }
    for _ in 0..u64::from(DEPENDENCIES) {
        handle.script_outcome(LoadStage::UploadSubmitted, CompletionOutcome::Succeeded);
        handle.script_outcome(LoadStage::Installing, CompletionOutcome::Succeeded);
    }
    for operation in submitted {
        handle.push_outcome(
            operation,
            registry.key_for_operation(operation).unwrap(),
            LoadStage::ReadSubmitted,
            CompletionOutcome::Succeeded,
        );
    }

    registry.drive(100, 2048).unwrap();
    assert_eq!(registry.stats().publications, u64::from(DEPENDENCIES));
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        usize::from(DEPENDENCIES)
    );
    let sqe = registry
        .resources()
        .snapshots()
        .find(|snapshot| snapshot.kind == ResourceKind::ReadSlot)
        .unwrap();
    assert_eq!(sqe.capacity, QD);
    assert_eq!(sqe.high_water, QD);
    assert_eq!(sqe.in_use, 0);
    let load_operations = registry
        .resources()
        .snapshots()
        .find(|snapshot| snapshot.kind == ResourceKind::LoadOperation)
        .unwrap();
    assert_eq!(load_operations.capacity, QD);
    assert_eq!(load_operations.high_water, QD);
    assert_eq!(load_operations.in_use, 0);

    let shutdown = registry.shutdown(200, 128).unwrap();
    assert!(shutdown.drained);
    assert_eq!(shutdown.active_grants, 0);
}

#[test]
fn physical_bridge_resource_high_water_uses_real_bytes() {
    let (mut registry, _) = registry(true);
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 32).unwrap();
    let high_water = |kind| {
        registry
            .resources()
            .snapshots()
            .find(|snapshot| snapshot.kind == kind)
            .unwrap()
            .high_water
    };
    assert_eq!(high_water(ResourceKind::PinnedHostBytes), BYTES);
    assert_eq!(high_water(ResourceKind::StorageReadBytes), BYTES);
    assert_eq!(high_water(ResourceKind::UploadBytes), BYTES);
    assert_eq!(high_water(ResourceKind::DeviceInstallBytes), BYTES);
    assert_eq!(high_water(ResourceKind::ResidentBytes), BYTES);
    assert_eq!(high_water(ResourceKind::ReadSlot), 1);
    assert_eq!(high_water(ResourceKind::InstallSlot), 1);
    assert_eq!(high_water(ResourceKind::LoadOperation), 1);
}

#[test]
fn physical_bridge_preserves_nonuniform_resource_plan() {
    let plan = MaterializationResourcePlan::new(
        MaterializationResourceRequirements {
            read_slots: 2,
            storage_read_bytes: 4096,
            pinned_host_bytes: 8192,
            upload_slots: 3,
            h2d_bytes: 12_288,
            install_slots: 4,
            device_install_bytes: 16_384,
        },
        20_480,
    )
    .unwrap();
    let limits = MaterializationResourceLimits {
        capacity: MaterializationResourceRequirements {
            device_install_bytes: plan.resident_bytes,
            ..plan.requirements
        },
        execution_reserve: MaterializationResourceRequirements::default(),
    }
    .validate()
    .unwrap();
    let (physical, handle) = MockPhysicalProvider::automatic();
    handle.set_plan_and_limits(plan, limits);
    let mut backend = SharedMaterializationProvider::new(Box::new(physical));
    let resolved = backend
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    assert!(matches!(resolved, MaterializationPreparation::Transfer(_)));
    let key = resolved.key();
    assert_eq!(backend.materialization_plan(key).unwrap(), plan);

    let direct_operation = OperationId::new(77);
    let reservation = backend.reserve(direct_operation, key, plan).unwrap();
    assert_eq!(
        reservation.slabs()[0].descriptor().len(),
        plan.requirements.pinned_host_bytes
    );
    backend
        .submit_read(direct_operation, key, &reservation, plan)
        .unwrap();
    backend
        .submit_upload(direct_operation, key, &reservation, plan)
        .unwrap();
    backend
        .poll_install(direct_operation, key, &reservation, plan)
        .unwrap();
    let completions = [
        backend.next_completion().unwrap(),
        backend.next_completion().unwrap(),
        backend.next_completion().unwrap(),
    ];
    assert_eq!(
        completions.map(|event| (event.stage, event.bytes)),
        [
            (
                LoadStage::ReadSubmitted,
                plan.requirements.storage_read_bytes
            ),
            (LoadStage::UploadSubmitted, plan.requirements.h2d_bytes),
            (
                LoadStage::Installing,
                plan.requirements.device_install_bytes
            ),
        ]
    );
    let commands = handle.commands();
    assert!(commands.contains(&MockPhysicalCommand::Reserve(direct_operation, key, plan,)));
    assert!(commands.contains(&MockPhysicalCommand::SubmitRead(
        direct_operation,
        key,
        plan,
    )));
    assert!(commands.contains(&MockPhysicalCommand::SubmitUpload(
        direct_operation,
        key,
        plan,
    )));
    assert!(commands.contains(&MockPhysicalCommand::PollInstall(
        direct_operation,
        key,
        plan,
    )));
    drop(reservation);

    let (registry_physical, registry_handle) = MockPhysicalProvider::automatic();
    registry_handle.set_plan_and_limits(plan, limits);
    let registry_backend = SharedMaterializationProvider::new(Box::new(registry_physical));
    let registry_resolved = registry_backend
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    assert!(matches!(
        registry_resolved,
        MaterializationPreparation::Transfer(_)
    ));
    let registry_key = registry_resolved.key();
    assert_eq!(registry_key, key);

    let resources = PhysicalResourceBroker::new(ResourceKind::ALL.map(|kind| {
        let capacity = match kind {
            ResourceKind::ReadSlot => plan.requirements.read_slots,
            ResourceKind::PinnedHostBytes => plan.requirements.pinned_host_bytes,
            ResourceKind::StorageReadBytes => plan.requirements.storage_read_bytes,
            ResourceKind::UploadSlot => plan.requirements.upload_slots,
            ResourceKind::UploadBytes => plan.requirements.h2d_bytes,
            ResourceKind::InstallSlot => plan.requirements.install_slots,
            ResourceKind::DeviceInstallBytes => plan.requirements.device_install_bytes,
            ResourceKind::ResidentBytes => plan.resident_bytes,
            ResourceKind::ResidencyLease | ResourceKind::LoadOperation => 1,
            ResourceKind::Arena
            | ResourceKind::KvPage
            | ResourceKind::Continuation
            | ResourceKind::Waiter
            | ResourceKind::ReadyCohort => 64,
        };
        PhysicalResourceLimit::new(kind, capacity, 0)
    }))
    .unwrap();
    let mut registry =
        LoadRegistry::new(registry_backend, resources, FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            ResourceDemand::required(ExecutionPhase::Prefill),
            [stage_request(
                registry_resolved,
                plan,
                ResourceDemand::required(ExecutionPhase::Prefill),
            )],
            1,
        )
        .unwrap();
    for now_ns in 100..=108 {
        registry.drive(now_ns, 32).unwrap();
    }
    assert!(registry.residency_binding(registry_key).is_some());

    let high_water = |kind| {
        registry
            .resources()
            .snapshots()
            .find(|snapshot| snapshot.kind == kind)
            .unwrap()
            .high_water
    };
    assert_eq!(
        high_water(ResourceKind::StorageReadBytes),
        plan.requirements.storage_read_bytes
    );
    assert_eq!(
        high_water(ResourceKind::PinnedHostBytes),
        plan.requirements.pinned_host_bytes
    );
    assert_eq!(
        high_water(ResourceKind::UploadBytes),
        plan.requirements.h2d_bytes
    );
    assert_eq!(
        high_water(ResourceKind::DeviceInstallBytes),
        plan.requirements.device_install_bytes
    );
    assert_eq!(high_water(ResourceKind::ResidentBytes), plan.resident_bytes);
    assert_eq!(
        high_water(ResourceKind::ReadSlot),
        plan.requirements.read_slots
    );
    assert_eq!(
        high_water(ResourceKind::UploadSlot),
        plan.requirements.upload_slots
    );
    assert_eq!(
        high_water(ResourceKind::InstallSlot),
        plan.requirements.install_slots
    );
}
