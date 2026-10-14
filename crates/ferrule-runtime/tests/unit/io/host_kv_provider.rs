#![expect(
    unsafe_code,
    reason = "Driver-free no-work physical attestations and adversarial routing tests; no device references exist"
)]
// Included as a private unit-test module; not a standalone integration target.
use super::*;
use crate::cache::{KvPageManager, KvReservationCommit};
use crate::io::host_kv_adapters::ReservedHostKvPlacement;
use crate::io::testing::{FakeMaterializationProvider, MockPhysicalProvider};
use ferrule_common::execution::{KvElementType, KvLayoutSchema, KvPlaneDescriptor, StateSlot};
use ferrule_common::host_kv::placement::{PlacementBudget, PlacementCredits};
use ferrule_common::host_kv::{HostKvPlacementState, HostKvStage};
#[derive(Debug)]
struct Recording {
    inner: FakeMaterializationProvider,
    calls: Arc<Mutex<Vec<HostKvPlan>>>,
    result: HostKvResult<()>,
    panic_after_record: bool,
}
impl RuntimeMaterializationProvider for Recording {
    fn submit_host_kv_plan(&mut self, plan: &HostKvPlan) -> HostKvResult<()> {
        self.calls.lock().unwrap().push(plan.clone());
        assert!(
            !self.panic_after_record,
            "injected provider unwind after retaining command"
        );
        self.result
    }
    fn preparation(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationPreparation, FailureReason> {
        self.inner.preparation(key)
    }

    fn promote_to_execution(
        &mut self,
        key: MaterializationKey,
    ) -> Result<ExecutionPromotion, FailureReason> {
        self.inner.promote_to_execution(key)
    }

    fn discard_preparation(&mut self, key: MaterializationKey) -> Result<(), FailureReason> {
        self.inner.discard_preparation(key)
    }

    fn release_execution_lease(&mut self, key: MaterializationKey) -> Result<(), FailureReason> {
        self.inner.release_execution_lease(key)
    }

    fn materialization_plan(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationResourcePlan, FailureReason> {
        self.inner.materialization_plan(key)
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        plan: MaterializationResourcePlan,
    ) -> Result<MaterializationOperationReservation, FailureReason> {
        self.inner.reserve(operation, key, plan)
    }

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        self.inner.submit_read(operation, key, reservation, plan)
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        self.inner.submit_upload(operation, key, reservation, plan)
    }

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        self.inner.poll_install(operation, key, reservation, plan)
    }

    fn cancel(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> Result<(), FailureReason> {
        self.inner.cancel(operation, key, stage, reason)
    }

    fn next_progress(&mut self) -> ProviderProgress {
        self.inner.next_progress()
    }
}
#[derive(Debug)]
struct Schema;
impl KvLayoutSchema for Schema {
    fn planes(&self) -> &[KvPlaneDescriptor] {
        static P: [KvPlaneDescriptor; 1] = [KvPlaneDescriptor::new("kv", 2, 1, KvElementType::F32)];
        &P
    }
    fn page_size(&self) -> usize {
        4
    }
    fn max_sequence_len(&self) -> usize {
        128
    }
}
fn setup() -> (KvPageManager, HostKvPlan) {
    let mut m = KvPageManager::new(Box::new(Schema), 8);
    let slot = StateSlot::new(0);
    m.alloc_sequence(slot, 7).unwrap();
    let r = m.reserve(slot, 7, 3).unwrap();
    let p = m
        .prepare_commit(vec![KvReservationCommit::new(r, 3)])
        .unwrap();
    let retired = m.publish_commit(p);
    m.confirm_page_retirement(retired).unwrap();
    let source = m.committed_kv_source(slot, 7, 0..3).unwrap();
    let plan = HostKvPlan {
        source,
        credits: PlacementCredits {
            pageable_host_bytes: 32,
            pinned_host_bytes: 32,
            staging_bytes: 32,
            staging_slots: 1,
            transfers: 1,
            consumers: 1,
        },
        fence: ferrule_common::host_kv::HostKvFence {
            owner: 2,
            operation: 5,
            generation: 3,
            stage: HostKvStage::HostCopy,
            stream: 17,
            event: 19,
        },
    };
    (m, plan)
}
fn recording(result: HostKvResult<()>) -> Recording {
    Recording {
        inner: FakeMaterializationProvider::new(),
        calls: Arc::default(),
        result,
        panic_after_record: false,
    }
}
#[test]
fn box_forwards_exact_host_plan() {
    let (_, plan) = setup();
    for result in [
        Ok(()),
        Err(HostKvError::ForeignOwner),
        Err(HostKvError::Unknown),
    ] {
        let mut direct = recording(result);
        assert_eq!(direct.submit_host_kv_plan(&plan), result);
        let calls = direct.calls.clone();
        let mut boxed: Box<dyn RuntimeMaterializationProvider> = Box::new(direct);
        assert_eq!(
            RuntimeMaterializationProvider::submit_host_kv_plan(&mut boxed, &plan),
            result
        );
        assert_eq!(*calls.lock().unwrap(), vec![plan.clone(), plan.clone()]);
    }
}
#[test]
fn shared_explicitly_rejects_without_checkpoint_commands() {
    let (_, plan) = setup();
    let (physical, handle) = MockPhysicalProvider::new(false);
    let mut shared = SharedMaterializationProvider::new(Box::new(physical));
    assert_eq!(
        shared.submit_host_kv_plan(&plan),
        Err(HostKvError::ProviderUnsupported)
    );
    assert!(handle.commands().is_empty());
}
#[test]
fn audited_no_reference_rejection_is_retryable_but_not_completion() {
    let (mut m, plan) = setup();
    let budget = PlacementBudget::new(plan.credits);
    let mut reserved =
        ReservedHostKvPlacement::reserve(&mut m, plan.source.clone(), plan.credits, budget.clone())
            .unwrap();
    let mut provider = recording(Err(HostKvError::ForeignOwner));
    // SAFETY: this test provider merely records metadata; it has no DMA,
    // callbacks or physical references, so its rejection is positively no-submit.
    assert_eq!(
        unsafe { reserved.submit_trusted(&m, &mut provider, 1, 0, 1) },
        Err(HostKvError::ForeignOwner)
    );
    assert_eq!(reserved.state(), Ok(HostKvPlacementState::Reserved));
    // A proven rejection retires this attempt, not its generation high-water
    // mark. Neither submission boundary may invoke the provider on replay.
    assert_eq!(
        reserved.submit(&m, &mut provider, 1, 0, 2),
        Err(HostKvError::FenceMismatch)
    );
    // SAFETY: same metadata-only provider; the stale attempt is rejected first.
    assert_eq!(
        unsafe { reserved.submit_trusted(&m, &mut provider, 1, 0, 2) },
        Err(HostKvError::FenceMismatch)
    );
    assert_eq!(provider.calls.lock().unwrap().len(), 1);
    assert_eq!(reserved.state(), Ok(HostKvPlacementState::Reserved));
    assert_eq!(budget.in_use(), plan.credits);
    // SAFETY: same no-reference provider, now with a fresh attempt identity.
    assert_eq!(
        unsafe { reserved.submit_trusted(&m, &mut provider, 2, 0, 2) },
        Err(HostKvError::ForeignOwner)
    );
    assert_eq!(provider.calls.lock().unwrap().len(), 2);
    assert_eq!(budget.in_use(), plan.credits);
    reserved.cancel(&mut m).unwrap();
    assert_eq!(budget.in_use(), PlacementCredits::default());
}
#[test]
fn accepted_and_unknown_commands_retain_custody_without_replay() {
    for result in [Ok(()), Err(HostKvError::Unknown)] {
        let (mut m, plan) = setup();
        let budget = PlacementBudget::new(plan.credits);
        let mut reserved =
            ReservedHostKvPlacement::reserve(&mut m, plan.source, plan.credits, budget.clone())
                .unwrap();
        let mut provider = recording(result);
        assert_eq!(reserved.submit(&m, &mut provider, 1, 0, 1), result);
        let expected = if result.is_ok() {
            HostKvPlacementState::HostCopyPending
        } else {
            HostKvPlacementState::Unknown
        };
        assert_eq!(reserved.state(), Ok(expected));
        assert!(reserved.submit(&m, &mut provider, 1, 0, 1).is_err());
        assert_eq!(provider.calls.lock().unwrap().len(), 1);
        assert!(reserved.cancel(&mut m).is_err());
        drop(reserved);
        assert_eq!(budget.in_use(), plan.credits);
        assert!(m.free_sequence_pages(StateSlot::new(0)).is_err());
    }
}

#[test]
fn safe_provider_error_never_certifies_no_references() {
    for error in [
        HostKvError::ProviderUnsupported,
        HostKvError::ForeignOwner,
        HostKvError::BudgetExceeded,
        HostKvError::FenceMismatch,
        HostKvError::Unknown,
    ] {
        let (mut m, plan) = setup();
        let budget = PlacementBudget::new(plan.credits);
        let mut reserved = ReservedHostKvPlacement::reserve(
            &mut m,
            plan.source.clone(),
            plan.credits,
            budget.clone(),
        )
        .unwrap();
        // This completely safe implementation records/retains the command then
        // claims any ordinary error. Its numeric error is not physical evidence.
        let mut provider = recording(Err(error));
        assert_eq!(reserved.submit(&m, &mut provider, 1, 0, 1), Err(error));
        assert_eq!(reserved.state(), Ok(HostKvPlacementState::Unknown));
        assert_eq!(provider.calls.lock().unwrap().len(), 1);
        assert!(reserved.cancel(&mut m).is_err());
        assert!(reserved.submit(&m, &mut provider, 2, 0, 2).is_err());
        assert_eq!(provider.calls.lock().unwrap().len(), 1);
        assert_eq!(budget.in_use(), plan.credits);
        drop(reserved);
        assert_eq!(budget.in_use(), plan.credits);
        assert!(
            m.reserve(plan.source.state_slot, plan.source.generation, 1)
                .is_err()
        );
    }
}
#[test]
fn provider_unwind_is_sticky_unknown_for_both_submission_boundaries() {
    for trusted in [false, true] {
        let (mut m, plan) = setup();
        let budget = PlacementBudget::new(plan.credits);
        let mut reserved = ReservedHostKvPlacement::reserve(
            &mut m,
            plan.source.clone(),
            plan.credits,
            budget.clone(),
        )
        .unwrap();
        let mut provider = recording(Err(HostKvError::ForeignOwner));
        provider.panic_after_record = true;
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if trusted {
                    // SAFETY: modeled provider has no physical resources; unwinding
                    // still must not be translated into a rejection certificate.
                    let _ = unsafe { reserved.submit_trusted(&m, &mut provider, 1, 0, 1) };
                } else {
                    let _ = reserved.submit(&m, &mut provider, 1, 0, 1);
                }
            }))
            .is_err()
        );
        assert_eq!(reserved.state(), Ok(HostKvPlacementState::Unknown));
        provider.panic_after_record = false;
        // SAFETY: this provider now rejects with no physical refs, but a NEW
        // known outcome is not evidence for the earlier interrupted command.
        assert!(unsafe { reserved.submit_trusted(&m, &mut provider, 2, 0, 2) }.is_err());
        assert_eq!(provider.calls.lock().unwrap().len(), 1);
        assert!(reserved.cancel(&mut m).is_err());
        drop(reserved);
        assert_eq!(budget.in_use(), plan.credits);
        assert!(
            m.reserve(plan.source.state_slot, plan.source.generation, 1)
                .is_err()
        );
    }
}
#[test]
fn audited_provider_unknown_is_not_rejection() {
    let (mut m, plan) = setup();
    let budget = PlacementBudget::new(plan.credits);
    let mut reserved =
        ReservedHostKvPlacement::reserve(&mut m, plan.source, plan.credits, budget.clone())
            .unwrap();
    let mut provider = recording(Err(HostKvError::Unknown));
    // SAFETY: no ordinary error is emitted; the modeled unknown remains charged.
    assert_eq!(
        unsafe { reserved.submit_trusted(&m, &mut provider, 1, 0, 1) },
        Err(HostKvError::Unknown)
    );
    assert_eq!(reserved.state(), Ok(HostKvPlacementState::Unknown));
    assert!(reserved.cancel(&mut m).is_err());
    assert_eq!(budget.in_use(), plan.credits);
}
