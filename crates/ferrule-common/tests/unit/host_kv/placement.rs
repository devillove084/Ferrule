#![expect(
    unsafe_code,
    reason = "Driver-free no-work physical attestations and adversarial routing tests; no device references exist"
)]
// SAFETY of the attestations in this private driver-free fixture: the modeled
// physical stages are synchronous/no-work; there are no device references or
// callbacks. Mismatched and replayed observations test rejection before mutation.
// Included as a private unit-test module; not a standalone integration target.
use super::*;
use crate::execution::{
    KvElementType, KvLayoutSchemaIdentity, KvPageId, KvPlaneDescriptor, StateSlot,
};
#[derive(Default)]
struct Authority {
    live: usize,
    fail_acquire: bool,
    fail_release: bool,
    drops: Rc<Cell<usize>>,
}
struct Lease(Rc<Cell<usize>>);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
impl CommittedKvReadAuthority for Authority {
    type Lease = Lease;
    fn acquire(&mut self, _: &CommittedKvSource) -> HostKvResult<Lease> {
        if self.fail_acquire {
            return Err(HostKvError::TransactionActive);
        }
        self.live += 1;
        Ok(Lease(self.drops.clone()))
    }
    fn validate(&self, _: &Lease) -> HostKvResult<()> {
        Ok(())
    }
    fn release(&mut self, _: &Lease) -> HostKvResult<()> {
        if self.fail_release {
            return Err(HostKvError::ReaderActive);
        }
        self.live -= 1;
        Ok(())
    }
}
fn source() -> CommittedKvSource {
    CommittedKvSource {
        owner: 1,
        state_slot: StateSlot::new(0),
        generation: 7,
        incarnation: 1,
        schema: KvLayoutSchemaIdentity::new(
            vec![KvPlaneDescriptor::new("kv", 2, 1, KvElementType::F32)],
            4,
            128,
        ),
        committed_tokens: 3,
        positions: 0..3,
        pages: vec![KvPageId(0)],
    }
}
fn credits() -> PlacementCredits {
    PlacementCredits::from_domains([32, 32, 32, 1, 1, 1])
}
fn fence(h: HostKvPlacementHandle, stage: HostKvStage) -> HostKvFence {
    HostKvFence {
        owner: h.owner(),
        operation: h.operation(),
        generation: 19,
        stage,
        stream: 2,
        event: 9,
    }
}
fn setup() -> (
    Authority,
    PlacementBudget,
    HostKvPlacementOwner<Lease>,
    HostKvPlacementHandle,
) {
    let mut a = Authority::default();
    let b = PlacementBudget::new(credits());
    let mut o = HostKvPlacementOwner::new(b.clone()).unwrap();
    let h = o.reserve(&mut a, &source(), credits()).unwrap();
    (a, b, o, h)
}
fn finish(o: &mut HostKvPlacementOwner<Lease>, a: &Authority, h: HostKvPlacementHandle) {
    for stage in [
        HostKvStage::HostCopy,
        HostKvStage::Upload,
        HostKvStage::Consumer,
    ] {
        let f = fence(h, stage);
        o.submit(a, h, f).unwrap();
        unsafe { o.observe_complete(h, HostKvCompletion { fence: f }) }.unwrap();
    }
}
#[test]
fn each_budget_domain_is_independent_and_failure_atomic() {
    for domain in 0..6 {
        let mut capacity = [100; 6];
        capacity[domain] = 1;
        let b = PlacementBudget::new(PlacementCredits::from_domains(capacity));
        let baseline = PlacementCredits::from_domains([1; 6]);
        b.acquire(baseline).unwrap();
        assert_eq!(b.acquire(baseline), Err(HostKvError::BudgetExceeded));
        assert_eq!(b.in_use(), baseline);
        b.release(baseline);
        assert_eq!(b.in_use(), PlacementCredits::default());
    }
}
#[test]
fn budget_overflow_retains_existing_charge() {
    for domain in 0..6 {
        let b = PlacementBudget::new(PlacementCredits::from_domains([u64::MAX; 6]));
        let mut first = [0; 6];
        first[domain] = u64::MAX;
        let first = PlacementCredits::from_domains(first);
        b.acquire(first).unwrap();
        assert_eq!(
            b.acquire(PlacementCredits::from_domains([1; 6])),
            Err(HostKvError::Overflow)
        );
        assert_eq!(b.in_use(), first);
    }
}
#[test]
fn authority_acquire_failure_rolls_back_all_domains() {
    let mut a = Authority {
        fail_acquire: true,
        ..Authority::default()
    };
    let b = PlacementBudget::new(credits());
    let mut o = HostKvPlacementOwner::new(b.clone()).unwrap();
    assert_eq!(
        o.reserve(&mut a, &source(), credits()),
        Err(HostKvError::TransactionActive)
    );
    assert_eq!(b.in_use(), PlacementCredits::default());
    a.fail_acquire = false;
    let h = o.reserve(&mut a, &source(), credits()).unwrap();
    o.cancel(&mut a, h).unwrap();
    assert_eq!(a.drops.get(), 1);
}
#[test]
fn ordered_fences_hold_credits_until_consumer_ack() {
    let (mut a, b, mut o, h) = setup();
    assert_eq!(
        o.submit(&a, h, fence(h, HostKvStage::Upload)),
        Err(HostKvError::InvalidTransition)
    );
    for stage in [
        HostKvStage::HostCopy,
        HostKvStage::Upload,
        HostKvStage::Consumer,
    ] {
        let f = fence(h, stage);
        o.submit(&a, h, f).unwrap();
        assert!(o.cancel(&mut a, h).is_err());
        assert!(o.release(&mut a, h).is_err());
        assert_eq!(b.in_use(), credits());
        unsafe { o.observe_complete(h, HostKvCompletion { fence: f }) }.unwrap();
        assert!(unsafe { o.observe_complete(h, HostKvCompletion { fence: f }) }.is_err());
    }
    o.release(&mut a, h).unwrap();
    assert_eq!(b.in_use(), PlacementCredits::default());
    assert_eq!(a.live, 0);
    assert_eq!(o.release(&mut a, h), Err(HostKvError::UnknownPlacement));
}
#[test]
fn unknown_retains_charge_even_after_exact_metadata_ack() {
    let stages = [
        HostKvStage::HostCopy,
        HostKvStage::Upload,
        HostKvStage::Consumer,
    ];
    for unknown_stage in 0..stages.len() {
        let (mut a, b, mut o, h) = setup();
        for &stage in &stages[..unknown_stage] {
            let f = fence(h, stage);
            o.submit(&a, h, f).unwrap();
            unsafe { o.observe_complete(h, HostKvCompletion { fence: f }) }.unwrap();
        }
        let f = fence(h, stages[unknown_stage]);
        o.submit(&a, h, f).unwrap();
        o.mark_unknown(h).unwrap();
        let mut wrong = f;
        wrong.event += 1;
        assert_eq!(
            unsafe { o.observe_complete(h, HostKvCompletion { fence: wrong }) },
            Err(HostKvError::FenceMismatch)
        );
        for _ in 0..2 {
            assert_eq!(
                unsafe { o.observe_complete(h, HostKvCompletion { fence: f }) },
                Err(HostKvError::Unknown)
            );
            assert_eq!(o.state(h), Ok(HostKvPlacementState::Unknown));
            assert!(unsafe { o.reject_submission(h, f) }.is_err());
            assert!(o.submit(&a, h, f).is_err());
            assert!(o.cancel(&mut a, h).is_err());
            assert!(o.release(&mut a, h).is_err());
            assert_eq!(b.in_use(), credits());
        }
        drop(o);
        assert_eq!(a.live, 1);
        assert_eq!(a.drops.get(), 0);
        assert_eq!(b.in_use(), credits());
    }
}
#[test]
fn release_failure_retains_owner_and_budget_for_retry() {
    let (mut a, b, mut o, h) = setup();
    finish(&mut o, &a, h);
    a.fail_release = true;
    assert_eq!(o.release(&mut a, h), Err(HostKvError::ReaderActive));
    assert_eq!(b.in_use(), credits());
    a.fail_release = false;
    o.release(&mut a, h).unwrap();
    assert_eq!(b.in_use(), PlacementCredits::default());
}
#[test]
fn dropping_owner_does_not_drop_registered_lease() {
    let (a, b, o, _) = setup();
    let drops = a.drops.clone();
    drop(o);
    assert_eq!(a.live, 1);
    assert_eq!(drops.get(), 0);
    assert_ne!(b.in_use(), PlacementCredits::default());
}
#[test]
fn every_fence_field_and_owner_handle_are_exact() {
    let (mut a, b, mut o, h) = setup();
    let f = fence(h, HostKvStage::HostCopy);
    o.submit(&a, h, f).unwrap();
    for field in 0..6 {
        let mut bad = f;
        match field {
            0 => bad.owner += 1,
            1 => bad.operation += 1,
            2 => bad.generation += 1,
            3 => bad.stream += 1,
            4 => bad.event += 1,
            _ => bad.stage = HostKvStage::Upload,
        }
        assert_eq!(
            unsafe { o.observe_complete(h, HostKvCompletion { fence: bad }) },
            Err(HostKvError::FenceMismatch)
        );
        assert_eq!(o.state(h), Ok(HostKvPlacementState::HostCopyPending));
    }
    let mut foreign = HostKvPlacementOwner::new(b.clone()).unwrap();
    assert_eq!(foreign.cancel(&mut a, h), Err(HostKvError::ForeignOwner));
    unsafe { o.observe_complete(h, HostKvCompletion { fence: f }) }.unwrap();
    let mut bad = fence(h, HostKvStage::Upload);
    bad.generation += 1;
    assert_eq!(o.submit(&a, h, bad), Err(HostKvError::FenceMismatch));
    assert_eq!(b.in_use(), credits());
}
#[test]
fn shared_budget_cannot_be_double_spent_and_unknown_stops_admission() {
    let mut a = Authority::default();
    let b = PlacementBudget::new(PlacementCredits::from_domains([100, 100, 100, 2, 2, 2]));
    let mut o = HostKvPlacementOwner::new(b.clone()).unwrap();
    let mut other = HostKvPlacementOwner::new(b.clone()).unwrap();
    let h = o.reserve(&mut a, &source(), credits()).unwrap();
    let h2 = other.reserve(&mut a, &source(), credits()).unwrap();
    assert_eq!(
        o.reserve(&mut a, &source(), credits()),
        Err(HostKvError::BudgetExceeded)
    );
    other.cancel(&mut a, h2).unwrap();
    let f = fence(h, HostKvStage::HostCopy);
    o.submit(&a, h, f).unwrap();
    o.mark_unknown(h).unwrap();
    assert_eq!(
        o.reserve(&mut a, &source(), credits()),
        Err(HostKvError::Unknown)
    );
    assert!(unsafe { o.reject_submission(h, f) }.is_err());
    let drops = a.drops.get();
    drop(o);
    assert_eq!(a.drops.get(), drops);
    assert_eq!(b.in_use(), credits());
}
#[test]
fn empty_metadata_credits_and_id_exhaustion_cannot_bypass_limits() {
    let (mut a, b, mut o, h) = setup();
    o.cancel(&mut a, h).unwrap();
    for domain in 3..6 {
        let mut c = credits().domains();
        c[domain] = 0;
        assert_eq!(
            o.reserve(&mut a, &source(), PlacementCredits::from_domains(c)),
            Err(HostKvError::InvalidGeometry)
        );
    }
    o.next_operation = u64::MAX;
    assert_eq!(
        o.reserve(&mut a, &source(), credits()),
        Err(HostKvError::Overflow)
    );
    assert_eq!(b.in_use(), PlacementCredits::default());
    assert_eq!(a.live, 0);
}

#[test]
fn metadata_tracing_is_sparse_and_contains_only_safe_fields() {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing_subscriber::{Layer, layer::Context, prelude::*};

    type Fields = BTreeMap<String, String>;
    #[derive(Clone)]
    struct Events(Arc<Mutex<Vec<Fields>>>);
    struct Visitor(Fields);
    impl Visit for Visitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0.insert(field.name().to_owned(), format!("{value:?}"));
        }
    }
    impl<S: tracing::Subscriber> Layer<S> for Events {
        fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
            if event.metadata().target() == "ferrule::host_kv" {
                let mut visitor = Visitor(BTreeMap::new());
                event.record(&mut visitor);
                self.0.lock().unwrap().push(visitor.0);
            }
        }
    }
    let events = Events(Arc::default());
    let subscriber = tracing_subscriber::registry().with(events.clone());
    let _guard = tracing::subscriber::set_default(subscriber);
    let (mut a, b, mut o, h) = setup();
    let f = fence(h, HostKvStage::HostCopy);
    o.submit(&a, h, f).unwrap();
    assert!(o.submit(&a, h, f).is_err());
    unsafe { o.reject_submission(h, f) }.unwrap();
    o.cancel(&mut a, h).unwrap();
    assert!(o.cancel(&mut a, h).is_err());
    let h = o.reserve(&mut a, &source(), credits()).unwrap();
    let f = fence(h, HostKvStage::HostCopy);
    o.submit(&a, h, f).unwrap();
    o.mark_unknown(h).unwrap();
    o.mark_unknown(h).unwrap();
    assert!(o.release(&mut a, h).is_err());
    drop(o);
    assert_eq!(b.in_use(), credits());
    let events = events.0.lock().unwrap();
    let transitions: Vec<_> = events.iter().map(|e| e["transition"].as_str()).collect();
    assert_eq!(
        transitions,
        [
            "\"submit\"",
            "\"reject\"",
            "\"retire\"",
            "\"submit\"",
            "\"unknown\""
        ]
    );
    let mut allowed = [
        "message",
        "scope",
        "transition",
        "owner",
        "operation",
        "stage",
        "pageable_host_bytes",
        "pinned_host_bytes",
        "staging_bytes",
        "staging_slots",
        "transfers",
        "consumers",
    ];
    allowed.sort_unstable();
    for event in events.iter() {
        assert_eq!(
            event.keys().map(String::as_str).collect::<Vec<_>>(),
            allowed
        );
        assert_eq!(event["scope"], "\"metadata\"");
        assert_eq!(event["pageable_host_bytes"], "32");
        assert_eq!(event["pinned_host_bytes"], "32");
        assert_eq!(event["staging_bytes"], "32");
        for count in ["staging_slots", "transfers", "consumers"] {
            assert_eq!(event[count], "1");
        }
    }
}

#[test]
fn audited_rejection_retry_cannot_replay_the_previous_fence() {
    let (mut a, b, mut o, h) = setup();
    let old = fence(h, HostKvStage::HostCopy);
    o.submit(&a, h, old).unwrap();
    // SAFETY: driver-free no-work provider; no physical refs exist.
    unsafe { o.reject_submission(h, old) }.unwrap();
    assert_eq!(o.state(h), Ok(HostKvPlacementState::Reserved));
    assert_eq!(b.in_use(), credits());
    assert!(unsafe { o.reject_submission(h, old) }.is_err());
    for generation in [old.generation - 1, old.generation] {
        let replay = HostKvFence {
            generation,
            event: old.event + 1,
            ..old
        };
        assert_eq!(o.submit(&a, h, replay), Err(HostKvError::FenceMismatch));
        assert_eq!(o.state(h), Ok(HostKvPlacementState::Reserved));
    }
    let next = HostKvFence {
        generation: old.generation + 1,
        ..old // Even recycled stream/event numbers cannot replay the old attempt.
    };
    o.submit(&a, h, next).unwrap();
    assert_eq!(
        unsafe { o.observe_complete(h, HostKvCompletion { fence: old }) },
        Err(HostKvError::FenceMismatch)
    );
    assert!(unsafe { o.reject_submission(h, old) }.is_err());
    assert!(o.cancel(&mut a, h).is_err());
    assert_eq!(b.in_use(), credits());
    assert_eq!(a.live, 1);
    unsafe { o.reject_submission(h, next) }.unwrap();
    o.cancel(&mut a, h).unwrap();
    assert_eq!(b.in_use(), PlacementCredits::default());
    assert_eq!(a.live, 0);
}

#[test]
fn cross_owner_attestations_never_discharge_equal_local_operation_ids() {
    let (mut a, b, mut o, h) = setup();
    let (mut other_a, other_b, mut other, other_h) = setup();
    assert_eq!(h.operation(), other_h.operation());
    assert_ne!(h.owner(), other_h.owner());
    let f = fence(h, HostKvStage::HostCopy);
    let other_f = fence(other_h, HostKvStage::HostCopy);
    o.submit(&a, h, f).unwrap();
    other.submit(&other_a, other_h, other_f).unwrap();
    // Driver-free malformed attestations exercise the checked routing boundary.
    assert_eq!(
        unsafe { other.observe_complete(other_h, HostKvCompletion { fence: f }) },
        Err(HostKvError::FenceMismatch)
    );
    assert_eq!(
        unsafe { other.observe_complete(h, HostKvCompletion { fence: f }) },
        Err(HostKvError::ForeignOwner)
    );
    assert!(unsafe { other.reject_submission(other_h, f) }.is_err());
    assert_eq!(
        unsafe { other.reject_submission(h, f) },
        Err(HostKvError::ForeignOwner)
    );
    assert!(o.cancel(&mut a, h).is_err());
    assert!(other.cancel(&mut other_a, other_h).is_err());
    assert_eq!(b.in_use(), credits());
    assert_eq!(other_b.in_use(), credits());
    assert_eq!((a.live, other_a.live), (1, 1));
}
