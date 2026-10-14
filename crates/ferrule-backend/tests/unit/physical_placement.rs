use super::*;
use std::cell::Cell;

#[derive(Clone, Copy)]
enum Fence {
    Done,
    Pending,
    Failed,
    Panic,
}
unsafe impl PageCompletionFence for Fence {
    fn is_complete(&self) -> Result<bool> {
        match self {
            Self::Done => Ok(true),
            Self::Pending => Ok(false),
            Self::Failed => Err(error("injected fence failure")),
            Self::Panic => panic!("injected fence unwind"),
        }
    }
}

struct Backing(Rc<Cell<usize>>);
impl Drop for Backing {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
type Custody = PagePlacementCustody<Backing, Fence>;

// Test-only trusted adapter: these tokens model actual domain-scoped evidence.
unsafe fn submitted(
    s: &PagePlacementSubmission,
    fences: [Fence; 3],
) -> [PageCompletionToken<Fence>; 3] {
    let [producer, dma, consumer] = fences;
    unsafe {
        [
            PageCompletionToken::after_submission(s, PageCompletionDomain::Producer, producer),
            PageCompletionToken::after_submission(s, PageCompletionDomain::Dma, dma),
            PageCompletionToken::after_submission(s, PageCompletionDomain::Consumer, consumer),
        ]
    }
}
unsafe fn drained(
    s: &PagePlacementSubmission,
    fences: [Fence; 3],
) -> [PageCompletionToken<Fence>; 3] {
    let [producer, dma, consumer] = fences;
    unsafe {
        [
            PageCompletionToken::after_full_drain(s, PageCompletionDomain::Producer, producer),
            PageCompletionToken::after_full_drain(s, PageCompletionDomain::Dma, dma),
            PageCompletionToken::after_full_drain(s, PageCompletionDomain::Consumer, consumer),
        ]
    }
}

fn allocation(
    domain: &PageAllocationDomain,
    base: u64,
    drops: &Rc<Cell<usize>>,
) -> PageAllocation<Backing> {
    // Driver-free synthetic address space. No memory accesses are performed;
    // the drop counter models ownership of one full 4 KiB grant.
    unsafe {
        PageAllocation::from_backing(
            Backing(drops.clone()),
            domain.clone(),
            base,
            4096,
            4096,
            PageAllocationKind::FixedPlane,
        )
        .unwrap()
    }
}
fn fixture() -> (Custody, Rc<Cell<usize>>) {
    let domain = PageAllocationDomain::new();
    let drops = Rc::new(Cell::new(0));
    let original = allocation(&domain, 0x1000, &drops)
        .placement(KvPageId(1), 0, 0..1024, 4)
        .unwrap();
    let staged = allocation(&domain, 0x3000, &drops)
        .placement(KvPageId(1), 0, 0..1024, 4)
        .unwrap();
    (Custody::new(vec![original], vec![staged]).unwrap(), drops)
}
fn proof(custody: &mut Custody) -> (PagePlacementSubmission, PagePlacementProof) {
    let submission = custody.begin().unwrap();
    unsafe {
        custody
            .record_fences(&submission, submitted(&submission, [Fence::Done; 3]))
            .unwrap();
    }
    let proof = custody.poll(&submission).unwrap().unwrap();
    (submission, proof)
}

#[test]
fn chunks_are_page_bounded_aligned_and_identity_bound() {
    let (custody, _) = fixture();
    let page = &custody.original()[0];
    let chunk = page.chunk(4, 8).unwrap();
    custody.validate_chunk(&chunk).unwrap();
    for (offset, bytes) in [(1020, 8), (0, 0), (1, 4), (0, 3), (usize::MAX, 4)] {
        assert!(page.chunk(offset, bytes).is_err());
    }
    let (foreign, _) = fixture();
    assert!(foreign.validate_chunk(&chunk).is_err());
    assert_eq!(chunk.byte_range(), 4..12);
    assert_eq!(chunk.page(), KvPageId(1));
    assert_eq!(chunk.plane(), 0);
    assert_eq!(chunk.identity().generation(), 1);
}

#[test]
fn overlap_checks_absolute_ranges_not_only_identity_or_slot() {
    let domain = PageAllocationDomain::new();
    let drops = Rc::new(Cell::new(0));
    let backing = allocation(&domain, 0x1000, &drops);
    let original = backing.placement(KvPageId(1), 0, 0..1024, 4).unwrap();
    let staged = backing.placement(KvPageId(2), 0, 512..1536, 4).unwrap();
    assert!(Custody::new(vec![original], vec![staged]).is_err());
    // Fault injection: even different allocation identities cannot hide aliases.
    let alias = allocation(&domain, 0x1200, &drops);
    let original = backing.placement(KvPageId(1), 0, 0..1024, 4).unwrap();
    let staged = alias.placement(KvPageId(2), 0, 0..1024, 4).unwrap();
    assert!(Custody::new(vec![original], vec![staged]).is_err());
    let a = backing.placement(KvPageId(1), 0, 0..1024, 4).unwrap();
    let b = backing.placement(KvPageId(2), 0, 0..1024, 4).unwrap();
    assert!(Custody::new(vec![], vec![a, b]).is_err());
}

#[test]
fn shared_fixed_plane_is_charged_once_and_disjoint_writes_are_allowed() {
    let domain = PageAllocationDomain::new();
    let drops = Rc::new(Cell::new(0));
    let backing = allocation(&domain, 0x1000, &drops);
    let original = backing.placement(KvPageId(1), 0, 0..1024, 4).unwrap();
    let staged = backing.placement(KvPageId(1), 0, 1024..2048, 4).unwrap();
    let mut custody = Custody::new(vec![original], vec![staged]).unwrap();
    assert_eq!(custody.metrics().live_allocation_bytes, 4096);
    let (_, proof) = proof(&mut custody);
    custody.release(proof).unwrap();
    assert_eq!(custody.metrics().live_allocation_bytes, 0);
    assert_eq!(custody.metrics().driver_freed_bytes, None);
    assert_eq!(drops.get(), 0); // shared backing still lives, no physical drop
    drop(backing);
    assert_eq!(drops.get(), 1);
}

#[test]
fn every_domain_must_complete_before_proof_and_release() {
    for pending_domain in 0..3 {
        let (mut custody, drops) = fixture();
        let submission = custody.begin().unwrap();
        let mut fences = [Fence::Done; 3];
        fences[pending_domain] = Fence::Pending;
        unsafe {
            custody
                .record_fences(&submission, submitted(&submission, fences))
                .unwrap();
        }
        assert!(custody.poll(&submission).unwrap().is_none());
        assert_eq!(drops.get(), 0);
        assert_eq!(custody.metrics().live_allocation_bytes, 8192);
        assert!(custody.begin().is_err());
        custody.mark_unknown(&submission).unwrap();
        let proof = unsafe {
            custody
                .reap(&submission, drained(&submission, [Fence::Done; 3]))
                .unwrap()
                .unwrap()
        };
        assert_eq!(drops.get(), 0); // physical evidence alone does not release Entry
        custody.release(proof).unwrap();
        assert_eq!(drops.get(), 2);
    }
}

#[test]
fn unknown_and_cancel_reject_late_acks_and_partial_or_failed_reap() {
    let (mut custody, drops) = fixture();
    let chunk = custody.original()[0].chunk(0, 4).unwrap();
    let submission = custody.begin().unwrap();
    custody.mark_unknown(&submission).unwrap();
    custody.cancel();
    assert!(custody.poll(&submission).is_err());
    assert!(custody.validate_chunk(&chunk).is_err());
    assert!(
        unsafe { custody.record_fences(&submission, submitted(&submission, [Fence::Done; 3])) }
            .is_err()
    );
    for domain in 0..3 {
        let mut drains = [Fence::Done; 3];
        drains[domain] = Fence::Pending;
        assert!(
            unsafe { custody.reap(&submission, drained(&submission, drains)) }
                .unwrap()
                .is_none()
        );
        drains[domain] = Fence::Failed;
        assert!(unsafe { custody.reap(&submission, drained(&submission, drains)) }.is_err());
        assert!(custody.is_unknown());
        assert_eq!(drops.get(), 0);
        assert!(custody.begin().is_err());
    }
    let proof = unsafe {
        custody
            .reap(&submission, drained(&submission, [Fence::Done; 3]))
            .unwrap()
            .unwrap()
    };
    custody.release(proof).unwrap();
    assert_eq!(drops.get(), 2);
}

#[test]
fn failed_or_panicking_probe_quarantines_without_dropping_backing() {
    for fault in [Fence::Failed, Fence::Panic] {
        let (mut custody, drops) = fixture();
        let submission = custody.begin().unwrap();
        unsafe {
            custody
                .record_fences(
                    &submission,
                    submitted(&submission, [Fence::Done, fault, Fence::Done]),
                )
                .unwrap();
        }
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| custody.poll(&submission)));
        assert!(result.is_err() || result.unwrap().is_err());
        assert!(custody.is_unknown());
        assert_eq!(drops.get(), 0);
        let proof = unsafe {
            custody
                .reap(&submission, drained(&submission, [Fence::Done; 3]))
                .unwrap()
                .unwrap()
        };
        custody.release(proof).unwrap();
        assert_eq!(drops.get(), 2);
    }
}

#[test]
fn abandoned_submission_pending_and_unknown_all_leak_instead_of_recycle() {
    for state in [State::Submitting, State::Pending, State::Unknown] {
        let (mut custody, drops) = fixture();
        let submission = custody.begin().unwrap();
        if state == State::Pending {
            unsafe {
                custody
                    .record_fences(&submission, submitted(&submission, [Fence::Pending; 3]))
                    .unwrap();
            }
        }
        if state == State::Unknown {
            custody.mark_unknown(&submission).unwrap();
        }
        drop(custody);
        assert_eq!(drops.get(), 0);
    }
    let (custody, drops) = fixture();
    drop(custody); // positively no work submitted
    assert_eq!(drops.get(), 2);
}

#[test]
fn slot_reuse_changes_generation_but_never_reports_vram_savings() {
    let (mut custody, drops) = fixture();
    let before = custody.metrics();
    let old_chunk = custody.original()[0].chunk(0, 4).unwrap();
    let (old_submission, proof) = proof(&mut custody);
    assert!(custody.poll(&old_submission).is_err()); // no duplicate proof
    custody.reuse(proof).unwrap();
    assert_eq!(custody.metrics().slot_reuses, 2);
    assert_eq!(
        custody.metrics().live_allocation_bytes,
        before.live_allocation_bytes
    );
    assert_eq!(custody.metrics().driver_freed_bytes, None);
    assert_eq!(drops.get(), 0);
    assert!(custody.validate_chunk(&old_chunk).is_err());
    assert!(custody.mark_unknown(&old_submission).is_err());
    let new_chunk = custody.original()[0].chunk(0, 4).unwrap();
    assert_eq!(new_chunk.identity().generation(), 2);
    custody.validate_chunk(&new_chunk).unwrap();
    let submission = custody.begin().unwrap();
    assert!(
        unsafe {
            custody.record_fences(
                &old_submission,
                submitted(&old_submission, [Fence::Done; 3]),
            )
        }
        .is_err()
    );
    unsafe {
        custody
            .record_fences(&submission, submitted(&submission, [Fence::Done; 3]))
            .unwrap();
    }
    let proof = custody.poll(&submission).unwrap().unwrap();
    custody.release(proof).unwrap();
    assert_eq!(drops.get(), 2);
}

#[test]
fn foreign_submission_and_proof_cannot_release_custody() {
    let (mut a, _) = fixture();
    let (mut b, drops) = fixture();
    let (submission, proof) = proof(&mut a);
    assert!(b.poll(&submission).is_err());
    assert!(b.release(proof).is_err());
    assert_eq!(drops.get(), 0);
    assert_eq!(b.metrics().live_allocation_bytes, 8192);
}

#[test]
fn invalid_extents_foreign_domains_and_single_page_slicing_are_rejected() {
    let domain = PageAllocationDomain::new();
    let drops = Rc::new(Cell::new(0));
    for (base, requested, granted) in [(0, 1, 1), (1, 0, 1), (1, 2, 1), (u64::MAX, 1, 1)] {
        assert!(
            unsafe {
                PageAllocation::from_backing(
                    Backing(drops.clone()),
                    domain.clone(),
                    base,
                    requested,
                    granted,
                    PageAllocationKind::SinglePage,
                )
            }
            .is_err()
        );
    }
    let a = allocation(&domain, 0x1000, &drops);
    for (range, alignment) in [(0..4097, 4), (0..0, 4), (1..8, 4), (0..4, 0), (0..4, 3)] {
        assert!(a.placement(KvPageId(1), 0, range, alignment).is_err());
    }
    let b = allocation(&PageAllocationDomain::new(), 0x3000, &drops);
    assert!(
        Custody::new(
            vec![a.placement(KvPageId(1), 0, 0..1024, 4).unwrap()],
            vec![b.placement(KvPageId(1), 0, 0..1024, 4).unwrap()],
        )
        .is_err()
    );
    let single = unsafe {
        PageAllocation::from_backing(
            Backing(drops),
            domain,
            0x5000,
            1024,
            4096,
            PageAllocationKind::SinglePage,
        )
    }
    .unwrap();
    assert!(single.placement(KvPageId(1), 0, 0..512, 4).is_err());
    let page = single.placement(KvPageId(1), 0, 0..1024, 4).unwrap();
    let mut custody = Custody::new(vec![page], vec![]).unwrap();
    let (_, proof) = proof(&mut custody);
    custody.reuse(proof).unwrap();
    assert_eq!(custody.metrics().slot_reuses, 0);
    assert_eq!(custody.metrics().live_allocation_bytes, 4096);
}

#[test]
fn rebinding_same_slot_and_reallocating_same_address_cannot_resurrect_metadata() {
    let domain = PageAllocationDomain::new();
    let drops = Rc::new(Cell::new(0));
    let grant = allocation(&domain, 0x1000, &drops);
    let page = grant.placement(KvPageId(1), 0, 0..1024, 4).unwrap();
    let old = page.chunk(0, 4).unwrap();
    let mut custody = Custody::new(vec![page], vec![]).unwrap();
    let (_, proof) = proof(&mut custody);
    custody.release(proof).unwrap();
    let rebound = grant.placement(KvPageId(1), 0, 0..1024, 4).unwrap();
    let custody = Custody::new(vec![rebound], vec![]).unwrap();
    assert!(custody.validate_chunk(&old).is_err());
    drop(custody);
    drop(grant);
    let reused_address = allocation(&domain, 0x1000, &drops);
    let page = reused_address
        .placement(KvPageId(1), 0, 0..1024, 4)
        .unwrap();
    let custody = Custody::new(vec![page], vec![]).unwrap();
    assert!(custody.validate_chunk(&old).is_err());
}

#[test]
fn canceled_placement_cannot_restart_even_after_proof() {
    let (mut custody, drops) = fixture();
    custody.cancel();
    assert!(custody.begin().is_err());
    drop(custody);
    assert_eq!(drops.get(), 2);
    let (mut custody, _) = fixture();
    let (_, proof) = proof(&mut custody);
    custody.cancel();
    assert!(custody.reuse(proof).is_err());
}

#[test]
fn complete_physical_drain_is_terminal_not_a_way_to_wash_sticky_unknown() {
    for attempt_reuse in [false, true] {
        let (mut custody, drops) = fixture();
        let chunk = custody.original()[0].chunk(0, 4).unwrap();
        let s = custody.begin().unwrap();
        custody.mark_unknown(&s).unwrap();
        let tokens = unsafe { drained(&s, [Fence::Done; 3]) };
        let proof = custody.reap(&s, tokens).unwrap().unwrap();
        assert!(custody.is_unknown());
        assert!(custody.begin().is_err());
        assert!(custody.poll(&s).is_err());
        assert!(custody.validate_chunk(&chunk).is_err());
        assert_eq!(drops.get(), 0);
        assert_eq!(custody.metrics().slot_reuses, 0);
        if attempt_reuse {
            assert!(custody.reuse(proof).is_err());
            assert!(custody.is_unknown());
        } else {
            custody.release(proof).unwrap();
            assert_eq!(drops.get(), 2);
        }
    }
}

#[test]
fn foreign_wrong_domain_and_late_ack_tokens_cannot_be_drains() {
    let (mut a, _) = fixture();
    let other = a.begin().unwrap();
    let (mut b, drops) = fixture();
    let s = b.begin().unwrap();
    // Correct outer handle cannot launder a foreign token in any one domain.
    for domain in 0..3 {
        let mut tokens = unsafe { submitted(&s, [Fence::Done; 3]) };
        let foreign = unsafe { submitted(&other, [Fence::Done; 3]) };
        tokens[domain] = foreign.into_iter().nth(domain).unwrap();
        assert!(b.record_fences(&s, tokens).is_err());
    }
    let mut tokens = unsafe { submitted(&s, [Fence::Done; 3]) };
    tokens.swap(0, 1);
    assert!(b.record_fences(&s, tokens).is_err());
    let tokens = unsafe { drained(&s, [Fence::Done; 3]) };
    assert!(b.record_fences(&s, tokens).is_err());
    b.mark_unknown(&s).unwrap();
    let late_acks = unsafe { submitted(&s, [Fence::Done; 3]) };
    assert!(b.reap(&s, late_acks).is_err());
    for domain in 0..3 {
        let mut tokens = unsafe { drained(&s, [Fence::Done; 3]) };
        let foreign = unsafe { drained(&other, [Fence::Done; 3]) };
        tokens[domain] = foreign.into_iter().nth(domain).unwrap();
        assert!(b.reap(&s, tokens).is_err());
        assert!(b.is_unknown());
        assert_eq!(drops.get(), 0);
    }
    let mut tokens = unsafe { drained(&s, [Fence::Done; 3]) };
    tokens.swap(1, 2);
    assert!(b.reap(&s, tokens).is_err());
    let tokens = unsafe { drained(&s, [Fence::Done; 3]) };
    let proof = b.reap(&s, tokens).unwrap().unwrap();
    b.release(proof).unwrap();
}

#[test]
fn opaque_submission_incarnation_and_generation_both_bind_completion() {
    let (mut custody, _) = fixture();
    let (old, proof) = proof(&mut custody);
    let old_tokens = unsafe { submitted(&old, [Fence::Done; 3]) };
    custody.reuse(proof).unwrap();
    let s = custody.begin().unwrap();
    assert!(custody.record_fences(&s, old_tokens).is_err());
    // Fault injection available only inside this private unit-test module:
    // copying all public-looking numbers still cannot impersonate a submission.
    let forged = PagePlacementSubmission {
        owner: s.owner.clone(),
        generation: s.generation,
        submission: Identity::new(),
    };
    let forged_tokens = unsafe { submitted(&forged, [Fence::Done; 3]) };
    assert!(custody.record_fences(&s, forged_tokens).is_err());
    let tokens = unsafe { submitted(&s, [Fence::Done; 3]) };
    custody.record_fences(&s, tokens).unwrap();
    let proof = custody.poll(&s).unwrap().unwrap();
    custody.release(proof).unwrap();
}

#[test]
fn generation_exhaustion_fails_without_wrap_or_releasing_backing() {
    let (mut custody, drops) = fixture();
    custody.generation = u64::MAX;
    let (_, proof) = proof(&mut custody);
    assert!(custody.reuse(proof).is_err());
    assert_eq!(custody.generation, u64::MAX);
    assert_eq!(custody.metrics().slot_reuses, 0);
    assert_eq!(drops.get(), 0);
}
