use super::*;
use crate::transformer::hybrid_parallel::mixed_decoded::DecodedIdentity;
use std::{cell::Cell, sync::Arc};
pub(super) const MIB: usize = 1 << 20;
pub(super) fn descriptor(id: u32, bytes: usize, layer: Option<usize>) -> DecodedDescriptor {
    DecodedDescriptor {
        identity: Arc::new(DecodedIdentity {
            owner: id,
            geometry: [bytes / 2, 1],
            parts: vec![],
        }),
        grants: bytes,
        routed_layer: layer,
        operands: vec![(id as usize, 0)],
    }
}
pub(super) fn pending<T>(capacity: usize, ds: &[DecodedDescriptor]) -> DecodedCustody<T> {
    let budget = HybridMixedDecodedCacheBudget::new(capacity, ds).unwrap();
    let ds = DecodedArenaDescriptors::new(budget, ds.to_vec()).unwrap();
    DecodedCustody::new(budget, 73, &ds).unwrap()
}
pub(super) fn cache(capacity: usize, ds: &[DecodedDescriptor]) -> DecodedCustody<usize> {
    let mut c = pending(capacity, ds);
    c.install_arena(usize::MAX, capacity, |_, offset, _| Ok(offset))
        .unwrap();
    c
}
pub(super) fn touch(c: &mut DecodedCustody<usize>, d: &DecodedDescriptor) -> bool {
    match c.acquire(d).unwrap() {
        Lookup::Hit(view) => {
            drop(view);
            false
        }
        Lookup::Decode(view) => {
            c.mark_ready(d).unwrap();
            drop(view);
            true
        }
        Lookup::Native => panic!("unexpected native fallback"),
    }
}
fn pair_inventory() -> Vec<DecodedDescriptor> {
    let mut ds = vec![descriptor(0, 58 * MIB, None)];
    for layer in 0..3 {
        ds.push(descriptor(1 + layer as u32 * 2, 2 * MIB, Some(layer)));
        ds.push(descriptor(2 + layer as u32 * 2, MIB, Some(layer)));
    }
    ds
}
#[test]
fn sixty_four_mib_a_b_a_uses_empty_slots_not_repeated_segment_reserves() {
    let ds = vec![descriptor(1, MIB, Some(0)), descriptor(2, MIB, Some(1))];
    let mut c = cache(64 * MIB, &ds);
    assert!(touch(&mut c, &ds[0]));
    c.complete(true).unwrap();
    assert!(touch(&mut c, &ds[1]));
    c.complete(true).unwrap();
    for _ in 0..4 {
        assert!(!touch(&mut c, &ds[0]));
        assert!(!touch(&mut c, &ds[1]));
        c.complete(true).unwrap();
    }
    assert_eq!(c.stats().evictions, 0);
    assert_eq!(c.physical_stats().fallbacks, 0);
    assert_eq!(c.physical_stats().charged(), 64 * MIB);
    assert_eq!(c.physical_stats().driver_growth, 64 * MIB);
    assert_eq!(c.physical_stats().unused_tail, 62 * MIB);
}
#[test]
fn class_full_evicts_true_same_class_lru_not_older_incompatible_slot() {
    let ds = pair_inventory();
    let mut c = cache(64 * MIB, &ds);
    for i in [0, 2, 1, 3, 1] {
        touch(&mut c, &ds[i]);
        c.complete(true).unwrap();
    }
    let old = c.roots[&ds[3].key(73)].view.clone();
    let address = Rc::as_ptr(&old);
    drop(old);
    assert!(touch(&mut c, &ds[5]));
    c.complete(true).unwrap();
    assert_eq!(Rc::as_ptr(&c.roots[&ds[5].key(73)].view), address);
    assert!(!c.roots.contains_key(&ds[3].key(73)));
    assert!(!touch(&mut c, &ds[2]), "older down is not a pair victim");
    assert!(!touch(&mut c, &ds[0]), "pins are not victims");
    c.complete(true).unwrap();
    assert_eq!(c.stats().evictions, 1);
    assert_eq!(c.physical_stats().reuses, 1);
}
#[test]
fn pending_and_consuming_cannot_supply_a_slot_but_same_stream_hit_is_allowed() {
    let ds = pair_inventory();
    let mut c = cache(64 * MIB, &ds);
    for i in [1, 3] {
        assert!(touch(&mut c, &ds[i]));
    }
    assert!(!touch(&mut c, &ds[1]));
    assert!(matches!(c.acquire(&ds[5]).unwrap(), Lookup::Native));
    assert_eq!(c.stats().evictions, 0);
    c.complete(true).unwrap();
    touch(&mut c, &ds[1]);
    touch(&mut c, &ds[3]);
    assert!(matches!(c.acquire(&ds[5]).unwrap(), Lookup::Native));
    c.complete(true).unwrap();
    assert!(touch(&mut c, &ds[5]));
    c.complete(true).unwrap();
}
#[test]
fn fenced_failed_decode_pools_slot_and_pin_is_redecoded_without_refund() {
    let ds = pair_inventory();
    let mut c = cache(64 * MIB, &ds);
    let Lookup::Decode(view) = c.acquire(&ds[1]).unwrap() else {
        panic!()
    };
    assert!(c.acquire(&ds[1]).is_err());
    drop(view);
    touch(&mut c, &ds[0]);
    c.complete(false).unwrap();
    assert_eq!(c.physical_stats().pooled, 2 * MIB);
    assert_eq!(c.physical_stats().charged(), 64 * MIB);
    assert_eq!(c.stats().pending_upload_bytes, 0);
    assert!(touch(&mut c, &ds[0]));
    assert!(touch(&mut c, &ds[3]));
    c.complete(true).unwrap();
    assert_eq!(c.stats().evictions, 0);
    assert_eq!(c.physical_stats().reuses, 1);
}
#[test]
fn unsupported_size_or_changed_descriptor_preflight_does_not_evict() {
    let ds = pair_inventory();
    let mut c = cache(64 * MIB, &ds);
    for i in [1, 3] {
        touch(&mut c, &ds[i]);
        c.complete(true).unwrap();
    }
    let before = c.snapshot(true, false, false, false, None);
    assert!(c.acquire(&descriptor(90, 3 * MIB, Some(0))).is_err());
    let mut changed = ds[1].clone();
    changed.grants = MIB;
    assert!(c.acquire(&changed).is_err());
    changed = ds[1].clone();
    changed.routed_layer = None;
    assert!(c.acquire(&changed).is_err());
    assert_eq!(c.snapshot(true, false, false, false, None), before);
    let duplicates = vec![ds[1].clone(), ds[1].clone()];
    let budget = HybridMixedDecodedCacheBudget::new(64 * MIB, &duplicates).unwrap();
    assert!(DecodedArenaDescriptors::new(budget, duplicates).is_err());
}
#[test]
fn surviving_strong_or_weak_alias_quarantines_entire_root_stickily() {
    let ds = pair_inventory();
    for weak_only in [false, true] {
        let mut c = cache(64 * MIB, &ds);
        let Lookup::Decode(view) = c.acquire(&ds[1]).unwrap() else {
            panic!()
        };
        c.mark_ready(&ds[1]).unwrap();
        let weak = Rc::downgrade(&view);
        let held = if weak_only {
            drop(view);
            None
        } else {
            Some(view)
        };
        assert!(c.complete(true).is_err());
        drop(held);
        assert!(c.complete(false).is_err());
        assert!(c.acquire(&ds[3]).is_err());
        assert_eq!(c.physical_stats().unknown, 64 * MIB);
        drop(c);
        assert!(weak.upgrade().is_some(), "Unknown retains view and root");
    }
}
#[test]
fn unknown_and_partial_install_failure_retain_root_and_all_slots() {
    struct Root(Rc<Cell<usize>>);
    impl Drop for Root {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let ds = pair_inventory();
    for fail_slice in [false, true] {
        let drops = Rc::new(Cell::new(0));
        let mut c = pending(64 * MIB, &ds);
        let mut calls = 0;
        let installed = c.install_arena(Root(drops.clone()), 64 * MIB, |r, _, _| {
            calls += 1;
            if fail_slice && calls == 2 {
                return Err(invalid("view failure"));
            }
            Ok(Root(r.0.clone()))
        });
        assert_eq!(installed.is_err(), fail_slice);
        c.quarantine();
        let frozen = c.physical_stats();
        assert!(c.complete(false).is_err());
        assert!(c.acquire(&ds[1]).is_err());
        assert_eq!(c.physical_stats(), frozen);
        assert_eq!(frozen.unknown, 64 * MIB);
        drop(c);
        assert_eq!(drops.get(), 0);
    }
}
#[test]
fn failed_or_unsettled_root_allocation_never_refunds_or_retries() {
    let ds = pair_inventory();
    let mut c = pending::<usize>(64 * MIB, &ds);
    c.arena_allocation_failed(32 * MIB).unwrap();
    assert_eq!(c.physical_stats().charged(), 64 * MIB);
    assert_eq!(c.physical_stats().unknown, 64 * MIB);
    assert!(c.complete(false).is_err());
    assert!(c.install_arena(0, 0, |_, o, _| Ok(o)).is_err());
    let mut c = pending::<usize>(64 * MIB, &ds);
    assert!(c.complete(false).is_err());
    assert_eq!(c.physical_stats().unknown, 64 * MIB);
    let mut c = pending::<usize>(64 * MIB, &ds);
    assert!(c.install_arena(0, 65 * MIB, |_, o, _| Ok(o)).is_err());
    assert_eq!(c.physical_stats().arena_backing, 64 * MIB);
}
#[test]
fn existing_allocator_holes_do_not_change_full_root_charge() {
    let ds = pair_inventory();
    let mut c = pending::<usize>(64 * MIB, &ds);
    c.install_arena(0, 0, |_, o, _| Ok(o)).unwrap();
    assert_eq!(c.physical_stats().charged(), 64 * MIB);
    assert_eq!(c.physical_stats().driver_growth, 0);
}
#[test]
fn pair_down_fit_set_is_stable_after_warmup_and_real_3366_mib_set_evicts() {
    let mut ds = vec![descriptor(0, 1340 * MIB, None)];
    for layer in 0..40 {
        for expert in 0..32 {
            let id = 1 + (layer * 32 + expert) as u32 * 2;
            ds.push(descriptor(id, 2 * MIB, Some(layer)));
            ds.push(descriptor(id + 1, MIB, Some(layer)));
        }
    }
    let mut c = cache(3072 * MIB, &ds);
    touch(&mut c, &ds[0]);
    c.complete(true).unwrap();
    for pass in 0..3 {
        for layer in 0..40 {
            for d in &ds[1 + layer * 64..1 + layer * 64 + 16] {
                assert_eq!(touch(&mut c, d), pass == 0);
            }
            c.complete(true).unwrap();
        }
    }
    assert_eq!(c.physical_stats().active, 2300 * MIB);
    assert_eq!(c.stats().evictions, 0);
    assert_eq!(c.physical_stats().fallbacks, 0);
    // 1122 complete experts = 3366 MiB routed: legitimately exceeds capacity.
    for layer in ds[1..1 + 1122 * 2].chunks(64) {
        for d in layer {
            touch(&mut c, d);
        }
        c.complete(true).unwrap();
    }
    assert!(c.stats().evictions > 0);
    assert_eq!(c.physical_stats().fallbacks, 0);
    assert_eq!(c.physical_stats().charged(), 3072 * MIB);
}

#[test]
fn every_class_at_capacity_has_no_warm_eviction_with_healthy_boundaries() {
    let ds = pair_inventory();
    let mut c = cache(64 * MIB, &ds);
    for pass in 0..8 {
        let order = if pass % 2 == 0 {
            [1, 2, 3, 4]
        } else {
            [4, 1, 3, 2]
        };
        for i in order {
            assert_eq!(touch(&mut c, &ds[i]), pass == 0);
        }
        c.complete(true).unwrap();
    }
    assert_eq!(c.physical_stats().free, 0);
    assert_eq!(c.stats().evictions, 0);
    assert_eq!(c.physical_stats().fallbacks, 0);
}

#[test]
fn fixed_and_exact_element_classes_get_deterministic_disjoint_slices() {
    let grant = ferrule_backend::cuda_allocation_grant_bytes(34).unwrap();
    let mut small = descriptor(4, grant, Some(0));
    small.identity = Arc::new(DecodedIdentity {
        owner: 4,
        geometry: [17, 1],
        parts: vec![],
    });
    let ds = vec![
        descriptor(0, MIB, None),
        descriptor(1, 2 * MIB, Some(0)),
        descriptor(2, MIB, None),
        descriptor(3, grant, Some(0)),
        small,
    ];
    let mut c = pending::<usize>(64 * MIB, &ds);
    let mut ranges = Vec::new();
    c.install_arena(0, 64 * MIB, |_, offset, elements| {
        ranges.push((offset, offset + elements));
        Ok(offset)
    })
    .unwrap();
    let mut end = 0;
    for (start, next) in ranges {
        assert!(start >= end);
        end = next;
    }
    assert!(end <= 32 * MIB);
    assert_eq!(c.plan().fixed[0].offset_words, 0);
    assert_eq!(c.plan().fixed[1].offset_words, MIB / 2);
    assert_eq!(c.plan().classes.len(), 3);
    assert_eq!(c.plan().slot_padding_bytes, grant - 34);
    let a = c.roots[&ds[0].key(73)].view.clone();
    let b = c.roots[&ds[2].key(73)].view.clone();
    assert!(!Rc::ptr_eq(&a, &b));
    drop((a, b));
    for d in &ds {
        touch(&mut c, d);
    }
    c.complete(true).unwrap();
}

#[test]
fn capped_class_inventory_redistributes_remaining_capacity() {
    let mut ds = vec![descriptor(0, 2 * MIB, Some(0))];
    for i in 1..=80 {
        ds.push(descriptor(i, MIB, Some(i as usize)));
    }
    let c = pending::<usize>(64 * MIB, &ds);
    let down = c
        .plan()
        .classes
        .iter()
        .find(|c| c.slot_bytes == MIB)
        .unwrap();
    assert_eq!(down.count, 62);
    assert_eq!(c.plan().unused_bytes, 0);
}

#[test]
fn partial_install_panic_retains_root_but_healthy_completed_drop_releases_it() {
    struct Root(Rc<Cell<usize>>);
    impl Drop for Root {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let ds = pair_inventory();
    let drops = Rc::new(Cell::new(0));
    let mut c = pending(64 * MIB, &ds);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        c.install_arena(Root(drops.clone()), 64 * MIB, |_, _, _| {
            panic!("slice panic")
        })
    }));
    assert!(result.is_err());
    drop(c);
    assert_eq!(drops.get(), 0);
    let drops = Rc::new(Cell::new(0));
    let mut c = pending(64 * MIB, &ds);
    c.install_arena(Root(drops.clone()), 64 * MIB, |r, _, _| {
        Ok(Root(r.0.clone()))
    })
    .unwrap();
    let count = 1 + c.plan().fixed.len() + c.plan().classes.iter().map(|c| c.count).sum::<usize>();
    c.complete(true).unwrap();
    drop(c);
    assert_eq!(drops.get(), count);
}

#[test]
fn custody_and_admission_compare_every_layout_field_not_just_budget_totals() {
    use crate::transformer::hybrid_parallel::mixed_decoded::DecodedCacheAdmission;
    let ds = pair_inventory();
    let budget = HybridMixedDecodedCacheBudget::new(64 * MIB, &ds).unwrap();
    let ds = DecodedArenaDescriptors::new(budget, ds).unwrap();
    let admission = DecodedCacheAdmission {
        budget,
        arena: ds.layout.clone(),
        physical_pages: 64,
        device_capacity: 128 * MIB,
        owner_margins: 0,
    };
    let cache = DecodedCustody::<usize>::new(budget, 73, &ds).unwrap();
    assert_eq!(cache.plan(), admission.arena.as_ref());
    assert!(Arc::ptr_eq(&cache.plan, &admission.arena));
    for field in 0..14 {
        let mut changed = ds.clone();
        let p = Arc::make_mut(&mut changed.layout);
        match field {
            0 => p.arena_grants += 1,
            1 => p.fixed[0].descriptor += 1,
            2 => p.fixed[0].offset_words += 1,
            3 => p.fixed[0].elements += 1,
            4 => p.fixed[0].slot_bytes += 1,
            5 => p.classes[0].offset_words += 1,
            6 => p.classes[0].elements += 1,
            7 => p.classes[0].slot_bytes += 1,
            8 => p.classes[0].count += 1,
            9 => p.classes[0].layer_minimum += 1,
            10 => p.routed_slot_bytes += 1,
            11 => p.unused_bytes += 1,
            12 => p.slot_padding_bytes += 1,
            _ => p.classes.swap(0, 1),
        }
        let mut other = admission.clone();
        other.arena = changed.layout.clone();
        assert_eq!(other.budget, admission.budget);
        assert_ne!(
            other, admission,
            "claim admission must include full layout field {field}"
        );
        assert!(
            DecodedCustody::<usize>::new(budget, 73, &changed).is_err(),
            "custody field {field}"
        );
    }
}

#[test]
fn aggregate_bytes_fit_does_not_imply_static_class_fit() {
    let mut ds = vec![descriptor(0, 1340 * MIB, None)];
    for i in 0..600 {
        ds.push(descriptor(1 + i * 2, 2 * MIB, Some(i as usize)));
        ds.push(descriptor(2 + i * 2, MIB, Some(i as usize)));
    }
    let mut c = cache(3072 * MIB, &ds);
    assert_eq!(c.plan().classes[0].count, 578);
    // This is deliberately NOT a complete pair/down expert working set.
    // 579 down-only slots fit in bytes but exceed the immutable down quota.
    assert!(c.pinned_allocated + 579 * MIB < c.plan().arena_grants);
    for d in ds.iter().skip(2).step_by(2).take(579) {
        touch(&mut c, d);
        c.complete(true).unwrap();
    }
    assert_eq!(c.stats().evictions, 1);
    assert_eq!(c.physical_stats().fallbacks, 0);
}

#[test]
fn completion_fault_quarantines_all_groups_even_after_partial_publication() {
    let ds = pair_inventory();
    for succeeded in [false, true] {
        let mut c = cache(64 * MIB, &ds);
        for i in [1, 2] {
            touch(&mut c, &ds[i]);
        }
        // Inject a premature/double ledger completion while outer custody still
        // owns the borrow. No other class may be recycled after this fault.
        let lease = c.pending[&ds[2].key(73)].clone();
        c.ledger.complete(&lease, true).unwrap();
        assert!(c.complete(succeeded).is_err());
        let frozen = c.snapshot(true, false, false, false, None);
        assert!(frozen.quarantined);
        assert_eq!(frozen.unknown_reserved_bytes, 64 * MIB);
        assert_eq!(frozen.evictions, 0);
        for d in &ds {
            assert!(c.acquire(d).is_err());
        }
        assert!(c.complete(true).is_err());
        assert!(c.complete(false).is_err());
        assert_eq!(c.snapshot(true, false, false, false, None), frozen);
    }
}
