use super::*;

#[derive(Debug, PartialEq, Eq)]
struct Snapshot<K, G> {
    stats: ExpertCacheStats,
    clock: u64,
    entries: Vec<(K, usize, u64, State, Option<G>)>,
    registration: Option<BTreeMap<K, G>>,
    completed: Option<BTreeMap<G, BTreeSet<(u64, K)>>>,
}

fn snapshot<K: Ord + Clone, G: Ord + Clone>(cache: &Cache<K, G>) -> Snapshot<K, G> {
    Snapshot {
        stats: cache.stats(),
        clock: cache.clock,
        entries: cache
            .entries
            .iter()
            .map(|(k, e)| (k.clone(), e.bytes, e.epoch, e.state, e.group.clone()))
            .collect(),
        registration: cache.groups.as_ref().map(|g| g.registration.clone()),
        completed: cache.groups.as_ref().map(|g| g.completed.clone()),
    }
}

fn cache(slots: usize, bytes: usize, diagnostics: bool) -> Cache<u64, u64> {
    let mut cache = Cache::with_groups(
        ExpertCacheLimits {
            max_experts: slots,
            max_bytes: bytes,
        },
        (0..16).map(|key| (key, key % 3)),
    )
    .unwrap();
    cache.stats.diagnostics_enabled = diagnostics;
    cache
}

fn idle(cache: &mut Cache<u64, u64>, key: u64, bytes: usize) -> Lease<u64> {
    let lease = cache
        .acquire(key, bytes, |_| panic!("unexpected eviction"))
        .unwrap();
    cache.complete(&lease, true).unwrap();
    lease
}

fn consistent(cache: &Cache<u64, u64>) {
    let mut resident_bytes = 0;
    let mut resident_experts = 0;
    let mut pending_bytes = 0;
    let mut completed = BTreeMap::<_, BTreeSet<_>>::new();
    let groups = cache.groups.as_ref().unwrap();
    for (key, entry) in &cache.entries {
        assert_eq!(entry.group.as_ref(), groups.registration.get(key));
        match entry.state {
            State::Uploading => pending_bytes += entry.bytes,
            State::Consuming | State::Idle => {
                resident_bytes += entry.bytes;
                resident_experts += 1;
            }
        }
        if entry.state == State::Idle && !cache.stats.quarantined {
            completed
                .entry(entry.group.unwrap())
                .or_default()
                .insert((entry.epoch, *key));
        }
    }
    assert_eq!(cache.stats.resident_bytes, resident_bytes);
    assert_eq!(cache.stats.resident_experts, resident_experts);
    assert_eq!(cache.stats.pending_upload_bytes, pending_bytes);
    assert!(cache.stats.charged_bytes() <= cache.limits.max_bytes);
    assert!(cache.entries.len() <= cache.limits.max_experts);
    let actual: BTreeMap<_, _> = groups
        .completed
        .iter()
        .filter(|(_, entries)| !entries.is_empty())
        .map(|(g, entries)| (*g, entries.clone()))
        .collect();
    assert_eq!(completed, actual);
}

#[test]
fn assessment_is_pure_and_idle_only_victims_preserve_lru() {
    for diagnostics in [false, true] {
        let mut cache = cache(3, 24, diagnostics);
        idle(&mut cache, 0, 8);
        idle(&mut cache, 1, 8);
        idle(&mut cache, 2, 8);
        let before = snapshot(&cache);
        for _ in 0..3 {
            assert_eq!(cache.assess_admission(&3, 8, 8).unwrap(), Admission::Fits);
            assert_eq!(cache.assess_admission(&2, 8, 8).unwrap(), Admission::Fits);
            assert_eq!(snapshot(&cache), before);
        }
        let mut victims = Vec::new();
        cache.scratch(8, |k| victims.push(*k)).unwrap();
        let lease = cache.acquire(3, 8, |k| victims.push(*k)).unwrap();
        assert_eq!(victims, [0, 1]);
        assert_eq!(cache.stats.charged_bytes(), 24);
        cache.complete(&lease, true).unwrap();
        consistent(&cache);
    }
}

#[test]
fn assessment_rejects_destructive_partial_eviction_before_it_happens() {
    for consuming in [false, true] {
        let mut cache = cache(3, 32, true);
        let lease = if consuming {
            idle(&mut cache, 0, 16);
            cache.acquire(0, 16, |_| panic!()).unwrap()
        } else {
            cache.acquire(0, 16, |_| panic!()).unwrap()
        };
        idle(&mut cache, 1, 8);
        cache.scratch(4, |_| panic!()).unwrap();
        let before = snapshot(&cache);
        assert_eq!(
            cache.assess_admission(&2, 16, 4).unwrap(),
            Admission::RequiresFlush
        );
        assert_eq!(snapshot(&cache), before);
        // The existing acquire path really would evict useful state and fail.
        let mut victims = Vec::new();
        assert!(cache.acquire(2, 16, |k| victims.push(*k)).is_err());
        assert_eq!(victims, [1]);
        cache.complete(&lease, true).unwrap();
        assert_eq!(cache.assess_admission(&2, 16, 4).unwrap(), Admission::Fits);
        consistent(&cache);
    }
}

#[test]
fn assessment_distinguishes_entry_pressure_and_same_key_overlap() {
    let mut cache = cache(1, 128, true);
    let lease = cache.acquire(0, 8, |_| panic!()).unwrap();
    let before = snapshot(&cache);
    for key in [0, 1] {
        assert_eq!(
            cache.assess_admission(&key, 8, 0).unwrap(),
            Admission::RequiresFlush
        );
        assert_eq!(snapshot(&cache), before);
    }
    cache.complete(&lease, true).unwrap();
    for key in [0, 1] {
        assert_eq!(cache.assess_admission(&key, 8, 0).unwrap(), Admission::Fits);
    }
}

#[test]
fn assessment_counts_total_scratch_once_and_never_refunds_active_payloads() {
    let mut cache = cache(3, 32, true);
    cache.scratch(4, |_| panic!()).unwrap();
    let pending = cache.acquire(0, 8, |_| panic!()).unwrap();
    idle(&mut cache, 1, 8);
    let hit = cache.acquire(1, 8, |_| panic!()).unwrap();
    let before = snapshot(&cache);
    assert_eq!(cache.assess_admission(&2, 8, 8).unwrap(), Admission::Fits);
    assert_eq!(
        cache.assess_admission(&2, 8, 9).unwrap(),
        Admission::RequiresFlush
    );
    assert_eq!(
        cache.assess_admission(&2, 8, 25).unwrap(),
        Admission::Impossible
    );
    assert_eq!(snapshot(&cache), before);
    cache.scratch(8, |_| panic!()).unwrap();
    let next = cache.acquire(2, 8, |_| panic!()).unwrap();
    assert_eq!(cache.stats.charged_bytes(), 32);
    cache
        .complete_batch(&[(&pending, true), (&hit, true), (&next, true)])
        .unwrap();
    consistent(&cache);
}

#[test]
fn assessment_protects_hit_payload_and_can_assess_explicit_scratch_reduction() {
    let mut cache = cache(2, 24, true);
    idle(&mut cache, 0, 8);
    let pending = cache.acquire(1, 8, |_| panic!()).unwrap();
    cache.scratch(8, |_| panic!()).unwrap();
    let before = snapshot(&cache);
    assert_eq!(cache.assess_admission(&0, 8, 8).unwrap(), Admission::Fits);
    assert_eq!(
        cache.assess_admission(&0, 8, 9).unwrap(),
        Admission::RequiresFlush
    );
    assert_eq!(
        cache.assess_admission(&2, 24, 0).unwrap(),
        Admission::RequiresFlush
    );
    assert_eq!(snapshot(&cache), before);
    cache.complete(&pending, true).unwrap();
    // Scratch may be reduced only after the caller releases the covered work.
    assert_eq!(cache.assess_admission(&2, 24, 0).unwrap(), Admission::Fits);
    cache.scratch(0, |_| panic!()).unwrap();
    let next = cache.acquire(2, 24, |_| {}).unwrap();
    cache.complete(&next, true).unwrap();
    consistent(&cache);
}

#[test]
fn assessment_binding_health_epoch_and_arithmetic_rejections_do_not_mutate() {
    for diagnostics in [false, true] {
        let mut cache = cache(2, 16, diagnostics);
        idle(&mut cache, 0, 8);
        let before = snapshot(&cache);
        assert!(cache.assess_admission(&0, 7, 0).is_err());
        assert!(cache.assess_admission(&16, 8, 0).is_err());
        for (bytes, scratch) in [(usize::MAX, 1), (1, usize::MAX), (17, 0), (0, 17)] {
            assert_eq!(
                cache.assess_admission(&1, bytes, scratch).unwrap(),
                Admission::Impossible
            );
            assert_eq!(snapshot(&cache), before);
        }
        cache.clock = u64::MAX;
        let before = snapshot(&cache);
        assert!(cache.assess_admission(&0, 8, 0).is_err());
        assert!(cache.assess_admission(&1, 8, 0).is_err());
        assert_eq!(snapshot(&cache), before);
        cache.quarantine();
        let before = snapshot(&cache);
        assert!(cache.assess_admission(&0, 8, 0).is_err());
        assert!(cache.assess_admission(&1, usize::MAX, 1).is_err());
        assert_eq!(snapshot(&cache), before);
    }
}

#[test]
fn assessment_handles_maximum_budget_and_counter_overflow_without_wrapping() {
    let mut cache = cache(3, usize::MAX, true);
    cache.scratch(1, |_| panic!()).unwrap();
    let active = cache.acquire(0, usize::MAX - 2, |_| panic!()).unwrap();
    cache.stats.misses = u64::MAX;
    cache.stats.hits = u64::MAX;
    cache.stats.admission_stalls = u64::MAX;
    let before = snapshot(&cache);
    assert_eq!(cache.assess_admission(&1, 1, 1).unwrap(), Admission::Fits);
    assert_eq!(
        cache.assess_admission(&1, 2, 1).unwrap(),
        Admission::RequiresFlush
    );
    assert_eq!(
        cache.assess_admission(&1, usize::MAX, 1).unwrap(),
        Admission::Impossible
    );
    assert_eq!(snapshot(&cache), before);
    let next = cache.acquire(1, 1, |_| panic!()).unwrap();
    assert_eq!(cache.stats.charged_bytes(), usize::MAX);
    assert!(cache.stats.counters_overflowed);
    cache
        .complete_batch(&[(&active, true), (&next, true)])
        .unwrap();
    let before = snapshot(&cache);
    assert_eq!(cache.assess_admission(&2, 1, 1).unwrap(), Admission::Fits);
    assert_eq!(snapshot(&cache), before);
    consistent(&cache);
}

#[test]
fn assessment_does_not_clone_even_the_next_key() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CLONES: AtomicUsize = AtomicUsize::new(0);
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Key(u64);
    impl Clone for Key {
        fn clone(&self) -> Self {
            CLONES.fetch_add(1, Ordering::Relaxed);
            Self(self.0)
        }
    }
    let mut cache = Cache::new(ExpertCacheLimits {
        max_experts: 1,
        max_bytes: 8,
    })
    .unwrap();
    let lease = cache.acquire(Key(0), 8, |_| panic!()).unwrap();
    cache.complete(&lease, true).unwrap();
    let before = CLONES.load(Ordering::Relaxed);
    assert_eq!(
        cache.assess_admission(&Key(0), 8, 0).unwrap(),
        Admission::Fits
    );
    assert_eq!(
        cache.assess_admission(&Key(1), 8, 0).unwrap(),
        Admission::Fits
    );
    assert_eq!(CLONES.load(Ordering::Relaxed), before);
}

#[test]
fn assessment_is_advisory_and_must_be_repeated_after_owner_mutation() {
    let mut cache = cache(1, 16, true);
    assert_eq!(cache.assess_admission(&0, 8, 0).unwrap(), Admission::Fits);
    let active = cache.acquire(1, 8, |_| panic!()).unwrap();
    let before = snapshot(&cache);
    assert_eq!(
        cache.assess_admission(&0, 8, 0).unwrap(),
        Admission::RequiresFlush
    );
    assert_eq!(snapshot(&cache), before);
    cache.complete(&active, true).unwrap();
    assert_eq!(cache.assess_admission(&0, 8, 0).unwrap(), Admission::Fits);
}

#[test]
fn assessment_fits_matches_actual_admission_over_small_states() {
    // Enumerate idle, consuming and uploading entries, both budgets, hits and
    // misses, and absolute scratch totals. Keep each successful query honest by
    // applying the existing scratch/acquire operations without policy changes.
    let mut fits = 0;
    let mut flushes = 0;
    let mut impossible = 0;
    for states in 0..27 {
        for slots in 3..=4 {
            for limit in [12, 16, 24] {
                for key in 0..=3 {
                    for scratch in [0, 4, 9, 25] {
                        let mut cache = cache(slots, limit, false);
                        let mut bits = states;
                        for k in 0..3 {
                            let state = bits % 3;
                            bits /= 3;
                            let lease = cache.acquire(k, 4, |_| panic!()).unwrap();
                            if state != 0 {
                                cache.complete(&lease, true).unwrap();
                                if state == 2 {
                                    cache.acquire(k, 4, |_| panic!()).unwrap();
                                }
                            }
                        }
                        let before = snapshot(&cache);
                        let assessment = cache.assess_admission(&key, 4, scratch).unwrap();
                        assert_eq!(snapshot(&cache), before);
                        match assessment {
                            Admission::Fits => {
                                fits += 1;
                                cache.scratch(scratch, |_| {}).unwrap();
                                let lease = cache.acquire(key, 4, |_| {}).unwrap();
                                cache.complete(&lease, true).unwrap();
                            }
                            Admission::RequiresFlush => {
                                flushes += 1;
                                let leases: Vec<_> = cache
                                    .entries
                                    .iter()
                                    .filter(|(_, e)| e.state != State::Idle)
                                    .map(|(key, e)| Lease {
                                        owner: cache.owner.clone(),
                                        key: *key,
                                        epoch: e.epoch,
                                    })
                                    .collect();
                                let completions: Vec<_> =
                                    leases.iter().map(|l| (l, true)).collect();
                                cache.complete_batch(&completions).unwrap();
                                assert_eq!(
                                    cache.assess_admission(&key, 4, scratch).unwrap(),
                                    Admission::Fits
                                );
                            }
                            Admission::Impossible => {
                                impossible += 1;
                                assert!(4usize.checked_add(scratch).is_none_or(|n| n > limit));
                            }
                        }
                        consistent(&cache);
                    }
                }
            }
        }
    }
    assert!(fits > 0 && flushes > 0 && impossible > 0);
}

#[test]
fn batch_success_has_identical_accounting_and_lru_to_single_completion() {
    for diagnostics in [false, true] {
        let mut batch = cache(5, 48, diagnostics);
        let mut single = cache(5, 48, diagnostics);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for key in 0..4 {
            if key >= 2 {
                idle(&mut batch, key, 8);
                idle(&mut single, key, 8);
            }
            a.push(batch.acquire(key, 8, |_| panic!()).unwrap());
            b.push(single.acquire(key, 8, |_| panic!()).unwrap());
        }
        batch.scratch(8, |_| panic!()).unwrap();
        single.scratch(8, |_| panic!()).unwrap();
        // Upload success, upload failure, hit success, hit failure in reverse
        // completion order must still use acquisition epochs for LRU.
        let completions = [(&a[3], false), (&a[2], true), (&a[1], false), (&a[0], true)];
        batch.complete_batch(&completions).unwrap();
        for (i, installed) in [(3, false), (2, true), (1, false), (0, true)] {
            single.complete(&b[i], installed).unwrap();
        }
        assert_eq!(snapshot(&batch), snapshot(&single));
        assert_eq!(batch.stats.uploads, 3);
        assert_eq!(batch.stats.resident_bytes, 16);
        assert_eq!(batch.stats.pending_upload_bytes, 0);
        assert_eq!(batch.stats.scratch_bytes, 8);
        let mut victims = Vec::new();
        batch.evict_one_completed(|k| victims.push(*k)).unwrap();
        batch.evict_one_completed(|k| victims.push(*k)).unwrap();
        assert_eq!(victims, [0, 2]);
        consistent(&batch);
    }
}

#[test]
fn batch_stale_missing_idle_and_foreign_members_leave_everything_unchanged() {
    for invalid in 0..4 {
        for invalid_first in [false, true] {
            let mut cache = cache(4, 32, true);
            let old = idle(&mut cache, 0, 8);
            let current = cache.acquire(0, 8, |_| panic!()).unwrap();
            let pending = cache.acquire(1, 8, |_| panic!()).unwrap();
            let completed = idle(&mut cache, 2, 8);
            let removed = cache.acquire(3, 8, |_| panic!()).unwrap();
            cache.complete(&removed, false).unwrap();
            let mut foreign_cache = self::cache(4, 32, true);
            let foreign = foreign_cache.acquire(0, 8, |_| panic!()).unwrap();
            let bad = match invalid {
                0 => &old,
                1 => &removed,
                2 => &completed,
                _ => &foreign,
            };
            let before = snapshot(&cache);
            let mut members = vec![(&pending, true), (&current, false)];
            if invalid_first {
                members.insert(0, (bad, true));
            } else {
                members.push((bad, true));
            }
            assert!(cache.complete_batch(&members).is_err());
            assert_eq!(snapshot(&cache), before);
            consistent(&cache);
        }
    }
}

#[test]
fn batch_duplicate_and_conflicting_outcomes_are_all_or_none() {
    for outcomes in [(true, true), (true, false), (false, true), (false, false)] {
        let mut cache = cache(3, 24, true);
        let a = cache.acquire(0, 8, |_| panic!()).unwrap();
        let b = cache.acquire(1, 8, |_| panic!()).unwrap();
        let duplicate = a.clone();
        let before = snapshot(&cache);
        assert!(
            cache
                .complete_batch(&[(&b, true), (&a, outcomes.0), (&duplicate, outcomes.1)])
                .is_err()
        );
        assert_eq!(snapshot(&cache), before);
        cache.complete_batch(&[(&b, true), (&a, true)]).unwrap();
        let before = snapshot(&cache);
        assert!(cache.complete_batch(&[(&a, true), (&b, true)]).is_err());
        assert_eq!(snapshot(&cache), before);
        consistent(&cache);
    }
}

#[test]
fn batch_unknown_is_sticky_with_live_and_idle_entries_and_late_proofs() {
    let mut cache = cache(4, 32, true);
    cache.scratch(8, |_| panic!()).unwrap();
    let uploading = cache.acquire(0, 8, |_| panic!()).unwrap();
    idle(&mut cache, 1, 8);
    let consuming = cache.acquire(1, 8, |_| panic!()).unwrap();
    let completed = idle(&mut cache, 2, 8);
    cache.quarantine();
    let before = snapshot(&cache);
    assert_eq!(cache.stats.unknown_quarantine_bytes, 32);
    for installed in [false, true] {
        assert!(
            cache
                .complete_batch(&[(&uploading, installed), (&consuming, installed)])
                .is_err()
        );
        assert!(cache.complete(&uploading, installed).is_err());
        assert!(cache.retire_completed(&completed).is_err());
        assert_eq!(snapshot(&cache), before);
    }
    assert!(cache.complete_batch(&[]).is_err());
    cache.quarantine();
    assert_eq!(snapshot(&cache), before);
    consistent(&cache);
}

#[test]
fn batch_empty_is_noop_and_counter_overflow_invalidates_without_partial_failure() {
    let mut cache = cache(3, 24, true);
    let a = cache.acquire(0, 8, |_| panic!()).unwrap();
    let b = cache.acquire(1, 8, |_| panic!()).unwrap();
    cache.stats.uploads = u64::MAX - 1;
    let before = snapshot(&cache);
    cache.complete_batch(&[]).unwrap();
    assert_eq!(snapshot(&cache), before);
    cache.complete_batch(&[(&a, true), (&b, true)]).unwrap();
    assert_eq!(cache.stats.uploads, u64::MAX);
    assert!(cache.stats.counters_overflowed);
    assert_eq!(cache.stats.resident_experts, 2);
    assert_eq!(cache.stats.pending_upload_bytes, 0);
    consistent(&cache);
}

#[test]
fn batch_completion_at_last_epoch_does_not_renew_or_wrap_it() {
    let mut cache = cache(2, 16, true);
    cache.clock = u64::MAX - 2;
    let a = cache.acquire(0, 8, |_| panic!()).unwrap();
    let b = cache.acquire(1, 8, |_| panic!()).unwrap();
    cache.complete_batch(&[(&b, true), (&a, true)]).unwrap();
    assert_eq!(cache.clock, u64::MAX);
    let before = snapshot(&cache);
    assert!(cache.assess_admission(&2, 8, 0).is_err());
    assert!(cache.acquire(2, 8, |_| panic!()).is_err());
    assert_eq!(snapshot(&cache), before);
    assert_eq!(cache.retire_completed(&b).unwrap(), 1);
    assert_eq!(cache.retire_completed(&a).unwrap(), 0);
    assert_eq!(cache.clock, u64::MAX);
    consistent(&cache);
}

#[test]
fn retirement_needs_actual_successful_completion_and_preserves_upload_history() {
    for diagnostics in [false, true] {
        let mut cache = cache(1, 16, diagnostics);
        cache.scratch(8, |_| panic!()).unwrap();
        let lease = cache.acquire(0, 8, |_| panic!()).unwrap();
        let before = snapshot(&cache);
        assert!(cache.retire_completed(&lease).is_err());
        assert_eq!(snapshot(&cache), before);
        cache.complete(&lease, true).unwrap();
        let before = cache.stats();
        assert_eq!(cache.retire_completed(&lease).unwrap(), lease.key);
        assert_eq!(cache.stats.resident_bytes, 0);
        assert_eq!(cache.stats.resident_experts, 0);
        assert_eq!(cache.stats.pending_upload_bytes, 0);
        assert_eq!(cache.stats.uploads, 1);
        assert_eq!(cache.stats.scratch_bytes, before.scratch_bytes);
        assert_eq!(cache.stats.peak_bytes, before.peak_bytes);
        assert_eq!(cache.stats.evictions, before.evictions + 1);
        assert_eq!(
            cache.stats.entry_limit_evictions,
            before.entry_limit_evictions
        );
        assert_eq!(
            cache.stats.byte_limit_evictions,
            before.byte_limit_evictions
        );
        assert_eq!(
            cache.stats.scratch_limit_evictions,
            before.scratch_limit_evictions
        );
        assert_eq!(cache.stats.misses, before.misses);
        let before = snapshot(&cache);
        assert!(cache.retire_completed(&lease).is_err());
        assert_eq!(snapshot(&cache), before);
        consistent(&cache);
    }
}

#[test]
fn retirement_never_releases_consuming_uploading_stale_or_failed_bindings() {
    let mut cache = cache(2, 16, true);
    let old = idle(&mut cache, 0, 8);
    let consuming = cache.acquire(0, 8, |_| panic!()).unwrap();
    let uploading = cache.acquire(1, 8, |_| panic!()).unwrap();
    let before = snapshot(&cache);
    for lease in [&old, &consuming, &uploading] {
        assert!(cache.retire_completed(lease).is_err());
        assert_eq!(snapshot(&cache), before);
    }
    cache
        .complete_batch(&[(&consuming, true), (&uploading, false)])
        .unwrap();
    let before = snapshot(&cache);
    for lease in [&old, &uploading] {
        assert!(cache.retire_completed(lease).is_err());
        assert_eq!(snapshot(&cache), before);
    }
    assert_eq!(cache.retire_completed(&consuming).unwrap(), 0);
    let reload = cache.acquire(0, 8, |_| panic!()).unwrap();
    cache.complete(&reload, true).unwrap();
    let before = snapshot(&cache);
    assert!(cache.retire_completed(&consuming).is_err());
    assert_eq!(snapshot(&cache), before);
    assert_eq!(cache.retire_completed(&reload).unwrap(), 0);
    consistent(&cache);
}

#[test]
fn foreign_owner_proofs_cannot_complete_or_retire_equal_key_and_epoch() {
    let mut a = cache(1, 8, true);
    let mut b = cache(1, 8, true);
    let x = a.acquire(0, 8, |_| panic!()).unwrap();
    let y = b.acquire(0, 8, |_| panic!()).unwrap();
    assert_eq!((&x.key, x.epoch), (&y.key, y.epoch));
    assert_ne!(x, y);
    let before = snapshot(&b);
    assert!(b.complete(&x, true).is_err());
    assert!(b.complete_batch(&[(&x, true)]).is_err());
    assert_eq!(snapshot(&b), before);
    b.complete(&y, true).unwrap();
    let before = snapshot(&b);
    assert!(b.retire_completed(&x).is_err());
    assert_eq!(snapshot(&b), before);
    a.complete(&x, true).unwrap();
    assert_eq!(a.retire_completed(&x).unwrap(), 0);
    assert_eq!(b.retire_completed(&y).unwrap(), 0);
    consistent(&a);
    consistent(&b);
}

#[test]
fn retirement_returns_exact_image_precision_source_and_role_binding() {
    let original = ExpertCacheKey {
        image_generation: 7,
        numeric_precision: Some(super::super::NumericFp8Precision::W8A16Bf16F32Accumulate),
        layer: 3,
        expert: 2,
        parameters: [
            (ParameterId::new(1), TensorRole::RoutedExpertGate),
            (ParameterId::new(2), TensorRole::RoutedExpertUp),
            (ParameterId::new(3), TensorRole::RoutedExpertDown),
        ],
    };
    let mut keys = vec![original.clone(); 7];
    keys[1].image_generation += 1;
    keys[2].numeric_precision = None;
    keys[3].layer += 1;
    keys[4].expert += 1;
    keys[5].parameters[0].0 = ParameterId::new(4);
    keys[6].parameters[0].1 = TensorRole::RoutedExpertUp;
    let mut cache = Cache::new(ExpertCacheLimits {
        max_experts: 7,
        max_bytes: 56,
    })
    .unwrap();
    let leases: Vec<_> = keys
        .iter()
        .map(|k| cache.acquire(k.clone(), 8, |_| panic!()).unwrap())
        .collect();
    let completions: Vec<_> = leases.iter().map(|l| (l, true)).collect();
    cache.complete_batch(&completions).unwrap();
    let mut physical: BTreeSet<_> = keys.iter().cloned().collect();
    for (key, lease) in keys.iter().zip(&leases).rev() {
        let returned = cache.retire_completed(lease).unwrap();
        assert_eq!(&returned, key);
        assert!(physical.remove(&returned));
        assert_eq!(cache.entries.len(), physical.len());
        for remaining in &physical {
            assert!(cache.entries.contains_key(remaining));
        }
    }
    assert!(physical.is_empty());
    assert_eq!(cache.stats.uploads, 7);
    assert_eq!(cache.stats.evictions, 7);
    assert_eq!(cache.stats.charged_bytes(), 0);
}

#[test]
fn retirement_counter_overflow_stays_invalid_without_touching_other_bindings() {
    let mut cache = cache(2, 16, true);
    let a = idle(&mut cache, 0, 8);
    let b = idle(&mut cache, 1, 8);
    cache.stats.evictions = u64::MAX;
    assert_eq!(cache.retire_completed(&a).unwrap(), 0);
    assert!(cache.stats.counters_overflowed);
    assert_eq!(cache.stats.evictions, u64::MAX);
    assert_eq!(cache.stats.uploads, 2);
    assert_eq!(cache.entries.get(&1).unwrap().epoch, b.epoch);
    assert_eq!(cache.retire_completed(&b).unwrap(), 1);
    assert!(cache.stats.counters_overflowed);
    consistent(&cache);
}
