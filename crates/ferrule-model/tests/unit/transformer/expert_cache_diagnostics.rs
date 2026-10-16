use super::*;
use crate::resource_observability::{ExpertCacheObservation, ResourceEvidence};

fn cache(entries: usize, bytes: usize, enabled: bool) -> Cache<u64> {
    let mut cache = Cache::new(ExpertCacheLimits {
        max_experts: entries,
        max_bytes: bytes,
    })
    .unwrap();
    cache.stats.diagnostics_enabled = enabled;
    cache
}

fn complete(cache: &mut Cache<u64>, key: u64, bytes: usize) {
    let lease = cache.acquire(key, bytes, |_| {}).unwrap();
    cache.complete(&lease, true).unwrap();
}

fn state<K: Ord + Clone, G: Ord + Clone>(cache: &Cache<K, G>) -> Vec<(K, usize, u64, State)> {
    cache
        .entries
        .iter()
        .map(|(k, e)| (k.clone(), e.bytes, e.epoch, e.state))
        .collect()
}

#[test]
fn diagnostics_on_off_preserve_victims_leases_and_completion_order() {
    for seed in [1u64, 17, 991] {
        let mut on = cache(4, 36, true);
        let mut off = cache(4, 36, false);
        let mut leases = Vec::new();
        let mut rng = seed;
        for _ in 0..3000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let mut a = Vec::new();
            let mut b = Vec::new();
            match rng % 6 {
                0..=2 => {
                    let key = (rng >> 8) % 9;
                    let x = on.acquire(key, 8, |k| a.push(*k));
                    let y = off.acquire(key, 8, |k| b.push(*k));
                    match (x, y) {
                        (Ok(x), Ok(y)) => {
                            assert_eq!((&x.key, x.epoch), (&y.key, y.epoch));
                            leases.push((x, y));
                        }
                        (Err(x), Err(y)) => assert_eq!(x.to_string(), y.to_string()),
                        pair => panic!("different acquisitions: {pair:?}"),
                    }
                }
                3 if !leases.is_empty() => {
                    let (x, y) = &leases[(rng as usize >> 16) % leases.len()];
                    let installed = rng & 128 == 0;
                    assert_eq!(
                        on.complete(x, installed).map_err(|e| e.to_string()),
                        off.complete(y, installed).map_err(|e| e.to_string())
                    );
                }
                4 => {
                    let bytes = (rng as usize >> 16) % 12;
                    assert_eq!(
                        on.scratch(bytes, |k| a.push(*k)).map_err(|e| e.to_string()),
                        off.scratch(bytes, |k| b.push(*k))
                            .map_err(|e| e.to_string())
                    );
                }
                _ => {
                    assert_eq!(
                        on.evict_one_completed(|k| a.push(*k)).unwrap(),
                        off.evict_one_completed(|k| b.push(*k)).unwrap()
                    );
                }
            }
            assert_eq!(a, b);
            assert_eq!(state(&on), state(&off));
            assert_eq!(on.clock, off.clock);
            let mut stats = on.stats();
            stats.diagnostics_enabled = false;
            stats.misses = 0;
            stats.entry_limit_evictions = 0;
            stats.byte_limit_evictions = 0;
            stats.scratch_limit_evictions = 0;
            stats.admission_stalls = 0;
            assert_eq!(stats, off.stats());
        }
    }
}

#[test]
fn pressure_classifies_actual_evictions_not_new_leases_or_upload_proofs() {
    let mut slots = cache(1, 40, true);
    complete(&mut slots, 0, 8);
    complete(&mut slots, 1, 8);
    complete(&mut slots, 1, 8);
    assert_eq!(slots.stats().misses, 2);
    assert_eq!(slots.stats().hits, 1);
    assert_eq!(slots.stats().entry_limit_evictions, 1);
    assert_eq!(slots.stats().byte_limit_evictions, 0);

    let mut bytes = cache(5, 24, true);
    for key in 0..4 {
        complete(&mut bytes, key, 8);
    }
    assert_eq!(bytes.stats().byte_limit_evictions, 1);
    assert_eq!(bytes.stats().entry_limit_evictions, 0);
    bytes.scratch(9, |_| {}).unwrap();
    assert_eq!(bytes.stats().byte_limit_evictions, 3);
    assert_eq!(bytes.stats().scratch_limit_evictions, 2);
    let lease = bytes.acquire(4, 8, |_| {}).unwrap();
    assert_eq!(bytes.stats().misses, 5);
    assert_eq!(bytes.stats().uploads, 4);
    assert_eq!(bytes.stats().pending_upload_bytes, 8);
    bytes.complete(&lease, false).unwrap();
    assert_eq!(bytes.stats().uploads, 4);
    assert_eq!(bytes.stats().misses, 5);

    let mut both = cache(1, 8, true);
    complete(&mut both, 0, 8);
    complete(&mut both, 1, 8);
    assert_eq!(both.stats().evictions, 1);
    assert_eq!(both.stats().entry_limit_evictions, 1);
    assert_eq!(both.stats().byte_limit_evictions, 1);
}

#[test]
fn pending_and_unknown_are_not_hits_cold_misses_or_release_proofs() {
    let mut cache = cache(1, 16, true);
    let lease = cache.acquire(0, 8, |_| panic!()).unwrap();
    assert!(cache.acquire(1, 8, |_| panic!()).is_err());
    assert_eq!(cache.stats().admission_stalls, 1);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 0);
    cache.quarantine();
    let before = (cache.stats(), cache.clock, state(&cache));
    let observation = ExpertCacheObservation::from(cache.stats());
    let ResourceEvidence::Observed(d) = observation.diagnostics else {
        panic!("expected historical observations, not completion authority");
    };
    assert_eq!(d.cold_misses, ResourceEvidence::Unobserved);
    assert_eq!(d.evicted_misses, ResourceEvidence::Unobserved);
    assert_eq!(d.evicted_selected, ResourceEvidence::Unobserved);
    assert_eq!(d.evicted_future_hint, ResourceEvidence::Unobserved);
    assert_eq!(d.history_bytes, 0);
    assert!(cache.complete(&lease, true).is_err());
    assert!(cache.acquire(0, 8, |_| panic!()).is_err());
    assert_eq!(before, (cache.stats(), cache.clock, state(&cache)));
}

#[test]
fn identity_registration_and_epoch_rejections_have_zero_mutation() {
    let key = ExpertCacheKey {
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
    let mut cache = Cache::with_groups(
        ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 8,
        },
        [(key.clone(), ())],
    )
    .unwrap();
    cache.stats.diagnostics_enabled = true;
    let first = cache.acquire(key.clone(), 8, |_| panic!()).unwrap();
    cache.complete(&first, true).unwrap();
    let before = (cache.stats(), cache.clock, state(&cache));
    for field in 0..6 {
        let mut wrong = key.clone();
        match field {
            0 => wrong.image_generation += 1,
            1 => wrong.numeric_precision = None,
            2 => wrong.layer += 1,
            3 => wrong.expert += 1,
            4 => wrong.parameters[0].0 = ParameterId::new(4),
            _ => wrong.parameters[0].1 = TensorRole::RoutedExpertUp,
        }
        assert!(cache.assess_admission(&wrong, 8, 0).is_err());
        assert!(cache.acquire(wrong, 8, |_| panic!()).is_err());
        assert_eq!(before, (cache.stats(), cache.clock, state(&cache)));
    }
    assert!(cache.acquire(key.clone(), 7, |_| panic!()).is_err());
    assert_eq!(before, (cache.stats(), cache.clock, state(&cache)));
    let next = cache.acquire(key.clone(), 8, |_| panic!()).unwrap();
    let before = (cache.stats(), cache.clock, state(&cache));
    assert!(cache.complete(&first, false).is_err());
    assert!(cache.acquire(key.clone(), 8, |_| panic!()).is_err());
    assert_eq!(before, (cache.stats(), cache.clock, state(&cache)));
    cache.complete(&next, true).unwrap();
    cache.clock = u64::MAX;
    let before = (cache.stats(), cache.clock, state(&cache));
    assert!(cache.acquire(key, 8, |_| panic!()).is_err());
    assert_eq!(before, (cache.stats(), cache.clock, state(&cache)));
}

#[test]
fn counter_overflow_invalidates_observations_without_changing_policy() {
    for counter in 0..4 {
        let mut cache = cache(1, 8, true);
        complete(&mut cache, 0, 8);
        match counter {
            0 => cache.stats.hits = u64::MAX,
            1 => cache.stats.uploads = u64::MAX,
            2 => cache.stats.evictions = u64::MAX,
            _ => cache.stats.misses = u64::MAX,
        }
        complete(&mut cache, u64::from(counter != 0), 8);
        assert!(cache.stats().counters_overflowed);
        assert_eq!(cache.stats().resident_bytes, 8);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(
            ExpertCacheObservation::from(cache.stats()).diagnostics,
            ResourceEvidence::Unknown
        );
        complete(&mut cache, 2, 8);
        assert!(cache.stats().counters_overflowed);
    }
    assert_eq!(
        ExpertCacheObservation::from(cache(1, 8, false).stats()).diagnostics,
        ResourceEvidence::Unobserved
    );
}
