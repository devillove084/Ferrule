use super::*;
use std::fmt::Debug;

fn limits() -> ExpertCacheLimits {
    ExpertCacheLimits {
        max_experts: 12,
        max_bytes: 48,
    }
}
fn grouped() -> Cache<u64, u64> {
    Cache::with_groups(limits(), (0..64).map(|k| (k, k % 3))).unwrap()
}

#[derive(Default, Debug)]
struct Work {
    // Full-table entry visits, not B-tree node comparisons.
    scanned_entries: usize,
    descriptor_lookups: usize,
    selected: usize,
}

// Retain the old evaluator verbatim in shape: sorted entries, Idle + predicate,
// then minimum epoch. All actual transitions still belong to Cache.
fn linear_evict<K: Ord + Clone, G: PartialEq, C: Ord + Clone>(
    cache: &mut Cache<K, C>,
    descriptors: &BTreeMap<K, G>,
    group: &G,
    work: &mut Work,
    evict: impl FnMut(&K),
) -> Result<bool> {
    cache.healthy()?;
    let victim = cache
        .entries
        .iter()
        .inspect(|_| work.scanned_entries += 1)
        .filter(|(key, entry)| {
            entry.state == State::Idle && {
                work.descriptor_lookups += 1;
                descriptors.get(key) == Some(group)
            }
        })
        .min_by_key(|(_, entry)| entry.epoch)
        .map(|(key, _)| key.clone());
    if let Some(key) = victim {
        work.selected += 1;
        cache.evict_completed_key(key, evict);
        Ok(true)
    } else {
        Ok(false)
    }
}

fn same_result<T: PartialEq + Debug>(a: Result<T>, b: Result<T>) -> Option<T> {
    match (a, b) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b);
            Some(a)
        }
        (Err(a), Err(b)) => {
            assert_eq!(a.to_string(), b.to_string());
            None
        }
        pair => panic!("different results: {pair:?}"),
    }
}

fn consistent(cache: &Cache<u64, u64>) {
    let groups = cache.groups.as_ref().unwrap();
    let actual: BTreeMap<_, _> = groups
        .completed
        .iter()
        .filter(|(_, set)| !set.is_empty())
        .map(|(g, set)| (*g, set.clone()))
        .collect();
    let mut expected = BTreeMap::<_, BTreeSet<_>>::new();
    if !cache.stats().quarantined {
        for (key, entry) in &cache.entries {
            assert_eq!(entry.group.as_ref(), groups.registration.get(key));
            if entry.state == State::Idle {
                expected
                    .entry(entry.group.unwrap())
                    .or_default()
                    .insert((entry.epoch, *key));
            }
        }
    }
    assert_eq!(actual, expected);
}

fn equivalent(index: &Cache<u64, u64>, reference: &Cache<u64>) {
    assert_eq!(index.stats(), reference.stats());
    assert_eq!(index.clock, reference.clock);
    let state = |key: &u64, bytes, epoch, state| (*key, bytes, epoch, state);
    assert_eq!(
        index
            .entries
            .iter()
            .map(|(k, e)| state(k, e.bytes, e.epoch, e.state))
            .collect::<Vec<_>>(),
        reference
            .entries
            .iter()
            .map(|(k, e)| state(k, e.bytes, e.epoch, e.state))
            .collect::<Vec<_>>()
    );
    consistent(index);
}

#[test]
fn randomized_operations_match_old_linear_victims_credits_and_leases() {
    let descriptors = (0..64).map(|k| (k, k % 3)).collect();
    for seed in [1u64, 17, 73, 111, 991, 12345, 54321, u64::MAX] {
        let mut rng = seed;
        let mut index = grouped();
        let mut reference = Cache::new(limits()).unwrap();
        let mut leases = Vec::new();
        let mut work = Work::default();
        for step in 0..5000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let key = (rng >> 16) % 64;
            let group = (rng >> 32) % 4; // Includes an empty group.
            let mut a = Vec::new();
            let mut b = Vec::new();
            match rng % 10 {
                0..=3 => {
                    let bytes = if rng & 256 == 0 {
                        1 + key as usize % 3
                    } else {
                        49
                    };
                    if let Some(lease) = same_result(
                        index.acquire(key, bytes, |k| a.push(*k)),
                        reference.acquire(key, bytes, |k| b.push(*k)),
                    ) {
                        leases.push(lease);
                    }
                }
                4 | 5 if !leases.is_empty() => {
                    // Keep historical leases to exercise stale and double completion.
                    let lease = &leases[(rng >> 40) as usize % leases.len()];
                    same_result(
                        index.complete(lease, rng & 128 != 0),
                        reference.complete(lease, rng & 128 != 0),
                    );
                }
                6 => {
                    same_result(
                        index.evict_one_completed_in_group(&group, |k| a.push(*k)),
                        linear_evict(&mut reference, &descriptors, &group, &mut work, |k| {
                            b.push(*k)
                        }),
                    );
                }
                7 => {
                    same_result(
                        index.evict_one_completed(|k| a.push(*k)),
                        reference.evict_one_completed(|k| b.push(*k)),
                    );
                }
                8 => {
                    same_result(
                        index.scratch(key as usize, |k| a.push(*k)),
                        reference.scratch(key as usize, |k| b.push(*k)),
                    );
                }
                _ => {
                    same_result(
                        index.preflight(key as usize, 4),
                        reference.preflight(key as usize, 4),
                    );
                }
            }
            assert_eq!(a, b, "seed={seed}, step={step}");
            equivalent(&index, &reference);
        }
        index.quarantine();
        reference.quarantine();
        for lease in &leases {
            same_result(index.complete(lease, true), reference.complete(lease, true));
            same_result(
                index.complete(lease, false),
                reference.complete(lease, false),
            );
        }
        for group in 0..4 {
            same_result(
                index.evict_one_completed_in_group(&group, |_| panic!()),
                linear_evict(
                    &mut reference,
                    &descriptors,
                    &group,
                    &mut work,
                    |_| panic!(),
                ),
            );
        }
        same_result(
            index.acquire(1, 1, |_| panic!()),
            reference.acquire(1, 1, |_| panic!()),
        );
        equivalent(&index, &reference);
    }
}

#[test]
fn registration_is_frozen_duplicate_or_unregistered_keys_cannot_rebind() {
    for second in [0, 1] {
        assert!(Cache::with_groups(limits(), [(7, 0), (7, second)]).is_err());
    }
    let mut cache = grouped();
    let lease = cache.acquire(1, 3, |_| panic!()).unwrap();
    cache.complete(&lease, true).unwrap();
    let before = cache.stats();
    assert!(cache.acquire(64, 3, |_| panic!()).is_err());
    assert!(
        !cache
            .evict_one_completed_in_group(&0, |_| panic!())
            .unwrap()
    );
    assert!(
        !cache
            .evict_one_completed_in_group(&99, |_| panic!())
            .unwrap()
    );
    assert_eq!(cache.stats(), before);
    assert!(
        cache
            .evict_one_completed_in_group(&1, |k| assert_eq!(*k, 1))
            .unwrap()
    );
    consistent(&cache);
    let mut plain = Cache::<u64>::new(limits()).unwrap();
    assert!(plain.groups.is_none());
    assert!(
        plain
            .evict_one_completed_in_group(&(), |_| panic!())
            .is_err()
    );
}

#[test]
fn completion_order_ties_epoch_overflow_and_live_leases_preserve_index() {
    let mut cache = grouped();
    // Force ties (normally unreachable) and reverse completion order. The old
    // min_by_key selects the smallest key, not the first inserted/completed one.
    let mut leases = Vec::new();
    for key in [6, 0, 3] {
        cache.clock = 10;
        leases.push(cache.acquire(key, 2, |_| panic!()).unwrap());
    }
    assert!(
        !cache
            .evict_one_completed_in_group(&0, |_| panic!())
            .unwrap()
    );
    for lease in leases.iter().rev() {
        cache.complete(lease, true).unwrap();
    }
    let mut victims = Vec::new();
    for _ in 0..3 {
        assert!(
            cache
                .evict_one_completed_in_group(&0, |k| victims.push(*k))
                .unwrap()
        );
    }
    assert_eq!(victims, [0, 3, 6]);
    cache.clock = u64::MAX - 1;
    let last = cache.acquire(0, 2, |_| panic!()).unwrap();
    assert!(
        !cache
            .evict_one_completed_in_group(&0, |_| panic!())
            .unwrap()
    );
    let before = cache.stats();
    assert!(cache.acquire(3, 2, |_| panic!()).is_err());
    assert_eq!(cache.stats(), before);
    assert_eq!(cache.clock, u64::MAX);
    cache.complete(&last, true).unwrap();
    assert!(cache.complete(&last, true).is_err());
    let before = cache.stats();
    assert!(cache.acquire(0, 2, |_| panic!()).is_err());
    assert_eq!(cache.stats(), before);
    consistent(&cache);
    assert!(
        cache
            .evict_one_completed_in_group(&0, |k| assert_eq!(*k, 0))
            .unwrap()
    );
}

#[test]
fn cancel_consuming_and_uploading_never_publish_and_stale_proof_cannot_reload() {
    let mut cache = grouped();
    let a = cache.acquire(0, 2, |_| panic!()).unwrap();
    cache.complete(&a, false).unwrap();
    assert!(
        !cache
            .evict_one_completed_in_group(&0, |_| panic!())
            .unwrap()
    );
    let b = cache.acquire(0, 2, |_| panic!()).unwrap();
    assert!(cache.complete(&a, true).is_err());
    cache.complete(&b, true).unwrap();
    let c = cache.acquire(0, 2, |_| panic!()).unwrap();
    assert!(
        !cache
            .evict_one_completed_in_group(&0, |_| panic!())
            .unwrap()
    );
    assert!(cache.complete(&b, true).is_err());
    cache.complete(&c, false).unwrap();
    assert!(cache.complete(&c, false).is_err());
    assert_eq!(cache.stats().charged_bytes(), 0);
    consistent(&cache);
    let idle = cache.acquire(0, 2, |_| panic!()).unwrap();
    cache.complete(&idle, true).unwrap();
    let pending = cache.acquire(1, 2, |_| panic!()).unwrap();
    cache.quarantine();
    let frozen = cache.stats();
    assert!(cache.complete(&pending, true).is_err());
    assert!(
        cache
            .evict_one_completed_in_group(&0, |_| panic!())
            .is_err()
    );
    assert_eq!(cache.stats(), frozen);
    assert_eq!(frozen.unknown_quarantine_bytes, 4);
    consistent(&cache);
}

#[test]
#[ignore = "CPU-only 1155-entry/20000-descriptor/50000-miss comparison; no timing assertion"]
fn cpu_grouped_selector_benchmark() {
    use crate::transformer::hybrid_parallel::mixed_decoded::{DecodedIdentity, DecodedKey};
    use std::{sync::Arc, time::Instant};
    let keys: Vec<_> = (0..20_000)
        .map(|owner| DecodedKey {
            generation: 73,
            identity: Arc::new(DecodedIdentity {
                owner,
                geometry: [if owner % 2 == 0 { 1024 } else { 512 }, 1],
                parts: vec![],
            }),
        })
        .collect();
    let descriptors: BTreeMap<_, _> = keys
        .iter()
        .map(|k| {
            (
                k.clone(),
                (k.identity.geometry[0], k.identity.geometry[0] * 2),
            )
        })
        .collect();
    let limits = ExpertCacheLimits {
        max_experts: 1155,
        max_bytes: 1155 * 2048,
    };
    let mut outputs = Vec::new();
    for indexed in [false, true] {
        let mut cache = if indexed {
            Cache::with_groups(limits, descriptors.iter().map(|(k, g)| (k.clone(), *g))).unwrap()
        } else {
            Cache::empty(limits, None).unwrap()
        };
        for key in &keys[..1155] {
            let lease = cache
                .acquire(key.clone(), descriptors[key].1, |_| panic!())
                .unwrap();
            cache.complete(&lease, true).unwrap();
        }
        let mut work = Work::default();
        let mut victims = Vec::with_capacity(50_000);
        let start = Instant::now();
        for n in 1155..51_155 {
            let key = &keys[n % keys.len()];
            let group = &descriptors[key]; // One incoming-descriptor check in both paths.
            let evicted = if indexed {
                let result = cache
                    .evict_one_completed_in_group(group, |k| victims.push(k.identity.owner))
                    .unwrap();
                work.selected += usize::from(result);
                result
            } else {
                linear_evict(&mut cache, &descriptors, group, &mut work, |k| {
                    victims.push(k.identity.owner)
                })
                .unwrap()
            };
            assert!(evicted);
            let lease = cache.acquire(key.clone(), group.1, |_| panic!()).unwrap();
            cache.complete(&lease, true).unwrap();
        }
        let elapsed = start.elapsed();
        assert_eq!(cache.stats().evictions, 50_000);
        assert_eq!(cache.stats().uploads, 51_155);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().resident_experts, 1155);
        assert_eq!(work.selected, 50_000);
        if indexed {
            assert_eq!((work.scanned_entries, work.descriptor_lookups), (0, 0));
        } else {
            assert_eq!(
                (work.scanned_entries, work.descriptor_lookups),
                (57_750_000, 57_750_000)
            );
        }
        let frozen_registration_lookups = usize::from(indexed) * 50_000;
        eprintln!(
            "CPU only indexed={indexed} entries=1155 classes=2 descriptors=20000 misses=50000 wall={elapsed:?} victim_work={work:?} incoming_descriptor_lookups=50000 frozen_registration_lookups={frozen_registration_lookups} (excludes preparation; includes lease/complete/index maintenance; not E2E)"
        );
        outputs.push((
            victims,
            cache.stats(),
            cache
                .entries
                .into_iter()
                .map(|(k, e)| (k, e.bytes, e.epoch, e.state))
                .collect::<Vec<_>>(),
        ));
    }
    assert_eq!(outputs[0], outputs[1]);
}
