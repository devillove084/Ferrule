use super::*;
use std::collections::{HashMap, HashSet, VecDeque};

type Key = (ContentNamespace, Vec<u32>);
struct Sample {
    prompt: usize,
    eligible: usize,
    theoretical: usize,
    hits: Vec<usize>,
    tokens: Vec<usize>,
}
fn keys(ns: &ContentNamespace, tokens: &[u32], size: usize) -> Vec<Key> {
    (0..tokens.len())
        .step_by(size)
        .map(|start| {
            (
                ns.clone(),
                tokens[..(start + size).min(tokens.len())].to_vec(),
            )
        })
        .collect()
}
fn touch(cache: &mut VecDeque<Key>, key: &Key, capacity: usize) {
    cache.retain(|existing| existing != key);
    if capacity > 0 {
        cache.push_front(key.clone());
        cache.truncate(capacity);
    }
}
fn ceil(n: usize, r: Ratio) -> usize {
    (n * r.numerator as usize).div_ceil(r.denominator as usize)
}

fn compare(input: &TraceInput, config: AnalyzerConfig) {
    let capacities = config.capacities.clone().unwrap();
    let mut caches: Vec<VecDeque<Key>> = capacities.iter().map(|_| VecDeque::new()).collect();
    let mut infinite = VecDeque::new();
    let mut published = HashSet::new();
    let namespaces: HashMap<_, _> = input
        .header
        .namespaces
        .iter()
        .map(|n| (n.id.clone(), n.namespace.clone()))
        .collect();
    let mut samples = Vec::new();
    let mut snapshot = if input.header.recency_model == RecencyModel::PublishRecencySnapshotV1 {
        Some(Analyzer::new(input.header.clone(), config.clone()).unwrap())
    } else {
        None
    };
    let mut deferred = if snapshot.is_none() {
        Some(DeferredAnalyzer::new(input.header.clone(), config.clone()).unwrap())
    } else {
        None
    };
    for event in &input.events {
        let summary = match &mut snapshot {
            Some(a) => a.ingest(event).unwrap(),
            None => deferred.as_mut().unwrap().ingest(event).unwrap(),
        };
        match event {
            TraceEvent::Publish(p) => {
                let flat: Vec<u32> = p
                    .pages
                    .iter()
                    .flat_map(|page| page.tokens[..page.valid_count].iter().copied())
                    .collect();
                for key in keys(&namespaces[&p.namespace_id], &flat, input.header.page_size) {
                    published.insert(key.clone());
                    touch(&mut infinite, &key, usize::MAX);
                    for (cache, &capacity) in caches.iter_mut().zip(&capacities) {
                        touch(cache, &key, capacity);
                    }
                }
            }
            TraceEvent::Demand(d) => {
                let summary = summary.unwrap();
                let eligible = match d.reuse_limit {
                    ReuseLimit::AllPromptTokens => d.prompt_tokens.len(),
                    ReuseLimit::BeforeLastPromptToken => d.prompt_tokens.len().saturating_sub(1),
                };
                let pages = keys(
                    &namespaces[&d.namespace_id],
                    &d.prompt_tokens[..eligible],
                    input.header.page_size,
                );
                let weights: Vec<_> = d.prompt_tokens[..eligible]
                    .chunks(input.header.page_size)
                    .map(<[u32]>::len)
                    .collect();
                let theoretical: usize = pages
                    .iter()
                    .zip(&weights)
                    .take_while(|(key, _)| published.contains(*key))
                    .map(|(_, weight)| weight)
                    .sum();
                assert_eq!(summary.theoretical_reusable_tokens, theoretical);
                let mut hits = Vec::new();
                let mut tokens = Vec::new();
                for (index, cache) in caches.iter_mut().enumerate() {
                    let bitmap: Vec<_> = pages.iter().map(|key| cache.contains(key)).collect();
                    let prefix = bitmap.iter().take_while(|&&hit| hit).count();
                    hits.push(bitmap.iter().filter(|&&hit| hit).count());
                    tokens.push(weights[..prefix].iter().sum());
                    if let Some(distances) = &summary.page_distances {
                        assert_eq!(bitmap,distances.iter().map(|d|matches!(d,StackDistance::Warm(distance) if *distance<capacities[index])).collect::<Vec<_>>());
                        for (key, &distance) in pages.iter().zip(distances) {
                            assert_eq!(
                                distance,
                                infinite
                                    .iter()
                                    .position(|other| other == key)
                                    .map_or(StackDistance::Cold, StackDistance::Warm)
                            );
                        }
                    } else {
                        assert_eq!(summary.capacity_results[index].page_hits, bitmap);
                        assert_eq!(
                            summary.capacity_results[index].contiguous_prefix_tokens,
                            tokens[index]
                        );
                        for key in &pages[..prefix] {
                            touch(cache, key, capacities[index]);
                        }
                    }
                }
                if config
                    .measurement_window
                    .is_none_or(|w| d.seq >= w.start_seq && d.seq < w.end_seq)
                {
                    samples.push(Sample {
                        prompt: d.prompt_tokens.len(),
                        eligible,
                        theoretical,
                        hits,
                        tokens,
                    });
                }
            }
            TraceEvent::Outcome(_) => assert!(summary.is_none()),
        }
    }
    assert!(
        published.len() <= *capacities.last().unwrap(),
        "oracle must include saturation capacity"
    );
    let report = match snapshot {
        Some(a) => a.finish().unwrap(),
        None => deferred.unwrap().finish().unwrap(),
    };
    assert_eq!(report, analyze(input, config.clone()).unwrap());
    let opportunities = samples.iter().filter(|s| s.theoretical > 0).count();
    assert_eq!(report.workload.opportunity_requests, opportunities as u64);
    for (ci, point) in report.capacity_points.iter().enumerate() {
        assert_eq!(
            point.page_hits,
            samples.iter().map(|s| s.hits[ci] as u64).sum::<u64>()
        );
        assert_eq!(
            point.contiguous_prefix_tokens,
            samples.iter().map(|s| s.tokens[ci] as u64).sum::<u64>()
        );
    }
    for (ai, coverage) in report.request_target_coverage.iter().enumerate() {
        let target = |s: &Sample| {
            ceil(
                match config.target_basis {
                    TargetBasis::TheoreticalReusable => s.theoretical,
                    TargetBasis::EligibleTokens => s.eligible,
                    TargetBasis::FullPrompt => s.prompt,
                },
                coverage.alpha,
            )
        };
        let expected: Vec<_> = (0..capacities.len())
            .map(|ci| {
                samples
                    .iter()
                    .filter(|s| s.theoretical > 0 && s.tokens[ci] >= target(s))
                    .count()
            })
            .collect();
        for (point, &n) in coverage.points.iter().zip(&expected) {
            assert_eq!(point.achieved_requests, n as u64);
        }
        assert_eq!(
            coverage.unreachable_target_requests,
            samples.iter().filter(|s| target(s) > s.theoretical).count() as u64
        );
        if let Some(budget) = config.capacity_budget {
            let ci = capacities.iter().position(|&c| c == budget).unwrap();
            assert_eq!(
                coverage.beyond_budget_requests,
                Some(
                    samples
                        .iter()
                        .filter(|s| s.theoretical > 0
                            && target(s) <= s.theoretical
                            && s.tokens[ci] < target(s))
                        .count() as u64
                )
            );
        }
        let denominator = match config.coverage_denominator {
            CoverageDenominator::Opportunity => opportunities,
            CoverageDenominator::AllDemands => samples.len(),
        };
        for answer in report
            .target_capacities
            .iter()
            .filter(|t| t.alpha == coverage.alpha)
        {
            let need = ceil(denominator, answer.p);
            let exact = if denominator == 0 {
                CapacityRequirement::Undefined
            } else if need == 0 {
                CapacityRequirement::Exact { capacity: 0 }
            } else {
                match expected.iter().position(|&n| n >= need) {
                    None => CapacityRequirement::Unreachable,
                    Some(ci)
                        if config
                            .capacity_budget
                            .is_some_and(|budget| capacities[ci] > budget) =>
                    {
                        CapacityRequirement::BeyondBudget {
                            budget: config.capacity_budget.unwrap(),
                        }
                    }
                    Some(ci)
                        if input.header.recency_model
                            == RecencyModel::DeferredAdmissionPrefixLruV1 =>
                    {
                        CapacityRequirement::AtSampledCapacity {
                            capacity: capacities[ci],
                        }
                    }
                    Some(ci) => CapacityRequirement::Exact {
                        capacity: capacities[ci],
                    },
                }
            };
            assert_eq!(answer.requirement, exact, "alpha index {ai}");
        }
    }
}

fn generated(seed: u32) -> TraceInput {
    let mut h = header(2);
    let mut other = namespace("other");
    other.namespace.run_scope = "other-run".into();
    h.namespaces.push(other);
    let mut events = Vec::new();
    let mut next = seed;
    for iteration in 0..36 {
        next = next.wrapping_mul(1664525).wrapping_add(1013904223);
        let length = (next as usize % 7) + 1;
        let base = (next >> 16) % 3;
        let tokens: Vec<_> = (0..length).map(|i| base + i as u32 % 3).collect();
        let id = format!("w{iteration}");
        let reader = format!("r{iteration}");
        let ns = if iteration % 3 == 0 { "other" } else { "n" };
        let mut d = demand(events.len() as u64, &id, &tokens);
        if let TraceEvent::Demand(v) = &mut d {
            v.namespace_id = ns.into();
        }
        events.push(d);
        let mut d = demand(events.len() as u64, &reader, &tokens);
        if let TraceEvent::Demand(v) = &mut d {
            v.namespace_id = ns.into();
            if iteration % 2 == 0 {
                v.reuse_limit = ReuseLimit::BeforeLastPromptToken;
            }
        }
        events.push(d);
        if iteration % 5 != 0 {
            for end in [length.saturating_sub(1), length] {
                let mut p = publish(events.len() as u64, &id, &tokens[..end], 2);
                if let TraceEvent::Publish(v) = &mut p {
                    v.namespace_id = ns.into();
                }
                events.push(p);
            }
        }
        events.push(outcome(
            events.len() as u64,
            &id,
            if iteration % 3 == 0 {
                RequestStatus::Cancelled
            } else {
                RequestStatus::Success
            },
        ));
        events.push(outcome(events.len() as u64, &reader, RequestStatus::Failed));
    }
    TraceInput { header: h, events }
}

#[test]
fn independent_oracle_all_small_capacities_targets_windows_and_models() {
    for seed in [1, 7, 42, 1234] {
        for model in [
            RecencyModel::PublishRecencySnapshotV1,
            RecencyModel::DeferredAdmissionPrefixLruV1,
        ] {
            for basis in [
                TargetBasis::TheoreticalReusable,
                TargetBasis::EligibleTokens,
                TargetBasis::FullPrompt,
            ] {
                for denominator in [
                    CoverageDenominator::Opportunity,
                    CoverageDenominator::AllDemands,
                ] {
                    let mut input = generated(seed);
                    input.header.recency_model = model;
                    let config = AnalyzerConfig {
                        capacities: Some((0..=80).collect()),
                        alphas: vec![fraction(0, 1), fraction(1, 3), fraction(1, 1)],
                        p_values: vec![
                            fraction(0, 1),
                            fraction(1, 4),
                            fraction(1, 2),
                            fraction(1, 1),
                        ],
                        target_basis: basis,
                        coverage_denominator: denominator,
                        capacity_budget: Some(4),
                        measurement_window: Some(MeasurementWindow {
                            start_seq: 9,
                            end_seq: 150,
                        }),
                    };
                    compare(&input, config);
                }
            }
        }
    }
}
