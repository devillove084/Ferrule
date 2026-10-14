use super::*;

#[test]
fn all_breakpoints_and_explicit_samples_have_same_exact_targets() {
    let (input, mut config) = fixture();
    config.capacities = None;
    let all = analyze(&input, config.clone()).unwrap();
    assert_eq!(
        all.capacity_points
            .iter()
            .map(|p| p.capacity_pages)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    config.capacities = Some(vec![0, 80]);
    assert_eq!(
        analyze(&input, config).unwrap().target_capacities,
        all.target_capacities
    );
}

#[test]
fn exact_ratios_and_defaults() {
    assert!(Ratio::new(0, 0).is_err());
    assert!(Ratio::new(2, 1).is_err());
    assert_eq!(
        fraction(u64::MAX - 1, u64::MAX).ceil_mul(u64::MAX).unwrap(),
        u64::MAX - 1
    );
    assert_eq!(fraction(95, 100).ceil_mul(101).unwrap(), 96);
    let (input, mut config) = fixture();
    config.alphas = vec![fraction(2, 2), fraction(1, 1), fraction(0, 9)];
    config.capacities = Some(vec![3, 0, 3, 1]);
    let report = analyze(&input, config).unwrap();
    assert_eq!(report.config.alphas, vec![fraction(0, 1), fraction(1, 1)]);
    assert_eq!(report.config.capacities, Some(vec![0, 1, 3]));
}

#[test]
fn no_opportunity_is_not_success_even_with_alpha_zero() {
    for model in [
        RecencyModel::PublishRecencySnapshotV1,
        RecencyModel::DeferredAdmissionPrefixLruV1,
    ] {
        let mut h = header(1);
        h.recency_model = model;
        let config = AnalyzerConfig {
            capacities: Some(vec![0, 2]),
            alphas: vec![fraction(0, 1)],
            p_values: vec![fraction(0, 1), fraction(1, 1)],
            coverage_denominator: CoverageDenominator::AllDemands,
            ..Default::default()
        };
        let report = analyze(
            &TraceInput {
                header: h,
                events: vec![demand(0, "a", &[1]), demand(1, "b", &[])],
            },
            config,
        )
        .unwrap();
        assert!(
            report.request_target_coverage[0]
                .points
                .iter()
                .all(|p| p.achieved_requests == 0 && p.coverage_opportunity.is_none())
        );
        assert_eq!(
            report.target_capacities[0].requirement,
            CapacityRequirement::Exact { capacity: 0 }
        );
        assert_eq!(
            report.target_capacities[1].requirement,
            CapacityRequirement::Unreachable
        );
        assert_eq!(report.workload.no_opportunity_requests, 2);
    }
}

#[test]
fn deferred_zero_coverage_is_exact_without_sampling_zero() {
    let (mut input, mut config) = fixture();
    input.header.recency_model = RecencyModel::DeferredAdmissionPrefixLruV1;
    config.capacities = Some(vec![2]);
    config.p_values = vec![fraction(0, 1)];
    let report = analyze(&input, config.clone()).unwrap();
    assert!(
        report
            .target_capacities
            .iter()
            .all(|t| t.requirement == CapacityRequirement::Exact { capacity: 0 })
    );
    input.events.clear();
    assert!(
        analyze(&input, config)
            .unwrap()
            .target_capacities
            .iter()
            .all(|t| t.requirement == CapacityRequirement::Undefined)
    );
}

#[test]
fn zero_denominators_empty_prompt_and_empty_trace_are_undefined() {
    let mut input = TraceInput {
        header: header(2),
        events: vec![],
    };
    for empty_request in [false, true] {
        if empty_request {
            input.events.push(demand(0, "a", &[]));
        }
        let report = analyze(&input, AnalyzerConfig::default()).unwrap();
        assert!(report.capacity_points[0].page_reference_hit_rate.is_none());
        assert!(
            report.capacity_points[0]
                .contiguous_prefix_token_hit_rate_prompt
                .is_none()
        );
        assert!(
            report
                .target_capacities
                .iter()
                .all(|t| t.requirement == CapacityRequirement::Undefined)
        );
    }
}

#[test]
fn distinguishes_mathematical_unreachable_from_budget_and_sampling() {
    let (mut input, mut config) = fixture();
    config.alphas = vec![fraction(1, 1)];
    config.p_values = vec![fraction(1, 1)];
    config.capacity_budget = Some(2);
    config.capacities = Some(vec![0, 80]);
    let r = analyze(&input, config.clone()).unwrap();
    assert_eq!(
        r.target_capacities[0].requirement,
        CapacityRequirement::BeyondBudget { budget: 2 }
    );
    assert_eq!(r.request_target_coverage[0].beyond_budget_requests, Some(1));
    config.target_basis = TargetBasis::FullPrompt;
    let r = analyze(&input, config.clone()).unwrap();
    assert_eq!(
        r.target_capacities[0].requirement,
        CapacityRequirement::Unreachable
    );
    assert_eq!(r.request_target_coverage[0].unreachable_target_requests, 2);
    config.target_basis = TargetBasis::TheoreticalReusable;
    config.capacity_budget = None;
    config.capacities = Some(vec![1]);
    input.header.recency_model = RecencyModel::DeferredAdmissionPrefixLruV1;
    assert_eq!(
        analyze(&input, config.clone()).unwrap().target_capacities[0].requirement,
        CapacityRequirement::NotReachedAtSampledCapacities
    );
    config.capacities = Some(vec![1, 80]);
    assert_eq!(
        analyze(&input, config).unwrap().target_capacities[0].requirement,
        CapacityRequirement::AtSampledCapacity { capacity: 80 }
    );
}

#[test]
fn full_prompt_eligible_and_theoretical_are_different_targets() {
    let input = TraceInput {
        header: header(2),
        events: vec![
            demand(0, "w", &[1, 2, 3, 4]),
            publish(1, "w", &[1, 2], 2),
            demand(2, "r", &[1, 2, 3]),
        ],
    };
    for (basis, unreachable) in [
        (TargetBasis::TheoreticalReusable, 0),
        (TargetBasis::EligibleTokens, 1),
        (TargetBasis::FullPrompt, 1),
    ] {
        let config = AnalyzerConfig {
            target_basis: basis,
            measurement_window: Some(MeasurementWindow {
                start_seq: 2,
                end_seq: 3,
            }),
            ..Default::default()
        };
        assert_eq!(
            analyze(&input, config).unwrap().request_target_coverage[0].unreachable_target_requests,
            unreachable
        );
    }
}

#[test]
fn per_capacity_demand_touch_diverges_from_publish_recency() {
    // Cache 2 sees A/B, then touching prefix A saves A when publishing unrelated C.
    let input = TraceInput {
        header: header(1),
        events: vec![
            demand(0, "w", &[1, 2]),
            publish(1, "w", &[1, 2], 1),
            demand(2, "touch", &[1]),
            demand(3, "c", &[9]),
            publish(4, "c", &[9], 1),
            demand(5, "probe", &[1]),
        ],
    };
    let config = AnalyzerConfig {
        capacities: Some(vec![1, 2, 3]),
        measurement_window: Some(MeasurementWindow {
            start_seq: 5,
            end_seq: 6,
        }),
        ..Default::default()
    };
    let snap = analyze(&input, config.clone()).unwrap();
    let mut deferred = input;
    deferred.header.recency_model = RecencyModel::DeferredAdmissionPrefixLruV1;
    let finite = analyze(&deferred, config).unwrap();
    assert_eq!(snap.capacity_points[1].contiguous_prefix_tokens, 0);
    assert_eq!(finite.capacity_points[1].contiguous_prefix_tokens, 1);
}

#[test]
fn reset_window_invalid_header_and_deferred_configuration_are_rejected() {
    let mut h = header(1);
    h.window_mode = WindowMode::ResetPerWindow;
    assert!(Analyzer::new(h, AnalyzerConfig::default()).is_err());
    assert!(Analyzer::new(header(0), AnalyzerConfig::default()).is_err());
    let mut h = header(1);
    h.schema_version = 2;
    assert!(Analyzer::new(h, AnalyzerConfig::default()).is_err());
    let mut h = header(1);
    h.namespaces[0].namespace.adapter.clear();
    assert!(Analyzer::new(h, AnalyzerConfig::default()).is_err());
    let mut h = header(1);
    h.namespaces.push(namespace("n"));
    assert!(Analyzer::new(h, AnalyzerConfig::default()).is_err());
    let mut h = header(1);
    h.recency_model = RecencyModel::DeferredAdmissionPrefixLruV1;
    assert!(DeferredAnalyzer::new(h.clone(), AnalyzerConfig::default()).is_err());
    let config = AnalyzerConfig {
        capacities: Some(vec![2]),
        capacity_budget: Some(1),
        ..Default::default()
    };
    assert!(DeferredAnalyzer::new(h, config).is_err());
}
