use super::*;

#[test]
fn histogram_insertion_is_preflighted_and_existing_buckets_are_free() {
    let limits = ResourceLimits {
        max_histogram_entries: 1,
        ..Default::default()
    };
    let mut hist = Histogram::new();
    let mut entries = 0;
    histogram_add(&mut hist, 2, 7, &mut entries, &limits).unwrap();
    histogram_add(&mut hist, 2, 8, &mut entries, &limits).unwrap();
    assert_eq!(hist[&2], 15);
    assert_eq!(entries, 1);
    assert_eq!(
        histogram_add(&mut hist, 3, 1, &mut entries, &limits),
        Err(KvSetError::ResourceLimit("histogram entries"))
    );
    assert_eq!(hist.len(), 1);
    assert!(!hist.contains_key(&3));
}

#[test]
fn warmup_large_demand_cannot_allocate_over_tiny_histogram_budget() {
    let mut h = header(1);
    h.limits.max_histogram_entries = 1;
    let c = AnalyzerConfig {
        measurement_window: Some(MeasurementWindow {
            start_seq: 2,
            end_seq: 3,
        }),
        ..Default::default()
    };
    let mut a = Analyzer::new(h, c).unwrap();
    a.ingest(&demand(0, "w", &[1, 2, 3, 4, 5, 6, 7, 8]))
        .unwrap();
    a.ingest(&publish(1, "w", &[1, 2, 3, 4, 5, 6, 7, 8], 1))
        .unwrap();
    assert_eq!(
        a.ingest(&demand(2, "r", &[1, 2, 3, 4, 5, 6, 7, 8])),
        Err(KvSetError::ResourceLimit("histogram entries"))
    );
    assert_eq!(a.engine.usage.histogram_entries, 1);
    assert_eq!(
        a.engine.page_histogram.len() + a.engine.token_histogram.len(),
        1
    );
    assert_eq!(a.report(), Err(KvSetError::Poisoned));
    assert_eq!(a.ingest(&demand(3, "x", &[])), Err(KvSetError::Poisoned));
}

#[test]
fn page_token_and_alpha_histograms_share_one_budget() {
    for budget in 0..=3 {
        let mut h = header(1);
        h.limits.max_histogram_entries = budget;
        let c = AnalyzerConfig {
            measurement_window: Some(MeasurementWindow {
                start_seq: 2,
                end_seq: 3,
            }),
            ..Default::default()
        };
        let mut a = Analyzer::new(h, c).unwrap();
        a.ingest(&demand(0, "w", &[1])).unwrap();
        a.ingest(&publish(1, "w", &[1], 1)).unwrap();
        assert_eq!(a.ingest(&demand(2, "r", &[1])).is_ok(), budget == 3);
        assert_eq!(a.engine.usage.histogram_entries, budget);
    }
}

#[test]
fn publish_rejects_token_expansion_before_keys_positions_or_later_invalid_page() {
    let mut h = header(1);
    h.limits.max_tokens_per_event = 1;
    let mut a = Analyzer::new(h, AnalyzerConfig::default()).unwrap();
    a.ingest(&demand(0, "w", &[1])).unwrap();
    let mut p = publish(1, "w", &[1, 2], 1);
    if let TraceEvent::Publish(p) = &mut p {
        p.pages.push(RawPage {
            tokens: vec![],
            valid_count: 0,
        });
    }
    assert_eq!(
        a.ingest(&p),
        Err(KvSetError::ResourceLimit("tokens per event"))
    );
    assert_eq!(a.engine.history.keys.len(), 1);
    assert_eq!(a.engine.history.fenwick.tree.len(), 0);
    assert_eq!(a.engine.pending["w"].known, vec![1]);
}

#[test]
fn publish_positions_and_pending_growth_checked_before_any_publish() {
    for position_limit in [true, false] {
        let mut h = header(1);
        if position_limit {
            h.limits.max_publish_positions = 1;
        } else {
            h.limits.max_pending_tokens = 1;
        }
        let mut a = Analyzer::new(h, AnalyzerConfig::default()).unwrap();
        a.ingest(&demand(0, "w", &[1])).unwrap();
        let expected = if position_limit {
            "publish positions"
        } else {
            "pending tokens"
        };
        assert_eq!(
            a.ingest(&publish(1, "w", &[1, 2], 1)),
            Err(KvSetError::ResourceLimit(expected))
        );
        assert_eq!(a.engine.history.fenwick.tree.len(), 0);
        assert_eq!(a.engine.history.keys.len(), 1);
        assert_eq!(a.engine.pending["w"].known.len(), 1);
    }
}

#[test]
fn huge_page_size_does_not_preallocate_page_size_times_pages() {
    let mut a = Analyzer::new(header(usize::MAX), AnalyzerConfig::default()).unwrap();
    a.ingest(&demand(0, "w", &[1])).unwrap();
    a.ingest(&publish(1, "w", &[1], usize::MAX)).unwrap();
    assert_eq!(a.report().unwrap().usage.publish_positions, 1);
}

#[test]
fn content_interning_limit_checks_before_copy() {
    let mut h = History::default();
    let mut limits = ResourceLimits {
        max_unique_keys: 1,
        max_identity_token_bytes: 4,
        ..Default::default()
    };
    h.intern(0, None, &[1], &limits).unwrap();
    assert_eq!(
        h.intern(0, None, &[2], &limits),
        Err(KvSetError::ResourceLimit("unique keys"))
    );
    assert_eq!(h.keys.len(), 1);
    assert_eq!(h.token_bytes, 4);
    limits.max_unique_keys = 2;
    assert_eq!(
        h.intern(0, None, &[2], &limits),
        Err(KvSetError::ResourceLimit("identity token bytes"))
    );
    assert_eq!(h.keys.len(), 1);
}

#[test]
fn finite_capacity_allocation_is_checked_before_reserve() {
    let mut h = header(1);
    h.recency_model = RecencyModel::DeferredAdmissionPrefixLruV1;
    let config = AnalyzerConfig {
        capacities: Some(vec![usize::MAX]),
        ..Default::default()
    };
    assert!(matches!(
        DeferredAnalyzer::new(h.clone(), config),
        Err(KvSetError::ResourceLimit("finite cache entries"))
    ));
    h.limits.max_total_capacity_entries = usize::MAX;
    let config = AnalyzerConfig {
        capacities: Some(vec![1, usize::MAX]),
        ..Default::default()
    };
    assert!(matches!(
        DeferredAnalyzer::new(h, config),
        Err(KvSetError::ArithmeticOverflow)
    ));
}

#[test]
fn report_breakpoints_and_rows_are_bounded_before_output_and_poison() {
    for row_limit in [false, true] {
        let mut h = header(1);
        if row_limit {
            h.limits.max_report_points = 5;
        } else {
            h.limits.max_capacity_points = 1;
        }
        let mut a = Analyzer::new(h, AnalyzerConfig::default()).unwrap();
        a.ingest(&demand(0, "w", &[1, 2])).unwrap();
        a.ingest(&publish(1, "w", &[1, 2], 1)).unwrap();
        a.ingest(&demand(2, "r", &[1, 2])).unwrap();
        assert_eq!(
            a.report(),
            Err(KvSetError::ResourceLimit(if row_limit {
                "report points"
            } else {
                "capacity points"
            }))
        );
        assert_eq!(a.report(), Err(KvSetError::Poisoned));
    }
}

#[test]
fn event_pending_summary_and_id_limits_reject_without_new_identity() {
    for which in 0..6 {
        let mut h = header(1);
        match which {
            0 => h.limits.max_events = 0,
            1 => h.limits.max_pending_requests = 0,
            2 => h.limits.max_pending_tokens = 0,
            3 => h.limits.max_demand_summary_entries = 0,
            4 => h.limits.max_request_id_bytes = 0,
            _ => h.limits.max_request_history_bytes = 0,
        }
        let mut a = Analyzer::new(h, AnalyzerConfig::default()).unwrap();
        assert!(matches!(
            a.ingest(&demand(0, "r", &[1])),
            Err(KvSetError::ResourceLimit(_))
        ));
        assert!(a.engine.pending.is_empty());
        assert!(a.engine.history.keys.is_empty());
    }
}

#[test]
fn namespace_and_parameter_limits_include_ids_and_pre_normalization_input() {
    let mut h = header(1);
    h.namespaces[0].id = "long-namespace-id".into();
    h.limits.max_namespace_bytes = 1;
    assert!(matches!(
        Analyzer::new(h, AnalyzerConfig::default()),
        Err(KvSetError::ResourceLimit("namespace bytes"))
    ));
    let mut h = header(1);
    h.limits.max_target_parameters = 3;
    let config = AnalyzerConfig {
        alphas: vec![Ratio::new(1, 1).unwrap(); 100],
        ..Default::default()
    };
    assert!(matches!(
        Analyzer::new(h, config),
        Err(KvSetError::ResourceLimit("target parameters"))
    ));
}

#[test]
fn arithmetic_overflow_and_histogram_overflow_fail_closed() {
    assert_eq!(add(usize::MAX, 1), Err(KvSetError::ArithmeticOverflow));
    assert_eq!(mul(usize::MAX, 2), Err(KvSetError::ArithmeticOverflow));
    let mut hist = Histogram::from([(1, u64::MAX)]);
    let mut entries = 1;
    assert_eq!(
        histogram_add(&mut hist, 1, 1, &mut entries, &ResourceLimits::default()),
        Err(KvSetError::ArithmeticOverflow)
    );
    assert_eq!(entries, 1);
    assert_eq!(hist[&1], u64::MAX);
}
