use super::*;

#[test]
fn concurrent_cold_failure_success_and_eof_never_implicitly_publish() {
    let mut a = Analyzer::new(header(2), AnalyzerConfig::default()).unwrap();
    for event in [
        demand(0, "a", &[1, 2]),
        demand(1, "b", &[1, 2]),
        outcome(2, "a", RequestStatus::Success),
        outcome(3, "b", RequestStatus::Failed),
    ] {
        a.ingest(&event).unwrap();
    }
    let cold = a.ingest(&demand(4, "c", &[1, 2])).unwrap().unwrap();
    assert_eq!(cold.page_distances, Some(vec![StackDistance::Cold]));
    a.ingest(&publish(5, "c", &[1, 2], 2)).unwrap();
    a.ingest(&outcome(6, "c", RequestStatus::Cancelled))
        .unwrap();
    let warm = a.ingest(&demand(7, "d", &[1, 2])).unwrap().unwrap();
    assert_eq!(warm.page_distances, Some(vec![StackDistance::Warm(0)]));
    assert_eq!(a.report().unwrap().outcomes.pending, 1);
    let report = a.finish().unwrap();
    assert_eq!(report.outcomes.abandoned_at_trace_end, 1);
    assert_eq!(report.outcomes.pending, 0);
    assert_eq!(report.usage.pending_tokens, 0);
    assert_eq!(report.workload.cold_prefix_requests, 3);
}

#[test]
fn publish_snapshot_does_not_touch_demand_and_cannot_cross_hole() {
    let mut a = Analyzer::new(header(1), AnalyzerConfig::default()).unwrap();
    a.ingest(&demand(0, "w", &[1, 2])).unwrap();
    a.ingest(&publish(1, "w", &[1, 2], 1)).unwrap();
    for (seq, id) in [(2, "r1"), (3, "r2")] {
        let summary = a.ingest(&demand(seq, id, &[1, 2])).unwrap().unwrap();
        assert_eq!(
            summary.page_distances,
            Some(vec![StackDistance::Warm(1), StackDistance::Warm(0)])
        );
    }
    let report = a.report().unwrap();
    assert_eq!(report.capacity_points[1].page_hits, 2);
    assert_eq!(report.capacity_points[1].contiguous_prefix_tokens, 0);
}

#[test]
fn committed_decode_extension_repetition_and_padding_are_exact() {
    let mut a = Analyzer::new(header(2), AnalyzerConfig::default()).unwrap();
    a.ingest(&demand(0, "w", &[1])).unwrap();
    let padded = TraceEvent::Publish(PublishEvent {
        seq: 1,
        request: request("w"),
        namespace_id: "n".into(),
        pages: vec![RawPage {
            tokens: vec![1, 999],
            valid_count: 1,
        }],
    });
    a.ingest(&padded).unwrap();
    assert_eq!(
        a.ingest(&demand(2, "r", &[1]))
            .unwrap()
            .unwrap()
            .theoretical_reusable_tokens,
        1
    );
    a.ingest(&publish(3, "w", &[1, 7, 8], 2)).unwrap();
    a.ingest(&publish(4, "w", &[1, 7, 8], 2)).unwrap();
    assert_eq!(a.report().unwrap().usage.pending_tokens, 4);
    a.ingest(&outcome(5, "w", RequestStatus::Success)).unwrap();
    assert_eq!(a.report().unwrap().usage.pending_tokens, 1);
    assert!(a.ingest(&publish(6, "w", &[1, 7, 8], 2)).is_err());
    assert_eq!(a.report(), Err(KvSetError::Poisoned));
}

#[test]
fn rejects_rewind_prompt_rewrite_and_decode_rewrite() {
    for bad in [vec![1], vec![2, 3, 4], vec![1, 3, 5]] {
        let mut a = Analyzer::new(header(2), AnalyzerConfig::default()).unwrap();
        a.ingest(&demand(0, "w", &[1])).unwrap();
        a.ingest(&publish(1, "w", &[1, 3, 4], 2)).unwrap();
        assert!(a.ingest(&publish(2, "w", &bad, 2)).is_err());
        assert_eq!(a.finish(), Err(KvSetError::Poisoned));
    }
}

#[test]
fn request_reuse_needs_terminal_and_strictly_larger_generation() {
    let mut a = Analyzer::new(header(1), AnalyzerConfig::default()).unwrap();
    a.ingest(&demand(0, "a", &[1])).unwrap();
    a.ingest(&outcome(1, "a", RequestStatus::Success)).unwrap();
    let mut next = demand(2, "a", &[2]);
    if let TraceEvent::Demand(d) = &mut next {
        d.request.generation = 1;
    }
    a.ingest(&next).unwrap();
    let mut duplicate = demand(3, "a", &[2]);
    if let TraceEvent::Demand(d) = &mut duplicate {
        d.request.generation = 2;
    }
    assert!(a.ingest(&duplicate).is_err());
    let mut b = Analyzer::new(header(1), AnalyzerConfig::default()).unwrap();
    b.ingest(&demand(0, "a", &[])).unwrap();
    b.ingest(&outcome(1, "a", RequestStatus::Success)).unwrap();
    assert!(b.ingest(&demand(2, "a", &[])).is_err());
}

#[test]
fn seq_can_have_gaps_but_never_repeat_or_reverse() {
    for bad_seq in [0, 9] {
        let mut a = Analyzer::new(header(1), AnalyzerConfig::default()).unwrap();
        a.ingest(&demand(9, "a", &[])).unwrap();
        assert!(a.ingest(&demand(bad_seq, "b", &[])).is_err());
    }
    let mut a = Analyzer::new(header(1), AnalyzerConfig::default()).unwrap();
    a.ingest(&demand(9, "a", &[])).unwrap();
    a.ingest(&outcome(u64::MAX, "a", RequestStatus::Failed))
        .unwrap();
    assert!(a.finish().is_ok());
}

#[test]
fn window_keeps_history_and_outcomes_follow_demand_membership() {
    let (input, config) = fixture();
    let mut a = Analyzer::new(input.header.clone(), config.clone()).unwrap();
    for chunk in input.events.chunks(2) {
        for event in chunk {
            a.ingest(event).unwrap();
        }
        assert!(!a.report().unwrap().finalized);
    }
    assert_eq!(
        a.finish().unwrap(),
        analyze(&input, config.clone()).unwrap()
    );
    let mut empty = config;
    empty.measurement_window = Some(MeasurementWindow {
        start_seq: 3,
        end_seq: 3,
    });
    let report = analyze(&input, empty).unwrap();
    assert_eq!(report.workload.demands, 0);
    assert_eq!(report.outcomes, OutcomeCounts::default());
    assert!(report.usage.publish_positions > 0);
}

#[test]
fn namespace_aliases_normalize_but_parent_and_descriptor_do_not() {
    let mut h = header(2);
    h.namespaces.push(namespace("alias"));
    let mut different = namespace("other");
    different.namespace.adapter = "adapter-v2".into();
    h.namespaces.push(different);
    let mut a = Analyzer::new(h, AnalyzerConfig::default()).unwrap();
    a.ingest(&demand(0, "w", &[1, 2, 7, 8])).unwrap();
    a.ingest(&publish(1, "w", &[1, 2, 7, 8], 2)).unwrap();
    for (seq, id, ns, tokens, expected) in [
        (2, "a", "alias", vec![1, 2, 7, 8], 4),
        (3, "b", "other", vec![1, 2, 7, 8], 0),
        (4, "c", "n", vec![9, 9, 7, 8], 0),
    ] {
        let mut event = demand(seq, id, &tokens);
        if let TraceEvent::Demand(d) = &mut event {
            d.namespace_id = ns.into();
        }
        assert_eq!(
            a.ingest(&event)
                .unwrap()
                .unwrap()
                .theoretical_reusable_tokens,
            expected
        );
    }
}

#[test]
fn actual_observations_are_separate_and_checked_even_outside_window() {
    let mut a = Analyzer::new(header(2), AnalyzerConfig::default()).unwrap();
    a.ingest(&demand(0, "a", &[1, 2])).unwrap();
    let mut event = outcome(1, "a", RequestStatus::Success);
    if let TraceEvent::Outcome(o) = &mut event {
        o.actual = Some(ActualObservation {
            reused_prefix_tokens: 2,
            reused_pages: 1,
            cache_capacity_pages: 1,
        });
    }
    a.ingest(&event).unwrap();
    let report = a.finish().unwrap();
    assert_eq!(report.actual.reused_prefix_tokens, 2);
    assert_eq!(report.workload.theoretical_reusable_tokens, 0);
    let mut a = Analyzer::new(header(2), AnalyzerConfig::default()).unwrap();
    a.ingest(&demand(0, "a", &[1])).unwrap();
    assert!(a.ingest(&event).is_err());
}
