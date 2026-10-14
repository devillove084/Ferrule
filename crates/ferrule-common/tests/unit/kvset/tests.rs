use super::*;

#[path = "lifecycle_tests.rs"]
mod lifecycle;
#[path = "metrics_tests.rs"]
mod metrics;
#[path = "oracle_tests.rs"]
mod oracle;
#[path = "schema_tests.rs"]
mod schema;

fn fraction(n: u64, d: u64) -> Ratio {
    Ratio::new(n, d).unwrap()
}
fn namespace(id: &str) -> NamespaceDefinition {
    NamespaceDefinition {
        id: id.into(),
        namespace: ContentNamespace {
            model: "model-v1".into(),
            adapter: "none".into(),
            position: "rope-origin-0".into(),
            layout: "gqa-f32".into(),
            backend: "cpu-reference".into(),
            plan: "causal-v1".into(),
            run_scope: "test-run".into(),
        },
    }
}
fn header(size: usize) -> TraceHeader {
    TraceHeader::new(size, vec![namespace("n")])
}
fn request(id: &str) -> RequestId {
    RequestId {
        id: id.into(),
        generation: 0,
    }
}
fn demand(seq: u64, id: &str, tokens: &[u32]) -> TraceEvent {
    TraceEvent::Demand(DemandEvent {
        seq,
        request: request(id),
        namespace_id: "n".into(),
        prompt_tokens: tokens.into(),
        reuse_limit: ReuseLimit::AllPromptTokens,
    })
}
fn publish(seq: u64, id: &str, tokens: &[u32], size: usize) -> TraceEvent {
    TraceEvent::Publish(PublishEvent {
        seq,
        request: request(id),
        namespace_id: "n".into(),
        pages: tokens
            .chunks(size)
            .map(|tokens| RawPage {
                tokens: tokens.into(),
                valid_count: tokens.len(),
            })
            .collect(),
    })
}
fn outcome(seq: u64, id: &str, status: RequestStatus) -> TraceEvent {
    TraceEvent::Outcome(OutcomeEvent {
        seq,
        request: request(id),
        status,
        actual: None,
    })
}
fn fixture() -> (TraceInput, AnalyzerConfig) {
    let mut events = vec![
        demand(0, "writer", &[1, 2, 3, 4]),
        publish(1, "writer", &[1, 2, 3], 2),
        demand(2, "r1", &[1, 2, 3, 9]),
        publish(3, "writer", &[1, 2, 3, 4], 2),
        demand(4, "r2", &[1, 2, 3, 9]),
        outcome(5, "writer", RequestStatus::Success),
        outcome(6, "r1", RequestStatus::Failed),
        outcome(7, "r2", RequestStatus::Cancelled),
    ];
    for index in [2, 4] {
        if let TraceEvent::Demand(d) = &mut events[index] {
            d.reuse_limit = ReuseLimit::BeforeLastPromptToken;
        }
    }
    let config = AnalyzerConfig {
        capacities: Some(vec![0, 1, 2, 3, 80]),
        alphas: vec![fraction(2, 3), fraction(1, 1)],
        p_values: vec![fraction(1, 2), fraction(1, 1)],
        measurement_window: Some(MeasurementWindow {
            start_seq: 2,
            end_seq: 5,
        }),
        ..Default::default()
    };
    (
        TraceInput {
            header: header(2),
            events,
        },
        config,
    )
}

#[test]
fn hand_calculated_partial_versions_and_three_metrics() {
    let (input, config) = fixture();
    let report = analyze(&input, config).unwrap();
    assert_eq!(
        report
            .capacity_points
            .iter()
            .map(|p| (p.capacity_pages, p.page_hits, p.contiguous_prefix_tokens))
            .collect::<Vec<_>>(),
        [(0, 0, 0), (1, 1, 0), (2, 3, 5), (3, 4, 6), (80, 4, 6)]
    );
    assert_eq!(report.workload.page_references, 4);
    assert_eq!(report.workload.eligible_tokens, 6);
    assert_eq!(report.workload.prompt_tokens, 8);
    assert_eq!(report.workload.theoretical_reusable_tokens, 6);
    assert_eq!(
        report.capacity_points[2].contiguous_prefix_token_hit_rate_eligible,
        Some(fraction(5, 6))
    );
    assert_eq!(
        report.capacity_points[2].contiguous_prefix_token_hit_rate_prompt,
        Some(fraction(5, 8))
    );
    assert_eq!(
        report.request_target_coverage[0].points[2].achieved_requests,
        2
    );
    assert_eq!(
        report.request_target_coverage[1].points[2].achieved_requests,
        1
    );
    assert_eq!(
        report
            .target_capacities
            .iter()
            .map(|t| t.requirement.clone())
            .collect::<Vec<_>>(),
        [
            CapacityRequirement::Exact { capacity: 2 },
            CapacityRequirement::Exact { capacity: 2 },
            CapacityRequirement::Exact { capacity: 2 },
            CapacityRequirement::Exact { capacity: 3 }
        ]
    );
    assert_eq!(report.outcomes.success, 0);
    assert_eq!(report.outcomes.failed, 1);
    assert_eq!(report.outcomes.cancelled, 1);
}
