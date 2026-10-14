use super::*;

#[path = "budget_tests.rs"]
mod budgets;

fn namespace(id: &str) -> NamespaceDefinition {
    NamespaceDefinition {
        id: id.into(),
        namespace: ContentNamespace {
            model: "m".into(),
            adapter: "none".into(),
            position: "p".into(),
            layout: "l".into(),
            backend: "b".into(),
            plan: "plan".into(),
            run_scope: "run".into(),
        },
    }
}
fn header(size: usize) -> TraceHeader {
    TraceHeader::new(size, vec![namespace("n")])
}
fn demand(seq: u64, id: &str, tokens: &[u32]) -> TraceEvent {
    TraceEvent::Demand(DemandEvent {
        seq,
        request: RequestId {
            id: id.into(),
            generation: 0,
        },
        namespace_id: "n".into(),
        prompt_tokens: tokens.into(),
        reuse_limit: ReuseLimit::AllPromptTokens,
    })
}
fn publish(seq: u64, id: &str, tokens: &[u32], size: usize) -> TraceEvent {
    TraceEvent::Publish(PublishEvent {
        seq,
        request: RequestId {
            id: id.into(),
            generation: 0,
        },
        namespace_id: "n".into(),
        pages: tokens
            .chunks(size)
            .map(|t| RawPage {
                tokens: t.into(),
                valid_count: t.len(),
            })
            .collect(),
    })
}

#[test]
fn fenwick_crosses_every_power_of_two_and_deletion() {
    let mut tree = Fenwick::default();
    for count in 1..=129 {
        tree.append().unwrap();
        assert_eq!(tree.sum(count), count);
    }
    for index in [1, 2, 4, 8, 16, 32, 64, 128] {
        tree.remove(index);
    }
    assert_eq!(tree.sum(129), 121);
    tree.append().unwrap();
    assert_eq!(tree.sum(130), 122);
}

#[test]
fn forced_hash_collisions_still_keep_full_identity() {
    let mut history = History {
        force_collision: true,
        ..Default::default()
    };
    let limits = ResourceLimits::default();
    let a = history.intern(0, None, &[1, 2], &limits).unwrap();
    let b = history.intern(0, None, &[2, 1], &limits).unwrap();
    let c = history.intern(0, Some(a), &[1, 2], &limits).unwrap();
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
    assert_eq!(history.intern(0, None, &[1, 2], &limits).unwrap(), a);
}

#[test]
fn equal_descriptors_share_normalized_namespace_identity() {
    let mut h = header(2);
    h.namespaces.push(namespace("alias"));
    let mut a = Analyzer::new(h, AnalyzerConfig::default()).unwrap();
    a.ingest(&demand(0, "w", &[1, 2])).unwrap();
    a.ingest(&publish(1, "w", &[1, 2], 2)).unwrap();
    let mut d = demand(2, "r", &[1, 2]);
    if let TraceEvent::Demand(event) = &mut d {
        event.namespace_id = "alias".into();
    }
    assert_eq!(
        a.ingest(&d).unwrap().unwrap().theoretical_reusable_tokens,
        2
    );
}
