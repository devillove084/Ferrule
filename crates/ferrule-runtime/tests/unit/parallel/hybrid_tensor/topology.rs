use super::*;

#[test]
fn optional_attachment_nonzero_source_and_invalid_sparse_kv_scope_fail_closed() {
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let spec = resources.spec();
    let l = layout(spec, 2, 2);
    assert_eq!(
        l.expert_group(0, options().expert_limits)
            .unwrap()
            .source_rank,
        rank(1)
    );
    assert_eq!(
        l.expert_group(1, options().expert_limits)
            .unwrap()
            .source_rank,
        rank(3)
    );
    let p = l.stages()[0].clone();
    let mut ep = p.expert_attachment().unwrap().clone();
    ep.source_rank = rank(91);
    let sparse =
        HybridParallelPlan::new(spec, 0..2, p.scope(), vec![rank(90), rank(91)], Some(ep)).unwrap();
    let sparse = HybridExpertTopology::new(spec, vec![sparse, l.stages()[1].clone()]).unwrap();
    assert!(
        PipelineParallelExecutor::new_hybrid_cpu_tensor(
            topology(2, 2),
            spec,
            sparse,
            config(),
            options(),
            |_| panic!("invalid mesh must fail before owner startup")
        )
        .is_err()
    );
}
