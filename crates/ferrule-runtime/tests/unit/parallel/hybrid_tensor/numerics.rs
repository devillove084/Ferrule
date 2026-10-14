use super::*;

pub(super) fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 3e-5, "{a} != {b}");
    }
}
fn snapshots_match(probe: &Probe, oracle: &Probe, layout: &HybridExpertTopology, id: u64) {
    let actual = probe.snapshots.lock().unwrap();
    let expected = oracle.snapshots.lock().unwrap();
    let full = &expected[&(0, id, false)];
    for stage in layout.stages() {
        for (lane, owner) in stage.tensor_members().iter().enumerate() {
            let local = &actual[&(owner.get(), id, false)];
            for (layer, kv) in &local.kv {
                close(&kv.key, &full.kv[layer].key);
                close(&kv.value, &full.kv[layer].value);
            }
            for layer in stage.layers() {
                if let HybridLocalAttention::GatedDelta {
                    shape,
                    qkv_channels,
                    value_heads,
                    ..
                } = &layer.attention[lane]
                {
                    let a = &local.recurrent[&layer.global_layer];
                    let b = &full.recurrent[&layer.global_layer];
                    assert_eq!(a.position(), b.position());
                    assert_eq!(a.shape(), *shape);
                    let conv = qkv_channels
                        .iter()
                        .flat_map(|r| {
                            b.conv_history()[r.start * shape.kernel..r.end * shape.kernel]
                                .iter()
                                .copied()
                        })
                        .collect::<Vec<_>>();
                    close(a.conv_history(), &conv);
                    let width = shape.key_dim * shape.value_dim;
                    close(
                        a.recurrent(),
                        &b.recurrent()[value_heads.start * width..value_heads.end * width],
                    );
                }
            }
        }
    }
}

#[test]
fn actual_pp2_tp2_ep2_matches_unsplit_owner_logits_local_state_and_kv() {
    let fixture = Arc::new(fixture::Fixture::new());
    let observed = Arc::new(Probe::default());
    let oracle = Arc::new(Probe::default());
    let mut split = pipeline(fixture.clone(), 2, 2, observed.clone());
    let mut full = pipeline(fixture.clone(), 1, 1, oracle.clone());
    let model_layout = layout(fixture.resources().spec(), 2, 2);
    assert_eq!(split.execution_scopes().kv_participants().len(), 4);
    let scopes = split.execution_scopes();
    let mut physical = scopes
        .kv_participants()
        .as_participants()
        .iter()
        .collect::<std::collections::BTreeSet<_>>();
    for (stage, plan) in model_layout.stages().iter().enumerate() {
        let attached = scopes
            .expert_dispatch_members(stage as u32, 0)
            .unwrap()
            .unwrap();
        let expected = plan.expert_attachment().unwrap();
        assert_eq!(
            attached.source_scope(),
            ExpertSourceScope::ExternalTensorSource
        );
        assert_eq!(attached.source_rank(), expected.source_rank);
        assert_eq!(attached.iter().collect::<Vec<_>>(), expected.members);
        assert!(
            attached
                .iter()
                .all(|rank| !scopes.kv_participants().as_participants().contains(rank))
        );
        physical.extend(attached.iter());
    }
    assert_eq!(physical.len(), 8);
    for (i, tokens) in [vec![1, 2, 3], vec![5], vec![6], vec![7]]
        .iter()
        .enumerate()
    {
        let phase = if i == 0 {
            ForwardPhase::Prefill
        } else {
            ForwardPhase::Decode
        };
        let a = split.forward(SessionId(73), tokens, phase).unwrap();
        let b = full.forward(SessionId(73), tokens, phase).unwrap();
        close(a.logits.values(), b.logits.values());
        snapshots_match(&observed, &oracle, &model_layout, a.transaction.get());
    }
    let stats = split.owner_stats().unwrap();
    assert_eq!(
        stats.iter().map(|s| s.experts.len()).collect::<Vec<_>>(),
        vec![0, 2, 0, 2]
    );
    for s in &stats {
        assert_eq!(s.expert_outstanding, 0);
        for e in &s.experts {
            assert!(e.rank.owner.get() >= 101);
            assert_eq!(e.last_context.unwrap().source_rank, s.rank.global);
        }
    }
    assert_eq!(split.coordinator().publication_count(), 4);
    split.shutdown().unwrap();
    full.shutdown().unwrap();
    assert_eq!(observed.stopped.load(Ordering::Relaxed), 4);
    assert_eq!(observed.expert_shutdowns.load(Ordering::Relaxed), 2);
}
