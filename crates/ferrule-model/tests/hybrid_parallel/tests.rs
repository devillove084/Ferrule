use ferrule_backend::cpu::gated_delta::{GatedDeltaInput, GatedDeltaWeights, gated_delta_step};
use ferrule_common::{ParallelRankId, ParallelTopologyId};
use ferrule_model::transformer::hybrid_parallel::*;
use ferrule_model::transformer::*;
use ferrule_runtime::parallel::hybrid_topology::HybridExpertTopology;

fn rank(n: usize) -> ParallelRankId {
    ParallelRankId::new(n as u32)
}
fn spec() -> DecoderModelSpec {
    let norm = || RmsNorm::new(4, 1e-6).unwrap();
    let rope = RotaryEmbedding::new(
        2,
        10000.0,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 2 },
        RotaryScaling::None,
    )
    .unwrap();
    let moe = Moe::new(
        4,
        5,
        MoeRouterSpec::new(
            4,
            2,
            RouterScoreFunction::Softmax,
            RouterSelection::TopK,
            true,
            1.0,
        )
        .unwrap(),
        false,
    )
    .unwrap()
    .with_shared_expert(SwiGlu::new(4, 7, false).unwrap())
    .unwrap()
    .with_shared_expert_gate(Linear::new(4, 1, false).unwrap())
    .unwrap();
    let layers = (0..4)
        .map(|i| {
            DecoderLayer::new(
                i,
                norm(),
                if i % 2 == 0 {
                    Attention::GatedDeltaNet(
                        GatedDeltaNetAttention::new(4, 4, 8, 2, 2, 3, 1e-6, false).unwrap(),
                    )
                } else {
                    Attention::Gqa(GqaAttention::new(4, 8, 2, 2, false, rope.clone()).unwrap())
                },
                Residual::Add,
                norm(),
                FeedForward::Moe(moe.clone()),
                Residual::Add,
            )
            .unwrap()
        })
        .collect();
    DecoderModelSpec::new(DecoderModelParts {
        architecture: "hybrid-tp-ep-reference".into(),
        hidden_size: 4,
        vocab_size: 8,
        max_sequence_length: Some(16),
        token_embedding: Embedding::new(8, 4, None).unwrap(),
        layers,
        final_norm: norm(),
        output: Linear::new(4, 8, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap()
}
fn plan(stage: usize, tp: usize) -> HybridParallelPlan {
    let members = (stage * 4..stage * 4 + tp).map(rank).collect::<Vec<_>>();
    let owners = (stage * 4..stage * 4 + 4).map(rank).collect::<Vec<_>>();
    let placement = ExpertPlacement::new(
        (stage * 2..stage * 2 + 2).flat_map(|l| (0..4).map(move |e| (l, e, rank(stage * 4 + e)))),
    )
    .unwrap();
    HybridParallelPlan::new(
        &spec(),
        stage * 2..stage * 2 + 2,
        HybridExpertScope {
            topology: ParallelTopologyId::new(17),
            replica: 2,
            stage: stage as u32,
        },
        members,
        Some(HybridExpertAttachment {
            source_rank: rank(stage * 4),
            members: owners,
            placement,
        }),
    )
    .unwrap()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 2e-5, "{x} != {y}");
    }
}

#[test]
fn pp_tp_ep_membership_is_a_union() {
    let topology = HybridExpertTopology::new(&spec(), vec![plan(0, 4), plan(1, 4)]).unwrap();
    assert_eq!(topology.physical_members().len(), 8); // NOT PP2 * TP4 * EP4.
    let group = topology
        .expert_group(
            1,
            ExpertDispatchLimits {
                max_tokens: 4,
                max_bytes: 64,
            },
        )
        .unwrap();
    assert_eq!(group.source_rank, rank(4));
    assert_eq!(group.layers, 2..4);
    assert_eq!(group.placement.owner(2, 3), Some(rank(7)));
    assert!(HybridExpertTopology::new(&spec(), vec![plan(1, 4), plan(0, 4)]).is_err());
    assert!(HybridExpertTopology::new(&spec(), vec![plan(0, 4)]).is_err());
    // PP x EP also uses the same contract without inventing tensor lanes.
    assert_eq!(
        HybridExpertTopology::new(&spec(), vec![plan(0, 1), plan(1, 1)])
            .unwrap()
            .physical_members()
            .len(),
        8
    );
}

#[test]
fn local_geometry_preserves_gqa_replica_owners_and_gdn_groups() {
    let p = plan(0, 4);
    for lane in 0..4 {
        let HybridLocalAttention::Gqa {
            query_heads,
            kv_heads,
            kv_owners,
            ..
        } = &p.layers()[1].attention[lane]
        else {
            panic!()
        };
        assert_eq!(*query_heads, lane * 2..lane * 2 + 2);
        assert_eq!(*kv_heads, lane / 2..lane / 2 + 1);
        assert_eq!(*kv_owners, vec![rank(lane / 2 * 2)]);
        let HybridLocalAttention::GatedDelta {
            key_heads,
            value_heads,
            shape,
            ..
        } = &p.layers()[0].attention[lane]
        else {
            panic!()
        };
        assert_eq!(*key_heads, lane..lane + 1);
        assert_eq!(*value_heads, lane * 2..lane * 2 + 2);
        assert_eq!((shape.key_heads, shape.value_heads), (1, 2));
        assert_eq!(shape.sizes().unwrap(), (8, 24, 8));
    }
    // TP <= KV partitions rather than replicates.
    let p2 = plan(0, 2);
    let HybridLocalAttention::Gqa {
        kv_heads,
        kv_owners,
        ..
    } = &p2.layers()[1].attention[1]
    else {
        panic!()
    };
    assert_eq!(*kv_heads, 1..2);
    assert_eq!(*kv_owners, vec![rank(1)]);
    let placement =
        ExpertPlacement::new((0..2).flat_map(|l| (0..4).map(move |e| (l, e, rank(e))))).unwrap();
    assert!(
        HybridParallelPlan::new(
            &spec(),
            0..2,
            p.scope(),
            (0..8).map(rank).collect(),
            Some(HybridExpertAttachment {
                source_rank: rank(0),
                members: (0..4).map(rank).collect(),
                placement: placement.clone()
            })
        )
        .is_err()
    );
    assert!(
        HybridParallelPlan::new(
            &spec(),
            0..2,
            p.scope(),
            vec![rank(0), rank(0)],
            Some(HybridExpertAttachment {
                source_rank: rank(0),
                members: (0..4).map(rank).collect(),
                placement
            })
        )
        .is_err()
    );
    assert!(
        HybridParallelPlan::new(
            &spec(),
            0..2,
            p.scope(),
            vec![rank(0)],
            Some(HybridExpertAttachment {
                source_rank: rank(0),
                members: vec![rank(0)],
                placement: ExpertPlacement::new([]).unwrap()
            })
        )
        .is_err()
    );
}

#[test]
fn gdn_local_state_and_multi_token_outputs_match_unsharded_recurrence() {
    let local = plan(0, 4);
    let global = plan(0, 1);
    let HybridLocalAttention::GatedDelta {
        shape: full_shape, ..
    } = global.layers()[0].attention[0]
    else {
        panic!()
    };
    let (channels, conv_len, recurrent_len) = full_shape.sizes().unwrap();
    let mut full_conv = vec![0.0; conv_len];
    let mut full_state = vec![0.0; recurrent_len];
    let mut states = vec![(vec![0.0; 24], vec![0.0; 8]); 4];
    let weights = (0..conv_len)
        .map(|i| 0.1 + i as f32 * 0.002)
        .collect::<Vec<_>>();
    let a_log = (0..8).map(|i| -0.5 + i as f32 * 0.05).collect::<Vec<_>>();
    let dt_bias = vec![0.2; 8];
    for token in 0..5 {
        let qkv = (0..channels)
            .map(|i| ((i + token * 3) as f32 * 0.17).sin())
            .collect::<Vec<_>>();
        let z = (0..16).map(|i| 0.2 + i as f32 * 0.03).collect::<Vec<_>>();
        let a = vec![0.1 * token as f32; 8];
        let b = vec![-0.2; 8];
        let expected = gated_delta_step(
            full_shape,
            GatedDeltaWeights {
                conv: &weights,
                a_log: &a_log,
                dt_bias: &dt_bias,
                norm: &[1.0, 0.8],
                norm_epsilon: 1e-6,
            },
            GatedDeltaInput {
                qkv: &qkv,
                z: &z,
                a: &a,
                b: &b,
            },
            &mut full_conv,
            &mut full_state,
        )
        .unwrap();
        let mut actual = Vec::new();
        for (lane, (conv, state)) in states.iter_mut().enumerate() {
            let HybridLocalAttention::GatedDelta {
                value_heads,
                qkv_channels,
                shape,
                ..
            } = &local.layers()[0].attention[lane]
            else {
                panic!()
            };
            let input = qkv_channels
                .iter()
                .flat_map(|r| qkv[r.clone()].iter().copied())
                .collect::<Vec<_>>();
            let w = qkv_channels
                .iter()
                .flat_map(|r| weights[r.start * 3..r.end * 3].iter().copied())
                .collect::<Vec<_>>();
            let output = gated_delta_step(
                *shape,
                GatedDeltaWeights {
                    conv: &w,
                    a_log: &a_log[value_heads.clone()],
                    dt_bias: &dt_bias[value_heads.clone()],
                    norm: &[1.0, 0.8],
                    norm_epsilon: 1e-6,
                },
                GatedDeltaInput {
                    qkv: &input,
                    z: &z[value_heads.start * 2..value_heads.end * 2],
                    a: &a[value_heads.clone()],
                    b: &b[value_heads.clone()],
                },
                conv,
                state,
            )
            .unwrap();
            actual.extend(output);
            close(
                state,
                &full_state[value_heads.start * 4..value_heads.end * 4],
            );
            let expected_conv = qkv_channels
                .iter()
                .flat_map(|r| full_conv[r.start * 3..r.end * 3].iter().copied())
                .collect::<Vec<_>>();
            close(conv, &expected_conv);
        }
        close(&actual, &expected);
    }
}
