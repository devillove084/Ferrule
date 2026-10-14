//! Complete-block CPU geometry tests; GPU integration is separately ignored.
#[allow(dead_code)]
#[path = "support/hybrid_tp_ep_fixture.rs"]
mod base;
use ferrule_common::{ParallelRankId, ParallelTopologyId};
use ferrule_model::transformer::{hybrid_parallel::*, *};
use std::sync::Arc;
fn parallel(
    r: &BoundDecoderResources,
    range: std::ops::Range<usize>,
    degree: usize,
) -> HybridParallelPlan {
    let experts = if r.spec().layers()[range.clone()]
        .iter()
        .any(|l| matches!(l.feed_forward(), FeedForward::Moe(_)))
    {
        HybridRoutedExpertExecution::LocalTensorShards
    } else {
        HybridRoutedExpertExecution::None
    };
    HybridParallelPlan::new_with_execution(
        r.spec(),
        range,
        HybridExpertScope {
            topology: ParallelTopologyId::new(907),
            replica: 0,
            stage: 0,
        },
        (0..degree).map(|i| ParallelRankId::new(i as u32)).collect(),
        experts,
        HybridRankPrecision::MixedBf16F32,
    )
    .unwrap()
}
#[test]
fn complete_block_uses_global_layer_local_kv_and_replicated_kv_heads() {
    let f = base::Fixture::for_degree(2, true);
    let r = f.resources();
    for layer in [1, 3] {
        let p = parallel(&r, layer..layer + 1, 2);
        for lane in 0..2 {
            let layout = Arc::new(p.rank_layout(lane).unwrap());
            let b = HybridMixedGqaBlockPlan::new(layout.clone(), 2, 1, 32, 2, 3).unwrap();
            assert_eq!(b.layer(), layer);
            assert_eq!(b.state_copies(), 3);
            assert_eq!(layout.state_schema().layer_range(), layer..layer + 1);
            assert_eq!(layout.state_schema().entries()[0].local_kv_layer, Some(0));
            let HybridLocalAttention::Gqa {
                query_heads,
                kv_heads,
                kv_owners,
                ..
            } = layout.attention(layer).unwrap()
            else {
                panic!()
            };
            assert_eq!(*query_heads, lane..lane + 1);
            assert_eq!(*kv_heads, 0..1);
            assert_eq!(kv_owners, &[ParallelRankId::new(0)]);
            assert_eq!(layout.state_budget(3).unwrap().total_grants, 0);
            use ferrule_common::execution::KvLayoutSchema;
            assert_eq!(
                layout.kv_planes(2, 32).unwrap().checked_page_bytes(),
                Some(64)
            );
            let sites = b.sites();
            assert_eq!(
                sites.iter().map(|s| s.semantic).collect::<Vec<_>>(),
                vec![
                    HybridCollectiveSemantic::AttentionOutput,
                    HybridCollectiveSemantic::DenseMlpDown
                ]
            );
            assert!(sites.iter().all(
                |s| s.kind == HybridCollectiveKind::ReduceSum && s.global_layer == Some(layer)
            ));
            assert_eq!(sites[0].shape.elements(1, 2).unwrap(), 16);
        }
    }
}
#[test]
fn complete_gqa_gdn_pair_admits_local_recurrent_state() {
    let f = base::Fixture::for_degree(2, true);
    let r = f.resources();
    let p = parallel(&r, 1..3, 2);
    for lane in 0..2 {
        let layout = Arc::new(p.rank_layout(lane).unwrap());
        let block =
            HybridMixedGqaBlockPlan::new_gqa_gdn_pair(layout.clone(), 2, 1, 32, 2, 3).unwrap();
        assert_eq!(block.layer(), 1);
        assert_eq!(block.recurrent_layer(), Some(2));
        let shape = block.recurrent_shape().unwrap().unwrap();
        assert_eq!(shape.key_heads, 1);
        assert_eq!(shape.value_heads, 2);
        assert_eq!(shape.key_dim, 3);
        assert_eq!(shape.value_dim, 2);
        assert!(block.recurrent_workspace_bytes().unwrap() > 0);
        let state = layout.state_budget(3).unwrap();
        assert_eq!(shape.sizes().unwrap(), (10, 30, 12));
        assert_eq!(state.per_sequence_bytes, 30 * 2 + 12 * 4);
        assert_eq!(state.total_grants, 3 * state.per_sequence_grants);
        assert!(state.copy_workspace_grants > 0);
        let HybridLocalAttention::GatedDelta {
            key_heads,
            value_heads,
            ..
        } = layout.attention(2).unwrap()
        else {
            panic!()
        };
        assert_eq!(*key_heads, lane..lane + 1);
        assert_eq!(*value_heads, lane * 2..lane * 2 + 2);
        use ferrule_backend::cuda_plan::GatedDeltaNetLayout;
        let native = GatedDeltaNetLayout {
            rows: 1,
            key_heads: 1,
            value_heads: 2,
            key_dim: 3,
            value_dim: 2,
        };
        let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
        let bytes = block.recurrent_workspace_bytes().unwrap();
        assert!(
            precision
                .gdn_chunk_plan(native, false, bytes)
                .unwrap()
                .is_some()
        );
        assert!(precision.gdn_chunk_plan(native, true, 0).unwrap().is_none());
        assert!(precision.gdn_chunk_plan(native, false, bytes - 1).is_err());
        assert_eq!(block.sites().len(), 4);
        assert_eq!(block.sites()[2].global_layer, Some(2));
    }
}

#[test]
fn complete_block_rejects_gdn_moe_multilayer_and_unbounded_views() {
    let f = base::Fixture::for_degree(2, true);
    let r = f.resources();
    for (range, degree) in [(0..2, 2), (1..2, 1)] {
        let p = parallel(&r, range, degree);
        assert!(
            HybridMixedGqaBlockPlan::new(Arc::new(p.rank_layout(0).unwrap()), 2, 1, 32, 2, 3)
                .is_err()
        );
    }
    let f = base::Fixture::for_degree(2, false);
    let r = f.resources();
    assert!(
        HybridMixedGqaBlockPlan::new(
            Arc::new(parallel(&r, 1..2, 2).rank_layout(0).unwrap()),
            2,
            1,
            32,
            2,
            3
        )
        .is_err()
    );
}
#[test]
fn complete_block_native_attention_views_are_rank_local_and_fail_closed() {
    use ferrule_backend::cuda_plan::bf16_attention::{
        Bf16AttentionGroup as G, Bf16AttentionShape as S,
    };
    let f = base::Fixture::for_degree(2, true);
    let r = f.resources();
    let b = HybridMixedGqaBlockPlan::new(
        Arc::new(parallel(&r, 3..4, 2).rank_layout(1).unwrap()),
        2,
        1,
        32,
        2,
        3,
    )
    .unwrap();
    let p = b
        .attention_plan(
            &[S { rows: 2, tokens: 2 }, S { rows: 1, tokens: 3 }],
            64 << 20,
            16 << 20,
        )
        .unwrap();
    assert!(
        p.view(&[G {
            row_start: 0,
            rows: 2,
            sequence: 0,
            tokens: 2
        }])
        .is_ok()
    );
    assert!(
        p.view(&[G {
            row_start: 0,
            rows: 1,
            sequence: 0,
            tokens: 4
        }])
        .is_err()
    );
    assert!(
        b.attention_plan(
            &[S {
                rows: 1,
                tokens: 33
            }],
            64 << 20,
            16 << 20
        )
        .is_err()
    );
    assert!(
        b.attention_plan(&[S { rows: 2, tokens: 2 }], 0, 16 << 20)
            .is_err()
    );
}
#[cfg(feature = "cuda")]
#[path = "support/hybrid_mixed_layer_gpu.rs"]
mod gpu;

#[test]
fn complete_pair_rejects_wrong_order_range_moe_and_capacity() {
    let f = base::Fixture::for_degree(2, true);
    let r = f.resources();
    for range in [0..2, 1..2, 0..4] {
        let p = parallel(&r, range, 2);
        assert!(
            HybridMixedGqaBlockPlan::new_gqa_gdn_pair(
                Arc::new(p.rank_layout(0).unwrap()),
                2,
                1,
                32,
                2,
                3
            )
            .is_err()
        );
    }
    let f = base::Fixture::for_degree(2, false);
    let r = f.resources();
    assert!(
        HybridMixedGqaBlockPlan::new_gqa_gdn_pair(
            Arc::new(parallel(&r, 1..3, 2).rank_layout(0).unwrap()),
            2,
            1,
            32,
            2,
            3
        )
        .is_err()
    );
}

#[test]
fn frozen_gdn_cuda_state_oracle_is_profile_matched_and_output_stable() {
    let cpu: serde_json::Value =
        serde_json::from_str(include_str!("support/hybrid_mixed_gdn_oracle.json")).unwrap();
    let cuda: serde_json::Value =
        serde_json::from_str(include_str!("support/hybrid_mixed_gdn_cuda_oracle.json")).unwrap();
    assert_eq!(cuda["schema"], "ferrule.mixed-gqa-gdn.tp2.cuda-ssm.v1");
    assert_eq!(cuda["ssm_dtype"], "f32");
    assert_eq!(cuda["policy"]["tf32"], false);
    assert_eq!(cuda["policy"]["bf16_reduced_precision_reduction"], false);
    assert_eq!(cuda["policy"]["deterministic"], true);
    assert_eq!(cuda["gpu_compute_capability"], serde_json::json!([8, 6]));
    assert_eq!(
        cuda["hf_cuda_source_pins"]["modeling_qwen3_5_moe"]["sha256"],
        "0f8ca77e875c31d448137c9ac291bed505c6b9927a85b5b8460af29d1e1ee647"
    );
    assert_eq!(
        cuda["cpu_fixture_sha256"],
        "98884445fb90c34747980f2c6a30161532b7a7ae59f7fe7cd218b8a3577cf2b7"
    );
    for case in ["prefill", "fresh_one", "cached_one"] {
        assert_eq!(cuda[case]["output"], cpu[case]["output"]);
        assert_eq!(cuda[case]["ranks"].as_array().unwrap().len(), 2);
        for lane in 0..2 {
            assert_eq!(
                cuda[case]["ranks"][lane]["conv"],
                cpu[case]["ranks"][lane]["conv"]
            );
            assert_eq!(
                cuda[case]["ranks"][lane]["ssm"].as_array().unwrap().len(),
                12
            );
        }
    }
}

#[test]
fn frozen_gdn_pair_oracle_has_native_local_state_and_distinct_history_cases() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("support/hybrid_mixed_gdn_oracle.json")).unwrap();
    assert_eq!(oracle["conv_dtype"], "bf16");
    assert_eq!(oracle["ssm_dtype"], "f32");
    for (case, rows) in [("prefill", 2), ("fresh_one", 1), ("cached_one", 1)] {
        assert_eq!(oracle[case]["output"].as_array().unwrap().len(), rows * 8);
        assert_eq!(oracle[case]["ranks"].as_array().unwrap().len(), 2);
        for rank in 0..2 {
            assert_eq!(
                oracle[case]["ranks"][rank]["conv"]
                    .as_array()
                    .unwrap()
                    .len(),
                30
            );
            assert_eq!(
                oracle[case]["ranks"][rank]["ssm"].as_array().unwrap().len(),
                12
            );
        }
        assert_ne!(oracle[case]["ranks"][0], oracle[case]["ranks"][1]);
    }
    assert_ne!(oracle["fresh_one"]["ranks"], oracle["cached_one"]["ranks"]);
}

struct BoundCollective {
    owner: ParallelRankId,
    members: Vec<ParallelRankId>,
    binding: Option<HybridBf16CollectiveBinding>,
    route_sites: Vec<HybridMixedRouteSite>,
}
impl HybridBf16Collective for BoundCollective {
    fn owner(&self) -> ParallelRankId {
        self.owner
    }
    fn members(&self) -> &[ParallelRankId] {
        &self.members
    }
    fn sealed_route_sites(&self) -> ferrule_common::Result<Vec<HybridMixedRouteSite>> {
        Ok(self.route_sites.clone())
    }
    fn sealed_binding(&self) -> ferrule_common::Result<HybridBf16CollectiveBinding> {
        self.binding
            .clone()
            .ok_or_else(|| ferrule_common::Error::Model {
                message: "unsealed".into(),
            })
    }
    fn exchange_bf16(
        &mut self,
        _: ferrule_common::execution::ExecutionTransactionId,
        _: u64,
        _: HybridCollectiveKind,
        _: Vec<u16>,
    ) -> ferrule_common::Result<Vec<u16>> {
        panic!("metadata preflight must not execute or append KV")
    }
    fn abort(&mut self) {}
}
#[test]
fn sealed_block_binding_rejects_every_identity_and_schedule_mismatch_before_enter() {
    let fixture = base::Fixture::for_degree(2, true);
    let resources = fixture.resources();
    let parallel = parallel(&resources, 1..3, 2);
    let plan = HybridMixedGqaBlockPlan::new_gqa_gdn_pair(
        Arc::new(parallel.rank_layout(0).unwrap()),
        2,
        1,
        32,
        2,
        3,
    )
    .unwrap();
    let expected = plan.collective_binding();
    let peer = |binding| BoundCollective {
        owner: ParallelRankId::new(0),
        members: expected.members.clone(),
        binding,
        route_sites: Vec::new(),
    };
    plan.validate_collective(&peer(Some(expected.clone())))
        .unwrap();
    for mismatch in 0..14 {
        let mut changed = expected.clone();
        match mismatch {
            0 => changed.sites[0].global_layer = Some(3), // same dimensions, wrong global layer
            1 => changed.sites[0].semantic = HybridCollectiveSemantic::DenseMlpDown,
            2 => changed.sites[0].ordinal += 4,
            3 => changed.sites.swap(0, 1),
            4 => changed.source = ParallelRankId::new(1),
            5 => changed.members.reverse(),
            6 => changed.epoch = ParallelTopologyId::new(908),
            7 => changed.sites[0].kind = HybridCollectiveKind::AllGather,
            8 => changed.sites[0].shape.per_row += 1,
            9 => {
                changed.sites.pop();
            }
            10 => changed.version += 1,
            11 => changed.precision = HybridCollectivePrecision::F32,
            12 => changed.max_rows += 1,
            13 => changed.max_sequences += 1,
            _ => unreachable!(),
        }
        let mut physical_enters = 0;
        let result = plan
            .validate_collective(&peer(Some(changed)))
            .map(|()| physical_enters += 1);
        assert!(result.is_err(), "mismatch {mismatch} was admitted");
        assert_eq!(physical_enters, 0);
    }
    assert!(plan.validate_collective(&peer(None)).is_err());
    let mut foreign = peer(Some(expected.clone()));
    foreign.owner = ParallelRankId::new(1);
    assert!(plan.validate_collective(&foreign).is_err());
    foreign.owner = ParallelRankId::new(0);
    foreign.members.reverse();
    assert!(plan.validate_collective(&foreign).is_err());
}

#[test]
fn runtime_adapter_exposes_authenticated_binding_before_model_preparation() {
    use ferrule_common::ParallelGroupId;
    use ferrule_runtime::parallel::{
        collective::HostCollectiveLimits,
        tensor::decoder_collective::{
            DecoderTensorCollective, MixedRankCollective, MixedRankCollectivePlan,
        },
    };
    for dense in [true, false] {
        let fixture = base::Fixture::for_degree(2, dense);
        let resources = fixture.resources();
        let parallel = parallel(&resources, 1..3, 2);
        let plan = HybridMixedRankProgramPlan::from_layout(
            Arc::new(parallel.rank_layout(0).unwrap()),
            2,
            1,
            32,
            2,
            3,
        )
        .unwrap();
        let sealed = Arc::new(
            MixedRankCollectivePlan::from_hybrid_sites(
                &parallel,
                ParallelGroupId::new(9),
                ParallelRankId::new(0),
                HYBRID_AGREEMENT_VERSION,
                &plan.sites(),
                HostCollectiveLimits {
                    max_ranks: 2,
                    max_elements_per_rank: 32,
                    max_host_bytes: 4096,
                },
                1,
                2,
            )
            .unwrap(),
        );
        let mut endpoints = DecoderTensorCollective::new_group_from_members(
            sealed.members(),
            sealed.epoch(),
            sealed.group(),
            sealed.limits(),
            std::time::Duration::from_secs(1),
        )
        .unwrap()
        .into_iter();
        let transport =
            MixedRankCollective::prepare(endpoints.next().unwrap(), ParallelRankId::new(0), sealed)
                .unwrap();
        assert_eq!(
            transport.sealed_binding().unwrap(),
            plan.collective_binding()
        );
        if dense {
            plan.validate_collective(&transport).unwrap();
        } else {
            assert!(transport.sealed_route_sites().unwrap().is_empty());
            assert!(!plan.route_sites().is_empty());
            assert!(
                plan.validate_collective(&transport).is_err(),
                "unextended runtime must reject before upload"
            );
        }
    }
}

#[test]
fn generic_program_lowers_every_bound_layer_and_distinct_state_slot() {
    use ferrule_common::execution::KvLayoutSchema;
    let fixture = base::Fixture::for_degree(2, true);
    let resources = fixture.resources();
    for range in [0..2, 0..4, 1..4, 2..4] {
        let topology = parallel(&resources, range.clone(), 2);
        for lane in 0..2 {
            let layout = Arc::new(topology.rank_layout(lane).unwrap());
            let plan =
                HybridMixedRankProgramPlan::from_layout(layout.clone(), 2, 1, 32, 2, 3).unwrap();
            assert_eq!(plan.layers(), range);
            let sites = plan.sites();
            assert_eq!(sites.len(), range.len() * 2);
            for (index, site) in sites.iter().enumerate() {
                assert_eq!(site.ordinal, index as u64);
                assert_eq!(site.global_layer, Some(range.start + index / 2));
                assert_eq!(site.shape.per_row, 8);
            }
            let recurrent = plan.recurrent_shapes().unwrap();
            let expected = range
                .clone()
                .filter(|layer| layer % 2 == 0)
                .collect::<Vec<_>>();
            assert_eq!(
                recurrent
                    .iter()
                    .map(|(layer, _)| *layer)
                    .collect::<Vec<_>>(),
                expected
            );
            for (_, shape) in &recurrent {
                assert_eq!(shape.sizes().unwrap(), (10, 30, 12));
            }
            assert_eq!(
                layout.state_budget(3).unwrap().per_sequence_bytes,
                expected.len() * (30 * 2 + 12 * 4)
            );
            let gqa_count = range.clone().filter(|layer| layer % 2 == 1).count();
            if gqa_count != 0 {
                assert_eq!(
                    layout.kv_planes(2, 32).unwrap().checked_page_bytes(),
                    Some(gqa_count * 64)
                );
            }
            let single = HybridMixedRankProgramPlan::from_layout(
                Arc::new(parallel(&resources, 0..2, 2).rank_layout(lane).unwrap()),
                2,
                1,
                32,
                2,
                3,
            )
            .unwrap();
            assert_eq!(
                plan.recurrent_workspace_bytes().unwrap(),
                single.recurrent_workspace_bytes().unwrap()
            );
        }
    }
}

#[test]
fn generic_routed_program_requires_authenticated_route_sites() {
    let fixture = base::Fixture::for_degree(2, false);
    let resources = fixture.resources();
    let parallel = parallel(&resources, 0..4, 2);
    let plan = HybridMixedRankProgramPlan::from_layout(
        Arc::new(parallel.rank_layout(0).unwrap()),
        2,
        1,
        32,
        2,
        3,
    )
    .unwrap();
    let mut transport = BoundCollective {
        owner: ParallelRankId::new(0),
        members: parallel.tensor_members().to_vec(),
        binding: Some(plan.collective_binding()),
        route_sites: Vec::new(),
    };
    assert!(plan.validate_collective(&transport).is_err());
    transport.route_sites = plan.route_sites();
    plan.validate_collective(&transport).unwrap();
    assert_eq!(plan.route_sites().len(), 3);
    for site in plan.route_sites() {
        assert_eq!(
            plan.sites()[site.output_site as usize].semantic,
            HybridCollectiveSemantic::RoutedExpertOutput
        );
        assert_eq!(
            plan.sites()
                .iter()
                .filter(|s| s.global_layer == Some(site.layer))
                .count(),
            2
        );
    }
    for field in 0..5 {
        transport.route_sites = plan.route_sites();
        match field {
            0 => transport.route_sites[0].layer += 1,
            1 => transport.route_sites[0].experts += 1,
            2 => transport.route_sites[0].top_k += 1,
            3 => transport.route_sites[0].output_site += 1,
            _ => transport.route_sites[0].max_rows += 1,
        }
        assert!(plan.validate_collective(&transport).is_err());
    }
}

#[test]
fn typed_route_control_preserves_ids_and_f32_bits_and_rejects_invalid_shapes() {
    let site = HybridMixedRouteSite {
        layer: 19,
        output_site: 7,
        experts: 512,
        top_k: 2,
        max_rows: 2,
        accumulation: HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
        combine: HybridMoeCombineOrder::LocalBf16AddThenReduce,
    };
    let routes = RouterRoutes::new(
        1,
        2,
        vec![511, 257],
        vec![f32::from_bits(0x3f000001), f32::from_bits(0x3effffff)],
    )
    .unwrap();
    let identity = HybridMixedRouteAgreement::new(site.clone(), &routes).unwrap();
    assert_eq!(identity.expert_ids(), &[511, 257]);
    assert_eq!(identity.weight_bits(), &[0x3f000001, 0x3effffff]);
    for ids in [vec![512, 0], vec![1, 1]] {
        assert!(
            HybridMixedRouteAgreement::new(
                site.clone(),
                &RouterRoutes::new(1, 2, ids, vec![0.5; 2]).unwrap()
            )
            .is_err()
        );
    }
    let mut bad = site.clone();
    bad.top_k = 1;
    assert!(HybridMixedRouteAgreement::new(bad, &routes).is_err());
    let mut bad = site;
    bad.max_rows = 0;
    assert!(HybridMixedRouteAgreement::new(bad, &routes).is_err());
}

#[test]
#[cfg(feature = "cuda")]
fn generic_program_operation_estimate_is_complete_and_one_byte_strict() {
    let fixture = base::Fixture::for_degree(2, true);
    let resources = gpu::native_resources(&fixture);
    let topology = parallel(&resources, 0..4, 2);
    let layout = Arc::new(topology.rank_layout(0).unwrap());
    let plan = HybridMixedRankProgramPlan::from_layout(layout, 2, 1, 32, 2, 3).unwrap();
    let estimate = plan.operation_estimate(&resources, 16).unwrap();
    assert!(estimate.storage_grants > 0);
    assert!(estimate.gemmex_storage_grants > 0);
    assert!(estimate.attention_storage_grants > 0);
    assert!(estimate.derived_rope_grants > 0);
    assert!(estimate.recurrent_workspace_grants > 0);
    assert!(estimate.operation_grants > 0);
    let total = estimate.total_device_grants().unwrap();
    assert!(total > estimate.storage_grants);
    assert!(estimate.require_device_capacity(total, 0).is_ok());
    assert!(estimate.require_device_capacity(total - 1, 0).is_err());
    assert!(estimate.require_device_capacity(total - 1, 1).is_err());
    assert!(estimate.require_device_capacity(total, 1).is_err());
    assert!(estimate.require_device_capacity(total, usize::MAX).is_err());
    let storage = plan
        .layout()
        .storage_budget(&resources, plan.segment(), 2, 32, 16, 3)
        .unwrap();
    assert_eq!(estimate.storage_grants, storage.storage_grants().unwrap());
    assert_eq!(
        estimate.state_copy_workspace_grants,
        storage.state.copy_workspace_grants
    );
    assert_eq!(estimate.gemmex_provider_bytes, 16 << 20);
    assert_eq!(estimate.attention_provider_bytes, 16 << 20);
    // Header-only estimates must not open or re-materialize checkpoint payloads.
    std::fs::remove_file(fixture.dir.join("mixed-layer.bf16")).unwrap();
    assert_eq!(plan.operation_estimate(&resources, 16).unwrap(), estimate);
}

#[test]
#[ignore = "explicit real checkpoint header path and hidden GPUs; no payload materialization"]
fn real_tp2_mixed_program_header_operation_estimate() {
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    let path = std::env::var("FERRULE_MIXED_HEADER_DIR").unwrap();
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(std::path::Path::new(&path))
            .unwrap();
    let layers = resources.spec().layers().len();
    let topology = parallel(&resources, 0..layers, 2);
    for lane in 0..2 {
        let program = HybridMixedRankProgramPlan::for_segment(
            Arc::new(topology.rank_layout(lane).unwrap()),
            LayerSegmentPlan::new(layers, 0..layers, true, true).unwrap(),
            2,
            2,
            1024,
            16,
            5,
        )
        .unwrap();
        let estimate = program.operation_estimate(&resources, 128).unwrap();
        let storage = program
            .layout()
            .storage_budget(&resources, program.segment(), 16, 1024, 128, 5)
            .unwrap();
        assert_eq!(storage.weight_grants, 18_390_764_032);
        assert_eq!(storage.state.total_grants, 162_201_600);
        assert_eq!(storage.kv_grants, 20_971_520);
        assert_eq!(program.recurrent_shapes().unwrap().len(), 30);
        assert_eq!(program.sites().len(), 81);
        assert_eq!(program.route_sites().len(), 40);
        let total = estimate.total_device_grants().unwrap();
        assert!(estimate.require_device_capacity(total - 1, 0).is_err());
        #[cfg(feature = "cuda")]
        {
            use ferrule_backend::cuda_plan::bf16_attention::{
                Bf16AttentionGroup as G, Bf16AttentionShape as S,
            };
            let shapes = [
                S { rows: 1, tokens: 1 },
                S { rows: 2, tokens: 2 },
                S {
                    rows: 1,
                    tokens: 996,
                },
                S {
                    rows: 2,
                    tokens: 996,
                },
            ];
            let (claim, context) = program
                .cuda_claim_inputs(&resources, 128, lane, 2 << 30, &shapes)
                .unwrap();
            context.validate_estimate(claim).unwrap();
            assert_eq!(claim.weight_bytes_upper_bound, storage.weight_grants);
            assert_eq!(
                claim.per_sequence_state_bytes,
                storage.state.per_sequence_grants
            );
            assert_eq!(claim.kv_bytes, storage.kv_grants);
            assert_eq!(
                claim.workspace_bytes_upper_bound,
                total - storage.storage_grants().unwrap()
            );
            let attention = program
                .attention_plan(
                    &shapes,
                    estimate.attention_storage_grants,
                    estimate.attention_provider_bytes,
                )
                .unwrap();
            attention
                .view(&[
                    G {
                        row_start: 0,
                        rows: 1,
                        sequence: 0,
                        tokens: 996,
                    },
                    G {
                        row_start: 1,
                        rows: 1,
                        sequence: 1,
                        tokens: 996,
                    },
                ])
                .unwrap();
            for (_, local) in program.recurrent_shapes().unwrap() {
                assert_eq!(
                    (local.key_dim, local.value_dim, local.value_heads),
                    (128, 128, 16)
                );
                let fresh = ferrule_backend::cuda_plan::Bf16ChunkPlan::new_prefill(
                    ferrule_backend::cuda_plan::GatedDeltaNetLayout {
                        rows: 1,
                        key_heads: local.key_heads,
                        value_heads: local.value_heads,
                        key_dim: local.key_dim,
                        value_dim: local.value_dim,
                    },
                    usize::MAX,
                )
                .unwrap();
                assert!(
                    fresh.workspace_elements() * 4 <= program.recurrent_workspace_bytes().unwrap()
                );
            }
            println!(
                "MODEL_HEADER_CLAIM lane={lane} geometry={:?} estimate={claim:?}",
                program
                    .layout()
                    .attention(program.gqa_layer().unwrap())
                    .unwrap()
            );
        }
        println!("MODEL_HEADER_ESTIMATE lane={lane} total={total} estimate={estimate:?}");
        println!(
            "MODEL_HEADER_22GIB_WITH_128M_MARGIN lane={lane} result={:?}",
            estimate.require_device_capacity(22 << 30, 128 << 20)
        );
    }
}

#[test]
fn generic_segment_endpoints_seal_only_the_owned_vocab_gather() {
    let fixture = base::Fixture::for_degree(2, true);
    let resources = fixture.resources();
    for (range, embedding, output) in [
        (0..4, true, true),
        (0..2, true, false),
        (2..4, false, true),
        (1..3, false, false),
    ] {
        let topology = parallel(&resources, range.clone(), 2);
        for lane in 0..2 {
            let layout = Arc::new(topology.rank_layout(lane).unwrap());
            let segment = LayerSegmentPlan::new(4, range.clone(), embedding, output).unwrap();
            let plan = HybridMixedRankProgramPlan::for_segment(
                layout.clone(),
                segment.clone(),
                2,
                1,
                32,
                2,
                3,
            )
            .unwrap();
            assert_eq!(plan.segment(), &segment);
            let sites = plan.sites();
            assert_eq!(sites.len(), 2 * range.len() + usize::from(output));
            assert!(sites.iter().all(|site| !matches!(
                site.semantic,
                HybridCollectiveSemantic::Agreement | HybridCollectiveSemantic::Completion
            )));
            if output {
                let site = sites.last().unwrap();
                assert_eq!(site.semantic, HybridCollectiveSemantic::OutputHead);
                assert_eq!(site.kind, HybridCollectiveKind::AllGather);
                assert_eq!(site.global_layer, None);
                assert_eq!(
                    site.shape.per_row,
                    resources.spec().vocab_size().div_ceil(2)
                );
            }
            let wrong_range = LayerSegmentPlan::new(4, 1..2, false, false).unwrap();
            assert!(
                HybridMixedRankProgramPlan::for_segment(layout, wrong_range, 2, 1, 32, 2, 3,)
                    .is_err()
            );
        }
    }
}
