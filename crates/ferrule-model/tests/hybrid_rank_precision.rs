//! Model-only CPU planning/materialization tests; never discover or open CUDA.
use ferrule_common::{ParallelRankId, ParallelTopologyId, execution::KvLayoutSchema};
use ferrule_model::{
    TensorRole, checkpoint::*, nn::*, transformer::hybrid_parallel::*, transformer::*,
};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

fn members(degree: usize) -> Vec<ParallelRankId> {
    [7, 19, 31, 63][..degree]
        .iter()
        .copied()
        .map(ParallelRankId::new)
        .collect()
}
fn scope() -> HybridExpertScope {
    HybridExpertScope {
        topology: ParallelTopologyId::new(907),
        replica: 1,
        stage: 0,
    }
}
fn spec() -> DecoderModelSpec {
    let norm = || RmsNorm::new(256, 1e-6).unwrap();
    let rope = RotaryEmbedding::new(
        16,
        10000.,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 16 },
        RotaryScaling::None,
    )
    .unwrap();
    let moe = Moe::new(
        256,
        256,
        MoeRouterSpec::new(
            4,
            2,
            RouterScoreFunction::Softmax,
            RouterSelection::TopK,
            true,
            1.,
        )
        .unwrap(),
        false,
    )
    .unwrap()
    .with_shared_expert(SwiGlu::new(256, 256, false).unwrap())
    .unwrap()
    .with_shared_expert_gate(Linear::new(256, 1, false).unwrap())
    .unwrap();
    let layers = [
        Attention::GatedDeltaNet(
            GatedDeltaNetAttention::new(256, 4, 8, 16, 16, 3, 1e-6, false).unwrap(),
        ),
        Attention::Gqa(GqaAttention::new(256, 4, 2, 16, false, rope).unwrap()),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, a)| {
        DecoderLayer::new(
            i,
            norm(),
            a,
            Residual::Add,
            norm(),
            FeedForward::Moe(moe.clone()),
            Residual::Add,
        )
        .unwrap()
    })
    .collect();
    DecoderModelSpec::new(DecoderModelParts {
        architecture: "rank-precision-test".into(),
        hidden_size: 256,
        vocab_size: 256,
        max_sequence_length: Some(32),
        token_embedding: Embedding::new(256, 256, None).unwrap(),
        layers,
        final_norm: norm(),
        output: Linear::new(256, 256, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap()
}
fn plan(precision: HybridRankPrecision, degree: usize) -> HybridParallelPlan {
    HybridParallelPlan::new_with_execution(
        &spec(),
        0..2,
        scope(),
        members(degree),
        HybridRoutedExpertExecution::LocalTensorShards,
        precision,
    )
    .unwrap()
}
fn attachment() -> HybridExpertAttachment {
    HybridExpertAttachment {
        source_rank: members(2)[1],
        members: vec![ParallelRankId::new(101)],
        placement: ExpertPlacement::new(
            (0..2).flat_map(|l| (0..4).map(move |e| (l, e, ParallelRankId::new(101)))),
        )
        .unwrap(),
    }
}
fn grant(bytes: usize) -> usize {
    ferrule_backend::cuda_allocation_grant_bytes(bytes).unwrap()
}

struct Fixture {
    path: PathBuf,
    binding: BoundParameter,
    bound: Arc<BoundStateDict>,
}
impl Fixture {
    fn new(
        role: TensorRole,
        shape: &[usize],
        residency: ParameterResidency,
        dtype: CheckpointDType,
        scale_dtype: Option<CheckpointDType>,
    ) -> Self {
        Self::with_data(role, shape, residency, dtype, scale_dtype, 1, None)
    }
    #[allow(clippy::too_many_arguments)]
    fn with_data(
        role: TensorRole,
        shape: &[usize],
        residency: ParameterResidency,
        dtype: CheckpointDType,
        scale_dtype: Option<CheckpointDType>,
        id: u64,
        data: Option<Vec<u8>>,
    ) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrule-rank-precision-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let elements = shape.iter().product::<usize>();
        let mut bytes = Vec::new();
        for i in 0..elements {
            match dtype {
                CheckpointDType::Bf16 => {
                    bytes.extend([0x8000_u16, 0x3f81, 0x7fc1, 0xbf83][i % 4].to_le_bytes())
                }
                CheckpointDType::F32 => bytes.extend((1.003_f32 + (i % 7) as f32).to_le_bytes()),
                CheckpointDType::F8E4M3 => bytes.push(0x38),
                _ => panic!("fixture dtype"),
            }
        }
        let weight = CheckpointTensorSlice {
            name: "weight".into(),
            role: role.clone(),
            path: path.clone(),
            offset: 0,
            bytes: bytes.len() as u64,
            shape: shape.to_vec(),
            dtype: dtype.clone(),
        };
        let module = ModulePath::new("projection").unwrap();
        let mut parameter = ParameterSpec::new(
            ParameterId::new(id),
            module.clone(),
            match dtype {
                CheckpointDType::Bf16 => ParameterDType::Bf16,
                CheckpointDType::F32 => ParameterDType::F32,
                _ => ParameterDType::F8E4M3,
            },
            shape.to_vec(),
            residency,
        )
        .unwrap();
        let mut slices = vec![weight];
        let mut mapper = ExactNameMapper::new();
        mapper
            .insert("weight", NameMapping::weight(module.clone()))
            .unwrap();
        if let Some(dtype) = scale_dtype {
            let shape = shape.iter().map(|n| n.div_ceil(128)).collect::<Vec<_>>();
            let offset = bytes.len();
            for _ in 0..shape.iter().product::<usize>() {
                if dtype == CheckpointDType::Bf16 {
                    bytes.extend(0x3f80_u16.to_le_bytes());
                } else {
                    bytes.extend(1_f32.to_le_bytes());
                }
            }
            parameter = parameter
                .with_required_scale(
                    if dtype == CheckpointDType::Bf16 {
                        ParameterDType::Bf16
                    } else {
                        ParameterDType::F32
                    },
                    shape.clone(),
                )
                .unwrap();
            slices.push(CheckpointTensorSlice {
                name: "scale".into(),
                role: role.clone(),
                path: path.clone(),
                offset: offset as u64,
                bytes: (bytes.len() - offset) as u64,
                shape,
                dtype,
            });
            mapper
                .insert("scale", NameMapping::scale(module.clone()))
                .unwrap();
        }
        if let Some(data) = data {
            assert_eq!(data.len(), bytes.len());
            bytes = data;
        }
        std::fs::write(&path, bytes).unwrap();
        let mut schema = StateDictSchema::builder();
        schema.register_with_role(parameter, role).unwrap();
        let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
            .bind_slices(slices)
            .unwrap();
        Self {
            path,
            binding: bound.get(&module).unwrap().clone(),
            bound: Arc::new(bound),
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[test]
fn mixed_rank_full40_tp2_geometry_is_not_execution_permission() {
    use ferrule_model::models::qwen35::{Qwen35Config, Qwen35Recipe};
    let json = serde_json::from_str(include_str!("qwen35_35b_fp8_config.json")).unwrap();
    let s = Qwen35Recipe::spec(&Qwen35Config::from_value(&json).unwrap()).unwrap();
    let p = HybridParallelPlan::new_with_execution(
        &s,
        0..s.layers().len(),
        scope(),
        members(2),
        HybridRoutedExpertExecution::LocalTensorShards,
        HybridRankPrecision::MixedBf16F32,
    )
    .unwrap();
    p.validate_spec(&s).unwrap();
    assert_eq!(p.layers().len(), 40);
    assert_eq!(p.physical_members().len(), 2);
    assert!(p.expert_attachment().is_none());
    assert!(p.validate_rank_execution().is_err());
    assert!(
        p.collective_schedule(&LayerSegmentPlan::new(40, 0..40, true, true).unwrap())
            .is_err()
    );
    for lane in 0..2 {
        let layout = p.rank_layout(lane).unwrap();
        assert_eq!(layout.state_schema().layers().len(), 40);
        assert_eq!(layout.state_schema().layer_range(), 0..40);
        assert_eq!(
            layout
                .state_schema()
                .layers()
                .iter()
                .filter(|l| matches!(
                    l,
                    ferrule_model::decoder::HybridLayerSchema::FullAttention { .. }
                ))
                .count(),
            10
        );
        for (layer, local) in s.layers().iter().zip(p.layers()) {
            match (layer.attention(), &local.attention[lane]) {
                (Attention::GatedDeltaNet(g), HybridLocalAttention::GatedDelta { shape, .. }) => {
                    assert_eq!(shape.key_heads * 2, g.num_key_heads());
                    assert_eq!(shape.value_heads * 2, g.num_value_heads());
                }
                (
                    Attention::Gqa(g),
                    HybridLocalAttention::Gqa {
                        query_heads,
                        kv_heads,
                        ..
                    },
                ) => {
                    assert_eq!(query_heads.len() * 2, g.num_heads());
                    assert_eq!(kv_heads.len() * 2, g.num_kv_heads());
                }
                _ => panic!("descriptor-driven rank geometry"),
            }
        }
    }
}

#[test]
fn mixed_rank_state_kv_and_cow_charge_physical_native_roots() {
    let f32 = plan(HybridRankPrecision::F32, 2).rank_layout(1).unwrap();
    let mixed = plan(HybridRankPrecision::MixedBf16F32, 2)
        .rank_layout(1)
        .unwrap();
    assert_ne!(f32, mixed);
    let mut bytes = 0;
    let mut grants = 0;
    let mut workspace = 0;
    for l in mixed.state_schema().layers() {
        if let ferrule_model::decoder::HybridLayerSchema::GatedDeltaNet(s) = l {
            let (_, c, r) = s.sizes().unwrap();
            bytes += 2 * c + 4 * r;
            grants += grant(2 * c) + grant(4 * r);
            workspace = workspace.max(grant(2 * c));
        }
    }
    let budget = mixed.state_budget(5).unwrap();
    assert_eq!(budget.per_sequence_bytes, bytes);
    assert_eq!(budget.per_sequence_grants, grants);
    assert_eq!(budget.total_grants, 5 * grants);
    assert_eq!(budget.copy_workspace_grants, workspace);
    assert_eq!(mixed.state_bytes().unwrap(), bytes);
    assert!(mixed.state_budget(0).is_err());
    assert!(mixed.state_budget(usize::MAX).is_err());
    assert_eq!(
        f32.state_bytes().unwrap(),
        f32.state_schema().per_sequence_state_bytes().unwrap()
    );
    assert_eq!(
        f32.kv_planes(16, 32).unwrap().checked_page_bytes(),
        mixed
            .kv_planes(16, 32)
            .unwrap()
            .checked_page_bytes()
            .map(|n| n * 2)
    );
    assert_eq!(mixed.precision().ssm_dtype(), CheckpointDType::F32);
    assert_eq!(
        mixed.precision().collective(),
        HybridCollectivePrecision::Bf16PartialsAndReduction
    );
}

#[test]
fn local_routed_gate_up_and_down_are_tensor_shards_not_expert_partitions() {
    for role in [
        TensorRole::RoutedExpertGate,
        TensorRole::RoutedExpertUp,
        TensorRole::RoutedExpertDown,
    ] {
        for expert in 0..4 {
            let f = Fixture::new(
                role.clone(),
                &[256, 256],
                ParameterResidency::Expert { layer: 0, expert },
                CheckpointDType::F8E4M3,
                Some(CheckpointDType::Bf16),
            );
            let p = plan(HybridRankPrecision::MixedBf16F32, 2);
            for lane in 0..2 {
                let layout = Arc::new(p.rank_layout(lane).unwrap());
                let n = NumericFp8RankPlan::new(&f.binding, layout.clone()).unwrap();
                assert_eq!(n.budget().weight_bytes, 256 * 128);
                assert_eq!(n.budget().scale_bytes, 4);
                let down = role == TensorRole::RoutedExpertDown;
                assert_eq!(
                    n.partition().rows,
                    vec![if down {
                        0..256
                    } else {
                        lane * 128..(lane + 1) * 128
                    }]
                );
                assert_eq!(
                    n.partition().columns,
                    if down {
                        lane * 128..(lane + 1) * 128
                    } else {
                        0..256
                    }
                );
                assert_eq!(n.partition().collective.is_some(), down);
                let b = layout.parameter_budget(&f.binding).unwrap();
                assert_eq!(b.allocation_roots, 2);
                assert_eq!(b.allocation_grant_bytes, grant(256 * 128) + grant(4));
                let reader = StateDictMaterializer::new(b.source_bytes as u64).unwrap();
                let prepared = reader
                    .prepare_numeric_fp8_rank_linear(&n, b.allocation_grant_bytes)
                    .unwrap();
                assert_eq!(
                    prepared.segments()[0]
                        .artifact()
                        .provenance()
                        .weight_read()
                        .columns(),
                    n.partition().columns
                );
                assert_eq!(
                    reader.checkpoint_read_counters().bytes_read,
                    b.source_bytes as u64
                );
            }
            let external =
                HybridParallelPlan::new(&spec(), 0..2, scope(), members(2), Some(attachment()))
                    .unwrap();
            assert!(
                external
                    .rank_layout(0)
                    .unwrap()
                    .parameter_partition(&f.binding)
                    .is_err()
            );
        }
    }
}

#[test]
fn mixed_rank_numeric_origins_and_scale_dtype_are_not_relabelled() {
    let p = plan(HybridRankPrecision::MixedBf16F32, 2);
    let layout = Arc::new(p.rank_layout(1).unwrap());
    let f = Fixture::new(
        TensorRole::LinearAttentionQkv,
        &[256, 256],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F8E4M3,
        Some(CheckpointDType::Bf16),
    );
    let n = NumericFp8RankPlan::new(&f.binding, layout.clone()).unwrap();
    assert_eq!(n.partition().rows, vec![32..64, 96..128, 192..256]);
    assert_eq!(n.reads().len(), 3);
    let budget = layout.parameter_budget(&f.binding).unwrap();
    assert_eq!(budget.allocation_roots, 6);
    assert_eq!(budget.scale_bytes, 12); // intersecting scale blocks remain per-segment resources
    let f = Fixture::new(
        TensorRole::LinearAttentionQkv,
        &[256, 256],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F8E4M3,
        Some(CheckpointDType::F32),
    );
    assert!(NumericFp8RankPlan::new(&f.binding, layout).is_err());
    assert!(
        NumericFp8RankPlan::new(
            &f.binding,
            Arc::new(plan(HybridRankPrecision::F32, 2).rank_layout(1).unwrap())
        )
        .is_ok()
    );
}

#[test]
fn f32_tp2_tp4_legacy_schedule_and_explicit_ownership_are_unchanged() {
    for degree in [2, 4] {
        let a = attachment();
        let legacy =
            HybridParallelPlan::new(&spec(), 0..2, scope(), members(degree), Some(a.clone()))
                .unwrap();
        let explicit = HybridParallelPlan::new_with_execution(
            &spec(),
            0..2,
            scope(),
            members(degree),
            HybridRoutedExpertExecution::ExternalExpertAttachment(a),
            HybridRankPrecision::F32,
        )
        .unwrap();
        assert_eq!(legacy, explicit);
        let segment = LayerSegmentPlan::new(2, 0..2, true, true).unwrap();
        assert_eq!(
            legacy.collective_schedule(&segment).unwrap().encode(),
            explicit.collective_schedule(&segment).unwrap().encode()
        );
        assert_eq!(
            legacy
                .collective_schedule(&segment)
                .unwrap()
                .agreement_version(),
            2
        );
        assert!(legacy.validate_rank_execution().is_ok());
        assert!(
            plan(HybridRankPrecision::F32, degree)
                .validate_rank_execution()
                .is_err()
        );
    }
}

#[test]
fn mixed_rank_native_budget_rejects_implicit_f32_mirrors() {
    let layout = plan(HybridRankPrecision::MixedBf16F32, 2)
        .rank_layout(0)
        .unwrap();
    let f = Fixture::new(
        TensorRole::SharedExpertGate,
        &[256, 256],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::Bf16,
        None,
    );
    assert_eq!(
        layout.parameter_budget(&f.binding).unwrap().weight_bytes,
        128 * 256 * 2
    );
    let f = Fixture::new(
        TensorRole::SharedExpertGate,
        &[256, 256],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F32,
        None,
    );
    assert!(layout.parameter_budget(&f.binding).is_err());
    let f = Fixture::new(
        TensorRole::LinearAttentionALog,
        &[8],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F32,
        None,
    );
    let budget = layout.parameter_budget(&f.binding).unwrap();
    assert_eq!((budget.source_bytes, budget.weight_bytes), (16, 8));
}

#[test]
fn mixed_rank_storage_ledger_counts_physical_kv_cow_slots_and_local_experts() {
    let f = Fixture::new(
        TensorRole::RoutedExpertDown,
        &[256, 256],
        ParameterResidency::Expert {
            layer: 0,
            expert: 3,
        },
        CheckpointDType::F8E4M3,
        Some(CheckpointDType::Bf16),
    );
    let resources = BoundDecoderResources::new(spec(), Arc::clone(&f.bound)).unwrap();
    let layout = plan(HybridRankPrecision::MixedBf16F32, 2)
        .rank_layout(0)
        .unwrap();
    let segment = LayerSegmentPlan::new(2, 0..2, true, true).unwrap();
    // Seven explicitly supplied physical slots, including shadows; not 7 tokens.
    let b = layout
        .storage_budget(&resources, &segment, 16, 32, 7, 5)
        .unwrap();
    assert_eq!(b.parameters.len(), 1);
    assert_eq!(b.parameters[0].parameter, f.binding.id());
    assert_eq!(b.parameters[0].scale_bytes, 4);
    assert_eq!(b.weight_grants, grant(256 * 128) + grant(4));
    let page = layout
        .kv_planes(16, 32)
        .unwrap()
        .checked_page_bytes()
        .unwrap();
    assert_eq!(b.kv_bytes, page * 7);
    assert_eq!(b.kv_grants, 2 * grant(page / 2 * 7));
    assert_eq!(b.state.total_grants, 5 * b.state.per_sequence_grants);
    assert_eq!(
        b.storage_grants().unwrap(),
        b.weight_grants + b.kv_grants + b.state.total_grants
    );
    assert!(
        layout
            .storage_budget(&resources, &segment, 16, 32, 0, 5)
            .is_err()
    );
    assert!(
        layout
            .storage_budget(&resources, &segment, 16, 32, usize::MAX, 5)
            .is_err()
    );
}

#[test]
fn native_bf16_rank_materialization_preserves_words_and_reads_only_local_rectangles() {
    for (role, shape) in [
        (TensorRole::SharedExpertGate, vec![256, 256]),
        (TensorRole::SharedExpertDown, vec![256, 256]),
        (TensorRole::LinearAttentionConv, vec![256, 1, 3]),
    ] {
        let f = Fixture::new(
            role,
            &shape,
            ParameterResidency::Layer { layer: 0 },
            CheckpointDType::Bf16,
            None,
        );
        let layout = Arc::new(
            plan(HybridRankPrecision::MixedBf16F32, 2)
                .rank_layout(1)
                .unwrap(),
        );
        let budget = layout.parameter_budget(&f.binding).unwrap();
        let reader = StateDictMaterializer::new(budget.source_bytes as u64).unwrap();
        assert!(
            reader
                .prepare_hybrid_rank_parameter(
                    &f.binding,
                    layout.clone(),
                    budget.allocation_grant_bytes - 1
                )
                .is_err()
        );
        assert_eq!(reader.checkpoint_read_counters().bytes_read, 0);
        let p = reader
            .prepare_hybrid_rank_parameter(
                &f.binding,
                layout.clone(),
                budget.allocation_grant_bytes,
            )
            .unwrap();
        assert_eq!(p.layout(), layout.as_ref());
        let HybridRankStorage::Bf16(words) = p.storage() else {
            panic!("not native BF16")
        };
        let width = if shape.len() == 3 { shape[2] } else { shape[1] };
        let expected = p
            .partition()
            .rows
            .iter()
            .flat_map(|r| {
                r.clone().flat_map(|row| {
                    p.partition().columns.clone().map(move |col| {
                        [0x8000_u16, 0x3f81, 0x7fc1, 0xbf83][(row * width + col) % 4]
                    })
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(words.as_ref(), expected);
        assert_eq!(words.len() * 2, budget.weight_bytes);
        assert_eq!(
            reader.checkpoint_read_counters().bytes_read,
            budget.source_bytes as u64
        );
        let clone = p.clone();
        let HybridRankStorage::Bf16(clone_words) = clone.storage() else {
            panic!()
        };
        assert!(Arc::ptr_eq(words, clone_words));
        let short = StateDictMaterializer::new(budget.source_bytes as u64 - 1).unwrap();
        assert!(
            short
                .prepare_hybrid_rank_parameter(&f.binding, layout, budget.allocation_grant_bytes)
                .is_err()
        );
        assert_eq!(short.checkpoint_read_counters().bytes_read, 0);
    }
}

#[test]
fn mixed_rank_alog_source_cast_is_bounded_but_other_f32_parameters_fail_before_io() {
    let layout = Arc::new(
        plan(HybridRankPrecision::MixedBf16F32, 2)
            .rank_layout(1)
            .unwrap(),
    );
    let f = Fixture::new(
        TensorRole::LinearAttentionALog,
        &[8],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F32,
        None,
    );
    let reader = StateDictMaterializer::new(16).unwrap();
    let p = reader
        .prepare_hybrid_rank_parameter(&f.binding, layout.clone(), 256)
        .unwrap();
    let HybridRankStorage::Bf16(words) = p.storage() else {
        panic!()
    };
    let expected = (4..8)
        .map(|i| half::bf16::from_f32(1.003 + (i % 7) as f32).to_bits())
        .collect::<Vec<_>>();
    assert_eq!(words.as_ref(), expected);
    assert_eq!(reader.checkpoint_read_counters().bytes_read, 16);

    let norm = Fixture::new(
        TensorRole::LinearAttentionNorm,
        &[8],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F32,
        None,
    );
    let budget = layout.parameter_budget(&norm.binding).unwrap();
    assert_eq!((budget.source_bytes, budget.weight_bytes), (32, 16));
    let norm_reader = StateDictMaterializer::new(32).unwrap();
    let full_norm = norm_reader.parameter(&norm.binding).unwrap();
    assert_eq!(
        full_norm.binding().weight().slice().dtype,
        CheckpointDType::F32
    );
    assert!(
        full_norm.values_bf16_words().is_err(),
        "native accessor cannot silently cast"
    );
    assert_eq!(full_norm.values_f32().unwrap().len(), 8);
    assert_eq!(norm_reader.checkpoint_read_counters().bytes_read, 32);

    let f = Fixture::new(
        TensorRole::SharedExpertDown,
        &[256, 256],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F32,
        None,
    );
    let reader = StateDictMaterializer::new(1 << 20).unwrap();
    assert!(
        reader
            .prepare_hybrid_rank_parameter(&f.binding, layout, 1 << 20)
            .is_err()
    );
    assert_eq!(reader.checkpoint_read_counters().bytes_read, 0);
}

#[test]
fn prepared_rank_binding_rejects_foreign_source_lane_precision_and_expert_mode() {
    let f = Fixture::new(
        TensorRole::SharedExpertGate,
        &[256, 256],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::Bf16,
        None,
    );
    let other = Fixture::new(
        TensorRole::SharedExpertGate,
        &[256, 256],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::Bf16,
        None,
    );
    let layout = Arc::new(
        plan(HybridRankPrecision::MixedBf16F32, 2)
            .rank_layout(0)
            .unwrap(),
    );
    let p = StateDictMaterializer::new(1 << 20)
        .unwrap()
        .prepare_hybrid_rank_parameter(&f.binding, layout.clone(), 1 << 20)
        .unwrap();
    assert!(p.validate_binding(&f.binding, &layout).is_ok());
    assert!(p.validate_binding(&other.binding, &layout).is_err());
    for foreign in [
        plan(HybridRankPrecision::MixedBf16F32, 2)
            .rank_layout(1)
            .unwrap(),
        plan(HybridRankPrecision::F32, 2).rank_layout(0).unwrap(),
        HybridParallelPlan::new_with_execution(
            &spec(),
            0..2,
            scope(),
            members(2),
            HybridRoutedExpertExecution::ExternalExpertAttachment(attachment()),
            HybridRankPrecision::MixedBf16F32,
        )
        .unwrap()
        .rank_layout(0)
        .unwrap(),
    ] {
        assert!(p.validate_binding(&f.binding, &foreign).is_err());
    }
}

#[test]
fn mixed_block_gemmex_rebases_only_complete_scale_blocks() {
    use ferrule_backend::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};
    let source = NumericFp8Layout {
        n: 128,
        k: 256,
        row_origin: 384,
        column_origin: 128,
        scale_type: NumericFp8ScaleType::Bf16,
    };
    let packed = HybridMixedProjectionArithmetic::packed_gemmex_layout(source).unwrap();
    assert_eq!(packed.row_origin, 0);
    assert_eq!(packed.column_origin, 0);
    assert_eq!(
        packed.storage_lengths().unwrap(),
        source.storage_lengths().unwrap()
    );
    for bad in [
        NumericFp8Layout {
            column_origin: 129,
            ..source
        },
        NumericFp8Layout {
            row_origin: 127,
            ..source
        },
        NumericFp8Layout { k: 129, ..source },
        NumericFp8Layout {
            scale_type: NumericFp8ScaleType::F32,
            ..source
        },
    ] {
        assert!(HybridMixedProjectionArithmetic::packed_gemmex_layout(bad).is_err());
    }
}

#[test]
fn mixed_block_requires_every_expert_and_exact_physical_grants() {
    let layout = Arc::new(
        plan(HybridRankPrecision::MixedBf16F32, 2)
            .rank_layout(1)
            .unwrap(),
    );
    let mut fixtures = Vec::new();
    for expert in 0..4 {
        for (r, role) in [
            TensorRole::RoutedExpertGate,
            TensorRole::RoutedExpertUp,
            TensorRole::RoutedExpertDown,
        ]
        .into_iter()
        .enumerate()
        {
            fixtures.push(Fixture::with_data(
                role,
                &[256, 256],
                ParameterResidency::Expert { layer: 0, expert },
                CheckpointDType::F8E4M3,
                Some(CheckpointDType::Bf16),
                (expert * 3 + r + 1) as u64,
                None,
            ));
        }
    }
    let materializer = StateDictMaterializer::new(1 << 20).unwrap();
    let prepared = fixtures
        .iter()
        .map(|f| {
            materializer
                .prepare_hybrid_rank_parameter(&f.binding, layout.clone(), usize::MAX)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let admit = |ps: &[PreparedHybridRankParameter], w, scratch, provider| {
        HybridMixedBlockPlan::new(
            layout.clone(),
            ps,
            2,
            HybridMixedProjectionArithmetic::NativeBf16GemmEx,
            w,
            scratch,
            provider,
        )
    };
    let block = admit(&prepared, usize::MAX, usize::MAX, usize::MAX).unwrap();
    assert!(block.all_expert_shards_resident(0));
    assert!(!block.all_expert_shards_resident(1));
    assert_eq!(
        block.weight_grant_bytes(),
        12 * (grant(256 * 128) + grant(4))
    );
    assert!(admit(&prepared[..11], usize::MAX, usize::MAX, usize::MAX).is_err());
    assert!(
        admit(
            &prepared,
            block.weight_grant_bytes() - 1,
            usize::MAX,
            usize::MAX
        )
        .is_err()
    );
    assert!(
        admit(
            &prepared,
            usize::MAX,
            block.workspace_grant_bytes() - 1,
            usize::MAX
        )
        .is_err()
    );
    assert!(
        admit(
            &prepared,
            usize::MAX,
            usize::MAX,
            block.provider_bytes() - 1
        )
        .is_err()
    );
    admit(
        &prepared,
        block.weight_grant_bytes(),
        block.workspace_grant_bytes(),
        block.provider_bytes(),
    )
    .unwrap();
    let w8 = HybridMixedBlockPlan::new(
        layout,
        &prepared,
        2,
        HybridMixedProjectionArithmetic::W8A16Bf16,
        block.weight_grant_bytes(),
        0,
        0,
    )
    .unwrap();
    assert_eq!(w8.workspace_grant_bytes(), 0);
    assert_eq!(w8.provider_bytes(), 0);
}

fn mixed_oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("support/hybrid_rank_mixed_oracle.json")).unwrap()
}
fn oracle_words(v: &serde_json::Value) -> Vec<u16> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| u16::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}
#[test]
fn frozen_torch_tp2_bf16_staging_distinguishes_both_rounding_orders() {
    use ferrule_model::execution::bf16_rne_word;
    let f = mixed_oracle();
    assert_eq!(f["schema"], "ferrule.tp2.bf16-local.v1");
    assert_eq!(f["oracle"], "independent-torch-cpu");
    let add = |a: &[u16], b: &[u16]| {
        a.iter()
            .zip(b)
            .map(|(&a, &b)| {
                bf16_rne_word(
                    f32::from_bits(u32::from(a) << 16) + f32::from_bits(u32::from(b) << 16),
                )
            })
            .collect::<Vec<_>>()
    };
    let reduced = add(&oracle_words(&f["part0"]), &oracle_words(&f["part1"]));
    assert_eq!(reduced, oracle_words(&f["reduced"]));
    assert_eq!(
        reduced
            .iter()
            .zip(oracle_words(&f["deferred"]))
            .filter(|(a, b)| **a != *b)
            .count(),
        91
    );
    let shared = [oracle_words(&f["shared"][0]), oracle_words(&f["shared"][1])];
    let routed = [oracle_words(&f["routed"][0]), oracle_words(&f["routed"][1])];
    assert_eq!(
        add(&add(&shared[0], &routed[0]), &add(&shared[1], &routed[1])),
        oracle_words(&f["late"])
    );
    assert_eq!(
        add(&add(&shared[0], &shared[1]), &add(&routed[0], &routed[1])),
        oracle_words(&f["early"])
    );
    assert_ne!(f["late"], f["early"]);
}

#[cfg(feature = "cuda")]
fn frozen_mixed_local_projection_gpu(native_weight: bool, math: HybridMixedProjectionArithmetic) {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_model::execution::bf16_rne_word;
    use ferrule_model::transformer::ExpertCachePolicy;
    use std::rc::Rc;
    assert_eq!(
        std::env::var("FERRULE_RUN_MIXED_RANK_GPU").as_deref(),
        Ok("1"),
        "separate two-GPU grant required"
    );
    let f = mixed_oracle();
    let (m, n, k) = (2, 128, 256);
    let codes = [0x38, 0xb4, 0x29, 0x42, 0xbb, 0x13, 0x48, 0xae];
    let values = [1., -0.75, 0.28125, 2.5, -1.375, 0.04296875, 4., -0.4375];
    let mut data = Vec::new();
    for r in 0..n {
        for c in 0..k {
            let ix = (r * 53 + c * 17 + (r * c) % 37) % 8;
            if native_weight {
                data.extend(
                    bf16_rne_word(values[ix] * if c < 128 { 0.75 } else { 1.25 }).to_le_bytes(),
                );
            } else {
                data.push(codes[ix]);
            }
        }
    }
    if !native_weight {
        data.extend(0x3f40_u16.to_le_bytes());
        data.extend(0x3fa0_u16.to_le_bytes());
    }
    let fixture = Fixture::with_data(
        TensorRole::AttentionOutput,
        &[n, k],
        ParameterResidency::Layer { layer: 0 },
        if native_weight {
            CheckpointDType::Bf16
        } else {
            CheckpointDType::F8E4M3
        },
        if native_weight {
            None
        } else {
            Some(CheckpointDType::Bf16)
        },
        1,
        Some(data),
    );
    let parallel = plan(HybridRankPrecision::MixedBf16F32, 2);
    let ops = [
        Rc::new(CudaOperators::new_on_device(0).unwrap()),
        Rc::new(CudaOperators::new_on_device(1).unwrap()),
    ];
    let mut owners = Vec::new();
    let mut tokens = Vec::new();
    for (lane, op) in ops.iter().enumerate() {
        let layout = Arc::new(parallel.rank_layout(lane).unwrap());
        let p = StateDictMaterializer::new(1 << 20)
            .unwrap()
            .prepare_hybrid_rank_parameter(&fixture.binding, layout.clone(), usize::MAX)
            .unwrap();
        let block = HybridMixedBlockPlan::new(
            layout,
            std::slice::from_ref(&p),
            m,
            math,
            1 << 20,
            64 << 20,
            16 << 20,
        )
        .unwrap();
        let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            op.clone(),
            std::slice::from_ref(&fixture.binding),
            ExpertCachePolicy::KeepAll,
            0,
            NumericFp8Precision::W8A16Bf16F32Accumulate,
        )
        .unwrap();
        tokens.push(
            owner
                .prepare_mixed_rank_block(block, &[p])
                .unwrap()
                .remove(0),
        );
        owners.push(owner);
    }
    // Both owners and complete local weight roots are resident simultaneously.
    let mut results = Vec::new();
    for lane in 0..2 {
        let x = (0..m)
            .flat_map(|r| {
                (lane * 128..(lane + 1) * 128).map(move |c| {
                    let i = r * k + c;
                    bf16_rne_word(((i * 29 + i / k * 11) % 509) as f32 / 128. - 254. / 128.)
                })
            })
            .collect::<Vec<_>>();
        let x = Rows::Cuda(
            CudaRows::bf16(
                RowsShape::new(m, 128).unwrap(),
                None,
                ops[lane].upload_bf16_words(&x).unwrap(),
            )
            .unwrap(),
        );
        assert!(
            owners[lane]
                .mixed_rank_linear_local(&tokens[1 - lane], &x)
                .is_err()
        );
        ops[lane].reset_counters();
        let result = owners[lane]
            .mixed_rank_linear_local(&tokens[lane], &x)
            .unwrap();
        assert_eq!(ops[lane].counters().artifact_uploads, 0);
        let actual = ops[lane]
            .download_bf16_buffer(result.cuda().unwrap().bf16_buffer().unwrap())
            .unwrap();
        let words = actual
            .iter()
            .copied()
            .map(bf16_rne_word)
            .collect::<Vec<_>>();
        assert_eq!(words, oracle_words(&f[format!("part{lane}")]));
        results.push(actual);
    }
    // Simulates BF16 collective arithmetic only, not runtime/NCCL transport.
    let a = ops[0]
        .upload_bf16_words(
            &results[0]
                .iter()
                .copied()
                .map(bf16_rne_word)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let b = ops[0]
        .upload_bf16_words(
            &results[1]
                .iter()
                .copied()
                .map(bf16_rne_word)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let sum = ops[0].residual_bf16(&a, &b).unwrap();
    let actual = ops[0]
        .download_bf16_buffer(&sum)
        .unwrap()
        .into_iter()
        .map(bf16_rne_word)
        .collect::<Vec<_>>();
    assert_eq!(actual, oracle_words(&f["reduced"]));
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "authenticated two-GPU production QKV concat path; fixed layer0 operands only"]
fn frozen_tp2_production_qkv_concat_rank_path() {
    production_qkv_concat_rank_path(1);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "authenticated two-GPU fixed Capital M5 QKV; independent pinned HF CUDA expected"]
fn frozen_tp2_production_capital_m5_qkv_rank_path() {
    production_qkv_concat_rank_path(5);
}

#[cfg(feature = "cuda")]
fn production_qkv_concat_rank_path(rows: usize) {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_model::transformer::ExpertCachePolicy;
    use std::rc::Rc;
    assert_eq!(
        std::env::var("FERRULE_RUN_MIXED_RANK_GPU").as_deref(),
        Ok("1")
    );
    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
    assert!(std::env::var("FERRULE_PAPER_WORKER_TOKEN").is_ok());
    let fixture = std::path::PathBuf::from(std::env::var_os("FERRULE_QKV_FIXTURE").unwrap());
    let expected = std::path::PathBuf::from(std::env::var_os("FERRULE_QKV_EXPECTED").unwrap());
    let inputs = std::env::var_os("FERRULE_QKV_INPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| fixture.clone());
    let mut raw_global = vec![0_u8; 8192 * 2048];
    let mut scale_global = vec![0_u8; 64 * 16 * 2];
    let ranges = [
        [0..1024, 2048..3072, 4096..6144],
        [1024..2048, 3072..4096, 6144..8192],
    ];
    for (rank, rank_ranges) in ranges.iter().enumerate() {
        let raw = std::fs::read(fixture.join(format!("rank{rank}.weight.fp8"))).unwrap();
        let scales = std::fs::read(fixture.join(format!("rank{rank}.scales.bf16"))).unwrap();
        let mut local_row = 0;
        let mut local_scale = 0;
        for range in rank_ranges {
            let width = range.end - range.start;
            raw_global[range.start * 2048..range.end * 2048]
                .copy_from_slice(&raw[local_row * 2048..(local_row + width) * 2048]);
            let scale_bytes = width / 128 * 16 * 2;
            scale_global[range.start / 128 * 16 * 2..range.end / 128 * 16 * 2]
                .copy_from_slice(&scales[local_scale..local_scale + scale_bytes]);
            local_row += width;
            local_scale += scale_bytes;
        }
    }
    let source = Fixture::with_data(
        TensorRole::LinearAttentionQkv,
        &[8192, 2048],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F8E4M3,
        Some(CheckpointDType::Bf16),
        991,
        Some([raw_global, scale_global].concat()),
    );
    let config = serde_json::from_str(include_str!("qwen35_35b_fp8_config.json")).unwrap();
    let spec = ferrule_model::models::qwen35::Qwen35Recipe::spec(
        &ferrule_model::models::qwen35::Qwen35Config::from_value(&config).unwrap(),
    )
    .unwrap();
    let parallel = HybridParallelPlan::new_with_execution(
        &spec,
        0..40,
        scope(),
        members(2),
        HybridRoutedExpertExecution::LocalTensorShards,
        HybridRankPrecision::MixedBf16F32,
    )
    .unwrap();
    let mut prepared_roots = Vec::new();
    for rank in 0..2 {
        let op = Rc::new(CudaOperators::new_on_device(rank).unwrap());
        let layout = Arc::new(parallel.rank_layout(rank).unwrap());
        let prepared = StateDictMaterializer::new(64 << 20)
            .unwrap()
            .prepare_hybrid_rank_parameter(&source.binding, layout.clone(), usize::MAX)
            .unwrap();
        let packed = match prepared.storage() {
            HybridRankStorage::Numeric(n) => n.packed_mixed_gemmex_qkv().unwrap(),
            _ => panic!("QKV fixture must remain numeric"),
        };
        assert_eq!(packed.layout().n, 4096);
        let budget = layout.parameter_budget(&source.binding).unwrap();
        let block = HybridMixedBlockPlan::new(
            layout,
            std::slice::from_ref(&prepared),
            rows,
            HybridMixedProjectionArithmetic::NativeBf16GemmEx,
            budget.allocation_grant_bytes,
            64 << 20,
            16 << 20,
        )
        .unwrap();
        let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            op.clone(),
            std::slice::from_ref(&source.binding),
            ExpertCachePolicy::KeepAll,
            0,
            NumericFp8Precision::W8A16Bf16F32Accumulate,
        )
        .unwrap();
        let token = owner
            .prepare_mixed_rank_block(block, &[prepared])
            .unwrap()
            .remove(0);
        let words = std::fs::read(inputs.join(format!("rank{rank}.normalized_input.bf16")))
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect::<Vec<_>>();
        let input = Rows::Cuda(
            CudaRows::bf16(
                RowsShape::new(rows, 2048).unwrap(),
                None,
                op.upload_bf16_words(&words).unwrap(),
            )
            .unwrap(),
        );
        prepared_roots.push((op, owner, token, input));
    }
    // Both rank images are resident before execution; no checkpoint fallback.
    std::fs::remove_file(&source.path).unwrap();
    for (rank, (op, mut owner, token, input)) in prepared_roots.into_iter().enumerate() {
        op.reset_counters();
        let result = owner.mixed_rank_linear_local(&token, &input).unwrap();
        let counters = op.counters();
        assert_eq!(counters.artifact_uploads, 0);
        assert_eq!(
            counters.device_allocation_attempts, 1,
            "only one native BF16 output allocation; no segmented packing"
        );
        assert_eq!(counters.host_to_device_copies, 0);
        let actual = op
            .download_bf16_buffer(result.cuda().unwrap().bf16_buffer().unwrap())
            .unwrap();
        let expected_bytes =
            std::fs::read(expected.join(format!("rank{rank}.hf_concat.qkv.f32"))).unwrap();
        let expected = expected_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        println!(
            "PRODUCTION_QKV_GPU_COUNTS rows={rows} rank={rank} exact_words={} local_numeric_segments=1 execute_artifact_uploads=0 output_allocations=1",
            actual.len()
        );
        op.sync_stream().unwrap();
        drop(result);
        drop(input);
        drop(owner);
        op.sync_stream().unwrap();
        op.trim_device_allocator().unwrap();
        assert_eq!(op.allocator_metrics().live_requested_bytes, 0);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires explicit two-GPU grant and CUBLAS_WORKSPACE_CONFIG=:0:0"]
fn frozen_tp2_native_bf16_rank_row_partials() {
    frozen_mixed_local_projection_gpu(true, HybridMixedProjectionArithmetic::NativeBf16GemmEx);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires explicit two-GPU grant and CUBLAS_WORKSPACE_CONFIG=:0:0"]
fn frozen_tp2_numeric_gemmex_rank_row_partials() {
    frozen_mixed_local_projection_gpu(false, HybridMixedProjectionArithmetic::NativeBf16GemmEx);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires explicit two-GPU grant"]
fn frozen_tp2_numeric_w8a16_rank_row_partials() {
    frozen_mixed_local_projection_gpu(false, HybridMixedProjectionArithmetic::W8A16Bf16);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires explicit two-GPU grant and CUBLAS_WORKSPACE_CONFIG=:0:0; HF-eager routed arithmetic only"]
fn frozen_tp2_local_experts_shared_late_add() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_model::execution::bf16_rne_word;
    use std::rc::Rc;
    assert_eq!(
        std::env::var("FERRULE_RUN_MIXED_RANK_GPU").as_deref(),
        Ok("1")
    );
    let oracle = mixed_oracle();
    let mut fixtures = Vec::new();
    for expert in 0..5 {
        let shared = expert == 4;
        let roles = if shared {
            [
                TensorRole::SharedExpertGate,
                TensorRole::SharedExpertUp,
                TensorRole::SharedExpertDown,
            ]
        } else {
            [
                TensorRole::RoutedExpertGate,
                TensorRole::RoutedExpertUp,
                TensorRole::RoutedExpertDown,
            ]
        };
        let factor = if shared { 1. } else { (expert + 1) as f32 };
        for (projection, role) in roles.into_iter().enumerate() {
            let mut data = vec![0x38; 256 * 256];
            for r in 0..2 {
                for c in 0..2 {
                    let scale = factor * (if projection == 2 { c + 1 } else { r + 1 }) as f32
                        / if projection == 1 { 128. } else { 256. };
                    data.extend(bf16_rne_word(scale).to_le_bytes());
                }
            }
            fixtures.push(Fixture::with_data(
                role,
                &[256, 256],
                if shared {
                    ParameterResidency::Layer { layer: 0 }
                } else {
                    ParameterResidency::Expert { layer: 0, expert }
                },
                CheckpointDType::F8E4M3,
                Some(CheckpointDType::Bf16),
                (expert * 3 + projection + 1) as u64,
                Some(data),
            ));
        }
    }
    let directory = fixtures
        .iter()
        .map(|f| f.binding.clone())
        .collect::<Vec<_>>();
    let parallel = plan(HybridRankPrecision::MixedBf16F32, 2);
    let ops = [
        Rc::new(CudaOperators::new_on_device(0).unwrap()),
        Rc::new(CudaOperators::new_on_device(1).unwrap()),
    ];
    let mut owners = Vec::new();
    let mut handles = Vec::new();
    for (lane, op) in ops.iter().enumerate() {
        let layout = Arc::new(parallel.rank_layout(lane).unwrap());
        let prepared = fixtures
            .iter()
            .map(|f| {
                StateDictMaterializer::new(1 << 20)
                    .unwrap()
                    .prepare_hybrid_rank_parameter(&f.binding, layout.clone(), usize::MAX)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let p = HybridMixedBlockPlan::new(
            layout,
            &prepared,
            2,
            HybridMixedProjectionArithmetic::NativeBf16GemmEx,
            4 << 20,
            64 << 20,
            16 << 20,
        )
        .unwrap();
        assert!(p.all_expert_shards_resident(0));
        let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            op.clone(),
            &directory,
            ExpertCachePolicy::KeepAll,
            0,
            NumericFp8Precision::W8A16Bf16F32Accumulate,
        )
        .unwrap();
        handles.push(owner.prepare_mixed_rank_block(p, &prepared).unwrap());
        owners.push(owner);
    }
    // Remove backing files after publication: execution must not read them or
    // silently populate an on-demand CPU/offload cache, even for experts 0/2.
    for f in &fixtures {
        std::fs::remove_file(&f.path).unwrap();
    }
    for lane in 0..2 {
        let op = &ops[lane];
        let x = [
            vec![bf16_rne_word(1. / 256.); 256],
            vec![bf16_rne_word(1. / 128.); 256],
        ]
        .concat();
        let x = Rows::Cuda(
            CudaRows::bf16(
                RowsShape::new(2, 256).unwrap(),
                None,
                op.upload_bf16_words(&x).unwrap(),
            )
            .unwrap(),
        );
        let routes =
            RouterRoutes::new(2, 2, vec![3, 1, 3, 1], vec![0.25, 0.75, 0.25, 0.75]).unwrap();
        op.reset_counters();
        let routed = owners[lane]
            .mixed_rank_routed_local(
                0,
                &x,
                &routes,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
            )
            .unwrap();
        let shared = owners[lane]
            .mixed_rank_swiglu_local(
                [&handles[lane][12], &handles[lane][13], &handles[lane][14]],
                &x,
            )
            .unwrap();
        let gate = Rows::Cuda(
            CudaRows::bf16(
                RowsShape::new(2, 1).unwrap(),
                None,
                op.upload_bf16_words(&[0, 0]).unwrap(),
            )
            .unwrap(),
        );
        let shared = owners[lane]
            .mixed_rank_shared_gate_local(shared, &gate)
            .unwrap();
        assert!(
            owners[lane]
                .mixed_rank_moe_local_add(
                    &shared,
                    &routed,
                    HybridMoeCombineOrder::SeparatelyReducedBf16Add
                )
                .is_err()
        );
        let combined = owners[lane]
            .mixed_rank_moe_local_add(
                &shared,
                &routed,
                HybridMoeCombineOrder::LocalBf16AddThenReduce,
            )
            .unwrap();
        assert_eq!(op.counters().artifact_uploads, 0);
        for (name, rows) in [
            ("routed", routed),
            ("shared", shared),
            ("combined", combined),
        ] {
            let actual = op
                .download_bf16_buffer(rows.cuda().unwrap().bf16_buffer().unwrap())
                .unwrap()
                .into_iter()
                .map(bf16_rne_word)
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                oracle_words(&oracle["moe"][lane.to_string()][name]),
                "lane {lane} {name}"
            );
        }
        assert!(!owners[lane].needs_quarantine());
    }
}

#[test]
fn mixed_gdn_qkv_packs_source_blocks_and_admits_native_width_only() {
    use ferrule_backend::cuda_plan::bf16_gemmex::Bf16GemmExPlan;
    use ferrule_model::models::qwen35::{Qwen35Config, Qwen35Recipe};
    let config = serde_json::from_str(include_str!("qwen35_35b_fp8_config.json")).unwrap();
    let spec = Qwen35Recipe::spec(&Qwen35Config::from_value(&config).unwrap()).unwrap();
    let parallel = |precision| {
        HybridParallelPlan::new_with_execution(
            &spec,
            0..40,
            scope(),
            members(2),
            HybridRoutedExpertExecution::LocalTensorShards,
            precision,
        )
        .unwrap()
    };
    let mut data = (0..8192 * 2048)
        .map(|i| 0x30 + ((i / 2048 / 128 + i % 2048 / 128) % 16) as u8)
        .collect::<Vec<_>>();
    for row in 0..64 {
        for col in 0..16 {
            data.extend((0x3e00_u16 + row * 16 + col).to_le_bytes());
        }
    }
    let f = Fixture::with_data(
        TensorRole::LinearAttentionQkv,
        &[8192, 2048],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F8E4M3,
        Some(CheckpointDType::Bf16),
        1,
        Some(data.clone()),
    );
    for rank in 0..2 {
        let layout = Arc::new(
            parallel(HybridRankPrecision::MixedBf16F32)
                .rank_layout(rank)
                .unwrap(),
        );
        let p = StateDictMaterializer::new(32 << 20)
            .unwrap()
            .prepare_hybrid_rank_parameter(&f.binding, layout.clone(), usize::MAX)
            .unwrap();
        let HybridRankStorage::Numeric(numeric) = p.storage() else {
            panic!()
        };
        assert_eq!(numeric.segments().len(), 3);
        let packed = numeric.packed_mixed_gemmex_qkv().unwrap();
        assert_eq!((packed.layout().n, packed.layout().k), (4096, 2048));
        assert_eq!(
            (packed.layout().row_origin, packed.layout().column_origin),
            (0, 0)
        );
        let ranges = [
            rank * 1024..(rank + 1) * 1024,
            2048 + rank * 1024..2048 + (rank + 1) * 1024,
            4096 + rank * 2048..4096 + (rank + 1) * 2048,
        ];
        let mut expected_weight = Vec::new();
        let mut expected_scales = Vec::new();
        for range in ranges {
            expected_weight.extend_from_slice(&data[range.start * 2048..range.end * 2048]);
            expected_scales.extend_from_slice(
                &data[8192 * 2048 + range.start / 128 * 32..8192 * 2048 + range.end / 128 * 32],
            );
        }
        assert_eq!(packed.weight_bytes(), expected_weight);
        assert_eq!(packed.scale_bytes(), expected_scales);
        let ledger = layout.parameter_budget(&f.binding).unwrap();
        let budget = grant(expected_weight.len()) + grant(expected_scales.len());
        assert!(budget <= ledger.allocation_grant_bytes);
        let workspace = Bf16GemmExPlan::new(packed.layout(), 2, usize::MAX, usize::MAX).unwrap();
        let block = |w, s, provider| {
            HybridMixedBlockPlan::new(
                layout.clone(),
                std::slice::from_ref(&p),
                2,
                HybridMixedProjectionArithmetic::NativeBf16GemmEx,
                w,
                s,
                provider,
            )
        };
        let b = block(
            budget,
            workspace.storage_grant_bytes(),
            workspace.provider_bytes(),
        )
        .unwrap();
        assert_eq!(b.weight_bytes(), packed.storage_bytes());
        assert_eq!(b.weight_grant_bytes(), budget);
        assert_eq!(b.workspace_grant_bytes(), workspace.storage_grant_bytes());
        assert!(block(budget - 1, usize::MAX, usize::MAX).is_err());
        assert!(block(budget, workspace.storage_grant_bytes() - 1, usize::MAX).is_err());
        let old_workspace = Bf16GemmExPlan::new(
            ferrule_common::numeric_fp8::NumericFp8Layout {
                n: 2048,
                ..packed.layout()
            },
            2,
            usize::MAX,
            usize::MAX,
        )
        .unwrap();
        assert!(block(budget, old_workspace.storage_grant_bytes(), usize::MAX).is_err());
        let w8 = HybridMixedBlockPlan::new(
            layout,
            std::slice::from_ref(&p),
            2,
            HybridMixedProjectionArithmetic::W8A16Bf16,
            ledger.allocation_grant_bytes,
            0,
            0,
        )
        .unwrap();
        assert_eq!(w8.weight_grant_bytes(), ledger.allocation_grant_bytes);
        assert_eq!(w8.workspace_grant_bytes(), 0);
        let f32 = NumericFp8RankPlan::new(
            &f.binding,
            Arc::new(
                parallel(HybridRankPrecision::F32)
                    .rank_layout(rank)
                    .unwrap(),
            ),
        )
        .unwrap();
        assert!(f32.mixed_gemmex_qkv_layout().unwrap().is_none());
        assert_eq!(f32.reads().len(), 3);
    }
    let layout = Arc::new(
        parallel(HybridRankPrecision::MixedBf16F32)
            .rank_layout(0)
            .unwrap(),
    );
    let p = StateDictMaterializer::new(32 << 20)
        .unwrap()
        .prepare_hybrid_rank_parameter(&f.binding, layout, usize::MAX)
        .unwrap();
    std::fs::write(&f.path, [0_u8]).unwrap();
    let HybridRankStorage::Numeric(numeric) = p.storage() else {
        panic!()
    };
    assert!(
        numeric.packed_mixed_gemmex_qkv().is_err(),
        "source mutation must fail before upload"
    );
}

#[test]
fn mixed_gdn_qkv_rejects_partial_scale_blocks_without_affecting_w8() {
    let f = Fixture::new(
        TensorRole::LinearAttentionQkv,
        &[256, 256],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F8E4M3,
        Some(CheckpointDType::Bf16),
    );
    let layout = Arc::new(
        plan(HybridRankPrecision::MixedBf16F32, 2)
            .rank_layout(0)
            .unwrap(),
    );
    let p = StateDictMaterializer::new(1 << 20)
        .unwrap()
        .prepare_hybrid_rank_parameter(&f.binding, layout.clone(), usize::MAX)
        .unwrap();
    let HybridRankStorage::Numeric(numeric) = p.storage() else {
        panic!()
    };
    assert!(numeric.plan().mixed_gemmex_qkv_layout().is_err());
    assert!(numeric.packed_mixed_gemmex_qkv().is_err());
    assert!(
        HybridMixedBlockPlan::new(
            layout.clone(),
            std::slice::from_ref(&p),
            1,
            HybridMixedProjectionArithmetic::NativeBf16GemmEx,
            usize::MAX,
            usize::MAX,
            usize::MAX
        )
        .is_err()
    );
    HybridMixedBlockPlan::new(
        layout,
        &[p],
        1,
        HybridMixedProjectionArithmetic::W8A16Bf16,
        usize::MAX,
        0,
        0,
    )
    .unwrap();
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "authenticated two-GPU fixed same-input shared MLP; independent HF separate-projection reference"]
fn frozen_tp2_capital_shared_mlp_source_geometry() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use std::rc::Rc;
    assert_eq!(
        std::env::var("FERRULE_RUN_MIXED_RANK_GPU").as_deref(),
        Ok("1")
    );
    assert!(std::env::var("FERRULE_PAPER_WORKER_TOKEN").is_ok());
    let dir = PathBuf::from(std::env::var_os("FERRULE_SHARED_GEOMETRY_FIXTURE").unwrap());
    let config = serde_json::from_str(include_str!("qwen35_35b_fp8_config.json")).unwrap();
    let spec = ferrule_model::models::qwen35::Qwen35Recipe::spec(
        &ferrule_model::models::qwen35::Qwen35Config::from_value(&config).unwrap(),
    )
    .unwrap();
    let parallel = HybridParallelPlan::new_with_execution(
        &spec,
        0..40,
        scope(),
        members(2),
        HybridRoutedExpertExecution::LocalTensorShards,
        HybridRankPrecision::MixedBf16F32,
    )
    .unwrap();
    let fixtures = [
        ("gate", TensorRole::SharedExpertGate, [512, 2048]),
        ("up", TensorRole::SharedExpertUp, [512, 2048]),
        ("down", TensorRole::SharedExpertDown, [2048, 512]),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (name, role, shape))| {
        let mut data = std::fs::read(dir.join(format!("{name}.fp8"))).unwrap();
        data.extend(std::fs::read(dir.join(format!("{name}.scales.bf16"))).unwrap());
        Fixture::with_data(
            role,
            &shape,
            ParameterResidency::Layer { layer: 2 },
            CheckpointDType::F8E4M3,
            Some(CheckpointDType::Bf16),
            i as u64 + 300,
            Some(data),
        )
    })
    .collect::<Vec<_>>();
    let bindings = fixtures
        .iter()
        .map(|f| f.binding.clone())
        .collect::<Vec<_>>();
    let words = |rank, name: &str| {
        std::fs::read(dir.join(format!("rank{rank}.{name}.bf16")))
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect::<Vec<_>>()
    };
    let mut roots = Vec::new();
    for rank in 0..2 {
        let op = Rc::new(CudaOperators::new_on_device(rank).unwrap());
        let layout = Arc::new(parallel.rank_layout(rank).unwrap());
        let materializer = StateDictMaterializer::new(8 << 20).unwrap();
        let parameters = bindings
            .iter()
            .map(|b| {
                materializer
                    .prepare_hybrid_rank_parameter(b, layout.clone(), usize::MAX)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let plan = HybridMixedBlockPlan::new(
            layout,
            &parameters,
            5,
            HybridMixedProjectionArithmetic::NativeBf16GemmEx,
            8 << 20,
            64 << 20,
            16 << 20,
        )
        .unwrap();
        let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            op.clone(),
            &bindings,
            ExpertCachePolicy::KeepAll,
            0,
            NumericFp8Precision::W8A16Bf16F32Accumulate,
        )
        .unwrap();
        let tokens = owner.prepare_mixed_rank_block(plan, &parameters).unwrap();
        let input = Rows::Cuda(
            CudaRows::bf16(
                RowsShape::new(5, 2048).unwrap(),
                None,
                op.upload_bf16_words(&words(rank, "input")).unwrap(),
            )
            .unwrap(),
        );
        roots.push((op, owner, tokens, input));
    }
    for f in &fixtures {
        std::fs::remove_file(&f.path).unwrap();
    }
    for (rank, (op, mut owner, tokens, input)) in roots.into_iter().enumerate() {
        let equal = |rows: &Rows, name: &str| {
            let actual = op
                .download_bf16_buffer(rows.cuda().unwrap().bf16_buffer().unwrap())
                .unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                words(rank, name)
                    .iter()
                    .map(|v| u32::from(*v) << 16)
                    .collect::<Vec<_>>(),
                "rank {rank} {name}"
            );
        };
        for (index, name) in [(0, "gate"), (1, "up")] {
            let value = owner
                .mixed_rank_linear_local(&tokens[index], &input)
                .unwrap();
            equal(&value, name);
        }
        op.reset_counters();
        let out = owner
            .mixed_rank_swiglu_local([&tokens[0], &tokens[1], &tokens[2]], &input)
            .unwrap();
        assert_eq!(op.counters().artifact_uploads, 0);
        assert_eq!(op.counters().host_to_device_copies, 0);
        equal(&out, "output");
        assert_ne!(
            words(rank, "output"),
            words(rank, "concat_output"),
            "fixture must distinguish native geometry"
        );
        println!(
            "SHARED_GEOMETRY_GPU rank={rank} gate_exact=1280 up_exact=1280 output_exact=10240 resident_no_source=true"
        );
        drop(out);
        drop(input);
        drop(owner);
        op.sync_stream().unwrap();
        op.trim_device_allocator().unwrap();
        assert_eq!(op.allocator_metrics().live_requested_bytes, 0);
    }
}

#[test]
fn shared_gemmex_plan_does_not_warm_a_routed_concatenation() {
    use ferrule_backend::cuda_plan::{
        NumericFp8Layout, NumericFp8ScaleType, bf16_gemmex::Bf16GemmExPlan,
    };
    let layout = Arc::new(
        plan(HybridRankPrecision::MixedBf16F32, 2)
            .rank_layout(0)
            .unwrap(),
    );
    let f = Fixture::new(
        TensorRole::SharedExpertGate,
        &[256, 256],
        ParameterResidency::Layer { layer: 0 },
        CheckpointDType::F8E4M3,
        Some(CheckpointDType::Bf16),
    );
    let p = StateDictMaterializer::new(1 << 20)
        .unwrap()
        .prepare_hybrid_rank_parameter(&f.binding, layout.clone(), usize::MAX)
        .unwrap();
    let native = Bf16GemmExPlan::new(
        NumericFp8Layout {
            n: 128,
            k: 256,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        },
        5,
        usize::MAX,
        usize::MAX,
    )
    .unwrap();
    let block = HybridMixedBlockPlan::new(
        layout,
        &[p],
        5,
        HybridMixedProjectionArithmetic::NativeBf16GemmEx,
        usize::MAX,
        native.storage_grant_bytes(),
        native.provider_bytes(),
    )
    .unwrap();
    assert_eq!(block.workspace_grant_bytes(), native.storage_grant_bytes());
}

#[path = "mixed_routed_down/admission.rs"]
mod mixed_routed_down_candidate;
