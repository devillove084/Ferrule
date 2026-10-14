//! CPU ownership checks: CUDA feature compiles types but no CUDA owner is opened.
use super::*;
use crate::transformer::hybrid_parallel::*;
use crate::transformer::materialize::numeric_rank_tests::Fixture;
use crate::transformer::*;
use ferrule_common::{ParallelRankId, ParallelTopologyId};
use std::sync::{Arc, Weak};

fn layout() -> Arc<HybridRankLayout> {
    let norm = || RmsNorm::new(256, 1e-6).unwrap();
    let rope = RotaryEmbedding::new(
        64,
        10000.,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 64 },
        RotaryScaling::None,
    )
    .unwrap();
    let layers = [
        Attention::GatedDeltaNet(
            GatedDeltaNetAttention::new(256, 4, 4, 64, 64, 3, 1e-6, false).unwrap(),
        ),
        Attention::Gqa(GqaAttention::new(256, 4, 2, 64, false, rope).unwrap()),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, attention)| {
        DecoderLayer::new(
            index,
            norm(),
            attention,
            Residual::Add,
            norm(),
            FeedForward::SwiGlu(SwiGlu::new(256, 256, false).unwrap()),
            Residual::Add,
        )
        .unwrap()
    })
    .collect();
    let spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: "mixed-retirement-cpu".into(),
        hidden_size: 256,
        vocab_size: 256,
        max_sequence_length: Some(16),
        token_embedding: Embedding::new(256, 256, None).unwrap(),
        layers,
        final_norm: norm(),
        output: Linear::new(256, 256, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap();
    Arc::new(
        HybridParallelPlan::new_with_execution(
            &spec,
            0..2,
            HybridExpertScope {
                topology: ParallelTopologyId::new(93),
                replica: 0,
                stage: 0,
            },
            vec![ParallelRankId::new(0), ParallelRankId::new(1)],
            HybridRoutedExpertExecution::None,
            HybridRankPrecision::MixedBf16F32,
        )
        .unwrap()
        .rank_layout(0)
        .unwrap(),
    )
}

fn linear(
    role: TensorRole,
    shape: [usize; 2],
    layer: Option<usize>,
) -> (
    Fixture,
    PreparedLinear,
    Weak<[PreparedNumericFp8RankSegment]>,
) {
    let fixture = Fixture::new(shape, role.clone(), layer, true);
    let materializer =
        StateDictMaterializer::for_hybrid_cuda_rank(4 << 20, layout(), 4 << 20).unwrap();
    let linear = materializer
        .prepared_linear(&fixture.binding, role)
        .unwrap();
    let HybridRankStorage::Numeric(payload) = linear.hybrid_rank().unwrap().storage() else {
        panic!()
    };
    let weak = payload.payload_liveness();
    (fixture, linear, weak)
}

#[test]
fn mixed_source_retirement_requires_final_completion_and_preserves_aliases() {
    let (_fixture, prepared, weak) = linear(TensorRole::DenseMlpGate, [256, 256], Some(0));
    let alias = prepared.clone();
    let pending = MixedSourceRetirement::default();
    assert!(prepared.mixed_runtime_binding(&pending).is_err());
    assert!(weak.upgrade().is_some());
    let runtime = prepared
        .mixed_runtime_binding(&MixedSourceRetirement::completed_for_test())
        .unwrap();
    assert!(matches!(
        prepared.hybrid_rank().unwrap().storage(),
        HybridRankStorage::Numeric(_)
    ));
    assert!(matches!(
        runtime.hybrid_rank().unwrap().storage(),
        HybridRankStorage::NumericBinding(_)
    ));
    drop(prepared);
    assert!(
        weak.upgrade().is_some(),
        "retirement must not mutate an external alias"
    );
    drop(alias);
    assert!(
        weak.upgrade().is_none(),
        "runtime binding retained raw segments"
    );
}

#[test]
fn mixed_source_retirement_preserves_provenance_shapes_and_fails_closed() {
    let (fixture, prepared, weak) = linear(TensorRole::LinearAttentionQkv, [768, 256], Some(0));
    let before = prepared.hybrid_rank().unwrap();
    let HybridRankStorage::Numeric(payload) = before.storage() else {
        panic!()
    };
    let reads = payload.plan().reads().to_vec();
    let partition = before.partition().clone();
    let local_shape = before.local_shape();
    let runtime = prepared
        .mixed_runtime_binding(&MixedSourceRetirement::completed_for_test())
        .unwrap();
    drop(prepared);
    assert!(weak.upgrade().is_none());
    let metadata = runtime.hybrid_rank().unwrap();
    metadata
        .validate_binding(&fixture.binding, &layout())
        .unwrap();
    assert_eq!(metadata.partition(), &partition);
    assert_eq!(metadata.local_shape(), local_shape);
    assert_eq!((runtime.out_features(), runtime.in_features()), (768, 256));
    let HybridRankStorage::NumericBinding(plan) = metadata.storage() else {
        panic!()
    };
    assert_eq!(plan.reads(), &reads);
    for read in plan.reads() {
        read.source().validate_source_identity().unwrap();
    }
    assert!(runtime.weight().is_err());
    assert!(runtime.parameter().weight().is_err());
    assert!(runtime.parameter().values_f32().is_err());
    assert!(runtime.parameter().values_bf16_words().is_err());
    assert!(
        HybridMixedBlockPlan::new(
            layout(),
            std::slice::from_ref(metadata),
            1,
            HybridMixedProjectionArithmetic::NativeBf16GemmEx,
            4 << 20,
            4 << 20,
            16 << 20
        )
        .is_err()
    );
    // Independent binders must not authenticate a retired source by shape/name.
    let foreign = Fixture::new([768, 256], TensorRole::LinearAttentionQkv, Some(0), true);
    assert!(
        metadata
            .validate_binding(&foreign.binding, &layout())
            .is_err()
    );
}

#[test]
fn mixed_source_retirement_error_and_unwind_leave_original_payload_live() {
    let (_fixture, prepared, weak) = linear(TensorRole::DenseMlpGate, [256, 256], Some(0));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _runtime = prepared
            .mixed_runtime_binding(&MixedSourceRetirement::completed_for_test())
            .unwrap();
        panic!("replacement construction failed before owner commit");
    }));
    assert!(result.is_err());
    assert!(weak.upgrade().is_some());
    assert!(matches!(
        prepared.hybrid_rank().unwrap().storage(),
        HybridRankStorage::Numeric(_)
    ));
    // A failed/dropped claim never changes the observation held by its image.
    let observation = MixedSourceRetirement::default();
    let abandoned = observation.clone();
    drop(abandoned);
    assert!(!observation.is_complete());
    assert!(prepared.mixed_runtime_binding(&observation).is_err());
    drop(prepared);
    assert!(weak.upgrade().is_none());
}

fn norm_from_source(fixture: &Fixture) -> PreparedNorm {
    use crate::checkpoint::CheckpointDType;
    use crate::nn::{ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec};
    let mut slice = fixture.binding.weight().slice().clone();
    slice.name = "norm.weight".into();
    slice.role = TensorRole::OutputNorm;
    slice.dtype = CheckpointDType::Bf16;
    slice.shape = vec![256];
    slice.bytes = 512;
    let path = ModulePath::new("norm").unwrap();
    let spec = ParameterSpec::new(
        ParameterId::new(71),
        path.clone(),
        ParameterDType::Bf16,
        vec![256],
        ParameterResidency::Static,
    )
    .unwrap();
    let mut schema = StateDictSchema::builder();
    schema
        .register_with_role(spec, TensorRole::OutputNorm)
        .unwrap();
    let mut mapper = ExactNameMapper::new();
    mapper
        .insert(&slice.name, NameMapping::weight(path.clone()))
        .unwrap();
    let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(vec![slice])
        .unwrap();
    let materializer = StateDictMaterializer::new(4096).unwrap();
    PreparedNorm::new(
        materializer.parameter(bound.get(&path).unwrap()).unwrap(),
        1e-6,
    )
    .unwrap()
}

#[test]
fn mixed_source_retirement_layer_and_delta_aliases_switch_together() {
    let (fixture, projection, weak) = linear(TensorRole::LinearAttentionQkv, [768, 256], Some(0));
    let norm = norm_from_source(&fixture);
    // Ownership fixture: all projection fields alias the same valid FP8 source.
    // No forward is executed; this detects an omitted field in the conversion.
    let parameter = Arc::new(norm.parameter().clone());
    let delta = PreparedGatedDeltaNetBlock {
        descriptor: GatedDeltaNetAttention::new(256, 4, 4, 64, 64, 3, 1e-6, false).unwrap(),
        vector_parameters: [parameter.clone(), parameter.clone(), parameter],
        input_norm: norm.clone(),
        qkv: projection.clone(),
        z: projection.clone(),
        a: projection.clone(),
        beta: projection.clone(),
        output: projection.clone(),
        conv: Arc::from([]),
        a_log: Arc::from([]),
        dt_bias: Arc::from([]),
        norm: norm.clone(),
    };
    let mut delta_directory = BTreeMap::from([(0, delta.clone())]);
    assert!(matches!(
        delta_directory[&0].qkv.hybrid_rank().unwrap().storage(),
        HybridRankStorage::Numeric(_)
    ));
    let ff = PreparedFeedForwardBlock {
        norm: norm.clone(),
        kind: PreparedFeedForwardKind::Routed {
            router: projection.clone(),
            descriptor: Moe::new(
                256,
                256,
                MoeRouterSpec::new(
                    2,
                    1,
                    RouterScoreFunction::Softmax,
                    RouterSelection::TopK,
                    true,
                    1.,
                )
                .unwrap(),
                false,
            )
            .unwrap(),
            shared: None,
            shared_gate: Some(projection.clone()),
            experts: Box::new([]),
        },
    };
    let mut layers = vec![TransformerLayer::new(
        0,
        Connected::new(AddResidual, PreparedAttentionBlock::GatedDeltaNet(delta)),
        Connected::new(AddResidual, ff),
    )];
    let mut output = Some(OutputPipeline::new(NoReduction, norm, projection));
    assert!(
        MixedRuntimeBindings::prepare(&layers, output.as_ref(), &MixedSourceRetirement::default())
            .is_err()
    );
    assert!(weak.upgrade().is_some());
    let runtime = MixedRuntimeBindings::prepare(
        &layers,
        output.as_ref(),
        &MixedSourceRetirement::completed_for_test(),
    )
    .unwrap();
    layers = runtime.layers;
    output = runtime.output;
    assert!(
        weak.upgrade().is_some(),
        "old GDN directory must remain an owner until replaced"
    );
    delta_directory = runtime.deltas;
    assert!(
        weak.upgrade().is_none(),
        "a runtime field still retains the FP8 source"
    );
    assert_eq!(layers[0].index(), 0);
    let delta = &delta_directory[&0];
    for linear in [
        &delta.qkv,
        &delta.z,
        &delta.a,
        &delta.beta,
        &delta.output,
        output.as_ref().unwrap().head(),
    ] {
        assert!(matches!(
            linear.hybrid_rank().unwrap().storage(),
            HybridRankStorage::NumericBinding(_)
        ));
    }
}

#[test]
fn mixed_source_retirement_swiglu_releases_all_three_sources() {
    let (_gate_file, gate, gate_weak) = linear(TensorRole::DenseMlpGate, [256, 256], Some(0));
    let (_up_file, up, up_weak) = linear(TensorRole::DenseMlpUp, [256, 256], Some(0));
    let (_down_file, down, down_weak) = linear(TensorRole::DenseMlpDown, [256, 256], Some(0));
    let prepared = PreparedSwiGlu::new(gate, up, down, Some(7.)).unwrap();
    let runtime = swiglu(&prepared, &MixedSourceRetirement::completed_for_test()).unwrap();
    assert_eq!(runtime.activation_limit(), prepared.activation_limit());
    drop(prepared);
    for weak in [gate_weak, up_weak, down_weak] {
        assert!(weak.upgrade().is_none());
    }
    for linear in [runtime.gate(), runtime.up(), runtime.down()] {
        assert!(matches!(
            linear.hybrid_rank().unwrap().storage(),
            HybridRankStorage::NumericBinding(_)
        ));
    }
}

#[test]
fn mixed_source_retirement_gqa_releases_all_four_sources() {
    let (query_file, query, query_weak) = linear(TensorRole::AttentionQuery, [256, 256], Some(1));
    let (_key_file, key, key_weak) = linear(TensorRole::AttentionKey, [128, 256], Some(1));
    let (_value_file, value, value_weak) = linear(TensorRole::AttentionValue, [128, 256], Some(1));
    let (_output_file, output, output_weak) =
        linear(TensorRole::AttentionOutput, [256, 256], Some(1));
    let norm = norm_from_source(&query_file);
    let rotary = RotaryEmbedding::new(
        64,
        10000.,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 64 },
        RotaryScaling::None,
    )
    .unwrap();
    let descriptor = DecoderLayer::new(
        1,
        RmsNorm::new(256, 1e-6).unwrap(),
        Attention::Gqa(GqaAttention::new(256, 4, 2, 64, false, rotary).unwrap()),
        Residual::Add,
        RmsNorm::new(256, 1e-6).unwrap(),
        FeedForward::SwiGlu(SwiGlu::new(256, 256, false).unwrap()),
        Residual::Add,
    )
    .unwrap();
    let rope = PreparedRope::new(1, 64, vec![1.; 32], vec![0.; 32]).unwrap();
    let prepared = PreparedAttentionBlock::Gqa(PreparedGqaBlock {
        descriptor,
        input_norm: norm.clone(),
        query,
        key,
        value,
        output,
        query_norm: Some(norm.clone()),
        key_norm: Some(norm),
        rope: rope.clone(),
    });
    let runtime = attention(&prepared, &MixedSourceRetirement::completed_for_test()).unwrap();
    drop(prepared);
    for weak in [query_weak, key_weak, value_weak, output_weak] {
        assert!(weak.upgrade().is_none());
    }
    let PreparedAttentionBlock::Gqa(runtime) = runtime else {
        panic!("retirement changed attention kind")
    };
    assert_eq!(runtime.rope, rope);
    assert!(runtime.query_norm.is_some());
    assert!(runtime.key_norm.is_some());
    assert!(runtime.input_norm.parameter().values_f32().is_ok());
    for linear in [
        &runtime.query,
        &runtime.key,
        &runtime.value,
        &runtime.output,
    ] {
        assert!(matches!(
            linear.hybrid_rank().unwrap().storage(),
            HybridRankStorage::NumericBinding(_)
        ));
        assert!(linear.weight().is_err());
    }
}
