//! Independent MoE source fixture; all route/data control uses MixedRankCollective.
use super::*;

pub(super) fn resources() -> (base::Fixture, BoundDecoderResources) {
    let original = base::Fixture::for_degree(2, true);
    let source = original.resources();
    let s = source.spec();
    let mut layers = s.layers().to_vec();
    let layer = &layers[3];
    let moe = Moe::new(
        8,
        12,
        MoeRouterSpec::new(
            16,
            8,
            RouterScoreFunction::Softmax,
            RouterSelection::TopK,
            true,
            1.,
        )
        .unwrap(),
        false,
    )
    .unwrap()
    .with_shared_expert(SwiGlu::new(8, 12, false).unwrap())
    .unwrap()
    .with_shared_expert_gate(Linear::new(8, 1, false).unwrap())
    .unwrap();
    layers[3] = DecoderLayer::new(
        3,
        layer.input_norm().clone(),
        layer.attention().clone(),
        Residual::Add,
        layer.post_attention_norm().clone(),
        FeedForward::Moe(moe),
        Residual::Add,
    )
    .unwrap();
    let spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: s.architecture().into(),
        hidden_size: 8,
        vocab_size: s.vocab_size(),
        max_sequence_length: Some(32),
        token_embedding: s.token_embedding().clone(),
        layers,
        final_norm: s.final_norm().clone(),
        output: s.output().clone(),
        tie_word_embeddings: false,
    })
    .unwrap();
    let mut oracle = original.oracle.clone();
    let tensors = oracle["tensors"].as_object_mut().unwrap();
    tensors.retain(|key, _| !key.starts_with("layers.3.feed_forward."));
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("hybrid_mixed_moe_cuda_oracle.json")).unwrap();
    tensors.extend(reference["tensors"].as_object().unwrap().clone());
    let fixture = base::Fixture::with_oracle(oracle);
    let resources = native_resources_from(&fixture, fixture.resources_for(spec));
    (fixture, resources)
}

pub(super) fn boundary_resources() -> (base::Fixture, BoundDecoderResources) {
    let (original, _) = resources();
    let (_, moe) = resources();
    let spec = moe.spec();
    let layers = spec.layers()[1..]
        .iter()
        .enumerate()
        .map(|(index, layer)| {
            DecoderLayer::new(
                index,
                layer.input_norm().clone(),
                layer.attention().clone(),
                Residual::Add,
                layer.post_attention_norm().clone(),
                layer.feed_forward().clone(),
                Residual::Add,
            )
            .unwrap()
        })
        .collect();
    let descriptor = DecoderModelSpec::new(DecoderModelParts {
        architecture: spec.architecture().into(),
        hidden_size: 8,
        vocab_size: 11,
        max_sequence_length: Some(32),
        token_embedding: spec.token_embedding().clone(),
        layers,
        final_norm: spec.final_norm().clone(),
        output: spec.output().clone(),
        tie_word_embeddings: false,
    })
    .unwrap();
    let mut oracle = original.oracle.clone();
    let tensors = oracle["tensors"].as_object_mut().unwrap();
    let old = tensors.clone();
    tensors.clear();
    for (name, value) in old {
        if name.starts_with("layers.0.") {
            continue;
        }
        let name = if name.starts_with("layers.") {
            let mut parts = name.split('.').map(str::to_owned).collect::<Vec<_>>();
            parts[1] = (parts[1].parse::<usize>().unwrap() - 1).to_string();
            parts.join(".")
        } else {
            name
        };
        tensors.insert(name, value);
    }
    let boundary: serde_json::Value =
        serde_json::from_str(include_str!("hybrid_mixed_boundary_oracle.json")).unwrap();
    tensors.insert(
        "token_embedding.weight".into(),
        boundary["embedding"].clone(),
    );
    let fixture = base::Fixture::with_oracle(oracle);
    let resources = native_resources_from(&fixture, fixture.resources_for(descriptor));
    (fixture, resources)
}

#[test]
fn cpu_prepare_all_native_expert_shards_preserves_source_ids_and_exact_grants() {
    let (_fixture, resources) = resources();
    let plan = parallel(&resources, 1..4, 2);
    for lane in 0..2 {
        let layout = Arc::new(plan.rank_layout(lane).unwrap());
        let reader =
            StateDictMaterializer::for_hybrid_cuda_rank(1 << 20, layout.clone(), 4 << 20).unwrap();
        let mut parameters = Vec::new();
        for binding in resources.state_dict().parameters() {
            if let ParameterResidency::Expert { layer, expert } = *binding.residency() {
                assert_eq!(layer, 3);
                let p = reader.expert_parameter(layer, expert, binding).unwrap();
                assert!(reader.expert_parameter(layer + 1, expert, binding).is_err());
                let rank = p.hybrid_rank().unwrap();
                rank.validate_binding(binding, &layout).unwrap();
                assert!(matches!(rank.storage(), HybridRankStorage::Bf16(_)));
                assert_eq!(
                    rank.local_shape(),
                    if binding.role() == &ferrule_model::TensorRole::RoutedExpertDown {
                        [8, 6]
                    } else {
                        [6, 8]
                    }
                );
                parameters.push(rank.clone());
            }
        }
        assert_eq!(parameters.len(), 16 * 3);
        let image = HybridMixedBlockPlan::new(
            layout.clone(),
            &parameters,
            2,
            HybridMixedProjectionArithmetic::NativeBf16GemmEx,
            4 << 20,
            64 << 20,
            16 << 20,
        )
        .unwrap();
        assert!(image.all_expert_shards_resident(3));
        assert_eq!(image.weight_bytes(), 16 * 3 * 48 * 2);
        assert_eq!(
            image.weight_grant_bytes(),
            parameters
                .iter()
                .map(|p| layout
                    .parameter_budget(p.binding())
                    .unwrap()
                    .allocation_grant_bytes)
                .sum::<usize>()
        );
        assert!(
            HybridMixedBlockPlan::new(
                layout,
                &parameters[..47],
                2,
                HybridMixedProjectionArithmetic::NativeBf16GemmEx,
                4 << 20,
                64 << 20,
                16 << 20
            )
            .is_err()
        );
    }
}
