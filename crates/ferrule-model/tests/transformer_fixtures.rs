use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferrule_model::ModelFamily;
use ferrule_model::TensorClass;
use ferrule_model::TensorRole;
use ferrule_model::checkpoint::{
    CheckpointDType, CheckpointTensorSlice, HfSafetensorsInventory, HfSafetensorsTensorInfo,
};
use ferrule_model::nn::{
    Module, ModuleNode, ModulePath, ModuleVisitError, ModuleVisitor, ParameterDType, ParameterId,
    ParameterPart, ParameterResidency, ParameterSpec, ParameterSpecError, StorageEncoding,
};

use ferrule_model::transformer::*;

fn path(value: &str) -> ModulePath {
    ModulePath::new(value).unwrap()
}

fn parameter(
    id: u64,
    path_value: &str,
    dtype: ParameterDType,
    shape: impl Into<Vec<usize>>,
    residency: ParameterResidency,
) -> ParameterSpec {
    ParameterSpec::new(
        ParameterId::new(id),
        path(path_value),
        dtype,
        shape,
        residency,
    )
    .unwrap()
}

#[test]
fn module_tree_visits_modules_and_parameters_in_registration_order() {
    let mut root = ModuleNode::new(ModulePath::root(), "Decoder").unwrap();
    let mut embedding = ModuleNode::new(path("embed_tokens"), "Embedding").unwrap();
    embedding
        .add_parameter(parameter(
            1,
            "embed_tokens.weight",
            ParameterDType::Bf16,
            [16, 8],
            ParameterResidency::Static,
        ))
        .unwrap();

    let mut layers = ModuleNode::new(path("layers"), "ModuleList").unwrap();
    let mut layer = ModuleNode::new(path("layers.0"), "DecoderLayer").unwrap();
    layer
        .add_parameter(parameter(
            2,
            "layers.0.input_norm",
            ParameterDType::Bf16,
            [8],
            ParameterResidency::layer(0),
        ))
        .unwrap();
    layers.add_child(layer).unwrap();
    root.add_child(embedding)
        .unwrap()
        .add_child(layers)
        .unwrap();

    #[derive(Default)]
    struct Visitor {
        events: Vec<String>,
    }

    impl ModuleVisitor for Visitor {
        fn enter_module(&mut self, path: &ModulePath, kind: &str) -> Result<(), ModuleVisitError> {
            self.events.push(format!("enter:{path}:{kind}"));
            Ok(())
        }

        fn visit_parameter(&mut self, parameter: &ParameterSpec) -> Result<(), ModuleVisitError> {
            self.events.push(format!("parameter:{}", parameter.path()));
            Ok(())
        }

        fn exit_module(&mut self, path: &ModulePath, kind: &str) -> Result<(), ModuleVisitError> {
            self.events.push(format!("exit:{path}:{kind}"));
            Ok(())
        }
    }

    let mut visitor = Visitor::default();
    let module: &dyn Module = &root;
    module.visit(&mut visitor).unwrap();
    assert_eq!(
        visitor.events,
        [
            "enter::Decoder",
            "enter:embed_tokens:Embedding",
            "parameter:embed_tokens.weight",
            "exit:embed_tokens:Embedding",
            "enter:layers:ModuleList",
            "enter:layers.0:DecoderLayer",
            "parameter:layers.0.input_norm",
            "exit:layers.0:DecoderLayer",
            "exit:layers:ModuleList",
            "exit::Decoder",
        ]
    );
}

#[test]
fn schema_rejects_duplicate_id_path_and_alias_cycles() {
    let first = parameter(
        1,
        "first.weight",
        ParameterDType::Bf16,
        [2, 2],
        ParameterResidency::Static,
    );
    let duplicate_path = parameter(
        2,
        "first.weight",
        ParameterDType::Bf16,
        [2, 2],
        ParameterResidency::Static,
    );
    let duplicate_id = parameter(
        1,
        "other.weight",
        ParameterDType::Bf16,
        [2, 2],
        ParameterResidency::Static,
    );

    let mut builder = StateDictSchema::builder();
    builder.register(first.clone()).unwrap();
    assert!(matches!(
        builder.register(duplicate_path),
        Err(StateDictSchemaError::DuplicatePath { .. })
    ));
    assert!(matches!(
        builder.register(duplicate_id),
        Err(StateDictSchemaError::DuplicateId { .. })
    ));

    let left = first.with_alias(ParameterId::new(2));
    let right = parameter(
        2,
        "second.weight",
        ParameterDType::Bf16,
        [2, 2],
        ParameterResidency::Static,
    )
    .with_alias(ParameterId::new(1));
    let mut cycle = StateDictSchema::builder();
    cycle.register(left).unwrap().register(right).unwrap();
    assert!(matches!(
        cycle.build(),
        Err(StateDictSchemaError::AliasCycle { .. })
    ));
}

#[test]
fn exact_mapper_and_binder_preserve_lazy_parts_identity_aliases_and_expert_residency() {
    let embedding = parameter(
        1,
        "embed_tokens.weight",
        ParameterDType::Bf16,
        [4, 2],
        ParameterResidency::Static,
    );
    let output = parameter(
        2,
        "lm_head.weight",
        ParameterDType::Bf16,
        [4, 2],
        ParameterResidency::Static,
    )
    .with_alias(ParameterId::new(1));
    let expert = parameter(
        3,
        "layers.0.experts.7.gate_proj.weight",
        ParameterDType::F8E4M3,
        [2, 2],
        ParameterResidency::expert(0, 7),
    )
    .with_required_scale(ParameterDType::F8E8M0, [1, 1])
    .unwrap();
    let optional_bias = parameter(
        4,
        "layers.0.self_attn.q_proj.bias",
        ParameterDType::Bf16,
        [2],
        ParameterResidency::layer(0),
    )
    .with_optional(true);
    let mut schema = StateDictSchema::builder();
    schema
        .register(embedding)
        .unwrap()
        .register(output)
        .unwrap()
        .register(expert)
        .unwrap()
        .register(optional_bias)
        .unwrap();
    let schema = schema.build().unwrap();

    let mut mapper = ExactNameMapper::new();
    mapper
        .insert(
            "model.embed_tokens.weight",
            NameMapping::weight(path("embed_tokens.weight")),
        )
        .unwrap()
        .insert(
            "model.layers.0.mlp.experts.7.gate_proj.weight",
            NameMapping::weight(path("layers.0.experts.7.gate_proj.weight")),
        )
        .unwrap()
        .insert(
            "model.layers.0.mlp.experts.7.gate_proj.weight_scale_inv",
            NameMapping::scale(path("layers.0.experts.7.gate_proj.weight")),
        )
        .unwrap();
    let object_safe: &dyn NameMapper = &mapper;

    let fixture = CheckpointFixture::new(32);
    let bound = StateDictBinder::new(&schema, object_safe)
        .bind_slices([
            fixture.slice(
                "model.embed_tokens.weight",
                0,
                16,
                CheckpointDType::Bf16,
                [4, 2],
            ),
            fixture.slice(
                "model.layers.0.mlp.experts.7.gate_proj.weight",
                16,
                4,
                CheckpointDType::F8E4M3,
                [2, 2],
            ),
            fixture.slice(
                "model.layers.0.mlp.experts.7.gate_proj.weight_scale_inv",
                20,
                1,
                CheckpointDType::F8E8M0,
                [1, 1],
            ),
        ])
        .unwrap();

    assert_eq!(bound.len(), 3);
    let embedding = bound.get(&path("embed_tokens.weight")).unwrap();
    let output = bound.get(&path("lm_head.weight")).unwrap();
    assert!(output.is_alias());
    assert!(embedding.shares_storage_with(output));
    assert_eq!(embedding.weight().slice().offset, 0);
    assert_eq!(embedding.weight().logical_shape(), [4, 2]);
    assert_eq!(
        embedding.weight().source_identity().catalog_path(),
        fixture.path()
    );

    let experts = bound.expert(0, 7).collect::<Vec<_>>();
    assert_eq!(experts.len(), 1);
    assert_eq!(experts[0].scale().unwrap().slice().bytes, 1);
    assert!(bound.validate_source_identities());
}

#[test]
fn prepared_executable_declares_real_decoder_stages_and_manifests() {
    let config = serde_json::json!({
        "vocab_size": 4,
        "hidden_size": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "intermediate_size": 4,
        "num_experts": 1,
        "experts_per_token": 1,
        "max_position_embeddings": 16,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": false
    });
    let recipe = SyntheticDecoderRecipe::new();
    let output = recipe.build(&config).unwrap();
    let fixture = CheckpointFixture::new(4096);
    let mut offset = 0u64;
    let slices = output
        .schema()
        .parameters()
        .iter()
        .filter(|parameter| parameter.alias_of().is_none())
        .map(|parameter| {
            let elements = parameter.shape().iter().product::<usize>();
            let bytes = elements as u64 * 4;
            let slice = fixture.slice(
                &SyntheticDecoderRecipe::external_name(parameter.path()),
                offset,
                bytes,
                CheckpointDType::F32,
                parameter.shape().to_vec(),
            );
            offset += bytes;
            slice
        })
        .collect::<Vec<_>>();
    let resources = DecoderLoadOptions::new(&recipe, &config)
        .bind_slices(slices)
        .unwrap();
    let executable = resources.prepared_executable(2).unwrap();

    assert!(!executable.resources().is_empty());
    assert_eq!(executable.stages().len(), 8);
    assert_eq!(
        executable.stages()[0].operation(),
        &ferrule_model::execution::TransformerStage::Embed
    );
    assert_eq!(
        executable.stages()[1].operation(),
        &ferrule_model::execution::TransformerStage::Attention { layer: 0 }
    );
    assert_eq!(
        executable.stages()[2].operation(),
        &ferrule_model::execution::TransformerStage::Router { layer: 0 }
    );
    assert_eq!(
        executable.stages()[3].operation(),
        &ferrule_model::execution::TransformerStage::FeedForward { layer: 0 }
    );
    assert_eq!(
        executable.stages().last().unwrap().operation(),
        &ferrule_model::execution::TransformerStage::Output
    );
    assert!(
        executable
            .stages()
            .iter()
            .filter(|stage| !matches!(
                stage.operation(),
                ferrule_model::execution::TransformerStage::FeedForward { .. }
            ))
            .all(|stage| !stage.resources().is_empty())
    );
}

#[test]
fn hf_inventory_binding_passes_metadata_and_keeps_checkpoint_slices_lazy() {
    struct MetadataMapper;

    impl NameMapper for MetadataMapper {
        fn map(
            &self,
            external_name: &str,
            meta: ExternalTensorMeta<'_>,
        ) -> Result<Option<NameMapping>, NameMapError> {
            if external_name != "model.projection.weight" {
                return Ok(None);
            }
            if meta.dtype != "BF16" || meta.shape != [2, 2] || meta.bytes != 8 {
                return Err(NameMapError::new("unexpected inventory metadata"));
            }
            Ok(Some(NameMapping::weight(path("projection.weight"))))
        }
    }

    let mut schema = StateDictSchema::builder();
    schema
        .register(parameter(
            1,
            "projection.weight",
            ParameterDType::Bf16,
            [2, 2],
            ParameterResidency::Static,
        ))
        .unwrap();
    let schema = schema.build().unwrap();
    let fixture = CheckpointFixture::new(8);
    let inventory = HfSafetensorsInventory {
        family: ModelFamily::Unknown("fixture".into()),
        total_size: Some(8),
        shard_count: 1,
        tensor_count: 1,
        tensors: vec![HfSafetensorsTensorInfo {
            name: "model.projection.weight".into(),
            shard: "model.safetensors".into(),
            dtype: "BF16".into(),
            shape: vec![2, 2],
            data_offset: 0,
            file_offset: 0,
            byte_size: 8,
            class: TensorClass::Unknown,
            role: TensorRole::AttentionQuery,
        }],
        dtype_counts: Vec::new(),
        class_counts: Vec::new(),
        role_counts: Vec::new(),
        shard_summaries: Vec::new(),
        index_only_tensors: Vec::new(),
        header_only_tensors: Vec::new(),
    };

    let bound = StateDictBinder::new(&schema, &MetadataMapper)
        .bind_hf(fixture.directory(), &inventory)
        .unwrap();
    let projection = bound.get(&path("projection.weight")).unwrap();
    assert_eq!(projection.weight().slice().path, fixture.path());
    assert_eq!(projection.weight().slice().role, TensorRole::AttentionQuery);
    assert_eq!(projection.weight().slice().offset, 0);
    assert_eq!(projection.weight().slice().bytes, 8);
}

#[test]
fn strict_binder_aggregates_missing_duplicate_unexpected_shape_dtype_and_pairing() {
    let fp8_parameter = |id, name| {
        ParameterSpec::new_encoded(
            ParameterId::new(id),
            path(name),
            ParameterDType::F8E4M3,
            [2, 2],
            [2, 2],
            StorageEncoding::Fp8Block128,
            ParameterResidency::Static,
        )
        .unwrap()
        .with_required_scale(ParameterDType::F8E8M0, [1, 1])
        .unwrap()
    };
    let first = fp8_parameter(1, "first.weight");
    let missing_scale = fp8_parameter(2, "missing_scale.weight");
    let orphan_scale = fp8_parameter(3, "orphan_scale.weight");
    let mut schema = StateDictSchema::builder();
    schema
        .register(first)
        .unwrap()
        .register(missing_scale)
        .unwrap()
        .register(orphan_scale)
        .unwrap();
    let schema = schema.build().unwrap();

    let mut mapper = ExactNameMapper::new();
    for (external, canonical, part) in [
        ("first.bad_shape", "first.weight", ParameterPart::Weight),
        ("first.duplicate", "first.weight", ParameterPart::Weight),
        ("first.bad_scale", "first.weight", ParameterPart::Scale),
        (
            "missing_scale.weight",
            "missing_scale.weight",
            ParameterPart::Weight,
        ),
        (
            "orphan_scale.scale",
            "orphan_scale.weight",
            ParameterPart::Scale,
        ),
    ] {
        mapper
            .insert(
                external,
                NameMapping::new(path(canonical), part, TensorTransform::Identity),
            )
            .unwrap();
    }

    let fixture = CheckpointFixture::new(64);
    let error = StateDictBinder::new(&schema, &mapper)
        .bind_slices([
            fixture.slice("first.bad_shape", 0, 4, CheckpointDType::F8E4M3, [4]),
            fixture.slice("first.duplicate", 4, 4, CheckpointDType::F8E4M3, [2, 2]),
            fixture.slice("first.bad_scale", 8, 2, CheckpointDType::Bf16, [1, 1]),
            fixture.slice(
                "missing_scale.weight",
                10,
                4,
                CheckpointDType::F8E4M3,
                [2, 2],
            ),
            fixture.slice("orphan_scale.scale", 14, 1, CheckpointDType::F8E8M0, [1, 1]),
            fixture.slice("totally.unexpected", 15, 2, CheckpointDType::Bf16, [1]),
        ])
        .unwrap_err();

    assert!(
        error
            .issues()
            .iter()
            .any(|issue| matches!(issue, BindingIssue::ShapeMismatch { .. }))
    );
    assert!(
        error
            .issues()
            .iter()
            .any(|issue| matches!(issue, BindingIssue::DTypeMismatch { .. }))
    );
    assert!(
        error
            .issues()
            .iter()
            .any(|issue| matches!(issue, BindingIssue::DuplicateTensorPart { .. }))
    );
    assert!(
        error
            .issues()
            .iter()
            .any(|issue| matches!(issue, BindingIssue::UnexpectedTensor { .. }))
    );
    assert!(
        error
            .issues()
            .iter()
            .any(|issue| matches!(issue, BindingIssue::MissingParameter { .. }))
    );
    assert!(
        error
            .issues()
            .iter()
            .any(|issue| matches!(issue, BindingIssue::MissingScale { .. }))
    );
    assert!(
        error
            .issues()
            .iter()
            .any(|issue| matches!(issue, BindingIssue::OrphanScale { .. }))
    );
}

#[test]
fn transpose_is_lazy_and_split_concat_fail_closed() {
    let parameter = parameter(
        1,
        "projection.weight",
        ParameterDType::Bf16,
        [2, 3],
        ParameterResidency::Static,
    );
    let mut schema = StateDictSchema::builder();
    schema.register(parameter).unwrap();
    let schema = schema.build().unwrap();
    let fixture = CheckpointFixture::new(32);

    let mut transpose = ExactNameMapper::new();
    transpose
        .insert(
            "projection.transposed",
            NameMapping::new(
                path("projection.weight"),
                ParameterPart::Weight,
                TensorTransform::transpose_2d(),
            ),
        )
        .unwrap();
    let bound = StateDictBinder::new(&schema, &transpose)
        .bind_slices([fixture.slice(
            "projection.transposed",
            0,
            12,
            CheckpointDType::Bf16,
            [3, 2],
        )])
        .unwrap();
    let projection = bound.get(&path("projection.weight")).unwrap();
    assert_eq!(projection.weight().slice().shape, [3, 2]);
    assert_eq!(projection.weight().logical_shape(), [2, 3]);
    assert_eq!(
        projection.weight().transform(),
        &TensorTransform::Transpose { axes: vec![1, 0] }
    );

    for transform in [
        TensorTransform::Split {
            axis: 0,
            index: 0,
            parts: 2,
        },
        TensorTransform::Concat {
            axis: 0,
            index: 0,
            parts: 2,
        },
    ] {
        let mut mapper = ExactNameMapper::new();
        mapper
            .insert(
                "projection.unsupported",
                NameMapping::new(path("projection.weight"), ParameterPart::Weight, transform),
            )
            .unwrap();
        let error = StateDictBinder::new(&schema, &mapper)
            .bind_slices([fixture.slice(
                "projection.unsupported",
                0,
                12,
                CheckpointDType::Bf16,
                [2, 3],
            )])
            .unwrap_err();
        assert!(
            error
                .issues()
                .iter()
                .any(|issue| matches!(issue, BindingIssue::InvalidTransform { .. }))
        );
    }
}

#[test]
fn qwen_like_gqa_fixture_is_family_neutral() {
    let rotary = RotaryEmbedding::new(
        4,
        10_000.0,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 4 },
        RotaryScaling::None,
    )
    .unwrap();
    let attention = GqaAttention::new(8, 2, 1, 4, false, rotary)
        .unwrap()
        .with_qk_norms(
            RmsNorm::new(4, 1e-6).unwrap(),
            RmsNorm::new(4, 1e-6).unwrap(),
        )
        .unwrap();
    let layer = DecoderLayer::new(
        0,
        RmsNorm::new(8, 1e-6).unwrap(),
        Attention::Gqa(attention),
        Residual::Add,
        RmsNorm::new(8, 1e-6).unwrap(),
        FeedForward::SwiGlu(SwiGlu::new(8, 16, false).unwrap()),
        Residual::Add,
    )
    .unwrap();
    let spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: "qwen-like-gqa".into(),
        hidden_size: 8,
        vocab_size: 32,
        max_sequence_length: Some(4096),
        token_embedding: Embedding::new(32, 8, None).unwrap(),
        layers: vec![layer],
        final_norm: RmsNorm::new(8, 1e-6).unwrap(),
        output: Linear::new(8, 32, false).unwrap(),
        tie_word_embeddings: true,
    })
    .unwrap();

    assert_eq!(spec.architecture(), "qwen-like-gqa");
    assert!(spec.tie_word_embeddings());
    let Attention::Gqa(attention) = spec.layers()[0].attention() else {
        panic!("expected GQA fixture")
    };
    assert_eq!(attention.num_heads(), 2);
    assert_eq!(attention.num_kv_heads(), 1);
}

#[test]
fn deepseek_like_mla_moe_hyper_fixture_is_family_neutral() {
    let rotary = RotaryEmbedding::new(
        4,
        10_000.0,
        RotaryPairing::Interleaved,
        RotaryRegion::Tail { dimensions: 2 },
        RotaryScaling::YaRN {
            factor: 4.0,
            original_max_position_embeddings: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: Some(1.1),
        },
    )
    .unwrap();
    let query = MlaQueryProjection::low_rank(8, 8, 4, 1e-6, false).unwrap();
    let attention = MlaAttention::new(
        MlaDimensions {
            hidden_size: 8,
            num_heads: 2,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 2,
            value_head_dim: 2,
            kv_lora_rank: 4,
        },
        query,
        1e-6,
        false,
        rotary,
    )
    .unwrap();
    let router = MoeRouterSpec::new(
        4,
        2,
        RouterScoreFunction::Sigmoid,
        RouterSelection::GroupLimitedTopK {
            groups: 2,
            selected_groups: 1,
        },
        true,
        1.0,
    )
    .unwrap();
    let moe = Moe::new(8, 4, router, false)
        .unwrap()
        .with_shared_expert(SwiGlu::new(8, 8, false).unwrap())
        .unwrap();
    let hyper = Residual::Hyper(HyperResidual::new(2, 1e-5, 10).unwrap());
    let layer = DecoderLayer::new(
        0,
        RmsNorm::new(8, 1e-6).unwrap(),
        Attention::Mla(attention),
        hyper.clone(),
        RmsNorm::new(8, 1e-6).unwrap(),
        FeedForward::Moe(moe),
        hyper,
    )
    .unwrap();
    let spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: "deepseek-like-mla-moe".into(),
        hidden_size: 8,
        vocab_size: 64,
        max_sequence_length: Some(16_384),
        token_embedding: Embedding::new(64, 8, None).unwrap(),
        layers: vec![layer],
        final_norm: RmsNorm::new(8, 1e-6).unwrap(),
        output: Linear::new(8, 64, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap();

    let Attention::Mla(attention) = spec.layers()[0].attention() else {
        panic!("expected MLA fixture")
    };
    assert_eq!(attention.kv_lora_rank(), 4);
    let FeedForward::Moe(moe) = spec.layers()[0].feed_forward() else {
        panic!("expected MoE fixture")
    };
    assert_eq!(moe.router_spec().experts_per_token(), 2);
    assert!(matches!(
        spec.layers()[0].attention_residual(),
        Residual::Hyper(_)
    ));
}

#[test]
fn decoder_recipe_is_object_safe_and_only_builds_metadata() {
    struct TinyRecipe;

    impl DecoderRecipe for TinyRecipe {
        fn build_spec(
            &self,
            config: &serde_json::Value,
        ) -> Result<DecoderModelSpec, DecoderRecipeError> {
            let vocab_size = config
                .get("vocab_size")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| DecoderRecipeError::invalid_config("missing vocab_size"))?
                as usize;
            let rotary = RotaryEmbedding::new(
                2,
                10_000.0,
                RotaryPairing::Interleaved,
                RotaryRegion::Prefix { dimensions: 2 },
                RotaryScaling::Linear { factor: 2.0 },
            )?;
            let layer = DecoderLayer::new(
                0,
                RmsNorm::new(4, 1e-6)?,
                Attention::Gqa(GqaAttention::new(4, 2, 1, 2, false, rotary)?),
                Residual::Add,
                RmsNorm::new(4, 1e-6)?,
                FeedForward::SwiGlu(SwiGlu::new(4, 8, false)?),
                Residual::Add,
            )?;
            Ok(DecoderModelSpec::new(DecoderModelParts {
                architecture: "tiny-recipe".into(),
                hidden_size: 4,
                vocab_size,
                max_sequence_length: Some(128),
                token_embedding: Embedding::new(vocab_size, 4, None)?,
                layers: vec![layer],
                final_norm: RmsNorm::new(4, 1e-6)?,
                output: Linear::new(4, vocab_size, false)?,
                tie_word_embeddings: true,
            })?)
        }

        fn build_schema(
            &self,
            config: &serde_json::Value,
        ) -> Result<StateDictSchema, DecoderRecipeError> {
            let vocab_size = config
                .get("vocab_size")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| DecoderRecipeError::invalid_config("missing vocab_size"))?
                as usize;
            let mut schema = StateDictSchema::builder();
            schema.register(parameter(
                1,
                "embed_tokens.weight",
                ParameterDType::Bf16,
                [vocab_size, 4],
                ParameterResidency::Static,
            ))?;
            schema.register(
                parameter(
                    2,
                    "lm_head.weight",
                    ParameterDType::Bf16,
                    [vocab_size, 4],
                    ParameterResidency::Static,
                )
                .with_alias(ParameterId::new(1)),
            )?;
            Ok(schema.build()?)
        }

        fn build_name_mapper(
            &self,
            _config: &serde_json::Value,
        ) -> Result<Arc<dyn NameMapper>, DecoderRecipeError> {
            let mut mapper = ExactNameMapper::new();
            mapper
                .insert(
                    "model.embed_tokens.weight",
                    NameMapping::weight(path("embed_tokens.weight")),
                )
                .map_err(DecoderRecipeError::name_mapper)?;
            Ok(Arc::new(mapper))
        }
    }

    let recipe: &dyn DecoderRecipe = &TinyRecipe;
    let output = recipe
        .build(&serde_json::json!({ "vocab_size": 16 }))
        .unwrap();
    assert_eq!(output.spec().vocab_size(), 16);
    assert_eq!(output.schema().len(), 2);
    assert!(
        output
            .name_mapper()
            .map(
                "model.embed_tokens.weight",
                ExternalTensorMeta {
                    dtype: "BF16",
                    shape: &[16, 4],
                    bytes: 128,
                },
            )
            .unwrap()
            .is_some()
    );
}

#[test]
fn encoded_parameter_specs_validate_physical_layout_and_scale_pairing() {
    let fp8 = ParameterSpec::new_encoded(
        ParameterId::new(1),
        path("fp8.weight"),
        ParameterDType::F8E4M3,
        [129, 257],
        [129, 257],
        StorageEncoding::Fp8Block128,
        ParameterResidency::Static,
    )
    .unwrap()
    .with_required_scale(ParameterDType::F8E8M0, [2, 3])
    .unwrap();
    let fp4 = ParameterSpec::new_encoded(
        ParameterId::new(2),
        path("fp4.weight"),
        ParameterDType::I8,
        [7, 64],
        [7, 32],
        StorageEncoding::PackedFp4X2 { block_size: 32 },
        ParameterResidency::expert(0, 0),
    )
    .unwrap()
    .with_required_scale(ParameterDType::F8E8M0, [7, 2])
    .unwrap();
    let dense = parameter(
        3,
        "dense.weight",
        ParameterDType::Bf16,
        [7, 64],
        ParameterResidency::Static,
    );
    let mut schema = StateDictSchema::builder();
    schema
        .register(fp8.clone())
        .unwrap()
        .register(fp4.clone())
        .unwrap()
        .register(dense.clone())
        .unwrap();
    let schema = schema.build().unwrap();

    assert_eq!(schema.storage_tensor_count(), 5);
    assert_eq!(fp8.weight().logical_shape(), [129, 257]);
    assert_eq!(fp8.weight().physical_shape(), [129, 257]);
    assert_eq!(fp8.scale().tensor().unwrap().physical_shape(), [2, 3]);
    assert_eq!(fp4.weight().logical_shape(), [7, 64]);
    assert_eq!(fp4.weight().physical_shape(), [7, 32]);
    assert_eq!(fp4.scale().tensor().unwrap().physical_shape(), [7, 2]);
    assert_eq!(dense.weight().encoding(), StorageEncoding::Dense);

    let missing_scale = ParameterSpec::new_encoded(
        ParameterId::new(4),
        path("missing.weight"),
        ParameterDType::I8,
        [1, 32],
        [1, 16],
        StorageEncoding::PackedFp4X2 { block_size: 32 },
        ParameterResidency::Static,
    )
    .unwrap();
    let mut invalid = StateDictSchema::builder();
    let error = invalid.register(missing_scale).unwrap_err();
    assert!(matches!(
        error,
        StateDictSchemaError::InvalidParameter {
            source: ParameterSpecError::MissingEncodingScale { .. },
            ..
        }
    ));

    let bad_scale = ParameterSpec::new_encoded(
        ParameterId::new(5),
        path("bad_scale.weight"),
        ParameterDType::F8E4M3,
        [129, 257],
        [129, 257],
        StorageEncoding::Fp8Block128,
        ParameterResidency::Static,
    )
    .unwrap()
    .with_required_scale(ParameterDType::F8E8M0, [1, 3])
    .unwrap();
    let mut invalid = StateDictSchema::builder();
    let error = invalid.register(bad_scale).unwrap_err();
    assert!(matches!(
        error,
        StateDictSchemaError::InvalidParameter {
            source: ParameterSpecError::InvalidEncodingScale { .. },
            ..
        }
    ));

    for dtype in [ParameterDType::Bf16, ParameterDType::F32] {
        let scaled_dense = ParameterSpec::new(
            ParameterId::new(6),
            path("scaled_dense.weight"),
            dtype,
            [7, 64],
            ParameterResidency::Static,
        )
        .unwrap()
        .with_required_scale(ParameterDType::F8E8M0, [1])
        .unwrap();
        let mut invalid = StateDictSchema::builder();
        let error = invalid.register(scaled_dense).unwrap_err();
        assert!(matches!(
            error,
            StateDictSchemaError::InvalidParameter {
                source: ParameterSpecError::DenseFloatScale,
                ..
            }
        ));
    }
}

#[test]
fn binder_and_materializer_preserve_packed_fp4_physical_shape() {
    let packed = ParameterSpec::new_encoded(
        ParameterId::new(1),
        path("expert.gate.weight"),
        ParameterDType::I8,
        [1, 32],
        [1, 16],
        StorageEncoding::PackedFp4X2 { block_size: 32 },
        ParameterResidency::expert(0, 0),
    )
    .unwrap()
    .with_required_scale(ParameterDType::F8E8M0, [1, 1])
    .unwrap();
    let mut schema = StateDictSchema::builder();
    schema
        .register_with_role(packed, TensorRole::RoutedExpertGate)
        .unwrap();
    let schema = schema.build().unwrap();
    let mut mapper = ExactNameMapper::new();
    mapper
        .insert(
            "expert.gate.weight",
            NameMapping::weight(path("expert.gate.weight")),
        )
        .unwrap()
        .insert(
            "expert.gate.scale",
            NameMapping::scale(path("expert.gate.weight")),
        )
        .unwrap();
    let fixture = CheckpointFixture::new(17);
    let bound = StateDictBinder::new(&schema, &mapper)
        .bind_slices([
            fixture.slice("expert.gate.weight", 0, 16, CheckpointDType::I8, [1, 16]),
            fixture.slice("expert.gate.scale", 16, 1, CheckpointDType::F8E8M0, [1, 1]),
        ])
        .unwrap();
    let binding = bound.get(&path("expert.gate.weight")).unwrap();
    assert_eq!(binding.weight().logical_shape(), [1, 32]);
    assert_eq!(binding.weight().physical_shape(), [1, 16]);
    assert_eq!(
        binding.weight().encoding(),
        StorageEncoding::PackedFp4X2 { block_size: 32 }
    );

    let prepared = StateDictMaterializer::new(32)
        .unwrap()
        .parameter(binding)
        .unwrap();
    assert_eq!(prepared.weight().slice.shape, [1, 16]);
    assert_eq!(prepared.scale().unwrap().slice.shape, [1, 1]);
    let linear = prepared.into_linear(TensorRole::RoutedExpertGate).unwrap();
    assert_eq!(linear.in_features(), 32);
    assert_eq!(linear.out_features(), 1);
}

struct CheckpointFixture {
    directory: PathBuf,
    path: PathBuf,
}

impl CheckpointFixture {
    fn new(bytes: usize) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("ferrule-state-dict-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("model.safetensors");
        std::fs::write(&path, vec![0; bytes]).unwrap();
        Self { directory, path }
    }

    fn directory(&self) -> &Path {
        &self.directory
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn slice(
        &self,
        name: &str,
        offset: u64,
        bytes: u64,
        dtype: CheckpointDType,
        shape: impl Into<Vec<usize>>,
    ) -> CheckpointTensorSlice {
        CheckpointTensorSlice {
            name: name.into(),
            role: TensorRole::Unknown,
            path: self.path.clone(),
            offset,
            bytes,
            dtype,
            shape: shape.into(),
        }
    }
}

impl Drop for CheckpointFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
