use ferrule_model::TensorRole;
use ferrule_model::models::qwen3::Qwen3MoeRecipe;
use ferrule_model::nn::{ModulePath, ParameterDType, ParameterResidency};
use ferrule_model::transformer::{
    Attention, DecoderRecipe, ExternalTensorMeta, FeedForward, NameMapper, Residual, RotaryPairing,
    RotaryRegion, RouterScoreFunction, RouterSelection,
};

#[test]
fn qwen3_recipe_builds_strict_bf16_gqa_moe_contract() {
    let value = qwen_config(false);

    let output = Qwen3MoeRecipe::new().build(&value).unwrap();
    let spec = output.spec();
    let schema = output.schema();

    assert_eq!(spec.architecture(), "Qwen3MoeForCausalLM");
    assert_eq!(spec.layers().len(), 48);
    assert_eq!(spec.hidden_size(), 2048);
    assert_eq!(spec.vocab_size(), 151_936);
    assert_eq!(spec.max_sequence_length(), Some(40_960));
    assert!(!spec.tie_word_embeddings());
    assert_eq!(schema.len(), 18_867);
    assert!(schema.parameters().iter().all(|parameter| {
        parameter.dtype().allowed() == [ParameterDType::Bf16]
            && parameter.scale().tensor().is_none()
            && !parameter.optional()
    }));

    for (expected, layer) in spec.layers().iter().enumerate() {
        assert_eq!(layer.index(), expected);
        assert_eq!(layer.attention_residual(), &Residual::Add);
        assert_eq!(layer.feed_forward_residual(), &Residual::Add);
        let Attention::Gqa(attention) = layer.attention() else {
            panic!("Qwen3 layer is not GQA")
        };
        assert_eq!(attention.num_heads(), 32);
        assert_eq!(attention.num_kv_heads(), 4);
        assert_eq!(attention.head_dim(), 128);
        assert_eq!(attention.query_norm().unwrap().hidden_size(), 128);
        assert_eq!(attention.key_norm().unwrap().hidden_size(), 128);
        assert_eq!(attention.rotary().pairing(), RotaryPairing::SplitHalf);
        assert_eq!(
            attention.rotary().region(),
            RotaryRegion::Prefix { dimensions: 128 }
        );
        let FeedForward::Moe(moe) = layer.feed_forward() else {
            panic!("Qwen3 layer is not MoE")
        };
        assert!(moe.shared_expert().is_none());
        assert_eq!(moe.router_spec().num_experts(), 128);
        assert_eq!(moe.router_spec().experts_per_token(), 8);
        assert_eq!(
            moe.router_spec().score_function(),
            RouterScoreFunction::Softmax
        );
        assert_eq!(moe.router_spec().selection(), &RouterSelection::TopK);
        assert!(moe.router_spec().normalize_selected());
        assert_eq!(moe.router_spec().route_scale(), 1.0);
    }

    let expert_parameters = schema
        .parameters()
        .iter()
        .filter(|parameter| matches!(parameter.residency(), ParameterResidency::Expert { .. }))
        .count();
    assert_eq!(expert_parameters, 48 * 128 * 3);
    assert_eq!(
        schema
            .get(&ModulePath::new("layers.47.attention.query_norm.weight").unwrap())
            .and_then(|parameter| schema.role(parameter.id())),
        Some(&TensorRole::AttentionQueryNorm)
    );
    assert!(
        schema
            .get(&ModulePath::new("layers.47.experts.127.down.weight").unwrap())
            .is_some()
    );
}

#[test]
fn qwen3_algorithmic_mapper_covers_every_real_expert_path() {
    let output = Qwen3MoeRecipe::new().build(&qwen_config(false)).unwrap();
    let mapper = output.name_mapper();
    for layer in 0..48 {
        for (external, canonical) in [
            ("input_layernorm", "input_norm"),
            ("post_attention_layernorm", "post_attention_norm"),
        ] {
            assert_mapping(
                mapper,
                &format!("model.layers.{layer}.{external}.weight"),
                &format!("layers.{layer}.{canonical}.weight"),
            );
        }
        for (external, canonical) in [
            ("q_proj", "query"),
            ("k_proj", "key"),
            ("v_proj", "value"),
            ("o_proj", "output"),
            ("q_norm", "query_norm"),
            ("k_norm", "key_norm"),
        ] {
            assert_mapping(
                mapper,
                &format!("model.layers.{layer}.self_attn.{external}.weight"),
                &format!("layers.{layer}.attention.{canonical}.weight"),
            );
        }
        assert_mapping(
            mapper,
            &format!("model.layers.{layer}.mlp.gate.weight"),
            &format!("layers.{layer}.router.weight"),
        );
        for expert in 0..128 {
            for (external, canonical) in [
                ("gate_proj", "gate"),
                ("up_proj", "up"),
                ("down_proj", "down"),
            ] {
                assert_mapping(
                    mapper,
                    &format!("model.layers.{layer}.mlp.experts.{expert}.{external}.weight"),
                    &format!("layers.{layer}.experts.{expert}.{canonical}.weight"),
                );
            }
        }
    }
    assert_mapping(
        mapper,
        "model.embed_tokens.weight",
        "token_embedding.weight",
    );
    assert_mapping(mapper, "model.norm.weight", "final_norm.weight");
    assert_mapping(mapper, "lm_head.weight", "output.weight");

    for illegal in [
        "model.layers.48.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.128.up_proj.weight",
        "model.layers.x.self_attn.k_norm.weight",
        "model.layers.0.self_attn.q_norm.bias",
        "model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.layers.0.mlp.experts.0.unknown.weight",
        "model.layers.0.mlp.gate.bias",
    ] {
        assert!(mapper.map(illegal, meta()).is_err(), "accepted {illegal}");
    }
    assert!(mapper.map("unrelated.weight", meta()).unwrap().is_none());
}

#[test]
fn tied_head_is_a_schema_alias_and_separate_hf_head_is_rejected() {
    let value = qwen_config(true);
    let output = Qwen3MoeRecipe::new().build(&value).unwrap();
    let schema = output.schema();
    let embedding = schema
        .get(&ModulePath::new("token_embedding.weight").unwrap())
        .unwrap();
    let output_head = schema
        .get(&ModulePath::new("output.weight").unwrap())
        .unwrap();
    assert_eq!(output_head.alias_of(), Some(embedding.id()));
    assert_eq!(schema.canonical_id(output_head.id()), Some(embedding.id()));

    let mapper = output.name_mapper();
    assert_mapping(
        mapper,
        "model.embed_tokens.weight",
        "token_embedding.weight",
    );
    assert!(mapper.map("lm_head.weight", meta()).is_err());
}

#[test]
fn qwen3_recipe_preserves_typed_config_error_source() {
    use std::error::Error as _;

    let error = Qwen3MoeRecipe::new()
        .build(&serde_json::json!({"model_type": "qwen3_moe"}))
        .err()
        .expect("incomplete Qwen config must fail");
    let source = error
        .source()
        .expect("recipe config errors retain a source");
    let schema = source
        .source()
        .expect("typed model config errors retain the serde source");
    assert!(schema.to_string().contains("missing field"));
}

#[test]
fn qwen3_config_rejects_dense_shared_and_non_bf16_variants() {
    let mut dense = qwen_config(false);
    dense["model_type"] = "qwen3".into();
    dense["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
    assert!(Qwen3MoeRecipe::new().build(&dense).is_err());

    let mut shared = qwen_config(false);
    shared["num_shared_experts"] = 1.into();
    assert!(Qwen3MoeRecipe::new().build(&shared).is_err());

    let mut dtype = qwen_config(false);
    dtype["torch_dtype"] = "float16".into();
    assert!(Qwen3MoeRecipe::new().build(&dtype).is_err());

    let mut sparse = qwen_config(false);
    sparse["decoder_sparse_step"] = 2.into();
    assert!(Qwen3MoeRecipe::new().build(&sparse).is_err());
}

fn assert_mapping(mapper: &dyn NameMapper, external: &str, canonical: &str) {
    let mapping = mapper
        .map(external, meta())
        .unwrap_or_else(|error| panic!("failed to map {external}: {error}"))
        .unwrap_or_else(|| panic!("did not recognize {external}"));
    assert_eq!(mapping.path.as_str(), canonical);
}

fn meta() -> ExternalTensorMeta<'static> {
    ExternalTensorMeta {
        dtype: "BF16",
        shape: &[1],
        bytes: 2,
    }
}

fn qwen_config(tie_word_embeddings: bool) -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "max_position_embeddings": 40960,
        "max_window_layers": 48,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 768,
        "norm_topk_prob": true,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "output_router_logits": false,
        "rms_norm_eps": 1e-6,
        "rope_scaling": null,
        "rope_theta": 1_000_000.0,
        "router_aux_loss_coef": 0.001,
        "sliding_window": null,
        "tie_word_embeddings": tie_word_embeddings,
        "torch_dtype": "bfloat16",
        "transformers_version": "test",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936
    })
}
