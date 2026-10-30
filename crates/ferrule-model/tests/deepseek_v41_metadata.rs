#![recursion_limit = "512"]

//! DeepSeek-V4.1 metadata-only tests.
//!
//! These tests never construct CUDA/runtime objects and never read tensor payloads.

use std::collections::BTreeMap;
use std::path::Path;

use ferrule_model::models::deepseek_v41::{
    DeepSeekV41Config, DeepSeekV41DType, DeepSeekV41Metadata,
};
use ferrule_model::{EnginePlanStatus, HfSafetensorsIndex, ModelFamily};
use serde_json::{Value, json};

pub fn config_value() -> Value {
    let mut compress_ratios = vec![0usize; 43];
    compress_ratios[2..20].fill(2);
    compress_ratios[20..40].fill(1);
    json!({
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "transformers_version": "5.6.0",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "image_token_id": 129264,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4"
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": 129280,
            "hidden_size": 5120,
            "moe_intermediate_size": 2304,
            "num_hidden_layers": 40,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-20,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "initializer_range": 0.02,
            "use_cache": true,
            "tie_word_embeddings": false,
            "max_position_embeddings": 1048576,
            "rope_theta": 10000,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536
            },
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.5,
            "sliding_window": 128,
            "compress_ratios": compress_ratios,
            "compress_rope_theta": 160000,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 512,
            "candidate_source_layer_id": 20,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384006168, 384016682],
            "engram_max_ngram_size": 4,
            "engram_vocab_size": 16000000,
            "engram_n_heads": 8,
            "engram_head_dim": 256,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 99092,
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [37, 38, 39],
            "dspark_markov_rank": 256,
            "dspark_n_routed_experts": 128,
            "dspark_num_experts_per_tok": 3
        },
        "vision_config": {
            "model_type": "deepseek_v41_vision",
            "num_hidden_layers": 32,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 2816,
            "patch_size": 14,
            "rope_theta": 10000,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": null
        }
    })
}

fn add(map: &mut BTreeMap<String, String>, name: impl Into<String>) {
    map.insert(name.into(), "model-00001-of-00001.safetensors".into());
}

fn add_pair(map: &mut BTreeMap<String, String>, base: impl Into<String>) {
    let base = base.into();
    add(map, format!("{base}.weight"));
    add(map, format!("{base}.scale"));
}

pub fn fixture_index() -> HfSafetensorsIndex {
    let mut map = BTreeMap::new();
    for name in [
        "embed.weight",
        "head.weight",
        "norm.weight",
        "aligner.w1.weight",
        "aligner.w1.bias",
        "aligner.w2.weight",
        "aligner.w2.bias",
        "image_start",
        "image_end",
        "image_newline",
    ] {
        add(&mut map, name);
    }
    for layer in 0..40 {
        let prefix = format!("layers.{layer}");
        for suffix in [
            "attn.attn_sink",
            "attn.kv_norm.weight",
            "attn.q_norm.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "ffn.gate.weight",
            "ffn.gate.bias",
            "ffn.gate.bias_vl",
        ] {
            add(&mut map, format!("{prefix}.{suffix}"));
        }
        for matrix in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
            add_pair(&mut map, format!("{prefix}.attn.{matrix}"));
        }
        for matrix in ["w1", "w2", "w3"] {
            add_pair(&mut map, format!("{prefix}.ffn.shared_experts.{matrix}"));
        }
        for expert in 0..384 {
            for matrix in ["w1", "w2", "w3"] {
                add_pair(&mut map, format!("{prefix}.ffn.experts.{expert}.{matrix}"));
            }
        }
        if [2, 8, 14, 20].contains(&layer) {
            for suffix in [
                "attn.indexer.k_norm.weight",
                "attn.indexer.weights_proj.weight",
                "attn.indexer.wk.weight",
            ] {
                add(&mut map, format!("{prefix}.{suffix}"));
            }
            add_pair(&mut map, format!("{prefix}.attn.indexer.wq_b"));
        } else if [24, 28, 32, 36].contains(&layer) {
            add(
                &mut map,
                format!("{prefix}.attn.indexer.weights_proj.weight"),
            );
            add_pair(&mut map, format!("{prefix}.attn.indexer.wq_b"));
        }
        match layer {
            2 | 8 | 14 => {
                for suffix in [
                    "attn.compressor.norm.weight",
                    "attn.compressor.wgate.weight",
                    "attn.compressor.wkv.weight",
                ] {
                    add(&mut map, format!("{prefix}.{suffix}"));
                }
            }
            20 => {
                for suffix in ["attn.compressor.norm.weight", "attn.compressor.wkv.weight"] {
                    add(&mut map, format!("{prefix}.{suffix}"));
                }
            }
            _ => {}
        }
        if [1, 14].contains(&layer) {
            for suffix in [
                "engram.embed.weight",
                "engram.embed.scale",
                "engram.k_weight",
                "engram.q_weight",
                "engram.wkv.weight",
                "engram.wkv.scale",
            ] {
                add(&mut map, format!("{prefix}.{suffix}"));
            }
        }
    }
    for stage in 0..3 {
        let prefix = format!("mtp.{stage}");
        for suffix in [
            "attn.attn_sink",
            "attn.kv_norm.weight",
            "attn.q_norm.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "ffn.gate.weight",
            "ffn.gate.bias",
            "ffn.gate.bias_vl",
        ] {
            add(&mut map, format!("{prefix}.{suffix}"));
        }
        for matrix in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
            add_pair(&mut map, format!("{prefix}.attn.{matrix}"));
        }
        for matrix in ["w1", "w2", "w3"] {
            add_pair(&mut map, format!("{prefix}.ffn.shared_experts.{matrix}"));
        }
        for expert in 0..128 {
            for matrix in ["w1", "w2", "w3"] {
                add_pair(&mut map, format!("{prefix}.ffn.experts.{expert}.{matrix}"));
            }
        }
        if stage == 0 {
            add_pair(&mut map, format!("{prefix}.main_proj"));
            add(&mut map, format!("{prefix}.main_norm.weight"));
        }
        if stage == 2 {
            add(&mut map, format!("{prefix}.confidence_head.proj.weight"));
            add(&mut map, format!("{prefix}.markov_head.embed.weight"));
            add(&mut map, format!("{prefix}.markov_head.head.weight"));
            add(&mut map, format!("{prefix}.norm.weight"));
        }
    }
    add(&mut map, "vision.norm.weight");
    add(&mut map, "vision.patch_embed.proj.weight");
    add(&mut map, "vision.patch_embed.proj.bias");
    for block in 0..32 {
        for suffix in [
            "attn.wo.weight",
            "attn.wo.bias",
            "attn.wqkv.weight",
            "attn.wqkv.bias",
            "mlp.w1.weight",
            "mlp.w2.weight",
            "norm1.weight",
            "norm2.weight",
        ] {
            add(&mut map, format!("vision.blocks.{block}.{suffix}"));
        }
    }
    HfSafetensorsIndex {
        total_size: Some(1),
        weight_map: map,
    }
}

#[test]
fn parses_independent_v41_contract_and_keeps_execution_metadata_only() {
    let config = DeepSeekV41Config::from_value(&config_value()).unwrap();
    assert_eq!(config.model_type, "deepseek_v41");
    assert_eq!(config.architectures, ["DeepseekV41ForCausalLM"]);
    assert_eq!(config.text.num_hidden_layers, 40);
    assert_eq!(config.text.n_routed_experts, 384);
    assert_eq!(config.text.n_shared_experts, 1);
    assert_eq!(config.text.num_experts_per_tok, 6);
    assert_eq!(config.text.max_position_embeddings, 1_048_576);
    assert_eq!(config.quantization.weight_block_size, [32, 32]);
    assert_eq!(config.quantization.expert_dtype, "fp4");
    assert_eq!(
        DeepSeekV41DType::parse("F8_E4M3").unwrap().as_str(),
        "F8_E4M3"
    );
    assert_eq!(
        DeepSeekV41DType::parse("F8_E8M0").unwrap().as_str(),
        "F8_E8M0"
    );

    let metadata = DeepSeekV41Metadata::from_value(
        Path::new("metadata-only-fixture"),
        &config_value(),
        fixture_index(),
    )
    .unwrap();
    assert!(
        metadata.inventory().is_none(),
        "fixture has no shard payload/header"
    );
    assert!(!metadata.summary().headers_validated);
    assert!(metadata.summary().scale_pair_count > 40 * 384 * 3);
    assert_eq!(metadata.descriptor().spec.family, ModelFamily::DeepSeekV41);
    assert!(!ModelFamily::DeepSeekV41.is_supported_runtime_family());
    assert_eq!(
        metadata.descriptor().engine_plan().status,
        EnginePlanStatus::MetadataOnly
    );
    assert!(!metadata.descriptor().engine_plan().is_executable());
    assert!(
        DeepSeekV41Metadata::from_config_and_index_real(
            Path::new("metadata-only-fixture"),
            config.clone(),
            fixture_index(),
        )
        .is_err()
    );
    assert_eq!(
        ModelFamily::from_architecture("DeepseekV41ForCausalLM"),
        ModelFamily::DeepSeekV41
    );
    assert_eq!(
        ModelFamily::from_architecture("deepseek_v41"),
        ModelFamily::DeepSeekV41
    );
    assert_eq!(
        ModelFamily::from_architecture("deepseek_v4"),
        ModelFamily::DeepSeekV4
    );
}

#[test]
fn rejects_unknown_and_unsupported_v41_config_fields() {
    type ConfigMutation = fn(&mut Value);
    let cases: [(&str, ConfigMutation); 13] = [
        ("model_type", |value: &mut Value| {
            value["model_type"] = json!("deepseek_v4")
        }),
        ("architecture", |value: &mut Value| {
            value["architectures"] = json!(["DeepseekV4ForCausalLM"])
        }),
        ("quant_method", |value: &mut Value| {
            value["quantization_config"]["quant_method"] = json!("int8")
        }),
        ("activation", |value: &mut Value| {
            value["quantization_config"]["activation_scheme"] = json!("static")
        }),
        ("block", |value: &mut Value| {
            value["quantization_config"]["weight_block_size"] = json!([16, 16])
        }),
        ("expert_dtype", |value: &mut Value| {
            value["quantization_config"]["expert_dtype"] = json!("fp8")
        }),
        ("experts", |value: &mut Value| {
            value["text_config"]["n_routed_experts"] = json!(256)
        }),
        ("top_k", |value: &mut Value| {
            value["text_config"]["num_experts_per_tok"] = json!(8)
        }),
        ("unknown", |value: &mut Value| {
            value["unknown_semantics"] = json!(true)
        }),
        ("optional_version_type", |value: &mut Value| {
            value["transformers_version"] = json!(5.6)
        }),
        ("optional_image_token_type", |value: &mut Value| {
            value["image_token_id"] = json!("129264")
        }),
        ("attention_bias", |value: &mut Value| {
            value["text_config"]["attention_bias"] = json!(true)
        }),
        ("candidate_type", |value: &mut Value| {
            value["text_config"]["candidate_block_size"] = json!("8")
        }),
    ];
    for (name, mutate) in cases {
        let mut value = config_value();
        mutate(&mut value);
        assert!(
            DeepSeekV41Config::from_value(&value).is_err(),
            "accepted {name}"
        );
    }
}

#[test]
#[ignore = "requires FERRULE_DEEPSEEK_V41_DIR; metadata/index and optional headers only, never payloads"]
fn real_nas_v41_metadata_only_contract() {
    let directory =
        std::env::var("FERRULE_DEEPSEEK_V41_DIR").expect("set FERRULE_DEEPSEEK_V41_DIR");
    let metadata = DeepSeekV41Metadata::open_hf_real(Path::new(&directory)).unwrap();
    assert_eq!(metadata.config().model_type, "deepseek_v41");
    assert_eq!(metadata.config().architectures, ["DeepseekV41ForCausalLM"]);
    assert_eq!(metadata.config().text.num_hidden_layers, 40);
    assert_eq!(metadata.config().text.n_routed_experts, 384);
    assert_eq!(metadata.config().text.n_shared_experts, 1);
    assert_eq!(metadata.config().text.num_experts_per_tok, 6);
    assert_eq!(metadata.index().tensor_count(), 96_085);
    assert_eq!(metadata.index().shard_count(), 48);
    assert_eq!(metadata.index().total_size, Some(510_286_023_000));
    assert_eq!(metadata.descriptor().spec.family, ModelFamily::DeepSeekV41);
    assert_eq!(
        metadata.descriptor().engine_plan().status,
        EnginePlanStatus::MetadataOnly
    );
    if let Some(inventory) = metadata.inventory() {
        assert_eq!(inventory.tensor_count, 96_085);
    }
    assert!(
        !metadata.summary().headers_validated,
        "NAS metadata-only artifact has no complete strict header geometry validation"
    );
}
