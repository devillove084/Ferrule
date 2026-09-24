//! Metadata-only tests: no forward, payload reads, CUDA or runtime registration.
use std::path::Path;

use ferrule_model::models::qwen35::{
    Qwen35Config, Qwen35HfNameMapper, Qwen35LayerType, Qwen35Metadata, Qwen35OutputGate,
    Qwen35Profile, Qwen35TensorPartitionKind, Qwen35TensorSpec,
};
use ferrule_model::nn::ParameterDType;
use ferrule_model::transformer::{ExternalTensorMeta, NameMapper};
use ferrule_model::{EnginePlanStatus, ModelFamily};
use serde_json::Value;

fn config_value() -> Value {
    serde_json::from_str(include_str!("qwen35_08b_config.json")).unwrap()
}

fn config() -> Qwen35Config {
    Qwen35Config::from_value(&config_value()).unwrap()
}

fn mapper() -> Qwen35HfNameMapper {
    Qwen35HfNameMapper::new(&config())
}

fn meta(spec: &Qwen35TensorSpec) -> ExternalTensorMeta<'_> {
    ExternalTensorMeta {
        dtype: spec.dtype.as_str(),
        shape: &spec.shape,
        bytes: spec.bytes(),
    }
}

#[test]
fn exact_family_identity_does_not_inherit_qwen3_runtime_support() {
    for (name, family) in [
        ("qwen3", ModelFamily::Qwen3),
        ("Qwen3ForCausalLM", ModelFamily::Qwen3),
        ("qwen3_moe", ModelFamily::QwenMoe),
        ("Qwen3MoeForCausalLM", ModelFamily::QwenMoe),
        ("qwen3_5", ModelFamily::Qwen35),
        ("qwen3_5_text", ModelFamily::Qwen35),
        ("Qwen3_5ForConditionalGeneration", ModelFamily::Qwen35),
        ("qwen3_5_moe", ModelFamily::Qwen35Moe),
        ("qwen3_5_moe_text", ModelFamily::Qwen35Moe),
        ("Qwen3_5MoeForConditionalGeneration", ModelFamily::Qwen35Moe),
    ] {
        assert_eq!(ModelFamily::from_architecture(name), family, "{name}");
    }
    for name in [
        "qwen3_next",
        "qwen35",
        "Qwen3ForCausalLMExtra",
        "other_qwen3",
        "qwen3_5_moe_extra",
        "qwen3_5_typo",
    ] {
        assert_eq!(
            ModelFamily::from_architecture(name),
            ModelFamily::Unknown(name.into())
        );
    }
    assert!(ModelFamily::Qwen35.is_supported_runtime_family());
    assert!(!ModelFamily::Qwen35Moe.is_supported_runtime_family());
}

#[test]
fn nested_config_resolves_hybrid_sequence_and_explicit_semantics() {
    let config = config();
    assert_eq!(config.profile(), Qwen35Profile::Dense08Bbf16);
    assert_eq!(config.architecture(), "Qwen3_5ForConditionalGeneration");
    assert_eq!(config.text().model_type, "qwen3_5_text");
    assert_eq!(config.layer_types().len(), 24);
    for &chunk in config.layer_types().as_chunks::<4>().0 {
        assert_eq!(
            chunk,
            [
                Qwen35LayerType::LinearAttention,
                Qwen35LayerType::LinearAttention,
                Qwen35LayerType::LinearAttention,
                Qwen35LayerType::FullAttention,
            ]
        );
    }
    let semantics = config.semantics();
    assert_eq!(semantics.output_gate, Qwen35OutputGate::PerHeadSigmoid);
    assert_eq!(semantics.norm_weight_offset, 1.0);
    assert_eq!(semantics.linear_gated_norm_weight_offset, 0.0);
    assert_eq!(semantics.rotary_dimensions, 64);
    assert!(semantics.rotary_prefix && semantics.rotary_split_half);
    assert!(config.text().tie_word_embeddings);

    for remove_layer_types in [false, true] {
        let mut value = config_value();
        if remove_layer_types {
            value["text_config"]
                .as_object_mut()
                .unwrap()
                .remove("layer_types");
        } else {
            value["text_config"]["layer_types"] = Value::Null;
        }
        assert_eq!(
            Qwen35Config::from_value(&value).unwrap().layer_types(),
            config.layer_types()
        );
    }
}

#[test]
fn rejects_moe_fp8_and_non_exact_profiles() {
    for (label, mut value) in [
        ("moe", config_value()),
        ("fp8", config_value()),
        ("architecture", config_value()),
        ("flat", config_value()),
    ] {
        match label {
            "moe" => value["model_type"] = "qwen3_5_moe".into(),
            "fp8" => value["quantization_config"] = serde_json::json!({"quant_method":"fp8"}),
            "architecture" => {
                value["architectures"][0] = "Qwen3_5MoeForConditionalGeneration".into()
            }
            "flat" => {
                let text = value["text_config"].take();
                value
                    .as_object_mut()
                    .unwrap()
                    .extend(text.as_object().unwrap().clone());
            }
            _ => unreachable!(),
        }
        assert!(
            Qwen35Config::from_value(&value).is_err(),
            "accepted {label}"
        );
    }
}

#[test]
fn maps_exact_text_schema_and_validates_attachments_without_binding_them() {
    let mapper = mapper();
    assert_eq!(mapper.tensors().count(), 488);
    assert_eq!(
        mapper.output_alias(),
        ("output.weight", "token_embedding.weight")
    );

    let qkv = mapper
        .tensors()
        .find(|spec| {
            spec.external_name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        })
        .unwrap();
    assert_eq!(qkv.partition, Qwen35TensorPartitionKind::Text);
    assert_eq!(qkv.dtype, ParameterDType::Bf16);
    assert_eq!(qkv.shape, [6144, 1024]);
    assert!(qkv.canonical_path.is_some());

    let alog = mapper
        .tensors()
        .find(|spec| spec.external_name == "model.language_model.layers.0.linear_attn.A_log")
        .unwrap();
    assert_eq!(alog.dtype, ParameterDType::F32);
    assert!(
        mapper
            .validate_tensor(&qkv.external_name, meta(qkv))
            .is_ok()
    );
    assert!(
        mapper
            .validate_tensor(&alog.external_name, meta(alog))
            .is_ok()
    );

    let visual = mapper
        .tensors()
        .find(|spec| spec.external_name == "model.visual.pos_embed.weight")
        .unwrap();
    assert_eq!(visual.partition, Qwen35TensorPartitionKind::Visual);
    assert!(visual.canonical_path.is_none());
    assert!(mapper.map(&visual.external_name, meta(visual)).is_err());

    assert!(
        mapper
            .map(
                "model.language_model.layers.0.linear_attn.unknown.weight",
                ExternalTensorMeta {
                    dtype: "BF16",
                    shape: &[1],
                    bytes: 2
                },
            )
            .is_err()
    );
    assert!(
        mapper
            .map(
                "lm_head.weight",
                ExternalTensorMeta {
                    dtype: "BF16",
                    shape: &[248320, 1024],
                    bytes: 248320 * 1024 * 2
                },
            )
            .is_err()
    );
}

#[test]
#[ignore = "requires FERRULE_QWEN35_08B_DIR; reads only config/index/headers"]
fn real_08b_headers_validate_text_and_known_attachment_partitions() {
    let directory = std::env::var("FERRULE_QWEN35_08B_DIR").expect("set FERRULE_QWEN35_08B_DIR");
    let path = Path::new(&directory);
    let metadata = Qwen35Metadata::open_hf(path).unwrap();
    assert_eq!(metadata.inventory().tensor_count, 488);
    assert_eq!(metadata.partition().text().len(), 320);
    assert_eq!(metadata.partition().visual().len(), 153);
    assert_eq!(metadata.partition().mtp().len(), 15);
    assert_eq!(metadata.config().text().eos_token_id, 248044);
    assert_eq!(metadata.descriptor().spec.family, ModelFamily::Qwen35);
    assert_eq!(
        metadata.descriptor().engine_plan().status,
        EnginePlanStatus::Executable
    );
    assert!(metadata.descriptor().engine_plan().is_executable());
    assert_eq!(
        metadata
            .inventory()
            .dtype_counts
            .iter()
            .find(|d| d.dtype == "F32")
            .unwrap()
            .tensors,
        36
    );
    assert_eq!(
        metadata
            .inventory()
            .tensors
            .iter()
            .map(|t| t.byte_size)
            .sum::<u64>(),
        1_746_882_752
    );
    let descriptor = ferrule_model::AutoConfig::from_pretrained(path).unwrap();
    assert_eq!(descriptor.descriptor().spec.family, ModelFamily::Qwen35);
    assert_eq!(descriptor.descriptor().spec.hidden_size, Some(1024));
}

#[test]
#[ignore = "requires FERRULE_QWEN35_35B_FP8_DIR; schema inspection only, no execution"]
fn real_35b_fp8_is_recognized_but_rejected_as_an_unimplemented_profile() {
    let directory =
        std::env::var("FERRULE_QWEN35_35B_FP8_DIR").expect("set FERRULE_QWEN35_35B_FP8_DIR");
    let path = Path::new(&directory);
    let value: Value =
        serde_json::from_str(&std::fs::read_to_string(path.join("config.json")).unwrap()).unwrap();
    assert_eq!(
        ModelFamily::from_architecture(value["model_type"].as_str().unwrap()),
        ModelFamily::Qwen35Moe
    );
    assert!(
        Qwen35Config::from_value(&value)
            .unwrap_err()
            .to_string()
            .contains("unsupported profile")
    );
    assert!(Qwen35Metadata::open_hf(path).is_err());
    assert!(ferrule_model::AutoConfig::from_pretrained(path).is_err());
    let inventory =
        ferrule_model::HfSafetensorsInventory::open(path, ModelFamily::Qwen35Moe).unwrap();
    assert_eq!(inventory.shard_count, 14);
    assert_eq!(inventory.tensor_count, 64196);
    let tensor = inventory
        .tensors
        .iter()
        .find(|t| t.name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        .unwrap();
    assert_eq!(tensor.dtype, "F8_E4M3");
    assert_eq!(tensor.shape, [8192, 2048]);
    let scale = inventory
        .tensors
        .iter()
        .find(|t| {
            t.name == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv"
        })
        .unwrap();
    assert_eq!(scale.dtype, "BF16");
    assert_eq!(scale.shape, [64, 16]);
}
