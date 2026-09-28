//! Hermetic config/schema and physical pair binding tests; no weight payload reads.
use ferrule_model::models::qwen35::*;
use ferrule_model::nn::{ModulePath, ParameterDType, ParameterResidency, StorageEncoding};
use ferrule_model::transformer::*;
use ferrule_model::{CheckpointDType, CheckpointTensorSlice, TensorRole};
use serde_json::{Value, json};

fn value() -> Value {
    serde_json::from_str(include_str!("qwen35_35b_fp8_config.json")).unwrap()
}
fn config() -> Qwen35Config {
    Qwen35Config::from_value(&value()).unwrap()
}
fn path(s: &str) -> ModulePath {
    ModulePath::new(s).unwrap()
}

#[test]
fn exact_35b_geometry_shared_gate_and_storage_contract() {
    let c = config();
    assert_eq!(c.profile(), Qwen35Profile::Moe35BA3Bfp8);
    assert!(!c.supports_execution());
    let spec = Qwen35Recipe::spec(&c).unwrap();
    assert_eq!(spec.layers().len(), 40);
    assert_eq!(spec.hidden_size(), 2048);
    assert!(!spec.tie_word_embeddings());
    assert_eq!(c.semantics().rotary_dimensions, 64);
    for (i, layer) in spec.layers().iter().enumerate() {
        assert!(layer.input_norm().one_plus_weight());
        assert!(layer.post_attention_norm().one_plus_weight());
        let FeedForward::Moe(m) = layer.feed_forward() else {
            panic!("not MoE")
        };
        assert_eq!(m.router_spec().num_experts(), 256);
        assert_eq!(m.router_spec().experts_per_token(), 8);
        assert_eq!(
            m.router_spec().score_function(),
            RouterScoreFunction::Softmax
        );
        assert_eq!(m.router_spec().selection(), &RouterSelection::TopK);
        assert!(m.router_spec().normalize_selected());
        assert_eq!(m.router_spec().route_scale(), 1.0);
        assert!(!m.router_spec().selection_bias());
        assert_eq!(m.expert().intermediate_size(), 512);
        assert_eq!(m.shared_expert().unwrap().intermediate_size(), 512);
        assert_eq!(m.shared_expert_gate().unwrap().weight_shape(), [1, 2048]);
        assert!(!m.shared_expert_gate().unwrap().has_bias());
        match layer.attention() {
            Attention::Gqa(a) => {
                assert_eq!(i % 4, 3);
                assert_eq!(a.query().weight_shape(), [8192, 2048]);
                assert!(a.gated_query());
                assert_eq!(a.num_heads(), 16);
                assert_eq!(a.num_kv_heads(), 2);
                assert_eq!(a.head_dim(), 256);
            }
            Attention::GatedDeltaNet(a) => {
                assert_ne!(i % 4, 3);
                assert_eq!(a.num_key_heads(), 16);
                assert_eq!(a.num_value_heads(), 32);
                assert_eq!(a.key_head_dim(), 128);
                assert_eq!(a.value_head_dim(), 128);
            }
            _ => panic!("unexpected attention"),
        }
    }
    let schema = Qwen35Recipe::schema(&c).unwrap();
    assert_eq!(schema.len(), 31333);
    assert_eq!(schema.storage_tensor_count(), 62303);
    for parameter in schema.parameters() {
        assert!(parameter.alias_of().is_none());
        assert_ne!(schema.role(parameter.id()), Some(&TensorRole::Unknown));
        if parameter.dtype().allowed() == [ParameterDType::F8E4M3] {
            assert_eq!(parameter.weight().encoding(), StorageEncoding::Dense);
            assert!(parameter.scale().is_required());
            let scale = parameter.scale().tensor().unwrap();
            assert_eq!(scale.dtype().allowed(), [ParameterDType::Bf16]);
            assert_eq!(
                scale.shape(),
                parameter
                    .shape()
                    .iter()
                    .map(|n| n.div_ceil(128))
                    .collect::<Vec<_>>()
            );
        } else {
            assert!(parameter.scale().tensor().is_none());
        }
    }
    let gate = schema
        .get(&path("layers.0.feed_forward.shared_expert_gate.weight"))
        .unwrap();
    assert_eq!(
        schema.role(gate.id()),
        Some(&TensorRole::SharedExpertOutputGate)
    );
    assert_eq!(gate.residency(), &ParameterResidency::layer(0));
    for expert in 0..256 {
        for (projection, role) in [
            ("gate", TensorRole::RoutedExpertGate),
            ("up", TensorRole::RoutedExpertUp),
            ("down", TensorRole::RoutedExpertDown),
        ] {
            let p = schema
                .get(&path(&format!(
                    "layers.39.feed_forward.experts.{expert}.{projection}.weight"
                )))
                .unwrap();
            assert_eq!(p.residency(), &ParameterResidency::expert(39, expert));
            assert_eq!(schema.role(p.id()), Some(&role));
        }
    }
}

#[test]
fn reject_other_architectures_geometries_and_quantization_semantics() {
    for (pointer, bad) in [
        ("/model_type", json!("qwen3_8_moe")),
        ("/model_type", json!("qwen4")),
        (
            "/architectures/0",
            json!("Qwen3_8MoeForConditionalGeneration"),
        ),
        ("/text_config/model_type", json!("qwen4_moe_text")),
        ("/text_config/hidden_size", json!(4096)),
        ("/text_config/num_hidden_layers", json!(41)),
        ("/text_config/num_experts", json!(128)),
        ("/text_config/num_experts_per_tok", json!(4)),
        ("/text_config/moe_intermediate_size", json!(1024)),
        ("/text_config/shared_expert_intermediate_size", json!(1024)),
        ("/text_config/linear_num_value_heads", json!(16)),
        ("/text_config/rms_norm_eps", json!(0.00001)),
        (
            "/text_config/rope_parameters/partial_rotary_factor",
            json!(0.5),
        ),
        ("/text_config/layer_types/0", json!("full_attention")),
        ("/tie_word_embeddings", json!(true)),
        ("/vision_config/depth", json!(12)),
        ("/quantization_config/weight_block_size", json!([64, 128])),
        ("/quantization_config/activation_scheme", json!("static")),
        ("/quantization_config/weight_per_tensor", json!(true)),
        ("/quantization_config/act_per_tensor", json!(true)),
        (
            "/quantization_config/modules_to_not_convert/0",
            json!("unknown.module"),
        ),
    ] {
        let mut v = value();
        *v.pointer_mut(pointer).unwrap() = bad;
        assert!(Qwen35Config::from_value(&v).is_err(), "accepted {pointer}");
    }
    let mut v = value();
    v.as_object_mut().unwrap().remove("quantization_config");
    assert!(
        Qwen35Config::from_value(&v)
            .unwrap_err()
            .to_string()
            .contains("BF16 packed experts are unsupported")
    );
    for parent in ["", "/text_config", "/quantization_config", "/vision_config"] {
        let mut v = value();
        v.pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), json!(true));
        assert!(Qwen35Config::from_value(&v).is_err());
    }
    let mut v = value();
    v["quantization_config"]["modules_to_not_convert"]
        .as_array_mut()
        .unwrap()
        .push(json!("lm_head"));
    assert!(Qwen35Config::from_value(&v).is_err());
}

#[test]
fn exact_names_and_parts_reject_packed_experts_and_unknown_scales() {
    let mapper = Qwen35HfNameMapper::new(&config());
    assert!(!mapper.tied_output());
    assert_eq!(mapper.output_alias(), None);
    assert_eq!(mapper.tensors().count(), 64196);
    let mut mappings = std::collections::BTreeSet::new();
    for t in mapper.tensors() {
        let meta = ExternalTensorMeta {
            dtype: t.dtype.as_str(),
            shape: &t.shape,
            bytes: t.bytes(),
        };
        mapper.validate_tensor(&t.external_name, meta).unwrap();
        if t.partition == Qwen35TensorPartitionKind::Text {
            let mapped = mapper.map(&t.external_name, meta).unwrap().unwrap();
            assert_eq!(mapped.part, t.part);
            assert!(mappings.insert((mapped.path, mapped.part)));
        } else {
            assert!(mapper.map(&t.external_name, meta).is_err());
        }
    }
    for name in [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.256.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.00.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight_scale_inv",
        "model.language_model.layers.0.mlp.unknown.weight",
        "mtp.unknown.weight",
        "model.visual.unknown.weight",
    ] {
        assert!(
            mapper
                .map(
                    name,
                    ExternalTensorMeta {
                        dtype: "BF16",
                        shape: &[1],
                        bytes: 2
                    }
                )
                .is_err(),
            "accepted {name}"
        );
    }
}

#[test]
fn numeric_fp8_binder_requires_exact_weight_scale_pair_and_roles() {
    let c = config();
    let mapper = Qwen35HfNameMapper::new(&c);
    let schema = Qwen35Recipe::schema(&c).unwrap();
    // Use actual recipe parameters and actual geometry, backed by one sparse file.
    // Binder only snapshots metadata; no FP8 or BF16 payload is read.
    let selected = [
        "layers.0.feed_forward.experts.0.gate.weight",
        "layers.0.feed_forward.experts.0.up.weight",
        "layers.0.feed_forward.experts.0.down.weight",
        "layers.0.feed_forward.shared_expert_gate.weight",
    ];
    let mut builder = StateDictSchema::builder();
    for name in selected {
        let p = schema.get(&path(name)).unwrap();
        builder
            .register_with_role(p.clone(), schema.role(p.id()).unwrap().clone())
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let file = std::env::temp_dir().join(format!("ferrule-qwen35-pairs-{}", std::process::id()));
    let mut offset = 0;
    let slices = mapper
        .tensors()
        .filter(|t| {
            t.canonical_path
                .as_ref()
                .is_some_and(|p| selected.contains(&p.as_str()))
        })
        .map(|t| {
            let s = CheckpointTensorSlice {
                name: t.external_name.clone(),
                path: file.clone(),
                offset,
                bytes: t.bytes(),
                dtype: CheckpointDType::from_safetensors_dtype(t.dtype.as_str()),
                shape: t.shape.clone(),
                role: TensorRole::Unknown,
            };
            offset += s.bytes;
            s
        })
        .collect::<Vec<_>>();
    std::fs::File::create_new(&file)
        .unwrap()
        .set_len(offset)
        .unwrap();
    let binder = StateDictBinder::new(&schema, &mapper);
    let state = binder.bind_slices(slices.clone()).unwrap();
    assert_eq!(state.len(), 4);
    for p in state.parameters() {
        if p.scale().is_some() {
            p.numeric_fp8_source(c.numeric_fp8_encoding().unwrap())
                .unwrap();
            assert_eq!(p.residency(), &ParameterResidency::expert(0, 0));
        } else {
            assert_eq!(p.role(), &TensorRole::SharedExpertOutputGate);
        }
    }
    for i in 0..slices.len() {
        let mut missing = slices.clone();
        missing.remove(i);
        assert!(
            binder.bind_slices(missing).is_err(),
            "accepted missing {}",
            slices[i].name
        );
    }
    let scale = slices
        .iter()
        .position(|s| s.name.ends_with("weight_scale_inv"))
        .unwrap();
    let mut wrong_dtype = slices.clone();
    wrong_dtype[scale].dtype = CheckpointDType::F32;
    assert!(binder.bind_slices(wrong_dtype).is_err());
    let mut wrong_shape = slices.clone();
    wrong_shape[scale].shape = vec![1, 1];
    assert!(binder.bind_slices(wrong_shape).is_err());
    let mut duplicate = slices.clone();
    duplicate.push(slices[scale].clone());
    assert!(binder.bind_slices(duplicate).is_err());
    let mut unknown = slices;
    unknown[scale].name += ".unknown";
    assert!(binder.bind_slices(unknown).is_err());
    std::fs::remove_file(file).unwrap();
}
