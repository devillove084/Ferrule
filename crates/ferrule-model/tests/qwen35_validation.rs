//! Hermetic negative coverage for the metadata boundary and EOS selection.
use std::collections::BTreeSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_model::models::qwen35::{
    Qwen35Config, Qwen35HfNameMapper, Qwen35Metadata, Qwen35TensorPartitionKind,
};
use ferrule_model::transformer::{ExternalTensorMeta, NameMapper};
use ferrule_model::{
    AutoConfig, EnginePlan, EnginePlanStatus, HfSafetensorsInventory, HfSafetensorsTensorInfo,
    ModelExecutionBackend, ModelFamily, TensorClass, TensorRole, TokenizerHandle,
};
use serde_json::{Value, json};

fn value() -> Value {
    serde_json::from_str(include_str!("qwen35_08b_config.json")).unwrap()
}
fn mapper() -> Qwen35HfNameMapper {
    Qwen35HfNameMapper::new(&Qwen35Config::from_value(&value()).unwrap())
}

fn inventory(visual: bool, mtp: bool) -> HfSafetensorsInventory {
    let mapper = mapper();
    let mut offset = 0;
    let tensors = mapper
        .tensors()
        .filter(|s| match s.partition {
            Qwen35TensorPartitionKind::Text => true,
            Qwen35TensorPartitionKind::Visual => visual,
            Qwen35TensorPartitionKind::Mtp => mtp,
        })
        .map(|s| {
            let tensor = HfSafetensorsTensorInfo {
                name: s.external_name.clone(),
                shard: "model.safetensors".into(),
                dtype: s.dtype.as_str().into(),
                shape: s.shape.clone(),
                data_offset: offset,
                file_offset: offset,
                byte_size: s.bytes(),
                class: TensorClass::Unknown,
                role: TensorRole::Unknown,
            };
            offset += s.bytes();
            tensor
        })
        .collect::<Vec<_>>();
    HfSafetensorsInventory {
        family: ModelFamily::Qwen35,
        total_size: Some(offset),
        shard_count: 1,
        tensor_count: tensors.len(),
        tensors,
        dtype_counts: vec![],
        class_counts: vec![],
        role_counts: vec![],
        shard_summaries: vec![],
        index_only_tensors: vec![],
        header_only_tensors: vec![],
    }
}

#[test]
fn rejects_invalid_config_fields_without_defaulting_away_semantics() {
    for (pointer, bad) in [
        ("/model_type", json!("qwen3")),
        ("/architectures", json!([])),
        (
            "/architectures",
            json!(["Qwen3_5ForConditionalGeneration", "Qwen3ForCausalLM"]),
        ),
        (
            "/architectures/0",
            json!("Qwen3_5ForConditionalGenerationExtra"),
        ),
        ("/tie_word_embeddings", json!(false)),
        ("/text_config/model_type", json!("qwen3_5_moe_text")),
        ("/text_config/dtype", json!("float16")),
        ("/text_config/hidden_size", json!(2048)),
        ("/text_config/num_hidden_layers", json!(0)),
        ("/text_config/num_key_value_heads", json!(0)),
        ("/text_config/linear_num_value_heads", json!(32)),
        ("/text_config/linear_conv_kernel_dim", json!(0)),
        ("/text_config/attention_bias", json!(true)),
        ("/text_config/attention_dropout", json!(0.1)),
        ("/text_config/attn_output_gate", json!(false)),
        ("/text_config/hidden_act", json!("gelu")),
        ("/text_config/rms_norm_eps", json!(0)),
        ("/text_config/tie_word_embeddings", json!(false)),
        ("/text_config/use_cache", json!(false)),
        ("/text_config/eos_token_id", json!(248320)),
        ("/text_config/mamba_ssm_dtype", json!("bfloat16")),
        ("/text_config/mlp_only_layers", json!([0])),
        ("/text_config/mtp_num_hidden_layers", json!(2)),
        ("/text_config/mtp_use_dedicated_embeddings", json!(true)),
        ("/text_config/full_attention_interval", json!(0)),
        ("/text_config/full_attention_interval", json!(3)),
        ("/text_config/max_position_embeddings", json!(0)),
        ("/text_config/layer_types", json!(["linear_attention"])),
        ("/text_config/layer_types/0", json!("full_attention")),
        ("/text_config/layer_types/0", json!("unknown")),
        (
            "/text_config/rope_parameters/partial_rotary_factor",
            json!(1.0),
        ),
        ("/text_config/rope_parameters/rope_type", json!("yarn")),
        (
            "/text_config/rope_parameters/mrope_section",
            json!([10, 10, 12]),
        ),
        (
            "/text_config/rope_parameters/mrope_interleaved",
            json!(false),
        ),
        ("/vision_config/model_type", json!("qwen3_5_moe")),
        ("/vision_config/depth", json!(13)),
        ("/vision_config/deepstack_visual_indexes", json!([0])),
    ] {
        let mut c = value();
        *c.pointer_mut(pointer).unwrap() = bad;
        assert!(
            Qwen35Config::from_value(&c).is_err(),
            "accepted {pointer}: {}",
            c.pointer(pointer).unwrap()
        );
    }
    for parent in [
        "",
        "/text_config",
        "/vision_config",
        "/text_config/rope_parameters",
    ] {
        let mut c = value();
        c.pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown_semantics".into(), json!(true));
        assert!(
            Qwen35Config::from_value(&c).is_err(),
            "unknown key under {parent}"
        );
    }
    let mut c = value();
    c["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("layer_types");
    c["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("full_attention_interval");
    assert_eq!(
        Qwen35Config::from_value(&c).unwrap().layer_types().len(),
        24
    );
    c["text_config"].as_object_mut().unwrap().remove("dtype");
    c["text_config"]["torch_dtype"] = json!("bfloat16");
    assert!(Qwen35Config::from_value(&c).is_ok());
}

#[test]
fn all_names_map_uniquely_and_dtype_shape_bytes_are_exact() {
    let mapper = mapper();
    let mut paths = BTreeSet::new();
    let mut f32_text = 0;
    for spec in mapper.tensors() {
        let meta = ExternalTensorMeta {
            dtype: spec.dtype.as_str(),
            shape: &spec.shape,
            bytes: spec.bytes(),
        };
        assert!(mapper.validate_tensor(&spec.external_name, meta).is_ok());
        if spec.partition == Qwen35TensorPartitionKind::Text {
            let mapped = mapper.map(&spec.external_name, meta).unwrap().unwrap();
            assert_eq!(Some(&mapped.path), spec.canonical_path.as_ref());
            assert!(paths.insert(mapped.path));
            if meta.dtype == "F32" {
                f32_text += 1;
                assert!(
                    spec.external_name.ends_with(".A_log")
                        || spec.external_name.ends_with("linear_attn.norm.weight")
                );
            }
        } else {
            assert!(mapper.map(&spec.external_name, meta).is_err());
        }
        assert!(
            mapper
                .validate_tensor(
                    &spec.external_name,
                    ExternalTensorMeta {
                        dtype: if meta.dtype == "F32" { "BF16" } else { "F32" },
                        ..meta
                    }
                )
                .is_err()
        );
        assert!(
            mapper
                .validate_tensor(
                    &spec.external_name,
                    ExternalTensorMeta {
                        bytes: meta.bytes - 1,
                        ..meta
                    }
                )
                .is_err()
        );
        assert!(
            mapper
                .validate_tensor(
                    &spec.external_name,
                    ExternalTensorMeta {
                        shape: &[1],
                        ..meta
                    }
                )
                .is_err()
        );
    }
    assert_eq!(paths.len(), 320);
    assert_eq!(f32_text, 36);
    let qgate = mapper
        .tensors()
        .find(|s| s.external_name == "model.language_model.layers.3.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(qgate.shape, [4096, 1024]);
    assert_eq!(
        qgate.canonical_path.as_ref().unwrap().as_str(),
        "layers.3.attention.query_gate.weight"
    );
    for name in [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.3.linear_attn.A_log",
        "model.language_model.layers.24.input_layernorm.weight",
        "model.language_model.layers.00.input_layernorm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.visual.unexpected.weight",
        "model.visual.blocks.12.norm1.weight",
        "mtp.layers.1.self_attn.q_proj.weight",
        "mtp.extra.weight",
        "unrelated.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv",
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
fn attachment_partition_is_all_or_none_and_unknowns_are_never_ignored() {
    let mapper = mapper();
    for visual in [false, true] {
        for mtp in [false, true] {
            let p = mapper.validate_inventory(&inventory(visual, mtp)).unwrap();
            assert_eq!(p.text().len(), 320);
            assert_eq!(p.visual().len(), if visual { 153 } else { 0 });
            assert_eq!(p.mtp().len(), if mtp { 15 } else { 0 });
        }
    }
    for prefix in ["model.language_model.", "model.visual.", "mtp."] {
        let mut i = inventory(true, true);
        let index = i
            .tensors
            .iter()
            .position(|t| t.name.starts_with(prefix))
            .unwrap();
        i.tensors.remove(index);
        i.tensor_count -= 1;
        assert!(
            mapper.validate_inventory(&i).is_err(),
            "missing tensor under {prefix}"
        );
        let mut i = inventory(true, true);
        i.tensors[index].name = format!("{prefix}unknown.weight");
        assert!(mapper.validate_inventory(&i).is_err());
    }
    let mut i = inventory(false, false);
    i.tensors.push(i.tensors[0].clone());
    i.tensor_count += 1;
    assert!(
        mapper
            .validate_inventory(&i)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    let mut i = inventory(false, false);
    i.index_only_tensors.push("missing.weight".into());
    assert!(mapper.validate_inventory(&i).is_err());
    let mut i = inventory(false, false);
    i.header_only_tensors.push("extra.weight".into());
    assert!(mapper.validate_inventory(&i).is_err());
}

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "ferrule-qwen35-metadata-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn json(&self, name: &str, value: &Value) {
        std::fs::write(self.0.join(name), serde_json::to_vec(value).unwrap()).unwrap();
    }
    fn artifact(&self) {
        self.json("config.json", &value());
        let i = inventory(false, false);
        let mut header = serde_json::Map::new();
        let mut weight_map = serde_json::Map::new();
        for t in &i.tensors {
            header.insert(t.name.clone(), json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [t.data_offset, t.data_offset + t.byte_size]}));
            weight_map.insert(t.name.clone(), json!(t.shard));
        }
        let mut bytes = serde_json::to_vec(&header).unwrap();
        while !bytes.len().is_multiple_of(8) {
            bytes.push(b' ');
        }
        let mut file = std::fs::File::create(self.0.join("model.safetensors")).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        // Sparse extent: the metadata reader must not materialize this 1.5GB payload.
        file.set_len(8 + bytes.len() as u64 + i.total_size.unwrap())
            .unwrap();
        self.json(
            "model.safetensors.index.json",
            &json!({"metadata":{"total_size":i.total_size}, "weight_map":weight_map}),
        );
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn descriptor_is_strict_header_only_and_reports_registered_backends() {
    let dir = TempDir::new();
    dir.artifact();
    let metadata = Qwen35Metadata::open_hf(&dir.0).unwrap();
    assert_eq!(metadata.partition().text().len(), 320);
    let descriptor = AutoConfig::from_pretrained(&dir.0)
        .unwrap()
        .into_descriptor();
    assert_eq!(descriptor.spec.family, ModelFamily::Qwen35);
    assert_eq!(descriptor.spec.hidden_size, Some(1024));
    assert_eq!(descriptor.spec.num_layers, Some(24));
    assert_eq!(descriptor.spec.semantics.rope_head_dim, Some(64));
    assert!(descriptor.spec.supports_current_runtime());
    for family in [ModelFamily::Qwen35, ModelFamily::Qwen35Moe] {
        let mut d = descriptor.clone();
        d.spec.family = family.clone();
        for backend in [ModelExecutionBackend::Cpu, ModelExecutionBackend::Cuda] {
            let plan = EnginePlan::from_contract_for_backend(&d.support_contract(), backend);
            let executable = family == ModelFamily::Qwen35
                && (backend == ModelExecutionBackend::Cpu || cfg!(feature = "cuda"));
            assert_eq!(plan.is_executable(), executable, "{plan:?}");
            assert_eq!(plan.backend_profile.is_some(), executable);
            assert_eq!(
                plan.status,
                if executable {
                    EnginePlanStatus::Executable
                } else {
                    EnginePlanStatus::Unsupported
                }
            );
        }
    }
    std::fs::remove_file(dir.0.join("model.safetensors.index.json")).unwrap();
    assert_eq!(
        Qwen35Metadata::open_hf(&dir.0)
            .unwrap()
            .partition()
            .text()
            .len(),
        320
    );
    let mut bad = value();
    bad["model_type"] = json!("qwen3");
    dir.json("config.json", &bad);
    assert!(
        AutoConfig::from_pretrained(&dir.0).is_err(),
        "contradictory model_type must not enter Qwen3"
    );
}

#[test]
fn invalid_index_shards_and_totals_fail_closed() {
    let dir = TempDir::new();
    dir.artifact();
    for shard in [
        "../escape.safetensors",
        "/absolute.safetensors",
        "",
        "nested/model.safetensors",
        "..\\escape.safetensors",
    ] {
        dir.json(
            "model.safetensors.index.json",
            &json!({"weight_map":{"model.language_model.norm.weight":shard}}),
        );
        assert!(
            Qwen35Metadata::open_hf(&dir.0)
                .unwrap_err()
                .to_string()
                .contains("invalid shard")
        );
    }
    dir.artifact();
    let path = dir.0.join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    index["metadata"]["total_size"] = json!(1);
    dir.json("model.safetensors.index.json", &index);
    assert!(
        Qwen35Metadata::open_hf(&dir.0)
            .unwrap_err()
            .to_string()
            .contains("total_size")
    );
    dir.artifact();
    let mut index: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    index["weight_map"]
        .as_object_mut()
        .unwrap()
        .remove("model.language_model.norm.weight");
    dir.json("model.safetensors.index.json", &index);
    assert!(Qwen35Metadata::open_hf(&dir.0).is_err());
}

#[test]
fn truncated_payload_extent_and_overlapping_offsets_are_rejected_without_payload_reads() {
    use std::io::{Read, Seek, SeekFrom};
    let dir = TempDir::new();
    dir.artifact();
    let path = dir.0.join("model.safetensors");
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(file.metadata().unwrap().len() - 1).unwrap();
    assert!(
        Qwen35Metadata::open_hf(&dir.0)
            .unwrap_err()
            .to_string()
            .contains("length differs")
    );
    dir.artifact();
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let mut len = [0u8; 8];
    file.read_exact(&mut len).unwrap();
    let header_len = u64::from_le_bytes(len) as usize;
    let mut bytes = vec![0u8; header_len];
    file.read_exact(&mut bytes).unwrap();
    let mut header: Value = serde_json::from_slice(&bytes).unwrap();
    let offsets = &mut header["model.language_model.norm.weight"]["data_offsets"];
    let size = offsets[1].as_u64().unwrap() - offsets[0].as_u64().unwrap();
    *offsets = json!([0, size]);
    let mut bytes = serde_json::to_vec(&header).unwrap();
    assert!(bytes.len() <= header_len);
    bytes.resize(header_len, b' ');
    file.seek(SeekFrom::Start(8)).unwrap();
    file.write_all(&bytes).unwrap();
    assert!(
        Qwen35Metadata::open_hf(&dir.0)
            .unwrap_err()
            .to_string()
            .contains("overlapping")
    );
}

#[test]
fn eos_generation_config_precedes_top_level_then_nested_text_fallback() {
    let dir = TempDir::new();
    tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
        .save(dir.0.join("tokenizer.json"), false)
        .unwrap();
    dir.json(
        "config.json",
        &json!({"text_config":{"eos_token_id":[3,4,3]}}),
    );
    assert_eq!(
        TokenizerHandle::load(&dir.0).unwrap().eos_token_ids(),
        [3, 4]
    );
    dir.json(
        "config.json",
        &json!({"eos_token_id":2,"text_config":{"eos_token_id":3}}),
    );
    assert_eq!(TokenizerHandle::load(&dir.0).unwrap().eos_token_ids(), [2]);
    dir.json("generation_config.json", &json!({"eos_token_id":[5,6,5]}));
    assert_eq!(
        TokenizerHandle::load(&dir.0).unwrap().eos_token_ids(),
        [5, 6]
    );
    dir.json("generation_config.json", &json!({"eos_token_id":null}));
    dir.json(
        "config.json",
        &json!({"eos_token_id":null,"text_config":{"eos_token_id":3}}),
    );
    assert_eq!(TokenizerHandle::load(&dir.0).unwrap().eos_token_ids(), [3]);
    dir.json("config.json", &json!({"text_config":{"eos_token_id":-1}}));
    assert!(TokenizerHandle::load(&dir.0).is_err());
    dir.json("generation_config.json", &json!({"eos_token_id":"invalid"}));
    dir.json("config.json", &json!({"text_config":{"eos_token_id":3}}));
    assert!(TokenizerHandle::load(&dir.0).is_err());
}
