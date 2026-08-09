use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ferrule_model::models::qwen3::Qwen3MoeCheckpoint;
use ferrule_model::nn::{ModulePath, ParameterDType};
use ferrule_model::{ModelDescriptor, ModelFamily, TensorRole};

#[test]
fn real_qwen3_30b_metadata_binds_to_the_generic_state_dict() {
    let model_dir = repository_root().join("models/Qwen3-30B-A3B");
    assert!(model_dir.join("config.json").is_file());
    assert!(model_dir.join("tokenizer.json").is_file());
    assert!(model_dir.join("model.safetensors.index.json").is_file());

    let checkpoint = Qwen3MoeCheckpoint::load_hf(&model_dir).unwrap();

    assert_eq!(checkpoint.resources().state_dict().len(), 18_867);
    assert_eq!(checkpoint.resources().experts().count(), 48 * 128);

    assert!(
        checkpoint
            .resources()
            .state_dict()
            .validate_source_identities()
    );

    let query_norm = checkpoint
        .resources()
        .state_dict()
        .get(&ModulePath::new("layers.47.attention.query_norm.weight").unwrap())
        .unwrap();
    let key_norm = checkpoint
        .resources()
        .state_dict()
        .get(&ModulePath::new("layers.47.attention.key_norm.weight").unwrap())
        .unwrap();
    assert_eq!(query_norm.role(), &TensorRole::AttentionQueryNorm);
    assert_eq!(key_norm.role(), &TensorRole::AttentionKeyNorm);
    assert_eq!(query_norm.weight().logical_shape(), [128]);
    assert_eq!(key_norm.weight().logical_shape(), [128]);

    let descriptor = ModelDescriptor::load(&model_dir).unwrap();
    assert_eq!(descriptor.spec.family, ModelFamily::QwenMoe);
    assert!(descriptor.spec.supports_current_runtime());
}

#[test]
fn synthetic_multishard_artifact_binds_without_reading_payloads() {
    let fixture = SyntheticFixture::create(false);
    let checkpoint = Qwen3MoeCheckpoint::load_hf(&fixture.dir).unwrap();

    assert_eq!(checkpoint.resources().state_dict().len(), 39);
    assert_eq!(checkpoint.resources().experts().count(), 6);

    assert!(
        checkpoint
            .resources()
            .state_dict()
            .validate_source_identities()
    );
    for parameter in checkpoint.resources().state_dict().parameters() {
        assert_eq!(parameter.spec().dtype().allowed(), [ParameterDType::Bf16]);
        assert_eq!(parameter.weight().slice().dtype.as_str(), "BF16");
    }
    let expert = checkpoint
        .resources()
        .experts()
        .require_shape(1, 2, TensorRole::RoutedExpertDown, &[4, 3])
        .unwrap();
    assert_eq!(
        expert.weight().slice().name,
        "model.layers.1.mlp.experts.2.down_proj.weight"
    );
}

#[test]
fn tied_artifact_uses_embedding_storage_for_the_logical_head() {
    let fixture = SyntheticFixture::create(true);
    let checkpoint = Qwen3MoeCheckpoint::load_hf(&fixture.dir).unwrap();

    assert!(checkpoint.resources().spec().tie_word_embeddings());

    let embedding = checkpoint
        .resources()
        .require_static_shape(TensorRole::TokenEmbedding, &[8, 4])
        .unwrap();
    let head = checkpoint
        .resources()
        .require_static_shape(TensorRole::OutputHead, &[8, 4])
        .unwrap();
    assert!(head.is_alias());
    assert!(head.shares_storage_with(embedding));
    assert_eq!(head.weight().slice().name, "model.embed_tokens.weight");
}

struct SyntheticFixture {
    dir: PathBuf,
}

impl SyntheticFixture {
    fn create(tied: bool) -> Self {
        let dir = unique_temp_dir("ferrule-qwen3-artifact");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&synthetic_config(tied)).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id":[2,3,2]}"#,
        )
        .unwrap();
        write_tokenizer(&dir);

        let shapes = synthetic_tensor_shapes(tied);
        let mut by_shard = BTreeMap::<String, Vec<(String, Vec<usize>)>>::new();
        for (name, shape) in shapes {
            by_shard
                .entry(shard_for(&name).into())
                .or_default()
                .push((name, shape));
        }
        let mut weight_map = serde_json::Map::new();
        let mut total_size = 0u64;
        for (shard, tensors) in &by_shard {
            write_fake_safetensors(&dir.join(shard), tensors);
            for (name, shape) in tensors {
                weight_map.insert(name.clone(), shard.clone().into());
                total_size += (shape.iter().product::<usize>() * 2) as u64;
            }
        }
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "metadata": { "total_size": total_size },
                "weight_map": weight_map,
            }))
            .unwrap(),
        )
        .unwrap();
        Self { dir }
    }
}

impl Drop for SyntheticFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn synthetic_config(tied: bool) -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "decoder_sparse_step": 1,
        "eos_token_id": 2,
        "head_dim": 2,
        "hidden_act": "silu",
        "hidden_size": 4,
        "initializer_range": 0.02,
        "intermediate_size": 8,
        "max_position_embeddings": 128,
        "max_window_layers": 2,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 3,
        "norm_topk_prob": true,
        "num_attention_heads": 2,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "output_router_logits": false,
        "rms_norm_eps": 1e-6,
        "rope_scaling": null,
        "rope_theta": 1_000_000.0,
        "router_aux_loss_coef": 0.001,
        "sliding_window": null,
        "tie_word_embeddings": tied,
        "torch_dtype": "bfloat16",
        "transformers_version": "test",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 8
    })
}

fn synthetic_tensor_shapes(tied: bool) -> BTreeMap<String, Vec<usize>> {
    let mut tensors = BTreeMap::from([
        ("model.embed_tokens.weight".into(), vec![8, 4]),
        ("model.norm.weight".into(), vec![4]),
    ]);
    if !tied {
        tensors.insert("lm_head.weight".into(), vec![8, 4]);
    }
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        tensors.insert(format!("{prefix}.input_layernorm.weight"), vec![4]);
        tensors.insert(format!("{prefix}.post_attention_layernorm.weight"), vec![4]);
        tensors.insert(format!("{prefix}.self_attn.q_proj.weight"), vec![4, 4]);
        tensors.insert(format!("{prefix}.self_attn.k_proj.weight"), vec![2, 4]);
        tensors.insert(format!("{prefix}.self_attn.v_proj.weight"), vec![2, 4]);
        tensors.insert(format!("{prefix}.self_attn.o_proj.weight"), vec![4, 4]);
        tensors.insert(format!("{prefix}.self_attn.q_norm.weight"), vec![2]);
        tensors.insert(format!("{prefix}.self_attn.k_norm.weight"), vec![2]);
        tensors.insert(format!("{prefix}.mlp.gate.weight"), vec![3, 4]);
        for expert in 0..3 {
            let expert = format!("{prefix}.mlp.experts.{expert}");
            tensors.insert(format!("{expert}.gate_proj.weight"), vec![3, 4]);
            tensors.insert(format!("{expert}.up_proj.weight"), vec![3, 4]);
            tensors.insert(format!("{expert}.down_proj.weight"), vec![4, 3]);
        }
    }
    tensors
}

fn shard_for(name: &str) -> &'static str {
    if name.ends_with("gate_proj.weight") {
        "model-00001-of-00003.safetensors"
    } else if name.ends_with("up_proj.weight") || name.contains("self_attn") {
        "model-00002-of-00003.safetensors"
    } else {
        "model-00003-of-00003.safetensors"
    }
}

fn write_fake_safetensors(path: &Path, tensors: &[(String, Vec<usize>)]) {
    let mut header = serde_json::Map::new();
    let mut payload_bytes = 0usize;
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<usize>() * 2;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [payload_bytes, payload_bytes + bytes],
            }),
        );
        payload_bytes += bytes;
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = Vec::with_capacity(8 + header.len() + 7 + payload_bytes);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    while file.len() % 8 != 0 {
        file.push(b' ');
    }
    file.resize(file.len() + payload_bytes, 0);
    std::fs::write(path, file).unwrap();
}

fn write_tokenizer(dir: &Path) {
    let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
    let tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
        .into_iter()
        .map(|token| tokenizers::AddedToken::from(token, false));
    assert_eq!(tokenizer.add_tokens(tokens).unwrap(), 8);
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()))
}
