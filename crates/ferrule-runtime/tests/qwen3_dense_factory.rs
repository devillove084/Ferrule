use std::path::PathBuf;

use ferrule_model::{AutoConfig, ChatTemplate, ModelExecutionBackend, ModelFamily};
use ferrule_runtime::{
    BackendSelection, BuiltinModelResolver, ModelFactoryOptions, ResidentModelPlanner,
    ResidentSchedulerConfig, ResidentTopKDriverConfig,
};

#[test]
fn qwen3_dense_factory_builds_single_file_cpu_engine() {
    let fixture = Fixture::new();
    let config = AutoConfig::from_pretrained(&fixture.0).unwrap();
    assert!(config.descriptor().engine_plan().is_executable());
    let plan = ResidentModelPlanner::new()
        .prepare(&config, BackendSelection::Auto, None, options())
        .unwrap();
    assert_eq!(plan.model_name(), "qwen3-dense");
    assert_eq!(plan.backend(), ModelExecutionBackend::Cpu);
    assert_eq!(plan.backend_profile(), "cpu-standard-decoder");
    assert_eq!(plan.chat_template(), ChatTemplate::Qwen3);
    let engine = plan.build().unwrap();
    let info = engine.model_info();
    assert_eq!(info.family, ModelFamily::Qwen3);
    assert_eq!(info.architecture.as_deref(), Some("Qwen3ForCausalLM"));
    assert_eq!(info.backend, "cpu");
    assert_eq!(info.num_layers, 1);
    assert_eq!(info.num_experts, 0);
    assert!(!fixture.0.join("model.safetensors.index.json").exists());

    // This rejection must remain true in CUDA-enabled builds too.
    let error = BuiltinModelResolver::new()
        .resolve(config.descriptor(), BackendSelection::Cuda)
        .unwrap_err();
    assert!(error.to_string().contains("supported: cpu"));
}

fn options() -> ModelFactoryOptions {
    ModelFactoryOptions {
        max_layers: None,
        max_tensor_mebibytes: 1,
        output_head_chunk_rows: 8,
        expert_reader_max_tensor_mebibytes: 1,
        expert_cache: Default::default(),
        moe_hotset_experts: 0,
        kv_cache_mebibytes: None,
        scheduler_config: ResidentSchedulerConfig {
            max_batch_tokens: 3,
            prefill_chunk_size: 3,
            max_active_sequences: 1,
            ..Default::default()
        },
        driver_config: ResidentTopKDriverConfig {
            ctx_size: 8,
            ..Default::default()
        },
    }
}

// std-only fixture: the runtime crate need not acquire tokenizer or JSON test dependencies.
struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ferrule-qwen3-dense-factory-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{
            "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3",
            "hidden_size":4, "intermediate_size":6, "num_hidden_layers":1,
            "num_attention_heads":4, "num_key_value_heads":2, "head_dim":2,
            "rms_norm_eps":0.000001, "rope_theta":1000000.0, "rope_scaling":null,
            "max_position_embeddings":8, "vocab_size":8, "tie_word_embeddings":true,
            "attention_bias":false, "attention_dropout":0.0, "hidden_act":"silu",
            "torch_dtype":"bfloat16", "use_cache":true, "use_sliding_window":false,
            "sliding_window":null, "max_window_layers":1, "initializer_range":0.02,
            "bos_token_id":1, "eos_token_id":2
        }"#,
        )
        .unwrap();
        std::fs::write(dir.join("tokenizer.json"), r#"{
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":null, "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel", "vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7},"unk_token":"<unk>"}
        }"#).unwrap();
        let mut entries = Vec::new();
        let mut payload = Vec::new();
        for (name, shape) in [
            ("model.embed_tokens.weight", vec![8, 4]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![8, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 8]),
            ("model.layers.0.self_attn.q_norm.weight", vec![2]),
            ("model.layers.0.self_attn.k_norm.weight", vec![2]),
            ("model.layers.0.mlp.gate_proj.weight", vec![6, 4]),
            ("model.layers.0.mlp.up_proj.weight", vec![6, 4]),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 6]),
        ] {
            let start = payload.len();
            for _ in 0..shape.iter().product::<usize>() {
                payload.extend_from_slice(&0x3f00u16.to_le_bytes());
            }
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",
                payload.len()
            ));
        }
        let mut header = format!("{{{}}}", entries.join(",")).into_bytes();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        Self(dir)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
