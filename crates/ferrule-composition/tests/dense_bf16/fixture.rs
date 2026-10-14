use ferrule_composition::{ModelFactoryOptions, ModelPrecision, ServingModelOptions};
use ferrule_model::AutoConfig;
use ferrule_runtime::{ResidentSchedulerConfig, ResidentTopKDriverConfig};

pub struct Fixture {
    pub dir: tempfile::TempDir,
    pub oracle: serde_json::Value,
}
impl Fixture {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../../ferrule-model/tests/fixtures/standard_bf16_hf.json"
        ))
        .unwrap();
        assert_eq!(oracle["schema"], "ferrule.standard-bf16-hf-eager.v2");
        assert_eq!(
            oracle["rotary_contract"]["inverse_frequency_dtype"],
            "float32"
        );
        let fixture = Self { dir, oracle };
        fixture.write(false);
        fixture
    }
    pub fn config(&self) -> AutoConfig {
        AutoConfig::from_pretrained(self.dir.path()).unwrap()
    }
    pub fn write(&self, f32: bool) {
        std::fs::write(
            self.dir.path().join("config.json"),
            self.oracle["config"].to_string(),
        )
        .unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for w in self.oracle["weights"].as_array().unwrap() {
            let start = payload.len();
            for v in w["values"].as_array().unwrap() {
                let v = v.as_f64().unwrap() as f32;
                assert_eq!(v.to_bits() & 0xffff, 0);
                if f32 {
                    payload.extend_from_slice(&v.to_le_bytes());
                } else {
                    payload.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
                }
            }
            header.insert(w["name"].as_str().unwrap().into(), serde_json::json!({
                "dtype": if f32 { "F32" } else { "BF16" }, "shape": w["shape"], "data_offsets": [start, payload.len()]
            }));
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(self.dir.path().join("model.safetensors"), file).unwrap();
        let vocab = (0..16)
            .map(|i| {
                (
                    if i == 0 {
                        "<unk>".into()
                    } else {
                        format!("t{i}")
                    },
                    serde_json::json!(i),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        std::fs::write(
            self.dir.path().join("tokenizer.json"),
            serde_json::json!({
                "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
                "normalizer":null, "pre_tokenizer":null, "post_processor":null, "decoder":null,
                "model":{"type":"WordLevel", "vocab":vocab,"unk_token":"<unk>"}
            })
            .to_string(),
        )
        .unwrap();
    }
}
pub fn options() -> ModelFactoryOptions {
    ModelFactoryOptions {
        max_layers: None,
        max_tensor_mebibytes: 1,
        output_head_chunk_rows: 16,
        expert_reader_max_tensor_mebibytes: 1,
        expert_cache: Default::default(),
        qwen35_moe_capacity: None,
        numeric_fp8_profile: None,
        qwen35_host_cache: None,
        moe_hotset_experts: 0,
        kv_cache_mebibytes: Some(1),
        scheduler_config: ResidentSchedulerConfig {
            max_batch_tokens: 3,
            prefill_chunk_size: 3,
            max_active_sequences: 2,
            prefix_cache_capacity_pages: 0,
            ..Default::default()
        },
        driver_config: ResidentTopKDriverConfig {
            ctx_size: 16,
            stop_at_eos: false,
            ..Default::default()
        },
    }
}
pub fn serving() -> ServingModelOptions {
    ServingModelOptions {
        model_precision: Some(ModelPrecision::Bf16),
        ..Default::default()
    }
}
