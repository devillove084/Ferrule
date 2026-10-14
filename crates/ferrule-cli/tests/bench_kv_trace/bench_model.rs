use serde_json::json;
use std::path::PathBuf;
pub struct Fixture(pub PathBuf);

impl Fixture {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrule-cli-bench-trace-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        let fixture = Self(path);
        std::fs::write(
            fixture.0.join("config.json"),
            json!({
                "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3",
                "hidden_size":4, "intermediate_size":8, "num_hidden_layers":2,
                "num_attention_heads":2, "num_key_value_heads":1, "head_dim":2,
                "rms_norm_eps":0.000001, "rope_theta":10000.0, "rope_scaling":null,
                "max_position_embeddings":4096, "vocab_size":8, "tie_word_embeddings":true,
                "attention_bias":false, "attention_dropout":0.0, "hidden_act":"silu",
                "torch_dtype":"bfloat16", "use_cache":true, "use_sliding_window":false,
                "sliding_window":null, "max_window_layers":2, "initializer_range":0.02,
                "bos_token_id":2, "eos_token_id":7
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(fixture.0.join("tokenizer.json"), json!({
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":{"type":"WhitespaceSplit"},
            "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel", "vocab":{"<unk>":0,"a":1,"hello":2,"b":3,"c":4,"d":5,"e":6,"<eos>":7},"unk_token":"<unk>"}
        }).to_string()).unwrap();
        let mut tensors = vec![
            ("model.embed_tokens.weight".to_owned(), vec![8, 4]),
            ("model.norm.weight".to_owned(), vec![4]),
        ];
        for layer in 0..2 {
            for (name, shape) in [
                ("input_layernorm.weight", vec![4]),
                ("post_attention_layernorm.weight", vec![4]),
                ("self_attn.q_proj.weight", vec![4, 4]),
                ("self_attn.k_proj.weight", vec![2, 4]),
                ("self_attn.v_proj.weight", vec![2, 4]),
                ("self_attn.o_proj.weight", vec![4, 4]),
                ("self_attn.q_norm.weight", vec![2]),
                ("self_attn.k_norm.weight", vec![2]),
                ("mlp.gate_proj.weight", vec![8, 4]),
                ("mlp.up_proj.weight", vec![8, 4]),
                ("mlp.down_proj.weight", vec![4, 8]),
            ] {
                tensors.push((format!("model.layers.{layer}.{name}"), shape));
            }
        }
        let mut payload = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, shape) in tensors {
            let start = payload.len();
            for index in 0..shape.iter().product::<usize>() {
                let value: f32 = if name == "model.embed_tokens.weight" {
                    if index / 4 == 1 { 0.5 } else { 0.125 }
                } else if shape.len() == 1 {
                    1.0
                } else {
                    0.0
                };
                payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            header.insert(
                name,
                json!({"dtype":"BF16", "shape":shape, "data_offsets":[start,payload.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(fixture.0.join("model.safetensors"), file).unwrap();
        fixture
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
