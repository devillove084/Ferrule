//! OLMoE model configuration — parsed from config.json.
use ferrule_core::Result;

#[derive(Debug, Clone)]
pub struct OlmoeConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub norm_topk_prob: bool,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

impl OlmoeConfig {
    pub fn from_json(json: &serde_json::Value) -> Result<Self> {
        fn token_id(v: Option<&serde_json::Value>) -> Option<u32> {
            match v? {
                serde_json::Value::Number(n) => n.as_u64().map(|id| id as u32),
                serde_json::Value::Array(ids) => ids.first()?.as_u64().map(|id| id as u32),
                _ => None,
            }
        }

        let h = json["hidden_size"].as_u64().unwrap_or(2048) as usize;
        let nh = json["num_attention_heads"].as_u64().unwrap_or(16) as usize;
        // Try multiple config key names for num KV heads
        let nkv = json["num_key_value_heads"]
            .as_u64()
            .or_else(|| json["num_kv_heads"].as_u64())
            .or_else(|| json["kv_heads"].as_u64())
            .map(|v| v as usize)
            .unwrap_or(nh); // fallback to nh; will be corrected from weights after loading
        let hd = h / nh;
        Ok(Self {
            hidden_size: h,
            num_layers: json["num_hidden_layers"].as_u64().unwrap_or(16) as usize,
            num_experts: json["num_experts"].as_u64().unwrap_or(64) as usize,
            num_experts_per_tok: json["num_experts_per_tok"].as_u64().unwrap_or(8) as usize,
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(1024) as usize,
            vocab_size: json["vocab_size"].as_u64().unwrap_or(50304) as usize,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            kv_dim: nkv * hd,
            rope_theta: json["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            rms_norm_eps: json
                .get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32,
            norm_topk_prob: json
                .get("norm_topk_prob")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            eos_token_id: token_id(json.get("eos_token_id")),
            pad_token_id: token_id(json.get("pad_token_id")),
        })
    }

    /// Correct kv_dim/nkv from actual weight dimensions (call after loading layer 0).
    pub fn correct_kv_heads(&mut self, actual_kv_dim: usize) {
        if actual_kv_dim != self.kv_dim && actual_kv_dim > 0 {
            tracing::info!(
                "  Correcting kv_dim: {} → {} (from weights)",
                self.kv_dim,
                actual_kv_dim
            );
            self.kv_dim = actual_kv_dim;
            self.num_kv_heads = actual_kv_dim / self.head_dim;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_json_minimal() {
        let json = serde_json::json!({});
        let config = OlmoeConfig::from_json(&json).unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 16);
        assert_eq!(config.num_experts, 64);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.num_kv_heads, 16);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.kv_dim, 2048);
        assert_eq!(config.rope_theta, 10000.0);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!(!config.norm_topk_prob);
        assert_eq!(config.eos_token_id, None);
    }

    #[test]
    fn test_from_json_with_kv_heads() {
        let json = serde_json::json!({
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_hidden_layers": 12
        });
        let config = OlmoeConfig::from_json(&json).unwrap();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.kv_dim, 256);
    }

    #[test]
    fn test_correct_kv_heads() {
        let json = serde_json::json!({});
        let mut config = OlmoeConfig::from_json(&json).unwrap();
        assert_eq!(config.kv_dim, 2048);
        assert_eq!(config.num_kv_heads, 16);
        config.correct_kv_heads(512);
        assert_eq!(config.kv_dim, 512);
        assert_eq!(config.num_kv_heads, 4); // 512 / 128 = 4
    }

    #[test]
    fn test_correct_kv_heads_noop() {
        let json = serde_json::json!({});
        let mut config = OlmoeConfig::from_json(&json).unwrap();
        let original_kv_dim = config.kv_dim;
        config.correct_kv_heads(original_kv_dim);
        assert_eq!(config.kv_dim, original_kv_dim);
    }

    #[test]
    fn test_eos_token_id_single() {
        let json = serde_json::json!({"eos_token_id": 50256});
        let config = OlmoeConfig::from_json(&json).unwrap();
        assert_eq!(config.eos_token_id, Some(50256));
    }

    #[test]
    fn test_eos_token_id_array() {
        let json = serde_json::json!({"eos_token_id": [50256, 0]});
        let config = OlmoeConfig::from_json(&json).unwrap();
        assert_eq!(config.eos_token_id, Some(50256));
    }
}
