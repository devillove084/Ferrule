//! Mixed-precision policy — modeled after llama.cpp's `ggml_ftype` approach.
//!
//! llama.cpp uses `enum ggml_ftype` for model-level quantization presets:
//!   GGML_FTYPE_MOSTLY_Q4_K  → all tensors Q4_K except 1D (norms/biases stay FP32)
//!   GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 → embed+output in F16, rest Q4_1
//!
//! Ferrule's PrecisionPolicy mirrors this: named presets with per-tensor overrides.
//! Individual tensor overrides are stored in qcache manifest for qcache-only startup.

use ferrule_core::QuantType;
use serde::{Deserialize, Serialize};

/// Named quantization preset — analogous to llama.cpp's `ggml_ftype`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantPreset {
    /// All FP32 (no quantization).
    F32,
    /// Most tensors Q4_0, norms+embed+lm_head stay FP32.
    MostlyQ4,
    /// Most tensors Q8_0, norms+embed+lm_head stay FP32.
    MostlyQ8,
    /// Most tensors Q4_0, embed+lm_head F16, norms FP32.
    MostlyQ4SomeF16,
    /// Fully Q4_0 except embedding (useful for memory-constrained edge).
    AggressiveQ4,
}

impl QuantPreset {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::MostlyQ4 => "mostly-q4",
            Self::MostlyQ8 => "mostly-q8",
            Self::MostlyQ4SomeF16 => "mostly-q4-some-f16",
            Self::AggressiveQ4 => "aggressive-q4",
        }
    }

    /// Compute the default dtype for a weight matrix given its shape.
    /// `dim0` and `dim1` follow llama.cpp conventions:
    ///   - 1D tensors (dim0==1 or dim1==1) are norms/biases → stay FP32
    ///   - Embedding/lm_head have special treatment in SOME_F16 presets
    pub fn default_dtype(self, dim0: usize, dim1: usize) -> QuantType {
        let is_1d = dim0 == 1 || dim1 == 1;
        match self {
            Self::F32 => QuantType::F32,
            Self::MostlyQ4 => {
                if is_1d {
                    QuantType::F32
                } else {
                    QuantType::Q4_0
                }
            }
            Self::MostlyQ8 => {
                if is_1d {
                    QuantType::F32
                } else {
                    QuantType::Q8_0
                }
            }
            Self::MostlyQ4SomeF16 => {
                if is_1d {
                    QuantType::F32
                } else {
                    QuantType::Q4_0
                }
                // Embedding/lm_head override handled separately via per-tensor policy
            }
            Self::AggressiveQ4 => {
                // Everything quantized — caller overrides embedding manually
                QuantType::Q4_0
            }
        }
    }
}

/// Per-tensor dtype override. Used for quality-sensitive tensors (embed, lm_head).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDtypeOverride {
    /// Tensor name pattern (prefix match, e.g. "model.embed_tokens").
    pub name_pattern: String,
    /// Forced dtype for matching tensors.
    pub dtype: QuantType,
}

/// Full precision policy = preset + optional per-tensor overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionPolicy {
    pub preset: QuantPreset,
    #[serde(default)]
    pub overrides: Vec<TensorDtypeOverride>,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self {
            preset: QuantPreset::MostlyQ4,
            overrides: Vec::new(),
        }
    }
}

impl PrecisionPolicy {
    /// Resolve the dtype for a specific tensor given its name and shape.
    pub fn resolve(&self, tensor_name: &str, dim0: usize, dim1: usize) -> QuantType {
        // Check per-tensor overrides first (specific beats general)
        for ov in &self.overrides {
            if tensor_name.starts_with(&ov.name_pattern) {
                return ov.dtype;
            }
        }
        self.preset.default_dtype(dim0, dim1)
    }

    /// Create the standard OLMoE policy: Q4_0 for projections, FP32 for norms/embed/lm_head.
    pub fn olmoe_q4() -> Self {
        Self {
            preset: QuantPreset::MostlyQ4,
            overrides: vec![
                TensorDtypeOverride {
                    name_pattern: "model.embed_tokens".into(),
                    dtype: QuantType::F32,
                },
                TensorDtypeOverride {
                    name_pattern: "lm_head".into(),
                    dtype: QuantType::F32,
                },
                TensorDtypeOverride {
                    name_pattern: "model.norm".into(),
                    dtype: QuantType::F32,
                },
            ],
        }
    }

    /// Create an aggressive edge-deployment policy.
    pub fn edge_q4() -> Self {
        Self {
            preset: QuantPreset::AggressiveQ4,
            overrides: vec![
                TensorDtypeOverride {
                    name_pattern: "model.embed_tokens".into(),
                    dtype: QuantType::F32,
                },
                TensorDtypeOverride {
                    name_pattern: "lm_head".into(),
                    dtype: QuantType::F16,
                },
            ],
        }
    }

    /// Estimate total quantized model size in bytes.
    pub fn estimate_bytes(
        &self,
        tensors: &[(&str, usize, usize)], // (name, dim0, dim1)
    ) -> f64 {
        tensors
            .iter()
            .map(|&(name, d0, d1)| {
                let dt = self.resolve(name, d0, d1);
                (d0 * d1) as f64 * dt.type_size()
            })
            .sum()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mostly_q4_skips_1d() {
        let p = PrecisionPolicy {
            preset: QuantPreset::MostlyQ4,
            overrides: vec![],
        };
        // Norm weight (1D-like: dim1=1 but we treat dim0×dim1 as matrix)
        assert_eq!(
            p.resolve("model.layers.0.input_layernorm.weight", 2048, 1),
            QuantType::F32
        );
        // Attention projection (2D)
        assert_eq!(
            p.resolve("model.layers.0.self_attn.q_proj.weight", 2048, 2048),
            QuantType::Q4_0
        );
    }

    #[test]
    fn per_tensor_override_wins() {
        let p = PrecisionPolicy {
            preset: QuantPreset::AggressiveQ4,
            overrides: vec![TensorDtypeOverride {
                name_pattern: "model.embed_tokens".into(),
                dtype: QuantType::F32,
            }],
        };
        assert_eq!(
            p.resolve("model.embed_tokens.weight", 50304, 2048),
            QuantType::F32
        );
        assert_eq!(
            p.resolve("model.layers.0.self_attn.q_proj.weight", 2048, 2048),
            QuantType::Q4_0
        );
    }

    #[test]
    fn olmoe_policy_correct() {
        let p = PrecisionPolicy::olmoe_q4();
        assert_eq!(p.preset, QuantPreset::MostlyQ4);
        // embed stays FP32 via override
        assert_eq!(
            p.resolve("model.embed_tokens.weight", 50304, 2048),
            QuantType::F32
        );
        assert_eq!(p.resolve("lm_head.weight", 50304, 2048), QuantType::F32);
        // projection is Q4_0 via preset
        // Router is [ne,d] = 2D → gets Q4_0 under MostlyQ4 (test verifies preset behavior)
        assert_eq!(
            p.resolve("model.layers.0.mlp.gate.weight", 64, 2048),
            QuantType::Q4_0
        );
    }

    #[test]
    fn edge_policy_quantizes_lm_head() {
        let p = PrecisionPolicy::edge_q4();
        assert_eq!(p.resolve("lm_head.weight", 50304, 2048), QuantType::F16);
        assert_eq!(
            p.resolve("model.embed_tokens.weight", 50304, 2048),
            QuantType::F32
        );
        assert_eq!(
            p.resolve("model.layers.0.self_attn.q_proj.weight", 2048, 2048),
            QuantType::Q4_0
        );
    }

    #[test]
    fn preset_names_unique() {
        let names: Vec<&str> = [
            QuantPreset::F32,
            QuantPreset::MostlyQ4,
            QuantPreset::MostlyQ8,
            QuantPreset::MostlyQ4SomeF16,
            QuantPreset::AggressiveQ4,
        ]
        .iter()
        .map(|p| p.name())
        .collect();
        // All unique
        let set: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(set.len(), names.len());
    }

    #[test]
    fn estimate_bytes_realistic() {
        let tensors = vec![
            ("embed", 50304, 2048),                      // 103M elements
            ("lm_head", 50304, 2048),                    // 103M elements
            ("norm", 2048, 1),                           // 2K elements
            ("q_proj_x16", 2048, 2048 * 16),             // 67M elements
            ("expert_down_x1024", 2048, 1024 * 64 * 16), // ~2.1B elements (huge!)
        ];
        let p = PrecisionPolicy::olmoe_q4();
        let bytes = p.estimate_bytes(&tensors);
        // Should be around: (103+103)M×4 + 2K×4 + 67M×0.5 + 2.1B×0.5 ≈ 1.9 GB
        assert!(bytes > 1e9);
        assert!(bytes < 3e9);
    }
}
