//! Synthetic DSV4 model builder using real config shapes.
//!
//! This module creates mock/synthetic weights matching real DeepSeek V4 shapes
//! (from the downloaded config.json and model.safetensors.index.json) so the
//! full 43-layer pipeline can be tested without the 166GB checkpoint.

use std::path::{Path, PathBuf};

use ferrule_core::Result;
use ferrule_model::TensorRole;

use crate::dsv4_param;
// removed unused
// removed unused
use crate::expert_streaming::{
    ExpertId, ExpertSource, ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertTensorComponent,
    ExpertTensorKey, ExpertTensorPayload, ExpertTensorSlice,
};
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{HyperConnectionConfig, HyperConnectionWeights};
use crate::layer_binding::LayerSourceBinding;
use crate::source_binding::{AttentionSourcePayload, RouterSourcePayload};
use crate::source_linear::SourceLinearPayload;
use crate::source_tensor::{SourceDType, SourceTensorPayload, SourceTensorSlice};
use crate::ExpertMatrixKind;

/// Minimal tokenizer mock for decode-smoke purposes.
pub struct SyntheticTokenizer {
    vocab_size: usize,
}

impl SyntheticTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: dsv4_param::VOCAB_SIZE,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Deterministic mock: each char maps to a token id.
        text.bytes()
            .map(|b| b as u32 % self.vocab_size as u32)
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|id| format!("[{id}]")).collect()
    }

    pub fn eos_token_id(&self) -> u32 {
        1
    }
}

/// Builds a tiny synthetic F32 linear payload with given [out, in].
pub fn f32_linear_identity(out: usize, input: usize, diagonal_value: f32) -> SourceLinearPayload {
    let mut values = vec![0.0f32; out * input];
    for i in 0..out.min(input) {
        values[i * input + i] = diagonal_value;
    }
    SourceLinearPayload::from_weight_and_scale(
        TensorRole::AttentionLatentQueryA,
        SourceTensorPayload {
            slice: SourceTensorSlice {
                name: "mock.weight".into(),
                role: TensorRole::Unknown,
                path: PathBuf::from("mock.safetensors"),
                offset: 0,
                bytes: (values.len() * 4) as u64,
                dtype: SourceDType::F32,
                shape: vec![out, input],
            },
            bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        None,
    )
    .unwrap()
}

/// Builds a synthetic attention payload matching DSV4 shapes.
fn mock_attention_payload() -> AttentionSourcePayload {
    let h = dsv4_param::HIDDEN_SIZE;
    let qr = dsv4_param::Q_LORA_RANK;
    let or = dsv4_param::O_LORA_RANK;
    let hd = dsv4_param::HEAD_DIM;
    let nh = dsv4_param::NUM_HEADS;
    let kv = hd; // kv dimension = head_dim

    AttentionSourcePayload {
        layer: 0,
        wq_a: f32_linear_identity(qr, h, 1.0 / h as f32),
        wq_b: f32_linear_identity(nh * hd, qr, 1.0 / (nh * hd) as f32),
        wkv: f32_linear_identity(kv, h, 1.0 / h as f32),
        wo_a: f32_linear_identity(or, nh * hd, 1.0 / (nh * hd) as f32),
        wo_b: f32_linear_identity(h, or, 1.0 / h as f32),
        q_norm: vec![1.0; qr],
        kv_norm: vec![1.0; kv],
        attention_sink: vec![0.0; nh],
        auxiliary: Vec::new(),
    }
}

/// Builds mock HC weights matching a DSV4 config.
fn mock_hc_weights(config: HyperConnectionConfig) -> HyperConnectionWeights {
    HyperConnectionWeights {
        function: vec![0.0; config.mix_hc() * (config.hc_mult * config.hidden_size)],
        scale: vec![1.0, 1.0, 1.0],
        base: vec![0.0; config.mix_hc()],
    }
}

/// Builds a synthetic shared FFN matching DSV4 shapes.
fn mock_shared_ffn() -> SwiGluFfnPayload {
    let h = dsv4_param::HIDDEN_SIZE;
    let inter = dsv4_param::MOE_INTERMEDIATE_SIZE;
    SwiGluFfnPayload {
        gate: f32_linear_identity(inter, h, 0.5 / h as f32),
        up: f32_linear_identity(inter, h, 0.5 / h as f32),
        down: f32_linear_identity(h, inter, 0.5 / inter as f32),
        swiglu_limit: dsv4_param::SWIGLU_LIMIT,
    }
}

/// Builds one tiny FP4 routed expert with weight/scale pairs.
fn tiny_fp4_payload(
    expert: ExpertId,
    matrix: ExpertMatrixKind,
    first_byte: u8,
) -> ExpertTensorPayload {
    let out = dsv4_param::MOE_INTERMEDIATE_SIZE;
    let packed_in = dsv4_param::HIDDEN_SIZE / 2;
    let mut bytes = vec![0u8; out * packed_in];
    bytes[0] = first_byte;
    ExpertTensorPayload {
        slice: ExpertTensorSlice {
            key: ExpertTensorKey { expert, matrix },
            component: ExpertTensorComponent::Weight,
            path: PathBuf::from("mock.safetensors"),
            offset: 0,
            bytes: bytes.len() as u64,
            dtype: "I8".into(),
            shape: vec![out, packed_in],
        },
        bytes,
    }
}

fn tiny_scale_payload(expert: ExpertId, matrix: ExpertMatrixKind) -> ExpertTensorPayload {
    let out = dsv4_param::MOE_INTERMEDIATE_SIZE;
    let scale_cols = dsv4_param::HIDDEN_SIZE / 32;
    let bytes = vec![127u8; out * scale_cols];
    ExpertTensorPayload {
        slice: ExpertTensorSlice {
            key: ExpertTensorKey { expert, matrix },
            component: ExpertTensorComponent::Scale,
            path: PathBuf::from("mock.safetensors"),
            offset: 0,
            bytes: bytes.len() as u64,
            dtype: "F8_E8M0".into(),
            shape: vec![out, scale_cols],
        },
        bytes,
    }
}

/// Registers a single tiny routed expert into a temp directory + planner.
pub fn register_mock_expert(
    dir: &Path,
    planner: &mut ExpertStreamingPlanner,
    layer: usize,
    expert: usize,
    gate_byte: u8,
    up_byte: u8,
    down_byte: u8,
) {
    let expert_id = ExpertId::new(layer, expert);
    let path = dir.join(format!("l{layer}e{expert}.bin"));
    let tensors = vec![
        tiny_fp4_payload(expert_id, ExpertMatrixKind::Gate, gate_byte),
        tiny_scale_payload(expert_id, ExpertMatrixKind::Gate),
        tiny_fp4_payload(expert_id, ExpertMatrixKind::Up, up_byte),
        tiny_scale_payload(expert_id, ExpertMatrixKind::Up),
        tiny_fp4_payload(expert_id, ExpertMatrixKind::Down, down_byte),
        tiny_scale_payload(expert_id, ExpertMatrixKind::Down),
    ];
    let mut bytes: Vec<u8> = Vec::new();
    for tensor in &tensors {
        bytes.extend(&tensor.bytes);
    }
    let _ = std::fs::write(&path, &bytes);
    let mut offset = 0u64;
    let slices: Vec<ExpertTensorSlice> = tensors
        .into_iter()
        .map(|tensor| {
            let b = tensor.bytes.len() as u64;
            let s = ExpertTensorSlice {
                offset,
                bytes: b,
                ..tensor.slice
            };
            offset += b;
            s
        })
        .collect();
    planner.register_source(expert_id, ExpertSource::LocalTensorSet { tensors: slices });
}

/// Builds the full compute state for one synthetic DSV4 layer.
pub fn build_mock_layer(layer: usize, config: HyperConnectionConfig) -> LayerSourceBinding {
    let is_hash = dsv4_param::is_hash_layer(layer);
    LayerSourceBinding {
        layer,
        attention: mock_attention_payload(),
        hc_attention: mock_hc_weights(config),
        hc_feed_forward: mock_hc_weights(config),
        router: RouterSourcePayload {
            layer,
            weight: f32_linear_identity(dsv4_param::N_ROUTED_EXPERTS, dsv4_param::HIDDEN_SIZE, 0.0),
            bias: if is_hash {
                None
            } else {
                Some(vec![0.0f32; dsv4_param::N_ROUTED_EXPERTS])
            },
            hash_table: if is_hash {
                Some((0..dsv4_param::N_ROUTED_EXPERTS).collect())
            } else {
                None
            },
            hash_rows: if is_hash { dsv4_param::VOCAB_SIZE } else { 0 },
            hash_cols: if is_hash {
                dsv4_param::N_ROUTED_EXPERTS
            } else {
                0
            },
        },
        shared_ffn: Some(mock_shared_ffn()),
        router_policy: dsv4_param::router_policy_for_layer(layer),
        attention_spec: dsv4_param::attention_spec(),
    }
}

/// Builds per-layer execution state for all 43 layers.
pub fn build_mock_execution_state(dir: &Path) -> crate::layer_binding::LayerExecutionState {
    let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
        gpu_slots_per_layer: dsv4_param::NUM_EXPERTS_PER_TOK,
        prefetch_per_layer: 0,
        preserve_source_quantization: true,
        allow_cpu_staging: false,
        allow_remote_sources: false,
    });
    // Register one tiny expert per layer for the mock path.
    for layer in 0..dsv4_param::NUM_LAYERS {
        register_mock_expert(dir, &mut planner, layer, 0, 0x42, 0x43, 0x22);
    }
    crate::layer_binding::LayerExecutionState::new(dsv4_param::HEAD_DIM, planner)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn mock_layer_uses_correct_shapes_from_official_config() {
        let config = dsv4_param::hc_config();
        let binding = build_mock_layer(0, config);
        assert_eq!(binding.layer, 0);
        assert_eq!(
            binding.attention.wq_a.format.in_features(),
            dsv4_param::HIDDEN_SIZE
        );
        assert_eq!(
            binding.attention.wq_a.format.out_features(),
            dsv4_param::Q_LORA_RANK
        );
        assert_eq!(
            binding.attention.wq_b.format.out_features(),
            dsv4_param::NUM_HEADS * dsv4_param::HEAD_DIM
        );
        assert_eq!(
            binding.attention.attention_sink.len(),
            dsv4_param::NUM_HEADS
        );
        assert_eq!(
            binding.router.weight.format.out_features(),
            dsv4_param::N_ROUTED_EXPERTS
        );
        assert_eq!(
            binding
                .shared_ffn
                .as_ref()
                .unwrap()
                .gate
                .format
                .in_features(),
            dsv4_param::HIDDEN_SIZE
        );
        assert_eq!(
            binding
                .shared_ffn
                .as_ref()
                .unwrap()
                .gate
                .format
                .out_features(),
            dsv4_param::MOE_INTERMEDIATE_SIZE
        );
        assert!(binding.router.hash_table.is_some());
        assert_eq!(
            binding.hc_attention.function.len(),
            config.mix_hc() * (config.hc_mult * config.hidden_size)
        );
    }

    #[test]
    fn mock_score_layer_has_bias_not_hash() {
        let config = dsv4_param::hc_config();
        let binding = build_mock_layer(3, config);
        assert!(binding.router.bias.is_some());
        assert!(binding.router.hash_table.is_none());
    }

    #[test]
    fn mock_layer_execution_state_registers_experts() {
        let dir = unique_temp_dir("ferrule-dsv4-mock-state");
        std::fs::create_dir_all(&dir).unwrap();
        let state = build_mock_execution_state(&dir);
        for layer in 0..dsv4_param::NUM_LAYERS {
            let resident = state.expert_planner.resident_experts(layer);
            assert_eq!(
                resident.len(),
                0,
                "layer {layer} should have no pre-loaded experts"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nonce}"))
}
