#![cfg(test)]

use super::attention::*;
use super::config::*;
use super::helpers::*;
use super::layer::*;
use super::sequence::DeepSeekV4SequenceExecutionState;

use std::path::{Path, PathBuf};

use crate::artifact::binding::{MlaAttentionArtifactPayload, RouterArtifactPayload};
use crate::artifact::linear::{artifact_linear_cache_key, ArtifactLinearPayload};
use crate::artifact::tensor::{ArtifactDType, ArtifactTensorPayload, ArtifactTensorSlice};
use crate::families::deepseek_v4;
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{HyperConnectionConfig, HyperConnectionWeights};
use crate::moe::executor::CpuReferenceExpertExecutor;
use crate::moe::routing::ExpertRouterPolicy;
use crate::moe::streaming::{
    ExpertId, ExpertLoadSource, ExpertMatrixKind, ExpertStreamingPlanner, ExpertStreamingPolicy,
    ExpertStreamingReader, ExpertTensorComponent, ExpertTensorKey, ExpertTensorPayload,
    ExpertTensorSlice,
};
use crate::TensorRole;

#[test]
fn attention_shape_contract_accepts_official_dimensions() {
    let cfg = official_tiny_cfg();
    let payload = attention_payload_for_cfg(cfg);
    let attention = DeepSeekV4Attention::new(0, cfg, payload).unwrap();
    assert_eq!(attention.config.output_group_input_dim(), 4096);
    assert_eq!(attention.config.output_latent_dim(), 8192);
}

#[test]
fn artifact_linear_cache_key_distinguishes_same_named_slices() {
    let first = f32_linear(TensorRole::OutputHead, "head", 2, 2);
    let mut second = first.clone();
    second.weight.slice.offset += 8;
    second.weight.slice.bytes = 8;

    assert_ne!(
        artifact_linear_cache_key(&first),
        artifact_linear_cache_key(&second)
    );
}

#[test]
fn non_compressed_attention_decode_runs_grouped_output_projection() {
    let cfg = DeepSeekV4AttentionConfig {
        hidden_size: 4,
        num_heads: 2,
        head_dim: 2,
        q_lora_rank: 4,
        rope_head_dim: 2,
        o_groups: 1,
        o_lora_rank: 4,
        window_size: 4,
        compress_ratio: 0,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
        compress_rope_theta: 160000.0,
        original_seq_len: 0,
        rope_factor: 1.0,
        beta_fast: 32,
        beta_slow: 1,
        index_n_heads: 2,
        index_head_dim: 2,
        index_topk: 4,
    };
    let payload = attention_payload_for_small_cfg(cfg);
    let attention = DeepSeekV4Attention::new(0, cfg, payload).unwrap();
    let mut cache = DeepSeekV4WindowKvCache::new(cfg.window_size, cfg.head_dim);
    let out = attention
        .decode_step_no_compress(&mut cache, &[1.0, 0.0, 0.0, 0.0], 0)
        .unwrap();
    assert_eq!(out.len(), 4);
    assert_eq!(cache.len(), 1);
    assert!(out.iter().all(|value| value.is_finite()));
}

#[test]
fn compressed_attention_decode_reference_updates_compressed_cache() {
    let cfg = DeepSeekV4AttentionConfig {
        hidden_size: 4,
        num_heads: 2,
        head_dim: 2,
        q_lora_rank: 4,
        rope_head_dim: 0,
        o_groups: 1,
        o_lora_rank: 4,
        window_size: 4,
        compress_ratio: 2,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
        compress_rope_theta: 160000.0,
        original_seq_len: 0,
        rope_factor: 1.0,
        beta_fast: 32,
        beta_slow: 1,
        index_n_heads: 2,
        index_head_dim: 2,
        index_topk: 4,
    };
    let payload = attention_payload_for_small_cfg(cfg);
    let compressed = DeepSeekV4CompressedAttentionPayload {
        compressor: tiny_compressor_payload(cfg.compress_ratio, cfg.hidden_size, cfg.head_dim),
        indexer: None,
    };
    let attention =
        DeepSeekV4Attention::new_with_compressed(2, cfg, payload, Some(compressed)).unwrap();
    let mut cache = DeepSeekV4AttentionCache::new(cfg);
    let first = attention
        .decode_step_reference(&mut cache, &[1.0, 0.0, 0.0, 0.0], 0)
        .unwrap();
    assert_eq!(first.len(), 4);
    assert_eq!(cache.compressed_len(), 0);
    let second = attention
        .decode_step_reference(&mut cache, &[0.0, 1.0, 0.0, 0.0], 1)
        .unwrap();
    assert_eq!(second.len(), 4);
    assert_eq!(cache.compressed_len(), 1);
    assert!(second.iter().all(|value| value.is_finite()));
}

#[test]
fn window_kv_indices_follow_official_ring_order_after_wrap() {
    let mut cache = DeepSeekV4WindowKvCache::new(4, 1);
    for pos in 0..6 {
        cache.append(pos, &[pos as f32]).unwrap();
    }
    assert_eq!(cache.topk_indices(5, 4), vec![2, 3, 0, 1]);
}

#[test]
fn official_prefill_topk_helpers_mask_future_positions() {
    assert_eq!(
        window_topk_indices_prefill(4, 5),
        vec![
            0, -1, -1, -1, // token 0
            0, 1, -1, -1, // token 1
            0, 1, 2, -1, // token 2
            0, 1, 2, 3, // token 3
            1, 2, 3, 4, // token 4
        ]
    );
    let (compressed, cols) = compress_topk_indices_prefill(4, 5, 5);
    assert_eq!(cols, 1);
    assert_eq!(compressed, vec![-1, -1, -1, 5, 5]);
}

#[test]
fn dsv4_layer_decode_step_runs_hc_attention_moe_shared_hc() {
    let dir = unique_temp_dir("ferrule-dsv4-layer-decode");
    std::fs::create_dir_all(&dir).unwrap();

    let hc_config = HyperConnectionConfig {
        hc_mult: 2,
        hidden_size: 32,
        sinkhorn_iters: 3,
        eps: 1e-6,
        norm_eps: 1e-6,
    };
    let attention_cfg = DeepSeekV4AttentionConfig {
        hidden_size: 32,
        num_heads: 1,
        head_dim: 32,
        q_lora_rank: 4,
        rope_head_dim: 0,
        o_groups: 1,
        o_lora_rank: 4,
        window_size: 4,
        compress_ratio: 0,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
        compress_rope_theta: 160000.0,
        original_seq_len: 0,
        rope_factor: 1.0,
        beta_fast: 32,
        beta_slow: 1,
        index_n_heads: 2,
        index_head_dim: 2,
        index_topk: 4,
    };
    let layer = DeepSeekV4Layer {
        layer: 0,
        hc_config,
        attn_norm: vec![1.0; 32],
        ffn_norm: vec![1.0; 32],
        attention: DeepSeekV4Attention::new(
            0,
            attention_cfg,
            attention_payload_for_vertical_cfg(attention_cfg),
        )
        .unwrap(),
        hc_attention: zero_hc_weights(hc_config),
        hc_feed_forward: zero_hc_weights(hc_config),
        router: RouterArtifactPayload {
            layer: 0,
            weight: f32_linear(TensorRole::RouterLogits, "router", 1, 32),
            bias: None,
            hash_table: Some(vec![0]),
            hash_rows: 1,
            hash_cols: 1,
        },
        shared_ffn: tiny_shared_ffn_32(),
        router_policy: ExpertRouterPolicy::sqrt_softplus_hash(1, 1.0),
    };

    let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(1));
    register_tiny_expert(&dir, &mut planner, 0, 0, 0x42, 0x43, 0x22);
    let mut expert_runtime = DeepSeekV4LayerExpertRuntime::new(planner);
    let mut state = DeepSeekV4LayerState::new(attention_cfg);
    let mut hc_state = vec![0.0f32; hc_config.hc_hidden_size()];
    hc_state[0] = 2.0;
    hc_state[33] = 3.0;

    let output = layer
        .decode_step_reference(
            &mut state,
            &mut expert_runtime,
            &hc_state,
            0,
            0,
            &[],
            &ExpertStreamingReader::new(4096),
            &CpuReferenceExpertExecutor::new(10.0),
        )
        .unwrap();
    assert_eq!(output.attention_hidden.len(), 32);
    assert_eq!(output.feed_forward_hidden.len(), 32);
    assert_eq!(output.hc_state.len(), hc_config.hc_hidden_size());
    assert_eq!(output.moe.routes.len(), 1);
    assert_eq!(output.moe.routes[0].expert, 0);
    assert_eq!(state.kv.len(), 1);
    assert!(output.hc_state.iter().all(|value| value.is_finite()));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dsv4_arena_shapes_deduplicate_43_layers_without_merging_compressor_variants() {
    let mut layers = Vec::with_capacity(43);
    for layer in 0..20 {
        layers.push(arena_shape_test_layer(layer, 0, None));
    }
    for layer in 20..30 {
        layers.push(arena_shape_test_layer(layer, 2, None));
    }
    for layer in 30..42 {
        layers.push(arena_shape_test_layer(layer, 4, Some(2)));
    }
    layers.push(arena_shape_test_layer(42, 4, Some(4)));

    let (layer_to_variant, representatives) = layer_arena_variant_layout(&layers);
    assert_eq!(layer_to_variant.len(), 43);
    assert_eq!(representatives.len(), 4);
    assert!(representatives.len() < layers.len());
    assert_eq!(layer_to_variant[0], layer_to_variant[19]);
    assert_ne!(layer_to_variant[19], layer_to_variant[20]);
    assert_ne!(layer_to_variant[29], layer_to_variant[30]);
    assert_ne!(layer_to_variant[41], layer_to_variant[42]);
}

#[test]
fn release_capacity_preserves_initialized_layer_slots() {
    let cfg = official_tiny_cfg();
    let mut state = DeepSeekV4LayerState::new(cfg);
    state.kv.compressed.reserve(128);
    let mut sequence = DeepSeekV4SequenceExecutionState::new(vec![state], 1);

    sequence.release_capacity();

    assert_eq!(sequence.layers.len(), 1);
    assert_eq!(sequence.layers[0].kv.compressed.capacity(), 0);
    assert!(sequence.layers[0].kv.is_empty());
}

fn official_tiny_cfg() -> DeepSeekV4AttentionConfig {
    DeepSeekV4AttentionConfig {
        hidden_size: deepseek_v4::HIDDEN_SIZE,
        num_heads: deepseek_v4::NUM_HEADS,
        head_dim: deepseek_v4::HEAD_DIM,
        q_lora_rank: deepseek_v4::Q_LORA_RANK,
        rope_head_dim: deepseek_v4::QK_ROPE_HEAD_DIM,
        o_groups: deepseek_v4::O_GROUPS,
        o_lora_rank: deepseek_v4::O_LORA_RANK,
        window_size: deepseek_v4::SLIDING_WINDOW,
        compress_ratio: 0,
        norm_eps: deepseek_v4::RMS_NORM_EPS,
        rope_theta: deepseek_v4::ROPE_THETA,
        compress_rope_theta: deepseek_v4::COMPRESS_ROPE_THETA,
        original_seq_len: deepseek_v4::ORIGINAL_MAX_POSITION_EMBEDDINGS,
        rope_factor: deepseek_v4::ROPE_FACTOR,
        beta_fast: deepseek_v4::ROPE_BETA_FAST,
        beta_slow: deepseek_v4::ROPE_BETA_SLOW,
        index_n_heads: deepseek_v4::INDEX_N_HEADS,
        index_head_dim: deepseek_v4::INDEX_HEAD_DIM,
        index_topk: deepseek_v4::INDEX_TOPK,
    }
}

fn attention_payload_for_cfg(cfg: DeepSeekV4AttentionConfig) -> MlaAttentionArtifactPayload {
    MlaAttentionArtifactPayload {
        layer: 0,
        query_a: f32_linear(
            TensorRole::AttentionLatentQueryA,
            "wq_a",
            cfg.q_lora_rank,
            cfg.hidden_size,
        ),
        query_b: f32_linear(
            TensorRole::AttentionLatentQueryB,
            "wq_b",
            cfg.q_full_dim(),
            cfg.q_lora_rank,
        ),
        key_value: f32_linear(
            TensorRole::AttentionLatentKv,
            "wkv",
            cfg.head_dim,
            cfg.hidden_size,
        ),
        output_a: f32_linear(
            TensorRole::AttentionLatentOutputA,
            "wo_a",
            cfg.output_latent_dim(),
            cfg.output_group_input_dim(),
        ),
        output_b: f32_linear(
            TensorRole::AttentionLatentOutputB,
            "wo_b",
            cfg.hidden_size,
            cfg.output_latent_dim(),
        ),
        query_norm: vec![1.0; cfg.q_lora_rank],
        key_value_norm: vec![1.0; cfg.head_dim],
        attention_sink: vec![0.0; cfg.num_heads],
        auxiliary: Vec::new(),
    }
}

fn attention_payload_for_small_cfg(cfg: DeepSeekV4AttentionConfig) -> MlaAttentionArtifactPayload {
    MlaAttentionArtifactPayload {
        layer: 0,
        query_a: identity_linear(TensorRole::AttentionLatentQueryA, "wq_a", 4),
        query_b: identity_linear(TensorRole::AttentionLatentQueryB, "wq_b", 4),
        key_value: f32_linear_values(
            TensorRole::AttentionLatentKv,
            "wkv",
            2,
            4,
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ),
        output_a: identity_linear(TensorRole::AttentionLatentOutputA, "wo_a", 4),
        output_b: identity_linear(TensorRole::AttentionLatentOutputB, "wo_b", 4),
        query_norm: vec![1.0; cfg.q_lora_rank],
        key_value_norm: vec![1.0; cfg.head_dim],
        attention_sink: vec![0.0; cfg.num_heads],
        auxiliary: Vec::new(),
    }
}

fn attention_payload_for_vertical_cfg(
    cfg: DeepSeekV4AttentionConfig,
) -> MlaAttentionArtifactPayload {
    MlaAttentionArtifactPayload {
        layer: 0,
        query_a: f32_linear_values(
            TensorRole::AttentionLatentQueryA,
            "wq_a",
            cfg.q_lora_rank,
            cfg.hidden_size,
            &one_hot_rows(
                cfg.q_lora_rank,
                cfg.hidden_size,
                &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
            ),
        ),
        query_b: f32_linear_values(
            TensorRole::AttentionLatentQueryB,
            "wq_b",
            cfg.q_full_dim(),
            cfg.q_lora_rank,
            &one_hot_rows(
                cfg.q_full_dim(),
                cfg.q_lora_rank,
                &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
            ),
        ),
        key_value: identity_linear(TensorRole::AttentionLatentKv, "wkv", cfg.head_dim),
        output_a: f32_linear_values(
            TensorRole::AttentionLatentOutputA,
            "wo_a",
            cfg.output_latent_dim(),
            cfg.output_group_input_dim(),
            &one_hot_rows(
                cfg.output_latent_dim(),
                cfg.output_group_input_dim(),
                &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
            ),
        ),
        output_b: f32_linear_values(
            TensorRole::AttentionLatentOutputB,
            "wo_b",
            cfg.hidden_size,
            cfg.output_latent_dim(),
            &one_hot_rows(
                cfg.hidden_size,
                cfg.output_latent_dim(),
                &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
            ),
        ),
        query_norm: vec![1.0; cfg.q_lora_rank],
        key_value_norm: vec![1.0; cfg.head_dim],
        attention_sink: vec![0.0; cfg.num_heads],
        auxiliary: Vec::new(),
    }
}

fn identity_linear(role: TensorRole, name: &str, dim: usize) -> ArtifactLinearPayload {
    let mut values = vec![0.0; dim * dim];
    for i in 0..dim {
        values[i * dim + i] = 1.0;
    }
    f32_linear_values(role, name, dim, dim, &values)
}

fn f32_linear(role: TensorRole, name: &str, out: usize, input: usize) -> ArtifactLinearPayload {
    f32_linear_values(role, name, out, input, &vec![0.0; out * input])
}

fn f32_linear_values(
    role: TensorRole,
    name: &str,
    out: usize,
    input: usize,
    values: &[f32],
) -> ArtifactLinearPayload {
    assert_eq!(values.len(), out * input);
    ArtifactLinearPayload::from_weight_and_scale(
        role,
        ArtifactTensorPayload {
            slice: ArtifactTensorSlice {
                name: format!("{name}.weight"),
                role: TensorRole::Unknown,
                path: PathBuf::from("synthetic.safetensors"),
                offset: 0,
                bytes: (values.len() * 4) as u64,
                dtype: ArtifactDType::F32,
                shape: vec![out, input],
            },
            bytes: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        },
        None,
    )
    .unwrap()
}

fn tiny_compressor_payload(
    ratio: usize,
    hidden_size: usize,
    head_dim: usize,
) -> DeepSeekV4CompressorPayload {
    DeepSeekV4CompressorPayload {
        compress_ratio: ratio,
        head_dim,
        overlap: false,
        rotate_for_indexer: false,
        ape: vec![0.0; ratio * head_dim],
        ape_rows: ratio,
        ape_cols: head_dim,
        norm: vec![1.0; head_dim],
        wkv: f32_linear_values(
            TensorRole::AttentionCompressor,
            "compressor.wkv",
            head_dim,
            hidden_size,
            &one_hot_rows(head_dim, hidden_size, &[(0, 0, 1.0), (1, 1, 1.0)]),
        ),
        wgate: f32_linear_values(
            TensorRole::AttentionCompressor,
            "compressor.wgate",
            head_dim,
            hidden_size,
            &vec![0.0; head_dim * hidden_size],
        ),
    }
}

fn arena_shape_test_layer(
    layer: usize,
    compress_ratio: usize,
    index_head_dim: Option<usize>,
) -> DeepSeekV4Layer {
    let hc_config = HyperConnectionConfig {
        hc_mult: 2,
        hidden_size: 32,
        sinkhorn_iters: 3,
        eps: 1e-6,
        norm_eps: 1e-6,
    };
    let cfg = DeepSeekV4AttentionConfig {
        hidden_size: 32,
        num_heads: 1,
        head_dim: 32,
        q_lora_rank: 4,
        rope_head_dim: 0,
        o_groups: 1,
        o_lora_rank: 4,
        window_size: 4,
        compress_ratio,
        norm_eps: 1e-6,
        rope_theta: 10_000.0,
        compress_rope_theta: 160_000.0,
        original_seq_len: 0,
        rope_factor: 1.0,
        beta_fast: 32,
        beta_slow: 1,
        index_n_heads: 2,
        index_head_dim: index_head_dim.unwrap_or(2),
        index_topk: 4,
    };
    let compressed = (compress_ratio != 0).then(|| DeepSeekV4CompressedAttentionPayload {
        compressor: tiny_compressor_payload(compress_ratio, cfg.hidden_size, cfg.head_dim),
        indexer: index_head_dim.map(|head_dim| {
            let mut compressor = tiny_compressor_payload(compress_ratio, cfg.hidden_size, head_dim);
            compressor.rotate_for_indexer = true;
            DeepSeekV4IndexerPayload {
                compressor,
                wq_b: f32_linear(
                    TensorRole::AuxIndexer,
                    "indexer.wq_b",
                    cfg.index_n_heads * head_dim,
                    cfg.q_lora_rank,
                ),
                weights_proj: f32_linear(
                    TensorRole::AuxIndexer,
                    "indexer.weights_proj",
                    cfg.index_n_heads,
                    cfg.hidden_size,
                ),
            }
        }),
    });
    DeepSeekV4Layer {
        layer,
        hc_config,
        attn_norm: vec![1.0; cfg.hidden_size],
        ffn_norm: vec![1.0; cfg.hidden_size],
        attention: DeepSeekV4Attention::new_with_compressed(
            layer,
            cfg,
            attention_payload_for_vertical_cfg(cfg),
            compressed,
        )
        .unwrap(),
        hc_attention: zero_hc_weights(hc_config),
        hc_feed_forward: zero_hc_weights(hc_config),
        router: RouterArtifactPayload {
            layer,
            weight: f32_linear(TensorRole::RouterLogits, "router", 256, cfg.hidden_size),
            bias: None,
            hash_table: None,
            hash_rows: 0,
            hash_cols: 0,
        },
        shared_ffn: tiny_shared_ffn_32(),
        router_policy: ExpertRouterPolicy::sqrt_softplus_score_topk(6, 1.0),
    }
}

fn tiny_shared_ffn_32() -> SwiGluFfnPayload {
    SwiGluFfnPayload {
        gate: f32_linear_values(
            TensorRole::SharedExpertGate,
            "shared_gate",
            1,
            32,
            &one_hot_rows(1, 32, &[(0, 0, 1.0)]),
        ),
        up: f32_linear_values(
            TensorRole::SharedExpertUp,
            "shared_up",
            1,
            32,
            &one_hot_rows(1, 32, &[(0, 1, 1.0)]),
        ),
        down: f32_linear_values(
            TensorRole::SharedExpertDown,
            "shared_down",
            32,
            1,
            &one_hot_rows(32, 1, &[(0, 0, 1.0)]),
        ),
        swiglu_limit: 10.0,
    }
}

fn zero_hc_weights(config: HyperConnectionConfig) -> HyperConnectionWeights {
    HyperConnectionWeights {
        function: vec![0.0; config.mix_hc() * config.hc_hidden_size()],
        scale: vec![1.0, 1.0, 1.0],
        base: vec![0.0; config.mix_hc()],
    }
}

fn one_hot_rows(rows: usize, cols: usize, entries: &[(usize, usize, f32)]) -> Vec<f32> {
    let mut values = vec![0.0f32; rows * cols];
    for &(row, col, value) in entries {
        values[row * cols + col] = value;
    }
    values
}

fn register_tiny_expert(
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
    let tensors = tiny_expert_tensors(expert_id, &path, gate_byte, up_byte, down_byte);
    let mut bytes = Vec::new();
    for tensor in &tensors {
        bytes.extend(&tensor.bytes);
    }
    std::fs::write(&path, bytes).unwrap();
    let mut offset = 0u64;
    let slices = tensors
        .into_iter()
        .map(|tensor| {
            let bytes = tensor.bytes.len() as u64;
            let slice = ExpertTensorSlice {
                offset,
                bytes,
                ..tensor.slice
            };
            offset += bytes;
            slice
        })
        .collect();
    planner.register_load_source(
        expert_id,
        ExpertLoadSource::LocalTensorSet { tensors: slices },
    );
}

fn tiny_expert_tensors(
    expert: ExpertId,
    path: &Path,
    gate_byte: u8,
    up_byte: u8,
    down_byte: u8,
) -> Vec<ExpertTensorPayload> {
    vec![
        tiny_fp4_payload(expert, path, ExpertMatrixKind::Gate, gate_byte),
        tiny_scale_payload(expert, path, ExpertMatrixKind::Gate),
        tiny_fp4_payload(expert, path, ExpertMatrixKind::Up, up_byte),
        tiny_scale_payload(expert, path, ExpertMatrixKind::Up),
        tiny_fp4_payload(expert, path, ExpertMatrixKind::Down, down_byte),
        tiny_scale_payload(expert, path, ExpertMatrixKind::Down),
    ]
}

fn tiny_fp4_payload(
    expert: ExpertId,
    path: &Path,
    matrix: ExpertMatrixKind,
    first_byte: u8,
) -> ExpertTensorPayload {
    let mut bytes = vec![0u8; 32 * 16];
    bytes[0] = first_byte;
    ExpertTensorPayload {
        slice: ExpertTensorSlice {
            key: ExpertTensorKey { expert, matrix },
            component: ExpertTensorComponent::Weight,
            path: path.to_path_buf(),
            offset: 0,
            bytes: bytes.len() as u64,
            dtype: "I8".into(),
            shape: vec![32, 16],
        },
        bytes,
    }
}

fn tiny_scale_payload(
    expert: ExpertId,
    path: &Path,
    matrix: ExpertMatrixKind,
) -> ExpertTensorPayload {
    let bytes = vec![127u8; 32];
    ExpertTensorPayload {
        slice: ExpertTensorSlice {
            key: ExpertTensorKey { expert, matrix },
            component: ExpertTensorComponent::Scale,
            path: path.to_path_buf(),
            offset: 0,
            bytes: bytes.len() as u64,
            dtype: "F8_E8M0".into(),
            shape: vec![32, 1],
        },
        bytes,
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nonce}"))
}
