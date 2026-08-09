#![cfg(test)]

#[cfg(feature = "cuda")]
mod cuda {
    use crate::transformer::cuda::*;

    use crate::transformer::attention::mla::*;
    use std::path::PathBuf;

    use crate::TensorRole;
    use crate::checkpoint::tensor::{
        CheckpointDType, CheckpointTensorPayload, CheckpointTensorSlice,
    };
    use crate::checkpoint::weight::LinearWeight;
    use crate::moe::RoutedMoePayload;
    use crate::moe::routing::RouterWeights;

    use crate::families::deepseek_v4;
    use crate::ffn::SwiGluFfnPayload;
    use crate::moe::routing::ExpertRouterPolicy;
    use crate::transformer::connection::{
        HyperConnection, HyperConnectionConfig, HyperConnectionPhase, HyperConnectionWeights,
    };

    #[test]
    fn attention_shape_contract_accepts_official_dimensions() {
        let cfg = official_tiny_cfg();
        let payload = attention_payload_for_cfg(cfg);
        let attention = PreparedMla::new(0, cfg, payload).unwrap();
        assert_eq!(attention.config().output_group_input_dim(), 4096);
        assert_eq!(attention.config().output_latent_dim(), 8192);
    }

    #[test]
    fn dsv4_arena_shapes_deduplicate_43_layers_without_merging_compressor_variants() {
        let mut layers = Vec::with_capacity(43);
        for layer in 0..20 {
            layers.push(arena_shape_test_layer(layer, 0, None));
        }
        for layer in 20..30 {
            layers.push(arena_shape_test_layer(layer, 128, None));
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

    fn official_tiny_cfg() -> MlaConfig {
        MlaConfig {
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

    fn attention_payload_for_cfg(cfg: MlaConfig) -> MlaWeights {
        MlaWeights {
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
        }
    }

    fn attention_payload_for_vertical_cfg(cfg: MlaConfig) -> MlaWeights {
        MlaWeights {
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
        }
    }

    fn identity_linear(role: TensorRole, name: &str, dim: usize) -> LinearWeight {
        let mut values = vec![0.0; dim * dim];
        for i in 0..dim {
            values[i * dim + i] = 1.0;
        }
        f32_linear_values(role, name, dim, dim, &values)
    }

    fn f32_linear(role: TensorRole, name: &str, out: usize, input: usize) -> LinearWeight {
        f32_linear_values(role, name, out, input, &vec![0.0; out * input])
    }

    fn f32_linear_values(
        role: TensorRole,
        name: &str,
        out: usize,
        input: usize,
        values: &[f32],
    ) -> LinearWeight {
        assert_eq!(values.len(), out * input);
        LinearWeight::from_weight_and_scale(
            role,
            CheckpointTensorPayload {
                slice: CheckpointTensorSlice {
                    name: format!("{name}.weight"),
                    role: TensorRole::Unknown,
                    path: PathBuf::from("synthetic.safetensors"),
                    offset: 0,
                    bytes: (values.len() * 4) as u64,
                    dtype: CheckpointDType::F32,
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

    fn tiny_compressor_payload(ratio: usize, hidden_size: usize, head_dim: usize) -> MlaCompressor {
        let overlap = ratio == 4;
        let out_dim = if overlap { 2 * head_dim } else { head_dim };
        MlaCompressor {
            compress_ratio: ratio,
            head_dim,
            overlap,
            rotate_for_indexer: false,
            ape: vec![0.0; ratio * out_dim],
            ape_rows: ratio,
            ape_cols: out_dim,
            norm: vec![1.0; head_dim],
            wkv: f32_linear_values(
                TensorRole::AttentionCompressor,
                "compressor.wkv",
                out_dim,
                hidden_size,
                &one_hot_rows(out_dim, hidden_size, &[(0, 0, 1.0), (1, 1, 1.0)]),
            ),
            wgate: f32_linear_values(
                TensorRole::AttentionCompressor,
                "compressor.wgate",
                out_dim,
                hidden_size,
                &vec![0.0; out_dim * hidden_size],
            ),
        }
    }

    fn arena_shape_test_layer(
        layer: usize,
        compress_ratio: usize,
        index_head_dim: Option<usize>,
    ) -> MlaHyperMoeLayer {
        let hc_config = HyperConnectionConfig {
            hc_mult: 2,
            hidden_size: 32,
            sinkhorn_iters: 3,
            eps: 1e-6,
            norm_eps: 1e-6,
        };
        let cfg = MlaConfig {
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
        let compressed = (compress_ratio != 0).then(|| MlaCompression {
            compressor: tiny_compressor_payload(compress_ratio, cfg.hidden_size, cfg.head_dim),
            indexer: index_head_dim.map(|head_dim| {
                let mut compressor =
                    tiny_compressor_payload(compress_ratio, cfg.hidden_size, head_dim);
                compressor.rotate_for_indexer = true;
                MlaIndexer {
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
        let hyper_connection = HyperConnection::new(
            hc_config,
            HyperConnectionPhase::new(zero_hc_weights(hc_config), vec![1.0; cfg.hidden_size]),
            HyperConnectionPhase::new(zero_hc_weights(hc_config), vec![1.0; cfg.hidden_size]),
        )
        .unwrap();
        MlaHyperMoeLayer {
            layer,
            hyper_connection,
            attention: PreparedMla::new_with_compressed(
                layer,
                cfg,
                attention_payload_for_vertical_cfg(cfg),
                compressed,
            )
            .unwrap(),
            feed_forward: RoutedMoePayload {
                router: RouterWeights {
                    layer,
                    weight: f32_linear(TensorRole::RouterLogits, "router", 256, cfg.hidden_size),
                    bias: None,
                    hash_table: None,
                    hash_rows: 0,
                    hash_cols: 0,
                },
                shared_expert: tiny_shared_ffn_32(),
                router_policy: ExpertRouterPolicy::sqrt_softplus_score_topk(6, 1.0),
            },
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
}

#[cfg(feature = "cuda")]
use std::path::Path;

use crate::nn::{ModulePathError, ParameterPart};

use super::DeepSeekV4Recipe;
#[cfg(feature = "cuda")]
use super::{DeepSeekV4Adapter, DeepSeekV4Checkpoint, DeepSeekV4PrepareOptions};

#[test]
fn name_mapper_preserves_module_path_error_source() {
    use std::error::Error as _;

    let error = super::name_mapper::mapping("invalid-path", ParameterPart::Weight)
        .expect_err("an invalid canonical path must fail");
    let source = error
        .source()
        .expect("name mapping errors must retain the canonical path source");

    assert!(source.downcast_ref::<ModulePathError>().is_some());
}

#[cfg(feature = "cuda")]
#[test]
fn prepare_options_keep_non_zero_production_defaults() {
    let options = DeepSeekV4PrepareOptions::default();
    assert_eq!(options.max_layers, crate::families::deepseek_v4::NUM_LAYERS);
    assert!(options.output_head_chunk_rows > 0);
    assert!(options.expert_reader_max_tensor_bytes > 0);
    assert!(options.reserved_device_bytes > 0);
}

#[test]
fn recipe_preserves_typed_config_error_source() {
    use std::error::Error as _;

    let error = crate::transformer::DecoderRecipe::build_spec(
        &DeepSeekV4Recipe::new(),
        &serde_json::json!([]),
    )
    .expect_err("non-object DeepSeek config must fail");
    let source = error
        .source()
        .expect("DeepSeek recipe config errors retain a typed source");
    assert!(source.to_string().contains("DeepSeek-V4 config"));
}

#[cfg(feature = "cuda")]
#[test]
fn checkpoint_rejects_a_zero_materialization_limit_before_io() {
    let error = DeepSeekV4Checkpoint::load_hf_with_limit(Path::new("missing"), 0)
        .err()
        .expect("zero parameter limit must be rejected");
    assert!(
        error
            .to_string()
            .contains("parameter limit must be positive")
    );

    let error = DeepSeekV4Adapter::load_hf_with_options(
        Path::new("missing"),
        0,
        DeepSeekV4PrepareOptions::default(),
    )
    .err()
    .expect("adapter must preserve checkpoint limit validation");
    assert!(
        error
            .to_string()
            .contains("parameter limit must be positive")
    );
}
