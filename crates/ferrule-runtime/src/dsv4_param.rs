//! DSV4 model config derived from official config.json.
//!
//! Synthetic fixtures use these constants so shapes and policy defaults
//! match the official model.

use crate::attention_backend::SparseAttentionSpec;
use crate::expert_routing::ExpertRouterPolicy;
use crate::hyper_connection::HyperConnectionConfig;

pub const HIDDEN_SIZE: usize = 4096;
pub const HC_MULT: usize = 4;
pub const HC_SINKHORN_ITERS: usize = 20;
pub const HC_EPS: f32 = 1e-6;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const NUM_LAYERS: usize = 43;
pub const NUM_HASH_LAYERS: usize = 3;
pub const N_ROUTED_EXPERTS: usize = 256;
pub const NUM_EXPERTS_PER_TOK: usize = 6;
pub const MOE_INTERMEDIATE_SIZE: usize = 2048;
pub const HEAD_DIM: usize = 512;
pub const NUM_HEADS: usize = 64;
pub const Q_LORA_RANK: usize = 1024;
pub const O_LORA_RANK: usize = 1024;
pub const SLIDING_WINDOW: usize = 128;
pub const VOCAB_SIZE: usize = 129280;
pub const SWIGLU_LIMIT: f32 = 10.0;
pub const ROUTED_SCALING_FACTOR: f32 = 1.5;
pub const DSPARK_BLOCK_SIZE: usize = 5;

pub fn is_hash_layer(layer: usize) -> bool {
    layer < NUM_HASH_LAYERS
}

pub fn router_policy_for_layer(layer: usize) -> ExpertRouterPolicy {
    if is_hash_layer(layer) {
        ExpertRouterPolicy::deepseek_v4_hash(NUM_EXPERTS_PER_TOK, ROUTED_SCALING_FACTOR)
    } else {
        ExpertRouterPolicy::deepseek_v4_score_topk(NUM_EXPERTS_PER_TOK, ROUTED_SCALING_FACTOR)
    }
}

pub fn hc_config() -> HyperConnectionConfig {
    HyperConnectionConfig {
        hc_mult: HC_MULT,
        hidden_size: HIDDEN_SIZE,
        sinkhorn_iters: HC_SINKHORN_ITERS,
        eps: HC_EPS,
        norm_eps: RMS_NORM_EPS,
    }
}

pub fn attention_spec() -> SparseAttentionSpec {
    SparseAttentionSpec {
        heads: NUM_HEADS,
        head_dim: HEAD_DIM,
        topk: SLIDING_WINDOW,
        softmax_scale: (HEAD_DIM as f32).powf(-0.5),
        has_attention_sink: true,
    }
}
