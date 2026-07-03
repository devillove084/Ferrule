//! Ferrule graph dialect op-key helpers.
//!
//! These helpers centralize stable opaque op names without turning graph nodes into
//! Rust enum variants or exposing public per-op backend methods.

use ferrule_core::Result;
use ferrule_graph::OpKey;

pub mod domain {
    pub const TENSOR: &str = "ferrule.tensor";
    pub const TRANSFORMER: &str = "ferrule.transformer";
    pub const STATE: &str = "ferrule.state";
}

pub mod tensor_ops {
    use super::{op, Result};
    use ferrule_graph::OpKey;

    pub fn residual_add() -> Result<OpKey> {
        op(super::domain::TENSOR, "residual_add", 1)
    }
}

pub mod transformer_ops {
    use super::{op, Result};
    use ferrule_graph::OpKey;

    pub fn token_embedding() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "token_embedding", 1)
    }

    pub fn rms_norm() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "rms_norm", 1)
    }

    pub fn linear() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "linear", 1)
    }

    pub fn rope() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "rope", 1)
    }

    pub fn causal_attention() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "causal_attention", 1)
    }

    pub fn swiglu_ffn() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "swiglu_ffn", 1)
    }

    pub fn logits_select() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "logits_select", 1)
    }

    pub fn transformer_state_init() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "transformer_state_init", 1)
    }

    pub fn transformer_layer() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "transformer_layer", 1)
    }

    pub fn output_projection() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "output_projection", 1)
    }

    pub fn hyper_connection_pre() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "hyper_connection_pre", 1)
    }

    pub fn hyper_connection_post() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "hyper_connection_post", 1)
    }

    pub fn latent_attention() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "latent_attention", 1)
    }

    pub fn routed_moe() -> Result<OpKey> {
        op(super::domain::TRANSFORMER, "routed_moe", 1)
    }
}

pub mod state_ops {
    use super::{op, Result};
    use ferrule_graph::OpKey;

    pub fn kv_state() -> Result<OpKey> {
        op(super::domain::STATE, "kv_state", 1)
    }
}

fn op(domain: &str, name: &str, version: u32) -> Result<OpKey> {
    OpKey::new(domain, name, version)
}
