//! Model-independent transformer connection components.

pub mod hyper;

pub use hyper::{
    HyperConnection, HyperConnectionConfig, HyperConnectionHead, HyperConnectionHeadWeights,
    HyperConnectionPhase, HyperConnectionPreOutput, HyperConnectionSplit, HyperConnectionStage,
    HyperConnectionWeights, PreparedHyperConnection, PreparedHyperConnectionHead,
    PreparedHyperConnectionWeights, hc_head_reference, hc_post_reference, hc_pre_reference,
    hc_split_sinkhorn_reference,
};
#[cfg(feature = "cuda")]
pub use hyper::{HyperConnectionPostBuffers, HyperConnectionPreBuffers};
