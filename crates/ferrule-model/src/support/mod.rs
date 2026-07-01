//! Generic model-support contract and engine-planning types.
//!
//! Keep model-specific tensor names at descriptor/binding boundaries. Runtime
//! executors should consume semantic roles and policies from this module.

pub mod binding;
pub mod contract;
pub mod layout;
pub mod plan;
pub mod policies;
pub mod roles;

pub use binding::{tensor_role_for_class, TensorBinding};
pub use contract::ModelSupportContract;
pub use layout::{AttentionLayout, FeedForwardLayout, LayerLayout, ModelLayout};
pub use plan::{EnginePlan, EnginePlanStatus, MissingPolicy, PolicyArea};
pub use policies::{
    AttentionPolicy, ExpertPolicy, KvPolicy, ParallelismPlan, PolicySet, QuantPolicy,
    ResidencyPolicy, RouterPolicy, SpeculationMode, SpeculationPolicy, TokenizerPolicy,
    ValidationPolicy,
};
pub use roles::{FeedForwardKind, KvCacheShape, TensorRole};

#[cfg(test)]
mod tests;
