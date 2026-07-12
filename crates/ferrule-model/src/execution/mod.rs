//! Model-side execution state, dynamic bindings, and reusable arena infrastructure.
//!
//! These types are model-family neutral. Runtime request correlation and backend-
//! specific resources remain outside this module.

mod arena;
mod backend;
mod binding;
mod plan;
mod sequence;

pub use arena::{ArenaLease, PersistentArenaPool, PersistentArenaPoolStats};
pub use backend::ModelExecutionBackend;
pub use binding::{ExecutionShapeKey, PreparedStepBinding};
pub use plan::PreparedModel;
pub use sequence::{SequenceStateCore, SequenceStepBinding};
