//! Model-side execution state, dynamic bindings, and reusable arena infrastructure.
//!
//! These types are model-family neutral. Runtime request correlation and backend-
//! specific resources remain outside this module.

mod arena;
mod backend;
mod binding;
mod plan;
mod resource;
mod sequence;
mod stage;

pub use arena::{ArenaLease, OwnedArenaCheckout, PersistentArenaPool, PersistentArenaPoolStats};
pub use backend::ModelExecutionBackend;
pub use binding::ExecutionShapeKey;
pub use plan::PreparedModel;
pub use resource::{ExecutionPlanError, ResourceBacking, ResourceLayout, ResourceManifest};
pub use sequence::{SequenceStateCore, SequenceStepBinding};
pub use stage::{
    ExecutableStage, MaterializedStage, PreparedExecutable, ResourceAccess, ResourceRetention,
    StageMaterializationRequest, StageResourceUse, TransformerResourceSlot, TransformerStage,
    WorkspaceClaim,
};
