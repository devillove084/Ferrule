//! Model-side execution state, dynamic bindings, and reusable arena infrastructure.
//!
//! These types are model-family neutral. Runtime request correlation and backend-
//! specific resources remain outside this module.

mod arena;
mod backend;
mod binding;

mod plan;
mod precision;
mod resource;
mod sequence;
mod stage;

pub use arena::{ArenaLease, OwnedArenaCheckout, PersistentArenaPool, PersistentArenaPoolStats};
pub use backend::ModelExecutionBackend;
pub use binding::ExecutionShapeKey;

pub use plan::PreparedModel;
pub use precision::{
    BoundaryPrecision, ExecutionPrecisionBoundary, ExecutionPrecisionPolicy, bf16_rne,
    bf16_rne_word, bf16_word_value,
};
pub use resource::{ExecutionPlanError, ResourceBacking, ResourceLayout, ResourceManifest};

pub use sequence::{SequenceStateCore, SequenceStepBinding, SequenceTopologyId};
pub use stage::{
    ExecutableStage, MaterializedStage, PreparedExecutable, ResolvedStage, ResolvedStageResource,
    ResourceAccess, ResourceRetention, StageMaterializationRequest, StageResourceUse,
    TransformerResourceSlot, TransformerStage, WorkspaceClaim,
};
