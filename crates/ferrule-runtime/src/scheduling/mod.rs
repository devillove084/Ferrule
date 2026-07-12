//! Request scheduling and session lifecycle.
//!
//! - `actions`: executable scheduling action vocabulary and planning helpers.
//! - `batch`: runtime-private lowering to the neutral execution ABI.
//! - `resident`: resident request/session scheduler over `SequenceKvCache`.
//! - `session`: sequence state, request IDs, session management.

pub mod actions;
pub(crate) mod batch;
pub mod resident;
pub mod session;

pub use actions::{
    plan_prefill_chunk, DecodeAction, LogitsSelection, PrefillChunkAction, SchedulerAction,
    DEFAULT_CHUNK_SIZE,
};
pub(crate) use batch::ScheduledBatch;
pub use resident::{ResidentScheduler, ResidentSchedulerConfig};
pub use session::{
    GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SequenceStatus, SessionId,
};
