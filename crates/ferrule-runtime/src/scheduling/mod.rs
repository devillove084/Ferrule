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
    DEFAULT_CHUNK_SIZE, DecodeAction, LogitsSelection, PrefillChunkAction, SchedulerAction,
    plan_prefill_chunk,
};
pub(crate) use batch::ScheduledBatch;
pub use resident::{CancelRequestResult, ResidentScheduler, ResidentSchedulerConfig};
pub use session::{
    GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SequenceStatus, SessionId,
};
