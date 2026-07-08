//! Request scheduling and session lifecycle.
//!
//! - `resident`: resident request/session scheduler over `SequenceKvCache`.
//! - `scheduler`: concrete scheduler actions, prefill/decode batching helpers, and legacy prototypes.
//! - `session`: sequence state, request IDs, session management.

pub mod resident;
pub mod scheduler;
pub mod session;

pub use resident::{ResidentScheduler, ResidentSchedulerConfig};
pub use scheduler::{
    plan_prefill_chunk, BatchedScheduler, DecodeAction, PreemptionPolicy, PrefillChunkAction,
    Scheduler, SchedulerAction,
};
pub use session::{
    GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SequenceStatus, SessionId,
};
