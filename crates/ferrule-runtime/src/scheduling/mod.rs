//! Request scheduling and session lifecycle.
//!
//! - `actions`: executable scheduling action vocabulary and planning helpers.
//! - `batch`: runtime-private lowering to the neutral execution ABI.
//! - `resident`: resident request/session scheduler over logical sequence slots.
//! - `session`: sequence state, request IDs, session management.

pub mod actions;
pub(crate) mod batch;
pub mod expert_io;
pub mod resident;
pub mod session;
pub mod slot_pool;

pub use actions::{
    DEFAULT_CHUNK_SIZE, DecodeAction, LogitsSelection, PrefillChunkAction, SchedulerAction,
    plan_prefill_chunk,
};
pub(crate) use batch::ScheduledBatch;
pub(crate) use expert_io::ModelExpertIoAdvisor;
pub use expert_io::{
    ExpertIoAdvisor, ExpertIoBudget, ExpertIoCandidate, ExpertIoDecisionTrace, ExpertIoEstimate,
    ExpertIoPhase, ExpertIoQueueClass, ExpertIoRejection, ZeroExpertIoAdvisor,
};
pub use resident::{CancelRequestResult, ResidentScheduler, ResidentSchedulerConfig};
pub use session::{
    GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SequenceStatus, SessionId,
};
pub use slot_pool::{FixedSequenceSlotPool, KvHandle, SequenceSlotPool};
