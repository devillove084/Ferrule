//! Async OpenAI-compatible serving for Ferrule.
//!
//! HTTP connections run on Tokio/Axum while one dedicated synchronous worker
//! owns each model driver. Bounded channels provide admission control and
//! per-request backpressure without placing the GPU execution path behind a
//! shared mutex.

mod config;
mod http;
mod openai;
mod worker;

pub use config::{ModelRegistration, WorkerConfig};
pub use ferrule_common::{
    ServingConfigError, ServingRequestError, SseSerializationError, WorkerExecutionError,
    WorkerOperation, WorkerRequestError, WorkerShutdownError, WorkerStartError,
};
pub use http::{ServerState, router, serve_with_shutdown};
pub use worker::{ModelWorker, ModelWorkerHandle, spawn_model_worker_with};
