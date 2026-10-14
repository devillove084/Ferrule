//! Driver-free checks exercise the same admission/mapping/Unknown code as Entry.
mod cpu {
    pub use ferrule_backend::cpu::{PagedKvBatch, PagedKvPrepare};
}
#[path = "chunked_kv/cpu_protocol.rs"]
mod cpu_protocol;
#[cfg(feature = "cuda")]
#[path = "chunked_kv/cuda_chunked.rs"]
mod cuda_chunked;
#[allow(dead_code)]
#[path = "../src/cuda/operators/kv/transaction/plan.rs"]
mod plan;
#[path = "../src/cuda/operators/kv/transaction/chunked_protocol.rs"]
mod protocol;
