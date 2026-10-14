//! KV layouts, page residency, and compression operators.

pub mod committed_gqa;
pub mod compressor;
pub use committed_gqa as host_placement;
pub mod page_pool;
mod read_protocol;
pub mod single_page_placement;
pub mod transaction;
pub use read_protocol::committed_gqa_credits;
pub use transaction::{
    CudaBf16GqaPlanes, CudaBf16GqaRows, CudaF32GqaPlanes, CudaKvView, CudaPagedKvPool,
    CudaPagedKvTransaction,
};
pub use transaction::{committed_bf16_kv_trace_host_bytes, CudaCommittedBf16KvChunk, CudaCommittedRead, CudaKvPhysicalStats, CudaKvReclaim};

#[doc(hidden)]
pub mod layout {
    pub use super::PagedPlaneLayout;
}

pub use crate::cuda::operator_owner::{
    CudaBf16Buffer, CudaCompressorRecurrentCheckpointSlab, CudaCompressorRecurrentState,
    CudaF32Buffer, CudaI32Buffer, CudaOperators,
};
pub use page_pool::{
    CudaKvPagePool, CudaKvPlaneStorage, KvHostPlaneSnapshot, KvPagePoolStats, KvPoolReservation,
    PagedPlaneLayout, TypedKvHostSnapshot, TypedPagedPlaneLayout,
};
