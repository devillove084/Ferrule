//! KV layouts, page residency, and compression operators.

pub mod compressor;
pub mod page_pool;

#[doc(hidden)]
pub mod layout {
    pub use super::PagedPlaneLayout;
}

pub use crate::cuda::context::{
    CudaBf16Buffer, CudaCompressorRecurrentCheckpointSlab, CudaCompressorRecurrentState,
    CudaF32Buffer, CudaI32Buffer, CudaOperators,
};
pub use page_pool::{
    CudaKvPagePool, CudaKvPlaneStorage, KvHostPlaneSnapshot, KvPagePoolStats, KvPoolReservation,
    PagedPlaneLayout, TypedKvHostSnapshot, TypedPagedPlaneLayout,
};
