//! Explicit allocation-only experiment; NOT a paged-GQA storage mode.
//!
//! Independent pages cannot be passed to the fixed-plane base+slot kernels.
//! There is deliberately no view/launch API here. For the integrated physical
//! release path, use `CudaPagedKvPool::new_chunked`: it retains chunks under the
//! existing Entry journal and exposes only bounded staging to attention. This
//! standalone allocation contract is not the chunked pool's ownership ledger.

use crate::cuda::operator_owner::{CudaComputeEvent, CudaF32Buffer, CudaOperators};
use crate::physical_placement::{
    PageAllocation, PageAllocationDomain, PageAllocationKind, PageCompletionFence,
    PagePhysicalPlacement,
};
use ferrule_common::execution::KvPageId;
use ferrule_common::{Error, Result};
use std::rc::Rc;

/// Opt-in façade over the existing canonical allocator (CudaDeviceBlock through
/// CudaF32Buffer). It creates no page manager, mapping table, registry or kernel ABI.
/// One façade defines an owner-local domain; different façades cannot be composed.
/// Merely constructing it allocates no device memory.
pub struct CudaSinglePagePlacementAllocator {
    ops: Rc<CudaOperators>,
    domain: PageAllocationDomain,
}
impl CudaSinglePagePlacementAllocator {
    pub fn new(ops: Rc<CudaOperators>) -> Self {
        Self {
            ops,
            domain: PageAllocationDomain::new(),
        }
    }

    /// Allocate one F32 page/plane, including its layers. The caller must obtain
    /// budget admission first. This does NOT install it in CudaPagedKvPool.
    pub fn allocate_f32(
        &self,
        page: KvPageId,
        plane: usize,
        elements: usize,
    ) -> Result<PagePhysicalPlacement<CudaF32Buffer>> {
        let invalid = || Error::Execution {
            message: "single-page allocation size overflow or empty page".into(),
        };
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .filter(|n| *n != 0)
            .ok_or_else(invalid)?;
        let granted = crate::cuda_allocation_grant_bytes(bytes).ok_or_else(invalid)?;
        let buffer = self.ops.zero_f32_buffer(elements)?;
        let base = buffer.as_device_buffer().cu_deviceptr();
        // Fresh root allocation (never a prefix/slice); zero_f32_buffer uses the
        // canonical device allocator and retains its context and retirement rules.
        let allocation = unsafe {
            PageAllocation::from_backing(
                buffer,
                self.domain.clone(),
                base,
                bytes,
                granted,
                PageAllocationKind::SinglePage,
            )?
        };
        allocation.placement(page, plane, 0..bytes, size_of::<f32>())
    }

    /// Context-wide telemetry, NOT per-page attributed savings. In particular,
    /// live_granted_bytes, pending_retirement_bytes, reserved_bytes and
    /// driver_free_bytes are distinct; dropping a page does not imply driver free.
    pub fn allocator_metrics(&self) -> crate::cuda::CudaAllocatorMetrics {
        self.ops.allocator_metrics()
    }
}

// Actual event query, never a client ACK. Binding the correct event to a specific
// submission/domain remains the unsafe PageCompletionToken constructor's duty.
// A compute event is not DMA/full-drain evidence without exact cross-stream
// dependencies; there is deliberately no safe event-to-token conversion here.
unsafe impl PageCompletionFence for CudaComputeEvent {
    fn is_complete(&self) -> Result<bool> {
        CudaComputeEvent::is_complete(self)
    }
}
