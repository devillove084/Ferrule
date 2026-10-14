//! Backend-private segmented CUDA device allocator.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, Weak};

#[cfg(test)]
use std::sync::Barrier;

use super::runtime::{
    CudaContext, CudaError, CudaResult, CudaRetirementEvent, DevicePtr, driver_alloc, driver_free,
};

use crate::cuda_allocation_layout::{
    ALLOCATOR_PAGE_BYTES, MIN_ALIGNMENT, SMALL_LIMIT, cuda_allocation_grant_bytes,
};
const LARGE_LIMIT: usize = 16 << 20;
const SEGMENT_BYTES: usize = 64 << 20;
const RETIREMENT_BATCH_BLOCKS: usize = 64;
const RETIREMENT_BATCH_BYTES: usize = 64 << 20;
// Driver-growth backpressure only: neither a reservation budget nor a limit on reuse.
const PENDING_RETIREMENT_BATCHES: usize = 256;
const PENDING_RETIREMENT_BYTES: usize = 256 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocationKind {
    Small,
    Large,
    Huge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentKind {
    Standard,
    Huge,
    Dedicated,
}

#[derive(Debug)]
struct Segment {
    id: u64,
    ptr: DevicePtr,
    bytes: usize,
    kind: SegmentKind,
    free: BTreeMap<usize, usize>,
    live_blocks: usize,
}

impl Segment {
    fn new(id: u64, ptr: DevicePtr, bytes: usize, kind: SegmentKind) -> Self {
        Self {
            id,
            ptr,
            bytes,
            kind,
            free: BTreeMap::from([(0, bytes)]),
            live_blocks: 0,
        }
    }

    fn allocate(&mut self, bytes: usize, alignment: usize) -> Option<usize> {
        let found = self.free.iter().find_map(|(&offset, &len)| {
            let absolute = self.ptr.checked_add(offset as u64)?;
            let aligned_absolute = align_up_u64(absolute, alignment as u64)?;
            let aligned = usize::try_from(aligned_absolute.checked_sub(self.ptr)?).ok()?;
            let padding = aligned.checked_sub(offset)?;
            let consumed = padding.checked_add(bytes)?;
            (consumed <= len).then_some((offset, len, aligned, consumed))
        });
        let (offset, len, aligned, consumed) = found?;
        self.free.remove(&offset);
        if aligned > offset {
            self.free.insert(offset, aligned - offset);
        }
        if consumed < len {
            self.free.insert(aligned + bytes, len - consumed);
        }
        self.live_blocks += 1;
        Some(aligned)
    }

    fn release(&mut self, offset: usize, bytes: usize) {
        debug_assert!(bytes > 0);
        debug_assert!(offset + bytes <= self.bytes);
        let mut start = offset;
        let mut len = bytes;
        if let Some((&previous, &previous_len)) = self.free.range(..offset).next_back()
            && previous + previous_len == offset
        {
            self.free.remove(&previous);
            start = previous;
            len += previous_len;
        }
        if let Some((&next, &next_len)) = self.free.range(start..).next()
            && start + len == next
        {
            self.free.remove(&next);
            len += next_len;
        }
        self.free.insert(start, len);
        self.live_blocks = self.live_blocks.saturating_sub(1);
    }

    fn is_empty(&self) -> bool {
        self.live_blocks == 0 && self.free.get(&0) == Some(&self.bytes) && self.free.len() == 1
    }
}

#[derive(Debug)]
struct RetiredBlock {
    segment: u64,
    offset: usize,
    granted: usize,
}

#[derive(Debug)]
enum RetirementFences {
    Unfenced,
    Recorded(Vec<CudaRetirementEvent>),
    Complete,
    // Installed before driver calls: unwinding must never imply completion.
    Retained,
}

#[derive(Debug)]
struct PendingRetirement {
    blocks: Vec<RetiredBlock>,
    bytes: usize,
    fences: Mutex<RetirementFences>,
    has_fences: AtomicBool,
}

#[derive(Debug, Default)]
struct MetricState {
    allocation_requests: u64,
    allocation_failures: u64,
    reuse_allocations: u64,
    small_allocations: u64,
    large_allocations: u64,
    huge_allocations: u64,
    requested_bytes: u64,
    granted_bytes: u64,
    live_requested_bytes: usize,
    live_granted_bytes: usize,
    peak_live_requested_bytes: usize,
    peak_live_granted_bytes: usize,
    reserved_bytes: usize,
    peak_reserved_bytes: usize,
    driver_allocations: u64,
    driver_allocation_bytes: u64,
    driver_frees: u64,
    driver_free_bytes: u64,
    capture_retirement_bytes: usize,
    retired_blocks: u64,
    retirement_batches: u64,
    retirement_created_fences: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum AllocatorShutdownState {
    #[default]
    Running,
    Quiescent,
    Retained,
}

#[derive(Debug, Default)]
struct AllocatorState {
    next_segment: u64,
    segments: Vec<Segment>,
    unfenced: Vec<RetiredBlock>,
    unfenced_bytes: usize,
    pending: Vec<Arc<PendingRetirement>>,
    capture_retirements: Vec<RetiredBlock>,
    metrics: MetricState,
    shutdown: bool,
    shutdown_state: AllocatorShutdownState,
}

/// Canonical point-in-time diagnostics for ordinary CUDA device allocations.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CudaAllocatorMetrics {
    pub allocation_requests: u64,
    pub allocation_failures: u64,
    pub reuse_allocations: u64,
    pub small_allocations: u64,
    pub large_allocations: u64,
    pub huge_allocations: u64,
    pub requested_bytes: u64,
    pub granted_bytes: u64,
    pub live_requested_bytes: usize,
    pub live_granted_bytes: usize,
    pub peak_live_requested_bytes: usize,
    pub peak_live_granted_bytes: usize,
    pub reserved_bytes: usize,
    pub peak_reserved_bytes: usize,
    pub free_bytes: usize,
    pub largest_free_extent: usize,
    pub pending_retirement_blocks: usize,
    pub pending_retirement_bytes: usize,
    pub capture_retirement_bytes: usize,
    /// Nonzero blocks retired after their final owner, including capture quarantine.
    pub retired_blocks: u64,
    /// Batches detached from the unfenced queue (including failed batches).
    pub retirement_batches: u64,
    /// Created fences returned by successful whole-batch recordings. Partial failed
    /// recordings are destroyed inside the runtime and are not observable here.
    pub retirement_created_fences: u64,
    pub pending_retirement_batches: usize,
    /// Includes recording/failed batches without a usable complete fence set.
    pub pending_unfenced_bytes: usize,
    /// Final shutdown proved completion and drained all nonzero block owners.
    pub shutdown_quiescent: bool,
    /// Shutdown/context teardown lacks a complete release proof (including unwind).
    pub shutdown_retained: bool,
    pub internal_fragmentation: f64,
    pub external_fragmentation: f64,
    pub segments: usize,
    pub driver_allocations: u64,
    pub driver_allocation_bytes: u64,
    pub driver_frees: u64,
    pub driver_free_bytes: u64,
}

pub(crate) struct CudaDeviceAllocator {
    context: Weak<CudaContext>,
    state: Mutex<AllocatorState>,
    #[cfg(test)]
    allocation_commit_hook: Mutex<Option<AllocationCommitHook>>,
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct AllocationCommitHook {
    pub(crate) reached: Arc<Barrier>,
    pub(crate) resume: Arc<Barrier>,
}

impl std::fmt::Debug for CudaDeviceAllocator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaDeviceAllocator")
            .field("metrics", &self.metrics())
            .finish()
    }
}

pub(crate) struct CudaDeviceBlock {
    allocator: Arc<CudaDeviceAllocator>,
    context: Arc<CudaContext>,
    segment: u64,
    ptr: DevicePtr,
    offset: usize,
    requested: usize,
    granted: usize,
}

impl std::fmt::Debug for CudaDeviceBlock {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaDeviceBlock")
            .field("ptr", &format_args!("{:#x}", self.ptr))
            .field("requested", &self.requested)
            .field("granted", &self.granted)
            .finish_non_exhaustive()
    }
}

impl Drop for CudaDeviceBlock {
    fn drop(&mut self) {
        self.allocator.retire(
            &self.context,
            self.segment,
            self.offset,
            self.requested,
            self.granted,
        );
    }
}

impl CudaDeviceBlock {
    pub(crate) fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub(crate) fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub(crate) fn requested_bytes(&self) -> usize {
        self.requested
    }
}

impl CudaDeviceAllocator {
    pub(crate) fn new(context: Weak<CudaContext>) -> Arc<Self> {
        Arc::new(Self {
            context,
            state: Mutex::new(AllocatorState {
                next_segment: 1,
                ..AllocatorState::default()
            }),
            #[cfg(test)]
            allocation_commit_hook: Mutex::new(None),
        })
    }

    pub(crate) fn allocate(
        self: &Arc<Self>,
        requested: usize,
        alignment: usize,
    ) -> CudaResult<Arc<CudaDeviceBlock>> {
        self.allocate_impl(requested, alignment, false)
    }

    pub(crate) fn allocate_dedicated(
        self: &Arc<Self>,
        requested: usize,
        alignment: usize,
    ) -> CudaResult<Arc<CudaDeviceBlock>> {
        self.allocate_impl(requested, alignment, true)
    }

    fn allocate_impl(
        self: &Arc<Self>,
        requested: usize,
        alignment: usize,
        dedicated: bool,
    ) -> CudaResult<Arc<CudaDeviceBlock>> {
        let context = self
            .context
            .upgrade()
            .ok_or_else(|| CudaError::internal("CUDA allocator context is unavailable"))?;
        let alignment = normalize_alignment(alignment)?;
        if self.lock().shutdown {
            return Err(CudaError::internal("CUDA allocator is shut down"));
        }
        self.poll_retirements();
        let layout = (requested != 0)
            .then(|| allocation_layout(requested))
            .transpose()?;
        #[cfg(test)]
        self.wait_at_allocation_commit_hook();
        let capture_state = context.capture_state();
        let mut state = self.lock();
        if state.shutdown {
            return Err(CudaError::internal("CUDA allocator is shut down"));
        }
        if capture_state.blocks_allocation() {
            state.metrics.allocation_failures = state.metrics.allocation_failures.saturating_add(1);
            return Err(CudaError::internal(if capture_state.completion_unknown() {
                "CUDA allocator completion Unknown; resources quarantined, restart worker"
            } else {
                "CUDA allocator capture active; retry after capture ends"
            }));
        }
        if requested == 0 {
            drop(state);
            drop(capture_state);
            return Ok(Arc::new(CudaDeviceBlock {
                allocator: Arc::clone(self),
                context,
                segment: 0,
                ptr: 0,
                offset: 0,
                requested: 0,
                granted: 0,
            }));
        }
        let (kind, granted) = layout.expect("non-zero CUDA allocation has a layout");
        state.metrics.allocation_requests = state.metrics.allocation_requests.saturating_add(1);
        state.metrics.requested_bytes = state
            .metrics
            .requested_bytes
            .saturating_add(requested as u64);
        state.metrics.granted_bytes = state.metrics.granted_bytes.saturating_add(granted as u64);
        match kind {
            AllocationKind::Small => {
                state.metrics.small_allocations = state.metrics.small_allocations.saturating_add(1)
            }
            AllocationKind::Large => {
                state.metrics.large_allocations = state.metrics.large_allocations.saturating_add(1)
            }
            AllocationKind::Huge => {
                state.metrics.huge_allocations = state.metrics.huge_allocations.saturating_add(1)
            }
        }

        let existing = if dedicated {
            None
        } else {
            find_allocation(&mut state.segments, kind, granted, alignment)
        };
        let (segment, offset, reused) = match existing {
            Some(value) => (value.0, value.1, true),
            None => {
                if pending_growth_pressure(&state) {
                    state.metrics.allocation_failures =
                        state.metrics.allocation_failures.saturating_add(1);
                    return Err(CudaError::internal(
                        "CUDA allocator driver growth deferred by ordinary retirement pressure; poll after registered-stream progress (not a device memory budget)",
                    ));
                }
                let segment_bytes = if dedicated {
                    granted
                } else {
                    segment_size(kind, granted)?
                };
                context.bind_to_thread()?;
                let ptr = match driver_alloc(segment_bytes) {
                    Ok(ptr) => ptr,
                    Err(error) => {
                        state.metrics.allocation_failures =
                            state.metrics.allocation_failures.saturating_add(1);
                        return Err(error);
                    }
                };
                let id = state.next_segment;
                state.next_segment = state.next_segment.saturating_add(1);
                let segment_kind = if dedicated {
                    SegmentKind::Dedicated
                } else if kind == AllocationKind::Huge {
                    SegmentKind::Huge
                } else {
                    SegmentKind::Standard
                };
                let mut segment = Segment::new(id, ptr, segment_bytes, segment_kind);
                let offset = segment
                    .allocate(granted, alignment)
                    .expect("fresh CUDA segment satisfies its allocation");
                state.segments.push(segment);
                state.metrics.reserved_bytes =
                    state.metrics.reserved_bytes.saturating_add(segment_bytes);
                state.metrics.peak_reserved_bytes = state
                    .metrics
                    .peak_reserved_bytes
                    .max(state.metrics.reserved_bytes);
                state.metrics.driver_allocations =
                    state.metrics.driver_allocations.saturating_add(1);
                state.metrics.driver_allocation_bytes = state
                    .metrics
                    .driver_allocation_bytes
                    .saturating_add(segment_bytes as u64);
                (id, offset, false)
            }
        };
        drop(capture_state);
        if reused {
            state.metrics.reuse_allocations = state.metrics.reuse_allocations.saturating_add(1);
        }
        state.metrics.live_requested_bytes =
            state.metrics.live_requested_bytes.saturating_add(requested);
        state.metrics.live_granted_bytes = state.metrics.live_granted_bytes.saturating_add(granted);
        state.metrics.peak_live_requested_bytes = state
            .metrics
            .peak_live_requested_bytes
            .max(state.metrics.live_requested_bytes);
        state.metrics.peak_live_granted_bytes = state
            .metrics
            .peak_live_granted_bytes
            .max(state.metrics.live_granted_bytes);
        let base = state
            .segments
            .iter()
            .find(|candidate| candidate.id == segment)
            .expect("allocated segment remains installed")
            .ptr;
        let ptr = base
            .checked_add(offset as u64)
            .ok_or_else(|| CudaError::internal("CUDA allocator pointer overflow"))?;
        drop(state);
        Ok(Arc::new(CudaDeviceBlock {
            allocator: Arc::clone(self),
            context,
            segment,
            ptr,
            offset,
            requested,
            granted,
        }))
    }

    fn retire(
        &self,
        context: &Arc<CudaContext>,
        segment: u64,
        offset: usize,
        requested: usize,
        granted: usize,
    ) {
        if requested == 0 {
            return;
        }
        let capture_state = context.capture_state();
        let mut state = self.lock();
        state.metrics.live_requested_bytes =
            state.metrics.live_requested_bytes.saturating_sub(requested);
        state.metrics.live_granted_bytes = state.metrics.live_granted_bytes.saturating_sub(granted);
        state.metrics.retired_blocks = state.metrics.retired_blocks.saturating_add(1);
        let block = RetiredBlock {
            segment,
            offset,
            granted,
        };
        if capture_state.blocks_allocation() {
            // Graph pointers remain isolated until final context teardown.
            state.metrics.capture_retirement_bytes = state
                .metrics
                .capture_retirement_bytes
                .saturating_add(granted);
            state.capture_retirements.push(block);
            return;
        }
        // Nothing enters a batch after recording starts. Raw-pointer users must
        // still finish submitting on registered streams before the last owner
        // drops; late submission / unregistered streams remain unsupported.
        let mut pressure = false;
        if state.unfenced_bytes.saturating_add(granted) > RETIREMENT_BATCH_BYTES {
            seal_locked(&mut state);
            pressure = true;
        }
        state.unfenced.push(block);
        state.unfenced_bytes += granted;
        if state.unfenced.len() >= RETIREMENT_BATCH_BLOCKS
            || state.unfenced_bytes >= RETIREMENT_BATCH_BYTES
        {
            seal_locked(&mut state);
            pressure = true;
        }
        drop(state);
        drop(capture_state);
        if pressure {
            self.poll_retirements();
        }
    }

    pub(crate) fn poll_retirements(&self) -> usize {
        let Some(context) = self.context.upgrade() else {
            return 0;
        };
        if context.is_capturing() {
            return 0;
        }
        let pending = {
            let mut state = self.lock();
            seal_locked(&mut state);
            state.pending.clone()
        };
        let mut released = 0;
        for batch in pending {
            // Concurrent/reentrant polling skips an in-flight batch. Retire can
            // always append to the next queue; no allocator lock spans events,
            // stream teardown, or the existing error-only completion fallback.
            let mut fences = match batch.fences.try_lock() {
                Ok(guard) => guard,
                Err(TryLockError::WouldBlock) => continue,
                Err(TryLockError::Poisoned(error)) => error.into_inner(),
            };
            let mut current = std::mem::replace(&mut *fences, RetirementFences::Retained);
            batch.has_fences.store(false, Ordering::Relaxed);
            if matches!(current, RetirementFences::Unfenced) {
                current = match context.record_retirement_events() {
                    Ok(None) => RetirementFences::Unfenced,
                    Ok(Some(events)) => {
                        let mut state = self.lock();
                        state.metrics.retirement_created_fences = state
                            .metrics
                            .retirement_created_fences
                            .saturating_add(events.len() as u64);
                        RetirementFences::Recorded(events)
                    }
                    Err(()) => match context.synchronize_registered_streams() {
                        Ok(()) => RetirementFences::Complete,
                        Err(_error) => {
                            #[cfg(feature = "cuda")]
                            tracing::error!(
                                target: "ferrule_backend::cuda::runtime",
                                error = ?_error,
                                completion_unknown = context.graph_completion_unknown(),
                                retained_blocks = batch.blocks.len(),
                                retained_bytes = batch.bytes,
                                "CUDA allocator retirement record/fallback failed; batch retained; investigate driver completion, restart worker if Unknown"
                            );
                            RetirementFences::Retained
                        }
                    },
                };
            }
            if let RetirementFences::Recorded(events) = current {
                let mut complete = true;
                let mut failed = false;
                for event in &events {
                    match event.query() {
                        Ok(true) => {}
                        Ok(false) => {
                            complete = false;
                            break;
                        }
                        Err(_query_error) => {
                            if let Err(_completion_error) = event.synchronize() {
                                // Failed completion is terminal, not a fence that a
                                // later unrelated local success may wash away.
                                context.mark_graph_unknown();
                                #[cfg(feature = "cuda")]
                                tracing::error!(
                                    target: "ferrule_backend::cuda::runtime",
                                    query_error = ?_query_error,
                                    completion_error = ?_completion_error,
                                    retained_blocks = batch.blocks.len(),
                                    retained_bytes = batch.bytes,
                                    "CUDA allocator retirement completion Unknown; batch retained; restart worker and investigate driver failure"
                                );
                                failed = true;
                                break;
                            }
                        }
                    }
                }
                current = if failed {
                    RetirementFences::Retained
                } else if complete {
                    RetirementFences::Complete
                } else {
                    RetirementFences::Recorded(events)
                };
            }
            // Serialize publication of free extents with capture/Unknown state.
            // Local event success cannot override a concurrent terminal failure.
            let capture_state = context.capture_state();
            if capture_state.completion_unknown() {
                drop(capture_state);
                drop(current);
                continue;
            }
            let may_release = !capture_state.blocks_allocation();
            // Capture is a temporary publication barrier, not loss of an
            // existing Recorded/Complete proof. Keep it for the next poll.
            batch.has_fences.store(
                matches!(
                    current,
                    RetirementFences::Recorded(_) | RetirementFences::Complete
                ),
                Ordering::Relaxed,
            );
            let complete = matches!(current, RetirementFences::Complete);
            *fences = current;
            drop(fences);
            if complete && may_release {
                released += self.release_batch(&batch);
            }
        }
        released
    }

    fn release_batch(&self, batch: &Arc<PendingRetirement>) -> usize {
        let mut state = self.lock();
        let Some(index) = state
            .pending
            .iter()
            .position(|candidate| Arc::ptr_eq(candidate, batch))
        else {
            return 0;
        };
        state.pending.swap_remove(index);
        for block in &batch.blocks {
            release_block_locked(&mut state, block.segment, block.offset, block.granted);
        }
        batch.blocks.len()
    }

    pub(crate) fn trim(&self) -> CudaResult<usize> {
        self.poll_retirements();
        let context = self
            .context
            .upgrade()
            .ok_or_else(|| CudaError::internal("CUDA allocator context is unavailable"))?;
        self.trim_with_context(&context)
    }

    fn trim_with_context(&self, context: &CudaContext) -> CudaResult<usize> {
        if context.is_capturing() {
            return Err(CudaError::internal(
                "CUDA allocator cannot trim during graph capture",
            ));
        }
        context.bind_to_thread()?;
        let mut state = self.lock();
        let mut released = 0;
        let mut index = 0;
        while index < state.segments.len() {
            if !state.segments[index].is_empty() {
                index += 1;
                continue;
            }
            let ptr = state.segments[index].ptr;
            let bytes = state.segments[index].bytes;
            driver_free(ptr)?;
            state.segments.swap_remove(index);
            released += bytes;
            state.metrics.reserved_bytes = state.metrics.reserved_bytes.saturating_sub(bytes);
            state.metrics.driver_frees = state.metrics.driver_frees.saturating_add(1);
            state.metrics.driver_free_bytes =
                state.metrics.driver_free_bytes.saturating_add(bytes as u64);
        }
        Ok(released)
    }

    #[cfg(test)]
    pub(crate) fn set_allocation_commit_hook(&self, hook: AllocationCommitHook) {
        *self
            .allocation_commit_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hook);
    }

    #[cfg(test)]
    fn wait_at_allocation_commit_hook(&self) {
        let hook = self
            .allocation_commit_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(hook) = hook {
            hook.reached.wait();
            hook.resume.wait();
        }
    }

    pub(crate) fn shutdown(&self) {
        let Some(context) = self.context.upgrade() else {
            self.lock().shutdown = true;
            return;
        };
        let _ = self.shutdown_with_context(&context, false);
    }

    pub(crate) fn shutdown_with_context(
        &self,
        context: &CudaContext,
        release_captured: bool,
    ) -> CudaResult<()> {
        let (pending, captured) = {
            let mut state = self.lock();
            state.shutdown = true;
            state.shutdown_state = AllocatorShutdownState::Retained;
            seal_locked(&mut state);
            (
                state.pending.clone(),
                if release_captured {
                    state.capture_retirements.len()
                } else {
                    0
                },
            )
        };
        // Only this pre-sync snapshot is covered. Concurrent last-owner drops
        // stay pending for a later poll / final context teardown.
        context.synchronize_registered_streams()?;
        let capture_state = context.capture_state();
        if capture_state.blocks_allocation() {
            return Err(CudaError::internal(
                "CUDA allocator shutdown: capture or Unknown after completion",
            ));
        }
        for batch in &pending {
            self.release_batch(batch);
        }
        let mut state = self.lock();
        let captured = state
            .capture_retirements
            .drain(..captured)
            .collect::<Vec<_>>();
        for block in captured {
            state.metrics.capture_retirement_bytes = state
                .metrics
                .capture_retirement_bytes
                .saturating_sub(block.granted);
            release_block_locked(&mut state, block.segment, block.offset, block.granted);
        }
        if !state.unfenced.is_empty()
            || !state.pending.is_empty()
            || !state.capture_retirements.is_empty()
            || state
                .segments
                .iter()
                .any(|segment| segment.live_blocks != 0)
        {
            return Err(CudaError::internal(
                "CUDA allocator shutdown: live, captured, or concurrently retired blocks remain",
            ));
        }
        drop(state);
        drop(capture_state);
        self.trim_with_context(context)?;
        let capture_state = context.capture_state();
        if capture_state.blocks_allocation() {
            return Err(CudaError::internal(
                "CUDA allocator shutdown: capture or Unknown during drain",
            ));
        }
        self.lock().shutdown_state = AllocatorShutdownState::Quiescent;
        Ok(())
    }

    // Preserve the ledger and raw event/segment custody, not only the primary
    // context retain. No recovery/reuse is authorized after failed final teardown.
    pub(crate) fn retain_after_context_failure(self: &Arc<Self>) -> CudaAllocatorMetrics {
        let mut state = self.lock();
        state.shutdown = true;
        state.shutdown_state = AllocatorShutdownState::Retained;
        drop(state);
        let metrics = self.metrics();
        std::mem::forget(Arc::clone(self));
        metrics
    }

    pub(crate) fn metrics(&self) -> CudaAllocatorMetrics {
        let state = self.lock();
        let free_bytes = state
            .segments
            .iter()
            .flat_map(|segment| segment.free.values())
            .copied()
            .sum::<usize>();
        let largest_free_extent = state
            .segments
            .iter()
            .flat_map(|segment| segment.free.values())
            .copied()
            .max()
            .unwrap_or(0);
        let pending_retirement_bytes = state
            .pending
            .iter()
            .map(|retirement| retirement.bytes)
            .sum::<usize>();
        CudaAllocatorMetrics {
            allocation_requests: state.metrics.allocation_requests,
            allocation_failures: state.metrics.allocation_failures,
            reuse_allocations: state.metrics.reuse_allocations,
            small_allocations: state.metrics.small_allocations,
            large_allocations: state.metrics.large_allocations,
            huge_allocations: state.metrics.huge_allocations,
            requested_bytes: state.metrics.requested_bytes,
            granted_bytes: state.metrics.granted_bytes,
            live_requested_bytes: state.metrics.live_requested_bytes,
            live_granted_bytes: state.metrics.live_granted_bytes,
            peak_live_requested_bytes: state.metrics.peak_live_requested_bytes,
            peak_live_granted_bytes: state.metrics.peak_live_granted_bytes,
            reserved_bytes: state.metrics.reserved_bytes,
            peak_reserved_bytes: state.metrics.peak_reserved_bytes,
            free_bytes,
            largest_free_extent,
            shutdown_quiescent: state.shutdown_state == AllocatorShutdownState::Quiescent,
            shutdown_retained: state.shutdown_state == AllocatorShutdownState::Retained,
            retired_blocks: state.metrics.retired_blocks,
            retirement_batches: state.metrics.retirement_batches,
            retirement_created_fences: state.metrics.retirement_created_fences,
            pending_retirement_batches: state.pending.len(),
            pending_unfenced_bytes: state.unfenced_bytes
                + state
                    .pending
                    .iter()
                    .filter(|batch| !batch.has_fences.load(Ordering::Relaxed))
                    .map(|batch| batch.bytes)
                    .sum::<usize>(),
            pending_retirement_blocks: state.unfenced.len()
                + state
                    .pending
                    .iter()
                    .map(|batch| batch.blocks.len())
                    .sum::<usize>()
                + state.capture_retirements.len(),
            pending_retirement_bytes: state.unfenced_bytes
                + pending_retirement_bytes
                + state
                    .capture_retirements
                    .iter()
                    .map(|retirement| retirement.granted)
                    .sum::<usize>(),
            capture_retirement_bytes: state.metrics.capture_retirement_bytes,
            internal_fragmentation: fragmentation(
                state
                    .metrics
                    .live_granted_bytes
                    .saturating_sub(state.metrics.live_requested_bytes),
                state.metrics.live_granted_bytes,
            ),
            external_fragmentation: if free_bytes == 0 {
                0.0
            } else {
                1.0 - largest_free_extent as f64 / free_bytes as f64
            },
            segments: state.segments.len(),
            driver_allocations: state.metrics.driver_allocations,
            driver_allocation_bytes: state.metrics.driver_allocation_bytes,
            driver_frees: state.metrics.driver_frees,
            driver_free_bytes: state.metrics.driver_free_bytes,
        }
    }

    fn lock(&self) -> MutexGuard<'_, AllocatorState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn seal_locked(state: &mut AllocatorState) {
    if state.unfenced.is_empty() {
        return;
    }
    let batch = Arc::new(PendingRetirement {
        blocks: std::mem::take(&mut state.unfenced),
        bytes: std::mem::take(&mut state.unfenced_bytes),
        fences: Mutex::new(RetirementFences::Unfenced),
        has_fences: AtomicBool::new(false),
    });
    state.pending.push(batch);
    state.metrics.retirement_batches = state.metrics.retirement_batches.saturating_add(1);
}

// Capture quarantine cannot progress through polling and belongs to context
// teardown, not the ordinary retirement-growth throttle. Existing safe extents
// are always tried first; externally admitted/reserved memory is not re-budgeted.
fn pending_growth_pressure(state: &AllocatorState) -> bool {
    state.pending.len() >= PENDING_RETIREMENT_BATCHES
        || state
            .unfenced_bytes
            .saturating_add(state.pending.iter().map(|batch| batch.bytes).sum::<usize>())
            >= PENDING_RETIREMENT_BYTES
}

fn find_allocation(
    segments: &mut [Segment],
    kind: AllocationKind,
    granted: usize,
    alignment: usize,
) -> Option<(u64, usize)> {
    segments.iter_mut().find_map(|segment| {
        let compatible = match kind {
            AllocationKind::Huge => segment.kind == SegmentKind::Huge,
            AllocationKind::Small | AllocationKind::Large => segment.kind == SegmentKind::Standard,
        };
        compatible
            .then(|| {
                segment
                    .allocate(granted, alignment)
                    .map(|offset| (segment.id, offset))
            })
            .flatten()
    })
}

fn release_block_locked(state: &mut AllocatorState, segment: u64, offset: usize, granted: usize) {
    if let Some(segment) = state
        .segments
        .iter_mut()
        .find(|candidate| candidate.id == segment)
    {
        segment.release(offset, granted);
    }
}

fn normalize_alignment(alignment: usize) -> CudaResult<usize> {
    let alignment = alignment.max(MIN_ALIGNMENT);
    if !alignment.is_power_of_two() {
        return Err(CudaError::internal(
            "CUDA allocation alignment must be a power of two",
        ));
    }
    Ok(alignment)
}

fn allocation_layout(requested: usize) -> CudaResult<(AllocationKind, usize)> {
    let kind = if requested <= SMALL_LIMIT {
        AllocationKind::Small
    } else if requested <= LARGE_LIMIT {
        AllocationKind::Large
    } else {
        AllocationKind::Huge
    };
    let granted = cuda_allocation_grant_bytes(requested)
        .ok_or_else(|| CudaError::internal("CUDA allocation size overflow"))?;
    Ok((kind, granted))
}

fn segment_size(kind: AllocationKind, granted: usize) -> CudaResult<usize> {
    match kind {
        AllocationKind::Small | AllocationKind::Large => Ok(SEGMENT_BYTES.max(granted)),
        AllocationKind::Huge => align_up(granted, ALLOCATOR_PAGE_BYTES)
            .ok_or_else(|| CudaError::internal("CUDA huge segment size overflow")),
    }
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|value| value & !(alignment - 1))
}

fn align_up_u64(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|value| value & !(alignment - 1))
}

fn fragmentation(unusable: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        unusable as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn device_block_auto_traits_follow_its_owners() {
        assert_send_sync::<CudaDeviceBlock>();
    }

    #[test]
    fn size_classes_cover_small_large_and_huge() {
        assert_eq!(allocation_layout(1).unwrap(), (AllocationKind::Small, 256));
        assert_eq!(
            allocation_layout(257).unwrap(),
            (AllocationKind::Small, 512)
        );
        assert_eq!(
            allocation_layout(SMALL_LIMIT + 1).unwrap(),
            (AllocationKind::Large, ALLOCATOR_PAGE_BYTES)
        );
        assert_eq!(
            allocation_layout(LARGE_LIMIT + 1).unwrap(),
            (AllocationKind::Huge, LARGE_LIMIT + ALLOCATOR_PAGE_BYTES)
        );
    }

    #[test]
    fn alignment_applies_to_the_absolute_device_address() {
        let mut segment = Segment::new(1, 0x1100, 8192, SegmentKind::Standard);
        let offset = segment.allocate(256, 4096).unwrap();
        assert_eq!((segment.ptr + offset as u64) % 4096, 0);
        assert_eq!(offset, 0x0f00);
    }

    #[test]
    fn free_ranges_coalesce_in_both_directions() {
        let mut segment = Segment::new(1, 0x1000, 4096, SegmentKind::Standard);
        let a = segment.allocate(256, 256).unwrap();
        let b = segment.allocate(512, 256).unwrap();
        let c = segment.allocate(256, 256).unwrap();
        segment.release(b, 512);
        segment.release(a, 256);
        segment.release(c, 256);
        assert!(segment.is_empty());
    }

    #[test]
    fn deterministic_allocation_churn_preserves_capacity_and_alignment() {
        let mut segment = Segment::new(7, 0x1000, 1 << 20, SegmentKind::Standard);
        let mut seed = 0x1234_5678_u64;
        let mut live = Vec::new();
        for step in 0..10_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            if !live.is_empty() && (seed & 3) == 0 {
                let index = (seed as usize) % live.len();
                let (offset, bytes) = live.swap_remove(index);
                segment.release(offset, bytes);
            } else {
                let requested = ((seed >> 17) as usize % 4096) + 1;
                let (_, granted) = allocation_layout(requested).unwrap();
                if let Some(offset) = segment.allocate(granted, MIN_ALIGNMENT) {
                    assert_eq!(offset % MIN_ALIGNMENT, 0, "step {step}");
                    live.push((offset, granted));
                }
            }
            let free = segment.free.values().sum::<usize>();
            let used = live.iter().map(|(_, bytes)| bytes).sum::<usize>();
            assert_eq!(free + used, segment.bytes, "step {step}");

            let mut extents = live.clone();
            extents.sort_unstable_by_key(|(offset, _)| *offset);
            for pair in extents.windows(2) {
                assert!(pair[0].0 + pair[0].1 <= pair[1].0, "step {step}");
            }
            let mut previous_end = 0;
            for (&offset, &bytes) in &segment.free {
                assert!(offset >= previous_end, "step {step}");
                assert!(offset + bytes <= segment.bytes, "step {step}");
                previous_end = offset + bytes;
            }
        }
        for (offset, bytes) in live {
            segment.release(offset, bytes);
        }
        assert!(segment.is_empty());
    }

    #[test]
    fn fragmentation_formulas_are_bounded() {
        assert_eq!(fragmentation(0, 0), 0.0);
        assert_eq!(fragmentation(0, 256), 0.0);
        assert!((fragmentation(128, 256) - 0.5).abs() < f64::EPSILON);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cuda/allocator.rs"]
mod dedicated_kv_tests;
