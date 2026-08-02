use std::collections::HashMap;
#[cfg(any(feature = "cuda", test))]
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(feature = "cuda")]
use ferrule_common::materialization_io::MaterializationResourceRequirements;
use ferrule_common::{CompletionHub, Error, Result};
#[cfg(feature = "cuda")]
use ferrule_common::{
    MaterializationKey, OperationId, RegisteredPinnedAlignedSlabLeaseDescriptor, RegistrationId,
    SlabId,
};
#[cfg(feature = "cuda")]
use ferrule_cuda::context::{CudaPinnedHostAllocator, CudaPinnedU8HostBuffer};
use io_uring::{IoUring, opcode, types};

use super::streaming::{ExpertIoStats, ExpertTensorPayload, ExpertTensorSlice};
use crate::runner::ModelCompletionReactor;

const DIRECT_IO_ALIGNMENT: usize = 4096;
const FIXED_FILE_CAPACITY: usize = 64;

#[repr(C, align(4096))]
#[derive(Clone)]
struct AlignedBlock([u8; DIRECT_IO_ALIGNMENT]);

#[cfg(feature = "cuda")]
pub(crate) struct PinnedExpertTensorPayload {
    pub(crate) slice: ExpertTensorSlice,
    pub(crate) bytes: CudaPinnedU8HostBuffer,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PinnedExpertReadTicket {
    operation_id: u64,
}

#[cfg(feature = "cuda")]
pub(crate) struct PinnedExpertReadPlan {
    extents: Vec<DirectReadExtent>,
    payload_count: usize,
    requirements: MaterializationResourceRequirements,
}

#[cfg(feature = "cuda")]
pub(crate) struct ReservedPinnedExpertRead {
    pub(crate) ticket: PinnedExpertReadTicket,
    pub(crate) slabs: Box<[RegisteredPinnedAlignedSlabLeaseDescriptor]>,
}

#[cfg(feature = "cuda")]
impl PinnedExpertReadPlan {
    pub(crate) const fn requirements(&self) -> MaterializationResourceRequirements {
        self.requirements
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct PinnedExpertReadResult {
    pub(crate) payloads: Vec<PinnedExpertTensorPayload>,
}

#[cfg(feature = "cuda")]
pub(crate) enum PinnedExpertReadPoll {
    Pending,
    Ready(PinnedExpertReadResult),
    Failed(Error),
    Cancelled,
}

enum RegisteredBufferBacking {
    Pageable(Box<[AlignedBlock]>),
    #[cfg(feature = "cuda")]
    CudaPinned(CudaPinnedU8HostBuffer),
}

struct RegisteredBuffer {
    backing: RegisteredBufferBacking,
    len: usize,
}

impl RegisteredBuffer {
    fn new_pageable(len: usize) -> Result<Self> {
        let len = align_up(len, DIRECT_IO_ALIGNMENT)?;
        let blocks = len / DIRECT_IO_ALIGNMENT;
        Ok(Self {
            backing: RegisteredBufferBacking::Pageable(
                vec![AlignedBlock([0; DIRECT_IO_ALIGNMENT]); blocks].into_boxed_slice(),
            ),
            len,
        })
    }

    #[cfg(feature = "cuda")]
    fn new_cuda_pinned(len: usize, allocator: &CudaPinnedHostAllocator) -> Result<Self> {
        let len = align_up(len, DIRECT_IO_ALIGNMENT)?;
        Ok(Self {
            backing: RegisteredBufferBacking::CudaPinned(
                allocator.allocate_u8_aligned(len, DIRECT_IO_ALIGNMENT)?,
            ),
            len,
        })
    }

    fn is_available(&self) -> bool {
        match &self.backing {
            RegisteredBufferBacking::Pageable(_) => true,
            #[cfg(feature = "cuda")]
            RegisteredBufferBacking::CudaPinned(buffer) => buffer.is_uniquely_owned(),
        }
    }

    #[cfg(feature = "cuda")]
    fn is_cuda_pinned(&self) -> bool {
        match &self.backing {
            RegisteredBufferBacking::Pageable(_) => false,
            #[cfg(feature = "cuda")]
            RegisteredBufferBacking::CudaPinned(_) => true,
        }
    }

    #[allow(unsafe_code)]
    fn as_mut_ptr(&mut self) -> Result<*mut u8> {
        match &mut self.backing {
            RegisteredBufferBacking::Pageable(blocks) => Ok(blocks.as_mut_ptr().cast()),
            #[cfg(feature = "cuda")]
            RegisteredBufferBacking::CudaPinned(buffer) => {
                // SAFETY: callers only request the pointer after `is_available`
                // confirms no payload view still owns the slab. The io_uring
                // state remains exclusively locked until the CQE is consumed.
                unsafe { buffer.as_mut_ptr_unique() }
            }
        }
    }

    fn iovec(&mut self) -> Result<libc::iovec> {
        Ok(libc::iovec {
            iov_base: self.as_mut_ptr()?.cast(),
            iov_len: self.len,
        })
    }

    #[allow(unsafe_code)]
    fn range(&self, offset: usize, len: usize) -> Result<&[u8]> {
        let end = offset.checked_add(len).ok_or_else(|| Error::Model {
            message: "io_uring expert slice range overflow".into(),
        })?;
        if end > self.len {
            return Err(Error::Model {
                message: format!(
                    "io_uring expert slice range exceeds registered buffer: {offset}+{len}>{}",
                    self.len
                ),
            });
        }
        match &self.backing {
            RegisteredBufferBacking::Pageable(blocks) => {
                // SAFETY: `AlignedBlock` contains exactly 4096 bytes with no
                // trailing padding, and boxed slices store blocks contiguously.
                // `self.len` is fixed to `blocks.len() * 4096`, and the shared
                // slice cannot mutate the backing allocation.
                let bytes =
                    unsafe { std::slice::from_raw_parts(blocks.as_ptr().cast::<u8>(), self.len) };
                Ok(&bytes[offset..end])
            }
            #[cfg(feature = "cuda")]
            RegisteredBufferBacking::CudaPinned(buffer) => Ok(&buffer.as_slice()[offset..end]),
        }
    }

    fn copy_range(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        Ok(self.range(offset, len)?.to_vec())
    }

    #[cfg(feature = "cuda")]
    fn pinned_range(&self, offset: usize, len: usize) -> Result<CudaPinnedU8HostBuffer> {
        match &self.backing {
            RegisteredBufferBacking::CudaPinned(buffer) => buffer.slice(offset, len),
            RegisteredBufferBacking::Pageable(_) => Err(Error::Model {
                message: "io_uring expert reader was not configured with CUDA pinned slabs".into(),
            }),
        }
    }
}

struct DirectReadView {
    slice_index: usize,
    slice: ExpertTensorSlice,
    payload_offset: usize,
    payload_len: usize,
}

struct DirectReadExtent {
    file_index: u32,
    aligned_offset: u64,
    aligned_len: usize,
    required_end: usize,
    views: Vec<DirectReadView>,
}

#[cfg(any(feature = "cuda", test))]
struct PinnedReadOperation<T> {
    extents: Vec<DirectReadExtent>,
    payloads: Vec<Option<T>>,
    completed_extents: Vec<bool>,
    extent_buffers: Vec<Option<usize>>,
    outstanding: usize,
    next_extent: usize,
    submission_authorized: bool,
    error: Option<Error>,
    cancelled: bool,
    #[cfg(feature = "cuda")]
    started: Instant,
    #[cfg(feature = "cuda")]
    timing_accounted: bool,
}

#[cfg(any(feature = "cuda", test))]
impl<T> PinnedReadOperation<T> {
    #[cfg(test)]
    fn new(extents: Vec<DirectReadExtent>, payload_count: usize) -> Self {
        Self::new_inner(extents, payload_count, true)
    }

    fn physical(extents: Vec<DirectReadExtent>, payload_count: usize) -> Self {
        Self::new_inner(extents, payload_count, false)
    }

    fn new_inner(
        extents: Vec<DirectReadExtent>,
        payload_count: usize,
        submission_authorized: bool,
    ) -> Self {
        let completed_extents = vec![false; extents.len()];
        let extent_buffers = vec![None; extents.len()];
        Self {
            extents,
            payloads: (0..payload_count).map(|_| None).collect(),
            completed_extents,
            extent_buffers,
            outstanding: 0,
            next_extent: 0,
            submission_authorized,
            error: None,
            cancelled: false,
            #[cfg(feature = "cuda")]
            started: Instant::now(),
            #[cfg(feature = "cuda")]
            timing_accounted: false,
        }
    }

    fn can_submit(&self) -> bool {
        self.submission_authorized
            && !self.cancelled
            && self.error.is_none()
            && self.next_extent < self.extents.len()
    }

    fn authorize_submission(&mut self) -> Result<()> {
        if self.submission_authorized {
            return Err(Error::Internal {
                message: "pinned expert read was submitted more than once".into(),
            });
        }
        if self.cancelled || self.error.is_some() || self.is_terminal() {
            return Err(Error::Internal {
                message: "terminal pinned expert read cannot be submitted".into(),
            });
        }
        self.submission_authorized = true;
        Ok(())
    }

    fn has_no_reservations(&self) -> bool {
        self.extent_buffers.iter().all(Option::is_none)
    }

    fn has_all_reservations(&self) -> bool {
        self.extent_buffers.iter().all(Option::is_some)
    }

    fn reserved_buffer(&self, extent_index: usize) -> Result<usize> {
        self.extent_buffers
            .get(extent_index)
            .copied()
            .flatten()
            .ok_or_else(|| Error::Internal {
                message: format!("pinned expert read extent {extent_index} has no reserved slab"),
            })
    }

    fn release_unqueued_reservations(&mut self) -> Vec<usize> {
        self.extent_buffers
            .iter_mut()
            .skip(self.next_extent)
            .filter_map(Option::take)
            .collect()
    }

    fn record_submission(&mut self, extent_index: usize) -> Result<()> {
        if extent_index != self.next_extent || extent_index >= self.extents.len() {
            return Err(Error::Internal {
                message: format!(
                    "pinned expert read submitted unexpected extent {extent_index}, next={} total={}",
                    self.next_extent,
                    self.extents.len()
                ),
            });
        }
        self.reserved_buffer(extent_index)?;
        self.next_extent += 1;
        self.outstanding = self
            .outstanding
            .checked_add(1)
            .ok_or_else(|| Error::Internal {
                message: "pinned expert read outstanding overflow".into(),
            })?;
        Ok(())
    }

    fn record_completion(&mut self, extent_index: usize) -> Result<()> {
        if extent_index >= self.next_extent || extent_index >= self.completed_extents.len() {
            return Err(Error::Internal {
                message: format!(
                    "pinned expert read completed unsubmitted extent {extent_index}, submitted={}",
                    self.next_extent
                ),
            });
        }
        if self.completed_extents[extent_index] {
            return Err(Error::Internal {
                message: format!(
                    "pinned expert read completed extent {extent_index} more than once"
                ),
            });
        }
        self.completed_extents[extent_index] = true;
        self.outstanding = self
            .outstanding
            .checked_sub(1)
            .ok_or_else(|| Error::Internal {
                message: "pinned expert read outstanding underflow".into(),
            })?;
        Ok(())
    }

    fn fail(&mut self, error: Error) -> Vec<usize> {
        if self.error.is_none() {
            self.error = Some(error);
        }
        self.release_unqueued_reservations()
    }

    fn cancel(&mut self) -> Option<Vec<usize>> {
        if self.cancelled || self.is_terminal() {
            return None;
        }
        self.cancelled = true;
        for payload in &mut self.payloads {
            *payload = None;
        }
        Some(self.release_unqueued_reservations())
    }

    fn is_terminal(&self) -> bool {
        self.outstanding == 0
            && (self.cancelled || self.error.is_some() || self.next_extent == self.extents.len())
    }

    #[cfg(feature = "cuda")]
    fn take_elapsed_if_terminal(&mut self) -> Option<std::time::Duration> {
        if !self.is_terminal() || self.timing_accounted {
            return None;
        }
        self.timing_accounted = true;
        Some(self.started.elapsed())
    }
}

#[cfg(any(feature = "cuda", test))]
fn take_terminal_pinned_operation<T>(
    operations: &mut HashMap<u64, PinnedReadOperation<T>>,
    operation_id: u64,
) -> Option<PinnedReadOperation<T>> {
    operations
        .get(&operation_id)
        .is_some_and(PinnedReadOperation::is_terminal)
        .then(|| {
            operations
                .remove(&operation_id)
                .expect("terminal operation exists")
        })
}

#[cfg(any(feature = "cuda", test))]
fn reap_detached_pinned_operations<T>(
    operations: &mut HashMap<u64, PinnedReadOperation<T>>,
    detached: &mut VecDeque<u64>,
) -> usize {
    let mut pending = VecDeque::with_capacity(detached.len());
    let mut reaped = 0usize;
    while let Some(operation_id) = detached.pop_front() {
        if !operations.contains_key(&operation_id)
            || take_terminal_pinned_operation(operations, operation_id).is_some()
        {
            reaped = reaped.saturating_add(1);
        } else {
            pending.push_back(operation_id);
        }
    }
    *detached = pending;
    reaped
}

#[cfg(any(feature = "cuda", test))]
fn unscheduled_pinned_operation_reason(
    scheduler_can_queue: bool,
    has_all_reservations: bool,
    has_no_reservations: bool,
    reusable_buffers: usize,
    extent_count: usize,
) -> Option<String> {
    if !scheduler_can_queue {
        return None;
    }
    if has_all_reservations {
        return Some(
            "pinned expert read retained all slabs but had no submitted or queued work".into(),
        );
    }
    (has_no_reservations && reusable_buffers >= extent_count).then(|| {
        format!(
            "pinned expert read could reserve {extent_count} slabs but scheduling made no progress"
        )
    })
}

#[cfg(any(feature = "cuda", test))]
fn reserve_all_extent_buffers<T>(
    operation: &mut PinnedReadOperation<T>,
    buffer_busy: &mut [bool],
    buffer_reusable: &[bool],
) -> Result<bool> {
    if buffer_busy.len() != buffer_reusable.len() {
        return Err(Error::Internal {
            message: format!(
                "pinned expert slab state length mismatch: busy={} reusable={}",
                buffer_busy.len(),
                buffer_reusable.len()
            ),
        });
    }
    if operation.has_all_reservations() {
        return Ok(true);
    }
    if !operation.has_no_reservations() {
        return Err(Error::Internal {
            message: "pinned expert read has a partial slab reservation".into(),
        });
    }

    let required = operation.extents.len();
    let candidates = buffer_busy
        .iter()
        .zip(buffer_reusable)
        .enumerate()
        .filter_map(|(index, (busy, reusable))| (!*busy && *reusable).then_some(index))
        .take(required)
        .collect::<Vec<_>>();
    if candidates.len() != required {
        return Ok(false);
    }
    for (slot, buffer_index) in operation.extent_buffers.iter_mut().zip(candidates) {
        buffer_busy[buffer_index] = true;
        *slot = Some(buffer_index);
    }
    Ok(true)
}

#[cfg(any(feature = "cuda", test))]
fn validate_pinned_extents(
    extents: &[DirectReadExtent],
    registered_buffers: usize,
    registered_buffer_bytes: usize,
    registered_files: usize,
) -> Result<()> {
    if extents.len() > registered_buffers {
        return Err(Error::Model {
            message: format!(
                "pinned expert read requires {} extents but only {registered_buffers} registered slabs exist; the operation can never make progress",
                extents.len()
            ),
        });
    }
    if registered_buffers > usize::from(u16::MAX) + 1 {
        return Err(Error::Model {
            message: format!(
                "pinned expert registered slab count {registered_buffers} exceeds the fixed-buffer u16 ABI"
            ),
        });
    }
    for (extent_index, extent) in extents.iter().enumerate() {
        if extent.aligned_len == 0 {
            return Err(Error::Model {
                message: format!("pinned expert extent {extent_index} is empty"),
            });
        }
        u32::try_from(extent.aligned_len).map_err(|_| Error::Model {
            message: format!(
                "pinned expert extent {extent_index} length {} exceeds the io_uring u32 read ABI",
                extent.aligned_len
            ),
        })?;
        if extent.aligned_len > registered_buffer_bytes {
            return Err(Error::Model {
                message: format!(
                    "pinned expert extent {extent_index} exceeds its registered slab: aligned={} slab={registered_buffer_bytes}",
                    extent.aligned_len
                ),
            });
        }
        if extent.required_end > extent.aligned_len {
            return Err(Error::Model {
                message: format!(
                    "pinned expert extent {extent_index} requires {} bytes from an aligned read of {} bytes",
                    extent.required_end, extent.aligned_len
                ),
            });
        }
        let file_index = usize::try_from(extent.file_index).map_err(|_| Error::Model {
            message: "pinned expert fixed-file index exceeds usize".into(),
        })?;
        if file_index >= registered_files || file_index >= FIXED_FILE_CAPACITY {
            return Err(Error::Model {
                message: format!(
                    "pinned expert extent {extent_index} references unregistered fixed file {} (registered={registered_files})",
                    extent.file_index
                ),
            });
        }
        for view in &extent.views {
            let end = view
                .payload_offset
                .checked_add(view.payload_len)
                .ok_or_else(|| Error::Model {
                    message: format!(
                        "pinned expert extent {extent_index} payload range overflows usize"
                    ),
                })?;
            if end > extent.required_end {
                return Err(Error::Model {
                    message: format!(
                        "pinned expert extent {extent_index} payload range {}..{end} exceeds required read end {}",
                        view.payload_offset, extent.required_end
                    ),
                });
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PinnedReadSubmissionState {
    Queued,
    Submitted,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PinnedReadSubmission {
    operation_id: u64,
    extent_index: usize,
    buffer_index: usize,
    state: PinnedReadSubmissionState,
}

#[cfg(any(feature = "cuda", test))]
fn allocate_durable_id(next: &mut u64, kind: &str) -> Result<u64> {
    let id = *next;
    if id == 0 {
        return Err(Error::Internal {
            message: format!("{kind} IDs are exhausted"),
        });
    }
    *next = id
        .checked_add(1)
        .filter(|next| *next != 0)
        .ok_or_else(|| Error::Internal {
            message: format!("{kind} IDs are exhausted"),
        })?;
    Ok(id)
}

struct PendingDirectReadExtent {
    path: PathBuf,
    start: u64,
    end: u64,
    slices: Vec<(usize, ExpertTensorSlice)>,
}

struct IoUringDirectState {
    ring: IoUring,
    completion_eventfd_registered: bool,
    queue_depth: usize,
    buffer_bytes: usize,
    buffers: Vec<RegisteredBuffer>,
    files: Vec<File>,
    file_indices: HashMap<PathBuf, u32>,
    stats: ExpertIoStats,
    #[cfg(feature = "cuda")]
    pinned_operations: HashMap<u64, PinnedReadOperation<PinnedExpertTensorPayload>>,
    #[cfg(feature = "cuda")]
    pinned_operation_order: VecDeque<u64>,
    #[cfg(feature = "cuda")]
    detached_pinned_operations: VecDeque<u64>,
    #[cfg(feature = "cuda")]
    pinned_submissions: HashMap<u64, PinnedReadSubmission>,
    #[cfg(feature = "cuda")]
    pinned_queued_submissions: VecDeque<u64>,
    #[cfg(feature = "cuda")]
    pinned_buffer_busy: Vec<bool>,

    #[cfg(feature = "cuda")]
    next_pinned_operation_id: u64,
    #[cfg(feature = "cuda")]
    next_pinned_submission_id: u64,
}

impl IoUringDirectState {
    fn new(queue_depth: usize, buffer_bytes: usize) -> Result<Self> {
        let buffers = (0..queue_depth)
            .map(|_| RegisteredBuffer::new_pageable(buffer_bytes))
            .collect::<Result<Vec<_>>>()?;
        Self::new_with_buffers(queue_depth, buffer_bytes, buffers)
    }

    #[cfg(feature = "cuda")]
    fn new_cuda_pinned(
        queue_depth: usize,
        buffer_bytes: usize,
        slab_count: usize,
        allocator: &CudaPinnedHostAllocator,
    ) -> Result<Self> {
        if slab_count < queue_depth {
            return Err(Error::Model {
                message: format!(
                    "CUDA pinned expert slab count must be at least queue depth: {slab_count} < {queue_depth}"
                ),
            });
        }
        if slab_count > u16::MAX as usize {
            return Err(Error::Model {
                message: format!(
                    "CUDA pinned expert slab count exceeds io_uring fixed-buffer limit: {slab_count}"
                ),
            });
        }
        let buffers = (0..slab_count)
            .map(|_| RegisteredBuffer::new_cuda_pinned(buffer_bytes, allocator))
            .collect::<Result<Vec<_>>>()?;
        Self::new_with_buffers(queue_depth, buffer_bytes, buffers)
    }

    #[allow(unsafe_code)]
    fn new_with_buffers(
        queue_depth: usize,
        buffer_bytes: usize,
        mut buffers: Vec<RegisteredBuffer>,
    ) -> Result<Self> {
        if queue_depth == 0 || queue_depth > u16::MAX as usize {
            return Err(Error::Model {
                message: format!(
                    "io_uring expert queue depth must be in 1..={}, got {queue_depth}",
                    u16::MAX
                ),
            });
        }
        let buffer_bytes = align_up(buffer_bytes, DIRECT_IO_ALIGNMENT)?;
        let entries = queue_depth.next_power_of_two().max(2);
        let entries = u32::try_from(entries).map_err(|_| Error::Model {
            message: "io_uring expert queue depth exceeds u32".into(),
        })?;
        let ring = IoUring::new(entries).map_err(|error| Error::Model {
            message: format!("create expert io_uring: {error}"),
        })?;
        ring.submitter()
            .register_files_sparse(FIXED_FILE_CAPACITY as u32)
            .map_err(|error| Error::Model {
                message: format!("register sparse expert io_uring files: {error}"),
            })?;

        let iovecs = buffers
            .iter_mut()
            .map(RegisteredBuffer::iovec)
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: every iovec points into a fixed-size boxed allocation owned by
        // this state. Buffers are never resized and the ring is unregistered or
        // dropped before the state releases those allocations.
        unsafe { ring.submitter().register_buffers(&iovecs) }.map_err(|error| Error::Model {
            message: format!("register expert io_uring buffers: {error}"),
        })?;

        #[cfg(feature = "cuda")]
        let buffer_count = buffers.len();
        Ok(Self {
            ring,
            completion_eventfd_registered: false,
            queue_depth,
            buffer_bytes,
            buffers,
            files: Vec::new(),
            file_indices: HashMap::new(),
            stats: ExpertIoStats::default(),
            #[cfg(feature = "cuda")]
            pinned_operations: HashMap::new(),
            #[cfg(feature = "cuda")]
            pinned_operation_order: VecDeque::new(),
            #[cfg(feature = "cuda")]
            detached_pinned_operations: VecDeque::new(),
            #[cfg(feature = "cuda")]
            pinned_submissions: HashMap::new(),
            #[cfg(feature = "cuda")]
            pinned_queued_submissions: VecDeque::new(),
            #[cfg(feature = "cuda")]
            pinned_buffer_busy: vec![false; buffer_count],

            #[cfg(feature = "cuda")]
            next_pinned_operation_id: 1,
            #[cfg(feature = "cuda")]
            next_pinned_submission_id: 1,
        })
    }

    fn register_completion_eventfd(&mut self, eventfd: &OwnedFd) -> Result<()> {
        self.ring
            .submitter()
            .register_eventfd(eventfd.as_raw_fd())
            .map_err(|error| Error::Model {
                message: format!("register expert io_uring completion eventfd: {error}"),
            })?;
        self.completion_eventfd_registered = true;
        Ok(())
    }

    fn unregister_completion_eventfd(&mut self) -> Result<()> {
        if !self.completion_eventfd_registered {
            return Ok(());
        }
        self.ring
            .submitter()
            .unregister_eventfd()
            .map_err(|error| Error::Model {
                message: format!("unregister expert io_uring completion eventfd: {error}"),
            })?;
        self.completion_eventfd_registered = false;
        Ok(())
    }

    fn register_file(&mut self, path: &Path) -> Result<u32> {
        if let Some(&index) = self.file_indices.get(path) {
            return Ok(index);
        }
        if self.files.len() >= FIXED_FILE_CAPACITY {
            return Err(Error::Model {
                message: format!(
                    "expert io_uring fixed-file table is full at {FIXED_FILE_CAPACITY} entries"
                ),
            });
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(|error| Error::Model {
                message: format!(
                    "open expert shard with O_DIRECT '{}': {error}",
                    path.display()
                ),
            })?;
        let index = u32::try_from(self.files.len()).map_err(|_| Error::Model {
            message: "expert fixed-file index exceeds u32".into(),
        })?;
        self.ring
            .submitter()
            .register_files_update(index, &[file.as_raw_fd()])
            .map_err(|error| Error::Model {
                message: format!(
                    "register expert shard '{}' at fixed index {index}: {error}",
                    path.display()
                ),
            })?;
        self.files.push(file);
        self.file_indices.insert(path.to_path_buf(), index);
        self.stats.fixed_file_registrations = self.stats.fixed_file_registrations.saturating_add(1);
        Ok(index)
    }

    fn plan(&mut self, slices: &[ExpertTensorSlice]) -> Result<Vec<DirectReadExtent>> {
        let mut ordered = slices.iter().cloned().enumerate().collect::<Vec<_>>();
        ordered.sort_by(|(_, left), (_, right)| {
            left.path
                .cmp(&right.path)
                .then_with(|| left.offset.cmp(&right.offset))
        });

        let mut extents = Vec::new();
        let mut current: Option<PendingDirectReadExtent> = None;
        for (slice_index, slice) in ordered {
            if slice.bytes == 0 {
                return Err(Error::Model {
                    message: "expert tensor slice is empty".into(),
                });
            }
            let slice_end = slice
                .offset
                .checked_add(slice.bytes)
                .ok_or_else(|| Error::Model {
                    message: "expert tensor slice end overflow".into(),
                })?;
            let can_merge = current.as_ref().is_some_and(|current| {
                if current.path != slice.path
                    || slice.offset > current.end.saturating_add(DIRECT_IO_ALIGNMENT as u64)
                {
                    return false;
                }
                let aligned_start =
                    current.start / DIRECT_IO_ALIGNMENT as u64 * DIRECT_IO_ALIGNMENT as u64;
                let merged_required = slice_end.max(current.end).saturating_sub(aligned_start);
                usize::try_from(merged_required)
                    .ok()
                    .and_then(|required| align_up(required, DIRECT_IO_ALIGNMENT).ok())
                    .is_some_and(|aligned| aligned <= self.buffer_bytes)
            });
            if can_merge {
                let current = current.as_mut().expect("checked above");
                current.end = current.end.max(slice_end);
                current.slices.push((slice_index, slice));
                continue;
            }
            if let Some(current) = current.take() {
                extents.push(self.build_extent(
                    current.path,
                    current.start,
                    current.end,
                    current.slices,
                )?);
            }
            current = Some(PendingDirectReadExtent {
                path: slice.path.clone(),
                start: slice.offset,
                end: slice_end,
                slices: vec![(slice_index, slice)],
            });
        }
        if let Some(current) = current {
            extents.push(self.build_extent(
                current.path,
                current.start,
                current.end,
                current.slices,
            )?);
        }
        Ok(extents)
    }

    fn build_extent(
        &mut self,
        path: PathBuf,
        start: u64,
        end: u64,
        slices: Vec<(usize, ExpertTensorSlice)>,
    ) -> Result<DirectReadExtent> {
        let aligned_offset = start / DIRECT_IO_ALIGNMENT as u64 * DIRECT_IO_ALIGNMENT as u64;
        let required_end = usize::try_from(end - aligned_offset).map_err(|_| Error::Model {
            message: "expert direct-read extent exceeds usize".into(),
        })?;
        let aligned_len = align_up(required_end, DIRECT_IO_ALIGNMENT)?;
        if aligned_len > self.buffer_bytes {
            return Err(Error::Model {
                message: format!(
                    "expert direct-read extent exceeds registered buffer: aligned={aligned_len} buffer={} path={}",
                    self.buffer_bytes,
                    path.display()
                ),
            });
        }
        let file_index = self.register_file(&path)?;
        let views = slices
            .into_iter()
            .map(|(slice_index, slice)| {
                let payload_offset =
                    usize::try_from(slice.offset - aligned_offset).map_err(|_| Error::Model {
                        message: "expert tensor alignment offset exceeds usize".into(),
                    })?;
                let payload_len = usize::try_from(slice.bytes).map_err(|_| Error::Model {
                    message: "expert tensor slice size exceeds usize".into(),
                })?;
                Ok(DirectReadView {
                    slice_index,
                    slice,
                    payload_offset,
                    payload_len,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DirectReadExtent {
            file_index,
            aligned_offset,
            aligned_len,
            required_end,
            views,
        })
    }

    #[allow(unsafe_code)]
    fn push_read(
        &mut self,
        extent_index: usize,
        buffer_index: usize,
        extent: &DirectReadExtent,
    ) -> Result<()> {
        let len = u32::try_from(extent.aligned_len).map_err(|_| Error::Model {
            message: "expert direct-read length exceeds u32".into(),
        })?;
        let buffer = &mut self.buffers[buffer_index];
        let user_data = ((extent_index as u64) << 32) | buffer_index as u64;
        let entry = opcode::ReadFixed::new(
            types::Fixed(extent.file_index),
            buffer.as_mut_ptr()?,
            len,
            buffer_index as u16,
        )
        .offset(extent.aligned_offset)
        .build()
        .user_data(user_data);
        let mut submission = self.ring.submission();
        // SAFETY: the fixed file and buffer registrations remain alive for the
        // entire state lifetime. A buffer is used by at most one SQE per wave and
        // is not read or reused until every CQE in that wave has been collected.
        unsafe { submission.push(&entry) }.map_err(|_| Error::Model {
            message: "expert io_uring submission queue is full".into(),
        })
    }

    #[cfg(feature = "cuda")]
    fn physical_resource_capacity(&self) -> Result<Option<MaterializationResourceRequirements>> {
        if !self
            .buffers
            .first()
            .is_some_and(RegisteredBuffer::is_cuda_pinned)
        {
            return Ok(None);
        }
        let pinned_host_bytes = self
            .buffers
            .len()
            .checked_mul(self.buffer_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Error::Model {
                message: "pinned io_uring capacity exceeds u64".into(),
            })?;
        Ok(Some(MaterializationResourceRequirements {
            read_slots: self.queue_depth as u64,
            storage_read_bytes: pinned_host_bytes,
            pinned_host_bytes,
            ..MaterializationResourceRequirements::default()
        }))
    }

    #[cfg(feature = "cuda")]
    fn plan_slices_pinned(&mut self, slices: &[ExpertTensorSlice]) -> Result<PinnedExpertReadPlan> {
        self.progress_detached_pinned_operations(self.queue_depth.max(1))?;
        if !self
            .buffers
            .first()
            .is_some_and(RegisteredBuffer::is_cuda_pinned)
        {
            return Err(Error::Model {
                message: "io_uring expert reader was not configured with CUDA pinned slabs".into(),
            });
        }
        let payload_bytes = slices.iter().try_fold(0u64, |total, slice| {
            total.checked_add(slice.bytes).ok_or_else(|| Error::Model {
                message: "pinned expert payload size overflow".into(),
            })
        })?;
        let extents = self.prepare_read(slices)?;
        validate_pinned_extents(
            &extents,
            self.buffers.len(),
            self.buffer_bytes,
            self.files.len(),
        )?;
        let storage_read_bytes = extents.iter().try_fold(0u64, |total, extent| {
            total
                .checked_add(extent.aligned_len as u64)
                .ok_or_else(|| Error::Model {
                    message: "aligned pinned read size overflow".into(),
                })
        })?;
        let pinned_host_bytes = u64::try_from(
            extents
                .len()
                .checked_mul(self.buffer_bytes)
                .ok_or_else(|| Error::Model {
                    message: "pinned slab demand overflow".into(),
                })?,
        )
        .map_err(|_| Error::Model {
            message: "pinned slab demand exceeds u64".into(),
        })?;
        Ok(PinnedExpertReadPlan {
            payload_count: slices.len(),
            extents,
            requirements: MaterializationResourceRequirements {
                read_slots: 1,
                storage_read_bytes,
                pinned_host_bytes,
                upload_slots: 1,
                h2d_bytes: payload_bytes,
                install_slots: 1,
                device_install_bytes: payload_bytes,
            },
        })
    }

    #[cfg(feature = "cuda")]
    fn reserve_slices_pinned(
        &mut self,
        plan: PinnedExpertReadPlan,
        protocol_operation: OperationId,
        key: MaterializationKey,
    ) -> Result<ReservedPinnedExpertRead> {
        if protocol_operation.is_zero() {
            return Err(Error::Model {
                message: "physical pinned read requires a non-zero registry operation".into(),
            });
        }
        if plan.extents.is_empty() {
            return Err(Error::Model {
                message: "physical pinned read requires at least one storage extent".into(),
            });
        }
        let operation_id = allocate_durable_id(
            &mut self.next_pinned_operation_id,
            "pinned expert read operation",
        )?;
        let operation = PinnedReadOperation::physical(plan.extents, plan.payload_count);
        self.pinned_operations.insert(operation_id, operation);
        match self.reserve_pinned_operation_buffers(operation_id) {
            Ok(true) => {}
            Ok(false) => {
                self.pinned_operations.remove(&operation_id);
                return Err(Error::Execution {
                    message: "registered pinned expert slabs are temporarily exhausted".into(),
                });
            }
            Err(error) => {
                let Some(mut operation) = self.pinned_operations.remove(&operation_id) else {
                    return Err(Error::Internal {
                        message: format!(
                            "physical pinned slab reservation failed ({error}) after its operation disappeared"
                        ),
                    });
                };
                let released = operation.release_unqueued_reservations();
                if let Err(release_error) = self.release_pinned_buffer_reservations(released) {
                    return Err(Error::Internal {
                        message: format!(
                            "physical pinned slab reservation failed ({error}); reservation cleanup also failed ({release_error})"
                        ),
                    });
                }
                return Err(error);
            }
        }

        let descriptor_result = (|| {
            let operation =
                self.pinned_operations
                    .get(&operation_id)
                    .ok_or_else(|| Error::Internal {
                        message: "physical pinned operation disappeared while describing its slabs"
                            .into(),
                    })?;
            operation
                .extents
                .iter()
                .enumerate()
                .map(|(extent_index, extent)| {
                    let buffer_index = operation.reserved_buffer(extent_index)?;
                    let buffer =
                        self.buffers
                            .get_mut(buffer_index)
                            .ok_or_else(|| Error::Internal {
                                message: format!(
                                    "physical pinned operation reserved missing slab {buffer_index}"
                                ),
                            })?;
                    let identity = u64::try_from(buffer_index)
                        .ok()
                        .and_then(|value| value.checked_add(1))
                        .ok_or_else(|| Error::Internal {
                            message: "pinned slab identity overflow".into(),
                        })?;
                    RegisteredPinnedAlignedSlabLeaseDescriptor::new(
                        protocol_operation,
                        SlabId::new(identity),
                        RegistrationId::new(identity),
                        buffer.as_mut_ptr()? as usize,
                        u64::try_from(buffer.len).map_err(|_| Error::Model {
                            message: "registered pinned slab length exceeds u64".into(),
                        })?,
                        0,
                        u64::try_from(extent.aligned_len).map_err(|_| Error::Model {
                            message: "registered pinned extent length exceeds u64".into(),
                        })?,
                        DIRECT_IO_ALIGNMENT,
                        key.source_generation(),
                        key.destination_generation(),
                    )
                    .map_err(Into::into)
                })
                .collect::<Result<Vec<_>>>()
        })();
        let slabs = match descriptor_result {
            Ok(slabs) => slabs,
            Err(error) => {
                let Some(mut operation) = self.pinned_operations.remove(&operation_id) else {
                    return Err(Error::Internal {
                        message: format!(
                            "physical pinned slab description failed ({error}) after its operation disappeared"
                        ),
                    });
                };
                let released = operation.release_unqueued_reservations();
                if let Err(release_error) = self.release_pinned_buffer_reservations(released) {
                    return Err(Error::Internal {
                        message: format!(
                            "physical pinned slab description failed ({error}); reservation cleanup also failed ({release_error})"
                        ),
                    });
                }
                return Err(error);
            }
        };
        self.account_pinned_operation_timing(operation_id);
        Ok(ReservedPinnedExpertRead {
            ticket: PinnedExpertReadTicket { operation_id },
            slabs: slabs.into_boxed_slice(),
        })
    }

    #[cfg(feature = "cuda")]
    fn submit_reserved_slices_pinned(&mut self, ticket: PinnedExpertReadTicket) -> Result<()> {
        let operation = self
            .pinned_operations
            .get_mut(&ticket.operation_id)
            .ok_or_else(|| Error::Model {
                message: format!(
                    "unknown or already consumed pinned expert read operation {}",
                    ticket.operation_id
                ),
            })?;
        operation.authorize_submission()?;
        self.pinned_operation_order.push_back(ticket.operation_id);
        self.schedule_pinned_reads()?;
        self.submit_queued_pinned_reads()?;
        self.detect_stuck_pinned_operation(ticket.operation_id)
    }

    #[cfg(feature = "cuda")]
    fn poll_slices_pinned(
        &mut self,
        ticket: PinnedExpertReadTicket,
        max_completions: usize,
    ) -> Result<PinnedExpertReadPoll> {
        if max_completions == 0 {
            return Err(Error::Model {
                message: "pinned expert read poll completion budget must be greater than zero"
                    .into(),
            });
        }
        if !self.pinned_operations.contains_key(&ticket.operation_id) {
            return Err(Error::Model {
                message: format!(
                    "unknown or already consumed pinned expert read operation {}",
                    ticket.operation_id
                ),
            });
        }

        self.drain_pinned_completions(max_completions)?;
        self.schedule_pinned_reads()?;
        self.submit_queued_pinned_reads()?;
        self.reap_detached_pinned_operations();
        self.detect_stuck_pinned_operation(ticket.operation_id)?;
        self.account_pinned_operation_timing(ticket.operation_id);
        self.take_pinned_operation_result(ticket.operation_id)
    }

    #[cfg(feature = "cuda")]
    fn cancel_slices_pinned(&mut self, ticket: PinnedExpertReadTicket) -> Result<bool> {
        let released = self
            .pinned_operations
            .get_mut(&ticket.operation_id)
            .ok_or_else(|| Error::Model {
                message: format!(
                    "unknown or already consumed pinned expert read operation {}",
                    ticket.operation_id
                ),
            })?
            .cancel();
        let Some(released) = released else {
            return Ok(false);
        };
        self.pinned_operation_order
            .retain(|operation_id| *operation_id != ticket.operation_id);
        self.release_pinned_buffer_reservations(released)?;
        self.account_pinned_operation_timing(ticket.operation_id);
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn detach_slices_pinned(&mut self, ticket: PinnedExpertReadTicket) -> Result<()> {
        let operation_id = ticket.operation_id;
        if !self.pinned_operations.contains_key(&operation_id) {
            return Ok(());
        }
        if !self.detached_pinned_operations.contains(&operation_id) {
            self.detached_pinned_operations.push_back(operation_id);
        }
        let cancel_error = self.cancel_slices_pinned(ticket).err();
        let progress_error = self
            .progress_detached_pinned_operations(self.queue_depth.max(1))
            .err();
        match (cancel_error, progress_error) {
            (Some(cancel), Some(progress)) => Err(Error::Internal {
                message: format!(
                    "detaching pinned expert read failed ({cancel}); reaping it also failed ({progress})"
                ),
            }),
            (Some(error), None) | (None, Some(error)) => Err(error),
            (None, None) => Ok(()),
        }
    }

    #[cfg(feature = "cuda")]
    fn reap_detached_pinned_operations(&mut self) -> usize {
        reap_detached_pinned_operations(
            &mut self.pinned_operations,
            &mut self.detached_pinned_operations,
        )
    }

    #[cfg(feature = "cuda")]
    fn progress_detached_pinned_operations(&mut self, max_completions: usize) -> Result<usize> {
        self.submit_queued_pinned_reads()?;
        self.drain_pinned_completions(max_completions)?;
        Ok(self.reap_detached_pinned_operations())
    }

    #[cfg(feature = "cuda")]
    fn react_to_pinned_completions(&mut self) -> Result<()> {
        loop {
            let drained = self.drain_pinned_completions(usize::MAX)?;
            self.schedule_pinned_reads()?;
            self.submit_queued_pinned_reads()?;
            self.reap_detached_pinned_operations();
            if drained == 0 {
                return Ok(());
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn release_pinned_buffer_reservations(&mut self, buffer_indices: Vec<usize>) -> Result<()> {
        for buffer_index in buffer_indices {
            let busy = self
                .pinned_buffer_busy
                .get_mut(buffer_index)
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "pinned expert reservation references missing slab {buffer_index}"
                    ),
                })?;
            if !*busy {
                return Err(Error::Internal {
                    message: format!("pinned expert reservation released idle slab {buffer_index}"),
                });
            }
            *busy = false;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn fail_pinned_operation(&mut self, operation_id: u64, error: Error) -> Result<()> {
        let released = self
            .pinned_operations
            .get_mut(&operation_id)
            .ok_or_else(|| Error::Internal {
                message: format!("cannot fail missing pinned expert operation {operation_id}"),
            })?
            .fail(error);
        self.pinned_operation_order
            .retain(|queued| *queued != operation_id);
        self.release_pinned_buffer_reservations(released)?;
        self.account_pinned_operation_timing(operation_id);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn reserve_pinned_operation_buffers(&mut self, operation_id: u64) -> Result<bool> {
        let reusable = self
            .buffers
            .iter()
            .map(RegisteredBuffer::is_available)
            .collect::<Vec<_>>();
        let operation = self
            .pinned_operations
            .get_mut(&operation_id)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "cannot reserve slabs for missing pinned expert operation {operation_id}"
                ),
            })?;
        reserve_all_extent_buffers(operation, &mut self.pinned_buffer_busy, &reusable)
    }

    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn schedule_pinned_reads(&mut self) -> Result<()> {
        let mut consecutively_blocked = 0usize;
        while self.pinned_submissions.len() < self.queue_depth {
            let sq_has_capacity = {
                let submission = self.ring.submission();
                !submission.is_full()
            };
            if !sq_has_capacity {
                break;
            }

            let Some(operation_id) = self.pinned_operation_order.pop_front() else {
                break;
            };
            let can_submit = self
                .pinned_operations
                .get(&operation_id)
                .is_some_and(PinnedReadOperation::can_submit);
            if !can_submit {
                continue;
            }
            if self.pinned_operations[&operation_id].has_no_reservations()
                && !self.reserve_pinned_operation_buffers(operation_id)?
            {
                self.pinned_operation_order.push_back(operation_id);
                consecutively_blocked = consecutively_blocked.saturating_add(1);
                if consecutively_blocked >= self.pinned_operation_order.len() {
                    break;
                }
                continue;
            }
            if !self.pinned_operations[&operation_id].has_all_reservations() {
                self.fail_pinned_operation(
                    operation_id,
                    Error::Internal {
                        message:
                            "pinned expert read reached scheduling with a partial slab reservation"
                                .into(),
                    },
                )?;
                continue;
            }
            consecutively_blocked = 0;

            let extent_index = self.pinned_operations[&operation_id].next_extent;
            let buffer_index =
                self.pinned_operations[&operation_id].reserved_buffer(extent_index)?;
            let extent = &self.pinned_operations[&operation_id].extents[extent_index];
            let len = u32::try_from(extent.aligned_len)
                .expect("pinned expert extent length was prevalidated");
            let file_index = extent.file_index;
            let aligned_offset = extent.aligned_offset;
            let buffer_ptr = match self.buffers[buffer_index].as_mut_ptr() {
                Ok(pointer) => pointer,
                Err(error) => {
                    self.fail_pinned_operation(operation_id, error)?;
                    continue;
                }
            };
            let submission_id = match allocate_durable_id(
                &mut self.next_pinned_submission_id,
                "pinned expert read submission",
            ) {
                Ok(submission_id) => submission_id,
                Err(error) => {
                    self.fail_pinned_operation(operation_id, error)?;
                    continue;
                }
            };
            let entry = opcode::ReadFixed::new(
                types::Fixed(file_index),
                buffer_ptr,
                len,
                u16::try_from(buffer_index)
                    .expect("pinned expert fixed-buffer index was prevalidated"),
            )
            .offset(aligned_offset)
            .build()
            .user_data(submission_id);
            let pushed = {
                let mut submission = self.ring.submission();
                // SAFETY: the fixed file and buffer registrations remain alive for
                // the state lifetime. Every extent owns a distinct reserved slab,
                // retained until its exact CQE is drained.
                unsafe { submission.push(&entry) }.is_ok()
            };
            if !pushed {
                self.pinned_operation_order.push_front(operation_id);
                break;
            }

            self.pinned_submissions.insert(
                submission_id,
                PinnedReadSubmission {
                    operation_id,
                    extent_index,
                    buffer_index,
                    state: PinnedReadSubmissionState::Queued,
                },
            );
            self.pinned_queued_submissions.push_back(submission_id);
            self.pinned_operations
                .get_mut(&operation_id)
                .expect("queued pinned operation exists")
                .record_submission(extent_index)
                .expect("pinned scheduler submits the next prevalidated extent");
            if self.pinned_operations[&operation_id].can_submit() {
                self.pinned_operation_order.push_back(operation_id);
            }

            self.stats.submitted_extents = self.stats.submitted_extents.saturating_add(1);
            self.stats.aligned_bytes = self.stats.aligned_bytes.saturating_add(u64::from(len));
            self.stats.peak_queue_depth = self
                .stats
                .peak_queue_depth
                .max(self.pinned_submissions.len());
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn submit_queued_pinned_reads(&mut self) -> Result<usize> {
        if self.pinned_queued_submissions.is_empty() {
            return Ok(0);
        }
        let accepted = self.ring.submit().map_err(|error| Error::Model {
            message: format!("submit expert io_uring reads: {error}"),
        })?;
        self.confirm_queued_pinned_submissions(accepted)?;
        Ok(accepted)
    }

    #[cfg(feature = "cuda")]
    fn confirm_queued_pinned_submissions(&mut self, accepted: usize) -> Result<()> {
        let confirmed = accepted.min(self.pinned_queued_submissions.len());
        for _ in 0..confirmed {
            let submission_id = self
                .pinned_queued_submissions
                .pop_front()
                .expect("confirmed queued submission exists");
            let submission = self
                .pinned_submissions
                .get_mut(&submission_id)
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "confirmed pinned submission {submission_id} has no ownership metadata"
                    ),
                })?;
            submission.state = PinnedReadSubmissionState::Submitted;
        }
        if accepted > confirmed {
            return Err(Error::Internal {
                message: format!(
                    "io_uring accepted {accepted} pinned submissions but only {confirmed} were tracked as queued"
                ),
            });
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn detect_stuck_pinned_operation(&mut self, operation_id: u64) -> Result<()> {
        let Some(operation) = self.pinned_operations.get(&operation_id) else {
            return Err(Error::Model {
                message: format!(
                    "unknown or already consumed pinned expert read operation {operation_id}"
                ),
            });
        };
        if operation.is_terminal() {
            return Ok(());
        }
        let tracked = self
            .pinned_submissions
            .values()
            .filter(|submission| submission.operation_id == operation_id)
            .count();
        let extent_count = operation.extents.len();
        let outstanding = operation.outstanding;
        let has_no_reservations = operation.has_no_reservations();
        let has_all_reservations = operation.has_all_reservations();
        let can_submit = operation.can_submit();

        let reason = if extent_count > self.buffers.len() {
            Some(format!(
                "pinned expert read requires {extent_count} slabs but only {} exist",
                self.buffers.len()
            ))
        } else if !has_no_reservations && !has_all_reservations {
            Some("pinned expert read has a partial slab reservation".into())
        } else if tracked != outstanding {
            Some(format!(
                "pinned expert read ownership mismatch: tracked={tracked} outstanding={outstanding}"
            ))
        } else if tracked == 0 && can_submit {
            let scheduler_can_queue = self.pinned_submissions.len() < self.queue_depth && {
                let submission = self.ring.submission();
                !submission.is_full()
            };
            let reusable = has_no_reservations
                .then(|| {
                    self.buffers
                        .iter()
                        .enumerate()
                        .filter(|(index, buffer)| {
                            !self.pinned_buffer_busy[*index] && buffer.is_available()
                        })
                        .count()
                })
                .unwrap_or(0);
            unscheduled_pinned_operation_reason(
                scheduler_can_queue,
                has_all_reservations,
                has_no_reservations,
                reusable,
                extent_count,
            )
        } else if tracked == 0 && !can_submit {
            Some("nonterminal pinned expert read has no submitted or queued work".into())
        } else {
            None
        };

        if let Some(reason) = reason {
            self.fail_pinned_operation(operation_id, Error::Internal { message: reason })?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn drain_pinned_completions(&mut self, max_completions: usize) -> Result<usize> {
        let completions = {
            let mut completion = self.ring.completion();
            completion
                .by_ref()
                .take(max_completions)
                .map(|entry| (entry.user_data(), entry.result()))
                .collect::<Vec<_>>()
        };
        let mut first_error = None;
        for (submission_id, result) in completions.iter().copied() {
            if let Err(error) = self.handle_pinned_completion(submission_id, result) {
                first_error.get_or_insert(error);
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(completions.len())
    }

    #[cfg(feature = "cuda")]
    fn handle_pinned_completion(&mut self, submission_id: u64, result: i32) -> Result<()> {
        let submission = self
            .pinned_submissions
            .remove(&submission_id)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "io_uring returned unknown pinned expert submission {submission_id}"
                ),
            })?;
        if submission.state == PinnedReadSubmissionState::Queued {
            self.pinned_queued_submissions
                .retain(|queued| *queued != submission_id);
        }
        let assigned_buffer = self
            .pinned_operations
            .get(&submission.operation_id)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "pinned expert submission {submission_id} references missing operation {}",
                    submission.operation_id
                ),
            })?
            .reserved_buffer(submission.extent_index)?;
        if assigned_buffer != submission.buffer_index {
            return Err(Error::Internal {
                message: format!(
                    "pinned expert submission {submission_id} completed slab {} but extent {} reserved slab {assigned_buffer}",
                    submission.buffer_index, submission.extent_index
                ),
            });
        }
        let Some(busy) = self.pinned_buffer_busy.get(submission.buffer_index) else {
            return Err(Error::Internal {
                message: format!(
                    "pinned expert submission {submission_id} references missing slab {}",
                    submission.buffer_index
                ),
            });
        };
        if !*busy {
            return Err(Error::Internal {
                message: format!(
                    "pinned expert submission {submission_id} completed an idle slab {}",
                    submission.buffer_index
                ),
            });
        }

        let (skip_payload, required_end, path, views) = {
            let operation = self
                .pinned_operations
                .get(&submission.operation_id)
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "pinned expert submission {submission_id} references missing operation {}",
                        submission.operation_id
                    ),
                })?;
            let extent = operation
                .extents
                .get(submission.extent_index)
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "pinned expert submission {submission_id} references missing extent {}",
                        submission.extent_index
                    ),
                })?;
            let path = extent
                .views
                .first()
                .map(|view| view.slice.path.clone())
                .ok_or_else(|| Error::Model {
                    message: "expert direct-read extent has no views".into(),
                })?;
            let views = extent
                .views
                .iter()
                .map(|view| {
                    (
                        view.slice_index,
                        view.slice.clone(),
                        view.payload_offset,
                        view.payload_len,
                    )
                })
                .collect::<Vec<_>>();
            (
                operation.cancelled || operation.error.is_some(),
                extent.required_end,
                path,
                views,
            )
        };

        let payload_result = if skip_payload {
            None
        } else if result < 0 {
            Some(Err(Error::Model {
                message: format!(
                    "expert io_uring read '{}' failed: {}",
                    path.display(),
                    std::io::Error::from_raw_os_error(-result)
                ),
            }))
        } else if (result as usize) < required_end {
            Some(Err(Error::Model {
                message: format!(
                    "short expert io_uring read '{}': got {}, need at least {required_end}",
                    path.display(),
                    result
                ),
            }))
        } else {
            Some(
                views
                    .into_iter()
                    .map(|(slice_index, slice, payload_offset, payload_len)| {
                        Ok((
                            slice_index,
                            PinnedExpertTensorPayload {
                                slice,
                                bytes: self.buffers[submission.buffer_index]
                                    .pinned_range(payload_offset, payload_len)?,
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>>>(),
            )
        };

        let mut failed_extent = false;
        let mut completed_extent = false;
        let released = {
            let operation = self
                .pinned_operations
                .get_mut(&submission.operation_id)
                .expect("completion operation validated above");
            operation.record_completion(submission.extent_index)?;
            match payload_result {
                None => Vec::new(),
                Some(Err(error)) => {
                    failed_extent = true;
                    operation.fail(error)
                }
                Some(Ok(payloads)) => {
                    let mut payload_error = None;
                    for (slice_index, payload) in payloads {
                        let Some(slot) = operation.payloads.get_mut(slice_index) else {
                            payload_error = Some(Error::Internal {
                                message: format!(
                                    "pinned expert payload references missing slice {slice_index}"
                                ),
                            });
                            break;
                        };
                        if slot.replace(payload).is_some() {
                            payload_error = Some(Error::Internal {
                                message: format!(
                                    "pinned expert payload produced slice {slice_index} more than once"
                                ),
                            });
                            break;
                        }
                    }
                    if let Some(error) = payload_error {
                        failed_extent = true;
                        operation.fail(error)
                    } else {
                        completed_extent = true;
                        Vec::new()
                    }
                }
            }
        };
        self.release_pinned_buffer_reservations(vec![submission.buffer_index])?;
        self.release_pinned_buffer_reservations(released)?;
        if failed_extent {
            self.stats.failed_extents = self.stats.failed_extents.saturating_add(1);
        }
        if completed_extent {
            self.stats.completed_extents = self.stats.completed_extents.saturating_add(1);
        }
        self.account_pinned_operation_timing(submission.operation_id);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn account_pinned_operation_timing(&mut self, operation_id: u64) {
        let elapsed = self
            .pinned_operations
            .get_mut(&operation_id)
            .and_then(PinnedReadOperation::take_elapsed_if_terminal);
        if let Some(elapsed) = elapsed {
            self.stats.read_us = self
                .stats
                .read_us
                .saturating_add(elapsed.as_micros().min(u128::from(u64::MAX)) as u64);
        }
    }

    #[cfg(feature = "cuda")]
    fn take_pinned_operation_result(&mut self, operation_id: u64) -> Result<PinnedExpertReadPoll> {
        let terminal = self
            .pinned_operations
            .get(&operation_id)
            .is_some_and(PinnedReadOperation::is_terminal);
        if !terminal {
            return Ok(PinnedExpertReadPoll::Pending);
        }
        let mut operation =
            take_terminal_pinned_operation(&mut self.pinned_operations, operation_id)
                .expect("terminal pinned operation exists");
        if operation.cancelled {
            return Ok(PinnedExpertReadPoll::Cancelled);
        }
        if let Some(error) = operation.error.take() {
            return Ok(PinnedExpertReadPoll::Failed(error));
        }
        match collect_payloads(operation.payloads) {
            Ok(payloads) => Ok(PinnedExpertReadPoll::Ready(PinnedExpertReadResult {
                payloads,
            })),
            Err(error) => Ok(PinnedExpertReadPoll::Failed(error)),
        }
    }

    fn execute_wave(
        &mut self,
        extents: &[DirectReadExtent],
        wave_start: usize,
        wave_end: usize,
    ) -> Result<Vec<(usize, usize)>> {
        let wave_len = wave_end - wave_start;
        let buffer_indices = self
            .buffers
            .iter()
            .enumerate()
            .filter_map(|(index, buffer)| buffer.is_available().then_some(index))
            .take(wave_len)
            .collect::<Vec<_>>();
        if buffer_indices.len() != wave_len {
            self.stats.slab_exhaustions = self.stats.slab_exhaustions.saturating_add(1);
            return Err(Error::Model {
                message: format!(
                    "expert io_uring pinned slab pool exhausted: available={} required={wave_len}",
                    buffer_indices.len()
                ),
            });
        }

        for (extent_index, &buffer_index) in (wave_start..wave_end).zip(buffer_indices.iter()) {
            self.push_read(extent_index, buffer_index, &extents[extent_index])?;
        }
        self.stats.submitted_extents = self.stats.submitted_extents.saturating_add(wave_len as u64);
        self.stats.aligned_bytes = self.stats.aligned_bytes.saturating_add(
            extents[wave_start..wave_end]
                .iter()
                .map(|extent| extent.aligned_len as u64)
                .sum::<u64>(),
        );
        self.stats.peak_queue_depth = self.stats.peak_queue_depth.max(wave_len);
        if let Err(error) = self.ring.submit_and_wait(wave_len) {
            self.stats.failed_extents = self.stats.failed_extents.saturating_add(wave_len as u64);
            return Err(Error::Model {
                message: format!("submit/wait expert io_uring reads: {error}"),
            });
        }

        let completions = {
            let mut completion = self.ring.completion();
            completion
                .by_ref()
                .take(wave_len)
                .map(|entry| (entry.user_data(), entry.result()))
                .collect::<Vec<_>>()
        };
        if completions.len() != wave_len {
            self.stats.failed_extents = self
                .stats
                .failed_extents
                .saturating_add((wave_len - completions.len()) as u64);
            return Err(Error::Model {
                message: format!(
                    "expert io_uring completion underflow: got {} expected {wave_len}",
                    completions.len()
                ),
            });
        }

        let mut completed = Vec::with_capacity(wave_len);
        for (user_data, result) in completions {
            let extent_index = (user_data >> 32) as usize;
            let buffer_index = (user_data & 0xffff_ffff) as usize;
            let extent = extents.get(extent_index).ok_or_else(|| Error::Model {
                message: format!("expert io_uring returned invalid extent index {extent_index}"),
            })?;
            if buffer_index >= self.buffers.len() {
                return Err(Error::Model {
                    message: format!(
                        "expert io_uring returned invalid buffer index {buffer_index}"
                    ),
                });
            }
            let path = extent
                .views
                .first()
                .map(|view| view.slice.path.as_path())
                .ok_or_else(|| Error::Model {
                    message: "expert direct-read extent has no views".into(),
                })?;
            if result < 0 {
                self.stats.failed_extents = self.stats.failed_extents.saturating_add(1);
                return Err(Error::Model {
                    message: format!(
                        "expert io_uring read '{}' at {} failed: {}",
                        path.display(),
                        extent.aligned_offset,
                        std::io::Error::from_raw_os_error(-result)
                    ),
                });
            }
            let bytes_read = result as usize;
            if bytes_read < extent.required_end {
                self.stats.failed_extents = self.stats.failed_extents.saturating_add(1);
                return Err(Error::Model {
                    message: format!(
                        "short expert io_uring read '{}': got {bytes_read}, need at least {}",
                        path.display(),
                        extent.required_end
                    ),
                });
            }
            self.stats.completed_extents = self.stats.completed_extents.saturating_add(1);
            completed.push((extent_index, buffer_index));
        }
        Ok(completed)
    }

    fn prepare_read(&mut self, slices: &[ExpertTensorSlice]) -> Result<Vec<DirectReadExtent>> {
        let extents = self.plan(slices)?;
        self.stats.requested_bytes = self
            .stats
            .requested_bytes
            .saturating_add(slices.iter().map(|slice| slice.bytes).sum::<u64>());
        self.stats.coalesced_slices = self
            .stats
            .coalesced_slices
            .saturating_add(slices.len().saturating_sub(extents.len()) as u64);
        Ok(extents)
    }

    fn read_slices(&mut self, slices: &[ExpertTensorSlice]) -> Result<Vec<ExpertTensorPayload>> {
        if slices.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "cuda")]
        self.progress_detached_pinned_operations(self.queue_depth.max(1))?;
        #[cfg(feature = "cuda")]
        if !self.pinned_operations.is_empty() {
            return Err(Error::Model {
                message:
                    "pageable io_uring reads cannot run while pinned read operations are active"
                        .into(),
            });
        }
        let started = Instant::now();
        let result = (|| {
            let extents = self.prepare_read(slices)?;
            let mut payloads = vec![None; slices.len()];
            for wave_start in (0..extents.len()).step_by(self.queue_depth) {
                let wave_end = (wave_start + self.queue_depth).min(extents.len());
                for (extent_index, buffer_index) in
                    self.execute_wave(&extents, wave_start, wave_end)?
                {
                    for view in &extents[extent_index].views {
                        let bytes = self.buffers[buffer_index]
                            .copy_range(view.payload_offset, view.payload_len)?;
                        payloads[view.slice_index] = Some(ExpertTensorPayload {
                            slice: view.slice.clone(),
                            bytes,
                        });
                    }
                }
            }
            collect_payloads(payloads)
        })();
        self.stats.read_us = self
            .stats
            .read_us
            .saturating_add(started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
        result
    }
}

fn collect_payloads<T>(payloads: Vec<Option<T>>) -> Result<Vec<T>> {
    payloads
        .into_iter()
        .enumerate()
        .map(|(index, payload)| {
            payload.ok_or_else(|| Error::Model {
                message: format!("expert io_uring did not produce payload for slice {index}"),
            })
        })
        .collect()
}

impl Drop for IoUringDirectState {
    fn drop(&mut self) {
        let _ = self.unregister_completion_eventfd();
        let _ = self.ring.submitter().unregister_buffers();
        let _ = self.ring.submitter().unregister_files();
    }
}

pub(crate) struct IoUringExpertReader {
    state: Mutex<IoUringDirectState>,
    completion_hub: CompletionHub,
    _completion_eventfd: OwnedFd,
    reactor_eventfd: Mutex<Option<OwnedFd>>,
}

impl IoUringExpertReader {
    pub(crate) fn new(
        queue_depth: usize,
        buffer_bytes: usize,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        Self::from_state(
            IoUringDirectState::new(queue_depth, buffer_bytes)?,
            completion_hub,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn new_cuda_pinned(
        queue_depth: usize,
        buffer_bytes: usize,
        slab_count: usize,
        allocator: &CudaPinnedHostAllocator,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        Self::from_state(
            IoUringDirectState::new_cuda_pinned(queue_depth, buffer_bytes, slab_count, allocator)?,
            completion_hub,
        )
    }

    fn from_state(mut state: IoUringDirectState, completion_hub: CompletionHub) -> Result<Self> {
        let (completion_eventfd, reactor_eventfd) = create_completion_eventfd_pair()?;
        state.register_completion_eventfd(&completion_eventfd)?;
        Ok(Self {
            state: Mutex::new(state),
            completion_hub,
            _completion_eventfd: completion_eventfd,
            reactor_eventfd: Mutex::new(Some(reactor_eventfd)),
        })
    }

    pub(crate) fn take_completion_reactor(self: &Arc<Self>) -> Option<ModelCompletionReactor> {
        let reactor_eventfd = self
            .reactor_eventfd
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()?;
        let reader = Arc::clone(self);
        let completion_hub = self.completion_hub.clone();
        let registration = IoUringCompletionEventfdRegistration {
            reader: Arc::clone(&reader),
        };
        Some(Box::pin(async move {
            let _registration = registration;
            let eventfd =
                tokio::io::unix::AsyncFd::new(reactor_eventfd).map_err(|error| Error::Model {
                    message: format!("attach expert io_uring completion eventfd to Tokio: {error}"),
                })?;
            loop {
                let mut ready = eventfd.readable().await.map_err(|error| Error::Model {
                    message: format!("await expert io_uring completion eventfd: {error}"),
                })?;
                drain_completion_eventfd(eventfd.get_ref())?;
                ready.clear_ready();
                let progress = reader.react_to_completions();
                completion_hub.notify();
                progress?;
            }
        }))
    }

    fn react_to_completions(&self) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| Error::Model {
            message: "expert io_uring state lock poisoned".into(),
        })?;
        #[cfg(feature = "cuda")]
        state.react_to_pinned_completions()?;
        #[cfg(not(feature = "cuda"))]
        let _ = &mut *state;
        Ok(())
    }

    fn unregister_completion_eventfd(&self) {
        if let Ok(mut state) = self.state.lock() {
            let _ = state.unregister_completion_eventfd();
        }
    }

    pub(crate) fn read_slices(
        &self,
        slices: &[ExpertTensorSlice],
    ) -> Result<Vec<ExpertTensorPayload>> {
        self.state
            .lock()
            .map_err(|_| Error::Model {
                message: "expert io_uring state lock poisoned".into(),
            })?
            .read_slices(slices)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn plan_slices_pinned(
        &self,
        slices: &[ExpertTensorSlice],
    ) -> Result<PinnedExpertReadPlan> {
        self.state
            .lock()
            .map_err(|_| Error::Model {
                message: "expert io_uring state lock poisoned".into(),
            })?
            .plan_slices_pinned(slices)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn reserve_slices_pinned(
        &self,
        plan: PinnedExpertReadPlan,
        operation: OperationId,
        key: MaterializationKey,
    ) -> Result<ReservedPinnedExpertRead> {
        self.state
            .lock()
            .map_err(|_| Error::Model {
                message: "expert io_uring state lock poisoned".into(),
            })?
            .reserve_slices_pinned(plan, operation, key)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn submit_reserved_slices_pinned(
        &self,
        ticket: PinnedExpertReadTicket,
    ) -> Result<()> {
        self.state
            .lock()
            .map_err(|_| Error::Model {
                message: "expert io_uring state lock poisoned".into(),
            })?
            .submit_reserved_slices_pinned(ticket)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cancel_slices_pinned(&self, ticket: PinnedExpertReadTicket) -> Result<bool> {
        let result = self
            .state
            .lock()
            .map_err(|_| Error::Model {
                message: "expert io_uring state lock poisoned".into(),
            })?
            .cancel_slices_pinned(ticket);
        self.completion_hub.notify();
        result
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn poll_slices_pinned(
        &self,
        ticket: PinnedExpertReadTicket,
        max_completions: usize,
    ) -> Result<PinnedExpertReadPoll> {
        let result = self
            .state
            .lock()
            .map_err(|_| Error::Model {
                message: "expert io_uring state lock poisoned".into(),
            })?
            .poll_slices_pinned(ticket, max_completions);
        if !matches!(&result, Ok(PinnedExpertReadPoll::Pending)) {
            self.completion_hub.notify();
        }
        result
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn detach_slices_pinned(&self, ticket: PinnedExpertReadTicket) -> Result<()> {
        let result = self
            .state
            .lock()
            .map_err(|_| Error::Model {
                message: "expert io_uring state lock poisoned".into(),
            })?
            .detach_slices_pinned(ticket);
        self.completion_hub.notify();
        result
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn physical_resource_capacity(
        &self,
    ) -> Result<Option<MaterializationResourceRequirements>> {
        self.state
            .lock()
            .map_err(|_| Error::Model {
                message: "expert io_uring state lock poisoned".into(),
            })?
            .physical_resource_capacity()
    }

    pub(crate) fn stats(&self) -> ExpertIoStats {
        self.state
            .lock()
            .map(|state| state.stats)
            .unwrap_or_default()
    }
}

struct IoUringCompletionEventfdRegistration {
    reader: Arc<IoUringExpertReader>,
}

impl Drop for IoUringCompletionEventfdRegistration {
    fn drop(&mut self) {
        self.reader.unregister_completion_eventfd();
    }
}

#[allow(unsafe_code)]
fn create_completion_eventfd_pair() -> Result<(OwnedFd, OwnedFd)> {
    let raw_eventfd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
    if raw_eventfd < 0 {
        return Err(Error::Model {
            message: format!(
                "create expert io_uring completion eventfd: {}",
                std::io::Error::last_os_error()
            ),
        });
    }
    // SAFETY: `eventfd` returned a new descriptor owned by this function.
    let completion_eventfd = unsafe { OwnedFd::from_raw_fd(raw_eventfd) };
    let raw_reactor_eventfd =
        unsafe { libc::fcntl(completion_eventfd.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
    if raw_reactor_eventfd < 0 {
        return Err(Error::Model {
            message: format!(
                "duplicate expert io_uring completion eventfd: {}",
                std::io::Error::last_os_error()
            ),
        });
    }
    // SAFETY: `F_DUPFD_CLOEXEC` returned a distinct descriptor owned here.
    let reactor_eventfd = unsafe { OwnedFd::from_raw_fd(raw_reactor_eventfd) };
    Ok((completion_eventfd, reactor_eventfd))
}

#[allow(unsafe_code)]
fn drain_completion_eventfd(eventfd: &OwnedFd) -> Result<()> {
    loop {
        let mut value = 0u64;
        let read = unsafe {
            libc::read(
                eventfd.as_raw_fd(),
                (&mut value as *mut u64).cast(),
                std::mem::size_of::<u64>(),
            )
        };
        if read == std::mem::size_of::<u64>() as isize {
            continue;
        }
        if read < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            if error.kind() == std::io::ErrorKind::WouldBlock {
                return Ok(());
            }
            return Err(Error::Model {
                message: format!("read expert io_uring completion eventfd: {error}"),
            });
        }
        return Err(Error::Model {
            message: format!("short expert io_uring completion eventfd read: got {read} bytes"),
        });
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::Model {
            message: format!("invalid direct-I/O alignment {alignment}"),
        });
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| Error::Model {
            message: "direct-I/O alignment overflow".into(),
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn synthetic_extents(count: usize) -> Vec<DirectReadExtent> {
        (0..count)
            .map(|index| DirectReadExtent {
                file_index: 0,
                aligned_offset: index as u64 * DIRECT_IO_ALIGNMENT as u64,
                aligned_len: DIRECT_IO_ALIGNMENT,
                required_end: DIRECT_IO_ALIGNMENT,
                views: Vec::new(),
            })
            .collect()
    }

    #[test]
    #[allow(unsafe_code)]
    fn completion_reactor_is_claimed_once_and_does_not_lose_an_early_wake() {
        let completion_hub = CompletionHub::new();
        let Ok(reader) = IoUringExpertReader::new(1, DIRECT_IO_ALIGNMENT, completion_hub.clone())
        else {
            return;
        };
        let reader = Arc::new(reader);
        let reactor = reader
            .take_completion_reactor()
            .expect("the io_uring completion reactor is initially claimable");
        assert!(reader.take_completion_reactor().is_none());

        let listener = completion_hub.listen();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .build()
            .expect("build current-thread Tokio runtime");
        let local = tokio::task::LocalSet::new();
        runtime.block_on(local.run_until(async move {
            let reactor_task = tokio::task::spawn_local(reactor);
            let value = 1u64;
            let written = unsafe {
                libc::write(
                    reader._completion_eventfd.as_raw_fd(),
                    (&value as *const u64).cast(),
                    std::mem::size_of::<u64>(),
                )
            };
            assert_eq!(written, std::mem::size_of::<u64>() as isize);
            assert!(matches!(
                listener.await,
                ferrule_common::CompletionWake::Progress(_)
            ));
            reactor_task.abort();
        }));
    }

    #[test]
    fn dropping_an_unpolled_completion_reactor_unregisters_eventfd() {
        let completion_hub = CompletionHub::new();
        let Ok(reader) = IoUringExpertReader::new(1, DIRECT_IO_ALIGNMENT, completion_hub) else {
            return;
        };
        let reader = Arc::new(reader);
        let reactor = reader
            .take_completion_reactor()
            .expect("the io_uring completion reactor is initially claimable");

        drop(reactor);

        assert!(
            !reader
                .state
                .lock()
                .expect("io_uring state lock")
                .completion_eventfd_registered
        );
    }

    #[test]
    fn durable_ids_are_monotonic_nonzero_and_fail_before_wraparound() {
        let mut next = 1;
        assert_eq!(allocate_durable_id(&mut next, "test").unwrap(), 1);
        assert_eq!(allocate_durable_id(&mut next, "test").unwrap(), 2);
        assert_eq!(next, 3);

        let mut exhausted = u64::MAX;
        assert!(allocate_durable_id(&mut exhausted, "test").is_err());
        assert_eq!(exhausted, u64::MAX);
    }

    fn reserve_test_operation<T>(
        operation: &mut PinnedReadOperation<T>,
        buffer_busy: &mut [bool],
        buffer_reusable: &[bool],
    ) -> bool {
        reserve_all_extent_buffers(operation, buffer_busy, buffer_reusable).unwrap()
    }

    #[test]
    fn queue_saturation_keeps_a_reservable_pinned_operation_waiting() {
        assert_eq!(
            unscheduled_pinned_operation_reason(false, false, true, 2, 2),
            None
        );
        assert!(
            unscheduled_pinned_operation_reason(true, false, true, 2, 2)
                .is_some_and(|reason| reason.contains("scheduling made no progress"))
        );
    }

    #[test]
    fn pinned_operation_accepts_out_of_order_extent_completions() {
        let mut operation = PinnedReadOperation::<()>::new(synthetic_extents(3), 0);
        let mut busy = vec![false; 3];
        assert!(reserve_test_operation(
            &mut operation,
            &mut busy,
            &[true; 3]
        ));
        for extent in 0..3 {
            operation.record_submission(extent).unwrap();
        }
        assert_eq!(operation.outstanding, 3);
        assert!(!operation.is_terminal());

        operation.record_completion(2).unwrap();
        assert_eq!(operation.outstanding, 2);
        operation.record_completion(0).unwrap();
        assert_eq!(operation.outstanding, 1);
        operation.record_completion(1).unwrap();

        assert_eq!(operation.outstanding, 0);
        assert!(operation.is_terminal());
        assert!(
            operation
                .completed_extents
                .iter()
                .all(|completed| *completed)
        );
    }

    #[test]
    fn physical_reservation_cannot_submit_before_explicit_authorization() {
        let mut operation = PinnedReadOperation::<()>::physical(synthetic_extents(2), 0);
        let mut busy = vec![false; 2];
        assert!(reserve_test_operation(
            &mut operation,
            &mut busy,
            &[true; 2]
        ));
        assert!(!operation.can_submit());
        assert!(operation.authorize_submission().is_ok());
        assert!(operation.can_submit());
        assert!(operation.authorize_submission().is_err());
    }

    #[test]
    fn cancelling_unsubmitted_physical_reservation_releases_every_slab() {
        let mut operation = PinnedReadOperation::<()>::physical(synthetic_extents(2), 0);
        let mut busy = vec![false; 2];
        assert!(reserve_test_operation(
            &mut operation,
            &mut busy,
            &[true; 2]
        ));
        let released = operation.cancel().unwrap();
        assert_eq!(released, vec![0, 1]);
        for index in released {
            busy[index] = false;
        }
        assert_eq!(busy, vec![false, false]);
        assert!(operation.is_terminal());
    }

    #[test]
    fn whole_operation_reservation_is_all_or_none_across_two_operations() {
        let mut first = PinnedReadOperation::<()>::new(synthetic_extents(3), 0);
        let mut second = PinnedReadOperation::<()>::new(synthetic_extents(2), 0);
        let mut busy = vec![false; 4];
        let mut reusable = vec![true; 4];

        assert!(reserve_test_operation(&mut first, &mut busy, &reusable));
        assert_eq!(first.extent_buffers, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(busy, vec![true, true, true, false]);

        first.record_submission(0).unwrap();
        first.record_completion(0).unwrap();
        busy[0] = false;
        reusable[0] = false;

        assert!(!reserve_test_operation(&mut second, &mut busy, &reusable));
        assert!(second.has_no_reservations());
        assert_eq!(busy, vec![false, true, true, false]);
        assert!(first.has_all_reservations());
    }

    #[test]
    fn pinned_extent_validation_rejects_an_operation_larger_than_the_slab_pool() {
        let error = validate_pinned_extents(&synthetic_extents(3), 2, DIRECT_IO_ALIGNMENT, 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("can never make progress"));
    }

    #[test]
    fn failed_operation_stops_new_submissions_and_releases_unqueued_slabs() {
        let mut operation = PinnedReadOperation::<()>::new(synthetic_extents(3), 0);
        let mut busy = vec![false; 3];
        assert!(reserve_test_operation(
            &mut operation,
            &mut busy,
            &[true; 3]
        ));
        operation.record_submission(0).unwrap();
        operation.record_submission(1).unwrap();
        let released = operation.fail(Error::Model {
            message: "synthetic read failure".into(),
        });

        assert_eq!(released, vec![2]);
        busy[released[0]] = false;
        assert_eq!(busy, vec![true, true, false]);
        assert!(!operation.can_submit());
        assert!(!operation.is_terminal());
        operation.record_completion(1).unwrap();
        assert!(!operation.is_terminal());
        operation.record_completion(0).unwrap();
        assert!(operation.is_terminal());
        assert_eq!(operation.next_extent, 2);
    }

    #[test]
    fn cancellation_releases_unsubmitted_reservations_but_waits_for_submitted_extent() {
        let mut operation = PinnedReadOperation::new(synthetic_extents(3), 2);
        let mut busy = vec![false; 3];
        assert!(reserve_test_operation(
            &mut operation,
            &mut busy,
            &[true; 3]
        ));
        operation.payloads[0] = Some(());
        operation.record_submission(0).unwrap();

        let released = operation.cancel().expect("operation was cancellable");
        assert_eq!(released, vec![1, 2]);
        for buffer in released {
            busy[buffer] = false;
        }
        assert_eq!(busy, vec![true, false, false]);
        assert!(operation.payloads.iter().all(Option::is_none));
        assert!(!operation.is_terminal());

        operation.record_completion(0).unwrap();
        assert!(
            busy[0],
            "submitted slab stays reserved until its CQE is handled"
        );
        busy[0] = false;
        assert_eq!(busy, vec![false, false, false]);
        assert!(operation.is_terminal());
    }

    #[test]
    fn detached_ticket_is_reaped_after_completion_and_only_once() {
        let operation_id = 7;
        let mut operation = PinnedReadOperation::<()>::new(synthetic_extents(1), 0);
        let mut busy = vec![false];
        assert!(reserve_test_operation(&mut operation, &mut busy, &[true]));
        operation.record_submission(0).unwrap();
        assert!(operation.cancel().unwrap().is_empty());

        let mut operations = HashMap::from([(operation_id, operation)]);
        let mut detached = VecDeque::from([operation_id]);
        assert_eq!(
            reap_detached_pinned_operations(&mut operations, &mut detached),
            0
        );
        assert!(operations.contains_key(&operation_id));
        assert_eq!(busy, vec![true]);

        operations
            .get_mut(&operation_id)
            .unwrap()
            .record_completion(0)
            .unwrap();
        assert!(
            busy[0],
            "the slab is still owned until completion handling releases it"
        );
        busy[0] = false;

        assert_eq!(
            reap_detached_pinned_operations(&mut operations, &mut detached),
            1
        );
        assert!(!operations.contains_key(&operation_id));
        assert!(detached.is_empty());
        assert_eq!(
            reap_detached_pinned_operations(&mut operations, &mut detached),
            0
        );
        assert!(take_terminal_pinned_operation(&mut operations, operation_id).is_none());
    }

    #[test]
    fn terminal_pinned_result_can_only_be_consumed_once() {
        let operation_id = 11;
        let operation = PinnedReadOperation::<()>::new(Vec::new(), 0);
        assert!(operation.is_terminal());
        let mut operations = HashMap::from([(operation_id, operation)]);

        assert!(take_terminal_pinned_operation(&mut operations, operation_id).is_some());
        assert!(take_terminal_pinned_operation(&mut operations, operation_id).is_none());
    }
}
