//! Shared io_uring reader: one ring, fixed-file table and registered slab owner.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ferrule_common::{CompletionHub, Error, Result};
use io_uring::{IoUring, opcode, types};

use super::streaming::source::{ExpertTensorPayload, ExpertTensorSlice};
use super::streaming::storage::{ExpertIoStats, ExpertIoTransport};
use crate::runner::ModelCompletionReactor;

#[cfg(feature = "cuda")]
mod cuda_pinned;
#[cfg(any(feature = "cuda", test))]
mod pinned_protocol;

#[cfg(feature = "cuda")]
#[allow(unused_imports)] // Keep the historical crate-private reader type paths.
pub(crate) use cuda_pinned::{
    PinnedExpertReadPlan, PinnedExpertReadPoll, PinnedExpertReadResult, PinnedExpertReadTicket,
    PinnedExpertTensorPayload, ReservedPinnedExpertRead,
};

pub(crate) const DIRECT_IO_ALIGNMENT: usize = 4096;
const FIXED_FILE_CAPACITY: usize = 64;

#[repr(C, align(4096))]
#[derive(Clone)]
struct AlignedBlock([u8; DIRECT_IO_ALIGNMENT]);

enum RegisteredBufferBacking {
    Pageable(Box<[AlignedBlock]>),
    #[cfg(feature = "cuda")]
    CudaPinned(cuda_pinned::CudaPinnedBuffer),
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

    fn is_available(&self) -> bool {
        match &self.backing {
            RegisteredBufferBacking::Pageable(_) => true,
            #[cfg(feature = "cuda")]
            RegisteredBufferBacking::CudaPinned(buffer) => buffer.is_uniquely_owned(),
        }
    }

    fn as_mut_ptr(&mut self) -> Result<*mut u8> {
        match &mut self.backing {
            RegisteredBufferBacking::Pageable(blocks) => Ok(blocks.as_mut_ptr().cast()),
            #[cfg(feature = "cuda")]
            RegisteredBufferBacking::CudaPinned(buffer) => buffer.as_mut_ptr(),
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

struct PendingDirectReadExtent {
    path: PathBuf,
    start: u64,
    end: u64,
    slices: Vec<(usize, ExpertTensorSlice)>,
}

struct IoUringReadState {
    ring: IoUring,
    transport: ExpertIoTransport,
    completion_eventfd_registered: bool,
    queue_depth: usize,
    buffer_bytes: usize,
    buffers: Vec<RegisteredBuffer>,
    files: Vec<File>,
    file_indices: HashMap<PathBuf, u32>,
    stats: ExpertIoStats,
    #[cfg(feature = "cuda")]
    pinned: cuda_pinned::CudaPinnedReadState,
}

impl IoUringReadState {
    fn new(queue_depth: usize, buffer_bytes: usize, transport: ExpertIoTransport) -> Result<Self> {
        let buffers = (0..queue_depth)
            .map(|_| RegisteredBuffer::new_pageable(buffer_bytes))
            .collect::<Result<Vec<_>>>()?;
        Self::new_with_buffers(queue_depth, buffer_bytes, buffers, transport)
    }

    #[allow(unsafe_code)]
    fn new_with_buffers(
        queue_depth: usize,
        buffer_bytes: usize,
        mut buffers: Vec<RegisteredBuffer>,
        transport: ExpertIoTransport,
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
        let ring = IoUring::new(entries)
            .map_err(|error| Error::context("create expert io_uring", error.into()))?;
        ring.submitter()
            .register_files_sparse(FIXED_FILE_CAPACITY as u32)
            .map_err(|error| {
                Error::context("register sparse expert io_uring files", error.into())
            })?;

        let iovecs = buffers
            .iter_mut()
            .map(RegisteredBuffer::iovec)
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: every iovec points into a fixed-size boxed allocation owned by
        // this state. Buffers are never resized and the ring is unregistered or
        // dropped before the state releases those allocations.
        unsafe { ring.submitter().register_buffers(&iovecs) }
            .map_err(|error| Error::context("register expert io_uring buffers", error.into()))?;

        #[cfg(feature = "cuda")]
        let buffer_count = buffers.len();
        Ok(Self {
            ring,
            transport,
            completion_eventfd_registered: false,
            queue_depth,
            buffer_bytes,
            buffers,
            files: Vec::new(),
            file_indices: HashMap::new(),
            stats: ExpertIoStats::default(),
            #[cfg(feature = "cuda")]
            pinned: cuda_pinned::CudaPinnedReadState::new(buffer_count),
        })
    }

    fn register_completion_eventfd(&mut self, eventfd: &OwnedFd) -> Result<()> {
        self.ring
            .submitter()
            .register_eventfd(eventfd.as_raw_fd())
            .map_err(|error| {
                Error::context("register expert io_uring completion eventfd", error.into())
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
            .map_err(|error| {
                Error::context(
                    "unregister expert io_uring completion eventfd",
                    error.into(),
                )
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
        let mut options = OpenOptions::new();
        options.read(true);
        if self.transport == ExpertIoTransport::DirectIoUring {
            options.custom_flags(libc::O_DIRECT);
        }
        let file = options.open(path).map_err(|error| Error::Model {
            message: format!(
                "open expert shard '{}' for {}: {error}",
                path.display(),
                self.transport.as_str()
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
            return Err(Error::context(
                "submit/wait expert io_uring reads",
                error.into(),
            ));
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
        self.prepare_pageable_read()?;
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

impl Drop for IoUringReadState {
    fn drop(&mut self) {
        let _ = self.unregister_completion_eventfd();
        let _ = self.ring.submitter().unregister_buffers();
        let _ = self.ring.submitter().unregister_files();
    }
}

pub(crate) struct IoUringExpertReader {
    state: Mutex<IoUringReadState>,
    completion_hub: CompletionHub,
    _completion_eventfd: OwnedFd,
    reactor_eventfd: Mutex<Option<OwnedFd>>,
    reactor_failure: Mutex<Option<ferrule_common::FailureReason>>,
}

impl IoUringExpertReader {
    pub(crate) fn new(
        queue_depth: usize,
        buffer_bytes: usize,
        transport: ExpertIoTransport,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        Self::from_state(
            IoUringReadState::new(queue_depth, buffer_bytes, transport)?,
            completion_hub,
        )
    }

    fn from_state(mut state: IoUringReadState, completion_hub: CompletionHub) -> Result<Self> {
        let (completion_eventfd, reactor_eventfd) = create_completion_eventfd_pair()?;
        state.register_completion_eventfd(&completion_eventfd)?;
        Ok(Self {
            state: Mutex::new(state),
            completion_hub,
            _completion_eventfd: completion_eventfd,
            reactor_eventfd: Mutex::new(Some(reactor_eventfd)),
            reactor_failure: Mutex::new(None),
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
            let result: Result<()> = async {
                let eventfd = tokio::io::unix::AsyncFd::new(reactor_eventfd).map_err(|error| {
                    Error::context(
                        "attach expert io_uring completion eventfd to Tokio",
                        error.into(),
                    )
                })?;
                loop {
                    let mut ready = eventfd.readable().await.map_err(|error| {
                        Error::context("await expert io_uring completion eventfd", error.into())
                    })?;
                    drain_completion_eventfd(eventfd.get_ref())?;
                    ready.clear_ready();
                    let progress = reader.react_to_completions();
                    completion_hub.notify();
                    progress?;
                }
            }
            .await;
            if let Err(error) = &result {
                *reader
                    .reactor_failure
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    Some(ferrule_common::FailureReason::ContractViolation {
                        message: error.to_string(),
                    });
                completion_hub.notify();
            }
            result
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
            return Err(Error::context(
                "read expert io_uring completion eventfd",
                error.into(),
            ));
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
#[path = "io_uring_reader/tests.rs"]
mod tests;
