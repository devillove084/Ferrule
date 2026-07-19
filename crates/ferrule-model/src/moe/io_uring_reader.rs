use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use ferrule_common::{Error, Result};
#[cfg(feature = "cuda")]
use ferrule_cuda::context::{CudaPinnedHostAllocator, CudaPinnedU8HostBuffer};
use io_uring::{IoUring, opcode, types};

use super::streaming::{ExpertTensorPayload, ExpertTensorSlice};

const DIRECT_IO_ALIGNMENT: usize = 4096;
const FIXED_FILE_CAPACITY: usize = 64;

#[repr(C, align(4096))]
#[derive(Clone)]
struct AlignedBlock([u8; DIRECT_IO_ALIGNMENT]);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct IoUringExpertReaderStats {
    pub(crate) submitted_extents: u64,
    pub(crate) completed_extents: u64,
    pub(crate) failed_extents: u64,
    pub(crate) requested_bytes: u64,
    pub(crate) aligned_bytes: u64,
    pub(crate) coalesced_slices: u64,
    pub(crate) fixed_file_registrations: u64,
    pub(crate) fallback_count: u64,
    pub(crate) slab_exhaustions: u64,
    pub(crate) peak_queue_depth: usize,
    pub(crate) read_us: u64,
}

#[cfg(feature = "cuda")]
pub(crate) struct PinnedExpertTensorPayload {
    pub(crate) slice: ExpertTensorSlice,
    pub(crate) bytes: CudaPinnedU8HostBuffer,
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
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::Model("io_uring expert slice range overflow".into()))?;
        if end > self.len {
            return Err(Error::Model(format!(
                "io_uring expert slice range exceeds registered buffer: {offset}+{len}>{}",
                self.len
            )));
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
            RegisteredBufferBacking::Pageable(_) => Err(Error::Model(
                "io_uring expert reader was not configured with CUDA pinned slabs".into(),
            )),
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

struct PendingDirectReadExtent {
    path: PathBuf,
    start: u64,
    end: u64,
    slices: Vec<(usize, ExpertTensorSlice)>,
}

struct IoUringDirectState {
    ring: IoUring,
    queue_depth: usize,
    buffer_bytes: usize,
    buffers: Vec<RegisteredBuffer>,
    files: Vec<File>,
    file_indices: HashMap<PathBuf, u32>,
    stats: IoUringExpertReaderStats,
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
            return Err(Error::Model(format!(
                "CUDA pinned expert slab count must be at least queue depth: {slab_count} < {queue_depth}"
            )));
        }
        if slab_count > u16::MAX as usize {
            return Err(Error::Model(format!(
                "CUDA pinned expert slab count exceeds io_uring fixed-buffer limit: {slab_count}"
            )));
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
            return Err(Error::Model(format!(
                "io_uring expert queue depth must be in 1..={}, got {queue_depth}",
                u16::MAX
            )));
        }
        let buffer_bytes = align_up(buffer_bytes, DIRECT_IO_ALIGNMENT)?;
        let entries = queue_depth.next_power_of_two().max(2);
        let entries = u32::try_from(entries)
            .map_err(|_| Error::Model("io_uring expert queue depth exceeds u32".into()))?;
        let ring = IoUring::new(entries)
            .map_err(|error| Error::Model(format!("create expert io_uring: {error}")))?;
        ring.submitter()
            .register_files_sparse(FIXED_FILE_CAPACITY as u32)
            .map_err(|error| {
                Error::Model(format!("register sparse expert io_uring files: {error}"))
            })?;

        let iovecs = buffers
            .iter_mut()
            .map(RegisteredBuffer::iovec)
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: every iovec points into a fixed-size boxed allocation owned by
        // this state. Buffers are never resized and the ring is unregistered or
        // dropped before the state releases those allocations.
        unsafe { ring.submitter().register_buffers(&iovecs) }
            .map_err(|error| Error::Model(format!("register expert io_uring buffers: {error}")))?;

        Ok(Self {
            ring,
            queue_depth,
            buffer_bytes,
            buffers,
            files: Vec::new(),
            file_indices: HashMap::new(),
            stats: IoUringExpertReaderStats::default(),
        })
    }

    fn register_file(&mut self, path: &Path) -> Result<u32> {
        if let Some(&index) = self.file_indices.get(path) {
            return Ok(index);
        }
        if self.files.len() >= FIXED_FILE_CAPACITY {
            return Err(Error::Model(format!(
                "expert io_uring fixed-file table is full at {FIXED_FILE_CAPACITY} entries"
            )));
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(|error| {
                Error::Model(format!(
                    "open expert shard with O_DIRECT '{}': {error}",
                    path.display()
                ))
            })?;
        let index = u32::try_from(self.files.len())
            .map_err(|_| Error::Model("expert fixed-file index exceeds u32".into()))?;
        self.ring
            .submitter()
            .register_files_update(index, &[file.as_raw_fd()])
            .map_err(|error| {
                Error::Model(format!(
                    "register expert shard '{}' at fixed index {index}: {error}",
                    path.display()
                ))
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
                return Err(Error::Model("expert tensor slice is empty".into()));
            }
            let slice_end = slice
                .offset
                .checked_add(slice.bytes)
                .ok_or_else(|| Error::Model("expert tensor slice end overflow".into()))?;
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
        let required_end = usize::try_from(end - aligned_offset)
            .map_err(|_| Error::Model("expert direct-read extent exceeds usize".into()))?;
        let aligned_len = align_up(required_end, DIRECT_IO_ALIGNMENT)?;
        if aligned_len > self.buffer_bytes {
            return Err(Error::Model(format!(
                "expert direct-read extent exceeds registered buffer: aligned={aligned_len} buffer={} path={}",
                self.buffer_bytes,
                path.display()
            )));
        }
        let file_index = self.register_file(&path)?;
        let views = slices
            .into_iter()
            .map(|(slice_index, slice)| {
                let payload_offset =
                    usize::try_from(slice.offset - aligned_offset).map_err(|_| {
                        Error::Model("expert tensor alignment offset exceeds usize".into())
                    })?;
                let payload_len = usize::try_from(slice.bytes)
                    .map_err(|_| Error::Model("expert tensor slice size exceeds usize".into()))?;
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
        let len = u32::try_from(extent.aligned_len)
            .map_err(|_| Error::Model("expert direct-read length exceeds u32".into()))?;
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
        unsafe { submission.push(&entry) }
            .map_err(|_| Error::Model("expert io_uring submission queue is full".into()))
    }

    #[cfg(feature = "cuda")]
    fn direct_expert_capacity(&self) -> Option<usize> {
        self.buffers
            .first()
            .is_some_and(RegisteredBuffer::is_cuda_pinned)
            .then_some(self.buffers.len() / 2)
    }

    #[cfg(feature = "cuda")]
    fn available_direct_experts(&self) -> Option<usize> {
        self.direct_expert_capacity().map(|_| {
            self.buffers
                .iter()
                .filter(|buffer| buffer.is_available())
                .count()
                / 2
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
            return Err(Error::Model(format!(
                "expert io_uring pinned slab pool exhausted: available={} required={wave_len}",
                buffer_indices.len()
            )));
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
            return Err(Error::Model(format!(
                "submit/wait expert io_uring reads: {error}"
            )));
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
            return Err(Error::Model(format!(
                "expert io_uring completion underflow: got {} expected {wave_len}",
                completions.len()
            )));
        }

        let mut completed = Vec::with_capacity(wave_len);
        for (user_data, result) in completions {
            let extent_index = (user_data >> 32) as usize;
            let buffer_index = (user_data & 0xffff_ffff) as usize;
            let extent = extents.get(extent_index).ok_or_else(|| {
                Error::Model(format!(
                    "expert io_uring returned invalid extent index {extent_index}"
                ))
            })?;
            if buffer_index >= self.buffers.len() {
                return Err(Error::Model(format!(
                    "expert io_uring returned invalid buffer index {buffer_index}"
                )));
            }
            let path = extent
                .views
                .first()
                .map(|view| view.slice.path.as_path())
                .ok_or_else(|| Error::Model("expert direct-read extent has no views".into()))?;
            if result < 0 {
                self.stats.failed_extents = self.stats.failed_extents.saturating_add(1);
                return Err(Error::Model(format!(
                    "expert io_uring read '{}' at {} failed: {}",
                    path.display(),
                    extent.aligned_offset,
                    std::io::Error::from_raw_os_error(-result)
                )));
            }
            let bytes_read = result as usize;
            if bytes_read < extent.required_end {
                self.stats.failed_extents = self.stats.failed_extents.saturating_add(1);
                return Err(Error::Model(format!(
                    "short expert io_uring read '{}': got {bytes_read}, need at least {}",
                    path.display(),
                    extent.required_end
                )));
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

    #[cfg(feature = "cuda")]
    fn read_slices_pinned(
        &mut self,
        slices: &[ExpertTensorSlice],
    ) -> Result<Vec<PinnedExpertTensorPayload>> {
        if slices.is_empty() {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let result = (|| {
            let extents = self.prepare_read(slices)?;
            let mut payloads = (0..slices.len()).map(|_| None).collect::<Vec<_>>();
            for wave_start in (0..extents.len()).step_by(self.queue_depth) {
                let wave_end = (wave_start + self.queue_depth).min(extents.len());
                for (extent_index, buffer_index) in
                    self.execute_wave(&extents, wave_start, wave_end)?
                {
                    for view in &extents[extent_index].views {
                        let bytes = self.buffers[buffer_index]
                            .pinned_range(view.payload_offset, view.payload_len)?;
                        payloads[view.slice_index] = Some(PinnedExpertTensorPayload {
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
            payload.ok_or_else(|| {
                Error::Model(format!(
                    "expert io_uring did not produce payload for slice {index}"
                ))
            })
        })
        .collect()
}

impl Drop for IoUringDirectState {
    fn drop(&mut self) {
        let _ = self.ring.submitter().unregister_buffers();
        let _ = self.ring.submitter().unregister_files();
    }
}

pub(crate) struct IoUringExpertReader {
    state: Mutex<IoUringDirectState>,
}

impl IoUringExpertReader {
    pub(crate) fn new(queue_depth: usize, buffer_bytes: usize) -> Result<Self> {
        Ok(Self {
            state: Mutex::new(IoUringDirectState::new(queue_depth, buffer_bytes)?),
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn new_cuda_pinned(
        queue_depth: usize,
        buffer_bytes: usize,
        slab_count: usize,
        allocator: &CudaPinnedHostAllocator,
    ) -> Result<Self> {
        Ok(Self {
            state: Mutex::new(IoUringDirectState::new_cuda_pinned(
                queue_depth,
                buffer_bytes,
                slab_count,
                allocator,
            )?),
        })
    }

    pub(crate) fn read_slices(
        &self,
        slices: &[ExpertTensorSlice],
    ) -> Result<Vec<ExpertTensorPayload>> {
        self.state
            .lock()
            .map_err(|_| Error::Model("expert io_uring state lock poisoned".into()))?
            .read_slices(slices)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn read_slices_pinned(
        &self,
        slices: &[ExpertTensorSlice],
    ) -> Result<Vec<PinnedExpertTensorPayload>> {
        self.state
            .lock()
            .map_err(|_| Error::Model("expert io_uring state lock poisoned".into()))?
            .read_slices_pinned(slices)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn direct_expert_capacity(&self) -> Option<usize> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.direct_expert_capacity())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn available_direct_experts(&self) -> Option<usize> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.available_direct_experts())
    }

    pub(crate) fn record_fallback(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.stats.fallback_count = state.stats.fallback_count.saturating_add(1);
        }
    }

    pub(crate) fn stats(&self) -> IoUringExpertReaderStats {
        self.state
            .lock()
            .map(|state| state.stats)
            .unwrap_or_default()
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::Model(format!(
            "invalid direct-I/O alignment {alignment}"
        )));
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| Error::Model("direct-I/O alignment overflow".into()))
}
