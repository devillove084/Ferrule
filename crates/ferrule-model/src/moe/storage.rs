//! Storage adapter for expert sources: transport selection and bounded reads.
//!
//! Completed reads cross the semantic boundary as artifact payloads. Pinned
//! reads retain the existing plan/ticket/poll boundary and source snapshots;
//! the io_uring reader remains the sole owner of physical operations and slabs.
//! Encoding validation and device publication are not performed here.

use std::sync::Arc;
#[cfg(all(target_os = "linux", feature = "cuda"))]
use std::{collections::HashMap, sync::Mutex};

use super::source::{
    ExpertArtifactPayload, ExpertId, ExpertLoadSource, ExpertTensorPayload, ExpertTensorSlice,
    checkpoint_read_plan_for_expert_source, slices_for_load_source,
};
use crate::checkpoint::CheckpointPositionedReader;
#[cfg(all(target_os = "linux", feature = "cuda"))]
use crate::checkpoint::{CheckpointReadPlan, CheckpointSourceFileIdentity};
#[cfg(target_os = "linux")]
use crate::moe::io_uring_reader;
use crate::runner::ModelCompletionReactor;
use ferrule_common::{CompletionHub, Error, Result, StaleReason};
use rayon::prelude::*;
use snafu::Snafu;

#[cfg(all(feature = "cuda-test-support", target_os = "linux"))]
#[path = "cuda_reader_test_support.rs"]
mod test_support;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertIoTransport {
    Positioned,
    BufferedIoUring,
    DirectIoUring,
}

impl ExpertIoTransport {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Positioned => "positioned",
            Self::BufferedIoUring => "buffered_io_uring",
            Self::DirectIoUring => "direct_io_uring",
        }
    }

    fn from_env() -> std::result::Result<Self, ExpertIoTransportError> {
        let value = std::env::var("FERRULE_EXPERT_IO_TRANSPORT")
            .unwrap_or_else(|_| default_expert_io_transport().as_str().to_owned());
        match value.to_ascii_lowercase().as_str() {
            "positioned" => Ok(Self::Positioned),
            "buffered_io_uring" => Ok(Self::BufferedIoUring),
            "direct_io_uring" => Ok(Self::DirectIoUring),
            _ => Err(ExpertIoTransportError::Unsupported { value }),
        }
    }

    const fn is_io_uring(self) -> bool {
        matches!(self, Self::BufferedIoUring | Self::DirectIoUring)
    }
}

#[derive(Debug, Snafu, PartialEq, Eq)]
pub enum ExpertIoTransportError {
    #[snafu(display(
        "unsupported FERRULE_EXPERT_IO_TRANSPORT '{value}'; expected positioned, buffered_io_uring, or direct_io_uring"
    ))]
    Unsupported { value: String },

    #[snafu(display(
        "io_uring expert streaming requires buffered_io_uring or direct_io_uring, got {}",
        transport.as_str()
    ))]
    IoUringReaderRequiresIoUring { transport: ExpertIoTransport },
}

fn transport_error(source: ExpertIoTransportError) -> Error {
    Error::ModelSource {
        source: Box::new(source),
    }
}

#[cfg(all(target_os = "linux", any(feature = "cuda", test)))]
#[derive(Debug, Snafu, PartialEq, Eq)]
enum ExpertIoPlanError {
    #[snafu(display("expert I/O execution reserve requires at least one operation slot"))]
    EmptyExecutionReserve,
    #[snafu(display("expert I/O execution reserve slot count {slots} exceeds u64"))]
    ExecutionReserveSlotsOutOfRange { slots: usize },
    #[snafu(display("expert I/O execution reserve {resource} byte calculation overflowed"))]
    ExecutionReserveByteOverflow { resource: &'static str },
}

#[cfg(all(target_os = "linux", any(feature = "cuda", test)))]
fn io_plan_error(source: ExpertIoPlanError) -> Error {
    Error::ModelSource {
        source: Box::new(source),
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertIoStats {
    pub submitted_extents: u64,
    pub completed_extents: u64,
    pub failed_extents: u64,
    pub requested_bytes: u64,
    pub aligned_bytes: u64,
    pub coalesced_slices: u64,
    pub fixed_file_registrations: u64,
    pub slab_exhaustions: u64,
    pub peak_queue_depth: usize,
    pub read_us: u64,
}

#[cfg(all(target_os = "linux", any(feature = "cuda", test)))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertIoPlan {
    pub queue_depth: usize,
    pub buffer_bytes: usize,
    pub slab_count: usize,
    pub maximum_extents_per_operation: usize,
}

#[cfg(all(target_os = "linux", any(feature = "cuda", test)))]
impl ExpertIoPlan {
    const DEFAULT_PINNED_BYTES: usize = 256 * 1024 * 1024;
    const MAX_QUEUE_DEPTH: usize = 32;

    #[cfg(feature = "cuda")]
    pub(crate) fn for_checkpoint_plans<'a>(
        plans: impl IntoIterator<Item = &'a CheckpointReadPlan>,
    ) -> Result<Self> {
        let mut maximum_extent_bytes = 0u64;
        let mut maximum_extents = 0usize;
        for plan in plans {
            maximum_extents = maximum_extents.max(plan.extents().len());
            for extent in plan.extents() {
                maximum_extent_bytes = maximum_extent_bytes.max(extent.bytes());
            }
        }
        Self::from_layout(maximum_extent_bytes, maximum_extents)
    }

    fn from_layout(maximum_extent_bytes: u64, maximum_extents: usize) -> Result<Self> {
        if maximum_extent_bytes == 0 || maximum_extents == 0 {
            return Err(Error::Model {
                message: "expert I/O planning requires non-empty checkpoint extents".into(),
            });
        }
        let alignment = io_uring_reader::DIRECT_IO_ALIGNMENT;
        let maximum_extent_bytes =
            usize::try_from(maximum_extent_bytes).map_err(|_| Error::Model {
                message: "expert I/O extent exceeds usize".into(),
            })?;
        // A tensor beginning at an unaligned file offset can require one extra
        // alignment unit even when its payload is alignment-sized.
        let buffer_bytes = maximum_extent_bytes
            .checked_add(alignment - 1)
            .and_then(|bytes| bytes.checked_add(alignment - 1))
            .map(|bytes| bytes / alignment * alignment)
            .ok_or_else(|| Error::Model {
                message: "expert I/O buffer size overflow".into(),
            })?;
        let raw_slabs = (Self::DEFAULT_PINNED_BYTES / buffer_bytes).max(maximum_extents);
        let slab_count = (raw_slabs / maximum_extents)
            .max(1)
            .checked_mul(maximum_extents)
            .ok_or_else(|| Error::Model {
                message: "expert I/O slab count overflow".into(),
            })?;
        let queue_depth = slab_count.clamp(1, Self::MAX_QUEUE_DEPTH);
        Ok(Self {
            queue_depth,
            buffer_bytes,
            slab_count,
            maximum_extents_per_operation: maximum_extents,
        })
    }

    #[cfg(feature = "cuda")]
    fn with_env_overrides(self) -> Result<Self> {
        let buffer_bytes =
            parse_expert_io_mib_override("FERRULE_EXPERT_IO_BUFFER_MIB", self.buffer_bytes)?;
        let plan = Self {
            queue_depth: parse_expert_io_usize("FERRULE_EXPERT_IO_QUEUE_DEPTH", self.queue_depth)?,
            buffer_bytes,
            slab_count: parse_expert_io_usize("FERRULE_EXPERT_IO_SLABS", self.slab_count)?,
            maximum_extents_per_operation: self.maximum_extents_per_operation,
        };
        if plan.slab_count < plan.maximum_extents_per_operation {
            return Err(Error::Model {
                message: format!(
                    "expert I/O slab count {} cannot hold one operation with {} extents",
                    plan.slab_count, plan.maximum_extents_per_operation
                ),
            });
        }
        Ok(plan)
    }

    pub(crate) fn execution_reserve(
        self,
        maximum_expert_bytes: u64,
        required_operation_slots: usize,
    ) -> Result<ferrule_common::materialization_io::MaterializationResourceRequirements> {
        if required_operation_slots == 0 {
            return Err(io_plan_error(ExpertIoPlanError::EmptyExecutionReserve));
        }
        let operation_slots = u64::try_from(required_operation_slots).map_err(|_| {
            io_plan_error(ExpertIoPlanError::ExecutionReserveSlotsOutOfRange {
                slots: required_operation_slots,
            })
        })?;
        let staged_bytes_per_operation = self
            .buffer_bytes
            .checked_mul(self.maximum_extents_per_operation)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                io_plan_error(ExpertIoPlanError::ExecutionReserveByteOverflow {
                    resource: "staged operation",
                })
            })?;
        let staged_bytes = staged_bytes_per_operation
            .checked_mul(operation_slots)
            .ok_or_else(|| {
                io_plan_error(ExpertIoPlanError::ExecutionReserveByteOverflow {
                    resource: "staged wave",
                })
            })?;
        let device_bytes = maximum_expert_bytes
            .checked_mul(operation_slots)
            .ok_or_else(|| {
                io_plan_error(ExpertIoPlanError::ExecutionReserveByteOverflow {
                    resource: "device wave",
                })
            })?;
        Ok(
            ferrule_common::materialization_io::MaterializationResourceRequirements {
                read_slots: operation_slots,
                storage_read_bytes: staged_bytes,
                pinned_host_bytes: staged_bytes,
                upload_slots: operation_slots,
                h2d_bytes: device_bytes,
                install_slots: operation_slots,
                device_install_bytes: device_bytes,
            },
        )
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
/// Completed pinned bytes plus their semantic descriptor, independent of transport.
/// Moving this value transfers the existing slab view; it creates no new lease.
pub(crate) struct PinnedExpertTensorPayload {
    pub(crate) slice: ExpertTensorSlice,
    pub(crate) bytes: ferrule_backend::cuda::operators::moe::CudaPinnedU8HostBuffer,
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub(crate) struct PinnedExpertArtifactPayload {
    pub(crate) expert: ExpertId,
    pub(crate) tensors: Vec<PinnedExpertTensorPayload>,
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub(crate) struct PinnedExpertLoadPlan {
    expert: ExpertId,
    reader: io_uring_reader::PinnedExpertReadPlan,
    source_files: Arc<[CheckpointSourceFileIdentity]>,
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
impl PinnedExpertLoadPlan {
    pub(crate) const fn requirements(
        &self,
    ) -> ferrule_common::materialization_io::MaterializationResourceRequirements {
        self.reader.requirements()
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PinnedExpertReadTicket {
    expert: ExpertId,
    reader: io_uring_reader::PinnedExpertReadTicket,
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub(crate) enum PinnedExpertReadPoll {
    Pending,
    Ready(PinnedExpertArtifactPayload),
    Failed(Error),
    Cancelled,
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub(crate) struct ReservedPinnedExpertLoad {
    pub(crate) ticket: PinnedExpertReadTicket,
    pub(crate) slabs: Box<[ferrule_common::RegisteredPinnedAlignedSlabLeaseDescriptor]>,
}

#[derive(Clone)]
pub struct ExpertStreamingReader {
    max_slice_bytes: u64,
    transport: ExpertIoTransport,
    completion_hub: CompletionHub,
    #[cfg(target_os = "linux")]
    io_uring: Option<Arc<io_uring_reader::IoUringExpertReader>>,
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pinned_source_files:
        Arc<Mutex<HashMap<PinnedExpertReadTicket, Arc<[CheckpointSourceFileIdentity]>>>>,
}

impl std::fmt::Debug for ExpertStreamingReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExpertStreamingReader")
            .field("max_slice_bytes", &self.max_slice_bytes)
            .field("backend", &self.backend_name())
            .finish()
    }
}

impl ExpertStreamingReader {
    pub fn new(max_slice_bytes: u64) -> Self {
        Self::new_with_completion_hub(max_slice_bytes, CompletionHub::new())
    }

    pub(crate) fn new_with_completion_hub(
        max_slice_bytes: u64,
        completion_hub: CompletionHub,
    ) -> Self {
        Self {
            max_slice_bytes,
            transport: ExpertIoTransport::Positioned,
            completion_hub,
            #[cfg(target_os = "linux")]
            io_uring: None,
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            pinned_source_files: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn from_env(max_slice_bytes: u64) -> Result<Self> {
        Self::from_env_with_completion_hub(max_slice_bytes, CompletionHub::new())
    }

    pub(crate) fn from_env_with_completion_hub(
        max_slice_bytes: u64,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        let transport = ExpertIoTransport::from_env().map_err(transport_error)?;
        if transport == ExpertIoTransport::Positioned {
            return Ok(Self::new_with_completion_hub(
                max_slice_bytes,
                completion_hub,
            ));
        }
        let queue_depth = parse_expert_io_usize("FERRULE_EXPERT_IO_QUEUE_DEPTH", 2)?;
        let buffer_mib = parse_expert_io_usize("FERRULE_EXPERT_IO_BUFFER_MIB", 16)?;
        let buffer_bytes = buffer_mib
            .checked_mul(1024 * 1024)
            .ok_or_else(|| Error::Model {
                message: "FERRULE_EXPERT_IO_BUFFER_MIB overflows usize".into(),
            })?;
        Self::with_transport_and_completion_hub(
            max_slice_bytes,
            queue_depth,
            buffer_bytes,
            transport,
            completion_hub,
        )
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn from_env_with_cuda_pinned(
        max_slice_bytes: u64,
        allocator: ferrule_backend::cuda::operators::moe::CudaPinnedHostAllocator,
        completion_hub: CompletionHub,
        plan: ExpertIoPlan,
    ) -> Result<(Self, ExpertIoPlan)> {
        let transport = ExpertIoTransport::from_env().map_err(transport_error)?;
        validate_io_uring_transport(transport).map_err(transport_error)?;
        let plan = plan.with_env_overrides()?;
        tracing::info!(
            transport = transport.as_str(),
            queue_depth = plan.queue_depth,
            buffer_bytes = plan.buffer_bytes,
            slab_count = plan.slab_count,
            pinned_bytes = plan.buffer_bytes.saturating_mul(plan.slab_count),
            "planned expert I/O pipeline"
        );
        Ok((
            Self {
                max_slice_bytes,
                transport,
                completion_hub: completion_hub.clone(),
                io_uring: Some(Arc::new(
                    io_uring_reader::IoUringExpertReader::new_cuda_pinned(
                        plan.queue_depth,
                        plan.buffer_bytes,
                        plan.slab_count,
                        &allocator,
                        transport,
                        completion_hub,
                    )?,
                )),
                pinned_source_files: Arc::new(Mutex::new(HashMap::new())),
            },
            plan,
        ))
    }

    #[cfg(target_os = "linux")]
    pub fn with_transport(
        max_slice_bytes: u64,
        queue_depth: usize,
        buffer_bytes: usize,
        transport: ExpertIoTransport,
    ) -> Result<Self> {
        Self::with_transport_and_completion_hub(
            max_slice_bytes,
            queue_depth,
            buffer_bytes,
            transport,
            CompletionHub::new(),
        )
    }

    #[cfg(target_os = "linux")]
    fn with_transport_and_completion_hub(
        max_slice_bytes: u64,
        queue_depth: usize,
        buffer_bytes: usize,
        transport: ExpertIoTransport,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        validate_io_uring_transport(transport).map_err(transport_error)?;
        Ok(Self {
            max_slice_bytes,
            transport,
            completion_hub: completion_hub.clone(),
            io_uring: Some(Arc::new(io_uring_reader::IoUringExpertReader::new(
                queue_depth,
                buffer_bytes,
                transport,
                completion_hub,
            )?)),
            #[cfg(feature = "cuda")]
            pinned_source_files: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn with_transport(
        max_slice_bytes: u64,
        queue_depth: usize,
        buffer_bytes: usize,
        transport: ExpertIoTransport,
    ) -> Result<Self> {
        Self::with_transport_and_completion_hub(
            max_slice_bytes,
            queue_depth,
            buffer_bytes,
            transport,
            CompletionHub::new(),
        )
    }

    #[cfg(not(target_os = "linux"))]
    fn with_transport_and_completion_hub(
        _max_slice_bytes: u64,
        _queue_depth: usize,
        _buffer_bytes: usize,
        _transport: ExpertIoTransport,
        _completion_hub: CompletionHub,
    ) -> Result<Self> {
        Err(Error::Model {
            message: "io_uring expert streaming is supported only on Linux".into(),
        })
    }

    pub const fn transport(&self) -> ExpertIoTransport {
        self.transport
    }

    pub const fn backend_name(&self) -> &'static str {
        self.transport.as_str()
    }

    pub fn max_slice_bytes(&self) -> u64 {
        self.max_slice_bytes
    }

    pub(crate) fn completion_hub(&self) -> CompletionHub {
        self.completion_hub.clone()
    }

    /// Transfers physical reader completion reactors to the owning model runner.
    pub fn take_completion_reactors(&self) -> Vec<ModelCompletionReactor> {
        #[cfg(target_os = "linux")]
        if let Some(reader) = self.io_uring.as_ref()
            && let Some(reactor) = reader.take_completion_reactor()
        {
            return vec![reactor];
        }
        Vec::new()
    }

    pub fn io_stats(&self) -> ExpertIoStats {
        #[cfg(target_os = "linux")]
        if let Some(reader) = self.io_uring.as_ref() {
            return reader.stats();
        }
        ExpertIoStats::default()
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn physical_resource_capacity(
        &self,
    ) -> Result<Option<ferrule_common::materialization_io::MaterializationResourceRequirements>>
    {
        self.io_uring
            .as_ref()
            .map_or(Ok(None), |reader| reader.physical_resource_capacity())
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn plan_checkpoint_source_pinned(
        &self,
        expert: ExpertId,
        read_plan: &CheckpointReadPlan,
        install_source: &ExpertLoadSource,
    ) -> Result<Option<PinnedExpertLoadPlan>> {
        let Some(reader) = self.io_uring.as_ref() else {
            return Ok(None);
        };
        read_plan
            .validate_source_identity()
            .map_err(stale_source_identity_error)?;
        let slices = self.bounded_slices_for_load_source(expert, install_source)?;
        if slices.len() != read_plan.extents().len()
            || slices
                .iter()
                .zip(read_plan.extents())
                .any(|(slice, extent)| {
                    slice.path != extent.path()
                        || slice.offset != extent.offset()
                        || slice.bytes != extent.bytes()
                })
        {
            return Err(Error::Model {
                message: format!(
                    "expert install descriptor {}:{} does not match its checkpoint read plan",
                    expert.layer, expert.expert
                ),
            });
        }
        Ok(Some(PinnedExpertLoadPlan {
            expert,
            reader: reader.plan_slices_pinned(&slices)?,
            source_files: Arc::from(read_plan.source_files()),
        }))
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn reserve_load_source_pinned(
        &self,
        plan: PinnedExpertLoadPlan,
        operation: ferrule_common::OperationId,
        key: ferrule_common::MaterializationKey,
    ) -> Result<ReservedPinnedExpertLoad> {
        let reader = self.io_uring.as_ref().ok_or_else(|| Error::Model {
            message: "pinned expert read plan requires the io_uring backend".into(),
        })?;
        let PinnedExpertLoadPlan {
            expert,
            reader: read_plan,
            source_files,
        } = plan;
        Self::require_source_files_current(&source_files)?;
        let reserved = reader.reserve_slices_pinned(read_plan, operation, key)?;
        let ticket = PinnedExpertReadTicket {
            expert,
            reader: reserved.ticket,
        };
        let mut pinned_source_files =
            self.pinned_source_files
                .lock()
                .map_err(|_| Error::Internal {
                    message: "pinned source identity registry is poisoned".into(),
                })?;
        if pinned_source_files.contains_key(&ticket) {
            drop(pinned_source_files);
            let cleanup = reader.detach_slices_pinned(ticket.reader);
            return Err(match cleanup {
                Ok(()) => Error::Internal {
                    message: "duplicate pinned expert source identity ticket".into(),
                },
                Err(error) => Error::Internal {
                    message: format!(
                        "duplicate pinned expert source identity ticket; read cleanup also failed ({error})"
                    ),
                },
            });
        }
        pinned_source_files.insert(ticket, source_files);
        drop(pinned_source_files);
        Ok(ReservedPinnedExpertLoad {
            ticket,
            slabs: reserved.slabs,
        })
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn submit_reserved_load_source_pinned(
        &self,
        ticket: PinnedExpertReadTicket,
    ) -> Result<()> {
        let reader = self.io_uring.as_ref().ok_or_else(|| Error::Model {
            message: "pinned expert read ticket requires the io_uring backend".into(),
        })?;
        self.pinned_source_files_for_ticket(ticket)?;
        reader.submit_reserved_slices_pinned(ticket.reader)
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn cancel_load_source_pinned(&self, ticket: PinnedExpertReadTicket) -> Result<bool> {
        let reader = self.io_uring.as_ref().ok_or_else(|| Error::Model {
            message: "pinned expert read ticket requires the io_uring backend".into(),
        })?;
        self.pinned_source_files_for_ticket(ticket)?;
        reader.cancel_slices_pinned(ticket.reader)
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn poll_load_source_pinned(
        &self,
        ticket: PinnedExpertReadTicket,
        max_completions: usize,
    ) -> Result<PinnedExpertReadPoll> {
        let reader = self.io_uring.as_ref().ok_or_else(|| Error::Model {
            message: "pinned expert read ticket requires the io_uring backend".into(),
        })?;
        let source_files = self.pinned_source_files_for_ticket(ticket)?;
        let stale = Self::require_source_files_current(&source_files).err();
        if stale.is_some() {
            reader.cancel_slices_pinned(ticket.reader)?;
        }
        match reader.poll_slices_pinned(ticket.reader, max_completions)? {
            io_uring_reader::PinnedExpertReadPoll::Pending => Ok(PinnedExpertReadPoll::Pending),
            io_uring_reader::PinnedExpertReadPoll::Ready(result) => {
                self.forget_pinned_source_files(ticket)?;
                if let Some(stale) = stale {
                    return Ok(PinnedExpertReadPoll::Failed(stale));
                }
                if let Err(stale) = Self::require_source_files_current(&source_files) {
                    return Ok(PinnedExpertReadPoll::Failed(stale));
                }
                Ok(PinnedExpertReadPoll::Ready(PinnedExpertArtifactPayload {
                    expert: ticket.expert,
                    tensors: result.payloads,
                }))
            }
            io_uring_reader::PinnedExpertReadPoll::Failed(error) => {
                self.forget_pinned_source_files(ticket)?;
                Ok(PinnedExpertReadPoll::Failed(stale.unwrap_or(error)))
            }
            io_uring_reader::PinnedExpertReadPoll::Cancelled => {
                self.forget_pinned_source_files(ticket)?;
                Ok(stale.map_or(
                    PinnedExpertReadPoll::Cancelled,
                    PinnedExpertReadPoll::Failed,
                ))
            }
        }
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn take_pinned_reactor_failure(&self) -> Option<ferrule_common::FailureReason> {
        self.io_uring
            .as_ref()
            .and_then(|reader| reader.take_reactor_failure())
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn quiesce_load_source_pinned(
        &self,
        ticket: PinnedExpertReadTicket,
    ) -> Result<bool> {
        let reader = self.io_uring.as_ref().ok_or_else(|| Error::Model {
            message: "pinned expert read ticket requires the io_uring backend".into(),
        })?;
        if !reader.quiesce_slices_pinned(ticket.reader)? {
            return Ok(false);
        }
        self.forget_pinned_source_files(ticket)?;
        Ok(true)
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub(crate) fn detach_load_source_pinned(&self, ticket: PinnedExpertReadTicket) -> Result<()> {
        let reader = self.io_uring.as_ref().ok_or_else(|| Error::Model {
            message: "pinned expert read ticket requires the io_uring backend".into(),
        })?;
        let detach = reader.detach_slices_pinned(ticket.reader);
        self.forget_pinned_source_files(ticket)?;
        detach
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    fn pinned_source_files_for_ticket(
        &self,
        ticket: PinnedExpertReadTicket,
    ) -> Result<Arc<[CheckpointSourceFileIdentity]>> {
        self.pinned_source_files
            .lock()
            .map_err(|_| Error::Internal {
                message: "pinned source identity registry is poisoned".into(),
            })?
            .get(&ticket)
            .cloned()
            .ok_or_else(|| Error::Internal {
                message: "pinned read ticket has no source identity snapshot".into(),
            })
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    fn forget_pinned_source_files(&self, ticket: PinnedExpertReadTicket) -> Result<()> {
        self.pinned_source_files
            .lock()
            .map_err(|_| Error::Internal {
                message: "pinned source identity registry is poisoned".into(),
            })?
            .remove(&ticket)
            .ok_or_else(|| Error::Internal {
                message: "pinned read ticket has no source identity snapshot".into(),
            })?;
        Ok(())
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    fn bounded_slices_for_load_source(
        &self,
        expert: ExpertId,
        load_source: &ExpertLoadSource,
    ) -> Result<Vec<ExpertTensorSlice>> {
        let slices = slices_for_load_source(expert, load_source)?;
        self.validate_bounded_slices(&slices)?;
        Ok(slices)
    }

    pub fn validate_source_identity(
        &self,
        load_source: &ExpertLoadSource,
    ) -> std::result::Result<(), StaleReason> {
        load_source.validate_source_identity()
    }

    pub fn read_load_source(
        &self,
        expert: ExpertId,
        load_source: &ExpertLoadSource,
    ) -> Result<ExpertArtifactPayload> {
        if matches!(load_source, ExpertLoadSource::HfLocalTensorSet { .. }) {
            return self.read_checkpoint_load_source(expert, load_source);
        }
        self.require_current_source_identity(load_source)?;
        let slices = slices_for_load_source(expert, load_source)?;
        let tensors = self.read_slices_with_backend(&slices, false)?;
        self.require_current_source_identity(load_source)?;
        Ok(ExpertArtifactPayload { expert, tensors })
    }

    fn read_checkpoint_load_source(
        &self,
        expert: ExpertId,
        load_source: &ExpertLoadSource,
    ) -> Result<ExpertArtifactPayload> {
        let slices = slices_for_load_source(expert, load_source)?;
        self.validate_bounded_slices(&slices)?;
        let plan = checkpoint_read_plan_for_expert_source(expert, load_source)?;
        if slices.len() != plan.extents().len()
            || slices.iter().zip(plan.extents()).any(|(slice, extent)| {
                slice.path != extent.path()
                    || slice.offset != extent.offset()
                    || slice.bytes != extent.bytes()
            })
        {
            return Err(Error::Model {
                message: format!(
                    "expert source {}:{} does not match its checkpoint read plan",
                    expert.layer, expert.expert
                ),
            });
        }
        let payloads = CheckpointPositionedReader::new(self.max_slice_bytes).read(&plan)?;
        let tensors = slices
            .into_iter()
            .zip(payloads)
            .map(|(slice, bytes)| ExpertTensorPayload { slice, bytes })
            .collect();
        Ok(ExpertArtifactPayload { expert, tensors })
    }

    pub fn read_local_slice(&self, slice: &ExpertTensorSlice) -> Result<ExpertTensorPayload> {
        if slice.bytes > self.max_slice_bytes {
            return Err(Error::Model {
                message: format!(
                    "expert tensor slice exceeds bounded read size: {} > {} bytes",
                    slice.bytes, self.max_slice_bytes
                ),
            });
        }
        Self::read_local_slice_positioned(slice)
    }

    /// Read a single tensor slice using positioned read (pread) with no shared
    /// file cursor or virtual-memory mapping.
    fn read_local_slice_positioned(slice: &ExpertTensorSlice) -> Result<ExpertTensorPayload> {
        use std::os::unix::fs::FileExt;
        let file = std::fs::File::open(&slice.path).map_err(|e| Error::Model {
            message: format!("expert tensor slice open '{}': {e}", slice.path.display()),
        })?;
        let mut bytes = vec![0u8; slice.bytes as usize];
        file.read_exact_at(&mut bytes, slice.offset)
            .map_err(|e| Error::Model {
                message: format!(
                    "expert tensor slice read_at '{}': {e}",
                    slice.path.display()
                ),
            })?;
        Ok(ExpertTensorPayload {
            slice: slice.clone(),
            bytes,
        })
    }

    /// Read all tensor slices for one expert concurrently. The configured
    /// backend is attempted first, with positioned reads as its fallback.
    pub fn read_load_source_concurrent(
        &self,
        expert: ExpertId,
        load_source: &ExpertLoadSource,
    ) -> Result<ExpertArtifactPayload> {
        if matches!(load_source, ExpertLoadSource::HfLocalTensorSet { .. }) {
            return self.read_checkpoint_load_source(expert, load_source);
        }
        self.require_current_source_identity(load_source)?;
        let slices = slices_for_load_source(expert, load_source)?;
        let tensors = self.read_slices_with_backend(&slices, true)?;
        self.require_current_source_identity(load_source)?;
        Ok(ExpertArtifactPayload { expert, tensors })
    }

    fn require_current_source_identity(&self, load_source: &ExpertLoadSource) -> Result<()> {
        self.validate_source_identity(load_source)
            .map_err(stale_source_identity_error)
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    fn require_source_files_current(source_files: &[CheckpointSourceFileIdentity]) -> Result<()> {
        if source_files
            .iter()
            .all(CheckpointSourceFileIdentity::is_current)
        {
            Ok(())
        } else {
            Err(stale_source_identity_error(
                StaleReason::SourceIdentityChanged,
            ))
        }
    }

    fn validate_bounded_slices(&self, slices: &[ExpertTensorSlice]) -> Result<()> {
        for slice in slices {
            if slice.bytes > self.max_slice_bytes {
                return Err(Error::Model {
                    message: format!(
                        "expert tensor slice exceeds bounded read size: {} > {} bytes",
                        slice.bytes, self.max_slice_bytes
                    ),
                });
            }
        }
        Ok(())
    }

    fn read_slices_positioned_unchecked(
        slices: &[ExpertTensorSlice],
        parallel: bool,
    ) -> Result<Vec<ExpertTensorPayload>> {
        if !parallel || slices.len() <= 1 {
            return slices
                .iter()
                .map(Self::read_local_slice_positioned)
                .collect();
        }

        let results = slices
            .par_iter()
            .map(Self::read_local_slice_positioned)
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    fn read_slices_with_backend(
        &self,
        slices: &[ExpertTensorSlice],
        parallel_fallback: bool,
    ) -> Result<Vec<ExpertTensorPayload>> {
        self.validate_bounded_slices(slices)?;

        #[cfg(target_os = "linux")]
        if let Some(reader) = self.io_uring.as_ref() {
            match reader.read_slices(slices) {
                Ok(payloads) => return Ok(payloads),
                Err(error) => {
                    tracing::debug!(
                        error = %error,
                        slices = slices.len(),
                        "expert io_uring read failed; falling back to positioned reads"
                    );
                }
            }
        }

        Self::read_slices_positioned_unchecked(slices, parallel_fallback)
    }
}

fn stale_source_identity_error(reason: StaleReason) -> Error {
    Error::Execution {
        message: format!("Stale expert source identity: {reason:?}"),
    }
}

#[cfg(any(target_os = "linux", test))]
fn validate_io_uring_transport(
    transport: ExpertIoTransport,
) -> std::result::Result<(), ExpertIoTransportError> {
    if transport.is_io_uring() {
        return Ok(());
    }
    Err(ExpertIoTransportError::IoUringReaderRequiresIoUring { transport })
}

const fn default_expert_io_transport() -> ExpertIoTransport {
    #[cfg(target_os = "linux")]
    {
        ExpertIoTransport::BufferedIoUring
    }
    #[cfg(not(target_os = "linux"))]
    {
        ExpertIoTransport::Positioned
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn parse_expert_io_mib_override(name: &str, default_bytes: usize) -> Result<usize> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(default_bytes);
    };
    let mib = value.parse::<usize>().map_err(|error| Error::Model {
        message: format!("invalid {name}='{value}': {error}"),
    })?;
    if mib == 0 {
        return Err(Error::Model {
            message: format!("{name} must be greater than zero"),
        });
    }
    mib.checked_mul(1024 * 1024).ok_or_else(|| Error::Model {
        message: format!("{name} overflows usize"),
    })
}

fn parse_expert_io_usize(name: &str, default: usize) -> Result<usize> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(default);
    };
    let parsed = value.parse::<usize>().map_err(|error| Error::Model {
        message: format!("invalid {name}='{value}': {error}"),
    })?;
    if parsed == 0 {
        return Err(Error::Model {
            message: format!("{name} must be greater than zero"),
        });
    }
    Ok(parsed)
}

#[cfg(all(test, target_os = "linux"))]
mod io_plan_tests {
    use super::ExpertIoPlan;

    #[test]
    fn checkpoint_layout_drives_bounded_nvme_pipeline() {
        let plan = ExpertIoPlan::from_layout(4 * 1024 * 1024, 6).unwrap();
        assert_eq!(plan.buffer_bytes, 4 * 1024 * 1024 + 4096);
        assert_eq!(plan.slab_count, 60);
        assert_eq!(plan.queue_depth, 32);
        assert_eq!(plan.maximum_extents_per_operation, 6);
        assert!(plan.buffer_bytes * plan.slab_count <= 256 * 1024 * 1024);

        let reserve = plan.execution_reserve(13_369_344, 6).unwrap();
        assert_eq!(reserve.read_slots, 6);
        assert_eq!(
            reserve.pinned_host_bytes,
            6 * 6 * (4 * 1024 * 1024 + 4096) as u64
        );
        assert_eq!(reserve.storage_read_bytes, reserve.pinned_host_bytes);
        assert_eq!(reserve.upload_slots, 6);
        assert_eq!(reserve.install_slots, 6);
        assert_eq!(reserve.h2d_bytes, 6 * 13_369_344);
        assert_eq!(reserve.device_install_bytes, reserve.h2d_bytes);
    }
}

/// Read multiple experts concurrently. Each expert's slices are read in
/// parallel, and multiple experts are read in parallel too.
pub fn read_experts_concurrent(
    reader: &ExpertStreamingReader,
    loads: &[(ExpertId, ExpertLoadSource)],
) -> Result<Vec<ExpertArtifactPayload>> {
    if loads.is_empty() {
        return Ok(Vec::new());
    }
    // Fast path: single expert.
    if loads.len() == 1 {
        let (expert, source) = &loads[0];
        return Ok(vec![reader.read_load_source_concurrent(*expert, source)?]);
    }

    // Parallel path: use rayon scope (no runtime creation overhead).
    // The previous tokio-based code created a new runtime per call, which
    // added ~1ms overhead per layer per token.
    let results: Vec<Result<ExpertArtifactPayload>> = loads
        .par_iter()
        .map(|(expert, source)| {
            let reader = reader.clone();
            reader.read_load_source_concurrent(*expert, source)
        })
        .collect();
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::super::source::{ExpertMatrixKind, ExpertTensorComponent, ExpertTensorKey};
    use super::super::tests::unique_temp_dir;
    use super::*;

    #[test]
    fn io_uring_reader_accepts_only_io_uring_transports() {
        validate_io_uring_transport(ExpertIoTransport::BufferedIoUring).unwrap();
        validate_io_uring_transport(ExpertIoTransport::DirectIoUring).unwrap();
        assert_eq!(
            validate_io_uring_transport(ExpertIoTransport::Positioned),
            Err(ExpertIoTransportError::IoUringReaderRequiresIoUring {
                transport: ExpertIoTransport::Positioned,
            })
        );
    }

    #[test]
    fn expert_io_transport_names_are_explicit_and_stable() {
        assert_eq!(ExpertIoTransport::Positioned.as_str(), "positioned");
        assert_eq!(
            ExpertIoTransport::BufferedIoUring.as_str(),
            "buffered_io_uring"
        );
        assert_eq!(ExpertIoTransport::DirectIoUring.as_str(), "direct_io_uring");
    }

    #[test]
    fn reader_rejects_slices_larger_than_bound() {
        let dir = unique_temp_dir("ferrule-expert-streaming-bound-test");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("slice.bin");
        std::fs::write(&shard, vec![0u8; 16]).unwrap();
        let slice = ExpertTensorSlice {
            key: ExpertTensorKey::new(0, 0, ExpertMatrixKind::Gate),
            component: ExpertTensorComponent::Weight,
            path: shard,
            offset: 0,
            bytes: 16,
            dtype: "I8".into(),
            shape: vec![16],
        };
        let err = ExpertStreamingReader::new(8)
            .read_local_slice(&slice)
            .unwrap_err();
        assert!(err.to_string().contains("bounded read size"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn positioned_reader_reads_the_requested_bytes() {
        let dir = unique_temp_dir("ferrule-expert-positioned-reader-test");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("slice.bin");
        std::fs::write(&shard, (0u8..16).collect::<Vec<_>>()).unwrap();
        let expert = ExpertId::new(0, 0);
        let source = ExpertLoadSource::LocalShard {
            path: shard,
            offset: 4,
            bytes: 6,
        };
        let reader = ExpertStreamingReader::new(16);
        let payload = reader.read_load_source(expert, &source).unwrap();

        assert_eq!(payload.expert, expert);
        assert_eq!(payload.tensors.len(), 1);
        assert_eq!(payload.tensors[0].bytes, vec![4, 5, 6, 7, 8, 9]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn positioned_payloads_preserve_descriptor_order_in_serial_and_parallel_reads() {
        let dir = unique_temp_dir("ferrule-expert-payload-order");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("slices.bin");
        std::fs::write(&path, (0u8..32).collect::<Vec<_>>()).unwrap();
        let expert = ExpertId::new(2, 7);
        let slices = [
            (ExpertMatrixKind::Down, 20, 3),
            (ExpertMatrixKind::Gate, 2, 4),
            (ExpertMatrixKind::Up, 12, 2),
        ]
        .map(|(matrix, offset, bytes)| ExpertTensorSlice {
            key: ExpertTensorKey { expert, matrix },
            component: ExpertTensorComponent::Weight,
            path: path.clone(),
            offset,
            bytes,
            dtype: "opaque".into(),
            shape: vec![bytes as usize],
        });
        let source = ExpertLoadSource::LocalTensorSet {
            tensors: slices.to_vec(),
        };
        let reader = ExpertStreamingReader::new(4);
        let serial = reader.read_load_source(expert, &source).unwrap();
        let parallel = reader.read_load_source_concurrent(expert, &source).unwrap();
        assert_eq!(serial, parallel);
        assert_eq!(serial.expert, expert);
        for (payload, descriptor) in serial.tensors.iter().zip(&slices) {
            assert_eq!(&payload.slice, descriptor);
            assert_eq!(
                payload.bytes,
                (descriptor.offset as u8..descriptor.end_offset() as u8).collect::<Vec<_>>()
            );
        }
        assert_eq!(
            read_experts_concurrent(&reader, &[(expert, source.clone()), (expert, source)])
                .unwrap(),
            vec![serial.clone(), serial]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn storage_returns_raw_payload_without_semantic_encoding_validation() {
        use super::super::source::ExpertComputeBundle;

        let dir = unique_temp_dir("ferrule-expert-raw-payload");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("malformed-bf16.bin");
        std::fs::write(&path, [0x12, 0x34, 0x56]).unwrap();
        let expert = ExpertId::new(1, 3);
        let source = ExpertLoadSource::LocalTensorSet {
            tensors: [
                ExpertMatrixKind::Gate,
                ExpertMatrixKind::Up,
                ExpertMatrixKind::Down,
            ]
            .into_iter()
            .map(|matrix| ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component: ExpertTensorComponent::Weight,
                path: path.clone(),
                offset: 0,
                bytes: 3,
                dtype: "BF16".into(),
                shape: vec![1, 1],
            })
            .collect(),
        };
        let reader = ExpertStreamingReader::new(3);
        for payload in [
            reader.read_load_source(expert, &source).unwrap(),
            reader.read_load_source_concurrent(expert, &source).unwrap(),
        ] {
            assert!(
                payload
                    .tensors
                    .iter()
                    .all(|tensor| tensor.bytes == [0x12, 0x34, 0x56])
            );
            let error = ExpertComputeBundle::from_artifact_payload(payload).unwrap_err();
            assert!(matches!(error, Error::Model { .. }));
            assert!(error.to_string().contains("byte length mismatch"));
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn read_bound_is_checked_before_open_and_short_reads_remain_errors() {
        let dir = unique_temp_dir("ferrule-expert-read-errors");
        let path = dir.join("missing.bin");
        let expert = ExpertId::new(0, 0);
        let source = ExpertLoadSource::LocalShard {
            path: path.clone(),
            offset: 0,
            bytes: 4,
        };
        let reader = ExpertStreamingReader::new(3);
        for error in [
            reader.read_load_source(expert, &source).unwrap_err(),
            reader
                .read_load_source_concurrent(expert, &source)
                .unwrap_err(),
        ] {
            assert!(matches!(error, Error::Model { .. }));
            assert!(error.to_string().contains("bounded read size"));
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, [1, 2]).unwrap();
        let error = ExpertStreamingReader::new(4)
            .read_load_source(expert, &source)
            .unwrap_err();
        assert!(matches!(error, Error::Model { .. }));
        assert!(error.to_string().contains("read_at"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
