//! Linux/CUDA-pinned storage adapter; physical read custody stays in io_uring.

use super::super::source::{ExpertId, ExpertLoadSource, ExpertTensorSlice, slices_for_load_source};
use super::{
    ExpertIoPlan, ExpertIoTransport, ExpertStreamingReader, parse_expert_io_usize,
    stale_source_identity_error, transport_error, validate_io_uring_transport,
};
use crate::checkpoint::{CheckpointReadPlan, CheckpointSourceFileIdentity};
use crate::moe::io_uring_reader;
use ferrule_common::{CompletionHub, Error, Result, StaleReason};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(super) type PinnedSourceFiles =
    Arc<Mutex<HashMap<PinnedExpertReadTicket, Arc<[CheckpointSourceFileIdentity]>>>>;

/// Completed pinned bytes plus their semantic descriptor, independent of transport.
/// Moving this value transfers the existing slab view; it creates no new lease.
pub(crate) struct PinnedExpertTensorPayload {
    pub(crate) slice: ExpertTensorSlice,
    pub(crate) bytes: ferrule_backend::cuda::operators::moe::CudaPinnedU8HostBuffer,
}

pub(crate) struct PinnedExpertArtifactPayload {
    pub(crate) expert: ExpertId,
    pub(crate) tensors: Vec<PinnedExpertTensorPayload>,
}

pub(crate) struct PinnedExpertLoadPlan {
    expert: ExpertId,
    reader: io_uring_reader::PinnedExpertReadPlan,
    source_files: Arc<[CheckpointSourceFileIdentity]>,
}

impl PinnedExpertLoadPlan {
    pub(crate) const fn requirements(
        &self,
    ) -> ferrule_common::materialization_io::MaterializationResourceRequirements {
        self.reader.requirements()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PinnedExpertReadTicket {
    expert: ExpertId,
    reader: io_uring_reader::PinnedExpertReadTicket,
}

pub(crate) enum PinnedExpertReadPoll {
    Pending,
    Ready(PinnedExpertArtifactPayload),
    Failed(Error),
    Cancelled,
}

pub(crate) struct ReservedPinnedExpertLoad {
    pub(crate) ticket: PinnedExpertReadTicket,
    pub(crate) slabs: Box<[ferrule_common::RegisteredPinnedAlignedSlabLeaseDescriptor]>,
}

impl ExpertIoPlan {
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
}

impl ExpertStreamingReader {
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
                pinned_source_files: PinnedSourceFiles::default(),
            },
            plan,
        ))
    }

    pub(crate) fn physical_resource_capacity(
        &self,
    ) -> Result<Option<ferrule_common::materialization_io::MaterializationResourceRequirements>>
    {
        self.io_uring
            .as_ref()
            .map_or(Ok(None), |reader| reader.physical_resource_capacity())
    }

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

    pub(crate) fn cancel_load_source_pinned(&self, ticket: PinnedExpertReadTicket) -> Result<bool> {
        let reader = self.io_uring.as_ref().ok_or_else(|| Error::Model {
            message: "pinned expert read ticket requires the io_uring backend".into(),
        })?;
        self.pinned_source_files_for_ticket(ticket)?;
        reader.cancel_slices_pinned(ticket.reader)
    }

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

    pub(crate) fn take_pinned_reactor_failure(&self) -> Option<ferrule_common::FailureReason> {
        self.io_uring
            .as_ref()
            .and_then(|reader| reader.take_reactor_failure())
    }

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

    pub(crate) fn detach_load_source_pinned(&self, ticket: PinnedExpertReadTicket) -> Result<()> {
        let reader = self.io_uring.as_ref().ok_or_else(|| Error::Model {
            message: "pinned expert read ticket requires the io_uring backend".into(),
        })?;
        let detach = reader.detach_slices_pinned(ticket.reader);
        self.forget_pinned_source_files(ticket)?;
        detach
    }

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

    fn bounded_slices_for_load_source(
        &self,
        expert: ExpertId,
        load_source: &ExpertLoadSource,
    ) -> Result<Vec<ExpertTensorSlice>> {
        let slices = slices_for_load_source(expert, load_source)?;
        self.validate_bounded_slices(&slices)?;
        Ok(slices)
    }

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
}

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
