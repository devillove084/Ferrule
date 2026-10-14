//! Host-only checkpoint layout and bounded I/O capacity planning.
//!
//! This module is Linux-gated for the io_uring alignment contract, not CUDA-gated.
//! Computing a plan or execution reserve allocates no physical pinned memory.

use crate::checkpoint::CheckpointReadPlan;
use crate::moe::io_uring_reader;
use ferrule_common::{Error, Result};
use snafu::Snafu;

#[derive(Debug, Snafu, PartialEq, Eq)]
enum ExpertIoPlanError {
    #[snafu(display("expert I/O execution reserve requires at least one operation slot"))]
    EmptyExecutionReserve,
    #[snafu(display("expert I/O execution reserve slot count {slots} exceeds u64"))]
    ExecutionReserveSlotsOutOfRange { slots: usize },
    #[snafu(display("expert I/O execution reserve {resource} byte calculation overflowed"))]
    ExecutionReserveByteOverflow { resource: &'static str },
}

#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
fn io_plan_error(source: ExpertIoPlanError) -> Error {
    Error::ModelSource {
        source: Box::new(source),
    }
}

#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertIoPlan {
    pub queue_depth: usize,
    pub buffer_bytes: usize,
    pub slab_count: usize,
    pub maximum_extents_per_operation: usize,
}

#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
impl ExpertIoPlan {
    const DEFAULT_PINNED_BYTES: usize = 256 * 1024 * 1024;
    const MAX_QUEUE_DEPTH: usize = 32;

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

#[cfg(test)]
mod tests {
    use super::ExpertIoPlan;

    #[test]
    fn empty_checkpoint_plans_remain_a_host_only_planning_error() {
        let error = ExpertIoPlan::for_checkpoint_plans(std::iter::empty()).unwrap_err();
        assert!(error.to_string().contains("non-empty checkpoint extents"));
    }

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
