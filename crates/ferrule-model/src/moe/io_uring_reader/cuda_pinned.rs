//! CUDA-pinned backing and physical transitions on the parent's single reader.
//!
//! The bookkeeping below contains no ring, files or slab allocations. They stay
//! in `IoUringReadState`, whose existing lock serializes pageable and pinned I/O.

use super::pinned_protocol::{
    PinnedReadOperation, allocate_durable_id, reap_detached_pinned_operations,
    reserve_all_extent_buffers, take_storage_local_operation, take_terminal_pinned_operation,
    unscheduled_pinned_operation_reason, validate_pinned_extents,
};
use super::{
    DIRECT_IO_ALIGNMENT, DirectReadExtent, ExpertIoTransport, ExpertTensorSlice,
    IoUringExpertReader, IoUringReadState, RegisteredBuffer, RegisteredBufferBacking, align_up,
    collect_payloads,
};
use ferrule_backend::cuda::operators::moe::{CudaPinnedHostAllocator, CudaPinnedU8HostBuffer};
use ferrule_common::materialization_io::MaterializationResourceRequirements;
use ferrule_common::{CompletionHub, Error, Result};
use ferrule_common::{
    MaterializationKey, OperationId, RegisteredPinnedAlignedSlabLeaseDescriptor, RegistrationId,
    SlabId,
};
use io_uring::{opcode, types};
use std::collections::{HashMap, VecDeque};

pub(crate) use super::super::streaming::storage::PinnedExpertTensorPayload;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PinnedExpertReadTicket {
    operation_id: u64,
}

pub(crate) struct PinnedExpertReadPlan {
    extents: Vec<DirectReadExtent>,
    payload_count: usize,
    requirements: MaterializationResourceRequirements,
}

pub(crate) struct ReservedPinnedExpertRead {
    pub(crate) ticket: PinnedExpertReadTicket,
    pub(crate) slabs: Box<[RegisteredPinnedAlignedSlabLeaseDescriptor]>,
}

impl PinnedExpertReadPlan {
    pub(crate) const fn requirements(&self) -> MaterializationResourceRequirements {
        self.requirements
    }
}

pub(crate) struct PinnedExpertReadResult {
    pub(crate) payloads: Vec<PinnedExpertTensorPayload>,
}

pub(crate) enum PinnedExpertReadPoll {
    Pending,
    Ready(PinnedExpertReadResult),
    Failed(Error),
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PinnedReadSubmissionState {
    Queued,
    Submitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PinnedReadSubmission {
    operation_id: u64,
    extent_index: usize,
    buffer_index: usize,
    state: PinnedReadSubmissionState,
}

pub(super) struct CudaPinnedBuffer(CudaPinnedU8HostBuffer);

impl CudaPinnedBuffer {
    pub(super) fn is_uniquely_owned(&self) -> bool {
        self.0.is_uniquely_owned()
    }

    #[allow(unsafe_code)]
    pub(super) fn as_mut_ptr(&mut self) -> Result<*mut u8> {
        // SAFETY: callers check availability before scheduling. The parent's
        // exclusive state lock retains the slab until its exact CQE is consumed.
        unsafe { self.0.as_mut_ptr_unique() }
    }

    pub(super) fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl RegisteredBuffer {
    fn new_cuda_pinned(len: usize, allocator: &CudaPinnedHostAllocator) -> Result<Self> {
        let len = align_up(len, DIRECT_IO_ALIGNMENT)?;
        Ok(Self {
            backing: RegisteredBufferBacking::CudaPinned(CudaPinnedBuffer(
                allocator.allocate_u8_aligned(len, DIRECT_IO_ALIGNMENT)?,
            )),
            len,
        })
    }

    fn is_cuda_pinned(&self) -> bool {
        match &self.backing {
            RegisteredBufferBacking::Pageable(_) => false,
            RegisteredBufferBacking::CudaPinned(_) => true,
        }
    }

    fn pinned_range(&self, offset: usize, len: usize) -> Result<CudaPinnedU8HostBuffer> {
        match &self.backing {
            RegisteredBufferBacking::CudaPinned(buffer) => buffer.0.slice(offset, len),
            RegisteredBufferBacking::Pageable(_) => Err(Error::Model {
                message: "io_uring expert reader was not configured with CUDA pinned slabs".into(),
            }),
        }
    }
}

pub(super) struct CudaPinnedReadState {
    operations: HashMap<u64, PinnedReadOperation<PinnedExpertTensorPayload>>,
    operation_order: VecDeque<u64>,
    detached_operations: VecDeque<u64>,
    submissions: HashMap<u64, PinnedReadSubmission>,
    queued_submissions: VecDeque<u64>,
    buffer_busy: Vec<bool>,
    next_operation_id: u64,
    next_submission_id: u64,
}

impl CudaPinnedReadState {
    pub(super) fn new(buffer_count: usize) -> Self {
        Self {
            operations: HashMap::new(),
            operation_order: VecDeque::new(),
            detached_operations: VecDeque::new(),
            submissions: HashMap::new(),
            queued_submissions: VecDeque::new(),
            buffer_busy: vec![false; buffer_count],
            next_operation_id: 1,
            next_submission_id: 1,
        }
    }
}

impl IoUringReadState {
    fn new_cuda_pinned(
        queue_depth: usize,
        buffer_bytes: usize,
        slab_count: usize,
        allocator: &CudaPinnedHostAllocator,
        transport: ExpertIoTransport,
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
        Self::new_with_buffers(queue_depth, buffer_bytes, buffers, transport)
    }

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
            &mut self.pinned.next_operation_id,
            "pinned expert read operation",
        )?;
        let operation = PinnedReadOperation::physical(plan.extents, plan.payload_count);
        self.pinned.operations.insert(operation_id, operation);
        match self.reserve_pinned_operation_buffers(operation_id) {
            Ok(true) => {}
            Ok(false) => {
                self.pinned.operations.remove(&operation_id);
                return Err(Error::Execution {
                    message: "registered pinned expert slabs are temporarily exhausted".into(),
                });
            }
            Err(error) => {
                let Some(mut operation) = self.pinned.operations.remove(&operation_id) else {
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
                self.pinned
                    .operations
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
                let Some(mut operation) = self.pinned.operations.remove(&operation_id) else {
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

    fn submit_reserved_slices_pinned(&mut self, ticket: PinnedExpertReadTicket) -> Result<()> {
        let operation = self
            .pinned
            .operations
            .get_mut(&ticket.operation_id)
            .ok_or_else(|| Error::Model {
                message: format!(
                    "unknown or already consumed pinned expert read operation {}",
                    ticket.operation_id
                ),
            })?;
        operation.authorize_submission()?;
        self.pinned.operation_order.push_back(ticket.operation_id);
        // Admit one extent from a newly authorized operation. The surrounding
        // registry drive authorizes a batch of experts; completion-side refill
        // can then order their remaining extents by physical checkpoint offset
        // instead of letting the first expert monopolize the SQ.
        self.schedule_pinned_reads(1)?;
        self.submit_queued_pinned_reads()?;
        self.detect_stuck_pinned_operation(ticket.operation_id)
    }

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
        if !self.pinned.operations.contains_key(&ticket.operation_id) {
            return Err(Error::Model {
                message: format!(
                    "unknown or already consumed pinned expert read operation {}",
                    ticket.operation_id
                ),
            });
        }

        self.drain_pinned_completions(max_completions)?;
        self.schedule_pinned_reads(self.queue_depth)?;
        self.submit_queued_pinned_reads()?;
        self.reap_detached_pinned_operations();
        self.detect_stuck_pinned_operation(ticket.operation_id)?;
        self.account_pinned_operation_timing(ticket.operation_id);
        self.take_pinned_operation_result(ticket.operation_id)
    }

    fn cancel_slices_pinned(&mut self, ticket: PinnedExpertReadTicket) -> Result<bool> {
        let released = self
            .pinned
            .operations
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
        self.pinned
            .operation_order
            .retain(|operation_id| *operation_id != ticket.operation_id);
        self.release_pinned_buffer_reservations(released)?;
        self.account_pinned_operation_timing(ticket.operation_id);
        Ok(true)
    }

    fn detach_slices_pinned(&mut self, ticket: PinnedExpertReadTicket) -> Result<()> {
        let operation_id = ticket.operation_id;
        if !self.pinned.operations.contains_key(&operation_id) {
            return Ok(());
        }
        if !self.pinned.detached_operations.contains(&operation_id) {
            self.pinned.detached_operations.push_back(operation_id);
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

    fn reap_detached_pinned_operations(&mut self) -> usize {
        reap_detached_pinned_operations(
            &mut self.pinned.operations,
            &mut self.pinned.detached_operations,
        )
    }

    fn progress_detached_pinned_operations(&mut self, max_completions: usize) -> Result<usize> {
        self.submit_queued_pinned_reads()?;
        self.drain_pinned_completions(max_completions)?;
        Ok(self.reap_detached_pinned_operations())
    }

    pub(super) fn react_to_pinned_completions(&mut self) -> Result<()> {
        loop {
            let drained = self.drain_pinned_completions(usize::MAX)?;
            self.schedule_pinned_reads(self.queue_depth)?;
            self.submit_queued_pinned_reads()?;
            self.reap_detached_pinned_operations();
            if drained == 0 {
                return Ok(());
            }
        }
    }

    fn release_pinned_buffer_reservations(&mut self, buffer_indices: Vec<usize>) -> Result<()> {
        for buffer_index in buffer_indices {
            let busy = self
                .pinned
                .buffer_busy
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

    fn fail_pinned_operation(&mut self, operation_id: u64, error: Error) -> Result<()> {
        let released = self
            .pinned
            .operations
            .get_mut(&operation_id)
            .ok_or_else(|| Error::Internal {
                message: format!("cannot fail missing pinned expert operation {operation_id}"),
            })?
            .fail(error);
        self.pinned
            .operation_order
            .retain(|queued| *queued != operation_id);
        self.release_pinned_buffer_reservations(released)?;
        self.account_pinned_operation_timing(operation_id);
        Ok(())
    }

    fn reserve_pinned_operation_buffers(&mut self, operation_id: u64) -> Result<bool> {
        let reusable = self
            .buffers
            .iter()
            .map(RegisteredBuffer::is_available)
            .collect::<Vec<_>>();
        let operation = self
            .pinned
            .operations
            .get_mut(&operation_id)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "cannot reserve slabs for missing pinned expert operation {operation_id}"
                ),
            })?;
        reserve_all_extent_buffers(operation, &mut self.pinned.buffer_busy, &reusable)
    }

    fn next_storage_local_operation(&mut self) -> Option<u64> {
        take_storage_local_operation(&mut self.pinned.operation_order, &self.pinned.operations)
    }

    #[allow(unsafe_code)]
    fn schedule_pinned_reads(&mut self, maximum_new_submissions: usize) -> Result<()> {
        let mut consecutively_blocked = 0usize;
        let mut scheduled = 0usize;
        while scheduled < maximum_new_submissions
            && self.pinned.submissions.len() < self.queue_depth
        {
            let sq_has_capacity = {
                let submission = self.ring.submission();
                !submission.is_full()
            };
            if !sq_has_capacity {
                break;
            }

            let Some(operation_id) = self.next_storage_local_operation() else {
                break;
            };
            let can_submit = self
                .pinned
                .operations
                .get(&operation_id)
                .is_some_and(PinnedReadOperation::can_submit);
            if !can_submit {
                continue;
            }
            if self.pinned.operations[&operation_id].has_no_reservations()
                && !self.reserve_pinned_operation_buffers(operation_id)?
            {
                self.pinned.operation_order.push_back(operation_id);
                consecutively_blocked = consecutively_blocked.saturating_add(1);
                if consecutively_blocked >= self.pinned.operation_order.len() {
                    break;
                }
                continue;
            }
            if !self.pinned.operations[&operation_id].has_all_reservations() {
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

            let extent_index = self.pinned.operations[&operation_id].next_extent;
            let buffer_index =
                self.pinned.operations[&operation_id].reserved_buffer(extent_index)?;
            let extent = &self.pinned.operations[&operation_id].extents[extent_index];
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
                &mut self.pinned.next_submission_id,
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
                self.pinned.operation_order.push_front(operation_id);
                break;
            }

            self.pinned.submissions.insert(
                submission_id,
                PinnedReadSubmission {
                    operation_id,
                    extent_index,
                    buffer_index,
                    state: PinnedReadSubmissionState::Queued,
                },
            );
            self.pinned.queued_submissions.push_back(submission_id);
            self.pinned
                .operations
                .get_mut(&operation_id)
                .expect("queued pinned operation exists")
                .record_submission(extent_index)
                .expect("pinned scheduler submits the next prevalidated extent");
            if self.pinned.operations[&operation_id].can_submit() {
                self.pinned.operation_order.push_back(operation_id);
            }

            scheduled += 1;
            self.stats.submitted_extents = self.stats.submitted_extents.saturating_add(1);
            self.stats.aligned_bytes = self.stats.aligned_bytes.saturating_add(u64::from(len));
            self.stats.peak_queue_depth = self
                .stats
                .peak_queue_depth
                .max(self.pinned.submissions.len());
        }
        Ok(())
    }

    fn submit_queued_pinned_reads(&mut self) -> Result<usize> {
        if self.pinned.queued_submissions.is_empty() {
            return Ok(0);
        }
        let accepted = self
            .ring
            .submit()
            .map_err(|error| Error::context("submit expert io_uring reads", error.into()))?;
        self.confirm_queued_pinned_submissions(accepted)?;
        Ok(accepted)
    }

    fn confirm_queued_pinned_submissions(&mut self, accepted: usize) -> Result<()> {
        let confirmed = accepted.min(self.pinned.queued_submissions.len());
        for _ in 0..confirmed {
            let submission_id = self
                .pinned
                .queued_submissions
                .pop_front()
                .expect("confirmed queued submission exists");
            let submission = self
                .pinned
                .submissions
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

    fn detect_stuck_pinned_operation(&mut self, operation_id: u64) -> Result<()> {
        let Some(operation) = self.pinned.operations.get(&operation_id) else {
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
            .pinned
            .submissions
            .values()
            .filter(|submission| submission.operation_id == operation_id)
            .count();
        let extent_count = operation.extents.len();
        let outstanding = operation.outstanding;
        let has_no_reservations = operation.has_no_reservations();
        let has_all_reservations = operation.has_all_reservations();
        let can_submit = operation.can_submit();
        let scheduler_queued = self
            .pinned
            .operation_order
            .iter()
            .any(|queued| *queued == operation_id);

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
            let scheduler_can_queue = self.pinned.submissions.len() < self.queue_depth && {
                let submission = self.ring.submission();
                !submission.is_full()
            };
            let reusable = if has_no_reservations {
                self.buffers
                    .iter()
                    .enumerate()
                    .filter(|(index, buffer)| {
                        !self.pinned.buffer_busy[*index] && buffer.is_available()
                    })
                    .count()
            } else {
                0
            };
            unscheduled_pinned_operation_reason(
                scheduler_queued,
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

    fn handle_pinned_completion(&mut self, submission_id: u64, result: i32) -> Result<()> {
        let submission = self
            .pinned
            .submissions
            .remove(&submission_id)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "io_uring returned unknown pinned expert submission {submission_id}"
                ),
            })?;
        if submission.state == PinnedReadSubmissionState::Queued {
            self.pinned
                .queued_submissions
                .retain(|queued| *queued != submission_id);
        }
        let assigned_buffer = self
            .pinned
            .operations
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
        let Some(busy) = self.pinned.buffer_busy.get(submission.buffer_index) else {
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
                .pinned
                .operations
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
                .pinned
                .operations
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

    fn account_pinned_operation_timing(&mut self, operation_id: u64) {
        let elapsed = self
            .pinned
            .operations
            .get_mut(&operation_id)
            .and_then(PinnedReadOperation::take_elapsed_if_terminal);
        if let Some(elapsed) = elapsed {
            self.stats.read_us = self
                .stats
                .read_us
                .saturating_add(elapsed.as_micros().min(u128::from(u64::MAX)) as u64);
        }
    }

    fn take_pinned_operation_result(&mut self, operation_id: u64) -> Result<PinnedExpertReadPoll> {
        let terminal = self
            .pinned
            .operations
            .get(&operation_id)
            .is_some_and(PinnedReadOperation::is_terminal);
        if !terminal {
            return Ok(PinnedExpertReadPoll::Pending);
        }
        let mut operation =
            take_terminal_pinned_operation(&mut self.pinned.operations, operation_id)
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

    pub(super) fn prepare_pageable_read(&mut self) -> Result<()> {
        self.progress_detached_pinned_operations(self.queue_depth.max(1))?;
        if !self.pinned.operations.is_empty() {
            return Err(Error::Model {
                message:
                    "pageable io_uring reads cannot run while pinned read operations are active"
                        .into(),
            });
        }
        Ok(())
    }
}

impl IoUringExpertReader {
    pub(crate) fn new_cuda_pinned(
        queue_depth: usize,
        buffer_bytes: usize,
        slab_count: usize,
        allocator: &CudaPinnedHostAllocator,
        transport: ExpertIoTransport,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        Self::from_state(
            IoUringReadState::new_cuda_pinned(
                queue_depth,
                buffer_bytes,
                slab_count,
                allocator,
                transport,
            )?,
            completion_hub,
        )
    }

    pub(crate) fn take_reactor_failure(&self) -> Option<ferrule_common::FailureReason> {
        self.reactor_failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    /// Unlike detach, success here proves the exact ticket's final CQE was consumed.
    pub(crate) fn quiesce_slices_pinned(&self, ticket: PinnedExpertReadTicket) -> Result<bool> {
        let mut state = self.state.lock().map_err(|_| Error::Internal {
            message: "expert io_uring state lock poisoned".into(),
        })?;
        state.cancel_slices_pinned(ticket)?;
        if take_terminal_pinned_operation(&mut state.pinned.operations, ticket.operation_id)
            .is_some()
        {
            return Ok(true);
        }
        Ok(false)
    }

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    fn temporary_shard(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "ferrule-{name}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, vec![0u8; DIRECT_IO_ALIGNMENT]).unwrap();
        path
    }

    fn gpu_reader_fixture(
        name: &str,
    ) -> (
        ferrule_backend::cuda::operators::moe::CudaOperators,
        IoUringExpertReader,
        Vec<ExpertTensorSlice>,
        MaterializationKey,
        PathBuf,
    ) {
        use crate::moe::streaming::{ExpertMatrixKind, ExpertTensorComponent, ExpertTensorKey};
        use ferrule_common::{
            BackendId, ContentHash, DestinationGeneration, DeviceId, ExpertId, LayerId,
            MaterializedResourceId, ModelInstanceId, PayloadEncodingId, SourceGeneration,
            SourceIdentityHash,
        };
        let ops = ferrule_backend::cuda::operators::moe::CudaOperators::new_on_device(0)
            .expect("real CUDA pinned allocator required");
        let reader = IoUringExpertReader::new_cuda_pinned(
            1,
            DIRECT_IO_ALIGNMENT,
            2,
            &ops.pinned_host_allocator(),
            ExpertIoTransport::DirectIoUring,
            CompletionHub::new(),
        )
        .expect("real registered-buffer io_uring required; no fallback");
        let path = temporary_shard(name);
        std::fs::write(&path, vec![0x3f; 2 * DIRECT_IO_ALIGNMENT]).unwrap();
        let slices = [0, DIRECT_IO_ALIGNMENT]
            .map(|offset| ExpertTensorSlice {
                key: ExpertTensorKey::new(0, 0, ExpertMatrixKind::Gate),
                component: ExpertTensorComponent::Weight,
                path: path.clone(),
                offset: offset as u64,
                bytes: DIRECT_IO_ALIGNMENT as u64,
                dtype: "BF16".into(),
                shape: vec![32, 64],
            })
            .to_vec();
        let key = MaterializationKey::new(
            ModelInstanceId::new(1),
            SourceIdentityHash::new([1; 32]),
            ContentHash::new([2; 32]),
            MaterializedResourceId::routed_expert(LayerId::new(0), ExpertId::new(0)),
            PayloadEncodingId::new(1),
            BackendId::new(1),
            DeviceId::new(0),
            SourceGeneration::new(1),
            DestinationGeneration::new(1),
        )
        .unwrap();
        (ops, reader, slices, key, path)
    }

    #[test]
    #[ignore = "requires real CUDA pinned memory and Linux direct io_uring"]
    fn gpu_read_cancel_retains_submitted_slab_until_exact_cqe() {
        let (_ops, reader, slices, key, path) = gpu_reader_fixture("cancel-cqe");
        let read = reader
            .reserve_slices_pinned(
                reader.plan_slices_pinned(&slices).unwrap(),
                OperationId::new(901),
                key,
            )
            .unwrap();
        reader.submit_reserved_slices_pinned(read.ticket).unwrap();
        {
            let state = reader.state.lock().unwrap();
            let operation = &state.pinned.operations[&read.ticket.operation_id];
            assert_eq!((operation.next_extent, operation.outstanding), (1, 1));
            assert_eq!(state.pinned.buffer_busy, [true, true]);
        }
        assert!(reader.cancel_slices_pinned(read.ticket).unwrap());
        assert!(
            !reader.quiesce_slices_pinned(read.ticket).unwrap(),
            "cancel is not a CQE"
        );
        {
            let state = reader.state.lock().unwrap();
            assert_eq!(state.pinned.buffer_busy, [true, false]);
            assert_eq!(state.pinned.submissions.len(), 1);
        }
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        loop {
            reader.react_to_completions().unwrap();
            if reader.quiesce_slices_pinned(read.ticket).unwrap() {
                break;
            }
            assert!(Instant::now() < deadline, "cancelled read CQE timeout");
            std::thread::yield_now();
        }
        assert!(
            reader.quiesce_slices_pinned(read.ticket).is_err(),
            "no second consumption"
        );
        let state = reader.state.lock().unwrap();
        assert!(state.pinned.operations.is_empty());
        assert!(state.pinned.submissions.is_empty());
        assert_eq!(state.pinned.buffer_busy, [false, false]);
        assert_eq!(state.stats.submitted_extents, 1);
        assert!(state.buffers.iter().all(RegisteredBuffer::is_available));
        drop(state);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires real CUDA pinned memory and Linux direct io_uring"]
    fn gpu_read_outer_poll_error_detaches_and_reaps_real_cqe_once() {
        let (_ops, reader, slices, key, path) = gpu_reader_fixture("detach-cqe");
        let read = reader
            .reserve_slices_pinned(
                reader.plan_slices_pinned(&slices).unwrap(),
                OperationId::new(902),
                key,
            )
            .unwrap();
        reader.submit_reserved_slices_pinned(read.ticket).unwrap();
        assert!(reader.poll_slices_pinned(read.ticket, 0).is_err());
        {
            let state = reader.state.lock().unwrap();
            assert_eq!(
                state.pinned.operations[&read.ticket.operation_id].outstanding,
                1
            );
            assert_eq!(state.pinned.buffer_busy, [true, true]);
        }
        reader.detach_slices_pinned(read.ticket).unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        loop {
            reader.react_to_completions().unwrap();
            let state = reader.state.lock().unwrap();
            if state.pinned.operations.is_empty() {
                break;
            }
            assert!(Instant::now() < deadline, "detached read CQE timeout");
            drop(state);
            std::thread::yield_now();
        }
        let mut state = reader.state.lock().unwrap();
        assert!(state.pinned.detached_operations.is_empty());
        assert!(state.pinned.submissions.is_empty());
        assert_eq!(state.pinned.buffer_busy, [false, false]);
        assert_eq!(state.reap_detached_pinned_operations(), 0);
        assert!(state.poll_slices_pinned(read.ticket, 1).is_err());
        assert!(state.buffers.iter().all(RegisteredBuffer::is_available));
        drop(state);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires real CUDA and io_uring; BF16 transfer, not routed FP4 success"]
    fn gpu_read_payload_custody_outlives_cqe_through_real_h2d_event() {
        use ferrule_backend::cuda::operators::moe::CudaArtifactLinearShape;
        let (ops, reader, slices, key, path) = gpu_reader_fixture("read-h2d");
        let read = reader
            .reserve_slices_pinned(
                reader.plan_slices_pinned(&slices[..1]).unwrap(),
                OperationId::new(903),
                key,
            )
            .unwrap();
        reader.submit_reserved_slices_pinned(read.ticket).unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        let mut result = loop {
            match reader.poll_slices_pinned(read.ticket, 1).unwrap() {
                PinnedExpertReadPoll::Ready(result) => break result,
                PinnedExpertReadPoll::Pending => {
                    assert!(Instant::now() < deadline, "read CQE timeout");
                    std::thread::yield_now();
                }
                _ => panic!("read unexpectedly failed or cancelled"),
            }
        };
        assert!(reader.poll_slices_pinned(read.ticket, 1).is_err());
        let payload = result.payloads.pop().unwrap();
        assert_eq!(payload.bytes.as_slice(), vec![0x3f; DIRECT_IO_ALIGNMENT]);
        let shape = CudaArtifactLinearShape::Bf16Bytes {
            out_features: 32,
            in_features: 64,
        };
        let mut frame = ops.allocate_artifact_linear_device(shape).unwrap();
        ops.reset_counters();
        let upload = ops
            .overwrite_artifact_linear_from_pinned_async(&mut frame, shape, payload.bytes, None)
            .unwrap();
        {
            let state = reader.state.lock().unwrap();
            assert_eq!(
                state.pinned.buffer_busy,
                [false, false],
                "CQE releases read reservation only"
            );
            assert!(
                !state.buffers[0].is_available(),
                "H2D ticket still owns the read slab"
            );
        }
        upload.synchronize().unwrap();
        assert!(upload.is_complete().unwrap());
        assert_eq!(ops.counters().host_to_device_copies, 1);
        assert_eq!(
            ops.counters().host_to_device_bytes,
            DIRECT_IO_ALIGNMENT as u64
        );
        assert_eq!(ops.counters().stream_wide_syncs, 0);
        drop(upload);
        assert!(
            reader
                .state
                .lock()
                .unwrap()
                .buffers
                .iter()
                .all(RegisteredBuffer::is_available)
        );
        // Reuse the original slab only after its upload ticket releases custody.
        let next = reader
            .reserve_slices_pinned(
                reader.plan_slices_pinned(&slices[..1]).unwrap(),
                OperationId::new(904),
                key,
            )
            .unwrap();
        assert_eq!(next.slabs[0].slab(), read.slabs[0].slab());
        assert!(reader.quiesce_slices_pinned(next.ticket).unwrap());
        drop(frame);
        assert_eq!(ops.allocator_metrics().live_requested_bytes, 0);
        std::fs::remove_file(path).unwrap();
    }
}
