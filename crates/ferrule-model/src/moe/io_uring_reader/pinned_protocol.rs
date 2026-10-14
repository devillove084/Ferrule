//! Pinned-read custody, reservation and cancellation protocol, without CUDA.
//!
//! No ring, file table or physical slab is owned here. The parent reader drives
//! these transitions and retains submitted slabs until their exact CQEs.

use super::{DirectReadExtent, FIXED_FILE_CAPACITY};
use ferrule_common::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

pub(super) struct PinnedReadOperation<T> {
    pub(super) extents: Vec<DirectReadExtent>,
    pub(super) payloads: Vec<Option<T>>,
    completed_extents: Vec<bool>,
    extent_buffers: Vec<Option<usize>>,
    pub(super) outstanding: usize,
    pub(super) next_extent: usize,
    submission_authorized: bool,
    pub(super) error: Option<Error>,
    pub(super) cancelled: bool,
    started: Instant,
    timing_accounted: bool,
}

impl<T> PinnedReadOperation<T> {
    #[cfg(test)]
    pub(super) fn new(extents: Vec<DirectReadExtent>, payload_count: usize) -> Self {
        Self::new_inner(extents, payload_count, true)
    }

    pub(super) fn physical(extents: Vec<DirectReadExtent>, payload_count: usize) -> Self {
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
            started: Instant::now(),
            timing_accounted: false,
        }
    }

    pub(super) fn can_submit(&self) -> bool {
        self.submission_authorized
            && !self.cancelled
            && self.error.is_none()
            && self.next_extent < self.extents.len()
    }

    pub(super) fn authorize_submission(&mut self) -> Result<()> {
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

    pub(super) fn has_no_reservations(&self) -> bool {
        self.extent_buffers.iter().all(Option::is_none)
    }

    pub(super) fn has_all_reservations(&self) -> bool {
        self.extent_buffers.iter().all(Option::is_some)
    }

    pub(super) fn reserved_buffer(&self, extent_index: usize) -> Result<usize> {
        self.extent_buffers
            .get(extent_index)
            .copied()
            .flatten()
            .ok_or_else(|| Error::Internal {
                message: format!("pinned expert read extent {extent_index} has no reserved slab"),
            })
    }

    pub(super) fn release_unqueued_reservations(&mut self) -> Vec<usize> {
        self.extent_buffers
            .iter_mut()
            .skip(self.next_extent)
            .filter_map(Option::take)
            .collect()
    }

    pub(super) fn record_submission(&mut self, extent_index: usize) -> Result<()> {
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

    pub(super) fn record_completion(&mut self, extent_index: usize) -> Result<()> {
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

    pub(super) fn fail(&mut self, error: Error) -> Vec<usize> {
        if self.error.is_none() {
            self.error = Some(error);
        }
        self.release_unqueued_reservations()
    }

    pub(super) fn cancel(&mut self) -> Option<Vec<usize>> {
        if self.cancelled || self.is_terminal() {
            return None;
        }
        self.cancelled = true;
        for payload in &mut self.payloads {
            *payload = None;
        }
        Some(self.release_unqueued_reservations())
    }

    pub(super) fn is_terminal(&self) -> bool {
        self.outstanding == 0
            && (self.cancelled || self.error.is_some() || self.next_extent == self.extents.len())
    }

    pub(super) fn take_elapsed_if_terminal(&mut self) -> Option<std::time::Duration> {
        if !self.is_terminal() || self.timing_accounted {
            return None;
        }
        self.timing_accounted = true;
        Some(self.started.elapsed())
    }
}

pub(super) fn take_terminal_pinned_operation<T>(
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

pub(super) fn reap_detached_pinned_operations<T>(
    operations: &mut HashMap<u64, PinnedReadOperation<T>>,
    detached: &mut VecDeque<u64>,
) -> usize {
    let mut pending = VecDeque::with_capacity(detached.len());
    let mut reaped = 0usize;
    while let Some(operation_id) = detached.pop_front() {
        if take_terminal_pinned_operation(operations, operation_id).is_some() {
            reaped = reaped.saturating_add(1);
        } else {
            // A missing map entry is not evidence that the reactor drained the
            // exact ticket; retain it in quarantine until a reaper proves it.
            pending.push_back(operation_id);
        }
    }
    *detached = pending;
    reaped
}

pub(super) fn unscheduled_pinned_operation_reason(
    scheduler_queued: bool,
    scheduler_can_queue: bool,
    has_all_reservations: bool,
    has_no_reservations: bool,
    reusable_buffers: usize,
    extent_count: usize,
) -> Option<String> {
    if scheduler_queued || !scheduler_can_queue {
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

pub(super) fn reserve_all_extent_buffers<T>(
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

pub(super) fn validate_pinned_extents(
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

pub(super) fn allocate_durable_id(next: &mut u64, kind: &str) -> Result<u64> {
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

pub(super) fn take_storage_local_operation<T>(
    order: &mut VecDeque<u64>,
    operations: &HashMap<u64, PinnedReadOperation<T>>,
) -> Option<u64> {
    let mut selected = None;
    let mut waiting = VecDeque::with_capacity(order.len());
    while let Some(operation_id) = order.pop_front() {
        let Some(operation) = operations.get(&operation_id) else {
            continue;
        };
        if !operation.can_submit() {
            continue;
        }
        let extent = &operation.extents[operation.next_extent];
        let key = (extent.file_index, extent.aligned_offset, operation_id);
        if selected.is_none_or(|(selected_key, _)| key < selected_key) {
            if let Some((_, previous)) = selected.replace((key, operation_id)) {
                waiting.push_back(previous);
            }
        } else {
            waiting.push_back(operation_id);
        }
    }
    *order = waiting;
    selected.map(|(_, operation_id)| operation_id)
}

#[cfg(test)]
mod tests {
    use super::super::DIRECT_IO_ALIGNMENT;
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
    fn storage_local_selection_prefers_file_then_offset_without_losing_fifo_work() {
        let mut operations = HashMap::new();
        let mut first = PinnedReadOperation::<()>::new(synthetic_extents(2), 0);
        first.extents[0].aligned_offset = 8 * DIRECT_IO_ALIGNMENT as u64;
        first.extents[1].aligned_offset = 9 * DIRECT_IO_ALIGNMENT as u64;
        let mut second = PinnedReadOperation::<()>::new(synthetic_extents(2), 0);
        second.extents[0].aligned_offset = 2 * DIRECT_IO_ALIGNMENT as u64;
        second.extents[1].aligned_offset = 3 * DIRECT_IO_ALIGNMENT as u64;
        operations.insert(1, first);
        operations.insert(2, second);
        let mut order = VecDeque::from([1, 2]);

        assert_eq!(
            take_storage_local_operation(&mut order, &operations),
            Some(2)
        );
        assert_eq!(order, VecDeque::from([1]));
        order.push_back(2);
        operations.get_mut(&2).unwrap().next_extent = 1;
        assert_eq!(
            take_storage_local_operation(&mut order, &operations),
            Some(2)
        );
        assert_eq!(order, VecDeque::from([1]));
    }

    #[test]
    fn queued_or_saturated_pinned_operation_remains_waiting() {
        assert_eq!(
            unscheduled_pinned_operation_reason(false, false, false, true, 2, 2),
            None
        );
        assert_eq!(
            unscheduled_pinned_operation_reason(true, true, true, false, 0, 2),
            None
        );
        assert!(
            unscheduled_pinned_operation_reason(false, true, false, true, 2, 2)
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
    fn detached_missing_operation_is_not_quiescence_proof() {
        let mut operations = HashMap::<u64, PinnedReadOperation<()>>::new();
        let mut detached = VecDeque::from([7]);
        assert_eq!(
            reap_detached_pinned_operations(&mut operations, &mut detached),
            0
        );
        assert_eq!(
            detached,
            VecDeque::from([7]),
            "unknown owner stays quarantined"
        );
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

    #[test]
    fn terminal_operation_timing_is_accounted_once() {
        let mut operation = PinnedReadOperation::<()>::physical(synthetic_extents(1), 0);
        assert!(operation.take_elapsed_if_terminal().is_none());
        operation.cancel().unwrap();
        assert!(operation.take_elapsed_if_terminal().is_some());
        assert!(operation.take_elapsed_if_terminal().is_none());
    }
}
