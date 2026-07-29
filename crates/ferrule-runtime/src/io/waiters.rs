//! Bidirectional physical-operation waiter indices and targeted continuation wakeup.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use ferrule_common::io_protocol::{ContinuationId, OperationId, WaiterId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaiterIndexError {
    DuplicateWaiter(WaiterId),
    UnknownWaiter(WaiterId),
}

impl std::fmt::Display for WaiterIndexError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "waiter index violation: {self:?}")
    }
}

impl std::error::Error for WaiterIndexError {}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct WaiterDetach {
    pub waiter: Option<WaiterId>,
    pub operations_losing_last_waiter: BTreeSet<OperationId>,
    pub continuation_became_empty: Option<ContinuationId>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OperationResolution {
    pub operation: Option<OperationId>,
    pub completed_waiters: BTreeSet<WaiterId>,
    pub ready_continuations: BTreeSet<ContinuationId>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ContinuationDetach {
    pub continuation: Option<ContinuationId>,
    pub removed_waiters: BTreeSet<WaiterId>,
    pub operations_losing_last_waiter: BTreeSet<OperationId>,
}

/// Exact two-way operation/waiter graph plus continuation unresolved sets.
#[derive(Debug, Default)]
pub struct WaiterIndex {
    load_to_waiters: BTreeMap<OperationId, BTreeSet<WaiterId>>,
    waiter_to_loads: BTreeMap<WaiterId, BTreeSet<OperationId>>,
    continuation_waiters: BTreeMap<ContinuationId, BTreeSet<WaiterId>>,
    continuation_unresolved: BTreeMap<ContinuationId, BTreeSet<OperationId>>,
    ready_queue: VecDeque<ContinuationId>,
    ready_set: BTreeSet<ContinuationId>,
    known_waiters: BTreeSet<WaiterId>,
}

impl WaiterIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        waiter: WaiterId,
        operations: impl IntoIterator<Item = OperationId>,
    ) -> Result<bool, WaiterIndexError> {
        if !self.known_waiters.insert(waiter) {
            return Err(WaiterIndexError::DuplicateWaiter(waiter));
        }
        let operations: BTreeSet<_> = operations.into_iter().collect();
        if operations.is_empty() {
            self.queue_ready(waiter.continuation());
            return Ok(true);
        }
        for operation in &operations {
            self.load_to_waiters
                .entry(*operation)
                .or_default()
                .insert(waiter);
        }
        self.waiter_to_loads.insert(waiter, operations.clone());
        self.continuation_waiters
            .entry(waiter.continuation())
            .or_default()
            .insert(waiter);
        self.continuation_unresolved
            .entry(waiter.continuation())
            .or_default()
            .extend(operations);
        self.remove_ready(waiter.continuation());
        Ok(false)
    }

    pub fn contains_waiter(&self, waiter: WaiterId) -> bool {
        self.waiter_to_loads.contains_key(&waiter)
    }

    pub fn has_seen_waiter(&self, waiter: WaiterId) -> bool {
        self.known_waiters.contains(&waiter)
    }

    pub fn active_waiters(&self) -> impl Iterator<Item = WaiterId> + '_ {
        self.waiter_to_loads.keys().copied()
    }

    pub fn waiter_count(&self, operation: OperationId) -> usize {
        self.load_to_waiters
            .get(&operation)
            .map_or(0, BTreeSet::len)
    }

    pub fn waiters_for(&self, operation: OperationId) -> impl Iterator<Item = WaiterId> + '_ {
        self.load_to_waiters
            .get(&operation)
            .into_iter()
            .flatten()
            .copied()
    }

    pub fn loads_for(&self, waiter: WaiterId) -> Option<&BTreeSet<OperationId>> {
        self.waiter_to_loads.get(&waiter)
    }

    pub fn unresolved_for(&self, continuation: ContinuationId) -> Option<&BTreeSet<OperationId>> {
        self.continuation_unresolved.get(&continuation)
    }

    pub fn detach_waiter(&mut self, waiter: WaiterId) -> Result<WaiterDetach, WaiterIndexError> {
        let operations = self
            .waiter_to_loads
            .remove(&waiter)
            .ok_or(WaiterIndexError::UnknownWaiter(waiter))?;
        let mut lost_last = BTreeSet::new();
        for operation in operations {
            let waiters = self
                .load_to_waiters
                .get_mut(&operation)
                .expect("bidirectional waiter edge must exist");
            waiters.remove(&waiter);
            if waiters.is_empty() {
                self.load_to_waiters.remove(&operation);
                lost_last.insert(operation);
            }
        }
        let continuation = waiter.continuation();
        let waiters = self
            .continuation_waiters
            .get_mut(&continuation)
            .expect("registered waiter must have a continuation edge");
        waiters.remove(&waiter);
        let continuation_became_empty = if waiters.is_empty() {
            self.continuation_waiters.remove(&continuation);
            self.continuation_unresolved.remove(&continuation);
            self.remove_ready(continuation);
            Some(continuation)
        } else {
            self.rebuild_unresolved(continuation);
            None
        };
        Ok(WaiterDetach {
            waiter: Some(waiter),
            operations_losing_last_waiter: lost_last,
            continuation_became_empty,
        })
    }

    /// Resolves exactly one operation and queues only continuations whose complete
    /// unresolved union became empty.
    pub fn satisfy_operation(&mut self, operation: OperationId) -> OperationResolution {
        let waiters = self.load_to_waiters.remove(&operation).unwrap_or_default();
        let mut completed_waiters = BTreeSet::new();
        let mut affected = BTreeSet::new();
        for waiter in waiters {
            affected.insert(waiter.continuation());
            let loads = self
                .waiter_to_loads
                .get_mut(&waiter)
                .expect("bidirectional waiter edge must exist");
            loads.remove(&operation);
            if loads.is_empty() {
                self.waiter_to_loads.remove(&waiter);
                if let Some(continuation_waiters) =
                    self.continuation_waiters.get_mut(&waiter.continuation())
                {
                    continuation_waiters.remove(&waiter);
                }
                completed_waiters.insert(waiter);
            }
        }
        let mut ready = BTreeSet::new();
        for continuation in affected {
            self.rebuild_unresolved(continuation);
            if self
                .continuation_unresolved
                .get(&continuation)
                .is_some_and(BTreeSet::is_empty)
            {
                self.continuation_waiters.remove(&continuation);
                self.continuation_unresolved.remove(&continuation);
                self.queue_ready(continuation);
                ready.insert(continuation);
            }
        }
        OperationResolution {
            operation: Some(operation),
            completed_waiters,
            ready_continuations: ready,
        }
    }

    /// Detaches all logical demand owned by a failed/cancelled continuation.
    pub fn detach_continuation(&mut self, continuation: ContinuationId) -> ContinuationDetach {
        let waiters = self
            .continuation_waiters
            .get(&continuation)
            .cloned()
            .unwrap_or_default();
        let mut result = ContinuationDetach {
            continuation: Some(continuation),
            ..ContinuationDetach::default()
        };
        for waiter in waiters {
            if let Ok(detached) = self.detach_waiter(waiter) {
                result.removed_waiters.insert(waiter);
                result
                    .operations_losing_last_waiter
                    .extend(detached.operations_losing_last_waiter);
            }
        }
        self.remove_ready(continuation);
        result
    }

    pub fn continuations_waiting_on(&self, operation: OperationId) -> BTreeSet<ContinuationId> {
        self.load_to_waiters
            .get(&operation)
            .into_iter()
            .flatten()
            .map(|waiter| waiter.continuation())
            .collect()
    }

    pub fn pop_ready(&mut self) -> Option<ContinuationId> {
        let continuation = self.ready_queue.pop_front()?;
        self.ready_set.remove(&continuation);
        Some(continuation)
    }

    pub fn requeue_ready_front(&mut self, continuation: ContinuationId) {
        if self.ready_set.insert(continuation) {
            self.ready_queue.push_front(continuation);
        }
    }

    pub fn ready_len(&self) -> usize {
        self.ready_queue.len()
    }

    pub fn ready_front(&self) -> Option<ContinuationId> {
        self.ready_queue.front().copied()
    }

    pub fn drain_ready(&mut self) -> Vec<ContinuationId> {
        self.ready_set.clear();
        self.ready_queue.drain(..).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.waiter_to_loads.is_empty()
            && self.load_to_waiters.is_empty()
            && self.continuation_waiters.is_empty()
            && self.continuation_unresolved.is_empty()
            && self.ready_queue.is_empty()
    }

    fn rebuild_unresolved(&mut self, continuation: ContinuationId) {
        let mut unresolved = BTreeSet::new();
        if let Some(waiters) = self.continuation_waiters.get(&continuation) {
            for waiter in waiters {
                if let Some(loads) = self.waiter_to_loads.get(waiter) {
                    unresolved.extend(loads.iter().copied());
                }
            }
        }
        if unresolved.is_empty() && !self.continuation_waiters.contains_key(&continuation) {
            self.continuation_unresolved.remove(&continuation);
        } else {
            self.continuation_unresolved
                .insert(continuation, unresolved);
        }
    }

    fn queue_ready(&mut self, continuation: ContinuationId) {
        if self.ready_set.insert(continuation) {
            self.ready_queue.push_back(continuation);
        }
    }

    fn remove_ready(&mut self, continuation: ContinuationId) {
        if self.ready_set.remove(&continuation) {
            self.ready_queue.retain(|queued| *queued != continuation);
        }
    }
}
