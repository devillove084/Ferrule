//! CUDA-independent custody ledger for one lexical owner-local submission scope.
//! No dtype, transaction, stream event, or allocation event is completion evidence.

pub(super) const MAX_ROOTS: usize = 16;
pub(super) const MAX_OPERATIONS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::transformer) enum BarrierReason {
    HostRead,
    CpuRouter,
    ExternalExperts,
    TensorCollective,
    KvPublish,
    RecurrentState,
    WorkspaceGrowth,
    CacheEvictionOrRebind,
    Diagnostic,
    PinBudget,
    SegmentExit,
    OpaqueOperator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Ready,
    Submitting,
    Pending,
    Unknown,
}

pub(super) struct SubmissionSegment<R> {
    roots: [Option<(u64, usize, R)>; MAX_ROOTS],
    limit: usize,
    bytes: usize,
    operations: usize,
    phase: Phase,
    external_error_ready: bool,
    pub(in crate::transformer) last_barrier: Option<BarrierReason>,
}

impl<R> SubmissionSegment<R> {
    pub fn new(limit: usize) -> Option<Self> {
        (limit != 0).then(|| Self {
            roots: std::array::from_fn(|_| None),
            limit,
            bytes: 0,
            operations: 0,
            phase: Phase::Ready,
            external_error_ready: false,
            last_barrier: None,
        })
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn root_count(&self) -> usize {
        self.roots.iter().flatten().count()
    }

    pub fn contains(&self, key: u64, bytes: usize) -> bool {
        self.roots
            .iter()
            .flatten()
            .any(|(k, n, _)| (*k, *n) == (key, bytes))
    }

    pub fn can_fit(&self, bytes: usize, roots: usize) -> bool {
        matches!(self.phase, Phase::Ready | Phase::Pending)
            && self.operations < MAX_OPERATIONS
            && self
                .bytes
                .checked_add(bytes)
                .is_some_and(|n| n <= self.limit)
            && self
                .root_count()
                .checked_add(roots)
                .is_some_and(|n| n <= MAX_ROOTS)
    }

    pub fn begin_operation(&mut self) -> bool {
        if !self.can_fit(0, 0) {
            return false;
        }
        self.external_error_ready = false;
        self.phase = Phase::Submitting;
        true
    }

    /// Must run before the first launch using the buffer. Duplicate exact views
    /// share one root; distinct views are conservatively charged separately.
    pub fn pin(&mut self, key: u64, bytes: usize, root: R) -> Result<(), R> {
        if self.phase != Phase::Submitting {
            return Err(root);
        }
        if self.contains(key, bytes) {
            return Ok(());
        }
        if self.bytes.checked_add(bytes).is_none_or(|n| n > self.limit) {
            return Err(root);
        }
        let Some(slot) = self.roots.iter_mut().find(|slot| slot.is_none()) else {
            return Err(root);
        };
        *slot = Some((key, bytes, root));
        self.bytes += bytes;
        Ok(())
    }

    pub fn can_defer_ordered_proof(&self) -> bool {
        self.phase == Phase::Submitting
    }

    pub fn finish_operation(&mut self) -> bool {
        if self.phase != Phase::Submitting {
            return false;
        }
        self.operations += 1;
        self.phase = Phase::Pending;
        true
    }

    pub fn has_pending(&self) -> bool {
        self.phase != Phase::Ready || self.root_count() != 0
    }

    /// An idle boundary needs no new proof: there is no local producer left to
    /// wait for. Still record it so an external error is distinguished from an
    /// error after local execution resumes (including opaque synchronous work).
    /// Leave `last_barrier` unchanged: this is not an additional consumer proof.
    pub(in crate::transformer) fn note_idle_barrier(&mut self, reason: BarrierReason) -> bool {
        if self.has_pending() {
            return false;
        }
        self.external_error_ready = reason == BarrierReason::ExternalExperts;
        true
    }

    /// This is local producer evidence only. External consumers remain under
    /// their owner's independent holds/drain protocol, even after scope exit.
    pub fn can_return_external_error(&self) -> bool {
        !self.has_pending() && self.external_error_ready
    }

    pub fn is_unknown(&self) -> bool {
        self.phase == Phase::Unknown
    }

    pub fn quarantine(&mut self) {
        self.external_error_ready = false;
        self.phase = Phase::Unknown;
    }

    /// Only the CUDA owner calls this, after its actual consumer and transfer
    /// proofs. A later successful sync cannot wash a previous Unknown.
    pub(in crate::transformer) fn retire_after_proof(&mut self, reason: BarrierReason) -> bool {
        if matches!(self.phase, Phase::Unknown | Phase::Submitting) {
            return false;
        }
        self.roots = std::array::from_fn(|_| None);
        self.bytes = 0;
        self.operations = 0;
        self.phase = Phase::Ready;
        self.external_error_ready = reason == BarrierReason::ExternalExperts;
        self.last_barrier = Some(reason);
        true
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/transformer/root_submission_state.rs"]
mod tests;
