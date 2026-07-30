//! Timestamped critical-path and uncovered-wait accounting.

use std::collections::{HashMap, HashSet};

use ahash::RandomState;

use ferrule_common::io_protocol::OperationId;

/// Stable owner identity for one runnable/parked cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CohortId(u64);

impl CohortId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable external output-token identity for immutable accounting snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputTokenId(u64);

impl OutputTokenId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CriticalPhase {
    Read,
    Upload,
    Publish,
    Wait,
    Resume,
    Commit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSpan {
    pub start_ns: u64,
    pub end_ns: u64,
}

impl TimeSpan {
    pub fn new(start_ns: u64, end_ns: u64) -> Result<Self, LedgerError> {
        if end_ns < start_ns {
            return Err(LedgerError::ReversedSpan { start_ns, end_ns });
        }
        Ok(Self { start_ns, end_ns })
    }

    const fn duration(self) -> u64 {
        self.end_ns - self.start_ns
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    ReversedSpan { start_ns: u64, end_ns: u64 },
    DuplicateOutputToken(OutputTokenId),
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "critical-path ledger violation: {self:?}")
    }
}

impl std::error::Error for LedgerError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhaseDurations {
    pub read_ns: u64,
    pub upload_ns: u64,
    pub publish_ns: u64,
    pub wait_ns: u64,
    pub resume_ns: u64,
    pub commit_ns: u64,
}

impl PhaseDurations {
    fn set(&mut self, phase: CriticalPhase, duration: u64) {
        match phase {
            CriticalPhase::Read => self.read_ns = duration,
            CriticalPhase::Upload => self.upload_ns = duration,
            CriticalPhase::Publish => self.publish_ns = duration,
            CriticalPhase::Wait => self.wait_ns = duration,
            CriticalPhase::Resume => self.resume_ns = duration,
            CriticalPhase::Commit => self.commit_ns = duration,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputTokenSnapshot {
    pub token: OutputTokenId,
    /// Number of externally visible tokens reconciled to the transaction commit
    /// that captured this immutable per-token snapshot.
    pub externally_committed_tokens: usize,
    pub captured_at_ns: u64,
    pub operation_phases: HashMap<OperationId, PhaseDurations, RandomState>,
    pub cohort_phases: HashMap<CohortId, PhaseDurations, RandomState>,
    pub read_ns: u64,
    pub upload_ns: u64,
    pub publish_ns: u64,
    pub wait_ns: u64,
    pub covered_wait_ns: u64,
    pub uncovered_wait_ns: u64,
    pub resume_ns: u64,
    pub commit_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SharedWait {
    operation: OperationId,
    cohort: CohortId,
    span: TimeSpan,
}

/// DAG-oriented interval ledger. Shared waits are identified by operation and
/// unioned before accounting, so N waiters never multiply one physical stall.
#[derive(Debug, Default)]
pub struct CriticalPathLedger {
    operation_phases: HashMap<(OperationId, CriticalPhase), Vec<TimeSpan>, RandomState>,
    cohort_phases: HashMap<(CohortId, CriticalPhase), Vec<TimeSpan>, RandomState>,
    shared_waits: Vec<SharedWait>,
    runnable_work: Vec<TimeSpan>,
    snapshots: HashMap<OutputTokenId, OutputTokenSnapshot, RandomState>,
}

impl CriticalPathLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_operation_phase(
        &mut self,
        operation: OperationId,
        phase: CriticalPhase,
        start_ns: u64,
        end_ns: u64,
    ) -> Result<(), LedgerError> {
        let span = TimeSpan::new(start_ns, end_ns)?;
        self.operation_phases
            .entry((operation, phase))
            .or_default()
            .push(span);
        Ok(())
    }

    pub fn record_cohort_phase(
        &mut self,
        cohort: CohortId,
        phase: CriticalPhase,
        start_ns: u64,
        end_ns: u64,
    ) -> Result<(), LedgerError> {
        let span = TimeSpan::new(start_ns, end_ns)?;
        self.cohort_phases
            .entry((cohort, phase))
            .or_default()
            .push(span);
        Ok(())
    }

    pub fn record_shared_wait(
        &mut self,
        operation: OperationId,
        cohort: CohortId,
        start_ns: u64,
        end_ns: u64,
    ) -> Result<(), LedgerError> {
        let span = TimeSpan::new(start_ns, end_ns)?;
        self.shared_waits.push(SharedWait {
            operation,
            cohort,
            span,
        });
        self.operation_phases
            .entry((operation, CriticalPhase::Wait))
            .or_default()
            .push(span);
        self.cohort_phases
            .entry((cohort, CriticalPhase::Wait))
            .or_default()
            .push(span);
        Ok(())
    }

    /// Records useful independent execution that legally covers dependency wait.
    pub fn record_runnable_work(&mut self, start_ns: u64, end_ns: u64) -> Result<(), LedgerError> {
        self.runnable_work.push(TimeSpan::new(start_ns, end_ns)?);
        Ok(())
    }

    pub fn snapshot_output(
        &mut self,
        token: OutputTokenId,
        externally_committed_tokens: usize,
        operations: impl IntoIterator<Item = OperationId>,
        cohorts: impl IntoIterator<Item = CohortId>,
        captured_at_ns: u64,
    ) -> Result<&OutputTokenSnapshot, LedgerError> {
        if self.snapshots.contains_key(&token) {
            return Err(LedgerError::DuplicateOutputToken(token));
        }
        let operations: HashSet<_, RandomState> = operations.into_iter().collect();
        let cohorts: HashSet<_, RandomState> = cohorts.into_iter().collect();
        let mut operation_phases = HashMap::default();
        for operation in &operations {
            operation_phases.insert(
                *operation,
                self.operation_durations(*operation, captured_at_ns),
            );
        }
        let mut cohort_phases = HashMap::default();
        for cohort in &cohorts {
            cohort_phases.insert(*cohort, self.cohort_durations(*cohort, captured_at_ns));
        }

        let read_ns = union_duration(self.selected_operation_spans(
            &operations,
            CriticalPhase::Read,
            captured_at_ns,
        ));
        let upload_ns = union_duration(self.selected_operation_spans(
            &operations,
            CriticalPhase::Upload,
            captured_at_ns,
        ));
        let publish_ns = union_duration(self.selected_operation_spans(
            &operations,
            CriticalPhase::Publish,
            captured_at_ns,
        ));
        let resume_ns = union_duration(self.selected_cohort_spans(
            &cohorts,
            CriticalPhase::Resume,
            captured_at_ns,
        ));
        let commit_ns = union_duration(self.selected_cohort_spans(
            &cohorts,
            CriticalPhase::Commit,
            captured_at_ns,
        ));
        let waits = merge_spans(
            self.shared_waits
                .iter()
                .filter(|wait| {
                    operations.contains(&wait.operation) || cohorts.contains(&wait.cohort)
                })
                .filter_map(|wait| clip(wait.span, captured_at_ns))
                .collect(),
        );
        let runnable = merge_spans(
            self.runnable_work
                .iter()
                .filter_map(|span| clip(*span, captured_at_ns))
                .collect(),
        );
        let wait_ns: u64 = waits.iter().copied().map(TimeSpan::duration).sum();
        let covered_wait_ns = intersection_duration(&waits, &runnable);
        let uncovered_wait_ns = wait_ns.saturating_sub(covered_wait_ns);

        self.snapshots.insert(
            token,
            OutputTokenSnapshot {
                token,
                externally_committed_tokens,
                captured_at_ns,
                operation_phases,
                cohort_phases,
                read_ns,
                upload_ns,
                publish_ns,
                wait_ns,
                covered_wait_ns,
                uncovered_wait_ns,
                resume_ns,
                commit_ns,
            },
        );
        Ok(self
            .snapshots
            .get(&token)
            .expect("snapshot was inserted immediately above"))
    }

    pub fn output(&self, token: OutputTokenId) -> Option<&OutputTokenSnapshot> {
        self.snapshots.get(&token)
    }

    fn operation_durations(&self, operation: OperationId, captured_at_ns: u64) -> PhaseDurations {
        let mut durations = PhaseDurations::default();
        for phase in [
            CriticalPhase::Read,
            CriticalPhase::Upload,
            CriticalPhase::Publish,
            CriticalPhase::Wait,
            CriticalPhase::Resume,
        ] {
            let spans = self
                .operation_phases
                .get(&(operation, phase))
                .into_iter()
                .flatten()
                .filter_map(|span| clip(*span, captured_at_ns))
                .collect();
            durations.set(phase, union_duration(spans));
        }
        durations
    }

    fn cohort_durations(&self, cohort: CohortId, captured_at_ns: u64) -> PhaseDurations {
        let mut durations = PhaseDurations::default();
        for phase in [
            CriticalPhase::Read,
            CriticalPhase::Upload,
            CriticalPhase::Publish,
            CriticalPhase::Wait,
            CriticalPhase::Resume,
            CriticalPhase::Commit,
        ] {
            let spans = self
                .cohort_phases
                .get(&(cohort, phase))
                .into_iter()
                .flatten()
                .filter_map(|span| clip(*span, captured_at_ns))
                .collect();
            durations.set(phase, union_duration(spans));
        }
        durations
    }

    fn selected_operation_spans(
        &self,
        operations: &HashSet<OperationId, RandomState>,
        phase: CriticalPhase,
        captured_at_ns: u64,
    ) -> Vec<TimeSpan> {
        operations
            .iter()
            .flat_map(|operation| {
                self.operation_phases
                    .get(&(*operation, phase))
                    .into_iter()
                    .flatten()
            })
            .filter_map(|span| clip(*span, captured_at_ns))
            .collect()
    }

    fn selected_cohort_spans(
        &self,
        cohorts: &HashSet<CohortId, RandomState>,
        phase: CriticalPhase,
        captured_at_ns: u64,
    ) -> Vec<TimeSpan> {
        cohorts
            .iter()
            .flat_map(|cohort| {
                self.cohort_phases
                    .get(&(*cohort, phase))
                    .into_iter()
                    .flatten()
            })
            .filter_map(|span| clip(*span, captured_at_ns))
            .collect()
    }
}

fn clip(span: TimeSpan, end_ns: u64) -> Option<TimeSpan> {
    let end = span.end_ns.min(end_ns);
    (end > span.start_ns).then_some(TimeSpan {
        start_ns: span.start_ns,
        end_ns: end,
    })
}

fn union_duration(spans: Vec<TimeSpan>) -> u64 {
    merge_spans(spans).into_iter().map(TimeSpan::duration).sum()
}

fn merge_spans(mut spans: Vec<TimeSpan>) -> Vec<TimeSpan> {
    spans.sort_unstable_by_key(|span| (span.start_ns, span.end_ns));
    let mut merged: Vec<TimeSpan> = Vec::with_capacity(spans.len());
    for span in spans {
        if let Some(last) = merged.last_mut()
            && span.start_ns <= last.end_ns
        {
            last.end_ns = last.end_ns.max(span.end_ns);
            continue;
        }
        merged.push(span);
    }
    merged
}

fn intersection_duration(left: &[TimeSpan], right: &[TimeSpan]) -> u64 {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut total = 0u64;
    while left_index < left.len() && right_index < right.len() {
        let start = left[left_index].start_ns.max(right[right_index].start_ns);
        let end = left[left_index].end_ns.min(right[right_index].end_ns);
        total = total.saturating_add(end.saturating_sub(start));
        if left[left_index].end_ns <= right[right_index].end_ns {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    total
}
