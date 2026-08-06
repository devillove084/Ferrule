//! Bounded aging and deficit-round-robin for runnable I/O transitions.

use std::collections::VecDeque;

use snafu::Snafu;

/// Deterministic fairness configuration. Costs and quanta use the same units
/// (normally marginal physical bytes, with metadata-only transitions costing one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FairQueueConfig {
    pub model_warmup_quantum: u64,
    pub transaction_prefetch_quantum: u64,
    pub required_quantum: u64,
    pub max_surplus: u64,
    pub debt_limit: u64,
    pub starvation_ticks: u64,
    pub max_transition_cost: u64,
}

impl Default for FairQueueConfig {
    fn default() -> Self {
        Self {
            model_warmup_quantum: 1,
            transaction_prefetch_quantum: 2 * 1024,
            required_quantum: 8 * 1024,
            max_surplus: 64 * 1024,
            debt_limit: 64 * 1024,
            starvation_ticks: 8,
            max_transition_cost: 64 * 1024,
        }
    }
}

impl FairQueueConfig {
    /// Derives byte-scaled fairness for a physical production backend.
    ///
    /// `limits` must already have passed `MaterializationResourceLimits::validate`. Total
    /// byte capacities are used directly instead of inferring a per-request size
    /// from slot counts. All Required work may consume the execution reserve and
    /// receives the largest scheduling quantum.
    pub fn for_production(
        limits: ferrule_common::materialization_io::MaterializationResourceLimits,
    ) -> Result<Self, FairQueueError> {
        const STARVATION_TICKS: u64 = 8;
        const MAX_COUNTER: u64 = i64::MAX as u64;

        let capacity = limits.capacity;
        let max_transition_cost = capacity
            .storage_read_bytes
            .max(capacity.pinned_host_bytes)
            .max(capacity.h2d_bytes)
            .max(capacity.device_install_bytes);
        if max_transition_cost == 0 {
            return Err(FairQueueError::ZeroMaxTransitionCost);
        }
        if max_transition_cost > MAX_COUNTER {
            return Err(FairQueueError::CostOutOfRange);
        }

        let progress_quantum = max_transition_cost
            .checked_div(STARVATION_TICKS)
            .expect("the production starvation interval is non-zero")
            .saturating_add(u64::from(
                !max_transition_cost.is_multiple_of(STARVATION_TICKS),
            ))
            .max(1);
        let scaled_quantum = |weight: u64| {
            progress_quantum
                .checked_mul(weight)
                .unwrap_or(MAX_COUNTER)
                .min(max_transition_cost)
                .min(MAX_COUNTER)
        };

        Self {
            model_warmup_quantum: progress_quantum,
            transaction_prefetch_quantum: scaled_quantum(2),
            required_quantum: scaled_quantum(8),
            max_surplus: max_transition_cost.min(MAX_COUNTER),
            debt_limit: max_transition_cost.min(MAX_COUNTER),
            starvation_ticks: STARVATION_TICKS,
            max_transition_cost,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, FairQueueError> {
        if self.model_warmup_quantum == 0
            || self.transaction_prefetch_quantum == 0
            || self.required_quantum == 0
        {
            return Err(FairQueueError::ZeroQuantum);
        }
        if self.max_transition_cost == 0 {
            return Err(FairQueueError::ZeroMaxTransitionCost);
        }
        if self.model_warmup_quantum > i64::MAX as u64
            || self.transaction_prefetch_quantum > i64::MAX as u64
            || self.required_quantum > i64::MAX as u64
            || self.max_transition_cost > i64::MAX as u64
            || self.max_surplus > i64::MAX as u64
            || self.debt_limit > i64::MAX as u64
        {
            return Err(FairQueueError::CostOutOfRange);
        }
        Ok(self)
    }

    const fn quantum(self, band: FairQueueBand) -> u64 {
        match band {
            FairQueueBand::Background => self.model_warmup_quantum,
            FairQueueBand::Prefetch => self.transaction_prefetch_quantum,
            FairQueueBand::Required => self.required_quantum,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum FairQueueError {
    #[snafu(display("fair queue quanta must be non-zero"))]
    ZeroQuantum,
    #[snafu(display("fair queue maximum transition cost must be non-zero"))]
    ZeroMaxTransitionCost,
    #[snafu(display("fair queue cost configuration exceeds the signed deficit range"))]
    CostOutOfRange,
    #[snafu(display("fair queue entries must have non-zero cost"))]
    ZeroCost,
    #[snafu(display("fair queue transition cost {cost} exceeds maximum {maximum}"))]
    TransitionTooLarge { cost: u64, maximum: u64 },
}

/// Owner-side classification of a queued item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FairQueueEntryState {
    /// The item is current and its hard physical claims can be admitted.
    Ready,
    /// The item is current but must wait for hard physical capacity.
    Blocked,
    /// The item no longer names current owner state and can be discarded.
    Stale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FairQueueBand {
    Background,
    Prefetch,
    Required,
}

impl FairQueueBand {
    const ALL: [Self; 3] = [Self::Background, Self::Prefetch, Self::Required];
}

#[derive(Debug)]
struct Entry<T> {
    item: T,
    band: FairQueueBand,
    cost: u64,
    enqueued_at: u64,
    sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateKind {
    Eligible,
    Forced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    band: FairQueueBand,
    kind: CandidateKind,
    age: u64,
    sequence: u64,
}

/// Work-conserving policy bands that reorder only hard-feasible work.
///
/// Each band is FIFO. A selection pass rotates blocked entries at most once and
/// drops stale entries lazily, so normal queue operations never scan or shift the
/// complete cross-band workload.
#[derive(Debug)]
pub struct FairQueue<T> {
    config: FairQueueConfig,
    queues: [VecDeque<Entry<T>>; 3],
    deficit: [i64; 3],
    len: usize,
    next_sequence: u64,
}

impl<T> FairQueue<T> {
    pub fn new(config: FairQueueConfig) -> Result<Self, FairQueueError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            queues: std::array::from_fn(|_| VecDeque::new()),
            deficit: [0; 3],
            len: 0,
            next_sequence: 1,
        })
    }

    pub const fn config(&self) -> FairQueueConfig {
        self.config
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn deficit(&self, band: FairQueueBand) -> i64 {
        self.deficit[band_index(band)]
    }

    pub fn push(
        &mut self,
        item: T,
        band: FairQueueBand,
        cost: u64,
        now: u64,
    ) -> Result<(), FairQueueError> {
        if cost == 0 {
            return Err(FairQueueError::ZeroCost);
        }
        if cost > self.config.max_transition_cost {
            return Err(FairQueueError::TransitionTooLarge {
                cost,
                maximum: self.config.max_transition_cost,
            });
        }
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.queues[band_index(band)].push_back(Entry {
            item,
            band,
            cost,
            enqueued_at: now,
            sequence,
        });
        self.len = self.len.saturating_add(1);
        Ok(())
    }

    /// Removes matching owner work. This is intended for explicit cancellation;
    /// hot-path stale removal belongs in `Self::pop_next_by`.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        for queue in &mut self.queues {
            let before = queue.len();
            queue.retain(|entry| keep(&entry.item));
            self.len -= before - queue.len();
        }
    }

    /// Reclassifies matching queued work without resetting age or identity.
    pub fn reclassify_where(&mut self, band: FairQueueBand, mut matches: impl FnMut(&T) -> bool) {
        let mut moved = Vec::new();
        for queue in &mut self.queues {
            let count = queue.len();
            for _ in 0..count {
                let mut entry = queue
                    .pop_front()
                    .expect("band queue length is stable during reclassification");
                if matches(&entry.item) {
                    entry.band = band;
                    moved.push(entry);
                } else {
                    queue.push_back(entry);
                }
            }
        }
        let target = &mut self.queues[band_index(band)];
        target.extend(moved);
        target.make_contiguous().sort_by_key(|entry| entry.sequence);
    }

    /// Selects one transition. Hard physical feasibility is authoritative: aging
    /// and debt never bypass it. Finite execution work may enter bounded debt after
    /// aging; speculative prefetch has no forced-progress entitlement.
    pub fn pop_next(&mut self, now: u64, mut hard_feasible: impl FnMut(&T) -> bool) -> Option<T> {
        self.pop_next_by(now, |item| {
            if hard_feasible(item) {
                FairQueueEntryState::Ready
            } else {
                FairQueueEntryState::Blocked
            }
        })
    }

    pub(crate) fn pop_next_by(
        &mut self,
        now: u64,
        mut state: impl FnMut(&T) -> FairQueueEntryState,
    ) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        self.add_quantum();

        let mut selected = None;
        for band in FairQueueBand::ALL {
            let Some(candidate) = self.prepare_candidate(band, now, &mut state) else {
                continue;
            };
            if selected.is_none_or(|current| candidate_precedes(candidate, current)) {
                selected = Some(candidate);
            }
        }

        let selected = selected?;
        let entry = self.queues[band_index(selected.band)]
            .pop_front()
            .expect("prepared fair-queue candidate remains at its band head");
        self.len -= 1;
        let deficit = &mut self.deficit[band_index(entry.band)];
        *deficit = deficit.saturating_sub(entry.cost as i64);
        Some(entry.item)
    }

    fn prepare_candidate(
        &mut self,
        band: FairQueueBand,
        now: u64,
        state: &mut impl FnMut(&T) -> FairQueueEntryState,
    ) -> Option<Candidate> {
        let index = band_index(band);
        let count = self.queues[index].len();
        for _ in 0..count {
            let entry = self.queues[index]
                .pop_front()
                .expect("band queue length is stable during one selection pass");
            match state(&entry.item) {
                FairQueueEntryState::Stale => {
                    self.len -= 1;
                }
                FairQueueEntryState::Blocked => self.queues[index].push_back(entry),
                FairQueueEntryState::Ready => {
                    let deficit = self.deficit[index];
                    let cost = entry.cost as i64;
                    let age = now.saturating_sub(entry.enqueued_at);
                    let kind = if deficit >= cost {
                        Some(CandidateKind::Eligible)
                    } else if band != FairQueueBand::Background
                        && age >= self.config.starvation_ticks
                        && deficit.saturating_sub(cost) >= -(self.config.debt_limit as i64)
                    {
                        Some(CandidateKind::Forced)
                    } else {
                        None
                    };
                    let sequence = entry.sequence;
                    self.queues[index].push_back(entry);
                    if let Some(kind) = kind {
                        self.queues[index].rotate_right(1);
                        return Some(Candidate {
                            band,
                            kind,
                            age,
                            sequence,
                        });
                    }
                }
            }
        }
        None
    }

    fn add_quantum(&mut self) {
        let cap = self.config.max_surplus as i64;
        for band in FairQueueBand::ALL {
            let quantum = self.config.quantum(band) as i64;
            let deficit = &mut self.deficit[band_index(band)];
            *deficit = deficit.saturating_add(quantum).min(cap);
        }
    }
}

const fn band_index(band: FairQueueBand) -> usize {
    match band {
        FairQueueBand::Background => 0,
        FairQueueBand::Prefetch => 1,
        FairQueueBand::Required => 2,
    }
}

const fn band_priority(band: FairQueueBand) -> u8 {
    match band {
        FairQueueBand::Background => 0,
        FairQueueBand::Prefetch => 1,
        FairQueueBand::Required => 2,
    }
}

fn candidate_precedes(candidate: Candidate, current: Candidate) -> bool {
    let forced = u8::from(candidate.kind == CandidateKind::Forced);
    let current_forced = u8::from(current.kind == CandidateKind::Forced);
    (
        forced,
        candidate.age,
        band_priority(candidate.band),
        u64::MAX - candidate.sequence,
    ) > (
        current_forced,
        current.age,
        band_priority(current.band),
        u64::MAX - current.sequence,
    )
}
