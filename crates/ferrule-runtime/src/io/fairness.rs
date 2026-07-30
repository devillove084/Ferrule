//! Bounded aging and deficit-round-robin for runnable I/O transitions.

use std::collections::HashMap;

use ahash::RandomState;

use crate::scheduling::ResourceClass;

/// Deterministic fairness configuration. Costs and quanta use the same units
/// (normally marginal physical bytes, with metadata-only transitions costing one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FairQueueConfig {
    pub prefetch_quantum: u64,
    pub prefill_quantum: u64,
    pub verification_quantum: u64,
    pub decode_quantum: u64,
    pub max_surplus: u64,
    pub debt_limit: u64,
    pub starvation_ticks: u64,
    pub max_transition_cost: u64,
}

impl Default for FairQueueConfig {
    fn default() -> Self {
        Self {
            prefetch_quantum: 1,
            prefill_quantum: 4 * 1024,
            verification_quantum: 8 * 1024,
            decode_quantum: 8 * 1024,
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
    /// `limits` must already have passed `ExpertIoResourceLimits::validate`. Total
    /// byte capacities are used directly instead of inferring a per-request size
    /// from slot counts. Verification and decode receive the largest quantum and
    /// are also the only classes allowed to consume the hard demand reserve.
    pub fn for_production(
        limits: ferrule_common::expert_io::ExpertIoResourceLimits,
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

        // One-eighth of a maximum transition gives even speculative prefetch
        // bounded deficit progress without allowing it to consume hard reserve.
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
            prefetch_quantum: progress_quantum,
            prefill_quantum: scaled_quantum(4),
            verification_quantum: scaled_quantum(8),
            decode_quantum: scaled_quantum(8),
            max_surplus: max_transition_cost.min(MAX_COUNTER),
            debt_limit: max_transition_cost.min(MAX_COUNTER),
            starvation_ticks: STARVATION_TICKS,
            max_transition_cost,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, FairQueueError> {
        if self.prefetch_quantum == 0
            || self.prefill_quantum == 0
            || self.verification_quantum == 0
            || self.decode_quantum == 0
        {
            return Err(FairQueueError::ZeroQuantum);
        }
        if self.max_transition_cost == 0 {
            return Err(FairQueueError::ZeroMaxTransitionCost);
        }
        if self.prefetch_quantum > i64::MAX as u64
            || self.prefill_quantum > i64::MAX as u64
            || self.verification_quantum > i64::MAX as u64
            || self.decode_quantum > i64::MAX as u64
            || self.max_transition_cost > i64::MAX as u64
            || self.max_surplus > i64::MAX as u64
            || self.debt_limit > i64::MAX as u64
        {
            return Err(FairQueueError::CostOutOfRange);
        }
        Ok(self)
    }

    pub const fn quantum(self, class: ResourceClass) -> u64 {
        match class {
            ResourceClass::Prefetch => self.prefetch_quantum,
            ResourceClass::Prefill => self.prefill_quantum,
            ResourceClass::Verification => self.verification_quantum,
            ResourceClass::Decode => self.decode_quantum,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FairQueueError {
    ZeroQuantum,
    ZeroMaxTransitionCost,
    CostOutOfRange,
    ZeroCost,
    TransitionTooLarge { cost: u64, maximum: u64 },
}

impl std::fmt::Display for FairQueueError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "fair queue configuration/entry error: {self:?}")
    }
}

impl std::error::Error for FairQueueError {}

#[derive(Debug)]
struct Entry<T> {
    item: T,
    class: ResourceClass,
    cost: u64,
    enqueued_at: u64,
    sequence: u64,
}

/// Work-conserving queue whose policy can reorder only hard-feasible work.
#[derive(Debug)]
pub struct FairQueue<T> {
    config: FairQueueConfig,
    entries: Vec<Entry<T>>,
    deficit: HashMap<ResourceClass, i64, RandomState>,
    next_sequence: u64,
}

impl<T> FairQueue<T> {
    pub fn new(config: FairQueueConfig) -> Result<Self, FairQueueError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            entries: Vec::new(),
            deficit: {
                let mut m = HashMap::with_hasher(RandomState::default());
                m.insert(ResourceClass::Prefetch, 0i64);
                m.insert(ResourceClass::Prefill, 0);
                m.insert(ResourceClass::Verification, 0);
                m.insert(ResourceClass::Decode, 0);
                m
            },
            next_sequence: 1,
        })
    }

    pub const fn config(&self) -> FairQueueConfig {
        self.config
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn deficit(&self, class: ResourceClass) -> i64 {
        self.deficit.get(&class).copied().unwrap_or_default()
    }

    pub fn push(
        &mut self,
        item: T,
        class: ResourceClass,
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
        self.entries.push(Entry {
            item,
            class,
            cost,
            enqueued_at: now,
            sequence,
        });
        Ok(())
    }

    /// Removes stale owner actions without treating hard infeasibility as staleness.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        self.entries.retain(|entry| keep(&entry.item));
    }

    /// Selects one transition. `hard_feasible` is authoritative: aging and debt
    /// never bypass it. Finite exact work may enter bounded debt after aging;
    /// speculative prefetch has no forced-progress entitlement.
    pub fn pop_next(&mut self, now: u64, mut hard_feasible: impl FnMut(&T) -> bool) -> Option<T> {
        if self.entries.is_empty() {
            return None;
        }
        self.add_quantum();

        let mut eligible = Vec::new();
        let mut forced = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            if !hard_feasible(&entry.item) {
                continue;
            }
            let deficit = self.deficit(entry.class);
            let cost = entry.cost as i64;
            if deficit >= cost {
                eligible.push(index);
                continue;
            }
            let age = now.saturating_sub(entry.enqueued_at);
            if entry.class != ResourceClass::Prefetch
                && age >= self.config.starvation_ticks
                && deficit.saturating_sub(cost) >= -(self.config.debt_limit as i64)
            {
                forced.push(index);
            }
        }

        let selected = if forced.is_empty() {
            self.best_candidate(&eligible, now)
        } else {
            self.best_candidate(&forced, now)
        }?;
        let entry = self.entries.remove(selected);
        let deficit = self
            .deficit
            .get_mut(&entry.class)
            .expect("all resource classes have a deficit account");
        *deficit = deficit.saturating_sub(entry.cost as i64);
        Some(entry.item)
    }

    fn add_quantum(&mut self) {
        let cap = self.config.max_surplus as i64;
        for class in [
            ResourceClass::Prefetch,
            ResourceClass::Prefill,
            ResourceClass::Verification,
            ResourceClass::Decode,
        ] {
            let quantum = self.config.quantum(class) as i64;
            let deficit = self
                .deficit
                .get_mut(&class)
                .expect("all resource classes have a deficit account");
            *deficit = deficit.saturating_add(quantum).min(cap);
        }
    }

    fn best_candidate(&self, candidates: &[usize], now: u64) -> Option<usize> {
        candidates.iter().copied().max_by_key(|index| {
            let entry = &self.entries[*index];
            let age = now.saturating_sub(entry.enqueued_at);
            (age, class_priority(entry.class), u64::MAX - entry.sequence)
        })
    }
}

const fn class_priority(class: ResourceClass) -> u8 {
    match class {
        ResourceClass::Prefetch => 0,
        ResourceClass::Prefill => 1,
        ResourceClass::Verification => 2,
        ResourceClass::Decode => 3,
    }
}
