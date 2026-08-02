//! Physical-resource admission for the runtime I/O reactor.
//!
//! Unlike token/KV admission, this broker accounts for physical resources that
//! remain occupied across scheduler ticks: storage queue entries, pinned
//! slabs, read/upload bytes, resident bytes, residency leases, continuations,
//! waiters, and ready cohorts. A submitted claim remains charged until the owner
//! observes the matching completion and explicitly returns provider custody.

use std::collections::HashMap;

use ahash::RandomState;
use snafu::Snafu;

/// Unit used by one resource capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceUnit {
    Bytes,
    Operations,
    Slots,
}

/// Fixed physical-credit domains owned by the runtime I/O reactor.
///
/// Unlike advisor estimates, these credits are correctness limits. A submitted
/// claim remains charged until the owner observes the matching completion and
/// explicitly returns provider custody.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceKind {
    ReadSlot,
    PinnedHostBytes,
    StorageReadBytes,
    UploadSlot,
    UploadBytes,
    InstallSlot,
    DeviceInstallBytes,
    ResidentBytes,
    ResidencyLease,
    Arena,
    KvPage,
    Continuation,
    Waiter,
    LoadOperation,
    ReadyCohort,
}

impl ResourceKind {
    pub const ALL: [Self; 15] = [
        Self::ReadSlot,
        Self::PinnedHostBytes,
        Self::StorageReadBytes,
        Self::UploadSlot,
        Self::UploadBytes,
        Self::InstallSlot,
        Self::DeviceInstallBytes,
        Self::ResidentBytes,
        Self::ResidencyLease,
        Self::Arena,
        Self::KvPage,
        Self::Continuation,
        Self::Waiter,
        Self::LoadOperation,
        Self::ReadyCohort,
    ];

    pub const fn unit(self) -> ResourceUnit {
        match self {
            Self::PinnedHostBytes
            | Self::StorageReadBytes
            | Self::UploadBytes
            | Self::DeviceInstallBytes
            | Self::ResidentBytes => ResourceUnit::Bytes,
            Self::ReadSlot | Self::LoadOperation => ResourceUnit::Operations,
            Self::UploadSlot
            | Self::InstallSlot
            | Self::ResidencyLease
            | Self::Arena
            | Self::KvPage
            | Self::Continuation
            | Self::Waiter
            | Self::ReadyCohort => ResourceUnit::Slots,
        }
    }
}

/// One amount in the fixed runtime physical-credit catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalResourceClaim {
    pub kind: ResourceKind,
    pub amount: u64,
}

impl PhysicalResourceClaim {
    pub const fn new(kind: ResourceKind, amount: u64) -> Self {
        Self { kind, amount }
    }
}

/// Capacity and admitted-execution reserve for one physical-credit domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalResourceLimit {
    pub kind: ResourceKind,
    pub capacity: u64,
    pub execution_reserve: u64,
}

impl PhysicalResourceLimit {
    pub const fn new(kind: ResourceKind, capacity: u64, execution_reserve: u64) -> Self {
        Self {
            kind,
            capacity,
            execution_reserve,
        }
    }
}

/// Priority class for a physical resource request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceClass {
    /// Speculative work that may be skipped or cancelled without affecting correctness.
    Prefetch,
    /// Admitted batch work optimized for aggregate progress.
    Throughput,
    /// Admitted work on a latency-sensitive committed frontier.
    LatencyCritical,
}

impl ResourceClass {
    const fn may_use_execution_reserve(self) -> bool {
        !matches!(self, Self::Prefetch)
    }
}

/// Stable identity of one non-cloneable physical-credit ownership token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalResourceGrantId(u64);

impl PhysicalResourceGrantId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Rejection or ownership violation in the physical-credit ledger.
#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum PhysicalResourceError {
    #[snafu(display("duplicate physical resource limit for {kind:?}"))]
    DuplicateLimit { kind: ResourceKind },
    #[snafu(display("missing physical resource limit for {kind:?}"))]
    MissingLimit { kind: ResourceKind },
    #[snafu(display("physical resource reserve {reserve} exceeds {kind:?} capacity {capacity}"))]
    ReserveExceedsCapacity {
        kind: ResourceKind,
        reserve: u64,
        capacity: u64,
    },
    #[snafu(display("physical resource claim overflow for {kind:?}"))]
    ClaimOverflow { kind: ResourceKind },
    #[snafu(display("physical resource request {requested} exceeds {kind:?} capacity {capacity}"))]
    ExceedsCapacity {
        kind: ResourceKind,
        requested: u64,
        capacity: u64,
    },
    #[snafu(display(
        "physical resource {kind:?} is temporarily unavailable: requested {requested}, available {available}"
    ))]
    TemporarilyUnavailable {
        kind: ResourceKind,
        requested: u64,
        available: u64,
    },
    #[snafu(display(
        "physical resource request {requested} would consume the {kind:?} execution reserve; base availability is {base_available}"
    ))]
    ExecutionReserve {
        kind: ResourceKind,
        requested: u64,
        base_available: u64,
    },
    #[snafu(display("unknown physical resource grant {grant:?}"))]
    UnknownGrant { grant: PhysicalResourceGrantId },
    #[snafu(display("physical resource grant owner mismatch for {grant:?}"))]
    GrantOwnerMismatch { grant: PhysicalResourceGrantId },
    #[snafu(display("physical resource grant {grant:?} has no {kind:?} claim"))]
    MissingClaim {
        grant: PhysicalResourceGrantId,
        kind: ResourceKind,
    },
    #[snafu(display("physical resource claim {kind:?} in grant {grant:?} was already submitted"))]
    AlreadySubmitted {
        grant: PhysicalResourceGrantId,
        kind: ResourceKind,
    },
    #[snafu(display("physical resource claim {kind:?} in grant {grant:?} was not submitted"))]
    NotSubmitted {
        grant: PhysicalResourceGrantId,
        kind: ResourceKind,
    },
    #[snafu(display(
        "submitted physical resource claim {kind:?} in grant {grant:?} cannot be revoked"
    ))]
    SubmittedClaimCannotBeRevoked {
        grant: PhysicalResourceGrantId,
        kind: ResourceKind,
    },
    #[snafu(display("physical resource grant identity space is exhausted"))]
    GrantIdExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HardClaimState {
    amount: u64,
    submitted: bool,
}

#[derive(Debug, Clone)]
struct HardLimitState {
    capacity: u64,
    execution_reserve: u64,
    in_use: u64,
    high_water: u64,
}

#[derive(Debug, Clone)]
struct HardGrantRecord {
    owner: u64,
    class: ResourceClass,
    claims: HashMap<ResourceKind, HardClaimState, RandomState>,
}

/// Non-cloneable RAII ownership token for a vector of hard credits.
///
/// The token deliberately cannot mutate the ledger from `Drop`: the registry is
/// the sole writer and must explicitly release held claims or return submitted
/// claims after a matching completion.
#[derive(Debug)]
#[must_use = "physical resource grants must be explicitly released by their owner"]
pub struct PhysicalResourceGrant {
    id: Option<PhysicalResourceGrantId>,
    owner: u64,
    claims: HashMap<ResourceKind, HardClaimState, RandomState>,
}

impl PhysicalResourceGrant {
    pub const fn id(&self) -> Option<PhysicalResourceGrantId> {
        self.id
    }

    pub const fn owner(&self) -> u64 {
        self.owner
    }

    pub fn is_released(&self) -> bool {
        self.id.is_none()
    }

    pub fn contains(&self, kind: ResourceKind) -> bool {
        self.claims.contains_key(&kind)
    }

    pub fn is_submitted(&self, kind: ResourceKind) -> bool {
        self.claims.get(&kind).is_some_and(|claim| claim.submitted)
    }

    pub fn claims(&self) -> impl ExactSizeIterator<Item = PhysicalResourceClaim> + '_ {
        self.claims
            .iter()
            .map(|(kind, state)| PhysicalResourceClaim {
                kind: *kind,
                amount: state.amount,
            })
    }
}

/// Snapshot of one fixed physical-credit domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalResourceSnapshot {
    pub kind: ResourceKind,
    pub unit: ResourceUnit,
    pub capacity: u64,
    pub execution_reserve: u64,
    pub in_use: u64,
    pub high_water: u64,
}

/// Owner-written hard-capacity ledger used by the runtime I/O registry.
#[derive(Debug)]
pub struct PhysicalResourceBroker {
    limits: HashMap<ResourceKind, HardLimitState, RandomState>,
    grants: HashMap<PhysicalResourceGrantId, HardGrantRecord, RandomState>,
    next_grant: u64,
}

impl PhysicalResourceBroker {
    pub fn new(
        limits: impl IntoIterator<Item = PhysicalResourceLimit>,
    ) -> std::result::Result<Self, PhysicalResourceError> {
        let mut states = HashMap::default();
        for limit in limits {
            if limit.execution_reserve > limit.capacity {
                return Err(PhysicalResourceError::ReserveExceedsCapacity {
                    kind: limit.kind,
                    reserve: limit.execution_reserve,
                    capacity: limit.capacity,
                });
            }
            if states
                .insert(
                    limit.kind,
                    HardLimitState {
                        capacity: limit.capacity,
                        execution_reserve: limit.execution_reserve,
                        in_use: 0,
                        high_water: 0,
                    },
                )
                .is_some()
            {
                return Err(PhysicalResourceError::DuplicateLimit { kind: limit.kind });
            }
        }
        for kind in ResourceKind::ALL {
            if !states.contains_key(&kind) {
                return Err(PhysicalResourceError::MissingLimit { kind });
            }
        }
        Ok(Self {
            limits: states,
            grants: HashMap::default(),
            next_grant: 1,
        })
    }

    /// A generous finite catalog used only by deterministic CPU tests.
    #[cfg(test)]
    pub fn testing_default() -> Self {
        Self::new(ResourceKind::ALL.map(|kind| {
            let capacity = match kind {
                ResourceKind::PinnedHostBytes
                | ResourceKind::StorageReadBytes
                | ResourceKind::UploadBytes
                | ResourceKind::DeviceInstallBytes
                | ResourceKind::ResidentBytes => 1 << 40,
                _ => 1 << 20,
            };
            PhysicalResourceLimit::new(kind, capacity, 0)
        }))
        .expect("the built-in physical resource catalog is valid")
    }

    /// Replace one runtime-owned limit before that resource has live ownership.
    /// This is used when an exact hardware/runtime topology (for example the KV
    /// page manager) becomes available after the base broker is constructed.
    pub fn reconfigure_limit(
        &mut self,
        kind: ResourceKind,
        capacity: u64,
        execution_reserve: u64,
    ) -> std::result::Result<(), PhysicalResourceError> {
        if execution_reserve > capacity {
            return Err(PhysicalResourceError::ReserveExceedsCapacity {
                kind,
                reserve: execution_reserve,
                capacity,
            });
        }
        let state = self
            .limits
            .get_mut(&kind)
            .ok_or(PhysicalResourceError::MissingLimit { kind })?;
        if state.in_use > capacity {
            return Err(PhysicalResourceError::ExceedsCapacity {
                kind,
                requested: state.in_use,
                capacity,
            });
        }
        state.capacity = capacity;
        state.execution_reserve = execution_reserve;
        Ok(())
    }

    pub fn snapshots(&self) -> impl ExactSizeIterator<Item = PhysicalResourceSnapshot> + '_ {
        self.limits
            .iter()
            .map(|(kind, state)| PhysicalResourceSnapshot {
                kind: *kind,
                unit: kind.unit(),
                capacity: state.capacity,
                execution_reserve: state.execution_reserve,
                in_use: state.in_use,
                high_water: state.high_water,
            })
    }

    pub fn in_use(&self, kind: ResourceKind) -> u64 {
        self.limits.get(&kind).map_or(0, |state| state.in_use)
    }

    pub fn active_grants(&self) -> usize {
        self.grants.len()
    }

    pub fn can_acquire(
        &self,
        class: ResourceClass,
        claims: impl IntoIterator<Item = PhysicalResourceClaim>,
    ) -> std::result::Result<(), PhysicalResourceError> {
        let claims = Self::canonical_claims(claims)?;
        self.validate_available(class, &claims)
    }

    pub fn acquire(
        &mut self,
        owner: u64,
        class: ResourceClass,
        claims: impl IntoIterator<Item = PhysicalResourceClaim>,
    ) -> std::result::Result<PhysicalResourceGrant, PhysicalResourceError> {
        let claims = Self::canonical_claims(claims)?;
        self.validate_available(class, &claims)?;
        let id = PhysicalResourceGrantId(self.next_grant);
        self.next_grant = self
            .next_grant
            .checked_add(1)
            .ok_or(PhysicalResourceError::GrantIdExhausted)?;
        let claims: HashMap<_, _, RandomState> = claims
            .into_iter()
            .map(|claim| {
                (
                    claim.kind,
                    HardClaimState {
                        amount: claim.amount,
                        submitted: false,
                    },
                )
            })
            .collect();
        for (kind, claim) in &claims {
            let state = self
                .limits
                .get_mut(kind)
                .expect("claims were validated against the complete catalog");
            state.in_use += claim.amount;
            state.high_water = state.high_water.max(state.in_use);
        }
        self.grants.insert(
            id,
            HardGrantRecord {
                owner,
                class,
                claims: claims.clone(),
            },
        );
        Ok(PhysicalResourceGrant {
            id: Some(id),
            owner,
            claims,
        })
    }

    pub fn promote(
        &mut self,
        grant: &PhysicalResourceGrant,
        class: ResourceClass,
    ) -> std::result::Result<(), PhysicalResourceError> {
        let (id, record) = self.record_for(grant)?;
        if class > record.class {
            self.grants
                .get_mut(&id)
                .expect("validated grant remains present")
                .class = class;
        }
        Ok(())
    }

    /// Transfers the named held claims into provider custody. A submitted claim
    /// cannot be revoked or dropped until `mark_returned` observes its completion.
    pub fn mark_submitted(
        &mut self,
        grant: &mut PhysicalResourceGrant,
        kinds: &[ResourceKind],
    ) -> std::result::Result<(), PhysicalResourceError> {
        let (id, _) = self.record_for(grant)?;
        for kind in kinds {
            let claim = grant
                .claims
                .get(kind)
                .ok_or(PhysicalResourceError::MissingClaim {
                    grant: id,
                    kind: *kind,
                })?;
            if claim.submitted {
                return Err(PhysicalResourceError::AlreadySubmitted {
                    grant: id,
                    kind: *kind,
                });
            }
        }
        for kind in kinds {
            grant
                .claims
                .get_mut(kind)
                .expect("claim presence was validated")
                .submitted = true;
            self.grants
                .get_mut(&id)
                .expect("validated grant remains present")
                .claims
                .get_mut(kind)
                .expect("token and ledger claims remain identical")
                .submitted = true;
        }
        Ok(())
    }

    /// Returns provider custody after the exact completion was validated. Credits
    /// remain owned until a separate explicit release or ownership transfer.
    pub fn mark_returned(
        &mut self,
        grant: &mut PhysicalResourceGrant,
        kinds: &[ResourceKind],
    ) -> std::result::Result<(), PhysicalResourceError> {
        let (id, _) = self.record_for(grant)?;
        for kind in kinds {
            let claim = grant
                .claims
                .get(kind)
                .ok_or(PhysicalResourceError::MissingClaim {
                    grant: id,
                    kind: *kind,
                })?;
            if !claim.submitted {
                return Err(PhysicalResourceError::NotSubmitted {
                    grant: id,
                    kind: *kind,
                });
            }
        }
        for kind in kinds {
            grant
                .claims
                .get_mut(kind)
                .expect("claim presence was validated")
                .submitted = false;
            self.grants
                .get_mut(&id)
                .expect("validated grant remains present")
                .claims
                .get_mut(kind)
                .expect("token and ledger claims remain identical")
                .submitted = false;
        }
        Ok(())
    }

    pub fn can_release_held(
        &self,
        grant: &PhysicalResourceGrant,
        kinds: &[ResourceKind],
    ) -> std::result::Result<(), PhysicalResourceError> {
        let (id, _) = self.record_for(grant)?;
        for kind in kinds {
            let claim = grant
                .claims
                .get(kind)
                .ok_or(PhysicalResourceError::MissingClaim {
                    grant: id,
                    kind: *kind,
                })?;
            if claim.submitted {
                return Err(PhysicalResourceError::SubmittedClaimCannotBeRevoked {
                    grant: id,
                    kind: *kind,
                });
            }
        }
        Ok(())
    }

    pub fn release_held(
        &mut self,
        grant: &mut PhysicalResourceGrant,
        kinds: &[ResourceKind],
    ) -> std::result::Result<(), PhysicalResourceError> {
        self.can_release_held(grant, kinds)?;
        let id = grant
            .id
            .expect("validated resource grant retains its identity");
        for kind in kinds {
            let claim = grant
                .claims
                .remove(kind)
                .expect("claim presence was validated");
            let state = self
                .limits
                .get_mut(kind)
                .expect("active claim references a configured limit");
            state.in_use -= claim.amount;
            self.grants
                .get_mut(&id)
                .expect("validated grant remains present")
                .claims
                .remove(kind);
        }
        if grant.claims.is_empty() {
            self.grants.remove(&id);
            grant.id = None;
        }
        Ok(())
    }

    pub fn can_release_all_held(
        &self,
        grant: &PhysicalResourceGrant,
    ) -> std::result::Result<(), PhysicalResourceError> {
        let kinds: Vec<_> = grant.claims.keys().copied().collect();
        self.can_release_held(grant, &kinds)
    }

    pub fn release_all_held(
        &mut self,
        grant: &mut PhysicalResourceGrant,
    ) -> std::result::Result<(), PhysicalResourceError> {
        let kinds: Vec<_> = grant.claims.keys().copied().collect();
        self.release_held(grant, &kinds)
    }

    fn canonical_claims(
        claims: impl IntoIterator<Item = PhysicalResourceClaim>,
    ) -> std::result::Result<Vec<PhysicalResourceClaim>, PhysicalResourceError> {
        let mut merged = HashMap::<ResourceKind, u64, RandomState>::default();
        for claim in claims {
            if claim.amount == 0 {
                continue;
            }
            let amount = merged.entry(claim.kind).or_default();
            *amount = amount
                .checked_add(claim.amount)
                .ok_or(PhysicalResourceError::ClaimOverflow { kind: claim.kind })?;
        }
        Ok(merged
            .into_iter()
            .map(|(kind, amount)| PhysicalResourceClaim { kind, amount })
            .collect())
    }

    fn validate_available(
        &self,
        class: ResourceClass,
        claims: &[PhysicalResourceClaim],
    ) -> std::result::Result<(), PhysicalResourceError> {
        for claim in claims {
            let state = self
                .limits
                .get(&claim.kind)
                .ok_or(PhysicalResourceError::MissingLimit { kind: claim.kind })?;
            if claim.amount > state.capacity {
                return Err(PhysicalResourceError::ExceedsCapacity {
                    kind: claim.kind,
                    requested: claim.amount,
                    capacity: state.capacity,
                });
            }
            let available = state.capacity.saturating_sub(state.in_use);
            if claim.amount > available {
                return Err(PhysicalResourceError::TemporarilyUnavailable {
                    kind: claim.kind,
                    requested: claim.amount,
                    available,
                });
            }
            if !class.may_use_execution_reserve() {
                let base_limit = state.capacity.saturating_sub(state.execution_reserve);
                let base_available = base_limit.saturating_sub(state.in_use);
                if claim.amount > base_available {
                    return Err(PhysicalResourceError::ExecutionReserve {
                        kind: claim.kind,
                        requested: claim.amount,
                        base_available,
                    });
                }
            }
        }
        Ok(())
    }

    fn record_for(
        &self,
        grant: &PhysicalResourceGrant,
    ) -> std::result::Result<(PhysicalResourceGrantId, &HardGrantRecord), PhysicalResourceError>
    {
        let id = grant.id.ok_or(PhysicalResourceError::GrantOwnerMismatch {
            grant: PhysicalResourceGrantId(0),
        })?;
        let record = self
            .grants
            .get(&id)
            .ok_or(PhysicalResourceError::UnknownGrant { grant: id })?;
        if record.owner != grant.owner || record.claims != grant.claims {
            return Err(PhysicalResourceError::GrantOwnerMismatch { grant: id });
        }
        Ok((id, record))
    }
}
