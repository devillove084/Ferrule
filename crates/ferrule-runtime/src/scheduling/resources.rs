//! Hardware-resource admission for storage-aware execution scheduling.
//!
//! Unlike token/KV admission, this broker accounts for physical resources that
//! remain occupied across scheduler ticks: storage queue entries, host buffers,
//! transfer lanes, device frames, scratch arenas, or any backend-defined domain.
//! Model code addresses resources through IDs registered by the hardware adapter;
//! the broker does not know model families or device vendors.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use ferrule_common::expert_io::{
    ExpertIoResourceAdmission, ExpertIoResourceClass, ExpertIoResourceControl,
    ExpertIoResourceDemand, ExpertIoResourceGrant, ExpertIoResourceLimits, ExpertIoResourcePermit,
    ExpertIoResourceStage,
};
use ferrule_common::{CompletionHub, Error, Result};

/// Stable resource identity within one broker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(u16);

impl ResourceId {
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Unit used by one resource capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceUnit {
    Bytes,
    Operations,
    Slots,
}

/// Fixed hard-credit domains owned by the runtime I/O reactor.
///
/// Unlike advisor estimates, these credits are correctness limits. A submitted
/// claim remains charged until the owner observes the matching completion and
/// explicitly returns provider custody.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceKind {
    Sqe,
    PinnedSlab,
    ReadBytes,
    UploadSlot,
    UploadBytes,
    ExpertFrame,
    Lease,
    Arena,
    KvPage,
    Continuation,
    Waiter,
    LoadOperation,
    ReadyCohort,
}

impl ResourceKind {
    pub const ALL: [Self; 13] = [
        Self::Sqe,
        Self::PinnedSlab,
        Self::ReadBytes,
        Self::UploadSlot,
        Self::UploadBytes,
        Self::ExpertFrame,
        Self::Lease,
        Self::Arena,
        Self::KvPage,
        Self::Continuation,
        Self::Waiter,
        Self::LoadOperation,
        Self::ReadyCohort,
    ];

    pub const fn unit(self) -> ResourceUnit {
        match self {
            Self::PinnedSlab | Self::ReadBytes | Self::UploadBytes | Self::ExpertFrame => {
                ResourceUnit::Bytes
            }
            Self::Sqe | Self::LoadOperation => ResourceUnit::Operations,
            Self::UploadSlot
            | Self::Lease
            | Self::Arena
            | Self::KvPage
            | Self::Continuation
            | Self::Waiter
            | Self::ReadyCohort => ResourceUnit::Slots,
        }
    }
}

/// One amount in the fixed runtime hard-credit catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HardResourceClaim {
    pub kind: ResourceKind,
    pub amount: u64,
}

impl HardResourceClaim {
    pub const fn new(kind: ResourceKind, amount: u64) -> Self {
        Self { kind, amount }
    }
}

/// Capacity and decode/verification reserve for one hard-credit domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardResourceLimit {
    pub kind: ResourceKind,
    pub capacity: u64,
    pub demand_reserve: u64,
}

impl HardResourceLimit {
    pub const fn new(kind: ResourceKind, capacity: u64, demand_reserve: u64) -> Self {
        Self {
            kind,
            capacity,
            demand_reserve,
        }
    }
}

/// Priority class for a physical resource request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceClass {
    /// Speculative work that may be skipped or cancelled without affecting correctness.
    Prefetch,
    /// Prompt processing required by an admitted request.
    Prefill,
    /// Exact target verification required to complete a speculative transaction.
    Verification,
    /// Latency-sensitive decode work on the committed frontier.
    Decode,
}

impl ResourceClass {
    const fn is_demand(self) -> bool {
        !matches!(self, Self::Prefetch)
    }

    const fn may_use_hard_reserve(self) -> bool {
        matches!(self, Self::Verification | Self::Decode)
    }
}

/// Stable identity of one non-cloneable hard-credit ownership token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HardResourceGrantId(u64);

impl HardResourceGrantId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Rejection or ownership violation in the hard-credit ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HardResourceError {
    DuplicateLimit(ResourceKind),
    MissingLimit(ResourceKind),
    ReserveExceedsCapacity {
        kind: ResourceKind,
        reserve: u64,
        capacity: u64,
    },
    ClaimOverflow(ResourceKind),
    ExceedsCapacity {
        kind: ResourceKind,
        requested: u64,
        capacity: u64,
    },
    TemporarilyUnavailable {
        kind: ResourceKind,
        requested: u64,
        available: u64,
    },
    DemandReserve {
        kind: ResourceKind,
        requested: u64,
        base_available: u64,
    },
    UnknownGrant(HardResourceGrantId),
    GrantOwnerMismatch(HardResourceGrantId),
    MissingClaim {
        grant: HardResourceGrantId,
        kind: ResourceKind,
    },
    AlreadySubmitted {
        grant: HardResourceGrantId,
        kind: ResourceKind,
    },
    NotSubmitted {
        grant: HardResourceGrantId,
        kind: ResourceKind,
    },
    SubmittedClaimCannotBeRevoked {
        grant: HardResourceGrantId,
        kind: ResourceKind,
    },
    GrantIdExhausted,
}

impl std::fmt::Display for HardResourceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "hard-resource violation: {self:?}")
    }
}

impl std::error::Error for HardResourceError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HardClaimState {
    amount: u64,
    submitted: bool,
}

#[derive(Debug, Clone)]
struct HardLimitState {
    capacity: u64,
    demand_reserve: u64,
    in_use: u64,
    high_water: u64,
}

#[derive(Debug, Clone)]
struct HardGrantRecord {
    owner: u64,
    class: ResourceClass,
    claims: BTreeMap<ResourceKind, HardClaimState>,
}

/// Non-cloneable RAII ownership token for a vector of hard credits.
///
/// The token deliberately cannot mutate the ledger from `Drop`: the registry is
/// the sole writer and must explicitly release held claims or return submitted
/// claims after a matching completion.
#[derive(Debug)]
#[must_use = "hard resource grants must be explicitly released by their owner"]
pub struct HardResourceGrant {
    id: Option<HardResourceGrantId>,
    owner: u64,
    claims: BTreeMap<ResourceKind, HardClaimState>,
}

impl HardResourceGrant {
    pub const fn id(&self) -> Option<HardResourceGrantId> {
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

    pub fn claims(&self) -> impl ExactSizeIterator<Item = HardResourceClaim> + '_ {
        self.claims.iter().map(|(kind, state)| HardResourceClaim {
            kind: *kind,
            amount: state.amount,
        })
    }
}

/// Snapshot of one fixed hard-credit domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardResourceSnapshot {
    pub kind: ResourceKind,
    pub unit: ResourceUnit,
    pub capacity: u64,
    pub demand_reserve: u64,
    pub in_use: u64,
    pub high_water: u64,
}

/// Owner-written hard-capacity ledger used by the runtime I/O registry.
#[derive(Debug)]
pub struct HardResourceBroker {
    limits: BTreeMap<ResourceKind, HardLimitState>,
    grants: BTreeMap<HardResourceGrantId, HardGrantRecord>,
    next_grant: u64,
}

impl HardResourceBroker {
    pub fn new(
        limits: impl IntoIterator<Item = HardResourceLimit>,
    ) -> std::result::Result<Self, HardResourceError> {
        let mut states = BTreeMap::new();
        for limit in limits {
            if limit.demand_reserve > limit.capacity {
                return Err(HardResourceError::ReserveExceedsCapacity {
                    kind: limit.kind,
                    reserve: limit.demand_reserve,
                    capacity: limit.capacity,
                });
            }
            if states
                .insert(
                    limit.kind,
                    HardLimitState {
                        capacity: limit.capacity,
                        demand_reserve: limit.demand_reserve,
                        in_use: 0,
                        high_water: 0,
                    },
                )
                .is_some()
            {
                return Err(HardResourceError::DuplicateLimit(limit.kind));
            }
        }
        for kind in ResourceKind::ALL {
            if !states.contains_key(&kind) {
                return Err(HardResourceError::MissingLimit(kind));
            }
        }
        Ok(Self {
            limits: states,
            grants: BTreeMap::new(),
            next_grant: 1,
        })
    }

    /// A generous finite catalog useful for CPU tests and embedders that have not
    /// yet supplied a hardware profile.
    pub fn testing_default() -> Self {
        Self::new(ResourceKind::ALL.map(|kind| {
            let capacity = match kind {
                ResourceKind::PinnedSlab
                | ResourceKind::ReadBytes
                | ResourceKind::UploadBytes
                | ResourceKind::ExpertFrame => 1 << 40,
                _ => 1 << 20,
            };
            HardResourceLimit::new(kind, capacity, 0)
        }))
        .expect("the built-in hard resource catalog is valid")
    }

    /// Replace one runtime-owned limit before that resource has live ownership.
    /// This is used when an exact hardware/runtime topology (for example the KV
    /// page manager) becomes available after the base broker is constructed.
    pub fn reconfigure_limit(
        &mut self,
        kind: ResourceKind,
        capacity: u64,
        demand_reserve: u64,
    ) -> std::result::Result<(), HardResourceError> {
        if demand_reserve > capacity {
            return Err(HardResourceError::ReserveExceedsCapacity {
                kind,
                reserve: demand_reserve,
                capacity,
            });
        }
        let state = self
            .limits
            .get_mut(&kind)
            .ok_or(HardResourceError::MissingLimit(kind))?;
        if state.in_use > capacity {
            return Err(HardResourceError::ExceedsCapacity {
                kind,
                requested: state.in_use,
                capacity,
            });
        }
        state.capacity = capacity;
        state.demand_reserve = demand_reserve;
        Ok(())
    }

    pub fn snapshots(&self) -> impl ExactSizeIterator<Item = HardResourceSnapshot> + '_ {
        self.limits
            .iter()
            .map(|(kind, state)| HardResourceSnapshot {
                kind: *kind,
                unit: kind.unit(),
                capacity: state.capacity,
                demand_reserve: state.demand_reserve,
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
        claims: impl IntoIterator<Item = HardResourceClaim>,
    ) -> std::result::Result<(), HardResourceError> {
        let claims = Self::canonical_claims(claims)?;
        self.validate_available(class, &claims)
    }

    pub fn acquire(
        &mut self,
        owner: u64,
        class: ResourceClass,
        claims: impl IntoIterator<Item = HardResourceClaim>,
    ) -> std::result::Result<HardResourceGrant, HardResourceError> {
        let claims = Self::canonical_claims(claims)?;
        self.validate_available(class, &claims)?;
        let id = HardResourceGrantId(self.next_grant);
        self.next_grant = self
            .next_grant
            .checked_add(1)
            .ok_or(HardResourceError::GrantIdExhausted)?;
        let claims: BTreeMap<_, _> = claims
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
        Ok(HardResourceGrant {
            id: Some(id),
            owner,
            claims,
        })
    }

    pub fn promote(
        &mut self,
        grant: &HardResourceGrant,
        class: ResourceClass,
    ) -> std::result::Result<(), HardResourceError> {
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
        grant: &mut HardResourceGrant,
        kinds: &[ResourceKind],
    ) -> std::result::Result<(), HardResourceError> {
        let (id, _) = self.record_for(grant)?;
        for kind in kinds {
            let claim = grant
                .claims
                .get(kind)
                .ok_or(HardResourceError::MissingClaim {
                    grant: id,
                    kind: *kind,
                })?;
            if claim.submitted {
                return Err(HardResourceError::AlreadySubmitted {
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
        grant: &mut HardResourceGrant,
        kinds: &[ResourceKind],
    ) -> std::result::Result<(), HardResourceError> {
        let (id, _) = self.record_for(grant)?;
        for kind in kinds {
            let claim = grant
                .claims
                .get(kind)
                .ok_or(HardResourceError::MissingClaim {
                    grant: id,
                    kind: *kind,
                })?;
            if !claim.submitted {
                return Err(HardResourceError::NotSubmitted {
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

    pub fn release_held(
        &mut self,
        grant: &mut HardResourceGrant,
        kinds: &[ResourceKind],
    ) -> std::result::Result<(), HardResourceError> {
        let (id, _) = self.record_for(grant)?;
        for kind in kinds {
            let claim = grant
                .claims
                .get(kind)
                .ok_or(HardResourceError::MissingClaim {
                    grant: id,
                    kind: *kind,
                })?;
            if claim.submitted {
                return Err(HardResourceError::SubmittedClaimCannotBeRevoked {
                    grant: id,
                    kind: *kind,
                });
            }
        }
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

    pub fn release_all_held(
        &mut self,
        grant: &mut HardResourceGrant,
    ) -> std::result::Result<(), HardResourceError> {
        let kinds: Vec<_> = grant.claims.keys().copied().collect();
        self.release_held(grant, &kinds)
    }

    fn canonical_claims(
        claims: impl IntoIterator<Item = HardResourceClaim>,
    ) -> std::result::Result<Vec<HardResourceClaim>, HardResourceError> {
        let mut merged = BTreeMap::<ResourceKind, u64>::new();
        for claim in claims {
            if claim.amount == 0 {
                continue;
            }
            let amount = merged.entry(claim.kind).or_default();
            *amount = amount
                .checked_add(claim.amount)
                .ok_or(HardResourceError::ClaimOverflow(claim.kind))?;
        }
        Ok(merged
            .into_iter()
            .map(|(kind, amount)| HardResourceClaim { kind, amount })
            .collect())
    }

    fn validate_available(
        &self,
        class: ResourceClass,
        claims: &[HardResourceClaim],
    ) -> std::result::Result<(), HardResourceError> {
        for claim in claims {
            let state = self
                .limits
                .get(&claim.kind)
                .ok_or(HardResourceError::MissingLimit(claim.kind))?;
            if claim.amount > state.capacity {
                return Err(HardResourceError::ExceedsCapacity {
                    kind: claim.kind,
                    requested: claim.amount,
                    capacity: state.capacity,
                });
            }
            let available = state.capacity.saturating_sub(state.in_use);
            if claim.amount > available {
                return Err(HardResourceError::TemporarilyUnavailable {
                    kind: claim.kind,
                    requested: claim.amount,
                    available,
                });
            }
            if !class.may_use_hard_reserve() {
                let base_limit = state.capacity.saturating_sub(state.demand_reserve);
                let base_available = base_limit.saturating_sub(state.in_use);
                if claim.amount > base_available {
                    return Err(HardResourceError::DemandReserve {
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
        grant: &HardResourceGrant,
    ) -> std::result::Result<(HardResourceGrantId, &HardGrantRecord), HardResourceError> {
        let id = grant
            .id
            .ok_or(HardResourceError::GrantOwnerMismatch(HardResourceGrantId(
                0,
            )))?;
        let record = self
            .grants
            .get(&id)
            .ok_or(HardResourceError::UnknownGrant(id))?;
        if record.owner != grant.owner || record.claims != grant.claims {
            return Err(HardResourceError::GrantOwnerMismatch(id));
        }
        Ok((id, record))
    }
}

/// One amount requested from a registered resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceClaim {
    pub resource: ResourceId,
    pub amount: u64,
}

impl ResourceClaim {
    pub const fn new(resource: ResourceId, amount: u64) -> Self {
        Self { resource, amount }
    }
}

/// Physical resource vector for one schedulable operation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceRequest {
    claims: Vec<ResourceClaim>,
}

impl ResourceRequest {
    pub fn new(claims: impl IntoIterator<Item = ResourceClaim>) -> Result<Self> {
        let mut merged = BTreeMap::<ResourceId, u64>::new();
        for claim in claims {
            if claim.amount == 0 {
                continue;
            }
            let amount = merged.entry(claim.resource).or_default();
            *amount = amount.checked_add(claim.amount).ok_or_else(|| {
                Error::Execution(format!(
                    "resource request overflow for resource {}",
                    claim.resource.get()
                ))
            })?;
        }
        Ok(Self {
            claims: merged
                .into_iter()
                .map(|(resource, amount)| ResourceClaim { resource, amount })
                .collect(),
        })
    }

    pub fn claims(&self) -> &[ResourceClaim] {
        &self.claims
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    fn amount(&self, resource: ResourceId) -> u64 {
        self.claims
            .binary_search_by_key(&resource, |claim| claim.resource)
            .ok()
            .map(|index| self.claims[index].amount)
            .unwrap_or(0)
    }

    fn subtract(&mut self, released: &ResourceRequest) -> Result<()> {
        for claim in released.claims() {
            let available = self.amount(claim.resource);
            if claim.amount > available {
                return Err(Error::Internal(format!(
                    "resource grant claim {} release {}/{} exceeds ownership",
                    claim.resource.get(),
                    claim.amount,
                    available
                )));
            }
        }
        let mut remaining = Vec::with_capacity(self.claims.len());
        for claim in &self.claims {
            let amount = claim.amount - released.amount(claim.resource);
            if amount != 0 {
                remaining.push(ResourceClaim::new(claim.resource, amount));
            }
        }
        self.claims = remaining;
        Ok(())
    }
}

/// Durable identity for one granted physical resource vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceGrantId(u64);

impl ResourceGrantId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Why a physical request could not be admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceRejection {
    UnknownResource(ResourceId),
    ExceedsCapacity {
        resource: ResourceId,
        requested: u64,
        capacity: u64,
    },
    DemandReserve {
        resource: ResourceId,
        requested: u64,
        prefetch_available: u64,
    },
    TemporarilyUnavailable {
        resource: ResourceId,
        requested: u64,
        available: u64,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceBrokerStats {
    pub grants: u64,
    pub releases: u64,
    pub partial_releases: u64,
    pub promotions: u64,
    pub rejected_capacity: u64,
    pub rejected_reserve: u64,
    pub rejected_unavailable: u64,
    pub peak_grants: usize,
}

#[derive(Debug, Clone)]
struct ResourceState {
    name: &'static str,
    unit: ResourceUnit,
    capacity: u64,
    demand_reserve: u64,
    in_use: u64,
    prefetch_in_use: u64,
}

#[derive(Debug, Clone)]
struct GrantedResources {
    owner: u64,
    class: ResourceClass,
    request: ResourceRequest,
}

/// Snapshot suitable for metrics or policy decisions without exposing grants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceSnapshot {
    pub id: ResourceId,
    pub name: &'static str,
    pub unit: ResourceUnit,
    pub capacity: u64,
    pub demand_reserve: u64,
    pub in_use: u64,
    pub prefetch_in_use: u64,
}

/// Builder used by a hardware adapter to define its resource topology.
#[derive(Debug, Default)]
pub struct ResourceBrokerBuilder {
    resources: Vec<ResourceState>,
}

impl ResourceBrokerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one resource domain. `demand_reserve` is capacity prefetch may
    /// never consume, guaranteeing room for selected/demand work.
    pub fn register(
        &mut self,
        name: &'static str,
        unit: ResourceUnit,
        capacity: u64,
        demand_reserve: u64,
    ) -> Result<ResourceId> {
        if name.is_empty() {
            return Err(Error::Execution("resource name must not be empty".into()));
        }
        if demand_reserve > capacity {
            return Err(Error::Execution(format!(
                "resource {name} demand reserve {demand_reserve} exceeds capacity {capacity}"
            )));
        }
        if self.resources.iter().any(|resource| resource.name == name) {
            return Err(Error::Execution(format!(
                "resource {name} is already registered"
            )));
        }
        let index = u16::try_from(self.resources.len())
            .map_err(|_| Error::Execution("resource catalog exceeds u16 identity space".into()))?;
        self.resources.push(ResourceState {
            name,
            unit,
            capacity,
            demand_reserve,
            in_use: 0,
            prefetch_in_use: 0,
        });
        Ok(ResourceId(index))
    }

    pub fn build(self) -> ResourceBroker {
        ResourceBroker {
            resources: self.resources,
            grants: BTreeMap::new(),
            next_grant: 1,
            stats: ResourceBrokerStats::default(),
        }
    }
}

/// Hard-capacity broker for physical resources held across execution ticks.
///
/// The broker is deliberately separate from predictive [`ExpertIoBudget`](super::ExpertIoBudget):
/// prediction may be wrong and may use forced progress, while a physical grant
/// can never exceed capacity or the demand reserve.
#[derive(Debug)]
pub struct ResourceBroker {
    resources: Vec<ResourceState>,
    grants: BTreeMap<ResourceGrantId, GrantedResources>,
    next_grant: u64,
    stats: ResourceBrokerStats,
}

impl ResourceBroker {
    pub fn snapshots(&self) -> impl ExactSizeIterator<Item = ResourceSnapshot> + '_ {
        self.resources
            .iter()
            .enumerate()
            .map(|(index, resource)| ResourceSnapshot {
                id: ResourceId(index as u16),
                name: resource.name,
                unit: resource.unit,
                capacity: resource.capacity,
                demand_reserve: resource.demand_reserve,
                in_use: resource.in_use,
                prefetch_in_use: resource.prefetch_in_use,
            })
    }

    pub const fn stats(&self) -> ResourceBrokerStats {
        self.stats
    }

    pub fn active_grants(&self) -> usize {
        self.grants.len()
    }

    pub fn owner(&self, grant: ResourceGrantId) -> Option<u64> {
        self.grants.get(&grant).map(|grant| grant.owner)
    }

    pub fn request(
        &mut self,
        owner: u64,
        class: ResourceClass,
        request: ResourceRequest,
    ) -> std::result::Result<ResourceGrantId, ResourceRejection> {
        for claim in request.claims() {
            let Some(resource) = self.resources.get(claim.resource.get() as usize) else {
                return Err(ResourceRejection::UnknownResource(claim.resource));
            };
            if claim.amount > resource.capacity {
                self.stats.rejected_capacity = self.stats.rejected_capacity.saturating_add(1);
                return Err(ResourceRejection::ExceedsCapacity {
                    resource: claim.resource,
                    requested: claim.amount,
                    capacity: resource.capacity,
                });
            }
            let available = resource.capacity.saturating_sub(resource.in_use);
            if claim.amount > available {
                self.stats.rejected_unavailable = self.stats.rejected_unavailable.saturating_add(1);
                return Err(ResourceRejection::TemporarilyUnavailable {
                    resource: claim.resource,
                    requested: claim.amount,
                    available,
                });
            }
            if !class.is_demand() {
                let prefetch_limit = resource.capacity.saturating_sub(resource.demand_reserve);
                let prefetch_available = prefetch_limit.saturating_sub(resource.prefetch_in_use);
                if claim.amount > prefetch_available {
                    self.stats.rejected_reserve = self.stats.rejected_reserve.saturating_add(1);
                    return Err(ResourceRejection::DemandReserve {
                        resource: claim.resource,
                        requested: claim.amount,
                        prefetch_available,
                    });
                }
            }
        }

        let id = ResourceGrantId(self.next_grant);
        self.next_grant =
            self.next_grant
                .checked_add(1)
                .ok_or(ResourceRejection::ExceedsCapacity {
                    resource: ResourceId(0),
                    requested: u64::MAX,
                    capacity: u64::MAX - 1,
                })?;
        for claim in request.claims() {
            let resource = &mut self.resources[claim.resource.get() as usize];
            resource.in_use += claim.amount;
            if !class.is_demand() {
                resource.prefetch_in_use += claim.amount;
            }
        }
        self.grants.insert(
            id,
            GrantedResources {
                owner,
                class,
                request,
            },
        );
        self.stats.grants = self.stats.grants.saturating_add(1);
        self.stats.peak_grants = self.stats.peak_grants.max(self.grants.len());
        Ok(id)
    }

    /// Promote a prefetch grant when selected work adopts it. No resource is
    /// recharged; only the prefetch share is released for future prefetches.
    pub fn promote(&mut self, grant: ResourceGrantId, class: ResourceClass) -> Result<()> {
        if !class.is_demand() {
            return Err(Error::Execution(
                "resource grant promotion requires a demand class".into(),
            ));
        }
        let granted = self.grants.get(&grant).cloned().ok_or_else(|| {
            Error::Execution(format!("resource grant {} is not active", grant.get()))
        })?;
        if granted.class.is_demand() {
            if class > granted.class {
                self.grants
                    .get_mut(&grant)
                    .expect("validated active grant")
                    .class = class;
            }
            return Ok(());
        }
        for claim in granted.request.claims() {
            let resource = &self.resources[claim.resource.get() as usize];
            if resource.prefetch_in_use < claim.amount {
                return Err(Error::Internal(format!(
                    "resource {} prefetch accounting underflow during promotion",
                    resource.name
                )));
            }
        }
        for claim in granted.request.claims() {
            self.resources[claim.resource.get() as usize].prefetch_in_use -= claim.amount;
        }
        self.grants
            .get_mut(&grant)
            .expect("validated active grant")
            .class = class;
        self.stats.promotions = self.stats.promotions.saturating_add(1);
        Ok(())
    }

    pub fn release(&mut self, grant: ResourceGrantId) -> Result<()> {
        let request = self
            .grants
            .get(&grant)
            .map(|granted| granted.request.clone())
            .ok_or_else(|| {
                Error::Execution(format!("resource grant {} is not active", grant.get()))
            })?;
        self.release_claims(grant, request)
    }

    pub fn release_claims(
        &mut self,
        grant: ResourceGrantId,
        released: ResourceRequest,
    ) -> Result<()> {
        if released.is_empty() {
            return Ok(());
        }
        let granted = self.grants.get(&grant).cloned().ok_or_else(|| {
            Error::Execution(format!("resource grant {} is not active", grant.get()))
        })?;
        for claim in released.claims() {
            if claim.amount > granted.request.amount(claim.resource) {
                return Err(Error::Execution(format!(
                    "resource grant {} does not own resource {} amount {}",
                    grant.get(),
                    claim.resource.get(),
                    claim.amount
                )));
            }
            let resource = self
                .resources
                .get(claim.resource.get() as usize)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "active grant references unknown resource {}",
                        claim.resource.get()
                    ))
                })?;
            if resource.in_use < claim.amount
                || (!granted.class.is_demand() && resource.prefetch_in_use < claim.amount)
            {
                return Err(Error::Internal(format!(
                    "resource {} accounting underflow during release",
                    resource.name
                )));
            }
        }
        for claim in released.claims() {
            let resource = &mut self.resources[claim.resource.get() as usize];
            resource.in_use -= claim.amount;
            if !granted.class.is_demand() {
                resource.prefetch_in_use -= claim.amount;
            }
        }
        let remaining = self
            .grants
            .get_mut(&grant)
            .expect("validated active grant remains present");
        remaining.request.subtract(&released)?;
        if remaining.request.is_empty() {
            self.grants.remove(&grant);
            self.stats.releases = self.stats.releases.saturating_add(1);
        } else {
            self.stats.partial_releases = self.stats.partial_releases.saturating_add(1);
        }
        Ok(())
    }

    /// Dominant share used by a fairness policy: the largest fraction of any
    /// registered capacity requested by this operation.
    pub fn dominant_share(&self, request: &ResourceRequest) -> Result<f64> {
        let mut dominant = 0.0f64;
        for claim in request.claims() {
            let resource = self
                .resources
                .get(claim.resource.get() as usize)
                .ok_or_else(|| {
                    Error::Execution(format!(
                        "resource {} is not registered",
                        claim.resource.get()
                    ))
                })?;
            if resource.capacity == 0 {
                if claim.amount != 0 {
                    return Err(Error::Execution(format!(
                        "resource {} has zero capacity",
                        resource.name
                    )));
                }
                continue;
            }
            dominant = dominant.max(claim.amount as f64 / resource.capacity as f64);
        }
        Ok(dominant)
    }
}

#[derive(Debug, Clone, Copy)]
struct ExpertIoResourceIds {
    read_slots: ResourceId,
    storage_read_bytes: ResourceId,
    pinned_host_bytes: ResourceId,
    upload_slots: ResourceId,
    h2d_bytes: ResourceId,
    install_slots: ResourceId,
    device_install_bytes: ResourceId,
}

impl ExpertIoResourceIds {
    fn request(self, demand: ExpertIoResourceDemand) -> Result<ResourceRequest> {
        ResourceRequest::new([
            ResourceClaim::new(self.read_slots, demand.read_slots),
            ResourceClaim::new(self.storage_read_bytes, demand.storage_read_bytes),
            ResourceClaim::new(self.pinned_host_bytes, demand.pinned_host_bytes),
            ResourceClaim::new(self.upload_slots, demand.upload_slots),
            ResourceClaim::new(self.h2d_bytes, demand.h2d_bytes),
            ResourceClaim::new(self.install_slots, demand.install_slots),
            ResourceClaim::new(self.device_install_bytes, demand.device_install_bytes),
        ])
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExpertIoResourceBrokerHandle {
    broker: Arc<Mutex<ResourceBroker>>,
}

impl ExpertIoResourceBrokerHandle {
    pub(crate) fn snapshots(&self) -> Result<Vec<ResourceSnapshot>> {
        let broker = self
            .broker
            .lock()
            .map_err(|_| Error::Internal("expert-I/O resource broker lock was poisoned".into()))?;
        Ok(broker.snapshots().collect())
    }

    pub(crate) fn stats(&self) -> Result<ResourceBrokerStats> {
        let broker = self
            .broker
            .lock()
            .map_err(|_| Error::Internal("expert-I/O resource broker lock was poisoned".into()))?;
        Ok(broker.stats())
    }

    pub(crate) fn active_grants(&self) -> Result<usize> {
        let broker = self
            .broker
            .lock()
            .map_err(|_| Error::Internal("expert-I/O resource broker lock was poisoned".into()))?;
        Ok(broker.active_grants())
    }
}

pub(crate) struct BrokerExpertIoResourceControl {
    broker: Arc<Mutex<ResourceBroker>>,
    ids: ExpertIoResourceIds,
    limits: ExpertIoResourceLimits,
    completion_hub: CompletionHub,
}

impl BrokerExpertIoResourceControl {
    pub(crate) fn new(
        limits: ExpertIoResourceLimits,
        completion_hub: CompletionHub,
    ) -> Result<(Self, ExpertIoResourceBrokerHandle)> {
        let limits = limits.validate()?;
        let mut builder = ResourceBrokerBuilder::new();
        let capacity = limits.capacity;
        let reserve = limits.demand_reserve;
        let ids = ExpertIoResourceIds {
            read_slots: builder.register(
                "expert_io.read_slots",
                ResourceUnit::Slots,
                capacity.read_slots,
                reserve.read_slots,
            )?,
            storage_read_bytes: builder.register(
                "expert_io.storage_read_bytes",
                ResourceUnit::Bytes,
                capacity.storage_read_bytes,
                reserve.storage_read_bytes,
            )?,
            pinned_host_bytes: builder.register(
                "expert_io.pinned_host_bytes",
                ResourceUnit::Bytes,
                capacity.pinned_host_bytes,
                reserve.pinned_host_bytes,
            )?,
            upload_slots: builder.register(
                "expert_io.upload_slots",
                ResourceUnit::Slots,
                capacity.upload_slots,
                reserve.upload_slots,
            )?,
            h2d_bytes: builder.register(
                "expert_io.h2d_bytes",
                ResourceUnit::Bytes,
                capacity.h2d_bytes,
                reserve.h2d_bytes,
            )?,
            install_slots: builder.register(
                "expert_io.install_slots",
                ResourceUnit::Slots,
                capacity.install_slots,
                reserve.install_slots,
            )?,
            device_install_bytes: builder.register(
                "expert_io.device_install_bytes",
                ResourceUnit::Bytes,
                capacity.device_install_bytes,
                reserve.device_install_bytes,
            )?,
        };
        let broker = Arc::new(Mutex::new(builder.build()));
        Ok((
            Self {
                broker: broker.clone(),
                ids,
                limits,
                completion_hub,
            },
            ExpertIoResourceBrokerHandle { broker },
        ))
    }

    fn request_for(&self, demand: ExpertIoResourceDemand) -> Result<ResourceRequest> {
        demand.validate_within(self.limits.capacity, "expert-I/O physical operation")?;
        self.ids.request(demand)
    }
}

fn broker_resource_class(class: ExpertIoResourceClass) -> ResourceClass {
    match class {
        ExpertIoResourceClass::Prefetch => ResourceClass::Prefetch,
        ExpertIoResourceClass::Prefill => ResourceClass::Prefill,
        ExpertIoResourceClass::Verification => ResourceClass::Verification,
        ExpertIoResourceClass::Decode => ResourceClass::Decode,
    }
}

impl ExpertIoResourceControl for BrokerExpertIoResourceControl {
    fn try_acquire(
        &mut self,
        owner: u64,
        operation_id: u64,
        class: ExpertIoResourceClass,
        demand: ExpertIoResourceDemand,
    ) -> Result<ExpertIoResourceAdmission> {
        if operation_id == 0 {
            return Err(Error::Execution(
                "expert-I/O resource operation ID must be non-zero".into(),
            ));
        }
        let request = self.request_for(demand)?;
        let grant = {
            let mut broker = self.broker.lock().map_err(|_| {
                Error::Internal("expert-I/O resource broker lock was poisoned".into())
            })?;
            match broker.request(owner, broker_resource_class(class), request) {
                Ok(grant) => grant,
                Err(
                    ResourceRejection::DemandReserve { .. }
                    | ResourceRejection::TemporarilyUnavailable { .. },
                ) => return Ok(ExpertIoResourceAdmission::TemporarilyUnavailable),
                Err(ResourceRejection::UnknownResource(resource)) => {
                    return Err(Error::Internal(format!(
                        "expert-I/O request references unknown resource {}",
                        resource.get()
                    )));
                }
                Err(ResourceRejection::ExceedsCapacity {
                    resource,
                    requested,
                    capacity,
                }) => {
                    return Err(Error::Execution(format!(
                        "expert-I/O request {requested} exceeds resource {} capacity {capacity}",
                        resource.get()
                    )));
                }
            }
        };
        let permit = Box::new(BrokerExpertIoResourcePermit {
            broker: self.broker.clone(),
            grant: Some(grant),
            ids: self.ids,
            remaining: demand,
            released_stages: 0,
            completion_hub: self.completion_hub.clone(),
        });
        ExpertIoResourceGrant::new(operation_id, demand, permit)
            .map(ExpertIoResourceAdmission::Granted)
    }
}

struct BrokerExpertIoResourcePermit {
    broker: Arc<Mutex<ResourceBroker>>,
    grant: Option<ResourceGrantId>,
    ids: ExpertIoResourceIds,
    remaining: ExpertIoResourceDemand,
    released_stages: u8,
    completion_hub: CompletionHub,
}

impl BrokerExpertIoResourcePermit {
    fn release_active(&mut self) -> Result<()> {
        let grant = self.grant.ok_or_else(|| {
            Error::Execution("expert-I/O resource permit is already released".into())
        })?;
        self.broker
            .lock()
            .map_err(|_| Error::Internal("expert-I/O resource broker lock was poisoned".into()))?
            .release(grant)?;
        self.grant = None;
        self.remaining = ExpertIoResourceDemand::default();
        self.completion_hub.notify();
        Ok(())
    }

    fn stage_bit(stage: ExpertIoResourceStage) -> u8 {
        match stage {
            ExpertIoResourceStage::Read => 1 << 0,
            ExpertIoResourceStage::PinnedHost => 1 << 1,
            ExpertIoResourceStage::Upload => 1 << 2,
            ExpertIoResourceStage::Install => 1 << 3,
        }
    }

    fn stage_demand(&self, stage: ExpertIoResourceStage) -> ExpertIoResourceDemand {
        match stage {
            ExpertIoResourceStage::Read => ExpertIoResourceDemand {
                read_slots: self.remaining.read_slots,
                storage_read_bytes: self.remaining.storage_read_bytes,
                ..ExpertIoResourceDemand::default()
            },
            ExpertIoResourceStage::PinnedHost => ExpertIoResourceDemand {
                pinned_host_bytes: self.remaining.pinned_host_bytes,
                ..ExpertIoResourceDemand::default()
            },
            ExpertIoResourceStage::Upload => ExpertIoResourceDemand {
                upload_slots: self.remaining.upload_slots,
                h2d_bytes: self.remaining.h2d_bytes,
                ..ExpertIoResourceDemand::default()
            },
            ExpertIoResourceStage::Install => ExpertIoResourceDemand {
                install_slots: self.remaining.install_slots,
                device_install_bytes: self.remaining.device_install_bytes,
                ..ExpertIoResourceDemand::default()
            },
        }
    }

    fn clear_stage(&mut self, stage: ExpertIoResourceStage) {
        match stage {
            ExpertIoResourceStage::Read => {
                self.remaining.read_slots = 0;
                self.remaining.storage_read_bytes = 0;
            }
            ExpertIoResourceStage::PinnedHost => self.remaining.pinned_host_bytes = 0,
            ExpertIoResourceStage::Upload => {
                self.remaining.upload_slots = 0;
                self.remaining.h2d_bytes = 0;
            }
            ExpertIoResourceStage::Install => {
                self.remaining.install_slots = 0;
                self.remaining.device_install_bytes = 0;
            }
        }
    }
}

impl ExpertIoResourcePermit for BrokerExpertIoResourcePermit {
    fn promote(&mut self, class: ExpertIoResourceClass) -> Result<()> {
        let grant = self.grant.ok_or_else(|| {
            Error::Execution("expert-I/O resource permit is already released".into())
        })?;
        self.broker
            .lock()
            .map_err(|_| Error::Internal("expert-I/O resource broker lock was poisoned".into()))?
            .promote(grant, broker_resource_class(class))
    }

    fn release_stage(&mut self, stage: ExpertIoResourceStage) -> Result<()> {
        let bit = Self::stage_bit(stage);
        if self.released_stages & bit != 0 {
            return Err(Error::Execution(format!(
                "expert-I/O resource stage {stage:?} is already released"
            )));
        }
        let grant = self.grant.ok_or_else(|| {
            Error::Execution("expert-I/O resource permit is already released".into())
        })?;
        let demand = self.stage_demand(stage);
        let request = self.ids.request(demand)?;
        if !request.is_empty() {
            self.broker
                .lock()
                .map_err(|_| {
                    Error::Internal("expert-I/O resource broker lock was poisoned".into())
                })?
                .release_claims(grant, request)?;
        }
        self.clear_stage(stage);
        self.released_stages |= bit;
        if self.remaining.is_empty() {
            self.grant = None;
        }
        self.completion_hub.notify();
        Ok(())
    }

    fn release(mut self: Box<Self>) -> Result<()> {
        self.release_active()
    }
}

impl Drop for BrokerExpertIoResourcePermit {
    fn drop(&mut self) {
        if self.grant.is_some() {
            let _ = self.release_active();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn broker() -> (ResourceBroker, ResourceId, ResourceId) {
        let mut builder = ResourceBrokerBuilder::new();
        let slabs = builder
            .register("nvme0.pinned_slabs", ResourceUnit::Slots, 8, 2)
            .unwrap();
        let h2d = builder
            .register("gpu0.h2d", ResourceUnit::Bytes, 1_000, 250)
            .unwrap();
        (builder.build(), slabs, h2d)
    }

    fn expert_io_limits(upload_slots: u64, demand_reserve: u64) -> ExpertIoResourceLimits {
        ExpertIoResourceLimits {
            capacity: ExpertIoResourceDemand {
                upload_slots,
                ..ExpertIoResourceDemand::default()
            },
            demand_reserve: ExpertIoResourceDemand {
                upload_slots: demand_reserve,
                ..ExpertIoResourceDemand::default()
            },
        }
    }

    #[test]
    fn expert_io_adapter_holds_hard_capacity_until_grant_release() {
        let (mut control, handle) =
            BrokerExpertIoResourceControl::new(expert_io_limits(1, 0), CompletionHub::new())
                .unwrap();
        let demand = ExpertIoResourceDemand {
            upload_slots: 1,
            ..ExpertIoResourceDemand::default()
        };
        let first = match control
            .try_acquire(1, 11, ExpertIoResourceClass::Decode, demand)
            .unwrap()
        {
            ExpertIoResourceAdmission::Granted(grant) => grant,
            ExpertIoResourceAdmission::TemporarilyUnavailable => panic!("first grant rejected"),
        };
        assert_eq!(handle.active_grants().unwrap(), 1);
        assert!(matches!(
            control
                .try_acquire(2, 12, ExpertIoResourceClass::Decode, demand)
                .unwrap(),
            ExpertIoResourceAdmission::TemporarilyUnavailable
        ));

        first.release().unwrap();

        assert_eq!(handle.active_grants().unwrap(), 0);
        let stats = handle.stats().unwrap();
        assert_eq!(stats.grants, 1);
        assert_eq!(stats.releases, 1);
        assert_eq!(stats.rejected_unavailable, 1);
    }

    #[test]
    fn expert_io_prefetch_promotion_returns_reserved_share_without_recharging() {
        let (mut control, handle) =
            BrokerExpertIoResourceControl::new(expert_io_limits(2, 1), CompletionHub::new())
                .unwrap();
        let demand = ExpertIoResourceDemand {
            upload_slots: 1,
            ..ExpertIoResourceDemand::default()
        };
        let mut prefetched = match control
            .try_acquire(1, 21, ExpertIoResourceClass::Prefetch, demand)
            .unwrap()
        {
            ExpertIoResourceAdmission::Granted(grant) => grant,
            ExpertIoResourceAdmission::TemporarilyUnavailable => panic!("prefetch rejected"),
        };
        assert!(matches!(
            control
                .try_acquire(2, 22, ExpertIoResourceClass::Prefetch, demand)
                .unwrap(),
            ExpertIoResourceAdmission::TemporarilyUnavailable
        ));

        prefetched
            .promote(ExpertIoResourceClass::Verification)
            .unwrap();
        let second = match control
            .try_acquire(2, 22, ExpertIoResourceClass::Prefetch, demand)
            .unwrap()
        {
            ExpertIoResourceAdmission::Granted(grant) => grant,
            ExpertIoResourceAdmission::TemporarilyUnavailable => {
                panic!("promotion did not return prefetch share")
            }
        };
        assert_eq!(handle.active_grants().unwrap(), 2);
        assert_eq!(handle.stats().unwrap().promotions, 1);

        drop(prefetched);
        drop(second);
        assert_eq!(handle.active_grants().unwrap(), 0);
        assert_eq!(handle.stats().unwrap().releases, 2);
    }

    #[test]
    fn expert_io_stages_release_physical_capacity_at_their_completion_boundary() {
        let demand = ExpertIoResourceDemand {
            read_slots: 1,
            storage_read_bytes: 4096,
            pinned_host_bytes: 8192,
            upload_slots: 1,
            h2d_bytes: 2048,
            install_slots: 1,
            device_install_bytes: 2048,
        };
        let (mut control, handle) = BrokerExpertIoResourceControl::new(
            ExpertIoResourceLimits {
                capacity: demand,
                demand_reserve: ExpertIoResourceDemand::default(),
            },
            CompletionHub::new(),
        )
        .unwrap();
        let mut grant = match control
            .try_acquire(1, 31, ExpertIoResourceClass::Decode, demand)
            .unwrap()
        {
            ExpertIoResourceAdmission::Granted(grant) => grant,
            ExpertIoResourceAdmission::TemporarilyUnavailable => panic!("grant rejected"),
        };

        grant.release_stage(ExpertIoResourceStage::Read).unwrap();
        let snapshots = handle.snapshots().unwrap();
        assert_eq!(
            snapshots
                .iter()
                .find(|resource| resource.name == "expert_io.read_slots")
                .unwrap()
                .in_use,
            0
        );
        assert_eq!(
            snapshots
                .iter()
                .find(|resource| resource.name == "expert_io.pinned_host_bytes")
                .unwrap()
                .in_use,
            demand.pinned_host_bytes
        );
        assert!(
            grant
                .release_stage(ExpertIoResourceStage::Read)
                .unwrap_err()
                .to_string()
                .contains("already released")
        );

        grant
            .release_stage(ExpertIoResourceStage::PinnedHost)
            .unwrap();
        grant.release_stage(ExpertIoResourceStage::Upload).unwrap();
        assert_eq!(handle.active_grants().unwrap(), 1);
        grant.release().unwrap();

        assert_eq!(handle.active_grants().unwrap(), 0);
        let stats = handle.stats().unwrap();
        assert_eq!(stats.partial_releases, 3);
        assert_eq!(stats.releases, 1);
    }

    #[test]
    fn request_merges_duplicate_claims_and_ignores_zeroes() {
        let resource = ResourceId(3);
        let request = ResourceRequest::new([
            ResourceClaim::new(resource, 2),
            ResourceClaim::new(resource, 0),
            ResourceClaim::new(resource, 5),
        ])
        .unwrap();
        assert_eq!(request.claims(), &[ResourceClaim::new(resource, 7)]);
    }

    #[test]
    fn prefetch_cannot_consume_demand_reserve() {
        let (mut broker, slabs, _) = broker();
        let request = ResourceRequest::new([ResourceClaim::new(slabs, 7)]).unwrap();
        assert_eq!(
            broker.request(1, ResourceClass::Prefetch, request),
            Err(ResourceRejection::DemandReserve {
                resource: slabs,
                requested: 7,
                prefetch_available: 6,
            })
        );
        assert_eq!(broker.snapshots().next().unwrap().in_use, 0);
    }

    #[test]
    fn demand_can_use_reserved_capacity_but_never_overcommit() {
        let (mut broker, slabs, _) = broker();
        let full = ResourceRequest::new([ResourceClaim::new(slabs, 8)]).unwrap();
        let grant = broker.request(1, ResourceClass::Decode, full).unwrap();
        let one = ResourceRequest::new([ResourceClaim::new(slabs, 1)]).unwrap();
        assert_eq!(
            broker.request(2, ResourceClass::Decode, one),
            Err(ResourceRejection::TemporarilyUnavailable {
                resource: slabs,
                requested: 1,
                available: 0,
            })
        );
        broker.release(grant).unwrap();
        assert_eq!(broker.snapshots().next().unwrap().in_use, 0);
    }

    #[test]
    fn multi_resource_request_is_failure_atomic() {
        let (mut broker, slabs, h2d) = broker();
        let request =
            ResourceRequest::new([ResourceClaim::new(slabs, 2), ResourceClaim::new(h2d, 1_001)])
                .unwrap();
        assert!(matches!(
            broker.request(1, ResourceClass::Decode, request),
            Err(ResourceRejection::ExceedsCapacity { resource, .. }) if resource == h2d
        ));
        assert!(broker.snapshots().all(|resource| resource.in_use == 0));
    }

    #[test]
    fn promotion_reuses_grant_and_returns_prefetch_share() {
        let (mut broker, slabs, _) = broker();
        let request = ResourceRequest::new([ResourceClaim::new(slabs, 6)]).unwrap();
        let grant = broker.request(7, ResourceClass::Prefetch, request).unwrap();
        assert_eq!(broker.snapshots().next().unwrap().prefetch_in_use, 6);
        broker.promote(grant, ResourceClass::Verification).unwrap();
        let snapshot = broker.snapshots().next().unwrap();
        assert_eq!(snapshot.in_use, 6);
        assert_eq!(snapshot.prefetch_in_use, 0);
        assert_eq!(broker.owner(grant), Some(7));
        assert_eq!(broker.stats().promotions, 1);
    }

    #[test]
    fn release_is_exactly_once() {
        let (mut broker, slabs, _) = broker();
        let request = ResourceRequest::new([ResourceClaim::new(slabs, 1)]).unwrap();
        let grant = broker.request(1, ResourceClass::Decode, request).unwrap();
        broker.release(grant).unwrap();
        assert!(
            broker
                .release(grant)
                .unwrap_err()
                .to_string()
                .contains("not active")
        );
        assert_eq!(broker.stats().releases, 1);
    }

    #[test]
    fn dominant_share_is_hardware_normalized() {
        let (broker, slabs, h2d) = broker();
        let request =
            ResourceRequest::new([ResourceClaim::new(slabs, 2), ResourceClaim::new(h2d, 500)])
                .unwrap();
        assert_eq!(broker.dominant_share(&request).unwrap(), 0.5);
    }

    #[test]
    fn hardware_adapter_can_register_independent_domains() {
        let mut builder = ResourceBrokerBuilder::new();
        let nvme0 = builder
            .register("nvme0.sqe", ResourceUnit::Operations, 64, 16)
            .unwrap();
        let nvme1 = builder
            .register("nvme1.sqe", ResourceUnit::Operations, 128, 32)
            .unwrap();
        assert_ne!(nvme0, nvme1);
        assert!(
            builder
                .register("nvme0.sqe", ResourceUnit::Operations, 1, 0)
                .unwrap_err()
                .to_string()
                .contains("already registered")
        );
    }
}
