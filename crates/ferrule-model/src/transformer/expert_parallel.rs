//! Bounded, transport-neutral expert-parallel activation dispatch.
//!
//! Dense Qwen does not use EP. This explicit MoE boundary supports synthetic
//! callers; wiring DeepSeek, IPC or HostCollective is a separate integration.
//! Router math, shared experts, residuals, weight residency and transaction
//! lifecycle remain outside this module. Messages contain activations/results,
//! never weights. `ExpertId` includes the layer to reject cross-layer replies.
//!
//! Each plan covers one source rank with a sequence identity per source row.
//! Callers must not reuse a transaction's layer/source-row identities for other
//! work. Every execution/publication boundary requires the caller's existing
//! transaction authority to reject cancelled or unknown identities. This module
//! has no transaction state machine or replay registry. The caller must serialize
//! cancellation with publication; a synchronous worker cannot be interrupted.
//! EP members here are an attached activation/expert group only; they do not
//! establish tensor-parallel or KV ownership semantics.

use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, ParallelRankId, Result};

use crate::moe::ExpertId;

pub use super::operators::{ExpertResultExecutor, RoutedSwiGluExecutor, RoutedSwiGluRequest};
use super::{
    CpuStandardDecoderOperators, ExpertAvailability, ExpertProvider, ExpertSwiGluOperator,
    HostRows, OperatorProgress, RouterRoutes, Rows, RowsDType, RowsShape, UnsupportedOperator,
};

type RouteIdentityKey = (ExpertId, usize, usize, ParallelRankId, ExpertPlacementEpoch);

fn route_identity_key(metadata: &ExpertRouteMetadata) -> RouteIdentityKey {
    (
        metadata.logical_expert,
        metadata.physical_slot,
        metadata.replica,
        metadata.owner_rank,
        metadata.placement_epoch,
    )
}

/// Monotonic identity for a logical-to-physical expert placement.
///
/// This is deliberately separate from `ExecutionTransactionId`: an epoch
/// describes the placement contract observed by a route, not completion,
/// cancellation, or publication authority.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertPlacementEpoch(u64);

impl ExpertPlacementEpoch {
    pub const INITIAL: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One physical slot for a logical expert. The slot is an activation-routing
/// identity only; it does not carry weights or change residency authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertPlacementEntry {
    pub logical_expert: ExpertId,
    pub physical_slot: usize,
    pub replica: usize,
    pub owner_rank: ParallelRankId,
}

/// Versioned route identity at the logical -> physical slot -> owner seam.
///
/// `RouterRoutes` remains the existing model-router output. This metadata is
/// the EP attachment's authenticated interpretation of one selected route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRouteMetadata {
    pub logical_expert: ExpertId,
    pub physical_slot: usize,
    pub replica: usize,
    pub owner_rank: ParallelRankId,
    pub placement_epoch: ExpertPlacementEpoch,
}

impl ExpertRouteMetadata {
    pub const fn new(
        logical_expert: ExpertId,
        physical_slot: usize,
        replica: usize,
        owner_rank: ParallelRankId,
        placement_epoch: ExpertPlacementEpoch,
    ) -> Self {
        Self {
            logical_expert,
            physical_slot,
            replica,
            owner_rank,
            placement_epoch,
        }
    }
}

/// Immutable logical expert -> physical replica/slot -> owner description.
/// No weight data, migration state, or transaction state is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPlacement {
    epoch: ExpertPlacementEpoch,
    replicas: BTreeMap<ExpertId, Vec<ExpertPlacementEntry>>,
    slots: BTreeMap<usize, ExpertPlacementEntry>,
    participants: BTreeSet<ParallelRankId>,
}

impl ExpertPlacement {
    /// Compatibility constructor for the original fixed one-slot-per-expert
    /// mapping. Physical slots are assigned deterministically by
    /// `(layer, logical expert, owner)` and the fixed map has the initial
    /// epoch; it never invents a fresh epoch for a route.
    pub fn new(entries: impl IntoIterator<Item = (usize, usize, ParallelRankId)>) -> Result<Self> {
        let mut logical_seen = BTreeSet::new();
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        entries.sort_unstable_by_key(|&(layer, expert, owner_rank)| (layer, expert, owner_rank));
        let entries = entries
            .into_iter()
            .enumerate()
            .map(|(physical_slot, (layer, expert, owner_rank))| {
                let logical_expert = ExpertId::new(layer, expert);
                if !logical_seen.insert(logical_expert) {
                    return Err(protocol_error(format!(
                        "duplicate placement for expert {layer}:{expert}"
                    )));
                }
                Ok(ExpertPlacementEntry {
                    logical_expert,
                    physical_slot,
                    replica: 0,
                    owner_rank,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_entries(ExpertPlacementEpoch::INITIAL, entries, [])
    }

    /// Build a versioned map with explicit physical slots and replica indexes.
    ///
    /// `participants` is the attached EP group. An empty participant iterator
    /// means "infer the group from owners", which keeps fixed maps concise.
    /// Every physical slot and `(logical expert, replica)` pair must be unique;
    /// replica indexes must be contiguous from zero for each logical expert.
    pub fn from_entries(
        epoch: ExpertPlacementEpoch,
        entries: impl IntoIterator<Item = ExpertPlacementEntry>,
        participants: impl IntoIterator<Item = ParallelRankId>,
    ) -> Result<Self> {
        let mut replicas = BTreeMap::<ExpertId, Vec<ExpertPlacementEntry>>::new();
        let mut slots = BTreeMap::new();
        let mut owner_participants = BTreeSet::new();
        for entry in entries {
            if slots.insert(entry.physical_slot, entry).is_some() {
                return Err(protocol_error(format!(
                    "duplicate physical slot {}",
                    entry.physical_slot
                )));
            }
            owner_participants.insert(entry.owner_rank);
            replicas
                .entry(entry.logical_expert)
                .or_default()
                .push(entry);
        }

        for (logical_expert, entries) in &mut replicas {
            entries.sort_unstable_by_key(|entry| entry.replica);
            let mut seen_replicas = BTreeSet::new();
            for (expected_replica, entry) in entries.iter().enumerate() {
                if !seen_replicas.insert(entry.replica) {
                    return Err(protocol_error(format!(
                        "duplicate replica {} for expert {}:{}",
                        entry.replica, logical_expert.layer, logical_expert.expert
                    )));
                }
                if entry.replica != expected_replica {
                    return Err(protocol_error(format!(
                        "invalid replica mapping for expert {}:{}",
                        logical_expert.layer, logical_expert.expert
                    )));
                }
            }
        }

        let requested_participants = participants.into_iter().collect::<BTreeSet<_>>();
        let participants = if requested_participants.is_empty() {
            owner_participants
        } else {
            if !owner_participants.is_subset(&requested_participants) {
                return Err(protocol_error("placement owner is not an EP participant"));
            }
            requested_participants
        };

        Ok(Self {
            epoch,
            replicas,
            slots,
            participants,
        })
    }

    pub fn epoch(&self) -> ExpertPlacementEpoch {
        self.epoch
    }

    /// Alias used by callers that describe placement versions as generations.
    pub fn generation(&self) -> ExpertPlacementEpoch {
        self.epoch
    }

    pub fn is_replicated(&self) -> bool {
        self.replicas.values().any(|entries| entries.len() > 1)
    }

    pub fn participants(&self) -> &BTreeSet<ParallelRankId> {
        &self.participants
    }

    pub fn logical_experts(&self) -> impl Iterator<Item = &ExpertId> {
        self.replicas.keys()
    }

    pub fn entries(&self) -> impl Iterator<Item = &ExpertPlacementEntry> {
        self.replicas.values().flat_map(|entries| entries.iter())
    }

    pub fn replicas(&self, logical_expert: ExpertId) -> Option<&[ExpertPlacementEntry]> {
        self.replicas.get(&logical_expert).map(Vec::as_slice)
    }

    pub fn entry(
        &self,
        logical_expert: ExpertId,
        physical_slot: usize,
    ) -> Option<&ExpertPlacementEntry> {
        self.slots
            .get(&physical_slot)
            .filter(|entry| entry.logical_expert == logical_expert)
    }

    pub fn primary(&self, logical_expert: ExpertId) -> Option<&ExpertPlacementEntry> {
        self.replicas.get(&logical_expert)?.first()
    }

    /// Compatibility lookup. For replicated logical experts this returns the
    /// deterministic primary replica (replica zero); callers needing a
    /// specific copy must use `replicas` or `entry`.
    pub fn owner(&self, layer: usize, expert: usize) -> Option<ParallelRankId> {
        self.primary(ExpertId::new(layer, expert))
            .map(|entry| entry.owner_rank)
    }

    pub fn route_metadata(
        &self,
        logical_expert: ExpertId,
        replica: usize,
    ) -> Option<ExpertRouteMetadata> {
        let entry = self.replicas(logical_expert)?.get(replica)?;
        Some(ExpertRouteMetadata::new(
            entry.logical_expert,
            entry.physical_slot,
            entry.replica,
            entry.owner_rank,
            self.epoch,
        ))
    }

    /// Validate a descriptive placement update without performing migration,
    /// changing residency, or changing owner authority.
    pub fn validate_update(&self, next: &Self) -> Result<()> {
        if next.epoch <= self.epoch {
            return Err(protocol_error(format!(
                "placement epoch {} is not newer than {}",
                next.epoch.get(),
                self.epoch.get()
            )));
        }
        if self.replicas.keys().ne(next.replicas.keys()) {
            return Err(protocol_error(
                "placement update changed the logical expert set",
            ));
        }
        Ok(())
    }
}

/// Caller-supplied identity, not a transaction allocator or lifecycle handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertDispatchContext {
    pub transaction: ExecutionTransactionId,
    pub source_rank: ParallelRankId,
    pub layer: usize,
}

/// Explicit admission bounds. There is intentionally no unbounded default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertDispatchLimits {
    /// Total routed tokens (`rows * top_k`), not a per-owner balanced quota.
    pub max_tokens: usize,
    /// Logical F32 activation payload bytes for each direction independently.
    /// Both requests and replies may coexist. Framing, metadata, allocations,
    /// weights and worker scratch are excluded. Transport adapters must bound
    /// framing/member counts and enforce limits before allocating peer data.
    pub max_bytes: usize,
}

/// One unweighted activation row. `weight` is metadata, never applied by owners.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertToken {
    pub transaction: ExecutionTransactionId,
    pub sequence: u64,
    pub source_rank: ParallelRankId,
    pub source_row: usize,
    pub route_slot: usize,
    pub expert: ExpertId,
    pub weight: f32,
    pub payload: Vec<f32>,
}

/// An unweighted expert output with the original identity and responding owner.
/// Combine uses its frozen router weights, not weights supplied by a peer.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertResult {
    pub transaction: ExecutionTransactionId,
    pub sequence: u64,
    pub source_rank: ParallelRankId,
    pub source_row: usize,
    pub route_slot: usize,
    pub expert: ExpertId,
    pub owner_rank: ParallelRankId,
    pub output: Vec<f32>,
}

impl ExpertResult {
    pub fn from_token(token: &ExpertToken, owner_rank: ParallelRankId, output: Vec<f32>) -> Self {
        Self {
            transaction: token.transaction,
            sequence: token.sequence,
            source_rank: token.source_rank,
            source_row: token.source_row,
            route_slot: token.route_slot,
            expert: token.expert,
            owner_rank,
            output,
        }
    }
}

/// An authenticated route view paired with the legacy activation token.
/// `ExpertDispatchPlan::route_view` is the only plan-owned constructor used by
/// execution; callers cannot make a stale or unknown view valid by changing the
/// legacy token identity alone.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertRouteToken {
    pub metadata: ExpertRouteMetadata,
    pub token: ExpertToken,
}

/// A route-aware worker reply. The metadata is checked before the legacy
/// `ExpertResult` is released to combine, so a replica cannot be collapsed to
/// the logical primary by an accidental worker implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertRouteResult {
    pub metadata: ExpertRouteMetadata,
    pub result: ExpertResult,
}

/// Dispatch retains a bucket for every member, including owners with no work.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertTokenBucket {
    pub owner_rank: ParallelRankId,
    pub tokens: Vec<ExpertToken>,
}

#[derive(Debug)]
struct ExpectedRoute {
    expert: ExpertId,
    metadata: ExpertRouteMetadata,
    sequence: u64,
    owner_index: usize,
    weight: f32,
}

/// Immutable plan in original member and `(source_row, route_slot)` order.
///
/// Construction accepts router outputs, never logits. Duplicate selected experts
/// within one row are invalid; the same expert across rows is valid. Combining
/// validates the entire result set before reducing and never mutates caller
/// storage. Exactly-once weighting is per combine call, not a transaction-wide
/// publication guarantee: liveness and replay protection belong to the caller.
#[derive(Debug)]
pub struct ExpertDispatchPlan {
    context: ExpertDispatchContext,
    placement_epoch: ExpertPlacementEpoch,
    placement_slots: BTreeMap<(ExpertId, usize), ExpertPlacementEntry>,
    logical_experts: BTreeSet<ExpertId>,
    members: Vec<ParallelRankId>,
    owner_counts: Vec<usize>,
    routes: Vec<ExpectedRoute>,
    route_identity_keys: BTreeSet<RouteIdentityKey>,
    rows: usize,
    top_k: usize,
    width: usize,
    required_bytes: usize,
}

impl ExpertDispatchPlan {
    /// Freeze one routed layer/source batch. `sequences[row]` is the caller's
    /// sequence identity for that source row; several rows may share it.
    /// `width` is both the input and output width of each routed expert.
    /// Like `RouterRoutes`, a plan is non-empty; individual owner buckets may
    /// be empty. No equal-sized partition or per-owner load balance is assumed.
    /// `context.source_rank` identifies the caller, not an expert worker. It may
    /// be external to `members`; the attachment must authenticate it before
    /// construction. This plan freezes that exact source and transaction and
    /// checks both on every token/result, independently of worker membership.
    pub fn new(
        context: ExpertDispatchContext,
        members: Vec<ParallelRankId>,
        placement: &ExpertPlacement,
        routes: &RouterRoutes,
        sequences: &[u64],
        width: usize,
        limits: ExpertDispatchLimits,
    ) -> Result<Self> {
        if placement.is_replicated() {
            return Err(protocol_error(
                "replicated placement requires explicit route metadata; legacy new cannot select a primary",
            ));
        }
        if placement.epoch() != ExpertPlacementEpoch::INITIAL {
            return Err(protocol_error(
                "unversioned RouterRoutes require the initial fixed placement; use new_with_route_metadata",
            ));
        }
        let metadata = routes
            .expert_ids()
            .iter()
            .map(|&expert| {
                let logical_expert = ExpertId::new(context.layer, expert);
                let entry = placement.primary(logical_expert).ok_or_else(|| {
                    protocol_error(format!(
                        "unknown expert logical id {}:{expert}",
                        context.layer
                    ))
                })?;
                Ok(ExpertRouteMetadata::new(
                    logical_expert,
                    entry.physical_slot,
                    entry.replica,
                    entry.owner_rank,
                    placement.epoch(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new_with_route_metadata(
            context, members, placement, routes, &metadata, sequences, width, limits,
        )
    }

    /// Construct a plan from explicit logical -> physical route metadata.
    ///
    /// This is the versioned boundary for a dynamic placement description.
    /// The old `new` constructor resolves the deterministic primary replica for
    /// fixed maps; it never manufactures a fresh epoch. No migration or weight
    /// movement is performed here.
    pub fn new_with_route_metadata(
        context: ExpertDispatchContext,
        members: Vec<ParallelRankId>,
        placement: &ExpertPlacement,
        routes: &RouterRoutes,
        route_metadata: &[ExpertRouteMetadata],
        sequences: &[u64],
        width: usize,
        limits: ExpertDispatchLimits,
    ) -> Result<Self> {
        if members.is_empty() || width == 0 {
            return Err(protocol_error(
                "members and activation width must be non-empty",
            ));
        }
        if sequences.len() != routes.rows() {
            return Err(protocol_error(
                "one sequence identity is required per source row",
            ));
        }
        let token_count = checked_mul(routes.rows(), routes.top_k())?;
        if route_metadata.len() != token_count {
            return Err(protocol_error(
                "one placement route metadata entry is required per route",
            ));
        }
        if token_count > limits.max_tokens {
            return Err(protocol_error("routed token limit exceeded"));
        }
        // Check activation bounds before allocating route indexes or payloads.
        let required_bytes = checked_mul(checked_mul(token_count, width)?, size_of::<f32>())?;
        if required_bytes > limits.max_bytes {
            return Err(protocol_error("dispatch byte limit exceeded"));
        }
        let mut member_indexes = BTreeMap::new();
        for (index, &member) in members.iter().enumerate() {
            if member_indexes.insert(member, index).is_some() {
                return Err(protocol_error("duplicate dispatch member"));
            }
        }
        let placement_slots = placement
            .entries()
            .map(|entry| ((entry.logical_expert, entry.physical_slot), *entry))
            .collect::<BTreeMap<_, _>>();
        let logical_experts = placement
            .logical_experts()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut owner_counts = vec![0; members.len()];
        let mut expected = Vec::with_capacity(token_count);
        for (row, &sequence) in sequences.iter().enumerate() {
            let (experts, weights) = routes.row(row)?;
            let mut seen = BTreeSet::new();
            for (route_slot, (&expert, &weight)) in experts.iter().zip(weights).enumerate() {
                if !seen.insert(expert) {
                    return Err(protocol_error(format!(
                        "duplicate expert in source row {row}"
                    )));
                }
                if !weight.is_finite() || weight < 0.0 {
                    return Err(protocol_error(
                        "route weight must be finite and non-negative",
                    ));
                }
                let logical_expert = ExpertId::new(context.layer, expert);
                let metadata = route_metadata[row * routes.top_k() + route_slot];
                if metadata.placement_epoch != placement.epoch() {
                    if metadata.placement_epoch < placement.epoch() {
                        return Err(protocol_error(format!(
                            "stale route placement epoch {} (current {})",
                            metadata.placement_epoch.get(),
                            placement.epoch().get()
                        )));
                    }
                    return Err(protocol_error(format!(
                        "route placement epoch {} does not match current {}",
                        metadata.placement_epoch.get(),
                        placement.epoch().get()
                    )));
                }
                if metadata.logical_expert != logical_expert {
                    return Err(protocol_error("route logical expert mismatch"));
                }
                if placement.replicas(logical_expert).is_none() {
                    return Err(protocol_error(format!(
                        "unknown expert logical id {}:{expert}",
                        context.layer
                    )));
                }
                let entry = placement
                    .entry(logical_expert, metadata.physical_slot)
                    .ok_or_else(|| protocol_error("unknown physical expert slot"))?;
                if entry.replica != metadata.replica {
                    return Err(protocol_error("route replica mismatch"));
                }
                if entry.owner_rank != metadata.owner_rank {
                    return Err(protocol_error("route owner mismatch"));
                }
                let owner_index = *member_indexes
                    .get(&metadata.owner_rank)
                    .ok_or_else(|| protocol_error("unknown expert owner rank"))?;
                owner_counts[owner_index] += 1;
                expected.push(ExpectedRoute {
                    expert: logical_expert,
                    metadata,
                    sequence,
                    owner_index,
                    weight,
                });
            }
        }
        let route_identity_keys = expected
            .iter()
            .map(|route| route_identity_key(&route.metadata))
            .collect();
        Ok(Self {
            context,
            placement_epoch: placement.epoch(),
            placement_slots,
            logical_experts,
            members,
            owner_counts,
            routes: expected,
            route_identity_keys,
            rows: routes.rows(),
            top_k: routes.top_k(),
            width,
            required_bytes,
        })
    }

    pub fn context(&self) -> ExpertDispatchContext {
        self.context
    }

    pub fn placement_epoch(&self) -> ExpertPlacementEpoch {
        self.placement_epoch
    }

    pub fn route_metadata(&self) -> impl Iterator<Item = &ExpertRouteMetadata> {
        self.routes.iter().map(|route| &route.metadata)
    }

    /// Authenticate a route identity against this immutable plan before it is
    /// attached to a transport or owner bucket. Epoch mismatch is checked
    /// before logical/physical identity so an old route cannot be refreshed by
    /// merely matching the current owner.
    pub fn validate_route_metadata(&self, metadata: &ExpertRouteMetadata) -> Result<()> {
        if metadata.placement_epoch != self.placement_epoch {
            if metadata.placement_epoch < self.placement_epoch {
                return Err(protocol_error(format!(
                    "stale route placement epoch {} (current {})",
                    metadata.placement_epoch.get(),
                    self.placement_epoch.get()
                )));
            }
            return Err(protocol_error(format!(
                "route placement epoch {} does not match current {}",
                metadata.placement_epoch.get(),
                self.placement_epoch.get()
            )));
        }
        if !self.logical_experts.contains(&metadata.logical_expert) {
            return Err(protocol_error(
                "unknown expert logical id in route metadata",
            ));
        }
        let entry = self
            .placement_slots
            .get(&(metadata.logical_expert, metadata.physical_slot))
            .ok_or_else(|| protocol_error("unknown physical expert slot"))?;
        if entry.replica != metadata.replica {
            return Err(protocol_error("route replica mismatch"));
        }
        if entry.owner_rank != metadata.owner_rank {
            return Err(protocol_error("route owner mismatch"));
        }
        if !self
            .route_identity_keys
            .contains(&route_identity_key(metadata))
        {
            return Err(protocol_error(
                "route metadata is not in this dispatch plan",
            ));
        }
        Ok(())
    }

    pub fn members(&self) -> &[ParallelRankId] {
        &self.members
    }

    pub fn token_count(&self) -> usize {
        self.routes.len()
    }

    pub fn required_bytes(&self) -> usize {
        self.required_bytes
    }

    /// Return route identities authenticated against this plan and bucket.
    /// The returned view carries the physical slot, replica, owner, and epoch
    /// that the route-aware worker must use.
    pub fn route_view(&self, bucket: &ExpertTokenBucket) -> Result<Vec<ExpertRouteToken>> {
        self.validate_bucket(bucket)?;
        bucket
            .tokens
            .iter()
            .map(|token| {
                let route = self.route(token.source_row, token.route_slot)?;
                self.validate_route_metadata(&route.metadata)?;
                Ok(ExpertRouteToken {
                    metadata: route.metadata,
                    token: token.clone(),
                })
            })
            .collect()
    }

    /// Buckets raw F32 row-major activations in member order. Tokens within each
    /// bucket retain the router's row/slot order, not expert-ID or score order.
    pub fn dispatch(
        &self,
        payload: &[f32],
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<Vec<ExpertTokenBucket>> {
        check_active(self.context.transaction)?;
        validate_values(payload, checked_mul(self.rows, self.width)?, "input")?;
        let mut buckets = self
            .members
            .iter()
            .zip(&self.owner_counts)
            .map(|(&owner_rank, &count)| ExpertTokenBucket {
                owner_rank,
                tokens: Vec::with_capacity(count),
            })
            .collect::<Vec<_>>();
        for (index, route) in self.routes.iter().enumerate() {
            let source_row = index / self.top_k;
            let start = source_row * self.width;
            buckets[route.owner_index].tokens.push(ExpertToken {
                transaction: self.context.transaction,
                sequence: route.sequence,
                source_rank: self.context.source_rank,
                source_row,
                route_slot: index % self.top_k,
                expert: route.expert,
                weight: route.weight,
                payload: payload[start..start + self.width].to_vec(),
            });
        }
        check_active(self.context.transaction)?;
        Ok(buckets)
    }

    /// Validate a complete received owner bucket before calling a worker. A
    /// transport may change arrival order, but cannot change route identity,
    /// owner, weight or shape, duplicate tokens, or omit work (including skew).
    pub fn validate_bucket(&self, bucket: &ExpertTokenBucket) -> Result<()> {
        let owner_index = self.owner_index(bucket.owner_rank)?;
        if bucket.tokens.len() != self.owner_counts[owner_index] {
            return Err(protocol_error("owner bucket token count mismatch"));
        }
        let mut seen = BTreeSet::new();
        for token in &bucket.tokens {
            self.validate_context(token.transaction, token.source_rank)?;
            let route = self.route(token.source_row, token.route_slot)?;
            if token.sequence != route.sequence {
                return Err(protocol_error("stale token sequence"));
            }
            if route.owner_index != owner_index || token.expert != route.expert {
                return Err(protocol_error("unknown token expert or wrong owner"));
            }
            if !seen.insert((token.source_row, token.route_slot)) {
                return Err(protocol_error("duplicate token"));
            }
            if token.weight.to_bits() != route.weight.to_bits() {
                return Err(protocol_error("token route weight mismatch"));
            }
            validate_values(&token.payload, self.width, "token payload")?;
        }
        Ok(())
    }

    /// Owner-local execution or a synchronous transport adapter. Empty buckets
    /// never invoke the worker/provider. Failed/cancelled calls return no partial
    /// result set; other owners' work has no publication semantics on its own.
    pub fn execute_bucket(
        &self,
        bucket: &ExpertTokenBucket,
        worker: &mut dyn ExpertWorker,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<Vec<ExpertResult>> {
        check_active(self.context.transaction)?;
        let routes = self.route_view(bucket)?;
        if worker.owner() != bucket.owner_rank {
            return Err(protocol_error("worker has the wrong owner rank"));
        }
        let routed_results = if routes.is_empty() {
            Vec::new()
        } else {
            worker.compute_routes(&routes)?
        };
        self.ordered_route_results(&routed_results, Some(bucket.owner_rank))?;
        check_active(self.context.transaction)?;
        Ok(routed_results
            .into_iter()
            .map(|routed| routed.result)
            .collect())
    }

    /// Validate all replies before reducing, then apply each frozen weight once
    /// in `(source_row, route_slot)` order. Arrival order never affects F32
    /// summation order. This neither renormalizes nor re-runs the router.
    pub fn combine(
        &self,
        results: &[ExpertResult],
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<HostRows> {
        check_active(self.context.transaction)?;
        let ordered = self.ordered_results(results, None)?;
        let mut output = vec![0.0; checked_mul(self.rows, self.width)?];
        for (route, result) in self.routes.iter().zip(ordered) {
            let start = result.source_row * self.width;
            for (target, value) in output[start..start + self.width]
                .iter_mut()
                .zip(&result.output)
            {
                *target += route.weight * value;
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(protocol_error("combined output is non-finite"));
        }
        check_active(self.context.transaction)?;
        HostRows::new(
            RowsShape::new(self.rows, self.width)?,
            RowsDType::F32,
            None,
            output,
        )
    }

    /// Admit all owner buckets, then authenticate replies independently of
    /// arrival order before allowing the immutable plan to combine them.
    pub(crate) fn execute_results(
        &self,
        executor: &mut dyn ExpertResultExecutor,
        buckets: Vec<ExpertTokenBucket>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<Vec<ExpertResult>> {
        check_active(self.context.transaction)?;
        if !executor.supports_batch() {
            let mut results = Vec::with_capacity(self.token_count());
            for bucket in buckets {
                check_active(self.context.transaction)?;
                let owner = bucket.owner_rank;
                let reply = executor.execute(bucket)?;
                check_active(self.context.transaction)?;
                self.ordered_results(&reply, Some(owner))?;
                results.extend(reply);
            }
            return Ok(results);
        }
        let mut expected = buckets
            .iter()
            .map(|b| b.owner_rank)
            .collect::<BTreeSet<_>>();
        let replies = executor.execute_batch(buckets)?;
        check_active(self.context.transaction)?;
        let mut results = Vec::with_capacity(self.token_count());
        for (owner, reply) in replies {
            if !expected.remove(&owner) {
                return Err(protocol_error("duplicate or unexpected batch owner"));
            }
            self.ordered_results(&reply, Some(owner))?;
            results.extend(reply);
        }
        if !expected.is_empty() {
            return Err(protocol_error("missing batch owner"));
        }
        Ok(results)
    }

    fn owner_index(&self, owner: ParallelRankId) -> Result<usize> {
        self.members
            .iter()
            .position(|&member| member == owner)
            .ok_or_else(|| protocol_error("unknown owner rank"))
    }

    fn route(&self, row: usize, slot: usize) -> Result<&ExpectedRoute> {
        if row >= self.rows || slot >= self.top_k {
            return Err(protocol_error("unknown row or route slot"));
        }
        Ok(&self.routes[row * self.top_k + slot])
    }

    fn validate_context(
        &self,
        transaction: ExecutionTransactionId,
        source_rank: ParallelRankId,
    ) -> Result<()> {
        if transaction != self.context.transaction {
            return Err(protocol_error("stale transaction"));
        }
        if source_rank != self.context.source_rank {
            return Err(protocol_error("unknown source rank"));
        }
        Ok(())
    }

    fn ordered_route_results<'a>(
        &self,
        results: &'a [ExpertRouteResult],
        owner: Option<ParallelRankId>,
    ) -> Result<Vec<&'a ExpertRouteResult>> {
        let expected_count = match owner {
            Some(owner) => self.owner_counts[self.owner_index(owner)?],
            None => self.routes.len(),
        };
        if results.len() != expected_count {
            return Err(protocol_error("incomplete or excess route result set"));
        }
        for routed in results {
            let result = &routed.result;
            let route = self.route(result.source_row, result.route_slot)?;
            self.validate_context(result.transaction, result.source_rank)?;
            if result.sequence != route.sequence {
                return Err(protocol_error("stale result sequence"));
            }
            if result.expert != route.expert
                || result.owner_rank != self.members[route.owner_index]
                || owner.is_some_and(|owner| owner != result.owner_rank)
            {
                return Err(protocol_error("unknown result expert or wrong owner"));
            }
            self.validate_route_metadata(&routed.metadata)?;
            if routed.metadata != route.metadata {
                return Err(protocol_error("route result metadata mismatch"));
            }
            validate_values(&result.output, self.width, "expert output")?;
        }
        let mut ordered = results.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|routed| (routed.result.source_row, routed.result.route_slot));
        if ordered.windows(2).any(|pair| {
            (pair[0].result.source_row, pair[0].result.route_slot)
                == (pair[1].result.source_row, pair[1].result.route_slot)
        }) {
            return Err(protocol_error("duplicate route result"));
        }
        Ok(ordered)
    }

    pub(crate) fn ordered_results<'a>(
        &self,
        results: &'a [ExpertResult],
        owner: Option<ParallelRankId>,
    ) -> Result<Vec<&'a ExpertResult>> {
        let expected_count = match owner {
            Some(owner) => self.owner_counts[self.owner_index(owner)?],
            None => self.routes.len(),
        };
        // Reject oversized peer replies before allocating a sorting index.
        if results.len() != expected_count {
            return Err(protocol_error("incomplete or excess result set"));
        }
        for result in results {
            self.validate_context(result.transaction, result.source_rank)?;
            let route = self.route(result.source_row, result.route_slot)?;
            if result.sequence != route.sequence {
                return Err(protocol_error("stale result sequence"));
            }
            if result.expert != route.expert
                || result.owner_rank != self.members[route.owner_index]
                || owner.is_some_and(|owner| owner != result.owner_rank)
            {
                return Err(protocol_error("unknown result expert or wrong owner"));
            }
            validate_values(&result.output, self.width, "expert output")?;
        }
        let mut ordered = results.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|result| (result.source_row, result.route_slot));
        if ordered.windows(2).any(|pair| {
            (pair[0].source_row, pair[0].route_slot) == (pair[1].source_row, pair[1].route_slot)
        }) {
            return Err(protocol_error("duplicate result"));
        }
        Ok(ordered)
    }
}

/// Owner boundary for CPU reference or future IPC/HostCollective adapters.
/// Return only **unweighted expert outputs**, echoing every token identity.
/// Call through `execute_bucket` to enforce the plan and external liveness gate;
/// network receivers must authenticate the responding owner, not trust its tag.
pub trait ExpertWorker {
    fn owner(&self) -> ParallelRankId;
    fn compute(&mut self, tokens: &[ExpertToken]) -> Result<Vec<ExpertResult>>;

    /// Compatibility bridge for existing workers. Legacy workers can execute
    /// fixed single-replica routes, but may not silently collapse replica N to
    /// logical replica zero. Replica-aware workers must override this method.
    fn compute_routes(&mut self, routes: &[ExpertRouteToken]) -> Result<Vec<ExpertRouteResult>> {
        if routes.iter().any(|route| route.metadata.replica != 0) {
            return Err(unsupported_route_error(
                "worker does not accept replicated physical slots",
            ));
        }
        let tokens = routes
            .iter()
            .map(|route| route.token.clone())
            .collect::<Vec<_>>();
        let results = self.compute(&tokens)?;
        if results.len() != routes.len() {
            return Err(protocol_error("route worker result count mismatch"));
        }
        Ok(results
            .into_iter()
            .zip(routes)
            .map(|(result, route)| ExpertRouteResult {
                metadata: route.metadata,
                result,
            })
            .collect())
    }
}

/// Synchronous owner multiplexer for local CPU workers. Workers/providers are
/// borrowed, not copied; only token buckets and computed results cross the seam.
/// A threaded/IPC implementation can instead implement `ExpertResultExecutor`
/// directly and construct the same CPU workers inside each owner thread.
pub struct CpuExpertResultExecutor<'a> {
    workers: BTreeMap<ParallelRankId, &'a mut dyn ExpertWorker>,
}

impl<'a> CpuExpertResultExecutor<'a> {
    pub fn new(workers: Vec<&'a mut dyn ExpertWorker>) -> Result<Self> {
        let mut by_owner = BTreeMap::new();
        for worker in workers {
            if by_owner.insert(worker.owner(), worker).is_some() {
                return Err(protocol_error("duplicate worker owner"));
            }
        }
        Ok(Self { workers: by_owner })
    }
}

impl ExpertResultExecutor for CpuExpertResultExecutor<'_> {
    fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
        if bucket.tokens.is_empty() {
            return Ok(Vec::new());
        }
        let worker = self.workers.get_mut(&bucket.owner_rank).ok_or_else(|| {
            protocol_error(format!(
                "missing worker for owner {}",
                bucket.owner_rank.get()
            ))
        })?;
        worker.compute(&bucket.tokens)
    }
}

/// Production routed-FFN adapter: existing router outputs -> owner execution ->
/// validated combine. Contains placement/transport configuration, not decoder
/// forward, weights, sequence cursors or transaction state. Group and per-call
/// admission checks happen before any bucket is sent.
///
/// `new` retains the legacy transport boundary. `new_versioned` is the
/// route-aware CPU/reference boundary and must be entered through
/// `execute_versioned`; it never sends a versioned bucket through the legacy
/// logical-only `ExpertResultExecutor`.
enum ExpertParallelExecution<'a> {
    Legacy(&'a mut dyn ExpertResultExecutor),
    Versioned(&'a mut dyn ExpertWorker),
}

pub struct ExpertParallelRoutedExecutor<'a> {
    members: Vec<ParallelRankId>,
    placement: &'a ExpertPlacement,
    limits: ExpertDispatchLimits,
    execution: ExpertParallelExecution<'a>,
}

impl<'a> ExpertParallelRoutedExecutor<'a> {
    pub fn new(
        members: Vec<ParallelRankId>,
        placement: &'a ExpertPlacement,
        limits: ExpertDispatchLimits,
        results: &'a mut dyn ExpertResultExecutor,
    ) -> Self {
        Self {
            members,
            placement,
            limits,
            execution: ExpertParallelExecution::Legacy(results),
        }
    }

    /// Construct the route-aware production adapter over an owner-local worker.
    /// The worker receives authenticated physical-slot route views through
    /// `execute_versioned`; it is never reached through the legacy result
    /// transport.
    pub fn new_versioned(
        members: Vec<ParallelRankId>,
        placement: &'a ExpertPlacement,
        limits: ExpertDispatchLimits,
        worker: &'a mut dyn ExpertWorker,
    ) -> Self {
        Self {
            members,
            placement,
            limits,
            execution: ExpertParallelExecution::Versioned(worker),
        }
    }

    /// Execute a routed request using explicit, versioned route metadata.
    /// `new_with_route_metadata` validation happens before dispatch; each bucket
    /// then goes through `ExpertDispatchPlan::execute_bucket`, preserving the
    /// existing provider/residency and transaction/cancellation boundaries.
    pub fn execute_versioned(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        route_metadata: &[ExpertRouteMetadata],
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        let worker = match &mut self.execution {
            ExpertParallelExecution::Versioned(worker) => worker,
            ExpertParallelExecution::Legacy(_) => {
                return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                    "expert_parallel_routed_swiglu",
                    "versioned execution requires new_versioned and execute_versioned",
                )));
            }
        };
        check_active(request.context.transaction)?;
        let input = request.input.host()?;
        if input.dtype() != RowsDType::F32 {
            return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                "expert_parallel_routed_swiglu",
                "the CPU EP executor requires F32 activations",
            )));
        }
        if request.routes.rows() != input.shape().rows() {
            return Err(protocol_error("routed input row count mismatch"));
        }
        let plan = ExpertDispatchPlan::new_with_route_metadata(
            request.context,
            self.members.clone(),
            self.placement,
            request.routes,
            route_metadata,
            request.sequences,
            input.shape().width(),
            self.limits,
        )?;
        let buckets = plan.dispatch(input.values(), check_active)?;
        let mut results = Vec::with_capacity(plan.token_count());
        for bucket in buckets {
            // The versioned constructor binds one owner-local worker. Empty
            // participant buckets remain part of dispatch accounting but do
            // not require a worker/provider call.
            if bucket.tokens.is_empty() {
                continue;
            }
            results.extend(plan.execute_bucket(&bucket, *worker, check_active)?);
        }
        let output = plan.combine(&results, check_active)?;
        Ok(OperatorProgress::Ready(Rows::Host(HostRows::new(
            output.shape(),
            RowsDType::F32,
            request.arena,
            output.into_values(),
        )?)))
    }
}

impl RoutedSwiGluExecutor for ExpertParallelRoutedExecutor<'_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        let results_executor = match &mut self.execution {
            ExpertParallelExecution::Legacy(results) => results,
            ExpertParallelExecution::Versioned(_) => {
                return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                    "expert_parallel_routed_swiglu",
                    "versioned executor requires execute_versioned",
                )));
            }
        };
        check_active(request.context.transaction)?;
        let input = request.input.host()?;
        if input.dtype() != RowsDType::F32 {
            return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                "expert_parallel_routed_swiglu",
                "the CPU EP executor requires F32 activations",
            )));
        }
        if request.routes.rows() != input.shape().rows() {
            return Err(protocol_error("routed input row count mismatch"));
        }
        let plan = ExpertDispatchPlan::new(
            request.context,
            self.members.clone(),
            self.placement,
            request.routes,
            request.sequences,
            input.shape().width(),
            self.limits,
        )?;
        let buckets = plan.dispatch(input.values(), check_active)?;
        let results = plan.execute_results(*results_executor, buckets, check_active)?;
        let output = plan.combine(&results, check_active)?;
        Ok(OperatorProgress::Ready(Rows::Host(HostRows::new(
            output.shape(),
            RowsDType::F32,
            request.arena,
            output.into_values(),
        )?)))
    }
}

/// F32 CPU reference adapter over existing owner-local prepared experts.
/// The physical slot selects the authenticated owner/route group, but the
/// existing provider still resolves a logical expert. This seam therefore
/// does not claim migrated weights or independently materialized physical
/// copies; storage, residency, and weight authority remain unchanged.
/// Waiting/Unsupported returns an error (no polling, retries or partial
/// publication in this minimal protocol).
pub struct CpuReferenceExpertWorker<'a> {
    owner: ParallelRankId,
    placement: &'a ExpertPlacement,
    experts: &'a mut dyn ExpertProvider,
    operators: CpuStandardDecoderOperators,
}

impl<'a> CpuReferenceExpertWorker<'a> {
    pub fn new(
        owner: ParallelRankId,
        placement: &'a ExpertPlacement,
        experts: &'a mut dyn ExpertProvider,
    ) -> Self {
        Self {
            owner,
            placement,
            experts,
            operators: CpuStandardDecoderOperators::default(),
        }
    }
}

impl CpuReferenceExpertWorker<'_> {
    fn validate_route(&self, route: &ExpertRouteToken) -> Result<()> {
        let metadata = route.metadata;
        if metadata.placement_epoch != self.placement.epoch() {
            if metadata.placement_epoch < self.placement.epoch() {
                return Err(protocol_error("stale route placement epoch at CPU worker"));
            }
            return Err(protocol_error(
                "route placement epoch does not match CPU worker placement",
            ));
        }
        if metadata.logical_expert != route.token.expert {
            return Err(protocol_error("route token logical expert mismatch"));
        }
        if self.placement.replicas(metadata.logical_expert).is_none() {
            return Err(protocol_error("unknown expert logical id at CPU worker"));
        }
        let entry = self
            .placement
            .entry(metadata.logical_expert, metadata.physical_slot)
            .ok_or_else(|| protocol_error("unknown physical expert slot at CPU worker"))?;
        if entry.replica != metadata.replica {
            return Err(protocol_error("route replica mismatch at CPU worker"));
        }
        if entry.owner_rank != metadata.owner_rank || metadata.owner_rank != self.owner {
            return Err(protocol_error("route owner mismatch at CPU worker"));
        }
        Ok(())
    }
}

impl ExpertWorker for CpuReferenceExpertWorker<'_> {
    fn owner(&self) -> ParallelRankId {
        self.owner
    }

    fn compute_routes(&mut self, routes: &[ExpertRouteToken]) -> Result<Vec<ExpertRouteResult>> {
        let mut groups = BTreeMap::<(ExpertId, usize), Vec<&ExpertRouteToken>>::new();
        for route in routes {
            self.validate_route(route)?;
            groups
                .entry((route.metadata.logical_expert, route.metadata.physical_slot))
                .or_default()
                .push(route);
        }
        let mut results = Vec::with_capacity(routes.len());
        for ((expert, _physical_slot), routes) in groups {
            let prepared = match self.experts.expert(expert.layer, expert.expert)? {
                ExpertAvailability::Ready(prepared) => prepared,
                ExpertAvailability::Waiting => {
                    return Err(protocol_error(format!("expert {expert:?} is waiting")));
                }
                ExpertAvailability::Unsupported(reason) => {
                    return Err(protocol_error(format!(
                        "expert {expert:?} is unsupported: {reason}"
                    )));
                }
            };
            let width = prepared.input_width();
            if prepared.output_width() != width {
                return Err(protocol_error("expert must preserve activation width"));
            }
            for route in &routes {
                validate_values(&route.token.payload, width, "CPU expert input")?;
            }
            let shape = RowsShape::new(routes.len(), width)?;
            let values = routes
                .iter()
                .flat_map(|route| route.token.payload.iter().copied())
                .collect();
            let input = Rows::Host(HostRows::new(shape, RowsDType::F32, None, values)?);
            let output = self
                .operators
                .apply_prepared_swiglu(&prepared, &input, None)?
                .into_host()?;
            if output.shape() != shape {
                return Err(protocol_error("expert output shape mismatch"));
            }
            for (route, row) in routes.into_iter().zip(output.values().chunks_exact(width)) {
                results.push(ExpertRouteResult {
                    metadata: route.metadata,
                    result: ExpertResult::from_token(&route.token, self.owner, row.to_vec()),
                });
            }
        }
        Ok(results)
    }

    fn compute(&mut self, tokens: &[ExpertToken]) -> Result<Vec<ExpertResult>> {
        let mut groups = BTreeMap::<ExpertId, Vec<&ExpertToken>>::new();
        for token in tokens {
            if self
                .placement
                .owner(token.expert.layer, token.expert.expert)
                != Some(self.owner)
            {
                return Err(protocol_error("CPU worker received a non-local expert"));
            }
            groups.entry(token.expert).or_default().push(token);
        }
        let mut results = Vec::with_capacity(tokens.len());
        for (expert, tokens) in groups {
            let prepared = match self.experts.expert(expert.layer, expert.expert)? {
                ExpertAvailability::Ready(prepared) => prepared,
                ExpertAvailability::Waiting => {
                    return Err(protocol_error(format!("expert {expert:?} is waiting")));
                }
                ExpertAvailability::Unsupported(reason) => {
                    return Err(protocol_error(format!(
                        "expert {expert:?} is unsupported: {reason}"
                    )));
                }
            };
            let width = prepared.input_width();
            if prepared.output_width() != width {
                return Err(protocol_error("expert must preserve activation width"));
            }
            for token in &tokens {
                validate_values(&token.payload, width, "CPU expert input")?;
            }
            let shape = RowsShape::new(tokens.len(), width)?;
            let values = tokens
                .iter()
                .flat_map(|token| token.payload.iter().copied())
                .collect();
            let input = Rows::Host(HostRows::new(shape, RowsDType::F32, None, values)?);
            let output = self
                .operators
                .apply_prepared_swiglu(&prepared, &input, None)?
                .into_host()?;
            if output.shape() != shape {
                return Err(protocol_error("CPU expert output shape mismatch"));
            }
            for (token, row) in tokens.into_iter().zip(output.values().chunks_exact(width)) {
                results.push(ExpertResult::from_token(token, self.owner, row.to_vec()));
            }
        }
        Ok(results)
    }
}

fn validate_values(values: &[f32], width: usize, label: &str) -> Result<()> {
    if values.len() != width {
        return Err(protocol_error(format!("{label} width/length mismatch")));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(protocol_error(format!(
            "{label} contains non-finite values"
        )));
    }
    Ok(())
}

fn checked_mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| protocol_error("dispatch size overflow"))
}

fn unsupported_route_error(message: impl Into<String>) -> Error {
    Error::ModelSource {
        source: Box::new(UnsupportedOperator::new(
            "expert_parallel_route",
            format!("unsupported route execution: {}", message.into()),
        )),
    }
}

fn protocol_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("expert parallel: {}", message.into()),
    }
}
