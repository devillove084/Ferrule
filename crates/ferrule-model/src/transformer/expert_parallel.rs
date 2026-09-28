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

/// Immutable `(layer, expert) -> owner rank` mapping. No weight data is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPlacement {
    owners: BTreeMap<ExpertId, ParallelRankId>,
}

impl ExpertPlacement {
    pub fn new(entries: impl IntoIterator<Item = (usize, usize, ParallelRankId)>) -> Result<Self> {
        let mut owners = BTreeMap::new();
        for (layer, expert, owner) in entries {
            if owners.insert(ExpertId::new(layer, expert), owner).is_some() {
                return Err(protocol_error(format!(
                    "duplicate placement for expert {layer}:{expert}"
                )));
            }
        }
        Ok(Self { owners })
    }

    pub fn owner(&self, layer: usize, expert: usize) -> Option<ParallelRankId> {
        self.owners.get(&ExpertId::new(layer, expert)).copied()
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

/// Dispatch retains a bucket for every member, including owners with no work.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertTokenBucket {
    pub owner_rank: ParallelRankId,
    pub tokens: Vec<ExpertToken>,
}

#[derive(Debug)]
struct ExpectedRoute {
    expert: ExpertId,
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
    members: Vec<ParallelRankId>,
    owner_counts: Vec<usize>,
    routes: Vec<ExpectedRoute>,
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
        let mut owner_counts = vec![0; members.len()];
        let mut expected = Vec::with_capacity(token_count);
        for (row, &sequence) in sequences.iter().enumerate() {
            let (experts, weights) = routes.row(row)?;
            let mut seen = BTreeSet::new();
            for (&expert, &weight) in experts.iter().zip(weights) {
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
                let owner = placement.owner(context.layer, expert).ok_or_else(|| {
                    protocol_error(format!("unknown expert {}:{expert}", context.layer))
                })?;
                let owner_index = *member_indexes
                    .get(&owner)
                    .ok_or_else(|| protocol_error("unknown expert owner rank"))?;
                owner_counts[owner_index] += 1;
                expected.push(ExpectedRoute {
                    expert: ExpertId::new(context.layer, expert),
                    sequence,
                    owner_index,
                    weight,
                });
            }
        }
        Ok(Self {
            context,
            members,
            owner_counts,
            routes: expected,
            rows: routes.rows(),
            top_k: routes.top_k(),
            width,
            required_bytes,
        })
    }

    pub fn context(&self) -> ExpertDispatchContext {
        self.context
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
        self.validate_bucket(bucket)?;
        if worker.owner() != bucket.owner_rank {
            return Err(protocol_error("worker has the wrong owner rank"));
        }
        let results = if bucket.tokens.is_empty() {
            Vec::new()
        } else {
            worker.compute(&bucket.tokens)?
        };
        self.ordered_results(&results, Some(bucket.owner_rank))?;
        check_active(self.context.transaction)?;
        Ok(results)
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
/// The synchronous F32 implementation returns Ready or an error (Unsupported
/// for non-F32 rows). It has no asynchronous continuation or automatic retry.
/// Shared layer math should inject it in place of `operators.routed_swiglu`,
/// before shared-expert and residual additions, not around the whole forward.
pub struct ExpertParallelRoutedExecutor<'a> {
    members: Vec<ParallelRankId>,
    placement: &'a ExpertPlacement,
    limits: ExpertDispatchLimits,
    results: &'a mut dyn ExpertResultExecutor,
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
            results,
        }
    }
}

impl RoutedSwiGluExecutor for ExpertParallelRoutedExecutor<'_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
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
        let results = plan.execute_results(self.results, buckets, check_active)?;
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
/// Provider residency/streaming remains local; Waiting/Unsupported returns an
/// error (no polling, retries or partial publication in this minimal protocol).
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

impl ExpertWorker for CpuReferenceExpertWorker<'_> {
    fn owner(&self) -> ParallelRankId {
        self.owner
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

fn protocol_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("expert parallel: {}", message.into()),
    }
}
