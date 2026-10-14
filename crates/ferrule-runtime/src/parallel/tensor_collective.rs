//! Bounded model-independent host collective endpoints for decoder execution.
//! Control handles abort/wake host waiters; neither wakeup nor host completion
//! proves GPU safety.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, ParallelGroupId, ParallelRankId, ParallelTopologyId, Result};
use ferrule_model::transformer::hybrid_parallel::DevicePayloadIdentity;
use ferrule_model::transformer::parallel::TensorParallelCollective;
use ferrule_model::transformer::{StandardTensorCollective, StandardTensorPlan};

use crate::parallel::collective::{
    CollectivePayload, HostCollectiveDescriptor, HostCollectiveGroup, HostCollectiveKind,
    HostCollectiveLimits, HostCollectiveStatus, TypedCollectivePoll,
};
use crate::parallel::pipeline::tensor_scope::{TensorBatchIdentity, TensorCollectiveSite};

#[path = "mixed_rank_collective.rs"]
mod mixed_rank;
pub use mixed_rank::{
    MixedCollectivePayloadBudget, MixedRankCollective, MixedRankCollectivePlan,
    MixedRouteControlBudget,
};

struct State {
    group: HostCollectiveGroup,
    site: Option<u64>,
    failure: Option<String>,
    singleton_aborted: bool,
    closed: Vec<bool>,
    execution: Option<(ExecutionTransactionId, Arc<AtomicBool>)>,
    execution_watermark: Option<ExecutionTransactionId>,
    sites: Option<Vec<TensorCollectiveSite>>,
    mixed_plan: Option<Arc<MixedRankCollectivePlan>>,
    registered: Vec<bool>,
    identity: Option<TensorBatchIdentity>,
    validated: Vec<bool>,
    completed: Vec<usize>,
    waiters: usize,
    routes: Option<mixed_rank::PendingRoutes>,
    routes_completed: Vec<usize>,
    device_votes: Vec<Option<DevicePayloadIdentity>>,
    device_permissions: Vec<Option<DevicePayloadIdentity>>,
    device_uncompleted: Vec<bool>,
    device_unknown: bool,
}
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    timeout: Duration,
    epoch: ParallelTopologyId,
}

#[derive(Clone)]
pub struct DecoderTensorCollectiveControl(Arc<Shared>);
impl DecoderTensorCollectiveControl {
    /// Register the immutable operator schedule on each owner during startup.
    /// In a formal pipeline, all owners must agree before a cohort may begin.
    pub fn register_sites(
        &self,
        owner: ParallelRankId,
        sites: Vec<TensorCollectiveSite>,
    ) -> Result<()> {
        self.register_sites_inner(owner, sites, None)
    }
    fn register_sites_inner(
        &self,
        owner: ParallelRankId,
        sites: Vec<TensorCollectiveSite>,
        mixed_plan: Option<Arc<MixedRankCollectivePlan>>,
    ) -> Result<()> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        let index = state.group.members().iter().position(|rank| *rank == owner);
        if index.is_none()
            || (sites.is_empty() && state.group.members().len() != 1)
            || sites
                .iter()
                .any(|site| site.validate(state.group.members().len()).is_err())
            || state.execution_watermark.is_some()
            || (state.sites.is_some() && state.mixed_plan != mixed_plan)
            || state.failure.is_some()
            || index.is_some_and(|index| state.registered[index])
            || state
                .sites
                .as_ref()
                .is_some_and(|expected| expected != &sites)
        {
            fail(
                &mut state,
                "invalid or mismatched tensor operator schedule".into(),
            );
            self.0.wake.notify_all();
            return Err(error("invalid or mismatched tensor operator schedule"));
        }
        state.mixed_plan = mixed_plan;
        state.sites = Some(sites);
        state.registered[index.expect("checked owner")] = true;
        Ok(())
    }

    /// Legacy collective-only admission. Once a stage registers a schedule it
    /// MUST use `begin_sealed`; the formal pipeline cannot bypass packed identity.
    pub fn begin(
        &self,
        transaction: ExecutionTransactionId,
        cancellation: Arc<AtomicBool>,
    ) -> Result<()> {
        self.admit(transaction, cancellation, None)
    }
    pub fn begin_sealed(
        &self,
        identity: TensorBatchIdentity,
        cancellation: Arc<AtomicBool>,
    ) -> Result<()> {
        self.admit(
            identity.binding().transaction(),
            cancellation,
            Some(identity),
        )
    }
    fn admit(
        &self,
        transaction: ExecutionTransactionId,
        cancellation: Arc<AtomicBool>,
        identity: Option<TensorBatchIdentity>,
    ) -> Result<()> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        mixed_rank::check_open(&mut state)?;
        if state.failure.is_some()
            || state.closed.iter().any(|closed| *closed)
            || state.group.active_descriptor().is_some()
            || state.execution.is_some()
            || state.routes.is_some()
            || state
                .execution_watermark
                .is_some_and(|previous| previous >= transaction)
        {
            return Err(error(
                "collective stage is failed, busy or has a stale transaction",
            ));
        }
        if state.sites.is_some() != identity.is_some()
            || (identity.is_some() && state.registered.iter().any(|registered| !registered))
            || identity.as_ref().is_some_and(|identity| {
                identity.binding().topology_id() != self.0.epoch
                    || state
                        .group
                        .members()
                        .iter()
                        .any(|rank| !identity.binding().participants().contains(*rank))
            })
        {
            return Err(error(
                "collective requires its sealed packed identity and complete owner schedule",
            ));
        }
        if let (Some(plan), Some(identity)) = (&state.mixed_plan, &identity)
            && (identity.rows() > plan.max_rows() || identity.sequences() > plan.max_sequences())
        {
            return Err(error(
                "packed cohort exceeds sealed Mixed collective bounds",
            ));
        }
        if cancellation.load(Ordering::Acquire) {
            return Err(error("collective cancelled before admission"));
        }
        if let (Some(sites), Some(identity)) = (&state.sites, &identity) {
            for site in sites {
                state
                    .group
                    .memory_for(
                        site.kind(),
                        site.elements(identity.sequences(), identity.rows())?,
                    )
                    .map_err(error)?;
            }
        }
        state.execution = Some((transaction, cancellation));
        state.execution_watermark = Some(transaction);
        state.identity = identity;
        state.validated.fill(false);
        state.completed.fill(0);
        state.device_votes.fill(None);
        state.device_permissions.fill(None);
        state.device_uncompleted.fill(false);
        state.device_unknown = false;
        state.routes_completed.fill(0);
        Ok(())
    }

    /// Called against the actual owner packed batch before physical enter.
    /// A foreign phase/layout poisons the group and wakes peers already waiting.
    pub fn validate_owner_execution(
        &self,
        owner: ParallelRankId,
        identity: &TensorBatchIdentity,
    ) -> Result<()> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        mixed_rank::check_open(&mut state)?;
        let index = state.group.members().iter().position(|rank| *rank == owner);
        if state.identity.as_ref() == Some(identity)
            && index.is_some_and(|index| !state.validated[index])
            && state
                .execution
                .as_ref()
                .is_some_and(|(_, flag)| flag.load(Ordering::Acquire))
        {
            fail_step(&mut state, "singleton owner cancelled before enter".into());
            self.0.wake.notify_all();
            return Err(error("owner cancelled before enter"));
        }
        if state.failure.is_some()
            || state
                .execution
                .as_ref()
                .is_none_or(|(_, flag)| flag.load(Ordering::Acquire))
            || state.identity.as_ref() != Some(identity)
            || index.is_none()
            || index.is_some_and(|index| state.validated[index])
        {
            fail(
                &mut state,
                "owner packed batch differs from sealed TP cohort".into(),
            );
            self.0.wake.notify_all();
            return Err(error("owner packed batch differs from sealed TP cohort"));
        }
        state.validated[index.expect("checked owner")] = true;
        self.0.wake.notify_all();
        Ok(())
    }
    /// Recheck an already validated owner at the decorator's execute boundary.
    /// This never admits an owner or substitutes for the journal's pre-enter check.
    pub(crate) fn check_owner_execution(
        &self,
        owner: ParallelRankId,
        transaction: ExecutionTransactionId,
        batch: &ferrule_model::decoder::PackedDecoderBatch,
        sessions: &[u64],
    ) -> Result<()> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        mixed_rank::check_open(&mut state)?;
        let result = (|| {
            let index = state.group.members().iter().position(|rank| *rank == owner);
            let identity = state
                .identity
                .as_ref()
                .ok_or_else(|| error("no sealed TP execution"))?;
            if state.failure.is_some()
                || state
                    .execution
                    .as_ref()
                    .is_none_or(|(id, flag)| *id != transaction || flag.load(Ordering::Acquire))
                || index.is_none_or(|index| !state.validated[index])
                || TensorBatchIdentity::from_packed(identity.binding(), batch, sessions)?
                    != *identity
            {
                return Err(error(
                    "owner execution differs from validated sealed TP cohort",
                ));
            }
            Ok(())
        })();
        if let Err(source) = &result {
            if state
                .execution
                .as_ref()
                .is_some_and(|(_, flag)| flag.load(Ordering::Acquire))
            {
                fail_step(&mut state, source.to_string());
            } else {
                fail(&mut state, source.to_string());
            }
            self.0.wake.notify_all();
        }
        result
    }

    /// Local schedule completion is not a KV ACK or device fence. The parent
    /// still calls `finish` after collecting every owner's execution reply.
    pub(crate) fn check_owner_completion(
        &self,
        owner: ParallelRankId,
        transaction: ExecutionTransactionId,
    ) -> Result<()> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        mixed_rank::check_open(&mut state)?;
        let index = state.group.members().iter().position(|rank| *rank == owner);
        if index
            .is_some_and(|i| state.device_permissions[i].is_some() || state.device_uncompleted[i])
        {
            fail(&mut state, "owner lacks device consumer completion".into());
            self.0.wake.notify_all();
            mixed_rank::check_open(&mut state)?;
            return Err(error("owner lacks device consumer completion"));
        }

        if state.failure.is_some()
            || state.mixed_plan.as_ref().is_some_and(|plan| {
                index.is_none_or(|i| state.routes_completed[i] != plan.route_sites().len())
            })
            || state
                .execution
                .as_ref()
                .is_none_or(|(id, flag)| *id != transaction || flag.load(Ordering::Acquire))
            || state.sites.as_ref().is_none_or(|sites| {
                index.is_none_or(|index| {
                    !state.validated[index] || state.completed[index] != sites.len()
                })
            })
        {
            fail_step(
                &mut state,
                "owner returned before sealed schedule completion".into(),
            );
            self.0.wake.notify_all();
            return Err(error("owner returned before sealed schedule completion"));
        }
        Ok(())
    }

    pub fn finish(&self, transaction: ExecutionTransactionId) -> Result<()> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        mixed_rank::check_open(&mut state)?;
        if state.device_permissions.iter().any(Option::is_some)
            || state.device_uncompleted.iter().any(|pending| *pending)
        {
            fail(
                &mut state,
                "transaction lacks device consumer completion".into(),
            );
            self.0.wake.notify_all();
            mixed_rank::check_open(&mut state)?;
            return Err(error("transaction lacks device consumer completion"));
        }

        if state
            .execution
            .as_ref()
            .is_none_or(|(active, flag)| *active != transaction || flag.load(Ordering::Acquire))
            || state.group.active_descriptor().is_some()
            || state.routes.is_some()
            || state.mixed_plan.as_ref().is_some_and(|plan| {
                state
                    .routes_completed
                    .iter()
                    .any(|n| *n != plan.route_sites().len())
            })
            || state.failure.is_some()
            || state.sites.as_ref().is_some_and(|sites| {
                state.validated.iter().any(|validated| !validated)
                    || state.completed.iter().any(|count| *count != sites.len())
            })
        {
            return Err(error("collective transaction completion mismatch"));
        }
        state.execution = None;
        state.identity = None;
        state.validated.fill(false);
        self.0.wake.notify_all();
        Ok(())
    }

    /// Shared abort+wake: does not borrow an endpoint and cannot fabricate a KV
    /// ACK or a GPU fence. Stale aborts cannot affect a newer execution.
    pub fn abort(&self, transaction: ExecutionTransactionId, reason: &str) -> Result<()> {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some((active, flag)) = &state.execution else {
            return Err(error("no active collective execution"));
        };
        if *active != transaction {
            return Err(error("stale collective abort"));
        }
        flag.store(true, Ordering::Release);
        fail_step(&mut state, reason.into());
        // Hold the condition mutex across the predicate change and notification:
        // a waiter cannot miss an abort between checking and parking.
        self.0.wake.notify_all();
        Ok(())
    }
    pub(crate) fn fail_execution(&self, transaction: ExecutionTransactionId) {
        let _ = self.abort(transaction, "pipeline tensor execution failed");
    }
    /// Called only by the parent after every journal rollback ACK and logical
    /// retirement. Singleton empty schedules have no communication lifetime to
    /// poison; protocol errors and multi-member failures remain terminal.
    pub(crate) fn complete_singleton_rollback(
        &self,
        transaction: ExecutionTransactionId,
    ) -> Result<()> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        if !singleton_step(&state) || state.execution.is_none() {
            return Ok(());
        }
        if state
            .execution
            .as_ref()
            .is_none_or(|(active, _)| *active != transaction)
            || state.failure.is_some()
            || state.closed.iter().any(|closed| *closed)
            || state.group.active_descriptor().is_some()
            || state.waiters != 0
        {
            return Err(error("singleton rollback admission mismatch"));
        }
        state.execution = None;
        state.identity = None;
        state.validated.fill(false);
        state.completed.fill(0);
        state.singleton_aborted = false;
        self.0.wake.notify_all();
        Ok(())
    }
    /// Failed communication lifetime, or singleton step awaiting journal proof.
    pub fn is_failed(&self) -> bool {
        self.0.state.lock().map_or(true, |state| {
            state.failure.is_some() || state.singleton_aborted
        })
    }
    /// Host waiters only. Useful for observability/testing, never custody evidence.
    pub fn waiting_members(&self) -> usize {
        self.0
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .waiters
    }
}

/// Non-cloneable producer: exactly one endpoint for each ordered member.
pub struct DecoderTensorCollective {
    shared: Arc<Shared>,
    members: Arc<[ParallelRankId]>,
    local: usize,
    epoch: ParallelTopologyId,
    group: ParallelGroupId,
    sequence: u64,
}
impl DecoderTensorCollective {
    /// Construct an ordered, model-independent collective cohort. The input
    /// order is authoritative for owner indices and AllGather layout; sparse or
    /// non-sorted rank IDs are valid, while duplicates are rejected by the
    /// bounded host group. No model geometry or weight plan is inspected.
    /// Callers authenticate membership and assign unique epoch/group lifetimes;
    /// this constructor does not authorize a model or enable production TP.
    pub fn new_group_from_members(
        members: &[ParallelRankId],
        epoch: ParallelTopologyId,
        group: ParallelGroupId,
        limits: HostCollectiveLimits,
        timeout: Duration,
    ) -> Result<Vec<Self>> {
        if timeout.is_zero() || Instant::now().checked_add(timeout).is_none() {
            return Err(error("invalid collective timeout"));
        }
        if members.is_empty() || members.len() > limits.max_ranks {
            return Err(error("empty or oversized collective roster"));
        }
        let host =
            HostCollectiveGroup::new(epoch, group, members.to_vec(), limits).map_err(error)?;
        let members: Arc<[ParallelRankId]> = members.into();
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                group: host,
                site: None,
                failure: None,
                singleton_aborted: false,
                closed: vec![false; members.len()],
                execution: None,
                execution_watermark: None,
                sites: None,
                mixed_plan: None,
                identity: None,
                registered: vec![false; members.len()],
                validated: vec![false; members.len()],
                completed: vec![0; members.len()],
                waiters: 0,
                routes: None,
                routes_completed: vec![0; members.len()],
                device_votes: vec![None; members.len()],
                device_permissions: vec![None; members.len()],
                device_uncompleted: vec![false; members.len()],
                device_unknown: false,
            }),
            wake: Condvar::new(),
            timeout,
            epoch,
        });
        Ok((0..members.len())
            .map(|local| Self {
                shared: Arc::clone(&shared),
                members: Arc::clone(&members),
                local,
                epoch,
                group,
                sequence: 0,
            })
            .collect())
    }

    /// Dense compatibility wrapper. The dense plan remains responsible for
    /// validating dense model geometry; this endpoint constructor only consumes
    /// its already-validated ordered owner roster.
    pub fn new_group(
        plan: &StandardTensorPlan,
        epoch: ParallelTopologyId,
        group: ParallelGroupId,
        limits: HostCollectiveLimits,
        timeout: Duration,
    ) -> Result<Vec<Self>> {
        let members = plan
            .placements()
            .iter()
            .map(|placement| placement.owner)
            .collect::<Vec<_>>();
        Self::new_group_from_members(&members, epoch, group, limits, timeout)
    }

    pub fn control(&self) -> DecoderTensorCollectiveControl {
        DecoderTensorCollectiveControl(Arc::clone(&self.shared))
    }

    /// Thread-only typed transport through the existing sealed schedule and lifetime.
    /// Non-F32 operations require registration and exact packed owner validation.
    /// No IPC wire version or legacy model collective trait is changed.
    pub fn exchange_payload(
        &mut self,
        transaction: ExecutionTransactionId,
        site: u64,
        kind: HostCollectiveKind,
        payload: CollectivePayload,
    ) -> Result<CollectivePayload> {
        let result = self.exchange_inner(transaction, site, kind, payload, None);
        if let Err(source) = &result {
            let mut state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
            let device_entry = state.device_votes.iter().any(Option::is_some)
                || state.device_permissions.iter().any(Option::is_some)
                || state.device_uncompleted.iter().any(|pending| *pending);
            fail(&mut state, source.to_string());
            self.shared.wake.notify_all();
            if device_entry {
                mixed_rank::check_open(&mut state)?;
                return Err(error(state.failure.as_deref().expect("failed exchange")));
            }
        }
        result
    }

    fn exchange_inner(
        &mut self,
        transaction: ExecutionTransactionId,
        site: u64,
        kind: HostCollectiveKind,
        values: CollectivePayload,
        device_identity: Option<&DevicePayloadIdentity>,
    ) -> Result<CollectivePayload> {
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| error("collective sequence exhausted"))?;
        self.sequence = sequence;
        let descriptor = HostCollectiveDescriptor {
            epoch: self.epoch,
            group: self.group,
            transaction,
            sequence,
            kind,
            count: values.len(),
        };
        let deadline = Instant::now()
            .checked_add(self.shared.timeout)
            .ok_or_else(|| error("collective deadline overflow"))?;
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        let mut payload = Some(values);
        if !matches!(
            kind,
            HostCollectiveKind::AllReduceSumF32 | HostCollectiveKind::AllGatherF32
        ) && state.sites.is_none()
        {
            return Err(error(
                "typed collective requires a sealed precision schedule",
            ));
        }
        if state.execution.is_none() {
            return Err(error("collective exchange requires an active begin"));
        }
        loop {
            mixed_rank::check_open(&mut state)?;
            if let Some(failure) = &state.failure {
                return Err(error(failure));
            }
            if let Some((expected, cancellation)) = &state.execution
                && (*expected != transaction || cancellation.load(Ordering::Acquire))
            {
                return Err(error("collective transaction cancelled or mismatched"));
            }
            if let Some(sites) = &state.sites {
                let identity = state
                    .identity
                    .as_ref()
                    .ok_or_else(|| error("TP collective has no sealed packed identity"))?;
                if !state.validated[self.local] {
                    return Err(error("TP owner did not validate packed identity"));
                }
                let expected = sites
                    .get(state.completed[self.local])
                    .ok_or_else(|| error("tensor site schedule exhausted"))?;
                if expected.site != site
                    || expected.kind() != descriptor.kind
                    || expected.elements(identity.sequences(), identity.rows())?
                        != device_identity.map_or(descriptor.count, |vote| vote.elements)
                {
                    return Err(error("tensor operator site/kind/packed-row shape mismatch"));
                }
            }
            if state.device_permissions[self.local].is_some() {
                return Err(error("device site has not acknowledged enqueue"));
            }
            mixed_rank::check_data_route(&state, self.local, site)?;
            if let Some(identity) = device_identity
                && payload.is_none()
                && state.group.status() == Some(HostCollectiveStatus::Ready)
            {
                let mut expected = identity.clone();
                for (index, vote) in state.device_votes.iter().enumerate() {
                    expected.owner = self.members[index];
                    if vote.as_ref() != Some(&expected) {
                        return Err(error("device metadata votes differ"));
                    }
                }
            }

            if payload.is_none()
                && let TypedCollectivePoll::Ready {
                    descriptor: received,
                    payload,
                } = state.group.take_typed_result(self.owner()).map_err(error)?
            {
                if received != descriptor {
                    return Err(error("collective result identity mismatch"));
                }
                if let Some(identity) = device_identity {
                    state.device_permissions[self.local] = Some(identity.clone());
                    if state.group.active_descriptor().is_none() {
                        state.device_votes.fill(None);
                    }
                } else {
                    state.completed[self.local] += 1;
                }

                self.shared.wake.notify_all();
                return Ok(payload);
            }
            if state.closed.iter().any(|closed| *closed) {
                return Err(error("collective peer closed"));
            }
            // No shared computation may consume values until EVERY physical
            // owner has checked its actual packed batch against the sealed one.
            let admitted =
                state.sites.is_none() || state.validated.iter().all(|validated| *validated);
            if payload.is_some() && admitted {
                let active = state.group.active_descriptor();
                if active.is_none_or(|op| op.sequence == sequence) {
                    if active.is_some() && state.site != Some(site) {
                        return Err(error("collective decoder site mismatch"));
                    }
                    if let Some(identity) = device_identity {
                        state.device_votes[self.local] = Some(identity.clone());
                    }
                    state
                        .group
                        .submit_typed(
                            self.owner(),
                            descriptor,
                            payload.take().expect("not submitted"),
                        )
                        .map_err(|e| error(e.error))?;
                    state.site = Some(site);
                    self.shared.wake.notify_all();
                    continue;
                }
                if active.is_some_and(|op| op.sequence > sequence) {
                    return Err(error("stale decoder collective sequence"));
                }
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(error("decoder collective timed out"));
            }
            state.waiters += 1;
            let (next, _) = self
                .shared
                .wake
                .wait_timeout(state, remaining)
                .map_err(|_| error("collective mutex poisoned"))?;
            state = next;
            state.waiters -= 1;
        }
    }
}
impl StandardTensorCollective for DecoderTensorCollective {
    fn owner(&self) -> ParallelRankId {
        self.members[self.local]
    }
    fn members(&self) -> &[ParallelRankId] {
        &self.members
    }
    fn exchange(
        &mut self,
        transaction: ExecutionTransactionId,
        site: u64,
        kind: TensorParallelCollective,
        values: Vec<f32>,
    ) -> Result<Vec<f32>> {
        let kind = match kind {
            TensorParallelCollective::Sum => HostCollectiveKind::AllReduceSumF32,
            TensorParallelCollective::AllGather => HostCollectiveKind::AllGatherF32,
        };
        match self.exchange_payload(transaction, site, kind, CollectivePayload::F32(values))? {
            CollectivePayload::F32(values) => Ok(values),
            _ => Err(error("legacy collective received a non-F32 result")),
        }
    }
    fn abort(&mut self) {
        let mut state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        fail(&mut state, "decoder forward aborted".into());
        self.shared.wake.notify_all();
    }
}
impl Drop for DecoderTensorCollective {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        close_peer(&mut state, self.local);
        self.shared.wake.notify_all();
    }
}
fn close_peer(state: &mut State, local: usize) {
    state.closed[local] = true;
    // A prepared Mixed endpoint owns an execution lifetime, not just a host
    // rendezvous. Ready/drained payloads cannot prove the lost owner's quiescence.
    if (state.mixed_plan.is_some() && state.execution.is_some())
        || state.group.status() == Some(HostCollectiveStatus::Pending)
    {
        fail(state, "decoder collective peer dropped".into());
    }
}
fn singleton_step(state: &State) -> bool {
    state.group.members().len() == 1 && state.sites.as_ref().is_some_and(Vec::is_empty)
}
fn fail_step(state: &mut State, message: String) {
    if singleton_step(state) && state.execution.is_some() {
        state.singleton_aborted = true;
        if let Some((_, flag)) = &state.execution {
            flag.store(true, Ordering::Release);
        }
    } else {
        fail(state, message);
    }
}
fn fail(state: &mut State, message: String) {
    // Host abort never retires device obligations, including drained metadata.
    state.device_unknown |= state.device_permissions.iter().any(Option::is_some)
        || state.device_uncompleted.iter().any(|pending| *pending);

    state.routes = None;
    if state.failure.is_none() {
        state.failure = Some(message);
    }
    if let Some((_, flag)) = &state.execution {
        flag.store(true, Ordering::Release);
    }
    if let Some(active) = state.group.active_descriptor() {
        let _ = state.group.abort(active);
    }
}
#[cfg(test)]
#[path = "../../tests/unit/parallel/tensor_scope/collective.rs"]
mod tests;

fn error(message: impl std::fmt::Display) -> Error {
    Error::Model {
        message: format!("decoder TP collective: {message}"),
    }
}
