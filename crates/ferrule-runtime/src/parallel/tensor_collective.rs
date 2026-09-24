//! Bounded host collective endpoints for the standard decoder. Control handles
//! abort/wake host waiters; neither wakeup nor host completion proves GPU safety.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, ParallelGroupId, ParallelRankId, ParallelTopologyId, Result};
use ferrule_model::transformer::parallel::TensorParallelCollective;
use ferrule_model::transformer::{StandardTensorCollective, StandardTensorPlan};

use crate::parallel::collective::{
    HostCollectiveDescriptor, HostCollectiveGroup, HostCollectiveKind, HostCollectiveLimits,
    HostCollectivePoll, HostCollectiveStatus,
};
use crate::parallel::pipeline::tensor_scope::{TensorBatchIdentity, TensorCollectiveSite};

struct State {
    group: HostCollectiveGroup,
    site: Option<u64>,
    failure: Option<String>,
    closed: Vec<bool>,
    execution: Option<(ExecutionTransactionId, Arc<AtomicBool>)>,
    execution_watermark: Option<ExecutionTransactionId>,
    sites: Option<Vec<TensorCollectiveSite>>,
    registered: Vec<bool>,
    identity: Option<TensorBatchIdentity>,
    validated: Vec<bool>,
    completed: Vec<usize>,
    waiters: usize,
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
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        let index = state.group.members().iter().position(|rank| *rank == owner);
        if index.is_none()
            || sites.is_empty()
            || sites.iter().any(|site| site.elements_per_row() == 0)
            || state.execution_watermark.is_some()
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
        if state.failure.is_some()
            || state.closed.iter().any(|closed| *closed)
            || state.group.active_descriptor().is_some()
            || state.execution.is_some()
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
        state.execution = Some((transaction, cancellation));
        state.execution_watermark = Some(transaction);
        state.identity = identity;
        state.validated.fill(false);
        state.completed.fill(0);
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
        let index = state.group.members().iter().position(|rank| *rank == owner);
        if state.failure.is_some()
            || state.execution.is_none()
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
    pub fn finish(&self, transaction: ExecutionTransactionId) -> Result<()> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        if state
            .execution
            .as_ref()
            .is_none_or(|(active, _)| *active != transaction)
            || state.group.active_descriptor().is_some()
            || state.failure.is_some()
            || state
                .sites
                .as_ref()
                .is_some_and(|sites| state.completed.iter().any(|count| *count != sites.len()))
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
        fail(&mut state, reason.into());
        // Hold the condition mutex across the predicate change and notification:
        // a waiter cannot miss an abort between checking and parking.
        self.0.wake.notify_all();
        Ok(())
    }
    pub(crate) fn fail_execution(&self, transaction: ExecutionTransactionId) {
        let _ = self.abort(transaction, "pipeline tensor execution failed");
    }
    /// Terminal for this collective lifetime. Rollback and finish never clear it.
    pub fn is_failed(&self) -> bool {
        self.0
            .state
            .lock()
            .map_or(true, |state| state.failure.is_some())
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
    pub fn new_group(
        plan: &StandardTensorPlan,
        epoch: ParallelTopologyId,
        group: ParallelGroupId,
        limits: HostCollectiveLimits,
        timeout: Duration,
    ) -> Result<Vec<Self>> {
        if timeout.is_zero() || Instant::now().checked_add(timeout).is_none() {
            return Err(error("invalid collective timeout"));
        }
        let members: Arc<[ParallelRankId]> = plan.placements().iter().map(|p| p.owner).collect();
        let host =
            HostCollectiveGroup::new(epoch, group, members.to_vec(), limits).map_err(error)?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                group: host,
                site: None,
                failure: None,
                closed: vec![false; members.len()],
                execution: None,
                execution_watermark: None,
                sites: None,
                identity: None,
                registered: vec![false; members.len()],
                validated: vec![false; members.len()],
                completed: vec![0; members.len()],
                waiters: 0,
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
    pub fn control(&self) -> DecoderTensorCollectiveControl {
        DecoderTensorCollectiveControl(Arc::clone(&self.shared))
    }

    fn exchange_inner(
        &mut self,
        transaction: ExecutionTransactionId,
        site: u64,
        kind: TensorParallelCollective,
        values: Vec<f32>,
    ) -> Result<Vec<f32>> {
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
            kind: match kind {
                TensorParallelCollective::Sum => HostCollectiveKind::AllReduceSumF32,
                TensorParallelCollective::AllGather => HostCollectiveKind::AllGatherF32,
            },
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
        loop {
            if let Some(failure) = &state.failure {
                return Err(error(failure));
            }
            if let Some((expected, cancellation)) = &state.execution {
                if *expected != transaction || cancellation.load(Ordering::Acquire) {
                    return Err(error("collective transaction cancelled or mismatched"));
                }
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
                    || expected.elements_per_row().checked_mul(identity.rows())
                        != Some(descriptor.count)
                {
                    return Err(error("tensor operator site/kind/packed-row shape mismatch"));
                }
            }
            if payload.is_none() {
                if let HostCollectivePoll::Ready(result) =
                    state.group.take_result(self.owner()).map_err(error)?
                {
                    if result.descriptor != descriptor {
                        return Err(error("collective result identity mismatch"));
                    }
                    state.completed[self.local] += 1;
                    self.shared.wake.notify_all();
                    return Ok(result.values);
                }
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
                    state
                        .group
                        .submit(
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
        let result = self.exchange_inner(transaction, site, kind, values);
        if let Err(source) = &result {
            let mut state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
            fail(&mut state, source.to_string());
            self.shared.wake.notify_all();
        }
        result
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
        state.closed[self.local] = true;
        if state.group.status() == Some(HostCollectiveStatus::Pending) {
            fail(&mut state, "decoder collective peer dropped".into());
        }
        self.shared.wake.notify_all();
    }
}
fn fail(state: &mut State, message: String) {
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
fn error(message: impl std::fmt::Display) -> Error {
    Error::Model {
        message: format!("decoder TP collective: {message}"),
    }
}
