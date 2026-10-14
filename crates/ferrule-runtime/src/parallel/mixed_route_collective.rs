//! Typed control events inside the existing decoder collective lifetime.
use super::*;
use crate::parallel::pipeline::tensor_scope::TensorCollectiveSemantic as Semantic;

pub(super) fn validate_sites(
    plan: &MixedRankCollectivePlan,
    sites: &[HybridMixedRouteSite],
) -> Result<()> {
    let outputs = plan
        .sites
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            matches!(
                s.operator,
                TensorCollectiveOperator::Protocol {
                    operation: Semantic::RoutedExpertOutput,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    if outputs.len() != sites.len() {
        return Err(error(
            "route sites must cover each complete routed FFN output exactly once",
        ));
    }
    for ((index, output), route) in outputs.into_iter().zip(sites) {
        if route.experts == 0
            || route.experts > 1024
            || route.top_k == 0
            || route.top_k > route.experts
            || route.top_k > 32
            || route.max_rows != plan.max_rows
            || route.max_rows > 32
            || route.accumulation != HybridRoutedAccumulation::HfEagerBf16ExpertOrder
            || route.combine != HybridMoeCombineOrder::LocalBf16AddThenReduce
            || route.output_site != output.site
            || output.layer != Some(route.layer)
            || output.kind() != HostCollectiveKind::AllReduceSumBf16
            || index == 0
            || plan.sites[index - 1].layer != Some(route.layer)
            || !matches!(
                plan.sites[index - 1].operator,
                TensorCollectiveOperator::Protocol {
                    operation: Semantic::AttentionOutput,
                    kind: HostCollectiveKind::AllReduceSumBf16,
                    ..
                }
            )
        {
            return Err(error(
                "route control must follow same-layer attention and precede one late BF16 combined FFN sum",
            ));
        }
    }
    Ok(())
}
pub(super) fn budget(
    sites: &[HybridMixedRouteSite],
    ranks: usize,
) -> Result<MixedRouteControlBudget> {
    if sites.is_empty() {
        return Ok(MixedRouteControlBudget::default());
    }
    let payload = sites
        .iter()
        .map(|site| {
            site.max_rows.checked_mul(site.top_k).and_then(|n| {
                n.checked_mul(std::mem::size_of::<usize>() + std::mem::size_of::<u32>())
            })
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| error("route payload overflow"))?
        .into_iter()
        .max()
        .unwrap();
    let service = payload
        .checked_mul(ranks)
        .ok_or_else(|| error("route service overflow"))?;
    let metadata = ranks
        .checked_mul(std::mem::size_of::<HybridMixedRouteAgreement>())
        .and_then(|n| n.checked_add(std::mem::size_of::<PendingRoutes>()))
        .and_then(|n| {
            n.checked_add(
                sites
                    .len()
                    .checked_mul(std::mem::size_of::<HybridMixedRouteSite>())?,
            )
        })
        .ok_or_else(|| error("route metadata overflow"))?;
    let total = service
        .checked_mul(2)
        .and_then(|n| n.checked_add(metadata))
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| error("route host budget overflow"))?;
    Ok(MixedRouteControlBudget {
        payload_bytes_per_owner: payload,
        service_payload_bytes: service,
        caller_payload_bytes: service,
        metadata_bytes: metadata,
        total_host_bytes: total,
    })
}
#[derive(PartialEq, Eq)]
struct RoutePayload {
    ids: Box<[usize]>,
    bits: Box<[u32]>,
}
pub(in super::super) struct PendingRoutes {
    sequence: u64,
    route_index: usize,
    deadline: Instant,
    payloads: [Option<RoutePayload>; 2],
    taken: [bool; 2],
}
impl PendingRoutes {
    fn ready(&self) -> bool {
        self.payloads.iter().all(Option::is_some)
    }
}
fn validate_payload(agreement: &HybridMixedRouteAgreement) -> Result<()> {
    let site = agreement.site();
    let count = agreement
        .rows()
        .checked_mul(site.top_k)
        .ok_or_else(|| error("route count overflow"))?;
    if agreement.rows() == 0
        || agreement.rows() > site.max_rows
        || site.top_k == 0
        || count != agreement.expert_ids().len()
        || count != agreement.weight_bits().len()
    {
        return Err(error("route row/slot count mismatch"));
    }
    for row in agreement.expert_ids().chunks_exact(site.top_k) {
        for (i, &id) in row.iter().enumerate() {
            if id >= site.experts || row[..i].contains(&id) {
                return Err(error("duplicate or invalid route expert"));
            }
        }
    }
    if agreement.weight_bits().iter().any(|b| {
        let w = f32::from_bits(*b);
        !w.is_finite() || w < 0.0
    }) {
        return Err(error("route weight must be finite and nonnegative"));
    }
    Ok(())
}
pub(in super::super) fn check_data_route(state: &State, local: usize, site: u64) -> Result<()> {
    if let Some(plan) = &state.mixed_plan
        && let Some(index) = plan
            .route_sites
            .iter()
            .position(|route| route.output_site == site)
        && state.routes_completed[local] <= index
    {
        return Err(error(
            "routed FFN data/expert execution requires completed typed route agreement",
        ));
    }
    Ok(())
}
impl MixedRankCollective {
    pub(super) fn agree_routes_inner(
        &mut self,
        transaction: ExecutionTransactionId,
        agreement: &HybridMixedRouteAgreement,
    ) -> Result<()> {
        validate_payload(agreement)?;
        let sequence = self
            .endpoint
            .sequence
            .checked_add(1)
            .ok_or_else(|| error("route sequence exhausted"))?;
        self.endpoint.sequence = sequence;
        let deadline = Instant::now()
            .checked_add(self.endpoint.shared.timeout)
            .ok_or_else(|| error("route deadline overflow"))?;
        let local = self.endpoint.local;
        let mut state = self
            .endpoint
            .shared
            .state
            .lock()
            .map_err(|_| error("collective mutex poisoned"))?;
        let mut submitted = false;
        loop {
            check_open(&mut state)?;
            if let Some(failure) = &state.failure {
                return Err(error(failure));
            }
            if state
                .execution
                .as_ref()
                .is_none_or(|(id, flag)| *id != transaction || flag.load(Ordering::Acquire))
            {
                return reject(
                    &mut state,
                    &self.endpoint.shared.wake,
                    "route transaction cancelled, inactive or stale",
                );
            }
            let route_index = state.routes_completed[local];
            let Some(expected) = self.plan.route_sites.get(route_index) else {
                return reject(
                    &mut state,
                    &self.endpoint.shared.wake,
                    "route schedule exhausted",
                );
            };
            let Some(next_data) = self.plan.sites.get(state.completed[local]) else {
                return reject(
                    &mut state,
                    &self.endpoint.shared.wake,
                    "route data schedule exhausted",
                );
            };
            if expected != agreement.site()
                || next_data.site != expected.output_site
                || state
                    .identity
                    .as_ref()
                    .is_none_or(|id| id.rows() != agreement.rows())
                || !state.validated[local]
            {
                return reject(
                    &mut state,
                    &self.endpoint.shared.wake,
                    "route layer/site/order/owner/packed rows differ from sealed execution",
                );
            }
            if !submitted {
                if state
                    .group
                    .sequence_watermark()
                    .is_some_and(|watermark| watermark >= sequence)
                {
                    return reject(
                        &mut state,
                        &self.endpoint.shared.wake,
                        "stale route sequence",
                    );
                }
                if state.routes.is_none() {
                    state.routes = Some(PendingRoutes {
                        sequence,
                        route_index,
                        deadline,
                        payloads: [None, None],
                        taken: [false; 2],
                    });
                }
                let pending = state.routes.as_mut().unwrap();
                if Instant::now() >= pending.deadline {
                    return reject(
                        &mut state,
                        &self.endpoint.shared.wake,
                        "typed route agreement timed out",
                    );
                }
                if pending.sequence != sequence
                    || pending.route_index != route_index
                    || pending.payloads[local].is_some()
                {
                    return reject(
                        &mut state,
                        &self.endpoint.shared.wake,
                        "route generation/site/duplicate owner mismatch",
                    );
                }
                pending.payloads[local] = Some(RoutePayload {
                    ids: agreement.expert_ids().into(),
                    bits: agreement.weight_bits().into(),
                });
                submitted = true;
                if pending.ready() && pending.payloads[0] != pending.payloads[1] {
                    fail(
                        &mut state,
                        "TP owners disagree on exact expert IDs or F32 route weight bits".into(),
                    );
                    self.endpoint.shared.wake.notify_all();
                    return reject(
                        &mut state,
                        &self.endpoint.shared.wake,
                        "TP owners disagree on exact expert IDs or F32 route weight bits",
                    );
                }
                self.endpoint.shared.wake.notify_all();
            }
            // check_open above dominates Ready. A host agreement never proves
            // CUDA completion or creates a KV ACK/publication receipt.
            if state
                .routes
                .as_ref()
                .is_some_and(|p| Instant::now() >= p.deadline)
            {
                return reject(
                    &mut state,
                    &self.endpoint.shared.wake,
                    "typed route agreement timed out",
                );
            }
            if state.routes.as_ref().is_some_and(PendingRoutes::ready) {
                let pending = state.routes.as_mut().unwrap();
                if pending.taken[local] {
                    return reject(
                        &mut state,
                        &self.endpoint.shared.wake,
                        "route result already consumed",
                    );
                }
                pending.taken[local] = true;
                let drained = pending.taken.iter().all(|taken| *taken);
                state.routes_completed[local] += 1;
                if drained {
                    state.routes = None;
                }
                self.endpoint.shared.wake.notify_all();
                return Ok(());
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return reject(
                    &mut state,
                    &self.endpoint.shared.wake,
                    "typed route agreement timed out",
                );
            }
            state.waiters += 1;
            let (next, _) = self
                .endpoint
                .shared
                .wake
                .wait_timeout(state, left)
                .map_err(|_| error("collective mutex poisoned"))?;
            state = next;
            state.waiters -= 1;
        }
    }
}

fn reject(state: &mut State, wake: &Condvar, message: &str) -> Result<()> {
    fail(state, message.into());
    wake.notify_all();
    Err(error(message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::pipeline::tensor_scope::{TensorCollectiveShape, tests::identity};
    use ferrule_common::error::ErrorChain;
    use ferrule_model::{
        decoder::CompletionUnknown,
        transformer::{RouterRoutes, hybrid_parallel::HybridBf16Collective},
    };
    #[test]
    fn ready_route_peer_loss_dominates_ready_and_retains_execution_watermarks() {
        for drop_transition in [true, false] {
            let id = identity();
            let tx = id.binding().transaction();
            let members = vec![ParallelRankId::new(0), ParallelRankId::new(1)];
            let site = HybridMixedRouteSite {
                layer: 0,
                output_site: 1,
                experts: 4,
                top_k: 2,
                max_rows: 1,
                accumulation: HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
                combine: HybridMoeCombineOrder::LocalBf16AddThenReduce,
            };
            let sites = [Semantic::AttentionOutput, Semantic::RoutedExpertOutput]
                .into_iter()
                .enumerate()
                .map(|(i, operation)| TensorCollectiveSite {
                    site: i as u64,
                    layer: Some(0),
                    operator: TensorCollectiveOperator::Protocol {
                        version: 2,
                        operation,
                        kind: HostCollectiveKind::AllReduceSumBf16,
                        shape: TensorCollectiveShape::PerRow(1),
                    },
                })
                .collect();
            let plan = Arc::new(
                MixedRankCollectivePlan::new(
                    id.binding().topology_id(),
                    ParallelGroupId::new(4),
                    members.clone(),
                    members[0],
                    sites,
                    HostCollectiveLimits {
                        max_ranks: 2,
                        max_elements_per_rank: 4,
                        max_host_bytes: 1024,
                    },
                    1,
                    1,
                )
                .unwrap()
                .with_route_sites(vec![site.clone()])
                .unwrap(),
            );
            let mut peers = DecoderTensorCollective::new_group_from_members(
                &members,
                plan.epoch(),
                plan.group(),
                plan.limits(),
                Duration::from_secs(5),
            )
            .unwrap()
            .into_iter()
            .map(|e| {
                let owner = e.owner();
                MixedRankCollective::prepare(e, owner, plan.clone()).unwrap()
            })
            .collect::<Vec<_>>();
            let control = peers[0].control();
            let cancelled = Arc::new(AtomicBool::new(false));
            control.begin_sealed(id.clone(), cancelled.clone()).unwrap();
            for p in &peers {
                p.validate_owner_execution(&id).unwrap();
            }
            std::thread::scope(|s| {
                let hs = peers
                    .iter_mut()
                    .map(|p| s.spawn(move || p.exchange_bf16(tx, 0, vec![0x3f80])))
                    .collect::<Vec<_>>();
                for h in hs {
                    h.join().unwrap().unwrap();
                }
            });
            let record = HybridMixedRouteAgreement::new(
                site,
                &RouterRoutes::new(1, 2, vec![0, 1], vec![0.5, 0.5]).unwrap(),
            )
            .unwrap();
            std::thread::scope(|s| {
                let h = s.spawn(|| peers[0].agree_routes(tx, &record));
                let end = Instant::now() + Duration::from_secs(2);
                while control.waiting_members() == 0 {
                    assert!(Instant::now() < end);
                    std::thread::yield_now();
                }
                {
                    let mut state = control.0.state.lock().unwrap();
                    let pending = state.routes.as_mut().unwrap();
                    assert_eq!(pending.sequence, 2);
                    pending.payloads[1] = Some(RoutePayload {
                        ids: record.expert_ids().into(),
                        bits: record.weight_bits().into(),
                    });
                    assert!(pending.ready());
                    if drop_transition {
                        close_peer(&mut state, 1);
                    } else {
                        state.closed[1] = true;
                    }
                    control.0.wake.notify_all();
                }
                let failure = h.join().unwrap().unwrap_err();
                assert!(
                    ErrorChain::new(&failure)
                        .downcast_ref::<CompletionUnknown>()
                        .is_some()
                );
            });
            assert!(cancelled.load(Ordering::Acquire));
            assert!(control.finish(tx).is_err());
            assert!(peers[0].complete_owner(tx).is_err());
            let state = control.0.state.lock().unwrap();
            assert_eq!(state.routes_completed, vec![0, 0]);
            assert_eq!(state.completed, vec![1, 1]);
            assert_eq!(state.execution_watermark, Some(tx));
            assert_eq!(state.group.sequence_watermark(), Some(1));
            assert!(state.routes.is_none());
        }
    }
}
