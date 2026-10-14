//! Test-only channel adapter: production authority stays on the existing trait/runtime.
use super::*;
use crate::transformer::RouterRoutes;
use std::{cell::RefCell, rc::Rc, thread::JoinHandle};

impl Bf16Channel {
    fn permit_device(
        &mut self,
        semantic: HybridCollectiveSemantic,
        elements: usize,
    ) -> Result<DevicePayloadPermit> {
        let result = (|| {
            self.ensure_known()?;
            let (transaction, rows, sequences) = self
                .active
                .ok_or_else(|| invalid("Mixed device participation outside invocation"))?;
            let site = self
                .sites
                .get(self.next)
                .ok_or_else(|| invalid("Mixed schedule exhausted"))?
                .clone();
            if self.failed
                || self.device_permit.is_some()
                || !matches!(
                    semantic,
                    HybridCollectiveSemantic::AttentionOutput
                        | HybridCollectiveSemantic::RoutedExpertOutput
                )
                || site.semantic != semantic
                || site.kind != HybridCollectiveKind::ReduceSum
                || rows == 0
                || rows > 8
                || elements == 0
                || elements != site.shape.elements(sequences, rows)?
            {
                return Err(invalid("Mixed device site/order/extent mismatch"));
            }
            if semantic == HybridCollectiveSemantic::RoutedExpertOutput {
                self.validate_routed_output(
                    site.global_layer
                        .ok_or_else(|| invalid("Mixed routed site lost layer"))?,
                )?;
            }
            let binding = self.transport.sealed_binding()?;
            if binding.sites != self.sites
                || rows > binding.max_rows
                || sequences > binding.max_sequences
            {
                return Err(invalid("Mixed device schedule differs from sealed binding"));
            }
            let members: [ParallelRankId; 2] = binding
                .members
                .as_slice()
                .try_into()
                .map_err(|_| invalid("Mixed device participation requires two owners"))?;
            let owner = self.transport.owner();
            if !members.contains(&owner) {
                return Err(invalid("Mixed device owner is not a sealed member"));
            }
            let identity = DevicePayloadIdentity {
                epoch: binding.epoch,
                transaction,
                owner,
                members,
                site,
                rows,
                sequences,
                elements,
            };
            let permit = self.transport.permit_device_payload(identity.clone())?;
            if permit.identity() != &identity {
                return Err(invalid(
                    "Mixed device transport returned a different permit",
                ));
            }
            self.device_permit = Some(identity);
            Ok(permit)
        })();
        if result.is_err() {
            self.abort();
        }
        result
    }
    fn check_device_permit(&mut self, permit: &DevicePayloadPermit) -> Result<()> {
        self.ensure_known()?;
        let result = if self.device_permit.as_ref() != Some(permit.identity()) {
            Err(invalid("Mixed device permit is not pending"))
        } else {
            self.transport.check_device_payload_permit(permit)
        };
        if result.is_err() {
            self.abort();
        }
        result
    }
    fn device_enqueued(&mut self, permit: DevicePayloadPermit) -> Result<()> {
        // An enqueue has already happened. Even acknowledgement validation errors
        // cannot downgrade custody to an ordinary, safely-retired failure.
        self.device_uncompleted = true;
        let result = (|| {
            self.ensure_known()?;
            if self.failed || self.device_permit.as_ref() != Some(permit.identity()) {
                return Err(invalid("Mixed device enqueue lost its permit"));
            }
            self.transport.device_payload_enqueued(permit)
        })();
        if let Err(source) = result {
            self.unknown = true;
            self.abort();
            return Err(CompletionUnknown::wrap(source));
        }
        self.device_permit = None;
        if self.sites[self.next].semantic == HybridCollectiveSemantic::RoutedExpertOutput {
            self.agreed_route = None;
        }
        self.next += 1;
        Ok(())
    }
    fn complete_device_payloads(&mut self, completion: Result<()>) -> Result<()> {
        let result = (|| {
            self.ensure_known()?;
            let (transaction, _, _) = self
                .active
                .ok_or_else(|| invalid("Mixed device completion outside invocation"))?;
            if self.failed
                || self.device_permit.is_some()
                || !self.device_uncompleted
                || self.next != self.sites.len()
            {
                return Err(invalid("Mixed device completion is missing or premature"));
            }
            self.transport
                .complete_device_payloads(transaction, completion)
        })();
        if let Err(source) = result {
            self.unknown = true;
            self.abort();
            return Err(CompletionUnknown::wrap(source));
        }
        self.device_uncompleted = false;
        Ok(())
    }
}

#[derive(Default)]
struct Calls {
    events: Vec<&'static str>,
    next: usize,
    votes: usize,
    checks: usize,
    enqueues: usize,
    acknowledgements: usize,
    fences: usize,
    host: usize,
    reports: usize,
    owner_ready: usize,
    aborts: usize,
    checked: bool,
    pending: Option<DevicePayloadIdentity>,
}
struct DeviceEndpoint {
    binding: HybridBf16CollectiveBinding,
    route: HybridMixedRouteSite,
    calls: Rc<RefCell<Calls>>,
    enabled: bool,
}
fn transaction() -> ExecutionTransactionId {
    ExecutionTransactionId::new(1).unwrap()
}
impl HybridBf16Collective for DeviceEndpoint {
    fn owner(&self) -> ParallelRankId {
        self.binding.members[0]
    }
    fn members(&self) -> &[ParallelRankId] {
        &self.binding.members
    }
    fn sealed_binding(&self) -> Result<HybridBf16CollectiveBinding> {
        Ok(self.binding.clone())
    }
    fn sealed_route_sites(&self) -> Result<Vec<HybridMixedRouteSite>> {
        Ok(vec![self.route.clone()])
    }
    fn data_location(&self, site: &HybridCollectiveSite) -> DataLocation {
        if self.enabled && site.kind == HybridCollectiveKind::ReduceSum {
            DataLocation::Device
        } else {
            DataLocation::Host
        }
    }
    fn participate_device(&mut self, identity: &DevicePayloadIdentity) -> Result<()> {
        let mut c = self.calls.borrow_mut();
        if !self.enabled
            || identity.epoch != self.binding.epoch
            || identity.transaction != transaction()
            || identity.owner != self.owner()
            || identity.members.as_slice() != self.members()
            || self.binding.sites.get(c.next) != Some(&identity.site)
            || identity.rows != 1
            || identity.sequences != 1
            || identity.elements != 1
            || c.pending.is_some()
        {
            return Err(invalid("mock canonical device vote mismatch"));
        }
        c.votes += 1;
        c.checked = false;
        c.pending = Some(identity.clone());
        c.events.push("vote");
        Ok(())
    }
    fn check_device_payload_permit(&mut self, permit: &DevicePayloadPermit) -> Result<()> {
        let mut c = self.calls.borrow_mut();
        if c.pending.as_ref() != Some(permit.identity()) {
            return Err(invalid("mock permit not pending"));
        }
        c.checks += 1;
        c.checked = true;
        c.events.push("check");
        Ok(())
    }
    fn device_payload_enqueued(&mut self, permit: DevicePayloadPermit) -> Result<()> {
        let mut c = self.calls.borrow_mut();
        if !c.checked
            || c.pending.as_ref() != Some(permit.identity())
            || c.enqueues != c.acknowledgements + 1
        {
            return Err(invalid(
                "mock enqueue acknowledgement lacks recheck/producer",
            ));
        }
        c.acknowledgements += 1;
        c.next += 1;
        c.pending = None;
        c.checked = false;
        c.events.push("ack");
        Ok(())
    }
    fn agree_routes(
        &mut self,
        tx: ExecutionTransactionId,
        agreement: &HybridMixedRouteAgreement,
    ) -> Result<()> {
        if tx != transaction() || agreement.site() != &self.route || agreement.rows() != 1 {
            return Err(invalid("mock exact route mismatch"));
        }
        self.calls.borrow_mut().events.push("routes");
        Ok(())
    }
    fn exchange_bf16(
        &mut self,
        tx: ExecutionTransactionId,
        site: u64,
        kind: HybridCollectiveKind,
        words: Vec<u16>,
    ) -> Result<Vec<u16>> {
        let mut c = self.calls.borrow_mut();
        let expected = self
            .binding
            .sites
            .get(c.next)
            .ok_or_else(|| invalid("mock schedule exhausted"))?;
        if tx != transaction()
            || expected.ordinal != site
            || expected.kind != kind
            || words.len() != 1
        {
            return Err(invalid("mock host site mismatch"));
        }
        c.next += 1;
        c.host += 1;
        c.events.push("host");
        Ok(if kind == HybridCollectiveKind::AllGather {
            words.repeat(2)
        } else {
            words
        })
    }
    fn complete_device_payloads(
        &mut self,
        tx: ExecutionTransactionId,
        completion: Result<()>,
    ) -> Result<()> {
        completion?;
        let mut c = self.calls.borrow_mut();
        if tx != transaction()
            || c.next != self.binding.sites.len()
            || c.pending.is_some()
            || c.acknowledgements == 0
            || c.fences != c.enqueues
        {
            return Err(invalid("mock lacks actual consumer fence"));
        }
        c.reports += 1;
        c.events.push("report");
        Ok(())
    }
    fn complete_owner(&mut self, tx: ExecutionTransactionId) -> Result<()> {
        let mut c = self.calls.borrow_mut();
        if tx != transaction()
            || c.next != self.binding.sites.len()
            || (c.acknowledgements != 0 && c.reports != 1)
        {
            return Err(invalid("mock cannot mark owner ready"));
        }
        c.owner_ready += 1;
        c.events.push("owner");
        Ok(())
    }
    fn abort(&mut self) {
        self.calls.borrow_mut().aborts += 1;
    }
}

fn device_channel(enabled: bool) -> (Bf16Channel, Rc<RefCell<Calls>>, HybridMixedRouteAgreement) {
    let members = vec![ParallelRankId::new(0), ParallelRankId::new(1)];
    let sites = [
        HybridCollectiveSemantic::AttentionOutput,
        HybridCollectiveSemantic::RoutedExpertOutput,
        HybridCollectiveSemantic::OutputHead,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, semantic)| HybridCollectiveSite {
        ordinal: i as u64,
        global_layer: (i < 2).then_some(7),
        semantic,
        kind: if i == 2 {
            HybridCollectiveKind::AllGather
        } else {
            HybridCollectiveKind::ReduceSum
        },
        shape: HybridCollectiveShape {
            fixed: 0,
            per_sequence: 0,
            per_row: 1,
        },
    })
    .collect::<Vec<_>>();
    let route = HybridMixedRouteSite {
        layer: 7,
        output_site: 1,
        experts: 4,
        top_k: 2,
        max_rows: 1,
        accumulation: HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
        combine: HybridMoeCombineOrder::LocalBf16AddThenReduce,
    };
    let agreement = HybridMixedRouteAgreement::new(
        route.clone(),
        &RouterRoutes::new(1, 2, vec![0, 1], vec![0.5, 0.5]).unwrap(),
    )
    .unwrap();
    let calls = Rc::new(RefCell::new(Calls::default()));
    let binding = HybridBf16CollectiveBinding {
        epoch: ParallelTopologyId::new(7),
        source: members[0],
        members,
        version: 2,
        precision: HybridCollectivePrecision::Bf16PartialsAndReduction,
        sites: sites.clone(),
        max_rows: 1,
        max_sequences: 1,
    };
    let mut channel = Bf16Channel::new(
        Box::new(DeviceEndpoint {
            binding,
            route,
            calls: calls.clone(),
            enabled,
        }),
        sites,
    );
    channel.begin(transaction(), 1, 1).unwrap();
    (channel, calls, agreement)
}

fn cpu_enqueue(calls: &Rc<RefCell<Calls>>, finite: bool) -> JoinHandle<Result<()>> {
    let mut c = calls.borrow_mut();
    c.enqueues += 1;
    c.events.push("enqueue");
    std::thread::spawn(move || {
        let words = if finite {
            [0x3f80u16, 0x4000]
        } else {
            [0x7f7fu16; 2]
        };
        let sum =
            f32::from_bits(u32::from(words[0]) << 16) + f32::from_bits(u32::from(words[1]) << 16);
        if !sum.is_finite() {
            return Err(invalid("CPU consumer finite flag failed"));
        }
        Ok(())
    })
}
fn cpu_fence(calls: &Rc<RefCell<Calls>>, jobs: Vec<JoinHandle<Result<()>>>) -> Result<()> {
    let mut result = Ok(());
    for job in jobs {
        let outcome = job.join().expect("CPU consumer panicked");
        let mut c = calls.borrow_mut();
        c.fences += 1;
        c.events.push("fence");
        if result.is_ok() {
            result = outcome;
        }
    }
    result
}
fn device_sites(
    c: &mut Bf16Channel,
    calls: &Rc<RefCell<Calls>>,
    agreement: &HybridMixedRouteAgreement,
    finite: bool,
) -> Vec<JoinHandle<Result<()>>> {
    let mut jobs = Vec::new();
    for semantic in [
        HybridCollectiveSemantic::AttentionOutput,
        HybridCollectiveSemantic::RoutedExpertOutput,
    ] {
        if semantic == HybridCollectiveSemantic::RoutedExpertOutput {
            c.agree_routes(agreement).unwrap();
        }
        let permit = c.permit_device(semantic, 1).unwrap();
        c.check_device_permit(&permit).unwrap();
        jobs.push(cpu_enqueue(calls, finite));
        c.device_enqueued(permit).unwrap();
    }
    assert_eq!(
        c.exchange(HybridCollectiveSemantic::OutputHead, vec![0x3f80])
            .unwrap(),
        [0x3f80; 2]
    );
    jobs
}

#[test]
fn scoped_protocol_calls_real_hooks_and_waits_for_cpu_consumers() {
    let (mut c, calls, agreement) = device_channel(true);
    let jobs = device_sites(&mut c, &calls, &agreement, true);
    assert_eq!(calls.borrow().reports, 0);
    assert_eq!(calls.borrow().owner_ready, 0);
    assert!(c.device_uncompleted);
    let outcome = cpu_fence(&calls, jobs);
    c.complete_device_payloads(outcome).unwrap();
    c.finish().unwrap();
    assert_eq!(
        calls.borrow().owner_ready,
        0,
        "local finish is not publication or an owner receipt"
    );
    c.transport.complete_owner(transaction()).unwrap();
    let p = calls.borrow();
    assert_eq!(
        (
            p.votes,
            p.checks,
            p.acknowledgements,
            p.host,
            p.reports,
            p.owner_ready
        ),
        (2, 2, 2, 1, 1, 1)
    );
    assert_eq!(
        p.events,
        [
            "vote", "check", "enqueue", "ack", "routes", "vote", "check", "enqueue", "ack", "host",
            "fence", "fence", "report", "owner"
        ]
    );
}

#[test]
fn host_mode_never_creates_device_work_or_completion_proof() {
    let (mut c, calls, agreement) = device_channel(false);
    assert_eq!(c.transport.data_location(&c.sites[0]), DataLocation::Host);
    c.exchange(HybridCollectiveSemantic::AttentionOutput, vec![0x3f80])
        .unwrap();
    c.agree_routes(&agreement).unwrap();
    c.exchange(HybridCollectiveSemantic::RoutedExpertOutput, vec![0x3f80])
        .unwrap();
    assert_eq!(
        c.exchange(HybridCollectiveSemantic::OutputHead, vec![0x3f80])
            .unwrap(),
        [0x3f80; 2]
    );
    c.finish().unwrap();
    let p = calls.borrow();
    assert_eq!(
        (
            p.votes,
            p.checks,
            p.enqueues,
            p.acknowledgements,
            p.fences,
            p.reports,
            p.owner_ready
        ),
        (0, 0, 0, 0, 0, 0, 0)
    );
    assert_eq!(p.host, 3);
}

#[test]
fn successful_result_without_actual_fence_never_marks_ready() {
    let (mut c, calls, agreement) = device_channel(true);
    let jobs = device_sites(&mut c, &calls, &agreement, true);
    let error = c.complete_device_payloads(Ok(())).unwrap_err();
    assert!(
        ErrorChain::new(&error)
            .downcast_ref::<CompletionUnknown>()
            .is_some()
    );
    assert!(c.completion_unknown());
    assert!(c.transport.complete_owner(transaction()).is_err());
    assert_eq!(calls.borrow().reports, 0);
    assert_eq!(calls.borrow().owner_ready, 0);
    let outcome = cpu_fence(&calls, jobs);
    assert!(
        c.complete_device_payloads(outcome).is_err(),
        "true cleanup cannot undo terminal Unknown"
    );
}

#[test]
fn per_site_fence_does_not_complete_full_invocation() {
    let (mut c, calls, _) = device_channel(true);
    let permit = c
        .permit_device(HybridCollectiveSemantic::AttentionOutput, 1)
        .unwrap();
    c.check_device_permit(&permit).unwrap();
    let job = cpu_enqueue(&calls, true);
    c.device_enqueued(permit).unwrap();
    let outcome = cpu_fence(&calls, vec![job]);
    assert!(c.complete_device_payloads(outcome).is_err());
    assert!(c.completion_unknown());
    assert_eq!(calls.borrow().reports, 0);
    assert_eq!(calls.borrow().owner_ready, 0);
}

#[test]
fn finite_flag_failure_retains_unknown_without_completion_or_ready() {
    let (mut c, calls, agreement) = device_channel(true);
    let jobs = device_sites(&mut c, &calls, &agreement, false);
    let outcome = cpu_fence(&calls, jobs);
    assert!(outcome.is_err());
    let error = c.complete_device_payloads(outcome).unwrap_err();
    assert!(
        ErrorChain::new(&error)
            .downcast_ref::<CompletionUnknown>()
            .is_some()
    );
    assert!(c.finish().is_err());
    assert!(c.transport.complete_owner(transaction()).is_err());
    assert_eq!(calls.borrow().reports, 0);
    assert_eq!(calls.borrow().owner_ready, 0);
}

#[test]
fn acknowledgement_requires_real_producer_and_prelaunch_recheck() {
    for missing_producer in [false, true] {
        let (mut c, calls, _) = device_channel(true);
        let permit = c
            .permit_device(HybridCollectiveSemantic::AttentionOutput, 1)
            .unwrap();
        let jobs = if missing_producer {
            c.check_device_permit(&permit).unwrap();
            Vec::new()
        } else {
            vec![cpu_enqueue(&calls, true)]
        };
        assert!(c.device_enqueued(permit).is_err());
        assert!(c.completion_unknown());
        assert_eq!(c.next, 0);
        assert_eq!(calls.borrow().acknowledgements, 0);
        assert_eq!(calls.borrow().owner_ready, 0);
        cpu_fence(&calls, jobs).unwrap();
    }
}
