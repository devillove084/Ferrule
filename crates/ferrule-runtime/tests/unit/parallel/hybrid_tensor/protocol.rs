use super::*;
use ferrule_common::execution::ExecutionBatch;

#[derive(Default)]
pub(super) struct Record {
    pub(super) events: Mutex<Vec<(&'static str, u32)>>,
    corrupt: AtomicBool,
    reject_ready: AtomicBool,
    pub(super) pending_rollback: AtomicBool,
}
struct Tap {
    inner: Rc<RefCell<Box<dyn PipelineTransport>>>,
    record: Arc<Record>,
}
impl Tap {
    fn record(&self, rank: PipelineRank, command: &PipelineCommand) {
        self.record
            .events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((command.completion_context(rank).stage, rank.global.get()));
    }
}
impl PipelineTransport for Tap {
    fn call_observed(
        &mut self,
        rank: PipelineRank,
        mut command: PipelineCommand,
        poll: &mut dyn FnMut(),
    ) -> Result<PipelineReply> {
        self.record(rank, &command);
        if matches!(command, PipelineCommand::Rollback(_))
            && self.record.pending_rollback.load(Ordering::Acquire)
        {
            return Ok(PipelineReply::Ack(KvEndProgress::Pending));
        }
        if matches!(command, PipelineCommand::Ready(_))
            && rank.global.get() == 3
            && self.record.reject_ready.swap(false, Ordering::AcqRel)
        {
            return Err(error("injected final readiness rejection"));
        }
        if rank.global.get() == 1
            && let PipelineCommand::Prepare { batch, .. } = &mut command
            && self.record.corrupt.swap(false, Ordering::AcqRel)
        {
            let mut tokens = batch.token_ids().to_vec();
            tokens[0] = (tokens[0] + 1) % 11;
            **batch = ExecutionBatch::new(
                batch.mode(),
                tokens,
                batch.positions().to_vec(),
                batch.kv_write_slots().to_vec(),
                batch.logits().to_vec(),
                batch.sequences().to_vec(),
                batch.kv_block_ids().to_vec(),
            )
            .with_intent(batch.intent());
        }
        self.inner.borrow_mut().call_observed(rank, command, poll)
    }
    fn call_batch_observed_with_failure(
        &mut self,
        commands: Vec<(PipelineRank, PipelineCommand)>,
        poll: &mut dyn FnMut(),
        failure: &mut dyn FnMut(),
    ) -> Result<Vec<PipelineReply>> {
        for (rank, command) in &commands {
            self.record(*rank, command);
        }
        self.inner
            .borrow_mut()
            .call_batch_observed_with_failure(commands, poll, failure)
    }
    fn outstanding(&self) -> usize {
        self.inner.borrow().outstanding()
    }
    fn shutdown(&mut self) -> Result<()> {
        self.inner.borrow_mut().shutdown()
    }
    fn quarantine(&mut self) {
        self.inner.borrow_mut().quarantine()
    }
}
pub(super) fn tap(pipeline: &mut PipelineParallelExecutor) -> Arc<Record> {
    let record = Arc::new(Record::default());
    let inner = pipeline.transport.clone();
    pipeline.transport = Rc::new(RefCell::new(Box::new(Tap {
        inner,
        record: record.clone(),
    })));
    record
}
#[test]
fn global_identity_mismatch_is_detected_before_any_state_write_or_dispatch() {
    let fixture = Arc::new(fixture::Fixture::new());
    let probe = Arc::new(Probe::default());
    let mut pp = pipeline(fixture, 2, 2, probe.clone());
    let record = tap(&mut pp);
    record.corrupt.store(true, Ordering::Release);
    assert!(
        pp.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
            .is_err()
    );
    assert_eq!(probe.calls.load(Ordering::Relaxed), 0);
    assert!(probe.snapshots.lock().unwrap().is_empty());
    assert_eq!(pp.coordinator().publication_count(), 0);
    assert!(!pp.is_quarantined());
    let stats = pp.owner_stats().unwrap();
    assert!(stats.iter().all(|s| s.kv.active_transactions == 0
        && s.kv.resident_pages == 0
        && s.expert_outstanding == 0));
    assert!(stats.iter().flat_map(|s| &s.experts).all(|e| e.calls == 0));
    assert!(
        record
            .events
            .lock()
            .unwrap()
            .iter()
            .all(|(op, _)| !matches!(*op, "execute" | "ready" | "install" | "publish"))
    );
    pp.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
        .unwrap();
    pp.shutdown().unwrap();
}
#[test]
fn all_kv_owners_ready_before_install_and_ep_owners_never_vote() {
    for reject in [false, true] {
        let fixture = Arc::new(fixture::Fixture::new());
        let probe = Arc::new(Probe::default());
        let mut pp = pipeline(fixture, 2, 2, probe);
        let record = tap(&mut pp);
        record.reject_ready.store(reject, Ordering::Release);
        let result = pp.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill);
        let events = record.events.lock().unwrap_or_else(|p| p.into_inner());
        assert!(events.iter().all(|(_, rank)| *rank < 4));
        let ready_ranks = events
            .iter()
            .filter(|(op, _)| *op == "ready")
            .map(|(_, rank)| *rank)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ready_ranks, [0, 1, 2, 3].into_iter().collect());
        if reject {
            assert!(result.is_err());
            assert!(
                events
                    .iter()
                    .all(|(op, _)| !matches!(*op, "install" | "publish"))
            );
            assert_eq!(pp.coordinator().publication_count(), 0);
        } else {
            result.unwrap();
            let install = events.iter().position(|(op, _)| *op == "install").unwrap();
            assert_eq!(
                events[..install]
                    .iter()
                    .filter(|(op, _)| *op == "ready")
                    .map(|(_, rank)| *rank)
                    .collect::<std::collections::BTreeSet<_>>(),
                [0, 1, 2, 3].into_iter().collect()
            );
            let publish = events.iter().position(|(op, _)| *op == "publish").unwrap();
            assert_eq!(
                events[..publish]
                    .iter()
                    .filter(|(op, _)| *op == "install")
                    .map(|(_, rank)| *rank)
                    .collect::<std::collections::BTreeSet<_>>(),
                [0, 1, 2, 3].into_iter().collect()
            );
            for name in [
                "prepare", "execute", "ready", "install", "publish", "retire", "finish",
            ] {
                assert!(events.iter().any(|(op, _)| *op == name), "{name}");
            }
            assert_eq!(pp.coordinator().publication_count(), 1);
        }
        drop(events);
        pp.shutdown().unwrap();
    }
}

#[test]
fn singleton_empty_wire_schedule_still_requires_sealed_owner_validation() {
    let mut peers = DecoderTensorCollective::new_group_from_members(
        &[rank(0)],
        topology(1, 1).topology_id(),
        ParallelGroupId::new(1),
        options().collective_limits,
        options().collective_timeout,
    )
    .unwrap();
    use ferrule_model::transformer::StandardTensorCollective;
    let peer = peers.pop().unwrap();
    let control = peer.control();
    control.register_sites(peer.owner(), vec![]).unwrap();
    use ferrule_common::execution::*;
    use ferrule_model::decoder::{DecoderKvPageStatus, LogicalExecutionIdentity};
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let schema = HybridStateSchema::for_segment(resources.spec(), 0..4).unwrap();
    let desc = description(
        resources.spec(),
        LayerSegmentPlan::new(4, 0..4, true, true).unwrap(),
        &schema,
        config(),
        None,
    )
    .unwrap();
    let batch = ExecutionBatch::new(
        ForwardMode::Prefill,
        vec![1],
        vec![0],
        vec![Some(KvWriteSlot::new(0))],
        vec![LogitsRequest::Full],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            ForwardPhase::Prefill,
            0..1,
            0,
            1,
            0..1,
        )],
        vec![KvBlockId::new(0)],
    );
    let reservation = KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: 0..1,
        newly_allocated: vec![KvPageId(0)],
        generation: 0,
        execution_generation: 0,
        cow_replacement: None,
    };
    let logical = LogicalExecutionIdentity::validate(
        &batch,
        &[reservation],
        &desc.execution_capabilities().unwrap(),
        2,
        &|_| DecoderKvPageStatus::Vacant,
        &[11],
    )
    .unwrap();
    let t = topology(1, 1);
    let binding = KvCommitBinding::new(
        ExecutionTransactionId::new(1).unwrap(),
        t.topology_id(),
        t.participants(),
        1,
    )
    .unwrap();
    let identity = super::super::super::tensor_scope::TensorBatchIdentity::from_logical(
        &binding,
        logical,
        batch.sequences().len(),
    );
    control
        .begin_sealed(identity.clone(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    assert!(control.finish(binding.transaction()).is_err());
    control
        .validate_owner_execution(rank(0), &identity)
        .unwrap();
    control.finish(binding.transaction()).unwrap();
}
