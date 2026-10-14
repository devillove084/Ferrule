//! Real owner execution, then a late fault before Ready. The extra owner
//! evidence is observation only: no model call, parent protocol, commit or publication.
use super::*;
use crate::parallel::pipeline::PhysicalKvLayout;
use ferrule_model::ModelExecutionBackend;
use std::sync::Barrier;
#[path = "late_fault_support.rs"]
mod support;
use support::{Lifecycle, PhysicalSnapshot, Pool, ReleasedState, physical_snapshot};

#[derive(Default)]
struct Evidence {
    physical: Mutex<BTreeMap<u32, Vec<PhysicalSnapshot>>>,
    released: Mutex<BTreeMap<u32, Vec<ReleasedState>>>,
    allocations: Mutex<BTreeMap<u32, Vec<KvPageId>>>,
    injected: AtomicUsize,
}
struct LateProgram {
    inner: Observed,
    pool: Pool,
    description: PipelineStageDescription,
    context: usize,
    evidence: Arc<Evidence>,
    barrier: Arc<Barrier>,
    cancel: Arc<AtomicBool>,
    cancel_fault: bool,
}
impl PipelineStateProgram for LateProgram {
    type SequenceState = HybridTensorSequenceState;
    type Lifecycle = Lifecycle;
    type KvView = CpuKvView;
    fn plan(&self) -> &LayerSegmentPlan {
        self.inner.plan()
    }
    fn hybrid_schema(&self) -> Option<&HybridStateSchema> {
        self.inner.hybrid_schema()
    }
    fn hybrid_reference_expert_group(&self) -> Option<&ExpertGroup> {
        self.inner.hybrid_reference_expert_group()
    }
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle> {
        Ok(Lifecycle {
            inner: self.inner.new_lifecycle()?,
            rank: self.inner.parallel.tensor_members()[self.inner.lane].get(),
            evidence: self.evidence.clone(),
        })
    }
    fn validate_execution(
        &self,
        binding: &KvCommitBinding,
        batch: &PackedDecoderBatch,
        ids: &[u64],
    ) -> Result<()> {
        self.inner.validate_execution(binding, batch, ids)
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [Self::SequenceState],
        view: &mut CpuKvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        assert!(
            context.experts.is_none(),
            "factory owns its authenticated experts"
        );
        let transaction = context.transaction;
        let check_active = context.check_active;
        let output = self.inner.execute_state(
            batch,
            input,
            states,
            view,
            PipelineExecutionContext {
                transaction,
                sequence_ids: context.sequence_ids,
                experts: None,
                check_active: &mut *check_active,
            },
        )?;
        if transaction.get() == 2 {
            let owner = self.inner.parallel.tensor_members()[self.inner.lane];
            self.evidence
                .allocations
                .lock()
                .unwrap()
                .insert(owner.get(), batch.new_pages().to_vec());
            self.barrier.wait();
            if self.plan().owns_output()
                && self.inner.lane + 1 == self.inner.parallel.tensor_members().len()
            {
                self.evidence.injected.fetch_add(1, Ordering::Relaxed);
                if self.cancel_fault {
                    self.cancel.store(true, Ordering::Release);
                    // Use the real Execute cancellation check after the parent
                    // observer mirrors the caller flag, before any Ready vote.
                    let deadline = std::time::Instant::now() + Duration::from_secs(3);
                    loop {
                        check_active(transaction)?;
                        assert!(
                            std::time::Instant::now() < deadline,
                            "parent did not mirror cancellation"
                        );
                        std::thread::yield_now();
                    }
                }
                return Err(error("late owner error after real model writes"));
            }
        }
        Ok(output)
    }
    fn validate_frontier(&self, b: &PackedDecoderBatch, s: &[Self::SequenceState]) -> Result<()> {
        self.inner.validate_frontier(b, s)
    }
    fn expert_outstanding(&self) -> usize {
        self.inner.expert_outstanding()
    }
    fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        let snapshot = physical_snapshot(&mut self.pool, &self.description, self.context);
        self.evidence
            .physical
            .lock()
            .unwrap()
            .entry(self.inner.parallel.tensor_members()[self.inner.lane].get())
            .or_default()
            .push(snapshot);
        self.inner.owner_stats()
    }
    fn shutdown(&mut self) -> Result<()> {
        self.inner.shutdown()
    }
}

fn run(context: usize, tp: usize, cancel_fault: bool) {
    let fixture = Arc::new(fixture::Fixture::new());
    let spec = fixture.resources().spec().clone();
    let layout = layout(&spec, 2, tp);
    let owner_layout = layout.clone();
    let observed = Arc::new(Probe::default());
    let probe = observed.clone();
    let evidence = Arc::new(Evidence::default());
    let owner_evidence = evidence.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    let owner_cancel = cancel.clone();
    let barriers = [Arc::new(Barrier::new(tp)), Arc::new(Barrier::new(tp))];
    let load = fixture.clone();
    // Production factory construction, with the existing private decoration
    // seam only for observation/fault injection. No test-owned forward loop.
    let mut pp = build(
        topology(2, tp),
        &spec,
        layout,
        config(),
        options(),
        move |_| Ok(load.resources()),
        move |rank, mut stage| {
            let index = rank.global.get() as usize;
            let pool = stage.backend.pool_mut().clone();
            let description = stage.boot_description().clone();
            Ok(stage.map_program(|inner| LateProgram {
                inner: Observed {
                    inner,
                    parallel: owner_layout.stages()[index / tp].clone(),
                    lane: index % tp,
                    probe,
                },
                pool,
                description,
                evidence: owner_evidence,
                barrier: barriers[index / tp].clone(),
                cancel: owner_cancel,
                cancel_fault,
                context,
            }))
        },
    )
    .unwrap();
    let record = protocol::tap(&mut pp);
    let prefix = &[1, 2, 3, 4][..context];
    pp.forward(SessionId(73), prefix, ForwardPhase::Prefill)
        .unwrap();
    let before = pp.owner_stats().unwrap();
    let logical = pp.page_manager().stats();
    let start = record.events.lock().unwrap().len();
    let failure = pp
        .forward_cancellable(SessionId(73), &[5], ForwardPhase::Decode, &cancel)
        .unwrap_err();
    assert!(!completion_boundary::is_unknown(&failure));
    assert_eq!(evidence.injected.load(Ordering::Relaxed), 1);
    assert_eq!(cancel.load(Ordering::Acquire), cancel_fault);
    assert_eq!(pp.coordinator().publication_count(), 1);
    assert_eq!(pp.coordinator().retained_transaction_count(), 0);
    assert_eq!(pp.coordinator().in_use_credits(), 0);
    let after_pages = pp.page_manager().stats();
    assert_eq!(after_pages.allocated_pages, logical.allocated_pages);
    assert_eq!(after_pages.committed_tokens, logical.committed_tokens);
    assert_eq!(after_pages.capacity_tokens, logical.capacity_tokens);
    assert_eq!(after_pages.retiring_pages, 0);
    // Logical page IDs are allocated lazily. The new ID at context4 returns to
    // the reusable free list; allocated/committed capacity stays unchanged.
    assert_eq!(
        after_pages.free_pages,
        logical.free_pages + usize::from(context == 4)
    );
    let after = pp.owner_stats().unwrap();
    for (old, new) in before.iter().zip(&after) {
        assert_eq!(old.rank, new.rank);
        assert_eq!(old.kv, new.kv);
        assert_eq!(new.kv.active_transactions, 0);
        assert_eq!(new.expert_outstanding, 0);
        assert_eq!(new.sessions, 1);
        let rank = new.rank.global.get();
        let physical = evidence.physical.lock().unwrap();
        assert_eq!(physical[&rank].len(), 2);
        assert_eq!(
            physical[&rank][0], physical[&rank][1],
            "owner {rank} changed committed KV/slots/free capacity"
        );
        let allocations = evidence.allocations.lock().unwrap();
        assert_eq!(allocations[&rank].len(), usize::from(context == 4));
        let snapshots = observed.snapshots.lock().unwrap();
        let committed = &snapshots[&(rank, 1, false)];
        let entered = &snapshots[&(rank, 2, true)];
        assert_eq!(committed.recurrent, entered.recurrent);
        for (layer, kv) in &committed.kv {
            assert_eq!(kv.key, entered.kv[layer].key);
            assert_eq!(kv.value, entered.kv[layer].value);
        }
        let late = &snapshots[&(rank, 2, false)];
        assert!(late.recurrent.values().all(|s| s.position() == context + 1));
        assert_ne!(late.recurrent, committed.recurrent);
        for (layer, kv) in &late.kv {
            assert!(kv.key.len() > committed.kv[layer].key.len());
            assert!(kv.value.len() > committed.kv[layer].value.len());
        }
    }
    let events = record.events.lock().unwrap();
    let failure_events = &events[start..];
    assert!(
        !failure_events
            .iter()
            .any(|(op, _)| matches!(*op, "ready" | "install" | "publish"))
    );
    assert_eq!(
        failure_events
            .iter()
            .filter(|(op, _)| *op == "rollback")
            .map(|(_, rank)| *rank)
            .collect::<std::collections::BTreeSet<_>>(),
        (0..(2 * tp) as u32).collect()
    );
    drop(events);
    if tp == 1 {
        assert!(!pp.is_quarantined());
        let mut oracle = pipeline(fixture, 1, 1, Arc::new(Probe::default()));
        oracle
            .forward(SessionId(73), prefix, ForwardPhase::Prefill)
            .unwrap();
        let expected = oracle
            .forward(SessionId(73), &[7], ForwardPhase::Decode)
            .unwrap();
        let actual = pp
            .forward(SessionId(73), &[7], ForwardPhase::Decode)
            .unwrap();
        assert_eq!(actual.transaction.get(), 3);
        close(actual.logits.values(), expected.logits.values());
        let snapshots = observed.snapshots.lock().unwrap();
        for rank in 0..2 {
            assert_eq!(
                snapshots[&(rank, 3, true)].recurrent,
                snapshots[&(rank, 1, false)].recurrent
            );
            for (layer, kv) in &snapshots[&(rank, 1, false)].kv {
                assert_eq!(kv.key, snapshots[&(rank, 3, true)].kv[layer].key);
                assert_eq!(kv.value, snapshots[&(rank, 3, true)].kv[layer].value);
            }
        }
        drop(snapshots);
        oracle.shutdown().unwrap();
    } else {
        assert!(pp.is_quarantined());
        let calls = observed.calls.load(Ordering::Relaxed);
        assert!(
            pp.forward(SessionId(73), &[7], ForwardPhase::Decode)
                .is_err()
        );
        assert_eq!(observed.calls.load(Ordering::Relaxed), calls);
    }
    pp.release_session(SessionId(73)).unwrap();
    let released = pp.owner_stats().unwrap();
    for owner in &released {
        assert_eq!(owner.sessions, 0);
        assert_eq!(owner.kv.resident_pages, 0);
        assert_eq!(owner.kv.free_pages, owner.kv.physical_pages);
        assert_eq!(owner.kv.active_transactions, 0);
        let rank = owner.rank.global.get();
        let physical = evidence.physical.lock().unwrap();
        assert!(physical[&rank].last().unwrap().slots.is_empty());
        assert_eq!(
            physical[&rank].last().unwrap().free,
            config()
                .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cpu))
                .unwrap()
        );
        let states = evidence.released.lock().unwrap();
        let snapshots = observed.snapshots.lock().unwrap();
        assert!(
            states[&rank]
                .iter()
                .any(|(position, state)| *position == context
                    && state == &snapshots[&(rank, 1, false)].recurrent),
            "owner {rank} source state changed during aborted step"
        );
    }
    assert_eq!(pp.page_manager().stats().allocated_pages, 0);
    pp.shutdown().unwrap();
    pp.shutdown().unwrap();
    assert_eq!(observed.stopped.load(Ordering::Relaxed), 2 * tp);
    assert_eq!(observed.expert_shutdowns.load(Ordering::Relaxed), 2);
}
macro_rules! case {
    ($name:ident, $context:expr, $tp:expr, $cancel:expr) => {
        #[test]
        fn $name() {
            run($context, $tp, $cancel);
        }
    };
}
case!(context3_half_page_tp1_late_error, 3, 1, false);
case!(context3_half_page_tp1_late_cancel, 3, 1, true);
case!(context3_half_page_tp2_late_error, 3, 2, false);
case!(context3_half_page_tp2_late_cancel, 3, 2, true);
case!(context4_new_page_tp1_late_error, 4, 1, false);
case!(context4_new_page_tp1_late_cancel, 4, 1, true);
case!(context4_new_page_tp2_late_error, 4, 2, false);
case!(context4_new_page_tp2_late_cancel, 4, 2, true);
