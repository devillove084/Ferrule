use super::*;

#[derive(Clone)]
pub(super) struct Snapshot {
    pub(super) recurrent: BTreeMap<usize, GatedDeltaNetState>,
    pub(super) kv: BTreeMap<usize, KvHistory>,
}
#[derive(Default)]
pub(super) struct Probe {
    pub(super) snapshots: Mutex<BTreeMap<(u32, u64, bool), Snapshot>>,
    // 1: known late failure, 2: Unknown late failure; only final source triggers.
    pub(super) fault: AtomicUsize,
    pub(super) stopped: AtomicUsize,
    pub(super) calls: AtomicUsize,
    pub(super) expert_shutdowns: AtomicUsize,
}
type RealProgram = TensorScopedStateProgram<HybridCpuTensorPipelineStageProgram>;
pub(super) struct Observed {
    pub(super) inner: RealProgram,
    pub(super) parallel: HybridParallelPlan,
    pub(super) lane: usize,
    pub(super) probe: Arc<Probe>,
}
impl Observed {
    fn capture(
        &self,
        batch: &PackedDecoderBatch,
        states: &[HybridTensorSequenceState],
        view: &CpuKvView,
        before: bool,
    ) {
        let end = if before {
            batch.sequences()[0].context_len()
        } else {
            batch.sequences()[0].sequence_len()
        };
        let local = self.parallel.rank_layout(self.lane).unwrap();
        let mut kv = BTreeMap::new();
        if end > 0 {
            for entry in local.state_schema().entries() {
                if let Some(layer) = entry.local_kv_layer {
                    let HybridLocalAttention::Gqa {
                        kv_heads, head_dim, ..
                    } = local.attention(entry.global_layer).unwrap()
                    else {
                        panic!()
                    };
                    kv.insert(
                        entry.global_layer,
                        view.history(layer, 0, end - 1, kv_heads.len(), *head_dim)
                            .unwrap(),
                    );
                }
            }
        }
        self.probe.snapshots.lock().unwrap().insert(
            (local.owner().get(), view.transaction().get(), before),
            Snapshot {
                recurrent: states[0].kv_state().recurrent(),
                kv,
            },
        );
    }
}
impl PipelineStateProgram for Observed {
    type SequenceState = HybridTensorSequenceState;
    type Lifecycle = HybridTensorSequenceLifecycle;
    type KvView = CpuKvView;
    fn plan(&self) -> &LayerSegmentPlan {
        self.inner.plan()
    }
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle> {
        self.inner.new_lifecycle()
    }
    fn hybrid_schema(&self) -> Option<&HybridStateSchema> {
        self.inner.hybrid_schema()
    }
    fn hybrid_reference_expert_group(&self) -> Option<&ExpertGroup> {
        self.inner.hybrid_reference_expert_group()
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
        self.capture(batch, states, view, true);
        self.probe.calls.fetch_add(1, Ordering::Relaxed);
        let transaction = context.transaction;
        let output = self
            .inner
            .execute_state(batch, input, states, view, context)?;
        self.capture(batch, states, view, false);
        if self.plan().owns_output() && self.lane + 1 == self.parallel.tensor_members().len() {
            match self.probe.fault.swap(0, Ordering::AcqRel) {
                1 => return Err(error("injected late hybrid EP failure")),
                2 => std::panic::panic_any(PanicQuiescence::Unknown),
                3 => {
                    return Err(ferrule_model::decoder::KvPrepareQuiescenceUnknown::wrap(
                        transaction,
                        error("secret checkpoint payload from typed unknown"),
                    ));
                }
                _ => (),
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
        self.inner.owner_stats()
    }
    fn shutdown(&mut self) -> Result<()> {
        let has_experts = self.inner.hybrid_reference_expert_group().is_some();
        let result = self.inner.shutdown();
        self.probe.stopped.fetch_add(1, Ordering::Relaxed);
        if has_experts && result.is_ok() {
            self.probe.expert_shutdowns.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}
pub(super) fn pipeline(
    fixture: Arc<fixture::Fixture>,
    pp: usize,
    tp: usize,
    probe: Arc<Probe>,
) -> PipelineParallelExecutor {
    let spec = fixture.resources().spec().clone();
    let layout = layout(&spec, pp, tp);
    let owner_layout = layout.clone();
    build(
        topology(pp, tp),
        &spec,
        layout,
        config(),
        options(),
        move |_| Ok(fixture.resources()),
        move |rank, stage| {
            let index = rank.global.get() as usize;
            Ok(stage.map_program(|inner| Observed {
                inner,
                parallel: owner_layout.stages()[index / tp].clone(),
                lane: index % tp,
                probe,
            }))
        },
    )
    .unwrap()
}
