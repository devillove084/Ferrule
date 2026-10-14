//! Shared model-contract setup. Does not implement PipelineStateProgram or drive
//! a parent protocol. Runtime is used only for bounded host collective endpoints.
#![allow(dead_code)] // Shared by independent PP, TP and review test targets.
#[path = "experts.rs"]
mod experts;
#[path = "local_transaction.rs"]
mod local_transaction;
#[path = "probe.rs"]
mod probe;
pub use experts::Experts;
use ferrule_common::{ParallelGroupId, ParallelRankId, ParallelTopologyId, Result};
use ferrule_model::{
    decoder::*, execution::ExecutionPrecisionPolicy, transformer::hybrid_parallel::*,
    transformer::*,
};
use ferrule_runtime::parallel::{
    collective::HostCollectiveLimits, tensor::decoder_collective::DecoderTensorCollective,
};
pub use local_transaction::{LocalTransaction, Work, tx};
pub use probe::{Capture, capture, close, compare_local};
use std::{ops::Range, time::Duration};
pub fn rank(n: usize) -> ParallelRankId {
    ParallelRankId::new(n as u32)
}
pub fn endpoints_for(members: &[ParallelRankId]) -> Vec<DecoderTensorCollective> {
    DecoderTensorCollective::new_group_from_members(
        members,
        ParallelTopologyId::new(907),
        ParallelGroupId::new(8),
        HostCollectiveLimits {
            max_ranks: members.len(),
            max_elements_per_rank: 4096,
            max_host_bytes: 1 << 20,
        },
        Duration::from_secs(10),
    )
    .unwrap()
}
pub fn endpoints() -> Vec<DecoderTensorCollective> {
    endpoints_for(&[rank(0), rank(1)])
}
pub fn plan(resources: &BoundDecoderResources) -> HybridParallelPlan {
    parallel(resources, 0..4, vec![rank(0), rank(1)], 0, 0)
}
pub fn parallel(
    resources: &BoundDecoderResources,
    range: Range<usize>,
    members: Vec<ParallelRankId>,
    source: usize,
    stage: u32,
) -> HybridParallelPlan {
    let routed = range
        .clone()
        .filter_map(
            |layer| match resources.spec().layers()[layer].feed_forward() {
                FeedForward::Moe(m) => Some((layer, m.router_spec().num_experts())),
                _ => None,
            },
        )
        .collect::<Vec<_>>();
    let attachment = (!routed.is_empty()).then(|| HybridExpertAttachment {
        source_rank: members[source],
        members: vec![rank(101), rank(207)],
        placement: ExpertPlacement::new(routed.into_iter().flat_map(|(l, n)| {
            (0..n).map(move |e| (l, e, rank(if e % 2 == 0 { 101 } else { 207 })))
        }))
        .unwrap(),
    });
    HybridParallelPlan::new(
        resources.spec(),
        range,
        HybridExpertScope {
            topology: ParallelTopologyId::new(907),
            replica: 0,
            stage,
        },
        members,
        attachment,
    )
    .unwrap()
}
pub struct Tensor {
    pub module: HybridTensorCpuSegment,
    pub lifecycle: HybridTensorSequenceLifecycle,
    pub local: LocalTransaction<HybridTensorSequenceState>,
    pub experts: Option<Experts>,
}
impl Tensor {
    pub fn new(
        resources: &BoundDecoderResources,
        plan: HybridParallelPlan,
        lane: usize,
        endpoint: impl StandardTensorCollective + 'static,
    ) -> Self {
        let range = plan.schema().layer_range();
        assert_eq!(endpoint.owner(), plan.tensor_members()[lane]);
        let module = HybridTensorCpuSegment::prepare(
            resources,
            LayerSegmentPlan::new(4, range.clone(), range.start == 0, range.end == 4).unwrap(),
            plan.clone(),
            lane,
            Box::new(endpoint),
            32,
            1 << 20,
        )
        .unwrap();
        assert_eq!(module.layout().owner(), plan.tensor_members()[lane]);
        assert!(module.validate_production_execution().is_err());
        let experts = plan
            .expert_attachment()
            .filter(|a| a.source_rank == module.layout().owner())
            .map(|a| Experts::prepare(resources, a, range));
        let mut lifecycle = module.lifecycle();
        let local = LocalTransaction::new(
            resources,
            module.layout().kv_planes(2, 32).unwrap(),
            vec![lifecycle.create().unwrap()],
        );
        Self {
            module,
            lifecycle,
            local,
            experts,
        }
    }
    pub fn enter(&mut self, id: u64, tokens: &[u32]) -> Work<HybridTensorSequenceState> {
        self.local.enter(&mut self.lifecycle, id, tokens)
    }
    pub fn execute(
        &mut self,
        work: &mut Work<HybridTensorSequenceState>,
        input: SegmentInput,
    ) -> Result<SegmentOutput> {
        let binding = self
            .experts
            .as_mut()
            .map(|ep| {
                self.module.bind_expert_call(
                    self.module.parallel_plan().scope(),
                    self.module.parallel_plan().expert_attachment().unwrap(),
                    ep,
                )
            })
            .transpose()?;
        let reads = self.module.checkpoint_read_counters();
        let result = self.module.execute(
            &work.batch,
            input,
            &mut work.states,
            &mut work.view,
            &[73],
            binding,
            &mut |_| Ok(()),
        );
        assert_eq!(
            reads,
            self.module.checkpoint_read_counters(),
            "no weight reads in forward"
        );
        result
    }
    pub fn snapshot(&self, work: &Work<HybridTensorSequenceState>) -> Capture {
        capture(
            self.module.schema(),
            work.states[0].kv_state().inner(),
            &work.batch,
            &work.view,
        )
    }
    pub fn commit_local(&mut self, work: Work<HybridTensorSequenceState>) {
        self.module
            .validate_frontier(&work.batch, &work.states)
            .unwrap();
        self.local.commit_local(&mut self.lifecycle, work);
    }
    pub fn rollback_local(&mut self, work: Work<HybridTensorSequenceState>) {
        self.local.rollback_local(&mut self.lifecycle, work);
    }
    pub fn release(&mut self) {
        self.local.release(&mut self.lifecycle);
    }
}
pub struct Oracle {
    module: HybridDecoderSegment,
    lifecycle: HybridSequenceLifecycle,
    local: LocalTransaction<HybridDecoderSequenceState>,
}
impl Oracle {
    pub fn new(resources: &BoundDecoderResources) -> Self {
        let module = HybridDecoderSegment::prepare(
            resources,
            LayerSegmentPlan::new(4, 0..4, true, true).unwrap(),
            ExecutionPrecisionPolicy::f32(),
            32,
            1 << 20,
        )
        .unwrap();
        let mut lifecycle = HybridSequenceLifecycle::new(module.schema().clone());
        let local = LocalTransaction::new(
            resources,
            module.schema().kv_planes(2, 32).unwrap(),
            vec![lifecycle.create().unwrap()],
        );
        Self {
            module,
            lifecycle,
            local,
        }
    }
    pub fn step(&mut self, id: u64, tokens: &[u32]) -> (Vec<f32>, Capture) {
        let mut work = self.local.enter(&mut self.lifecycle, id, tokens);
        let output = self
            .module
            .execute(
                &work.batch,
                SegmentInput::Tokens,
                &mut work.states,
                &mut work.view,
            )
            .unwrap();
        let capture = capture(
            self.module.schema(),
            work.states[0].kv_state(),
            &work.batch,
            &work.view,
        );
        self.module
            .validate_frontier(&work.batch, &work.states)
            .unwrap();
        self.local.commit_local(&mut self.lifecycle, work);
        (logits(output), capture)
    }
    pub fn release(&mut self) {
        self.local.release(&mut self.lifecycle);
    }
}
pub fn logits(output: SegmentOutput) -> Vec<f32> {
    match output {
        SegmentOutput::Logits(rows) => rows.values().to_vec(),
        _ => panic!("logits"),
    }
}
pub fn hidden(output: SegmentOutput) -> SegmentInput {
    match output {
        SegmentOutput::Hidden { next_layer, rows } => SegmentInput::Hidden { next_layer, rows },
        _ => panic!("hidden"),
    }
}
pub fn assert_routes(tensor: &Tensor, id: u64, rows: usize) {
    let p = tensor.module.parallel_plan();
    let Some(ep) = &tensor.experts else { return };
    let calls = ep
        .calls
        .iter()
        .filter(|c| c.transaction == tx(id))
        .collect::<Vec<_>>();
    let layers = p
        .layers()
        .iter()
        .filter(|l| l.routed_experts() > 0)
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), layers.len());
    for (call, layer) in calls.iter().zip(layers) {
        assert_eq!(call.source, tensor.module.layout().owner());
        assert_eq!(call.layer, layer.global_layer);
        assert_eq!(call.sequences, vec![73; rows]);
        assert_eq!(call.routes.len(), rows * 2);
        for (ids, weights) in call
            .routes
            .as_chunks::<2>()
            .0
            .iter()
            .zip(call.weights.as_chunks::<2>().0)
        {
            assert!(*ids == [0, 1] || *ids == [1, 0]);
            assert!((weights[0] + weights[1] - 1.).abs() < 1e-6);
            assert!((weights[0] - weights[1]).abs() > 1e-5);
        }
    }
}
