//! Logical packed-row identity for standard TP. No physical KV slots, sequence
//! topology IDs, device pointers or rank-local execution indices participate.

use std::ops::Range;

use ferrule_common::execution::{
    ExecutionIntent, ForwardMode, ForwardPhase, KvCowReplacement, KvPageId, KvWriteSlot,
};
use ferrule_common::{ParallelRankId, Result};
use ferrule_model::TensorRole;
use ferrule_model::decoder::{KvCommitBinding, LogitsPlan, PackedDecoderBatch};
use ferrule_model::transformer::parallel::{
    TensorParallelLinearPartition, TensorParallelLinearPlan,
};
use ferrule_model::transformer::{
    BoundDecoderResources, LayerSegmentPlan, SegmentInput, SegmentOutput,
};

use super::{PipelineExecutionContext, PipelineStageProgram, error};
use crate::parallel::collective::HostCollectiveKind;
use crate::parallel::tensor::decoder_collective::DecoderTensorCollectiveControl;

#[derive(Debug, Clone, PartialEq, Eq)]
struct LogicalSequence {
    session: u64,
    generation: u64,
    phase: ForwardPhase,
    query: Range<usize>,
    context_len: usize,
    sequence_len: usize,
    block_table: Vec<KvPageId>,
}

/// Exact identity, not a hash. Logical page IDs/write slots are included, but
/// physical pool-slot mappings and owner-local sequence generations are not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorBatchIdentity {
    binding: KvCommitBinding,
    intent: ExecutionIntent,
    mode: ForwardMode,
    page_size: usize,
    tokens: Vec<u32>,
    positions: Vec<usize>,
    logical_write_slots: Vec<KvWriteSlot>,
    sequences: Vec<LogicalSequence>,
    row_to_sequence: Vec<usize>,
    sequence_major_rows: Vec<usize>,
    new_pages: Vec<KvPageId>,
    writable_pages: Vec<KvPageId>,
    protected_pages: Vec<KvPageId>,
    cow: Vec<KvCowReplacement>,
    logits: LogitsPlan,
}
impl TensorBatchIdentity {
    pub fn from_packed(
        binding: &KvCommitBinding,
        batch: &PackedDecoderBatch,
        sessions: &[u64],
    ) -> Result<Self> {
        if batch.is_empty() || sessions.len() != batch.sequences().len() {
            return Err(error(
                "tensor batch identity requires the exact packed session order",
            ));
        }
        Ok(Self {
            binding: binding.clone(),
            intent: batch.intent(),
            mode: batch.mode(),
            page_size: batch.page_size(),
            tokens: batch.token_ids().to_vec(),
            positions: batch.positions().to_vec(),
            logical_write_slots: batch.write_slots().to_vec(),
            sequences: batch
                .sequences()
                .iter()
                .zip(sessions)
                .map(|(sequence, &session)| LogicalSequence {
                    session,
                    generation: sequence.page_generation(),
                    phase: sequence.phase(),
                    query: sequence.query(),
                    context_len: sequence.context_len(),
                    sequence_len: sequence.sequence_len(),
                    block_table: sequence.block_table().to_vec(),
                })
                .collect(),
            row_to_sequence: batch.row_to_sequence().to_vec(),
            sequence_major_rows: batch.sequence_major_rows().to_vec(),
            new_pages: batch.new_pages().to_vec(),
            writable_pages: batch.writable_pages().to_vec(),
            protected_pages: batch.protected_pages().to_vec(),
            cow: batch.cow_replacements().to_vec(),
            logits: batch.logits_plan().clone(),
        })
    }
    pub fn binding(&self) -> &KvCommitBinding {
        &self.binding
    }
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }
}

/// One collective-producing operator, in forward order. `elements_per_row` is
/// the padded per-rank transport width, derived from the full operator shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCollectiveSite {
    pub site: u64,
    /// Global decoder layer; `None` denotes the vocabulary head.
    pub layer: Option<usize>,
    pub operator: TensorParallelLinearPlan,
}
impl TensorCollectiveSite {
    pub fn kind(&self) -> HostCollectiveKind {
        match self.operator.partition() {
            TensorParallelLinearPartition::Row => HostCollectiveKind::AllReduceSumF32,
            TensorParallelLinearPartition::Column => HostCollectiveKind::AllGatherF32,
        }
    }
    pub fn elements_per_row(&self) -> usize {
        match self.operator.partition() {
            TensorParallelLinearPartition::Row => self.operator.out_features(),
            TensorParallelLinearPartition::Column => {
                self.operator.out_features().div_ceil(self.operator.ranks())
            }
        }
    }
    pub fn standard(
        resources: &BoundDecoderResources,
        segment: &LayerSegmentPlan,
        tp: usize,
    ) -> Result<Vec<Self>> {
        if tp == 0 {
            return Err(error("zero tensor degree"));
        }
        let mut sites = Vec::new();
        for layer in segment.layers() {
            for role in [TensorRole::AttentionOutput, TensorRole::DenseMlpDown] {
                let binding = resources.require_layer(layer, role)?;
                let [out_features, in_features] = binding.weight().logical_shape() else {
                    return Err(error("tensor collective requires a matrix operator"));
                };
                sites.push(Self {
                    site: binding.id().get(),
                    layer: Some(layer),
                    operator: TensorParallelLinearPlan::new(
                        *out_features,
                        *in_features,
                        tp,
                        TensorParallelLinearPartition::Row,
                    )?,
                });
            }
        }
        if segment.owns_output() {
            sites.push(Self {
                site: resources.require_static(TensorRole::OutputHead)?.id().get(),
                layer: None,
                operator: TensorParallelLinearPlan::new(
                    resources.spec().vocab_size(),
                    resources.spec().hidden_size(),
                    tp,
                    TensorParallelLinearPartition::Column,
                )?,
            });
        }
        Ok(sites)
    }
}

/// Runtime decorator: keeps model math/traits unchanged and checks the actual
/// owner packed batch before entering the physical KV backend or model program.
pub struct TensorScopedProgram<P> {
    inner: P,
    control: DecoderTensorCollectiveControl,
    owner: ParallelRankId,
}
impl<P> TensorScopedProgram<P> {
    pub fn new(inner: P, control: DecoderTensorCollectiveControl, owner: ParallelRankId) -> Self {
        Self {
            inner,
            control,
            owner,
        }
    }
}
impl<P: PipelineStageProgram> PipelineStageProgram for TensorScopedProgram<P> {
    type KvView = P::KvView;
    fn plan(&self) -> &LayerSegmentPlan {
        self.inner.plan()
    }
    fn validate_execution(
        &self,
        binding: &KvCommitBinding,
        batch: &PackedDecoderBatch,
        sessions: &[u64],
    ) -> Result<()> {
        let identity = TensorBatchIdentity::from_packed(binding, batch, sessions)?;
        self.control
            .validate_owner_execution(self.owner, &identity)?;
        self.inner.validate_execution(binding, batch, sessions)
    }
    fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        view: &mut Self::KvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        self.inner.execute(batch, input, view, context)
    }
    fn expert_outstanding(&self) -> usize {
        self.inner.expert_outstanding()
    }
    fn owner_stats(&mut self) -> Result<Vec<crate::parallel::expert::ExpertOwnerStats>> {
        self.inner.owner_stats()
    }
    fn shutdown(&mut self) -> Result<()> {
        self.inner.shutdown()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::collective::HostCollectiveLimits;
    use crate::parallel::tensor::decoder_collective::DecoderTensorCollective;
    use ferrule_common::execution::{
        ExecutionBatch, ExecutionSequence, ExecutionTransactionId, KvBlockId, KvReservationView,
        LogitsRequest, StateSlot,
    };
    use ferrule_common::{
        ParallelGroupId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
    };
    use ferrule_model::decoder::{DecoderKvPageStatus, GenericDecoderSequenceState};
    use ferrule_model::models::qwen3::Qwen3DenseRecipe;
    use ferrule_model::transformer::parallel::TensorParallelCollective;
    use ferrule_model::transformer::{
        DecoderRecipe, StandardTensorCollective, StandardTensorPlacement, StandardTensorPlan,
    };
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    fn tx(value: u64) -> ExecutionTransactionId {
        ExecutionTransactionId::new(value).unwrap()
    }
    fn topology() -> ValidatedParallelTopology {
        ValidatedParallelTopology::new(
            ParallelTopologyId::new(501),
            2,
            ParallelRankId::new(0),
            ParallelismPlan::validated(1, 2, 1, 1, 1, 1).unwrap(),
        )
        .unwrap()
    }
    fn binding() -> KvCommitBinding {
        KvCommitBinding::new(
            tx(1),
            topology().topology_id(),
            topology().participants(),
            17,
        )
        .unwrap()
    }
    fn packed(phase: ForwardPhase, local_slot: u32, generation: u64) -> PackedDecoderBatch {
        let description = super::super::PipelineStageDescription {
            plan: LayerSegmentPlan::new(1, 0..1, true, true).unwrap(),
            config: super::super::PipelineConfig {
                page_size: 2,
                max_pages: 8,
                max_positions: 16,
                max_batch_tokens: 4,
                session_capacity: 2,
                max_parameter_bytes: 1 << 30,
                precision: ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
                max_ack_polls: 2,
            },
            hidden: 4,
            vocabulary: 4,
            kv_heads: 1,
            head_dim: 2,
            expert_group: None,
        };
        let batch = ExecutionBatch::new(
            if phase == ForwardPhase::Prefill {
                ForwardMode::Prefill
            } else {
                ForwardMode::Decode
            },
            vec![3],
            vec![1],
            vec![Some(KvWriteSlot::new(15))],
            vec![LogitsRequest::Full],
            vec![ExecutionSequence::new(
                StateSlot::new(local_slot),
                phase,
                0..1,
                1,
                2,
                0..1,
            )],
            vec![KvBlockId::new(7)],
        );
        let reservation = KvReservationView {
            state_slot: StateSlot::new(9),
            execution_state_slot: StateSlot::new(local_slot),
            positions: 1..2,
            newly_allocated: vec![],
            generation,
            execution_generation: 0,
            cow_replacement: None,
        };
        let states = [
            GenericDecoderSequenceState::with_position(1, (), ()),
            GenericDecoderSequenceState::with_position(1, (), ()),
        ];
        PackedDecoderBatch::lower(
            &batch,
            &[reservation],
            &states,
            &description.execution_capabilities().unwrap(),
            2,
            &|_| DecoderKvPageStatus::Resident,
        )
        .unwrap()
    }
    fn identity() -> TensorBatchIdentity {
        TensorBatchIdentity::from_packed(&binding(), &packed(ForwardPhase::Decode, 0, 5), &[11])
            .unwrap()
    }
    fn peers() -> Vec<DecoderTensorCollective> {
        let spec = Qwen3DenseRecipe::new()
            .build_spec(&serde_json::json!({
                "architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","torch_dtype":"bfloat16",
                "hidden_act":"silu","vocab_size":4,"hidden_size":4,"num_hidden_layers":1,
                "num_attention_heads":2,"num_key_value_heads":2,"head_dim":2,"intermediate_size":8,
                "max_position_embeddings":16,"rms_norm_eps":0.00001,"rope_theta":10000.0,
                "tie_word_embeddings":true,"attention_bias":false,"use_sliding_window":false,
                "attention_dropout":0.0,"use_cache":true,"max_window_layers":1,
                "initializer_range":0.02,"bos_token_id":1,"eos_token_id":2
            }))
            .unwrap();
        let plan = StandardTensorPlan::new(
            &spec,
            (0..2)
                .map(|rank| StandardTensorPlacement {
                    owner: ParallelRankId::new(rank),
                    device: rank as usize,
                })
                .collect(),
        )
        .unwrap();
        let peers = DecoderTensorCollective::new_group(
            &plan,
            topology().topology_id(),
            ParallelGroupId::new(1),
            HostCollectiveLimits {
                max_ranks: 2,
                max_elements_per_rank: 16,
                max_host_bytes: 4096,
            },
            Duration::from_secs(30),
        )
        .unwrap();
        for peer in &peers {
            peer.control()
                .register_sites(
                    peer.owner(),
                    vec![TensorCollectiveSite {
                        site: 77,
                        layer: Some(0),
                        operator: TensorParallelLinearPlan::new(
                            4,
                            4,
                            2,
                            TensorParallelLinearPartition::Row,
                        )
                        .unwrap(),
                    }],
                )
                .unwrap();
        }
        peers
    }
    fn wait(control: &DecoderTensorCollectiveControl) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while control.waiting_members() == 0 {
            assert!(Instant::now() < deadline, "peer did not park");
            std::thread::yield_now();
        }
    }
    #[test]
    fn packed_identity_excludes_owner_indices_but_includes_phase_and_logical_generation() {
        let expected = identity();
        let other = packed(ForwardPhase::Decode, 1, 5);
        assert_ne!(
            other.source_batch(),
            packed(ForwardPhase::Decode, 0, 5).source_batch()
        );
        assert_eq!(
            expected,
            TensorBatchIdentity::from_packed(&binding(), &other, &[11]).unwrap()
        );
        for actual in [
            TensorBatchIdentity::from_packed(
                &binding(),
                &packed(ForwardPhase::Prefill, 0, 5),
                &[11],
            )
            .unwrap(),
            TensorBatchIdentity::from_packed(
                &binding(),
                &packed(ForwardPhase::Decode, 0, 6),
                &[11],
            )
            .unwrap(),
            TensorBatchIdentity::from_packed(
                &binding(),
                &packed(ForwardPhase::Decode, 0, 5),
                &[12],
            )
            .unwrap(),
        ] {
            assert_ne!(actual, expected);
        }
    }
    #[test]
    fn sealed_cohort_rejects_every_logical_layout_axis_and_kv_binding_change() {
        let original = identity();
        let mut variants = Vec::new();
        let mut change = |mutate: fn(&mut TensorBatchIdentity)| {
            let mut value = original.clone();
            mutate(&mut value);
            variants.push(value);
        };
        change(|x| x.tokens[0] = 2);
        change(|x| x.positions[0] = 0);
        change(|x| x.mode = ForwardMode::Prefill);
        change(|x| x.sequences[0].phase = ForwardPhase::Prefill);
        change(|x| x.sequences[0].generation += 1);
        change(|x| x.sequences[0].context_len = 0);
        change(|x| x.sequences[0].query = 1..2);
        change(|x| x.sequences[0].session = 12);
        change(|x| x.row_to_sequence[0] = 1);
        change(|x| x.sequence_major_rows[0] = 1);
        change(|x| x.logical_write_slots[0] = KvWriteSlot::new(14));
        change(|x| x.sequences[0].block_table[0] = KvPageId(8));
        change(|x| {
            x.binding = KvCommitBinding::new(
                tx(1),
                topology().topology_id(),
                topology().participants(),
                18,
            )
            .unwrap()
        });
        change(|x| {
            x.binding = KvCommitBinding::new(
                tx(2),
                topology().topology_id(),
                topology().participants(),
                17,
            )
            .unwrap()
        });
        for actual in variants {
            let peers = peers();
            let control = peers[0].control();
            assert!(
                control
                    .begin(tx(1), Arc::new(AtomicBool::new(false)))
                    .is_err()
            );
            control
                .begin_sealed(original.clone(), Arc::new(AtomicBool::new(false)))
                .unwrap();
            control
                .validate_owner_execution(peers[0].owner(), &original)
                .unwrap();
            assert!(
                control
                    .validate_owner_execution(peers[1].owner(), &actual)
                    .is_err()
            );
            assert!(control.is_failed());
            assert!(control.finish(tx(1)).is_err());
        }
    }
    #[test]
    fn phase_mismatch_wakes_peer_waiting_for_packed_identity_admission() {
        let mut peers = peers();
        let second = peers.pop().unwrap();
        let mut first = peers.pop().unwrap();
        let control = first.control();
        let original = identity();
        control
            .begin_sealed(original.clone(), Arc::new(AtomicBool::new(false)))
            .unwrap();
        control
            .validate_owner_execution(first.owner(), &original)
            .unwrap();
        let waiter = std::thread::spawn(move || {
            first.exchange(tx(1), 77, TensorParallelCollective::Sum, vec![1.0; 4])
        });
        wait(&control);
        let wrong = TensorBatchIdentity::from_packed(
            &binding(),
            &packed(ForwardPhase::Prefill, 1, 5),
            &[11],
        )
        .unwrap();
        let start = Instant::now();
        assert!(
            control
                .validate_owner_execution(second.owner(), &wrong)
                .is_err()
        );
        assert!(waiter.join().unwrap().is_err());
        assert!(start.elapsed() < Duration::from_secs(1));
    }
    #[test]
    fn sealed_site_schedule_rejects_shapes_kind_and_sequence_and_accepts_matching_rows() {
        for case in 0..4 {
            let mut peers = peers();
            let mut second = peers.pop().unwrap();
            let mut first = peers.pop().unwrap();
            let control = first.control();
            let original = identity();
            control
                .begin_sealed(original.clone(), Arc::new(AtomicBool::new(false)))
                .unwrap();
            control
                .validate_owner_execution(first.owner(), &original)
                .unwrap();
            control
                .validate_owner_execution(second.owner(), &original)
                .unwrap();
            let waiter = std::thread::spawn(move || {
                first.exchange(tx(1), 77, TensorParallelCollective::Sum, vec![1.0; 4])
            });
            wait(&control);
            let site = if case == 0 { 78 } else { 77 };
            let kind = if case == 1 {
                TensorParallelCollective::AllGather
            } else {
                TensorParallelCollective::Sum
            };
            let count = if case == 2 { 8 } else { 4 };
            let actual = second.exchange(tx(1), site, kind, vec![2.0; count]);
            let peer = waiter.join().unwrap();
            if case == 3 {
                assert_eq!(actual.unwrap(), vec![3.0; 4]);
                assert_eq!(peer.unwrap(), vec![3.0; 4]);
                control.finish(tx(1)).unwrap();
            } else {
                assert!(actual.is_err());
                assert!(peer.is_err());
                assert!(control.finish(tx(1)).is_err());
            }
        }
    }

    #[test]
    fn operator_registration_binds_global_layer_and_complete_shape_not_just_count() {
        for layer_mismatch in [false, true] {
            let mut peers = peers();
            // Fresh independent group: changing either global layer or input
            // shape is invalid even though the output/transport width is 4.
            let tensor=StandardTensorPlan::new(
                &Qwen3DenseRecipe::new().build_spec(&serde_json::json!({
                    "architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","torch_dtype":"bfloat16",
                    "hidden_act":"silu","vocab_size":4,"hidden_size":4,"num_hidden_layers":1,
                    "num_attention_heads":2,"num_key_value_heads":2,"head_dim":2,"intermediate_size":8,
                    "max_position_embeddings":16,"rms_norm_eps":0.00001,"rope_theta":10000.0,
                    "tie_word_embeddings":true,"attention_bias":false,"use_sliding_window":false,
                    "attention_dropout":0.0,"use_cache":true,"max_window_layers":1,
                    "initializer_range":0.02,"bos_token_id":1,"eos_token_id":2
                })).unwrap(),
                (0..2).map(|rank| StandardTensorPlacement {owner:ParallelRankId::new(rank),device:rank as usize}).collect()
            ).unwrap();
            peers.clear();
            let peers = DecoderTensorCollective::new_group(
                &tensor,
                topology().topology_id(),
                ParallelGroupId::new(1),
                HostCollectiveLimits {
                    max_ranks: 2,
                    max_elements_per_rank: 16,
                    max_host_bytes: 4096,
                },
                Duration::from_secs(30),
            )
            .unwrap();
            let control = peers[0].control();
            control
                .register_sites(
                    peers[0].owner(),
                    vec![TensorCollectiveSite {
                        site: 77,
                        layer: Some(0),
                        operator: TensorParallelLinearPlan::new(
                            4,
                            4,
                            2,
                            TensorParallelLinearPartition::Row,
                        )
                        .unwrap(),
                    }],
                )
                .unwrap();
            assert!(
                control
                    .register_sites(
                        peers[1].owner(),
                        vec![TensorCollectiveSite {
                            site: 77,
                            layer: Some(usize::from(layer_mismatch)),
                            operator: TensorParallelLinearPlan::new(
                                4,
                                if layer_mismatch { 4 } else { 8 },
                                2,
                                TensorParallelLinearPartition::Row
                            )
                            .unwrap(),
                        }]
                    )
                    .is_err()
            );
            assert!(control.is_failed());
        }
    }

    #[test]
    fn formal_pipeline_observer_wakes_collective_on_inflight_cancel_and_owner_failure() {
        use crate::SessionId;
        use crate::parallel::data::PanicQuiescence;
        use crate::parallel::pipeline::{
            PipelineConfig, PipelineParallelExecutor, PipelineStage, PipelineStageDescription,
        };
        use ferrule_model::decoder::{
            CpuKvView, CpuPagedKvPool, DenseLogits, PagedKvBackend, StandardGqaPlanes,
        };
        use std::sync::Mutex;
        use std::sync::atomic::Ordering;
        struct Program {
            plan: LayerSegmentPlan,
            endpoint: DecoderTensorCollective,
            control: DecoderTensorCollectiveControl,
            cancel: Arc<AtomicBool>,
            mode: usize,
        }
        impl PipelineStageProgram for Program {
            type KvView = CpuKvView;
            fn plan(&self) -> &LayerSegmentPlan {
                &self.plan
            }
            fn execute(
                &mut self,
                batch: &PackedDecoderBatch,
                _: SegmentInput,
                _: &mut CpuKvView,
                ctx: PipelineExecutionContext<'_>,
            ) -> Result<SegmentOutput> {
                if self.endpoint.owner().get() == 1 {
                    wait(&self.control);
                    match self.mode {
                        0 => {
                            self.cancel.store(true, Ordering::Release);
                            let deadline = Instant::now() + Duration::from_secs(2);
                            loop {
                                (ctx.check_active)(ctx.transaction)?;
                                assert!(
                                    Instant::now() < deadline,
                                    "pipeline did not observe cancellation"
                                );
                                std::thread::yield_now();
                            }
                        }
                        1 => return Err(error("injected owner failure while peer is waiting")),
                        _ => std::panic::panic_any(PanicQuiescence::Unknown),
                    }
                }
                let output = self.endpoint.exchange(
                    ctx.transaction,
                    77,
                    TensorParallelCollective::Sum,
                    vec![1.0; batch.len() * 4],
                )?;
                Ok(SegmentOutput::Logits(DenseLogits::new(
                    batch.len(),
                    4,
                    output,
                )?))
            }
        }
        for mode in 0..3 {
            let peers = peers();
            let control = peers[0].control();
            let endpoints = Arc::new(Mutex::new(peers.into_iter().map(Some).collect::<Vec<_>>()));
            let cancel = Arc::new(AtomicBool::new(false));
            let factory_cancel = Arc::clone(&cancel);
            let cfg = PipelineConfig {
                page_size: 2,
                max_pages: 8,
                max_positions: 16,
                max_batch_tokens: 4,
                session_capacity: 2,
                max_parameter_bytes: 1 << 30,
                precision: ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
                max_ack_polls: 2,
            };
            let mut pipeline = PipelineParallelExecutor::new_thread_tensor_with_program_inner(
                topology(),
                vec![LayerSegmentPlan::new(1, 0..1, true, true).unwrap()],
                cfg,
                move |owner, _, plan| {
                    let endpoint = endpoints.lock().unwrap()[owner.global.get() as usize]
                        .take()
                        .unwrap();
                    let control = endpoint.control();
                    let program = TensorScopedProgram::new(
                        Program {
                            plan: plan.clone(),
                            endpoint,
                            control: control.clone(),
                            cancel: factory_cancel,
                            mode,
                        },
                        control,
                        owner.global,
                    );
                    let description = PipelineStageDescription {
                        plan,
                        config: cfg,
                        hidden: 4,
                        vocabulary: 4,
                        kv_heads: 1,
                        head_dim: 2,
                        expert_group: None,
                    };
                    let capabilities = description.execution_capabilities()?;
                    let planes = StandardGqaPlanes::new(
                        1,
                        1,
                        2,
                        2,
                        16,
                        ferrule_common::execution::KvElementType::F32,
                    )?;
                    PipelineStage::new(
                        program,
                        PagedKvBackend::new(CpuPagedKvPool::from_strategy(&planes, 8)?),
                        description,
                        capabilities,
                    )
                },
            )
            .unwrap();
            pipeline.tensor_controls = vec![control.clone()];
            let started = Instant::now();
            assert!(
                pipeline
                    .forward_cancellable(SessionId(11), &[1], ForwardPhase::Prefill, &cancel)
                    .is_err()
            );
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "wait expired instead of being woken"
            );
            assert!(control.is_failed());
            assert_eq!(pipeline.outstanding(), 0);
            assert_eq!(pipeline.coordinator().publication_count(), 0);
            assert!(pipeline.is_quarantined());
            cancel.store(false, Ordering::Release);
            let next_transaction = pipeline.next_transaction;
            assert!(
                pipeline
                    .forward(SessionId(11), &[1], ForwardPhase::Prefill)
                    .is_err()
            );
            assert!(pipeline.create_session(SessionId(12)).is_err());
            assert!(pipeline.fork_session(SessionId(11), SessionId(12)).is_err());
            assert_eq!(
                pipeline.next_transaction, next_transaction,
                "no retry admission or replay"
            );
            assert!(pipeline.is_quarantined());
            if mode == 2 {
                assert!(pipeline.is_quarantined());
                assert_eq!(
                    pipeline.coordinator().pending_ranks(tx(1)).unwrap().len(),
                    2
                );
                assert_eq!(pipeline.coordinator().in_use_credits(), 2);
                assert_eq!(pipeline.page_manager().allocated_pages(), 1);
                assert_eq!(pipeline.coordinator().retained_transaction_count(), 1);
                assert!(pipeline.release_session(SessionId(11)).is_err());
                assert!(pipeline.shutdown().is_err());
                assert_eq!(pipeline.page_manager().allocated_pages(), 1);
                assert_eq!(pipeline.coordinator().in_use_credits(), 2);
            } else {
                assert!(pipeline.is_quarantined());
                assert_eq!(pipeline.coordinator().in_use_credits(), 0);
                assert_eq!(pipeline.coordinator().retained_transaction_count(), 0);
                assert_eq!(pipeline.page_manager().allocated_pages(), 0);
                assert!(
                    pipeline
                        .owner_stats()
                        .unwrap()
                        .iter()
                        .all(|owner| owner.kv.active_transactions == 0)
                );
                pipeline.release_session(SessionId(11)).unwrap();
                assert!(pipeline.is_quarantined());
                pipeline.shutdown().unwrap();
                assert!(control.is_failed());
                assert!(pipeline.is_quarantined());
                assert!(
                    pipeline
                        .forward(SessionId(11), &[1], ForwardPhase::Prefill)
                        .is_err()
                );
            }
        }
    }
}
