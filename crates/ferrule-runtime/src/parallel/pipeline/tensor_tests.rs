//! CPU fault injection at the production PP×TP worker/transport boundary.
//! No test-owned pipeline loop, transaction coordinator, or worker threads.
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::tensor_scope::{TensorCollectiveSite, TensorScopedProgram};
use crate::parallel::collective::HostCollectiveLimits;
use crate::parallel::data::PanicQuiescence;
use crate::parallel::pipeline::{
    PipelineConfig, PipelineExecutionContext, PipelineParallelExecutor, PipelineStage,
    PipelineStageDescription, PipelineStageProgram,
};
use crate::parallel::tensor::decoder_collective::DecoderTensorCollective;
use crate::{SessionId, TransactionState};
use ferrule_common::execution::{ForwardPhase, KvElementType};
use ferrule_common::{
    Error, ParallelGroupId, ParallelRankId, ParallelTopologyId, ParallelismPlan, Result,
    ValidatedParallelTopology,
};
use ferrule_model::decoder::{
    CpuKvView, CpuPagedKvPool, DenseLogits, PackedDecoderBatch, PagedKvBackend, StandardGqaPlanes,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::models::qwen3::Qwen3DenseRecipe;
use ferrule_model::transformer::parallel::TensorParallelCollective;
use ferrule_model::transformer::parallel::{
    TensorParallelLinearPartition, TensorParallelLinearPlan,
};
use ferrule_model::transformer::{
    DecoderRecipe, HostRows, LayerSegmentPlan, RowsDType, RowsShape, SegmentInput, SegmentOutput,
    StandardTensorCollective, StandardTensorPlacement, StandardTensorPlan,
};

fn error(message: &str) -> Error {
    Error::Execution {
        message: message.into(),
    }
}
fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: 2,
        max_pages: 8,
        max_positions: 16,
        max_batch_tokens: 8,
        session_capacity: 2,
        max_parameter_bytes: 1 << 30,
        precision: ExecutionPrecisionPolicy::f32(),
        max_ack_polls: 2,
    }
}
struct Program {
    plan: LayerSegmentPlan,
    local: usize,
    endpoint: DecoderTensorCollective,
    fault: Arc<AtomicUsize>,
    external_cancel: Arc<AtomicBool>,
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
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        let fault = self.fault.load(Ordering::Acquire);
        if fault == 1 && self.local == 0 {
            return Err(error("injected fenced failure"));
        }
        if fault == 2 && self.local == 0 {
            std::panic::panic_any(PanicQuiescence::Unknown);
        }
        if fault == 3 && self.local == 0 {
            self.external_cancel.store(true, Ordering::Release);
        }
        if (1..=3).contains(&fault) {
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                (context.check_active)(context.transaction)?;
                assert!(
                    Instant::now() < deadline,
                    "batch must propagate failure/cancellation to every admitted peer"
                );
                std::thread::sleep(Duration::from_micros(100));
            }
        }
        let values = self.endpoint.exchange(
            context.transaction,
            1,
            TensorParallelCollective::Sum,
            vec![self.local as f32 + 1.0; batch.len() * 4],
        )?;
        let mut values = values;
        if fault == 4 && self.local == 1 {
            values[0] += 1.0;
        }
        if self.plan.owns_output() {
            Ok(SegmentOutput::Logits(DenseLogits::new(
                batch.len(),
                4,
                values,
            )?))
        } else {
            Ok(SegmentOutput::Hidden {
                next_layer: self.plan.layers().end,
                rows: HostRows::new(
                    RowsShape::new(batch.len(), 4)?,
                    RowsDType::F32,
                    None,
                    values,
                )?,
            })
        }
    }
}
fn build(
    pp: usize,
    tp: usize,
    fault: Arc<AtomicUsize>,
    cancel: Arc<AtomicBool>,
) -> PipelineParallelExecutor {
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(501),
        (pp * tp) as u32,
        ParallelRankId::new(0),
        ParallelismPlan::validated(1, tp, 1, 1, 1, pp).unwrap(),
    )
    .unwrap();
    let spec = Qwen3DenseRecipe::new()
        .build_spec(&serde_json::json!({
            "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3", "torch_dtype":"bfloat16",
            "hidden_act":"silu", "vocab_size":4, "hidden_size":4, "num_hidden_layers":2,
            "num_attention_heads":4, "num_key_value_heads":4, "head_dim":2, "intermediate_size":8,
            "max_position_embeddings":16, "rms_norm_eps":0.00001, "rope_theta":10000.0,
            "tie_word_embeddings":true, "attention_bias":false, "use_sliding_window":false,
            "attention_dropout":0.0, "use_cache":true, "max_window_layers":2,
            "initializer_range":0.02, "bos_token_id":1, "eos_token_id":2
        }))
        .unwrap();
    let mut endpoints = Vec::new();
    let mut controls = Vec::new();
    for stage in 0..pp {
        let plan = StandardTensorPlan::new(
            &spec,
            (0..tp)
                .map(|local| StandardTensorPlacement {
                    owner: ParallelRankId::new((stage * tp + local) as u32),
                    device: stage * tp + local,
                })
                .collect(),
        )
        .unwrap();
        let group = DecoderTensorCollective::new_group(
            &plan,
            topology.topology_id(),
            ParallelGroupId::new(stage as u32 + 1),
            HostCollectiveLimits {
                max_ranks: tp,
                max_elements_per_rank: 32,
                max_host_bytes: 8192,
            },
            Duration::from_millis(200),
        )
        .unwrap();
        let control = group[0].control();
        for peer in &group {
            control
                .register_sites(
                    peer.owner(),
                    vec![TensorCollectiveSite {
                        site: 1,
                        layer: Some(stage * 2 / pp),
                        operator: TensorParallelLinearPlan::new(
                            4,
                            4,
                            tp,
                            TensorParallelLinearPartition::Row,
                        )
                        .unwrap(),
                    }],
                )
                .unwrap();
        }
        controls.push(control);
        endpoints.extend(group.into_iter().map(Some));
    }
    let endpoints = Arc::new(Mutex::new(endpoints));
    let plans = (0..pp)
        .map(|stage| {
            LayerSegmentPlan::new(
                2,
                stage * 2 / pp..(stage + 1) * 2 / pp,
                stage == 0,
                stage + 1 == pp,
            )
            .unwrap()
        })
        .collect();
    let mut pipeline = PipelineParallelExecutor::new_thread_tensor_with_program_inner(
        topology,
        plans,
        config(),
        move |owner, coordinate, plan| {
            let planes =
                StandardGqaPlanes::new(plan.layer_count(), 4 / tp, 2, 2, 16, KvElementType::F32)?;
            let backend = PagedKvBackend::new(CpuPagedKvPool::from_strategy(&planes, 8)?);
            let description = PipelineStageDescription {
                plan: plan.clone(),
                config: config(),
                physical_pages: config().max_pages,
                hidden: 4,
                vocabulary: 4,
                kv_heads: 4 / tp,
                head_dim: 2,
                expert_group: None,
            };
            let capabilities = description.execution_capabilities()?;
            let program = Program {
                plan,
                local: coordinate.tensor() as usize,
                endpoint: endpoints.lock().unwrap()[owner.global.get() as usize]
                    .take()
                    .unwrap(),
                fault,
                external_cancel: cancel,
            };
            let control = program.endpoint.control();
            PipelineStage::new(
                TensorScopedProgram::new(program, control, owner.global),
                backend,
                description,
                capabilities,
            )
        },
    )
    .unwrap();
    pipeline.tensor_controls = controls;
    pipeline
}
fn drained(p: &PipelineParallelExecutor) {
    assert!(!p.is_quarantined());
    assert_eq!(p.outstanding(), 0);
    assert_eq!(p.coordinator().in_use_credits(), 0);
    assert_eq!(p.coordinator().retained_transaction_count(), 0);
    for owner in p.owner_stats().unwrap() {
        assert_eq!(owner.kv.active_transactions, 0);
    }
}
#[test]
fn thread_pp_tp_collective_dispatch_and_all_owner_commit_cancel_fork() {
    for (pp, tp) in [(1, 2), (1, 4), (2, 2)] {
        let flag = Arc::new(AtomicBool::new(false));
        let mut p = build(pp, tp, Arc::new(AtomicUsize::new(0)), Arc::clone(&flag));
        let output = p
            .forward(SessionId(1), &[1, 2, 3], ForwardPhase::Prefill)
            .unwrap();
        assert_eq!(output.logits.values(), vec![(tp * (tp + 1) / 2) as f32; 12]);
        drained(&p);
        p.fork_session(SessionId(1), SessionId(2)).unwrap();
        let pages = p.page_manager().allocated_pages();
        assert!(
            p.forward_observed(
                SessionId(2),
                &[2],
                ForwardPhase::Decode,
                &flag,
                |progress| {
                    if progress.state == TransactionState::Preparing {
                        assert_eq!(progress.pending_ranks.len(), pp * tp);
                        flag.store(true, Ordering::Release);
                    }
                }
            )
            .is_err()
        );
        assert_eq!(p.page_manager().allocated_pages(), pages);
        assert_eq!(p.coordinator().publication_count(), 1);
        drained(&p);
        flag.store(false, Ordering::Release);
        p.forward(SessionId(2), &[2], ForwardPhase::Decode).unwrap();
        drained(&p);
        p.shutdown().unwrap();
    }
}
#[test]
fn batch_failure_and_inflight_cancel_drain_but_poisoned_lifetime_stays_unavailable() {
    for mode in [1, 3, 4] {
        let fault = Arc::new(AtomicUsize::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        let mut p = build(2, 2, Arc::clone(&fault), Arc::clone(&cancel));
        p.forward(SessionId(1), &[1, 2, 3], ForwardPhase::Prefill)
            .unwrap();
        let pages = p.page_manager().allocated_pages();
        fault.store(mode, Ordering::Release);
        let start = Instant::now();
        assert!(
            p.forward_cancellable(SessionId(1), &[2], ForwardPhase::Decode, &cancel)
                .is_err()
        );
        assert!(start.elapsed() < Duration::from_secs(2));
        assert_eq!(p.coordinator().publication_count(), 1);
        assert_eq!(p.page_manager().allocated_pages(), pages);
        assert!(p.is_quarantined());
        assert_eq!(p.outstanding(), 0);
        assert_eq!(p.coordinator().in_use_credits(), 0);
        assert_eq!(p.coordinator().retained_transaction_count(), 0);
        assert!(
            p.owner_stats()
                .unwrap()
                .iter()
                .all(|owner| owner.kv.active_transactions == 0)
        );
        fault.store(0, Ordering::Release);
        cancel.store(false, Ordering::Release);
        assert!(p.forward(SessionId(1), &[2], ForwardPhase::Decode).is_err());
        assert!(p.create_session(SessionId(2)).is_err());
        assert!(p.fork_session(SessionId(1), SessionId(2)).is_err());
        assert_eq!(p.coordinator().publication_count(), 1);
        assert_eq!(p.page_manager().allocated_pages(), pages);
        assert!(p.is_quarantined());
        p.release_session(SessionId(1)).unwrap();
        assert!(p.is_quarantined());
        p.shutdown().unwrap();
        assert!(p.is_quarantined());
    }
}
#[test]
fn unknown_owner_retains_entire_cohort_and_logical_custody() {
    let mut p = build(
        2,
        2,
        Arc::new(AtomicUsize::new(2)),
        Arc::new(AtomicBool::new(false)),
    );
    assert!(
        p.forward(SessionId(1), &[1, 2, 3], ForwardPhase::Prefill)
            .is_err()
    );
    assert!(p.is_quarantined());
    assert_eq!(p.outstanding(), 0);
    assert_eq!(p.coordinator().publication_count(), 0);
    assert_eq!(p.coordinator().retained_transaction_count(), 1);
    assert_eq!(p.coordinator().in_use_credits(), 4);
    assert_eq!(
        p.coordinator()
            .pending_ranks(ferrule_common::execution::ExecutionTransactionId::new(1).unwrap())
            .unwrap()
            .len(),
        4
    );
    assert!(p.page_manager().allocated_pages() > 0);
    assert!(
        p.forward(SessionId(1), &[1], ForwardPhase::Prefill)
            .is_err()
    );
    assert!(p.shutdown().is_err());
}
