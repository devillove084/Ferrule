//! PP1 x TP2/TP4 real CUDA standard decoder coverage.
//! Test scaffolding drives the existing physical KV backend; all decoder math
//! and collective computation are production implementations.
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_common::execution::{
    ExecutionBatch, ExecutionSequence, ExecutionTransactionId, ForwardMode, ForwardPhase,
    KvBlockId, KvElementType, KvPageId, KvReservationView, KvWriteSlot, LogitsRequest, StateSlot,
};
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::decoder::{
    CpuPagedKvPool, CudaPagedKvPool, DecoderKvBackend, DecoderKvPageSnapshot, DecoderKvPrepare,
    DecoderKvSequenceCustody, DenseLogits, GenericDecoderOptions, GenericDecoderSequenceState,
    KvEndProgress, PackedDecoderBatch, PagedKvBackend, StandardGqaPlanes,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::models::qwen3::Qwen3DenseRecipe;
use ferrule_model::transformer::{
    Attention, BoundDecoderResources, DecoderLoadOptions, DecoderRecipe, LayerSegmentPlan,
    SegmentInput, SegmentOutput, StandardDecoderSegment, StandardTensorPlacement,
    StandardTensorPlan, SyntheticDecoderRecipe,
};
use ferrule_model::{ModelFamily, TensorRole, WeightSource};

const PAGE: usize = 2;
const POSITIONS: usize = 16;
const LIMIT: u64 = 1024 * 1024;

// Reuse ordinary dense metadata, but explicitly allow F32 checkpoint storage
// for this family-neutral test. Production Qwen3 remains BF16-only.
struct DenseFixtureRecipe {
    bf16: bool,
}
impl DecoderRecipe for DenseFixtureRecipe {
    fn build_spec(
        &self,
        config: &serde_json::Value,
    ) -> Result<
        ferrule_model::transformer::DecoderModelSpec,
        ferrule_model::transformer::DecoderRecipeError,
    > {
        Qwen3DenseRecipe::new().build_spec(config)
    }
    fn build_schema(
        &self,
        config: &serde_json::Value,
    ) -> Result<
        ferrule_model::transformer::StateDictSchema,
        ferrule_model::transformer::DecoderRecipeError,
    > {
        use ferrule_model::nn::{ParameterDType, ParameterSpec};
        let source = Qwen3DenseRecipe::new().build_schema(config)?;
        if self.bf16 {
            return Ok(source);
        }
        let mut builder = ferrule_model::transformer::StateDictSchema::builder();
        for p in source.parameters() {
            let mut parameter = ParameterSpec::new(
                p.id(),
                p.path().clone(),
                ParameterDType::F32,
                p.shape().to_vec(),
                p.residency().clone(),
            )?;
            if let Some(alias) = p.alias_of() {
                parameter = parameter.with_alias(alias);
            }
            builder.register_with_role(parameter, source.role(p.id()).unwrap().clone())?;
        }
        Ok(builder.build()?)
    }
    fn build_name_mapper(
        &self,
        config: &serde_json::Value,
    ) -> Result<
        std::sync::Arc<dyn ferrule_model::transformer::NameMapper>,
        ferrule_model::transformer::DecoderRecipeError,
    > {
        Qwen3DenseRecipe::new().build_name_mapper(config)
    }
}

struct Fixture {
    directory: PathBuf,
    resources: BoundDecoderResources,
}
impl Fixture {
    fn new(dense: bool, tied: bool, bf16: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "ferrule-full-gpu-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let recipe: Box<dyn DecoderRecipe> = if dense {
            Box::new(DenseFixtureRecipe { bf16 })
        } else {
            Box::new(SyntheticDecoderRecipe::new())
        };
        let config = if dense {
            serde_json::json!({
                "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3", "torch_dtype":"bfloat16",
                "hidden_act":"silu", "vocab_size":11, "hidden_size":8, "num_hidden_layers":2,
                "num_attention_heads":8, "num_key_value_heads":4, "head_dim":2, "intermediate_size":13,
                "max_position_embeddings":POSITIONS, "rms_norm_eps":0.00001, "rope_theta":10000.0,
                "tie_word_embeddings":tied, "attention_bias":false, "use_sliding_window":false,
                "attention_dropout":0.0, "use_cache":true, "max_window_layers":2,
                "initializer_range":0.02, "bos_token_id":1, "eos_token_id":2
            })
        } else {
            serde_json::json!({
                "vocab_size":8, "hidden_size":4, "num_attention_heads":2, "num_key_value_heads":1,
                "head_dim":2, "intermediate_size":4, "num_experts":2, "experts_per_token":2,
                "max_position_embeddings":POSITIONS, "rms_norm_eps":0.00001, "rope_theta":10000.0,
                "tie_word_embeddings":tied
            })
        };
        let output = recipe.build(&config).unwrap();
        let slices = output
            .schema()
            .parameters()
            .iter()
            .filter(|p| p.alias_of().is_none())
            .map(|parameter| {
                let seed = parameter
                    .path()
                    .as_str()
                    .bytes()
                    .fold(0usize, |s, b| (s * 31 + usize::from(b)) % 251);
                let values = (0..parameter.shape().iter().product::<usize>()).map(|i| {
                    if parameter.shape().len() == 1 {
                        0.91 + i as f32 * 0.03
                    } else {
                        ((seed + i * 17 + i * i * 3) % 41) as f32 * 0.012 - 0.24
                    }
                });
                let payload = if bf16 {
                    values
                        .flat_map(|v| ferrule_backend::cpu::bf16_rne_word(v).to_le_bytes())
                        .collect::<Vec<_>>()
                } else {
                    values.flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
                };
                let canonical = parameter.path().as_str();
                let name = if !dense {
                    SyntheticDecoderRecipe::external_name(parameter.path())
                } else if canonical == "token_embedding.weight" {
                    "model.embed_tokens.weight".into()
                } else if canonical == "output.weight" {
                    "lm_head.weight".into()
                } else if canonical == "final_norm.weight" {
                    "model.norm.weight".into()
                } else {
                    format!(
                        "model.{}",
                        canonical
                            .replace(".input_norm.", ".input_layernorm.")
                            .replace(".post_attention_norm.", ".post_attention_layernorm.")
                            .replace(".attention.query_norm.", ".self_attn.q_norm.")
                            .replace(".attention.key_norm.", ".self_attn.k_norm.")
                            .replace(".attention.query.", ".self_attn.q_proj.")
                            .replace(".attention.key.", ".self_attn.k_proj.")
                            .replace(".attention.value.", ".self_attn.v_proj.")
                            .replace(".attention.output.", ".self_attn.o_proj.")
                            .replace(".feed_forward.gate.", ".mlp.gate_proj.")
                            .replace(".feed_forward.up.", ".mlp.up_proj.")
                            .replace(".feed_forward.down.", ".mlp.down_proj.")
                    )
                };
                let path = directory.join(format!("{canonical}.bin"));
                std::fs::write(&path, &payload).unwrap();
                CheckpointTensorSlice {
                    name,
                    role: TensorRole::Unknown,
                    path,
                    offset: 0,
                    bytes: payload.len() as u64,
                    dtype: if bf16 {
                        CheckpointDType::Bf16
                    } else {
                        CheckpointDType::F32
                    },
                    shape: parameter.shape().to_vec(),
                }
            })
            .collect::<Vec<_>>();
        let resources = DecoderLoadOptions::new(recipe.as_ref(), &config)
            .bind_slices(slices)
            .unwrap();
        assert_eq!(resources.spec().layers().len(), 2);
        Self {
            directory,
            resources,
        }
    }

    fn planes(&self, layers: usize) -> StandardGqaPlanes {
        let Attention::Gqa(gqa) = self.resources.spec().layers()[0].attention() else {
            panic!("GQA")
        };
        StandardGqaPlanes::new(
            layers,
            gqa.num_kv_heads(),
            gqa.head_dim(),
            PAGE,
            POSITIONS,
            KvElementType::F32,
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn enter<B: DecoderKvBackend<SequenceState = GenericDecoderSequenceState>>(
    fixture: &Fixture,
    backend: &mut B,
    state: &mut GenericDecoderSequenceState,
    id: u64,
    tokens: &[u32],
    cow: Option<ferrule_common::execution::KvCowReplacement>,
) -> (PackedDecoderBatch, B::Transaction, B::KvView) {
    let start = state.core().position();
    let end = start + tokens.len();
    // Intentionally unrelated to physical GPU slots.
    let mut pages = (0..end.div_ceil(PAGE))
        .map(|p| KvPageId(700 + p as u32 * 11))
        .collect::<Vec<_>>();
    if let Some(cow) = cow {
        pages[cow.logical_page] = cow.replacement;
    }
    let slot = StateSlot::new(0);
    let phase = if start == 0 {
        ForwardPhase::Prefill
    } else {
        ForwardPhase::Decode
    };
    let mode = if start == 0 {
        ForwardMode::Prefill
    } else {
        ForwardMode::Decode
    };
    let batch = ExecutionBatch::new(
        mode,
        tokens.to_vec(),
        (start..end).map(|p| p as u32).collect(),
        (start..end)
            .map(|p| {
                Some(KvWriteSlot::new(
                    pages[p / PAGE].0 * PAGE as u32 + (p % PAGE) as u32,
                ))
            })
            .collect(),
        vec![LogitsRequest::Full; tokens.len()],
        vec![ExecutionSequence::new(
            slot,
            phase,
            0..tokens.len() as u32,
            start as u32,
            end as u32,
            0..pages.len() as u32,
        )],
        pages.iter().map(|p| KvBlockId::new(p.0)).collect(),
    );
    let reservations = [KvReservationView {
        state_slot: slot,
        execution_state_slot: slot,
        positions: start..end,
        newly_allocated: pages[start.div_ceil(PAGE)..].to_vec(),
        generation: state.core().generation(),
        execution_generation: state.core().generation(),
        cow_replacement: cow,
    }];
    let options = GenericDecoderOptions::standard_cpu(
        fixture.resources.spec(),
        ModelFamily::Unknown("gpu-test".into()),
        WeightSource::Safetensors,
        PAGE,
        POSITIONS,
        POSITIONS,
        1,
        LIMIT,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap();
    let packed = PackedDecoderBatch::lower(
        &batch,
        &reservations,
        std::slice::from_ref(state),
        &options.capabilities(),
        PAGE,
        &|p| backend.page_status(p),
    )
    .unwrap();
    let custody = packed
        .sequences()
        .iter()
        .map(|s| DecoderKvSequenceCustody {
            source_index: s.state_index(),
            topology_id: s.topology_id(),
            page_state_slot: s.page_state_slot(),
            page_generation: s.page_generation(),
            execution_generation: s.execution_generation(),
            context_len: s.context_len(),
            query_len: s.query_len(),
        })
        .collect::<Vec<_>>();
    let statuses = packed
        .protected_pages()
        .iter()
        .map(|&page| DecoderKvPageSnapshot {
            page,
            status: backend.page_status(page),
        })
        .collect::<Vec<_>>();
    let mut transaction = backend
        .prepare(DecoderKvPrepare {
            transaction: ExecutionTransactionId::new(id).unwrap(),
            sequences: &custody,
            new_pages: packed.new_pages(),
            writable_pages: packed.writable_pages(),
            cow_replacements: packed.cow_replacements(),
            protected_pages: packed.protected_pages(),
            capacity: backend.capacity(),
            page_statuses: &statuses,
        })
        .unwrap();
    backend
        .enter(&mut transaction, &packed, std::slice::from_mut(state))
        .unwrap();
    let view = backend.active_view(&mut transaction).unwrap();
    (packed, transaction, view)
}

fn finish<B: DecoderKvBackend<SequenceState = GenericDecoderSequenceState>>(
    backend: &mut B,
    state: &mut GenericDecoderSequenceState,
    mut tx: B::Transaction,
    rows: usize,
) {
    let binding = state.core().begin_step().unwrap();
    backend.leave(&mut tx).unwrap();
    assert_eq!(
        backend.commit(&mut Some(tx)).unwrap(),
        KvEndProgress::Complete
    );
    state.core_mut().commit_step(binding, rows).unwrap();
}

fn logits(output: SegmentOutput) -> DenseLogits {
    let SegmentOutput::Logits(logits) = output else {
        panic!("expected logits")
    };
    logits
}

fn close(actual: &DenseLogits, expected: &DenseLogits) {
    assert_eq!(actual.rows(), expected.rows());
    assert_eq!(actual.width(), expected.width());
    for (index, (&a, &e)) in actual.values().iter().zip(expected.values()).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 2e-5 + 2e-4 * e.abs(),
            "logit {index}: CUDA={a} CPU={e}"
        );
    }
}

use ferrule_common::{ParallelGroupId, ParallelRankId, ParallelTopologyId};
use ferrule_runtime::parallel::collective::HostCollectiveLimits;
use ferrule_runtime::parallel::tensor::decoder_collective::DecoderTensorCollective;
use std::time::Duration;

fn tensor(fixture: &Fixture, degree: usize) -> StandardTensorPlan {
    StandardTensorPlan::new(
        fixture.resources.spec(),
        (0..degree)
            .map(|device| StandardTensorPlacement {
                // Deliberately unrelated to device/local rank; no PP/KV owner collision.
                owner: ParallelRankId::new(20 + device as u32 * 3),
                device,
            })
            .collect(),
    )
    .unwrap()
}
fn endpoints(plan: &StandardTensorPlan) -> Vec<DecoderTensorCollective> {
    DecoderTensorCollective::new_group(
        plan,
        ParallelTopologyId::new(81),
        ParallelGroupId::new(19),
        HostCollectiveLimits {
            max_ranks: plan.ranks(),
            max_elements_per_rank: 1024,
            max_host_bytes: 1024 * 1024,
        },
        Duration::from_secs(30),
    )
    .unwrap()
}

// Fork the partial tail, execute then roll back one COW, retry, and append to the
// unchanged source. Page IDs are shared logically but slots/KV heads are private.
fn exercise<B: DecoderKvBackend<SequenceState = GenericDecoderSequenceState>>(
    fixture: &Fixture,
    backend: &mut B,
    mut forward: impl FnMut(&PackedDecoderBatch, &mut B::KvView) -> SegmentOutput,
) -> Vec<DenseLogits> {
    use ferrule_common::execution::KvCowReplacement;
    let mut source = GenericDecoderSequenceState::new((), ());
    let (batch, tx, mut view) = enter(fixture, backend, &mut source, 1, &[1, 2, 3], None);
    let mut output = vec![logits(forward(&batch, &mut view))];
    drop(view);
    finish(backend, &mut source, tx, 3);
    let mut branch = source.logical_fork().unwrap();
    let cow = KvCowReplacement {
        logical_page: 1,
        source: KvPageId(711),
        replacement: KvPageId(999),
    };
    let capacity = backend.capacity().free_pages;
    let (batch, mut tx, mut view) = enter(fixture, backend, &mut branch, 2, &[7], Some(cow));
    let _provisional = forward(&batch, &mut view);
    drop(view);
    backend.leave(&mut tx).unwrap();
    assert_eq!(
        backend.rollback(&mut Some(tx)).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(backend.capacity().free_pages, capacity);
    assert_eq!(branch.core().position(), 3);
    let (batch, tx, mut view) = enter(fixture, backend, &mut branch, 3, &[4], Some(cow));
    output.push(logits(forward(&batch, &mut view)));
    drop(view);
    finish(backend, &mut branch, tx, 1);
    assert_eq!(source.core().position(), 3);
    for (id, token) in [(4, 5), (5, 6)] {
        let (batch, tx, mut view) = enter(fixture, backend, &mut source, id, &[token], None);
        output.push(logits(forward(&batch, &mut view)));
        drop(view);
        finish(backend, &mut source, tx, 1);
    }
    assert_eq!(branch.core().position(), 4);
    assert_eq!(backend.capacity().active_transactions, 0);
    backend
        .release(&[KvPageId(700), KvPageId(711), KvPageId(722), KvPageId(999)])
        .unwrap();
    assert_eq!(
        backend.capacity().free_pages,
        backend.capacity().physical_pages
    );
    backend.shutdown().unwrap();
    output
}

fn cpu(fixture: &Fixture) -> Vec<DenseLogits> {
    let segment = StandardDecoderSegment::prepare(
        &fixture.resources,
        LayerSegmentPlan::new(2, 0..2, true, true).unwrap(),
        ExecutionPrecisionPolicy::f32(),
        POSITIONS,
        LIMIT,
    )
    .unwrap();
    let mut backend =
        PagedKvBackend::new(CpuPagedKvPool::from_strategy(&fixture.planes(2), 8).unwrap());
    exercise(fixture, &mut backend, |batch, view| {
        segment.execute(batch, SegmentInput::Tokens, view).unwrap()
    })
}
fn gpu(
    fixture: &Fixture,
    tp: Option<(StandardTensorPlan, usize, DecoderTensorCollective)>,
) -> (Vec<DenseLogits>, usize) {
    let ordinal = tp.as_ref().map_or(0, |(_, local, _)| *local);
    let ops = Rc::new(CudaOperators::new_on_device(ordinal).unwrap());
    let plan = LayerSegmentPlan::new(2, 0..2, true, true).unwrap();
    let planes = tp.as_ref().map_or_else(
        || fixture.planes(2),
        |(tp, _, _)| tp.kv_planes(PAGE, POSITIONS).unwrap(),
    );
    let mut segment = match tp {
        Some((tp, rank, collective)) => StandardDecoderSegment::prepare_cuda_tensor(
            &fixture.resources,
            plan,
            tp,
            ParallelRankId::new(rank as u32),
            POSITIONS,
            LIMIT,
            Rc::clone(&ops),
            Box::new(collective),
        )
        .unwrap(),
        None => StandardDecoderSegment::prepare_cuda(
            &fixture.resources,
            plan,
            ExecutionPrecisionPolicy::f32(),
            POSITIONS,
            LIMIT,
            Rc::clone(&ops),
        )
        .unwrap(),
    };
    let bytes = segment.operators().resident_parameter_bytes();
    let mut backend =
        PagedKvBackend::new(CudaPagedKvPool::from_schema(Rc::clone(&ops), &planes, 8).unwrap());
    let output = exercise(fixture, &mut backend, |batch, view| {
        let output = segment.execute(batch, SegmentInput::Tokens, view).unwrap();
        assert_eq!(segment.operators().resident_parameter_bytes(), bytes);
        output
    });
    segment.quiesce().unwrap();
    assert!(!segment.needs_quarantine());
    (output, bytes)
}

#[test]
#[ignore = "requires four CUDA GPUs; FERRULE_CUDA_ARCH=sm_86 and --test-threads=1"]
fn pp1_tp2_tp4_logits_decode_cow_match_cpu_and_gpu_tp1() {
    for (tied, bf16) in [(true, true), (false, true), (false, false)] {
        let fixture = Fixture::new(true, tied, bf16);
        let expected = cpu(&fixture);
        let (tp1, full_bytes) = gpu(&fixture, None);
        for (a, e) in tp1.iter().zip(&expected) {
            close(a, e);
        }
        for degree in [2, 4] {
            let plan = tensor(&fixture, degree);
            let threads = endpoints(&plan)
                .into_iter()
                .enumerate()
                .map(|(rank, endpoint)| {
                    let plan = plan.clone();
                    std::thread::spawn(move || {
                        let fixture = Fixture::new(true, tied, bf16);
                        gpu(&fixture, Some((plan, rank, endpoint)))
                    })
                })
                .collect::<Vec<_>>();
            let results = threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>();
            for (rank, (output, bytes)) in results.iter().enumerate() {
                assert!(
                    *bytes < full_bytes,
                    "rank must retain only local projection weights"
                );
                assert_eq!(output.len(), expected.len());
                for ((a, e), one) in output.iter().zip(&expected).zip(&tp1) {
                    close(a, e);
                    close(a, one);
                }
                eprintln!(
                    "PASS sm86 PP1 TP={degree} rank={rank} tied={tied} bf16_weights={bf16} resident={bytes}/{full_bytes}; prefill/decode/COW/rollback/release"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU; rejects before a collective can block"]
fn tp_rejects_foreign_placement_owner_and_unsharded_or_bf16_kv() {
    // Even missing payloads must not obscure the placement admission error.
    {
        let fixture = Fixture::new(true, true, true);
        let plan = tensor(&fixture, 2);
        let mut peers = endpoints(&plan);
        let endpoint = peers.remove(0);
        std::fs::remove_dir_all(&fixture.directory).unwrap();
        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let error = StandardDecoderSegment::prepare_cuda_tensor(
            &fixture.resources,
            LayerSegmentPlan::new(2, 0..2, true, true).unwrap(),
            plan,
            ParallelRankId::new(1),
            POSITIONS,
            LIMIT,
            ops,
            Box::new(endpoint),
        )
        .unwrap_err();
        assert!(error.to_string().contains("placement mismatch"), "{error}");
    }
    for case in 0..3 {
        let fixture = Fixture::new(true, true, true);
        let plan = tensor(&fixture, 2);
        let mut peers = endpoints(&plan);
        let endpoint = peers.remove(0);
        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let mut segment = StandardDecoderSegment::prepare_cuda_tensor(
            &fixture.resources,
            LayerSegmentPlan::new(2, 0..2, true, true).unwrap(),
            plan.clone(),
            ParallelRankId::new(0),
            POSITIONS,
            LIMIT,
            Rc::clone(&ops),
            Box::new(endpoint),
        )
        .unwrap();
        let planes = match case {
            0 => fixture.planes(2), // Full KV heads cannot be reused by a shard.
            1 => plan.kv_planes(PAGE, POSITIONS).unwrap(),
            _ => StandardGqaPlanes::new(2, 2, 2, PAGE, POSITIONS, KvElementType::Bf16).unwrap(),
        };
        let pool_ops = if case == 1 {
            Rc::new(CudaOperators::new_on_device(0).unwrap())
        } else {
            Rc::clone(&ops)
        };
        let pool = CudaPagedKvPool::from_schema(pool_ops, &planes, 8);
        if case == 2 {
            let error = pool
                .err()
                .expect("BF16 KV must be rejected at construction");
            assert!(error.to_string().contains("only F32"), "{error}");
            continue;
        }
        let mut backend = PagedKvBackend::new(pool.unwrap());
        let mut state = GenericDecoderSequenceState::new((), ());
        let (batch, mut tx, mut view) =
            enter(&fixture, &mut backend, &mut state, 1, &[1, 2, 3], None);
        let error = segment
            .execute(&batch, SegmentInput::Tokens, &mut view)
            .unwrap_err();
        eprintln!("rejected KV case={case}: {error}");
        drop(view);
        backend.leave(&mut tx).unwrap();
        assert_eq!(
            backend.rollback(&mut Some(tx)).unwrap(),
            KvEndProgress::Complete
        );
        assert_eq!(backend.capacity().free_pages, 8);
        assert_eq!(state.core().position(), 0);
        let (batch, mut tx, mut view) = enter(&fixture, &mut backend, &mut state, 2, &[1], None);
        let error = segment
            .execute(&batch, SegmentInput::Tokens, &mut view)
            .unwrap_err();
        assert!(error.to_string().contains("lifetime has failed"), "{error}");
        drop(view);
        backend.leave(&mut tx).unwrap();
        assert_eq!(
            backend.rollback(&mut Some(tx)).unwrap(),
            KvEndProgress::Complete
        );
        segment.quiesce().unwrap();
        assert!(!segment.needs_quarantine());
        backend.shutdown().unwrap();
    }
}

#[path = "support/standard_tensor_pipeline.rs"]
mod pipeline;
