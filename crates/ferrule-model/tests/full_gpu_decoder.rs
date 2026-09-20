//! Opt-in real CUDA decoder checks. Oracles use the shared CPU decoder, never
//! reimplemented transformer math. No model download or NAS access is needed.
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
use ferrule_model::nn::ParameterResidency;
use ferrule_model::transformer::{
    Attention, BoundDecoderResources, CudaStandardDecoderOperators, DecoderLoadOptions,
    DecoderRecipe, DeviceSegmentInput, DeviceSegmentOutput, HostRows, LayerSegmentPlan,
    OperatorProgress, PreparedEmbedding, PreparedLinear, Rows, RowsDType, RowsDevice, RowsShape,
    SegmentInput, SegmentOutput, StandardDecoderOperators, StandardDecoderSegment,
    StateDictMaterializer, SyntheticDecoderRecipe,
};
use ferrule_model::{ModelFamily, TensorRole, WeightSource};

const PAGE: usize = 2;
const POSITIONS: usize = 16;
const LIMIT: u64 = 1024 * 1024;

struct Fixture {
    directory: PathBuf,
    resources: BoundDecoderResources,
}
impl Fixture {
    fn new(dense: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "ferrule-full-gpu-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let recipe: Box<dyn DecoderRecipe> = if dense {
            Box::new(Qwen3DenseRecipe::new())
        } else {
            Box::new(SyntheticDecoderRecipe::new())
        };
        let config = if dense {
            serde_json::json!({
                "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3", "torch_dtype":"bfloat16",
                "hidden_act":"silu", "vocab_size":8, "hidden_size":4, "num_hidden_layers":2,
                "num_attention_heads":4, "num_key_value_heads":2, "head_dim":2, "intermediate_size":6,
                "max_position_embeddings":POSITIONS, "rms_norm_eps":0.00001, "rope_theta":10000.0,
                "tie_word_embeddings":true, "attention_bias":false, "use_sliding_window":false,
                "attention_dropout":0.0, "use_cache":true, "max_window_layers":2,
                "initializer_range":0.02, "bos_token_id":1, "eos_token_id":2
            })
        } else {
            serde_json::json!({
                "vocab_size":8, "hidden_size":4, "num_attention_heads":2, "num_key_value_heads":1,
                "head_dim":2, "intermediate_size":4, "num_experts":2, "experts_per_token":2,
                "max_position_embeddings":POSITIONS, "rms_norm_eps":0.00001, "rope_theta":10000.0,
                "tie_word_embeddings":true
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
                let payload = if dense {
                    values
                        .flat_map(|v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                        .collect::<Vec<_>>()
                } else {
                    values.flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
                };
                let canonical = parameter.path().as_str();
                let name = if !dense {
                    SyntheticDecoderRecipe::external_name(parameter.path())
                } else if canonical == "token_embedding.weight" {
                    "model.embed_tokens.weight".into()
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
                    dtype: if dense {
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
) -> (PackedDecoderBatch, B::Transaction, B::KvView) {
    let start = state.core().position();
    let end = start + tokens.len();
    // Intentionally unrelated to physical GPU slots.
    let pages = (0..end.div_ceil(PAGE))
        .map(|p| KvPageId(700 + p as u32 * 11))
        .collect::<Vec<_>>();
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
        cow_replacement: None,
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

fn run_full(dense: bool) {
    let fixture = Fixture::new(dense);
    let ops = Rc::new(CudaOperators::new().unwrap());
    let plan = LayerSegmentPlan::new(2, 0..2, true, true).unwrap();
    let cpu = StandardDecoderSegment::prepare(
        &fixture.resources,
        plan.clone(),
        ExecutionPrecisionPolicy::f32(),
        POSITIONS,
        LIMIT,
    )
    .unwrap();
    let mut gpu = StandardDecoderSegment::prepare_cuda(
        &fixture.resources,
        plan,
        ExecutionPrecisionPolicy::f32(),
        POSITIONS,
        LIMIT,
        Rc::clone(&ops),
    )
    .unwrap();
    let planes = fixture.planes(2);
    let mut cpu_kv = PagedKvBackend::new(CpuPagedKvPool::from_strategy(&planes, 8).unwrap());
    let mut gpu_kv =
        PagedKvBackend::new(CudaPagedKvPool::from_schema(Rc::clone(&ops), &planes, 8).unwrap());
    let mut cpu_state = GenericDecoderSequenceState::new((), ());
    let mut gpu_state = GenericDecoderSequenceState::new((), ());
    let mut stable_bytes = 0;
    for (step, tokens) in [vec![1, 2, 3], vec![4], vec![5]].iter().enumerate() {
        let (batch, tx, mut kv) = enter(
            &fixture,
            &mut cpu_kv,
            &mut cpu_state,
            step as u64 + 1,
            tokens,
        );
        let expected = logits(cpu.execute(&batch, SegmentInput::Tokens, &mut kv).unwrap());
        drop(kv);
        finish(&mut cpu_kv, &mut cpu_state, tx, tokens.len());
        let (batch, tx, mut kv) = enter(
            &fixture,
            &mut gpu_kv,
            &mut gpu_state,
            step as u64 + 1,
            tokens,
        );
        let output = gpu
            .execute_cuda_bound(&batch, DeviceSegmentInput::Tokens, &mut kv)
            .unwrap();
        let DeviceSegmentOutput::Logits(ref rows) = output else {
            panic!("GPU logits")
        };
        assert_eq!(rows.dtype(), RowsDType::F32);
        assert_eq!(
            rows.device(),
            RowsDevice::Cuda {
                ordinal: ops.device_ordinal() as u32
            }
        );
        let actual = logits(gpu.download_output(output).unwrap());
        close(&actual, &expected);
        drop(kv);
        finish(&mut gpu_kv, &mut gpu_state, tx, tokens.len());
        let bytes = gpu.operators().resident_parameter_bytes();
        if step == 0 {
            stable_bytes = bytes;
        } else {
            assert_eq!(bytes, stable_bytes);
        }
    }
    assert!(!gpu.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA device and native provider"]
fn full_two_layer_qwen_bf16_weights_f32_activations_prefill_decode() {
    run_full(true);
}

#[test]
#[ignore = "requires CUDA device and native provider"]
fn full_two_layer_moe_f32_weights_prefill_decode() {
    run_full(false);
}

#[test]
#[ignore = "requires CUDA device and native provider"]
fn assigned_layer_scope_and_foreign_owner_or_image_are_rejected() {
    let fixture = Fixture::new(true);
    let ops = Rc::new(CudaOperators::new().unwrap());
    let plan = LayerSegmentPlan::new(2, 1..2, false, false).unwrap();
    // Destroy every unowned payload. A layers-only stage must not read them.
    for parameter in fixture.resources.state_dict().parameters() {
        if !matches!(
            parameter.residency(),
            ParameterResidency::Layer { layer: 1 }
        ) && !parameter.is_alias()
        {
            let _ =
                std::fs::remove_file(fixture.directory.join(format!("{}.bin", parameter.path())));
        }
    }
    let mut stage = StandardDecoderSegment::prepare_cuda(
        &fixture.resources,
        plan,
        ExecutionPrecisionPolicy::f32(),
        POSITIONS,
        LIMIT,
        Rc::clone(&ops),
    )
    .unwrap();
    assert!(
        stage
            .parameters()
            .iter()
            .all(|p| matches!(p.residency(), ParameterResidency::Layer { layer: 1 }))
    );
    let mut kv = PagedKvBackend::new(
        CudaPagedKvPool::from_schema(Rc::clone(&ops), &fixture.planes(1), 8).unwrap(),
    );
    let mut state = GenericDecoderSequenceState::new((), ());
    let (batch, tx, mut view) = enter(&fixture, &mut kv, &mut state, 1, &[1, 2]);
    let input = HostRows::new(
        RowsShape::new(2, 4).unwrap(),
        RowsDType::F32,
        None,
        vec![0.2; 8],
    )
    .unwrap();
    let output = stage
        .execute_cuda_bound(
            &batch,
            DeviceSegmentInput::Hidden {
                next_layer: 1,
                rows: Rows::Host(input),
            },
            &mut view,
        )
        .unwrap();
    assert!(matches!(
        output,
        DeviceSegmentOutput::Hidden {
            next_layer: 2,
            rows: Rows::Cuda(_)
        }
    ));
    drop(view);
    finish(&mut kv, &mut state, tx, 2);

    let source = Fixture::new(true);
    let foreign = Fixture::new(true);
    let mut operators = CudaStandardDecoderOperators::new(
        Rc::clone(&ops),
        ExecutionPrecisionPolicy::f32(),
        source.resources.state_dict().parameters(),
    )
    .unwrap();
    let embedding = |fixture: &Fixture| {
        let materializer = StateDictMaterializer::new(LIMIT).unwrap();
        let binding = fixture
            .resources
            .require_static_shape(TensorRole::TokenEmbedding, &[8, 4])
            .unwrap();
        PreparedEmbedding::new(
            PreparedLinear::from_parameter(
                materializer.static_parameter(binding).unwrap(),
                TensorRole::TokenEmbedding,
            )
            .unwrap(),
        )
    };
    assert!(matches!(
        operators
            .embedding(&embedding(&source), &[1], None)
            .unwrap(),
        OperatorProgress::Ready(_)
    ));
    assert!(
        operators
            .embedding(&embedding(&foreign), &[1], None)
            .is_err()
    );
    let foreign_ops = CudaOperators::new_on_device(ops.device_ordinal()).unwrap();
    let rows = ferrule_model::transformer::CudaRows::f32(
        RowsShape::new(1, 4).unwrap(),
        None,
        foreign_ops.zero_f32_buffer(4).unwrap(),
    )
    .unwrap();
    assert!(operators.bind_rows(Rows::Cuda(rows)).is_err());
    assert!(
        CudaStandardDecoderOperators::new(ops, ExecutionPrecisionPolicy::bf16_compatibility(), &[])
            .is_err()
    );
}

#[test]
#[ignore = "requires at least two CUDA devices"]
fn cuda_rows_report_the_actual_nonzero_ordinal() {
    let ops = Rc::new(CudaOperators::new_on_device(1).unwrap());
    let mut operators =
        CudaStandardDecoderOperators::new(Rc::clone(&ops), ExecutionPrecisionPolicy::f32(), &[])
            .unwrap();
    let rows = operators
        .bind_rows(Rows::Host(
            HostRows::new(
                RowsShape::new(1, 3).unwrap(),
                RowsDType::F32,
                None,
                vec![1.0, 2.0, 3.0],
            )
            .unwrap(),
        ))
        .unwrap();
    assert_eq!(rows.device(), RowsDevice::Cuda { ordinal: 1 });
    assert_eq!(
        operators.download_rows(rows).unwrap().values(),
        &[1.0, 2.0, 3.0]
    );
}
