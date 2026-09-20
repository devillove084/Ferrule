//! Opt-in NAS checkpoint validation, not a fixture or a serving/performance claim.
//! Run with FERRULE_CUDA_ARCH=sm_86 and an external 300s deadline. Missing model,
//! devices, CUDA errors and oracle mismatches are failures, never runtime skips.
#![cfg(all(unix, feature = "cuda"))]

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_backend::cuda::providers::CudaContext;
use ferrule_common::execution::ForwardPhase;
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::ModelFamily;
use ferrule_model::checkpoint::CheckpointDType;
use ferrule_model::decoder::DenseLogits;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::models::qwen3::Qwen3DenseRecipe;
use ferrule_model::transformer::{BoundDecoderResources, DecoderLoadOptions, LayerSegmentPlan};
use ferrule_runtime::parallel::pipeline::{
    PipelineConfig, PipelineParallelExecutor, PipelineStage,
};
use ferrule_runtime::{SessionId, TransactionState};

const MODEL: &str = "/mnt/nas1/hf/Qwen3-0.6B";
const LAYERS: usize = 28;
const VOCAB: usize = 151_936;
const PAGE: usize = 2;
const PARAMETER_LIMIT: u64 = 1024 * 1024 * 1024;
const SESSION: SessionId = SessionId(1);

fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: PAGE,
        max_pages: 16,
        max_positions: 16,
        max_batch_tokens: 8,
        session_capacity: 2,
        max_parameter_bytes: PARAMETER_LIMIT,
        precision: ExecutionPrecisionPolicy::f32(),
        max_ack_polls: 8,
    }
}

fn load() -> BoundDecoderResources {
    let path = std::env::var_os("FERRULE_QWEN3_06B_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(MODEL));
    let value =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).expect("NAS config"))
            .expect("valid Qwen3 config");
    let resources = DecoderLoadOptions::new(&Qwen3DenseRecipe::new(), &value)
        .open_hf(&path, ModelFamily::Qwen3)
        .expect("bind complete Qwen3-0.6B checkpoint");
    assert_eq!(resources.spec().architecture(), "Qwen3ForCausalLM");
    assert_eq!(
        resources.spec().layers().len(),
        LAYERS,
        "no layer truncation"
    );
    assert_eq!(resources.spec().hidden_size(), 1024);
    assert_eq!(resources.spec().vocab_size(), VOCAB);
    assert!(resources.spec().tie_word_embeddings());
    assert_eq!(resources.state_dict().len(), 3 + LAYERS * 11);
    assert!(resources.state_dict().validate_source_identities());
    let mut largest = 0;
    for parameter in resources.state_dict().parameters() {
        let slice = parameter.weight().slice();
        assert_eq!(slice.dtype, CheckpointDType::Bf16, "{}", parameter.path());
        assert!(slice.bytes <= PARAMETER_LIMIT, "{}", parameter.path());
        largest = largest.max(slice.bytes);
    }
    assert_eq!(largest, (VOCAB * 1024 * 2) as u64);
    eprintln!(
        "checkpoint={} layers={LAYERS} weights=BF16 execute=F32 parameter_limit={PARAMETER_LIMIT} largest_tensor={largest}",
        path.display()
    );
    resources
}

fn pipeline(degree: usize, cuda: bool) -> PipelineParallelExecutor {
    assert!(degree == 1 || degree == 2);
    let plans = (0..degree)
        .map(|stage| {
            LayerSegmentPlan::new(
                LAYERS,
                stage * LAYERS / degree..(stage + 1) * LAYERS / degree,
                stage == 0,
                stage + 1 == degree,
            )
            .unwrap()
        })
        .collect();
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(86),
        degree as u32,
        ParallelRankId::new(0),
        ParallelismPlan::validated(1, 1, 1, 1, 1, degree).unwrap(),
    )
    .unwrap();
    let config = config();
    let start = Instant::now();
    // Capture no CUDA object: checkpoint binding, materialization, device and KV
    // construction all run inside the production persistent owner factory.
    let pipeline = if cuda {
        PipelineParallelExecutor::new_with_program(topology, plans, config, move |rank, plan| {
            let resources = load();
            let ops = Rc::new(CudaOperators::new_on_device(rank.local.get() as usize)?);
            PipelineStage::prepare_cuda(&resources, plan, config, ops)
        })
    } else {
        PipelineParallelExecutor::new(topology, plans, config, move |_, plan| {
            PipelineStage::prepare_cpu(&load(), plan, config)
        })
    }
    .expect("prepare all 28 decoder layers on persistent owners");
    let owners = pipeline.owner_stats().unwrap();
    assert_eq!(owners.len(), degree);
    assert_eq!(
        owners
            .iter()
            .flat_map(|owner| owner.layers.clone())
            .collect::<Vec<_>>(),
        (0..LAYERS).collect::<Vec<_>>()
    );
    for (index, owner) in owners.iter().enumerate() {
        assert_ne!(owner.thread, std::thread::current().id());
        assert!(
            owners[..index]
                .iter()
                .all(|other| other.thread != owner.thread)
        );
        assert_eq!(owner.executions, 0);
        assert!(owner.experts.is_empty());
    }
    eprintln!("prepared cuda={cuda} PP{degree} in {:?}", start.elapsed());
    pipeline
}

fn drained(pipeline: &PipelineParallelExecutor) {
    assert!(!pipeline.is_quarantined());
    assert_eq!(pipeline.outstanding(), 0);
    assert_eq!(pipeline.coordinator().in_use_credits(), 0);
    assert_eq!(pipeline.coordinator().retained_transaction_count(), 0);
    assert_eq!(pipeline.coordinator().retained_operation_count(), 0);
    assert_eq!(pipeline.page_manager().stats().retiring_pages, 0);
    for owner in pipeline.owner_stats().unwrap() {
        assert_eq!(owner.kv.active_transactions, 0);
        assert_eq!(owner.expert_outstanding, 0);
    }
}

fn forward(
    pipeline: &mut PipelineParallelExecutor,
    session: SessionId,
    tokens: &[u32],
    phase: ForwardPhase,
) -> DenseLogits {
    let start = Instant::now();
    let before = pipeline.coordinator().publication_count();
    let logits = pipeline
        .forward(session, tokens, phase)
        .expect("GPU/CPU forward must not be skipped")
        .logits;
    assert_eq!(logits.rows(), tokens.len());
    assert_eq!(logits.width(), VOCAB);
    for row in 0..logits.rows() {
        assert!(logits.row(row).unwrap().iter().all(|v| v.is_finite()));
    }
    assert_eq!(pipeline.coordinator().publication_count(), before + 1);
    drained(pipeline);
    eprintln!(
        "PP{} {phase:?} tokens={tokens:?} took {:?}",
        pipeline.stage_descriptions().len(),
        start.elapsed()
    );
    logits
}

fn close(label: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), VOCAB);
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0f32;
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (a - e).abs();
        assert!(
            a.is_finite() && e.is_finite() && error <= atol + rtol * e.abs(),
            "{label}[{index}]: actual={a} expected={e} abs_error={error}"
        );
        maximum = maximum.max(error);
    }
    assert_eq!(
        argmax(actual),
        argmax(expected),
        "{label}: greedy token mismatch"
    );
    eprintln!(
        "{label}: width={VOCAB} max_abs_error={maximum:.8e} argmax={}",
        argmax(actual)
    );
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32
}

fn release(pipeline: &mut PipelineParallelExecutor, sessions: &[SessionId]) {
    for &session in sessions {
        pipeline.release_session(session).unwrap();
    }
    drained(pipeline);
    assert_eq!(pipeline.page_manager().active_sequences(), 0);
    assert_eq!(pipeline.page_manager().allocated_pages(), 0);
    for owner in pipeline.owner_stats().unwrap() {
        assert_eq!(owner.sessions, 0);
        assert_eq!(owner.kv.resident_pages, 0);
        assert_eq!(owner.kv.preempted_pages, 0);
        assert_eq!(owner.kv.free_pages, owner.kv.physical_pages);
    }
    pipeline.shutdown().unwrap();
}

#[test]
#[ignore = "requires local Qwen3-0.6B, two CUDA GPUs, external 300s timeout and --test-threads=1"]
fn qwen3_06b_all_28_layers_bf16_weights_f32_gpu_pp1_pp2_prefill_decode() {
    let start = Instant::now();
    assert!(CudaContext::device_count().expect("CUDA device enumeration") >= 2);
    let mut tokens = vec![151_643, 9707, 11];
    let mut reference = Vec::new();
    let mut pp1 = pipeline(1, true);
    reference.push(forward(&mut pp1, SESSION, &tokens, ForwardPhase::Prefill));
    for _ in 0..3 {
        let last = reference.last().unwrap();
        let token = argmax(last.row(last.rows() - 1).unwrap());
        tokens.push(token);
        reference.push(forward(&mut pp1, SESSION, &[token], ForwardPhase::Decode));
    }
    assert_eq!(pp1.owner_stats().unwrap()[0].executions, 4);
    release(&mut pp1, &[SESSION]);
    drop(pp1);

    let mut pp2 = pipeline(2, true);
    let owners = pp2.owner_stats().unwrap();
    let prefill = forward(&mut pp2, SESSION, &tokens[..3], ForwardPhase::Prefill);
    for row in 0..3 {
        close(
            "PP2/PP1 prefill",
            prefill.row(row).unwrap(),
            reference[0].row(row).unwrap(),
            2e-5,
            2e-5,
        );
    }
    // This observer runs after real compute/readiness, before the sole commit
    // decision. An aborted full-model decode must leave the prefix untouched.
    let slot = pp2.session_slot(SESSION).unwrap();
    let pages = pp2
        .page_manager()
        .block_table(slot)
        .unwrap()
        .pages()
        .to_vec();
    let cancel = AtomicBool::new(false);
    let mut observed = false;
    let error = pp2
        .forward_observed(
            SESSION,
            &[tokens[3]],
            ForwardPhase::Decode,
            &cancel,
            |progress| {
                if progress.state == TransactionState::Preparing {
                    observed = true;
                    cancel.store(true, Ordering::Release);
                }
            },
        )
        .expect_err("cancel before commit must abort");
    assert!(
        observed && error.to_string().contains("cancelled before decision"),
        "{error}"
    );
    drained(&pp2);
    assert_eq!(pp2.coordinator().publication_count(), 1);
    assert_eq!(pp2.page_manager().block_table(slot).unwrap().pages(), pages);
    assert_eq!(
        pp2.page_manager()
            .block_table(slot)
            .unwrap()
            .committed_tokens(),
        3
    );
    for (index, &token) in tokens[3..].iter().enumerate() {
        let actual = forward(&mut pp2, SESSION, &[token], ForwardPhase::Decode);
        close(
            "PP2/PP1 decode",
            actual.row(0).unwrap(),
            reference[index + 1].row(0).unwrap(),
            2e-5,
            2e-5,
        );
    }
    assert_eq!(
        pp2.page_manager()
            .block_table(slot)
            .unwrap()
            .committed_tokens(),
        tokens.len()
    );

    let replay_session = SessionId(2);
    let replay = forward(&mut pp2, replay_session, &tokens, ForwardPhase::Prefill);
    for (index, expected) in reference.iter().enumerate().skip(1) {
        close(
            "causal replay/decode",
            replay.row(index + 2).unwrap(),
            expected.row(0).unwrap(),
            2e-3,
            2e-4,
        );
    }
    for (before, after) in owners.iter().zip(pp2.owner_stats().unwrap()) {
        assert_eq!(before.thread, after.thread);
        assert_eq!(before.layers, after.layers);
        assert_eq!(after.executions, 6); // prefill, aborted decode, 3 decodes, replay
    }
    release(&mut pp2, &[SESSION, replay_session]);
    drop(pp2);

    // Narrow in token count only: this CPU oracle still executes all 28 layers
    // and the complete tied output head, using independent CPU operator kernels.
    let mut cpu = pipeline(1, false);
    let cpu_prefill = forward(&mut cpu, SESSION, &tokens[..3], ForwardPhase::Prefill);
    for row in 0..cpu_prefill.rows() {
        close(
            "GPU/CPU narrow oracle prefill",
            reference[0].row(row).unwrap(),
            cpu_prefill.row(row).unwrap(),
            2e-3,
            2e-4,
        );
    }
    let cpu_decode = forward(&mut cpu, SESSION, &[tokens[3]], ForwardPhase::Decode);
    close(
        "GPU/CPU narrow oracle decode",
        reference[1].row(0).unwrap(),
        cpu_decode.row(0).unwrap(),
        2e-3,
        2e-4,
    );
    release(&mut cpu, &[SESSION]);
    eprintln!(
        "PASS full 28-layer BF16-checkpoint/F32-execution GPU PP1/PP2 + narrow CPU oracle in {:?}",
        start.elapsed()
    );
}
