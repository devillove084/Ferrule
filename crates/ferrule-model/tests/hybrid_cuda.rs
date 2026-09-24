//! Same generic hybrid forward on CUDA, with no device-state downloads.
#![cfg(feature = "cuda")]
#[path = "hybrid_cpu.rs"]
mod reference;
use ferrule_common::execution::*;
use ferrule_model::decoder::*;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::runner::{
    MultiSessionBatchProgress, MultiSessionRunner, TransactionEndIntent, TransactionEndProgress,
};
use ferrule_model::tokenizer::TokenizerHandle;
use ferrule_model::{CheckpointDType, CheckpointTensorReader, CheckpointTensorSlice, TensorRole};
use ferrule_model::{ModelFamily, WeightSource};
use reference::{Fixture, floats};
use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
// F64 arithmetic also prevents F32 overflow from making infinity <= infinity.
fn within_tolerance(actual: f32, expected: f32, atol: f64, rtol: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (f64::from(actual) - f64::from(expected)).abs()
            <= atol + rtol * f64::from(expected).abs()
}
fn close(actual: &[f32], expected: &[f32], label: &str) {
    compare(actual, expected, label, 2e-5, 3e-4);
}
fn compare(actual: &[f32], expected: &[f32], label: &str, atol: f64, rtol: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut max_error = 0.0f64;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            within_tolerance(a, b, atol, rtol),
            "{label}[{i}]: {a} != {b}"
        );
        max_error = max_error.max((f64::from(a) - f64::from(b)).abs());
    }
    eprintln!("{label}: elements={} max_abs={max_error:.9e}", actual.len());
}
#[test]
fn gpu_oracle_comparison_rejects_nonfinite_and_f32_overflow() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for good in [0.0, 1.0, f32::MAX] {
            assert!(!within_tolerance(good, bad, 2e-4, 2e-4));
            assert!(!within_tolerance(bad, good, 2e-4, 2e-4));
        }
        assert!(!within_tolerance(bad, bad, 2e-4, 2e-4));
    }
    assert!(!within_tolerance(f32::MAX, -f32::MAX, 0.0, 1.5));
    assert!(within_tolerance(1.0001, 1.0, 2e-4, 2e-4));
}
type GpuRunner = GenericDecoderRunner<HybridCudaDecoder>;
enum Ending {
    Publish,
    Abort,
    Cancel,
}
fn snapshot(states: &[CudaHybridSequenceState]) -> Vec<(u64, usize, u64)> {
    states
        .iter()
        .map(|s| {
            (
                s.topology_id().get(),
                s.core().position(),
                s.core().generation(),
            )
        })
        .collect()
}
fn gpu_options(fixture: &Fixture, read_limit: u64) -> GenericDecoderOptions {
    let resources = fixture.resources();
    GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Unknown("hybrid-oracle".into()),
        WeightSource::Safetensors,
        2,
        32,
        32,
        4,
        read_limit,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap()
}
fn gpu(fixture: &Fixture) -> (GpuRunner, HybridCudaDevice) {
    let resources = fixture.resources();
    let options = gpu_options(fixture, 1 << 20);
    let estimate = HybridCudaMemoryEstimate::for_resources(&resources, &options, 64).unwrap();
    let budget = HybridCudaMemoryBudget {
        kv_pages: 64,
        state_bytes: estimate.per_sequence_state_bytes * 16,
        weight_bytes: estimate.weight_bytes_upper_bound,
        workspace_bytes: estimate.workspace_bytes_upper_bound,
    };
    let device = HybridCudaDevice::new_on_device(0, budget).unwrap();
    let runner = GpuRunner::hybrid_cuda(
        resources,
        TokenizerHandle::load(&fixture.dir).unwrap(),
        options,
        device.clone(),
    )
    .unwrap();
    (runner, device)
}
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_preparation_clean_failure_retries_same_device_clone() {
    let fixture = Fixture::new();
    let options = gpu_options(&fixture, 1 << 20);
    let estimate =
        HybridCudaMemoryEstimate::for_resources(&fixture.resources(), &options, 64).unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 64,
            state_bytes: estimate.per_sequence_state_bytes * 16,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    let prepare = |limit| {
        GpuRunner::hybrid_cuda(
            fixture.resources(),
            TokenizerHandle::load(&fixture.dir).unwrap(),
            gpu_options(&fixture, limit),
            device.clone(),
        )
    };
    let error = prepare(1)
        .err()
        .expect("one byte read limit must fail preparation");
    eprintln!("expected low read-limit failure: {error}");
    assert!(!device.needs_quarantine());
    assert_eq!(device.live_state_bytes(), 0);
    device.operators().failpoints().arm_allocation();
    assert!(
        prepare(1 << 20).is_err(),
        "injected clean allocation failure"
    );
    assert!(!device.needs_quarantine());
    assert_eq!(device.live_state_bytes(), 0);
    let mut runner = prepare(1 << 20).expect("clean preparation failure must return claim");
    assert!(
        prepare(1 << 20).is_err(),
        "a live image owns the factory exclusively"
    );
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let logits = step(
        &mut runner,
        &mut states,
        &mut [Vec::new()],
        &[(&[1, 3], ForwardPhase::Prefill)],
        1,
        Ending::Publish,
    );
    for (row, expected) in logits
        .iter()
        .zip(fixture.oracle["logits"].as_array().unwrap())
    {
        close(row, &floats(expected), "retried image");
    }
    drop(runner);
    assert!(
        prepare(1 << 20).is_err(),
        "surviving states must not gain a new image authority"
    );
    drop(states);
    assert_eq!(device.live_state_bytes(), 0);
    assert!(
        prepare(1 << 20).is_err(),
        "successful images consume the factory"
    );
}

fn step(
    runner: &mut GpuRunner,
    states: &mut [CudaHybridSequenceState],
    pages: &mut [Vec<KvPageId>],
    inputs: &[(&[u32], ForwardPhase)],
    id: u64,
    ending: Ending,
) -> Vec<Vec<f32>> {
    let publish = matches!(ending, Ending::Publish);
    let original = snapshot(states);
    let mut staged_pages = pages.to_vec();
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut writes = Vec::new();
    let mut sequences = Vec::new();
    let mut blocks = Vec::new();
    let mut reservations = Vec::new();
    for (i, (input, phase)) in inputs.iter().enumerate() {
        let context = states[i].core().position();
        let start = tokens.len();
        let block_start = blocks.len();
        let mut new_pages = Vec::new();
        while staged_pages[i].len() < (context + input.len()).div_ceil(2) {
            let page = KvPageId((id * 64 + i as u64 * 16 + staged_pages[i].len() as u64) as u32);
            staged_pages[i].push(page);
            new_pages.push(page);
        }
        for (offset, &token) in input.iter().enumerate() {
            let position = context + offset;
            tokens.push(token);
            positions.push(position as u32);
            writes.push(Some(KvWriteSlot::new(
                staged_pages[i][position / 2].0 * 2 + (position % 2) as u32,
            )));
        }
        blocks.extend(staged_pages[i].iter().map(|page| KvBlockId::new(page.0)));
        sequences.push(ExecutionSequence::new(
            StateSlot::new(i as u32),
            *phase,
            start as u32..tokens.len() as u32,
            context as u32,
            (context + input.len()) as u32,
            block_start as u32..blocks.len() as u32,
        ));
        reservations.push(KvReservationView {
            state_slot: StateSlot::new(i as u32),
            execution_state_slot: StateSlot::new(i as u32),
            positions: context..(context + input.len()),
            newly_allocated: new_pages,
            generation: states[i].core().generation(),
            execution_generation: states[i].core().generation(),
            cow_replacement: None,
        });
    }
    let prefill = inputs.iter().any(|(_, p)| *p == ForwardPhase::Prefill);
    let decode = inputs.iter().any(|(_, p)| *p == ForwardPhase::Decode);
    let mode = match (prefill, decode) {
        (true, true) => ForwardMode::Mixed,
        (true, false) => ForwardMode::Prefill,
        _ => ForwardMode::Decode,
    };
    let logits = vec![LogitsRequest::Full; tokens.len()];
    let batch = ExecutionBatch::new(mode, tokens, positions, writes, logits, sequences, blocks);
    let tx = ExecutionTransactionId::new(id).unwrap();
    runner
        .prepare_multi_session_batch(tx, states, &batch, &reservations)
        .unwrap();
    if matches!(ending, Ending::Cancel) {
        finish(runner, states, tx, TransactionEndIntent::Abort);
        assert_eq!(snapshot(states), original);
        return Vec::new();
    }
    let output = match runner
        .execute_multi_session_batch_progress(tx, states, &batch)
        .unwrap()
    {
        MultiSessionBatchProgress::Complete(output) => output,
        _ => panic!("CUDA hybrid unexpectedly suspended"),
    };
    assert_eq!(
        snapshot(states),
        original,
        "forward must only mutate transaction working copies"
    );
    finish(
        runner,
        states,
        tx,
        if publish {
            TransactionEndIntent::Publish
        } else {
            TransactionEndIntent::Abort
        },
    );
    if publish {
        pages.clone_from_slice(&staged_pages);
    } else {
        assert_eq!(
            snapshot(states),
            original,
            "abort must preserve all committed states"
        );
    }
    output
        .logits
        .into_iter()
        .map(|row| match row.logits {
            LogitsOutput::Full(values) => values,
            _ => panic!("expected full logits"),
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_hybrid_prefill_decode_abort_fork_reset_matches_tf_and_cpu() {
    let fixture = Fixture::new();
    let (mut runner, device) = gpu(&fixture);
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let per_state = states[0].kv_state().resident_bytes();
    let mut pages = vec![Vec::new()];
    let all = step(
        &mut runner,
        &mut states,
        &mut pages,
        &[(&[1, 3, 2, 5, 7], ForwardPhase::Prefill)],
        1,
        Ending::Publish,
    );
    for (row, want) in all.iter().zip(fixture.oracle["logits"].as_array().unwrap()) {
        close(row, &floats(want), "GPU TF full logits");
    }
    assert_eq!(device.live_state_bytes(), 2 * per_state);
    let mut cpu = fixture.runner();
    let mut cpu_states = vec![cpu.create_sequence_state().unwrap()];
    let expected = reference::step(
        &mut cpu,
        &mut cpu_states,
        &mut [Vec::new()],
        &[(&[1, 3, 2, 5, 7], ForwardPhase::Prefill)],
        1,
        true,
    );
    for (a, b) in all.iter().zip(expected) {
        close(a, &b, "GPU/CPU logits");
    }
    device.operators().reset_counters();
    runner.reset_sequence_state(&mut states[0]).unwrap();
    assert_eq!(
        device.operators().counters().device_to_host_copies,
        0,
        "reset must stay device resident"
    );
    pages[0].clear();
    step(
        &mut runner,
        &mut states,
        &mut pages,
        &[(&[1, 3], ForwardPhase::Prefill)],
        2,
        Ending::Publish,
    );
    step(
        &mut runner,
        &mut states,
        &mut pages,
        &[(&[2, 9], ForwardPhase::Prefill)],
        3,
        Ending::Abort,
    );
    step(
        &mut runner,
        &mut states,
        &mut pages,
        &[(&[2], ForwardPhase::Decode)],
        4,
        Ending::Cancel,
    );
    step(
        &mut runner,
        &mut states,
        &mut pages,
        &[(&[2, 5], ForwardPhase::Prefill)],
        5,
        Ending::Publish,
    );
    let parent = snapshot(&states);
    device.operators().reset_counters();
    let branch = runner.fork_sequence_state_from(&states[0], 4).unwrap();
    assert_eq!(
        device.operators().counters().device_to_host_copies,
        0,
        "fork must be D2D, not host copy"
    );
    assert!(device.completed_state_operations() > 0);
    assert!(runner.fork_sequence_state_from(&states[0], 3).is_err());
    let mut branches = vec![branch];
    let mut branch_pages = pages.clone();
    let branch_logits = step(
        &mut runner,
        &mut branches,
        &mut branch_pages,
        &[(&[9], ForwardPhase::Decode)],
        6,
        Ending::Publish,
    );
    assert_eq!(snapshot(&states), parent);
    let parent_logits = step(
        &mut runner,
        &mut states,
        &mut pages,
        &[(&[7], ForwardPhase::Decode)],
        7,
        Ending::Publish,
    );
    close(
        &parent_logits[0],
        &floats(&fixture.oracle["logits"][4]),
        "parent after divergent branch",
    );
    assert_ne!(branch_logits, parent_logits);
    for state in states.into_iter().chain(branches) {
        runner.try_release_sequence_state(state).unwrap();
    }
    assert_eq!(
        device.live_state_bytes(),
        per_state,
        "only default state remains"
    );
    drop(runner);
    assert_eq!(device.live_state_bytes(), 0);
    assert!(!device.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_hybrid_multitoken_continuations_match_verified_full_recompute() {
    let fixture = Fixture::new();
    let tokens = [1, 3, 2, 5, 7];
    let mut cpu = fixture.runner();
    let mut cpu_states = vec![cpu.create_sequence_state().unwrap()];
    let full = reference::step(
        &mut cpu,
        &mut cpu_states,
        &mut [Vec::new()],
        &[(&tokens, ForwardPhase::Prefill)],
        1,
        true,
    );
    let tf = fixture.oracle["logits"].as_array().unwrap();
    assert_eq!(full.len(), tokens.len());
    assert_eq!(tf.len(), tokens.len());
    // oracle.logits is TF's entire sequence from an empty cache, not the buggy
    // TF 5.2 cached multi-token continuation path. Verify CPU full before reuse.
    for (row, expected) in full.iter().zip(tf) {
        close(row, &floats(expected), "CPU full / TF full recompute");
    }
    let (mut runner, device) = gpu(&fixture);
    let mut tx = 1;
    for chunks in [&[2usize, 3][..], &[1usize, 2, 2][..]] {
        let mut states = vec![runner.create_sequence_state().unwrap()];
        let mut pages = vec![Vec::new()];
        let mut start = 0;
        for &size in chunks {
            assert_eq!(states[0].core().position(), start);
            let end = start + size;
            let rows = step(
                &mut runner,
                &mut states,
                &mut pages,
                &[(&tokens[start..end], ForwardPhase::Prefill)],
                tx,
                Ending::Publish,
            );
            assert_eq!(rows.len(), size);
            for (offset, row) in rows.iter().enumerate() {
                let label = format!("chunks={chunks:?} context={start} row={offset}");
                close(row, &full[start + offset], &label);
                close(
                    row,
                    &floats(&tf[start + offset]),
                    &format!("{label} TF full"),
                );
            }
            assert_eq!(states[0].core().position(), end);
            start = end;
            tx += 1;
        }
        assert_eq!(start, tokens.len());
        runner
            .try_release_sequence_state(states.pop().unwrap())
            .unwrap();
    }
    assert!(!device.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_hybrid_ragged_mixed_cohort_stays_isolated() {
    let fixture = Fixture::new();
    let (mut runner, device) = gpu(&fixture);
    let mut states = vec![
        runner.create_sequence_state().unwrap(),
        runner.create_sequence_state().unwrap(),
    ];
    let mut pages = vec![Vec::new(), Vec::new()];
    step(
        &mut runner,
        &mut states,
        &mut pages,
        &[
            (&[1, 3], ForwardPhase::Prefill),
            (&[1], ForwardPhase::Prefill),
        ],
        1,
        Ending::Publish,
    );
    let logits = step(
        &mut runner,
        &mut states,
        &mut pages,
        &[
            (&[2], ForwardPhase::Decode),
            (&[3, 2], ForwardPhase::Prefill),
        ],
        2,
        Ending::Publish,
    );
    close(
        &logits[0],
        &floats(&fixture.oracle["logits"][2]),
        "mixed decode",
    );
    close(
        &logits[1],
        &floats(&fixture.oracle["logits"][1]),
        "mixed prefill first",
    );
    close(
        &logits[2],
        &floats(&fixture.oracle["logits"][2]),
        "mixed prefill second",
    );
    assert!(!device.needs_quarantine());
}

fn finish(
    runner: &mut GpuRunner,
    states: &mut [CudaHybridSequenceState],
    tx: ExecutionTransactionId,
    intent: TransactionEndIntent,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match runner.end_transaction(tx, states, intent).unwrap() {
            TransactionEndProgress::Complete => return,
            TransactionEndProgress::Pending => assert!(
                std::time::Instant::now() < deadline,
                "CUDA terminal protocol timed out"
            ),
        }
        std::thread::yield_now();
    }
}

struct NasReference {
    path: PathBuf,
    header: Value,
    data_start: u64,
    report: Value,
}
impl NasReference {
    fn open(directory: &Path, manifest: &Value, case: &str) -> Self {
        let entry = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == case)
            .unwrap();
        assert_eq!(entry["self_checks_passed"], true);
        let report: Value = serde_json::from_slice(
            &std::fs::read(directory.join(entry["report"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let path = directory.join(entry["file"].as_str().unwrap());
        assert_eq!(report["sha256"], entry["sha256"]);
        // Test-only integrity check; no new production crypto/serialization dependencies.
        let checksum = Command::new("sha256sum")
            .arg(&path)
            .output()
            .expect("sha256sum is required for the opt-in oracle test");
        assert!(checksum.status.success());
        assert_eq!(
            String::from_utf8(checksum.stdout)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap(),
            entry["sha256"].as_str().unwrap()
        );
        let mut file = std::fs::File::open(&path).unwrap();
        let mut len = [0u8; 8];
        file.read_exact(&mut len).unwrap();
        let len = u64::from_le_bytes(len);
        assert!(len < 16 * 1024 * 1024);
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header).unwrap();
        Self {
            path,
            header: serde_json::from_slice(&header).unwrap(),
            data_start: 8 + len,
            report,
        }
    }
    fn logits(&self, stage: &str, rows: usize) -> Vec<f32> {
        let name = format!("{stage}.logits");
        let t = &self.header[&name];
        assert_eq!(t["dtype"], "F32");
        assert_eq!(t["shape"], serde_json::json!([1, rows, 248320]));
        assert_eq!(t["shape"], self.report["tensors"][&name]["shape"]);
        let start = t["data_offsets"][0].as_u64().unwrap();
        let end = t["data_offsets"][1].as_u64().unwrap();
        assert_eq!(end - start, (rows * 248320 * 4) as u64);
        assert_eq!(t["dtype"], self.report["tensors"][&name]["dtype"]);
        let slice = CheckpointTensorSlice {
            name,
            role: TensorRole::OutputHead,
            path: self.path.clone(),
            offset: self.data_start + start,
            bytes: end - start,
            dtype: CheckpointDType::F32,
            shape: vec![1, rows, 248320],
        };
        let payload = CheckpointTensorReader::new(16 * 1024 * 1024)
            .read_slice(&slice)
            .unwrap();
        payload
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }
}
fn argmax(values: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best as u32
}

#[test]
#[ignore = "requires local Qwen3.5-0.8B artifact, TF5.2 reference, and CUDA GPU"]
fn qwen35_08b_full_24_layer_gpu_matches_local_tf_reference() {
    use ferrule_model::models::qwen35::Qwen35Adapter;
    let model = std::env::var_os("FERRULE_QWEN35_08B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-0.8B".into());
    let oracle_dir = std::env::var_os("FERRULE_QWEN35_ORACLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/validation/qwen35-reference-cpu-f32")
        });
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(oracle_dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], "ferrule.qwen35-reference.v1");
    assert_eq!(manifest["status"], "complete");
    assert_eq!(manifest["execution"]["device"], "cpu");
    assert_eq!(manifest["execution"]["dtype"], "F32");
    let (_, resources) = Qwen35Adapter::bind_hf_metadata(&model).unwrap();
    assert_eq!(resources.spec().layers().len(), 24);
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Qwen35,
        WeightSource::Safetensors,
        2,
        32,
        8,
        1,
        1 << 30,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap();
    let estimate = HybridCudaMemoryEstimate::for_resources(&resources, &options, 32).unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 32,
            state_bytes: estimate.per_sequence_state_bytes * 6,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    eprintln!("24-layer CUDA memory admission: {estimate:?}");
    let mut runner = GpuRunner::hybrid_cuda(
        resources,
        TokenizerHandle::load(&model).unwrap(),
        options,
        device.clone(),
    )
    .unwrap();
    for (case_index, case) in ["capital", "hello"].iter().enumerate() {
        let oracle = NasReference::open(&oracle_dir, &manifest, case);
        let mut states = vec![runner.create_sequence_state().unwrap()];
        let mut pages = vec![Vec::new()];
        let mut predictions = Vec::new();
        for (call_index, stage) in ["prefill", "decode.0", "decode.1"].iter().enumerate() {
            let call = &oracle.report["calls"][call_index];
            assert_eq!(call["stage"], *stage);
            assert_eq!(
                call["position_start"].as_u64().unwrap() as usize,
                states[0].core().position()
            );
            let tokens: Vec<u32> = call["input_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect();
            if call_index > 0 {
                assert_eq!(tokens, [predictions[call_index - 1]]);
            }
            let logits = step(
                &mut runner,
                &mut states,
                &mut pages,
                &[(
                    &tokens,
                    if call_index == 0 {
                        ForwardPhase::Prefill
                    } else {
                        ForwardPhase::Decode
                    },
                )],
                (case_index * 3 + call_index + 1) as u64,
                Ending::Publish,
            );
            assert_eq!(logits.len(), tokens.len());
            assert!(logits.iter().all(|row| row.len() == 248320));
            let actual: Vec<f32> = logits.iter().flatten().copied().collect();
            let expected = oracle.logits(stage, tokens.len());
            compare(&actual, &expected, &format!("{case}/{stage}"), 2e-4, 2e-4);
            for (a, b) in logits.iter().zip(expected.chunks_exact(248320)) {
                assert_eq!(argmax(a), argmax(b), "{case}/{stage} row argmax");
            }
            let prediction = argmax(logits.last().unwrap());
            predictions.push(prediction);
            assert_eq!(prediction, call["next_token_id"].as_u64().unwrap() as u32);
        }
        assert!(
            states[0]
                .kv_state()
                .layers()
                .iter()
                .filter_map(Option::as_ref)
                .all(|s| s.position() == states[0].core().position())
        );
        eprintln!(
            "{case}: predictions={predictions:?} position={}",
            states[0].core().position()
        );
        runner
            .try_release_sequence_state(states.pop().unwrap())
            .unwrap();
    }
    assert!(!device.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_hybrid_dk256_head_repeat_rotary64_matches_cpu() {
    use ferrule_model::transformer::*;
    let norm = |width| RmsNorm::new(width, 1e-6).unwrap().with_one_plus_weight();
    let rope = RotaryEmbedding::new(
        256,
        10000.0,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 64 },
        RotaryScaling::None,
    )
    .unwrap();
    let full = GqaAttention::new(8, 2, 1, 256, false, rope)
        .unwrap()
        .with_gated_query()
        .unwrap()
        .with_qk_norms(norm(256), norm(256))
        .unwrap();
    let spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: "wide-hybrid-oracle".into(),
        hidden_size: 8,
        vocab_size: 11,
        max_sequence_length: Some(32),
        token_embedding: Embedding::new(11, 8, None).unwrap(),
        layers: vec![
            DecoderLayer::new(
                0,
                norm(8),
                Attention::GatedDeltaNet(
                    GatedDeltaNetAttention::new(8, 1, 2, 256, 2, 3, 1e-6, false).unwrap(),
                ),
                Residual::Add,
                norm(8),
                FeedForward::SwiGlu(SwiGlu::new(8, 12, false).unwrap()),
                Residual::Add,
            )
            .unwrap(),
            DecoderLayer::new(
                1,
                norm(8),
                Attention::Gqa(full),
                Residual::Add,
                norm(8),
                FeedForward::SwiGlu(SwiGlu::new(8, 12, false).unwrap()),
                Residual::Add,
            )
            .unwrap(),
        ],
        final_norm: norm(8),
        output: Linear::new(8, 11, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap();
    let mut oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hybrid_cpu/oracle.json")).unwrap();
    let tensors = oracle["tensors"].as_object_mut().unwrap();
    tensors.retain(|name, _| !name.starts_with("layers.2.") && !name.starts_with("layers.3."));
    for (name, shape) in [
        ("layers.0.attention.qkv.weight", vec![516, 8]),
        ("layers.0.attention.conv.weight", vec![516, 1, 3]),
        ("layers.1.attention.query.weight", vec![1024, 8]),
        ("layers.1.attention.key.weight", vec![256, 8]),
        ("layers.1.attention.value.weight", vec![256, 8]),
        ("layers.1.attention.output.weight", vec![8, 512]),
        ("layers.1.attention.query_norm.weight", vec![256]),
        ("layers.1.attention.key_norm.weight", vec![256]),
    ] {
        let values = (0..shape.iter().product::<usize>())
            .map(|i| (i as f32 * 0.13).sin() * 0.08)
            .collect::<Vec<_>>();
        tensors[name] = serde_json::json!({"shape":shape,"values":values});
    }
    let fixture = Fixture::with_oracle(oracle);
    let resources = fixture.resources_for(spec);
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Unknown("wide-hybrid".into()),
        WeightSource::Safetensors,
        2,
        32,
        8,
        1,
        1 << 20,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap();
    let estimate = HybridCudaMemoryEstimate::for_resources(&resources, &options, 32).unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 32,
            state_bytes: estimate.per_sequence_state_bytes * 8,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    let mut gpu = GpuRunner::hybrid_cuda(
        resources.clone(),
        TokenizerHandle::load(&fixture.dir).unwrap(),
        options.clone(),
        device.clone(),
    )
    .unwrap();
    let mut cpu = GenericDecoderRunner::<HybridCpuDecoder>::hybrid_cpu(
        resources,
        TokenizerHandle::load(&fixture.dir).unwrap(),
        options,
    )
    .unwrap();
    cpu.configure_kv_page_capacity(32).unwrap();
    let mut gs = vec![gpu.create_sequence_state().unwrap()];
    let mut cs = vec![cpu.create_sequence_state().unwrap()];
    let mut gp = vec![Vec::new()];
    let mut cp = vec![Vec::new()];
    for (id, ids, phase) in [
        (1, &[1, 3, 2][..], ForwardPhase::Prefill),
        (2, &[5][..], ForwardPhase::Decode),
    ] {
        let got = step(
            &mut gpu,
            &mut gs,
            &mut gp,
            &[(ids, phase)],
            id,
            Ending::Publish,
        );
        let expected = reference::step(&mut cpu, &mut cs, &mut cp, &[(ids, phase)], id, true);
        for (a, b) in got.iter().zip(expected) {
            close(a, &b, "wide-head GPU/CPU");
        }
    }
    assert!(!device.needs_quarantine());
}
