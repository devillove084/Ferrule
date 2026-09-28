use super::*;
use ferrule_common::execution::*;
use ferrule_model::decoder::{CudaHybridSequenceState, HybridCudaDecoder};
use ferrule_model::runner::{
    MultiSessionBatchProgress, MultiSessionRunner, ResidentModelRunner, TransactionEndIntent,
    TransactionEndProgress,
};
use ferrule_model::tokenizer::TokenizerHandle;
use ferrule_model::{CheckpointDType, CheckpointTensorReader, CheckpointTensorSlice, TensorRole};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;

struct Oracle {
    path: PathBuf,
    header: Value,
    data_start: u64,
    report: Value,
}

impl Oracle {
    fn open(directory: &Path, case: &str) -> Self {
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["schema"], "ferrule.qwen35-fp8-reference.f32.v1");
        assert_eq!(manifest["status"], "complete");
        assert_eq!(manifest["full_reference_complete"], true);
        assert_eq!(manifest["execution"]["precision_profile"], "f32");
        assert_eq!(manifest["execution"]["tf32"], false);
        let entry = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == case)
            .unwrap();
        let report: Value = serde_json::from_slice(
            &std::fs::read(directory.join(entry["report"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        assert_eq!(report["sha256"], entry["sha256"]);
        assert_eq!(report["calls"].as_array().unwrap().len(), 2);
        let path = directory.join(report["file"].as_str().unwrap());
        let checksum = Command::new("sha256sum").arg(&path).output().unwrap();
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
        let mut len = [0; 8];
        file.read_exact(&mut len).unwrap();
        let len = u64::from_le_bytes(len);
        assert!(len < 16 * 1024 * 1024);
        let mut header = vec![0; len as usize];
        file.read_exact(&mut header).unwrap();
        Self {
            path,
            header: serde_json::from_slice(&header).unwrap(),
            data_start: 8 + len,
            report,
        }
    }

    fn values(&self, name: &str) -> Vec<f32> {
        let tensor = &self.header[name];
        assert_eq!(tensor["dtype"], "F32");
        let start = tensor["data_offsets"][0].as_u64().unwrap();
        let end = tensor["data_offsets"][1].as_u64().unwrap();
        let shape = serde_json::from_value(tensor["shape"].clone()).unwrap();
        CheckpointTensorReader::new(64 * 1024 * 1024)
            .read_slice(&CheckpointTensorSlice {
                name: name.into(),
                role: TensorRole::Unknown,
                path: self.path.clone(),
                offset: self.data_start + start,
                bytes: end - start,
                dtype: CheckpointDType::F32,
                shape,
            })
            .unwrap()
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn ids(&self, name: &str) -> Vec<usize> {
        let tensor = &self.header[name];
        assert_eq!(tensor["dtype"], "I64");
        let start = tensor["data_offsets"][0].as_u64().unwrap();
        let end = tensor["data_offsets"][1].as_u64().unwrap();
        let shape = serde_json::from_value(tensor["shape"].clone()).unwrap();
        CheckpointTensorReader::new(16 * 1024 * 1024)
            .read_slice(&CheckpointTensorSlice {
                name: name.into(),
                role: TensorRole::Unknown,
                path: self.path.clone(),
                offset: self.data_start + start,
                bytes: end - start,
                dtype: CheckpointDType::I64,
                shape,
            })
            .unwrap()
            .bytes
            .chunks_exact(8)
            .map(|bytes| usize::try_from(i64::from_le_bytes(bytes.try_into().unwrap())).unwrap())
            .collect()
    }
}

fn finish(
    runner: &mut GenericDecoderRunner<HybridCudaDecoder>,
    states: &mut [CudaHybridSequenceState],
    transaction: ExecutionTransactionId,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match runner
            .end_transaction(transaction, states, TransactionEndIntent::Publish)
            .unwrap()
        {
            TransactionEndProgress::Complete => return,
            TransactionEndProgress::Pending => assert!(
                std::time::Instant::now() < deadline,
                "strict EP terminal transaction timed out"
            ),
        }
        std::thread::yield_now();
    }
}

fn full_logits_step(
    runner: &mut GenericDecoderRunner<HybridCudaDecoder>,
    states: &mut [CudaHybridSequenceState],
    pages: &mut [Vec<KvPageId>],
    tokens: &[u32],
    phase: ForwardPhase,
    transaction_id: u64,
) -> Vec<Vec<f32>> {
    let state = &states[0];
    let position = state.core().position();
    let mut staged_pages = pages.to_vec();
    let mut writes = Vec::with_capacity(tokens.len());
    let mut blocks = Vec::new();
    let mut new_pages = Vec::new();
    while staged_pages[0].len() < (position + tokens.len()).div_ceil(16) {
        let page = KvPageId((transaction_id * 64 + staged_pages[0].len() as u64) as u32);
        staged_pages[0].push(page);
        new_pages.push(page);
    }
    let mut positions = Vec::with_capacity(tokens.len());
    for (offset, _) in tokens.iter().enumerate() {
        let absolute = position + offset;
        positions.push(absolute as u32);
        writes.push(Some(KvWriteSlot::new(
            staged_pages[0][absolute / 16].0 * 16 + (absolute % 16) as u32,
        )));
    }
    blocks.extend(staged_pages[0].iter().map(|page| KvBlockId::new(page.0)));
    let batch = ExecutionBatch::new(
        match phase {
            ForwardPhase::Prefill => ForwardMode::Prefill,
            ForwardPhase::Decode => ForwardMode::Decode,
        },
        tokens.to_vec(),
        positions,
        writes,
        vec![LogitsRequest::Full; tokens.len()],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            phase,
            0..tokens.len() as u32,
            position as u32,
            (position + tokens.len()) as u32,
            0..blocks.len() as u32,
        )],
        blocks,
    );
    let reservation = KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: position..position + tokens.len(),
        newly_allocated: new_pages,
        generation: state.core().generation(),
        execution_generation: state.core().generation(),
        cow_replacement: None,
    };
    let transaction = ExecutionTransactionId::new(transaction_id).unwrap();
    runner
        .prepare_multi_session_batch(transaction, states, &batch, &[reservation])
        .unwrap();
    let output = match runner
        .execute_multi_session_batch_progress(transaction, states, &batch)
        .unwrap()
    {
        MultiSessionBatchProgress::Complete(output) => output,
        _ => panic!("strict EP forward unexpectedly suspended"),
    };
    finish(runner, states, transaction);
    pages.clone_from_slice(&staged_pages);
    output
        .logits
        .into_iter()
        .map(|row| match row.logits {
            LogitsOutput::Full(values) => values,
            other => panic!("strict test expected full logits, got {other:?}"),
        })
        .collect()
}

#[derive(Default)]
struct StrictReport {
    metrics: Vec<Value>,
    failures: Vec<String>,
    logits: usize,
}
impl StrictReport {
    fn values(&mut self, actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} shape");
        let mut max_abs = 0.0f64;
        let mut max_tolerance_ratio = 0.0f64;
        let mut sum_squared = 0.0f64;
        let mut failures = 0;
        let mut nonfinite = 0;
        let mut first_failure = None;
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            let finite = a.is_finite() && e.is_finite();
            let diff = (f64::from(a) - f64::from(e)).abs();
            let tolerance = 2e-4 + 2e-4 * f64::from(e).abs();
            if finite {
                max_abs = max_abs.max(diff);
                max_tolerance_ratio = max_tolerance_ratio.max(diff / tolerance);
                sum_squared += diff * diff;
            } else {
                nonfinite += 1;
            }
            if !finite || diff > tolerance {
                failures += 1;
                first_failure.get_or_insert(serde_json::json!({"index":index,"actual":a,"expected":e,"abs_error":diff,"tolerance":tolerance}));
            }
        }
        let rms = (sum_squared / actual.len() as f64).sqrt();
        eprintln!(
            "{label}: n={} max_abs={max_abs:.9e} rms={rms:.9e} max_tolerance_ratio={max_tolerance_ratio:.9e} nonfinite={nonfinite} failures={failures} first={first_failure:?}",
            actual.len()
        );
        self.metrics.push(serde_json::json!({"label":label,"elements":actual.len(),"max_abs":max_abs,"rms":rms,
            "max_tolerance_ratio":max_tolerance_ratio,"nonfinite":nonfinite,"failures":failures,"first_failure":first_failure}));
        if failures > 0 {
            self.failures.push(label.into());
        }
    }
}

fn compare_routes(
    report: &mut StrictReport,
    routes: &BTreeMap<usize, (usize, Vec<usize>, Vec<f32>)>,
    oracle: &Oracle,
    case: &str,
    stage: &str,
    rows: usize,
) {
    assert_eq!(routes.len(), 40, "{stage} route layer count");
    for (layer, (actual_rows, actual_ids, actual_weights)) in routes {
        assert_eq!(*actual_rows, rows);
        let prefix = format!("{stage}.layers.{layer:02}");
        let expected_ids = oracle.ids(&format!("{prefix}.selected_expert_ids"));
        let expected_weights = oracle.values(&format!("{prefix}.routing_weights"));
        assert_eq!(actual_ids.len(), rows * 8);
        assert_eq!(expected_ids.len(), actual_ids.len());
        assert_eq!(actual_weights.len(), rows * 8);
        let mismatches = actual_ids
            .iter()
            .zip(&expected_ids)
            .filter(|(a, e)| a != e)
            .count();
        if mismatches > 0 {
            eprintln!(
                "FIRST ROUTE DIVERGENCE {case}/{prefix}: actual={actual_ids:?} expected={expected_ids:?}"
            );
            report
                .failures
                .push(format!("{case}/{prefix} ordered route IDs"));
        }
        // Compare original rank-slot order; do not sort away routing differences.
        report.metrics.push(serde_json::json!({"label":format!("{case}/{prefix} ordered route IDs"), "elements":actual_ids.len(),
            "failures":mismatches, "actual_ids":actual_ids, "expected_ids":expected_ids}));
        report.values(
            actual_weights,
            &expected_weights,
            &format!("{case}/{prefix} routing_weights"),
        );
    }
}

#[test]
#[ignore = "requires NAS Qwen3.5 FP8, eight CUDA GPUs, and 900s; strict all-row F32 full40 oracle"]
fn qwen35_ep8_factory_full40_f32_strict_oracle() {
    ferrule_common::observability::init_tracing();
    let model = std::env::var_os("FERRULE_NUMERIC_FP8_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8"));
    let oracle_dir = std::env::var_os("FERRULE_NUMERIC_FP8_F32_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/validation/qwen35-fp8-reference-full40-f32-v1")
        });
    let oracles = [
        Oracle::open(&oracle_dir, "hello"),
        Oracle::open(&oracle_dir, "capital"),
    ];
    let config = ferrule_model::AutoConfig::from_pretrained(&model).unwrap();
    let options = ModelFactoryOptions {
        max_layers: None,
        max_tensor_mebibytes: 1024,
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_mebibytes: 64,
        expert_cache: ExpertCacheOptions::default(),
        qwen35_moe_capacity: None,
        qwen35_host_cache: Some(
            ferrule_model::transformer::host_experts::HostExpertCacheOptions {
                max_bytes: 32u64 << 30,
                ..Default::default()
            },
        ),
        moe_hotset_experts: 0,
        kv_cache_mebibytes: Some(1024),
        scheduler_config: ResidentSchedulerConfig {
            max_active_sequences: 1,
            max_batch_tokens: 32,
            max_decode_batch: 1,
            prefill_chunk_size: 32,
            ..Default::default()
        },
        driver_config: ResidentTopKDriverConfig {
            ctx_size: 1024,
            enable_native_proposals: false,
            ..Default::default()
        },
    };
    let plan = ResidentModelPlanner::new()
        .prepare_qwen35_expert_parallel(
            &config,
            BackendSelection::Auto,
            None,
            options,
            PipelineBuildOptions {
                parallelism: ferrule_common::ParallelismPlan {
                    expert_parallel: 8,
                    ..Default::default()
                },
                devices: Some((0..8).collect()),
                ..Default::default()
            },
        )
        .unwrap();
    let request = plan.request;
    let (_, resources) = Qwen35Adapter::bind_hf_metadata(&request.model_path).unwrap();
    let tokenizer = TokenizerHandle::load(&request.model_path).unwrap();
    let mut prepared = super::prepare_bound(&request, resources, tokenizer).unwrap();
    assert!(Arc::ptr_eq(
        prepared.runner.resources().host_experts().unwrap(),
        &prepared.host
    ));
    assert_eq!(prepared.host.stats().experts, 10240);
    assert_eq!(prepared.host.hits(), 30720);
    let host_identity = Arc::as_ptr(&prepared.host);
    let module = prepared.runner.forward_executor().module();
    assert_eq!(
        module.numeric_fp8_precision(),
        Some(NumericFp8Precision::F32Tf32x3)
    );
    let routes = Rc::new(RefCell::new(BTreeMap::new()));
    let observed = Rc::clone(&routes);
    module.set_diagnostic_trace(move |event| {
        if event.name == "router" {
            let route = event.routes.expect("strict router trace");
            let previous = observed.borrow_mut().insert(
                event.layer.unwrap(),
                (
                    event.shape.rows(),
                    route.expert_ids().to_vec(),
                    route.weights().to_vec(),
                ),
            );
            assert!(
                previous.is_none(),
                "duplicate router event would hide a row/layer"
            );
        }
        Ok(())
    });
    let mut state = prepared.runner.create_sequence_state().unwrap();
    let mut pages = vec![Vec::new()];
    let mut transaction = 1u64;
    let mut report = StrictReport::default();
    let started = Instant::now();
    for (case, oracle) in ["hello", "capital"].into_iter().zip(&oracles) {
        let case_started = Instant::now();
        let mut position = 0usize;
        for (index, stage) in ["prefill", "decode.0"].into_iter().enumerate() {
            let tokens: Vec<u32> =
                serde_json::from_value(oracle.report["calls"][index]["input_token_ids"].clone())
                    .unwrap();
            assert_eq!(oracle.report["calls"][index]["position_start"], position);
            if index == 1 {
                assert_eq!(tokens.len(), 1);
            }
            let phase = if index == 0 {
                ForwardPhase::Prefill
            } else {
                ForwardPhase::Decode
            };
            let actual = full_logits_step(
                &mut prepared.runner,
                std::slice::from_mut(&mut state),
                &mut pages,
                &tokens,
                phase,
                transaction,
            );
            position += tokens.len();
            assert_eq!(
                oracle.report["calls"][index]["sequence_length_after"],
                position
            );
            assert_eq!(actual.len(), tokens.len());
            assert!(actual.iter().all(|row| row.len() == 248320));
            assert_eq!(
                oracle.header[format!("{stage}.logits")]["shape"],
                serde_json::json!([1, tokens.len(), 248320])
            );
            let expected = oracle.values(&format!("{stage}.logits"));
            report.logits += actual.len() * 248320;
            report.values(
                &actual.iter().flatten().copied().collect::<Vec<_>>(),
                &expected,
                &format!("{case}/{stage} logits"),
            );
            compare_routes(
                &mut report,
                &routes.borrow(),
                oracle,
                case,
                stage,
                tokens.len(),
            );
            routes.borrow_mut().clear();
            let layers = state.kv_state().layers();
            assert_eq!(layers.len(), 40);
            assert_eq!(layers.iter().flatten().count(), 30);
            for (layer, state) in layers.iter().enumerate() {
                if let Some(state) = state {
                    assert_eq!(state.position(), position);
                    let (conv, recurrent) = state.diagnostic_snapshot().unwrap();
                    report.values(
                        &conv,
                        &oracle.values(&format!("{stage}.cache.conv_states.{layer:02}")),
                        &format!("{case}/{stage} conv {layer}"),
                    );
                    report.values(
                        &recurrent,
                        &oracle.values(&format!("{stage}.cache.recurrent_states.{layer:02}")),
                        &format!("{case}/{stage} recurrent {layer}"),
                    );
                }
            }
            assert_eq!(state.core().position(), position);
            assert_eq!(Arc::as_ptr(&prepared.host), host_identity);
            assert_eq!(prepared.host.hits(), 30720);
            let root_cache = prepared
                .runner
                .forward_executor()
                .module()
                .expert_cache_stats()
                .unwrap();
            assert_eq!(root_cache.uploads, 0);
            assert_eq!(root_cache.resident_experts, 0);
            assert_eq!(root_cache.evictions, 0);
            assert!(!prepared.device.needs_quarantine());
            assert!(case_started.elapsed() < Duration::from_secs(900));
            transaction += 1;
        }
        prepared.runner.reset_sequence_state(&mut state).unwrap();
        pages.clear();
        pages.push(Vec::new());
        assert_eq!(state.core().position(), 0);
        eprintln!(
            "EP8 strict case={case} complete in {:?}",
            case_started.elapsed()
        );
    }
    assert_eq!(report.logits, 1_986_560);
    prepared
        .runner
        .forward_executor()
        .module()
        .clear_diagnostic_trace();
    prepared.runner.try_release_sequence_state(state).unwrap();
    prepared.runner.shutdown_physical().unwrap();
    assert!(!prepared.device.needs_quarantine());
    let output_dir = std::env::var_os("FERRULE_EP_STRICT_REPORT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/validation/qwen35-ep-strict")
        });
    std::fs::create_dir_all(&output_dir).unwrap();
    let passed = report.failures.is_empty();
    let summary = serde_json::json!({"passed":passed,"logits_compared":report.logits,"atol":2e-4,"rtol":2e-4,
        "comparison":"F64; both operands finite; original route rank-slot order", "cases":["hello","capital"],
        "oracle_directory":oracle_dir,"host_generation":prepared.host.generation(),"host_hits":prepared.host.hits(),
        "elapsed_forward_and_compare_seconds":started.elapsed().as_secs_f64(),"failures":report.failures,"metrics":report.metrics});
    std::fs::write(
        output_dir.join("metrics.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    drop(prepared.runner);
    assert_eq!(prepared.device.live_state_bytes(), 0);
    assert!(!prepared.device.needs_quarantine());
    eprintln!(
        "EP8 strict full40: logits={} passed={passed} report={} failures={:?}; shutdown complete",
        report.logits,
        output_dir.join("metrics.json").display(),
        report.failures
    );
    assert!(
        passed,
        "strict EP8 differences; see metrics.json: {:?}",
        report.failures
    );
}
