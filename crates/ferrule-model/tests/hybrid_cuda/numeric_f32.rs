//! Explicit F32 arithmetic acceptance, independent of the BF16 diagnostic oracle.
use super::*;
use ferrule_model::runner::ModelRunner;
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn numeric_precision_admission_uses_the_selected_workspace_plan() {
    let f = fixture::Fixture::new();
    let r = resources(&f);
    let limits = ExpertCacheLimits {
        max_experts: 1,
        max_bytes: 8192,
    };
    let legacy = options(&r)
        .with_hybrid_cuda_numeric_fp8(limits, 4096)
        .unwrap();
    assert_eq!(
        legacy.hybrid_cuda_numeric_fp8_precision(),
        Some(NumericFp8Precision::Bf16RneF32Accumulate)
    );
    let explicit = options(&r)
        .with_hybrid_cuda_numeric_fp8_precision(limits, 4096, NumericFp8Precision::F32Tf32x3)
        .unwrap();
    assert_eq!(
        explicit.hybrid_cuda_numeric_fp8(),
        legacy.hybrid_cuda_numeric_fp8()
    );
    let a = HybridCudaMemoryEstimate::for_resources(&r, &legacy, 32).unwrap();
    let b = HybridCudaMemoryEstimate::for_resources(&r, &explicit, 32).unwrap();
    // Same reservation, not the same allocation: F32 uses four-byte weight tiles
    // and no extra activation pack. Admission calls that exact backend plan.
    assert_eq!(a.weight_bytes_upper_bound, b.weight_bytes_upper_bound);
    assert_eq!(a.workspace_bytes_upper_bound, b.workspace_bytes_upper_bound);
    for precision in [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::F32Tf32x3,
    ] {
        let too_small = options(&r)
            .with_hybrid_cuda_numeric_fp8_precision(limits, 1, precision)
            .unwrap();
        assert!(HybridCudaMemoryEstimate::for_resources(&r, &too_small, 32).is_err());
    }
    // For eight rows the BF16 activation pack alone costs >=128 bytes, while
    // this tiny F32 image's largest decoded weight row fits in 64 bytes.
    let f32_small = options(&r)
        .with_hybrid_cuda_numeric_fp8_precision(limits, 64, NumericFp8Precision::F32Tf32x3)
        .unwrap();
    HybridCudaMemoryEstimate::for_resources(&r, &f32_small, 32).unwrap();
    let bf16_small = options(&r)
        .with_hybrid_cuda_numeric_fp8(limits, 64)
        .unwrap();
    assert!(HybridCudaMemoryEstimate::for_resources(&r, &bf16_small, 32).is_err());
    typed(
        GenericDecoderRunner::<HybridCpuDecoder>::hybrid_cpu(r, tokenizer(&f), explicit)
            .err()
            .expect("no F32 numeric CPU fallback"),
    );
}

#[test]
#[ignore = "requires single CUDA GPU + NAS F32 full40 oracle; strict all rows, 900s timeout"]
fn qwen35_35b_numeric_f32_full40_hello_prefill_decode_matches_reference() {
    full40_case("hello", NumericFp8Precision::F32Tf32x3);
}

#[test]
#[ignore = "requires single CUDA GPU + NAS F32 full40 oracle; strict all rows, 900s timeout"]
fn qwen35_35b_numeric_f32_full40_capital_prefill_decode_matches_reference() {
    full40_case("capital", NumericFp8Precision::F32Tf32x3);
}

fn oracle_ids(oracle: &Oracle, name: &str) -> Vec<usize> {
    let t = &oracle.header[name];
    assert_eq!(t["dtype"], "I64");
    let a = t["data_offsets"][0].as_u64().unwrap();
    let b = t["data_offsets"][1].as_u64().unwrap();
    let payload = CheckpointTensorReader::new(1 << 20)
        .read_slice(&CheckpointTensorSlice {
            name: name.into(),
            role: TensorRole::Unknown,
            path: oracle.path.clone(),
            offset: oracle.start + a,
            bytes: b - a,
            dtype: CheckpointDType::I64,
            shape: serde_json::from_value(t["shape"].clone()).unwrap(),
        })
        .unwrap();
    payload
        .bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| usize::try_from(i64::from_le_bytes(*b)).unwrap())
        .collect()
}

pub(super) fn full40_case(case: &str, precision: NumericFp8Precision) {
    let (profile, schema, reference_env, default_dir, numeric_scratch) = match precision {
        NumericFp8Precision::F32Tf32x3 => (
            "f32",
            "ferrule.qwen35-fp8-reference.f32.v1",
            "FERRULE_NUMERIC_FP8_F32_REFERENCE_DIR",
            "qwen35-fp8-reference-full40-f32-v1",
            64 * 1024 * 1024,
        ),
        NumericFp8Precision::W8A16Bf16F32Accumulate => (
            "bf16_rne",
            "ferrule.qwen35-fp8-reference.v1",
            "FERRULE_NUMERIC_FP8_W8A16_REFERENCE_DIR",
            "qwen35-fp8-reference-full40-w8a16-bf16-rne-v2",
            0,
        ),
        _ => panic!("full40 requires an explicit F32 or fused W8A16 profile"),
    };
    use ferrule_model::models::qwen35::Qwen35Adapter;
    let started = std::time::Instant::now();
    let model = std::env::var_os("FERRULE_NUMERIC_FP8_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8".into());
    // Separate reference selection prevents accidentally certifying W8A16 against F32.
    let dir = std::env::var_os(reference_env)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/validation")
                .join(default_dir)
        });
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["execution"]["precision_profile"], profile);
    assert_eq!(manifest["precision_contract"]["profile"], profile);
    assert_eq!(manifest["arguments"]["precision"], profile);
    assert_eq!(manifest["execution"]["device"], "cpu");
    assert_eq!(manifest["execution"]["tf32"], false);
    assert_eq!(manifest["status"], "complete");
    let oracle = Oracle::open_schema(&dir, case, schema);
    assert_eq!(oracle.report["calls"].as_array().unwrap().len(), 2);
    // Diagnostic capture is additive: the historical CPU acceptance still runs.
    let full_trace = std::env::var_os("FERRULE_NUMERIC_FULL_TRACE_DIR").map(PathBuf::from);
    let cuda_reference = full_trace.as_ref().map(|_| {
        assert_eq!(precision, NumericFp8Precision::W8A16Bf16F32Accumulate);
        let directory = PathBuf::from(
            std::env::var_os("FERRULE_NUMERIC_CUDA_REFERENCE_DIR")
                .expect("full trace requires the independent CUDA reference"),
        );
        let pin =
            std::env::var("FERRULE_NUMERIC_CUDA_REFERENCE_SHA256").expect("CUDA manifest pin");
        let checksum = Command::new("sha256sum")
            .arg(directory.join("manifest.json"))
            .output()
            .unwrap();
        assert!(checksum.status.success());
        assert_eq!(
            String::from_utf8(checksum.stdout)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap(),
            pin
        );
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["schema"], "ferrule.qwen35-fp8-cuda-reference.v1");
        assert_eq!(manifest["status"], "complete");
        assert_eq!(manifest["full_reference_complete"], true);
        assert_eq!(manifest["execution"]["device"], "cuda");
        assert_eq!(manifest["execution"]["precision_profile"], "bf16_rne");
        assert_eq!(manifest["execution"]["tf32"], false);
        let cpu_manifest: Value =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(
            manifest["checkpoint"]["config_sha256"],
            cpu_manifest["checkpoint"]["config_sha256"]
        );
        assert_eq!(manifest["tokenizer_files"], cpu_manifest["tokenizer_files"]);
        let native = Oracle::open_schema(&directory, case, "ferrule.qwen35-fp8-cuda-reference.v1");
        let entry = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == case)
            .unwrap();
        assert_eq!(entry["sha256"], native.report["sha256"]);
        let checksum = Command::new("sha256sum")
            .arg(directory.join(entry["report"].as_str().unwrap()))
            .output()
            .unwrap();
        assert!(checksum.status.success());
        assert_eq!(
            String::from_utf8(checksum.stdout)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap(),
            entry["report_sha256"].as_str().unwrap()
        );
        assert_eq!(native.report["calls"].as_array().unwrap().len(), 2);
        for i in 0..2 {
            for field in [
                "stage",
                "input_token_ids",
                "position_start",
                "sequence_length_after",
            ] {
                assert_eq!(
                    oracle.report["calls"][i][field], native.report["calls"][i][field],
                    "different reference trajectory"
                );
            }
        }
        serde_json::json!({"directory": directory, "manifest_sha256": pin,
            "artifact_sha256": native.report["sha256"], "calls": native.report["calls"]})
    });
    let prefill_rows = oracle.report["calls"][0]["input_token_ids"]
        .as_array()
        .unwrap()
        .len();
    assert!(prefill_rows > 0 && prefill_rows < 8);
    let (_, r) = Qwen35Adapter::bind_hf_metadata(&model).unwrap();
    assert_eq!(r.spec().layers().len(), 40);
    assert_eq!(
        r.spec()
            .layers()
            .iter()
            .filter(|l| matches!(l.attention(), Attention::GatedDeltaNet(_)))
            .count(),
        30
    );
    let vocab = r.spec().vocab_size();
    let opts = GenericDecoderOptions::standard_cpu(
        r.spec(),
        ModelFamily::Qwen35,
        WeightSource::Safetensors,
        2,
        8,
        prefill_rows,
        1,
        1 << 30,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap()
    .with_hybrid_cuda_numeric_fp8_precision(
        ExpertCacheLimits {
            max_experts: 64,
            max_bytes: 320 * 1024 * 1024,
        },
        numeric_scratch,
        precision,
    )
    .unwrap();
    let estimate = HybridCudaMemoryEstimate::for_resources(&r, &opts, 8).unwrap();
    eprintln!("{profile} full40 {case}: admission={estimate:?}");
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 8,
            state_bytes: estimate.per_sequence_state_bytes * 4,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    eprintln!(
        "{profile} full40 free/total={:?}",
        device.operators().memory_info().unwrap()
    );
    let mut runner = GpuRunner::hybrid_cuda(
        r,
        TokenizerHandle::load(&model).unwrap(),
        opts,
        device.clone(),
    )
    .unwrap();
    assert_eq!(runner.model_info().backend, precision.hybrid_backend_name());
    let module = runner.forward_executor().module();
    assert_eq!(module.expert_cache_stats().unwrap().uploads, 0);
    assert!(module.resident_parameter_bytes() <= estimate.weight_bytes_upper_bound);
    eprintln!(
        "{profile} full40 prepared resident={}, elapsed={:?}",
        module.resident_parameter_bytes(),
        started.elapsed()
    );
    // Only already-materialized top-k metadata is observed. No activation D2H
    // and no permanent trace cache; each call takes/clears at most 40 entries.
    let routes = Rc::new(RefCell::new(std::collections::BTreeMap::new()));
    let observed = Rc::clone(&routes);
    // A bounded opt-in first-divergence capture; no production logging or alternate forward.
    let trace_stage = Rc::new(std::cell::Cell::new("prefill"));
    let observed_stage = Rc::clone(&trace_stage);
    let first_trace = std::env::var_os("FERRULE_NUMERIC_FIRST_TRACE_DIR").map(PathBuf::from);
    assert!(
        !(first_trace.is_some() && full_trace.is_some()),
        "select first or full trace, not both"
    );
    let full_capture = full_trace.is_some();
    let trace_max_bytes = std::env::var("FERRULE_NUMERIC_FULL_TRACE_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2 * 1024 * 1024 * 1024);
    assert!(
        (1..=8).contains(&(trace_max_bytes / (1024 * 1024 * 1024))),
        "trace budget must be 1..8 GiB"
    );
    let capture_dir = full_trace.as_ref().map(|path| path.join(case));
    let trace = full_trace.or(first_trace).map(|path| {
        let path = path.join(case);
        std::fs::create_dir(&path).unwrap();
        let log = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.join("events.jsonl"))
            .unwrap();
        (path, log)
    });
    if let Some(directory) = &capture_dir {
        std::fs::write(directory.join("capture.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "ferrule.w8a16-causal-capture.v1", "status": "running", "case": case,
            "precision": profile, "cpu_reference_sha256": oracle.report["sha256"],
            "independent_cuda_reference": cuda_reference,
            "scope": "all existing operator seams, both stages; no alternate forward",
            "physical_kv_values": "not exposed by CudaDiagnosticEvent; do not label reconstructed KV as captured",
            "max_trace_bytes": trace_max_bytes,
            "source_files": ["crates/ferrule-model/tests/hybrid_cuda/numeric_f32.rs"],
        })).unwrap()).unwrap();
    }
    let mut trace = trace;
    let mut sequence = 0usize;
    let trace_bytes = Rc::new(std::cell::Cell::new(0usize));
    let observer_bytes = Rc::clone(&trace_bytes);
    module.set_diagnostic_trace(move |event| {
        if let Some((dir, log)) = &mut trace
            && (full_capture || (observed_stage.get() == "prefill"
                && (event.layer.is_some_and(|layer| layer < 4) || event.name == "embedding")))
        {
            use std::io::Write;
            let file = format!("{sequence:05}.f32");
            let bytes = observer_bytes.get().checked_add(event.shape.elements().checked_mul(4).unwrap()).unwrap();
            assert!(bytes <= trace_max_bytes, "diagnostic capture exceeds configured per-case budget");
            observer_bytes.set(bytes);
            let values = event.operators.download_f32_buffer(event.buffer)?;
            let payload = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            std::fs::write(dir.join(&file), payload)?;
            let tensor = |s: &CheckpointTensorSlice| serde_json::json!({
                "name": s.name, "path": s.path, "offset": s.offset, "bytes": s.bytes,
                "shape": s.shape, "dtype": format!("{:?}", s.dtype),
            });
            let parameter = event.parameter;
            let record = serde_json::json!({
                "sequence": sequence, "stage": observed_stage.get(), "layer": event.layer,
                "event": event.name, "shape": [event.shape.rows(), event.shape.width()], "file": file,
                "parameter": parameter.map(|p| p.path().to_string()),
                "role": parameter.map(|p| format!("{:?}", p.role())),
                "native_module": parameter.map(|p| p.weight().slice().name.replace("model.language_model.", "model.").trim_end_matches(".weight").to_owned()),
                "weight": parameter.map(|p| tensor(p.weight().slice())),
                "scale": parameter.and_then(|p| p.scale()).map(|p| tensor(p.slice())),
                "numeric": parameter.is_some_and(|p| p.numeric_fp8_encoding().is_some()),
                "precision": profile, "epsilon": event.norm.map(|p| p.epsilon()),
                "one_plus_weight": event.norm.map(|p| p.one_plus_weight()),
                "ids": event.routes.map(|r| r.expert_ids()),
                "weights": event.routes.map(|r| r.weights()),
            });
            writeln!(log, "{record}")?;
            sequence += 1;
        }
        if event.name == "router" {
            let r = event.routes.expect("router routes");
            let previous = observed.borrow_mut().insert(
                event.layer.unwrap(),
                (
                    event.shape.rows(),
                    r.expert_ids().to_vec(),
                    r.weights().to_vec(),
                ),
            );
            assert!(previous.is_none(), "one router per layer and call");
        }
        Ok(())
    });
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let mut pages = vec![Vec::new()];
    let evidence = if precision == NumericFp8Precision::W8A16Bf16F32Accumulate {
        std::env::var_os("FERRULE_W8A16_FULL40_REPORT_DIR").map(PathBuf::from)
    } else {
        None
    };
    let mut calls = Vec::new();
    let mut mismatches = Vec::new();
    let mut position = 0;
    for (i, stage) in ["prefill", "decode.0"].into_iter().enumerate() {
        trace_stage.set(stage);
        let tokens: Vec<u32> =
            serde_json::from_value(oracle.report["calls"][i]["input_token_ids"].clone()).unwrap();
        assert_eq!(oracle.report["calls"][i]["position_start"], position);
        if i == 1 {
            assert_eq!(tokens.len(), 1);
        }
        let logits = step(
            &mut runner,
            &mut states,
            &mut pages,
            &[(
                &tokens,
                if i == 0 {
                    ForwardPhase::Prefill
                } else {
                    ForwardPhase::Decode
                },
            )],
            i as u64 + 1,
            Ending::Publish,
        );
        position += tokens.len();
        assert_eq!(oracle.report["calls"][i]["sequence_length_after"], position);
        assert_eq!(logits.len(), tokens.len());
        assert!(logits.iter().all(|r| r.len() == vocab));
        let actual: Vec<f32> = logits.iter().flatten().copied().collect();
        if let Some(directory) = &capture_dir {
            write_causal_tensor(
                directory,
                &format!("{stage}.logits"),
                &actual,
                &trace_bytes,
                trace_max_bytes,
            );
        }
        if let Some(evidence) = &evidence {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(evidence.join(format!("{case}-{stage}.f32")))
                .unwrap();
            for value in &actual {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        calls.push(serde_json::json!({
            "stage": stage, "input_token_ids": tokens, "shape": [1, tokens.len(), vocab],
            "file": format!("{case}-{stage}.f32"),
        }));
        let label = format!("{profile} full40 {case}/{stage} all logits");
        assert_eq!(
            oracle.header[format!("{stage}.logits")]["shape"],
            serde_json::json!([1, tokens.len(), vocab])
        );
        if !report_diff(
            &actual,
            &oracle.values(&format!("{stage}.logits")),
            &label,
            2e-4,
            2e-4,
        ) {
            mismatches.push(label);
        }
        let argmax = argmax(logits.last().unwrap());
        if argmax != oracle.report["calls"][i]["next_token_id"].as_u64().unwrap() as u32 {
            mismatches.push(format!("{case}/{stage} argmax"));
        }
        let captured = std::mem::take(&mut *routes.borrow_mut());
        assert_eq!(captured.len(), 40);
        let mut slot_mismatches = 0;
        let mut set_mismatches = 0;
        for (layer, (rows, ids, weights)) in captured {
            assert_eq!(rows, tokens.len());
            let prefix = format!("{stage}.layers.{layer:02}");
            let expected_ids = oracle_ids(&oracle, &format!("{prefix}.selected_expert_ids"));
            let expected_weights = oracle.values(&format!("{prefix}.routing_weights"));
            assert_eq!(ids.len(), rows * 8);
            assert_eq!(ids.len(), expected_ids.len());
            assert_eq!(weights.len(), expected_weights.len());
            let mut aligned = Vec::with_capacity(weights.len());
            for row in 0..rows {
                let a = &ids[row * 8..(row + 1) * 8];
                let e = &expected_ids[row * 8..(row + 1) * 8];
                if a != e {
                    slot_mismatches += 1;
                    eprintln!("{case}/{prefix} row={row}: actual route={a:?}, reference={e:?}");
                }
                if a.iter().any(|id| !e.contains(id)) {
                    set_mismatches += 1;
                    mismatches.push(format!("{case}/{prefix} route row={row}"));
                }
                for id in e {
                    aligned.push(
                        a.iter()
                            .position(|v| v == id)
                            .map_or(f32::NAN, |j| weights[row * 8 + j]),
                    );
                }
            }
            if !report_diff(
                &aligned,
                &expected_weights,
                &format!("{case}/{prefix} routing_weights"),
                2e-4,
                2e-4,
            ) {
                mismatches.push(format!("{case}/{prefix} routing weights"));
            }
        }
        let stats = runner
            .forward_executor()
            .module()
            .expert_cache_stats()
            .unwrap();
        let scratch = runner
            .forward_executor()
            .module()
            .numeric_workspace_usage()
            .unwrap();
        eprintln!(
            "{profile} full40 {case}/{stage}: elapsed={:?}, argmax={argmax}, route_slot_mismatches={slot_mismatches}, route_set_mismatches={set_mismatches}, cache={stats:?}, workspace={scratch:?}",
            started.elapsed()
        );
        assert!(stats.resident_experts <= 64 && stats.evictions > 0);
        assert!(stats.peak_bytes <= 320 * 1024 * 1024);
        assert_eq!(stats.pending_upload_bytes, 0);
        assert_eq!(stats.unknown_quarantine_bytes, 0);
        assert_eq!(stats.scratch_bytes, numeric_scratch);
        assert!(scratch.1 <= scratch.0 && scratch.0 == numeric_scratch);
        if precision == NumericFp8Precision::W8A16Bf16F32Accumulate {
            assert_eq!(scratch.1, 0);
            assert!(scratch.4 > 0, "fused W8A16 must actually submit");
        }
        assert!(!stats.quarantined && !device.needs_quarantine());
        assert_eq!(states[0].core().position(), position);
        for (layer, state) in states[0].kv_state().layers().iter().enumerate() {
            if let Some(state) = state {
                assert_eq!(state.position(), position);
                let (conv, recurrent) = state.diagnostic_snapshot().unwrap();
                for (field, actual) in [("conv_states", conv), ("recurrent_states", recurrent)] {
                    let label = format!("{stage}.cache.{field}.{layer:02}");
                    if let Some(directory) = &capture_dir {
                        write_causal_tensor(
                            directory,
                            &label,
                            &actual,
                            &trace_bytes,
                            trace_max_bytes,
                        );
                    }
                    if !report_diff(
                        &actual,
                        &oracle.values(&label),
                        &format!("{case}/{label}"),
                        2e-4,
                        2e-4,
                    ) {
                        mismatches.push(label);
                    }
                }
            }
        }
    }
    runner.forward_executor().module().clear_diagnostic_trace();
    runner
        .try_release_sequence_state(states.pop().unwrap())
        .unwrap();
    assert_eq!(device.live_state_bytes(), estimate.per_sequence_state_bytes);
    assert!(!device.needs_quarantine());
    if let Some(evidence) = evidence {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(evidence.join(format!("{case}.json")))
            .unwrap();
        let report = serde_json::json!({
            "schema": "ferrule.w8a16-full40-observation.v1", "case": case,
            "backend": precision.hybrid_backend_name(), "reference_profile": profile,
            "reference_sha256": oracle.report["sha256"], "calls": calls,
            "atol": 2e-4, "rtol": 2e-4, "strict_failures": mismatches,
            "strict_passed": mismatches.is_empty(),
            "scope": "single GPU full40 prefill + one decode; all logits, routes, GDN state; not process EP",
        });
        file.write_all(serde_json::to_string_pretty(&report).unwrap().as_bytes())
            .unwrap();
    }
    if let Some(directory) = &capture_dir {
        let path = directory.join("capture.json");
        let mut capture: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        capture["status"] = "complete".into();
        capture["captured_payload_bytes"] = trace_bytes.get().into();
        capture["historical_cpu_strict_passed"] = mismatches.is_empty().into();
        capture["calls"] = serde_json::json!(
            calls
                .iter()
                .map(|call| {
                    let mut call = call.clone();
                    call["file"] = format!("{}.logits.f32", call["stage"].as_str().unwrap()).into();
                    call
                })
                .collect::<Vec<_>>()
        );
        capture["events_sha256"] = causal_sha256(&directory.join("events.jsonl")).into();
        capture["test_binary_sha256"] = causal_sha256(&std::env::current_exe().unwrap()).into();
        let mut files = std::fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| {
                Path::new(name)
                    .extension()
                    .is_some_and(|extension| extension == "f32")
            })
            .collect::<Vec<_>>();
        files.sort();
        let hashes = Command::new("sha256sum")
            .current_dir(directory)
            .args(&files)
            .output()
            .unwrap();
        assert!(hashes.status.success());
        let hashes = String::from_utf8(hashes.stdout).unwrap();
        let mut payloads = serde_json::Map::new();
        for line in hashes.lines() {
            let (digest, name) = line.split_once("  ").unwrap();
            payloads.insert(name.to_owned(), digest.into());
        }
        assert_eq!(payloads.len(), files.len());
        capture["payload_sha256"] = payloads.into();
        std::fs::write(path, serde_json::to_vec_pretty(&capture).unwrap()).unwrap();
    }
    assert!(
        mismatches.is_empty(),
        "{profile} full40 {case} strict failures (both calls executed): {mismatches:?}"
    );
}

fn write_causal_tensor(
    directory: &Path,
    key: &str,
    values: &[f32],
    used: &std::cell::Cell<usize>,
    limit: usize,
) {
    let bytes = used
        .get()
        .checked_add(values.len().checked_mul(4).unwrap())
        .unwrap();
    assert!(
        bytes <= limit,
        "diagnostic capture exceeds configured per-case budget"
    );
    used.set(bytes);
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(format!("{key}.f32")))
        .unwrap();
    for value in values {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
}

fn causal_sha256(path: &Path) -> String {
    let result = Command::new("sha256sum").arg(path).output().unwrap();
    assert!(result.status.success());
    String::from_utf8(result.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned()
}
