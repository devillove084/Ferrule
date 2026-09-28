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
    full40_case("hello");
}

#[test]
#[ignore = "requires single CUDA GPU + NAS F32 full40 oracle; strict all rows, 900s timeout"]
fn qwen35_35b_numeric_f32_full40_capital_prefill_decode_matches_reference() {
    full40_case("capital");
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
        .chunks_exact(8)
        .map(|b| usize::try_from(i64::from_le_bytes(b.try_into().unwrap())).unwrap())
        .collect()
}

fn full40_case(case: &str) {
    use ferrule_model::models::qwen35::Qwen35Adapter;
    let started = std::time::Instant::now();
    let model = std::env::var_os("FERRULE_NUMERIC_FP8_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8".into());
    // Deliberately distinct from the BF16 oracle environment variable/directory.
    let dir = std::env::var_os("FERRULE_NUMERIC_FP8_F32_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/validation/qwen35-fp8-reference-full40-f32-v1")
        });
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["execution"]["precision_profile"], "f32");
    assert_eq!(manifest["execution"]["tf32"], false);
    assert_eq!(manifest["status"], "complete");
    let oracle = Oracle::open_schema(&dir, case, "ferrule.qwen35-fp8-reference.f32.v1");
    assert_eq!(oracle.report["calls"].as_array().unwrap().len(), 2);
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
        64 * 1024 * 1024,
        NumericFp8Precision::F32Tf32x3,
    )
    .unwrap();
    let estimate = HybridCudaMemoryEstimate::for_resources(&r, &opts, 8).unwrap();
    eprintln!("F32 full40 {case}: admission={estimate:?}");
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
        "F32 full40 free/total={:?}",
        device.operators().memory_info().unwrap()
    );
    let mut runner = GpuRunner::hybrid_cuda(
        r,
        TokenizerHandle::load(&model).unwrap(),
        opts,
        device.clone(),
    )
    .unwrap();
    assert_eq!(
        runner.model_info().backend,
        "cuda-hybrid-numeric-fp8-f32-tf32x3"
    );
    let module = runner.forward_executor().module();
    assert_eq!(module.expert_cache_stats().unwrap().uploads, 0);
    assert!(module.resident_parameter_bytes() <= estimate.weight_bytes_upper_bound);
    eprintln!(
        "F32 full40 prepared resident={}, elapsed={:?}",
        module.resident_parameter_bytes(),
        started.elapsed()
    );
    // Only already-materialized top-k metadata is observed. No activation D2H
    // and no permanent trace cache; each call takes/clears at most 40 entries.
    let routes = Rc::new(RefCell::new(std::collections::BTreeMap::new()));
    let observed = Rc::clone(&routes);
    module.set_diagnostic_trace(move |event| {
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
    let mut mismatches = Vec::new();
    let mut position = 0;
    for (i, stage) in ["prefill", "decode.0"].into_iter().enumerate() {
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
        let label = format!("F32 full40 {case}/{stage} all logits");
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
        assert_eq!(
            argmax,
            oracle.report["calls"][i]["next_token_id"].as_u64().unwrap() as u32
        );
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
            "F32 full40 {case}/{stage}: elapsed={:?}, argmax={argmax}, route_slot_mismatches={slot_mismatches}, route_set_mismatches={set_mismatches}, cache={stats:?}, workspace={scratch:?}",
            started.elapsed()
        );
        assert!(stats.resident_experts <= 64 && stats.evictions > 0);
        assert!(stats.peak_bytes <= 320 * 1024 * 1024);
        assert_eq!(stats.pending_upload_bytes, 0);
        assert_eq!(stats.unknown_quarantine_bytes, 0);
        assert_eq!(stats.scratch_bytes, 64 * 1024 * 1024);
        assert!(scratch.1 <= scratch.0 && scratch.0 == 64 * 1024 * 1024);
        assert!(!stats.quarantined && !device.needs_quarantine());
        assert_eq!(states[0].core().position(), position);
        for (layer, state) in states[0].kv_state().layers().iter().enumerate() {
            if let Some(state) = state {
                assert_eq!(state.position(), position);
                let (conv, recurrent) = state.diagnostic_snapshot().unwrap();
                for (field, actual) in [("conv_states", conv), ("recurrent_states", recurrent)] {
                    let label = format!("{stage}.cache.{field}.{layer:02}");
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
    assert!(
        mismatches.is_empty(),
        "F32 full40 {case} strict failures (both calls executed): {mismatches:?}"
    );
}
