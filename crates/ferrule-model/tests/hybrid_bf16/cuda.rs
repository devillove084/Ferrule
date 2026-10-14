use super::fixture::*;
use ferrule_common::execution::*;
use ferrule_model::decoder::*;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::runner::{
    MultiSessionBatchProgress, MultiSessionRunner, TransactionEndIntent, TransactionEndProgress,
};
use ferrule_model::tokenizer::TokenizerHandle;
use ferrule_model::{ModelFamily, WeightSource};
use serde_json::Value;
use std::{cell::RefCell, rc::Rc};
type Runner = GenericDecoderRunner<HybridCudaDecoder>;

fn build(
    f: &Fixture,
    precision: ferrule_model::transformer::NumericFp8Precision,
) -> (Runner, HybridCudaDevice) {
    let options = GenericDecoderOptions::standard_cpu(
        f.resources.spec(),
        ModelFamily::Unknown("native-hybrid-bf16".into()),
        WeightSource::Safetensors,
        2,
        32,
        8,
        1,
        1 << 20,
        ExecutionPrecisionPolicy::bf16_compatibility(),
    )
    .unwrap();
    let numeric = f.oracle["schema"] == "ferrule.numeric-fp8-model-bf16.v1";
    let options = if numeric {
        options
            .with_hybrid_cuda_numeric_fp8_bf16_precision(
                ferrule_model::transformer::ExpertCacheLimits {
                    max_experts: 1,
                    max_bytes: 64 << 10,
                },
                if matches!(precision,
                    ferrule_model::transformer::NumericFp8Precision::Bf16ModelTiledF32Accumulate
                    | ferrule_model::transformer::NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate
                    | ferrule_model::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate)
                {
                    4096
                } else {
                    0
                },
                precision,
            )
            .unwrap()
    } else {
        options
    };
    let estimate = HybridCudaMemoryEstimate::for_resources(&f.resources, &options, 32).unwrap();
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
    let chunk_bytes = HybridMemoryPlan::bf16_chunk_workspace_bytes(&f.resources, &options).unwrap();
    let runner = Runner::hybrid_cuda(
        f.resources.clone(),
        TokenizerHandle::load(&f.dir).unwrap(),
        options,
        device.clone(),
    )
    .unwrap();
    let module = runner.forward_executor().module();
    assert_eq!(module.bf16_chunk_workspace_bytes(), chunk_bytes);
    assert_eq!(module.bf16_profile().is_some(), !numeric);
    assert_eq!(module.numeric_bf16_profile().is_some(), numeric);
    if numeric {
        assert_eq!(module.numeric_fp8_precision(), Some(precision));
        assert!(module.resident_parameter_bytes() < estimate.weight_bytes_upper_bound);
        assert_eq!(module.expert_cache_stats().unwrap().resident_experts, 0);
    } else {
        assert!(module.numeric_fp8_precision().is_none());
        assert_eq!(
            module.resident_parameter_bytes(),
            estimate.weight_bytes_upper_bound
        );
    }
    (runner, device)
}

// A single-sequence transaction request, using the public runner's existing
// prepare/execute/end API. The production runner/journal is not replaced.
fn step(
    runner: &mut Runner,
    state: &mut CudaHybridSequenceState,
    pages: &mut Vec<KvPageId>,
    tokens: &[u32],
    id: u64,
    publish: bool,
    cow: bool,
) -> Vec<f32> {
    let phase = if state.core().position() == 0 {
        ForwardPhase::Prefill
    } else {
        ForwardPhase::Decode
    };
    step_in_phase(runner, state, pages, tokens, (id, publish, cow), phase)
}

fn step_in_phase(
    runner: &mut Runner,
    state: &mut CudaHybridSequenceState,
    pages: &mut Vec<KvPageId>,
    tokens: &[u32],
    transaction: (u64, bool, bool),
    phase: ForwardPhase,
) -> Vec<f32> {
    let (id, publish, cow) = transaction;
    let start = state.core().position();
    let generation = state.core().generation();
    let mut staged = pages.clone();
    let mut newly = Vec::new();
    let replacement = if cow {
        assert!(!start.is_multiple_of(2));
        let logical = start / 2;
        let target = KvPageId((id * 64) as u32);
        let source = staged[logical];
        staged[logical] = target;
        Some(KvCowReplacement {
            logical_page: logical,
            source,
            replacement: target,
        })
    } else {
        None
    };
    while staged.len() < (start + tokens.len()).div_ceil(2) {
        let page = KvPageId((id * 64 + staged.len() as u64 + 1) as u32);
        staged.push(page);
        newly.push(page);
    }
    let batch = ExecutionBatch::new(
        match phase {
            ForwardPhase::Prefill => ForwardMode::Prefill,
            ForwardPhase::Decode => ForwardMode::Decode,
        },
        tokens.to_vec(),
        (start..start + tokens.len()).map(|p| p as u32).collect(),
        (start..start + tokens.len())
            .map(|p| Some(KvWriteSlot::new(staged[p / 2].0 * 2 + (p % 2) as u32)))
            .collect(),
        vec![LogitsRequest::Full; tokens.len()],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            phase,
            0..tokens.len() as u32,
            start as u32,
            (start + tokens.len()) as u32,
            0..staged.len() as u32,
        )],
        staged.iter().map(|p| KvBlockId::new(p.0)).collect(),
    );
    let reservations = [KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: start..start + tokens.len(),
        newly_allocated: newly,
        generation,
        execution_generation: generation,
        cow_replacement: replacement,
    }];
    let tx = ExecutionTransactionId::new(id).unwrap();
    runner
        .prepare_multi_session_batch(tx, std::slice::from_mut(state), &batch, &reservations)
        .unwrap();
    let MultiSessionBatchProgress::Complete(output) = runner
        .execute_multi_session_batch_progress(tx, std::slice::from_mut(state), &batch)
        .unwrap()
    else {
        panic!("native BF16 unexpectedly waits")
    };
    assert_eq!(
        state.core().position(),
        start,
        "working state must not publish during forward"
    );
    let intent = if publish {
        TransactionEndIntent::Publish
    } else {
        TransactionEndIntent::Abort
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match runner
            .end_transaction(tx, std::slice::from_mut(state), intent)
            .unwrap()
        {
            TransactionEndProgress::Complete => break,
            TransactionEndProgress::Pending => {
                assert!(std::time::Instant::now() < deadline, "terminal deadline")
            }
        }
        std::thread::yield_now();
    }
    if publish {
        *pages = staged;
    } else {
        assert_eq!(state.core().position(), start);
    }
    output
        .logits
        .into_iter()
        .flat_map(|r| match r.logits {
            LogitsOutput::Full(v) => v,
            _ => panic!("full logits"),
        })
        .collect()
}
fn close(actual: &[f32], expected: &[f32], label: &str, bf16: bool) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut max = 0.0_f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && e.is_finite(), "{label}[{i}]: nonfinite");
        let error = (f64::from(a) - f64::from(e)).abs();
        max = max.max(error);
        assert!(
            error <= 0.00025 + 0.012 * f64::from(e).abs(),
            "{label}[{i}]: CUDA={a}, HF={e}"
        );
        if bf16 {
            assert_eq!(
                a.to_bits() & 0xffff,
                0,
                "{label} must cross an actual BF16 boundary"
            );
        }
    }
    eprintln!("{label}: elements={} max_abs={max:.9e}", actual.len());
}
fn state_values(state: &CudaHybridSequenceState) -> (Vec<f32>, Vec<f32>) {
    assert!(
        state.kv_state().layers().is_empty(),
        "no F32 history mirror"
    );
    state.kv_state().bf16_layers().unwrap()[0]
        .as_ref()
        .unwrap()
        .diagnostic_snapshot()
        .unwrap()
}
fn check_state(state: &CudaHybridSequenceState, case: &Value) {
    let (conv, ssm) = state_values(state);
    close(&conv, &values(&case["conv"]), "conv history", true);
    close(&ssm, &values(&case["ssm"]), "F32 SSM", false);
    assert_eq!(state.kv_state().resident_bytes(), 512);
}
fn check_kv(runner: &mut Runner, pages: &[KvPageId], case: &Value) {
    let mut planes = [Vec::new(), Vec::new()];
    for &page in pages {
        runner.preempt_kv_pages(&[page]).unwrap();
        let snapshot = runner.backend().pool().typed_host_snapshot(page).unwrap();
        for (i, plane) in snapshot.planes.iter().enumerate() {
            assert_eq!(plane.element_type(), KvElementType::Bf16);
            assert!(plane.f32_bits().is_none());
            assert_eq!(plane.byte_len(), 2 * 8 * 2);
            planes[i].extend(
                plane
                    .bf16_words()
                    .unwrap()
                    .iter()
                    .map(|&v| f32::from_bits(u32::from(v) << 16)),
            );
        }
        runner.restore_kv_pages(&[page]).unwrap();
    }
    for (i, key) in ["key", "value"].iter().enumerate() {
        let expected = values(&case[key]);
        planes[i].truncate(expected.len());
        close(&planes[i], &expected, key, true);
    }
}
#[derive(Default)]
struct Trace {
    residuals: Vec<Vec<f32>>,
    routes: Vec<(Vec<usize>, Vec<f32>)>,
    working_conv: Vec<Vec<f32>>,
    working_ssm: Vec<Vec<f32>>,
}
fn check_trace(trace: &Rc<RefCell<Trace>>, case: &Value) {
    let mut trace = trace.borrow_mut();
    assert_eq!(trace.routes.len(), 2);
    for (i, (ids, weights)) in trace.routes.iter().enumerate() {
        let expected_ids: Vec<Vec<usize>> =
            serde_json::from_value(case["routes"][i]["ids"].clone()).unwrap();
        assert_eq!(
            ids,
            &expected_ids.into_iter().flatten().collect::<Vec<_>>(),
            "router experts"
        );
        close(
            weights,
            &values(&case["routes"][i]["weights"]),
            "F32 route weights",
            false,
        );
    }
    // Check routes first: a different expert is not a same-input math failure.
    // Each MoE emits routed+shared addition between its two layer residuals.
    assert_eq!(trace.residuals.len(), 6);
    for (expected, index) in [0, 2, 3, 5].into_iter().enumerate() {
        close(
            &trace.residuals[index],
            &values(&case["residuals"][expected]),
            "hidden/residual",
            true,
        );
    }
    assert_eq!(trace.working_conv.len(), 1);
    assert_eq!(trace.working_ssm.len(), 1);
    close(
        &trace.working_conv[0],
        &values(&case["conv"]),
        "working conv",
        true,
    );
    close(
        &trace.working_ssm[0],
        &values(&case["ssm"]),
        "working F32 SSM",
        false,
    );
    trace.residuals.clear();
    trace.routes.clear();
    trace.working_conv.clear();
    trace.working_ssm.clear();
}

#[test]
#[ignore = "sole GPU owner only; native HF full model/cached decode/COW/rollback/restore"]
fn native_bf16_hybrid_full_state_decode_cow_abort_restore_matches_hf() {
    full_state(Fixture::new());
}

#[test]
#[ignore = "sole GPU owner only; raw FP8/BF16 scales against explicit HF profile tie policy"]
fn numeric_fp8_bf16_hybrid_full_state_decode_cow_abort_restore_matches_hf() {
    full_state(Fixture::numeric_profile());
}

fn full_state(fixture: Fixture) {
    full_state_with_precision(
        fixture,
        ferrule_model::transformer::NumericFp8Precision::Bf16ModelF32Accumulate,
    );
}

pub(super) fn full_state_with_precision(
    fixture: Fixture,
    precision: ferrule_model::transformer::NumericFp8Precision,
) {
    let (mut runner, device) = build(&fixture, precision);
    eprintln!(
        "hybrid fixture: schema={} profile={} oracle={} router_ties={}; rows/logits/KV/conv=BF16 SSM=F32",
        fixture.oracle["schema"],
        fixture.oracle["profile"],
        fixture.oracle["oracle_contract"],
        fixture.oracle["router_tie_policy"],
    );
    let trace = Rc::new(RefCell::new(Trace::default()));
    let capture = Rc::clone(&trace);
    let mut diagnostic_file = std::env::var_os("FERRULE_TINY_BF16_TRACE")
        .map(|path| std::fs::File::create(path).unwrap());
    runner
        .forward_executor()
        .module()
        .set_diagnostic_trace(move |event| {
            if let Some(file) = diagnostic_file.as_mut() {
                use std::io::Write;
                let record = serde_json::json!({
                    "layer": event.layer, "name": event.name,
                    "parameter": event.parameter.map(|p| format!("{:?}", p.id())),
                    "residency": event.parameter.map(|p| format!("{:?}", p.residency())),
                    "shape": [event.shape.rows(), event.shape.width()],
                    "values": event.operators.download_f32_buffer(event.buffer)?,
                    "ids": event.routes.map(|r| r.expert_ids()),
                    "weights": event.routes.map(|r| r.weights()),
                });
                writeln!(file, "{record}").unwrap();
            }
            if event.name == "residual.output" {
                capture
                    .borrow_mut()
                    .residuals
                    .push(event.operators.download_f32_buffer(event.buffer)?);
            }
            if event.name == "gdn.conv_state" {
                capture
                    .borrow_mut()
                    .working_conv
                    .push(event.operators.download_f32_buffer(event.buffer)?);
            }
            if event.name == "gdn.recurrent_state" {
                capture
                    .borrow_mut()
                    .working_ssm
                    .push(event.operators.download_f32_buffer(event.buffer)?);
            }
            if let Some(routes) = event.routes {
                capture
                    .borrow_mut()
                    .routes
                    .push((routes.expert_ids().to_vec(), routes.weights().to_vec()));
            }
            Ok(())
        });
    let mut parent = runner.create_sequence_state().unwrap();
    let mut pages = Vec::new();
    let cases = &fixture.oracle["cases"];
    let logits = step(
        &mut runner,
        &mut parent,
        &mut pages,
        &[1, 3, 2],
        1,
        true,
        false,
    );
    let expected: Vec<Vec<f32>> = serde_json::from_value(cases[0]["logits"].clone()).unwrap();
    close(
        &logits,
        &expected.into_iter().flatten().collect::<Vec<_>>(),
        "prefill logits",
        true,
    );
    check_trace(&trace, &cases[0]);
    check_state(&parent, &cases[0]);
    check_kv(&mut runner, &pages, &cases[0]);
    device.operators().reset_counters();
    let mut fork = runner.fork_sequence_state_from(&parent, 3).unwrap();
    assert_eq!(device.operators().counters().device_to_host_copies, 0);
    assert!(runner.fork_sequence_state_from(&parent, 2).is_err());
    let mut fork_pages = pages.clone();
    let original = state_values(&parent);
    let aborted = step(&mut runner, &mut parent, &mut pages, &[5], 2, false, true);
    close(
        &aborted,
        &values(&cases[1]["logits"][0]),
        "aborted COW logits",
        true,
    );
    assert_eq!(original, state_values(&parent));
    check_trace(&trace, &cases[1]);
    check_kv(&mut runner, &pages, &cases[0]);
    for (case, id, tokens, cow) in [(1, 3, vec![5], true), (2, 4, vec![7], false)] {
        let got = step(&mut runner, &mut parent, &mut pages, &tokens, id, true, cow);
        let expected: Vec<Vec<f32>> =
            serde_json::from_value(cases[case]["logits"].clone()).unwrap();
        close(
            &got,
            &expected.into_iter().flatten().collect::<Vec<_>>(),
            "cached decode logits",
            true,
        );
        check_trace(&trace, &cases[case]);
        check_state(&parent, &cases[case]);
        check_kv(&mut runner, &pages, &cases[case]);
    }
    let parent_before = state_values(&parent);
    let got = step(&mut runner, &mut fork, &mut fork_pages, &[9], 5, true, true);
    close(&got, &values(&cases[3]["logits"][0]), "fork logits", true);
    check_trace(&trace, &cases[3]);
    check_state(&fork, &cases[3]);
    check_kv(&mut runner, &fork_pages, &cases[3]);
    assert_eq!(parent_before, state_values(&parent));
    runner.try_release_sequence_state(fork).unwrap();
    runner.reset_sequence_state(&mut parent).unwrap();
    pages.clear();
    let got = step(
        &mut runner,
        &mut parent,
        &mut pages,
        &[1, 3, 2, 5, 7],
        6,
        true,
        false,
    );
    let expected: Vec<Vec<f32>> = serde_json::from_value(cases[4]["logits"].clone()).unwrap();
    close(
        &got,
        &expected.into_iter().flatten().collect::<Vec<_>>(),
        "whole prefill logits",
        true,
    );
    check_trace(&trace, &cases[4]);
    check_state(&parent, &cases[4]);
    check_kv(&mut runner, &pages, &cases[4]);
    runner.try_release_sequence_state(parent).unwrap();
    assert_eq!(device.live_state_bytes(), 512, "only default state remains");
    if fixture.oracle["schema"] == "ferrule.numeric-fp8-model-bf16.v1" {
        let module = runner.forward_executor().module();
        let cache = module.expert_cache_stats().unwrap();
        assert!(cache.evictions > 0);
        assert!(cache.resident_experts <= 1);
        assert!(cache.peak_bytes <= 64 << 10);
        let (reserved, allocated, allocations, reuses, submissions) =
            module.numeric_workspace_usage().unwrap();
        if precision == ferrule_model::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate
        {
            // Precision rebinding retires the previous workspace; the final M1
            // routed baseline is fused, while multi-row calls used bounded tiled scratch.
            assert_eq!(reserved, 4096);
            assert!(allocated <= reserved);
            assert!(allocations > 0 && reuses > 0);
        } else if matches!(
            precision,
            ferrule_model::transformer::NumericFp8Precision::Bf16ModelTiledF32Accumulate
                | ferrule_model::transformer::NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate
        ) {
            use ferrule_backend::cuda_plan::{
                NumericFp8Layout, NumericFp8LinearPlan, NumericFp8Precision, NumericFp8ScaleType,
            };
            assert_eq!(reserved, 4096);
            // This sequence executes M1/M3/M5. The owner retains the largest
            // submitted tile, not zero (fused) or the unexecuted M8 capacity.
            let expected = fixture
                .resources
                .state_dict()
                .parameters()
                .iter()
                .filter_map(|p| {
                    p.numeric_fp8_encoding()
                        .map(|encoding| p.numeric_fp8_source(encoding).unwrap())
                })
                .flat_map(|source| {
                    let [n, k] = source.matrix_shape();
                    [1, 3, 5].map(move |rows| {
                        NumericFp8LinearPlan::new(
                            NumericFp8Layout {
                                n,
                                k,
                                row_origin: 0,
                                column_origin: 0,
                                scale_type: NumericFp8ScaleType::Bf16,
                            },
                            rows,
                            reserved,
                            NumericFp8Precision::Bf16RneF32Accumulate,
                        )
                        .unwrap()
                        .workspace_requirements()
                        .bytes as usize
                    })
                })
                .max()
                .unwrap();
            assert_eq!(allocated, expected);
            assert!(allocated <= reserved);
            assert!(allocations > 0 && reuses > 0);
        } else {
            assert_eq!(allocated, 0);
        }
        assert!(submissions > 0);
        eprintln!(
            "numeric BF16 cache={cache:?}; numeric workspace={allocated}; submissions={submissions}"
        );
    }
    runner.shutdown().unwrap();
    drop(runner);
    assert_eq!(device.live_state_bytes(), 0);
    assert!(!device.needs_quarantine());
    eprintln!("hybrid lifecycle complete: state_bytes=0; quarantine=false");
}

#[path = "chunk/history_gpu.rs"]
mod history_gpu;
