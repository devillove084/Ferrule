//! Constructor admission on CPU and the actual opt-in CUDA runner/provider path.
#[allow(dead_code)]
#[path = "hybrid_bf16/fixture.rs"]
mod fixture;

use ferrule_model::decoder::{GenericDecoderOptions, HybridMemoryPlan};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{
    ExpertCacheLimits, ExpertReadPipelineOptions, ExpertReadSchedule, NumericFp8Precision,
};
use ferrule_model::{ModelFamily, WeightSource};

fn read_options(schedule: ExpertReadSchedule) -> ExpertReadPipelineOptions {
    ExpertReadPipelineOptions {
        schedule,
        max_host_bytes: 2 << 20,
        max_metadata_bytes: 256 << 10,
        worker_stack_bytes: 1 << 20,
    }
}

fn options(f: &fixture::Fixture, bf16: bool) -> GenericDecoderOptions {
    let base = GenericDecoderOptions::standard_cpu(
        f.resources.spec(),
        ModelFamily::Unknown("expert-read-constructor".into()),
        WeightSource::Safetensors,
        2,
        32,
        1,
        1,
        1 << 20,
        if bf16 {
            ExecutionPrecisionPolicy::bf16_compatibility()
        } else {
            ExecutionPrecisionPolicy::f32()
        },
    )
    .unwrap();
    let limits = ExpertCacheLimits {
        max_experts: 1,
        max_bytes: 64 << 10,
    };
    if bf16 {
        base.with_hybrid_cuda_numeric_fp8_bf16(limits, 0).unwrap()
    } else {
        base.with_hybrid_cuda_numeric_fp8_precision(
            limits,
            0,
            NumericFp8Precision::W8A16Bf16F32Accumulate,
        )
        .unwrap()
    }
}

#[test]
fn ordinary_options_default_off_and_admit_both_schedules_without_payload_reads() {
    let f = fixture::Fixture::numeric();
    for bf16 in [false, true] {
        let base = options(&f, bf16);
        assert_eq!(base.expert_read_pipeline_options(), None);
        let baseline = HybridMemoryPlan::for_resources(&f.resources, &base, 32).unwrap();
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = read_options(schedule);
            let enabled = base.clone().with_expert_read_pipeline(read).unwrap();
            assert_eq!(enabled.expert_read_pipeline_options(), Some(read));
            assert_eq!(
                HybridMemoryPlan::for_resources(&f.resources, &enabled, 32).unwrap(),
                baseline
            );
        }
    }
}

#[test]
fn constructor_source_and_metadata_underbudget_and_stale_bindings_reject() {
    let f = fixture::Fixture::numeric();
    let base = options(&f, true);
    let read = read_options(ExpertReadSchedule::Lookahead);
    for bad in [
        ExpertReadPipelineOptions {
            // Three 16x16 FP8 matrices with one BF16 scale each, twice.
            max_host_bytes: read.max_metadata_bytes + read.worker_stack_bytes as u64 + 1547,
            ..read
        },
        ExpertReadPipelineOptions {
            max_metadata_bytes: 1,
            ..read
        },
    ] {
        let enabled = base.clone().with_expert_read_pipeline(bad).unwrap();
        assert!(HybridMemoryPlan::for_resources(&f.resources, &enabled, 32).is_err());
    }
    let enabled = base.with_expert_read_pipeline(read).unwrap();
    let parameter = f
        .resources
        .state_dict()
        .parameters()
        .iter()
        .find(|p| p.role() == &ferrule_model::TensorRole::RoutedExpertGate)
        .unwrap();
    let original = &parameter.weight().slice().path;
    let replacement = f.dir.join("replacement.bin");
    std::fs::write(&replacement, std::fs::read(original).unwrap()).unwrap();
    std::fs::rename(replacement, original).unwrap();
    assert!(HybridMemoryPlan::for_resources(&f.resources, &enabled, 32).is_err());
}

#[test]
fn unsupported_nonnumeric_and_cpu_constructors_never_ignore_the_enabled_option() {
    let f = fixture::Fixture::new();
    let enabled = options(&f, true)
        .with_expert_read_pipeline(read_options(ExpertReadSchedule::Blocking))
        .unwrap();
    assert!(HybridMemoryPlan::for_resources(&f.resources, &enabled, 32).is_err());
    let f = fixture::Fixture::numeric();
    let enabled = options(&f, false)
        .with_expert_read_pipeline(read_options(ExpertReadSchedule::Blocking))
        .unwrap();
    assert!(
        ferrule_model::decoder::GenericDecoderRunner::hybrid_cpu(
            f.resources.clone(),
            ferrule_model::tokenizer::TokenizerHandle::load(&f.dir).unwrap(),
            enabled,
        )
        .is_err()
    );
}

fn assert_full_logits(
    output: &ferrule_common::execution::ExecutionOutput,
    expected_rows: usize,
    vocabulary: usize,
) -> usize {
    use ferrule_common::execution::LogitsOutput;
    assert!(expected_rows > 0 && vocabulary > 0);
    assert_eq!(output.logits.len(), expected_rows);
    for (index, row) in output.logits.iter().enumerate() {
        assert_eq!(row.input_row, index as u32);
        let LogitsOutput::Full(values) = &row.logits else {
            panic!("constructor requires full vocabulary logits")
        };
        assert_eq!(values.len(), vocabulary);
        assert!(values.iter().all(|value| value.is_finite()));
    }
    expected_rows * vocabulary
}

#[test]
fn constructor_full_logits_report_rejects_empty_rows_vocabulary_and_nonfinite() {
    use ferrule_common::execution::{ExecutionOutput, LogitsOutput, LogitsRow};
    let output = ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::Full(vec![1.; 32]))]);
    assert_eq!(assert_full_logits(&output, 1, 32), 32);
    for bad in [
        ExecutionOutput::default(),
        ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::Full(vec![]))]),
        ExecutionOutput::new(vec![LogitsRow::new(
            0,
            LogitsOutput::Full(vec![f32::NAN; 32]),
        )]),
    ] {
        assert!(std::panic::catch_unwind(|| assert_full_logits(&bad, 1, 32)).is_err());
    }
}

#[cfg(feature = "cuda")]
fn persist_constructor_report(file: &mut std::fs::File, report: &serde_json::Value) {
    use std::io::{Seek, Write};
    file.seek(std::io::SeekFrom::Start(0)).unwrap();
    file.set_len(0).unwrap();
    file.write_all(&serde_json::to_vec_pretty(report).unwrap())
        .unwrap();
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires exclusive real GPU ownership; compile-check with GPUs hidden"]
fn normal_cuda_constructor_reaches_configured_provider_for_both_model_boundaries() {
    use ferrule_common::execution::*;
    use ferrule_model::decoder::*;
    use ferrule_model::runner::{
        MultiSessionBatchProgress, MultiSessionRunner, ResidentModelRunner, TransactionEndIntent,
        TransactionEndProgress,
    };
    use ferrule_model::tokenizer::TokenizerHandle;
    let path = std::env::var_os("FERRULE_EXPERT_READ_CONSTRUCTOR_REPORT")
        .expect("explicit new report file path; no default or overwrite");
    let mut report_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    let mut report = serde_json::json!({"schema":"ferrule.expert-read-constructor-quality.v1","complete":false,"passed":false,"expected_cells":4,"cells":[],"observer_mode":"ordinary_no_diagnostic_observer","diagnostic_trace_installed":false,"read_only_resource_snapshots":true,"gpu_overlap_claim":false,"expected_logit_rows_per_cell":1,"expected_vocabulary":32,"quality_scope":"tiny actual constructor/provider evidence and nonempty finite shape; not full-model oracle acceptance","ordinary_no_observer_claim":true,"route_checks":null,"state_oracle_checks":null});
    persist_constructor_report(&mut report_file, &report);
    for bf16 in [false, true] {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let f = fixture::Fixture::numeric();
            let read = read_options(schedule);
            let options = options(&f, bf16).with_expert_read_pipeline(read).unwrap();
            let estimate =
                HybridCudaMemoryEstimate::for_resources(&f.resources, &options, 32).unwrap();
            let device = HybridCudaDevice::new_on_device(
                0,
                HybridCudaMemoryBudget {
                    kv_pages: 32,
                    state_bytes: estimate.per_sequence_state_bytes * 4,
                    weight_bytes: estimate.weight_bytes_upper_bound,
                    workspace_bytes: estimate.workspace_bytes_upper_bound,
                },
            )
            .unwrap();
            let mut runner = GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda(
                f.resources.clone(),
                TokenizerHandle::load(&f.dir).unwrap(),
                options,
                device.clone(),
            )
            .unwrap();
            let module = runner.forward_executor().module();
            assert_eq!(module.expert_read_pipeline_options(), Some(read));
            assert_eq!(module.expert_read_pipeline_stats().submitted_reads, 0);
            let configured = module.expert_read_pipeline_observation();
            assert!(configured.enabled);
            assert_eq!(configured.requested_mode, Some(schedule));
            assert_eq!(configured.effective_mode, None);
            assert!(!configured.worker_read_observed);
            assert_eq!(module.expert_cache_stats().unwrap().resident_experts, 0);
            let mut state = runner.create_sequence_state().unwrap();
            let generation = state.core().generation();
            let batch = ExecutionBatch::new(
                ForwardMode::Prefill,
                vec![1],
                vec![0],
                vec![Some(KvWriteSlot::new(2))],
                vec![LogitsRequest::Full],
                vec![ExecutionSequence::new(
                    StateSlot::new(0),
                    ForwardPhase::Prefill,
                    0..1,
                    0,
                    1,
                    0..1,
                )],
                vec![KvBlockId::new(1)],
            );
            let reservations = [KvReservationView {
                state_slot: StateSlot::new(0),
                execution_state_slot: StateSlot::new(0),
                positions: 0..1,
                newly_allocated: vec![KvPageId(1)],
                generation,
                execution_generation: generation,
                cow_replacement: None,
            }];
            let before = runner.physical_resource_snapshot();
            let cell_index = report["cells"].as_array().unwrap().len();
            report["cells"].as_array_mut().unwrap().push(serde_json::json!({"model_boundary":if bf16 {"Bf16ModelF32Accumulate"}else{"W8A16Bf16F32Accumulate"},"requested":read,"configured":configured,"before_prediction":before,"after_prediction":null,"delta":null,"passed":false}));
            persist_constructor_report(&mut report_file, &report);
            let tx = ExecutionTransactionId::new(1).unwrap();
            let executed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runner
                    .prepare_multi_session_batch(
                        tx,
                        std::slice::from_mut(&mut state),
                        &batch,
                        &reservations,
                    )
                    .unwrap();
                runner
                    .execute_multi_session_batch_progress(
                        tx,
                        std::slice::from_mut(&mut state),
                        &batch,
                    )
                    .unwrap()
            }));
            let after = runner.physical_resource_snapshot();
            report["cells"][cell_index]["after_prediction"] = serde_json::json!(after);
            report["cells"][cell_index]["delta"] = serde_json::json!(
                before
                    .expert_read_pipeline
                    .value()
                    .zip(after.expert_read_pipeline.value())
                    .and_then(|(a, b)| b.stats.checked_delta(a.stats))
            );
            persist_constructor_report(&mut report_file, &report);
            let MultiSessionBatchProgress::Complete(output) = (match executed {
                Ok(output) => output,
                Err(panic) => std::panic::resume_unwind(panic),
            }) else {
                panic!("synchronous numeric forward unexpectedly waits")
            };
            let compared = assert_full_logits(&output, 1, f.resources.spec().vocab_size());
            assert_eq!(f.resources.spec().vocab_size(), 32);
            report["cells"][cell_index]["logit_values_checked"] = serde_json::json!(compared);
            let LogitsOutput::Full(values) = &output.logits[0].logits else {
                unreachable!()
            };
            let greedy =
                values.iter().enumerate().fold(
                    0,
                    |best, (i, value)| if *value > values[best] { i } else { best },
                );
            report["cells"][cell_index]["greedy_lower_id_token"] = serde_json::json!(greedy);
            let stats = runner
                .forward_executor()
                .module()
                .expert_read_pipeline_stats();
            let observed = runner
                .forward_executor()
                .module()
                .expert_read_pipeline_observation();
            assert!(observed.enabled && observed.worker_read_observed);
            assert_eq!(observed.requested_mode, Some(schedule));
            assert_eq!(observed.effective_mode, Some(schedule));
            assert!(stats.completed_reads > 0 && stats.verified_reads > 0);
            assert!(stats.reader_interval_calls > 0 && stats.reader_interval_bytes > 0);
            assert!(stats.proved_read_bytes > 0);
            assert_eq!(
                stats.failed_reads + stats.cancelled_reads + stats.unknown_reads,
                0
            );
            assert_eq!(observed.current_charge.host_charged_bytes, 0);
            let payload_limit =
                read.max_host_bytes - read.max_metadata_bytes - read.worker_stack_bytes as u64;
            assert!(stats.peak_packed_bytes > 0 && stats.peak_packed_bytes <= payload_limit / 2);
            assert!(
                stats.peak_artifact_bytes > 0 && stats.peak_artifact_bytes <= payload_limit / 2
            );
            assert!(stats.peak_payload_bytes <= payload_limit);
            assert_eq!(stats.peak_metadata_bytes, read.max_metadata_bytes);
            assert_eq!(
                stats.peak_requested_stack_bytes,
                read.worker_stack_bytes as u64
            );
            let physical = runner.physical_resource_snapshot();
            assert_eq!(physical.expert_read_pipeline.value(), Some(&observed));
            assert_eq!(runner.physical_resource_snapshot(), physical);
            assert_eq!(
                physical.release,
                ferrule_model::resource_observability::ResourceEvidence::Unobserved
            );
            assert!(before.scope.value().is_some());
            assert!(before.prepared_plan_id.value().is_some());
            assert!(before.image_generation.value().is_some());
            assert!(report["cells"][cell_index]["delta"].is_object());
            assert_eq!(before.scope, after.scope);
            assert_eq!(before.prepared_plan_id, after.prepared_plan_id);
            assert_eq!(before.image_generation, after.image_generation);
            report["cells"][cell_index]["observed"] = serde_json::json!(observed);
            report["cells"][cell_index]["charged_limits_satisfied"] = serde_json::json!(
                stats.peak_host_charged_bytes <= read.max_host_bytes
                    && stats.peak_payload_bytes <= payload_limit
            );
            report["cells"][cell_index]["positive_worker_read"] = serde_json::json!(
                observed.worker_read_observed
                    && stats.submitted_reads > 0
                    && stats.completed_reads > 0
                    && stats.verified_reads > 0
                    && stats.proved_read_bytes > 0
            );
            report["cells"][cell_index]["same_owner"] = serde_json::json!(true);
            persist_constructor_report(&mut report_file, &report);
            assert_eq!(stats.submitted_reads, 6);
            assert_eq!(stats.unknown_bytes, 0);
            assert!(stats.peak_host_charged_bytes <= read.max_host_bytes);
            assert_eq!(
                stats.speculative_reads,
                if schedule == ExpertReadSchedule::Lookahead {
                    4
                } else {
                    0
                }
            );
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                match runner
                    .end_transaction(
                        tx,
                        std::slice::from_mut(&mut state),
                        TransactionEndIntent::Abort,
                    )
                    .unwrap()
                {
                    TransactionEndProgress::Complete => break,
                    TransactionEndProgress::Pending => {
                        assert!(std::time::Instant::now() < deadline)
                    }
                }
                std::thread::yield_now();
            }
            runner.try_release_sequence_state(state).unwrap();
            runner.shutdown().unwrap();
            drop(runner);
            assert_eq!(device.live_state_bytes(), 0);
            assert!(!device.needs_quarantine());
            report["cells"][cell_index]["state_bytes_after_shutdown"] =
                serde_json::json!(device.live_state_bytes());
            report["cells"][cell_index]["quarantine"] =
                serde_json::json!(device.needs_quarantine());
            report["cells"][cell_index]["passed"] = serde_json::json!(true);
            persist_constructor_report(&mut report_file, &report);
        }
    }
    assert_eq!(report["cells"].as_array().unwrap().len(), 4);
    let compared = report["cells"]
        .as_array()
        .unwrap()
        .iter()
        .map(|cell| cell["logit_values_checked"].as_u64().unwrap())
        .sum::<u64>();
    assert_eq!(compared, 128);
    report["total_logit_values_checked"] = serde_json::json!(compared);
    report["complete"] = serde_json::json!(true);
    report["passed"] = serde_json::json!(
        report["cells"]
            .as_array()
            .unwrap()
            .iter()
            .all(|cell| cell["passed"] == true)
    );
    persist_constructor_report(&mut report_file, &report);
}
