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

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires exclusive real GPU ownership; compile-check with GPUs hidden"]
fn normal_cuda_constructor_reaches_configured_provider_for_both_model_boundaries() {
    use ferrule_common::execution::*;
    use ferrule_model::decoder::*;
    use ferrule_model::runner::{
        MultiSessionBatchProgress, MultiSessionRunner, TransactionEndIntent, TransactionEndProgress,
    };
    use ferrule_model::tokenizer::TokenizerHandle;
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
            let tx = ExecutionTransactionId::new(1).unwrap();
            runner
                .prepare_multi_session_batch(
                    tx,
                    std::slice::from_mut(&mut state),
                    &batch,
                    &reservations,
                )
                .unwrap();
            let MultiSessionBatchProgress::Complete(output) = runner
                .execute_multi_session_batch_progress(tx, std::slice::from_mut(&mut state), &batch)
                .unwrap()
            else {
                panic!("synchronous numeric forward unexpectedly waits")
            };
            assert!(output.logits.into_iter().all(|row| match row.logits {
                LogitsOutput::Full(values) => values.into_iter().all(f32::is_finite),
                _ => false,
            }));
            let stats = runner
                .forward_executor()
                .module()
                .expert_read_pipeline_stats();
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
        }
    }
}
