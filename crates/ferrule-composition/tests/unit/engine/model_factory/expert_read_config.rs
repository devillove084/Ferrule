//! Dedicated CPU metadata tests. Sparse headers are not inference-ready weights.
use super::*;
use ferrule_model::transformer::{ExpertReadPipelineOptions, ExpertReadSchedule};

use super::mixed_serving::fixture;

fn read(schedule: ExpertReadSchedule) -> ExpertReadPipelineOptions {
    ExpertReadPipelineOptions {
        schedule,
        max_host_bytes: 64 << 20,
        max_metadata_bytes: 8 << 20,
        worker_stack_bytes: 2 << 20,
    }
}
fn options() -> ModelFactoryOptions {
    let mut o = tests::options();
    o.max_tensor_mebibytes = 1024;
    o.driver_config.ctx_size = 128;
    o.driver_config.enable_native_proposals = false;
    o.scheduler_config.max_batch_tokens = 8;
    o.scheduler_config.max_active_sequences = 2;
    o.scheduler_config.max_decode_batch = 1;
    o.scheduler_config.prefix_cache_capacity_pages = 0;
    o.qwen35_moe_capacity = Some(Qwen35MoeCapacityLimits {
        max_experts: 8,
        max_bytes: 320 << 20,
        scratch_bytes: 0,
        workspace_bytes: 2 << 30,
        max_positions: 128,
        max_sequences: 2,
        max_batch_tokens: 8,
        ..Default::default()
    });
    o
}
fn prepare(
    f: &fixture::Fixture,
    read: Option<ExpertReadPipelineOptions>,
) -> Result<ResidentModelBuildPlan> {
    ResidentModelPlanner.prepare_numeric_fp8_bf16_serving(
        &AutoConfig::from_pretrained(f.dir.path())?,
        BackendSelection::Cuda,
        None,
        options(),
        ServingModelOptions {
            expert_read_pipeline: read,
            model_precision: Some(ModelPrecision::Bf16),
            ..Default::default()
        },
        NumericStoragePolicy::E4M3FnBf16Scales,
    )
}

#[test]
fn expert_read_disabled_cpu_profile_is_a_noop_and_opt_in_rejects_before_io() {
    let config =
        AutoConfig::from_descriptor(tests::descriptor(ModelFamily::Qwen35, "not-opened", 24));
    let plan = ResidentModelPlanner
        .prepare_serving(
            &config,
            BackendSelection::Cpu,
            None,
            tests::options(),
            ServingModelOptions::default(),
        )
        .unwrap();
    assert_eq!(plan.expert_read_pipeline_options(), None);
    for backend in [BackendSelection::Cpu, BackendSelection::Cuda] {
        let error = ResidentModelPlanner
            .prepare_serving(
                &config,
                backend,
                None,
                tests::options(),
                ServingModelOptions {
                    expert_read_pipeline: Some(read(ExpertReadSchedule::Blocking)),
                    ..Default::default()
                },
            )
            .err()
            .expect("unsupported without source IO");
        assert!(
            error.to_string().contains("expert read pipeline"),
            "{error}"
        );
    }
}

#[test]
fn expert_read_invalid_bounds_reject_before_source_io() {
    for value in [
        ExpertReadPipelineOptions {
            max_host_bytes: u64::MAX,
            ..read(ExpertReadSchedule::Blocking)
        },
        ExpertReadPipelineOptions {
            max_host_bytes: 0,
            ..read(ExpertReadSchedule::Blocking)
        },
        ExpertReadPipelineOptions {
            max_metadata_bytes: 0,
            ..read(ExpertReadSchedule::Blocking)
        },
        ExpertReadPipelineOptions {
            worker_stack_bytes: 0,
            ..read(ExpertReadSchedule::Blocking)
        },
        ExpertReadPipelineOptions {
            max_metadata_bytes: u64::MAX,
            ..read(ExpertReadSchedule::Blocking)
        },
        ExpertReadPipelineOptions {
            max_metadata_bytes: u64::MAX - (1 << 20),
            worker_stack_bytes: 2 << 20,
            ..read(ExpertReadSchedule::Lookahead)
        },
        ExpertReadPipelineOptions {
            max_metadata_bytes: 62 << 20,
            worker_stack_bytes: 2 << 20,
            ..read(ExpertReadSchedule::Blocking)
        },
    ] {
        let error = ServingModelOptions {
            expert_read_pipeline: Some(value),
            ..Default::default()
        }
        .validate_before_model_io(
            BackendSelection::Cuda,
            &tests::options(),
            &Default::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("expert read pipeline"));
    }
}

#[test]
fn expert_read_both_schedules_reuse_admission_and_preserve_device_ledger() {
    let f = fixture::Fixture::new();
    let disabled = prepare(&f, None).unwrap();
    assert!(disabled.prepared_source_plan.is_none());
    assert!(disabled.serving_resource_requirements().unwrap().is_none());
    assert!(disabled.mixed_tp2_cache_requirements().unwrap().is_none());
    let base = disabled.numeric_bf16_memory_plan().unwrap().unwrap();
    assert_eq!(base.expert_read_pipeline_options(), None);
    assert_eq!(base.runner_options().expert_read_pipeline_options(), None);
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let plan = prepare(&f, Some(read(schedule))).unwrap();
        assert_eq!(plan.expert_read_pipeline_options(), Some(read(schedule)));
        let source = plan.prepared_source_plan.as_ref().unwrap().clone();
        assert!(plan.serving_resource_requirements().unwrap().is_none());
        assert!(plan.mixed_tp2_cache_requirements().unwrap().is_none());
        assert!(plan.mixed_tp2_device_requirements().unwrap().is_none());
        let admitted = plan.numeric_bf16_memory_plan().unwrap().unwrap();
        assert_eq!(
            admitted.expert_read_pipeline_options(),
            Some(read(schedule))
        );
        assert_eq!(
            admitted.runner_options().expert_read_pipeline_options(),
            Some(read(schedule))
        );
        assert_eq!(
            admitted
                .runner_options()
                .hybrid_cuda_numeric_fp8_precision(),
            base.runner_options().hybrid_cuda_numeric_fp8_precision()
        );
        assert_eq!(admitted.model_estimate(), base.model_estimate());
        assert_eq!(admitted.model_budget(), base.model_budget());
        assert_eq!(admitted.total_device_bytes(), base.total_device_bytes());
        assert!(admitted.resources().host_experts().is_none());
        assert!(std::sync::Arc::ptr_eq(
            &source,
            plan.prepared_source_plan.as_ref().unwrap()
        ));
        plan.build_with(|request| -> Result<()> {
            assert_eq!(
                request.resolved.execution.expert_read_pipeline,
                Some(read(schedule))
            );
            assert!(std::sync::Arc::ptr_eq(
                &source,
                request.prepared_source_plan.as_ref().unwrap()
            ));
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn expert_read_resource_queries_reject_missing_tp2_and_crossed_source_payloads() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::expert_read_resource_queries_reject_missing_tp2_and_crossed_source_payloads"
        ),
        || {
            let f = fixture::Fixture::new();
            let mut tp1 = prepare(&f, Some(read(ExpertReadSchedule::Lookahead))).unwrap();
            let mut tp2_options = options();
            tp2_options.max_layers = None;
            tp2_options.qwen35_moe_capacity = None;
            tp2_options.scheduler_config.prefill_chunk_size = 8;
            tp2_options.kv_cache_mebibytes = Some(16);
            let mut tp2 = ResidentModelPlanner
                .prepare_serving(
                    &AutoConfig::from_pretrained(f.dir.path()).unwrap(),
                    BackendSelection::Cuda,
                    None,
                    tp2_options,
                    ServingModelOptions {
                        model_profile: Some(ServingModelProfile::Qwen35MixedBf16),
                        parallel: Some(PipelineBuildOptions {
                            parallelism: ferrule_common::ParallelismPlan {
                                tensor_parallel: 2,
                                ..Default::default()
                            },
                            devices: Some(vec![3, 1]),
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(tp1.serving_resource_requirements().unwrap().is_none());
            assert_eq!(
                tp2.serving_resource_requirements().unwrap().unwrap().len(),
                2
            );
            let source = tp2.prepared_source_plan.take();
            assert!(tp2.serving_resource_requirements().is_err());
            assert!(tp2.mixed_tp2_cache_requirements().is_err());
            tp2.prepared_source_plan = source;
            std::mem::swap(&mut tp1.prepared_source_plan, &mut tp2.prepared_source_plan);
            for mut plan in [tp1, tp2] {
                assert!(plan.serving_resource_requirements().is_err());
                assert!(plan.mixed_tp2_cache_requirements().is_err());
                assert!(plan.refresh_facts().is_err());
                assert!(
                    plan.build_with(|_| -> Result<()> { panic!("owner must not run") })
                        .is_err()
                );
            }
        },
    );
}

#[test]
fn expert_read_host_and_metadata_underbudget_reject_before_owner() {
    let f = fixture::Fixture::new();
    for value in [
        ExpertReadPipelineOptions {
            max_host_bytes: (10 << 20) + 1,
            ..read(ExpertReadSchedule::Blocking)
        },
        ExpertReadPipelineOptions {
            max_metadata_bytes: 1,
            ..read(ExpertReadSchedule::Lookahead)
        },
    ] {
        let error = prepare(&f, Some(value))
            .err()
            .expect("source-derived host cap must reject");
        assert!(
            error.to_string().contains("host budget")
                || error.to_string().contains("metadata budget"),
            "{error}"
        );
    }
}

#[test]
fn expert_read_readonly_budget_and_schedule_drift_invalidates_prepared_admission() {
    let f = fixture::Fixture::new();
    let mut plan = prepare(&f, Some(read(ExpertReadSchedule::Blocking))).unwrap();
    let frozen = plan.request.clone();
    for field in 0..4 {
        plan.request = frozen.clone();
        let value = plan
            .request
            .execution
            .expert_read_pipeline
            .as_mut()
            .unwrap();
        match field {
            0 => value.schedule = ExpertReadSchedule::Lookahead,
            1 => value.max_host_bytes += 1 << 20,
            2 => value.max_metadata_bytes += 1 << 20,
            _ => value.worker_stack_bytes += 1 << 20,
        }
        assert!(
            plan.numeric_bf16_memory_plan()
                .unwrap_err()
                .to_string()
                .contains("changed")
        );
        assert!(plan.serving_resource_requirements().is_err());
        assert!(plan.mixed_tp2_cache_requirements().is_err());
        assert!(
            plan.prepared_source_plan
                .as_ref()
                .unwrap()
                .validate(&plan.request)
                .unwrap_err()
                .to_string()
                .contains("changed")
        );
    }
    let called = std::cell::Cell::new(false);
    let error = plan
        .build_with(|_| -> Result<()> {
            called.set(true);
            Ok(())
        })
        .unwrap_err();
    assert!(!called.get());
    assert!(error.to_string().contains("changed"), "{error}");
}

#[test]
fn expert_read_metadata_and_payload_snapshot_drift_reject_before_owner() {
    let f = fixture::Fixture::new();
    let plan = prepare(&f, Some(read(ExpertReadSchedule::Blocking))).unwrap();
    let config = f.dir.path().join("config.json");
    let original = std::fs::read(&config).unwrap();
    let mut changed = original.clone();
    changed.push(b' ');
    std::fs::write(&config, &changed).unwrap();
    assert!(plan.serving_resource_requirements().is_err());
    assert!(plan.mixed_tp2_cache_requirements().is_err());
    assert!(
        plan.build_with(|_| -> Result<()> { panic!("owner must not run") })
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
    std::fs::write(config, original).unwrap();
    let plan = prepare(&f, Some(read(ExpertReadSchedule::Lookahead))).unwrap();
    let weights = f.dir.path().join("model.safetensors");
    let len = std::fs::metadata(&weights).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(weights)
        .unwrap()
        .set_len(len + 1)
        .unwrap();
    assert!(plan.serving_resource_requirements().is_err());
    assert!(plan.mixed_tp2_cache_requirements().is_err());
    assert!(
        plan.build_with(|_| -> Result<()> { panic!("owner must not run") })
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
}
