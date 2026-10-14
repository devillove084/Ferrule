//! CPU-only CLI/read-policy tests; no model payload or device owner.
use super::*;
use crate::args::ExpertReadArgs;
use ferrule_model::transformer::{ExpertReadPipelineOptions, ExpertReadSchedule};

use super::model_precision_tests::fixture;

#[test]
fn expert_read_disabled_is_not_an_implicit_blocking_pipeline() {
    let args = serve(&["ferrule", "serve", "not-opened"]);
    assert_eq!(args.expert_read.options().unwrap(), None);
    assert_eq!(serving_options(&args).unwrap().expert_read_pipeline, None);
    for fields in [
        ExpertReadArgs {
            schedule: Some("blocking".into()),
            ..Default::default()
        },
        ExpertReadArgs {
            metadata_mb: Some(8),
            ..Default::default()
        },
        ExpertReadArgs {
            worker_stack_mb: Some(2),
            ..Default::default()
        },
    ] {
        assert!(
            fields
                .options()
                .unwrap_err()
                .to_string()
                .contains("require --expert-read-host-mb")
        );
    }
}

#[test]
fn expert_read_flags_require_explicit_host_cap_and_validate_schedule() {
    for (flag, value) in [
        ("--expert-read-schedule", "blocking"),
        ("--expert-read-schedule", "lookahead"),
        ("--expert-read-metadata-mb", "8"),
        ("--expert-read-worker-stack-mb", "2"),
    ] {
        assert!(Cli::try_parse_from(["ferrule", "serve", "not-opened", flag, value]).is_err());
    }
    assert!(
        Cli::try_parse_from([
            "ferrule",
            "serve",
            "not-opened",
            "--expert-read-host-mb",
            "64",
            "--expert-read-schedule",
            "async",
        ])
        .is_err()
    );
}

#[test]
fn expert_read_schedules_share_host_limits_without_changing_device_budget() {
    for (schedule, expected) in [
        (None, ExpertReadSchedule::Blocking),
        (Some("blocking"), ExpertReadSchedule::Blocking),
        (Some("lookahead"), ExpertReadSchedule::Lookahead),
    ] {
        let mut argv = vec![
            "ferrule",
            "serve",
            "not-opened",
            "--expert-read-host-mb",
            "64",
        ];
        if let Some(schedule) = schedule {
            argv.extend(["--expert-read-schedule", schedule]);
        }
        let args = serve(&argv);
        let read = serving_options(&args)
            .unwrap()
            .expert_read_pipeline
            .unwrap();
        assert_eq!(read.schedule, expected);
        assert_eq!(read.max_host_bytes, 64 << 20);
        assert_eq!(read.max_metadata_bytes, 8 << 20);
        assert_eq!(read.worker_stack_bytes, 2 << 20);
        assert_eq!(
            model_factory_options(&args).unwrap().qwen35_moe_capacity,
            None
        );
        assert_eq!(
            model_factory_options(&args).unwrap().qwen35_host_cache,
            None
        );
    }
    let args = serve(&[
        "ferrule",
        "serve",
        "not-opened",
        "--expert-read-host-mb",
        "64",
        "--expert-read-metadata-mb",
        "4",
        "--expert-read-worker-stack-mb",
        "3",
        "--cuda-expert-device-bytes",
        "335544320",
    ]);
    let read = serving_options(&args)
        .unwrap()
        .expert_read_pipeline
        .unwrap();
    assert_eq!(
        (read.max_metadata_bytes, read.worker_stack_bytes),
        (4 << 20, 3 << 20)
    );
    assert_eq!(
        model_factory_options(&args)
            .unwrap()
            .qwen35_moe_capacity
            .unwrap()
            .max_bytes,
        335544320
    );
}

#[test]
fn expert_read_budget_zero_overflow_and_overhead_reject() {
    for flag in [
        "--expert-read-host-mb",
        "--expert-read-metadata-mb",
        "--expert-read-worker-stack-mb",
    ] {
        for value in ["0", "18446744073709551615"] {
            let mut argv = vec!["ferrule", "serve", "not-opened"];
            if flag != "--expert-read-host-mb" {
                argv.extend(["--expert-read-host-mb", "64"]);
            }
            argv.extend([flag, value]);
            assert!(serving_options(&serve(&argv)).is_err(), "{flag} {value}");
        }
    }
    for host in ["1", "10"] {
        assert!(
            serving_options(&serve(&[
                "ferrule",
                "serve",
                "not-opened",
                "--expert-read-host-mb",
                host,
            ]))
            .is_err()
        );
    }
}

#[test]
fn expert_read_unsupported_profiles_reject_before_model_io() {
    for flags in [
        vec![],
        vec!["--backend", "cpu"],
        vec!["--model-precision", "bf16"],
        vec!["--pipeline-parallel", "2"],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--tensor-parallel",
            "2",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--engine",
            "pipeline",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--kv-execution-policy",
            "host-kv-decode",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--expert-prewarm-workers",
            "1",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--expert-host-cache-entries",
            "1",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--numeric-fp8-precision",
            "w8a16-bf16",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--expert-prewarm",
            "full",
        ],
    ] {
        let mut argv = vec![
            "ferrule",
            "serve",
            "not-opened",
            "--expert-read-host-mb",
            "64",
            "--max-batch-tokens",
            "8",
        ];
        argv.extend(flags);
        let error = prepare_model(&serve(&argv))
            .err()
            .expect("unsupported before IO");
        assert!(!error.to_string().contains("No such file"), "{error}");
        assert!(
            error.to_string().contains("expert read")
                || error.to_string().contains("prewarm")
                || error.to_string().contains("expert options")
                || error.to_string().contains("CPU execution is test-only"),
            "{error}"
        );
    }
}

#[test]
fn expert_read_stack_and_metadata_sum_overflow_rejects_before_model_io() {
    let args = serve(&[
        "ferrule",
        "serve",
        "not-opened",
        "--expert-read-host-mb",
        "64",
        "--expert-read-metadata-mb",
        "17592186044415",
        "--expert-read-worker-stack-mb",
        "2",
    ]);
    let error = prepare_model(&args).err().expect("host overhead overflow");
    assert!(
        error.to_string().contains("host overhead overflows"),
        "{error}"
    );
}

fn storage_middle64_args(model: &str, schedule: Option<&str>) -> ServeArgs {
    let mut argv = vec![
        "ferrule",
        "serve",
        model,
        "--model-profile",
        "qwen35-mixed-bf16",
        "--expert-prewarm",
        "lazy",
        "--kv-execution-policy",
        "resident",
        "--ctx-size",
        "128",
        "--max-active-sequences",
        "2",
        "--max-batch-tokens",
        "8",
        "--prefill-chunk-size",
        "8",
        "--decode-cohort-target",
        "1",
        "--decode-cohort-max-deferrals",
        "0",
        "--kv-cache-mb",
        "16",
        "--max-tensor-mb",
        "1024",
        "--cuda-expert-device-entries",
        "64",
        "--cuda-expert-device-bytes",
        "335544320",
        "--cuda-expert-scratch-mb",
        "8",
        "--cuda-workspace-mb",
        "2048",
        "--cuda-device-budget-mb",
        "7168",
    ];
    if let Some(schedule) = schedule {
        argv.extend([
            "--expert-read-host-mb",
            "64",
            "--expert-read-schedule",
            schedule,
            "--expert-read-metadata-mb",
            "8",
            "--expert-read-worker-stack-mb",
            "2",
        ]);
    }
    serve(&argv)
}

// Mirror the conservative composition admission ledger, not an allocator/RSS
// measurement. Whole projections have weight/scale extents, and path/name lengths
// matter: a sparse fixture under a long validation TMPDIR is not the study source.
#[derive(Debug, serde::Serialize)]
struct SourceReadBudget {
    routed_layers: usize,
    experts_per_layer: usize,
    experts_per_token: usize,
    max_batch_tokens: usize,
    selected_experts_per_layer: usize,
    max_projection_extents: usize,
    max_source_path_bytes: usize,
    required_metadata_bytes: u64,
    executor_metadata_bytes: u64,
    max_whole_expert_payload_bytes: u64,
}

fn source_read_budget(
    memory: &ferrule_composition::model_factory::Qwen35NumericBf16MemoryPlan,
) -> SourceReadBudget {
    use ferrule_model::checkpoint::CheckpointTensorReader;
    use ferrule_model::nn::ParameterResidency;
    use ferrule_model::transformer::FeedForward;
    use std::collections::BTreeMap;

    let resources = memory.resources();
    let runner = memory.runner_options();
    let reader = CheckpointTensorReader::new(runner.max_parameter_bytes());
    let mut experts = BTreeMap::<(usize, usize), (u64, u64, u64, usize)>::new();
    let mut budget = SourceReadBudget {
        routed_layers: 0,
        experts_per_layer: 0,
        experts_per_token: 0,
        max_batch_tokens: runner.capabilities().max_batch_tokens,
        selected_experts_per_layer: 0,
        max_projection_extents: 0,
        max_source_path_bytes: 0,
        required_metadata_bytes: 0,
        executor_metadata_bytes: 0,
        max_whole_expert_payload_bytes: 0,
    };
    for parameter in resources.state_dict().parameters() {
        let ParameterResidency::Expert { layer, expert } = *parameter.residency() else {
            continue;
        };
        assert_eq!(parameter.id(), parameter.canonical_id());
        let source = parameter
            .numeric_fp8_source(parameter.numeric_fp8_encoding().unwrap())
            .unwrap();
        assert!(source.expert_count().is_none());
        let [rows, columns] = source.matrix_shape();
        let plan = source
            .plan_tile(&reader, None, 0..rows, 0..columns)
            .unwrap();
        let entry = experts.entry((layer, expert)).or_insert((0, 4096, 4096, 0));
        entry.0 = entry
            .0
            .checked_add(plan.read_plan().storage_bytes())
            .unwrap();
        let mut variable = 0u64;
        for slice in [source.weight(), source.scale()] {
            let path_bytes = slice.path.as_os_str().len();
            budget.max_source_path_bytes = budget.max_source_path_bytes.max(path_bytes);
            variable = variable
                .checked_add(path_bytes as u64)
                .and_then(|n| n.checked_add(slice.name.len() as u64))
                .unwrap();
        }
        budget.max_projection_extents = budget
            .max_projection_extents
            .max(plan.read_plan().extents().len());
        for extent in plan.read_plan().extents() {
            variable = variable
                .checked_add(extent.path().as_os_str().len() as u64)
                .and_then(|n| n.checked_add(256))
                .unwrap();
        }
        entry.1 = entry
            .1
            .checked_add(variable.checked_mul(32).unwrap() + 8192)
            .unwrap();
        entry.2 = entry
            .2
            .checked_add(variable.checked_mul(16).unwrap() + 4096)
            .unwrap();
        entry.3 += 1;
    }
    for layer in resources.spec().layers() {
        let FeedForward::Moe(moe) = layer.feed_forward() else {
            continue;
        };
        let count = moe.router_spec().num_experts();
        let per_token = moe.router_spec().experts_per_token();
        let selected = budget
            .max_batch_tokens
            .checked_mul(per_token)
            .unwrap()
            .min(count);
        budget.routed_layers += 1;
        budget.experts_per_layer = budget.experts_per_layer.max(count);
        budget.experts_per_token = budget.experts_per_token.max(per_token);
        budget.selected_experts_per_layer = budget.selected_experts_per_layer.max(selected);
        let mut metadata = Vec::with_capacity(count);
        let mut executor_metadata = Vec::with_capacity(count);
        for expert in 0..count {
            let &(payload, composition, executor, projections) =
                experts.get(&(layer.index(), expert)).unwrap();
            assert_eq!(projections, 3);
            assert!(payload <= runner.max_parameter_bytes());
            budget.max_whole_expert_payload_bytes =
                budget.max_whole_expert_payload_bytes.max(payload);
            metadata.push(composition);
            executor_metadata.push(executor);
        }
        let worst = |mut bytes: Vec<u64>| {
            bytes.sort_unstable_by(|a, b| b.cmp(a));
            bytes
                .into_iter()
                .take(selected)
                .try_fold(4096u64, u64::checked_add)
                .unwrap()
        };
        budget.required_metadata_bytes = budget.required_metadata_bytes.max(worst(metadata));
        budget.executor_metadata_bytes =
            budget.executor_metadata_bytes.max(worst(executor_metadata));
    }
    assert_eq!(budget.routed_layers, 40);
    assert_eq!(budget.experts_per_layer, 256);
    assert_eq!(budget.experts_per_token, 8);
    assert_eq!(budget.max_batch_tokens, 8);
    assert_eq!(budget.selected_experts_per_layer, 64);
    assert_eq!(budget.max_projection_extents, 2);
    assert_eq!(budget.max_whole_expert_payload_bytes, 3 * ((1 << 20) + 128));
    budget
}

fn fixture_storage_middle64_args(
    model: &str,
    schedule: &str,
    source: &SourceReadBudget,
) -> ServeArgs {
    let mut args = storage_middle64_args(model, Some(schedule));
    // Only the synthetic source's path-dependent metadata allowance changes.
    // Keep the real study argv, host/device caps and scheduler unchanged.
    args.expert_read.metadata_mb = Some(source.required_metadata_bytes.div_ceil(1 << 20));
    let read = serving_options(&args)
        .unwrap()
        .expert_read_pipeline
        .unwrap();
    assert!(source.required_metadata_bytes <= read.max_metadata_bytes);
    assert!(read.max_metadata_bytes - source.required_metadata_bytes < 1 << 20);
    assert!(source.executor_metadata_bytes <= source.required_metadata_bytes);
    assert_eq!(read.max_host_bytes, 64 << 20);
    assert_eq!(read.worker_stack_bytes, 2 << 20);
    assert!(
        2 * source.max_whole_expert_payload_bytes
            <= read.max_host_bytes - read.max_metadata_bytes - read.worker_stack_bytes as u64
    );
    println!(
        "sparse_fixture_source_budget={} admitted_metadata_bytes={}",
        serde_json::to_string(source).unwrap(),
        read.max_metadata_bytes
    );
    args
}

#[test]
fn expert_read_sparse_fixture_metadata_budget_tracks_source_geometry() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::expert_read_sparse_fixture_metadata_budget_tracks_source_geometry"
        ),
        || {
            let f = fixture::Fixture::new();
            // Force the geometry that failed under the frozen build's long TMPDIR,
            // independent of the caller's temp directory. Payloads remain sparse.
            let long_path = f.dir.path().join("source-header-only-".repeat(8));
            std::fs::create_dir(&long_path).unwrap();
            for name in ["config.json", "model.safetensors"] {
                std::fs::rename(f.dir.path().join(name), long_path.join(name)).unwrap();
            }
            let model = long_path.to_str().unwrap();
            let base = prepare_model(&storage_middle64_args(model, None)).unwrap();
            let source = source_read_budget(&base.numeric_bf16_memory_plan().unwrap().unwrap());
            assert!(source.required_metadata_bytes > 8 << 20);
            for schedule in ["blocking", "lookahead"] {
                let error = prepare_model(&storage_middle64_args(model, Some(schedule)))
                    .err()
                    .expect("unmodified study cap rejects the long fixture path");
                assert!(error.to_string().contains("metadata budget"), "{error}");
                let args = fixture_storage_middle64_args(model, schedule, &source);
                let plan = prepare_model(&args).unwrap();
                assert!(plan.serving_resource_requirements().unwrap().is_none());
                // The next smaller CLI MiB budget is below the calculated requirement.
                let mut under = args;
                under.expert_read.metadata_mb = Some(under.expert_read.metadata_mb.unwrap() - 1);
                let error = prepare_model(&under)
                    .err()
                    .expect("computed metadata boundary");
                assert!(error.to_string().contains("metadata budget"), "{error}");
            }
        },
    );
}

#[test]
#[ignore = "requires local Qwen3.5-35B headers; CPU source admission, not inference-ready weights"]
fn expert_read_local_35b_headers_v3_cli_source_preflight_no_gpu() {
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    assert_eq!(
        std::env::var("NVIDIA_VISIBLE_DEVICES").as_deref(),
        Ok("none")
    );
    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
    let model = "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8";
    let cases: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../papers/arxiv/experiments/storage_demand/paired_read_v1/cases-v3.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let base = prepare_model(&storage_middle64_args(model, None)).unwrap();
    let base_memory = base.numeric_bf16_memory_plan().unwrap().unwrap();
    let source = source_read_budget(&base_memory);
    println!(
        "source_geometry={}",
        serde_json::to_string(&source).unwrap()
    );
    let mut failures = Vec::new();
    for (schedule, expected) in [
        ("blocking", ExpertReadSchedule::Blocking),
        ("lookahead", ExpertReadSchedule::Lookahead),
    ] {
        let argv = std::iter::once("ferrule")
            .chain(
                cases[schedule]["argv_after_binary"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_str().unwrap()),
            )
            .collect::<Vec<_>>();
        println!("v3_serve_argv={}", serde_json::to_string(&argv).unwrap());
        let args = serve(&argv);
        assert_eq!(args.model, model);
        assert_eq!(args.request_queue_capacity, 2);
        assert_eq!(args.max_inflight_requests, 2);
        assert_eq!(
            args.served_model_name.as_deref(),
            Some("qwen35-storage-demand-paper")
        );
        assert_eq!(args.host.to_string(), "127.0.0.1");
        assert_eq!(args.port, 18369);
        worker_config(&args).unwrap();
        let requested = serving_options(&args)
            .unwrap()
            .expert_read_pipeline
            .unwrap();
        assert_eq!(requested.schedule, expected);
        assert_eq!(requested.max_host_bytes, 64 << 20);
        assert_eq!(requested.max_metadata_bytes, 8 << 20);
        assert_eq!(requested.worker_stack_bytes, 2 << 20);
        match prepare_model(&args) {
            Ok(plan) => {
                assert!(plan.serving_resource_requirements().unwrap().is_none());
                assert!(serving_resource_report(&plan).unwrap().is_none());
                let memory = plan.numeric_bf16_memory_plan().unwrap().unwrap();
                let effective = plan.effective_options();
                let scheduler = effective.scheduler_config;
                let limits = plan.qwen35_moe_capacity_limits().unwrap().unwrap();
                assert_eq!(effective.driver_config.ctx_size, 128);
                assert_eq!(scheduler.max_active_sequences, 2);
                assert_eq!(scheduler.max_decode_batch, 2);
                assert_eq!(scheduler.max_batch_tokens, 8);
                assert_eq!(scheduler.prefill_chunk_size, 8);
                assert_eq!(scheduler.decode_cohort_target, 1);
                assert_eq!(scheduler.decode_cohort_max_deferrals, 0);
                assert_eq!(scheduler.prefill_reserve_tokens, 1);
                assert!(scheduler.allow_mixed_batches);
                assert_eq!(scheduler.prefix_cache_capacity_pages, 0);
                assert_eq!(
                    (
                        limits.max_positions,
                        limits.max_sequences,
                        limits.max_batch_tokens
                    ),
                    (128, 2, 8)
                );
                assert_eq!(limits.max_experts, 64);
                assert_eq!(limits.max_bytes, 320 << 20);
                assert_eq!(limits.scratch_bytes, 8 << 20);
                assert_eq!(limits.workspace_bytes, 2 << 30);
                assert_eq!(limits.kv_bytes, 16 << 20);
                assert_eq!(limits.device_bytes, 7 << 30);
                assert_eq!(limits.allocator_margin_bytes, 512 << 20);
                assert_eq!(
                    plan.selected_implementation().kind,
                    SelectedImplementationKind::Resident
                );
                assert_eq!(plan.expert_read_pipeline_options(), Some(requested));
                assert_eq!(
                    memory.runner_options().expert_read_pipeline_options(),
                    Some(requested)
                );
                assert!(memory.resources().host_experts().is_none());
                assert_eq!(memory.model_budget(), base_memory.model_budget());
                assert_eq!(
                    memory.total_device_bytes(),
                    base_memory.total_device_bytes()
                );
                assert!(source.required_metadata_bytes <= requested.max_metadata_bytes);
                let payload_cap = requested.max_host_bytes
                    - requested.max_metadata_bytes
                    - requested.worker_stack_bytes as u64;
                assert!(2 * source.max_whole_expert_payload_bytes <= payload_cap);
                println!(
                    "source_admission={}",
                    serde_json::json!({
                        "schedule": schedule, "status": "PASS", "model": model,
                        "effective_cli_max_metadata_bytes": requested.max_metadata_bytes,
                        "required_composition_metadata_bytes": source.required_metadata_bytes,
                        "required_executor_metadata_bytes": source.executor_metadata_bytes,
                        "metadata_admitted": true, "host_cap_bytes": requested.max_host_bytes,
                        "worker_stack_bytes": requested.worker_stack_bytes,
                        "worker_payload_max_bytes": payload_cap / 2,
                        "double_buffer_payload_cap_bytes": payload_cap,
                        "max_whole_expert_payload_bytes": source.max_whole_expert_payload_bytes,
                        "double_buffer_required_bytes": 2 * source.max_whole_expert_payload_bytes,
                        "source_reader_batch_cap_bytes": memory.runner_options().max_parameter_bytes(),
                        "device_required_bytes": memory.total_device_bytes(),
                        "device_cap_bytes": limits.device_bytes,
                        "model_budget": format!("{:?}", memory.model_budget()),
                        "selected_plan": format!("{:?}", plan.selected_implementation().kind),
                        "backend_profile": plan.backend_profile(), "context": 128,
                        "max_active_sequences": scheduler.max_active_sequences,
                        "max_decode_batch": scheduler.max_decode_batch,
                        "max_batch_tokens": scheduler.max_batch_tokens,
                        "prefill_chunk_size": scheduler.prefill_chunk_size,
                        "decode_cohort_target": scheduler.decode_cohort_target,
                        "decode_cohort_max_deferrals": scheduler.decode_cohort_max_deferrals,
                        "prefill_reserve_tokens": scheduler.prefill_reserve_tokens,
                        "allow_mixed_batches": scheduler.allow_mixed_batches,
                        "prefix_cache_capacity_pages": scheduler.prefix_cache_capacity_pages,
                        "cache_entries": limits.max_experts, "cache_including_scratch_bytes": limits.max_bytes,
                        "scratch_bytes": limits.scratch_bytes, "workspace_ceiling_bytes": limits.workspace_bytes,
                        "kv_ceiling_bytes": limits.kv_bytes, "allocator_margin_bytes": limits.allocator_margin_bytes,
                        "serving_resource_requirements": null,
                        "owner_constructed": false, "numeric_payload_read": false,
                        "gpu_hidden": true, "cuda_context_created": false,
                    })
                );
                for line in execution_report(&plan, args.rank_timeout_ms) {
                    println!("{line}");
                }
            }
            Err(error) => {
                println!(
                    "source_admission={}",
                    serde_json::json!({
                        "schedule": schedule, "status": "FAIL", "error": error.to_string(),
                        "effective_cli_max_metadata_bytes": requested.max_metadata_bytes,
                        "required_composition_metadata_bytes": source.required_metadata_bytes,
                        "required_executor_metadata_bytes": source.executor_metadata_bytes,
                        "owner_constructed": false, "numeric_payload_read": false, "gpu_hidden": true,
                    })
                );
                failures.push(format!("{schedule}: {error}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "V3 source admission blocks launch: {failures:?}"
    );
}

#[test]
fn expert_read_v3_study_cohort_is_explicit_without_changing_cli_defaults() {
    let defaults = serve(&["ferrule", "serve", "not-opened"]);
    let scheduler = model_factory_options(&defaults).unwrap().scheduler_config;
    assert_eq!(defaults.decode_cohort_target, 4);
    assert_eq!(defaults.decode_cohort_max_deferrals, 3);
    assert_eq!(scheduler.decode_cohort_target, 4);
    assert_eq!(scheduler.decode_cohort_max_deferrals, 3);
    for schedule in [None, Some("blocking"), Some("lookahead")] {
        let args = storage_middle64_args("not-opened", schedule);
        let scheduler = model_factory_options(&args).unwrap().scheduler_config;
        assert_eq!(args.decode_cohort_target, 1);
        assert_eq!(args.decode_cohort_max_deferrals, 0);
        assert_eq!(scheduler.decode_cohort_target, 1);
        assert_eq!(scheduler.decode_cohort_max_deferrals, 0);
    }
}

#[test]
fn expert_read_cli_resource_report_rejects_stale_prepared_sources_before_owner() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::expert_read_cli_resource_report_rejects_stale_prepared_sources_before_owner"
        ),
        || {
            for schedule in ["blocking", "lookahead"] {
                for name in ["config.json", "model.safetensors"] {
                    let f = fixture::Fixture::new();
                    let model = f.dir.path().to_str().unwrap();
                    let base = prepare_model(&storage_middle64_args(model, None)).unwrap();
                    let budget =
                        source_read_budget(&base.numeric_bf16_memory_plan().unwrap().unwrap());
                    let args = fixture_storage_middle64_args(model, schedule, &budget);
                    let plan = prepare_model(&args).unwrap();
                    assert!(serving_resource_report(&plan).unwrap().is_none());
                    let path = f.dir.path().join(name);
                    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
                    file.set_len(file.metadata().unwrap().len() + 1).unwrap();
                    let error = serving_resource_report(&plan).unwrap_err();
                    assert!(
                        error.to_string().contains("stale"),
                        "{schedule} {name}: {error}"
                    );
                }
            }
        },
    );
}

#[test]
fn expert_read_cli_lowering_freezes_both_schedules_on_existing_storage_middle64_owner() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::expert_read_cli_lowering_freezes_both_schedules_on_existing_storage_middle64_owner"
        ),
        || {
            let f = fixture::Fixture::new();
            let model = f.dir.path().to_str().unwrap();
            let base_args = storage_middle64_args(model, None);
            let base = prepare_model(&base_args).unwrap();
            let base_memory = base.numeric_bf16_memory_plan().unwrap().unwrap();
            let source_budget = source_read_budget(&base_memory);
            assert!(base.mixed_tp2_cache_requirements().unwrap().is_none());
            assert!(base.serving_resource_requirements().unwrap().is_none());
            assert!(serving_resource_report(&base).unwrap().is_none());
            assert_eq!(base.expert_read_pipeline_options(), None);
            assert_eq!(
                base_memory.runner_options().expert_read_pipeline_options(),
                None
            );
            assert!(base_memory.resources().host_experts().is_none());
            assert!(
                execution_report(&base, base_args.rank_timeout_ms)
                    .iter()
                    .all(|line| !line.contains("expert read host staging"))
            );
            let base_limits = base.qwen35_moe_capacity_limits().unwrap().unwrap();
            assert_eq!(
                (
                    base_limits.max_experts,
                    base_limits.max_bytes,
                    base_limits.scratch_bytes
                ),
                (64, 320 << 20, 8 << 20)
            );
            assert_eq!(
                (
                    base_limits.device_bytes,
                    base_limits.workspace_bytes,
                    base_limits.kv_bytes
                ),
                (7 << 30, 2 << 30, 16 << 20)
            );
            assert_eq!(base_limits.allocator_margin_bytes, 512 << 20);
            for (schedule, expected) in [
                ("blocking", ExpertReadSchedule::Blocking),
                ("lookahead", ExpertReadSchedule::Lookahead),
            ] {
                let args = fixture_storage_middle64_args(model, schedule, &source_budget);
                let read = ExpertReadPipelineOptions {
                    schedule: expected,
                    max_host_bytes: 64 << 20,
                    max_metadata_bytes: source_budget.required_metadata_bytes.div_ceil(1 << 20)
                        << 20,
                    worker_stack_bytes: 2 << 20,
                };
                assert_eq!(
                    serving_options(&args).unwrap().expert_read_pipeline,
                    Some(read)
                );
                assert!(
                    model_factory_options(&args)
                        .unwrap()
                        .qwen35_host_cache
                        .is_none()
                );
                let plan = prepare_model(&args).unwrap();
                let memory = plan.numeric_bf16_memory_plan().unwrap().unwrap();
                assert!(plan.mixed_tp2_cache_requirements().unwrap().is_none());
                assert!(plan.serving_resource_requirements().unwrap().is_none());
                assert!(serving_resource_report(&plan).unwrap().is_none());
                assert_eq!(
                    plan.selected_implementation().kind,
                    SelectedImplementationKind::Resident
                );
                assert_eq!(plan.backend_profile(), base.backend_profile());
                assert_eq!(plan.expert_read_pipeline_options(), Some(read));
                assert_eq!(memory.expert_read_pipeline_options(), Some(read));
                assert_eq!(
                    memory.runner_options().expert_read_pipeline_options(),
                    Some(read)
                );
                assert!(memory.resources().host_experts().is_none());
                assert_eq!(memory.model_budget(), base_memory.model_budget());
                assert_eq!(
                    memory.total_device_bytes(),
                    base_memory.total_device_bytes()
                );
                assert_eq!(
                    plan.qwen35_moe_capacity_limits().unwrap().unwrap(),
                    base_limits
                );
                assert_eq!(
                    plan.effective_options().scheduler_config,
                    base.effective_options().scheduler_config
                );
                let scheduler = plan.effective_options().scheduler_config;
                assert_eq!(scheduler.max_decode_batch, 2);
                assert_eq!(scheduler.decode_cohort_target, 1);
                assert_eq!(scheduler.decode_cohort_max_deferrals, 0);
                assert!(
                    read.max_host_bytes - read.max_metadata_bytes - read.worker_stack_bytes as u64
                        >= 2 * source_budget.max_whole_expert_payload_bytes
                );
                assert_eq!(
                    base_limits.device_bytes as u64 + read.max_host_bytes,
                    7_583_301_632
                );
                let report = execution_report(&plan, args.rank_timeout_ms);
                let expected_report = format!(
                    "expert read host staging: schedule={expected:?} host_cap_bytes=67108864 metadata_bytes={} worker_stack_bytes=2097152; metadata and requested stack included in host cap; separate from CUDA cache/device budgets; OS/TLS/allocator headroom not included; no GPU-overlap claim",
                    read.max_metadata_bytes,
                );
                assert_eq!(
                    report
                        .iter()
                        .filter(|line| line.starts_with("expert read host staging:"))
                        .collect::<Vec<_>>(),
                    vec![&expected_report]
                );
            }
        },
    );
}
