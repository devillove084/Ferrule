use super::*;

#[test]
fn qwen35_reference_cli_is_usable_and_explicit_moe_options_are_rejected() {
    use ferrule_model::models::qwen35::{
        Qwen35Config, Qwen35HfNameMapper, Qwen35TensorPartitionKind,
    };
    use std::io::Write;
    let fixture = PlanFixture::new(false);
    let value: serde_json::Value = serde_json::from_str(include_str!(
        "../../../ferrule-model/tests/qwen35_08b_config.json"
    ))
    .unwrap();
    std::fs::write(fixture.0.join("config.json"), value.to_string()).unwrap();
    let mapper = Qwen35HfNameMapper::new(&Qwen35Config::from_value(&value).unwrap());
    let mut header = serde_json::Map::new();
    let mut bytes = 0;
    for tensor in mapper
        .tensors()
        .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
    {
        header.insert(tensor.external_name.clone(), serde_json::json!({"dtype":tensor.dtype.as_str(), "shape":tensor.shape, "data_offsets":[bytes,bytes+tensor.bytes()]}));
        bytes += tensor.bytes();
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = std::fs::File::create(fixture.0.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.set_len(8 + header.len() as u64 + bytes).unwrap();
    for flags in [
        vec![],
        vec!["--max-batch-tokens", "512", "--prefill-chunk-size", "512"],
        vec![
            "--max-batch-tokens",
            "32",
            "--prefill-chunk-size",
            "32",
            "--ctx-size",
            "8",
        ],
        vec!["--max-batch-tokens", "8", "--prefill-chunk-size", "4"],
    ] {
        let args = fixture.reference_args(&flags);
        let plan = prepare_model(&args).unwrap();
        let requested = plan.requested_options();
        let effective = plan.effective_options();
        assert_eq!(
            requested.scheduler_config.max_batch_tokens,
            args.max_batch_tokens
        );
        assert_eq!(
            requested.scheduler_config.prefill_chunk_size,
            args.prefill_chunk_size
        );
        assert_eq!(
            effective.scheduler_config.max_batch_tokens,
            args.max_batch_tokens.min(32).min(args.ctx_size)
        );
        assert_eq!(
            effective.scheduler_config.prefill_chunk_size,
            args.prefill_chunk_size
                .min(effective.scheduler_config.max_batch_tokens)
        );
        assert_eq!(plan.observability().ctx_size, args.ctx_size);
        if args.max_batch_tokens != effective.scheduler_config.max_batch_tokens {
            let report = plan.adjustment_report();
            let change = report
                .adjustments()
                .iter()
                .find(|a| a.field == "scheduler_config.max_batch_tokens")
                .unwrap();
            assert_eq!(change.requested, args.max_batch_tokens.to_string());
            assert_eq!(
                change.effective,
                effective.scheduler_config.max_batch_tokens.to_string()
            );
            assert!(change.reason.contains("min(32, context)"));
        }
        assert!(
            plan.resolution_report()
                .contains("owner live admission required")
        );
        eprintln!("{}", plan.resolution_report());
    }
    let args = fixture.reference_args(&[]);
    let plan = prepare_model(&args).unwrap();
    assert_eq!(plan.model_name(), "qwen3.5-0.8b");
    assert!(
        prepare_model(&fixture.reference_args(&["--cuda-expert-device-entries", "1024"])).is_err()
    );
    assert_eq!(plan.chat_template(), ferrule_model::ChatTemplate::Qwen35);
    for (flag, values) in [
        ("--expert-host-cache-entries", ["64", "0"]),
        ("--expert-host-cache-mb", ["1024", "0"]),
        ("--expert-pinned-cache-entries", ["16", "0"]),
        ("--expert-pinned-cache-mb", ["256", "0"]),
        ("--moe-hotset-experts", ["1", "0"]),
    ] {
        for value in values {
            assert!(
                prepare_model(&fixture.reference_args(&[flag, value]))
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("explicit MoE")
            );
        }
    }
    for args in [["--max-layers", "1"], ["--tensor-parallel", "2"]] {
        assert!(prepare_model(&fixture.reference_args(&args)).is_err());
    }
    if cfg!(feature = "cuda") {
        assert!(prepare_model(&fixture.reference_args(&["--backend", "cuda"])).is_ok());
    }
}

#[test]
fn parsed_omitted_cache_options_prepare_pipeline_plans() {
    for moe in [false, true] {
        let fixture = PlanFixture::new(moe);
        for backend in ["cpu", "cuda"] {
            if backend == "cuda" && !cfg!(feature = "cuda") {
                continue;
            }
            for rank in ["thread", "process"] {
                if rank == "process" && !cfg!(unix) {
                    continue;
                }
                for engine in ["auto", "pipeline"] {
                    let mut flags = vec![
                        "--engine",
                        engine,
                        "--backend",
                        backend,
                        "--pipeline-parallel",
                        "2",
                        "--rank-backend",
                        rank,
                    ];
                    if moe {
                        flags.extend(["--expert-parallel", "2"]);
                    }
                    let args = fixture.reference_args(&flags);
                    assert_eq!(args.expert_host_cache_entries, None);
                    assert_eq!(args.expert_host_cache_mb, None);
                    assert_eq!(args.expert_pinned_cache_entries, None);
                    assert_eq!(args.expert_pinned_cache_mb, None);
                    assert_eq!(
                        prepare_model(&args)
                            .unwrap()
                            .requested_options()
                            .expert_cache,
                        ExpertCacheOptions::default()
                    );
                    let plan = prepare_model(&args).unwrap();
                    let expected_backend = if backend == "cpu" {
                        ferrule_model::ModelExecutionBackend::Cpu
                    } else {
                        ferrule_model::ModelExecutionBackend::Cuda
                    };
                    assert_eq!(plan.backend(), expected_backend);
                    assert_eq!(
                        plan.requested_backend(),
                        BackendSelection::parse(backend).unwrap()
                    );
                    assert_eq!(
                        plan.requested_options().scheduler_config.max_decode_batch,
                        args.max_active_sequences
                    );
                    let effective = plan.effective_options();
                    assert_eq!(effective.scheduler_config.max_decode_batch, 1);
                    assert_eq!(effective.scheduler_config.decode_cohort_max_deferrals, 0);
                    assert!(!effective.driver_config.enable_native_proposals);
                    assert!(
                        plan.adjustment_report()
                            .render()
                            .contains("generic pipeline is serial")
                    );
                    assert_eq!(plan.observability().max_layers, 2);
                    assert_eq!(
                        plan.pipeline_options()
                            .unwrap()
                            .parallelism
                            .pipeline_parallel,
                        2
                    );
                }
            }
        }
        // Explicit pipeline at PP1 must also avoid resident CLI defaults.
        assert!(
            prepare_model(&fixture.reference_args(&["--engine", "pipeline"]))
                .unwrap()
                .pipeline_options()
                .is_some()
        );
    }
}

#[test]
fn parsed_explicit_pipeline_cache_constraints_are_never_discarded() {
    let fixture = PlanFixture::new(false);
    for engine in ["auto", "pipeline"] {
        for (flag, values) in [
            ("--expert-host-cache-entries", ["64", "256", "0"]),
            ("--expert-host-cache-mb", ["1024", "0", "1"]),
            ("--expert-pinned-cache-entries", ["16", "64", "0"]),
            ("--expert-pinned-cache-mb", ["256", "0", "1"]),
        ] {
            for value in values {
                let args = fixture.reference_args(&[
                    "--engine",
                    engine,
                    "--pipeline-parallel",
                    "2",
                    flag,
                    value,
                ]);
                let error = prepare_model(&args)
                    .err()
                    .expect("explicit policy must be rejected");
                assert!(
                    error.to_string().contains("expert_cache overrides"),
                    "{error:#}"
                );
                assert!(error.to_string().contains(flag), "{error:#}");
            }
        }
    }
}

#[test]
fn parsed_resident_cache_defaults_and_explicit_limits_are_preserved() {
    let fixture = PlanFixture::new(false);
    for engine in ["auto", "resident"] {
        let args = fixture.reference_args(&["--engine", engine]);
        assert!(prepare_model(&args).unwrap().pipeline_options().is_none());
        assert_eq!(
            prepare_model(&args)
                .unwrap()
                .requested_options()
                .expert_cache,
            ExpertCacheOptions {
                host_entries: 64,
                host_mebibytes: 1024,
                pinned_entries: 16,
                pinned_mebibytes: 256,
            }
        );
        let args = fixture.reference_args(&["--engine", engine, "--expert-host-cache-mb", "7"]);
        assert!(prepare_model(&args).unwrap().pipeline_options().is_none());
        assert_eq!(
            prepare_model(&args)
                .unwrap()
                .requested_options()
                .expert_cache,
            ExpertCacheOptions {
                host_entries: 64,
                host_mebibytes: 7,
                pinned_entries: 16,
                pinned_mebibytes: 256,
            }
        );
        let args = fixture.reference_args(&[
            "--engine",
            engine,
            "--expert-host-cache-entries",
            "0",
            "--expert-host-cache-mb",
            "0",
            "--expert-pinned-cache-entries",
            "3",
            "--expert-pinned-cache-mb",
            "2",
        ]);
        assert!(prepare_model(&args).unwrap().pipeline_options().is_none());
        assert_eq!(
            prepare_model(&args)
                .unwrap()
                .requested_options()
                .expert_cache,
            ExpertCacheOptions {
                host_entries: 0,
                host_mebibytes: 0,
                pinned_entries: 3,
                pinned_mebibytes: 2,
            }
        );
    }
}

#[test]
fn parsed_pipeline_plan_still_rejects_unsupported_runtime_options() {
    let fixture = PlanFixture::new(false);
    for (flag, value, message) in [
        ("--max-layers", "1", "requires all checkpoint layers"),
        (
            "--moe-hotset-experts",
            "1",
            "does not implement moe_hotset_experts",
        ),
        (
            "--rank-restarts",
            "1",
            "does not support rank restart/replay",
        ),
    ] {
        let args = fixture.reference_args(&["--engine", "pipeline", flag, value]);
        let error = prepare_model(&args)
            .err()
            .expect("unsupported option must be rejected");
        assert!(error.to_string().contains(message), "{error:#}");
    }
}
