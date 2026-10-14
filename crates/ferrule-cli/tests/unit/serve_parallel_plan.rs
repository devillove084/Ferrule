use super::*;

#[cfg(feature = "cpu-reference")]
#[test]
fn dense_hybrid_pp2_routes_to_real_pipeline_planner_without_ready_claim() {
    use ferrule_composition::model_factory::{ArithmeticPrecision, SelectedImplementationKind};
    let fixture = PlanFixture::new(false);
    std::fs::write(
        fixture.0.join("config.json"),
        include_str!("../../../ferrule-model/tests/qwen35_08b_config.json"),
    )
    .unwrap();
    // Metadata-only sparse checkpoint. It is removed before build so this test
    // cannot accidentally allocate/run a full 0.8B model or claim readiness.
    use ferrule_model::models::qwen35::{
        Qwen35Config, Qwen35HfNameMapper, Qwen35TensorPartitionKind,
    };
    use std::io::Write;
    let value = serde_json::from_str(include_str!(
        "../../../ferrule-model/tests/qwen35_08b_config.json"
    ))
    .unwrap();
    let mapper = Qwen35HfNameMapper::new(&Qwen35Config::from_value(&value).unwrap());
    let mut header = serde_json::Map::new();
    let mut size = 0;
    for tensor in mapper
        .tensors()
        .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
    {
        header.insert(tensor.external_name.clone(), serde_json::json!({"dtype":tensor.dtype.as_str(),"shape":tensor.shape,"data_offsets":[size,size+tensor.bytes()]}));
        size += tensor.bytes();
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let checkpoint = fixture.0.join("model.safetensors");
    let hidden = fixture.0.join("hidden.safetensors");
    let mut file = std::fs::File::create(&checkpoint).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.set_len(8 + header.len() as u64 + size).unwrap();
    drop(file);
    for (engine, rank) in ["auto", "pipeline"]
        .into_iter()
        .flat_map(|e| ["thread", "process"].map(|r| (e, r)))
    {
        if rank == "process" && !cfg!(unix) {
            continue;
        }
        let args = fixture.args(&[
            "--engine",
            engine,
            "--pipeline-parallel",
            "2",
            "--backend",
            "cpu",
            "--rank-backend",
            rank,
            "--ctx-size",
            "128",
            "--prefill-chunk-size",
            "8",
            "--max-batch-tokens",
            "8",
            "--kv-cache-mb",
            "32",
        ]);
        let plan = prepare_model(&args).unwrap();
        assert_eq!(
            plan.selected_implementation().kind,
            SelectedImplementationKind::Pipeline
        );
        resolution_tests::assert_pipeline_topology(
            &plan,
            2,
            1,
            1,
            if rank == "process" {
                PipelineRankBackend::Process
            } else {
                PipelineRankBackend::Thread
            },
            None,
        );
        let report = execution_report(&plan, 1234).join("\n");
        assert!(report.contains("pipeline PP=2 TP=1 EP=1 cpu"), "{report}");
        assert!(report.contains("precision=F32"), "{report}");
        assert!(!report.contains("numeric hybrid"), "{report}");
        assert!(!report.contains("CUDA ordinals"), "{report}");
        assert!(report.contains("serial greedy decode, no mixed batches/cohort deferral, cancellation at chunk/token boundaries; no restart/replay"), "{report}");
        #[cfg(unix)]
        if rank == "process" {
            let frame_ms = ferrule_runtime::parallel::process::ProcessChildConfig::default()
                .io_timeout
                .as_millis()
                .max(1234);
            assert!(report.contains(&format!("process startup/command/shutdown absolute deadline=1234ms; child frame I/O absolute deadline={frame_ms}ms once readable; healthy idle has no first-byte deadline (no restart/replay)")), "{report}");
        } else {
            assert!(!report.contains("deadline"), "{report}");
        }
        assert_eq!(plan.support().arithmetic, ArithmeticPrecision::F32);
        assert!(plan.qwen35_expert_placement().is_none());
        assert_eq!(
            plan.backend_profile().contains("process"),
            rank == "process"
        );
        assert_eq!(
            plan.pipeline_options().unwrap().rank_backend,
            if rank == "process" {
                PipelineRankBackend::Process
            } else {
                PipelineRankBackend::Thread
            }
        );
        #[cfg(unix)]
        if rank == "process" {
            let launch = plan
                .pipeline_options()
                .unwrap()
                .process_launch
                .as_ref()
                .unwrap();
            assert!(launch.args.iter().any(|a| a == "134217728"));
        }
        assert_eq!(
            plan.pipeline_options()
                .unwrap()
                .parallelism
                .pipeline_parallel,
            2
        );
        assert_eq!(plan.observability().ctx_size, 128);
        assert_eq!(
            plan.effective_options().scheduler_config.max_decode_batch,
            1
        );
        assert!(
            !plan
                .effective_options()
                .scheduler_config
                .allow_mixed_batches
        );
        std::fs::rename(&checkpoint, &hidden).unwrap();
        assert!(
            plan.build().is_err(),
            "missing checkpoint must not yield a ready owner"
        );
        std::fs::rename(&hidden, &checkpoint).unwrap();
    }
    #[cfg(unix)]
    {
        std::fs::write(
            fixture.0.join("tokenizer.json"),
            serde_json::json!({
                "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
                "normalizer":null, "pre_tokenizer":{"type":"WhitespaceSplit"},
                "post_processor":null, "decoder":null,
                "model":{"type":"WordLevel", "vocab":{"t0":0}, "unk_token":"t0"}
            })
            .to_string(),
        )
        .unwrap();
        let args = fixture.args(&[
            "--engine",
            "pipeline",
            "--pipeline-parallel",
            "2",
            "--rank-backend",
            "process",
            "--backend",
            "cpu",
            "--ctx-size",
            "128",
            "--max-batch-tokens",
            "32",
            "--prefill-chunk-size",
            "32",
            "--kv-cache-mb",
            "32",
        ]);
        let plan = prepare_model(&args).unwrap();
        let error = plan
            .build()
            .err()
            .expect("full vocabulary must fail preflight before any child launch");
        assert!(error.to_string().contains("128 MiB"), "{error}");
    }
    for extra in [
        vec!["--rank-backend", "process", "--rank-timeout-ms", "0"],
        vec!["--expert-parallel", "2"],
        vec!["--tensor-parallel", "2"],
        vec!["--rank-restarts", "1"],
        vec!["--engine", "resident"],
    ] {
        let mut flags = vec!["--pipeline-parallel", "2", "--backend", "cpu"];
        flags.extend(extra);
        assert!(prepare_model(&fixture.args(&flags)).is_err());
    }
}

#[test]
fn numeric_process_ep_cli_preserves_launch_timeout_and_rejects_unfinished_compositions() {
    let fixture = PlanFixture::new(false);
    std::fs::write(
        fixture.0.join("config.json"),
        include_str!("../../../ferrule-model/tests/qwen35_35b_fp8_config.json"),
    )
    .unwrap();
    for degree in [2, 4, 8] {
        let devices = (0..degree)
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let args = fixture.args(&[
            "--expert-parallel",
            &degree.to_string(),
            "--devices",
            &devices,
            "--rank-backend",
            "process",
            "--rank-timeout-ms",
            "1234",
        ]);
        let options = pipeline_options(&args).unwrap();
        assert_eq!(options.rank_backend, PipelineRankBackend::Process);
        assert_eq!(options.rank_timeout, Duration::from_millis(1234));
        #[cfg(unix)]
        {
            let launch = options.process_launch.unwrap();
            assert_eq!(launch.executable, std::env::current_exe().unwrap());
            assert!(launch.args.iter().any(|v| v == "__rank-worker"));
            assert!(launch.args.iter().any(|v| v == "--max-frame-bytes"));
        }
        // CPU builds reject CUDA explicitly; never turn missing support into Ready.
        if !cfg!(feature = "cuda") {
            assert!(prepare_model(&args).is_err());
        }
    }
    for extra in [
        vec!["--pipeline-parallel", "2"],
        vec!["--tensor-parallel", "2"],
        vec!["--rank-restarts", "1"],
        vec!["--rank-timeout-ms", "0"],
        vec!["--expert-prewarm", "lazy"],
    ] {
        let mut flags = vec![
            "--expert-parallel",
            "2",
            "--devices",
            "0,1",
            "--rank-backend",
            "process",
        ];
        flags.extend(extra);
        assert!(prepare_model(&fixture.args(&flags)).is_err());
    }
}
