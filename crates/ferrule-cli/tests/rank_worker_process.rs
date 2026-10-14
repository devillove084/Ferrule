#[cfg(all(unix, not(feature = "cuda")))]
mod linear_boot {
    use ferrule_runtime::parallel::process::*;
    use serde_json::{Value, json};
    use std::time::Duration;

    pub(super) fn fixture() -> (Value, ProcessFrameLimits) {
        let config = json!({
            "spec": {
                "mode": "dp", "ranks": 1, "rank": 0, "device": 0,
                "rows": 1, "in_features": 1, "out_features": 1
            },
            "weight": [1.0]
        });
        // Checked wire budgets for this one-element linear fixture.
        let payload = 32 + 4096;
        let limits = ProcessFrameLimits {
            max_frame_bytes: payload + 4096,
            max_config_bytes: payload,
            max_command_bytes: payload,
            max_output_bytes: payload,
            max_error_bytes: 4096,
        };
        (config, limits)
    }

    pub(super) fn reject(
        config: &Value,
        limits: ProcessFrameLimits,
        rank: u32,
    ) -> ProcessHandlerError {
        let launch = ProcessLaunch::new(env!("CARGO_BIN_EXE_ferrule"))
            .arg("__rank-worker")
            .arg("--max-frame-bytes")
            .arg(limits.max_frame_bytes.to_string())
            .arg("--io-timeout-ms")
            .arg("1000");
        let options = ProcessOwnerConfig {
            frame_limits: limits,
            startup_timeout: Duration::from_secs(2),
            command_timeout: Duration::from_secs(2),
            terminate_grace: Duration::from_millis(20),
            kill_grace: Duration::from_secs(1),
        };
        let error = ProcessRankOwner::<Value, Value, Value>::spawn(
            launch,
            ProcessIdentity::new(
                ProcessGroupEpoch::new(1).unwrap(),
                ferrule_common::ParallelRankId::new(rank),
                ProcessOwnerInstanceId::new(u64::from(rank) + 1).unwrap(),
            ),
            config,
            options,
        )
        .unwrap_err();
        let source = match &error.source {
            ProcessError::QuiescenceUnknown { source, .. } => source.as_ref(),
            source => source,
        };
        let ProcessError::RemoteFailure { failure } = source else {
            panic!("expected linear startup failure: {error}");
        };
        assert_eq!(failure.kind, ProcessFailureKind::Startup);
        assert_eq!(failure.quiescence, ProcessQuiescence::Fenced);
        failure.clone()
    }
}

#[cfg(all(unix, not(feature = "cuda")))]
#[test]
fn production_process_rank_owner_reports_no_cuda_from_hidden_child() {
    let (config, limits) = linear_boot::fixture();
    let failure = linear_boot::reject(&config, limits, 0);
    assert!(failure.message.contains("CUDA-only"), "{failure:?}");
    assert!(failure.message.contains("no CPU fallback"), "{failure:?}");
}

#[cfg(all(unix, not(feature = "cuda")))]
#[test]
fn linear_boot_validation_precedes_no_cuda_refusal() {
    use ferrule_runtime::parallel::process::ProcessFrameLimits;
    use serde_json::json;

    for case in 0..6 {
        let (mut config, mut limits) = linear_boot::fixture();
        let mut rank = 0;
        let expected = match case {
            0 => {
                config = json!({});
                "missing field `spec`"
            }
            1 => {
                config["unknown"] = true.into();
                "unknown field"
            }
            2 => {
                config["spec"]["ranks"] = 0.into();
                "invalid worker rank geometry"
            }
            3 => {
                config["weight"] = json!([]);
                "Boot weight length mismatch"
            }
            4 => {
                rank = 1;
                "Boot identity mismatch"
            }
            _ => {
                limits = ProcessFrameLimits::default();
                "Boot limits do not match checked geometry"
            }
        };
        let failure = linear_boot::reject(&config, limits, rank);
        assert!(failure.message.contains(expected), "{failure:?}");
        assert!(!failure.message.contains("CUDA-only"), "{failure:?}");
    }
}

#[cfg(unix)]
#[cfg(not(feature = "cuda"))]
#[test]
fn hidden_rank_worker_help_is_available_without_initializing_cuda() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ferrule"))
        .args(["__rank-worker", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("max-frame-bytes")
    );
    assert!(output.stderr.is_empty());
}

#[cfg(all(unix, not(feature = "cuda")))]
#[test]
fn numeric_v2_boot_reaches_runtime_endpoint_and_never_falls_back_to_linear_or_cpu() {
    use ferrule_common::{ParallelRankId, topology::ExpertSourceScope};
    use ferrule_runtime::parallel::expert_process::{
        NUMERIC_EXPERT_WIRE_VERSION, NumericBoot, NumericSource,
    };
    use ferrule_runtime::parallel::process::decoder::{DecoderDevice, ExpertPlacementFrame};
    use ferrule_runtime::parallel::process::*;
    use serde_json::Value;
    use std::time::Duration;

    let numeric = NumericBoot {
        numeric_expert_boot: NUMERIC_EXPERT_WIRE_VERSION,
        numeric_profile: ferrule_model::transformer::NumericFp8Precision::expert_wire_default(),
        checkpoint: "not-read-on-cpu".into(),
        total_layers: 2,
        layers: 1..2,
        placement: ExpertPlacementFrame {
            source_scope: ExpertSourceScope::ExternalStage,
            source: 3,
            members: vec![7],
            entries: vec![(1, 0, 7)],
            max_tokens: 4,
            max_bytes: 4096,
            devices: vec![DecoderDevice::Cuda { ordinal: 0 }],
            timeout_ms: 1000,
        },
        owner: 7,
        hidden: 4,
        max_rows: 1,
        max_parameter_bytes: 4096,
        host_max_bytes: 4096,
        host_max_experts: 1,
        host_workers: 1,
        host_headroom_bytes: 0,
        device_bytes: 4096,
        workspace_bytes: 1024,
        expected: vec![NumericSource {
            layer: 1,
            expert: 0,
            identity: [1; 32],
            content_hash: [2; 32],
            encoding: 1,
            generation: 1,
            bytes: 16,
        }],
    };
    numeric.group().unwrap();
    for case in 0..3 {
        let mut boot = serde_json::to_value(&numeric).unwrap();
        let expected = match case {
            0 => "numeric expert requires CUDA; no CPU fallback",
            1 => {
                boot["numeric_expert_boot"] = 1.into();
                "invalid numeric v2 boot"
            }
            _ => {
                boot["unknown"] = true.into();
                "unknown field"
            }
        };
        let limits = ProcessFrameLimits::default();
        let launch = ProcessLaunch::new(env!("CARGO_BIN_EXE_ferrule"))
            .arg("__rank-worker")
            .arg("--max-frame-bytes")
            .arg(limits.max_frame_bytes.to_string())
            .arg("--io-timeout-ms")
            .arg("1000");
        let options = ProcessOwnerConfig {
            startup_timeout: Duration::from_secs(3),
            command_timeout: Duration::from_secs(2),
            terminate_grace: Duration::from_millis(20),
            kill_grace: Duration::from_secs(1),
            ..Default::default()
        };
        let error = ProcessRankOwner::<Value, Value, Value>::spawn(
            launch,
            ProcessIdentity::new(
                ProcessGroupEpoch::new(1).unwrap(),
                ParallelRankId::new(7),
                ProcessOwnerInstanceId::new(1).unwrap(),
            ),
            &boot,
            options,
        )
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert!(
            !error.to_string().contains("__rank-worker is CUDA-only"),
            "numeric Boot must not be retried as linear"
        );
    }
}

#[cfg(unix)]
#[test]
fn dense_hybrid_worker_accepts_fitted_ceiling_before_strict_checkpoint_binding() {
    use ferrule_runtime::parallel::process::decoder::*;
    use ferrule_runtime::parallel::process::*;
    use std::time::Duration;
    let dir = tempfile::tempdir().unwrap();
    let boot = DecoderBoot {
        version: DECODER_WIRE_VERSION,
        rank: 0,
        checkpoint: dir.path().to_owned(),
        recipe: DecoderRecipeKind::Qwen35DenseHybrid,
        segment: SegmentFrame {
            total_layers: 2,
            layers: 0..2,
            embedding: true,
            output: true,
        },
        precision: DecoderPrecision::F32,
        device: DecoderDevice::Cpu,
        kv: KvConfigFrame {
            page_size: 16,
            max_pages: 4,
            physical_pages: 8,
            max_positions: 32,
            max_batch_tokens: 8,
            session_capacity: 1,
            max_parameter_bytes: 1 << 30,
            max_ack_polls: 8,
        },
        experts: None,
        hybrid: Some(HybridBootFrame {
            hidden: 1024,
            vocabulary: 248320,
            cuda_budget: None,
            layers: vec![
                HybridLayerFrame::GatedDeltaNet {
                    key_heads: 1,
                    value_heads: 2,
                    key_dim: 3,
                    value_dim: 2,
                    kernel: 3,
                },
                HybridLayerFrame::FullAttention {
                    kv_heads: 2,
                    head_dim: 256,
                },
            ],
        }),
    };
    let limits = boot
        .fit_frame_limits(ProcessFrameLimits::default())
        .unwrap();
    assert!(limits.max_frame_bytes > ProcessFrameLimits::default().max_frame_bytes);
    let launch = ProcessLaunch::new(env!("CARGO_BIN_EXE_ferrule"))
        .arg("__rank-worker")
        .arg("--max-frame-bytes")
        .arg(ProcessFrameLimits::MAX_FRAME_BYTES.to_string())
        .arg("--io-timeout-ms")
        .arg("30000");
    let options = ProcessOwnerConfig {
        frame_limits: limits,
        startup_timeout: Duration::from_secs(5),
        ..Default::default()
    };
    let identity = ProcessIdentity::new(
        ProcessGroupEpoch::new(1).unwrap(),
        ferrule_common::ParallelRankId::new(0),
        ProcessOwnerInstanceId::new(1).unwrap(),
    );
    let error = ProcessRankOwner::<DecoderBoot, DecoderCommand, DecoderReply>::spawn(
        launch, identity, &boot, options,
    )
    .unwrap_err();
    // The absent config must fail in the strict decoder binder, not negotiation
    // or the unrelated linear/CUDA endpoint. No checkpoint payload is read.
    let source = match &error.source {
        ProcessError::QuiescenceUnknown { source, .. } => source.as_ref(),
        source => source,
    };
    let ProcessError::RemoteFailure { failure } = source else {
        panic!("expected decoder startup failure: {error}");
    };
    assert_eq!(failure.kind, ProcessFailureKind::Startup);
    assert_eq!(failure.quiescence, ProcessQuiescence::Fenced);
    assert!(
        failure.message.contains("No such file"),
        "{}",
        failure.message
    );
}
