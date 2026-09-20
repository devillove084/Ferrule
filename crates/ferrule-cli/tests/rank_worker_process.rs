#[cfg(unix)]
#[cfg(not(feature = "cuda"))]
#[test]
fn production_process_rank_owner_reports_no_cuda_from_hidden_child() {
    use ferrule_runtime::parallel::process::{
        ProcessFrameLimits, ProcessLaunch, ProcessOwnerConfig, ProcessRankOwner,
    };
    use serde_json::json;
    use std::time::Duration;

    let limits = ProcessFrameLimits::default();
    let launch = ProcessLaunch::new(env!("CARGO_BIN_EXE_ferrule"))
        .arg("__rank-worker")
        .arg("--max-frame-bytes")
        .arg(limits.max_frame_bytes.to_string())
        .arg("--io-timeout-ms")
        .arg("1000");
    let options = ProcessOwnerConfig {
        startup_timeout: Duration::from_secs(2),
        command_timeout: Duration::from_secs(2),
        terminate_grace: Duration::from_millis(20),
        kill_grace: Duration::from_secs(1),
        ..ProcessOwnerConfig::default()
    };
    let error = ProcessRankOwner::<serde_json::Value, serde_json::Value, serde_json::Value>::spawn(
        launch,
        ferrule_runtime::parallel::process::ProcessIdentity::new(
            ferrule_runtime::parallel::process::ProcessGroupEpoch::new(1).unwrap(),
            ferrule_common::ParallelRankId::new(0),
            ferrule_runtime::parallel::process::ProcessOwnerInstanceId::new(1).unwrap(),
        ),
        &json!({}),
        options,
    )
    .unwrap_err();
    assert!(error.to_string().contains("CUDA-only"), "{error}");
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
