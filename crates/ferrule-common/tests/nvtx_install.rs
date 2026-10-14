//! Subprocesses isolate global tracing initialization and environment choices.
use std::process::Command;

use ferrule_common::observability::{TracingInitStatus, init_tracing_once};

#[test]
fn runtime_and_format_filter_matrix() {
    for runtime in [None, Some("0"), Some("1"), Some("true"), Some("01")] {
        for format in ["compact", "json"] {
            for log in ["off", "info"] {
                let mut child = Command::new(std::env::current_exe().unwrap());
                child
                    .args(["--exact", "install_child", "--nocapture"])
                    .env("FERRULE_NVTX_INSTALL_CHILD", "1")
                    .env("FERRULE_LOG", log)
                    .env("FERRULE_LOG_FORMAT", format)
                    .env_remove("FERRULE_PROFILE_NVTX")
                    .env_remove("NVTX_INJECTION32_PATH")
                    .env_remove("NVTX_INJECTION64_PATH");
                if let Some(runtime) = runtime {
                    child.env("FERRULE_PROFILE_NVTX", runtime);
                }
                let output = child.output().unwrap();
                assert!(
                    output.status.success(),
                    "runtime={runtime:?} format={format} log={log}:\n{}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                let stderr = String::from_utf8(output.stderr).unwrap();
                assert_eq!(stderr.contains("ordinary info"), log == "info");
                for private in [
                    "private prompt",
                    "/private/checkpoint",
                    "credential",
                    "hidden debug",
                ] {
                    assert!(!stderr.contains(private), "{stderr}");
                }
            }
        }
    }
}

#[test]
fn install_child() {
    if std::env::var("FERRULE_NVTX_INSTALL_CHILD").as_deref() != Ok("1") {
        return;
    }
    assert_eq!(init_tracing_once(), TracingInitStatus::Installed);
    let profiling = cfg!(feature = "nvtx-profile")
        && std::env::var("FERRULE_PROFILE_NVTX").as_deref() == Ok("1");
    let span = tracing::debug_span!(target: "ferrule_profile", "read_subset",
        operation = 1u64, source_generation = 1u64, requested_bytes = 16u64,
        prompt = "private prompt", path = "/private/checkpoint", token = "credential");
    assert_eq!(!span.is_disabled(), profiling);
    span.in_scope(|| tracing::debug!("hidden debug"));
    assert!(tracing::debug_span!(target: "unrelated", "read_subset").is_disabled());
    assert!(tracing::debug_span!(target: "ferrule_profile", "unlisted").is_disabled());
    assert_eq!(
        tracing::enabled!(tracing::Level::INFO),
        std::env::var("FERRULE_LOG").as_deref() == Ok("info")
    );
    tracing::info!("ordinary info");
}
