use std::process::Command;

#[test]
fn network_help_describes_tls_one_shot_and_explicit_remote_bind() {
    let output = Command::new(env!("CARGO_BIN_EXE_ferrule"))
        .args(["rank-serve", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    for flag in [
        "--config",
        "--bind",
        "--allow-remote-bind",
        "mTLS",
        "127.0.0.1:9443",
        "TCP readiness",
    ] {
        assert!(help.contains(flag), "missing {flag}: {help}");
    }
    let output = Command::new(env!("CARGO_BIN_EXE_ferrule"))
        .args(["bench-parallel", "--help"])
        .output()
        .unwrap();
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("--remote-manifest")
    );
}

#[test]
fn remote_manifest_cannot_be_mixed_with_explicit_thread_process_or_device_placement() {
    for flags in [
        ["--rank-backend", "thread"],
        ["--rank-backend", "process"],
        ["--devices", "0,1"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_ferrule"))
            .args([
                "bench-parallel",
                "--mode",
                "tp-row",
                "--ranks",
                "2",
                "--fixture",
                "not-read.json",
                "--remote-manifest",
                "not-read-manifest.json",
            ])
            .args(flags)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("cannot be used with"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[cfg(not(all(unix, feature = "cuda")))]
#[test]
fn network_production_endpoints_never_select_cpu_or_plaintext_fallback() {
    for args in [
        vec!["rank-serve", "--config", "not-read.json"],
        vec![
            "bench-parallel",
            "--mode",
            "dp",
            "--ranks",
            "2",
            "--fixture",
            "not-read.json",
            "--remote-manifest",
            "not-read.json",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_ferrule"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("CUDA"));
        assert!(String::from_utf8_lossy(&output.stderr).contains("fallback"));
    }
}
