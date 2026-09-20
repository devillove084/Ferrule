use std::process::Command;

#[test]
fn bench_parallel_help_describes_the_cuda_fixture_interface() {
    let output = Command::new(env!("CARGO_BIN_EXE_ferrule"))
        .args(["bench-parallel", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "CUDA",
        "--fixture",
        "--mode",
        "dp",
        "tp-column",
        "tp-row",
        "--ranks",
        "--rows",
        "--in-features",
        "--out-features",
        "--warmup",
        "--iterations",
        "--devices",
        "--json",
        "--emit-values",
    ] {
        assert!(help.contains(expected), "missing {expected}: {help}");
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn default_build_rejects_parallel_bench_on_stderr_without_stdout_or_fallback() {
    for mode in ["dp", "tp-column", "tp-row"] {
        let output = Command::new(env!("CARGO_BIN_EXE_ferrule"))
            .args([
                "bench-parallel",
                "--mode",
                mode,
                "--ranks",
                "2",
                "--fixture",
                "fixture-is-not-read-without-cuda.json",
                "--json",
                "--emit-values",
            ])
            .env("FERRULE_LOG", "trace")
            .env("FERRULE_LOG_FORMAT", "json")
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            output.stdout.is_empty(),
            "unexpected stdout: {:?}",
            output.stdout
        );
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("CUDA-only"), "{error}");
        assert!(error.contains("--features cuda"), "{error}");
        assert!(error.contains("no CPU fallback"), "{error}");
    }
}
