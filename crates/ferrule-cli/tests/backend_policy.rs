use std::process::Command;

#[test]
fn cpu_policy_is_enforced_before_opening_a_model() {
    let missing = tempfile::tempdir().unwrap().path().join("no-model");
    for command in ["chat", "serve", "bench-interactive"] {
        for backend in ["cpu", "cpu-reference", "reference"] {
            let output = Command::new(env!("CARGO_BIN_EXE_ferrule"))
                .arg(command)
                .arg(&missing)
                .args(["--backend", backend])
                .output()
                .unwrap();
            assert!(!output.status.success());
            let error = String::from_utf8_lossy(&output.stderr);
            if cfg!(feature = "cpu-reference") {
                assert!(!error.contains("CPU execution is test-only"), "{error}");
                assert!(error.contains("no-model"), "{error}");
            } else {
                assert!(error.contains("CPU execution is test-only"), "{error}");
                assert!(error.contains("--features cpu-reference"), "{error}");
            }
        }
    }
}
