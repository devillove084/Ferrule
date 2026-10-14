//! Text-only architectural guard: requires rg, not an AST/dependency proof.
#[test]
fn parallel_has_no_concrete_family_import_or_factory_backedge() {
    let output = std::process::Command::new("rg")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "--line-number", "--glob", "*.rs",
            r"models\s*::|engine\s*::\s*model_factory|ModelFamily\s*::\s*(Qwen\w*|DeepSeek\w*)|\b(Qwen|DeepSeek)\w*(Adapter|Recipe|Profile|Config)\b",
            "src/parallel",
        ])
        .output().expect("architecture guard requires ripgrep on PATH");
    assert_eq!(
        output.status.code(),
        Some(1),
        "parallel must not bind concrete families (wire recipe tags may remain):\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn runtime_helpers_have_single_backend_neutral_paths() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for path in [
        "src/cache/kv_read_authority.rs",
        "src/io/host_kv_adapters.rs",
        "src/engine/driver/kv_execution.rs",
        "src/engine/resident_composition.rs",
        "src/engine/runner_executor.rs",
        "src/engine/runner_executor/kv_dispatch.rs",
        "src/parallel/pipeline/completion_boundary.rs",
        "src/parallel/pipeline/dense_bf16_memory.rs",
        "src/parallel/pipeline/kv_capacity.rs",
        "tests/kv_execution_policy.rs",
        "tests/unit/cache/kv_read_authority.rs",
        "tests/unit/cache/kv_read_authority/append_contract.rs",
        "tests/unit/engine/driver_kv_execution.rs",
        "tests/unit/io/host_kv_adapters.rs",
    ] {
        assert!(root.join(path).is_file(), "missing canonical path: {path}");
    }
    for path in [
        "src/cache/host_kv.rs",
        "src/io/host_kv.rs",
        "src/engine/composition.rs",
        "src/engine/driver/host_kv.rs",
        "src/engine/runner_executor/host_kv.rs",
        "src/parallel/observability.rs",
        "src/parallel/pipeline/bf16.rs",
        "src/parallel/pipeline/capacity.rs",
        "tests/host_kv_policy.rs",
        "tests/unit/cache/host_kv.rs",
        "tests/unit/cache/host_kv",
        "tests/unit/engine/driver_host_kv.rs",
        "tests/unit/io/host_kv.rs",
        "src/engine/native_executor.rs",
        "src/engine/native_executor",
        "src/engine/model_factory",
        "tests/unit/engine/model_factory",
        "src/engine/model_factory/qwen35_ep_memory.rs",
        "tests/unit/engine/model_factory/qwen35_ep_memory.rs",
    ] {
        assert!(!root.join(path).exists(), "duplicate legacy path: {path}");
    }
    let executor = include_str!("../src/engine/runner_executor.rs");
    assert!(executor.contains("pub struct RunnerMultiSessionExecutor<R: MultiSessionRunner> {"));
    assert!(executor.contains("pub use RunnerMultiSessionExecutor as NativeMultiSessionExecutor;"));
    assert!(!executor.contains("pub struct NativeMultiSessionExecutor"));
    for source in [
        include_str!("../src/engine/runner_executor/kv_dispatch.rs"),
        include_str!("../src/engine/kv_policy.rs"),
    ] {
        assert!(!source.contains(r#"cfg(any(feature = "cuda", test))"#));
    }
}
