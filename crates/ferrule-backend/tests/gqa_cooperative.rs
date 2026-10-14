#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::Command;

fn run_harness(mode: &str) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = manifest.join("tests/support/gqa_cooperative.cu");
    let native_cuda = manifest.join("native/cuda");

    let binary = std::env::temp_dir().join(format!("ferrule-gqa-{mode}-{}", std::process::id()));
    let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let compile = Command::new(nvcc)
        .args([
            "-std=c++17",
            "-O3",
            "-Xcompiler=-Wall,-Wextra",
            "--Werror=all-warnings",
        ])
        .arg(format!(
            "-arch={}",
            env!("FERRULE_BACKEND_CUDA_COMPILED_TARGET")
        ))
        .arg("-I")
        .arg(&native_cuda)
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .status()
        .expect("failed to invoke nvcc");
    assert!(
        compile.success(),
        "nvcc failed to compile direct GQA harness"
    );
    let mut command = Command::new(&binary);
    if mode != "oracle" {
        command.arg(format!("--{mode}"));
    }
    let status = command.status().expect("failed to run direct GQA harness");
    let _ = std::fs::remove_file(binary);
    assert!(status.success(), "direct GQA {mode} failed");
}

#[test]
fn gqa_cooperative_compile_only() {
    // Compile the actual CUDA fixture too, without initializing a CUDA device.
    run_harness("compile-only");
}

#[test]
#[ignore = "requires nvcc and native CUDA GPU; fixed-input bitwise old/new GPU oracle"]
fn gqa_cooperative_fixed_input() {
    run_harness("oracle");
}

#[test]
#[ignore = "requires nvcc and exclusive native CUDA GPU; CUDA-event kernel timings"]
fn gqa_cooperative_kernel_benchmark() {
    run_harness("bench");
}
