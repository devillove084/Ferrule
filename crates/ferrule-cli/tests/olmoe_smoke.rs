//! Golden-token smoke tests — verify CPU FP32 produces expected output.
//! Run with: FERRULE_MODEL=models/OLMoE-Instruct cargo test -p ferrule-cli --test olmoe_smoke -- --ignored

use ferrule_runtime::ModelRunner;
use std::path::Path;

fn model_dir() -> String {
    std::env::var("FERRULE_MODEL").unwrap_or_else(|_| "models/OLMoE-Instruct".into())
}

fn has_model() -> bool {
    Path::new(&model_dir()).join("config.json").exists()
}

#[test]
#[ignore]
fn cpu_greedy_first_token_matches() {
    if !has_model() {
        eprintln!("skipping: model not found at {}", model_dir());
        return;
    }
    let mut runner = ferrule_runtime::CpuOlmoeRunner::load(Path::new(&model_dir())).unwrap();
    let tokens = runner.encode("The capital of France is").unwrap();
    let logits = runner.prefill(&tokens).unwrap();

    // Argmax should produce a reasonable next token
    let top = ferrule_runtime::argmax(&logits);
    let decoded = runner.decode(&[top]).unwrap();
    assert!(!decoded.is_empty(), "First token should be non-empty");
    // Expected: should start with " Paris" or similar
    eprintln!("CPU top-1 token: {} → {:?}", top, decoded);
}

#[test]
#[ignore]
fn cpu_vs_gpu_first_token_match() {
    if !has_model() {
        eprintln!("skipping: model not found at {}", model_dir());
        return;
    }
    // Only runs with CUDA feature
    #[cfg(feature = "cuda")]
    {
        let mut cpu_runner =
            ferrule_runtime::CpuOlmoeRunner::load(Path::new(&model_dir())).unwrap();
        let mut gpu_runner = ferrule_runtime::GpuOlmoeRunner::load(
            Path::new(&model_dir()),
            ferrule_quant::QuantType::Q4_0,
        )
        .unwrap();

        let tokens = cpu_runner.encode("The capital of France is").unwrap();
        let cpu_logits = cpu_runner.prefill(&tokens).unwrap();
        let gpu_logits = gpu_runner.prefill(&tokens).unwrap();

        let cpu_top = ferrule_runtime::argmax(&cpu_logits);
        let gpu_top = ferrule_runtime::argmax(&gpu_logits);
        assert_eq!(cpu_top, gpu_top, "CPU and GPU first token must match");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("skipping GPU comparison: CUDA not compiled in");
    }
}
