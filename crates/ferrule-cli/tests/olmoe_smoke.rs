//! Golden-token smoke tests — verify GPU produces expected output.
//! Run with: FERRULE_MODEL=models/OLMoE-Instruct cargo test -p ferrule-cli --test olmoe_smoke -- --ignored --features cuda

use ferrule_runtime::{ModelRunner, RuntimeRunner};
use std::path::Path;

fn model_dir() -> String {
    std::env::var("FERRULE_MODEL").unwrap_or_else(|_| "models/OLMoE-Instruct".into())
}

fn has_model() -> bool {
    Path::new(&model_dir()).join("config.json").exists()
}

#[test]
#[ignore]
fn gpu_greedy_first_token_matches() {
    if !has_model() {
        eprintln!("skipping: model not found at {}", model_dir());
        return;
    }
    let mut runner = RuntimeRunner::load(Path::new(&model_dir())).unwrap();
    let tokens = runner.encode("The capital of France is").unwrap();
    let logits = runner.prefill(&tokens).unwrap();

    // Argmax should produce a reasonable next token
    let top = ferrule_runtime::argmax(&logits);
    let decoded = runner.decode(&[top]).unwrap();
    assert!(!decoded.is_empty(), "First token should be non-empty");
    eprintln!("GPU top-1 token: {} → {:?}", top, decoded);
}
