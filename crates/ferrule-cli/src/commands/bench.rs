#[cfg(feature = "cuda")]
use ferrule_runtime::{
    InferenceEngine, ModelGenerationDefaults, ModelRunner, RuntimeRunner, SamplingConfig,
};
#[cfg(feature = "cuda")]
use std::path::Path;

// ── bench-infer ──────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn cmd_bench_infer(
    model_dir: &str,
    prompt: &str,
    max_tokens: usize,
    quant: &str,
    warmup: usize,
    repeat: usize,
    json: bool,
    ctx_size: usize,
) -> anyhow::Result<()> {
    let qt = super::run::parse_quant(quant);

    let mut sc = SamplingConfig::greedy();
    if let Some(def) = ModelGenerationDefaults::load(Path::new(model_dir)) {
        def.apply_to_config(&mut sc);
    }
    let gen_cfg = ferrule_runtime::GenerationConfig {
        max_new_tokens: max_tokens,
        stop: Vec::new(),
        logprobs_k: 0,
        ctx_size,
        ..ferrule_runtime::GenerationConfig::default()
    };

    // Warmup
    for i in 0..warmup {
        let runner = RuntimeRunner::load_with_quant(Path::new(model_dir), qt)?;
        let mut engine = InferenceEngine::new(runner, sc.clone());
        let _ = engine.generate_text(prompt, &gen_cfg, |_| Ok(()))?;
        if !json {
            tracing::info!("  warmup {}/{}", i + 1, warmup);
        }
    }

    let runner = RuntimeRunner::load_with_quant(Path::new(model_dir), qt)?;
    let mut engine = InferenceEngine::new(runner, sc);
    let mut total_pp_secs = Vec::with_capacity(repeat);
    let mut total_tg_secs = Vec::with_capacity(repeat);
    let mut total_tps = Vec::with_capacity(repeat);

    for _ in 0..repeat {
        let result = engine.generate_text(prompt, &gen_cfg, |_| Ok(()))?;
        total_pp_secs.push(result.stats.prefill_time.as_secs_f64());
        total_tg_secs.push(result.stats.decode_time.as_secs_f64());
        total_tps.push(
            result.stats.generated_tokens as f64 / result.stats.decode_time.as_secs_f64().max(1e-6),
        );
    }

    let pp = median(&mut total_pp_secs);
    let tg = median(&mut total_tg_secs);
    let tok_s = median(&mut total_tps);

    let prompt_tokens = engine.runner().encode(prompt)?.len();
    if prompt_tokens > ctx_size {
        tracing::warn!("prompt ({prompt_tokens} tokens) exceeds ctx_size ({ctx_size})");
    }

    if json {
        let out = serde_json::json!({
            "model": model_dir,
            "backend": "gpu",
            "quant": format!("{qt:?}"),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": max_tokens,
            "prompt_seconds": pp,
            "prompt_tok_per_s": prompt_tokens as f64 / pp.max(1e-6),
            "decode_seconds": tg,
            "decode_tok_per_s": tok_s,
            "total_seconds": pp + tg,
            "total_tok_per_s": (prompt_tokens + max_tokens) as f64 / (pp + tg).max(1e-6),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        tracing::info!("bench-infer  model={model_dir}  backend=gpu  quant={qt:?}");
        println!(
            "  prompt: {prompt_tokens} tokens  {pp:.3}s  {:.1} tok/s",
            prompt_tokens as f64 / pp.max(1e-6)
        );
        tracing::info!("  decode: {max_tokens} tokens  {tg:.3}s  {tok_s:.1} tok/s");
        println!(
            "  total:  {} tokens  {:.3}s  {:.1} tok/s",
            prompt_tokens + max_tokens,
            pp + tg,
            (prompt_tokens + max_tokens) as f64 / (pp + tg).max(1e-6)
        );
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn cmd_bench_infer(
    _model_dir: &str,
    _prompt: &str,
    _max_tokens: usize,
    _quant: &str,
    _warmup: usize,
    _repeat: usize,
    _json: bool,
    _ctx_size: usize,
) -> anyhow::Result<()> {
    anyhow::bail!("bench-infer requires --features cuda")
}

#[cfg(feature = "cuda")]
pub fn median(v: &mut [f64]) -> f64 {
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_median_odd() {
        let mut v = vec![3.0, 1.0, 2.0];
        assert!((median(&mut v) - 2.0).abs() < 1e-10);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_median_even() {
        let mut v = vec![4.0, 1.0, 2.0, 3.0];
        assert!((median(&mut v) - 2.5).abs() < 1e-10);
    }
}
