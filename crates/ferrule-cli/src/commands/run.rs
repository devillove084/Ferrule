use crate::SamplingArgs;
use ferrule_runtime::{
    GenerateStats, InferenceEngine, Logprobs, ModelGenerationDefaults, ModelRunner, RuntimeRunner,
};
use std::path::Path;

use super::info::print_model_info;

// ── run (auto-dispatch) ──────────────────────────────────────────────────

pub fn cmd_run(
    model_dir: &str,
    prompt: &str,
    max_tokens: usize,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    let runner = RuntimeRunner::load(Path::new(model_dir))?;
    print_model_info(&runner.model_info());

    let mut sc = sampling.sampling_config();
    if let Some(def) = ModelGenerationDefaults::load(Path::new(model_dir)) {
        def.apply_to_config(&mut sc);
    }
    let prompt_tokens = runner.encode(prompt)?.len();
    println!("Prompt: \"{prompt}\" → {prompt_tokens} tokens");
    if prompt_tokens > sampling.ctx_size {
        tracing::info!(
            "warning: prompt ({prompt_tokens} tokens) exceeds ctx_size ({})",
            sampling.ctx_size
        );
    }

    let gen_cfg = sampling.generation_config(max_tokens);

    let mut engine = InferenceEngine::new(runner, sc);
    let t0 = std::time::Instant::now();
    let result = engine.generate_text(prompt, &gen_cfg, |event| {
        let disp = format_token_display(event.token, &event.text, sampling.verbose_tokens());
        println!(
            "  [{:>3}] {disp:30} {:.0}ms",
            event.index,
            t0.elapsed().as_secs_f64() * 1000.0
        );
        if let Some(ref lp) = event.logprobs {
            print_logprobs(lp);
        }
        Ok(())
    })?;

    println!("\n{}", result.text);
    print_generation_stats(&result.stats);
    Ok(())
}

// ── helpers ──────────────────────────────────────────────────────────────

pub fn format_token_display(token: u32, text: &str, verbose: bool) -> String {
    if verbose {
        format!("{text}[{token}]")
    } else {
        text.to_string()
    }
}

pub fn print_logprobs(lp: &Logprobs) {
    println!("  logprobs top-{}:", lp.entries.len());
    for (tid, prob) in &lp.entries {
        println!("    {tid:>6}  {prob:.4}");
    }
}

pub fn print_generation_stats(stats: &GenerateStats) {
    println!(
        "{:.1}s total, {:.1} tok/s decode, {:.1} tok/s total",
        stats.total_time().as_secs_f64(),
        stats.decode_tokens_per_second(),
        stats.total_tokens_per_second()
    );
}

// ── parse_quant (CUDA only) ──────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub fn parse_quant(quant: &str) -> ferrule_quant::QuantType {
    match quant.to_ascii_lowercase().as_str() {
        "q2" | "q2s" => ferrule_quant::QuantType::Q2S,
        "t1" | "t1s" => ferrule_quant::QuantType::T1S,
        "q8" | "q8_0" => ferrule_quant::QuantType::Q8_0,
        _ => ferrule_quant::QuantType::Q4_0,
    }
}

// ── gpu-run ──────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub fn cmd_gpu_run(
    model_dir: &str,
    prompt: &str,
    max_tokens: usize,
    quant: &str,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    let qt = parse_quant(quant);
    tracing::info!("Uploading to GPU (quant: {qt:?})...");
    let runner = RuntimeRunner::load_with_quant(Path::new(model_dir), qt)?;
    print_model_info(&runner.model_info());

    let mut sc = sampling.sampling_config();
    if let Some(def) = ModelGenerationDefaults::load(Path::new(model_dir)) {
        def.apply_to_config(&mut sc);
    }

    let gen_cfg = sampling.generation_config(max_tokens);

    let prompt_tokens = runner.encode(prompt)?.len();
    println!("Prompt: \"{prompt}\" → {prompt_tokens} tokens");
    if prompt_tokens > sampling.ctx_size {
        tracing::info!(
            "warning: prompt ({prompt_tokens} tokens) exceeds ctx_size ({})",
            sampling.ctx_size
        );
    }

    let mut engine = InferenceEngine::new(runner, sc);
    let t0 = std::time::Instant::now();
    let result = engine.generate_text(prompt, &gen_cfg, |event| {
        let disp = format_token_display(event.token, &event.text, sampling.verbose_tokens());
        println!(
            "  [{:>3}] {disp:30} {:.0}ms",
            event.index,
            t0.elapsed().as_secs_f64() * 1000.0
        );
        if let Some(ref lp) = event.logprobs {
            print_logprobs(lp);
        }
        Ok(())
    })?;

    println!("\n{}", result.text);
    print_generation_stats(&result.stats);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn cmd_gpu_run(
    _model_dir: &str,
    _prompt: &str,
    _max_tokens: usize,
    _quant: &str,
    _sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    println!("gpu-run requires --features cuda");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_token_display_verbose() {
        let result = format_token_display(42, "hello", true);
        assert_eq!(result, "hello[42]");
    }

    #[test]
    fn test_format_token_display_plain() {
        let result = format_token_display(42, "hello", false);
        assert_eq!(result, "hello");
    }
}
