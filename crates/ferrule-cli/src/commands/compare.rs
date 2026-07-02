#[cfg(feature = "cuda")]
use ferrule_runtime::ModelRunner;
#[cfg(feature = "cuda")]
use std::path::Path;

// ── compare-logits ───────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub fn cmd_compare_logits(
    model_dir: &str,
    prompt: &str,
    max_tokens: usize,
    quant: &str,
    _free_run: bool,
    ctx_size: usize,
) -> anyhow::Result<()> {
    let qt = super::run::parse_quant(quant);
    tracing::info!("Loading model (quant: {qt:?})...");
    let mut runner = ferrule_runtime::RuntimeRunner::load_with_quant(Path::new(model_dir), qt)?;

    let tokens = runner.encode(prompt)?;
    println!("Prompt: \"{prompt}\" → {} tokens", tokens.len());
    if tokens.len() > ctx_size {
        anyhow::bail!(
            "prompt ({} tokens) exceeds ctx_size ({ctx_size})",
            tokens.len()
        );
    }

    // Prefill
    let logits = runner.prefill(&tokens)?;
    let mut token = ferrule_runtime::argmax(&logits);
    let mut text = String::new();
    let mut step = 0;

    let eos = runner.eos_token_id();

    loop {
        if step >= max_tokens || tokens.len() + step >= ctx_size {
            break;
        }
        if Some(token) == eos {
            break;
        }

        let piece = runner.decode(&[token])?;
        text.push_str(&piece);

        step += 1;

        let next_logits = runner.decode_token(token)?;
        token = ferrule_runtime::argmax(&next_logits);
    }

    println!();
    tracing::info!("=== generation report ===");
    tracing::info!("model:       {model_dir}");
    tracing::info!("quant:       {qt:?}");
    println!("prompt:      \"{prompt}\"");
    tracing::info!("tokens:      {step}");
    println!();
    println!("output: {text}");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn cmd_compare_logits(
    _model_dir: &str,
    _prompt: &str,
    _max_tokens: usize,
    _quant: &str,
    _free_run: bool,
    _ctx_size: usize,
) -> anyhow::Result<()> {
    anyhow::bail!("compare-logits requires --features cuda")
}
