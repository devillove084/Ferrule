use crate::SamplingArgs;
use ferrule_runtime::{
    detect_chat_template, ChatTemplate, InferenceEngine, ModelGenerationDefaults, ModelRunner,
};
use std::io::Write;
use std::path::Path;

use super::info::print_model_info;
use super::run::{format_token_display, print_logprobs};

// ── resolve_template ─────────────────────────────────────────────────────────

fn resolve_template(model_dir: &Path, chat_template_override: Option<&str>) -> ChatTemplate {
    if let Some(name) = chat_template_override {
        ChatTemplate::from_name(name).unwrap_or(ChatTemplate::Plain)
    } else {
        detect_chat_template(model_dir)
    }
}

// ── chat ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub fn cmd_chat(
    model_dir: &str,
    max_tokens: usize,
    quant: &str,
    sampling: &SamplingArgs,
    chat_template_override: Option<&str>,
) -> anyhow::Result<()> {
    let template = resolve_template(Path::new(model_dir), chat_template_override);

    let mut sc = sampling.sampling_config();
    if let Some(def) = ModelGenerationDefaults::load(Path::new(model_dir)) {
        def.apply_to_config(&mut sc);
    }

    let gen_cfg = sampling.generation_config(max_tokens);

    if matches!(quant.to_ascii_lowercase().as_str(), "cpu" | "f32" | "fp32") {
        let runner = ferrule_runtime::CpuOlmoeRunner::load(Path::new(model_dir))?;
        let engine = InferenceEngine::new(runner, sc);
        run_chat_loop(engine, max_tokens, &gen_cfg, template, sampling)
    } else {
        let qt = super::run::parse_quant(quant);
        tracing::info!("Uploading to GPU (quant: {qt:?})...");
        let runner = ferrule_runtime::GpuOlmoeRunner::load(Path::new(model_dir), qt)?;
        let engine = InferenceEngine::new(runner, sc);
        run_chat_loop(engine, max_tokens, &gen_cfg, template, sampling)
    }
}

#[cfg(not(feature = "cuda"))]
pub fn cmd_chat(
    model_dir: &str,
    _max_tokens: usize,
    quant: &str,
    sampling: &SamplingArgs,
    chat_template_override: Option<&str>,
) -> anyhow::Result<()> {
    let template = resolve_template(Path::new(model_dir), chat_template_override);

    let mut sc = sampling.sampling_config();
    if let Some(def) = ModelGenerationDefaults::load(Path::new(model_dir)) {
        def.apply_to_config(&mut sc);
    }

    let gen_cfg = sampling.generation_config(_max_tokens);

    if matches!(quant.to_ascii_lowercase().as_str(), "cpu" | "f32" | "fp32") {
        let runner = ferrule_runtime::CpuOlmoeRunner::load(Path::new(model_dir))?;
        let engine = InferenceEngine::new(runner, sc);
        run_chat_loop(engine, _max_tokens, &gen_cfg, template, sampling)
    } else {
        anyhow::bail!("chat -q {quant} requires --features cuda")
    }
}

// ── run_chat_loop ────────────────────────────────────────────────────────────

pub fn run_chat_loop<R: ModelRunner>(
    mut engine: InferenceEngine<R>,
    _max_tokens: usize,
    generation: &ferrule_runtime::GenerationConfig,
    chat_template: ChatTemplate,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    use console::style;
    use rustyline::error::ReadlineError;

    print_model_info(&engine.runner().model_info());
    println!(
        "{} Type /exit or Ctrl-D to quit. Template: {}.",
        style("Chat ready.").cyan(),
        chat_template.name()
    );
    println!("  /reset      clear session state\n  /stats      show session stats\n  /experts    show expert activation counts\n  /metrics    show observability metrics\n  /ctx        show context window usage");

    let mut first_turn = true;
    let mut rl = rustyline::DefaultEditor::new()?;
    loop {
        let line = match rl.readline(&format!("{} ", style("You>").green().bold())) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/exit" | "/quit") {
            break;
        }
        if input == "/reset" {
            engine.reset_session()?;
            first_turn = true;
            println!("{} session reset.", style("Ferrule>").cyan().bold());
            continue;
        }
        if input == "/stats" {
            let hist = engine.history();
            println!(
                "{} session: {} tokens, {} generated",
                style("Ferrule>").cyan().bold(),
                hist.len(),
                hist.len().saturating_sub(
                    engine
                        .runner()
                        .encode(&chat_template.format_turn("", true))?
                        .len()
                )
            );
            continue;
        }
        if input == "/experts" {
            match engine.runner().expert_report() {
                Some(report) => print!("{report}"),
                None => println!(
                    "{} expert report not available (GPU MoE only).",
                    style("Ferrule>").cyan().bold()
                ),
            }
            continue;
        }
        if input == "/metrics" {
            let snap = ferrule_core::observability::METRICS.snapshot();
            println!("{} {}", style("Ferrule>").cyan().bold(), snap);
            continue;
        }
        if input == "/ctx" {
            let hist = engine.history();
            let usage_pct = if generation.ctx_size > 0 {
                hist.len() as f64 / generation.ctx_size as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "{} context: {} / {} tokens ({:.0}%)",
                style("Ferrule>").cyan().bold(),
                hist.len(),
                generation.ctx_size,
                usage_pct
            );
            continue;
        }
        let _ = rl.add_history_entry(input);

        let prompt = chat_template.format_turn(input, first_turn);
        let prefill = engine.prefill_text_checked(&prompt, generation.ctx_size)?;
        first_turn = false;
        if prefill.logits.is_empty() {
            continue;
        }

        print!("{} ", style("Ferrule>").cyan().bold());
        std::io::stdout().flush()?;
        let _ = engine.generate_from_logits(
            prefill.logits,
            prefill.tokens.len(),
            prefill.prefill_time,
            generation,
            |event| {
                let disp =
                    format_token_display(event.token, &event.text, sampling.verbose_tokens());
                print!("{disp}");
                std::io::stdout().flush()?;
                if let Some(ref lp) = event.logprobs {
                    print_logprobs(lp);
                }
                Ok(())
            },
        )?;
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_template_override_known() {
        let result = resolve_template(Path::new("/nonexistent"), Some("plain"));
        assert_eq!(result.name(), "plain");
    }

    #[test]
    fn test_resolve_template_override_unknown_fallback() {
        let result = resolve_template(Path::new("/nonexistent"), Some("nonexistent_template"));
        assert_eq!(result.name(), "plain");
    }
}
