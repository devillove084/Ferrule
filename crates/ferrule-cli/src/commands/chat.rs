use crate::SamplingArgs;
use ferrule_model::{
    detect_chat_template,
    models::deepseek_v4::{DeepSeekV4ArtifactModel, DeepSeekV4PrepareOptions, DeepSeekV4Runner},
    ChatTemplate, ModelDescriptor, ModelExecutionBackend, ModelFamily, ModelRunner, PrefillMode,
};
use ferrule_runtime::{
    GenerationConfig, LazyEngineWorker, SamplingConfig, SessionId, TopKDecodeStep, TopKFinishReason,
};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::info::print_model_info;

// ── resolve_template ─────────────────────────────────────────────────────────

fn resolve_template(model_dir: &Path, chat_template_override: Option<&str>) -> ChatTemplate {
    if let Some(name) = chat_template_override {
        ChatTemplate::from_name(name).unwrap_or(ChatTemplate::Plain)
    } else {
        detect_chat_template(model_dir)
    }
}

fn resolve_template_for_family(
    model_dir: &Path,
    family: &ModelFamily,
    chat_template_override: Option<&str>,
) -> ChatTemplate {
    if chat_template_override.is_some() {
        return resolve_template(model_dir, chat_template_override);
    }
    if matches!(family, ModelFamily::DeepSeekV4) {
        ChatTemplate::DeepSeekV4
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
    let model_path = Path::new(model_dir);
    let descriptor = ModelDescriptor::load(model_path)?;
    let template =
        resolve_template_for_family(model_path, &descriptor.spec.family, chat_template_override);
    let gen_cfg = sampling.generation_config(max_tokens);

    if !matches!(descriptor.spec.family, ModelFamily::DeepSeekV4) {
        anyhow::bail!(
            "chat currently only supports DeepSeek-V4 models; use deepseek-v4-generate or deepseek-v4-probe for diagnostics"
        );
    }
    let backend = if matches!(quant.to_ascii_lowercase().as_str(), "cpu" | "f32" | "fp32") {
        ModelExecutionBackend::Cpu
    } else {
        ModelExecutionBackend::Cuda
    };
    run_deepseek_v4_chat(model_path, backend, &gen_cfg, template, sampling)
}

#[cfg(not(feature = "cuda"))]
pub fn cmd_chat(
    model_dir: &str,
    _max_tokens: usize,
    quant: &str,
    sampling: &SamplingArgs,
    chat_template_override: Option<&str>,
) -> anyhow::Result<()> {
    let model_path = Path::new(model_dir);
    let descriptor = ModelDescriptor::load(model_path)?;
    let template =
        resolve_template_for_family(model_path, &descriptor.spec.family, chat_template_override);
    let gen_cfg = sampling.generation_config(_max_tokens);

    if matches!(descriptor.spec.family, ModelFamily::DeepSeekV4) {
        if !matches!(quant.to_ascii_lowercase().as_str(), "cpu" | "f32" | "fp32") {
            anyhow::bail!("DeepSeek-V4 chat -q {quant} requires --features cuda; use -q cpu for the slow reference path")
        }
        return run_deepseek_v4_chat(
            model_path,
            ModelExecutionBackend::Cpu,
            &gen_cfg,
            template,
            sampling,
        );
    }
    anyhow::bail!(
        "chat currently only supports DeepSeek-V4 models; use deepseek-v4-generate or deepseek-v4-probe for diagnostics"
    );
}

// ── DeepSeek-V4 chat ─────────────────────────────────────────────────────────

fn deepseek_v4_chat_options() -> DeepSeekV4PrepareOptions {
    DeepSeekV4PrepareOptions {
        output_head_chunk_rows: 4096,
        // Match the ROADMAP interactive goal: keep a bounded per-layer GPU hotset
        // and feed it with predictor-driven lookahead prefetches instead of the
        // old unbounded managed expert cache.
        moe_prefetch_experts: 32,
        moe_hotset_experts: 48,
        ..DeepSeekV4PrepareOptions::default()
    }
}

fn run_deepseek_v4_chat(
    model_path: &Path,
    backend: ModelExecutionBackend,
    generation: &GenerationConfig,
    chat_template: ChatTemplate,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    let sampling_config = sampling.sampling_config();
    let options = deepseek_v4_chat_options();
    if can_use_deepseek_v4_fast_greedy(&sampling_config, generation) {
        run_deepseek_v4_greedy_chat_loop_lazy(
            model_path.to_path_buf(),
            backend,
            options,
            generation,
            chat_template,
            sampling,
        )
    } else {
        anyhow::bail!(
            "DeepSeek-V4 non-greedy/logprob chat is not yet supported; use --temp 0 --repeat-penalty 1 --logprobs 0 for the top-k fast path"
        );
    }
}

fn can_use_deepseek_v4_fast_greedy(
    sampling: &SamplingConfig,
    generation: &GenerationConfig,
) -> bool {
    sampling.temperature <= 0.0
        && (sampling.repeat_penalty - 1.0).abs() < f32::EPSILON
        && generation.logprobs_k == 0
}

fn run_deepseek_v4_greedy_chat_loop_lazy(
    model_path: PathBuf,
    backend: ModelExecutionBackend,
    options: DeepSeekV4PrepareOptions,
    generation: &GenerationConfig,
    chat_template: ChatTemplate,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    use console::style;
    use rustyline::error::ReadlineError;

    let load_path = model_path.clone();
    let mut lazy_engine = LazyEngineWorker::spawn(
        SessionId(0),
        move || DeepSeekV4ArtifactModel::load_hf_with_limit(&load_path, 128 * 1024 * 1024),
        move |model| DeepSeekV4Runner::new_with_operator_backend(model, options, backend),
    );
    let mut model_info_printed = false;
    println!(
        "{} Type /exit or Ctrl-D to quit. Template: {}. DeepSeek-V4 greedy top-k fast path.",
        style("Chat ready.").cyan(),
        chat_template.name()
    );
    println!(
        "  model is loading in the background; the first prompt waits only if loading is not finished"
    );
    println!("  /reset      clear session state\n  /stats      show session stats\n  /experts    show DSV4 layer/cache stats\n  /ctx        show context window usage");

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
        if input == "/reset" || input == "/clear" {
            if let Some(worker) = lazy_engine.worker_mut() {
                worker.reset()?;
            }
            first_turn = true;
            println!("{} session reset.", style("Ferrule>").cyan().bold());
            continue;
        }
        if input == "/stats" {
            if let Some(worker) = lazy_engine.worker() {
                let stats = worker.stats();
                println!(
                    "{} position={} generated={} turns={} bound_layers={}",
                    style("Ferrule>").cyan().bold(),
                    stats.position,
                    stats.generated_tokens,
                    stats.turns,
                    stats.bound_layers.unwrap_or(0)
                );
            } else {
                println!(
                    "{} model loading for {:.1}s; generated=0 bound_layers=0",
                    style("Ferrule>").cyan().bold(),
                    lazy_engine.load_started().elapsed().as_secs_f64()
                );
            }
            continue;
        }
        if input == "/experts" {
            if let Some(worker) = lazy_engine.worker() {
                match worker.expert_report() {
                    Some(report) => print!("{report}"),
                    None => println!(
                        "{} expert report not available.",
                        style("Ferrule>").cyan().bold()
                    ),
                }
            } else {
                println!("{} model still loading.", style("Ferrule>").cyan().bold());
            }
            continue;
        }
        if input == "/ctx" {
            let position = lazy_engine.position_if_loaded();
            let usage_pct = if generation.ctx_size > 0 {
                position as f64 / generation.ctx_size as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "{} context: {} / {} tokens ({:.0}%)",
                style("Ferrule>").cyan().bold(),
                position,
                generation.ctx_size,
                usage_pct
            );
            continue;
        }

        let _ = rl.add_history_entry(input);
        let load_started = lazy_engine.load_started();
        let worker = lazy_engine.ensure_loaded()?;
        if !model_info_printed {
            print_model_info(&worker.runner().model_info());
            eprintln!(
                "[load] DeepSeek-V4 artifact loaded in {:.2}s; CUDA runtime initialized on first use",
                load_started.elapsed().as_secs_f64()
            );
            model_info_printed = true;
        }
        let prompt = chat_template.format_turn(input, first_turn);
        let prompt_tokens = worker.encode(&prompt)?;
        first_turn = false;

        print!("{} ", style("Ferrule>").cyan().bold());
        std::io::stdout().flush()?;

        let turn_start = Instant::now();
        let mut first_token_time = None;
        let mut decode =
            worker.append_prompt(&prompt_tokens, generation, PrefillMode::Interactive, 1)?;
        let turn = loop {
            match worker.decode_next(&mut decode)? {
                TopKDecodeStep::Token(event) => {
                    first_token_time.get_or_insert_with(|| turn_start.elapsed());
                    if sampling.verbose_tokens() {
                        eprint!("[{}:{:.4}]", event.token, event.logit);
                    }
                    print!("{}", event.text);
                    std::io::stdout().flush()?;
                }
                TopKDecodeStep::Finished(turn) => break turn,
            }
        };
        println!();

        if generation.max_new_tokens == 0 {
            println!("{} max_new_tokens is 0.", style("Ferrule>").cyan().bold());
        } else if turn.finish_reason == TopKFinishReason::MaxTokens {
            println!(
                "{} turn stopped at max_tokens; use a larger -n or /reset if the next turn looks malformed.",
                style("Ferrule>").cyan().bold()
            );
        }
        if turn.stopped_by_context {
            println!(
                "{} turn stopped because the context window is full.",
                style("Ferrule>").cyan().bold()
            );
        }
        let prefill_s = turn.prefill_time.as_secs_f64().max(1e-6);
        let decode_s = turn.decode_time.as_secs_f64().max(1e-6);
        let ttft_ms = first_token_time
            .map(|duration| format!("{:.1}ms", duration.as_secs_f64() * 1000.0))
            .unwrap_or_else(|| "n/a".into());
        println!(
            "{} ttft={} prefill={:.1}ms ({:.2} tok/s) decode={:.1}ms ({:.2} tok/s) pos={}",
            style("stats>").dim(),
            ttft_ms,
            turn.prefill_time.as_secs_f64() * 1000.0,
            turn.prompt_tokens as f64 / prefill_s,
            turn.decode_time.as_secs_f64() * 1000.0,
            turn.tokens.len() as f64 / decode_s,
            turn.final_position
        );
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
