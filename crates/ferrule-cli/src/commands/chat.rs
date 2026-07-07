use crate::SamplingArgs;
use ferrule_model::{ModelDescriptor, ModelFamily};
use ferrule_runtime::{
    detect_chat_template,
    models::deepseek_v4::{
        DeepSeekV4ArtifactModel, DeepSeekV4OperatorBackend, DeepSeekV4ReferenceOptions,
        DeepSeekV4ReferenceRunner,
    },
    ChatTemplate, GenerationConfig, InferenceEngine, ModelGenerationDefaults, ModelRunner,
    RuntimeRunner, SamplingConfig,
};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::Instant;

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

    if matches!(descriptor.spec.family, ModelFamily::DeepSeekV4) {
        let backend = if matches!(quant.to_ascii_lowercase().as_str(), "cpu" | "f32" | "fp32") {
            DeepSeekV4OperatorBackend::Cpu
        } else {
            DeepSeekV4OperatorBackend::Cuda
        };
        return run_deepseek_v4_chat(model_path, backend, &gen_cfg, template, sampling);
    }

    let mut sc = sampling.sampling_config();
    if let Some(def) = ModelGenerationDefaults::load(model_path) {
        def.apply_to_config(&mut sc);
    }

    if matches!(quant.to_ascii_lowercase().as_str(), "cpu" | "f32" | "fp32") {
        let runner = RuntimeRunner::load(model_path)?;
        let engine = InferenceEngine::new(runner, sc);
        run_chat_loop(engine, max_tokens, &gen_cfg, template, sampling)
    } else {
        let qt = super::run::parse_quant(quant);
        tracing::info!("Uploading to GPU (quant: {qt:?})...");
        let runner = RuntimeRunner::load_with_quant(model_path, qt)?;
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
            DeepSeekV4OperatorBackend::Cpu,
            &gen_cfg,
            template,
            sampling,
        );
    }

    let mut sc = sampling.sampling_config();
    if let Some(def) = ModelGenerationDefaults::load(model_path) {
        def.apply_to_config(&mut sc);
    }

    if matches!(quant.to_ascii_lowercase().as_str(), "cpu" | "f32" | "fp32") {
        let runner = RuntimeRunner::load(model_path)?;
        let engine = InferenceEngine::new(runner, sc);
        run_chat_loop(engine, _max_tokens, &gen_cfg, template, sampling)
    } else {
        anyhow::bail!("chat -q {quant} requires --features cuda")
    }
}

// ── DeepSeek-V4 chat ─────────────────────────────────────────────────────────

fn deepseek_v4_chat_options() -> DeepSeekV4ReferenceOptions {
    DeepSeekV4ReferenceOptions {
        output_head_chunk_rows: 4096,
        // Match the ROADMAP interactive goal: use the observed per-layer hotset
        // for predictive residency instead of naive low expert IDs. This improves
        // steady chat turns while keeping FERRULE_CUDA_MOE_TC available for A/B.
        moe_prefetch_experts: 8,
        ..DeepSeekV4ReferenceOptions::default()
    }
}

fn run_deepseek_v4_chat(
    model_path: &Path,
    backend: DeepSeekV4OperatorBackend,
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
        let runner = DeepSeekV4ReferenceRunner::load_hf_with_options_and_backend(
            model_path,
            128 * 1024 * 1024,
            options,
            backend,
        )?;
        eprintln!(
            "note: DeepSeek-V4 non-greedy/logprob chat uses the full-logits path; use --temp 0 --repeat-penalty 1 --logprobs 0 for the top-k fast path"
        );
        let engine = InferenceEngine::new(runner, sampling_config);
        run_chat_loop(
            engine,
            generation.max_new_tokens,
            generation,
            chat_template,
            sampling,
        )
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

struct DeepSeekV4LazyRunner {
    backend: DeepSeekV4OperatorBackend,
    options: DeepSeekV4ReferenceOptions,
    loader: Option<JoinHandle<anyhow::Result<DeepSeekV4ArtifactModel>>>,
    runner: Option<DeepSeekV4ReferenceRunner>,
    load_started: Instant,
    model_info_printed: bool,
}

impl DeepSeekV4LazyRunner {
    fn loading(
        model_path: PathBuf,
        backend: DeepSeekV4OperatorBackend,
        options: DeepSeekV4ReferenceOptions,
    ) -> Self {
        let load_path = model_path.clone();
        let loader = std::thread::spawn(move || {
            Ok(DeepSeekV4ArtifactModel::load_hf_with_limit(
                &load_path,
                128 * 1024 * 1024,
            )?)
        });
        Self {
            backend,
            options,
            loader: Some(loader),
            runner: None,
            load_started: Instant::now(),
            model_info_printed: false,
        }
    }

    fn position_if_loaded(&self) -> usize {
        self.runner
            .as_ref()
            .map(DeepSeekV4ReferenceRunner::position)
            .unwrap_or(0)
    }

    fn runner_mut_if_loaded(&mut self) -> Option<&mut DeepSeekV4ReferenceRunner> {
        self.runner.as_mut()
    }

    fn ensure_loaded(&mut self) -> anyhow::Result<&mut DeepSeekV4ReferenceRunner> {
        if self.runner.is_none() {
            let loader = self
                .loader
                .take()
                .ok_or_else(|| anyhow::anyhow!("DeepSeek-V4 model loader was not started"))?;
            let model = loader
                .join()
                .map_err(|_| anyhow::anyhow!("DeepSeek-V4 model loader panicked"))??;
            let runner = DeepSeekV4ReferenceRunner::new_with_operator_backend(
                model,
                self.options,
                self.backend,
            )?;
            self.runner = Some(runner);
        }
        if !self.model_info_printed {
            if let Some(runner) = self.runner.as_ref() {
                print_model_info(&runner.model_info());
                eprintln!(
                    "[load] DeepSeek-V4 artifact loaded in {:.2}s; CUDA runtime initialized on first use",
                    self.load_started.elapsed().as_secs_f64()
                );
            }
            self.model_info_printed = true;
        }
        Ok(self.runner.as_mut().expect("runner initialized above"))
    }
}

fn run_deepseek_v4_greedy_chat_loop_lazy(
    model_path: PathBuf,
    backend: DeepSeekV4OperatorBackend,
    options: DeepSeekV4ReferenceOptions,
    generation: &GenerationConfig,
    chat_template: ChatTemplate,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    use console::style;
    use rustyline::error::ReadlineError;

    let mut lazy_runner = DeepSeekV4LazyRunner::loading(model_path, backend, options);
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
    let mut generated_total = 0usize;
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
            if let Some(runner) = lazy_runner.runner_mut_if_loaded() {
                runner.reset_session()?;
            }
            first_turn = true;
            generated_total = 0;
            println!("{} session reset.", style("Ferrule>").cyan().bold());
            continue;
        }
        if input == "/stats" {
            if let Some(runner) = lazy_runner.runner.as_ref() {
                println!(
                    "{} position={} generated={} bound_layers={}",
                    style("Ferrule>").cyan().bold(),
                    runner.position(),
                    generated_total,
                    runner.bound_layer_count()
                );
            } else {
                println!(
                    "{} model loading for {:.1}s; generated=0 bound_layers=0",
                    style("Ferrule>").cyan().bold(),
                    lazy_runner.load_started.elapsed().as_secs_f64()
                );
            }
            continue;
        }
        if input == "/experts" {
            if let Some(runner) = lazy_runner.runner.as_ref() {
                match runner.expert_report() {
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
            let position = lazy_runner.position_if_loaded();
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
        let runner = lazy_runner.ensure_loaded()?;
        let prompt = chat_template.format_turn(input, first_turn);
        let prompt_tokens = runner.encode(&prompt)?;
        ensure_deepseek_v4_context_room(
            runner.position(),
            prompt_tokens.len(),
            generation.ctx_size,
        )?;
        first_turn = false;

        let prefill_start = std::time::Instant::now();
        let mut top = runner.prefill_tokens_topk_interactive(&prompt_tokens, 1)?;
        if top.is_empty() {
            println!();
            continue;
        }

        print!("{} ", style("Ferrule>").cyan().bold());
        std::io::stdout().flush()?;

        let mut text = String::new();
        let mut turn_generated = 0usize;
        let mut stopped_by_eos = false;
        let mut stopped_by_string = None;
        let eos = runner.eos_token_id();
        let decode_start = std::time::Instant::now();

        for step in 0..generation.max_new_tokens {
            if runner.position() >= generation.ctx_size {
                break;
            }
            let Some(&next) = top.first() else {
                break;
            };

            if sampling.verbose_tokens() {
                eprint!("[{}:{:.4}]", next.token_id, next.logit);
            }

            if eos == Some(next.token_id) {
                if generation.append_eos_to_session {
                    runner.feed_token(next.token_id)?;
                }
                stopped_by_eos = true;
                break;
            }

            let piece = runner.decode(&[next.token_id]).unwrap_or_default();
            print!("{piece}");
            std::io::stdout().flush()?;
            text.push_str(&piece);
            turn_generated += 1;
            generated_total += 1;

            if let Some(stop) = matched_deepseek_v4_stop(&text, &generation.stop) {
                runner.feed_token(next.token_id)?;
                stopped_by_string = Some(stop.to_string());
                break;
            }

            if step + 1 == generation.max_new_tokens {
                runner.feed_token(next.token_id)?;
                break;
            }

            top = runner.decode_token_topk(next.token_id, 1)?;
        }
        println!();

        if generation.max_new_tokens == 0 {
            println!("{} max_new_tokens is 0.", style("Ferrule>").cyan().bold());
        } else if !stopped_by_eos
            && stopped_by_string.is_none()
            && turn_generated == generation.max_new_tokens
        {
            println!(
                "{} turn stopped at max_tokens; use a larger -n or /reset if the next turn looks malformed.",
                style("Ferrule>").cyan().bold()
            );
        }
        println!(
            "{} prefill={:.1}ms decode={:.1}ms pos={}",
            style("stats>").dim(),
            prefill_start.elapsed().as_secs_f64() * 1000.0,
            decode_start.elapsed().as_secs_f64() * 1000.0,
            runner.position()
        );
    }

    Ok(())
}

fn matched_deepseek_v4_stop<'a>(text: &str, stop: &'a [String]) -> Option<&'a str> {
    stop.iter()
        .find(|candidate| !candidate.is_empty() && text.ends_with(candidate.as_str()))
        .map(String::as_str)
}

fn ensure_deepseek_v4_context_room(
    current_tokens: usize,
    new_tokens: usize,
    ctx_size: usize,
) -> anyhow::Result<()> {
    if ctx_size == 0 {
        anyhow::bail!("ctx_size must be greater than zero");
    }
    let requested = current_tokens.saturating_add(new_tokens);
    if requested > ctx_size {
        anyhow::bail!("context length {requested} exceeds ctx_size {ctx_size}");
    }
    Ok(())
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
