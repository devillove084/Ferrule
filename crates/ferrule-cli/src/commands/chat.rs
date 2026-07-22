use crate::{GenerationConfig, SamplingArgs};
use ferrule_model::{
    ChatTemplate, ModelDescriptor, ModelExecutionBackend, ModelFamily, ModelRunner,
    detect_chat_template,
    models::deepseek_v4::{DeepSeekV4ArtifactModel, DeepSeekV4PrepareOptions, DeepSeekV4Runner},
};
use ferrule_runtime::{
    FixedSequenceSlotPool, GenerateRequest, RequestId, ResidentActionKind, ResidentDriverStep,
    ResidentSchedulerConfig, ResidentTopKDriver, ResidentTopKDriverConfig, SequenceFinishReason,
    SessionId,
};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::info::print_model_info;
use super::resident::build_resident_topk_driver;

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
            anyhow::bail!(
                "DeepSeek-V4 chat -q {quant} requires --features cuda; use -q cpu for the slow reference path"
            )
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
    let options = deepseek_v4_chat_options();
    if sampling.supports_fast_greedy() {
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

type DeepSeekV4ChatDriver = ResidentTopKDriver<DeepSeekV4Runner, FixedSequenceSlotPool>;

struct LazyResidentChatDriver {
    load_started: Instant,
    loader: Option<std::thread::JoinHandle<anyhow::Result<DeepSeekV4Runner>>>,
    driver: Option<DeepSeekV4ChatDriver>,
    scheduler_config: ResidentSchedulerConfig,
    driver_config: ResidentTopKDriverConfig,
    session_id: SessionId,
}

impl LazyResidentChatDriver {
    fn spawn(
        load: impl FnOnce() -> anyhow::Result<DeepSeekV4Runner> + Send + 'static,
        scheduler_config: ResidentSchedulerConfig,
        driver_config: ResidentTopKDriverConfig,
        session_id: SessionId,
    ) -> Self {
        Self {
            load_started: Instant::now(),
            loader: Some(std::thread::spawn(load)),
            driver: None,
            scheduler_config,
            driver_config,
            session_id,
        }
    }

    fn poll(&mut self) -> anyhow::Result<()> {
        if self
            .loader
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished)
        {
            self.join_loader()?;
        }
        Ok(())
    }

    fn join_loader(&mut self) -> anyhow::Result<()> {
        let loader = self
            .loader
            .take()
            .ok_or_else(|| anyhow::anyhow!("resident chat loader is unavailable"))?;
        let runner = loader
            .join()
            .map_err(|_| anyhow::anyhow!("resident chat loader thread panicked"))??;
        let schema = runner.kv_layout_schema().clone();
        let mut driver = build_resident_topk_driver(
            runner,
            Box::new(schema),
            self.scheduler_config,
            self.driver_config,
        )?;
        driver.retain_session(self.session_id)?;
        self.driver = Some(driver);
        Ok(())
    }

    fn driver_if_ready(&mut self) -> anyhow::Result<Option<&mut DeepSeekV4ChatDriver>> {
        self.poll()?;
        Ok(self.driver.as_mut())
    }

    fn ensure_loaded(&mut self) -> anyhow::Result<&mut DeepSeekV4ChatDriver> {
        if self.driver.is_none() {
            self.join_loader()?;
        }
        self.driver
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("resident chat driver failed to load"))
    }
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

    let session_id = SessionId(0);
    let driver_config = ResidentTopKDriverConfig {
        ctx_size: generation.ctx_size,
        stop_at_eos: generation.stop_at_eos,
        append_eos_to_session: generation.append_eos_to_session,
        dspark_confidence_threshold: 0.2,
        max_steps_per_run: generation
            .ctx_size
            .saturating_add(generation.max_new_tokens)
            .saturating_add(64),
    };
    let scheduler_config = ResidentSchedulerConfig {
        prefill_chunk_size: 4096,
        max_active_sequences: 1,
        max_decode_batch: 1,
        ..Default::default()
    };
    let mut lazy_driver = LazyResidentChatDriver::spawn(
        move || {
            let model =
                DeepSeekV4ArtifactModel::load_hf_with_limit(&model_path, 128 * 1024 * 1024)?;
            DeepSeekV4Runner::new_with_operator_backend(model, options, backend).map_err(Into::into)
        },
        scheduler_config,
        driver_config,
        session_id,
    );
    let mut model_info_printed = false;
    let mut generated_tokens = 0usize;
    let mut turns = 0u64;
    println!(
        "{} Type /exit or Ctrl-D to quit. Template: {}. DeepSeek-V4 greedy top-k fast path.",
        style("Chat ready.").cyan(),
        chat_template.name()
    );
    println!(
        "  model is loading in the background; the first prompt waits only if loading is not finished"
    );
    println!(
        "  /reset      clear session state\n  /stats      show session stats\n  /experts    show DSV4 layer/cache stats\n  /ctx        show context window usage"
    );

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
            if let Some(driver) = lazy_driver.driver_if_ready()? {
                driver.reset_session(session_id)?;
            }
            first_turn = true;
            generated_tokens = 0;
            turns = 0;
            println!("{} session reset.", style("Ferrule>").cyan().bold());
            continue;
        }
        if input == "/stats" {
            if let Some(driver) = lazy_driver.driver_if_ready()? {
                println!(
                    "{} position={} generated={} turns={} bound_layers={}",
                    style("Ferrule>").cyan().bold(),
                    driver.retained_session_position(session_id).unwrap_or(0),
                    generated_tokens,
                    turns,
                    driver.executor().runner().bound_layer_count()
                );
            } else {
                println!(
                    "{} model loading for {:.1}s; generated=0 bound_layers=0",
                    style("Ferrule>").cyan().bold(),
                    lazy_driver.load_started.elapsed().as_secs_f64()
                );
            }
            continue;
        }
        if input == "/experts" {
            if let Some(driver) = lazy_driver.driver_if_ready()? {
                match driver.executor().runner().expert_report() {
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
            let position = lazy_driver
                .driver_if_ready()?
                .and_then(|driver| driver.retained_session_position(session_id))
                .unwrap_or(0);
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
        let load_started = lazy_driver.load_started;
        let driver = lazy_driver.ensure_loaded()?;
        if !model_info_printed {
            print_model_info(&driver.executor().runner().model_info());
            eprintln!(
                "[load] DeepSeek-V4 artifact loaded in {:.2}s; CUDA runtime initialized on first use",
                load_started.elapsed().as_secs_f64()
            );
            model_info_printed = true;
        }
        let prompt = chat_template.format_turn(input, first_turn);
        let prompt_tokens = driver.executor().runner().encode(&prompt)?;
        first_turn = false;

        print!("{} ", style("Ferrule>").cyan().bold());
        std::io::stdout().flush()?;

        turns = turns.saturating_add(1);
        driver.submit(GenerateRequest {
            id: RequestId(turns),
            session_id: Some(session_id),
            prompt_tokens: prompt_tokens.clone(),
            max_new_tokens: generation.max_new_tokens,
            stop: generation.stop.clone(),
            ignore_eos: !generation.stop_at_eos,
        });
        let turn_start = Instant::now();
        let mut first_token_time = None;
        let mut prefill_time = std::time::Duration::ZERO;
        let mut decode_time = std::time::Duration::ZERO;
        let mut turn_tokens = 0usize;
        loop {
            let step_started = Instant::now();
            let step = driver.step(&mut |event| {
                first_token_time.get_or_insert_with(|| turn_start.elapsed());
                if sampling.verbose_tokens() {
                    eprint!("[{}:{:.4}]", event.token, event.logit.unwrap_or(f32::NAN));
                }
                print!("{}", event.text);
                std::io::stdout().flush()?;
                turn_tokens = turn_tokens.saturating_add(1);
                Ok(())
            })?;
            let elapsed = step_started.elapsed();
            match step {
                ResidentDriverStep::Executed { action_kind, .. } => match action_kind {
                    ResidentActionKind::Prefill | ResidentActionKind::Mixed => {
                        prefill_time += elapsed;
                    }
                    ResidentActionKind::Decode => decode_time += elapsed,
                    ResidentActionKind::Finish | ResidentActionKind::Cancel => {}
                },
                ResidentDriverStep::Idle => break,
                ResidentDriverStep::Blocked => {
                    anyhow::bail!("resident chat driver blocked while running a turn")
                }
            }
        }
        println!();
        generated_tokens = generated_tokens.saturating_add(turn_tokens);

        let finished = driver.drain_finished();
        let turn = finished
            .last()
            .ok_or_else(|| anyhow::anyhow!("resident chat driver produced no finished sequence"))?;
        if generation.max_new_tokens == 0 {
            println!("{} max_new_tokens is 0.", style("Ferrule>").cyan().bold());
        } else if turn.finish_reason == Some(SequenceFinishReason::MaxTokens) {
            println!(
                "{} turn stopped at max_tokens; use a larger -n or /reset if the next turn looks malformed.",
                style("Ferrule>").cyan().bold()
            );
        }
        if turn.finish_reason == Some(SequenceFinishReason::Context) {
            println!(
                "{} turn stopped because the context window is full.",
                style("Ferrule>").cyan().bold()
            );
        }
        let prefill_s = prefill_time.as_secs_f64().max(1e-6);
        let decode_s = decode_time.as_secs_f64().max(1e-6);
        let ttft_ms = first_token_time
            .map(|duration| format!("{:.1}ms", duration.as_secs_f64() * 1000.0))
            .unwrap_or_else(|| "n/a".into());
        println!(
            "{} ttft={} prefill={:.1}ms ({:.2} tok/s) decode={:.1}ms ({:.2} tok/s) pos={}",
            style("stats>").dim(),
            ttft_ms,
            prefill_time.as_secs_f64() * 1000.0,
            prompt_tokens.len() as f64 / prefill_s,
            decode_time.as_secs_f64() * 1000.0,
            turn_tokens as f64 / decode_s,
            turn.position
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
