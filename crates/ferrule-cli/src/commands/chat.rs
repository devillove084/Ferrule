#[cfg(feature = "cuda")]
use crate::GenerationConfig;
use crate::SamplingArgs;
#[cfg(any(feature = "cuda", test))]
use ferrule_model::{ChatTemplate, detect_chat_template};
#[cfg(feature = "cuda")]
use ferrule_model::{
    ModelDescriptor, ModelExecutionBackend, ModelFamily,
    models::deepseek_v4::{DeepSeekV4ArtifactModel, DeepSeekV4PrepareOptions, DeepSeekV4Runner},
};
#[cfg(feature = "cuda")]
use ferrule_runtime::{
    GenerateRequest, LocalResidentInferenceEngine, RequestId, ResidentActionKind,
    ResidentDriverStep, ResidentSchedulerConfig, ResidentTopKDriverConfig, SequenceFinishReason,
    SessionId,
};
#[cfg(feature = "cuda")]
use std::io::Write;
#[cfg(feature = "cuda")]
use std::os::unix::io::AsRawFd;
#[cfg(any(feature = "cuda", test))]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use super::info::print_model_info;
#[cfg(feature = "cuda")]
use super::resident::{block_on_local_inference, build_resident_topk_driver};

// ── resolve_template ─────────────────────────────────────────────────────────

#[cfg(any(feature = "cuda", test))]
fn resolve_template(model_dir: &Path, chat_template_override: Option<&str>) -> ChatTemplate {
    if let Some(name) = chat_template_override {
        ChatTemplate::from_name(name).unwrap_or(ChatTemplate::Plain)
    } else {
        detect_chat_template(model_dir)
    }
}

#[cfg(feature = "cuda")]
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
            "chat currently only supports DeepSeek-V4 models; use deepseek-v4-generate for diagnostics"
        );
    }
    run_deepseek_v4_chat(model_path, &gen_cfg, template, sampling)
}

#[cfg(not(feature = "cuda"))]
pub fn cmd_chat(
    model_dir: &str,
    max_tokens: usize,
    sampling: &SamplingArgs,
    chat_template_override: Option<&str>,
) -> anyhow::Result<()> {
    let _ = (model_dir, max_tokens, sampling, chat_template_override);
    anyhow::bail!("resident chat requires building ferrule-cli with --features cuda")
}

// ── DeepSeek-V4 chat ─────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn deepseek_v4_chat_options() -> DeepSeekV4PrepareOptions {
    DeepSeekV4PrepareOptions {
        output_head_chunk_rows: 4096,
        // Match the bench-interactive configuration: a bounded per-layer GPU hotset
        // without speculative prefetch. The predictive advisor was removed; prefetch
        // without prediction competes with selected experts for limited slots.
        moe_prefetch_experts: 0,
        moe_hotset_experts: 48,
        ..DeepSeekV4PrepareOptions::default()
    }
}

#[cfg(feature = "cuda")]
fn run_deepseek_v4_chat(
    model_path: &Path,
    generation: &GenerationConfig,
    chat_template: ChatTemplate,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    let options = deepseek_v4_chat_options();
    if sampling.supports_fast_greedy() {
        run_deepseek_v4_greedy_chat_loop(
            model_path.to_path_buf(),
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

#[cfg(feature = "cuda")]
fn run_deepseek_v4_greedy_chat_loop(
    model_path: PathBuf,
    options: DeepSeekV4PrepareOptions,
    generation: &GenerationConfig,
    chat_template: ChatTemplate,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    block_on_local_inference(run_deepseek_v4_greedy_chat_loop_async(
        model_path,
        options,
        generation,
        chat_template,
        sampling,
    ))
}

#[cfg(feature = "cuda")]
async fn run_deepseek_v4_greedy_chat_loop_async(
    model_path: PathBuf,
    options: DeepSeekV4PrepareOptions,
    generation: &GenerationConfig,
    chat_template: ChatTemplate,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    use console::style;
    use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
    use rustyline::error::ReadlineError;

    // Redirect stderr (tracing INFO lines) to a tmp file for a clean terminal.
    let log_path = std::env::temp_dir().join(format!("ferrule-chat-{}.log", std::process::id()));
    let _log_guard;
    if std::env::var_os("FERRULE_CHAT_KEEP_STDERR").is_none() {
        let log_file = std::fs::File::create(&log_path)?;
        let log_fd = log_file.as_raw_fd();
        // SAFETY: dup2 replaces stderr fd before any multi-threaded work starts.
        unsafe {
            libc::dup2(log_fd, 2);
        }
        _log_guard = Some(log_file);
    } else {
        _log_guard = None;
    }
    eprintln!("[log] stderr -> {}", log_path.display());

    let session_id = SessionId(0);
    let driver_config = ResidentTopKDriverConfig {
        ctx_size: generation.ctx_size,
        stop_at_eos: generation.stop_at_eos,
        proposal_confidence_threshold: 0.2,
    };
    let scheduler_config = ResidentSchedulerConfig {
        prefill_chunk_size: 4096,
        max_active_sequences: 1,
        max_decode_batch: 1,
        ..Default::default()
    };
    let load_started = Instant::now();
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_path, 128 * 1024 * 1024)?;
    let runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let schema = runner.kv_layout_schema().clone();
    let mut driver = LocalResidentInferenceEngine::new(build_resident_topk_driver(
        runner,
        Box::new(schema),
        scheduler_config,
        driver_config,
    )?);
    driver.retain_session(session_id)?;
    print_model_info(&driver.model_info());
    println!(
        "[load] DeepSeek-V4 artifact and CUDA owner initialized in {:.2}s",
        load_started.elapsed().as_secs_f64()
    );
    let mut generated_tokens = 0usize;
    let mut turns = 0u64;
    println!(
        "{} Type /exit or Ctrl-D to quit. Template: {}. DeepSeek-V4 greedy top-k fast path.",
        style("Chat ready.").cyan(),
        chat_template.name()
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
            driver.reset_session(session_id)?;
            first_turn = true;
            generated_tokens = 0;
            turns = 0;
            println!("{} session reset.", style("Ferrule>").cyan().bold());
            continue;
        }
        if input == "/stats" {
            println!(
                "{} position={} generated={} turns={} bound_layers={}",
                style("Ferrule>").cyan().bold(),
                driver.retained_session_position(session_id).unwrap_or(0),
                generated_tokens,
                turns,
                driver.bound_layer_count().unwrap_or_default()
            );
            continue;
        }
        if input == "/experts" {
            match driver.expert_report() {
                Some(report) => print!("{report}"),
                None => println!(
                    "{} expert report not available.",
                    style("Ferrule>").cyan().bold()
                ),
            }
            continue;
        }
        if input == "/ctx" {
            let position = driver.retained_session_position(session_id).unwrap_or(0);
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
        let prompt = chat_template.format_turn(input, first_turn);
        let prompt_tokens = driver.encode(&prompt)?;
        first_turn = false;

        // Spinner while waiting for the first token.
        // Write to stdout so it appears in the terminal even when stderr is redirected.
        let spinner = ProgressBar::with_draw_target(Some(80), ProgressDrawTarget::stdout());
        spinner.set_style(
            ProgressStyle::with_template("{spinner:.dim} {msg}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        spinner.set_message("thinking…");
        spinner.enable_steady_tick(std::time::Duration::from_millis(80));
        let mut spinner_active = true;

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
            let step = driver
                .step(&mut |event| {
                    if spinner_active {
                        spinner.finish_and_clear();
                        spinner_active = false;
                        print!("{} ", style("Ferrule>").cyan().bold());
                        std::io::stdout().flush()?;
                    }
                    first_token_time.get_or_insert_with(|| turn_start.elapsed());
                    if sampling.verbose_tokens() {
                        eprint!("[{}:{:.4}]", event.token, event.logit.unwrap_or(f32::NAN));
                    }
                    print!("{}", event.text);
                    std::io::stdout().flush()?;
                    turn_tokens = turn_tokens.saturating_add(1);
                    Ok(())
                })
                .await?;
            let elapsed = step_started.elapsed();
            match step {
                ResidentDriverStep::Executed { action_kind, .. } => match action_kind {
                    ResidentActionKind::Prefill | ResidentActionKind::Mixed => {
                        prefill_time += elapsed;
                    }
                    ResidentActionKind::Decode => decode_time += elapsed,
                    ResidentActionKind::Finish | ResidentActionKind::Cancel => {}
                },
                ResidentDriverStep::WaitingForModelProgress(_) => decode_time += elapsed,
                ResidentDriverStep::Idle => break,
                ResidentDriverStep::Blocked => {
                    if spinner_active {
                        spinner.finish_and_clear();
                    }
                    anyhow::bail!("resident chat driver blocked while running a turn")
                }
            }
        }
        if spinner_active {
            spinner.finish_and_clear();
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
        let stats_line = format!(
            "ttft={} prefill={:.1}ms ({:.2} tok/s) decode={:.1}ms ({:.2} tok/s) pos={}",
            ttft_ms,
            prefill_time.as_secs_f64() * 1000.0,
            prompt_tokens.len() as f64 / prefill_s,
            decode_time.as_secs_f64() * 1000.0,
            turn_tokens as f64 / decode_s,
            turn.position,
        );
        // Write full stats to a tmp file; show a dim one-liner on stderr.
        let stats_path =
            std::env::temp_dir().join(format!("ferrule-chat-stats-{}.log", std::process::id()));
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&stats_path)
            .and_then(|mut f| writeln!(f, "[turn {turns}] {stats_line}"))
            .ok();
        eprintln!("{} {}", style("stats>").dim(), style(&stats_line).dim());
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
