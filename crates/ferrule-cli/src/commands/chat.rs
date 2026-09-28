use crate::{GenerationConfig, SamplingArgs};
use ferrule_model::AutoConfig;
use ferrule_runtime::engine::LocalSessionInferenceEngine;
use ferrule_runtime::engine::model_factory::ExpertCacheOptions;
use ferrule_runtime::{
    BackendSelection, GenerateRequest, ModelFactoryOptions, RequestId, ResidentActionKind,
    ResidentDriverStep, ResidentModelBuildPlan, ResidentModelPlanner, SequenceFinishReason,
    SessionId,
};
use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

use super::info::print_model_info;
use super::resident::{
    block_on_local_inference, require_finished_request, resident_driver_config,
    single_sequence_scheduler_config,
};

// ── chat ─────────────────────────────────────────────────────────────────────

pub fn cmd_chat(
    model_dir: &str,
    max_tokens: usize,
    sampling: &SamplingArgs,
    backend_override: Option<&str>,
    chat_template_override: Option<&str>,
) -> anyhow::Result<()> {
    if !sampling.supports_fast_greedy() {
        anyhow::bail!(
            "resident non-greedy/logprob chat is not yet supported; use --temp 0 --repeat-penalty 1 --logprobs 0 for the top-k fast path"
        );
    }

    let config = AutoConfig::from_pretrained(model_dir)?;
    let generation = sampling.generation_config(max_tokens);
    let backend = backend_override
        .map(BackendSelection::parse)
        .transpose()?
        .unwrap_or_default();
    let options = chat_model_options(&config, &generation, sampling)?;
    let planner = ResidentModelPlanner::new();
    let prepared = if sampling.expert_parallel != 1 || sampling.devices.is_some() {
        planner.prepare_qwen35_expert_parallel(
            &config,
            backend,
            chat_template_override,
            options,
            ferrule_runtime::engine::model_factory::PipelineBuildOptions {
                parallelism: ferrule_common::ParallelismPlan {
                    expert_parallel: sampling.expert_parallel,
                    ..Default::default()
                },
                devices: sampling.devices.clone(),
                ..Default::default()
            },
        )?
    } else {
        planner.prepare(&config, backend, chat_template_override, options)?
    };
    // Print before chat redirects stderr: adjustments must remain user-visible.
    eprintln!("[model plan] {}", prepared.resolution_report());
    let model_name = prepared.model_name();
    let backend_profile = prepared.backend_profile();
    let chat_template = prepared.chat_template();

    run_greedy_chat_loop(
        prepared,
        model_name,
        backend_profile,
        &generation,
        chat_template,
        sampling,
    )
}

fn chat_model_options(
    config: &AutoConfig,
    generation: &GenerationConfig,
    sampling: &SamplingArgs,
) -> anyhow::Result<ModelFactoryOptions> {
    let mut driver_config = resident_driver_config(generation.ctx_size, generation.stop_at_eos);
    if matches!(
        config.descriptor().spec.family,
        ferrule_model::ModelFamily::Qwen35 | ferrule_model::ModelFamily::Qwen35Moe
    ) {
        driver_config.enable_native_proposals = false;
    }
    Ok(ModelFactoryOptions {
        max_layers: None,
        max_tensor_mebibytes: if matches!(
            config.descriptor().spec.family,
            ferrule_model::ModelFamily::Qwen35 | ferrule_model::ModelFamily::Qwen35Moe
        ) {
            1024
        } else {
            128
        },
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_mebibytes: 64,
        expert_cache: ExpertCacheOptions::default(),
        qwen35_moe_capacity: sampling.cuda_expert_device.capacity_limits(),
        qwen35_host_cache: sampling.expert_prewarm.options(
            sampling.expert_host_cache_entries,
            sampling.expert_host_cache_mb,
        )?,
        moe_hotset_experts: 0,
        kv_cache_mebibytes: None,
        scheduler_config: single_sequence_scheduler_config(4096),
        driver_config,
    })
}

fn run_greedy_chat_loop(
    prepared: ResidentModelBuildPlan,
    adapter_name: &'static str,
    backend_profile: &'static str,
    generation: &GenerationConfig,
    chat_template: ferrule_model::ChatTemplate,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    block_on_local_inference(run_greedy_chat_loop_async(
        prepared,
        adapter_name,
        backend_profile,
        generation,
        chat_template,
        sampling,
    ))
}

async fn run_greedy_chat_loop_async(
    prepared: ResidentModelBuildPlan,
    adapter_name: &'static str,
    backend_profile: &'static str,
    generation: &GenerationConfig,
    chat_template: ferrule_model::ChatTemplate,
    sampling: &SamplingArgs,
) -> anyhow::Result<()> {
    use console::style;
    use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
    use rustyline::error::ReadlineError;

    // Full host prewarm may take minutes on cold NAS: keep its startup admission
    // and progress visible rather than hiding them behind an apparently stuck REPL.
    let show_host_prewarm = prepared.qwen35_host_cache_options().is_some();
    // Other profiles retain the historical clean-terminal log redirection.
    let log_path = std::env::temp_dir().join(format!("ferrule-chat-{}.log", std::process::id()));
    let _log_guard = if !show_host_prewarm && std::env::var_os("FERRULE_CHAT_KEEP_STDERR").is_none()
    {
        let log_file = std::fs::File::create(&log_path)?;
        let log_fd = log_file.as_raw_fd();
        // SAFETY: dup2 replaces stderr fd before any multi-threaded work starts.
        #[allow(
            unsafe_code,
            reason = "redirecting the inherited stderr file descriptor requires libc"
        )]
        unsafe {
            libc::dup2(log_fd, 2);
        }
        Some(log_file)
    } else {
        None
    };
    if _log_guard.is_some() {
        eprintln!("[log] stderr -> {}", log_path.display());
    }

    let session_id = SessionId(0);
    let load_started = Instant::now();
    let mut driver = LocalSessionInferenceEngine::new(prepared.build()?);
    driver.initialize().await?;
    driver.retain_session(session_id)?;
    print_model_info(&driver.model_info());
    println!(
        "[load] {adapter_name} artifact and {backend_profile} owner initialized in {:.2}s",
        load_started.elapsed().as_secs_f64()
    );
    if show_host_prewarm && let Some(report) = driver.expert_report() {
        println!("{report}");
    }
    let mut generated_tokens = 0usize;
    let mut turns = 0u64;
    println!(
        "{} Type /exit or Ctrl-D to quit. Template: {}. {adapter_name} greedy top-k fast path.",
        style("Chat ready.").cyan(),
        chat_template.name()
    );

    println!(
        "  /reset      clear session state\n  /stats      show session stats\n  /experts    show model layer/cache stats\n  /ctx        show context window usage"
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
        let request_id = RequestId(turns);
        driver.submit(GenerateRequest {
            id: request_id,
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
        let turn = loop {
            let step_started = Instant::now();
            let step = driver
                .step(&mut |event| {
                    if spinner_active {
                        spinner.finish_and_clear();
                        spinner_active = false;
                        print!("{} ", style("Ferrule>").cyan().bold());
                        std::io::stdout()
                            .flush()
                            .map_err(ferrule_common::Error::from)?;
                    }
                    first_token_time.get_or_insert_with(|| turn_start.elapsed());
                    if sampling.verbose_tokens() {
                        eprint!("[{}:{:.4}]", event.token, event.logit.unwrap_or(f32::NAN));
                    }
                    print!("{}", event.text);
                    std::io::stdout()
                        .flush()
                        .map_err(ferrule_common::Error::from)?;
                    turn_tokens = turn_tokens.saturating_add(1);
                    Ok(())
                })
                .await?;
            let elapsed = step_started.elapsed();
            let step_is_idle = matches!(&step, ResidentDriverStep::Idle);
            match step {
                ResidentDriverStep::Executed { action_kind, .. } => match action_kind {
                    ResidentActionKind::Prefill | ResidentActionKind::Mixed => {
                        prefill_time += elapsed;
                    }
                    ResidentActionKind::Decode => decode_time += elapsed,
                    ResidentActionKind::Finish | ResidentActionKind::Cancel => {}
                },
                ResidentDriverStep::WaitingForModelProgress(_) => decode_time += elapsed,
                ResidentDriverStep::Idle => {}
                ResidentDriverStep::Blocked => {
                    if spinner_active {
                        spinner.finish_and_clear();
                    }
                    anyhow::bail!("resident chat driver blocked while running a turn")
                }
            }
            if let Some(terminal) = driver.take_request_terminal(request_id) {
                break require_finished_request(terminal, "resident chat turn")?;
            }
            if step_is_idle {
                anyhow::bail!("resident chat became idle before request terminalization");
            }
        };
        if spinner_active {
            spinner.finish_and_clear();
        }
        println!();
        generated_tokens = generated_tokens.saturating_add(turn_tokens);

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

    driver.shutdown().await?;
    Ok(())
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    use crate::args::{Cli, Command};
    use clap::Parser;
    use ferrule_model::{
        AttentionKind, ModelDescriptor, ModelFamily, MoeSpec, TransformerSemantics,
        TransformerSpec, WeightSource,
    };

    #[test]
    fn parsed_chat_defaults_and_context_reach_requested_effective_report() {
        let config = AutoConfig::from_descriptor(ModelDescriptor {
            path: "metadata-only-no-model".into(),
            spec: TransformerSpec {
                family: ModelFamily::Qwen35,
                architecture: Some("Qwen3_5ForConditionalGeneration".into()),
                weight_source: WeightSource::Safetensors,
                hidden_size: Some(64),
                num_layers: Some(24),
                vocab_size: Some(128),
                num_heads: Some(4),
                num_kv_heads: Some(4),
                head_dim: Some(16),
                attention: AttentionKind::GroupedQuery,
                moe: MoeSpec::none(),
                semantics: TransformerSemantics::default(),
                tensor_count: None,
                quantization: Vec::new(),
                notes: Vec::new(),
            },
            tensor_classes: Vec::new(),
        });
        for flags in [
            vec![],
            vec!["--ctx-size", "4096"],
            vec!["--ctx-size", "8", "--backend", "cpu"],
        ] {
            let mut argv = vec!["ferrule", "chat", "metadata-only-no-model"];
            argv.extend(flags);
            let Command::Chat {
                sampling,
                max_tokens,
                backend,
                chat_template,
                ..
            } = Cli::try_parse_from(argv).unwrap().command
            else {
                panic!("chat parse")
            };
            let generation = sampling.generation_config(max_tokens);
            let options = chat_model_options(&config, &generation, &sampling).unwrap();
            let selection = backend
                .as_deref()
                .map(BackendSelection::parse)
                .transpose()
                .unwrap()
                .unwrap_or_default();
            let plan = ResidentModelPlanner
                .prepare(&config, selection, chat_template.as_deref(), options)
                .unwrap();
            assert_eq!(plan.requested_backend(), selection);
            assert_eq!(
                plan.requested_options().scheduler_config.prefill_chunk_size,
                4096
            );
            assert_eq!(
                plan.requested_options().driver_config.ctx_size,
                generation.ctx_size
            );
            let effective = plan.effective_options();
            assert_eq!(
                effective.scheduler_config.max_batch_tokens,
                32.min(generation.ctx_size)
            );
            assert_eq!(
                effective.scheduler_config.prefill_chunk_size,
                effective.scheduler_config.max_batch_tokens
            );
            assert!(!effective.driver_config.enable_native_proposals);
            assert!(
                plan.adjustment_report()
                    .render()
                    .contains("min(32, context)")
            );
            assert!(
                plan.resolution_report()
                    .contains("owner live admission required")
            );
            assert_eq!(plan.backend_profile(), "cpu-hybrid-f32-qwen35-0.8b");
        }
    }
}
