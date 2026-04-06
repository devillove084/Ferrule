use candle_nn::Optimizer;
use clap::{Parser, Subcommand};
use ferrule_candle::CandlePolicy;
use ferrule_core::{AppConfig, FerruleError, FerruleResult, SamplingParams, init_observability};
use ferrule_runtime::{
    AgentBaselineMode, AgentGeneration, AgentTask, BaselineMode, CalcTool, EchoToolEnv,
    FinishReward, OnPolicyBuffer, ToolRegistry, batch_reinforce_stats, build_scored_trajectory,
    compute_agent_objectives, load_agent_tasks_jsonl, make_sequence_sample, normalize_answer,
    run_agent_episode, run_agent_episode_with_trace, run_episode,
};
use tracing::info;

#[derive(Debug, Parser)]
#[command(name = "ferrule")]
#[command(about = "Ferrule bootstrap CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Doctor {
        #[arg(long, default_value = "configs/rollout.toml")]
        config: String,

        #[arg(long)]
        prompt: Option<String>,
    },
    Rollout {
        #[arg(long, default_value = "configs/rollout.toml")]
        config: String,
    },
    Train {
        #[arg(long, default_value = "configs/train.toml")]
        config: String,
    },
    Generate {
        #[arg(long, default_value = "configs/rollout.toml")]
        config: String,

        #[arg(long)]
        prompt: String,
    },
    Score {
        #[arg(long, default_value = "configs/rollout.toml")]
        config: String,

        #[arg(long)]
        prompt: String,

        #[arg(long)]
        completion: String,
    },
    AgentDemo {
        #[arg(long, default_value = "configs/agent.toml")]
        config: String,
    },
    AgentTrainStep {
        #[arg(long, default_value = "configs/agent.toml")]
        config: String,
    },
}

fn agent_reward_from_ctx(
    ctx: &ferrule_runtime::EpisodeContext,
    expected_final: &Option<String>,
) -> f32 {
    let mut reward = 0.0;

    let tool_calls = ctx
        .transcript
        .iter()
        .filter(|x| x.starts_with("TOOL_CALL "))
        .count();

    let invalid_actions = ctx
        .transcript
        .iter()
        .filter(|x| x.starts_with("INVALID_ACTION:"))
        .count();

    reward -= 0.05 * tool_calls as f32;
    reward -= 0.20 * invalid_actions as f32;

    for line in &ctx.transcript {
        if let Some(rest) = line.strip_prefix("FINAL: ") {
            match expected_final {
                Some(target) => {
                    if normalize_answer(rest) == *target {
                        reward += 1.0;
                    }
                }
                None => {
                    reward += 1.0;
                }
            }
        }
    }

    reward
}

async fn eval_agent_policy(
    policy: &CandlePolicy,
    tasks: &[AgentTask],
    tools: &ToolRegistry,
    agent_cfg: &ferrule_core::AgentConfig,
    rollout_seed: u64,
) -> FerruleResult<Vec<serde_json::Value>> {
    let mut rows = Vec::new();

    for (episode_idx, task) in tasks.iter().enumerate().take(4) {
        let mut call_idx = 0usize;

        let (_ctx, traj) = run_agent_episode_with_trace(
            &task.initial_observation,
            agent_cfg.max_steps,
            tools,
            |prompt| {
                let params = SamplingParams {
                    seed: rollout_seed + episode_idx as u64 * 100 + call_idx as u64,
                    temperature: agent_cfg.temperature,
                    top_p: agent_cfg.top_p,
                    top_k: agent_cfg.top_k,
                    max_new_tokens: agent_cfg.max_new_tokens,
                    stop_strings: vec!["</ACTION>".to_string()],
                    repeat_penalty: 1.02,
                    repeat_last_n: 64,
                };
                call_idx += 1;

                let out = policy.generate_text_once_with_bias(prompt, &params)?;
                Ok(AgentGeneration {
                    text: out.text,
                    token_ids: out.token_ids,
                })
            },
            |ctx| Ok(agent_reward_from_ctx(ctx, &task.expected_final)),
        )
        .await?;

        rows.push(serde_json::json!({
            "task": task.initial_observation,
            "total_reward": traj.total_reward,
            "finished": traj.finished,
            "steps": traj.steps.len(),
            "last_action": traj.steps.last().map(|s| s.action_text.clone()),
        }));
    }

    Ok(rows)
}

async fn run() -> FerruleResult<()> {
    let cli = Cli::parse();

    let config_path = match &cli.command {
        Command::Doctor { config, .. } => config,
        Command::Rollout { config } => config,
        Command::Train { config } => config,
        Command::Generate { config, .. } => config,
        Command::Score { config, .. } => config,
        Command::AgentDemo { config } => config,
        Command::AgentTrainStep { config } => config,
    };

    let cfg = AppConfig::from_file(config_path)?;
    init_observability(&cfg.observability)?;

    match cli.command {
        Command::Doctor { prompt, .. } => {
            let policy = CandlePolicy::from_config(&cfg.model)?;
            policy.candle_sanity_check()?;

            let info = policy.doctor_info();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "backend": info.backend,
                    "family": info.family,
                    "model_id": info.model_id,
                    "device_spec": info.device_spec,
                    "resolved_device": info.resolved_device,
                    "compiled_backends": info.compiled_backends,
                    "root_dir": info.root_dir,
                    "tokenizer_json": info.tokenizer_json,
                    "config_json": info.config_json,
                    "weight_files": info.weight_files,
                    "vocab_size_hint": info.vocab_size_hint,
                    "chat_template": info.chat_template,
                    "dtype": info.dtype,
                    "use_flash_attn": info.use_flash_attn,
                    "use_kv_cache": info.use_kv_cache,
                    "real_backend_loaded": info.real_backend_loaded
                }))?
            );

            if cfg.model.backend == "real" && cfg.model.family == "llama" {
                let prompt = prompt.unwrap_or_else(|| "Hello from Ferrule.".to_string());
                let ids = policy.encode_text(&prompt, true)?;
                let logits = policy.llama_prefill_logits(&ids)?;
                println!("llama prefill ok, logits shape = {:?}", logits.shape());
            }

            info!("doctor checks passed");
        }
        Command::Rollout { .. } => {
            let policy = CandlePolicy::from_config(&cfg.model)?;
            policy.candle_sanity_check()?;

            let mut env = EchoToolEnv;
            let reward = FinishReward;

            let traj = run_episode(
                &policy,
                &mut env,
                &reward,
                &[1, 2, 3, 4],
                SamplingParams::default(),
                cfg.rollout.max_steps,
                cfg.rollout.seed,
            )
            .await?;

            println!("{}", serde_json::to_string_pretty(&traj)?);
        }

        Command::Train { .. } => {
            if cfg.model.backend != "real" {
                return Err(FerruleError::Config(
                    "train currently requires model.backend = \"real\"".to_string(),
                ));
            }

            let train_cfg = cfg.train.clone().ok_or_else(|| {
                FerruleError::Config("missing [train] section in config".to_string())
            })?;

            let baseline_mode = BaselineMode::parse(&train_cfg.baseline_mode)?;
            let policy = CandlePolicy::from_config(&cfg.model)?;
            policy.candle_sanity_check()?;

            let dataset = if let Some(path) = &train_cfg.dataset_path {
                ferrule_runtime::load_jsonl_dataset(path)?
            } else {
                vec![ferrule_runtime::PromptExample {
                    prompt: train_cfg
                        .prompt
                        .clone()
                        .ok_or_else(|| FerruleError::Config("missing train.prompt".to_string()))?,
                    expected: train_cfg.expected_substring.clone(),
                }]
            };

            let mut buffer = OnPolicyBuffer::new();
            let mut sample_rows = Vec::new();

            for i in 0..train_cfg.num_samples {
                let ex = &dataset[i % dataset.len()];
                let prompt = &ex.prompt;
                let prompt_ids = policy.encode_text(prompt, true)?;

                let params = SamplingParams {
                    seed: cfg.rollout.seed + i as u64,
                    temperature: 0.2,
                    top_p: Some(0.9),
                    top_k: Some(20),
                    max_new_tokens: train_cfg.max_new_tokens,
                    stop_strings: vec![],
                    repeat_penalty: 1.05,
                    repeat_last_n: 32,
                };

                let out = policy.generate_text_once(prompt, &params)?;
                let token_logprobs = policy.score_completion_ids(&prompt_ids, &out.token_ids)?;
                let normalized = ferrule_runtime::normalize_answer(&out.text);

                let reward = match &ex.expected {
                    Some(target) => {
                        if normalized == *target {
                            1.0
                        } else if normalized.starts_with(target) {
                            0.5
                        } else if normalized.contains(target) {
                            0.2
                        } else {
                            0.0
                        }
                    }
                    None => 1.0,
                };

                let sample = make_sequence_sample(
                    prompt_ids,
                    out.token_ids.clone(),
                    out.text.clone(),
                    token_logprobs,
                    reward,
                    out.finish_reason.clone(),
                )?;

                sample_rows.push(serde_json::json!({
                    "idx": i,
                    "prompt": prompt,
                    "expected": ex.expected,
                    "reward": reward,
                    "finish_reason": out.finish_reason,
                    "completion_tokens": out.usage.completion_tokens,
                    "text": out.text,
                    "normalized_text": normalized,
                }));

                buffer.push(sample);
            }

            let stats = buffer.stats();
            let train = batch_reinforce_stats(buffer.samples(), baseline_mode);

            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "buffer": {
                        "num_samples": stats.num_samples,
                        "mean_reward": stats.mean_reward,
                        "mean_completion_tokens": stats.mean_completion_tokens
                    },
                    "train": {
                        "baseline_mode": train_cfg.baseline_mode,
                        "mean_reward": train.mean_reward,
                        "mean_advantage": train.mean_advantage,
                        "mean_logprob_sum": train.mean_logprob_sum,
                        "mean_objective": train.mean_objective
                    },
                    "samples": sample_rows
                }))?
            );
        }

        Command::Generate { prompt, .. } => {
            if cfg.model.backend != "real" {
                return Err(FerruleError::Config(
                    "generate currently requires model.backend = \"real\"".to_string(),
                ));
            }

            let policy = CandlePolicy::from_config(&cfg.model)?;
            policy.candle_sanity_check()?;

            let out = policy.generate_text_once(&prompt, &SamplingParams::default())?;

            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "text": out.text,
                    "finish_reason": out.finish_reason,
                    "token_ids": out.token_ids,
                    "usage": out.usage
                }))?
            );
        }

        Command::Score {
            prompt, completion, ..
        } => {
            if cfg.model.backend != "real" {
                return Err(FerruleError::Config(
                    "score currently requires model.backend = \"real\"".to_string(),
                ));
            }

            let policy = CandlePolicy::from_config(&cfg.model)?;
            policy.candle_sanity_check()?;

            let lps = policy.score_completion(&prompt, &completion)?;
            let sum = lps.iter().copied().sum::<f32>();

            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "prompt": prompt,
                    "completion": completion,
                    "token_logprobs": lps,
                    "logprob_sum": sum
                }))?
            );
        }

        Command::AgentDemo { .. } => {
            let agent_cfg = cfg.agent.clone().ok_or_else(|| {
                FerruleError::Config("missing [agent] section in config".to_string())
            })?;

            let mut tools = ToolRegistry::new();
            tools.register(CalcTool);

            if cfg.model.backend == "mock" {
                let mut calls = 0usize;

                let (ctx, total_reward) = run_agent_episode(
                    &agent_cfg.initial_observation,
                    agent_cfg.max_steps,
                    &tools,
                    |_prompt| {
                        let out = match calls {
                            0 => "<ACTION>\nTOOL: calc\nINPUT: 2+2\n</ACTION>".to_string(),
                            _ => "<ACTION>\nFINAL: 4\n</ACTION>".to_string(),
                        };
                        calls += 1;
                        Ok(out)
                    },
                    |ctx| {
                        let mut reward = 0.0;

                        for line in &ctx.transcript {
                            if line.starts_with("TOOL_CALL ") {
                                reward += 0.1;
                            }

                            if line.starts_with("INVALID_ACTION:") {
                                reward -= 0.2;
                            }

                            if let Some(rest) = line.strip_prefix("FINAL: ") {
                                match &agent_cfg.expected_final {
                                    Some(target) => {
                                        if normalize_answer(rest) == *target {
                                            reward += 1.0;
                                        }
                                    }
                                    None => reward += 1.0,
                                }
                            }
                        }

                        Ok(reward)
                    },
                )
                .await?;

                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "mode": "mock",
                        "status": ctx.status,
                        "steps": ctx.step_idx,
                        "total_reward": total_reward,
                        "transcript": ctx.transcript
                    }))?
                );
            } else {
                let policy = CandlePolicy::from_config(&cfg.model)?;
                policy.candle_sanity_check()?;

                let mut call_idx = 0usize;
                let (ctx, traj) = run_agent_episode_with_trace(
                    &agent_cfg.initial_observation,
                    agent_cfg.max_steps,
                    &tools,
                    |prompt| {
                        let params = SamplingParams {
                            seed: cfg.rollout.seed + call_idx as u64,
                            temperature: agent_cfg.temperature,
                            top_p: agent_cfg.top_p,
                            top_k: agent_cfg.top_k,
                            max_new_tokens: agent_cfg.max_new_tokens,
                            stop_strings: vec!["</ACTION>".to_string()],
                            repeat_penalty: 1.02,
                            repeat_last_n: 64,
                        };
                        call_idx += 1;

                        let out = policy.generate_text_once(prompt, &params)?;
                        Ok(AgentGeneration {
                            text: out.text,
                            token_ids: out.token_ids,
                        })
                    },
                    |ctx| {
                        let mut reward = 0.0;
                        let tool_calls = ctx
                            .transcript
                            .iter()
                            .filter(|x| x.starts_with("TOOL_CALL "))
                            .count();

                        let invalid_actions = ctx
                            .transcript
                            .iter()
                            .filter(|x| x.starts_with("INVALID_ACTION:"))
                            .count();

                        reward -= 0.05 * (tool_calls as f32);
                        reward -= 0.20 * (invalid_actions as f32);

                        for line in &ctx.transcript {
                            if let Some(rest) = line.strip_prefix("FINAL: ") {
                                match &agent_cfg.expected_final {
                                    Some(target) => {
                                        if normalize_answer(rest) == *target {
                                            reward += 1.0;
                                        }
                                    }
                                    None => {
                                        reward += 1.0;
                                    }
                                }
                            }
                        }

                        Ok(reward)
                    },
                )
                .await?;

                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "mode": "real",
                        "status": ctx.status,
                        "steps": ctx.step_idx,
                        "total_reward": traj.total_reward,
                        "transcript": ctx.transcript,
                        "trajectory": traj
                    }))?
                );
            }
        }

        Command::AgentTrainStep { .. } => {
            let agent_cfg = cfg.agent.clone().ok_or_else(|| {
                FerruleError::Config("missing [agent] section in config".to_string())
            })?;

            if cfg.model.backend != "real" {
                return Err(FerruleError::Config(
                    "agent-train-step currently requires model.backend = \"real\"".to_string(),
                ));
            }

            let baseline_mode = AgentBaselineMode::parse(&agent_cfg.baseline_mode)?;
            let policy = CandlePolicy::from_config(&cfg.model)?;
            policy.candle_sanity_check()?;

            let vars = policy.trainable_vars()?;
            let mut opt = candle_nn::optim::AdamW::new_lr(vars, 1e-1)
                .map_err(|e| FerruleError::Model(format!("failed to build AdamW: {e}")))?;

            let mut tools = ToolRegistry::new();
            tools.register(CalcTool);

            let tasks = if let Some(path) = &agent_cfg.dataset_path {
                load_agent_tasks_jsonl(path)?
            } else {
                vec![AgentTask {
                    initial_observation: agent_cfg.initial_observation.clone(),
                    expected_final: agent_cfg.expected_final.clone(),
                }]
            };

            let before =
                eval_agent_policy(&policy, &tasks, &tools, &agent_cfg, cfg.rollout.seed).await?;

            let mut scored = Vec::new();

            for episode_idx in 0..agent_cfg.num_episodes {
                let task = &tasks[episode_idx % tasks.len()];
                let mut call_idx = 0usize;

                let (_ctx, traj) = run_agent_episode_with_trace(
                    &task.initial_observation,
                    agent_cfg.max_steps,
                    &tools,
                    |prompt| {
                        let params = SamplingParams {
                            seed: cfg.rollout.seed + episode_idx as u64 * 100 + call_idx as u64,
                            temperature: agent_cfg.temperature,
                            top_p: agent_cfg.top_p,
                            top_k: agent_cfg.top_k,
                            max_new_tokens: agent_cfg.max_new_tokens,
                            stop_strings: vec!["</ACTION>".to_string()],
                            repeat_penalty: 1.02,
                            repeat_last_n: 64,
                        };
                        call_idx += 1;

                        let out = policy.generate_text_once_with_bias(prompt, &params)?;
                        Ok(AgentGeneration {
                            text: out.text,
                            token_ids: out.token_ids,
                        })
                    },
                    |ctx| Ok(agent_reward_from_ctx(ctx, &task.expected_final)),
                )
                .await?;

                let mut per_step_logprobs = Vec::with_capacity(traj.steps.len());

                for step in &traj.steps {
                    let prompt_ids = policy.encode_text(&step.prompt_text, true)?;
                    let lps = policy
                        .score_completion_ids_with_bias(&prompt_ids, &step.action_token_ids)?;
                    per_step_logprobs.push(lps);
                }

                scored.push(build_scored_trajectory(
                    &traj,
                    per_step_logprobs,
                    agent_cfg.gamma,
                )?);
            }

            let (rows, stats) = compute_agent_objectives(&scored, baseline_mode);

            let mut loss_terms = Vec::new();
            for row in &rows {
                let step = &scored[row.trajectory_idx].steps[row.step_idx];
                let prompt_ids = policy.encode_text(&step.prompt_text, true)?;

                let logprob_sum = policy
                    .differentiable_action_logprob_sum(&prompt_ids, &step.action_token_ids)?;

                let adv =
                    candle_core::Tensor::new(&[-row.advantage], policy.device()).map_err(|e| {
                        FerruleError::Model(format!("failed to build advantage tensor: {e}"))
                    })?;

                let term = logprob_sum.broadcast_mul(&adv).map_err(|e| {
                    FerruleError::Model(format!("failed to multiply loss term: {e}"))
                })?;

                loss_terms.push(term);
            }

            let loss = candle_core::Tensor::stack(&loss_terms, 0)
                .map_err(|e| FerruleError::Model(format!("failed to stack loss terms: {e}")))?
                .mean(0)
                .map_err(|e| FerruleError::Model(format!("failed to mean-reduce loss: {e}")))?;

            opt.backward_step(&loss)
                .map_err(|e| FerruleError::Model(format!("optimizer step failed: {e}")))?;

            let after =
                eval_agent_policy(&policy, &tasks, &tools, &agent_cfg, cfg.rollout.seed).await?;

            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "objective_stats": stats,
                    "eval_before": before,
                    "eval_after": after
                }))?
            );
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("fatal: {err}");
        std::process::exit(1);
    }
}
