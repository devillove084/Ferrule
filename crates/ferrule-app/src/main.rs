use clap::{Parser, Subcommand};
use ferrule_candle::CandlePolicy;
use ferrule_core::{AppConfig, FerruleError, FerruleResult, SamplingParams, init_observability};
use ferrule_runtime::{EchoToolEnv, FinishReward, run_episode};
use tracing::{info, warn};

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
}

async fn run() -> FerruleResult<()> {
    let cli = Cli::parse();

    let config_path = match &cli.command {
        Command::Doctor { config, .. } => config,
        Command::Rollout { config } => config,
        Command::Train { config } => config,
        Command::Generate { config, .. } => config,
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
                    "root_dir": info.root_dir,
                    "tokenizer_json": info.tokenizer_json,
                    "config_json": info.config_json,
                    "weight_files": info.weight_files,
                    "vocab_size_hint": info.vocab_size_hint,
                    "chat_template": info.chat_template,
                    "dtype": info.dtype,
                    "use_flash_attn": info.use_flash_attn,
                    "use_kv_cache": info.use_kv_cache,
                    "real_backend_loaded": info.real_backend_loaded,
                    "device_spec": info.device_spec,
                    "resolved_device": info.resolved_device,
                    "compiled_backends": info.compiled_backends,
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
            warn!("training loop is not wired yet; next step is buffer + logprobs + sequence loss");
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
