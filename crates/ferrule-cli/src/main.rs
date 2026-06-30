use clap::{Args, Parser, Subcommand};
#[cfg(feature = "cuda")]
use ferrule_runtime::{detect_chat_template, InferenceEngine, ModelGenerationDefaults};
use ferrule_runtime::{CpuOlmoeRunner, GenerationConfig, ModelRunner, SamplingConfig};
use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

mod commands;
mod server;

use commands::bench::cmd_bench_infer;
use commands::chat::cmd_chat;
use commands::compare::cmd_compare_logits;
use commands::info::{cmd_info, print_model_info};
use commands::inspect::cmd_inspect_cache;
#[cfg(feature = "cuda")]
use commands::run::parse_quant;
use commands::run::{cmd_gpu_run, cmd_run};

// ── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "ferrule", version = "0.2")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print model architecture and vocabulary size.
    Info { model: String },
    /// Run inference on CPU (FP32).
    Run {
        model: String,
        #[arg(short = 'p', long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(short = 'n', long, default_value = "16")]
        max_tokens: usize,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Verify CUDA and benchmark GEMV.
    Cuda,
    /// Run inference on GPU.
    GpuRun {
        model: String,
        #[arg(short = 'p', long, default_value = "Hello")]
        prompt: String,
        #[arg(short = 'n', long, default_value = "4")]
        max_tokens: usize,
        #[arg(short = 'q', long, default_value = "q4")]
        quant: String,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Interactive chat REPL.
    Chat {
        model: String,
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: usize,
        #[arg(short = 'q', long, default_value = "q4")]
        quant: String,
        #[command(flatten)]
        sampling: SamplingArgs,
        /// Override auto-detected chat template.
        #[arg(long = "chat-template")]
        chat_template: Option<String>,
    },
    /// Benchmark prompt/decode throughput (no model-load timing).
    BenchInfer {
        model: String,
        #[arg(short = 'p', long, default_value = "Hello")]
        prompt: String,
        #[arg(short = 'n', long, default_value = "128")]
        max_tokens: usize,
        #[arg(short = 'q', long, default_value = "q4")]
        quant: String,
        /// Warmup runs before measuring.
        #[arg(long, default_value = "2")]
        warmup: usize,
        /// Repeat count for statistics.
        #[arg(long, default_value = "3")]
        repeat: usize,
        /// JSON output for machine consumption.
        #[arg(long)]
        json: bool,
        /// Context window size for token limit warnings.
        #[arg(long, default_value = "4096")]
        ctx_size: usize,
    },
    /// Compare CPU FP32 vs GPU quantized logits for the same prompt.
    CompareLogits {
        model: String,
        #[arg(short = 'p', long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(short = 'n', long, default_value = "16")]
        max_tokens: usize,
        #[arg(short = 'q', long, default_value = "q4")]
        quant: String,
        /// Allow CPU and GPU to sample independently (no teacher forcing).
        #[arg(long)]
        free_run: bool,
        /// Context window size for token limit warnings.
        #[arg(long, default_value = "4096")]
        ctx_size: usize,
    },
    /// Inspect a qcache file header.
    InspectCache { path: String },
    /// Start a minimal OpenAI-compatible HTTP server.
    Server {
        model: String,
        #[arg(short = 'q', long, default_value = "q4")]
        quant: String,
        /// Host to bind.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on.
        #[arg(long, default_value = "8080")]
        port: u16,
    },
    /// Compute perplexity over a text file (teacher-force, CPU only).
    Perplexity {
        model: String,
        /// Text file to evaluate (one line = one prompt, or freeform).
        #[arg(short = 'f', long)]
        file: String,
        /// Context window size for batching.
        #[arg(long, default_value = "512")]
        ctx_size: usize,
    },
}

// ── Args helpers ───────────────────────────────────────────────────────────

#[derive(Args, Clone)]
struct SamplingArgs {
    /// Sampling temperature. Use 0 for greedy decoding.
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
    /// Keep only the best K tokens before sampling. Use 0 to disable.
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    /// Nucleus sampling probability mass. Use 1.0 to disable.
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    /// Minimum probability relative to the best token. Use 0 to disable.
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    /// Penalize repeated tokens. Use 1.0 to disable.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,
    /// Number of recent tokens considered by repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
    /// Deterministic sampler seed. Use 0 for Ferrule's default seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Stop generation when the decoded text ends with this string.
    #[arg(long = "stop")]
    stop: Vec<String>,
    /// Print top-K logprobs for each generated token.
    #[arg(long, default_value_t = 0)]
    logprobs: usize,
    /// Print each token id alongside its decoded text.
    #[arg(long)]
    verbose_tokens: bool,
    /// Context window size (max tokens for KV cache).
    #[arg(long, default_value = "4096")]
    ctx_size: usize,
}

impl SamplingArgs {
    fn sampling_config(&self) -> SamplingConfig {
        SamplingConfig {
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            seed: self.seed,
        }
    }

    fn generation_config(&self, max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: max_tokens,
            stop: self.stop.clone(),
            logprobs_k: self.logprobs,
            ctx_size: self.ctx_size,
            ..GenerationConfig::default()
        }
    }

    fn verbose_tokens(&self) -> bool {
        self.verbose_tokens
    }
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    ferrule_core::observability::init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Info { model } => cmd_info(&model),
        Command::Run {
            model,
            prompt,
            max_tokens,
            sampling,
        } => cmd_run(&model, &prompt, max_tokens, &sampling),
        Command::Cuda => cmd_cuda(),
        Command::GpuRun {
            model,
            prompt,
            max_tokens,
            quant,
            sampling,
        } => cmd_gpu_run(&model, &prompt, max_tokens, &quant, &sampling),
        Command::Chat {
            model,
            max_tokens,
            quant,
            sampling,
            chat_template,
        } => cmd_chat(
            &model,
            max_tokens,
            &quant,
            &sampling,
            chat_template.as_deref(),
        ),
        Command::BenchInfer {
            model,
            prompt,
            max_tokens,
            quant,
            warmup,
            repeat,
            json,
            ctx_size,
        } => cmd_bench_infer(
            &model, &prompt, max_tokens, &quant, warmup, repeat, json, ctx_size,
        ),
        Command::CompareLogits {
            model,
            prompt,
            max_tokens,
            quant,
            free_run,
            ctx_size,
        } => cmd_compare_logits(&model, &prompt, max_tokens, &quant, free_run, ctx_size),
        Command::InspectCache { path } => cmd_inspect_cache(&path),
        Command::Server {
            model,
            quant,
            host,
            port,
        } => cmd_server(&model, &quant, &host, port),
        Command::Perplexity {
            model,
            file,
            ctx_size,
        } => cmd_perplexity(&model, &file, ctx_size),
    }
}

// ── cuda ───────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn cmd_cuda() -> anyhow::Result<()> {
    println!("=== CUDA Probe ===");
    ferrule_cuda::forward::cuda_probe()?;

    println!("\n=== GEMV Benchmark (2048×2048) ===");
    let d = 2048usize;
    let x: Vec<f32> = (0..d).map(|i| (i as f32).sin()).collect();
    let w: Vec<f32> = (0..d * d).map(|i| (i as f32).cos()).collect();

    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    ctx.bind_to_thread()
        .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let module =
        ferrule_cuda::kernels::kernels::load(&ctx).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let s = ctx.default_stream();
    let xd = DeviceBuffer::from_host(&s, &x).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let wd = DeviceBuffer::from_host(&s, &w).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let mut yd = DeviceBuffer::<f32>::zeroed(&s, d).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;

    for _ in 0..10 {
        module
            .gemv_f32(
                &s,
                LaunchConfig::for_num_elems(d as u32),
                &xd,
                &wd,
                &mut yd,
                d as u32,
            )
            .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }

    let t0 = std::time::Instant::now();
    let n_iter = 2000;
    for _ in 0..n_iter {
        module
            .gemv_f32(
                &s,
                LaunchConfig::for_num_elems(d as u32),
                &xd,
                &wd,
                &mut yd,
                d as u32,
            )
            .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

    let mut rms_buf =
        DeviceBuffer::<f32>::zeroed(&s, 1).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let dummy = DeviceBuffer::<f32>::zeroed(&s, 1).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let t0 = std::time::Instant::now();
    let n_empty = 5000;
    for _ in 0..n_empty {
        module
            .compute_rms(
                &s,
                LaunchConfig::for_num_elems(1u32),
                &dummy,
                &mut rms_buf,
                1u32,
                1e-5f32,
            )
            .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }
    let empty_us = t0.elapsed().as_secs_f64() * 1e6 / n_empty as f64;

    let hidden_buf =
        DeviceBuffer::<f32>::zeroed(&s, d).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let t0 = std::time::Instant::now();
    let n_rms = 1000;
    for _ in 0..n_rms {
        module
            .compute_rms(
                &s,
                LaunchConfig::for_num_elems(1u32),
                &hidden_buf,
                &mut rms_buf,
                d as u32,
                1e-5f32,
            )
            .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }
    let rms_us = t0.elapsed().as_secs_f64() * 1e6 / n_rms as f64;

    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        let mut out = vec![0f32; d];
        for j in 0..d {
            let row = &w[j * d..(j + 1) * d];
            out[j] = row.iter().zip(x.iter()).map(|(r, xi)| r * xi).sum();
        }
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

    println!("  CPU: {cpu_ms:.2} ms");
    println!("  GPU GEMV (kernel only): {gpu_ms:.3} ms");
    println!("  Kernel launch overhead: {empty_us:.0} µs");
    println!("  compute_rms(d=2048): {rms_us:.0} µs");
    println!("  Speedup: {:.0}x", cpu_ms / gpu_ms);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn cmd_cuda() -> anyhow::Result<()> {
    println!("cuda requires --features cuda");
    Ok(())
}

// ── server ─────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn cmd_server(model_dir: &str, quant: &str, host: &str, port: u16) -> anyhow::Result<()> {
    let qt = parse_quant(quant);
    let template = detect_chat_template(Path::new(model_dir));

    tracing::info!("Loading server model once (quant: {qt:?})...");
    let runner = ferrule_runtime::GpuOlmoeRunner::load(Path::new(model_dir), qt)?;
    let mut sc = SamplingConfig::greedy();
    if let Some(def) = ModelGenerationDefaults::load(Path::new(model_dir)) {
        def.apply_to_config(&mut sc);
    }
    let engine = Arc::new(Mutex::new(InferenceEngine::new(runner, sc)));

    let gen_fn: server::GenFn = Box::new({
        let engine = engine.clone();
        move |prompt: &str, max_tokens: usize| {
            let gen_cfg = GenerationConfig {
                max_new_tokens: max_tokens,
                stop: Vec::new(),
                logprobs_k: 0,
                ..GenerationConfig::default()
            };

            let mut engine = engine
                .lock()
                .map_err(|_| anyhow::anyhow!("server engine mutex poisoned"))?;
            let result = engine.generate_text(prompt, &gen_cfg, |_| Ok(()))?;

            Ok((
                result.text,
                result.stats.prompt_tokens,
                result.stats.generated_tokens,
            ))
        }
    });

    let stream_fn: server::StreamingGenFn = Box::new({
        let engine = engine.clone();
        move |prompt: &str, max_tokens: usize, on_token: Box<dyn Fn(&str) + Send>| {
            let gen_cfg = GenerationConfig {
                max_new_tokens: max_tokens,
                stop: Vec::new(),
                logprobs_k: 0,
                ..GenerationConfig::default()
            };

            let mut engine = engine
                .lock()
                .map_err(|_| anyhow::anyhow!("server engine mutex poisoned"))?;
            let result = engine.generate_text(prompt, &gen_cfg, |event| {
                on_token(&event.text);
                Ok(())
            })?;

            Ok((result.stats.prompt_tokens, result.stats.generated_tokens))
        }
    });

    server::run(
        gen_fn,
        Some(stream_fn),
        model_dir.to_string(),
        template,
        host,
        port,
    )
}

#[cfg(not(feature = "cuda"))]
fn cmd_server(_model_dir: &str, _quant: &str, _host: &str, _port: u16) -> anyhow::Result<()> {
    anyhow::bail!("server requires --features cuda")
}

// ── perplexity ──────────────────────────────────────────────────────────────

fn cmd_perplexity(model_dir: &str, file: &str, ctx_size: usize) -> anyhow::Result<()> {
    use ferrule_runtime::perplexity;

    let mut runner = CpuOlmoeRunner::load(Path::new(model_dir))?;
    print_model_info(&runner.model_info());

    let text = std::fs::read_to_string(file).map_err(|e| anyhow::anyhow!("read {file}: {e}"))?;
    let tokens = runner.encode(&text)?;
    println!("File: {file}  tokens: {}", tokens.len());

    // Chunk tokens into context windows
    let total = tokens.len();
    let mut sum_nll = 0.0f64;
    let mut loss_tokens = 0usize;
    let t0 = std::time::Instant::now();

    for chunk_start in (0..total).step_by(ctx_size) {
        let chunk_end = (chunk_start + ctx_size + 1).min(total); // +1 for next-token target
        let chunk = &tokens[chunk_start..chunk_end];
        if chunk.len() < 2 {
            continue;
        }

        runner.reset_session()?;
        let result =
            perplexity::compute_perplexity(chunk, runner.model_info().vocab_size, |token| {
                runner.decode_token(token)
            })?;
        sum_nll += result.sum_nll;
        loss_tokens += result.loss_tokens;
    }

    let dur = t0.elapsed();
    let perplexity = if loss_tokens > 0 {
        (sum_nll / loss_tokens as f64).exp()
    } else {
        f64::INFINITY
    };

    println!(
        "total_tokens={total}  loss_tokens={loss_tokens}  perplexity={perplexity:.2}  duration={:.1}s  tok/s={:.1}",
        dur.as_secs_f64(),
        total as f64 / dur.as_secs_f64().max(1e-6)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_arguments_are_unique() {
        Cli::command().debug_assert();
    }
}
