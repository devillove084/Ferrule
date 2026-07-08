use clap::{Args, Parser, Subcommand};

mod bench;
mod commands;

use commands::bench_interactive::cmd_bench_interactive;
use commands::chat::cmd_chat;
use commands::info::cmd_info;
use commands::inspect::{
    cmd_deepseek_v4_generate, cmd_deepseek_v4_probe, cmd_expert_stream_smoke,
    cmd_inspect_weightpack,
};

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
    /// Verify CUDA and benchmark GEMV.
    Cuda,
    /// Interactive chat REPL.
    Chat {
        model: String,
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: usize,
        #[arg(short = 'q', long, default_value = "cuda")]
        quant: String,
        #[command(flatten)]
        sampling: SamplingArgs,
        /// Override auto-detected chat template.
        #[arg(long = "chat-template")]
        chat_template: Option<String>,
    },
    /// Benchmark multi-turn interactive chat latency.
    #[command(name = "bench-interactive")]
    BenchInteractive {
        model: String,
        /// Prompts to feed, one per turn. Can be repeated.
        #[arg(short = 'p', long = "prompt", default_value = "Hello")]
        prompts: Vec<String>,
        /// Max new tokens per turn.
        #[arg(short = 'n', long = "max-tokens", default_value_t = 1)]
        max_tokens: usize,
        /// Chat template name (e.g. deepseek-v4).
        #[arg(long = "chat-template")]
        chat_template: Option<String>,
        /// Path to a golden interactive trace JSON for correctness comparison.
        #[arg(long = "golden")]
        golden: Option<String>,
        /// JSON output for machine consumption.
        #[arg(long)]
        json: bool,
    },
    /// Inspect a WeightPack file header.
    #[command(name = "inspect-weightpack")]
    InspectWeightPack { path: String },
    /// Smoke-test artifact-preserving expert streaming from local HF shards.
    #[command(name = "expert-stream-smoke")]
    ExpertStreamSmoke {
        model: String,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, default_value_t = 0)]
        expert: usize,
        #[arg(long = "max-slice-mb", default_value_t = 64)]
        max_slice_mb: u64,
    },
    /// Generate greedily from real local DeepSeek-V4 HF shards.
    #[command(name = "deepseek-v4-generate")]
    DeepSeekV4Generate {
        model: String,
        #[arg(short = 'p', long, default_value = "Hello")]
        prompt: String,
        /// Number of new tokens to generate greedily.
        #[arg(short = 'n', long = "max-tokens", default_value_t = 4)]
        max_tokens: usize,
        /// Number of DSV4 base layers to execute.
        #[arg(long, default_value_t = 43)]
        max_layers: usize,
        /// lm_head chunk size in rows for full-vocab top-1 scans.
        #[arg(long, default_value_t = 4096)]
        output_head_chunk_rows: usize,
        /// Maximum single artifact tensor read size for top-level/layer tensors.
        #[arg(long = "max-tensor-mb", default_value_t = 128)]
        max_tensor_mb: u64,
        /// Maximum single expert artifact read size.
        #[arg(long = "expert-max-slice-mb", default_value_t = 64)]
        expert_reader_max_slice_mb: u64,
        /// Operator backend: cuda or cpu.
        #[arg(long, default_value = "cuda")]
        backend: String,
        /// Do not stop when eos_token_id is generated.
        #[arg(long)]
        no_stop_eos: bool,
        /// Print generated token ids/logits to stderr.
        #[arg(long)]
        verbose_tokens: bool,
        /// Wrap --prompt with the official DeepSeek-V4 chat encoding.
        #[arg(long)]
        chat: bool,
        /// Emit machine-readable benchmark counters instead of streamed text.
        #[arg(long)]
        json: bool,
        /// Number of warmup decode tokens before timing.
        #[arg(long, default_value_t = 0)]
        warmup_tokens: usize,
        /// Number of routed experts per layer to predictively prefetch.
        #[arg(long, default_value_t = 0)]
        moe_prefetch_experts: usize,
        /// Bound resident routed experts per layer (0 = managed default).
        #[arg(long, default_value_t = 0)]
        moe_hotset_experts: usize,
    },
    /// Probe real local DeepSeek-V4 HF shards through the DSV4-specific reference path.
    #[command(name = "deepseek-v4-probe")]
    DeepSeekV4Probe {
        model: String,
        #[arg(short = 'p', long, default_value = "Hello")]
        prompt: String,
        /// Number of DSV4 base layers to execute. Use 0 for fast top-level IO smoke.
        #[arg(long, default_value_t = 0)]
        max_layers: usize,
        /// First lm_head row to print when not using --full-vocab-topk.
        #[arg(long, default_value_t = 0)]
        start_row: usize,
        /// Number of lm_head rows to print when not using --full-vocab-topk.
        #[arg(long, default_value_t = 16)]
        row_count: usize,
        /// Top-K logits to print.
        #[arg(long, default_value_t = 8)]
        top_k: usize,
        /// Scan all lm_head rows in chunks and print full-vocab top-K.
        #[arg(long)]
        full_vocab_topk: bool,
        /// lm_head chunk size in rows for full-vocab logits/top-K scans.
        #[arg(long, default_value_t = 1024)]
        output_head_chunk_rows: usize,
        /// Maximum single artifact tensor read size for top-level/layer tensors.
        #[arg(long = "max-tensor-mb", default_value_t = 128)]
        max_tensor_mb: u64,
        /// Maximum single expert artifact read size.
        #[arg(long = "expert-max-slice-mb", default_value_t = 64)]
        expert_reader_max_slice_mb: u64,
        /// Operator backend: cpu or cuda.
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Optional official/reference JSON to compare prompt tokens and logits against.
        #[arg(long = "reference-json")]
        reference_json: Option<String>,
        /// Absolute tolerance for --reference-json logit comparisons.
        #[arg(long = "reference-atol", default_value_t = 1e-3)]
        reference_atol: f32,
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
    fn sampling_config(&self) -> ferrule_runtime::SamplingConfig {
        ferrule_runtime::SamplingConfig {
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            seed: self.seed,
        }
    }

    fn generation_config(&self, max_tokens: usize) -> ferrule_runtime::GenerationConfig {
        ferrule_runtime::GenerationConfig {
            max_new_tokens: max_tokens,
            stop: self.stop.clone(),
            logprobs_k: self.logprobs,
            ctx_size: self.ctx_size,
            ..ferrule_runtime::GenerationConfig::default()
        }
    }

    fn verbose_tokens(&self) -> bool {
        self.verbose_tokens
    }
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    ferrule_common::observability::init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Info { model } => cmd_info(&model),
        Command::Cuda => cmd_cuda(),
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
        Command::BenchInteractive {
            model,
            prompts,
            max_tokens,
            chat_template,
            golden,
            json,
        } => cmd_bench_interactive(
            &model,
            &prompts,
            max_tokens,
            chat_template.as_deref(),
            golden.as_deref(),
            json,
        ),
        Command::InspectWeightPack { path } => cmd_inspect_weightpack(&path),
        Command::ExpertStreamSmoke {
            model,
            layer,
            expert,
            max_slice_mb,
        } => cmd_expert_stream_smoke(&model, layer, expert, max_slice_mb),
        Command::DeepSeekV4Generate {
            model,
            prompt,
            max_tokens,
            max_layers,
            output_head_chunk_rows,
            max_tensor_mb,
            expert_reader_max_slice_mb,
            backend,
            no_stop_eos,
            verbose_tokens,
            chat,
            json,
            warmup_tokens,
            moe_prefetch_experts,
            moe_hotset_experts,
        } => cmd_deepseek_v4_generate(
            &model,
            &prompt,
            max_tokens,
            max_layers,
            output_head_chunk_rows,
            max_tensor_mb,
            expert_reader_max_slice_mb,
            &backend,
            !no_stop_eos,
            verbose_tokens,
            chat,
            json,
            warmup_tokens,
            moe_prefetch_experts,
            moe_hotset_experts,
        ),
        Command::DeepSeekV4Probe {
            model,
            prompt,
            max_layers,
            start_row,
            row_count,
            top_k,
            full_vocab_topk,
            output_head_chunk_rows,
            max_tensor_mb,
            expert_reader_max_slice_mb,
            backend,
            reference_json,
            reference_atol,
        } => cmd_deepseek_v4_probe(
            &model,
            &prompt,
            max_layers,
            start_row,
            row_count,
            top_k,
            full_vocab_topk,
            output_head_chunk_rows,
            max_tensor_mb,
            expert_reader_max_slice_mb,
            &backend,
            reference_json.as_deref(),
            reference_atol,
        ),
    }
}

// ── cuda ───────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn cmd_cuda() -> anyhow::Result<()> {
    println!("=== CUDA Probe ===");
    ferrule_cuda::cuda_probe()?;

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
        unsafe {
            module.gemv_f32(
                &s,
                LaunchConfig::for_num_elems(d as u32),
                &xd,
                &wd,
                &mut yd,
                d as u32,
                d as u32,
            )
        }
        .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }

    let t0 = std::time::Instant::now();
    let n_iter = 2000;
    for _ in 0..n_iter {
        unsafe {
            module.gemv_f32(
                &s,
                LaunchConfig::for_num_elems(d as u32),
                &xd,
                &wd,
                &mut yd,
                d as u32,
                d as u32,
            )
        }
        .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

    let mut rms_buf =
        DeviceBuffer::<f32>::zeroed(&s, 1).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let dummy = DeviceBuffer::<f32>::zeroed(&s, 1).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let t0 = std::time::Instant::now();
    let n_empty = 5000;
    for _ in 0..n_empty {
        unsafe {
            module.compute_rms(
                &s,
                LaunchConfig::for_num_elems(1u32),
                &dummy,
                &mut rms_buf,
                1u32,
                1e-5f32,
            )
        }
        .map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    }
    let empty_us = t0.elapsed().as_secs_f64() * 1e6 / n_empty as f64;

    let hidden_buf =
        DeviceBuffer::<f32>::zeroed(&s, d).map_err(|e| anyhow::anyhow!("CUDA {e:?}"))?;
    let t0 = std::time::Instant::now();
    let n_rms = 1000;
    for _ in 0..n_rms {
        unsafe {
            module.compute_rms(
                &s,
                LaunchConfig::for_num_elems(1u32),
                &hidden_buf,
                &mut rms_buf,
                d as u32,
                1e-5f32,
            )
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_arguments_are_unique() {
        Cli::command().debug_assert();
    }
}
