use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ferrule", version = "0.2")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
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
        /// Route the benchmark through the ResidentTopKDriver serving spine.
        #[arg(long = "runtime-driver")]
        runtime_driver: bool,
        /// Number of warmup decode tokens before measured turns.
        #[arg(long, default_value_t = 0)]
        warmup_tokens: usize,
        /// Number of DSV4 base layers to execute.
        #[arg(long, default_value_t = 43)]
        max_layers: usize,
        /// Runtime scheduler prefill chunk size for --runtime-driver.
        #[arg(long = "prefill-chunk-size", default_value_t = 4096)]
        prefill_chunk_size: usize,
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

#[derive(Args, Clone)]
pub(crate) struct SamplingArgs {
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
    pub(crate) fn sampling_config(&self) -> ferrule_runtime::SamplingConfig {
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

    pub(crate) fn generation_config(&self, max_tokens: usize) -> ferrule_runtime::GenerationConfig {
        ferrule_runtime::GenerationConfig {
            max_new_tokens: max_tokens,
            stop: self.stop.clone(),
            logprobs_k: self.logprobs,
            ctx_size: self.ctx_size,
            ..ferrule_runtime::GenerationConfig::default()
        }
    }

    pub(crate) fn verbose_tokens(&self) -> bool {
        self.verbose_tokens
    }
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
