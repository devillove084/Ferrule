use std::net::IpAddr;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Clone)]
pub(crate) struct GenerationConfig {
    pub(crate) max_new_tokens: usize,
    pub(crate) stop: Vec<String>,
    pub(crate) stop_at_eos: bool,
    pub(crate) append_eos_to_session: bool,
    pub(crate) ctx_size: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 16,
            stop: Vec::new(),
            stop_at_eos: true,
            append_eos_to_session: true,
            ctx_size: 4096,
        }
    }
}

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
    /// Serve an OpenAI-compatible asynchronous HTTP API.
    Serve(ServeArgs),
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

        /// Number of warmup decode tokens before measured turns.
        #[arg(long, default_value_t = 0)]
        warmup_tokens: usize,
        /// Number of DSV4 base layers to execute.
        #[arg(long, default_value_t = 43)]
        max_layers: usize,
        /// Runtime scheduler prefill chunk size.
        #[arg(long = "prefill-chunk-size", default_value_t = 4096)]
        prefill_chunk_size: usize,
        /// lm_head chunk size in rows for full-vocabulary top-1 scans.
        #[arg(long = "output-head-chunk-rows", default_value_t = 4096)]
        output_head_chunk_rows: usize,
        /// Number of routed experts per layer to prefetch during the benchmark.
        #[arg(long = "moe-prefetch-experts", default_value_t = 0)]
        moe_prefetch_experts: usize,
        /// Bound resident routed experts per layer (0 = managed default).
        #[arg(long = "moe-hotset-experts", default_value_t = 48)]
        moe_hotset_experts: usize,
        /// Path to a golden interactive trace JSON for correctness comparison.
        #[arg(long = "golden")]
        golden: Option<String>,
        /// JSON output for machine consumption.
        #[arg(long)]
        json: bool,
        /// Replay one identical decode across independent sequence states to measure
        /// capture, resident-with-head, and resident-body-only target passes.
        #[arg(long)]
        resident_replay: bool,
        /// Run the resident target-verification roofline at V=2/4, the
        /// checkpoint-reference width (`dspark_block_size + 1`), and experimental V=8.
        #[arg(long)]
        verify_width_sweep: bool,
        /// Number of measured all-resident samples per verification width.
        #[arg(long, default_value_t = 3)]
        verify_iterations: usize,
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
        #[arg(long, default_value_t = 32)]
        moe_prefetch_experts: usize,
        /// Bound resident routed experts per layer (0 = managed default).
        #[arg(long, default_value_t = 48)]
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
    /// Compare batched/device prefill vs token-loop append, reporting the first
    /// diverging layer and top-1 token at cut points (1L/5L/23L/43L).
    #[command(name = "deepseek-v4-prefill-parity")]
    DeepSeekV4PrefillParity {
        model: String,
        #[arg(short = 'p', long, default_value = "Hello")]
        prompt: String,
        /// Number of DSV4 base layers to execute (should match the depth under test).
        #[arg(long, default_value_t = 43)]
        max_layers: usize,
        /// Maximum single artifact tensor read size for top-level/layer tensors.
        #[arg(long = "max-tensor-mb", default_value_t = 128)]
        max_tensor_mb: u64,
        /// Maximum single expert artifact read size.
        #[arg(long = "expert-max-slice-mb", default_value_t = 64)]
        expert_reader_max_slice_mb: u64,
        /// Operator backend: cuda or cpu.
        #[arg(long, default_value = "cuda")]
        backend: String,
        /// Wrap --prompt with the official DeepSeek-V4 chat encoding.
        #[arg(long)]
        chat: bool,
        /// Absolute tolerance for HC state comparison.
        #[arg(long, default_value_t = 1e-4)]
        atol: f32,
        /// Layer-depth cut points to report top-1 for. Can be repeated.
        #[arg(long = "cut")]
        cuts: Vec<usize>,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Args, Clone)]
pub(crate) struct ServeArgs {
    /// Local Hugging Face model directory.
    pub(crate) model: String,
    /// Public model ID returned by /v1/models and accepted by requests.
    #[arg(long = "served-model-name", default_value = "deepseek-v4")]
    pub(crate) served_model_name: String,
    /// Listening address.
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) host: IpAddr,
    /// Listening TCP port.
    #[arg(long, default_value_t = 8000)]
    pub(crate) port: u16,
    /// Operator backend: cuda or cpu.
    #[arg(long, default_value = "cuda")]
    pub(crate) backend: String,
    /// Override the model-family default chat template.
    #[arg(long = "chat-template")]
    pub(crate) chat_template: Option<String>,
    /// Maximum context tokens retained per active request.
    #[arg(long = "ctx-size", default_value_t = 1024)]
    pub(crate) ctx_size: usize,
    /// Maximum simultaneously resident requests.
    #[arg(long = "max-active-sequences", default_value_t = 4)]
    pub(crate) max_active_sequences: usize,
    /// Maximum prompt tokens processed by one prefill chunk.
    #[arg(long = "prefill-chunk-size", default_value_t = 512)]
    pub(crate) prefill_chunk_size: usize,
    /// Maximum packed prefill plus decode tokens in one scheduler action.
    #[arg(long = "max-batch-tokens", default_value_t = 512)]
    pub(crate) max_batch_tokens: usize,
    /// Hard budget for the preallocated physical CUDA KV data planes in MiB.
    #[arg(long = "kv-cache-mb", default_value_t = 1024)]
    pub(crate) kv_cache_mb: u64,
    /// Bounded requests waiting for model-worker admission.
    #[arg(long = "request-queue-capacity", default_value_t = 256)]
    pub(crate) request_queue_capacity: usize,
    /// Bounded token events buffered independently per request.
    #[arg(long = "event-queue-capacity", default_value_t = 32)]
    pub(crate) event_queue_capacity: usize,
    /// Maximum time in seconds to wait for tokenizer/runtime admission.
    #[arg(long = "admission-timeout-secs", default_value_t = 30)]
    pub(crate) admission_timeout_secs: u64,
    /// Number of DSV4 base layers to execute.
    #[arg(long, default_value_t = 43)]
    pub(crate) max_layers: usize,
    /// lm_head chunk size in rows for full-vocabulary top-1 scans.
    #[arg(long, default_value_t = 4096)]
    pub(crate) output_head_chunk_rows: usize,
    /// Maximum single top-level/layer artifact tensor read size.
    #[arg(long = "max-tensor-mb", default_value_t = 128)]
    pub(crate) max_tensor_mb: u64,
    /// Maximum single expert artifact read size.
    #[arg(long = "expert-max-slice-mb", default_value_t = 64)]
    pub(crate) expert_reader_max_slice_mb: u64,
    /// Number of routed experts per layer to predictively prefetch.
    #[arg(long, default_value_t = 0)]
    pub(crate) moe_prefetch_experts: usize,
    /// Bound resident routed experts per layer (0 = managed default).
    #[arg(long, default_value_t = 96)]
    pub(crate) moe_hotset_experts: usize,
    /// Predicted incremental expert source bytes admitted per scheduler batch in MiB (0 = unbounded).
    #[arg(long = "expert-io-batch-mb", default_value_t = 2693)]
    pub(crate) expert_io_batch_mb: u64,
    /// Maximum whole experts retained in pageable host memory (0 disables retention).
    #[arg(long = "expert-host-cache-entries", default_value_t = 64)]
    pub(crate) expert_host_cache_entries: usize,
    /// Pageable host expert-cache budget in MiB (0 = entry-limited only).
    #[arg(long = "expert-host-cache-mb", default_value_t = 1024)]
    pub(crate) expert_host_cache_mb: u64,
    /// Maximum whole experts retained in pinned host memory (0 disables retention).
    #[arg(long = "expert-pinned-cache-entries", default_value_t = 16)]
    pub(crate) expert_pinned_cache_entries: usize,
    /// Pinned host expert-cache budget in MiB (0 = entry-limited only).
    #[arg(long = "expert-pinned-cache-mb", default_value_t = 256)]
    pub(crate) expert_pinned_cache_mb: u64,
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
    pub(crate) fn supports_fast_greedy(&self) -> bool {
        self.temp <= 0.0 && (self.repeat_penalty - 1.0).abs() < f32::EPSILON && self.logprobs == 0
    }

    pub(crate) fn generation_config(&self, max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: max_tokens,
            stop: self.stop.clone(),
            ctx_size: self.ctx_size,
            ..GenerationConfig::default()
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

    #[test]
    fn serve_defaults_to_the_measured_dspark_hotset() {
        let cli = Cli::try_parse_from(["ferrule", "serve", "model"]).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("serve command was not parsed");
        };
        assert_eq!(args.moe_hotset_experts, 96);
    }
}
