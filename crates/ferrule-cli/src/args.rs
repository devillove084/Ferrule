use std::net::IpAddr;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone)]
pub(crate) struct GenerationConfig {
    pub(crate) max_new_tokens: usize,
    pub(crate) stop: Vec<String>,
    pub(crate) stop_at_eos: bool,
    pub(crate) ctx_size: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 16,
            stop: Vec::new(),
            stop_at_eos: true,
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
        #[command(flatten)]
        sampling: SamplingArgs,
        /// Override the model execution backend (cpu or cuda).
        #[arg(long)]
        backend: Option<String>,
        /// Override auto-detected chat template.
        #[arg(long = "chat-template")]
        chat_template: Option<String>,
    },
    /// Serve an OpenAI-compatible asynchronous HTTP API.
    Serve(ServeArgs),
    /// Benchmark multi-turn chat latency and runtime-owned materialization critical paths.
    #[command(name = "bench-interactive")]
    BenchInteractive {
        model: String,
        /// Prompts to feed, one per turn. Can be repeated.
        #[arg(short = 'p', long = "prompt", default_value = "Hello")]
        prompts: Vec<String>,
        /// Max new tokens per turn.
        #[arg(short = 'n', long = "max-tokens", default_value_t = 1)]
        max_tokens: usize,
        /// Chat template name.
        #[arg(long = "chat-template")]
        chat_template: Option<String>,

        /// Number of warmup decode tokens before measured turns.
        #[arg(long, default_value_t = 0)]
        warmup_tokens: usize,
        /// Maximum number of model layers to execute.
        #[arg(long, default_value_t = 43)]
        max_layers: usize,
        /// Runtime scheduler prefill chunk size.
        #[arg(long = "prefill-chunk-size", default_value_t = 4096)]
        prefill_chunk_size: usize,
        /// lm_head chunk size in rows for full-vocabulary top-1 scans.
        #[arg(long = "output-head-chunk-rows", default_value_t = 4096)]
        output_head_chunk_rows: usize,

        /// Routed-expert slots per layer (0 = automatic device-budget planning).
        #[arg(long = "moe-hotset-experts", default_value_t = 0)]
        moe_hotset_experts: usize,
        /// Path to a golden interactive trace JSON for correctness comparison.
        #[arg(long = "golden")]
        golden: Option<String>,
        /// Emit the versioned interactive benchmark JSON schema (currently v2).
        #[arg(long)]
        json: bool,
    },

    /// Benchmark CUDA F32 linear DP/TP using production executors (host-to-host).
    #[command(name = "bench-parallel")]
    BenchParallel(BenchParallelArgs),

    /// Private framed rank-worker protocol endpoint, not a user benchmark.
    #[command(name = "__rank-worker", hide = true)]
    RankWorker(RankWorkerArgs),

    /// Inspect a WeightPack file header.
    #[command(name = "inspect-weightpack")]
    InspectWeightPack { path: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum BenchParallelMode {
    Dp,
    TpColumn,
    TpRow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum RankBackend {
    Thread,
    Process,
}

#[derive(Debug, Args, Clone)]
pub(crate) struct RankWorkerArgs {
    /// Trusted child-local protocol ceiling; limited by the production wire API.
    #[arg(long)]
    pub(crate) max_frame_bytes: usize,
    /// Finite idle pipe deadline; parent independently bounds device work.
    #[arg(long)]
    pub(crate) io_timeout_ms: u64,
}

#[derive(Debug, Args, Clone)]
pub(crate) struct BenchParallelArgs {
    #[arg(long, value_enum)]
    pub(crate) mode: BenchParallelMode,
    /// Number of CUDA ranks. DP processes this many full batches per iteration.
    #[arg(long)]
    pub(crate) ranks: usize,
    /// Rank isolation backend. Process mode creates CUDA only inside child Boot.
    #[arg(long, value_enum, default_value = "thread")]
    pub(crate) rank_backend: RankBackend,
    /// Process startup/command/shutdown deadline in ms; does not bound thread-mode CUDA.
    #[arg(long, default_value_t = 30000)]
    pub(crate) rank_timeout_ms: u64,
    /// Reserved replay budget. Only 0 is supported: no automatic epoch retry or replay.
    #[arg(long, default_value_t = 0)]
    pub(crate) rank_restarts: usize,
    /// Optional shape overrides; each must match the fixture.
    #[arg(long)]
    pub(crate) rows: Option<usize>,
    #[arg(long)]
    pub(crate) in_features: Option<usize>,
    #[arg(long)]
    pub(crate) out_features: Option<usize>,
    #[arg(long, default_value_t = 10)]
    pub(crate) warmup: usize,
    #[arg(long, default_value_t = 100)]
    pub(crate) iterations: usize,
    /// Required JSON: rows, in_features, out_features, weight, input (flat F32 arrays).
    #[arg(long)]
    pub(crate) fixture: PathBuf,
    /// Comma-separated visible CUDA ordinals in rank order; defaults to 0..ranks.
    #[arg(long, value_delimiter = ',')]
    pub(crate) devices: Option<Vec<usize>>,
    /// Emit only the versioned JSON report on stdout; diagnostics use stderr.
    #[arg(long)]
    pub(crate) json: bool,
    /// Include every output element from the last iteration for every rank.
    #[arg(long)]
    pub(crate) emit_values: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServeEngine {
    Auto,
    Resident,
    Pipeline,
}

/// Not pageable/pinned host retention and not per-layer hotset preloading.
#[derive(Debug, Args, Clone, Default)]
pub(crate) struct CudaExpertDeviceArgs {
    /// Qwen3.5 FP8 CUDA cache entries across all layers (default: 1024).
    #[arg(long = "cuda-expert-device-entries")]
    pub(crate) entries: Option<usize>,
    /// Qwen3.5 FP8 CUDA cache hard byte cap, INCLUDING 64 MiB scratch (default: 4294967296).
    /// Admission also enforces the model's total device budget and actual free VRAM.
    #[arg(long = "cuda-expert-device-bytes")]
    pub(crate) bytes: Option<usize>,
}

impl CudaExpertDeviceArgs {
    pub(crate) fn capacity_limits(
        &self,
    ) -> Option<ferrule_runtime::engine::model_factory::Qwen35MoeCapacityLimits> {
        if self.entries.is_none() && self.bytes.is_none() {
            return None;
        }
        let mut limits = ferrule_runtime::engine::model_factory::Qwen35MoeCapacityLimits::default();
        if let Some(entries) = self.entries {
            limits.max_experts = entries;
        }
        if let Some(bytes) = self.bytes {
            limits.max_bytes = bytes;
        }
        Some(limits)
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum ExpertPrewarmArg {
    Full,
    Lazy,
}

#[derive(Debug, Args, Clone, Default)]
pub(crate) struct ExpertPrewarmArgs {
    /// 35B startup: full retains every compressed routed expert before ready;
    /// lazy explicitly allows demand NAS reads and disables host retention.
    #[arg(long = "expert-prewarm", value_enum)]
    pub(crate) mode: Option<ExpertPrewarmArg>,
    /// Bounded CPU pread/validation workers for full host prewarm (default: 4, max: 8).
    #[arg(long = "expert-prewarm-workers")]
    pub(crate) workers: Option<usize>,
}

impl ExpertPrewarmArgs {
    pub(crate) fn options(
        &self,
        entries: Option<usize>,
        mb: Option<u64>,
    ) -> anyhow::Result<Option<ferrule_model::transformer::host_experts::HostExpertCacheOptions>>
    {
        use ferrule_model::transformer::host_experts::{ExpertPrewarmMode, HostExpertCacheOptions};
        if self.mode.is_none() && self.workers.is_none() && entries.is_none() && mb.is_none() {
            return Ok(None);
        }
        let mut options = HostExpertCacheOptions::default();
        if let Some(mode) = self.mode {
            options.mode = match mode {
                ExpertPrewarmArg::Full => ExpertPrewarmMode::Full,
                ExpertPrewarmArg::Lazy => ExpertPrewarmMode::Lazy,
            };
        }
        if let Some(workers) = self.workers {
            options.workers = workers;
        }
        if let Some(entries) = entries {
            options.max_experts = entries;
        }
        if let Some(mb) = mb {
            options.max_bytes = mb
                .checked_mul(1 << 20)
                .ok_or_else(|| anyhow::anyhow!("host expert MiB budget overflows bytes"))?;
        }
        options.validate()?;
        Ok(Some(options))
    }
}

fn parse_admission_limit(value: &str) -> Result<usize, String> {
    let limit = value.parse::<usize>().map_err(|error| error.to_string())?;
    if limit == 0 || limit > isize::MAX as usize {
        return Err(format!("admission limit must be in 1..={}", isize::MAX));
    }
    Ok(limit)
}

#[derive(Args, Clone)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub(crate) expert_prewarm: ExpertPrewarmArgs,
    #[command(flatten)]
    pub(crate) cuda_expert_device: CudaExpertDeviceArgs,
    /// Local Hugging Face model directory.
    pub(crate) model: String,
    /// Public model ID returned by /v1/models and accepted by requests.
    /// Defaults to the resolved model adapter ID.
    #[arg(long = "served-model-name")]
    pub(crate) served_model_name: Option<String>,
    /// Override the model execution backend (cpu or cuda).
    #[arg(long)]
    pub(crate) backend: Option<String>,
    /// Serial greedy CPU/CUDA PP, MoE EP, or dense CUDA thread PP x TP.
    #[arg(long, value_enum, default_value = "auto")]
    pub(crate) engine: ServeEngine,
    #[arg(long, default_value_t = 1)]
    pub(crate) pipeline_parallel: usize,
    /// Dense CUDA tensor owners per stage (1, 2 or 4); thread only, no EP x TP.
    #[arg(long, default_value_t = 1)]
    pub(crate) tensor_parallel: usize,
    /// Expert owners: Qwen3-MoE pipeline or Qwen3.5 GPU-thread EP (2/4/8); F32.
    #[arg(long, default_value_t = 1)]
    pub(crate) expert_parallel: usize,
    /// Rank isolation. Process uses private child pipes; there is no automatic replay.
    #[arg(long, value_enum, default_value = "thread")]
    pub(crate) rank_backend: RankBackend,
    /// CUDA ordinals: TP uses PP-stage-major then TP-rank order (PP * TP distinct devices).
    /// Otherwise PP owners, then stage-ordered EP owners; only non-TP allows colocation.
    /// Qwen3.5 EP requires an explicit distinct list: root = first, experts = same list.
    /// Other models default to 0..owner-count.
    #[arg(long, value_delimiter = ',')]
    pub(crate) devices: Option<Vec<usize>>,
    /// Process startup/command/shutdown deadline (ms), also applied to expert children.
    /// Thread mode only accepts the default; it cannot preempt a kernel.
    #[arg(long, default_value_t = 30000)]
    pub(crate) rank_timeout_ms: u64,
    /// Reserved replay budget. Only 0 is supported.
    #[arg(long, default_value_t = 0)]
    pub(crate) rank_restarts: usize,
    /// Listening address.
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) host: IpAddr,
    /// Listening TCP port.
    #[arg(long, default_value_t = 8000)]
    pub(crate) port: u16,
    /// Override the model-family default chat template.
    #[arg(long = "chat-template")]
    pub(crate) chat_template: Option<String>,
    /// Maximum context tokens retained per active request.
    #[arg(long = "ctx-size", default_value_t = 1024)]
    pub(crate) ctx_size: usize,
    /// Maximum simultaneously resident requests.
    #[arg(long = "max-active-sequences", default_value_t = 4)]
    pub(crate) max_active_sequences: usize,
    /// Desired number of ready decode requests before dispatching a cohort.
    #[arg(long = "decode-cohort-target", default_value_t = 4)]
    pub(crate) decode_cohort_target: usize,
    /// Maximum prefill decisions used to form a decode cohort before forcing progress.
    #[arg(long = "decode-cohort-max-deferrals", default_value_t = 3)]
    pub(crate) decode_cohort_max_deferrals: usize,
    /// Maximum prompt tokens processed by one prefill chunk.
    #[arg(long = "prefill-chunk-size", default_value_t = 512)]
    pub(crate) prefill_chunk_size: usize,
    /// Maximum packed prefill plus decode tokens in one scheduler action.
    /// Qwen3.5 resolves this to min(requested, 32, context); adjustments are reported.
    #[arg(long = "max-batch-tokens", default_value_t = 512)]
    pub(crate) max_batch_tokens: usize,
    /// Hard budget for the model's physical KV data planes in MiB.
    #[arg(long = "kv-cache-mb", default_value_t = 1024)]
    pub(crate) kv_cache_mb: u64,
    /// Bounded requests waiting for model-worker admission.
    #[arg(long = "request-queue-capacity", default_value_t = 256)]
    pub(crate) request_queue_capacity: usize,
    /// Bounded token events buffered independently per request.
    #[arg(long = "event-queue-capacity", default_value_t = 32)]
    pub(crate) event_queue_capacity: usize,
    /// Maximum server request leases, including responses and pending cleanup.
    #[arg(long, default_value_t = 1024, value_parser = parse_admission_limit)]
    pub(crate) max_inflight_requests: usize,
    /// Aggregate UTF-8 ingress and prompt byte budget (not tokenizer/GPU memory).
    #[arg(long, default_value_t = 64 * 1024 * 1024, value_parser = parse_admission_limit)]
    pub(crate) max_prompt_bytes: usize,
    /// Maximum bytes per HTTP body; aggregate prompt capacity may reject earlier.
    #[arg(long, default_value_t = 16 * 1024 * 1024, value_parser = parse_admission_limit)]
    pub(crate) max_body_bytes: usize,
    /// Override runtime waiting capacity (resident default: 1024).
    #[arg(long, value_parser = parse_admission_limit)]
    pub(crate) runtime_max_waiting_requests: Option<usize>,
    /// Override runtime request identities (resident default: 4096).
    #[arg(long, value_parser = parse_admission_limit)]
    pub(crate) runtime_max_request_identities: Option<usize>,
    /// Override runtime session identities (resident default: 4096).
    #[arg(long, value_parser = parse_admission_limit)]
    pub(crate) runtime_max_session_identities: Option<usize>,
    /// Maximum time in seconds to wait for tokenizer/runtime admission.
    #[arg(long = "admission-timeout-secs", default_value_t = 30)]
    pub(crate) admission_timeout_secs: u64,
    /// Maximum model layers to execute (defaults to the descriptor layer count).
    #[arg(long)]
    pub(crate) max_layers: Option<usize>,
    /// lm_head chunk size in rows for full-vocabulary top-1 scans.
    #[arg(long, default_value_t = 4096)]
    pub(crate) output_head_chunk_rows: usize,
    /// Single tensor materialization limit in MiB (Qwen3.5: 1024, others: 128).
    #[arg(long = "max-tensor-mb")]
    pub(crate) max_tensor_mb: Option<u64>,
    /// Maximum single expert artifact read size.
    #[arg(long = "expert-max-slice-mb", default_value_t = 64)]
    pub(crate) expert_reader_max_slice_mb: u64,

    /// Routed-expert slots per layer (0 = automatic device-budget planning).
    #[arg(long)]
    pub(crate) moe_hotset_experts: Option<usize>,
    /// Maximum whole experts retained in pageable host memory (0 disables retention).
    /// 35B full-prewarm default: 10240; other resident models: 64. Pipeline unsupported.
    #[arg(long = "expert-host-cache-entries")]
    pub(crate) expert_host_cache_entries: Option<usize>,
    /// Pageable host expert-cache budget in MiB (0 = entry-limited only).
    /// 35B default: 40960, including staging/metadata (0 is invalid for full prewarm).
    /// Other resident models: 1024. Pipeline unsupported.
    #[arg(long = "expert-host-cache-mb")]
    pub(crate) expert_host_cache_mb: Option<u64>,
    /// Maximum whole experts retained in pinned host memory (0 disables retention).
    /// Resident default: 16. Explicit cache options are unsupported by pipeline/Qwen3.5.
    #[arg(long = "expert-pinned-cache-entries")]
    pub(crate) expert_pinned_cache_entries: Option<usize>,
    /// Pinned host expert-cache budget in MiB (0 = entry-limited only).
    /// Resident default: 256. Explicit cache options are unsupported by pipeline/Qwen3.5.
    #[arg(long = "expert-pinned-cache-mb")]
    pub(crate) expert_pinned_cache_mb: Option<u64>,
}

#[derive(Args, Clone)]
pub(crate) struct SamplingArgs {
    /// Qwen3.5 GPU-thread expert owners (2/4/8); defaults to single GPU.
    #[arg(long, default_value_t = 1)]
    pub(crate) expert_parallel: usize,
    /// Explicit distinct CUDA ordinals for Qwen3.5 EP; root = first, experts = same list.
    #[arg(long, value_delimiter = ',')]
    pub(crate) devices: Option<Vec<usize>>,
    #[command(flatten)]
    pub(crate) expert_prewarm: ExpertPrewarmArgs,
    /// 35B full host prewarm expert cap (default: 10240), separate from CUDA cache.
    #[arg(long = "expert-host-cache-entries")]
    pub(crate) expert_host_cache_entries: Option<usize>,
    /// 35B pageable compressed host budget in MiB including staging (default: 40960).
    #[arg(long = "expert-host-cache-mb")]
    pub(crate) expert_host_cache_mb: Option<u64>,
    #[command(flatten)]
    pub(crate) cuda_expert_device: CudaExpertDeviceArgs,
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
    fn host_prewarm_flags_have_separate_bounded_semantics_for_chat_and_serve() {
        use ferrule_model::transformer::host_experts::{ExpertPrewarmMode, HostExpertCacheOptions};
        for command in ["chat", "serve"] {
            let cli = Cli::try_parse_from([
                "ferrule",
                command,
                "model",
                "--expert-host-cache-mb",
                "40960",
                "--expert-host-cache-entries",
                "10240",
                "--expert-prewarm-workers",
                "4",
                "--expert-prewarm",
                "full",
            ])
            .unwrap();
            let options = match cli.command {
                Command::Serve(args) => {
                    assert!(args.cuda_expert_device.capacity_limits().is_none());
                    args.expert_prewarm
                        .options(args.expert_host_cache_entries, args.expert_host_cache_mb)
                        .unwrap()
                }
                Command::Chat { sampling, .. } => {
                    assert!(sampling.cuda_expert_device.capacity_limits().is_none());
                    sampling
                        .expert_prewarm
                        .options(
                            sampling.expert_host_cache_entries,
                            sampling.expert_host_cache_mb,
                        )
                        .unwrap()
                }
                _ => panic!("wrong command"),
            };
            assert_eq!(options, Some(HostExpertCacheOptions::default()));
        }
        assert!(
            ExpertPrewarmArgs::default()
                .options(None, None)
                .unwrap()
                .is_none()
        );
        assert!(ExpertPrewarmArgs::default().options(None, Some(0)).is_err());
        assert!(
            ExpertPrewarmArgs::default()
                .options(None, Some(u64::MAX))
                .is_err()
        );
        assert!(
            ExpertPrewarmArgs {
                workers: Some(128),
                ..Default::default()
            }
            .options(None, None)
            .is_err()
        );
        let lazy = ExpertPrewarmArgs {
            mode: Some(ExpertPrewarmArg::Lazy),
            ..Default::default()
        }
        .options(Some(0), Some(0))
        .unwrap()
        .unwrap();
        assert_eq!(lazy.mode, ExpertPrewarmMode::Lazy);
    }

    #[test]
    fn cuda_expert_device_flags_are_shared_by_chat_and_serve_not_host_cache() {
        for command in ["chat", "serve"] {
            let parsed = Cli::try_parse_from([
                "ferrule",
                command,
                "model",
                "--cuda-expert-device-entries",
                "1024",
                "--cuda-expert-device-bytes",
                "4294967296",
            ])
            .unwrap();
            let device = match parsed.command {
                Command::Chat { sampling, .. } => sampling.cuda_expert_device,
                Command::Serve(args) => {
                    assert_eq!(args.expert_host_cache_entries, None);
                    assert_eq!(args.expert_pinned_cache_entries, None);
                    args.cuda_expert_device
                }
                _ => unreachable!(),
            };
            let limits = device.capacity_limits().unwrap();
            assert_eq!(
                (limits.max_experts, limits.max_bytes, limits.scratch_bytes),
                (1024, 4usize << 30, 64 << 20)
            );
        }
        assert!(CudaExpertDeviceArgs::default().capacity_limits().is_none());
    }

    #[test]
    fn qwen35_ep_flags_are_explicit_for_chat_and_serve() {
        for command in ["chat", "serve"] {
            let defaults = Cli::try_parse_from(["ferrule", command, "model"]).unwrap();
            let (degree, devices) = match defaults.command {
                Command::Chat { sampling, .. } => (sampling.expert_parallel, sampling.devices),
                Command::Serve(args) => (args.expert_parallel, args.devices),
                _ => unreachable!(),
            };
            assert_eq!(degree, 1);
            assert_eq!(devices, None);
            for (degree, list) in [(2, "3,1"), (4, "3,2,1,0"), (8, "0,1,2,3,4,5,6,7")] {
                let parsed = Cli::try_parse_from([
                    "ferrule",
                    command,
                    "model",
                    "--expert-parallel",
                    &degree.to_string(),
                    "--devices",
                    list,
                ])
                .unwrap();
                let (actual, devices) = match parsed.command {
                    Command::Chat { sampling, .. } => (sampling.expert_parallel, sampling.devices),
                    Command::Serve(args) => (args.expert_parallel, args.devices),
                    _ => unreachable!(),
                };
                assert_eq!(actual, degree);
                assert_eq!(
                    devices.unwrap(),
                    list.split(',')
                        .map(|v| v.parse::<usize>().unwrap())
                        .collect::<Vec<_>>()
                );
            }
            for invalid in ["-1", "not-a-device", "0,,1"] {
                assert!(
                    Cli::try_parse_from([
                        "ferrule",
                        command,
                        "model",
                        "--expert-parallel",
                        "2",
                        "--devices",
                        invalid
                    ])
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn cli_arguments_are_unique() {
        Cli::command().debug_assert();
    }

    #[test]
    fn cli_exposes_only_supported_commands() {
        let mut command = Cli::command();
        let command_names = command
            .get_subcommands()
            .filter(|subcommand| {
                subcommand.get_name() != "help" && subcommand.get_name() != "__rank-worker"
            })
            .map(|subcommand| subcommand.get_name().to_owned())
            .collect::<Vec<_>>();

        assert_eq!(
            command_names,
            [
                "info",
                "cuda",
                "chat",
                "serve",
                "bench-interactive",
                "bench-parallel",
                "inspect-weightpack",
            ]
        );

        let help = command.render_long_help().to_string();
        assert!(help.contains("inspect-weightpack"));
        assert!(!help.contains("expert-stream-smoke"));
        assert!(!help.contains("deepseek-v4-generate"));
    }

    #[test]
    fn removed_inspect_commands_are_rejected() {
        for command in ["expert-stream-smoke", "deepseek-v4-generate"] {
            assert!(Cli::try_parse_from(["ferrule", command]).is_err());
        }
    }

    #[test]
    fn bench_parallel_parses_modes_defaults_and_explicit_options() {
        for (mode, expected) in [
            ("dp", BenchParallelMode::Dp),
            ("tp-column", BenchParallelMode::TpColumn),
            ("tp-row", BenchParallelMode::TpRow),
        ] {
            let cli = Cli::try_parse_from([
                "ferrule",
                "bench-parallel",
                "--mode",
                mode,
                "--ranks",
                "2",
                "--fixture",
                "input.json",
            ])
            .unwrap();
            let Command::BenchParallel(args) = cli.command else {
                panic!("wrong command")
            };
            assert_eq!(args.mode, expected);
            assert_eq!(args.ranks, 2);
            assert_eq!(args.rows, None);
            assert_eq!(args.in_features, None);
            assert_eq!(args.out_features, None);
            assert_eq!(args.devices, None);
            assert_eq!(args.warmup, 10);
            assert_eq!(args.iterations, 100);
            assert!(!args.json && !args.emit_values);
            assert_eq!(args.rank_backend, RankBackend::Thread);
            assert_eq!(args.rank_timeout_ms, 30_000);
            assert_eq!(args.rank_restarts, 0);
        }
        let cli = Cli::try_parse_from([
            "ferrule",
            "bench-parallel",
            "--mode",
            "tp-row",
            "--ranks",
            "2",
            "--fixture",
            "input.json",
            "--devices",
            "3,1",
            "--rows",
            "7",
            "--in-features",
            "19",
            "--out-features",
            "17",
            "--warmup",
            "0",
            "--iterations",
            "4",
            "--json",
            "--emit-values",
        ])
        .unwrap();
        let Command::BenchParallel(args) = cli.command else {
            panic!("wrong command")
        };
        assert_eq!(args.devices, Some(vec![3, 1]));
        assert_eq!(args.rows, Some(7));
        assert_eq!(args.in_features, Some(19));
        assert_eq!(args.out_features, Some(17));
        assert_eq!(args.warmup, 0);
        assert_eq!(args.iterations, 4);
        assert!(args.json && args.emit_values);
        assert_eq!(args.rank_backend, RankBackend::Thread);
        assert_eq!(args.rank_timeout_ms, 30_000);
        assert_eq!(args.rank_restarts, 0);
    }

    #[test]
    fn bench_parallel_process_options_and_private_worker_are_hidden() {
        let cli = Cli::try_parse_from([
            "ferrule",
            "bench-parallel",
            "--mode",
            "dp",
            "--ranks",
            "2",
            "--fixture",
            "input.json",
            "--rank-backend",
            "process",
            "--rank-timeout-ms",
            "7",
            "--rank-restarts",
            "0",
        ])
        .unwrap();
        let Command::BenchParallel(args) = cli.command else {
            panic!("wrong command")
        };
        assert_eq!(args.rank_backend, RankBackend::Process);
        assert_eq!(args.rank_timeout_ms, 7);
        assert_eq!(args.rank_restarts, 0);
        let worker = Cli::try_parse_from([
            "ferrule",
            "__rank-worker",
            "--max-frame-bytes",
            "1024",
            "--io-timeout-ms",
            "10",
        ])
        .unwrap();
        let Command::RankWorker(args) = worker.command else {
            panic!("wrong hidden command")
        };
        assert_eq!(args.max_frame_bytes, 1024);
        assert_eq!(args.io_timeout_ms, 10);
    }

    #[test]
    fn bench_parallel_requires_fixture_and_rejects_unknown_modes_and_options() {
        assert!(
            Cli::try_parse_from(["ferrule", "bench-parallel", "--mode", "dp", "--ranks", "2"])
                .is_err()
        );
        for mode in ["cpu", "tp", "TP-ROW"] {
            assert!(
                Cli::try_parse_from([
                    "ferrule",
                    "bench-parallel",
                    "--mode",
                    mode,
                    "--ranks",
                    "2",
                    "--fixture",
                    "input.json"
                ])
                .is_err()
            );
        }
        for extra in ["--backend", "--unknown"] {
            assert!(
                Cli::try_parse_from([
                    "ferrule",
                    "bench-parallel",
                    "--mode",
                    "dp",
                    "--ranks",
                    "2",
                    "--fixture",
                    "input.json",
                    extra,
                    "cpu"
                ])
                .is_err()
            );
        }
        for devices in ["", "0,", "0,-1", "a,1"] {
            assert!(
                Cli::try_parse_from([
                    "ferrule",
                    "bench-parallel",
                    "--mode",
                    "dp",
                    "--ranks",
                    "2",
                    "--fixture",
                    "input.json",
                    "--devices",
                    devices
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn serve_defaults_to_automatic_device_budget_and_model_layers() {
        let cli = Cli::try_parse_from(["ferrule", "serve", "model"]).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("serve command was not parsed");
        };
        assert_eq!(args.moe_hotset_experts, None);
        assert_eq!(args.max_layers, None);
        assert_eq!(args.served_model_name, None);
        assert_eq!(args.backend, None);
    }
}
