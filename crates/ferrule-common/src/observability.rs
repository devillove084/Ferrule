//! Observability — tracing, structured metrics, and OpenTelemetry integration.
//!
//! Modeled after SGLang's metrics: TTFT, TPOT, cache hit rate, queue depth,
//! GPU memory, scheduler state, and per-request lifecycle tracking.
//!
//! Environment:
//!   FERRULE_LOG=info                    → log filter (env-filter syntax)
//!   FERRULE_LOG_FORMAT=json             → JSON log output
//!   FERRULE_METRICS_INTERVAL=5          → periodic metrics dump interval (seconds)

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;
use std::sync::Mutex;
use std::time::Instant;

// ── Tracing init ───────────────────────────────────────────────────────

/// Initialize tracing subscriber. Call once at program start.
pub fn init_tracing() {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::EnvFilter;

    let env_filter =
        EnvFilter::try_from_env("FERRULE_LOG").unwrap_or_else(|_| EnvFilter::new("info"));

    match std::env::var("FERRULE_LOG_FORMAT").as_deref() {
        Ok("json") => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(tracing_subscriber::fmt::layer().json())
                .init();
        }
        _ => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_target(false)
                        .with_thread_ids(false)
                        .compact(),
                )
                .init();
        }
    }
}

pub fn shutdown() {}

// ═══════════════════════════════════════════════════════════════════════
// Metrics — SGLang-style observability
// ═══════════════════════════════════════════════════════════════════════

/// Global metrics singleton.
pub static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);

pub struct Metrics {
    // ── Token throughput ──────────────────────────────────────────
    pub prompt_tokens: AtomicU64,
    pub generated_tokens: AtomicU64,

    // ── Request lifecycle ─────────────────────────────────────────
    pub total_requests: AtomicU64,
    pub active_requests: AtomicU64,
    pub finished_requests: AtomicU64,

    // ── Latency histograms (approximate via sum+count) ────────────
    ttft_sum_us: AtomicU64, // sum of time-to-first-token in µs
    ttft_count: AtomicU64,
    tpot_sum_us: AtomicU64, // sum of time-per-output-token in µs
    tpot_count: AtomicU64,
    e2e_sum_us: AtomicU64, // sum of end-to-end request latency
    e2e_count: AtomicU64,
    queue_sum_us: AtomicU64, // sum of queue wait time
    queue_count: AtomicU64,

    // ── Cache ─────────────────────────────────────────────────────
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub prefix_hits: AtomicU64,
    pub prefix_misses: AtomicU64,

    // ── Scheduler ─────────────────────────────────────────────────
    pub max_queue_depth: AtomicU64,
    pub max_running: AtomicU64,
    pub preemptions: AtomicU64,

    // ── Memory ────────────────────────────────────────────────────
    gpu_used_bytes: AtomicU64,
    gpu_total_bytes: AtomicU64,

    // ── Timing of last metrics dump ───────────────────────────────
    last_dump: Mutex<Instant>,
}

impl Metrics {
    fn new() -> Self {
        Self {
            prompt_tokens: AtomicU64::new(0),
            generated_tokens: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            finished_requests: AtomicU64::new(0),
            ttft_sum_us: AtomicU64::new(0),
            ttft_count: AtomicU64::new(0),
            tpot_sum_us: AtomicU64::new(0),
            tpot_count: AtomicU64::new(0),
            e2e_sum_us: AtomicU64::new(0),
            e2e_count: AtomicU64::new(0),
            queue_sum_us: AtomicU64::new(0),
            queue_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            prefix_hits: AtomicU64::new(0),
            prefix_misses: AtomicU64::new(0),
            max_queue_depth: AtomicU64::new(0),
            max_running: AtomicU64::new(0),
            preemptions: AtomicU64::new(0),
            gpu_used_bytes: AtomicU64::new(0),
            gpu_total_bytes: AtomicU64::new(0),
            last_dump: Mutex::new(Instant::now()),
        }
    }

    // ── Record methods ──────────────────────────────────────────────

    pub fn record_ttft(&self, us: u64) {
        self.ttft_sum_us.fetch_add(us, Ordering::Relaxed);
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_tpot(&self, us: u64) {
        self.tpot_sum_us.fetch_add(us, Ordering::Relaxed);
        self.tpot_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_e2e_latency(&self, us: u64) {
        self.e2e_sum_us.fetch_add(us, Ordering::Relaxed);
        self.e2e_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_queue_time(&self, us: u64) {
        self.queue_sum_us.fetch_add(us, Ordering::Relaxed);
        self.queue_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_gpu_memory(&self, used_bytes: u64, total_bytes: u64) {
        self.gpu_used_bytes.store(used_bytes, Ordering::Relaxed);
        self.gpu_total_bytes.store(total_bytes, Ordering::Relaxed);
    }

    pub fn request_started(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        let active = self.active_requests.fetch_add(1, Ordering::Relaxed) + 1;
        self.max_running.fetch_max(active, Ordering::Relaxed);
    }

    pub fn request_finished(&self) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.finished_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_queue_depth(&self, depth: u64) {
        self.max_queue_depth.fetch_max(depth, Ordering::Relaxed);
    }

    // ── Snapshot ─────────────────────────────────────────────────────

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            generated_tokens: self.generated_tokens.load(Ordering::Relaxed),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            active_requests: self.active_requests.load(Ordering::Relaxed),
            finished_requests: self.finished_requests.load(Ordering::Relaxed),
            avg_ttft_ms: avg(
                self.ttft_sum_us.load(Ordering::Relaxed),
                self.ttft_count.load(Ordering::Relaxed),
            ) / 1000.0,
            avg_tpot_ms: avg(
                self.tpot_sum_us.load(Ordering::Relaxed),
                self.tpot_count.load(Ordering::Relaxed),
            ) / 1000.0,
            avg_e2e_ms: avg(
                self.e2e_sum_us.load(Ordering::Relaxed),
                self.e2e_count.load(Ordering::Relaxed),
            ) / 1000.0,
            avg_queue_ms: avg(
                self.queue_sum_us.load(Ordering::Relaxed),
                self.queue_count.load(Ordering::Relaxed),
            ) / 1000.0,
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            prefix_hits: self.prefix_hits.load(Ordering::Relaxed),
            prefix_misses: self.prefix_misses.load(Ordering::Relaxed),
            max_queue_depth: self.max_queue_depth.load(Ordering::Relaxed),
            max_running: self.max_running.load(Ordering::Relaxed),
            preemptions: self.preemptions.load(Ordering::Relaxed),
            gpu_used_mb: self.gpu_used_bytes.load(Ordering::Relaxed) as f64 / 1_048_576.0,
            gpu_total_mb: self.gpu_total_bytes.load(Ordering::Relaxed) as f64 / 1_048_576.0,
        }
    }

    /// Dump metrics to log if interval has elapsed. Returns true if dumped.
    pub fn maybe_dump(&self) -> bool {
        let interval_secs: u64 = std::env::var("FERRULE_METRICS_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if interval_secs == 0 {
            return false;
        }
        let mut last = self.last_dump.lock().unwrap();
        if last.elapsed().as_secs() >= interval_secs {
            let snap = self.snapshot();
            tracing::info!(target: "ferrule_metrics", "{}", snap);
            *last = Instant::now();
            true
        } else {
            false
        }
    }
}

fn avg(sum: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum as f64 / count as f64
    }
}

// ── Snapshot ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub total_requests: u64,
    pub active_requests: u64,
    pub finished_requests: u64,
    pub avg_ttft_ms: f64,
    pub avg_tpot_ms: f64,
    pub avg_e2e_ms: f64,
    pub avg_queue_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefix_hits: u64,
    pub prefix_misses: u64,
    pub max_queue_depth: u64,
    pub max_running: u64,
    pub preemptions: u64,
    pub gpu_used_mb: f64,
    pub gpu_total_mb: f64,
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cache_hit_pct = if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64 * 100.0
        } else {
            0.0
        };
        let prefix_hit_pct = if self.prefix_hits + self.prefix_misses > 0 {
            self.prefix_hits as f64 / (self.prefix_hits + self.prefix_misses) as f64 * 100.0
        } else {
            0.0
        };

        write!(
            f,
            "req(total={} active={} done={}) tokens(prompt={} gen={}) \
             ttft={:.1}ms tpot={:.1}ms e2e={:.1}ms queue={:.1}ms \
             cache(hit={} miss={} {:.1}%) prefix(hit={} miss={} {:.1}%) \
             sched(queue_max={} run_max={} preempt={}) \
             gpu({:.0}/{:.0}MB)",
            self.total_requests,
            self.active_requests,
            self.finished_requests,
            self.prompt_tokens,
            self.generated_tokens,
            self.avg_ttft_ms,
            self.avg_tpot_ms,
            self.avg_e2e_ms,
            self.avg_queue_ms,
            self.cache_hits,
            self.cache_misses,
            cache_hit_pct,
            self.prefix_hits,
            self.prefix_misses,
            prefix_hit_pct,
            self.max_queue_depth,
            self.max_running,
            self.preemptions,
            self.gpu_used_mb,
            self.gpu_total_mb,
        )
    }
}
