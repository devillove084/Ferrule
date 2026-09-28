//! Process-local tracing and opt-in cumulative metrics; no OTel/exporter integration.
//!
//! Runtime-owned engine/driver snapshots describe one runtime's work and current
//! resources. They are not increments for these process counters: adding full
//! snapshots on each poll would double count. These relaxed atomic snapshots are
//! best-effort observations, never admission or cleanup authority.
//!
//! The CLI calls `init_tracing`; embedded callers retain subscriber ownership.
//! Request counters require an explicit owner-held `RequestMetricsGuard`. The
//! server/worker hook is a separate integration step, not implicitly installed
//! by tracing initialization. No timer, metrics endpoint, or exporter is spawned.
//!
//! Environment:
//!   FERRULE_LOG=info                    → log filter (env-filter syntax)
//!   FERRULE_LOG_FORMAT=json             → JSON log output on stderr
//!   FERRULE_METRICS_INTERVAL=5          → minimum dump interval when polled (seconds)
//! Missing/invalid/zero metrics interval disables dumps.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests;

// ── Tracing init ───────────────────────────────────────────────────────

/// Result of the one process-wide initialization attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracingInitStatus {
    Installed,
    /// Another subscriber/logging owner was already installed; it is retained.
    AlreadyInstalled,
}

static TRACING_INIT: OnceLock<TracingInitStatus> = OnceLock::new();

/// Compatibility entry point. Safe to call repeatedly or concurrently.
pub fn init_tracing() {
    let _ = init_tracing_once();
}

/// Initialize once and return the cached result; environment is read only once.
pub fn init_tracing_once() -> TracingInitStatus {
    *TRACING_INIT.get_or_init(install_tracing)
}

fn install_tracing() -> TracingInitStatus {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let env_filter =
        EnvFilter::try_from_env("FERRULE_LOG").unwrap_or_else(|_| EnvFilter::new("info"));

    let result = match std::env::var("FERRULE_LOG_FORMAT").as_deref() {
        Ok("json") => tracing_subscriber::registry()
            .with(env_filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(std::io::stderr)
                    .json(),
            )
            .try_init(),
        _ => tracing_subscriber::registry()
            .with(env_filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(std::io::stderr)
                    .with_target(false)
                    .with_thread_ids(false)
                    .compact(),
            )
            .try_init(),
    };
    if result.is_ok() {
        TracingInitStatus::Installed
    } else {
        TracingInitStatus::AlreadyInstalled
    }
}

/// Compatibility no-op: there is no exporter, timer, or owned buffer to flush.
/// Does not reset counters, finish requests, or uninstall the global subscriber.
pub fn shutdown() {}

// ═══════════════════════════════════════════════════════════════════════
// Metrics — SGLang-style observability
// ═══════════════════════════════════════════════════════════════════════

/// Process-local singleton, cumulative from first use; not a runtime snapshot.
pub static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);

/// Explicitly recorded process counters, gauges, high-water marks and averages.
/// Public atomic fields remain for compatibility. Prefer `start_request()` for
/// paired accounting; do not mix raw lifecycle writes with guards.
pub struct Metrics {
    // ── Token throughput ──────────────────────────────────────────
    pub prompt_tokens: AtomicU64,
    pub generated_tokens: AtomicU64,

    // ── Request lifecycle ─────────────────────────────────────────
    pub total_requests: AtomicU64,
    pub active_requests: AtomicU64,
    pub finished_requests: AtomicU64,

    // ── Latency averages (sum/count, not histograms or percentiles) ──
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

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// An isolated accumulator, useful for embedded owners and deterministic tests.
    pub fn new() -> Self {
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

    /// Start one admitted generation's observation lifetime. Move the guard to
    /// the sole request owner; drop/finish ends accounting, not physical cleanup.
    pub fn start_request(&self) -> RequestMetricsGuard<'_> {
        self.request_started();
        RequestMetricsGuard {
            metrics: self,
            active: true,
        }
    }

    /// Legacy unpaired hook. New callers should retain a request guard instead.
    pub fn request_started(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        let active = self.active_requests.fetch_add(1, Ordering::Relaxed) + 1;
        self.max_running.fetch_max(active, Ordering::Relaxed);
    }

    /// Legacy unpaired hook; ignores unmatched finishes rather than underflowing.
    /// It cannot identify duplicate request IDs. Use a guard for exactly-once
    /// accounting and never call this alongside a guard for the same request.
    pub fn request_finished(&self) {
        if self
            .active_requests
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |active| {
                active.checked_sub(1)
            })
            .is_ok()
        {
            self.finished_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn update_queue_depth(&self, depth: u64) {
        self.max_queue_depth.fetch_max(depth, Ordering::Relaxed);
    }

    // ── Snapshot ─────────────────────────────────────────────────────

    /// Read-only, non-transactional observation; concurrent updates may be mixed.
    /// Never use it to decide readiness, capacity, cancellation, or quiescence.
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

    /// Poll an optional log dump. Missing/invalid/zero interval disables it.
    /// No counters are reset and no background task is created.
    pub fn maybe_dump(&self) -> bool {
        let interval_secs: u64 = std::env::var("FERRULE_METRICS_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        self.maybe_dump_every(Duration::from_secs(interval_secs))
    }

    /// Explicit dump boundary for callers that already own a polling loop.
    /// True means a log event was attempted (the subscriber may filter it).
    /// Only the throttle timestamp changes. Contended/poisoned locks skip output.
    pub fn maybe_dump_every(&self, interval: Duration) -> bool {
        if interval.is_zero() {
            return false;
        }
        let Ok(mut last) = self.last_dump.try_lock() else {
            return false;
        };
        if last.elapsed() >= interval {
            let snap = self.snapshot();
            tracing::info!(target: "ferrule_metrics", "{}", snap);
            *last = Instant::now();
            true
        } else {
            false
        }
    }
}

/// Non-cloneable accounting token. Drop is a fallback for early returns/unwind;
/// it is NOT evidence of successful completion, permit release, or quiescence.
/// `finished_requests` counts ended observation lifetimes, including failures.
#[must_use = "retain the guard for the request observation lifetime"]
pub struct RequestMetricsGuard<'a> {
    metrics: &'a Metrics,
    active: bool,
}

impl RequestMetricsGuard<'_> {
    /// End at most once. Repeated finish and subsequent drop are no-ops.
    pub fn finish(&mut self) -> bool {
        if !std::mem::replace(&mut self.active, false) {
            return false;
        }
        self.metrics.request_finished();
        true
    }
}

impl Drop for RequestMetricsGuard<'_> {
    fn drop(&mut self) {
        self.finish();
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

/// Process accumulator view. Counts/tokens/cache/preemptions are cumulative;
/// active requests and GPU memory are gauges (GPU values are last-writer, not
/// an implicit multi-device sum); queue/running maxima are high-water marks.
/// Latencies are lifetime arithmetic means of explicitly recorded samples.
#[derive(Debug, Clone, Default, PartialEq)]
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
        let cache_total = self.cache_hits as f64 + self.cache_misses as f64;
        let cache_hit_pct = if cache_total > 0.0 {
            self.cache_hits as f64 / cache_total * 100.0
        } else {
            0.0
        };
        let prefix_total = self.prefix_hits as f64 + self.prefix_misses as f64;
        let prefix_hit_pct = if prefix_total > 0.0 {
            self.prefix_hits as f64 / prefix_total * 100.0
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
