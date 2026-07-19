//! Kernel-level profiling — count launches, time major regions, and
//! measure arbitrary code blocks.
//!
//! Gated behind `FERRULE_PROFILE=1` env var; zero overhead when disabled.

use std::time::Instant;

/// Regions tracked by the profiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Region {
    Embed,
    AttnProj,
    AttnScore,
    Router,
    ExpertLoop,
    LmHead,
    Other,
}

impl Region {
    pub fn name(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::AttnProj => "attn_proj",
            Self::AttnScore => "attn_score",
            Self::Router => "router",
            Self::ExpertLoop => "expert_loop",
            Self::LmHead => "lm_head",
            Self::Other => "other",
        }
    }
}

/// Lightweight profiler — accumulates per-region time and launch counts.
#[derive(Debug, Clone, Default)]
pub struct Profiler {
    enabled: bool,
    /// Per-region accumulated time in microseconds.
    times: [u64; 7],
    /// Per-region launch count.
    launches: [u64; 7],
    /// Total tokens processed.
    total_tokens: u64,
    /// Current region start time (if timing).
    current: Option<(Region, Instant)>,
}

impl Profiler {
    pub fn new() -> Self {
        let enabled = std::env::var_os("FERRULE_PROFILE").is_some();
        Self {
            enabled,
            ..Default::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Begin timing a region. Nesting is not supported — ends previous region.
    pub fn begin(&mut self, region: Region) {
        if !self.enabled {
            return;
        }
        self.end_current();
        self.current = Some((region, Instant::now()));
    }

    /// End the current region and accumulate time.
    pub fn end(&mut self, region: Region) {
        if !self.enabled {
            return;
        }
        if let Some((cur, t0)) = self.current.take() {
            let elapsed = t0.elapsed().as_micros() as u64;
            self.times[cur as usize] += elapsed;
        }
        self.launches[region as usize] += 1;
    }

    /// Record a launch without timing.
    pub fn launch(&mut self, region: Region) {
        if !self.enabled {
            return;
        }
        self.launches[region as usize] += 1;
    }

    /// Mark one token processed.
    pub fn token(&mut self) {
        self.total_tokens += 1;
    }

    fn end_current(&mut self) {
        if let Some((cur, t0)) = self.current.take() {
            let elapsed = t0.elapsed().as_micros() as u64;
            self.times[cur as usize] += elapsed;
        }
    }

    /// Generate a human-readable report.
    pub fn report(&self) -> String {
        if !self.enabled {
            return "profiler disabled (set FERRULE_PROFILE=1)".into();
        }
        let mut out = String::new();
        use std::fmt::Write;
        let _ = writeln!(out, "=== Profile ({:>3} tokens) ===", self.total_tokens);
        let _ = writeln!(out, "  region        launches   time_ms   pct");
        let total_us: u64 = self.times.iter().sum();
        for (i, region) in [
            Region::Embed,
            Region::AttnProj,
            Region::AttnScore,
            Region::Router,
            Region::ExpertLoop,
            Region::LmHead,
            Region::Other,
        ]
        .iter()
        .enumerate()
        {
            let launches = self.launches[i];
            let us = self.times[i];
            let pct = if total_us > 0 {
                us as f64 / total_us as f64 * 100.0
            } else {
                0.0
            };
            if launches > 0 || us > 0 {
                let _ = writeln!(
                    out,
                    "  {:<12} {:>6} {:>9.1} {:>5.1}%",
                    region.name(),
                    launches,
                    us as f64 / 1000.0,
                    pct
                );
            }
        }
        if total_us > 0 {
            let _ = writeln!(
                out,
                "  total                 {:>9.1}",
                total_us as f64 / 1000.0
            );
            let _ = writeln!(
                out,
                "  avg/token             {:>9.1}",
                total_us as f64 / 1000.0 / self.total_tokens.max(1) as f64
            );
        }
        out
    }
}

// ── Lightweight kernel-level timer (P3.2) ─────────────────────────────

/// Accumulates named timing regions over many invocations.
///
/// A simpler alternative to the region-based [`Profiler`]: call
/// [`KernelProfiler::time`] with any label and a closure to measure and
/// accumulate elapsed wall-clock time across calls.
#[derive(Debug, Clone, Default)]
pub struct KernelProfiler {
    pub regions: Vec<TimedRegion>,
}

/// A single named region tracked by [`KernelProfiler`].
#[derive(Debug, Clone)]
pub struct TimedRegion {
    pub label: String,
    pub elapsed_us: u64,
    pub calls: u64,
}

impl KernelProfiler {
    /// Time execution of `f`, associating elapsed microseconds with `label`.
    ///
    /// If `label` has been seen before, elapsed time and call count are
    /// accumulated; otherwise a new [`TimedRegion`] is created.
    pub fn time<R>(&mut self, label: &str, f: impl FnOnce() -> R) -> R {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_micros() as u64;
        if let Some(region) = self.regions.iter_mut().find(|r| r.label == label) {
            region.elapsed_us += elapsed;
            region.calls += 1;
        } else {
            self.regions.push(TimedRegion {
                label: label.to_string(),
                elapsed_us: elapsed,
                calls: 1,
            });
        }
        result
    }

    /// Total elapsed microseconds across all timed regions.
    pub fn total_us(&self) -> u64 {
        self.regions.iter().map(|r| r.elapsed_us).sum()
    }

    /// Clear all accumulated timing data.
    pub fn reset(&mut self) {
        self.regions.clear();
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn profiler_disabled_by_default() {
        let p = Profiler::new();
        assert!(!p.enabled());
    }

    #[test]
    fn profiler_accumulates() {
        let mut p = Profiler {
            enabled: true,
            ..Default::default()
        };
        p.begin(Region::AttnProj);
        thread::sleep(Duration::from_millis(2));
        p.end(Region::AttnProj);

        p.begin(Region::ExpertLoop);
        thread::sleep(Duration::from_millis(1));
        p.end(Region::ExpertLoop);

        p.token();
        p.token();

        let report = p.report();
        assert!(report.contains("attn_proj"));
        assert!(report.contains("expert_loop"));
        assert!(report.contains("2 tokens"));
    }

    #[test]
    fn profiler_launch_count() {
        let mut p = Profiler {
            enabled: true,
            ..Default::default()
        };
        for _ in 0..5 {
            p.launch(Region::Router);
        }
        let report = p.report();
        assert!(report.contains("router"));
        // report should show 5 launches
    }

    #[test]
    fn profiler_disabled_no_overhead() {
        let mut p = Profiler::new();
        // begin/end/token should be no-ops
        p.begin(Region::Embed);
        p.end(Region::Embed);
        p.token();
        p.launch(Region::Other);
        let report = p.report();
        assert!(report.contains("disabled"));
    }

    // ── KernelProfiler tests ───────────────────────────────────────

    #[test]
    fn kernel_profiler_single_region() {
        let mut kp = KernelProfiler::default();
        let result = kp.time("my_kernel", || 42);
        assert_eq!(result, 42);
        assert_eq!(kp.regions.len(), 1);
        assert_eq!(kp.regions[0].label, "my_kernel");
        assert_eq!(kp.regions[0].calls, 1);
    }

    #[test]
    fn kernel_profiler_accumulates_same_label() {
        let mut kp = KernelProfiler::default();
        kp.time("gemv", || {
            std::thread::sleep(std::time::Duration::from_micros(100))
        });
        kp.time("gemv", || {
            std::thread::sleep(std::time::Duration::from_micros(100))
        });
        assert_eq!(kp.regions.len(), 1);
        assert_eq!(kp.regions[0].calls, 2);
        assert!(kp.regions[0].elapsed_us >= 200);
    }

    #[test]
    fn kernel_profiler_multiple_labels() {
        let mut kp = KernelProfiler::default();
        kp.time("embed", || ());
        kp.time("attn", || ());
        kp.time("router", || ());
        assert_eq!(kp.regions.len(), 3);
        let labels: Vec<&str> = kp.regions.iter().map(|r| r.label.as_str()).collect();
        assert!(labels.contains(&"embed"));
        assert!(labels.contains(&"attn"));
        assert!(labels.contains(&"router"));
    }

    #[test]
    fn kernel_profiler_total_us() {
        let mut kp = KernelProfiler::default();
        kp.time("a", || ());
        kp.time("b", || ());
        let total = kp.total_us();
        let sum: u64 = kp.regions.iter().map(|r| r.elapsed_us).sum();
        assert_eq!(total, sum);
    }

    #[test]
    fn kernel_profiler_reset() {
        let mut kp = KernelProfiler::default();
        kp.time("x", || ());
        assert_eq!(kp.regions.len(), 1);
        kp.reset();
        assert!(kp.regions.is_empty());
    }

    #[test]
    fn kernel_profiler_default_is_empty() {
        let kp = KernelProfiler::default();
        assert!(kp.regions.is_empty());
        assert_eq!(kp.total_us(), 0);
    }
}
