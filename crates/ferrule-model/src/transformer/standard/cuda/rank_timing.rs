//! Fixed-size host-wall aggregates. Never GPU timing or completion authority.
use super::performance::HostCallTiming;
use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum Phase {
    Other,
    AttentionProjection,
    Gqa,
    Gdn,
    Router,
    RouteHostRead,
    RouteAgreement,
    SharedFfn,
    DenseFfn,
    RoutedFfn,
    Collective,
    Head,
    SourceRetirement,
}
pub(super) const PHASES: usize = 13;
#[derive(Clone, Copy, Default)]
pub(super) struct Sample {
    pub calls: u64,
    pub failures: u64,
    pub elapsed: Duration,
}
impl Sample {
    pub fn of(t: &HostCallTiming) -> Self {
        Self {
            calls: t.calls,
            failures: t.failures,
            elapsed: t.elapsed,
        }
    }
    pub fn since(self, before: Self) -> Self {
        Self {
            calls: self.calls.saturating_sub(before.calls),
            failures: self.failures.saturating_sub(before.failures),
            elapsed: self.elapsed.saturating_sub(before.elapsed),
        }
    }
}
pub(super) struct Ledger {
    phases: [HostCallTiming; PHASES],
    active: Phase,
    last: Duration,
    pub drain: HostCallTiming,
    pub upload_wait_skips: u64,
    pub routed_h2d_calls: u64,
    pub routed_h2d_bytes: u64,
    pub retired_source_logical_bytes: u64,
    pub retirement_batches: u64,
    #[cfg_attr(test, allow(dead_code))] // Standalone timing tests omit the rank observer.
    pub decoded_cache: [u64; 4],
}
impl Default for Ledger {
    fn default() -> Self {
        Self {
            phases: std::array::from_fn(|_| HostCallTiming::default()),
            active: Phase::Other,
            last: Duration::ZERO,
            drain: HostCallTiming::default(),
            upload_wait_skips: 0,
            routed_h2d_calls: 0,
            routed_h2d_bytes: 0,
            retired_source_logical_bytes: 0,
            retirement_batches: 0,
            decoded_cache: [0; 4],
        }
    }
}
impl Ledger {
    fn charge(&mut self, now: Duration) {
        self.phases[self.active as usize].elapsed += now.saturating_sub(self.last);
        self.last = now;
    }
    pub fn enter(&mut self, phase: Phase, now: Duration) -> Phase {
        self.charge(now);
        let previous = self.active;
        self.active = phase;
        self.phases[phase as usize].calls += 1;
        previous
    }
    pub fn leave(&mut self, phase: Phase, previous: Phase, now: Duration, succeeded: bool) {
        self.charge(now);
        self.phases[phase as usize].failures += u64::from(!succeeded);
        self.active = previous;
    }
    pub fn snapshot(&mut self, now: Duration) -> [Sample; PHASES] {
        self.charge(now);
        std::array::from_fn(|i| Sample::of(&self.phases[i]))
    }
}
pub(super) struct Timings {
    pub started: Instant,
    pub ledger: RefCell<Ledger>,
}
impl Timings {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            ledger: RefCell::new(Ledger::default()),
        }
    }
}
pub(super) struct PhaseGuard {
    state: Rc<Timings>,
    phase: Phase,
    previous: Phase,
    succeeded: bool,
}
impl PhaseGuard {
    pub fn start(state: Option<&Rc<Timings>>, phase: Phase) -> Option<Self> {
        Self::start_with(state, phase, |s| s.started.elapsed())
    }
    // Lazy clock makes the disabled contract directly testable without CUDA.
    pub fn start_with(
        state: Option<&Rc<Timings>>,
        phase: Phase,
        clock: impl FnOnce(&Timings) -> Duration,
    ) -> Option<Self> {
        let state = state?;
        let previous = state.ledger.borrow_mut().enter(phase, clock(state));
        Some(Self {
            state: Rc::clone(state),
            phase,
            previous,
            succeeded: false,
        })
    }
    pub fn finish(mut self, succeeded: bool) {
        self.succeeded = succeeded;
    }
}
impl Drop for PhaseGuard {
    fn drop(&mut self) {
        self.state.ledger.borrow_mut().leave(
            self.phase,
            self.previous,
            self.state.started.elapsed(),
            self.succeeded,
        );
    }
}
