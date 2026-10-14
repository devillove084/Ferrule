//! CPU-only tests of production timing aggregation; no CUDA initialization.
#[path = "../src/transformer/standard/performance.rs"]
mod performance;
#[path = "../src/transformer/standard/cuda/rank_timing.rs"]
mod rank_timing;
use rank_timing::*;
use std::{rc::Rc, time::Duration};
fn us(n: u64) -> Duration {
    Duration::from_micros(n)
}
#[test]
fn disabled_phase_never_reads_clock_or_allocates_a_guard() {
    assert!(PhaseGuard::start_with(None, Phase::Gdn, |_| panic!("disabled clock")).is_none());
    assert!(PhaseGuard::start(None, Phase::Gqa).is_none());
}
#[test]
fn nested_phases_are_exclusive_and_accumulate_counts_and_failures() {
    let mut ledger = Ledger::default();
    let outer = ledger.enter(Phase::Router, us(2));
    let inner = ledger.enter(Phase::RouteHostRead, us(5));
    ledger.leave(Phase::RouteHostRead, inner, us(12), true);
    ledger.leave(Phase::Router, outer, us(15), false);
    let first = ledger.snapshot(us(20));
    assert_eq!(first[Phase::Other as usize].elapsed, us(7));
    assert_eq!(first[Phase::Router as usize].elapsed, us(6));
    assert_eq!(first[Phase::RouteHostRead as usize].elapsed, us(7));
    assert_eq!(first[Phase::Router as usize].failures, 1);
    assert_eq!(first.iter().map(|p| p.elapsed).sum::<Duration>(), us(20));
    let previous = ledger.enter(Phase::Router, us(20));
    ledger.leave(Phase::Router, previous, us(24), true);
    let delta =
        ledger.snapshot(us(24))[Phase::Router as usize].since(first[Phase::Router as usize]);
    assert_eq!((delta.calls, delta.failures, delta.elapsed), (1, 0, us(4)));
}
#[test]
fn guard_success_and_unwind_restore_parent_without_completion_effects() {
    let state = Rc::new(Timings::new());
    PhaseGuard::start(Some(&state), Phase::Head)
        .unwrap()
        .finish(true);
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = PhaseGuard::start(Some(&state), Phase::Collective).unwrap();
        panic!("operator panic");
    }));
    assert!(failure.is_err());
    assert_eq!(Rc::strong_count(&state), 1);
    let samples = state.ledger.borrow_mut().snapshot(state.started.elapsed());
    assert_eq!(samples[Phase::Head as usize].calls, 1);
    assert_eq!(samples[Phase::Head as usize].failures, 0);
    assert_eq!(samples[Phase::Collective as usize].failures, 1);
}
#[test]
fn all_phase_slots_and_observation_only_counters_are_available() {
    let state = Timings::new();
    let mut ledger = state.ledger.borrow_mut();
    for phase in [
        Phase::AttentionProjection,
        Phase::Gqa,
        Phase::Gdn,
        Phase::RouteAgreement,
        Phase::SharedFfn,
        Phase::DenseFfn,
        Phase::RoutedFfn,
        Phase::SourceRetirement,
    ] {
        let parent = ledger.enter(phase, us(0));
        ledger.leave(phase, parent, us(0), true);
    }
    ledger.drain.measure(|| Ok::<_, ()>(())).unwrap();
    ledger.upload_wait_skips += 1;
    ledger.routed_h2d_calls += 2;
    ledger.routed_h2d_bytes += 96;
    ledger.retired_source_logical_bytes += 4096;
    ledger.retirement_batches += 1;
    assert_eq!(
        (
            ledger.upload_wait_skips,
            ledger.routed_h2d_calls,
            ledger.routed_h2d_bytes,
            ledger.retired_source_logical_bytes,
            ledger.retirement_batches
        ),
        (1, 2, 96, 4096, 1)
    );
}

#[test]
fn disabled_production_gate_precedes_clocks_allocations_and_counter_snapshots() {
    let code = include_str!("../src/transformer/standard/cuda/rank_performance.rs");
    let begin = code
        .split("fn begin_rank_performance")
        .nth(1)
        .unwrap()
        .split("fn rank_snapshot")
        .next()
        .unwrap();
    let reject = begin.find("return false").unwrap();
    for work in [
        "Rc::new(Timings::new())",
        "self.rank_snapshot",
        "ProofTiming::default()",
    ] {
        assert!(reject < begin.find(work).unwrap());
    }
    assert!(begin.contains("!self.rank_performance_enabled"));
    for forbidden in [
        ".record_compute_event(",
        ".sync_stream(",
        ".sync_upload_stream(",
        ".download_f32_buffer(",
        ".download_bf16_buffer(",
        ".bf16_to_f32(",
        ".memory_info(",
    ] {
        assert!(
            !code.contains(forbidden),
            "observer introduced GPU work: {forbidden}"
        );
    }
    let owner = include_str!("../src/transformer/standard/cuda.rs");
    assert!(owner.contains("rank_performance_enabled: false"));
}

#[test]
#[ignore = "manual CPU-only clock/guard overhead calibration; not model performance"]
fn calibrate_host_clock_and_phase_guard_overhead() {
    use std::hint::black_box;
    use std::time::Instant;
    const N: usize = 100_000;
    let start = Instant::now();
    for _ in 0..N {
        black_box(PhaseGuard::start(None, Phase::Gdn));
    }
    let disabled_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    for _ in 0..N {
        black_box(Instant::now());
    }
    let clock_ns = start.elapsed().as_nanos();
    let state = Rc::new(Timings::new());
    let start = Instant::now();
    for _ in 0..N {
        PhaseGuard::start(Some(&state), Phase::Gdn)
            .unwrap()
            .finish(true);
    }
    let guard_ns = start.elapsed().as_nanos();
    eprintln!(
        "CPU_CLOCK_CALIBRATION iterations={N} disabled_total_ns={disabled_ns} clock_total_ns={clock_ns} guard_total_ns={guard_ns}"
    );
}

#[test]
fn production_schema_and_wiring_keep_profile_separate_from_numeric_observer() {
    let code = include_str!("../src/transformer/standard/cuda/rank_performance.rs");
    for field in [
        "mixed_rank_performance",
        "schema_version",
        "scope",
        "transaction",
        "host_wall_us",
        "attention_projection_host_us",
        "gdn_host_us",
        "gqa_host_us",
        "route_host_read_us",
        "route_agreement_host_us",
        "shared_ffn_host_us",
        "routed_ffn_host_us",
        "collective_host_us",
        "head_host_us",
        "consumer_event_calls",
        "compute_sync_calls",
        "upload_sync_calls",
        "boundary_drain_calls",
        "allocation_attempts",
        "route_metadata_upload_bytes",
        "source_retired_logical_bytes",
        "upload_wait_actual_skips",
    ] {
        assert!(code.contains(field), "missing schema field {field}");
    }
    let owner = include_str!("../src/transformer/standard/cuda.rs");
    let policy = owner
        .split("fn mixed_upload_proof_scope")
        .nth(1)
        .unwrap()
        .split("fn synchronous<T>")
        .next()
        .unwrap();
    assert!(
        !policy.contains("rank_performance"),
        "profiling must not disable the measured upload-skip policy"
    );
    let mixed = include_str!("../src/transformer/standard/cuda/hybrid_rank/mixed_layer.rs");
    assert!(mixed.contains("self.begin_rank_layer(layer.index())"));
    assert!(mixed.contains("self.end_rank_layer(\"ok\")"));
    assert!(mixed.contains("source_retirement_logical_bytes"));
}
