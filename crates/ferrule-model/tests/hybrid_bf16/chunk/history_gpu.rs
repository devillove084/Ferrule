//! Existing runner/journal protocol, not a replacement lifecycle.
use super::*;

fn frontier(state: &CudaHybridSequenceState) -> usize {
    let position = state.core().position();
    for layer in state.kv_state().bf16_layers().unwrap().iter().flatten() {
        assert_eq!(layer.position(), position);
    }
    position
}

#[test]
#[ignore = "sole GPU owner; M1 fresh abort/retry, retained prefill vs decode, fork/COW/reset"]
fn mixed_m1_history_survives_abort_fork_and_prefill_continuation() {
    let fixture = Fixture::numeric_profile();
    let (mut runner, device) = build(
        &fixture,
        ferrule_model::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate,
    );
    let scratch = runner
        .forward_executor()
        .module()
        .bf16_chunk_workspace_bytes();
    assert!(scratch > 0);
    let mut parent = runner.create_sequence_state().unwrap();
    let mut pages = Vec::new();
    let initial = state_values(&parent);
    assert_eq!(frontier(&parent), 0);
    let aborted = step_in_phase(
        &mut runner,
        &mut parent,
        &mut pages,
        &[1],
        (41, false, false),
        ForwardPhase::Prefill,
    );
    assert_eq!(frontier(&parent), 0);
    assert_eq!(state_values(&parent), initial);
    let first = step_in_phase(
        &mut runner,
        &mut parent,
        &mut pages,
        &[1],
        (42, true, false),
        ForwardPhase::Prefill,
    );
    assert_eq!(first, aborted);
    assert_eq!(frontier(&parent), 1);
    let mut fork = runner.fork_sequence_state_from(&parent, 1).unwrap();
    let mut fork_pages = pages.clone();
    assert_eq!(frontier(&fork), 1);
    let continued_prefill = step_in_phase(
        &mut runner,
        &mut parent,
        &mut pages,
        &[3],
        (43, true, true),
        ForwardPhase::Prefill,
    );
    let cached_decode = step_in_phase(
        &mut runner,
        &mut fork,
        &mut fork_pages,
        &[3],
        (44, true, true),
        ForwardPhase::Decode,
    );
    assert_eq!(continued_prefill, cached_decode);
    assert_eq!(state_values(&parent), state_values(&fork));
    assert_eq!(frontier(&parent), 2);
    assert_eq!(frontier(&fork), 2);
    runner.try_release_sequence_state(fork).unwrap();
    runner.reset_sequence_state(&mut parent).unwrap();
    pages.clear();
    assert_eq!(frontier(&parent), 0);
    assert_eq!(state_values(&parent), initial);
    let fresh_again = step_in_phase(
        &mut runner,
        &mut parent,
        &mut pages,
        &[1],
        (45, true, false),
        ForwardPhase::Prefill,
    );
    assert_eq!(fresh_again, first);
    assert_eq!(
        runner
            .forward_executor()
            .module()
            .bf16_chunk_workspace_bytes(),
        scratch
    );
    runner.try_release_sequence_state(parent).unwrap();
    runner.shutdown().unwrap();
    drop(runner);
    assert_eq!(device.live_state_bytes(), 0);
    assert!(!device.needs_quarantine());
}
