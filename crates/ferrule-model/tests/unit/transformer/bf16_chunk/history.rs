//! CPU policy specification using the existing sequence cursor lifecycle.
use super::NumericFp8Precision as P;
use crate::execution::SequenceStateCore;
use ferrule_common::execution::ForwardPhase;

fn selection(profile: P, rows: usize, state: &SequenceStateCore) -> bool {
    profile.gdn_uses_chunk(rows, state.position() != 0)
}

#[test]
fn fresh_single_row_and_retained_single_row_follow_history_not_scheduler_phase() {
    for phase in [ForwardPhase::Prefill, ForwardPhase::Decode] {
        // HF's predicate does not inspect the scheduler's phase label.
        assert!(
            selection(P::Bf16ModelMixedF32Accumulate, 1, &SequenceStateCore::new()),
            "{phase:?} with no previous state must use chunk arithmetic"
        );
        for position in [1, 4, 65, 129] {
            assert!(
                !selection(
                    P::Bf16ModelMixedF32Accumulate,
                    1,
                    &SequenceStateCore::with_position(position)
                ),
                "{phase:?} at {position}: even a last prefill token uses cached recurrence"
            );
        }
    }
}

#[test]
fn mixed_cohorts_are_selected_independently_and_profiles_stay_explicit() {
    let cohorts = [(1, 0), (1, 5), (5, 0), (1, 65), (2, 5), (0, 0)];
    for profile in [
        P::default(),
        P::F32Tf32x3,
        P::W8A16Bf16F32Accumulate,
        P::Bf16ModelF32Accumulate,
        P::Bf16ModelTiledF32Accumulate,
        P::Bf16ModelSlicedK2F32Accumulate,
        P::Bf16ModelMixedF32Accumulate,
    ] {
        let selected: Vec<_> = cohorts
            .map(|(rows, position)| {
                selection(profile, rows, &SequenceStateCore::with_position(position))
            })
            .into_iter()
            .collect();
        assert_eq!(
            selected,
            if profile == P::Bf16ModelMixedF32Accumulate {
                vec![true, false, true, false, true, false]
            } else {
                vec![false; cohorts.len()]
            }
        );
    }
}

#[test]
fn history_policy_tracks_commit_abort_fork_reset_and_restore_without_extra_flag() {
    let profile = P::Bf16ModelMixedF32Accumulate;
    let mut committed = SequenceStateCore::new();
    let empty_snapshot = committed.clone();
    let mut working = committed.clone();
    working
        .commit_step(working.begin_step().unwrap(), 1)
        .unwrap();
    assert!(!selection(profile, 1, &working));
    // Aborting discards the working frontier, not a separately mutated flag.
    assert!(selection(profile, 1, &committed));
    committed
        .commit_step(committed.begin_step().unwrap(), 5)
        .unwrap();
    assert!(!selection(profile, 1, &committed));
    let retained_snapshot = committed.clone();
    let mut fork = committed.forked().unwrap();
    assert!(!selection(profile, 1, &fork));
    fork.reset();
    assert!(selection(profile, 1, &fork));
    assert!(!selection(profile, 1, &committed));
    committed.restore_from(&empty_snapshot);
    assert!(selection(profile, 1, &committed));
    committed.restore_from(&retained_snapshot);
    assert!(!selection(profile, 1, &committed));
}

#[test]
fn row_capacity_one_still_requires_a_fresh_sequence_chunk_plan() {
    // Admission must budget for first use, not only steady-state decode.
    let capacity = 1;
    assert!(selection(
        P::Bf16ModelMixedF32Accumulate,
        capacity,
        &SequenceStateCore::new()
    ));
    assert!(!selection(
        P::Bf16ModelTiledF32Accumulate,
        capacity,
        &SequenceStateCore::new()
    ));
}
