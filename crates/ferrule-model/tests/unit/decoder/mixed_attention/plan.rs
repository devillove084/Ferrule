use super::*;
use NumericFp8Precision as P;

fn capacity(rows: usize, sequences: usize, positions: usize) -> Result<MixedAttentionCapacity> {
    Ok(MixedAttentionCapacity::for_profile(
        Some(P::Bf16ModelMixedF32Accumulate),
        true,
        rows,
        sequences,
        positions,
    )?
    .unwrap())
}

#[test]
fn mixed_context_limit_is_initial_admission_and_does_not_change_other_profiles() {
    for n in [1, 128, 1024] {
        assert!(capacity(8, 2, n).is_ok());
    }
    for n in [0, 1025, usize::MAX] {
        assert!(
            capacity(8, 2, n)
                .unwrap_err()
                .to_string()
                .contains("initial image admission")
        );
    }
    for precision in [
        None,
        Some(P::default()),
        Some(P::F32Tf32x3),
        Some(P::W8A16Bf16F32Accumulate),
        Some(P::Bf16ModelF32Accumulate),
        Some(P::Bf16ModelTiledF32Accumulate),
        Some(P::Bf16ModelSlicedK2F32Accumulate),
    ] {
        assert_eq!(
            MixedAttentionCapacity::for_profile(precision, true, 8, 2, 8192).unwrap(),
            None
        );
    }
    assert_eq!(
        MixedAttentionCapacity::for_profile(
            Some(P::Bf16ModelMixedF32Accumulate),
            false,
            8,
            2,
            8192
        )
        .unwrap(),
        None
    );
}

#[test]
fn invalid_row_and_sequence_capacities_reject_without_cuda() {
    for (rows, sequences) in [(0, 1), (1, 0), (usize::MAX, 1), (1, usize::MAX)] {
        assert!(capacity(rows, sequences, 128).is_err());
    }
}

#[test]
fn capacity_one_accepts_initial_prefill_and_every_cached_length() {
    let c = capacity(1, 2, 128).unwrap();
    for position in 0..128 {
        c.validate_cohorts(
            &[position],
            &[0],
            [(0..1, position, position + 1)].into_iter(),
        )
        .unwrap();
    }
    assert!(
        c.validate_cohorts(&[128], &[0], [(0..1, 128, 129)].into_iter())
            .is_err()
    );
}

#[test]
fn all_bounded_single_cohort_views_are_complete_unpadded_matrices() {
    let c = capacity(8, 2, 128).unwrap();
    for rows in 1..=8 {
        for tokens in rows..=128 {
            let context = tokens - rows;
            c.validate_cohorts(
                &(context..tokens).collect::<Vec<_>>(),
                &vec![0; rows],
                [(0..rows, context, tokens)].into_iter(),
            )
            .unwrap();
        }
    }
}

#[test]
fn ragged_prefill_decode_cohort_uses_each_sequences_own_history() {
    let c = capacity(8, 2, 128).unwrap();
    c.validate_cohorts(
        &[0, 1, 2, 3, 4, 127],
        &[0, 0, 0, 0, 0, 1],
        [(0..5, 0, 5), (5..6, 127, 128)].into_iter(),
    )
    .unwrap();
    c.validate_cohorts(
        &[99, 100, 101, 0, 1],
        &[0, 0, 0, 1, 1],
        [(0..3, 99, 102), (3..5, 0, 2)].into_iter(),
    )
    .unwrap();
}

#[test]
fn malformed_or_overbudget_cohorts_reject_before_append() {
    let c = capacity(8, 2, 128).unwrap();
    for cohorts in [
        vec![],
        vec![(0..0, 0, 0)],
        vec![(1..2, 0, 1)],
        vec![(0..1, 0, 1)],
        vec![(0..3, 0, 3)],
        vec![(0..2, 0, 129)],
        vec![(0..2, usize::MAX, 1)],
        vec![(0..2, 0, 1)],
        vec![(0..1, 0, 1), (0..1, 0, 1)],
    ] {
        assert!(
            c.validate_cohorts(&[0, 1], &[0, 0], cohorts.into_iter())
                .is_err()
        );
    }
    assert!(
        c.validate_cohorts(&[0, 1], &[0], [(0..2, 0, 2)].into_iter())
            .is_err()
    );
    assert!(
        c.validate_cohorts(&[0, 2], &[0, 0], [(0..2, 0, 2)].into_iter())
            .is_err()
    );
    assert!(
        c.validate_cohorts(&[0, 1], &[0, 1], [(0..2, 0, 2)].into_iter())
            .is_err()
    );
    assert!(
        capacity(1, 2, 128)
            .unwrap()
            .validate_cohorts(&[0, 1], &[0, 0], [(0..2, 0, 2)].into_iter())
            .is_err()
    );
    assert!(
        capacity(8, 1, 128)
            .unwrap()
            .validate_cohorts(&[0, 0], &[0, 1], [(0..1, 0, 1), (1..2, 0, 1)].into_iter())
            .is_err()
    );
}

#[test]
fn validated_packed_batch_uses_history_not_prefill_decode_label() {
    use crate::decoder::{DecoderKvPageStatus, GenericDecoderSequenceState};
    use ferrule_common::execution::*;
    use std::num::NonZeroU32;
    let caps = ExecutionCapabilities {
        max_batch_tokens: 4,
        max_sequences: 2,
        max_prefill_query_tokens_per_sequence: 4,
        max_decode_query_tokens_per_sequence: 4,
        max_top_k: NonZeroU32::new(4),
        supports_prefill: true,
        supports_decode: true,
        supports_mixed: true,
        full_logits_width: NonZeroU32::new(4),
        kv_binding_mode: KvBindingMode::Paged,
        logits_row_policy: LogitsRowPolicy::Any,
    };
    for (mode, phase) in [
        (ForwardMode::Prefill, ForwardPhase::Prefill),
        (ForwardMode::Decode, ForwardPhase::Decode),
    ] {
        let batch = ExecutionBatch::new(
            mode,
            vec![3],
            vec![1],
            vec![Some(KvWriteSlot::new(15))],
            vec![LogitsRequest::Full],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                phase,
                0..1,
                1,
                2,
                0..1,
            )],
            vec![KvBlockId::new(7)],
        );
        let packed = PackedDecoderBatch::lower(
            &batch,
            &[KvReservationView {
                state_slot: StateSlot::new(9),
                execution_state_slot: StateSlot::new(0),
                positions: 1..2,
                newly_allocated: vec![],
                generation: 5,
                execution_generation: 0,
                cow_replacement: None,
            }],
            &[GenericDecoderSequenceState::with_position(1, (), ())],
            &caps,
            2,
            &|_| DecoderKvPageStatus::Resident,
        )
        .unwrap();
        capacity(1, 1, 2).unwrap().validate_batch(&packed).unwrap();
        assert!(capacity(1, 1, 1).unwrap().validate_batch(&packed).is_err());
    }
}
