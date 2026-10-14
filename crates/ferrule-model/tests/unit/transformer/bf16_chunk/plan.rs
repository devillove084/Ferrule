use super::NumericFp8Precision as P;
use ferrule_backend::cuda_plan::GatedDeltaNetLayout;
fn layout(rows: usize) -> GatedDeltaNetLayout {
    GatedDeltaNetLayout {
        rows,
        key_heads: 16,
        value_heads: 32,
        key_dim: 128,
        value_dim: 128,
    }
}
#[test]
fn mixed_chunk_is_per_sequence_history_aware_and_keeps_other_profiles() {
    for precision in [
        P::default(),
        P::F32Tf32x3,
        P::W8A16Bf16F32Accumulate,
        P::Bf16ModelF32Accumulate,
        P::Bf16ModelTiledF32Accumulate,
        P::Bf16ModelSlicedK2F32Accumulate,
        P::Bf16ModelMixedF32Accumulate,
    ] {
        for (rows, previous) in [1, 2, 5, 9, 64, 65, 129]
            .into_iter()
            .flat_map(|rows| [false, true].map(|previous| (rows, previous)))
        {
            let plan = precision
                .gdn_chunk_plan(layout(rows), previous, 17_850_368)
                .unwrap();
            assert_eq!(
                plan.is_some(),
                precision == P::Bf16ModelMixedF32Accumulate && (!previous || rows > 1)
            );
            if let Some(plan) = plan {
                assert_eq!(plan.workspace_elements(), 4_462_592);
                let backend = ferrule_backend::cuda_plan::Bf16ChunkPlan::new_prefill(
                    layout(rows),
                    17_850_368,
                )
                .unwrap();
                assert_eq!(plan.launch_count(), backend.launch_count());
            }
        }
    }
    // A packed multi-sequence decode with one row per cohort is still scan.
    for _ in 0..8 {
        assert!(
            P::Bf16ModelMixedF32Accumulate
                .gdn_chunk_plan(layout(1), true, 0)
                .unwrap()
                .is_none()
        );
    }
}
#[test]
fn mixed_chunk_rejects_underbudget_and_backend_unsupported_shapes() {
    let mixed = P::Bf16ModelMixedF32Accumulate;
    assert!(mixed.gdn_chunk_plan(layout(5), false, 17_850_367).is_err());
    assert!(mixed.gdn_chunk_plan(layout(5), false, 0).is_err());
    for l in [
        GatedDeltaNetLayout {
            key_dim: 513,
            ..layout(5)
        },
        GatedDeltaNetLayout {
            value_dim: 513,
            ..layout(5)
        },
        GatedDeltaNetLayout {
            key_heads: 3,
            ..layout(5)
        },
        GatedDeltaNetLayout {
            rows: usize::MAX,
            ..layout(5)
        },
    ] {
        assert!(mixed.gdn_chunk_plan(l, false, usize::MAX).is_err());
    }
}

#[test]
fn fresh_one_row_has_full_budget_execution_identity_and_cached_one_row_is_unchanged() {
    let profile = P::Bf16ModelMixedF32Accumulate;
    assert!(
        profile
            .gdn_chunk_plan(layout(1), false, 17_850_367)
            .is_err()
    );
    let fresh = profile
        .gdn_chunk_plan(layout(1), false, 17_850_368)
        .unwrap()
        .unwrap();
    assert_eq!(fresh.workspace_requirements().bytes, 17_850_368);
    assert_eq!(fresh.launch_count(), 15);
    assert_eq!(
        fresh.execution(),
        ferrule_backend::cuda_plan::Bf16ChunkExecution::PrefillChunk
    );
    let cached = ferrule_backend::cuda_plan::Bf16ChunkPlan::new(layout(1), 0).unwrap();
    assert_ne!(fresh, cached);
    assert_eq!(cached.workspace_elements(), 0);
    assert_eq!(cached.launch_count(), 1);
    assert!(
        profile
            .gdn_chunk_plan(layout(1), true, 0)
            .unwrap()
            .is_none()
    );
}
