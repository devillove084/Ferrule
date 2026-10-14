use ferrule_backend::cuda_plan::{Bf16ChunkPlan, GatedDeltaNetLayout};
fn shape(rows: usize) -> GatedDeltaNetLayout {
    GatedDeltaNetLayout {
        rows,
        key_heads: 16,
        value_heads: 32,
        key_dim: 128,
        value_dim: 128,
    }
}
#[test]
fn chunk_budget_exact_and_bounded_in_rows() {
    for (rows, launches) in [(2, 15), (5, 15), (32, 15), (64, 15), (65, 30), (129, 45)] {
        let p = Bf16ChunkPlan::new(shape(rows), 17_850_368).unwrap();
        assert_eq!(p.workspace_elements(), 4_462_592);
        assert_eq!(p.workspace_requirements().bytes, 17_850_368);
        assert_eq!(p.workspace_requirements().alignment, 4);
        assert_eq!(p.launch_count(), launches);
        assert!(Bf16ChunkPlan::new(shape(rows), 17_850_367).is_err());
    }
}
#[test]
fn single_row_preserves_existing_no_scratch_contract() {
    let p = Bf16ChunkPlan::new(shape(1), 0).unwrap();
    assert_eq!(p.workspace_elements(), 0);
    assert_eq!(p.launch_count(), 1);
}
#[test]
fn invalid_shapes_overflow_and_native_capability() {
    let s = shape(5);
    for bad in [
        GatedDeltaNetLayout { rows: 0, ..s },
        GatedDeltaNetLayout { key_heads: 0, ..s },
        GatedDeltaNetLayout { key_heads: 3, ..s },
        GatedDeltaNetLayout {
            value_heads: usize::MAX,
            ..s
        },
        GatedDeltaNetLayout {
            rows: usize::MAX,
            ..s
        },
        GatedDeltaNetLayout {
            key_dim: usize::MAX,
            ..s
        },
        GatedDeltaNetLayout { key_dim: 513, ..s },
        GatedDeltaNetLayout {
            value_dim: 513,
            ..s
        },
        GatedDeltaNetLayout {
            key_heads: 1,
            value_heads: 65536,
            ..s
        },
        GatedDeltaNetLayout {
            key_heads: 1,
            value_heads: 65535,
            key_dim: 512,
            value_dim: 512,
            rows: 2,
        },
    ] {
        assert!(Bf16ChunkPlan::new(bad, usize::MAX).is_err(), "{bad:?}");
    }
}
#[test]
fn dimensions_and_grouped_heads_have_checked_formula() {
    for (dk, dv, kh, hv) in [
        (1, 3, 1, 2),
        (17, 9, 2, 6),
        (64, 32, 4, 4),
        (512, 511, 1, 2),
    ] {
        let l = GatedDeltaNetLayout {
            rows: 65,
            key_heads: kh,
            value_heads: hv,
            key_dim: dk,
            value_dim: dv,
        };
        let bytes = 4 * hv * (6 * 64 * dk + 7 * 64 * dv + 4 * 64 * 64 + 3 * 64 + dk * dv);
        assert_eq!(
            Bf16ChunkPlan::new(l, bytes)
                .unwrap()
                .workspace_requirements()
                .bytes,
            bytes as u64
        );
    }
}

#[test]
fn explicit_prefill_one_row_is_not_cached_recurrence() {
    use ferrule_backend::cuda_plan::Bf16ChunkExecution;
    let cached = Bf16ChunkPlan::new(shape(1), 0).unwrap();
    let prefill = Bf16ChunkPlan::new_prefill(shape(1), 17_850_368).unwrap();
    assert_eq!(cached.execution(), Bf16ChunkExecution::CachedRecurrent);
    assert_eq!(prefill.execution(), Bf16ChunkExecution::PrefillChunk);
    assert_eq!(cached.launch_count(), 1);
    assert_eq!(prefill.launch_count(), 15);
    assert_eq!(prefill.workspace_elements(), 4_462_592);
    assert_eq!(prefill.workspace_requirements().bytes, 17_850_368);
    assert_ne!(cached, prefill);
    assert!(Bf16ChunkPlan::new_prefill(shape(1), 17_850_367).is_err());
    for rows in [2, 5, 64, 65, 129] {
        assert_eq!(
            Bf16ChunkPlan::new_prefill(shape(rows), usize::MAX).unwrap(),
            Bf16ChunkPlan::new(shape(rows), usize::MAX).unwrap()
        );
    }
}

#[test]
fn explicit_prefill_checks_single_row_capability_and_overflow() {
    let s = shape(1);
    for bad in [
        GatedDeltaNetLayout { rows: 0, ..s },
        GatedDeltaNetLayout {
            rows: usize::MAX,
            ..s
        },
        GatedDeltaNetLayout { key_dim: 513, ..s },
        GatedDeltaNetLayout {
            value_dim: 513,
            ..s
        },
        GatedDeltaNetLayout { key_heads: 3, ..s },
        GatedDeltaNetLayout {
            value_heads: usize::MAX,
            ..s
        },
    ] {
        assert!(Bf16ChunkPlan::new_prefill(bad, usize::MAX).is_err());
    }
    // Cached admission has not been narrowed to the chunk capability.
    assert!(Bf16ChunkPlan::new(GatedDeltaNetLayout { key_dim: 513, ..s }, 0).is_ok());
}
