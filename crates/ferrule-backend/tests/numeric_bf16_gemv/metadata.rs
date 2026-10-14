use super::*;
fn layout() -> NumericFp8Layout {
    NumericFp8Layout {
        n: 2048,
        k: 4096,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    }
}
const P: NumericFp8Precision = NumericFp8Precision::Bf16M1GemvF32Accumulate;
#[test]
fn gemv_fixed_geometry_and_provider_reserve_are_separate() {
    for budget in [NUMERIC_BF16_GEMV_SCRATCH_BYTES, 8 << 20] {
        let p = NumericFp8LinearPlan::new(layout(), 1, budget, P).unwrap();
        assert_eq!(p.selected_kernel(), NumericFp8SelectedKernel::Bf16M1Gemv);
        assert_eq!(p.tile_rows(), 257);
        assert_eq!(p.tile_count(), 8);
        assert_eq!(p.kernel_launches(), 17);
        assert_eq!(p.provider_bytes(), 16 << 20);
        assert!(p.validate_provider_budget(p.provider_bytes() - 1).is_err());
        p.validate_provider_budget(p.provider_bytes()).unwrap();
        assert_eq!(
            p.workspace_requirements().bytes as usize,
            p.activation_bytes() + p.weight_tile_bytes() + 2048 * 4
        );
        assert!(p.workspace_requirements().bytes <= budget as u64);
    }
    assert!(
        NumericFp8LinearPlan::new(layout(), 1, NUMERIC_BF16_GEMV_SCRATCH_BYTES - 1, P).is_err()
    );
}
#[test]
fn gemv_rejects_uncovered_shapes_scales_origins_and_multirow() {
    for l in [
        NumericFp8Layout {
            n: 2047,
            ..layout()
        },
        NumericFp8Layout {
            k: 4095,
            ..layout()
        },
        NumericFp8Layout {
            row_origin: 128,
            ..layout()
        },
        NumericFp8Layout {
            column_origin: 128,
            ..layout()
        },
        NumericFp8Layout {
            scale_type: NumericFp8ScaleType::F32,
            ..layout()
        },
    ] {
        assert!(NumericFp8LinearPlan::new(l, 1, 8 << 20, P).is_err());
    }
    for m in [0, 2, 5, 8] {
        assert!(NumericFp8LinearPlan::new(layout(), m, 8 << 20, P).is_err());
    }
}
#[test]
fn old_numeric_profiles_never_charge_or_select_gemv() {
    for p in [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::Bf16SlicedK2F32Accumulate,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
        NumericFp8Precision::F32Tf32x3,
    ] {
        let p = NumericFp8LinearPlan::new(layout(), 1, 8 << 20, p).unwrap();
        assert_eq!(p.provider_bytes(), 0);
        assert_ne!(p.selected_kernel(), NumericFp8SelectedKernel::Bf16M1Gemv);
    }
}
