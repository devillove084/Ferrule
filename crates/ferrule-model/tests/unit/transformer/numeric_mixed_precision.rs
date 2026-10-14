use super::*;
use crate::support::TensorRole as R;
use ferrule_backend::cuda_plan::{
    NumericFp8Layout, NumericFp8ScaleType, NumericFp8SelectedKernel as K,
};
const MIXED: NumericFp8Precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
const SCRATCH: usize = 8 << 20;
fn layout(n: usize, k: usize) -> NumericFp8Layout {
    NumericFp8Layout {
        n,
        k,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    }
}

#[test]
fn mixed_role_shape_schedule_and_unverified_routed_baseline() {
    for (role, n, k, expected) in [
        (R::LinearAttentionQkv, 8192, 2048, K::Bf16TiledGemm),
        (R::LinearAttentionZ, 4096, 2048, K::Bf16TiledSlicedK2Gemm),
        (R::AttentionOutput, 2048, 4096, K::Bf16M1Gemv),
        (R::SharedExpertGate, 512, 2048, K::W8A16FusedBf16Vector),
        (R::SharedExpertUp, 512, 2048, K::W8A16FusedBf16Vector),
        (R::SharedExpertDown, 2048, 512, K::W8A16FusedBf16Vector),
        (R::RoutedExpertGate, 512, 2048, K::W8A16FusedBf16Vector),
        (R::RoutedExpertUp, 512, 2048, K::W8A16FusedBf16Vector),
        (R::RoutedExpertDown, 2048, 512, K::W8A16FusedBf16Vector),
    ] {
        let l = layout(n, k);
        let plan = MIXED.linear_plan(&role, l, 1, SCRATCH).unwrap();
        assert_eq!(plan.selected_kernel(), expected);
        let upper = NumericFp8Precision::Bf16ModelTiledF32Accumulate
            .linear_plan(&role, l, 1, SCRATCH)
            .unwrap();
        assert!(plan.workspace_requirements().bytes <= upper.workspace_requirements().bytes);
        assert!(
            plan.workspace_allocation_grant_bytes().unwrap()
                <= upper.workspace_allocation_grant_bytes().unwrap()
        );
        for m in [2, 5, 8, 9] {
            let actual = MIXED.linear_plan(&role, l, m, SCRATCH).unwrap();
            let upper = NumericFp8Precision::Bf16ModelTiledF32Accumulate
                .linear_plan(&role, l, m, SCRATCH)
                .unwrap();
            assert_eq!(
                actual.selected_kernel(),
                if role == R::LinearAttentionQkv {
                    K::Bf16TiledSlicedK2Gemm
                } else {
                    K::Bf16TiledGemm
                }
            );
            assert_eq!(
                actual.workspace_requirements(),
                upper.workspace_requirements()
            );
        }
    }
    let qkv = MIXED
        .linear_plan(&R::LinearAttentionQkv, layout(8192, 2048), 1, SCRATCH)
        .unwrap();
    assert_eq!(qkv.workspace_requirements().bytes, SCRATCH as u64);
    assert_eq!(qkv.tile_rows(), 2047);
    assert!(
        MIXED
            .linear_plan(&R::LinearAttentionQkv, layout(8192, 2048), 1, 8191)
            .is_err()
    );
}

#[test]
fn mixed_guards_fall_back_without_reinterpreting_unknown_shapes_or_strides() {
    for (role, n, k) in [
        (R::LinearAttentionZ, 4095, 2048),
        (R::LinearAttentionZ, 4096, 2047),
        (R::LinearAttentionZ, 16, 16),
        (R::AttentionOutput, 2048, 2048),
        (R::SharedExpertGate, 1024, 2048),
        (R::SharedExpertUp, 512, 1024),
        (R::SharedExpertDown, 2048, 1024),
        (R::OutputHead, 4096, 2048),
        (R::RouterLogits, 4096, 2048),
        (R::AttentionQuery, 8192, 2048),
    ] {
        assert_eq!(
            MIXED
                .linear_plan(&role, layout(n, k), 1, SCRATCH)
                .unwrap()
                .selected_kernel(),
            K::Bf16TiledGemm
        );
    }
    for l in [
        NumericFp8Layout {
            row_origin: 128,
            ..layout(4096, 2048)
        },
        NumericFp8Layout {
            column_origin: 128,
            ..layout(4096, 2048)
        },
        NumericFp8Layout {
            scale_type: NumericFp8ScaleType::F32,
            ..layout(4096, 2048)
        },
    ] {
        assert_eq!(
            MIXED
                .linear_plan(&R::LinearAttentionZ, l, 1, SCRATCH)
                .unwrap()
                .selected_kernel(),
            K::Bf16TiledGemm
        );
    }
    assert!(
        MIXED
            .linear_plan(&R::LinearAttentionZ, layout(4096, 2048), 0, SCRATCH)
            .is_err()
    );
}

#[test]
fn mixed_identity_admission_preserves_defaults_and_old_profiles() {
    assert_eq!(
        MIXED.hybrid_backend_name(),
        "cuda-hybrid-numeric-fp8-model-bf16-mixed-f32-ssm"
    );
    assert_eq!(
        MIXED.standard_backend_name(),
        "cuda-standard-numeric-fp8-model-bf16-mixed-f32-accumulate"
    );
    assert!(MIXED.has_bf16_model_boundaries());
    assert!(!MIXED.allows_zero_scratch());
    assert_eq!(
        NumericFp8Precision::default(),
        NumericFp8Precision::Bf16RneF32Accumulate
    );
    let limits = ExpertCacheLimits {
        max_experts: 1,
        max_bytes: 16 << 20,
    };
    let base = options(ExecutionPrecisionPolicy::bf16_compatibility());
    assert!(
        base.clone()
            .with_hybrid_cuda_numeric_fp8_bf16_precision(limits, 0, MIXED)
            .is_err()
    );
    assert!(
        options(ExecutionPrecisionPolicy::f32())
            .with_hybrid_cuda_numeric_fp8_bf16_precision(limits, SCRATCH, MIXED)
            .is_err()
    );
    let image = base
        .clone()
        .with_hybrid_cuda_numeric_fp8_bf16_precision(limits, SCRATCH, MIXED)
        .unwrap();
    assert_eq!(image.hybrid_cuda_numeric_fp8_precision(), Some(MIXED));
    assert_eq!(
        base.with_hybrid_cuda_numeric_fp8_bf16(limits, 0)
            .unwrap()
            .hybrid_cuda_numeric_fp8_precision(),
        Some(NumericFp8Precision::Bf16ModelF32Accumulate)
    );
    for p in [
        NumericFp8Precision::Bf16ModelF32Accumulate,
        NumericFp8Precision::Bf16ModelTiledF32Accumulate,
        NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate,
    ] {
        assert_ne!(p.hybrid_backend_name(), MIXED.hybrid_backend_name());
    }
    assert_eq!(
        NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate
            .linear_plan(&R::LinearAttentionZ, layout(512, 2048), 1, SCRATCH)
            .unwrap()
            .selected_kernel(),
        K::Bf16TiledSlicedK2Gemm
    );
}

#[test]
fn mixed_gemv_guard_and_persistent_charge_do_not_leak_to_other_roles() {
    let p = MIXED
        .linear_plan(&R::AttentionOutput, layout(2048, 4096), 1, SCRATCH)
        .unwrap();
    assert_eq!(
        p.provider_bytes(),
        ferrule_backend::cuda_plan::NUMERIC_BF16_GEMV_PROVIDER_BYTES
    );
    assert!(
        MIXED
            .linear_plan(
                &R::AttentionOutput,
                layout(2048, 4096),
                1,
                ferrule_backend::cuda_plan::NUMERIC_BF16_GEMV_SCRATCH_BYTES - 1
            )
            .is_err()
    );
    for l in [
        layout(2047, 4096),
        layout(2048, 4095),
        NumericFp8Layout {
            row_origin: 128,
            ..layout(2048, 4096)
        },
        NumericFp8Layout {
            column_origin: 128,
            ..layout(2048, 4096)
        },
        NumericFp8Layout {
            scale_type: NumericFp8ScaleType::F32,
            ..layout(2048, 4096)
        },
    ] {
        let p = MIXED
            .linear_plan(&R::AttentionOutput, l, 1, SCRATCH)
            .unwrap();
        assert_eq!(p.selected_kernel(), K::Bf16TiledGemm);
        assert_eq!(p.provider_bytes(), 0);
    }
    for role in [
        R::SharedExpertDown,
        R::RoutedExpertDown,
        R::OutputHead,
        R::AttentionQuery,
    ] {
        let p = MIXED
            .linear_plan(&role, layout(2048, 4096), 1, SCRATCH)
            .unwrap();
        assert_ne!(p.selected_kernel(), K::Bf16M1Gemv);
        assert_eq!(p.provider_bytes(), 0);
    }
}

#[path = "numeric_mixed_kv.rs"]
mod kv;

#[test]
fn mixed_qkv_rows_preserve_tiled_budget_and_plan_identity() {
    let l = layout(8192, 2048);
    for m in [1, 5, 9] {
        for tile in [1, 64, 257, 1024, 2048 - m] {
            let budget = (m + tile) * 2048 * 2;
            let plan = MIXED
                .linear_plan(&R::LinearAttentionQkv, l, m, budget)
                .unwrap();
            let tiled = NumericFp8Precision::Bf16ModelTiledF32Accumulate
                .linear_plan(&R::LinearAttentionQkv, l, m, budget)
                .unwrap();
            assert_eq!(
                plan.selected_kernel(),
                if m == 1 {
                    K::Bf16TiledGemm
                } else {
                    K::Bf16TiledSlicedK2Gemm
                }
            );
            assert_eq!(plan.tile_rows(), tile);
            assert_eq!(plan.activation_bytes(), m * 2048 * 2);
            assert_eq!(
                plan.workspace_requirements(),
                tiled.workspace_requirements()
            );
            assert_eq!(
                plan.workspace_allocation_grant_bytes().unwrap(),
                tiled.workspace_allocation_grant_bytes().unwrap()
            );
            assert_eq!(
                plan.artifact_allocation_grant_bytes().unwrap(),
                tiled.artifact_allocation_grant_bytes().unwrap()
            );
            assert_eq!(plan.kernel_launches(), tiled.kernel_launches());
            assert_eq!(plan.provider_bytes(), 0);
            plan.validate_provider_budget(0).unwrap();
            assert_eq!(plan == tiled, m == 1);
            assert_eq!(plan.precision() == tiled.precision(), m == 1);
        }
        let minimum = (m + 1) * 2048 * 2;
        assert!(
            MIXED
                .linear_plan(&R::LinearAttentionQkv, l, m, minimum - 1)
                .is_err()
        );
        assert!(MIXED.linear_plan(&R::LinearAttentionQkv, l, m, 0).is_err());
    }
}

#[test]
fn mixed_qkv_guards_do_not_change_other_roles_or_profiles() {
    for m in [1, 5, 9] {
        for l in [
            layout(8192, 2047),
            layout(8192, 2112),
            layout(32, 16),
            NumericFp8Layout {
                scale_type: NumericFp8ScaleType::F32,
                ..layout(8192, 2048)
            },
            NumericFp8Layout {
                row_origin: 128,
                ..layout(8192, 2048)
            },
            NumericFp8Layout {
                column_origin: 128,
                ..layout(8192, 2048)
            },
        ] {
            assert_eq!(
                MIXED
                    .linear_plan(&R::LinearAttentionQkv, l, m, SCRATCH)
                    .unwrap()
                    .selected_kernel(),
                K::Bf16TiledGemm
            );
        }
        for precision in [
            NumericFp8Precision::Bf16RneF32Accumulate,
            NumericFp8Precision::Bf16ModelTiledF32Accumulate,
            NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate,
        ] {
            assert_eq!(
                precision
                    .linear_plan(&R::LinearAttentionQkv, layout(8192, 2048), m, SCRATCH)
                    .unwrap()
                    .selected_kernel(),
                K::Bf16TiledGemm
            );
        }
    }
    for m in [5, 9] {
        for role in [
            R::AttentionQuery,
            R::AttentionKey,
            R::AttentionValue,
            R::LinearAttentionZ,
            R::AttentionOutput,
            R::SharedExpertGate,
            R::SharedExpertUp,
            R::SharedExpertDown,
            R::RoutedExpertGate,
            R::RoutedExpertUp,
            R::RoutedExpertDown,
            R::RouterLogits,
            R::OutputHead,
        ] {
            assert_eq!(
                MIXED
                    .linear_plan(&role, layout(8192, 2048), m, SCRATCH)
                    .unwrap()
                    .selected_kernel(),
                K::Bf16TiledGemm
            );
        }
    }
    // Selection is semantic, not a hardcoded output width or layer index.
    assert_eq!(
        MIXED
            .linear_plan(&R::LinearAttentionQkv, layout(512, 2048), 5, SCRATCH)
            .unwrap()
            .selected_kernel(),
        K::Bf16TiledSlicedK2Gemm
    );
    for (m, k) in [(0, 2048), (5, 0)] {
        assert!(
            MIXED
                .linear_plan(&R::LinearAttentionQkv, layout(8192, k), m, SCRATCH)
                .is_err()
        );
    }
}
