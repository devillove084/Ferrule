use super::NumericFp8Precision as P;
use crate::support::TensorRole as R;
use ferrule_backend::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};
const MIXED: P = P::Bf16ModelMixedF32Accumulate;
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
fn gemmex_roles_rows_profiles_and_full_weight_budget() {
    for m in [1, 2, 5, 8, 16, 32] {
        for role in [
            R::LinearAttentionQkv,
            R::LinearAttentionZ,
            R::AttentionQuery,
            R::AttentionKey,
            R::AttentionValue,
            R::AttentionOutput,
            R::SharedExpertGate,
            R::SharedExpertUp,
            R::SharedExpertDown,
            R::RoutedExpertDown,
        ] {
            let p = MIXED
                .gemmex_plan(&role, layout(8192, 2048), m, usize::MAX, usize::MAX)
                .unwrap()
                .unwrap();
            assert_eq!(p.staging_bytes(), 2 * ((m + 8192) * 2048 + m * 8192));
            assert_eq!(p.storage_bytes(), p.staging_bytes() + (32 << 20));
            assert_eq!(p.provider_bytes(), 16 << 20);
            assert!(
                MIXED
                    .gemmex_plan(&role, p.layout(), m, p.storage_grant_bytes() - 1, 16 << 20)
                    .is_err()
            );
            assert!(
                MIXED
                    .gemmex_plan(
                        &role,
                        p.layout(),
                        m,
                        p.storage_grant_bytes(),
                        (16 << 20) - 1
                    )
                    .is_err()
            );
        }
    }
    assert_eq!(
        MIXED
            .gemmex_plan(
                &R::LinearAttentionQkv,
                layout(8192, 2048),
                5,
                usize::MAX,
                usize::MAX
            )
            .unwrap()
            .unwrap()
            .storage_bytes(),
        67_211_264
    );
    for p in [
        P::default(),
        P::F32Tf32x3,
        P::W8A16Bf16F32Accumulate,
        P::Bf16ModelF32Accumulate,
        P::Bf16ModelTiledF32Accumulate,
        P::Bf16ModelSlicedK2F32Accumulate,
    ] {
        assert!(
            p.gemmex_plan(&R::LinearAttentionQkv, layout(8192, 2048), 5, 0, 0)
                .unwrap()
                .is_none()
        );
    }
}
#[test]
fn gemmex_fallback_excludes_vocab_router_and_separate_routed_halves() {
    for r in [
        R::OutputHead,
        R::RouterLogits,
        R::RoutedExpertGate,
        R::RoutedExpertUp,
    ] {
        assert!(
            MIXED
                .gemmex_plan(&r, layout(262144, 2048), 5, 0, 0)
                .unwrap()
                .is_none()
        );
    }
    for l in [
        layout(127, 128),
        layout(128, 127),
        layout(0, 128),
        NumericFp8Layout {
            row_origin: 128,
            ..layout(128, 128)
        },
        NumericFp8Layout {
            column_origin: 128,
            ..layout(128, 128)
        },
        NumericFp8Layout {
            scale_type: NumericFp8ScaleType::F32,
            ..layout(128, 128)
        },
    ] {
        assert!(
            MIXED
                .gemmex_plan(&R::AttentionOutput, l, 5, 0, 0)
                .unwrap()
                .is_none()
        );
    }
    for m in [0, 33, usize::MAX] {
        assert!(
            MIXED
                .gemmex_plan(&R::AttentionOutput, layout(128, 128), m, 0, 0)
                .unwrap()
                .is_none()
        );
    }
}
#[test]
fn gemmex_native_concat_checks_layout_and_never_charges_two_handles() {
    let l = layout(512, 2048);
    let p = MIXED
        .gemmex_gate_up_plan(l, l, 2, usize::MAX, usize::MAX)
        .unwrap()
        .unwrap();
    assert_eq!(p.layout(), layout(1024, 2048));
    assert_eq!(p.provider_bytes(), 16 << 20);
    assert_eq!(
        p.storage_bytes(),
        2 * ((2 + 1024) * 2048 + 2 * 1024) + (32 << 20)
    );
    for other in [
        layout(256, 2048),
        layout(512, 1024),
        NumericFp8Layout {
            row_origin: 128,
            ..l
        },
        NumericFp8Layout {
            scale_type: NumericFp8ScaleType::F32,
            ..l
        },
    ] {
        assert!(
            MIXED
                .gemmex_gate_up_plan(l, other, 2, usize::MAX, usize::MAX)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn resident_control_is_mixed_only_full_budget_and_never_vocab() {
    for m in [1, 5, 32] {
        let p = MIXED
            .resident_bf16_gemmex_plan(&R::RouterLogits, 256, 2048, m, usize::MAX, usize::MAX)
            .unwrap()
            .unwrap();
        assert!(
            MIXED
                .resident_bf16_gemmex_plan(
                    &R::RouterLogits,
                    256,
                    2048,
                    m,
                    p.storage_grant_bytes() - 1,
                    16 << 20
                )
                .is_err()
        );
        for profile in [
            P::default(),
            P::Bf16ModelF32Accumulate,
            P::Bf16ModelTiledF32Accumulate,
            P::Bf16ModelSlicedK2F32Accumulate,
        ] {
            assert!(
                profile
                    .resident_bf16_gemmex_plan(&R::RouterLogits, 256, 2048, m, 0, 0)
                    .unwrap()
                    .is_none()
            );
        }
    }
    for role in [R::OutputHead, R::AttentionQuery] {
        assert!(
            MIXED
                .resident_bf16_gemmex_plan(&role, 248320, 2048, 1, 0, 0)
                .unwrap()
                .is_none()
        );
    }
    for (n, k, m) in [(0, 2048, 1), (256, 0, 1), (256, 2048, 33)] {
        assert!(
            MIXED
                .resident_bf16_gemmex_plan(&R::RouterLogits, n, k, m, 0, 0)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn resident_ab_tail_plans_share_capacity_without_relaxing_numeric_fp8() {
    for role in [
        R::LinearAttentionA,
        R::LinearAttentionBeta,
        R::SharedExpertOutputGate,
    ] {
        for rows in [1, 5, 32] {
            for n in [1, 3, 7, 8, 31, 32, 40, 136] {
                let plan = MIXED
                    .resident_bf16_gemmex_plan(&role, n, 2048, rows, usize::MAX, usize::MAX)
                    .unwrap()
                    .unwrap();
                assert!(plan.is_resident());
                assert_eq!(plan.staging_bytes() % 256, 0);
                assert!(plan.storage_grant_bytes() <= 69_206_016);
                assert!(
                    P::Bf16ModelF32Accumulate
                        .resident_bf16_gemmex_plan(&role, n, 2048, rows, 0, 0)
                        .unwrap()
                        .is_none()
                );
                assert!(
                    MIXED
                        .resident_bf16_gemmex_plan(
                            &role,
                            n,
                            2048,
                            rows,
                            plan.storage_grant_bytes() - 1,
                            16 << 20
                        )
                        .is_err()
                );
            }
        }
    }
}
