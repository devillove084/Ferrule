use super::*;

#[test]
fn mixed_kv_m1_is_explicit_fused_with_no_extra_scratch_or_provider_reserve() {
    for role in [R::AttentionKey, R::AttentionValue] {
        let l = layout(512, 2048);
        let p = MIXED.linear_plan(&role, l, 1, SCRATCH).unwrap();
        assert_eq!(p.selected_kernel(), K::W8A16FusedBf16Vector);
        assert_eq!(p.workspace_requirements().bytes, 0);
        assert_eq!(p.provider_bytes(), 0);
        assert_eq!(p.workspace_allocation_grant_bytes().unwrap(), 0);
        let old = NumericFp8Precision::Bf16ModelTiledF32Accumulate
            .linear_plan(&role, l, 1, SCRATCH)
            .unwrap();
        assert_eq!(old.selected_kernel(), K::Bf16TiledGemm);
        assert!(p.workspace_requirements().bytes <= old.workspace_requirements().bytes);
        for precision in [
            NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate,
            NumericFp8Precision::Bf16RneF32Accumulate,
        ] {
            assert_eq!(
                precision
                    .linear_plan(&role, l, 1, SCRATCH)
                    .unwrap()
                    .selected_kernel(),
                K::Bf16TiledGemm
            );
        }
        for m in [2, 5, 8] {
            let p = MIXED.linear_plan(&role, l, m, SCRATCH).unwrap();
            assert_eq!(p.selected_kernel(), K::Bf16TiledGemm);
            assert_eq!(
                p,
                NumericFp8Precision::Bf16ModelTiledF32Accumulate
                    .linear_plan(&role, l, m, SCRATCH)
                    .unwrap()
            );
        }
    }
}

#[test]
fn mixed_kv_guard_rejects_generalization_to_shapes_scales_origins_or_other_roles() {
    for role in [R::AttentionKey, R::AttentionValue] {
        for l in [
            layout(511, 2048),
            layout(512, 2047),
            layout(1024, 2048),
            NumericFp8Layout {
                row_origin: 128,
                ..layout(512, 2048)
            },
            NumericFp8Layout {
                column_origin: 128,
                ..layout(512, 2048)
            },
            NumericFp8Layout {
                scale_type: NumericFp8ScaleType::F32,
                ..layout(512, 2048)
            },
        ] {
            assert_eq!(
                MIXED
                    .linear_plan(&role, l, 1, SCRATCH)
                    .unwrap()
                    .selected_kernel(),
                K::Bf16TiledGemm
            );
        }
    }
    for role in [R::AttentionQuery, R::RouterLogits, R::OutputHead] {
        assert_eq!(
            MIXED
                .linear_plan(&role, layout(512, 2048), 1, SCRATCH)
                .unwrap()
                .selected_kernel(),
            K::Bf16TiledGemm
        );
    }
    assert_eq!(
        MIXED
            .linear_plan(&R::SharedExpertDown, layout(2048, 512), 1, SCRATCH)
            .unwrap()
            .selected_kernel(),
        K::W8A16FusedBf16Vector
    );
    assert_eq!(
        MIXED
            .linear_plan(&R::AttentionOutput, layout(2048, 4096), 1, SCRATCH)
            .unwrap()
            .selected_kernel(),
        K::Bf16M1Gemv
    );
}
