use super::*;
use crate::support::TensorRole;
use ferrule_backend::cuda_plan::{
    NumericFp8Layout, NumericFp8ScaleType, NumericFp8SelectedKernel as K,
};

#[test]
fn semantic_gate_schedule_is_explicit_role_shape_based_and_defaults_stay_tiled() {
    use NumericFp8Precision as P;
    let scheduled = P::Bf16ModelSlicedK2F32Accumulate;
    for (role, m, n, k, expected) in [
        (
            TensorRole::LinearAttentionZ,
            1,
            4096,
            2048,
            K::Bf16TiledSlicedK2Gemm,
        ),
        (
            TensorRole::LinearAttentionZ,
            1,
            128,
            64,
            K::Bf16TiledSlicedK2Gemm,
        ),
        (
            TensorRole::LinearAttentionZ,
            5,
            4096,
            2048,
            K::Bf16TiledGemm,
        ),
        (TensorRole::LinearAttentionZ, 1, 16, 16, K::Bf16TiledGemm),
        (
            TensorRole::LinearAttentionQkv,
            1,
            8192,
            2048,
            K::Bf16TiledGemm,
        ),
        (TensorRole::RouterLogits, 1, 256, 2048, K::Bf16TiledGemm),
        (TensorRole::RoutedExpertGate, 1, 512, 2048, K::Bf16TiledGemm),
        (TensorRole::SharedExpertGate, 1, 512, 2048, K::Bf16TiledGemm),
        (TensorRole::OutputHead, 1, 248320, 2048, K::Bf16TiledGemm),
    ] {
        let layout = NumericFp8Layout {
            n,
            k,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        };
        let plan = scheduled.linear_plan(&role, layout, m, 8 << 20).unwrap();
        let tiled = P::Bf16ModelTiledF32Accumulate
            .linear_plan(&role, layout, m, 8 << 20)
            .unwrap();
        assert_eq!(plan.selected_kernel(), expected);
        assert_eq!(tiled.selected_kernel(), K::Bf16TiledGemm);
        assert_eq!(
            plan.workspace_requirements(),
            tiled.workspace_requirements()
        );
        assert_eq!(
            plan.workspace_allocation_grant_bytes().unwrap(),
            tiled.workspace_allocation_grant_bytes().unwrap()
        );
        assert_eq!(plan.kernel_launches(), tiled.kernel_launches());
        assert_ne!(
            scheduled.hybrid_backend_name(),
            P::Bf16ModelTiledF32Accumulate.hybrid_backend_name()
        );
        assert!(scheduled.linear_plan(&role, layout, m, 0).is_err());
    }
}
