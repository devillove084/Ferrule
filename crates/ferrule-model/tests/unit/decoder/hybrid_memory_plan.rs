use super::*;
fn route(rows: usize, top_k: usize, hidden: usize, bucket: usize) -> Result<usize> {
    SwiGluScratchPlan::route(rows, top_k, hidden, bucket)?.total_bytes()
}

#[test]
fn qwen_bucket_table_and_gpu_indices_are_charged_exactly() {
    let rows = 32;
    let top_k = 8;
    let hidden = 2048;
    let table = rows * top_k * hidden * 4;
    assert_eq!(table, 2 * 1024 * 1024);
    let expected = table + 2 * rows * hidden * 4 + (2 * rows * top_k + 2 * rows) * 4;
    assert_eq!(route(rows, top_k, hidden, rows).unwrap(), expected);
    assert_eq!(
        route(rows, top_k, hidden, 7).unwrap(),
        expected - 2 * (rows - 7) * 4
    );
    assert_eq!(
        route(rows, top_k + 1, hidden, rows).unwrap() - expected,
        rows * (hidden + 2) * 4
    );
}

#[test]
fn bucket_geometry_overflow_and_i32_abi_fail_without_cuda() {
    for (rows, top_k, hidden, bucket) in [
        (usize::MAX, 2, 1, 0),
        (2, 1, usize::MAX, 0),
        (i32::MAX as usize, 2, 1, 0),
        (1, 1, i32::MAX as usize + 1, 0),
        (1, 0, 1, 0),
        (1, 1, 0, 0),
        (1, 1, 8, 2),
    ] {
        assert!(route(rows, top_k, hidden, bucket).is_err());
    }
    assert!(add(usize::MAX, 1).is_err());
    assert!(mul(usize::MAX, 4).is_err());
}

#[test]
fn numeric_abi_plans_have_feature_independent_exact_values() {
    use ferrule_backend::cuda_plan::{NumericFp8Precision as P, NumericFp8SelectedKernel as K};
    let layout = NumericFp8Layout {
        n: 257,
        k: 130,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    for (precision, bytes, activation, tile, tiles, launches, kernel) in [
        (
            P::Bf16RneF32Accumulate,
            4080,
            2448,
            6,
            43,
            87,
            K::Bf16TiledGemm,
        ),
        (P::F32Tf32x3, 3640, 0, 7, 37, 74, K::F32TiledTf32x3),
        (
            P::W8A16Bf16F32Accumulate,
            0,
            0,
            0,
            0,
            1,
            K::W8A16FusedBf16Mma,
        ),
    ] {
        let plan = NumericFp8LinearPlan::new(layout, 9, 4096, precision).unwrap();
        assert_eq!(plan.workspace_requirements().bytes, bytes);
        assert_eq!(plan.workspace_requirements().alignment, 16);
        assert_eq!(plan.activation_bytes(), activation);
        assert_eq!(plan.tile_rows(), tile);
        assert_eq!(plan.tile_count(), tiles);
        assert_eq!(plan.kernel_launches(), launches);
        assert_eq!(plan.selected_kernel(), kernel);
        assert_eq!(plan.artifact_allocation_grant_bytes().unwrap(), 65_792);
        assert_eq!(
            plan.workspace_allocation_grant_bytes().unwrap(),
            if bytes == 0 { 0 } else { 4096 }
        );
        #[cfg(feature = "cuda")]
        {
            let execution_plan =
                ferrule_backend::cuda::operators::linear::NumericFp8LinearPlan::new(
                    layout, 9, 4096, precision,
                )
                .unwrap();
            assert_eq!(plan, execution_plan);
        }
    }
    assert_eq!(
        NumericFp8LinearPlan::new(layout, 1, 0, P::F32Tf32x3)
            .unwrap()
            .selected_kernel(),
        K::F32FusedM1
    );
    assert_eq!(
        NumericFp8LinearPlan::new(layout, 8, 0, P::W8A16Bf16F32Accumulate)
            .unwrap()
            .selected_kernel(),
        K::W8A16FusedBf16Vector
    );
    assert!(NumericFp8LinearPlan::new(layout, 9, 2719, P::Bf16RneF32Accumulate).is_err());
    assert!(NumericFp8LinearPlan::new(layout, usize::MAX, 4096, P::F32Tf32x3).is_err());
    assert!(
        NumericFp8LinearPlan::new(
            NumericFp8Layout {
                k: usize::MAX,
                ..layout
            },
            1,
            0,
            P::F32Tf32x3
        )
        .is_err()
    );
}

#[test]
fn recurrent_abi_geometry_rejects_overflow_without_an_owner() {
    let conv = CausalConv1dLayout {
        rows: 32,
        channels: 10,
        kernel_size: 3,
    };
    assert_eq!(conv.row_elements().unwrap(), 320);
    assert_eq!(conv.history_elements().unwrap(), 30);
    let delta = GatedDeltaNetLayout {
        rows: 32,
        key_heads: 1,
        value_heads: 2,
        key_dim: 3,
        value_dim: 2,
    };
    assert_eq!(delta.qkv_elements().unwrap(), 320);
    assert_eq!(delta.state_elements().unwrap(), 12);
    assert_eq!(delta.gate_elements().unwrap(), 64);
    assert_eq!(delta.output_elements().unwrap(), 128);
    for rows in [0, usize::MAX, i32::MAX as usize] {
        assert!(CausalConv1dLayout { rows, ..conv }.validate().is_err());
        assert!(GatedDeltaNetLayout { rows, ..delta }.validate().is_err());
    }
    assert!(
        GatedDeltaNetLayout {
            key_heads: 0,
            ..delta
        }
        .validate()
        .is_err()
    );
    assert!(
        GatedDeltaNetLayout {
            key_heads: 3,
            ..delta
        }
        .validate()
        .is_err()
    );
}

#[test]
fn legacy_budget_reexport_is_the_same_metadata_type() {
    let requirements = HybridReservationRequirements {
        kv_pages: 2,
        state_bytes: 3,
        weight_bytes: 5,
        workspace_bytes: 7,
    };
    let legacy: crate::decoder::HybridCudaMemoryBudget = requirements;
    let public: crate::decoder::HybridReservationRequirements = legacy;
    assert_eq!(public, requirements);
}
