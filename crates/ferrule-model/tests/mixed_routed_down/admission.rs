//! Candidate-only CPU admission gates. No CUDA discovery or tolerance comparisons.
use super::*;
use ferrule_backend::cuda_plan::{
    NumericFp8Layout, NumericFp8Precision as Precision, NumericFp8ScaleType,
    NumericFp8SelectedKernel,
};

fn block(
    degree: usize,
    math: HybridMixedProjectionArithmetic,
) -> (HybridMixedBlockPlan, Vec<BoundParameter>) {
    let layout = Arc::new(
        plan(HybridRankPrecision::MixedBf16F32, degree)
            .rank_layout(degree - 1)
            .unwrap(),
    );
    let fixtures = (0..4)
        .flat_map(|expert| {
            [
                TensorRole::RoutedExpertGate,
                TensorRole::RoutedExpertUp,
                TensorRole::RoutedExpertDown,
            ]
            .into_iter()
            .enumerate()
            .map(move |(r, role)| {
                Fixture::with_data(
                    role,
                    &[256, 256],
                    ParameterResidency::Expert { layer: 0, expert },
                    CheckpointDType::F8E4M3,
                    Some(CheckpointDType::Bf16),
                    (expert * 3 + r + 1) as u64,
                    None,
                )
            })
        })
        .collect::<Vec<_>>();
    let materializer = StateDictMaterializer::new(1 << 20).unwrap();
    let prepared = fixtures
        .iter()
        .map(|f| {
            materializer
                .prepare_hybrid_rank_parameter(&f.binding, layout.clone(), usize::MAX)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let block = HybridMixedBlockPlan::new(
        layout,
        &prepared,
        8,
        math,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
    .unwrap();
    (
        block,
        fixtures
            .iter()
            .map(|fixture| fixture.binding.clone())
            .collect(),
    )
}

fn shape() -> NumericFp8Layout {
    NumericFp8Layout {
        n: 256,
        k: 128,
        row_origin: 0,
        column_origin: 128,
        scale_type: NumericFp8ScaleType::Bf16,
    }
}

#[test]
fn explicit_opt_in_exact_capability_and_native_fallback_reserves() {
    let (native, bindings) = block(2, HybridMixedProjectionArithmetic::NativeBf16GemmEx);
    let down = &bindings[2];
    assert!(
        native
            .routed_down_m1_candidate_plan(down, shape(), 1, 1)
            .is_none()
    );
    assert_eq!(native.candidate_accumulator_grant_bytes(), 0);
    let candidate = native
        .clone()
        .with_routed_down_m1_candidate(4 * grant(256 * 4))
        .unwrap();
    assert_eq!(
        candidate.candidate_accumulator_grant_bytes(),
        4 * grant(256 * 4)
    );
    assert!(
        native
            .clone()
            .with_routed_down_m1_candidate(4 * grant(256 * 4) - 1)
            .is_err()
    );
    assert_eq!(
        candidate.arithmetic(),
        HybridMixedProjectionArithmetic::NativeBf16GemmEx
    );
    assert_eq!(candidate.weight_grant_bytes(), native.weight_grant_bytes());
    assert_eq!(candidate.weight_bytes(), native.weight_bytes());
    assert_eq!(
        candidate.workspace_grant_bytes(),
        native.workspace_grant_bytes()
    );
    assert_eq!(candidate.provider_bytes(), native.provider_bytes());
    for column_origin in [0, 128] {
        let p = candidate
            .routed_down_m1_candidate_plan(
                down,
                NumericFp8Layout {
                    column_origin,
                    ..shape()
                },
                1,
                1,
            )
            .unwrap();
        assert_eq!(p.layout().column_origin, 0);
        assert_eq!(p.precision(), Precision::W8A16Bf16F32Accumulate);
        assert_eq!(p.rows(), 1);
        assert_eq!(
            p.selected_kernel(),
            NumericFp8SelectedKernel::W8A16FusedBf16Vector
        );
        assert_eq!(
            p.selected_kernel().name(),
            "numeric-w8a16-fused-bf16-vector"
        );
        assert_eq!(p.kernel_launches(), 1); // numeric call only; BF16 writeback is separate
        assert_eq!(p.workspace_allocation_grant_bytes().unwrap(), 0);
        assert_eq!(p.activation_bytes(), 0);
        assert_eq!(p.weight_tile_bytes(), 0);
        assert_eq!(p.provider_bytes(), 0);
        assert_eq!(p.fused_shared_bytes_per_cta(), Some(0));
        let capability = candidate
            .routed_down_m1_candidate_capability(
                down,
                NumericFp8Layout {
                    column_origin,
                    ..shape()
                },
                1,
                1,
            )
            .unwrap();
        assert_eq!(capability.n, 256);
        assert_eq!(capability.k, 128);
        assert_eq!(capability.kernel, p.selected_kernel());
        assert_eq!(capability.kernel_name, p.selected_kernel().name());
        assert_eq!(capability.launches, 1);
        assert_eq!(capability.workspace_bytes, 0);
        assert_eq!(capability.provider_bytes, 0);
        assert_eq!(capability.fused_shared_bytes_per_cta, Some(0));
        assert_eq!(capability.minimum_sm, 80);
        assert_eq!(capability.f32_accumulator_grants, grant(256 * 4));
        assert_eq!(capability.bf16_output_grants, grant(256 * 2));
        assert_eq!(capability.bf16_writeback_launches, 1);
        assert!(!capability.production_validated);
    }
}

#[test]
fn role_actual_invocation_rows_shape_scale_and_legacy_are_fail_closed() {
    let (native, bindings) = block(2, HybridMixedProjectionArithmetic::NativeBf16GemmEx);
    let down = &bindings[2];
    let candidate = native.with_routed_down_m1_candidate(usize::MAX).unwrap();
    for role in [
        TensorRole::RoutedExpertGate,
        TensorRole::RoutedExpertUp,
        TensorRole::SharedExpertGate,
        TensorRole::SharedExpertUp,
        TensorRole::SharedExpertDown,
        TensorRole::DenseMlpGate,
        TensorRole::DenseMlpUp,
        TensorRole::DenseMlpDown,
        TensorRole::RouterLogits,
        TensorRole::SharedExpertOutputGate,
        TensorRole::AttentionQuery,
        TensorRole::AttentionKey,
        TensorRole::AttentionValue,
        TensorRole::AttentionOutput,
        TensorRole::LinearAttentionQkv,
        TensorRole::LinearAttentionZ,
        TensorRole::LinearAttentionA,
        TensorRole::LinearAttentionBeta,
        TensorRole::OutputHead,
    ] {
        let foreign_role = Fixture::with_data(
            role.clone(),
            &[256, 256],
            ParameterResidency::Expert {
                layer: 0,
                expert: 0,
            },
            CheckpointDType::F8E4M3,
            Some(CheckpointDType::Bf16),
            3,
            None,
        );
        assert!(
            candidate
                .routed_down_m1_candidate_plan(&foreign_role.binding, shape(), 1, 1)
                .is_none(),
            "{role:?}"
        );
    }
    let foreign_layer = Fixture::with_data(
        TensorRole::RoutedExpertDown,
        &[256, 256],
        ParameterResidency::Expert {
            layer: 1,
            expert: 0,
        },
        CheckpointDType::F8E4M3,
        Some(CheckpointDType::Bf16),
        3,
        None,
    );
    assert!(
        candidate
            .routed_down_m1_candidate_plan(&foreign_layer.binding, shape(), 1, 1)
            .is_none(),
        "same shape from another layer must not authorize the candidate"
    );

    for (invocation, projection) in [
        (0, 1),
        (1, 0),
        (2, 1),
        (5, 1),
        (8, 1),
        (1, 2),
        (2, 2),
        (32, 1),
    ] {
        assert!(
            candidate
                .routed_down_m1_candidate_plan(down, shape(), invocation, projection)
                .is_none()
        );
    }
    for bad in [
        NumericFp8Layout { n: 128, ..shape() },
        NumericFp8Layout { k: 256, ..shape() },
        NumericFp8Layout { k: 129, ..shape() },
        NumericFp8Layout {
            row_origin: 1,
            ..shape()
        },
        NumericFp8Layout {
            column_origin: 129,
            ..shape()
        },
        NumericFp8Layout {
            scale_type: NumericFp8ScaleType::F32,
            ..shape()
        },
        NumericFp8Layout {
            k: usize::MAX,
            ..shape()
        },
    ] {
        assert!(
            candidate
                .routed_down_m1_candidate_plan(down, bad, 1, 1)
                .is_none()
        );
    }
    assert!(
        block(1, HybridMixedProjectionArithmetic::NativeBf16GemmEx)
            .0
            .with_routed_down_m1_candidate(usize::MAX)
            .is_err()
    );
    assert!(
        block(2, HybridMixedProjectionArithmetic::W8A16Bf16)
            .0
            .with_routed_down_m1_candidate(usize::MAX)
            .is_err()
    );
}

#[test]
fn full_budget_includes_candidate_outputs_and_owner_margins_at_22gib() {
    let (native, _) = block(2, HybridMixedProjectionArithmetic::NativeBf16GemmEx);
    let candidate = native.with_routed_down_m1_candidate(usize::MAX).unwrap();
    let estimate = HybridMixedRankOperationEstimate {
        storage_grants: candidate.weight_grant_bytes(),
        gemmex_storage_grants: candidate.workspace_grant_bytes(),
        gemmex_provider_bytes: candidate.provider_bytes(),
        attention_storage_grants: 0,
        attention_provider_bytes: 0,
        derived_rope_grants: 0,
        attention_metadata_grants: 0,
        recurrent_workspace_grants: 0,
        state_copy_workspace_grants: 0,
        operation_grants: candidate.candidate_accumulator_grant_bytes(),
        boundary_device_grants: 0,
        pinned_host_bytes: 0,
        host_staging_bytes: 0,
    };
    let total = estimate.total_device_grants().unwrap();
    assert!(
        estimate
            .require_device_capacity(22 << 30, 128 << 20)
            .is_ok()
    );
    assert_eq!(
        estimate
            .require_device_capacity(total + (128 << 20), 128 << 20)
            .unwrap(),
        total + (128 << 20)
    );
    assert!(
        estimate
            .require_device_capacity(total + (128 << 20) - 1, 128 << 20)
            .is_err()
    );
    let mut over = estimate;
    over.operation_grants = 22 << 30;
    assert!(over.require_device_capacity(22 << 30, 128 << 20).is_err());
    over.operation_grants = usize::MAX;
    assert!(over.total_device_grants().is_err());
}

#[test]
fn same_id_role_shape_but_foreign_layer_expert_or_source_is_rejected() {
    let (native, bindings) = block(2, HybridMixedProjectionArithmetic::NativeBf16GemmEx);
    let candidate = native.with_routed_down_m1_candidate(usize::MAX).unwrap();
    let down = &bindings[2];
    assert!(
        candidate
            .routed_down_m1_candidate_plan(&down.clone(), shape(), 1, 1)
            .is_some()
    );
    for residency in [
        ParameterResidency::Expert {
            layer: 1,
            expert: 0,
        },
        ParameterResidency::Expert {
            layer: 0,
            expert: 1,
        },
        ParameterResidency::Expert {
            layer: 0,
            expert: 4,
        },
        ParameterResidency::Expert {
            layer: 0,
            expert: 0,
        },
        ParameterResidency::Layer { layer: 0 },
    ] {
        let foreign = Fixture::with_data(
            TensorRole::RoutedExpertDown,
            &[256, 256],
            residency,
            CheckpointDType::F8E4M3,
            Some(CheckpointDType::Bf16),
            3,
            None,
        );
        assert_eq!(foreign.binding.id(), down.id());
        assert_eq!(foreign.binding.role(), down.role());
        assert_eq!(
            foreign.binding.weight().logical_shape(),
            down.weight().logical_shape()
        );
        assert!(
            candidate
                .routed_down_m1_candidate_plan(&foreign.binding, shape(), 1, 1)
                .is_none()
        );
        assert!(
            candidate
                .routed_down_m1_candidate_capability(&foreign.binding, shape(), 1, 1)
                .is_none()
        );
    }
    for wrong_origin in [
        NumericFp8Layout {
            row_origin: 128,
            ..shape()
        },
        NumericFp8Layout {
            column_origin: 256,
            ..shape()
        },
    ] {
        assert!(
            candidate
                .routed_down_m1_candidate_plan(down, wrong_origin, 1, 1)
                .is_none()
        );
    }
}
