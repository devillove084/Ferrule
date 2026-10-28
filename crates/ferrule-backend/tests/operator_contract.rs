use ferrule_backend::cuda_plan::{
    NUMERIC_BF16_GEMV_SCRATCH_BYTES, NumericFp8Layout, NumericFp8LinearPlan, NumericFp8Precision,
};
use ferrule_backend::operator_contract::{
    AdvisoryStorageIdentity, AliasPolicy, CapturePolicy, CastSemantics, LinearCapability,
    LinearFusionPlan, LinearOperatorDescriptor, LinearStageKind, LogicalDType,
    OperatorContractError, QuantizationDescriptor, RetentionPolicy, ScaleOperand,
    TensorViewDescriptor, build_numeric_fp8_operator_contract,
};
use ferrule_common::numeric_fp8::NumericFp8ScaleType;

fn numeric_descriptor<'a>(
    input: TensorViewDescriptor<'a>,
    weight: TensorViewDescriptor<'a>,
    output: TensorViewDescriptor<'a>,
    scale: TensorViewDescriptor<'a>,
    layout: NumericFp8Layout,
) -> LinearOperatorDescriptor<'a> {
    let quantization = QuantizationDescriptor::from_numeric_fp8_layout(scale, layout).unwrap();
    LinearOperatorDescriptor::new(
        input,
        weight,
        output,
        Some(quantization),
        AliasPolicy::Disjoint,
        CapturePolicy::CaptureSafe,
        RetentionPolicy::UntilCompletion,
    )
}

#[test]
fn tensor_views_validate_dense_stride_offset_and_alignment() {
    let dims = [2, 3];
    let strides = [32, 4];
    let view = TensorViewDescriptor::new(LogicalDType::F32, &dims, &strides, 16, 4, None);

    view.validate().unwrap();
    assert_eq!(view.byte_range().unwrap(), 16..60);

    let bad_alignment = TensorViewDescriptor::new(LogicalDType::F32, &dims, &strides, 16, 2, None);
    assert!(matches!(
        bad_alignment.validate(),
        Err(OperatorContractError::InvalidField {
            field: "alignment",
            ..
        })
    ));

    let bad_offset = TensorViewDescriptor::new(LogicalDType::F32, &dims, &strides, 2, 4, None);
    assert!(matches!(
        bad_offset.validate(),
        Err(OperatorContractError::InvalidField {
            field: "byte_offset",
            ..
        })
    ));
}

#[test]
fn numeric_scale_descriptor_preserves_ragged_origin_and_grid() {
    let layout = NumericFp8Layout {
        n: 129,
        k: 257,
        row_origin: 7,
        column_origin: 11,
        scale_type: NumericFp8ScaleType::F32,
    };
    let scale_dims = [2, 3];
    let scale_strides = [12, 4];
    let scale =
        TensorViewDescriptor::new(LogicalDType::F32, &scale_dims, &scale_strides, 0, 4, None);

    let descriptor = QuantizationDescriptor::from_numeric_fp8_layout(scale, layout).unwrap();
    assert_eq!(descriptor.origin, [7, 11]);
    assert_eq!(descriptor.group_shape, [128, 128]);
    assert_eq!(descriptor.scale.dims, [2, 3]);

    let wrong_dims = [2, 2];
    let wrong_strides = [8, 4];
    let wrong_scale =
        TensorViewDescriptor::new(LogicalDType::F32, &wrong_dims, &wrong_strides, 0, 4, None);
    assert!(matches!(
        QuantizationDescriptor::from_numeric_fp8_layout(wrong_scale, layout),
        Err(OperatorContractError::ShapeMismatch {
            field: "scale grid",
            ..
        })
    ));
}

#[test]
fn numeric_scale_shape_errors_keep_layout_and_overflow_types() {
    let zero_layout = NumericFp8Layout {
        n: 0,
        k: 8,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let scale_dims = [1, 1];
    let scale_strides = [4, 4];
    let scale =
        TensorViewDescriptor::new(LogicalDType::F32, &scale_dims, &scale_strides, 0, 4, None);
    assert!(matches!(
        QuantizationDescriptor::from_numeric_fp8_layout(scale, zero_layout),
        Err(OperatorContractError::InvalidLayout {
            field: "numeric FP8 shape",
            ..
        })
    ));

    let overflow_layout = NumericFp8Layout {
        n: 1,
        k: 1,
        row_origin: usize::MAX,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    assert!(matches!(
        QuantizationDescriptor::from_numeric_fp8_layout(scale, overflow_layout),
        Err(OperatorContractError::ShapeOverflow {
            field: "row origin",
        })
    ));
}

#[test]
fn contract_rejects_overflow_and_linear_shape_mismatch() {
    let overflow_dims = [usize::MAX, 2];
    let overflow_strides = [4, 4];
    let overflow = TensorViewDescriptor::new(
        LogicalDType::F32,
        &overflow_dims,
        &overflow_strides,
        0,
        4,
        None,
    );
    assert!(matches!(
        overflow.validate(),
        Err(OperatorContractError::ShapeOverflow { .. })
    ));

    let input_dims = [2, 3];
    let input_strides = [12, 4];
    let weight_dims = [5, 4];
    let weight_strides = [16, 4];
    let output_dims = [2, 5];
    let output_strides = [20, 4];
    let descriptor = LinearOperatorDescriptor::new(
        TensorViewDescriptor::new(LogicalDType::F32, &input_dims, &input_strides, 0, 4, None),
        TensorViewDescriptor::new(LogicalDType::F32, &weight_dims, &weight_strides, 0, 4, None),
        TensorViewDescriptor::new(LogicalDType::F32, &output_dims, &output_strides, 0, 4, None),
        None,
        AliasPolicy::Disjoint,
        CapturePolicy::CaptureSafe,
        RetentionPolicy::UntilCompletion,
    );
    assert!(matches!(
        descriptor.validate(),
        Err(OperatorContractError::ShapeMismatch {
            field: "linear dimensions",
            ..
        })
    ));
}

#[test]
fn numeric_capability_rejects_unrepresentable_view_offset() {
    let layout = NumericFp8Layout {
        n: 4,
        k: 8,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    };
    let input_dims = [2, 8];
    let input_strides = [32, 4];
    let weight_dims = [4, 8];
    let weight_strides = [8, 1];
    let output_dims = [2, 4];
    let output_strides = [16, 4];
    let scale_dims = [1, 1];
    let scale_strides = [2, 2];
    let descriptor = numeric_descriptor(
        TensorViewDescriptor::new(LogicalDType::F32, &input_dims, &input_strides, 4, 4, None),
        TensorViewDescriptor::new(
            LogicalDType::E4M3Fn,
            &weight_dims,
            &weight_strides,
            0,
            1,
            None,
        ),
        TensorViewDescriptor::new(LogicalDType::F32, &output_dims, &output_strides, 0, 4, None),
        TensorViewDescriptor::new(LogicalDType::Bf16, &scale_dims, &scale_strides, 0, 2, None),
        layout,
    );

    descriptor.validate().unwrap();
    assert!(matches!(
        descriptor.validate_for(LinearCapability::NumericFp8),
        Err(OperatorContractError::Unsupported {
            capability: "numeric FP8 view offsets",
            ..
        })
    ));
}

#[test]
fn existing_numeric_plan_builds_a_valid_descriptor() {
    let layout = NumericFp8Layout {
        n: 129,
        k: 257,
        row_origin: 7,
        column_origin: 11,
        scale_type: NumericFp8ScaleType::F32,
    };
    let plan = NumericFp8LinearPlan::new(layout, 3, 0, NumericFp8Precision::W8A16Bf16F32Accumulate)
        .unwrap();
    let scale_grid = layout.scale_shape().unwrap();
    let input_dims = [plan.rows(), layout.k];
    let input_strides = [(layout.k + 5) * 2, 2];
    let weight_dims = [layout.n, layout.k];
    let weight_strides = [layout.k, 1];
    let output_dims = [plan.rows(), layout.n];
    let output_strides = [(layout.n + 3) * 4, 4];
    let scale_strides = [scale_grid[1] * 4, 4];
    let descriptor = numeric_descriptor(
        TensorViewDescriptor::new(LogicalDType::Bf16, &input_dims, &input_strides, 0, 2, None),
        TensorViewDescriptor::new(
            LogicalDType::E4M3Fn,
            &weight_dims,
            &weight_strides,
            0,
            1,
            None,
        ),
        TensorViewDescriptor::new(LogicalDType::F32, &output_dims, &output_strides, 0, 4, None),
        TensorViewDescriptor::new(LogicalDType::F32, &scale_grid, &scale_strides, 0, 4, None),
        layout,
    );

    descriptor
        .validate_for(LinearCapability::NumericFp8)
        .unwrap();
    assert_eq!(descriptor.input.dims, [3, 257]);
    assert_eq!(descriptor.output.dims, [3, 129]);
}

#[test]
fn production_builder_validates_f32_bf16_and_eager_gemv_contracts() {
    let layout = NumericFp8Layout {
        n: 129,
        k: 257,
        row_origin: 7,
        column_origin: 11,
        scale_type: NumericFp8ScaleType::F32,
    };
    let plan = NumericFp8LinearPlan::new(layout, 3, 0, NumericFp8Precision::W8A16Bf16F32Accumulate)
        .unwrap();
    let f32_contract = build_numeric_fp8_operator_contract(
        plan,
        LogicalDType::F32,
        layout.k + 5,
        layout.n + 3,
        |descriptor| {
            let policy = descriptor.capture_policy;
            descriptor.validate_for(LinearCapability::NumericFp8)?;
            let stages = descriptor.fusion_plan.stages();
            Ok((
                descriptor.input.dtype,
                policy,
                [stages[0], stages[1], stages[2], stages[3]],
            ))
        },
    )
    .unwrap();
    let bf16_contract = build_numeric_fp8_operator_contract(
        plan,
        LogicalDType::Bf16,
        layout.k + 5,
        layout.n + 3,
        |descriptor| {
            let policy = descriptor.capture_policy;
            descriptor.validate_for(LinearCapability::NumericFp8)?;
            let stages = descriptor.fusion_plan.stages();
            Ok((
                descriptor.input.dtype,
                policy,
                [stages[0], stages[1], stages[2]],
            ))
        },
    )
    .unwrap();
    assert_eq!(f32_contract.0, LogicalDType::F32);
    assert_eq!(f32_contract.1, CapturePolicy::CaptureSafe);
    assert_eq!(f32_contract.2[0].kind, LinearStageKind::Decode);
    assert_eq!(f32_contract.2[0].input_dtype, LogicalDType::E4M3Fn);
    assert_eq!(f32_contract.2[0].output_dtype, LogicalDType::Bf16);
    assert_eq!(f32_contract.2[0].scale_operand, ScaleOperand::Weight);
    assert_eq!(f32_contract.2[1].kind, LinearStageKind::Cast);
    assert_eq!(f32_contract.2[1].input_dtype, LogicalDType::F32);
    assert_eq!(f32_contract.2[1].output_dtype, LogicalDType::Bf16);
    assert_eq!(
        f32_contract.2[1].cast_semantics,
        CastSemantics::RoundToNearestEven
    );
    assert!(!f32_contract.2[1].materialized);
    assert_eq!(f32_contract.2[1].alias_policy, AliasPolicy::MayAlias);
    assert_eq!(f32_contract.2[2].kind, LinearStageKind::Linear);
    assert_eq!(f32_contract.2[2].input_dtype, LogicalDType::Bf16);
    assert_eq!(f32_contract.2[2].weight_dtype, Some(LogicalDType::Bf16));
    assert_eq!(f32_contract.2[3].kind, LinearStageKind::Accumulate);
    assert_eq!(f32_contract.2[3].output_dtype, LogicalDType::F32);
    assert!(f32_contract.2[3].materialized);
    assert_eq!(f32_contract.2[3].alias_policy, AliasPolicy::Disjoint);

    assert_eq!(bf16_contract.0, LogicalDType::Bf16);
    assert_eq!(bf16_contract.1, CapturePolicy::CaptureSafe);
    assert_eq!(bf16_contract.2[1].input_dtype, LogicalDType::Bf16);
    assert_eq!(bf16_contract.2[1].weight_dtype, Some(LogicalDType::Bf16));

    let f32_precision_plan = NumericFp8LinearPlan::new(
        layout,
        3,
        layout.n * layout.k * LogicalDType::F32.byte_width(),
        NumericFp8Precision::F32Tf32x3,
    )
    .unwrap();
    let f32_precision_stages = build_numeric_fp8_operator_contract(
        f32_precision_plan,
        LogicalDType::F32,
        layout.k,
        layout.n,
        |descriptor| {
            descriptor.validate_for(LinearCapability::NumericFp8)?;
            let stages = descriptor.fusion_plan.stages();
            Ok([stages[0], stages[1], stages[2]])
        },
    )
    .unwrap();
    assert_eq!(f32_precision_stages[0].output_dtype, LogicalDType::F32);
    assert_eq!(
        f32_precision_stages[1].weight_dtype,
        Some(LogicalDType::F32)
    );

    let f32_m1_plan =
        NumericFp8LinearPlan::new(layout, 1, 0, NumericFp8Precision::F32Tf32x3).unwrap();
    let f32_m1_stages = build_numeric_fp8_operator_contract(
        f32_m1_plan,
        LogicalDType::F32,
        layout.k,
        layout.n,
        |descriptor| {
            descriptor.validate_for(LinearCapability::NumericFp8)?;
            let stages = descriptor.fusion_plan.stages();
            Ok([stages[0], stages[1], stages[2]])
        },
    )
    .unwrap();
    assert_eq!(f32_m1_stages[0].output_dtype, LogicalDType::F32);
    assert!(!f32_m1_stages[0].materialized);
    assert_eq!(f32_m1_stages[0].alias_policy, AliasPolicy::MayAlias);
    assert_eq!(f32_m1_stages[1].kind, LinearStageKind::Linear);
    assert_eq!(f32_m1_stages[1].input_dtype, LogicalDType::F32);
    assert_eq!(f32_m1_stages[1].weight_dtype, Some(LogicalDType::F32));
    assert_eq!(f32_m1_stages[2].kind, LinearStageKind::Accumulate);

    let gemv_layout = NumericFp8Layout {
        n: 2048,
        k: 4096,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    };
    let gemv_plan = NumericFp8LinearPlan::new(
        gemv_layout,
        1,
        NUMERIC_BF16_GEMV_SCRATCH_BYTES,
        NumericFp8Precision::Bf16M1GemvF32Accumulate,
    )
    .unwrap();
    let gemv_contract = build_numeric_fp8_operator_contract(
        gemv_plan,
        LogicalDType::Bf16,
        gemv_layout.k,
        gemv_layout.n,
        |descriptor| {
            let policy = descriptor.capture_policy;
            descriptor.validate_for(LinearCapability::NumericFp8)?;
            let stages = descriptor.fusion_plan.stages();
            Ok((policy, [stages[0], stages[1], stages[2]]))
        },
    )
    .unwrap();
    assert_eq!(gemv_contract.0, CapturePolicy::EagerOnly);
    assert_eq!(gemv_contract.1[0].kind, LinearStageKind::Decode);
    assert_eq!(gemv_contract.1[0].output_dtype, LogicalDType::Bf16);
    assert!(gemv_contract.1[0].materialized);
    assert_eq!(gemv_contract.1[0].alias_policy, AliasPolicy::Disjoint);
    assert_eq!(gemv_contract.1[1].input_dtype, LogicalDType::Bf16);
    assert_eq!(gemv_contract.1[1].weight_dtype, Some(LogicalDType::Bf16));
    assert!(gemv_contract.1[2].materialized);

    let error = build_numeric_fp8_operator_contract(
        gemv_plan,
        LogicalDType::F32,
        gemv_layout.k,
        gemv_layout.n,
        |_| Ok(()),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperatorContractError::Unsupported {
            capability: "numeric BF16 GEMV input",
            ..
        }
    ));
}

#[test]
fn f32_precision_rejects_bf16_input_at_contract_boundary() {
    let layout = NumericFp8Layout {
        n: 129,
        k: 257,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let plan = NumericFp8LinearPlan::new(
        layout,
        3,
        layout.n * layout.k * LogicalDType::F32.byte_width(),
        NumericFp8Precision::F32Tf32x3,
    )
    .unwrap();
    let error =
        build_numeric_fp8_operator_contract(plan, LogicalDType::Bf16, layout.k, layout.n, |_| {
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(
        error,
        OperatorContractError::Unsupported {
            capability: "numeric F32 precision input",
            ..
        }
    ));
}

#[test]
fn ordinary_bf16_precision_accepts_bf16_and_f32_inputs() {
    let layout = NumericFp8Layout {
        n: 129,
        k: 256,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let scratch_budget = (3 + layout.n) * layout.k * LogicalDType::Bf16.byte_width();
    let plan = NumericFp8LinearPlan::new(
        layout,
        3,
        scratch_budget,
        NumericFp8Precision::Bf16RneF32Accumulate,
    )
    .unwrap();
    for input_dtype in [LogicalDType::Bf16, LogicalDType::F32] {
        build_numeric_fp8_operator_contract(plan, input_dtype, layout.k, layout.n, |descriptor| {
            descriptor.validate_for(LinearCapability::NumericFp8)
        })
        .unwrap();
    }
}

#[test]
fn tiled_f32_cast_stage_is_materialized_and_w8a16_cast_is_inline() {
    let layout = NumericFp8Layout {
        n: 129,
        k: 256,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let scratch_budget = (3 + layout.n) * layout.k * LogicalDType::Bf16.byte_width();
    for precision in [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::Bf16SlicedK2F32Accumulate,
    ] {
        let plan = NumericFp8LinearPlan::new(layout, 3, scratch_budget, precision).unwrap();
        let stages = build_numeric_fp8_operator_contract(
            plan,
            LogicalDType::F32,
            layout.k,
            layout.n,
            |descriptor| {
                descriptor.validate_for(LinearCapability::NumericFp8)?;
                let stages = descriptor.fusion_plan.stages();
                Ok([stages[0], stages[1], stages[2], stages[3]])
            },
        )
        .unwrap();
        assert_eq!(stages[1].kind, LinearStageKind::Cast);
        assert!(stages[1].materialized);
        assert_eq!(stages[1].alias_policy, AliasPolicy::Disjoint);
        assert_eq!(stages[1].cast_semantics, CastSemantics::RoundToNearestEven);
    }

    let plan = NumericFp8LinearPlan::new(layout, 3, 0, NumericFp8Precision::W8A16Bf16F32Accumulate)
        .unwrap();
    let stages = build_numeric_fp8_operator_contract(
        plan,
        LogicalDType::F32,
        layout.k,
        layout.n,
        |descriptor| {
            descriptor.validate_for(LinearCapability::NumericFp8)?;
            let stages = descriptor.fusion_plan.stages();
            Ok([stages[0], stages[1], stages[2], stages[3]])
        },
    )
    .unwrap();
    assert_eq!(stages[1].kind, LinearStageKind::Cast);
    assert!(!stages[1].materialized);
    assert_eq!(stages[1].alias_policy, AliasPolicy::MayAlias);
    assert_eq!(stages[1].cast_semantics, CastSemantics::RoundToNearestEven);
}

#[test]
fn w8a16_stage_sequence_describes_nonmaterialized_decode_output() {
    let layout = NumericFp8Layout {
        n: 129,
        k: 257,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let plan = NumericFp8LinearPlan::new(layout, 3, 0, NumericFp8Precision::W8A16Bf16F32Accumulate)
        .unwrap();
    let stages = build_numeric_fp8_operator_contract(
        plan,
        LogicalDType::F32,
        layout.k,
        layout.n,
        |descriptor| {
            descriptor.validate_for(LinearCapability::NumericFp8)?;
            let stages = descriptor.fusion_plan.stages();
            Ok([stages[0], stages[1], stages[2], stages[3]])
        },
    )
    .unwrap();

    assert_eq!(stages[0].kind, LinearStageKind::Decode);
    assert_eq!(stages[0].input_dtype, LogicalDType::E4M3Fn);
    assert_eq!(stages[0].output_dtype, LogicalDType::Bf16);
    assert_eq!(stages[0].scale_operand, ScaleOperand::Weight);
    assert!(!stages[0].materialized);
    assert_eq!(stages[0].alias_policy, AliasPolicy::MayAlias);
    assert_eq!(stages[1].kind, LinearStageKind::Cast);
    assert_eq!(stages[1].input_dtype, LogicalDType::F32);
    assert_eq!(stages[1].output_dtype, LogicalDType::Bf16);
    assert_eq!(stages[1].cast_semantics, CastSemantics::RoundToNearestEven);
    assert_eq!(stages[2].kind, LinearStageKind::Linear);
    assert_eq!(stages[2].input_dtype, LogicalDType::Bf16);
    assert_eq!(stages[2].weight_dtype, Some(LogicalDType::Bf16));
    assert_eq!(stages[2].output_dtype, LogicalDType::F32);
    assert_eq!(stages[3].kind, LinearStageKind::Accumulate);
    assert_eq!(stages[3].input_dtype, LogicalDType::F32);
    assert_eq!(stages[3].output_dtype, LogicalDType::F32);
    assert!(stages[3].materialized);
    assert_eq!(stages[3].alias_policy, AliasPolicy::Disjoint);
}

#[test]
fn descriptor_rejects_bad_stage_order_dtype_and_scale() {
    let input_dims = [1, 8];
    let input_strides = [32, 4];
    let weight_dims = [4, 8];
    let weight_strides = [8, 1];
    let output_dims = [1, 4];
    let output_strides = [16, 4];
    let scale_dims = [1, 1];
    let scale_strides = [4, 4];
    let layout = NumericFp8Layout {
        n: 4,
        k: 8,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let base = numeric_descriptor(
        TensorViewDescriptor::new(LogicalDType::F32, &input_dims, &input_strides, 0, 4, None),
        TensorViewDescriptor::new(
            LogicalDType::E4M3Fn,
            &weight_dims,
            &weight_strides,
            0,
            1,
            None,
        ),
        TensorViewDescriptor::new(LogicalDType::F32, &output_dims, &output_strides, 0, 4, None),
        TensorViewDescriptor::new(LogicalDType::F32, &scale_dims, &scale_strides, 0, 4, None),
        layout,
    );
    let stages = base.fusion_plan.stages();

    let mut bad_order = [stages[0], stages[1], stages[2], stages[3]];
    bad_order.swap(0, 1);
    let bad_order_descriptor = LinearOperatorDescriptor::with_fusion_plan(
        base.input,
        base.weight,
        base.output,
        base.quantization,
        LinearFusionPlan::from_four(bad_order),
        base.alias_policy,
        base.capture_policy,
        base.retention_policy,
    );
    assert!(matches!(
        bad_order_descriptor.validate(),
        Err(OperatorContractError::FusionStageOrder { index: 0, .. })
    ));

    let mut bad_dtype = [stages[0], stages[1], stages[2], stages[3]];
    bad_dtype[1].input_dtype = LogicalDType::Bf16;
    let bad_dtype_descriptor = LinearOperatorDescriptor::with_fusion_plan(
        base.input,
        base.weight,
        base.output,
        base.quantization,
        LinearFusionPlan::from_four(bad_dtype),
        base.alias_policy,
        base.capture_policy,
        base.retention_policy,
    );
    assert!(matches!(
        bad_dtype_descriptor.validate(),
        Err(OperatorContractError::FusionStageMismatch {
            index: 1,
            field: "cast dtype",
            ..
        })
    ));

    let mut bad_scale = [stages[0], stages[1], stages[2], stages[3]];
    bad_scale[0].scale_operand = ScaleOperand::None;
    let bad_scale_descriptor = LinearOperatorDescriptor::with_fusion_plan(
        base.input,
        base.weight,
        base.output,
        base.quantization,
        LinearFusionPlan::from_four(bad_scale),
        base.alias_policy,
        base.capture_policy,
        base.retention_policy,
    );
    assert!(matches!(
        bad_scale_descriptor.validate(),
        Err(OperatorContractError::FusionStageMismatch {
            index: 0,
            field: "scale operand",
            ..
        })
    ));
}

#[test]
fn descriptor_rejects_unmaterialized_or_alias_inconsistent_output_stage() {
    let input_dims = [1, 8];
    let input_strides = [32, 4];
    let weight_dims = [4, 8];
    let weight_strides = [8, 1];
    let output_dims = [1, 4];
    let output_strides = [16, 4];
    let scale_dims = [1, 1];
    let scale_strides = [4, 4];
    let layout = NumericFp8Layout {
        n: 4,
        k: 8,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let base = numeric_descriptor(
        TensorViewDescriptor::new(LogicalDType::F32, &input_dims, &input_strides, 0, 4, None),
        TensorViewDescriptor::new(
            LogicalDType::E4M3Fn,
            &weight_dims,
            &weight_strides,
            0,
            1,
            None,
        ),
        TensorViewDescriptor::new(LogicalDType::F32, &output_dims, &output_strides, 0, 4, None),
        TensorViewDescriptor::new(LogicalDType::F32, &scale_dims, &scale_strides, 0, 4, None),
        layout,
    );
    let stages = base.fusion_plan.stages();

    let mut bad_decode_storage = [stages[0], stages[1], stages[2], stages[3]];
    bad_decode_storage[0].materialized = false;
    let descriptor = LinearOperatorDescriptor::with_fusion_plan(
        base.input,
        base.weight,
        base.output,
        base.quantization,
        LinearFusionPlan::from_four(bad_decode_storage),
        base.alias_policy,
        base.capture_policy,
        base.retention_policy,
    );
    assert!(matches!(
        descriptor.validate(),
        Err(OperatorContractError::FusionStageMismatch {
            index: 0,
            field: "alias policy",
            ..
        })
    ));

    let mut bad_cast_storage = [stages[0], stages[1], stages[2], stages[3]];
    bad_cast_storage[1].materialized = true;
    let descriptor = LinearOperatorDescriptor::with_fusion_plan(
        base.input,
        base.weight,
        base.output,
        base.quantization,
        LinearFusionPlan::from_four(bad_cast_storage),
        base.alias_policy,
        base.capture_policy,
        base.retention_policy,
    );
    assert!(matches!(
        descriptor.validate(),
        Err(OperatorContractError::FusionStageMismatch {
            index: 1,
            field: "alias policy",
            ..
        })
    ));

    let mut bad_linear_storage = [stages[0], stages[1], stages[2], stages[3]];
    bad_linear_storage[2].alias_policy = AliasPolicy::Disjoint;
    let descriptor = LinearOperatorDescriptor::with_fusion_plan(
        base.input,
        base.weight,
        base.output,
        base.quantization,
        LinearFusionPlan::from_four(bad_linear_storage),
        base.alias_policy,
        base.capture_policy,
        base.retention_policy,
    );
    assert!(matches!(
        descriptor.validate(),
        Err(OperatorContractError::FusionStageMismatch {
            index: 2,
            field: "alias policy",
            ..
        })
    ));

    let mut unmaterialized = [stages[0], stages[1], stages[2], stages[3]];
    unmaterialized[3].materialized = false;
    let descriptor = LinearOperatorDescriptor::with_fusion_plan(
        base.input,
        base.weight,
        base.output,
        base.quantization,
        LinearFusionPlan::from_four(unmaterialized),
        base.alias_policy,
        base.capture_policy,
        base.retention_policy,
    );
    assert!(matches!(
        descriptor.validate(),
        Err(OperatorContractError::FusionStageMismatch {
            index: 3,
            field: "materialized output",
            ..
        })
    ));

    let mut aliased = [stages[0], stages[1], stages[2], stages[3]];
    aliased[3].alias_policy = AliasPolicy::MayAlias;
    let descriptor = LinearOperatorDescriptor::with_fusion_plan(
        base.input,
        base.weight,
        base.output,
        base.quantization,
        LinearFusionPlan::from_four(aliased),
        base.alias_policy,
        base.capture_policy,
        base.retention_policy,
    );
    assert!(matches!(
        descriptor.validate(),
        Err(OperatorContractError::FusionStageMismatch {
            index: 3,
            field: "alias policy",
            ..
        })
    ));
}

#[test]
fn capability_rejects_alias_and_short_retention() {
    let input_dims = [1, 2];
    let input_strides = [8, 4];
    let weight_dims = [3, 2];
    let weight_strides = [2, 1];
    let output_dims = [1, 3];
    let output_strides = [12, 4];
    let scale_dims = [1, 1];
    let scale_strides = [4, 4];
    let layout = NumericFp8Layout {
        n: 3,
        k: 2,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let base = numeric_descriptor(
        TensorViewDescriptor::new(LogicalDType::F32, &input_dims, &input_strides, 0, 4, None),
        TensorViewDescriptor::new(
            LogicalDType::E4M3Fn,
            &weight_dims,
            &weight_strides,
            0,
            1,
            None,
        ),
        TensorViewDescriptor::new(LogicalDType::F32, &output_dims, &output_strides, 0, 4, None),
        TensorViewDescriptor::new(LogicalDType::F32, &scale_dims, &scale_strides, 0, 4, None),
        layout,
    );
    let descriptor = LinearOperatorDescriptor::new(
        base.input,
        base.weight,
        base.output,
        base.quantization,
        AliasPolicy::MayAlias,
        CapturePolicy::CaptureSafe,
        RetentionPolicy::UntilCompletion,
    );

    assert!(matches!(
        descriptor.validate_for(LinearCapability::NumericFp8),
        Err(OperatorContractError::Unsupported {
            capability: "numeric FP8 aliasing",
            ..
        })
    ));

    let retention_descriptor = LinearOperatorDescriptor::new(
        base.input,
        base.weight,
        base.output,
        base.quantization,
        AliasPolicy::Disjoint,
        CapturePolicy::CaptureSafe,
        RetentionPolicy::SubmitOnly,
    );
    assert!(matches!(
        retention_descriptor.validate_for(LinearCapability::NumericFp8),
        Err(OperatorContractError::Unsupported {
            capability: "numeric FP8 retention",
            ..
        })
    ));
}

#[test]
fn disjoint_policy_detects_overlapping_known_storage() {
    let input_dims = [1, 2];
    let input_strides = [8, 4];
    let weight_dims = [2, 2];
    let output_dims = [1, 2];
    let output_strides = [8, 4];
    let descriptor = LinearOperatorDescriptor::new(
        TensorViewDescriptor::new(
            LogicalDType::F32,
            &input_dims,
            &input_strides,
            0,
            4,
            Some(AdvisoryStorageIdentity::new(1)),
        ),
        TensorViewDescriptor::new(
            LogicalDType::F32,
            &weight_dims,
            &input_strides,
            16,
            4,
            Some(AdvisoryStorageIdentity::new(2)),
        ),
        TensorViewDescriptor::new(
            LogicalDType::F32,
            &output_dims,
            &output_strides,
            0,
            4,
            Some(AdvisoryStorageIdentity::new(1)),
        ),
        None,
        AliasPolicy::Disjoint,
        CapturePolicy::CaptureSafe,
        RetentionPolicy::UntilCompletion,
    );

    let error = descriptor.validate().unwrap_err();
    assert!(
        matches!(
            error,
            OperatorContractError::AdvisoryAliasing {
                first: "input",
                second: "output",
            }
        ),
        "unexpected contract error: {error:?}"
    );
}
