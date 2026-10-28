use ferrule_backend::cuda_plan::{
    NUMERIC_BF16_GEMV_SCRATCH_BYTES, NumericFp8Layout, NumericFp8LinearPlan, NumericFp8Precision,
};
use ferrule_backend::operator_contract::{
    AdvisoryStorageIdentity, AliasPolicy, CapturePolicy, LinearCapability,
    LinearOperatorDescriptor, LogicalDType, OperatorContractError, QuantizationDescriptor,
    RetentionPolicy, TensorViewDescriptor, build_numeric_fp8_operator_contract,
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
            Ok((descriptor.input.dtype, policy))
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
            Ok((descriptor.input.dtype, policy))
        },
    )
    .unwrap();
    assert_eq!(
        f32_contract,
        (LogicalDType::F32, CapturePolicy::CaptureSafe)
    );
    assert_eq!(
        bf16_contract,
        (LogicalDType::Bf16, CapturePolicy::CaptureSafe)
    );

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
    let gemv_policy = build_numeric_fp8_operator_contract(
        gemv_plan,
        LogicalDType::Bf16,
        gemv_layout.k,
        gemv_layout.n,
        |descriptor| {
            let policy = descriptor.capture_policy;
            descriptor.validate_for(LinearCapability::NumericFp8)?;
            Ok(policy)
        },
    )
    .unwrap();
    assert_eq!(gemv_policy, CapturePolicy::EagerOnly);

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
