use ferrule_backend::cpu::{
    CpuExecutionPrecision, CpuOperatorProvider, HostRows, LinearRef, LinearWeight,
    NativeCpuProvider, ReferenceCpuProvider, RowsShape, SwiGluRef,
};

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ferrule_backend::cpu::bf16_rne_word(*value).to_le_bytes())
        .collect()
}

fn rows(values: Vec<f32>, row_count: usize, width: usize) -> HostRows {
    HostRows::new(
        RowsShape::new(row_count, width).unwrap(),
        ferrule_backend::cpu::RowsDType::F32,
        None,
        values,
    )
    .unwrap()
}

#[test]
fn native_dense_operators_match_reference_at_bf16_boundaries() {
    let reference = ReferenceCpuProvider;
    let native = NativeCpuProvider;
    let weight = bf16_bytes(&[
        1.0, -2.0, 0.5, 0.25, 1.0, -1.0, -0.5, 0.75, 2.0, 1.5, -0.25, 0.125,
    ]);
    let linear = LinearRef {
        weight: LinearWeight::Bf16(&weight),
        out_features: 3,
        in_features: 4,
        bias: Some(&[0.25, -0.5, 1.0]),
    };
    let input = rows(vec![1.125, -2.25, 0.375, 4.5, -1.0, 0.25, 3.0, -0.5], 2, 4);
    let expected = reference
        .linear(linear, &input, None, CpuExecutionPrecision::Bf16)
        .unwrap();
    let actual = native
        .linear(linear, &input, None, CpuExecutionPrecision::Bf16)
        .unwrap();
    assert_eq!(actual, expected);

    let embedding = LinearRef {
        weight: LinearWeight::Bf16(&weight),
        out_features: 3,
        in_features: 4,
        bias: None,
    };
    let expected = reference
        .embedding(embedding, &[2, 0], None, CpuExecutionPrecision::Bf16)
        .unwrap();
    let actual = native
        .embedding(embedding, &[2, 0], None, CpuExecutionPrecision::Bf16)
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn native_rms_norm_and_swiglu_match_reference() {
    let reference = ReferenceCpuProvider;
    let native = NativeCpuProvider;
    let input = rows(vec![1.125, -2.25, 0.375, 4.5], 1, 4);
    let norm_weight = [1.0, 0.75, 1.25, -0.5];
    let expected = reference
        .rms_norm(
            &input,
            &norm_weight,
            1e-6,
            1,
            None,
            CpuExecutionPrecision::Bf16,
        )
        .unwrap();
    let actual = native
        .rms_norm(
            &input,
            &norm_weight,
            1e-6,
            1,
            None,
            CpuExecutionPrecision::Bf16,
        )
        .unwrap();
    assert_eq!(actual, expected);

    let gate = bf16_bytes(&[1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 1.5, 0.25]);
    let up = bf16_bytes(&[0.5, 1.0, -1.0, 0.25, 1.0, -0.5, 0.75, 2.0]);
    let down = bf16_bytes(&[1.0, 2.0, -1.0, 0.5, 0.25, -2.0]);
    let expert = SwiGluRef {
        gate: LinearRef {
            weight: LinearWeight::Bf16(&gate),
            out_features: 2,
            in_features: 4,
            bias: Some(&[0.25, -0.125]),
        },
        up: LinearRef {
            weight: LinearWeight::Bf16(&up),
            out_features: 2,
            in_features: 4,
            bias: None,
        },
        down: LinearRef {
            weight: LinearWeight::Bf16(&down),
            out_features: 3,
            in_features: 2,
            bias: Some(&[0.0, 0.25, -0.5]),
        },
        activation_limit: Some(10.0),
    };
    let expected = reference
        .swiglu(expert, &input, None, CpuExecutionPrecision::Bf16)
        .unwrap();
    let actual = native
        .swiglu(expert, &input, None, CpuExecutionPrecision::Bf16)
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn native_capabilities_fail_open_to_reference_semantics() {
    let capabilities = NativeCpuProvider::runtime_capabilities();
    assert_eq!(capabilities.native_bf16_linear, capabilities.avx2);
    assert_eq!(capabilities.native_bf16_embedding, capabilities.avx2);
    assert_eq!(capabilities.native_rms_norm, capabilities.avx2);
    assert_eq!(capabilities.native_bf16_swiglu, capabilities.avx2);
    assert!(!capabilities.native_paged_gqa);
    assert!(!capabilities.native_routed_moe);
}
