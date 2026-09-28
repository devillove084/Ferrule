//! The backend's public payload contract must compile without its CUDA feature.
use ferrule_backend::numeric_fp8::{
    ImmutableValidatedNumericFp8Payload, NumericFp8Layout, NumericFp8ScaleType,
};

#[test]
fn backend_payload_api_needs_no_cuda() {
    let layout = NumericFp8Layout {
        n: 2,
        k: 2,
        row_origin: 127,
        column_origin: 255,
        scale_type: NumericFp8ScaleType::F32,
    };
    let payload = ImmutableValidatedNumericFp8Payload::new(
        layout,
        vec![0, 0x80, 0x7e, 0xfe],
        [1.0f32; 4]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(payload.layout(), layout);
    assert_eq!(payload.storage_bytes(), 20);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA; tiny proof/raw equivalence and prelaunch alias rejection"]
fn gpu_proof_upload_preserves_output_and_alias_checks() {
    use ferrule_backend::cuda::operators::linear::*;
    use ferrule_backend::cuda::providers::{CudaContext, DeviceBuffer};

    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let layout = NumericFp8Layout {
        n: 2,
        k: 2,
        row_origin: 127,
        column_origin: 127,
        scale_type: NumericFp8ScaleType::F32,
    };
    let proof = ImmutableValidatedNumericFp8Payload::new(
        layout,
        vec![0x38, 0xb8, 0x40, 0x38],
        [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let validated = CudaNumericFp8Artifact::upload_validated(&stream, &proof).unwrap();
    let raw =
        CudaNumericFp8Artifact::upload(&stream, layout, proof.weight_bytes(), proof.scale_bytes())
            .unwrap();
    assert_eq!(validated.storage_bytes(), proof.storage_bytes());
    let activation = DeviceBuffer::from_host(&stream, &[1.0f32, 2.0]).unwrap();
    for precision in [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::F32Tf32x3,
    ] {
        let plan = NumericFp8LinearPlan::new(layout, 1, 64, precision).unwrap();
        let mut workspace = CudaNumericFp8Workspace::from_buffer_with_precision(
            DeviceBuffer::<u8>::zeroed(&stream, plan.workspace_requirements().bytes as usize)
                .unwrap(),
            precision,
        );
        // Safe DeviceBuffer views may alias even though the CPU proof cannot
        // mutate. Proof reuse must not bypass the existing address-span checks.
        let mut alias = activation.slice(0, 2).unwrap();
        let error = numeric_fp8_linear(
            &stream,
            &validated,
            &activation,
            &mut alias,
            &mut workspace,
            plan,
            2,
            2,
        )
        .unwrap_err();
        assert!(error.to_string().contains("overlapping"));
        assert!(!workspace.is_poisoned());
        assert_eq!(activation.to_host_vec(&stream).unwrap(), [1.0, 2.0]);
        for artifact in [&validated, &raw] {
            let mut output = DeviceBuffer::<f32>::zeroed(&stream, 2).unwrap();
            numeric_fp8_linear(
                &stream,
                artifact,
                &activation,
                &mut output,
                &mut workspace,
                plan,
                2,
                2,
            )
            .unwrap();
            assert_eq!(output.to_host_vec(&stream).unwrap(), [-3.0, 14.0]);
        }
    }
}
