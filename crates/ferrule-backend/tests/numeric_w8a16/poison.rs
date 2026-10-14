use super::*;
use crate::cuda::operators::linear::CudaOperators;

#[test]
#[ignore = "GPU owner required; injected quarantine is not an asynchronous device-fault test"]
fn w8a16_poison_survives_both_input_apis_and_capture() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let l = NumericFp8Layout {
        n: 1,
        k: 1,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let artifact = op
        .upload_numeric_fp8_linear(l, &[0x38], &1.0f32.to_le_bytes())
        .unwrap();
    let input = op.upload_f32_buffer(&[1.0; 2]).unwrap();
    let bfinput = op.upload_bf16_words(&[0x3f80; 2]).unwrap();
    let mut output = op.upload_f32_buffer(&[73.0; 2]).unwrap();
    let p =
        NumericFp8LinearPlan::new(l, 2, 0, NumericFp8Precision::W8A16Bf16F32Accumulate).unwrap();
    let mut ws = op.numeric_fp8_linear_workspace(p).unwrap();
    ws.poison();
    op.reset_counters();
    assert!(
        op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut ws, p, 1, 1)
            .unwrap_err()
            .to_string()
            .contains("quarantined")
    );
    assert!(
        op.numeric_fp8_bf16_linear_into(&artifact, &bfinput, &mut output, &mut ws, p, 1, 1)
            .unwrap_err()
            .to_string()
            .contains("quarantined")
    );
    assert!(
        op.capture_decode_graph(|| op.numeric_fp8_bf16_linear_into(
            &artifact,
            &bfinput,
            &mut output,
            &mut ws,
            p,
            1,
            1
        ))
        .is_err()
    );
    assert!(ws.is_poisoned());
    assert!(!op.is_capture_safe());
    assert_eq!(op.counters().compute_kernel_launches, 0);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.download_f32_buffer(&output).unwrap(), [73.0; 2]);
}
