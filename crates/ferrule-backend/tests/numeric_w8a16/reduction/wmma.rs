//! Hermetic positive-sum/cancellation probes, independent of model artifacts.
use super::{ScalarLayout, ferrule_test_w8a16_scalar, within_bound};
use crate::oracle::{bf16_bits, dot};
use ferrule_backend::cuda::operators::linear::*;

fn next(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
#[ignore = "GPU owner + FERRULE_CUDA_TEST_ORACLE=1; seeded WMMA accumulation regression"]
fn seeded_segment_carry_and_dispatch_boundaries() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let mut failures = 0;
    let mut elements = 0;
    for (n, k) in [(33, 2048), (33, 2051), (7, 16384), (7, 16385), (7, 32769)] {
        for scale_type in [NumericFp8ScaleType::Bf16, NumericFp8ScaleType::F32] {
            let layout = NumericFp8Layout {
                n,
                k,
                row_origin: 127,
                column_origin: 125,
                scale_type,
            };
            let [sr, sc] = layout.scale_shape().unwrap();
            // FP8 1 times these non-power-of-two scales has BF16 low mantissa
            // bits whose contributions disappear in a long-lived MMA sum.
            let weights = vec![0x38; n * k];
            let scales: Vec<f32> = (0..sr * sc)
                .map(|i| (128 + i * 13 % 8) as f32 / 128.0)
                .collect();
            let scale_bytes: Vec<u8> = scales
                .iter()
                .flat_map(|&s| match scale_type {
                    NumericFp8ScaleType::Bf16 => bf16_bits(s).to_le_bytes().to_vec(),
                    NumericFp8ScaleType::F32 => s.to_le_bytes().to_vec(),
                })
                .collect();
            let artifact = op
                .upload_numeric_fp8_linear(layout, &weights, &scale_bytes)
                .unwrap();
            for m in [8, 9, 32] {
                for cancel in [false, true] {
                    let lda = k + 3;
                    let ldd = n + 5;
                    let mut state = 0x574d_4d41;
                    let mut a = vec![7331.; m * lda];
                    for row in 0..m {
                        for inner in 0..k {
                            let value = (128 + (next(&mut state) & 7)) as f32 / 128.0;
                            a[row * lda + inner] = if cancel && inner >= k / 2 {
                                -value
                            } else {
                                value
                            };
                        }
                    }
                    let input = op.upload_f32_buffer(&a).unwrap();
                    let bfinput = op
                        .upload_bf16_words(&a.iter().copied().map(bf16_bits).collect::<Vec<_>>())
                        .unwrap();
                    let mut output = op.upload_f32_buffer(&vec![7331.; m * ldd + 7]).unwrap();
                    let mut bfoutput = op.upload_f32_buffer(&vec![7331.; m * ldd + 7]).unwrap();
                    let plan = NumericFp8LinearPlan::new(
                        layout,
                        m,
                        0,
                        NumericFp8Precision::W8A16Bf16F32Accumulate,
                    )
                    .unwrap();
                    assert_eq!(plan.workspace_requirements().bytes, 0);
                    let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
                    op.reset_counters();
                    op.numeric_fp8_linear_into(
                        &artifact,
                        &input,
                        &mut output,
                        &mut workspace,
                        plan,
                        lda,
                        ldd,
                    )
                    .unwrap();
                    op.numeric_fp8_bf16_linear_into(
                        &artifact,
                        &bfinput,
                        &mut bfoutput,
                        &mut workspace,
                        plan,
                        lda,
                        ldd,
                    )
                    .unwrap();
                    assert_eq!(op.counters().device_allocation_attempts, 0);
                    assert_eq!(op.counters().compute_kernel_launches, 2);
                    op.record_compute_event().unwrap().synchronize().unwrap();
                    let actual = op.download_f32_buffer(&output).unwrap();
                    let bfactual = op.download_f32_buffer(&bfoutput).unwrap();
                    assert!(
                        actual
                            .iter()
                            .zip(&bfactual)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                    op.stream_clone().context().bind_to_thread().unwrap();
                    let scalar_layout = ScalarLayout {
                        m: m as u64,
                        n: n as u64,
                        k: k as u64,
                        row_origin: 127,
                        column_origin: 125,
                        lda: lda as u64,
                        ldd: ldd as u64,
                        scale_bytes: scale_type.element_bytes() as u64,
                    };
                    let mut scalar = vec![7331.; actual.len()];
                    let mut runtime = 0;
                    #[allow(
                        unsafe_code,
                        reason = "test-only independent CUDA reference; validated host spans and bound context"
                    )]
                    let status = unsafe {
                        ferrule_test_w8a16_scalar(
                            &scalar_layout,
                            a.as_ptr(),
                            weights.as_ptr(),
                            scale_bytes.as_ptr().cast(),
                            scalar.as_mut_ptr(),
                            &mut runtime,
                        )
                    };
                    assert_eq!(status, 0);
                    let mut case_failures = 0;
                    let mut scalar_failures = 0;
                    let mut max_fraction = 0.0f64;
                    for row in 0..m {
                        for col in 0..n {
                            let (truth, norm, serial) =
                                dot(layout, &weights, &scales, &a, lda, row, col);
                            let value = f64::from(actual[row * ldd + col]);
                            case_failures += usize::from(!within_bound(value, truth, norm));
                            scalar_failures += usize::from(!within_bound(
                                f64::from(scalar[row * ldd + col]),
                                truth,
                                norm,
                            ));
                            assert_eq!(scalar[row * ldd + col].to_bits(), serial.to_bits());
                            max_fraction =
                                max_fraction.max((value - truth).abs() / (1e-5 + 2e-6 * norm));
                        }
                        assert!(
                            actual[row * ldd + n..(row + 1) * ldd]
                                .iter()
                                .all(|&v| v == 7331.)
                        );
                        assert!(
                            scalar[row * ldd + n..(row + 1) * ldd]
                                .iter()
                                .all(|&v| v == 7331.)
                        );
                    }
                    assert!(actual[m * ldd..].iter().all(|&v| v == 7331.));
                    assert!(scalar[m * ldd..].iter().all(|&v| v == 7331.));
                    assert_eq!(op.download_f32_buffer(&input).unwrap(), a);
                    eprintln!(
                        "seeded m={m} n={n} k={k} scale={scale_type:?} cancel={cancel} failures={case_failures} scalar_failures={scalar_failures} max_fraction={max_fraction}"
                    );
                    failures += case_failures;
                    elements += m * n;
                }
            }
        }
    }
    eprintln!("seeded total: {elements} outputs, {failures} production bound failures");
    assert_eq!(
        failures, 0,
        "fixed F32-output operator bound; scalar metrics are independent diagnostics"
    );
}
