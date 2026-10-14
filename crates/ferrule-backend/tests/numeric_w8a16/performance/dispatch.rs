use super::oracle::*;
use ferrule_backend::cuda::operators::linear::*;

const PROFILE: NumericFp8Precision = NumericFp8Precision::W8A16Bf16F32Accumulate;

#[test]
#[ignore = "GPU owner required; packed loads, scale crossings, partial row groups and WMMA dispatch boundary"]
fn packed_and_byte_paths_preserve_bf16_oracle_and_canaries() {
    let op = CudaOperators::new_on_device(0).unwrap();
    for m in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
        for (n, k) in [
            (7, 4),
            (33, 32),
            (131, 132),
            (17, 2052),
            (17, 2051),
            (7, 8192),
        ] {
            for ty in [NumericFp8ScaleType::F32, NumericFp8ScaleType::Bf16] {
                // Residue 125 puts a scale boundary inside a four-byte packed load.
                let l = layout(n, k, ty);
                let w = weight(l);
                let (sb, sv) = scales(l);
                let artifact = op.upload_numeric_fp8_linear(l, &w, &sb).unwrap();
                let lda = k + 3;
                let ldd = n + 5;
                let a: Vec<_> = (0..m * lda)
                    .map(|i| (i as i32 % 37 - 18) as f32 * 0.03173)
                    .collect();
                let input = op.upload_f32_buffer(&a).unwrap();
                let binput = op
                    .upload_bf16_words(&a.iter().copied().map(bf16_bits).collect::<Vec<_>>())
                    .unwrap();
                let mut out = op.upload_f32_buffer(&vec![7331.0; m * ldd + 7]).unwrap();
                let mut bout = op.upload_f32_buffer(&vec![7331.0; m * ldd + 7]).unwrap();
                let plan = NumericFp8LinearPlan::new(l, m, 0, PROFILE).unwrap();
                let mut ws = op.numeric_fp8_linear_workspace(plan).unwrap();
                op.reset_counters();
                op.numeric_fp8_linear_into(&artifact, &input, &mut out, &mut ws, plan, lda, ldd)
                    .unwrap();
                op.numeric_fp8_bf16_linear_into(
                    &artifact, &binput, &mut bout, &mut ws, plan, lda, ldd,
                )
                .unwrap();
                assert_eq!(op.counters().compute_kernel_launches, 2);
                assert_eq!(op.counters().device_allocation_attempts, 0);
                assert_eq!(op.counters().stream_wide_syncs, 0);
                let got = op.download_f32_buffer(&out).unwrap();
                assert_eq!(got, op.download_f32_buffer(&bout).unwrap());
                for row in 0..m {
                    for col in 0..n {
                        let (expected, norm, _) = dot(l, &w, &sv, &a, lda, row, col);
                        let actual = f64::from(got[row * ldd + col]);
                        assert!(
                            actual.is_finite() && (actual - expected).abs() <= 1e-5 + 2e-6 * norm,
                            "{m}x{n}x{k} {ty:?} ({row},{col}): {actual} vs {expected}"
                        );
                    }
                    assert!(
                        got[row * ldd + n..(row + 1) * ldd]
                            .iter()
                            .all(|v| *v == 7331.0)
                    );
                }
                assert!(got[m * ldd..].iter().all(|v| *v == 7331.0));
                assert_eq!(op.download_f32_buffer(&input).unwrap(), a);
            }
        }
    }
}

#[test]
#[ignore = "GPU owner required; distinguish true BF16 operands from F32 fallback and premature scale rounding"]
fn vector_and_mma_do_not_silently_change_precision() {
    let op = CudaOperators::new_on_device(0).unwrap();
    for m in [1, 2, 4, 8, 9, 32] {
        let l = NumericFp8Layout {
            n: 1,
            k: 4,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::F32,
        };
        let p = NumericFp8LinearPlan::new(l, m, 0, PROFILE).unwrap();
        let mut ws = op.numeric_fp8_linear_workspace(p).unwrap();
        for (w, scale, row, expected) in [
            (
                [0x38, 0x38, 0, 0],
                1.0f32,
                [1.0001f32, -1.0, 0.0, 0.0],
                0.0f32,
            ),
            (
                [0x3e, 0, 0, 0],
                1.003f32,
                [1.0f32, 0.0, 0.0, 0.0],
                bf16(1.75 * 1.003),
            ),
        ] {
            let artifact = op
                .upload_numeric_fp8_linear(l, &w, &scale.to_le_bytes())
                .unwrap();
            let input = op.upload_f32_buffer(&row.repeat(m)).unwrap();
            let mut output = op.zero_f32_buffer(m).unwrap();
            op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut ws, p, 4, 1)
                .unwrap();
            assert_eq!(op.download_f32_buffer(&output).unwrap(), vec![expected; m]);
        }
    }
    assert_ne!(1.0001f32 - 1.0, 0.0);
    assert_ne!(bf16(1.75 * 1.003), bf16(1.75 * bf16(1.003)));
}
