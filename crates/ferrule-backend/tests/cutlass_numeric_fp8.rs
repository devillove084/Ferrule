#![cfg(feature = "cuda")]

use ferrule_backend::cuda::operators::linear::*;
use ferrule_backend::cuda::providers::cutlass::{CutlassKernelId, discover_provider};
use ferrule_backend::cuda::providers::{COMPILED_TARGET, CudaContext, DeviceBuffer};

const PRECISION: NumericFp8Precision = NumericFp8Precision::Bf16RneF32Accumulate;
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn round(x: f32) -> f32 {
    f32::from_bits(u32::from(bf16(x)) << 16)
}
fn decode(b: u8) -> f32 {
    let sign = if b & 128 == 0 { 1.0 } else { -1.0 };
    let e = (b >> 3) & 15;
    let m = b & 7;
    sign * if e == 0 {
        f32::from(m) / 512.0
    } else {
        2.0f32.powi(i32::from(e) - 7) * (1.0 + f32::from(m) / 8.0)
    }
}
fn layout(n: usize, k: usize, scale_type: NumericFp8ScaleType) -> NumericFp8Layout {
    NumericFp8Layout {
        n,
        k,
        row_origin: 0,
        column_origin: 0,
        scale_type,
    }
}
fn scales(l: NumericFp8Layout) -> (Vec<u8>, Vec<f32>) {
    let [r, c] = l.scale_shape().unwrap();
    let values: Vec<_> = (0..r * c)
        .map(|i| 0.1731 + (i % 19) as f32 * 0.02173)
        .map(|v| {
            if l.scale_type == NumericFp8ScaleType::Bf16 {
                round(v)
            } else {
                v
            }
        })
        .collect();
    let bytes = values
        .iter()
        .flat_map(|&v| match l.scale_type {
            NumericFp8ScaleType::Bf16 => bf16(v).to_le_bytes().to_vec(),
            NumericFp8ScaleType::F32 => v.to_le_bytes().to_vec(),
        })
        .collect();
    (bytes, values)
}

#[test]
fn exact_budget_and_invalid_metadata() {
    let l = layout(259, 259, NumericFp8ScaleType::F32);
    let min = (3 + 1) * 264 * 2;
    assert!(NumericFp8LinearPlan::new(l, 3, min - 1, PRECISION).is_err());
    let p = NumericFp8LinearPlan::new(l, 3, (3 + 17) * 264 * 2 + 7, PRECISION).unwrap();
    assert_eq!(p.tile_rows(), 17);
    assert_eq!(p.tile_count(), 16);
    assert_eq!(p.activation_bytes(), 1584);
    assert_eq!(p.weight_tile_bytes(), 8976);
    assert_eq!(p.workspace_requirements().bytes, 10560);
    assert_eq!(p.kernel_launches(), 33);
    for rows in [0, usize::MAX] {
        assert!(NumericFp8LinearPlan::new(l, rows, usize::MAX, PRECISION).is_err());
    }
    for bad in [
        NumericFp8Layout { k: 0, ..l },
        NumericFp8Layout { n: usize::MAX, ..l },
        NumericFp8Layout {
            column_origin: usize::MAX,
            ..l
        },
    ] {
        assert!(bad.storage_lengths().is_err());
    }
    for ty in [NumericFp8ScaleType::F32, NumericFp8ScaleType::Bf16] {
        let l = layout(1, 1, ty);
        for bad in [0.0, -0.0, -1.0, f32::INFINITY, f32::NAN] {
            let b = if ty == NumericFp8ScaleType::F32 {
                bad.to_le_bytes().to_vec()
            } else {
                bf16(bad).to_le_bytes().to_vec()
            };
            assert!(l.validate_payload(&[0x38], &b).is_err());
        }
        let (s, _) = scales(l);
        assert!(l.validate_payload(&[0x7f], &s).is_err());
        assert!(l.validate_payload(&[0xff], &s).is_err());
        assert!(l.validate_payload(&[], &s).is_err());
        assert!(l.validate_payload(&[0x38], &s[..s.len() - 1]).is_err());
    }
}

#[test]
#[ignore = "requires actual sm80+ CUDA GPU; bounded numeric FP8 to BF16 TensorOp"]
fn gpu_odd_tails_scales_large_k_strides_offsets_and_canaries() {
    assert!(
        discover_provider()
            .unwrap()
            .supports(CutlassKernelId::Bf16Gemm)
    );
    eprintln!("compiled target: {COMPILED_TARGET}");
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let consumer = ctx.new_stream().unwrap();
    for ty in [NumericFp8ScaleType::Bf16, NumericFp8ScaleType::F32] {
        for (m, n, k, chunk, ro, co) in [
            (1, 7, 3, 1, 0, 0),
            (3, 259, 259, 17, 127, 125),
            (65, 131, 129, 33, 0, 0),
            (3, 17, 8193, 3, 255, 127),
        ] {
            let l = NumericFp8Layout {
                row_origin: ro,
                column_origin: co,
                ..layout(n, k, ty)
            };
            let (sb, sv) = scales(l);
            let codes = [0u8, 0x80, 1, 0x87, 0x21, 0x39, 0xbd, 0x4e, 0xcf, 0x7e, 0xfe];
            let w: Vec<_> = (0..n * k)
                .map(|i| codes[(i * 7 + i / k) % codes.len()])
                .collect();
            let artifact = CudaNumericFp8Artifact::upload(&stream, l, &w, &sb).unwrap();
            assert_eq!(artifact.storage_bytes(), w.len() + sb.len());
            let lda = k + 3;
            let ldd = n + 5;
            let a: Vec<_> = (0..m * lda)
                .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 503.7)
                .collect();
            let ar = DeviceBuffer::from_host(&stream, &a).unwrap();
            let output_root =
                DeviceBuffer::from_host(&stream, &vec![7331.0f32; 3 + m * ldd + 4]).unwrap();
            let mut out = output_root.slice(3, m * ldd).unwrap();
            let kp = k.div_ceil(8) * 8;
            let budget = (m + chunk) * kp * 2;
            let p = NumericFp8LinearPlan::new(l, m, budget, PRECISION).unwrap();
            assert_eq!(p.tile_rows(), chunk);
            assert_eq!(p.workspace_requirements().bytes as usize, budget);
            let scratch_root =
                DeviceBuffer::from_host(&stream, &vec![0xa5u8; 16 + budget + 32]).unwrap();
            let mut scratch =
                CudaNumericFp8Workspace::from_buffer(scratch_root.slice(16, budget).unwrap());
            numeric_fp8_linear(&stream, &artifact, &ar, &mut out, &mut scratch, p, lda, ldd)
                .unwrap();
            let done = stream.record_event(None).unwrap();
            consumer.wait(&done).unwrap();
            let got = output_root.to_host_vec(&consumer).unwrap();
            assert_eq!(&got[..3], &[7331.0; 3]);
            assert_eq!(&got[3 + m * ldd..], &[7331.0; 4]);
            let mut ab = vec![0u16; m * kp];
            let mut wb = vec![0u16; n * kp];
            for r in 0..m {
                for c in 0..k {
                    ab[r * kp + c] = bf16(a[r * lda + c]);
                }
            }
            let sc = l.scale_shape().unwrap()[1];
            for r in 0..n {
                for c in 0..k {
                    let s = sv[((ro % 128 + r) / 128) * sc + (co % 128 + c) / 128];
                    wb[r * kp + c] = bf16(decode(w[r * k + c]) * s);
                }
            }
            // Independent full BF16 GPU reference exists only in this test.
            let adev = DeviceBuffer::from_host(&stream, &ab).unwrap();
            let wdev = DeviceBuffer::from_host(&stream, &wb).unwrap();
            let mut reference = DeviceBuffer::<f32>::zeroed(&stream, m * n).unwrap();
            bf16_gemm(
                &stream,
                &adev,
                &wdev,
                &mut reference,
                Bf16GemmLayout::contiguous(m, n, kp),
            )
            .unwrap();
            let ref_gpu = reference.to_host_vec(&stream).unwrap();
            let mut max_error = 0.0f32;
            for r in 0..m {
                assert!(
                    got[3 + r * ldd + n..3 + (r + 1) * ldd]
                        .iter()
                        .all(|&v| v == 7331.0)
                );
                for c in 0..n {
                    let actual = got[3 + r * ldd + c];
                    let expected = (0..k)
                        .map(|t| {
                            f64::from(f32::from_bits(u32::from(ab[r * kp + t]) << 16))
                                * f64::from(f32::from_bits(u32::from(wb[c * kp + t]) << 16))
                        })
                        .sum::<f64>();
                    let norm = (0..k)
                        .map(|t| {
                            (f64::from(f32::from_bits(u32::from(ab[r * kp + t]) << 16))
                                * f64::from(f32::from_bits(u32::from(wb[c * kp + t]) << 16)))
                            .abs()
                        })
                        .sum::<f64>();
                    assert!(
                        actual.is_finite()
                            && (f64::from(actual) - expected).abs() <= 1e-5 + 2e-6 * norm,
                        "{ty:?} {m}x{n}x{k} ({r},{c}): {actual} vs BF16 {expected}"
                    );
                    assert_eq!(actual, ref_gpu[r * n + c], "tiled vs full BF16 CUTLASS");
                    max_error = max_error.max((f64::from(actual) - expected).abs() as f32);
                }
            }
            let scratch_bytes = scratch_root.to_host_vec(&stream).unwrap();
            assert!(
                scratch_bytes[..16]
                    .iter()
                    .chain(&scratch_bytes[16 + budget..])
                    .all(|&b| b == 0xa5)
            );
            assert_eq!(ar.to_host_vec(&stream).unwrap(), a);
            eprintln!(
                "{ty:?} M={m} N={n} K={k}: encoded={} scratch={} A={} Wtile={} tiles={} launches={} max BF16-reference error={max_error}",
                artifact.storage_bytes(),
                budget,
                p.activation_bytes(),
                p.weight_tile_bytes(),
                p.tile_count(),
                p.kernel_launches()
            );
        }
    }
}

#[test]
#[ignore = "requires actual CUDA GPU; owner/shape/budget failure atomicity"]
fn gpu_rejects_invalid_before_launch() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let other = CudaOperators::new_on_device(0).unwrap();
    let l = layout(7, 3, NumericFp8ScaleType::F32);
    let s = 1.0f32.to_le_bytes();
    let proof = ferrule_common::numeric_fp8::ImmutableValidatedNumericFp8Payload::new(
        l,
        vec![0x38; 21],
        s.to_vec(),
    )
    .unwrap();
    let w = op.upload_validated_numeric_fp8_linear(&proof).unwrap();
    let foreign = other.upload_validated_numeric_fp8_linear(&proof).unwrap();
    let a = op.upload_f32_buffer(&[1.0; 6]).unwrap();
    let foreign_a = other.upload_f32_buffer(&[1.0; 6]).unwrap();
    let mut d = op.upload_f32_buffer(&[73.0; 14]).unwrap();
    let mut foreign_d = other.upload_f32_buffer(&[73.0; 14]).unwrap();
    let p = NumericFp8LinearPlan::new(l, 2, 48, PRECISION).unwrap();
    let mut ws = op.numeric_fp8_linear_workspace(p).unwrap();
    let mut foreign_ws = other.numeric_fp8_linear_workspace(p).unwrap();
    let mut short = CudaNumericFp8Workspace::from_buffer(
        DeviceBuffer::<u8>::zeroed(&op.stream_clone(), 47).unwrap(),
    );
    let mut long = CudaNumericFp8Workspace::from_buffer(
        DeviceBuffer::<u8>::zeroed(&op.stream_clone(), 49).unwrap(),
    );
    let mut misaligned = CudaNumericFp8Workspace::from_buffer(
        DeviceBuffer::<u8>::zeroed(&op.stream_clone(), 49)
            .unwrap()
            .slice(1, 48)
            .unwrap(),
    );
    op.reset_counters();
    assert!(
        op.numeric_fp8_linear_into(&foreign, &a, &mut d, &mut ws, p, 3, 7)
            .is_err()
    );
    assert!(
        op.numeric_fp8_linear_into(&w, &foreign_a, &mut d, &mut ws, p, 3, 7)
            .is_err()
    );
    assert!(
        op.numeric_fp8_linear_into(&w, &a, &mut foreign_d, &mut ws, p, 3, 7)
            .is_err()
    );
    for scratch in [&mut foreign_ws, &mut short, &mut long, &mut misaligned] {
        assert!(
            op.numeric_fp8_linear_into(&w, &a, &mut d, scratch, p, 3, 7)
                .is_err()
        );
        assert!(!scratch.is_poisoned());
    }
    for (lda, ldd) in [
        (2, 7),
        (3, 6),
        (usize::MAX, 7),
        (3, usize::MAX),
        (4, 7),
        (3, 8),
    ] {
        assert!(
            op.numeric_fp8_linear_into(&w, &a, &mut d, &mut ws, p, lda, ldd)
                .is_err()
        );
    }
    let wrong =
        NumericFp8LinearPlan::new(NumericFp8Layout { row_origin: 1, ..l }, 2, 48, PRECISION)
            .unwrap();
    assert!(
        op.numeric_fp8_linear_into(&w, &a, &mut d, &mut ws, wrong, 3, 7)
            .is_err()
    );
    for value in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            op.upload_numeric_fp8_linear(l, &[0x38; 21], &value.to_le_bytes())
                .is_err()
        );
    }
    assert_eq!(op.counters().compute_kernel_launches, 0);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().artifact_uploads, 0);
    assert_eq!(op.download_f32_buffer(&d).unwrap(), [73.0; 14]);
    assert_eq!(other.download_f32_buffer(&foreign_d).unwrap(), [73.0; 14]);
    // A different current CUDA context must not change the exact owner of work.
    other.stream_clone().context().bind_to_thread().unwrap();
    op.numeric_fp8_linear_into(&w, &a, &mut d, &mut ws, p, 3, 7)
        .unwrap();
    op.record_compute_event().unwrap().synchronize().unwrap();
    assert_eq!(op.download_f32_buffer(&d).unwrap(), [3.0; 14]);
}

#[test]
#[ignore = "requires actual CUDA GPU; explicit BF16 rounding and captured multi-tile completion"]
fn gpu_precision_and_graph_replay() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let l = layout(3, 3, NumericFp8ScaleType::F32);
    let w = op
        .upload_numeric_fp8_linear(
            l,
            &[0x38, 0xb8, 0, 0x38, 0, 0, 0, 0x38, 0],
            &1.001f32.to_le_bytes(),
        )
        .unwrap();
    let values = [
        1.0001f32,
        1.0,
        0.0,
        f32::from_bits(0x3f80_8000),
        f32::from_bits(0x3f81_8000),
        0.0,
    ];
    let mut a = op.upload_f32_buffer(&values).unwrap();
    let mut out = op.zero_f32_buffer(6).unwrap();
    let p = NumericFp8LinearPlan::new(l, 2, 48, PRECISION).unwrap();
    let mut scratch = op.numeric_fp8_linear_workspace(p).unwrap();
    op.numeric_fp8_linear_into(&w, &a, &mut out, &mut scratch, p, 3, 3)
        .unwrap();
    op.record_compute_event().unwrap().synchronize().unwrap();
    op.reset_counters();
    op.enable_capture_safe();
    let graph = op
        .capture_decode_graph(|| {
            op.numeric_fp8_linear_into(&w, &a, &mut out, &mut scratch, p, 3, 3)
        })
        .unwrap();
    op.disable_capture_safe();
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().stream_wide_syncs, 0);
    for factor in [1.0, 2.0, -1.0] {
        op.overwrite_f32_buffer(&values.map(|v| v * factor), &mut a)
            .unwrap();
        op.launch_graph(&graph).unwrap();
        op.record_compute_event().unwrap().synchronize().unwrap();
        let got = op.download_f32_buffer(&out).unwrap();
        let x = values.map(|v| round(v * factor));
        assert_eq!(got, [x[0] - x[1], x[0], x[1], x[3] - x[4], x[3], x[4]]);
        assert_eq!(got[0], 0.0); // F32-exact would preserve ~0.0001: deliberately BF16.
    }
}

#[test]
#[ignore = "requires actual CUDA GPU; every finite E4M3 byte, decode tail padding, alias rejection"]
fn gpu_all_finite_codes_and_overlap_rejection() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let w: Vec<u8> = (0..=255).filter(|b| b & 0x7f != 0x7f).collect();
    let l = layout(1, w.len(), NumericFp8ScaleType::F32);
    let scale_values = [f32::from_bits(0x3f80_8000), f32::from_bits(0x3f81_8000)];
    let scales: Vec<_> = scale_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let artifact = CudaNumericFp8Artifact::upload(&stream, l, &w, &scales).unwrap();
    let plan = NumericFp8LinearPlan::new(l, 1, 1024, PRECISION).unwrap();
    let a = DeviceBuffer::from_host(&stream, &vec![1.0f32; w.len()]).unwrap();
    let root = DeviceBuffer::<u8>::zeroed(&stream, 1024).unwrap();
    let mut scratch = CudaNumericFp8Workspace::from_buffer(root.slice(0, 1024).unwrap());
    let mut output = DeviceBuffer::from_host(&stream, &[713.0f32]).unwrap();
    // Safe slice aliases still must fail before any conversion writes/launch.
    let mut alias = a.slice(0, 1).unwrap();
    assert!(
        numeric_fp8_linear(
            &stream,
            &artifact,
            &a,
            &mut alias,
            &mut scratch,
            plan,
            l.k,
            1
        )
        .is_err()
    );
    assert_eq!(a.to_host_vec(&stream).unwrap(), vec![1.0; w.len()]);
    numeric_fp8_linear(
        &stream,
        &artifact,
        &a,
        &mut output,
        &mut scratch,
        plan,
        l.k,
        1,
    )
    .unwrap();
    let raw = root.to_host_vec(&stream).unwrap();
    let words: Vec<_> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(u16::from_le_bytes)
        .collect();
    assert!(words[..l.k].iter().all(|&v| v == bf16(1.0)));
    assert_eq!(&words[l.k..256], &[0, 0]);
    for (i, &byte) in w.iter().enumerate() {
        assert_eq!(
            words[256 + i],
            bf16(decode(byte) * scale_values[i / 128]),
            "code {byte:#x}"
        );
    }
    assert_eq!(&words[510..], &[0, 0]);
}
