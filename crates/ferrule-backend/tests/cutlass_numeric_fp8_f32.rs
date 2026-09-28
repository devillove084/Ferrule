#![cfg(feature = "cuda")]

use ferrule_backend::cuda::operators::linear::*;
use ferrule_backend::cuda::providers::cutlass::{
    CutlassKernelId, F32GemmLayout, discover_provider, f32_gemm,
};
use ferrule_backend::cuda::providers::{CudaContext, DeviceBuffer};

const F32: NumericFp8Precision = NumericFp8Precision::F32Tf32x3;
const BF16: NumericFp8Precision = NumericFp8Precision::Bf16RneF32Accumulate;

fn bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}
fn fp8(b: u8) -> f32 {
    let e = (b >> 3) & 15;
    let f = b & 7;
    let v = if e == 0 {
        f32::from(f) / 512.0
    } else {
        2.0f32.powi(i32::from(e) - 7) * (1.0 + f32::from(f) / 8.0)
    };
    if b & 128 == 0 { v } else { -v }
}
fn layout(n: usize, k: usize, scale_type: NumericFp8ScaleType) -> NumericFp8Layout {
    NumericFp8Layout {
        n,
        k,
        row_origin: 127,
        column_origin: 125,
        scale_type,
    }
}
fn scales(l: NumericFp8Layout) -> (Vec<u8>, Vec<f32>) {
    let [r, c] = l.scale_shape().unwrap();
    let v: Vec<_> = (0..r * c)
        .map(|i| 0.71317 + (i % 23) as f32 * 0.03173)
        .map(|v| {
            if l.scale_type == NumericFp8ScaleType::Bf16 {
                bf16(v)
            } else {
                v
            }
        })
        .collect();
    let bytes = v
        .iter()
        .flat_map(|v| match l.scale_type {
            NumericFp8ScaleType::Bf16 => ((v.to_bits() >> 16) as u16).to_le_bytes().to_vec(),
            NumericFp8ScaleType::F32 => v.to_le_bytes().to_vec(),
        })
        .collect();
    (bytes, v)
}
fn decode(l: NumericFp8Layout, bytes: &[u8], scales: &[f32]) -> Vec<f32> {
    let cols = l.scale_shape().unwrap()[1];
    bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let r = (l.row_origin % 128 + i / l.k) / 128;
            let c = (l.column_origin % 128 + i % l.k) / 128;
            fp8(b) * scales[r * cols + c]
        })
        .collect()
}

#[test]
fn explicit_profile_identity_and_exact_budgets() {
    let l = layout(259, 259, NumericFp8ScaleType::F32);
    let p = NumericFp8LinearPlan::new(l, 3, 17 * 259 * 4 + 7, F32).unwrap();
    assert_eq!(p.precision(), F32);
    assert_eq!(p.padded_k(), 259);
    assert_eq!(p.activation_bytes(), 0);
    assert_eq!(p.weight_tile_bytes(), 17612);
    assert_eq!(p.workspace_requirements().bytes, 17612);
    assert_eq!(p.tile_rows(), 17);
    assert_eq!(p.tile_count(), 16);
    assert_eq!(p.kernel_launches(), 32);
    assert!(NumericFp8LinearPlan::new(l, 3, 4 * 259 - 1, F32).is_err());
    assert_eq!(
        NumericFp8LinearPlan::new(l, 3, 4 * 259, F32)
            .unwrap()
            .tile_rows(),
        1
    );
    let b = NumericFp8LinearPlan::new(l, 3, 2 * 264 * 4, BF16).unwrap();
    assert_eq!(b.tile_rows(), 1);
    assert_eq!(b.activation_bytes(), 1584);
    assert_eq!(b.workspace_requirements().bytes, 2112);
    assert!(NumericFp8LinearPlan::new(l, 3, 2111, BF16).is_err());
    // Identical layout, rows, budget and total bytes, but distinct profiles.
    let l = layout(3, 8, NumericFp8ScaleType::F32);
    let a = NumericFp8LinearPlan::new(l, 1, 64, BF16).unwrap();
    let b = NumericFp8LinearPlan::new(l, 1, 64, F32).unwrap();
    assert_eq!(a.workspace_requirements(), b.workspace_requirements());
    assert_ne!(a, b);
    for rows in [0, usize::MAX] {
        assert!(NumericFp8LinearPlan::new(l, rows, 64, F32).is_err());
    }
}

#[test]
#[ignore = "actual CUDA sm80+; independent F32/F64 reference, same weights under explicit profiles"]
fn gpu_f32_reference_odd_large_k_strides_tail_canaries() {
    let manifest = discover_provider().unwrap();
    assert!(manifest.supports(CutlassKernelId::F32Gemm));
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let consumer = ctx.new_stream().unwrap();
    for ty in [NumericFp8ScaleType::Bf16, NumericFp8ScaleType::F32] {
        for (m, n, k, tile) in [
            (1, 7, 3, 1),
            (3, 259, 259, 17),
            (65, 131, 129, 33),
            (3, 17, 8193, 3),
        ] {
            let l = layout(n, k, ty);
            let (s, sv) = scales(l);
            let codes = [0x00u8, 0x80, 0x01, 0x87, 0x28, 0x35, 0xb9, 0x3b, 0xc1];
            let raw: Vec<_> = (0..n * k)
                .map(|i| codes[(i * 7 + i / k) % codes.len()])
                .collect();
            let w = decode(l, &raw, &sv);
            let artifact = CudaNumericFp8Artifact::upload(&stream, l, &raw, &s).unwrap();
            let lda = k + 3;
            let ldd = n + 5;
            let a: Vec<_> = (0..m * lda)
                .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 503.7)
                .collect();
            let input_root =
                DeviceBuffer::from_host(&stream, &[&[29.0f32][..], &a, &[31.0][..]].concat())
                    .unwrap();
            let input = input_root.slice(1, a.len()).unwrap();
            let output_root =
                DeviceBuffer::from_host(&stream, &vec![7331.0f32; 3 + m * ldd + 4]).unwrap();
            let mut output = output_root.slice(3, m * ldd).unwrap();
            let budget = 4 * k * tile;
            let plan = NumericFp8LinearPlan::new(l, m, budget, F32).unwrap();
            let scratch_root =
                DeviceBuffer::from_host(&stream, &vec![0xa5u8; 16 + budget + 32]).unwrap();
            let mut scratch = CudaNumericFp8Workspace::from_buffer_with_precision(
                scratch_root.slice(16, budget).unwrap(),
                F32,
            );
            numeric_fp8_linear(
                &stream,
                &artifact,
                &input,
                &mut output,
                &mut scratch,
                plan,
                lda,
                ldd,
            )
            .unwrap();
            consumer.wait(&stream.record_event(None).unwrap()).unwrap();
            let got = output_root.to_host_vec(&consumer).unwrap();
            let scratch_bytes = scratch_root.to_host_vec(&consumer).unwrap();
            assert!(
                scratch_bytes[..16]
                    .iter()
                    .chain(&scratch_bytes[16 + budget..])
                    .all(|&v| v == 0xa5)
            );
            assert!(
                got[..3]
                    .iter()
                    .chain(&got[3 + m * ldd..])
                    .all(|&v| v == 7331.0)
            );
            // Final GPU-decoded tile is checked bitwise against F32 multiplication,
            // independently of GEMM and of either reduction order.
            let first = (n - 1) / tile * tile;
            for (i, expected) in w[first * k..].iter().enumerate() {
                let b = &scratch_bytes[16 + i * 4..16 + i * 4 + 4];
                assert_eq!(
                    u32::from_le_bytes(b.try_into().unwrap()),
                    expected.to_bits()
                );
            }
            // Full F32 reference is test-only; production retains just the tile.
            let wdev = DeviceBuffer::from_host(&stream, &w).unwrap();
            let mut direct = DeviceBuffer::<f32>::zeroed(&stream, m * n).unwrap();
            f32_gemm(
                &stream,
                &input,
                &wdev,
                &mut direct,
                F32GemmLayout {
                    activation_stride: lda,
                    ..F32GemmLayout::contiguous(m, n, k)
                },
            )
            .unwrap();
            let direct = direct.to_host_vec(&stream).unwrap();
            let mut max_error = 0.0f64;
            for r in 0..m {
                assert!(
                    got[3 + r * ldd + n..3 + (r + 1) * ldd]
                        .iter()
                        .all(|&v| v == 7331.0)
                );
                for c in 0..n {
                    let expected = (0..k)
                        .map(|t| f64::from(a[r * lda + t]) * f64::from(w[c * k + t]))
                        .sum::<f64>();
                    let serial =
                        (0..k).fold(0.0f32, |v, t| a[r * lda + t].mul_add(w[c * k + t], v));
                    let norm = (0..k)
                        .map(|t| (f64::from(a[r * lda + t]) * f64::from(w[c * k + t])).abs())
                        .sum::<f64>();
                    // Same norm-based accuracy bound as existing standard F32 tests.
                    let tolerance = 1e-6 + 3e-7 * norm + 3e-6 * expected.abs();
                    let actual = got[3 + r * ldd + c];
                    assert!(
                        actual.is_finite() && (f64::from(actual) - expected).abs() <= tolerance,
                        "{ty:?} {m}x{n}x{k} ({r},{c}): {actual}, F64={expected}, serial F32={serial}, tol={tolerance}"
                    );
                    assert_eq!(actual, direct[r * n + c], "tiled vs existing CUTLASS F32");
                    max_error = max_error.max((f64::from(actual) - expected).abs());
                }
            }
            assert_eq!(input.to_host_vec(&stream).unwrap(), a);
            // Same encoded operator and original input, explicitly select BF16.
            let bp =
                NumericFp8LinearPlan::new(l, m, 2 * (m + tile) * k.div_ceil(8) * 8, BF16).unwrap();
            let mut bs = CudaNumericFp8Workspace::from_buffer(
                DeviceBuffer::<u8>::zeroed(&stream, bp.workspace_requirements().bytes as usize)
                    .unwrap(),
            );
            numeric_fp8_linear(
                &stream,
                &artifact,
                &input,
                &mut output,
                &mut bs,
                bp,
                lda,
                ldd,
            )
            .unwrap();
            let bgot = output.to_host_vec(&stream).unwrap();
            for r in 0..m {
                for c in 0..n {
                    let expected = (0..k)
                        .map(|t| f64::from(bf16(a[r * lda + t])) * f64::from(bf16(w[c * k + t])))
                        .sum::<f64>();
                    let norm = (0..k)
                        .map(|t| {
                            (f64::from(bf16(a[r * lda + t])) * f64::from(bf16(w[c * k + t]))).abs()
                        })
                        .sum::<f64>();
                    assert!((f64::from(bgot[r * ldd + c]) - expected).abs() <= 1e-5 + 2e-6 * norm);
                }
            }
            eprintln!(
                "{ty:?} F32 {m}x{n}x{k}: scratch={budget}, tiles={}, launches={}, max F64 error={max_error}",
                plan.tile_count(),
                plan.kernel_launches()
            );
        }
    }
}

#[test]
#[ignore = "actual CUDA sm80+; small/large K cancellation without BF16 activation or weight rounding"]
fn gpu_cancellation_and_precision_boundaries() {
    let op = CudaOperators::new_on_device(0).unwrap();
    for k in [3usize, 8193] {
        let l = NumericFp8Layout {
            row_origin: 0,
            column_origin: 0,
            ..layout(3, k, NumericFp8ScaleType::F32)
        };
        let scale: Vec<_> = (0..k.div_ceil(128))
            .map(|i| 1.000137f32 + i as f32 * 0.0000117)
            .collect();
        let sb: Vec<_> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut raw = vec![0u8; 3 * k];
        let mut a = vec![1.0f32; k];
        for t in 0..k - 1 {
            raw[t] = if t % 2 == 0 { 0x38 } else { 0xb8 };
            a[t] = if t % 2 == 0 { 1.000217 } else { 1.0 };
        }
        raw[k] = 0x38;
        raw[2 * k + 1] = 0x38;
        let w = decode(l, &raw, &scale);
        let artifact = op.upload_numeric_fp8_linear(l, &raw, &sb).unwrap();
        let input = op.upload_f32_buffer(&a).unwrap();
        let mut output = op.zero_f32_buffer(3).unwrap();
        let p = NumericFp8LinearPlan::new(l, 1, 4 * k, F32).unwrap();
        let mut scratch = op.numeric_fp8_linear_workspace(p).unwrap();
        op.reset_counters();
        op.enable_capture_safe();
        op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut scratch, p, k, 3)
            .unwrap();
        op.disable_capture_safe();
        assert_eq!(op.counters().compute_kernel_launches, 6);
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(op.counters().stream_wide_syncs, 0);
        op.record_compute_event().unwrap().synchronize().unwrap();
        let got = op.download_f32_buffer(&output).unwrap();
        let direct_weight = op
            .upload_f32_linear(
                &w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                3,
                k,
            )
            .unwrap();
        let mut direct = op.zero_f32_buffer(3).unwrap();
        op.linear_f32_into(&direct_weight, &input, 1, &mut direct)
            .unwrap();
        assert_eq!(got, op.download_f32_buffer(&direct).unwrap());
        for c in 0..3 {
            let expected = (0..k)
                .map(|t| f64::from(a[t]) * f64::from(w[c * k + t]))
                .sum::<f64>();
            let serial = (0..k).fold(0.0f32, |sum, t| a[t].mul_add(w[c * k + t], sum));
            let norm = (0..k)
                .map(|t| (f64::from(a[t]) * f64::from(w[c * k + t])).abs())
                .sum::<f64>();
            // Existing standard F32 contract, not an output-relative "F32 exact"
            // promise. Large-K cancellation also affects the direct F32 GEMM.
            let tolerance = 1e-6 + 3e-7 * norm + 3e-6 * expected.abs();
            assert!((f64::from(got[c]) - expected).abs() <= tolerance);
            if k == 3 || c != 0 {
                assert!((f64::from(got[c]) - expected).abs() <= 2e-6 + 2e-5 * expected.abs());
            }
            eprintln!(
                "serial F32={serial}, absolute error={}, standard F32 bound={tolerance}",
                (f64::from(got[c]) - expected).abs()
            );
            eprintln!(
                "cancellation K={k} col={c}: TF32x3={} F64={expected}",
                got[c]
            );
        }
        let p = NumericFp8LinearPlan::new(l, 1, 4 * k.div_ceil(8) * 8, BF16).unwrap();
        let mut scratch = op.numeric_fp8_linear_workspace(p).unwrap();
        op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut scratch, p, k, 3)
            .unwrap();
        op.record_compute_event().unwrap().synchronize().unwrap();
        assert_eq!(op.download_f32_buffer(&output).unwrap(), [0.0, 1.0, 1.0]);
        assert!(got[0] > 0.0002 && got[1] > 1.0003 && got[2] > 1.0001);
    }
}

#[test]
#[ignore = "actual CUDA sm80+; profile identity, owners, alias, strides and budget preflight"]
fn gpu_f32_rejects_invalid_before_launch() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let other = CudaOperators::new_on_device(0).unwrap();
    let l = NumericFp8Layout {
        row_origin: 0,
        column_origin: 0,
        ..layout(3, 8, NumericFp8ScaleType::F32)
    };
    let w = op
        .upload_numeric_fp8_linear(l, &[0x38; 24], &1.0f32.to_le_bytes())
        .unwrap();
    let foreign_w = other
        .upload_numeric_fp8_linear(l, &[0x38; 24], &1.0f32.to_le_bytes())
        .unwrap();
    let input = op.upload_f32_buffer(&[1.0001; 8]).unwrap();
    let foreign_a = other.upload_f32_buffer(&[1.0001; 8]).unwrap();
    let mut output = op.upload_f32_buffer(&[73.0; 3]).unwrap();
    let mut foreign_d = other.upload_f32_buffer(&[73.0; 3]).unwrap();
    let p = NumericFp8LinearPlan::new(l, 1, 64, F32).unwrap();
    let bp = NumericFp8LinearPlan::new(l, 1, 64, BF16).unwrap();
    let mut ws = op.numeric_fp8_linear_workspace(p).unwrap();
    let mut bws = op.numeric_fp8_linear_workspace(bp).unwrap();
    let mut foreign_ws = other.numeric_fp8_linear_workspace(p).unwrap();
    let stream = op.stream_clone();
    let mut short = CudaNumericFp8Workspace::from_buffer_with_precision(
        DeviceBuffer::<u8>::zeroed(&stream, 63).unwrap(),
        F32,
    );
    let mut long = CudaNumericFp8Workspace::from_buffer_with_precision(
        DeviceBuffer::<u8>::zeroed(&stream, 65).unwrap(),
        F32,
    );
    let mut misaligned = CudaNumericFp8Workspace::from_buffer_with_precision(
        DeviceBuffer::<u8>::zeroed(&stream, 65)
            .unwrap()
            .slice(1, 64)
            .unwrap(),
        F32,
    );
    let mut legacy =
        CudaNumericFp8Workspace::from_buffer(DeviceBuffer::<u8>::zeroed(&stream, 64).unwrap());
    assert_eq!(legacy.precision(), BF16);
    assert_eq!(ws.precision(), F32);
    assert_eq!(bws.allocated_bytes(), ws.allocated_bytes());
    op.reset_counters();
    // Matching bytes does not authorize reinterpretation under a different profile.
    assert!(
        op.numeric_fp8_linear_into(&w, &input, &mut output, &mut bws, p, 8, 3)
            .is_err()
    );
    assert!(
        op.numeric_fp8_linear_into(&w, &input, &mut output, &mut ws, bp, 8, 3)
            .is_err()
    );
    for bad in [
        &mut foreign_ws,
        &mut short,
        &mut long,
        &mut misaligned,
        &mut legacy,
    ] {
        assert!(
            op.numeric_fp8_linear_into(&w, &input, &mut output, bad, p, 8, 3)
                .is_err()
        );
        assert!(!bad.is_poisoned());
    }
    assert!(
        op.numeric_fp8_linear_into(&foreign_w, &input, &mut output, &mut ws, p, 8, 3)
            .is_err()
    );
    assert!(
        op.numeric_fp8_linear_into(&w, &foreign_a, &mut output, &mut ws, p, 8, 3)
            .is_err()
    );
    assert!(
        op.numeric_fp8_linear_into(&w, &input, &mut foreign_d, &mut ws, p, 8, 3)
            .is_err()
    );
    for (lda, ldd) in [(7, 3), (usize::MAX, 3), (8, 2), (8, usize::MAX)] {
        assert!(
            op.numeric_fp8_linear_into(&w, &input, &mut output, &mut ws, p, lda, ldd)
                .is_err()
        );
    }
    let wrong =
        NumericFp8LinearPlan::new(NumericFp8Layout { row_origin: 1, ..l }, 1, 64, F32).unwrap();
    assert!(
        op.numeric_fp8_linear_into(&w, &input, &mut output, &mut ws, wrong, 8, 3)
            .is_err()
    );
    let mut alias = input.as_device_buffer().slice(0, 3).unwrap();
    assert!(
        numeric_fp8_linear(
            &stream,
            &w,
            input.as_device_buffer(),
            &mut alias,
            &mut ws,
            p,
            8,
            3
        )
        .is_err()
    );
    assert_eq!(op.counters().compute_kernel_launches, 0);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.download_f32_buffer(&output).unwrap(), [73.0; 3]);
    assert_eq!(other.download_f32_buffer(&foreign_d).unwrap(), [73.0; 3]);
    other.stream_clone().context().bind_to_thread().unwrap();
    op.numeric_fp8_linear_into(&w, &input, &mut output, &mut ws, p, 8, 3)
        .unwrap();
    op.record_compute_event().unwrap().synchronize().unwrap();
    assert!(!ws.is_poisoned());
    assert_eq!(op.counters().compute_kernel_launches, 4);
}

#[test]
#[ignore = "actual CUDA sm80+; allocation-free graph replay and exact cross-stream completion"]
fn gpu_f32_graph_replay_and_cross_stream_scratch_reuse() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let stream = op.stream_clone();
    let consumer = stream.context().new_stream().unwrap();
    let l = layout(7, 129, NumericFp8ScaleType::F32);
    let (s, sv) = scales(l);
    let raw: Vec<_> = (0..l.n * l.k)
        .map(|i| if i % 3 == 0 { 0xbc } else { 0x39 })
        .collect();
    let w = decode(l, &raw, &sv);
    let artifact = op.upload_numeric_fp8_linear(l, &raw, &s).unwrap();
    let values: Vec<_> = (0..2 * l.k)
        .map(|i| ((i * 17 % 89) as f32 - 43.0) * 0.07123)
        .collect();
    let mut input = op.upload_f32_buffer(&values).unwrap();
    let mut output = op.zero_f32_buffer(2 * l.n).unwrap();
    let p = NumericFp8LinearPlan::new(l, 2, l.k * 4 * 2, F32).unwrap();
    let mut scratch = op.numeric_fp8_linear_workspace(p).unwrap();
    op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut scratch, p, l.k, l.n)
        .unwrap();
    op.record_compute_event().unwrap().synchronize().unwrap();
    op.reset_counters();
    op.enable_capture_safe();
    let graph = op
        .capture_decode_graph(|| {
            op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut scratch, p, l.k, l.n)
        })
        .unwrap();
    op.disable_capture_safe();
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert_eq!(op.counters().stream_wide_syncs, 0);
    let mut consume_output = DeviceBuffer::<f32>::zeroed(&consumer, 2 * l.n).unwrap();
    let mut reuse_finished = consumer.record_event(None).unwrap();
    for factor in [1.0, -0.71, 1.0317] {
        reuse_finished.synchronize().unwrap();
        let a: Vec<_> = values.iter().map(|v| v * factor).collect();
        op.overwrite_f32_buffer(&a, &mut input).unwrap();
        op.launch_graph(&graph).unwrap();
        let produced = stream.record_event(None).unwrap();
        consumer.wait(&produced).unwrap();
        // Reuse the same scratch on another stream only after the graph's exact
        // event. The final event also gates the next activation/graph overwrite.
        numeric_fp8_linear(
            &consumer,
            &artifact,
            input.as_device_buffer(),
            &mut consume_output,
            &mut scratch,
            p,
            l.k,
            l.n,
        )
        .unwrap();
        reuse_finished = consumer.record_event(None).unwrap();
        reuse_finished.synchronize().unwrap();
        let got = op.download_f32_buffer(&output).unwrap();
        assert_eq!(got, consume_output.to_host_vec(&consumer).unwrap());
        for r in 0..2 {
            for c in 0..l.n {
                let expected = (0..l.k)
                    .map(|t| f64::from(a[r * l.k + t]) * f64::from(w[c * l.k + t]))
                    .sum::<f64>();
                let norm = (0..l.k)
                    .map(|t| (f64::from(a[r * l.k + t]) * f64::from(w[c * l.k + t])).abs())
                    .sum::<f64>();
                assert!(
                    (f64::from(got[r * l.n + c]) - expected).abs()
                        <= 1e-6 + 3e-7 * norm + 3e-6 * expected.abs()
                );
            }
        }
    }
}
