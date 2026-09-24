#![cfg(feature = "cuda")]

use ferrule_backend::cuda::context::CudaOperators;
use ferrule_backend::cuda::providers::cutlass::{self, CutlassKernelId};

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn reference(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    (0..m * n)
        .map(|index| {
            let row = index / n;
            let col = index % n;
            (0..k)
                .map(|t| f64::from(a[row * k + t]) * f64::from(w[col * k + t]))
                .sum::<f64>() as f32
        })
        .collect()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() < 3e-5 * (1.0 + e.abs()),
            "element {i}: {a} vs {e}"
        );
    }
}

#[test]
#[ignore = "requires native CUDA GPU; CUTLASS standard linear regression"]
fn standard_linear_f32_decode_prefill_and_large_k() {
    assert!(
        cutlass::discover_provider()
            .unwrap()
            .supports(CutlassKernelId::F32Gemm)
    );
    let op = CudaOperators::new_on_device(0).unwrap();
    for (m, n, k) in [(1, 7, 3), (7, 19, 65), (65, 131, 257), (3, 17, 8193)] {
        let a: Vec<_> = (0..m * k)
            .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 503.7)
            .collect();
        let w: Vec<_> = (0..n * k)
            .map(|i| ((i * 17 % 997) as f32 - 498.0) / 501.3)
            .collect();
        let weight = op.upload_f32_linear(&bytes(&w), n, k).unwrap();
        let input = op.upload_f32_buffer(&a).unwrap();
        let mut output = op.upload_f32_buffer(&vec![f32::NAN; m * n]).unwrap();
        op.reset_counters();
        op.enable_capture_safe();
        op.linear_f32_into(&weight, &input, m, &mut output).unwrap();
        op.disable_capture_safe();
        let counters = op.counters();
        assert_eq!(counters.compute_kernel_launches, 1);
        assert_eq!(counters.device_allocation_attempts, 0);
        assert_eq!(counters.stream_wide_syncs, 0);
        // Existing event boundary covers the new provider launch as before.
        op.record_compute_event().unwrap().synchronize().unwrap();
        let actual = op.download_f32_buffer(&output).unwrap();
        let expected = reference(&a, &w, m, n, k);
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        let max_serial_error = (0..m * n)
            .map(|i| {
                let serial = (0..k).fold(0.0f32, |sum, t| {
                    a[(i / n) * k + t].mul_add(w[(i % n) * k + t], sum)
                });
                (serial - expected[i]).abs()
            })
            .fold(0.0f32, f32::max);
        eprintln!(
            "{m}x{n}x{k}: maximum F64-reference error CUTLASS={max_error}, serial F32 FMA={max_serial_error}"
        );
        // Cancellation makes a fixed output-relative tolerance inappropriate:
        // bound multiplication/accumulation error using the dot-product norm.
        for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let sum_abs = (0..k)
                .map(|t| (f64::from(a[(i / n) * k + t]) * f64::from(w[(i % n) * k + t])).abs())
                .sum::<f64>();
            let tolerance = 1e-6 + 3e-7 * sum_abs + 3e-6 * f64::from(want.abs());
            assert!(
                got.is_finite() && f64::from((got - want).abs()) <= tolerance,
                "{m}x{n}x{k} element {i}: {got} vs {want}, tolerance={tolerance}"
            );
        }
        assert_eq!(op.download_f32_buffer(&input).unwrap(), a);
    }
}

#[test]
#[ignore = "requires native CUDA GPU; allocation-free capture/replay regression"]
fn standard_linear_f32_graph_capture_replay() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let w = [1.000217f32, -1.0, 0.333123, 0.718923, 1.217833, -0.981723];
    let a = [1.000123f32, 1.0, -0.883713, 1.000331];
    let weight = op.upload_f32_linear(&bytes(&w), 3, 2).unwrap();
    let mut input = op.upload_f32_buffer(&a).unwrap();
    let mut output = op.zero_f32_buffer(6).unwrap();
    let mut consumed = op.zero_f32_buffer(6).unwrap();
    let addresses = (
        input.as_device_buffer().cu_deviceptr(),
        output.as_device_buffer().cu_deviceptr(),
    );
    op.linear_f32_into(&weight, &input, 2, &mut output).unwrap();
    let mut quiesced = op.record_compute_event().unwrap();
    quiesced.synchronize().unwrap();
    op.enable_capture_safe();
    let graph = op
        .capture_decode_graph(|| op.linear_f32_into(&weight, &input, 2, &mut output))
        .unwrap();
    op.disable_capture_safe();

    let mut previous = reference(&a, &w, 2, 3, 2);
    for (round, values) in [
        [1.000123f32, 1.0, 0.381713, -2.000331],
        [-0.371231, 1.241723, 2.18231, 0.417931],
        [0.0, -3.37123, -1.72913, 0.618273],
    ]
    .into_iter()
    .enumerate()
    {
        // Graphs borrow device addresses, not buffer ownership. Retain all
        // allocations, and finish the previous consumer before any overwrite.
        quiesced.synchronize().unwrap();
        assert!(quiesced.is_complete().unwrap());
        op.overwrite_f32_buffer(&values, &mut input).unwrap();
        let poison = if round % 2 == 0 { f32::NAN } else { -777.0 };
        op.overwrite_f32_buffer(&[poison; 6], &mut output).unwrap();
        op.overwrite_f32_buffer(&[poison; 6], &mut consumed)
            .unwrap();
        assert_eq!(
            addresses,
            (
                input.as_device_buffer().cu_deviceptr(),
                output.as_device_buffer().cu_deviceptr(),
            )
        );
        let expected = reference(&values, &w, 2, 3, 2);
        assert_ne!(expected, previous);
        op.reset_counters();
        op.enable_capture_safe();
        op.launch_graph(&graph).unwrap();
        let completed = op.record_compute_event().unwrap();
        // A real device-side consumer reads D before the next round may reuse it.
        op.copy_f32_range(&output, 0, &mut consumed, 0, 6).unwrap();
        quiesced = op.compute_stream_authority().record_event().unwrap();
        op.disable_capture_safe();
        completed.synchronize().unwrap();
        quiesced.synchronize().unwrap();
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(op.counters().stream_wide_syncs, 0);
        let actual = op.download_f32_buffer(&output).unwrap();
        close(&actual, &expected);
        close(&op.download_f32_buffer(&consumed).unwrap(), &expected);
        assert_eq!(op.download_f32_buffer(&input).unwrap(), values);
        if round == 0 {
            // A single-TF32/BF16 multiplication loses this cancellation residual.
            assert!((actual[0] - expected[0]).abs() < 2e-7);
        }
        previous = expected;
    }
    quiesced.synchronize().unwrap();
    drop(graph); // Destroy the graph before releasing its borrowed allocations.
}

#[test]
#[ignore = "requires native CUDA GPU; invalid standard inputs never launch"]
fn standard_linear_f32_rejects_invalid_shapes_storage_and_owners() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let weights = bytes(&[1.0; 15]);
    let weight = op.upload_f32_linear(&weights, 3, 5).unwrap();
    let other_weight = foreign.upload_f32_linear(&weights, 3, 5).unwrap();
    let bf16 = op.upload_bf16_linear(&vec![0; 30], 3, 5).unwrap();
    let a = op.upload_f32_buffer(&[1.0; 10]).unwrap();
    let other_a = foreign.upload_f32_buffer(&[1.0; 10]).unwrap();
    let mut d = op.upload_f32_buffer(&[73.0; 6]).unwrap();
    let mut other_d = foreign.upload_f32_buffer(&[73.0; 6]).unwrap();
    let short_a = op.upload_f32_buffer(&[1.0; 9]).unwrap();
    let mut short_d = op.upload_f32_buffer(&[73.0; 5]).unwrap();
    op.reset_counters();
    for rows in [0, 1, 3, usize::MAX] {
        assert!(op.linear_f32_into(&weight, &a, rows, &mut d).is_err());
    }
    assert!(op.linear_f32_into(&bf16, &a, 2, &mut d).is_err());
    assert!(op.linear_f32_into(&other_weight, &a, 2, &mut d).is_err());
    assert!(op.linear_f32_into(&weight, &other_a, 2, &mut d).is_err());
    assert!(op.linear_f32_into(&weight, &a, 2, &mut other_d).is_err());
    assert!(op.linear_f32_into(&weight, &short_a, 2, &mut d).is_err());
    assert!(op.linear_f32_into(&weight, &a, 2, &mut short_d).is_err());
    assert_eq!(op.counters().compute_kernel_launches, 0);
    assert_eq!(op.download_f32_buffer(&d).unwrap(), [73.0; 6]);
    assert_eq!(op.download_f32_buffer(&short_d).unwrap(), [73.0; 5]);
    assert_eq!(foreign.download_f32_buffer(&other_d).unwrap(), [73.0; 6]);
}
