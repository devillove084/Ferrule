//! FP8 E4M3 Tensor Core smoke tests for GB10 (`sm_121a`).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrule_cuda::context::CudaArtifactOperatorContext;
use ferrule_cuda::kernels::kernels;
use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_guard() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .expect("CUDA FP8 MMA smoke test lock poisoned")
}

fn has_cuda() -> bool {
    CudaContext::new(0).is_ok()
}

fn fp8_e4m3fn(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        return sign * mantissa as f32 / 512.0;
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    sign * 2.0f32.powi(exponent as i32 - 7) * (1.0 + mantissa as f32 / 8.0)
}

fn e8m0_scale(byte: u8) -> f32 {
    if byte == 0 {
        2.0f32.powi(-127)
    } else {
        2.0f32.powi(byte as i32 - 127)
    }
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let round_bit = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + round_bit);
    f32::from_bits(rounded & 0xffff_0000)
}

fn fp8_nearest_byte(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    if value == 0.0 {
        return sign;
    }
    let magnitude = value.abs().min(448.0);
    let mut best = 0u8;
    let mut best_error = magnitude;
    for code in 1..=0x7e {
        let candidate = fp8_e4m3fn(code);
        let error = (candidate - magnitude).abs();
        if error < best_error {
            best = code;
            best_error = error;
        }
    }
    sign | best
}

fn activation_pack_reference(values: &[f32], row_width: usize) -> (Vec<u8>, Vec<u8>) {
    const BLOCK: usize = 128;
    let rows = values.len() / row_width;
    let blocks_per_row = row_width / BLOCK;
    let mut packed = vec![0u8; values.len()];
    let mut scales = vec![0u8; rows * blocks_per_row];
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let start = row * row_width + block * BLOCK;
            let amax = values[start..start + BLOCK]
                .iter()
                .fold(1e-4f32, |acc, value| acc.max(value.abs()));
            let scale_byte = ((amax / 448.0).log2().ceil() as i32 + 127).clamp(0, 255) as u8;
            let scale = e8m0_scale(scale_byte);
            scales[row * blocks_per_row + block] = scale_byte;
            for i in 0..BLOCK {
                packed[start + i] =
                    fp8_nearest_byte((values[start + i] / scale).clamp(-448.0, 448.0));
            }
        }
    }
    (packed, scales)
}

#[test]
#[allow(
    unsafe_code,
    reason = "validated CUDA smoke buffers and launch geometry require a raw kernel launch"
)]
fn fp8_activation_pack_matches_cpu_reference() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    const ROWS: usize = 2;
    const K: usize = 256;
    const BLOCK: usize = 128;

    let mut values = vec![0.0f32; ROWS * K];
    for row in 0..ROWS {
        for col in 0..K {
            let centered = (col % 31) as f32 - 15.0;
            let magnitude = if col < BLOCK { 0.25 } else { 0.000_75 };
            values[row * K + col] = centered * magnitude * (row + 1) as f32;
        }
    }
    let (expected_packed, expected_scales) = activation_pack_reference(&values, K);

    let ctx = CudaContext::new(0).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let module =
        kernels::load(&ctx).unwrap_or_else(|error| panic!("module load failed: {error:?}"));
    let stream = ctx.default_stream();
    let values_dev = DeviceBuffer::from_host(&stream, &values).expect("values");
    let mut packed_dev = DeviceBuffer::<u8>::zeroed(&stream, values.len()).expect("packed");
    let mut scales_dev = DeviceBuffer::<u8>::zeroed(&stream, ROWS * K / BLOCK).expect("scales");
    unsafe {
        module.fp8_e4m3fn_e8m0_quantize_f32_packed(
            &stream,
            LaunchConfig::for_num_elems((ROWS * K / BLOCK) as u32),
            &values_dev,
            &mut packed_dev,
            &mut scales_dev,
            values.len() as u32,
            K as u32,
            BLOCK as u32,
        )
    }
    .expect("pack launch");

    let actual_packed = packed_dev.to_host_vec(&stream).expect("packed download");
    let actual_scales = scales_dev.to_host_vec(&stream).expect("scale download");
    assert_eq!(actual_scales, expected_scales);
    assert_eq!(actual_packed, expected_packed);
}

#[test]
#[allow(
    unsafe_code,
    reason = "fixed-size CUDA smoke buffers match the one-warp kernel contract"
)]
fn fp8_mma_tile_matches_scalar_reference() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    const M: usize = 16;
    const N: usize = 8;
    const K: usize = 128;

    let positive_codes = [0x20u8, 0x28, 0x30, 0x38, 0x3c, 0x40, 0x44];
    let mut weight = vec![0u8; M * K];
    let mut x = vec![0u8; N * K];
    for row in 0..M {
        for col in 0..K {
            let sign = if (row * 7 + col * 3).is_multiple_of(11) {
                0x80
            } else {
                0
            };
            weight[row * K + col] = positive_codes[(row + col * 5) % positive_codes.len()] | sign;
        }
    }
    for row in 0..N {
        for col in 0..K {
            let sign = if (row * 5 + col).is_multiple_of(13) {
                0x80
            } else {
                0
            };
            x[row * K + col] = positive_codes[(row * 3 + col * 2) % positive_codes.len()] | sign;
        }
    }
    let weight_scales = vec![126u8];
    let x_scales: Vec<u8> = (0..N).map(|row| 125 + (row % 4) as u8).collect();
    let mut expected = vec![0.0f32; N * M];
    for batch in 0..N {
        for row in 0..M {
            let mut dot = 0.0f32;
            for col in 0..K {
                dot += fp8_e4m3fn(weight[row * K + col]) * fp8_e4m3fn(x[batch * K + col]);
            }
            expected[batch * M + row] =
                dot * e8m0_scale(weight_scales[0]) * e8m0_scale(x_scales[batch]);
        }
    }

    let ctx = CudaContext::new(0).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let module =
        kernels::load(&ctx).unwrap_or_else(|error| panic!("module load failed: {error:?}"));
    let stream = ctx.default_stream();
    let x_dev = DeviceBuffer::from_host(&stream, &x).expect("x");
    let xs_dev = DeviceBuffer::from_host(&stream, &x_scales).expect("x scales");
    let weight_dev = DeviceBuffer::from_host(&stream, &weight).expect("weight");
    let ws_dev = DeviceBuffer::from_host(&stream, &weight_scales).expect("weight scales");
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N * M).expect("output");
    unsafe {
        module.gemm_fp8_e4m3fn_e8m0_2d_mma(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &x_dev,
            &xs_dev,
            &weight_dev,
            &ws_dev,
            &mut output_dev,
            N as u32,
            M as u32,
            K as u32,
            1,
        )
    }
    .expect("MMA launch");
    let actual = output_dev.to_host_vec(&stream).expect("output download");

    let mut max_abs = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let error = (actual - expected).abs();
        max_abs = max_abs.max(error);
        let tolerance = 2e-3f32.max(expected.abs() * 2e-4);
        assert!(
            error <= tolerance,
            "FP8 MMA mismatch at {index}: actual={actual} expected={expected} error={error} tolerance={tolerance}"
        );
    }
    println!("FP8 MMA tile OK: max_abs={max_abs}");
}

#[test]
fn grouped_output_a_bf16_mma_matches_grouped_reference() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    const ROWS: usize = 8;
    const GROUPS: usize = 2;
    const GROUP_IN: usize = 128;
    const RANK: usize = 16;
    const OUT: usize = GROUPS * RANK;

    let positive_codes = [0x20u8, 0x28, 0x30, 0x38, 0x3c, 0x40, 0x44];
    let mut weight = vec![0u8; OUT * GROUP_IN];
    for row in 0..OUT {
        for col in 0..GROUP_IN {
            let sign = if (row * 3 + col * 5).is_multiple_of(19) {
                0x80
            } else {
                0
            };
            weight[row * GROUP_IN + col] =
                positive_codes[(row * 7 + col) % positive_codes.len()] | sign;
        }
    }
    let weight_scales = vec![126u8];
    let mut context = vec![0.0f32; ROWS * GROUPS * GROUP_IN];
    for token in 0..ROWS {
        for group in 0..GROUPS {
            for col in 0..GROUP_IN {
                let centered = (col % 23) as f32 - 11.0;
                context[(token * GROUPS + group) * GROUP_IN + col] =
                    centered * 0.015_625 * (token + group + 1) as f32;
            }
        }
    }

    let mut expected = vec![0.0f32; ROWS * OUT];
    let weight_scale = e8m0_scale(weight_scales[0]);
    for token in 0..ROWS {
        for out_row in 0..OUT {
            let group = out_row / RANK;
            let mut dot = 0.0f32;
            for col in 0..GROUP_IN {
                let x = bf16_round(context[(token * GROUPS + group) * GROUP_IN + col]);
                let w = bf16_round(fp8_e4m3fn(weight[out_row * GROUP_IN + col]) * weight_scale);
                dot += x * w;
            }
            expected[token * OUT + out_row] = bf16_round(dot);
        }
    }

    let ops = CudaArtifactOperatorContext::new().expect("operator context");
    let handle = ops
        .upload_fp8_e4m3_e8m0_linear(&weight, &weight_scales, OUT, GROUP_IN, 128, 128)
        .expect("upload grouped WO-A");
    assert!(ops.grouped_output_a_bf16_mma_supported(&handle, OUT, GROUP_IN, RANK));
    let context_dev = ops.upload_f32_buffer(&context).expect("upload context");
    let mut output_dev = ops.zero_f32_buffer(ROWS * OUT).expect("output");
    ops.grouped_output_a_bf16_mma_from_device_into(
        &context_dev,
        ROWS,
        &handle,
        OUT,
        GROUP_IN,
        RANK,
        &mut output_dev,
    )
    .expect("grouped WO-A MMA");
    let actual = ops
        .download_f32_buffer(&output_dev)
        .expect("output download");

    let mut max_abs = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let error = (actual - expected).abs();
        max_abs = max_abs.max(error);
        let tolerance = 0.031_25f32.max(expected.abs() * 0.01);
        assert!(
            error <= tolerance,
            "grouped BF16 MMA mismatch at {index}: actual={actual} expected={expected} error={error} tolerance={tolerance}"
        );
    }
    println!("grouped WO-A BF16 MMA OK: max_abs={max_abs}");
}

#[test]
fn fp8_mma_rows_match_allocation_free_matvec() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    const ROWS: usize = 8;
    const OUT: usize = 16;
    const K: usize = 1024;

    let positive_codes = [0x20u8, 0x28, 0x30, 0x38, 0x3c, 0x40, 0x44];
    let mut weight = vec![0u8; OUT * K];
    for row in 0..OUT {
        for col in 0..K {
            let sign = if (row + col * 3).is_multiple_of(17) {
                0x80
            } else {
                0
            };
            weight[row * K + col] = positive_codes[(row * 5 + col) % positive_codes.len()] | sign;
        }
    }
    let weight_scales = (0..K / 128)
        .map(|block| 125 + (block % 5) as u8)
        .collect::<Vec<_>>();
    let mut input = vec![0.0f32; ROWS * K];
    for row in 0..ROWS {
        for col in 0..K {
            let centered = (col % 29) as f32 - 14.0;
            let block_scale = 2.0f32.powi((col / 128) as i32 - 4);
            input[row * K + col] = centered * 0.031_25 * (row + 1) as f32 * block_scale;
        }
    }

    let ops = CudaArtifactOperatorContext::new().expect("operator context");
    let handle = ops
        .upload_fp8_e4m3_e8m0_linear(&weight, &weight_scales, OUT, K, 128, 128)
        .expect("upload linear");
    assert!(ops.artifact_linear_uses_fp8_mma(&handle));
    let input_dev = ops.upload_f32_buffer(&input).expect("upload input");
    let rows_output = ops
        .artifact_linear_rows_from_device(&handle, &input_dev, ROWS)
        .expect("rows MMA");
    let rows_output = ops
        .download_f32_buffer(&rows_output)
        .expect("rows download");

    for row in 0..ROWS {
        let matvec = ops
            .artifact_linear_matvec(&handle, &input[row * K..(row + 1) * K])
            .expect("matvec MMA");
        for col in 0..OUT {
            let rows_value = rows_output[row * OUT + col];
            let matvec_value = matvec[col];
            assert_eq!(
                rows_value.to_bits(),
                matvec_value.to_bits(),
                "rows/matvec mismatch at row={row} col={col}: rows={rows_value} matvec={matvec_value}"
            );
        }
    }
}
