#![cfg(feature = "cuda")]

//! Artifact-format CUDA kernel smoke tests.
//!
//! These tests skip on machines without CUDA. On CUDA machines, launch failures
//! and deterministic output mismatches fail the test instead of only printing a
//! `[FAIL]` line.

use ferrule_backend::cuda::context::cuda_gemv_fp8_e4m3fn_e8m0_2d;
use ferrule_backend::cuda::kernels::kernels;
use ferrule_backend::cuda::runtime::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use ferrule_common::Result;
use std::sync::Arc;

fn rc<T, E>(result: std::result::Result<T, E>) -> Result<T>
where
    E: std::error::Error + Send + Sync + 'static,
{
    result.map_err(|source| ferrule_common::Error::Backend {
        source: Box::new(source),
    })
}

fn has_cuda() -> bool {
    CudaContext::new(0).is_ok()
}

fn load() -> Result<(Arc<CudaContext>, kernels::LoadedModule, Arc<CudaStream>)> {
    let ctx = rc(CudaContext::new(0))?;
    rc(ctx.bind_to_thread())?;
    let module = rc(kernels::load(&ctx))?;
    let stream = ctx.default_stream();
    Ok((ctx, module, stream))
}

fn assert_cuda<T, E: std::fmt::Debug>(r: std::result::Result<T, E>, label: &str) -> T {
    match r {
        Ok(value) => {
            eprintln!("  [PASS] {label}");
            value
        }
        Err(err) => panic!("  [FAIL] {label}: {err:?}"),
    }
}

fn assert_close(actual: f32, expected: f32, tolerance: f32, label: &str) {
    assert!(
        actual.is_finite(),
        "{label}: expected finite value, got {actual}"
    );
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: expected {expected}, got {actual}"
    );
}

fn assert_close_slice(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: output length mismatch"
    );
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, tolerance, &format!("{label}[{idx}]"));
    }
}

#[test]
#[allow(
    unsafe_code,
    reason = "validated smoke test buffers and launch geometry require raw kernel launches"
)]
fn bf16_linear_matches_cpu_reference_for_real_dsv4_shapes() {
    if !has_cuda() {
        eprintln!("SKIP: no CUDA");
        return;
    }

    let (_ctx, module, stream) = assert_cuda(load(), "load CUDA kernel module");
    let k = 4096usize;
    let x = (0..k)
        .map(|col| ((col * 5 % 23) as i32 - 11) as f32 / 32.0)
        .collect::<Vec<_>>();
    let x_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &x),
        "upload BF16 GEMV input",
    );

    for n in [64usize, 256, 512, 1024] {
        let mut weight = Vec::with_capacity(n * k * 2);
        for row in 0..n {
            for col in 0..k {
                let value = ((row * 17 + col * 13) % 29) as i32 - 14;
                let bf16 = ((value as f32 / 64.0).to_bits() >> 16) as u16;
                weight.extend_from_slice(&bf16.to_le_bytes());
            }
        }
        let weight_dev = assert_cuda(
            DeviceBuffer::from_host(&stream, &weight),
            &format!("upload BF16 GEMV {n}x{k} weight"),
        );
        let mut output_dev = assert_cuda(
            DeviceBuffer::<f32>::zeroed(&stream, n),
            "allocate BF16 linear output",
        );
        let mut block_pair_first_dev = assert_cuda(
            DeviceBuffer::<f32>::zeroed(&stream, n),
            "allocate first block-pair BF16 GEMV output",
        );
        let mut block_pair_second_dev = assert_cuda(
            DeviceBuffer::<f32>::zeroed(&stream, n),
            "allocate second block-pair BF16 GEMV output",
        );

        assert_cuda(
            unsafe {
                module.linear_bf16_from_f32(
                    &stream,
                    LaunchConfig {
                        grid_dim: (n as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &x_dev,
                    &weight_dev,
                    &mut output_dev,
                    n as u32,
                    k as u32,
                )
            },
            &format!("BF16 linear {n}x{k}"),
        );
        assert_cuda(
            unsafe {
                module.dual_linear_bf16_from_f32(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((2 * n) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &x_dev,
                    &weight_dev,
                    &weight_dev,
                    &mut block_pair_first_dev,
                    &mut block_pair_second_dev,
                    n as u32,
                    n as u32,
                    k as u32,
                )
            },
            &format!("block-pair BF16 GEMV {n}x{k}"),
        );

        let output = assert_cuda(output_dev.to_host_vec(&stream), "download BF16 linear");
        let block_pair_first = assert_cuda(
            block_pair_first_dev.to_host_vec(&stream),
            "download first block-pair BF16 GEMV",
        );
        let block_pair_second = assert_cuda(
            block_pair_second_dev.to_host_vec(&stream),
            "download second block-pair BF16 GEMV",
        );
        assert_eq!(
            block_pair_first, output,
            "first dual-linear output must match the standalone operator for {n}x{k}"
        );
        assert_eq!(
            block_pair_second, output,
            "second dual-linear output must match the standalone operator for {n}x{k}"
        );
        for row in 0..n {
            let expected = (0..k)
                .map(|col| {
                    let offset = (row * k + col) * 2;
                    let bits = u16::from_le_bytes([weight[offset], weight[offset + 1]]);
                    f32::from_bits(u32::from(bits) << 16) * x[col]
                })
                .sum::<f32>();
            let tolerance = 5e-4f32.max(expected.abs() * 5e-4);
            assert_close(
                output[row],
                expected,
                tolerance,
                &format!("BF16 linear {n}x{k} row {row}"),
            );
        }
    }
}

#[test]
#[allow(
    unsafe_code,
    reason = "validated smoke test buffers and launch geometry require raw kernel launches"
)]
fn bf16_rows_cover_batch_and_channel_tails() {
    if !has_cuda() {
        eprintln!("SKIP: no CUDA");
        return;
    }

    const ROWS: usize = 9;
    const N: usize = 65;
    const K: usize = 16;

    let (_ctx, module, stream) = assert_cuda(load(), "load CUDA kernel module");
    let input = assert_cuda(
        DeviceBuffer::from_host(&stream, &vec![1.0f32; ROWS * K]),
        "upload BF16 rows input",
    );
    let weight = assert_cuda(
        DeviceBuffer::from_host(&stream, &vec![0x80u8, 0x3f].repeat(N * K)),
        "upload BF16 rows weight",
    );
    let mut output = assert_cuda(
        DeviceBuffer::<f32>::zeroed(&stream, ROWS * N),
        "allocate BF16 rows output",
    );

    assert_cuda(
        unsafe {
            module.linear_rows_bf16_from_f32(
                &stream,
                LaunchConfig {
                    grid_dim: (N.div_ceil(64) as u32, ROWS.div_ceil(8) as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                &input,
                &weight,
                &mut output,
                ROWS as u32,
                N as u32,
                K as u32,
            )
        },
        "launch BF16 rows tail smoke",
    );

    let output = assert_cuda(output.to_host_vec(&stream), "download BF16 rows output");
    assert_eq!(output, vec![16.0f32; ROWS * N]);
}

#[test]
#[allow(
    unsafe_code,
    reason = "validated smoke test buffers and launch geometry require raw kernel launches"
)]
fn payload_encoding_kernels_produce_expected_tiny_outputs() {
    if !has_cuda() {
        eprintln!("SKIP: no CUDA");
        return;
    }

    let fp8 = assert_cuda(
        cuda_gemv_fp8_e4m3fn_e8m0_2d(&[1.0, 2.0], &[0x38u8; 4], &[127u8; 4], 2, 2, 1, 1),
        "standalone FP8 GEMV 2D",
    );
    assert_close_slice(&fp8, &[3.0, 3.0], 1e-4, "standalone FP8 GEMV 2D");

    let (_ctx, module, stream) = assert_cuda(load(), "load CUDA kernel module");

    let (rope_dim, heads) = (64usize, 4usize);
    let mut qk = vec![0.0f32; heads * rope_dim];
    for (idx, value) in qk.iter_mut().enumerate() {
        *value = idx as f32 * 0.01;
    }
    let mut qk_dev = assert_cuda(DeviceBuffer::from_host(&stream, &qk), "upload RoPE input");
    let cos = vec![0.5f32; rope_dim / 2];
    let sin = vec![0.866f32; rope_dim / 2];
    let cos_dev = assert_cuda(DeviceBuffer::from_host(&stream, &cos), "upload RoPE cos");
    let sin_dev = assert_cuda(DeviceBuffer::from_host(&stream, &sin), "upload RoPE sin");
    assert_cuda(
        unsafe {
            module.rope_yarn(
                &stream,
                LaunchConfig::for_num_elems((heads * rope_dim) as u32),
                &mut qk_dev,
                &cos_dev,
                &sin_dev,
                (heads * rope_dim) as u32,
                rope_dim as u32,
                rope_dim as u32,
            )
        },
        "RoPE/YaRN launch",
    );
    let qk_out = assert_cuda(qk_dev.to_host_vec(&stream), "download RoPE output");
    assert_close(qk_out[0], -0.00866, 1e-4, "RoPE first component");
    assert_close(qk_out[1], 0.005, 1e-4, "RoPE second component");
}
