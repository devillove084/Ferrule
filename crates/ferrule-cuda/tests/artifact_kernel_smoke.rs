//! Artifact-format CUDA kernel smoke tests.
//!
//! These tests skip on machines without CUDA. On CUDA machines, launch failures
//! and deterministic output mismatches fail the test instead of only printing a
//! `[FAIL]` line.

use cuda_core::stream::CudaStream;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrule_common::Result;
use ferrule_cuda::context::{
    cuda_gemv_fp4_e2m1_e8m0, cuda_gemv_fp8_e4m3fn_e8m0_2d, cuda_sparse_attention_sink_f32,
};
use ferrule_cuda::kernels::kernels;
use std::sync::Arc;

fn rc<T, E: std::fmt::Debug>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| ferrule_common::Error::Internal(format!("{e:?}")))
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
fn bf16_block_gemv_matches_legacy_for_real_dsv4_shapes() {
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
        let mut legacy_dev = assert_cuda(
            DeviceBuffer::<f32>::zeroed(&stream, n),
            "allocate legacy BF16 GEMV output",
        );
        let mut block_dev = assert_cuda(
            DeviceBuffer::<f32>::zeroed(&stream, n),
            "allocate block BF16 GEMV output",
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
                module.gemv_bf16_bytes(
                    &stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &x_dev,
                    &weight_dev,
                    &mut legacy_dev,
                    n as u32,
                    k as u32,
                )
            },
            &format!("legacy BF16 GEMV {n}x{k}"),
        );
        assert_cuda(
            unsafe {
                module.gemv_bf16_bytes_block(
                    &stream,
                    LaunchConfig {
                        grid_dim: (n as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &x_dev,
                    &weight_dev,
                    &mut block_dev,
                    n as u32,
                    k as u32,
                )
            },
            &format!("block BF16 GEMV {n}x{k}"),
        );
        assert_cuda(
            unsafe {
                module.gemv_bf16_bytes_block_pair(
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

        let legacy = assert_cuda(legacy_dev.to_host_vec(&stream), "download legacy BF16 GEMV");
        let block = assert_cuda(block_dev.to_host_vec(&stream), "download block BF16 GEMV");
        let block_pair_first = assert_cuda(
            block_pair_first_dev.to_host_vec(&stream),
            "download first block-pair BF16 GEMV",
        );
        let block_pair_second = assert_cuda(
            block_pair_second_dev.to_host_vec(&stream),
            "download second block-pair BF16 GEMV",
        );
        assert_eq!(
            block_pair_first, block,
            "first block-pair output must preserve standalone block reduction for {n}x{k}"
        );
        assert_eq!(
            block_pair_second, block,
            "second block-pair output must preserve standalone block reduction for {n}x{k}"
        );
        for (row, (&expected, &actual)) in legacy.iter().zip(&block).enumerate() {
            let tolerance = 5e-4f32.max(expected.abs() * 5e-4);
            assert_close(
                actual,
                expected,
                tolerance,
                &format!("BF16 block GEMV {n}x{k} row {row}"),
            );
        }
    }
}

#[test]
#[allow(
    unsafe_code,
    reason = "validated smoke test buffers and launch geometry require raw kernel launches"
)]
fn artifact_format_kernels_produce_expected_tiny_outputs() {
    if !has_cuda() {
        eprintln!("SKIP: no CUDA");
        return;
    }

    let fp4 = assert_cuda(
        cuda_gemv_fp4_e2m1_e8m0(&[1.0; 32], &[0x42u8; 16], &[127u8; 1], 1, 32),
        "standalone FP4 GEMV",
    );
    assert_close_slice(&fp4, &[48.0], 1e-4, "standalone FP4 GEMV");

    let fp8 = assert_cuda(
        cuda_gemv_fp8_e4m3fn_e8m0_2d(&[1.0, 2.0], &[0x38u8; 4], &[127u8; 4], 2, 2, 1, 1),
        "standalone FP8 GEMV 2D",
    );
    assert_close_slice(&fp8, &[3.0, 3.0], 1e-4, "standalone FP8 GEMV 2D");

    let (_ctx, module, stream) = assert_cuda(load(), "load CUDA kernel module");

    let x = vec![1.0f32; 64];
    let packed = vec![0x42u8; 16];
    let scales = vec![127u8; 1];
    let x_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &x),
        "upload batched FP4 input",
    );
    let packed_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &packed),
        "upload batched FP4 weight",
    );
    let scales_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &scales),
        "upload batched FP4 scale",
    );
    let mut y_dev = assert_cuda(
        DeviceBuffer::<f32>::zeroed(&stream, 2),
        "alloc batched FP4 output",
    );
    assert_cuda(
        unsafe {
            module.gemm_fp4_e2m1_e8m0(
                &stream,
                LaunchConfig::for_num_elems(2),
                &x_dev,
                &packed_dev,
                &scales_dev,
                &mut y_dev,
                2,
                1,
                32,
            )
        },
        "batched FP4 GEMM launch",
    );
    let y = assert_cuda(y_dev.to_host_vec(&stream), "download batched FP4 output");
    assert_close_slice(&y, &[48.0, 48.0], 1e-4, "batched FP4 GEMM");

    let (hidden_size, q_rank, q_dim) = (4usize, 2usize, 3usize);
    let hidden = vec![1.0f32, 2.0, 3.0, 4.0];
    let query_a = vec![
        1.0, 0.0, 0.0, 0.0, // latent 0 = hidden[0]
        0.0, 1.0, 0.0, 0.0, // latent 1 = hidden[1]
    ];
    let query_b = vec![
        1.0, 0.0, // out 0 = normed[0]
        0.0, 1.0, // out 1 = normed[1]
        1.0, 1.0, // out 2 = normed[0] + normed[1]
    ];
    let query_norm = vec![1.0f32, 0.5];
    let hidden_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &hidden),
        "upload MLA hidden",
    );
    let query_a_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &query_a),
        "upload MLA query_a",
    );
    let query_b_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &query_b),
        "upload MLA query_b",
    );
    let query_norm_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &query_norm),
        "upload MLA query_norm",
    );
    let mut q_dev = assert_cuda(
        DeviceBuffer::<f32>::zeroed(&stream, q_dim),
        "alloc MLA output",
    );
    assert_cuda(
        unsafe {
            module.mla_q_projection_f32(
                &stream,
                LaunchConfig::for_num_elems(q_dim as u32),
                &hidden_dev,
                &query_a_dev,
                &query_b_dev,
                &query_norm_dev,
                &mut q_dev,
                hidden_size as u32,
                q_rank as u32,
                q_dim as u32,
                1e-6,
            )
        },
        "MLA Q projection launch",
    );
    let q = assert_cuda(q_dev.to_host_vec(&stream), "download MLA output");
    let rms = 1.0f32 / ((1.0f32 * 1.0 + 2.0 * 2.0) / 2.0 + 1e-6).sqrt();
    let norm0 = 1.0 * rms * 1.0;
    let norm1 = 2.0 * rms * 0.5;
    assert_close_slice(&q, &[norm0, norm1, norm0 + norm1], 1e-3, "MLA Q projection");

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

    let (head_dim, num_heads, kv_len, topk) = (8usize, 2usize, 4usize, 2usize);
    let q = vec![0.1f32; num_heads * head_dim];
    let kv = vec![0.5f32; kv_len * head_dim];
    let topk_idx: Vec<i32> = (0..topk).map(|i| i as i32).collect();
    let sink = vec![0.0f32; num_heads];
    let q_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &q),
        "upload sparse attention q",
    );
    let kv_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &kv),
        "upload sparse attention kv",
    );
    let topk_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &topk_idx),
        "upload sparse attention topk",
    );
    let sink_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &sink),
        "upload sparse attention sink",
    );
    let mut out_dev = assert_cuda(
        DeviceBuffer::<f32>::zeroed(&stream, num_heads * head_dim),
        "alloc sparse attention output",
    );
    assert_cuda(
        unsafe {
            module.sparse_attn_tiled_sink_f32(
                &stream,
                LaunchConfig::for_num_elems(num_heads as u32),
                &q_dev,
                &kv_dev,
                &topk_dev,
                &sink_dev,
                &mut out_dev,
                num_heads as u32,
                1,
                kv_len as u32,
                num_heads as u32,
                head_dim as u32,
                topk as u32,
                0.5,
            )
        },
        "sparse attention launch",
    );
    let out = assert_cuda(
        out_dev.to_host_vec(&stream),
        "download sparse attention output",
    );
    let expected_attention_value = 1.0 / (2.0 + (-0.2f32).exp());
    assert!(
        out.iter().all(|v| v.is_finite()),
        "sparse attention produced non-finite output: {out:?}"
    );
    assert_close(
        out[0],
        expected_attention_value,
        2e-3,
        "sparse attention first component",
    );

    let wrapped_out = assert_cuda(
        cuda_sparse_attention_sink_f32(
            &q, &kv, &topk_idx, &sink, 1, kv_len, num_heads, head_dim, topk, 0.5,
        ),
        "standalone sparse attention wrapper",
    );
    assert_close_slice(
        &wrapped_out,
        &out,
        1e-5,
        "standalone sparse attention wrapper output",
    );

    let (intermediate, hidden) = (32usize, 8usize);
    let gate = vec![0.5f32; intermediate];
    let up = vec![0.3f32; intermediate];
    let down_packed = vec![0x42u8; hidden * intermediate / 2];
    let down_scales = vec![127u8; hidden * intermediate / 32];
    let gate_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &gate),
        "upload SwiGLU gate",
    );
    let up_dev = assert_cuda(DeviceBuffer::from_host(&stream, &up), "upload SwiGLU up");
    let down_packed_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &down_packed),
        "upload SwiGLU down weight",
    );
    let down_scales_dev = assert_cuda(
        DeviceBuffer::from_host(&stream, &down_scales),
        "upload SwiGLU down scale",
    );
    let mut swiglu_out_dev = assert_cuda(
        DeviceBuffer::<f32>::zeroed(&stream, hidden),
        "alloc SwiGLU output",
    );
    assert_cuda(
        unsafe {
            module.swiglu_down_accumulate(
                &stream,
                LaunchConfig::for_num_elems(hidden as u32),
                &gate_dev,
                &up_dev,
                &down_packed_dev,
                &down_scales_dev,
                &mut swiglu_out_dev,
                intermediate as u32,
                hidden as u32,
                1.0,
                10.0,
            )
        },
        "SwiGLU accumulate launch",
    );
    let swiglu_out = assert_cuda(
        swiglu_out_dev.to_host_vec(&stream),
        "download SwiGLU output",
    );
    let silu_gate = 0.5f32 / (1.0 + (-0.5f32).exp());
    let expected_swiglu = 16.0 * silu_gate * 0.3 * (1.0 + 2.0);
    assert_close(
        swiglu_out[0],
        expected_swiglu,
        5e-2,
        "SwiGLU first component",
    );
}
