//! Numerical regression for the DSV4 head-dim-512 warp sparse-attention path.

use cuda_core::CudaContext;
use ferrule_cuda::context::cuda_sparse_attention_sink_f32;

fn has_cuda() -> bool {
    CudaContext::new(0).is_ok()
}

#[derive(Clone, Copy)]
struct SparseAttentionShape {
    tokens: usize,
    kv_len: usize,
    heads: usize,
    head_dim: usize,
    topk_len: usize,
    softmax_scale: f32,
}

fn sparse_attention_reference(
    query: &[f32],
    values: &[f32],
    topk: &[i32],
    sink: &[f32],
    shape: SparseAttentionShape,
) -> Vec<f32> {
    let SparseAttentionShape {
        tokens,
        kv_len,
        heads,
        head_dim,
        topk_len,
        softmax_scale,
    } = shape;
    let mut output = vec![0.0f32; tokens * heads * head_dim];
    for token in 0..tokens {
        for (head, &sink_score) in sink.iter().take(heads).enumerate() {
            let query_offset = (token * heads + head) * head_dim;
            let mut max_score = sink_score;
            for slot in 0..topk_len {
                let kv_index = topk[token * topk_len + slot];
                if kv_index < 0 || kv_index as usize >= kv_len {
                    continue;
                }
                let kv_offset = kv_index as usize * head_dim;
                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    dot += query[query_offset + dim] * values[kv_offset + dim];
                }
                max_score = max_score.max(dot * softmax_scale);
            }

            let mut denominator = libm::expf(sink_score - max_score);
            let output_offset = query_offset;
            for slot in 0..topk_len {
                let kv_index = topk[token * topk_len + slot];
                if kv_index < 0 || kv_index as usize >= kv_len {
                    continue;
                }
                let kv_offset = kv_index as usize * head_dim;
                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    dot += query[query_offset + dim] * values[kv_offset + dim];
                }
                let weight = libm::expf(dot * softmax_scale - max_score);
                denominator += weight;
                for dim in 0..head_dim {
                    output[output_offset + dim] += weight * values[kv_offset + dim];
                }
            }
            for dim in 0..head_dim {
                output[output_offset + dim] /= denominator;
            }
        }
    }
    output
}

#[test]
fn dsv4_warp_sparse_attention_matches_scalar_reference() {
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    const TOKENS: usize = 3;
    const KV_LEN: usize = 7;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 512;
    const TOPK: usize = 4;
    let softmax_scale = (HEAD_DIM as f32).sqrt().recip();

    let mut query = vec![0.0f32; TOKENS * HEADS * HEAD_DIM];
    for (index, value) in query.iter_mut().enumerate() {
        let centered = (index % 37) as f32 - 18.0;
        *value = centered * 0.003_906_25;
    }
    let mut values = vec![0.0f32; KV_LEN * HEAD_DIM];
    for (index, value) in values.iter_mut().enumerate() {
        let centered = (index % 43) as f32 - 21.0;
        *value = centered * 0.007_812_5;
    }
    let topk = vec![0, 2, 5, 6, 1, 3, 4, -1, 6, 5, 2, 0];
    let sink = vec![-0.25f32, 0.125];
    let expected = sparse_attention_reference(
        &query,
        &values,
        &topk,
        &sink,
        SparseAttentionShape {
            tokens: TOKENS,
            kv_len: KV_LEN,
            heads: HEADS,
            head_dim: HEAD_DIM,
            topk_len: TOPK,
            softmax_scale,
        },
    );
    let actual = cuda_sparse_attention_sink_f32(
        &query,
        &values,
        &topk,
        &sink,
        TOKENS,
        KV_LEN,
        HEADS,
        HEAD_DIM,
        TOPK,
        softmax_scale,
    )
    .expect("warp sparse attention");

    let mut max_abs = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let error = (actual - expected).abs();
        max_abs = max_abs.max(error);
        let tolerance = 2e-5f32.max(expected.abs() * 2e-4);
        assert!(
            error <= tolerance,
            "sparse attention mismatch at {index}: actual={actual} expected={expected} error={error} tolerance={tolerance}"
        );
    }
    println!("DSV4 warp sparse attention OK: max_abs={max_abs}");
}
