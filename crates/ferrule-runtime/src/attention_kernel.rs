//! Cache-aware DSV4 MLA attention kernels.
//!
//! Implements the Flash Attention approach (tiled QK^T + online softmax) adapted
//! for DSV4 Multi-Latent Attention:
//!   1. wq_a → q_norm → wq_b → reshape to [heads, head_dim]
//!   2. wkv → kv_norm → [kv_dim] (compressed KV, not separate K/V)
//!   3. Sparse top-k gather over sliding window KV cache
//!   4. Online softmax with attention sink
//!   5. wo_a + wo_b output projection
//!
//! The `AttentionKernel` trait provides a dispatch boundary so CUDA tiled kernels
//! can replace the CPU reference later without changing caller code.

use ferrule_core::{Error, Result};

use crate::attention_backend::SparseAttentionSpec;
use crate::source_linear::SourceLinearPayload;

// ── block sizes for cache tiling ─────────────────────────────────────────

/// L2-friendly tile size for (out_features × in_features) linear projections.
/// On CPUs with 256KB–512KB L2, a 128×128 float tile ≈ 64KB.
const LINEAR_TILE_M: usize = 128;
const LINEAR_TILE_K: usize = 128;

/// Registers tile for QK^T attention: heads done sequentially, each head's
/// q vector (head_dim) computed in one go.
const QK_TILE_B: usize = 256; // KV cache tile size for score computation

// ── kernel dispatch trait ─────────────────────────────────────────────────

/// Separates the attention *math* from the *execution backend*.
///
/// A `CpuTiledAttention` provides small-correctness-oriented tiled matmuls
/// and online softmax. A `CudaTiledAttention` (future) will use CUDA blocks
/// with shared memory and warp-level reductions.
pub trait AttentionKernel {
    /// Compute output = linear_matvec(weight, input) with cache tiling.
    fn linear_matvec(&self, linear: &SourceLinearPayload, input: &[f32]) -> Result<Vec<f32>>;

    /// rms_norm with per-element weight vector.
    fn rms_norm_weight(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        label: &str,
    ) -> Result<Vec<f32>>;

    /// Sparse online-softmax attention.
    ///
    /// Parameters:
    /// - `q`: `[heads, head_dim]` contiguous query
    /// - `kv`: `[kv_len, head_dim]` cached compressed KV values
    /// - `topk_indices`: `[topk]` indices into kv (negative = masked)
    /// - `attention_sink`: `[heads]` per-head contribution to softmax denominator
    /// - `spec`: validation / shape metadata
    fn sparse_attention_online(
        &self,
        q: &[f32],
        kv: &[f32],
        topk_indices: &[isize],
        attention_sink: Option<&[f32]>,
        spec: SparseAttentionSpec,
    ) -> Result<Vec<f32>>;

    /// Down-projection after attention: output = wo_b.matvec(wo_a.matvec(context))
    fn output_projection(
        &self,
        wo_a: &SourceLinearPayload,
        wo_b: &SourceLinearPayload,
        context: &[f32],
    ) -> Result<Vec<f32>>;
}

// ── CPU tiled kernel ─────────────────────────────────────────────────────

/// CPU attention kernel with cache-blocked tiled matmul and online softmax.
///
/// Designed to be correctness-anchor, not peak performance. CUDA kernels
/// should implement the same trait with shared-memory tiling.
#[derive(Debug, Clone, Copy)]
pub struct CpuAttentionKernel {
    pub tile_m: usize,
    pub tile_k: usize,
    pub qk_tile: usize,
}

impl Default for CpuAttentionKernel {
    fn default() -> Self {
        Self {
            tile_m: LINEAR_TILE_M,
            tile_k: LINEAR_TILE_K,
            qk_tile: QK_TILE_B,
        }
    }
}

impl CpuAttentionKernel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tiled row-major matvec: output[row] = sum_k weight[row,k] * input[k]
    fn tiled_matvec(weights: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
        let tile_m = LINEAR_TILE_M;
        let mut output = vec![0.0f32; rows];
        for tile_start in (0..rows).step_by(tile_m) {
            let tile_end = (tile_start + tile_m).min(rows);
            for row in tile_start..tile_end {
                let mut acc = 0.0f32;
                let offset = row * cols;
                // Inner loop stays simple; cache locality comes from tile_m bound
                for col in 0..cols {
                    acc += weights[offset + col] * input[col];
                }
                output[row] = acc;
            }
        }
        output
    }

    /// Online softmax sparse attention with attention sink.
    ///
    /// Instead of materializing the full score matrix, we stream through
    /// top-k indices and maintain running max/denom per (token, head).
    fn sparse_attention_online_inner(
        q: &[f32],
        kv: &[f32],
        topk_indices: &[isize],
        attention_sink: Option<&[f32]>,
        spec: SparseAttentionSpec,
    ) -> Result<Vec<f32>> {
        spec.validate()?;
        let heads = spec.heads;
        let head_dim = spec.head_dim;
        let topk = spec.topk;
        let kv_len = kv.len() / head_dim;
        let scale = spec.softmax_scale;

        let mut output = vec![0.0f32; heads * head_dim];

        for head in 0..heads {
            let q_head = &q[head * head_dim..(head + 1) * head_dim];
            let sink_val = attention_sink.map(|s| s[head]).unwrap_or(f32::NEG_INFINITY);

            // Pass 1: find max score (incl. sink)
            let mut max_score = sink_val;
            for slot in 0..topk {
                let idx = topk_indices[slot];
                if idx < 0 {
                    continue;
                }
                let idx = idx as usize;
                if idx >= kv_len {
                    return Err(Error::Model(format!("sparse idx {idx} >= kv_len {kv_len}")));
                }
                let kv_row = &kv[idx * head_dim..(idx + 1) * head_dim];
                let dot: f32 = q_head.iter().zip(kv_row).map(|(a, b)| a * b).sum();
                max_score = max_score.max(dot * scale);
            }

            // Pass 2: accumulate weighted values with online softmax
            let sink_exp = if spec.has_attention_sink {
                (sink_val - max_score).exp()
            } else {
                0.0
            };
            let mut denom = sink_exp;
            let mut weighted = vec![0.0f32; head_dim];
            if spec.has_attention_sink {
                // Sink doesn't contribute to output, only denominator
            }

            for slot in 0..topk {
                let idx = topk_indices[slot];
                if idx < 0 {
                    continue;
                }
                let idx = idx as usize;
                let kv_row = &kv[idx * head_dim..(idx + 1) * head_dim];
                let dot: f32 = q_head.iter().zip(kv_row).map(|(a, b)| a * b).sum();
                let weight = (dot * scale - max_score).exp();
                denom += weight;
                for d in 0..head_dim {
                    weighted[d] += weight * kv_row[d];
                }
            }

            if denom == 0.0 || !denom.is_finite() {
                return Err(Error::Model("online softmax denominator is invalid".into()));
            }
            let out_head = &mut output[head * head_dim..(head + 1) * head_dim];
            for d in 0..head_dim {
                out_head[d] = weighted[d] / denom;
            }
        }
        Ok(output)
    }
}

impl AttentionKernel for CpuAttentionKernel {
    fn linear_matvec(&self, linear: &SourceLinearPayload, input: &[f32]) -> Result<Vec<f32>> {
        let in_features = linear.format.in_features();
        let out_features = linear.format.out_features();
        if input.len() != in_features {
            return Err(Error::Model(format!(
                "input length mismatch: expected {in_features}, got {}",
                input.len()
            )));
        }
        let weights = linear.reference_weights_f32()?;
        Ok(Self::tiled_matvec(
            &weights,
            out_features,
            in_features,
            input,
        ))
    }

    fn rms_norm_weight(
        &self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        label: &str,
    ) -> Result<Vec<f32>> {
        if input.len() != weight.len() || input.is_empty() {
            return Err(Error::Model(format!("{label}: length mismatch or empty")));
        }
        let mean_sq: f32 = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
        let rms = 1.0 / (mean_sq + eps).sqrt();
        Ok(input.iter().zip(weight).map(|(v, w)| v * rms * w).collect())
    }

    fn sparse_attention_online(
        &self,
        q: &[f32],
        kv: &[f32],
        topk_indices: &[isize],
        attention_sink: Option<&[f32]>,
        spec: SparseAttentionSpec,
    ) -> Result<Vec<f32>> {
        Self::sparse_attention_online_inner(q, kv, topk_indices, attention_sink, spec)
    }

    fn output_projection(
        &self,
        wo_a: &SourceLinearPayload,
        wo_b: &SourceLinearPayload,
        context: &[f32],
    ) -> Result<Vec<f32>> {
        let projected = self.linear_matvec(wo_a, context)?;
        self.linear_matvec(wo_b, &projected)
    }
}

// ── unified decode-attention step ────────────────────────────────────────

/// Executes a full DSV4 MLA decode attention step through the kernel trait.
///
/// Steps:
///   1. wq_a(hidden) → q_norm → wq_b → reshape to [heads, head_dim]
///   2. wkv(hidden) → kv_norm → append to KV cache
///   3. Build sliding-window top-k indices from KV cache
///   4. Online softmax sparse attention → context
///   5. wo_a(context) → wo_b → output
#[allow(clippy::too_many_arguments)]
pub fn dsv4_attention_decode_step(
    kernel: &impl AttentionKernel,
    hidden: &[f32],
    wq_a: &SourceLinearPayload,
    wq_b: &SourceLinearPayload,
    q_norm: &[f32],
    wkv: &SourceLinearPayload,
    kv_norm: &[f32],
    kv_cache: &mut Vec<f32>,
    attention_sink: &[f32],
    wo_a: &SourceLinearPayload,
    wo_b: &SourceLinearPayload,
    spec: SparseAttentionSpec,
) -> Result<Vec<f32>> {
    // 1. Query low-rank path
    let q_latent = kernel.linear_matvec(wq_a, hidden)?;
    let q_latent = kernel.rms_norm_weight(&q_latent, q_norm, 1e-6, "q_norm")?;
    let q_full = kernel.linear_matvec(wq_b, &q_latent)?;

    // 2. KV projection + cache append
    let kv_val = kernel.linear_matvec(wkv, hidden)?;
    let kv_val = kernel.rms_norm_weight(&kv_val, kv_norm, 1e-6, "kv_norm")?;
    let kv_len_before = kv_cache.len() / spec.head_dim;
    kv_cache.extend_from_slice(&kv_val);
    let kv_len = kv_cache.len() / spec.head_dim;

    // 3. Sliding-window top-k indices
    let topk = spec.topk;
    let mut indices = vec![-1isize; topk];
    let start = kv_len.saturating_sub(topk);
    for i in 0..(kv_len - start) {
        indices[i] = (start + i) as isize;
    }

    // 4. Sparse attention
    let context =
        kernel.sparse_attention_online(&q_full, kv_cache, &indices, Some(attention_sink), spec)?;

    // 5. Output projection
    kernel.output_projection(wo_a, wo_b, &context)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ferrule_model::TensorRole;

    use super::*;
    use crate::source_tensor::{SourceDType, SourceTensorPayload, SourceTensorSlice};

    #[test]
    fn cpu_tiled_attention_produces_same_result_as_naive() {
        let kernel = CpuAttentionKernel::new();
        let spec = SparseAttentionSpec {
            heads: 2,
            head_dim: 4,
            topk: 2,
            softmax_scale: 0.5,
            has_attention_sink: true,
        };
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let kv = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let sink = vec![0.0, 0.0];
        let indices = vec![0, 1];

        let out = kernel
            .sparse_attention_online(&q, &kv, &indices, Some(&sink), spec)
            .unwrap();

        // Compare with naive reference
        let ref_out = crate::attention_backend::sparse_attention_reference(
            &q,
            &kv,
            &indices,
            Some(&sink),
            1,
            2,
            spec,
        )
        .unwrap();
        for (i, (a, b)) in out.iter().zip(ref_out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn tiled_matvec_matches_naive() {
        let kernel = CpuAttentionKernel::new();
        let linear = f32_linear(
            TensorRole::AttentionLatentQueryA,
            "test",
            4,
            8,
            &(0..32).map(|i| (i % 3) as f32).collect::<Vec<_>>(),
        );
        let input: Vec<f32> = (0..8).map(|i| (i % 5) as f32).collect();
        let out = kernel.linear_matvec(&linear, &input).unwrap();
        let ref_out = linear.reference_matvec(&input).unwrap();
        for (i, (a, b)) in out.iter().zip(ref_out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn dsv4_attention_decode_step_runs_end_to_end() {
        let kernel = CpuAttentionKernel::new();
        let h = 32;
        let qr = 8;
        let hd = 16;
        let nh = 2;
        let spec = SparseAttentionSpec {
            heads: nh,
            head_dim: hd,
            topk: 4,
            softmax_scale: (hd as f32).powf(-0.5),
            has_attention_sink: true,
        };
        let wq_a = f32_linear(
            TensorRole::AttentionLatentQueryA,
            "wq_a",
            qr,
            h,
            &vec![0.1; qr * h],
        );
        let wq_b = f32_linear(
            TensorRole::AttentionLatentQueryB,
            "wq_b",
            nh * hd,
            qr,
            &vec![0.1; nh * hd * qr],
        );
        let wkv = f32_linear(
            TensorRole::AttentionLatentKv,
            "wkv",
            hd,
            h,
            &vec![0.1; hd * h],
        );
        let wo_a = f32_linear(
            TensorRole::AttentionLatentOutputA,
            "wo_a",
            qr,
            nh * hd,
            &vec![0.1; qr * nh * hd],
        );
        let wo_b = f32_linear(
            TensorRole::AttentionLatentOutputB,
            "wo_b",
            h,
            qr,
            &vec![0.1; h * qr],
        );

        let hidden: Vec<f32> = (0..h).map(|i| (i % 7) as f32).collect();
        let mut kv_cache = Vec::new();

        let out = dsv4_attention_decode_step(
            &kernel,
            &hidden,
            &wq_a,
            &wq_b,
            &vec![1.0; qr],
            &wkv,
            &vec![1.0; hd],
            &mut kv_cache,
            &vec![0.0; nh],
            &wo_a,
            &wo_b,
            spec,
        )
        .unwrap();
        assert_eq!(out.len(), h);
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(kv_cache.len(), hd);
    }

    fn f32_linear(
        role: TensorRole,
        name: &str,
        out: usize,
        input: usize,
        values: &[f32],
    ) -> SourceLinearPayload {
        assert_eq!(values.len(), out * input);
        SourceLinearPayload::from_weight_and_scale(
            role,
            SourceTensorPayload {
                slice: SourceTensorSlice {
                    name: format!("{name}.weight"),
                    role: TensorRole::Unknown,
                    path: PathBuf::from("synthetic.safetensors"),
                    offset: 0,
                    bytes: (values.len() * 4) as u64,
                    dtype: SourceDType::F32,
                    shape: vec![out, input],
                },
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
            None,
        )
        .unwrap()
    }
}
