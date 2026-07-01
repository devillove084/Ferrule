//! Attention backend contracts and reference kernels.
//!
//! DeepSeek V4's attention is not a plain dense causal MHA kernel. The runtime
//! needs a generic attention surface that can route to: CPU/reference correctness,
//! sparse FlashAttention-style CUDA for MLA/compressed KV, or dense FlashAttention
//! for future Llama/Qwen-like models. This module defines that surface without
//! concrete model-family tensor names.

use ferrule_core::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackendKind {
    /// Scalar CPU/reference implementation for golden tests.
    Reference,
    /// Sparse top-k gather + online-softmax style CUDA backend.
    CudaSparseFlash,
    /// Dense causal/windowed CUDA backend for standard MHA/GQA families.
    CudaDenseFlash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMaskKind {
    ExplicitTopK,
    SlidingWindow { window_size: usize },
    Causal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseAttentionSpec {
    pub heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    pub softmax_scale: f32,
    pub has_attention_sink: bool,
}

impl SparseAttentionSpec {
    pub fn validate(&self) -> Result<()> {
        if self.heads == 0 || self.head_dim == 0 || self.topk == 0 {
            return Err(Error::Model(format!(
                "invalid sparse attention shape: heads={}, head_dim={}, topk={}",
                self.heads, self.head_dim, self.topk
            )));
        }
        if !self.softmax_scale.is_finite() || self.softmax_scale <= 0.0 {
            return Err(Error::Model(format!(
                "invalid sparse attention softmax scale {}",
                self.softmax_scale
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionBackendPlan {
    pub backend: AttentionBackendKind,
    pub mask: AttentionMaskKind,
    pub sparse: Option<SparseAttentionSpec>,
}

impl AttentionBackendPlan {
    pub fn dsv4_sparse_flash(heads: usize, head_dim: usize, topk: usize) -> Self {
        Self {
            backend: AttentionBackendKind::CudaSparseFlash,
            mask: AttentionMaskKind::ExplicitTopK,
            sparse: Some(SparseAttentionSpec {
                heads,
                head_dim,
                topk,
                softmax_scale: (head_dim as f32).powf(-0.5),
                has_attention_sink: true,
            }),
        }
    }

    pub fn reference_sparse(spec: SparseAttentionSpec) -> Self {
        Self {
            backend: AttentionBackendKind::Reference,
            mask: AttentionMaskKind::ExplicitTopK,
            sparse: Some(spec),
        }
    }
}

/// CPU/reference sparse attention with DSV4 sink semantics.
///
/// Shapes:
/// - `q`: `[tokens, heads, head_dim]`
/// - `kv`: `[kv_len, head_dim]`
/// - `topk_indices`: `[tokens, topk]`, negative values are masked
/// - `attention_sink`: optional `[heads]`; it contributes to the denominator only
/// - output: `[tokens, heads, head_dim]`
pub fn sparse_attention_reference(
    q: &[f32],
    kv: &[f32],
    topk_indices: &[isize],
    attention_sink: Option<&[f32]>,
    tokens: usize,
    kv_len: usize,
    spec: SparseAttentionSpec,
) -> Result<Vec<f32>> {
    spec.validate()?;
    let heads = spec.heads;
    let head_dim = spec.head_dim;
    let topk = spec.topk;
    if q.len() != tokens * heads * head_dim {
        return Err(Error::Model(format!(
            "sparse attention q length mismatch: expected {}, got {}",
            tokens * heads * head_dim,
            q.len()
        )));
    }
    if kv.len() != kv_len * head_dim {
        return Err(Error::Model(format!(
            "sparse attention kv length mismatch: expected {}, got {}",
            kv_len * head_dim,
            kv.len()
        )));
    }
    if topk_indices.len() != tokens * topk {
        return Err(Error::Model(format!(
            "sparse attention topk length mismatch: expected {}, got {}",
            tokens * topk,
            topk_indices.len()
        )));
    }
    if let Some(sink) = attention_sink {
        if sink.len() != heads {
            return Err(Error::Model(format!(
                "attention sink length mismatch: expected {heads}, got {}",
                sink.len()
            )));
        }
    } else if spec.has_attention_sink {
        return Err(Error::Model(
            "sparse attention spec requires attention sink".into(),
        ));
    }

    let mut out = vec![0.0f32; q.len()];
    let mut scores = vec![0.0f32; topk];
    for token in 0..tokens {
        for head in 0..heads {
            let q_offset = (token * heads + head) * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];
            let mut max_score = attention_sink
                .map(|sink| sink[head])
                .unwrap_or(f32::NEG_INFINITY);
            for slot in 0..topk {
                let idx = topk_indices[token * topk + slot];
                let score = if idx < 0 {
                    f32::NEG_INFINITY
                } else {
                    let idx = usize::try_from(idx).map_err(|_| {
                        Error::Model(format!("negative sparse attention index {idx}"))
                    })?;
                    if idx >= kv_len {
                        return Err(Error::Model(format!(
                            "sparse attention index {idx} exceeds kv_len {kv_len}"
                        )));
                    }
                    dot(q_vec, &kv[idx * head_dim..(idx + 1) * head_dim]) * spec.softmax_scale
                };
                scores[slot] = score;
                max_score = max_score.max(score);
            }

            let mut denom = attention_sink
                .map(|sink| (sink[head] - max_score).exp())
                .unwrap_or(0.0);
            for &score in &scores {
                if score.is_finite() {
                    denom += (score - max_score).exp();
                }
            }
            if denom == 0.0 || !denom.is_finite() {
                return Err(Error::Model(
                    "sparse attention denominator is invalid".into(),
                ));
            }

            for slot in 0..topk {
                let idx = topk_indices[token * topk + slot];
                if idx < 0 || !scores[slot].is_finite() {
                    continue;
                }
                let idx = idx as usize;
                let weight = (scores[slot] - max_score).exp() / denom;
                for d in 0..head_dim {
                    out[q_offset + d] += weight * kv[idx * head_dim + d];
                }
            }
        }
    }
    Ok(out)
}

/// Reference helper for DSV4 sliding-window top-k indices, single batch.
pub fn sliding_window_topk_indices(
    window_size: usize,
    seq_len: usize,
    start_pos: usize,
) -> Vec<isize> {
    if seq_len == 0 || window_size == 0 {
        return Vec::new();
    }
    let mut out = vec![-1isize; seq_len * window_size];
    if start_pos > 0 {
        let row = if start_pos >= window_size - 1 {
            let pivot = start_pos % window_size;
            (pivot + 1..window_size)
                .chain(0..=pivot)
                .map(|v| v as isize)
                .collect::<Vec<_>>()
        } else {
            (0..=start_pos)
                .map(|v| v as isize)
                .chain(std::iter::repeat(-1).take(window_size - start_pos - 1))
                .collect::<Vec<_>>()
        };
        for token in 0..seq_len {
            out[token * window_size..(token + 1) * window_size].copy_from_slice(&row);
        }
        return out;
    }

    for token in 0..seq_len {
        let start = token.saturating_sub(window_size - 1);
        for slot in 0..window_size.min(token + 1) {
            out[token * window_size + slot] = (start + slot) as isize;
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_attention_reference_matches_sink_denominator_semantics() {
        let spec = SparseAttentionSpec {
            heads: 1,
            head_dim: 2,
            topk: 2,
            softmax_scale: 1.0,
            has_attention_sink: true,
        };
        let q = vec![1.0, 0.0];
        let kv = vec![1.0, 2.0, 3.0, 5.0];
        let out = sparse_attention_reference(&q, &kv, &[0, 1], Some(&[0.0]), 1, 2, spec).unwrap();
        let e1 = 1.0f32.exp();
        let e3 = 3.0f32.exp();
        let denom = e1 + e3 + 1.0; // sink exp(0)
        let expected0 = (e1 * 1.0 + e3 * 3.0) / denom;
        let expected1 = (e1 * 2.0 + e3 * 5.0) / denom;
        assert!(
            (out[0] - expected0).abs() < 1e-6,
            "{} vs {expected0}",
            out[0]
        );
        assert!(
            (out[1] - expected1).abs() < 1e-6,
            "{} vs {expected1}",
            out[1]
        );
    }

    #[test]
    fn sparse_attention_reference_masks_negative_indices() {
        let spec = SparseAttentionSpec {
            heads: 1,
            head_dim: 1,
            topk: 2,
            softmax_scale: 1.0,
            has_attention_sink: false,
        };
        let out = sparse_attention_reference(&[1.0], &[7.0], &[0, -1], None, 1, 1, spec).unwrap();
        assert_eq!(out, vec![7.0]);
    }

    #[test]
    fn sliding_window_topk_prefill_is_causal_window() {
        assert_eq!(
            sliding_window_topk_indices(3, 4, 0),
            vec![0, -1, -1, 0, 1, -1, 0, 1, 2, 1, 2, 3]
        );
    }

    #[test]
    fn sliding_window_topk_decode_wraps_ring_buffer() {
        assert_eq!(sliding_window_topk_indices(4, 1, 5), vec![2, 3, 0, 1]);
    }
}
