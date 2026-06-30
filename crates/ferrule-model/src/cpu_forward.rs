//! CPU FP32 reference forward pass (single-token decode).
use ferrule_core::Result;

use crate::OlmoeModel;

impl OlmoeModel {
    /// CPU forward with real attention + RoPE + KV cache (single token).
    /// Supports GQA: k/v cache entries are kv_dim per position.
    pub fn forward(
        &self,
        token_ids: &[u32],
        k_cache: &mut [Vec<f32>],
        v_cache: &mut [Vec<f32>],
        pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.config.hidden_size;
        let nh = self.config.num_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let kv_dim = self.config.kv_dim;
        let mut hidden = vec![0f32; d];
        // Embedding lookup (last token only — prefill handled by caller)
        if let Some(&tid) = token_ids.last() {
            let s = tid as usize * d;
            if s + d <= self.embed.len() {
                hidden.copy_from_slice(&self.embed[s..s + d]);
            }
        }
        // Precompute RoPE cos/sin for this position
        let mut rope_cos = vec![1.0f32; hd / 2];
        let mut rope_sin = vec![0.0f32; hd / 2];
        let theta = self.config.rope_theta;
        for i in 0..hd / 2 {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / hd as f32);
            let angle = pos as f32 * freq;
            rope_cos[i] = angle.cos();
            rope_sin[i] = angle.sin();
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // Attention norm
            let mut normed = hidden.clone();
            rms_norm(&mut normed, &layer.attn_norm, self.config.rms_norm_eps);
            // Q/K/V projections — k,v use kv_dim outputs (GQA)
            let mut q = vec![0f32; d];
            let mut k = vec![0f32; kv_dim];
            let mut v = vec![0f32; kv_dim];
            layer.attn.q_proj.forward(&normed, &mut q);
            layer.attn.k_proj.forward(&normed, &mut k);
            layer.attn.v_proj.forward(&normed, &mut v);
            // Q/K head norms
            rms_norm(&mut q, &layer.attn.q_norm, self.config.rms_norm_eps);
            rms_norm(&mut k, &layer.attn.k_norm, self.config.rms_norm_eps);
            // RoPE on Q and K (k has nkv heads, q has nh heads)
            rope_cpu(&mut q, &rope_cos, &rope_sin);
            rope_cpu(&mut k, &rope_cos, &rope_sin);
            // Append K,V to cache (kv_dim per position)
            k_cache[li].extend_from_slice(&k);
            v_cache[li].extend_from_slice(&v);
            let seq_len = k_cache[li].len() / kv_dim;
            // Attention with GQA support
            let attn_out = cpu_attention_gqa(&q, &k_cache[li], &v_cache[li], seq_len, nh, nkv, hd);
            // O projection
            let mut ao = vec![0f32; d];
            layer.attn.o_proj.forward(&attn_out, &mut ao);
            for j in 0..d {
                hidden[j] += ao[j];
            }

            // FFN
            let mut ffn_in = hidden.clone();
            rms_norm(&mut ffn_in, &layer.ffn_norm, self.config.rms_norm_eps);
            let mut rl = vec![0f32; self.config.num_experts];
            layer.router.forward(&ffn_in, &mut rl);
            let mut idx: Vec<(usize, f32)> = rl.iter().copied().enumerate().collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(self.config.num_experts_per_tok);
            let max_l = rl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let all_sum: f32 = rl.iter().map(|v| (*v - max_l).exp()).sum();
            let exps: Vec<f32> = idx.iter().map(|(_, v)| (v - max_l).exp()).collect();
            let topk_sum: f32 = exps.iter().sum();
            let sum = if self.config.norm_topk_prob {
                topk_sum
            } else {
                all_sum
            };
            let mut fo = vec![0f32; d];
            let mid = self.config.intermediate_size;
            for (k, &(eid, _)) in idx.iter().enumerate() {
                let w = exps[k] / sum;
                let ex = &layer.experts[eid];
                let (mut g, mut u) = (vec![0f32; mid], vec![0f32; mid]);
                ex.gate.forward(&ffn_in, &mut g);
                ex.up.forward(&ffn_in, &mut u);
                for j in 0..mid {
                    g[j] = silu(g[j]) * u[j];
                }
                let mut dn = vec![0f32; d];
                ex.down.forward(&g, &mut dn);
                for j in 0..d {
                    fo[j] += w * dn[j];
                }
            }
            for j in 0..d {
                hidden[j] += fo[j];
            }
        }

        // Final layer norm
        rms_norm(&mut hidden, &self.final_norm, self.config.rms_norm_eps);
        let mut logits = vec![0f32; self.config.vocab_size];
        for j in 0..self.config.vocab_size {
            let row = &self.lm_head[j * d..(j + 1) * d];
            logits[j] = row.iter().zip(hidden.iter()).map(|(r, h)| r * h).sum();
        }
        // Debug: print top-8 token ids
        if std::env::var_os("FERRULE_DEBUG_TOPK").is_some() {
            let mut top5: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            top5.truncate(8);
            tracing::info!(
                "CPU top8 ids: {:?}",
                top5.iter().map(|(i, _)| i).collect::<Vec<_>>()
            );
        }

        Ok((hidden, logits))
    }
}

fn rope_cpu(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let d = x.len();
    let hd2 = cos.len();
    for h in 0..d / (hd2 * 2) {
        let off = h * hd2 * 2;
        for i in 0..hd2 {
            let c = cos[i];
            let s = sin[i];
            let x0 = x[off + i];
            let x1 = x[off + hd2 + i];
            x[off + i] = x0 * c - x1 * s;
            x[off + hd2 + i] = x0 * s + x1 * c;
        }
    }
}

/// GQA attention: q has nh heads, k/v have nkv heads.
/// For each q head h, use kv head h * nkv / nh.
fn cpu_attention_gqa(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) -> Vec<f32> {
    let d_out = nh * hd;
    let kv_dim = nkv * hd;
    let sm_scale = 1.0 / (hd as f32).sqrt();
    let n_rep = nh / nkv; // how many q heads share one kv head
    let mut out = vec![0f32; d_out];
    for h in 0..nh {
        let kv_h = h / n_rep; // which kv head this q head maps to
        let mut scores = vec![0f32; seq_len];
        let mut max_s = f32::NEG_INFINITY;
        for p in 0..seq_len {
            let mut dot = 0.0;
            for j in 0..hd {
                dot += q[h * hd + j] * k_cache[p * kv_dim + kv_h * hd + j];
            }
            scores[p] = dot * sm_scale;
            if scores[p] > max_s {
                max_s = scores[p];
            }
        }
        let mut sum = 0.0;
        for p in 0..seq_len {
            scores[p] = (scores[p] - max_s).exp();
            sum += scores[p];
        }
        for j in 0..hd {
            let mut val = 0.0;
            for p in 0..seq_len {
                val += scores[p] * v_cache[p * kv_dim + kv_h * hd + j];
            }
            out[h * hd + j] = val / sum;
        }
    }
    out
}

pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let r = 1.0 / (ms + eps).sqrt();
    for (xi, wi) in x.iter_mut().zip(weight) {
        *xi *= r * *wi;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OlmoeConfig;
    use crate::weights::{AttnWeights, ExpertWeights, LayerWeights, LinearWeight};
    use std::path::PathBuf;

    /// Build a minimal 1-layer model for smoke-testing forward pass.
    fn minimal_model() -> OlmoeModel {
        let d = 4usize; // hidden_size
        let nh = 2; // num_heads
        let nkv = 1; // num_kv_heads
        let hd = d / nh; // head_dim = 2
        let kv_dim = nkv * hd; // kv_dim = 2
        let mid = 6; // intermediate_size
        let ne = 2; // num_experts
        let ntok = 2; // num_experts_per_tok (use both experts)
        let vocab = 8;

        let config = OlmoeConfig {
            hidden_size: d,
            num_layers: 1,
            num_experts: ne,
            num_experts_per_tok: ntok,
            intermediate_size: mid,
            vocab_size: vocab,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            kv_dim,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            norm_topk_prob: false,
            eos_token_id: Some(0),
            pad_token_id: None,
        };

        // Embedding: vocab × d
        let embed = vec![0.1_f32; vocab * d];

        // Layer 0 weights: fill with small random-like values
        let attn = AttnWeights {
            q_proj: LinearWeight {
                w: vec![0.01_f32; d * d],
                out_f: d,
                in_f: d,
            },
            k_proj: LinearWeight {
                w: vec![0.01_f32; kv_dim * d],
                out_f: kv_dim,
                in_f: d,
            },
            v_proj: LinearWeight {
                w: vec![0.01_f32; kv_dim * d],
                out_f: kv_dim,
                in_f: d,
            },
            o_proj: LinearWeight {
                w: vec![0.01_f32; d * d],
                out_f: d,
                in_f: d,
            },
            q_norm: vec![1.0_f32; d],
            k_norm: vec![1.0_f32; kv_dim],
        };

        let router = LinearWeight {
            w: vec![0.01_f32; ne * d],
            out_f: ne,
            in_f: d,
        };

        let experts: Vec<ExpertWeights> = (0..ne)
            .map(|_| ExpertWeights {
                gate: LinearWeight {
                    w: vec![0.01_f32; mid * d],
                    out_f: mid,
                    in_f: d,
                },
                up: LinearWeight {
                    w: vec![0.01_f32; mid * d],
                    out_f: mid,
                    in_f: d,
                },
                down: LinearWeight {
                    w: vec![0.01_f32; d * mid],
                    out_f: d,
                    in_f: mid,
                },
            })
            .collect();

        let layer = LayerWeights {
            attn_norm: vec![1.0_f32; d],
            attn,
            ffn_norm: vec![1.0_f32; d],
            router,
            experts,
        };

        // Create a minimal tokenizer (needed but not used by forward)
        let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());

        OlmoeModel {
            config,
            embed,
            lm_head: vec![0.01_f32; vocab * d],
            final_norm: vec![1.0_f32; d],
            layers: vec![layer],
            model_dir: PathBuf::from("/tmp"),
            tokenizer,
        }
    }

    #[test]
    fn test_forward_output_dimensions() {
        let model = minimal_model();
        let d = model.config.hidden_size;
        let vocab = model.config.vocab_size;
        let num_layers = model.config.num_layers;

        let mut k_cache: Vec<Vec<f32>> = (0..num_layers).map(|_| Vec::new()).collect();
        let mut v_cache: Vec<Vec<f32>> = (0..num_layers).map(|_| Vec::new()).collect();

        let (hidden, logits) = model
            .forward(&[0], &mut k_cache, &mut v_cache, 0)
            .expect("forward should succeed");

        assert_eq!(hidden.len(), d, "hidden state dimension");
        assert_eq!(logits.len(), vocab, "logits dimension");
    }

    #[test]
    fn test_forward_multiple_tokens() {
        let model = minimal_model();
        let num_layers = model.config.num_layers;
        let vocab = model.config.vocab_size;

        let mut k_cache: Vec<Vec<f32>> = (0..num_layers).map(|_| Vec::new()).collect();
        let mut v_cache: Vec<Vec<f32>> = (0..num_layers).map(|_| Vec::new()).collect();

        // Forward two tokens sequentially (simulating decode)
        let (_h0, _logits0) = model
            .forward(&[0], &mut k_cache, &mut v_cache, 0)
            .expect("token 0");
        let (_h1, logits1) = model
            .forward(&[1], &mut k_cache, &mut v_cache, 1)
            .expect("token 1");

        assert_eq!(logits1.len(), vocab);
    }

    #[test]
    fn test_rms_norm_unit_weight() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = vec![1.0_f32; 4];
        rms_norm(&mut x, &w, 1e-6);
        // After RMS norm with unit weights: x_i * (1 / sqrt(mean(x^2) + eps))
        let ms = (1.0_f32 + 4.0 + 9.0 + 16.0) / 4.0; // 7.5
        let r = 1.0 / (ms + 1e-6).sqrt();
        for (i, &val) in x.iter().enumerate() {
            let expected = (i + 1) as f32 * r;
            assert!(
                (val - expected).abs() < 1e-6,
                "x[{}]: {} != {}",
                i,
                val,
                expected
            );
        }
    }
}
