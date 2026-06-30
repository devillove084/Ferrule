//! GPU OLMoE forward pass — quantized weights, scratch pool, zero CPU roundtrips.
#![allow(unsafe_code)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrule_core::{Error, Result};
use ferrule_quant::QuantType;
use std::sync::Arc;

use crate::context::{cu, gemv_quant, gemv_quant_off};

// Re-export public items that moved to sub-modules (preserve API paths).
pub use crate::context::{cuda_gemv, cuda_probe};
pub use crate::graph::{cuda_graph_enabled, flash_attn_enabled};

// ── Attention optimization plan (P10.4) ─────────────────────────────────
//
// Current status: standard MHA/GQA attention with per-token score computation
// and in-GPU softmax+combine. This is O(seq_len²) in compute and memory.
//
// Planned improvements:
//   1. FlashAttention-2 kernel: tile-based online softmax that avoids
//      materializing the full N×N attention matrix. Reduces HBM traffic
//      from O(N²) to O(N) and enables much longer contexts.
//   2. PagedAttention integration: the FA kernel reads the block table
//      directly, so physical K/V blocks can be non-contiguous.
//   3. Decode-phase optimization: for seq_len>1 and single-query decode,
//      use a dedicated FlashDecoding kernel (tile over KV, reduce).
//
// When FERRULE_FLASH_ATTN=1 is set, the attention path below should
// dispatch to a FlashAttention kernel instead of the current two-kernel
// (scores + softmax-combine) approach.

// ── Quantized weight storage ──────────────────────────────────────────

/// Per-layer weights stored in quantized format on GPU.
/// K/V projections use kv_dim outputs (GQA), Q uses d.
#[allow(dead_code)]
pub(crate) struct QLayer {
    // Attention norm weights (f32 — small, precision matters)
    pub(crate) an: DeviceBuffer<f32>,  // d
    pub(crate) qn: DeviceBuffer<f32>,  // d (Q norm)
    pub(crate) kn: DeviceBuffer<f32>,  // kv_dim (K norm)
    pub(crate) fn_: DeviceBuffer<f32>, // d

    // Quantized attention projection weights
    pub(crate) qp_packed: DeviceBuffer<u8>,
    pub(crate) qp_scales: DeviceBuffer<f32>,
    pub(crate) kp_packed: DeviceBuffer<u8>,
    pub(crate) kp_scales: DeviceBuffer<f32>,
    pub(crate) vp_packed: DeviceBuffer<u8>,
    pub(crate) vp_scales: DeviceBuffer<f32>,
    pub(crate) op_packed: DeviceBuffer<u8>,
    pub(crate) op_scales: DeviceBuffer<f32>,

    // Router (f32 — small)
    pub(crate) rt: DeviceBuffer<f32>, // ne × d

    // Expert weights, concatenated across all experts
    pub(crate) ex_gate_packed: DeviceBuffer<u8>,
    pub(crate) ex_gate_scales: DeviceBuffer<f32>,
    pub(crate) ex_up_packed: DeviceBuffer<u8>,
    pub(crate) ex_up_scales: DeviceBuffer<f32>,
    pub(crate) ex_down_packed: DeviceBuffer<u8>,
    pub(crate) ex_down_scales: DeviceBuffer<f32>,
}

pub struct GpuOlmoeModel {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) s: Arc<cuda_core::stream::CudaStream>,
    pub(crate) module: crate::kernels::kernels::LoadedModule,
    pub(crate) emb: DeviceBuffer<f32>,
    pub(crate) lm_head: DeviceBuffer<f32>,
    pub(crate) final_norm: DeviceBuffer<f32>,
    pub(crate) layers: Vec<QLayer>,
    pub(crate) d: usize,
    pub(crate) kv_dim: usize,
    pub(crate) ne: usize,
    pub(crate) na: usize,
    pub(crate) mid: usize,
    pub(crate) vocab: usize,
    pub(crate) eps: f32,
    pub(crate) norm_topk_prob: bool,
    pub(crate) qt: QuantType,
    pub(crate) scratch: Scratch,
    // ── KV cache (GPU) + RoPE (GPU) ──
    pub(crate) nh: usize,
    pub(crate) nkv: usize,
    pub(crate) hd: usize,
    #[allow(dead_code)]
    pub(crate) max_seq: usize,
    pub(crate) k_cache: Vec<DeviceBuffer<f32>>, // [num_layers][max_seq × kv_dim]
    pub(crate) v_cache: Vec<DeviceBuffer<f32>>,
    pub(crate) rope_cos: DeviceBuffer<f32>,
    pub(crate) rope_sin: DeviceBuffer<f32>,
    pub(crate) cur_seq: usize,
    pub(crate) scores_buf: DeviceBuffer<f32>, // [nh × max_seq]
    /// Per-layer per-expert activation count.
    pub expert_hits: Vec<Vec<usize>>,
    /// Total tokens processed.
    pub total_tokens: usize,
}

/// Pre-allocated GPU buffers — allocated once, reused every token.
pub(crate) struct Scratch {
    pub(crate) hidden: DeviceBuffer<f32>,     // d
    pub(crate) normed: DeviceBuffer<f32>,     // d
    pub(crate) q: DeviceBuffer<f32>,          // d
    pub(crate) k: DeviceBuffer<f32>,          // kv_dim
    pub(crate) v: DeviceBuffer<f32>,          // kv_dim
    pub(crate) q_tmp: DeviceBuffer<f32>,      // d
    pub(crate) k_tmp: DeviceBuffer<f32>,      // kv_dim
    pub(crate) ao: DeviceBuffer<f32>,         // d
    pub(crate) h_tmp1: DeviceBuffer<f32>,     // d
    pub(crate) h_tmp2: DeviceBuffer<f32>,     // d (zero-filled, used as zero operand)
    pub(crate) ffn_in: DeviceBuffer<f32>,     // d
    pub(crate) router_out: DeviceBuffer<f32>, // ne (also used as temp for GPU top-k)
    pub(crate) fo: DeviceBuffer<f32>,         // d
    pub(crate) gb: DeviceBuffer<f32>,         // mid
    pub(crate) ub: DeviceBuffer<f32>,         // mid
    pub(crate) gb2: DeviceBuffer<f32>,        // mid
    pub(crate) db: DeviceBuffer<f32>,         // d
    pub(crate) logits: DeviceBuffer<f32>,     // vocab
    pub(crate) topk_idx: DeviceBuffer<f32>,   // na (top-k expert indices as f32, GPU-side)
    pub(crate) topk_w: DeviceBuffer<f32>,     // na (top-k softmax weights)
    /// GPU-side vocab top-k buffers (avoid full logits download).
    pub(crate) topk_vocab_idx: DeviceBuffer<f32>, // K
    pub(crate) topk_vocab_val: DeviceBuffer<f32>, // K
}

impl GpuOlmoeModel {
    pub fn from_cpu(model: &ferrule_model::OlmoeModel, qt: QuantType) -> Result<Self> {
        Self::build_from_cpu(model, qt)
    }

    /// Build GPU model from a lightweight CPU model + pre-existing qcache.
    /// Skips full FP32 weight loading — norms, router, embed, lm_head come
    /// from the lightweight model; quantized attention/expert weights come
    /// from the cache.
    pub fn from_lightweight(
        model: &ferrule_model::OlmoeModel,
        cache: &crate::qcache::QCacheReader,
        qt: QuantType,
    ) -> Result<Self> {
        Self::build_from_lightweight(model, cache, qt)
    }

    pub fn forward(&mut self, tid: u32) -> Result<Vec<f32>> {
        let _span = tracing::debug_span!("gpu_forward", token = tid).entered();
        if self.cur_seq >= self.max_seq {
            return Err(Error::Internal(format!(
                "GPU context length {} exceeds max_seq {}",
                self.cur_seq + 1,
                self.max_seq
            )));
        }
        cu(self.ctx.bind_to_thread())?;
        let m = &self.module;
        let s = &self.s;
        let d = self.d;
        let kv_dim = self.kv_dim;
        let nkv = self.nkv;
        let cfg = |n: usize| LaunchConfig::for_num_elems(n as u32);
        // 1-block launch for fused reduce+apply kernels (max 1024 threads/block)
        let cfg1 = |n: usize| LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n.min(1024) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let Scratch {
            hidden,
            normed,
            q,
            k,
            v,
            q_tmp,
            k_tmp,
            ao,
            h_tmp1,
            h_tmp2,
            ffn_in,
            router_out,
            fo,
            gb,
            ub,
            gb2,
            db,
            logits,
            topk_idx,
            topk_w,
            topk_vocab_idx,
            topk_vocab_val,
        } = &mut self.scratch;

        // ── Embedding lookup ──
        {
            let _s = tracing::trace_span!("embed").entered();
            cu(m.embed_lookup(s, cfg(d), &self.emb, hidden, tid, d as u32))?;
        }

        // Expert stride constants for quantized packed/scales offsets
        let mid = self.mid;
        let (gate_bytes_per_exp, gate_scales_per_exp, down_bytes_per_exp, down_scales_per_exp) =
            match self.qt {
                QuantType::Q4_0 => {
                    let gate_bytes_per_row = d.div_ceil(32) * 16;
                    let down_bytes_per_row = mid.div_ceil(32) * 16;
                    (
                        mid * gate_bytes_per_row,
                        mid * d.div_ceil(32),
                        d * down_bytes_per_row,
                        d * mid.div_ceil(32),
                    )
                }
                QuantType::Q8_0 => (
                    mid * d, // 1 byte per value
                    mid * d.div_ceil(32),
                    d * mid,
                    d * mid.div_ceil(32),
                ),
                QuantType::Q2S => (
                    mid * (d + 3) / 4,
                    mid * d.div_ceil(64),
                    d * (mid + 3) / 4,
                    d * mid.div_ceil(64),
                ),
                QuantType::T1S => (
                    mid * (d + 3) / 4,
                    mid * d.div_ceil(64),
                    d * (mid + 3) / 4,
                    d * mid.div_ceil(64),
                ),
            };

        for (li, layer) in self.layers.iter().enumerate() {
            let qt = self.qt;
            // ── Attention norm ──
            cu(m.rms_norm_fused(s, cfg1(d), hidden, &layer.an, normed, d as u32, self.eps))?;

            // ── Q/K/V projections (GQA: k/v have kv_dim outputs, q has d) ──
            if qt == QuantType::Q4_0 && kv_dim == d {
                // MHA: all three outputs same size, use fused triple kernel
                cu(m.gemv_triple_q4(
                    s,
                    cfg(d),
                    normed,
                    &layer.qp_packed,
                    &layer.qp_scales,
                    q,
                    &layer.kp_packed,
                    &layer.kp_scales,
                    k,
                    &layer.vp_packed,
                    &layer.vp_scales,
                    v,
                    d as u32,
                ))?;
            } else {
                // Q projection (d outputs)
                gemv_quant(
                    m,
                    s,
                    qt,
                    cfg(d),
                    normed,
                    &layer.qp_packed,
                    &layer.qp_scales,
                    q,
                    d as u32,
                )?;
                // K projection (kv_dim outputs)
                gemv_quant(
                    m,
                    s,
                    qt,
                    cfg(kv_dim),
                    normed,
                    &layer.kp_packed,
                    &layer.kp_scales,
                    k,
                    d as u32,
                )?;
                // V projection (kv_dim outputs)
                gemv_quant(
                    m,
                    s,
                    qt,
                    cfg(kv_dim),
                    normed,
                    &layer.vp_packed,
                    &layer.vp_scales,
                    v,
                    d as u32,
                )?;
            }

            // ── Q/K head norms (GQA: k norm has kv_dim elements) ──
            cu(m.rms_norm_fused(s, cfg1(d), q, &layer.qn, q_tmp, d as u32, self.eps))?;
            cu(m.rms_norm_fused(
                s,
                cfg1(kv_dim),
                k,
                &layer.kn,
                k_tmp,
                kv_dim as u32,
                self.eps,
            ))?;

            // ── RoPE on GPU: q_tmp→q (nh heads), k_tmp→k (nkv heads) ──
            let pos = self.cur_seq;
            cu(m.rope(
                s,
                cfg(d),
                q_tmp,
                &self.rope_cos,
                &self.rope_sin,
                q,
                pos as u32,
                self.nh as u32,
                self.hd as u32,
            ))?;
            cu(m.rope(
                s,
                cfg(kv_dim),
                k_tmp,
                &self.rope_cos,
                &self.rope_sin,
                k,
                pos as u32,
                nkv as u32,
                self.hd as u32,
            ))?;

            // Copy K_rot, V into GPU KV cache (kv_dim elements per position)
            let offset = (pos * kv_dim * 4) as usize;
            let k_size = kv_dim * 4;
            unsafe {
                cuda_bindings::cuMemcpyDtoD_v2(
                    self.k_cache[li].cu_deviceptr() + offset as u64,
                    k.cu_deviceptr(),
                    k_size,
                );
                cuda_bindings::cuMemcpyDtoD_v2(
                    self.v_cache[li].cu_deviceptr() + offset as u64,
                    v.cu_deviceptr(),
                    k_size,
                );
            }
            let seq_len = pos + 1;

            // GPU: attention scores (GQA-aware) — stays on GPU
            // TODO: FlashAttention kernel for long-context prefill
            let sm_scale = 1.0 / (self.hd as f32).sqrt();
            cu(m.attn_scores(
                s,
                LaunchConfig::for_num_elems((self.nh * seq_len) as u32),
                q,
                &self.k_cache[li],
                &mut self.scores_buf,
                seq_len as u32,
                self.nh as u32,
                nkv as u32,
                self.hd as u32,
                sm_scale,
            ))?;

            // GPU: inline softmax + V combine (fused, no CPU round-trip, GQA-aware)
            cu(m.attn_combine_softmax(
                s,
                cfg(d),
                &self.scores_buf,
                &self.v_cache[li],
                ao,
                seq_len as u32,
                self.nh as u32,
                nkv as u32,
                self.hd as u32,
            ))?;

            // ── O projection ──
            gemv_quant(
                m,
                s,
                qt,
                cfg(d),
                ao,
                &layer.op_packed,
                &layer.op_scales,
                q,
                d as u32,
            )?;
            cu(m.add(s, cfg(d), hidden, q, h_tmp1))?;
            cu(m.add(s, cfg(d), h_tmp1, h_tmp2, hidden))?;

            // ── FFN norm ──
            cu(m.rms_norm_fused(s, cfg1(d), hidden, &layer.fn_, ffn_in, d as u32, self.eps))?;

            // ── Router (f32, small) + GPU top-k ──
            cu(m.gemv_f32(s, cfg(self.ne), ffn_in, &layer.rt, router_out, d as u32))?;
            // GPU-side top-k expert selection (no CPU round-trip for logits)
            cu(m.router_topk(
                s,
                cfg1(self.ne),
                router_out,
                topk_idx,
                topk_w,
                self.ne as u32,
                self.na as u32,
                self.norm_topk_prob as u32,
            ))?;
            // P10.1: Download only k indices + k weights (tiny: na×4 bytes, e.g. 8×4=32).
            // This is a single host sync per token, negligible overhead vs the
            // ~100+ μs of expert GEMV kernels. Already optimal — no further work needed.
            let tk_idx = cu(topk_idx.to_host_vec(s))?;
            let tk_w = cu(topk_w.to_host_vec(s))?;

            // ── Expert FFN ──
            {
                let _s = tracing::trace_span!("expert_loop", layer = li).entered();
                cu(m.mul(s, cfg(d), hidden, h_tmp2, fo))?; // fo = 0

                for k in 0..self.na {
                    let eid = tk_idx[k] as usize;
                    self.expert_hits[li][eid] = self.expert_hits[li][eid].saturating_add(1);
                    let w = tk_w[k];
                    let gate_packed_off = eid as u32 * gate_bytes_per_exp as u32;
                    let gate_scales_off = eid as u32 * gate_scales_per_exp as u32;
                    let down_packed_off = eid as u32 * down_bytes_per_exp as u32;
                    let down_scales_off = eid as u32 * down_scales_per_exp as u32;

                    // gate + up (fused dual for Q4_0)
                    if qt == QuantType::Q4_0 {
                        cu(m.gemv_dual_q4_off(
                            s,
                            cfg(self.mid),
                            ffn_in,
                            &layer.ex_gate_packed,
                            &layer.ex_gate_scales,
                            gb,
                            gate_packed_off as u32,
                            gate_scales_off as u32,
                            &layer.ex_up_packed,
                            &layer.ex_up_scales,
                            ub,
                            gate_packed_off as u32,
                            gate_scales_off as u32,
                            d as u32,
                        ))?;
                    } else {
                        gemv_quant_off(
                            m,
                            s,
                            qt,
                            cfg(self.mid),
                            ffn_in,
                            &layer.ex_gate_packed,
                            &layer.ex_gate_scales,
                            gb,
                            d as u32,
                            gate_packed_off as u32,
                            gate_scales_off as u32,
                        )?;
                        gemv_quant_off(
                            m,
                            s,
                            qt,
                            cfg(self.mid),
                            ffn_in,
                            &layer.ex_up_packed,
                            &layer.ex_up_scales,
                            ub,
                            d as u32,
                            gate_packed_off as u32,
                            gate_scales_off as u32,
                        )?;
                    }

                    // SiLU(gate) * up (fused silu+mul → saves 1 launch)
                    cu(m.silu_mul(s, cfg(self.mid), gb, ub, gb2))?;

                    // down
                    gemv_quant_off(
                        m,
                        s,
                        qt,
                        cfg(d),
                        gb2,
                        &layer.ex_down_packed,
                        &layer.ex_down_scales,
                        db,
                        self.mid as u32,
                        down_packed_off as u32,
                        down_scales_off as u32,
                    )?;

                    // fo += w * db
                    cu(m.saxpy(s, cfg(d), w, db, fo))?;
                }

                // ── Residual ──
            } // expert_loop span
            cu(m.add(s, cfg(d), hidden, fo, h_tmp1))?;
            cu(m.add(s, cfg(d), h_tmp1, h_tmp2, hidden))?;
        }

        self.cur_seq += 1;
        self.total_tokens += 1;
        ferrule_core::observability::METRICS
            .generated_tokens
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // ── Final layer norm (model.norm.weight) ──
        cu(m.rms_norm_fused(
            s,
            cfg1(d),
            hidden,
            &self.final_norm,
            normed,
            d as u32,
            self.eps,
        ))?;

        // ── lm_head on GPU (uses normed hidden state) ──
        let vocab = self.vocab;
        cu(m.gemv_f32(
            s,
            LaunchConfig::for_num_elems(vocab as u32),
            normed,
            &self.lm_head,
            logits,
            d as u32,
        ))?;
        // P10.5: Optional GPU vocab top-K. It is opt-in because returning
        // sparse logits changes full-distribution sampling semantics
        // (top-p/min-p/logprobs/repeat penalties for omitted tokens).
        // Set FERRULE_GPU_TOPK=N to enable, capped by the fixed buffers.
        const GPU_TOPK_CAP: u32 = 40;
        let k: u32 = std::env::var_os("FERRULE_GPU_TOPK")
            .and_then(|s| s.to_string_lossy().parse().ok())
            .unwrap_or(0)
            .min(GPU_TOPK_CAP);
        let result = if k > 0 && (k as usize) < vocab {
            let k = k.min(vocab as u32);
            let one_block = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            cu(m.topk_vocab(
                s,
                one_block,
                logits,
                topk_vocab_idx,
                topk_vocab_val,
                vocab as u32,
                k,
            ))?;
            // Download only top-K indices + values
            let idx = cu(topk_vocab_idx.to_host_vec(s))?;
            let val = cu(topk_vocab_val.to_host_vec(s))?;
            // Reconstruct full logits: fill with -inf, set top-K
            let mut full = vec![f32::NEG_INFINITY; vocab];
            for j in 0..k as usize {
                let id = idx[j] as usize;
                if id < vocab {
                    full[id] = val[j];
                }
            }
            Ok(full)
        } else {
            cu(logits.to_host_vec(s))
        };
        if std::env::var_os("FERRULE_DEBUG_TOPK").is_some() {
            if let Ok(ref top) = result {
                let mut top5: Vec<(usize, f32)> = top.iter().copied().enumerate().collect();
                top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                top5.truncate(8);
                tracing::debug!(
                    "GPU top8 ids: {:?}",
                    top5.iter().map(|(i, _)| i).collect::<Vec<_>>()
                );
            }
        }
        result
    }

    /// Placeholder for CUDA graph capture of the decode loop.
    /// When implemented, captures the entire per-token forward pass
    /// into a CUDA graph for replay, eliminating per-kernel launch overhead.
    pub fn capture_decode_graph(&mut self) -> Result<()> {
        if cuda_graph_enabled() {
            tracing::info!("CUDA graph capture requested but not yet implemented");
        }
        Ok(())
    }

    pub fn reset_session(&mut self) {
        self.cur_seq = 0;
    }

    /// Generate a human-readable expert activation report.
    pub fn expert_report(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "Expert activation report ({} tokens):",
            self.total_tokens
        );
        for (li, layer_hits) in self.expert_hits.iter().enumerate() {
            let total_hits: usize = layer_hits.iter().sum();
            if total_hits == 0 {
                continue;
            }
            // Find top-5 experts by hit count
            let mut ranked: Vec<(usize, usize)> = layer_hits.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1));
            let _ = write!(out, "  layer {li:>2}: ");
            for (eid, count) in ranked.iter().take(5) {
                if *count > 0 {
                    let _ = write!(out, "e{eid}:{count} ");
                }
            }
            let _ = writeln!(out);
        }
        out
    }
}
