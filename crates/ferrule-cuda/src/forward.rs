//! GPU OLMoE forward pass — quantized weights, scratch pool, zero CPU roundtrips.
#![allow(unsafe_code)]

use cuda_core::{CudaContext, DeviceBuffer};
use ferrule_core::Result;
use ferrule_quant::QuantType;
use std::sync::Arc;

use crate::transformer::{CudaContiguousKvCache, CudaTransformerExecutor};

// Re-export public items that moved to sub-modules (preserve API paths).
pub use crate::context::{cuda_gemv, cuda_probe};
pub use crate::graph::{cuda_graph_enabled, flash_attn_enabled};

// ── Attention optimization plan (P10.4) ─────────────────────────────────
//
// Current status: standard MHA/GQA attention with per-token score computation
// and in-GPU softmax+combine. Planned upgrades should plug into the attention
// step abstraction rather than duplicating the whole model forward path.

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
    // ── Attention geometry + state ──
    pub(crate) nh: usize,
    pub(crate) nkv: usize,
    pub(crate) hd: usize,
    pub(crate) rope_cos: DeviceBuffer<f32>,
    pub(crate) rope_sin: DeviceBuffer<f32>,
    pub(crate) kv: CudaContiguousKvCache,
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

    /// Build GPU model from a lightweight CPU model + pre-existing weight pack.
    /// Skips full FP32 weight loading — norms, router, embed, lm_head come
    /// from the lightweight model; quantized attention/expert weights come
    /// from the weight pack.
    pub fn from_lightweight(
        model: &ferrule_model::OlmoeModel,
        weightpack: &crate::weightpack::WeightPackReader,
        qt: QuantType,
    ) -> Result<Self> {
        Self::build_from_lightweight(model, weightpack, qt)
    }

    pub fn forward(&mut self, tid: u32) -> Result<Vec<f32>> {
        let _span = tracing::debug_span!("gpu_forward", token = tid).entered();
        CudaTransformerExecutor::new(self).forward_token(tid)
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
        self.kv.reset();
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
