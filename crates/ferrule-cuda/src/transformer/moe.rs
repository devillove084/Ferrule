use cuda_core::LaunchConfig;
use ferrule_core::{Error, Result};
use ferrule_quant::QuantType;

use crate::context::{cu, gemv_quant_off};
use crate::forward::Scratch;
use crate::transformer::CudaTransformerExecutor;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertQuantOffsets {
    pub(crate) gate_bytes_per_exp: usize,
    pub(crate) gate_scales_per_exp: usize,
    pub(crate) down_bytes_per_exp: usize,
    pub(crate) down_scales_per_exp: usize,
}

impl CudaTransformerExecutor<'_> {
    pub(crate) fn expert_quant_offsets(&self) -> ExpertQuantOffsets {
        match self.model.qt {
            QuantType::Q4_0 => {
                let gate_bytes_per_row = self.model.d.div_ceil(32) * 16;
                let down_bytes_per_row = self.model.mid.div_ceil(32) * 16;
                ExpertQuantOffsets {
                    gate_bytes_per_exp: self.model.mid * gate_bytes_per_row,
                    gate_scales_per_exp: self.model.mid * self.model.d.div_ceil(32),
                    down_bytes_per_exp: self.model.d * down_bytes_per_row,
                    down_scales_per_exp: self.model.d * self.model.mid.div_ceil(32),
                }
            }
            QuantType::Q8_0 => ExpertQuantOffsets {
                gate_bytes_per_exp: self.model.mid * self.model.d,
                gate_scales_per_exp: self.model.mid * self.model.d.div_ceil(32),
                down_bytes_per_exp: self.model.d * self.model.mid,
                down_scales_per_exp: self.model.d * self.model.mid.div_ceil(32),
            },
            QuantType::Q2S | QuantType::T1S => ExpertQuantOffsets {
                gate_bytes_per_exp: self.model.mid * (self.model.d + 3) / 4,
                gate_scales_per_exp: self.model.mid * self.model.d.div_ceil(64),
                down_bytes_per_exp: self.model.d * (self.model.mid + 3) / 4,
                down_scales_per_exp: self.model.d * self.model.mid.div_ceil(64),
            },
        }
    }

    /// Run one Transformer MoE block for the current token hidden state.
    ///
    /// This preserves the existing kernel sequence: FFN norm → router → GPU
    /// top-k → selected expert gate/up/down loop → residual.
    pub(crate) fn moe_step(&mut self, li: usize, offsets: ExpertQuantOffsets) -> Result<()> {
        let model = &mut self.model;
        let m = &model.module;
        let s = &model.s;
        let d = model.d;
        let mid = model.mid;
        let ne = model.ne;
        let na = model.na;
        let eps = model.eps;
        let norm_topk_prob = model.norm_topk_prob;
        let qt = model.qt;
        let layer = &model.layers[li];
        let hits = &mut model.expert_hits[li];
        let cfg = |n: usize| LaunchConfig::for_num_elems(n as u32);
        let cfg1 = |n: usize| LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n.min(1024) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let Scratch {
            hidden,
            h_tmp1,
            h_tmp2,
            ffn_in,
            router_out,
            fo,
            gb,
            ub,
            gb2,
            db,
            topk_idx,
            topk_w,
            ..
        } = &mut model.scratch;

        // FFN norm.
        cu(m.rms_norm_fused(s, cfg1(d), hidden, &layer.fn_, ffn_in, d as u32, eps))?;

        // Router (f32, small) + GPU top-k.
        cu(m.gemv_f32(
            s,
            cfg(ne),
            ffn_in,
            &layer.rt,
            router_out,
            ne as u32,
            d as u32,
        ))?;
        cu(m.router_topk(
            s,
            cfg1(ne),
            router_out,
            topk_idx,
            topk_w,
            ne as u32,
            na as u32,
            norm_topk_prob as u32,
        ))?;

        // Tiny host sync: only top-k expert indices and weights.
        let tk_idx = cu(topk_idx.to_host_vec(s))?;
        let tk_w = cu(topk_w.to_host_vec(s))?;

        cu(m.mul(s, cfg(d), hidden, h_tmp2, fo, d as u32))?; // fo = 0

        for rank in 0..na {
            let eid = tk_idx[rank] as usize;
            let hit = hits.get_mut(eid).ok_or_else(|| {
                Error::Internal(format!("router selected expert {eid}, but model has {ne}"))
            })?;
            *hit = hit.saturating_add(1);
            let w = tk_w[rank];
            let gate_packed_off = eid as u32 * offsets.gate_bytes_per_exp as u32;
            let gate_scales_off = eid as u32 * offsets.gate_scales_per_exp as u32;
            let down_packed_off = eid as u32 * offsets.down_bytes_per_exp as u32;
            let down_scales_off = eid as u32 * offsets.down_scales_per_exp as u32;

            // gate + up (fused dual for Q4_0)
            if qt == QuantType::Q4_0 {
                cu(m.gemv_dual_q4_off(
                    s,
                    cfg(mid),
                    ffn_in,
                    &layer.ex_gate_packed,
                    &layer.ex_gate_scales,
                    gb,
                    gate_packed_off,
                    gate_scales_off,
                    &layer.ex_up_packed,
                    &layer.ex_up_scales,
                    ub,
                    gate_packed_off,
                    gate_scales_off,
                    mid as u32,
                    d as u32,
                ))?;
            } else {
                gemv_quant_off(
                    m,
                    s,
                    qt,
                    cfg(mid),
                    ffn_in,
                    &layer.ex_gate_packed,
                    &layer.ex_gate_scales,
                    gb,
                    mid as u32,
                    d as u32,
                    gate_packed_off,
                    gate_scales_off,
                )?;
                gemv_quant_off(
                    m,
                    s,
                    qt,
                    cfg(mid),
                    ffn_in,
                    &layer.ex_up_packed,
                    &layer.ex_up_scales,
                    ub,
                    mid as u32,
                    d as u32,
                    gate_packed_off,
                    gate_scales_off,
                )?;
            }

            // SiLU(gate) * up.
            cu(m.silu_mul(s, cfg(mid), gb, ub, gb2, mid as u32))?;

            // down.
            gemv_quant_off(
                m,
                s,
                qt,
                cfg(d),
                gb2,
                &layer.ex_down_packed,
                &layer.ex_down_scales,
                db,
                d as u32,
                mid as u32,
                down_packed_off,
                down_scales_off,
            )?;

            // fo += w * db.
            cu(m.saxpy(s, cfg(d), w, db, fo, d as u32))?;
        }

        // Residual.
        cu(m.add(s, cfg(d), hidden, fo, h_tmp1, d as u32))?;
        cu(m.add(s, cfg(d), h_tmp1, h_tmp2, hidden, d as u32))?;
        Ok(())
    }
}
