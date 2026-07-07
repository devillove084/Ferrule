use cuda_core::LaunchConfig;
use ferrule_core::Result;
use ferrule_quant::QuantType;

use crate::context::{cu, gemv_quant};
use crate::forward::Scratch;
use crate::transformer::CudaTransformerExecutor;

impl CudaTransformerExecutor<'_> {
    /// Run one Transformer attention block for the current token position.
    ///
    /// This preserves the existing kernel sequence: norm → Q/K/V projections →
    /// Q/K norms → RoPE → contiguous KV append → scores → softmax+combine → O
    /// projection → residual.
    pub(crate) fn attention_step(&mut self, li: usize) -> Result<()> {
        let model = &mut self.model;
        let m = &model.module;
        let s = &model.s;
        let d = model.d;
        let kv_dim = model.kv_dim;
        let nh = model.nh;
        let nkv = model.nkv;
        let hd = model.hd;
        let eps = model.eps;
        let qt = model.qt;
        let pos = model.kv.current_position();
        let layer = &model.layers[li];
        let cfg = |n: usize| LaunchConfig::for_num_elems(n as u32);
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
            ..
        } = &mut model.scratch;

        // Attention norm.
        cu(unsafe { m.rms_norm_fused(s, cfg1(d), hidden, &layer.an, normed, d as u32, eps) })?;

        // Q/K/V projections (GQA: k/v have kv_dim outputs, q has d).
        if qt == QuantType::Q4_0 && kv_dim == d {
            cu(unsafe {
                m.gemv_triple_q4(
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
                    d as u32,
                )
            })?;
        } else {
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
                d as u32,
            )?;
            gemv_quant(
                m,
                s,
                qt,
                cfg(kv_dim),
                normed,
                &layer.kp_packed,
                &layer.kp_scales,
                k,
                kv_dim as u32,
                d as u32,
            )?;
            gemv_quant(
                m,
                s,
                qt,
                cfg(kv_dim),
                normed,
                &layer.vp_packed,
                &layer.vp_scales,
                v,
                kv_dim as u32,
                d as u32,
            )?;
        }

        // Q/K head norms (GQA: k norm has kv_dim elements).
        cu(unsafe { m.rms_norm_fused(s, cfg1(d), q, &layer.qn, q_tmp, d as u32, eps) })?;
        cu(unsafe { m.rms_norm_fused(s, cfg1(kv_dim), k, &layer.kn, k_tmp, kv_dim as u32, eps) })?;

        // RoPE on GPU: q_tmp→q (nh heads), k_tmp→k (nkv heads).
        cu(unsafe {
            m.rope(
                s,
                cfg(d),
                q_tmp,
                &model.rope_cos,
                &model.rope_sin,
                q,
                pos as u32,
                nh as u32,
                hd as u32,
            )
        })?;
        cu(unsafe {
            m.rope(
                s,
                cfg(kv_dim),
                k_tmp,
                &model.rope_cos,
                &model.rope_sin,
                k,
                pos as u32,
                nkv as u32,
                hd as u32,
            )
        })?;

        // Append K_rot/V into the contiguous GPU KV cache.
        model.kv.append_layer(li, pos, k, v)?;
        let seq_len = model.kv.current_seq_len_after_append();
        let (k_cache, v_cache, scores_buf) = model.kv.layer_buffers_mut(li)?;

        // Attention scores (GQA-aware) — stays on GPU.
        let sm_scale = 1.0 / (hd as f32).sqrt();
        cu(unsafe {
            m.attn_scores(
                s,
                LaunchConfig::for_num_elems((nh * seq_len) as u32),
                q,
                k_cache,
                scores_buf,
                seq_len as u32,
                nh as u32,
                nkv as u32,
                hd as u32,
                sm_scale,
            )
        })?;

        // Inline softmax + V combine (fused, no CPU round-trip, GQA-aware).
        cu(unsafe {
            m.attn_combine_softmax(
                s,
                cfg(d),
                scores_buf,
                v_cache,
                ao,
                seq_len as u32,
                nh as u32,
                nkv as u32,
                hd as u32,
            )
        })?;

        // O projection + residual.
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
            d as u32,
        )?;
        cu(unsafe { m.add(s, cfg(d), hidden, q, h_tmp1, d as u32) })?;
        cu(unsafe { m.add(s, cfg(d), h_tmp1, h_tmp2, hidden, d as u32) })?;
        Ok(())
    }
}
