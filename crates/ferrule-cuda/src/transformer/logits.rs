use cuda_core::LaunchConfig;
use ferrule_core::Result;

use crate::context::cu;
use crate::forward::Scratch;
use crate::transformer::CudaTransformerExecutor;

impl CudaTransformerExecutor<'_> {
    /// Run final norm + lm_head projection and return logits.
    pub(crate) fn logits_step(&mut self) -> Result<Vec<f32>> {
        let model = &mut self.model;
        let m = &model.module;
        let s = &model.s;
        let d = model.d;
        let vocab = model.vocab;
        let eps = model.eps;
        let cfg1 = |n: usize| LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n.min(1024) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let Scratch {
            hidden,
            normed,
            logits,
            topk_vocab_idx,
            topk_vocab_val,
            ..
        } = &mut model.scratch;

        // Final layer norm.
        cu(m.rms_norm_fused(s, cfg1(d), hidden, &model.final_norm, normed, d as u32, eps))?;

        // lm_head on GPU.
        cu(m.gemv_f32(
            s,
            LaunchConfig::for_num_elems(vocab as u32),
            normed,
            &model.lm_head,
            logits,
            vocab as u32,
            d as u32,
        ))?;

        // Optional GPU vocab top-K. Opt-in because returning sparse logits
        // changes full-distribution sampling semantics.
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
            let idx = cu(topk_vocab_idx.to_host_vec(s))?;
            let val = cu(topk_vocab_val.to_host_vec(s))?;
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
                let mut top8: Vec<(usize, f32)> = top.iter().copied().enumerate().collect();
                top8.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                top8.truncate(8);
                tracing::debug!(
                    "GPU top8 ids: {:?}",
                    top8.iter().map(|(i, _)| i).collect::<Vec<_>>()
                );
            }
        }

        result
    }
}
