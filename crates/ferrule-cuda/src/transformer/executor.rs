use cuda_core::LaunchConfig;
use ferrule_core::Result;

use crate::context::cu;
use crate::forward::GpuOlmoeModel;

/// Concrete CUDA Transformer executor over an owned model state.
///
/// This is the Rust-side equivalent of a PyTorch `Module.forward(...)` wrapper:
/// it composes embedding, per-layer attention/MLP steps, and logits projection
/// while the model object owns weights, scratch buffers, and KV state. Future
/// families should vary the step policies/layouts behind this executor rather
/// than duplicating a full runner.
pub(crate) struct CudaTransformerExecutor<'model> {
    pub(crate) model: &'model mut GpuOlmoeModel,
}

impl<'model> CudaTransformerExecutor<'model> {
    pub(crate) fn new(model: &'model mut GpuOlmoeModel) -> Self {
        Self { model }
    }

    pub(crate) fn forward_token(mut self, tid: u32) -> Result<Vec<f32>> {
        self.model.kv.ensure_token_room()?;
        cu(self.model.ctx.bind_to_thread())?;

        self.embed_token(tid)?;

        let expert_offsets = self.expert_quant_offsets();
        for li in 0..self.model.layers.len() {
            self.attention_step(li)?;
            self.moe_step(li, expert_offsets)?;
        }

        self.model.kv.advance_token();
        self.model.total_tokens += 1;
        ferrule_core::observability::METRICS
            .generated_tokens
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        self.logits_step()
    }

    fn embed_token(&mut self, tid: u32) -> Result<()> {
        let _span = tracing::trace_span!("embed").entered();
        let cfg = LaunchConfig::for_num_elems(self.model.d as u32);
        cu(unsafe {
            self.model.module.embed_lookup(
                &self.model.s,
                cfg,
                &self.model.emb,
                &mut self.model.scratch.hidden,
                tid,
                self.model.d as u32,
            )
        })
    }
}
