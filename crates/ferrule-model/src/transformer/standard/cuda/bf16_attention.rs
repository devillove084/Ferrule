//! Image-owned prepared attention; KV mutation authority stays in the typed adapter.
use super::*;
use ferrule_backend::cuda::operators::attention::CudaPreparedBf16Attention;
use ferrule_backend::cuda::operators::rope::CudaI32Buffer;
use ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionPlan;

pub(super) struct PreparedAttention {
    pub provider: CudaPreparedBf16Attention,
    pub status: CudaI32Buffer,
    active: Option<ExecutionTransactionId>,
}
impl PreparedAttention {
    pub fn is_poisoned(&self) -> bool {
        self.provider.is_poisoned()
    }
    pub fn begin(
        &mut self,
        ops: &CudaOperators,
        transaction: ExecutionTransactionId,
    ) -> Result<()> {
        if self.active.is_some() {
            return Err(cuda_error("attention invocation still active"));
        }
        self.provider.begin_invocation(ops, &mut self.status)?;
        self.active = Some(transaction);
        Ok(())
    }
    pub fn validate_transaction(&self, transaction: ExecutionTransactionId) -> Result<()> {
        if self.active != Some(transaction) || self.is_poisoned() {
            return Err(cuda_error(
                "attention invocation transaction mismatch or Unknown",
            ));
        }
        Ok(())
    }
    pub fn finish(
        &mut self,
        ops: &CudaOperators,
        transaction: ExecutionTransactionId,
        abort: bool,
    ) -> Result<()> {
        if self.active.is_none() {
            return Ok(());
        }
        self.validate_transaction(transaction)?;
        let result = self.provider.finish_invocation(ops, &self.status);
        if !self.is_poisoned() {
            // Owned status completion consumed the invocation, including a semantic
            // rejection. Abort must still allow the existing journal to roll back.
            self.active = None;
            if abort {
                return Ok(());
            }
        }
        result
    }
}
impl CudaStandardDecoderOperators {
    pub(super) fn prepare_bf16_attention(&mut self, plan: Option<Bf16AttentionPlan>) -> Result<()> {
        let Some(plan) = plan else {
            return Ok(());
        };
        if self.bf16_attention.is_some() {
            return Err(cuda_error("Mixed attention already prepared"));
        }
        self.synchronous(|this| {
            let status = this.ops.zero_i32_buffer(1)?;
            match this.ops.prepare_bf16_attention(plan) {
                Ok(provider) => {
                    this.bf16_attention =
                        Some(Rc::new(std::cell::RefCell::new(PreparedAttention {
                            provider,
                            status,
                            active: None,
                        })))
                }
                Err(error) => {
                    if CudaPreparedBf16Attention::preparation_completion_unknown(&this.ops) {
                        this.poisoned = true;
                        this.cache_quarantine();
                        std::mem::forget(status);
                    }
                    return Err(error);
                }
            }
            Ok(())
        })
    }
    pub(super) fn begin_bf16_attention(
        &mut self,
        transaction: ExecutionTransactionId,
    ) -> Result<()> {
        if self.needs_quarantine() {
            return Err(cuda_error("attention owner completion Unknown"));
        }
        if let Some(owner) = &self.bf16_attention {
            owner.borrow_mut().begin(&self.ops, transaction)?;
        }
        Ok(())
    }
    pub(super) fn finish_bf16_attention(
        &mut self,
        transaction: ExecutionTransactionId,
        abort: bool,
    ) -> Result<()> {
        if let Some(owner) = &self.bf16_attention {
            owner.borrow_mut().finish(&self.ops, transaction, abort)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/transformer/bf16_attention/gpu.rs"]
mod tests;
