//! Call-scoped HostKV authority for the ordinary synchronous transformer loop.
use super::invocation::InvocationAdapter;
use super::{DecoderInvocation, DecoderInvocationScope, PackedDecoderBatch};
use ferrule_backend::cuda::operators::kv::transaction::append_bridge::AppendPrefixAuthority;
use ferrule_common::execution::{ExecutionTransactionId, ForwardMode, ForwardPhase};
use ferrule_common::host_kv::placement::PlacementBudget;
use ferrule_common::{Error, Result};

/// Borrows live reservation authorities, never copied witness authorization.
/// No continuation owns this context. All selected KV layers must return their
/// physical output custody before this synchronous invocation may complete.
///
/// ```compile_fail
/// use ferrule_model::decoder::CudaHostKvAppendContext;
/// fn escape<'a>(context: CudaHostKvAppendContext<'a>) -> CudaHostKvAppendContext<'static> {
///     context
/// }
/// ```
pub struct CudaHostKvAppendContext<'a> {
    authorities: Vec<&'a dyn AppendPrefixAuthority>,
    budget: &'a PlacementBudget,
    block_tokens: usize,
}
impl<'a> CudaHostKvAppendContext<'a> {
    pub fn new(
        authorities: Vec<&'a dyn AppendPrefixAuthority>,
        budget: &'a PlacementBudget,
        block_tokens: usize,
    ) -> Result<Self> {
        let first = authorities
            .first()
            .ok_or_else(|| error("no GQA layer authority"))?;
        if block_tokens == 0
            || first.witness().source().committed_tokens == 0
            || first
                .witness()
                .source()
                .schema
                .planes()
                .iter()
                .any(|p| p.layer_count != authorities.len())
        {
            return Err(error("requires nonempty prefix and every GQA layer"));
        }
        for (layer, authority) in authorities.iter().enumerate() {
            authority.validate()?;
            if authority.layer() != layer || authority.witness() != first.witness() {
                return Err(error(
                    "layer authorities differ in live reservation identity",
                ));
            }
        }
        Ok(Self {
            authorities,
            budget,
            block_tokens,
        })
    }
    pub(crate) fn layer(&self, layer: usize) -> Result<&dyn AppendPrefixAuthority> {
        let authority = *self
            .authorities
            .get(layer)
            .ok_or_else(|| error("missing layer"))?;
        authority.validate()?;
        Ok(authority)
    }
    pub(crate) fn budget(&self) -> &PlacementBudget {
        self.budget
    }
    pub(crate) fn block_tokens(&self) -> usize {
        self.block_tokens
    }
    pub(crate) fn invocation_scope(&self) -> Result<DecoderInvocationScope<'_>> {
        DecoderInvocationScope::with_adapter(InvocationAdapter::CudaHostKv(self))
    }
}
impl DecoderInvocation for CudaHostKvAppendContext<'_> {
    fn validate_batch(
        &self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) -> Result<()> {
        let witness = self.authorities[0].witness();
        if batch.mode() != ForwardMode::Decode
            || batch.len() != 1
            || batch.sequences().len() != 1
            || batch.sequences()[0].phase() != ForwardPhase::Decode
            || transaction != witness.transaction()
            || batch.positions() != [witness.source().committed_tokens]
            || batch.sequences()[0].block_table() != witness.pages()
        {
            return Err(error(
                "only fresh single-sequence single-token decode is supported; no prefill fallback",
            ));
        }
        for authority in &self.authorities {
            authority.validate()?;
        }
        Ok(())
    }
    fn layer_count(&self) -> usize {
        self.authorities.len()
    }
    fn validate_layer(&self, layer: usize) -> Result<()> {
        self.layer(layer).map(|_| ())
    }
}
impl DecoderInvocationScope<'_> {
    pub(crate) fn cuda_host_kv(&self) -> Result<&CudaHostKvAppendContext<'_>> {
        match &self.adapter {
            InvocationAdapter::CudaHostKv(append) => Ok(append),
            InvocationAdapter::Validation(_) => {
                Err(error("invocation has no CUDA HostKV append adapter"))
            }
        }
    }
}
fn error(message: &str) -> Error {
    Error::Execution {
        message: format!("HostKV append: {message}"),
    }
}
