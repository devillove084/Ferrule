//! Borrowed invocation policy, not a KV owner or physical backend capability.
use super::PackedDecoderBatch;
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, Result};
use std::cell::Cell;

/// Live validation supplied by a call-local adapter. Implementing this protocol
/// grants no physical read/write authority: CUDA still requires its audited
/// append authority and authenticated transaction ticket.
///
/// Validators must consult the retained live owner, including generation and
/// transaction identity, on every call rather than cache a successful check.
pub trait DecoderInvocation {
    fn validate_batch(
        &self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) -> Result<()>;
    fn layer_count(&self) -> usize;
    fn validate_layer(&self, layer: usize) -> Result<()>;
}

// Only this capability projection varies with the compiled physical adapter;
// the validator protocol and invocation lifecycle have one shape in every build.
pub(super) enum InvocationAdapter<'a> {
    Validation(&'a dyn DecoderInvocation),
    #[cfg(feature = "cuda")]
    CudaHostKv(&'a super::CudaHostKvAppendContext<'a>),
}
impl InvocationAdapter<'_> {
    fn protocol(&self) -> &dyn DecoderInvocation {
        match self {
            Self::Validation(invocation) => *invocation,
            #[cfg(feature = "cuda")]
            Self::CudaHostKv(append) => *append,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fresh,
    Ready(usize),
    Running(usize),
    Finished,
}

/// One synchronous invocation borrows its validator and keeps only ordering
/// state. It owns no reservation, witness, ticket, resource, or continuation.
///
/// The borrow cannot escape into a continuation with a longer lifetime:
/// ```compile_fail
/// use ferrule_model::decoder::{DecoderInvocationScope, DecoderTransactionContext};
/// struct Continuation { scope: Option<&'static DecoderInvocationScope<'static>> }
/// fn retain(context: &DecoderTransactionContext<'_>, continuation: &mut Continuation) {
///     continuation.scope = context.invocation();
/// }
/// ```
/// Nor can the borrowed validator be promoted to static ownership:
/// ```compile_fail
/// use ferrule_model::decoder::DecoderInvocationScope;
/// fn escape<'a>(scope: DecoderInvocationScope<'a>) -> DecoderInvocationScope<'static> {
///     scope
/// }
/// ```
pub struct DecoderInvocationScope<'a> {
    pub(super) adapter: InvocationAdapter<'a>,
    layer_count: usize,
    phase: Cell<Phase>,
}
impl<'a> DecoderInvocationScope<'a> {
    pub fn new(invocation: &'a dyn DecoderInvocation) -> Result<Self> {
        Self::with_adapter(InvocationAdapter::Validation(invocation))
    }
    pub(super) fn with_adapter(adapter: InvocationAdapter<'a>) -> Result<Self> {
        let layer_count = adapter.protocol().layer_count();
        if layer_count == 0 {
            return Err(error("invocation requires at least one layer"));
        }
        Ok(Self {
            adapter,
            layer_count,
            phase: Cell::new(Phase::Fresh),
        })
    }
    fn validate_live(&self) -> Result<()> {
        if self.adapter.protocol().layer_count() != self.layer_count {
            return Err(error("invocation layer count changed"));
        }
        for layer in 0..self.layer_count {
            self.adapter.protocol().validate_layer(layer)?;
        }
        Ok(())
    }
    pub(crate) fn begin(
        &self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) -> Result<()> {
        if self.phase.get() != Phase::Fresh {
            return Err(error("invocation replay"));
        }
        self.validate_batch(transaction, batch)?;
        self.phase.set(Phase::Ready(0));
        Ok(())
    }
    pub fn validate_batch(
        &self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) -> Result<()> {
        self.adapter.protocol().validate_batch(transaction, batch)?;
        self.validate_live()
    }
    pub fn layer_count(&self) -> usize {
        self.layer_count
    }
    pub fn begin_layer(&self, layer: usize) -> Result<()> {
        if layer >= self.layer_count || self.phase.get() != Phase::Ready(layer) {
            return Err(error("layer replay or previous output not retired"));
        }
        self.adapter.protocol().validate_layer(layer)?;
        self.phase.set(Phase::Running(layer));
        Ok(())
    }
    /// Call only after the adapter has returned the exact physical output
    /// custody. This changes invocation ordering, not backend retirement state.
    pub fn retire_layer(&self, layer: usize) -> Result<()> {
        if self.phase.get() != Phase::Running(layer) {
            return Err(error("layer retirement without matching invocation"));
        }
        self.adapter.protocol().validate_layer(layer)?;
        self.phase.set(Phase::Ready(layer + 1));
        Ok(())
    }
    pub fn finish(&self) -> Result<()> {
        if self.phase.get() != Phase::Ready(self.layer_count) && self.phase.get() != Phase::Finished
        {
            return Err(error("forward did not retire all layer outputs"));
        }
        self.validate_live()?;
        self.phase.set(Phase::Finished);
        Ok(())
    }
}
fn error(message: &str) -> Error {
    Error::Execution {
        message: format!("decoder synchronous invocation: {message}"),
    }
}
