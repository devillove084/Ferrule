//! Single-owner GPU state custody. Mutable device state deliberately is not Clone.
use super::*;
use crate::execution::SequenceTopologyId;
use crate::runner::SequenceStateReleaseError;
use crate::transformer::{Attention, BoundDecoderResources, FeedForward, UnsupportedOperator};
use ferrule_backend::cpu::gated_delta::GatedDeltaShape;
use ferrule_backend::cuda::operators::linear::{CudaF32Buffer, CudaOperators};
use ferrule_backend::cuda::operators::recurrent::{CausalConv1dLayout, GatedDeltaNetLayout};
use ferrule_common::execution::KvLayoutSchema;
use ferrule_common::{Error, Result};
use std::{cell::Cell, fmt, rc::Rc};

/// Completion evidence for external expert work, independent of root KV fences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridCudaExpertProgress {
    Complete,
    Pending,
    /// Permanent loss of completion evidence. The owner must retain custody.
    Unknown,
}

/// Owner-local attachment to the existing routed operator seam (no Send bound).
///
/// The executor must never publish KV or allocate a decoder transaction. It must
/// return same-owner F32 CUDA rows, and own all transport tickets/device resources
/// until `drain` proves completion. Borrowed request metadata cannot escape a call.
/// Waiting/Unsupported are fail-closed on this synchronous forward, not fallbacks.
/// A blocked model thread cannot process cancellation callbacks: transport waits
/// must have their own bounded deadline/ticket failure policy.
pub trait HybridCudaRoutedExecutor:
    crate::transformer::expert_parallel::RoutedSwiGluExecutor
{
    /// Poll all submitted work. Complete proves every ticket retired; Pending
    /// and Err retain custody and may be retried. Lost evidence MUST be Unknown,
    /// not Err followed by a later optimistic Complete.
    fn drain(&mut self) -> Result<HybridCudaExpertProgress>;
    /// Close external owners after drain; the same completion contract applies.
    fn shutdown(&mut self) -> Result<HybridCudaExpertProgress>;
    /// Notify transport cancellation after a forward failure; does not release custody.
    fn on_error(&mut self, _transaction: ferrule_common::execution::ExecutionTransactionId) {}
}

/// Optional external routed-only execution. Router/shared experts remain on root.
/// Construct on the root owner thread; use `HybridCudaDevice::operators()` for
/// the existing CUDA result-combine adapter, not another same-ordinal context.
pub struct HybridCudaRoutedExperts {
    pub(crate) closed: bool,
    pub(crate) owner: Option<HybridCudaDevice>,
    pub(crate) source_rank: ferrule_common::ParallelRankId,
    pub(crate) executor: Option<Box<dyn HybridCudaRoutedExecutor>>,
    pub(crate) check_active:
        Option<Box<dyn FnMut(ferrule_common::execution::ExecutionTransactionId) -> Result<()>>>,
}
impl HybridCudaRoutedExperts {
    pub fn new(
        source_rank: ferrule_common::ParallelRankId,
        executor: Box<dyn HybridCudaRoutedExecutor>,
    ) -> Self {
        Self {
            closed: false,
            owner: None,
            source_rank,
            executor: Some(executor),
            check_active: None,
        }
    }
    /// Optional already-registered cancellation authority. This supplements,
    /// never replaces, the model's exact transaction/batch/phase checks. The
    /// callback must not require processing events on the blocked owner thread.
    pub fn with_active_check(
        mut self,
        check: impl FnMut(ferrule_common::execution::ExecutionTransactionId) -> Result<()> + 'static,
    ) -> Self {
        self.check_active = Some(Box::new(check));
        self
    }
}
impl fmt::Debug for HybridCudaRoutedExperts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HybridCudaRoutedExperts")
            .field("source_rank", &self.source_rank)
            .finish_non_exhaustive()
    }
}

impl Drop for HybridCudaRoutedExperts {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        // Also covers rejected preparation before a module takes custody.
        let complete = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let Some(executor) = self.executor.as_mut() else {
                return true;
            };
            matches!(executor.drain(), Ok(HybridCudaExpertProgress::Complete))
                && matches!(executor.shutdown(), Ok(HybridCudaExpertProgress::Complete))
        }));
        if !matches!(complete, Ok(true))
            || self
                .owner
                .as_ref()
                .is_some_and(HybridCudaDevice::needs_quarantine)
        {
            if let Some(owner) = self.owner.take() {
                owner.quarantine();
                std::mem::forget(owner);
            }
            std::mem::forget(self.executor.take());
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HybridCudaMemoryBudget {
    /// Physical slots including transaction shadows/COW, not just committed pages.
    pub kv_pages: usize,
    /// Default state, active sequences, forks and working copies all consume this cap.
    pub state_bytes: usize,
    /// Non-expert resident bindings plus the expert cache ceiling less its
    /// numeric scratch reservation. Obtain this from `for_resources`.
    pub weight_bytes: usize,
    /// Conservative operation workspace plus numeric scratch (counted once),
    /// not an allocator reservation. Obtain this from `for_resources`.
    pub workspace_bytes: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridCudaMemoryEstimate {
    pub per_sequence_state_bytes: usize,
    pub kv_bytes: usize,
    pub weight_bytes_upper_bound: usize,
    pub workspace_bytes_upper_bound: usize,
}
#[path = "hybrid_cuda_admission.rs"]
mod admission;

#[derive(Clone)]
pub struct HybridCudaDevice {
    pub(crate) inner: Rc<DeviceInner>,
}
pub(crate) struct DeviceInner {
    pub ops: Rc<CudaOperators>,
    budget: HybridCudaMemoryBudget,
    poisoned: Cell<bool>,
    live_state_bytes: Cell<usize>,
    completed_operations: Cell<u64>,
    claimed: Cell<bool>,
    #[cfg(test)]
    reject_next_fence: Cell<bool>,
}
impl fmt::Debug for HybridCudaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HybridCudaDevice")
            .field("device", &self.device_ordinal())
            .field("quarantined", &self.needs_quarantine())
            .field("live_state_bytes", &self.live_state_bytes())
            .finish()
    }
}
impl HybridCudaDevice {
    pub fn new_on_device(ordinal: usize, budget: HybridCudaMemoryBudget) -> Result<Self> {
        if budget.kv_pages == 0
            || budget.state_bytes == 0
            || budget.weight_bytes == 0
            || budget.workspace_bytes == 0
        {
            return Err(error("hybrid CUDA budgets must be nonzero"));
        }
        Ok(Self {
            inner: Rc::new(DeviceInner {
                ops: Rc::new(CudaOperators::new_on_device(ordinal)?),
                budget,
                poisoned: Cell::new(false),
                live_state_bytes: Cell::new(0),
                completed_operations: Cell::new(0),
                claimed: Cell::new(false),
                #[cfg(test)]
                reject_next_fence: Cell::new(false),
            }),
        })
    }
    pub fn device_ordinal(&self) -> usize {
        self.inner.ops.device_ordinal()
    }
    pub fn budget(&self) -> HybridCudaMemoryBudget {
        self.inner.budget
    }
    pub fn live_state_bytes(&self) -> usize {
        self.inner.live_state_bytes.get()
    }
    pub fn completed_state_operations(&self) -> u64 {
        self.inner.completed_operations.get()
    }
    pub fn needs_quarantine(&self) -> bool {
        self.inner.poisoned.get()
    }
    /// Shared owner-local context for external routed result combine. Cloning
    /// this Rc preserves allocator/stream identity; it grants no KV authority.
    pub fn operators(&self) -> &Rc<CudaOperators> {
        &self.inner.ops
    }
    pub(crate) fn ensure_ready(&self) -> Result<()> {
        if self.needs_quarantine() {
            Err(unknown(error("hybrid CUDA owner is quarantined")))
        } else {
            Ok(())
        }
    }
    pub(crate) fn quarantine(&self) {
        self.inner.poisoned.set(true);
    }
    pub(crate) fn admit(
        &self,
        estimate: HybridCudaMemoryEstimate,
        max_sequences: usize,
    ) -> Result<HybridCudaPreparationClaim> {
        self.ensure_ready()?;
        let b = self.budget();
        let minimum = max_sequences
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(estimate.per_sequence_state_bytes))
            .ok_or_else(|| error("hybrid state admission overflow"))?;
        if estimate.weight_bytes_upper_bound > b.weight_bytes
            || estimate.workspace_bytes_upper_bound > b.workspace_bytes
            || minimum > b.state_bytes
        {
            return Err(error(format!(
                "hybrid CUDA budget too small: estimate={estimate:?}, minimum states={minimum}, budget={b:?}"
            )));
        }
        let required = b
            .weight_bytes
            .checked_add(b.workspace_bytes)
            .and_then(|n| n.checked_add(b.state_bytes))
            .and_then(|n| n.checked_add(estimate.kv_bytes))
            .ok_or_else(|| error("hybrid total budget overflow"))?;
        let (free, total) = self.inner.ops.memory_info()?;
        if required > free || required > total {
            return Err(error(format!(
                "hybrid CUDA budget exceeds device memory: required={required}, free={free}, total={total}, estimate={estimate:?}, budget={b:?}"
            )));
        }
        if self.inner.claimed.replace(true) {
            return Err(error("hybrid device factory already owns an image"));
        }
        Ok(HybridCudaPreparationClaim {
            device: self.clone(),
            committed: false,
            device_owners: Rc::strong_count(&self.inner),
            operator_owners: Rc::strong_count(self.operators()),
        })
    }
    pub(crate) fn fence(&self) -> Result<()> {
        self.ensure_ready()?;
        #[cfg(test)]
        if self.inner.reject_next_fence.replace(false) {
            self.quarantine();
            return Err(unknown(error("injected missing CUDA completion evidence")));
        }
        let compute = self
            .inner
            .ops
            .record_compute_event()
            .and_then(|event| event.synchronize());
        let upload = self
            .inner
            .ops
            .record_upload_event()
            .and_then(|event| event.synchronize());
        let failures = [compute, upload]
            .into_iter()
            .filter_map(Result::err)
            .collect::<Vec<_>>();
        if !failures.is_empty() {
            self.quarantine();
            return Err(unknown(
                Error::failures("hybrid CUDA completion fences", failures).unwrap_err(),
            ));
        }
        self.inner
            .completed_operations
            .set(self.inner.completed_operations.get().saturating_add(1));
        Ok(())
    }
    fn complete<T>(&self, operation: Result<T>) -> Result<T> {
        match self.fence() {
            Ok(()) => operation,
            Err(fence) => match operation {
                Ok(value) => {
                    std::mem::forget(value);
                    Err(fence)
                }
                Err(source) => Err(Error::with_cleanup(
                    "hybrid CUDA state operation",
                    source,
                    Err(fence),
                )),
            },
        }
    }
}

/// Failed preparation returns its claim only after temporary owners have
/// dropped and completion is proven. A successful image consumes the factory:
/// clones, including surviving states, never gain a second image authority.
pub(crate) struct HybridCudaPreparationClaim {
    device: HybridCudaDevice,
    committed: bool,
    device_owners: usize,
    operator_owners: usize,
}
impl HybridCudaPreparationClaim {
    fn finish<T>(mut self, result: Result<T>) -> Result<T> {
        match result {
            Ok(value) => {
                self.committed = true;
                Ok(value)
            }
            Err(source) => {
                let cleanup = self.rollback();
                // Do not retry a failed cleanup from Drop.
                self.committed = true;
                Err(Error::with_cleanup(
                    "hybrid CUDA preparation",
                    source,
                    cleanup,
                ))
            }
        }
    }
    fn rollback(&self) -> Result<()> {
        // Lower-level Drop may retain an owner on unknown completion without
        // returning an error. A later successful fence cannot forgive that custody.
        if std::thread::panicking()
            || self.device.live_state_bytes() != 0
            || Rc::strong_count(&self.device.inner) > self.device_owners
            || Rc::strong_count(self.device.operators()) > self.operator_owners
        {
            self.device.quarantine();
        }
        if let Err(failure) = self.device.fence() {
            std::mem::forget(self.device.clone());
            return Err(failure);
        }
        self.device.inner.claimed.set(false);
        Ok(())
    }
}
impl Drop for HybridCudaPreparationClaim {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.rollback();
        }
    }
}

pub type CudaHybridSequenceState = DecoderSequenceState<(), CudaHybridLayerStates>;
pub struct CudaGatedDeltaState {
    pub(crate) shape: GatedDeltaShape,
    pub(crate) position: usize,
    pub(crate) conv: CudaF32Buffer,
    pub(crate) recurrent: CudaF32Buffer,
    pub(crate) device: HybridCudaDevice,
}
impl fmt::Debug for CudaGatedDeltaState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaGatedDeltaState")
            .field("shape", &self.shape)
            .field("position", &self.position)
            .finish()
    }
}
impl CudaGatedDeltaState {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn shape(&self) -> GatedDeltaShape {
        self.shape
    }
    /// Explicit diagnostic readback, never used by forward, fork or rollback.
    /// Returns owned host snapshots without exposing mutable device custody.
    pub fn diagnostic_snapshot(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        self.device.fence()?;
        let result = (|| {
            Ok((
                self.device.operators().download_f32_buffer(&self.conv)?,
                self.device
                    .operators()
                    .download_f32_buffer(&self.recurrent)?,
            ))
        })();
        self.device.complete(result)
    }
    pub(crate) fn validate_owner(&self, ops: &CudaOperators) -> Result<()> {
        self.device.ensure_ready()?;
        if !std::ptr::eq(self.device.operators().as_ref(), ops) {
            return Err(error("foreign CUDA recurrent state owner"));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct CudaHybridLayerStates {
    schema: HybridStateSchema,
    layers: Vec<Option<CudaGatedDeltaState>>,
    device: HybridCudaDevice,
    charged_bytes: usize,
    release_proven: bool,
}
impl CudaHybridLayerStates {
    pub fn schema(&self) -> &HybridStateSchema {
        &self.schema
    }
    pub fn layers(&self) -> &[Option<CudaGatedDeltaState>] {
        &self.layers
    }
    pub fn resident_bytes(&self) -> usize {
        self.charged_bytes
    }
    pub fn needs_quarantine(&self) -> bool {
        self.device.needs_quarantine()
    }
    fn validate(&self, schema: &HybridStateSchema, position: usize) -> Result<()> {
        self.device.ensure_ready()?;
        if &self.schema != schema || self.layers.len() != schema.layers().len() {
            return Err(error("CUDA hybrid schema mismatch"));
        }
        for (description, state) in schema.layers().iter().zip(&self.layers) {
            match (description, state) {
                (HybridLayerSchema::FullAttention { .. }, None) => {}
                (HybridLayerSchema::GatedDeltaNet(shape), Some(state))
                    if state.shape == *shape && state.position == position =>
                {
                    state.validate_owner(self.device.operators())?;
                    let (_, conv, recurrent) = shape.sizes()?;
                    if conv != state.conv.len() || recurrent != state.recurrent.len() {
                        return Err(error("CUDA recurrent extents mismatch"));
                    }
                }
                _ => return Err(error("CUDA recurrent/conv frontier mismatch")),
            }
        }
        Ok(())
    }
}
impl Drop for CudaHybridLayerStates {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.device.quarantine();
        }
        if self.device.needs_quarantine() || (!self.release_proven && self.device.fence().is_err())
        {
            // Never reclaim buffers, owner, or budget until completion is proven.
            std::mem::forget(std::mem::take(&mut self.layers));
            std::mem::forget(self.device.clone());
            return;
        }
        self.device
            .inner
            .live_state_bytes
            .set(self.device.live_state_bytes() - self.charged_bytes);
    }
}
impl StandardSequenceState for CudaHybridSequenceState {
    fn validate_standard_schema(_: &HybridStateSchema) -> Result<()> {
        Ok(())
    }
    fn validate_standard_state_at(
        &self,
        schema: &HybridStateSchema,
        position: usize,
    ) -> Result<()> {
        self.kv_state().validate(schema, position)
    }
    fn linear_state_mut(&mut self, _: usize) -> Result<&mut GatedDeltaNetState> {
        Err(unsupported("CUDA recurrent state has no CPU fallback"))
    }
    fn linear_state(&mut self, layer: usize) -> Result<GatedDeltaStateRef<'_>> {
        self.kv_state().device.ensure_ready()?;
        match self.kv_state_mut().layers.get_mut(layer) {
            Some(Some(state)) => Ok(GatedDeltaStateRef::Cuda(state)),
            _ => Err(error("missing CUDA linear state")),
        }
    }
}

pub struct CudaHybridSequenceLifecycle {
    schema: HybridStateSchema,
    device: HybridCudaDevice,
}
impl CudaHybridSequenceLifecycle {
    pub fn new(schema: HybridStateSchema, device: HybridCudaDevice) -> Result<Self> {
        state_bytes(&schema)?;
        Ok(Self { schema, device })
    }
    fn allocate(&self) -> Result<CudaHybridLayerStates> {
        self.device.ensure_ready()?;
        let bytes = state_bytes(&self.schema)?;
        let charged = self
            .device
            .live_state_bytes()
            .checked_add(bytes)
            .filter(|n| *n <= self.device.budget().state_bytes)
            .ok_or_else(|| {
                error("hybrid CUDA state budget exhausted (working copies/forks included)")
            })?;
        self.device.inner.live_state_bytes.set(charged);
        let mut state = CudaHybridLayerStates {
            schema: self.schema.clone(),
            layers: Vec::new(),
            device: self.device.clone(),
            charged_bytes: bytes,
            release_proven: false,
        };
        let result = (|| {
            for layer in self.schema.layers() {
                state.layers.push(match layer {
                    HybridLayerSchema::FullAttention { .. } => None,
                    HybridLayerSchema::GatedDeltaNet(shape) => {
                        let (_, conv, recurrent) = shape.sizes()?;
                        Some(CudaGatedDeltaState {
                            shape: *shape,
                            position: 0,
                            conv: self.device.operators().zero_f32_buffer(conv)?,
                            recurrent: self.device.operators().zero_f32_buffer(recurrent)?,
                            device: self.device.clone(),
                        })
                    }
                });
            }
            Ok(())
        })();
        self.device.complete(result)?;
        Ok(state)
    }
    fn validate(&self, source: &CudaHybridSequenceState) -> Result<()> {
        if !Rc::ptr_eq(&self.device.inner, &source.kv_state().device.inner) {
            return Err(error("foreign hybrid CUDA lifecycle"));
        }
        source.validate_standard_state(&self.schema)
    }
    fn copy(&self, source: &CudaHybridSequenceState) -> Result<CudaHybridLayerStates> {
        self.validate(source)?;
        source.core().begin_step()?;
        self.device.fence()?;
        let mut destination = self.allocate()?;
        let copy = (|| {
            for (src, dst) in source.kv_state().layers.iter().zip(&mut destination.layers) {
                if let (Some(src), Some(dst)) = (src, dst) {
                    self.device.operators().copy_f32_range(
                        &src.conv,
                        0,
                        &mut dst.conv,
                        0,
                        src.conv.len(),
                    )?;
                    self.device.operators().copy_f32_range(
                        &src.recurrent,
                        0,
                        &mut dst.recurrent,
                        0,
                        src.recurrent.len(),
                    )?;
                    dst.position = src.position;
                }
            }
            Ok(())
        })();
        self.device.complete(copy)?;
        Ok(destination)
    }
}
impl DecoderSequenceLifecycle<CudaHybridSequenceState> for CudaHybridSequenceLifecycle {
    fn create(&mut self) -> Result<CudaHybridSequenceState> {
        Ok(DecoderSequenceState::new((), self.allocate()?))
    }
    fn checkout(
        &mut self,
        _: DecoderSequenceCheckout,
        source: &CudaHybridSequenceState,
    ) -> Result<CudaHybridSequenceState> {
        Ok(DecoderSequenceState::from_parts(
            source.core().clone(),
            source.topology_id(),
            (),
            self.copy(source)?,
        ))
    }
    fn logical_fork(
        &mut self,
        source: &CudaHybridSequenceState,
        expected_position: usize,
    ) -> Result<CudaHybridSequenceState> {
        if source.core().position() != expected_position {
            return Err(unsupported(
                "CUDA hybrid only supports exact frontier forks; partial retain unsupported",
            ));
        }
        Ok(DecoderSequenceState::from_parts(
            source.core().forked()?,
            SequenceTopologyId::take(),
            (),
            self.copy(source)?,
        ))
    }
    fn reset(&mut self, state: &mut CudaHybridSequenceState) -> Result<()> {
        self.validate(state)?;
        // Allocate/zero replacement first, so allocation failure leaves committed state intact.
        let replacement = self.allocate()?;
        state.kv_state_mut().release_proven = true;
        *state.kv_state_mut() = replacement;
        state.core_mut().reset();
        Ok(())
    }
    fn try_release(
        &mut self,
        mut state: CudaHybridSequenceState,
    ) -> std::result::Result<(), SequenceStateReleaseError<CudaHybridSequenceState>> {
        // Aborted working states may have partial layer frontiers; custody still belongs here.
        if !Rc::ptr_eq(&self.device.inner, &state.kv_state().device.inner)
            || state.kv_state().schema != self.schema
        {
            return Err(SequenceStateReleaseError::new(
                error("CUDA hybrid release owner/schema mismatch"),
                state,
            ));
        }
        if let Err(e) = self.device.fence() {
            return Err(SequenceStateReleaseError::new(e, state));
        }
        state.kv_state_mut().release_proven = true;
        drop(state);
        Ok(())
    }
}
fn state_bytes(schema: &HybridStateSchema) -> Result<usize> {
    schema.layers().iter().try_fold(0usize, |sum, layer| {
        let HybridLayerSchema::GatedDeltaNet(s) = layer else {
            return Ok(sum);
        };
        CausalConv1dLayout {
            rows: 1,
            channels: s.sizes()?.0,
            kernel_size: s.kernel,
        }
        .validate()?;
        GatedDeltaNetLayout {
            rows: 1,
            key_heads: s.key_heads,
            value_heads: s.value_heads,
            key_dim: s.key_dim,
            value_dim: s.value_dim,
        }
        .validate()?;
        let (_, c, r) = s.sizes()?;
        c.checked_add(r)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(sum))
            .ok_or_else(|| error("CUDA state byte overflow"))
    })
}
pub(crate) fn unsupported(message: &str) -> Error {
    Error::ModelSource {
        source: Box::new(UnsupportedOperator::new("hybrid_cuda", message)),
    }
}
fn error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: message.into(),
    }
}

/// Composition marker for the existing generic transaction runner.
pub struct HybridCudaDecoder;
pub type HybridCudaKvBackend = PagedKvBackend<TypedCudaPagedKvPool<CudaHybridSequenceState>>;
impl DecoderComposition for HybridCudaDecoder {
    type Resources = std::sync::Arc<BoundDecoderResources>;
    type SequenceState = CudaHybridSequenceState;
    type KvBackend = HybridCudaKvBackend;
    type ForwardExecutor =
        crate::transformer::TransformerForwardExecutor<crate::transformer::CudaHybridModule>;
    type Proposal = NoProposal;
    type SequenceLifecycle = CudaHybridSequenceLifecycle;
    type ResourceManager = NoResourceManager;
    type Observer = StandardDecoderObserver;
    type Snapshot = GenericDecoderObservabilitySnapshot;
    type Continuation =
        crate::transformer::TransformerContinuation<crate::transformer::CudaHybridModule>;
    type ProposalContinuation = std::convert::Infallible;
    type TerminalGuard = NoTerminalGuard;
}
impl GenericDecoderRunner<HybridCudaDecoder> {
    /// Single-device F32 activations/KV/state on the generic transaction path.
    /// Default options retain dense F32/BF16-checkpoint behavior. Explicit
    /// `options.with_hybrid_cuda_numeric_fp8(limits, scratch_bytes)` enables
    /// compressed linear bindings and bounded routed/shared MoE execution with
    /// BF16-RNE operands. Use `with_hybrid_cuda_numeric_fp8_precision` with
    /// `NumericFp8Precision::F32Tf32x3` for unrounded F32 operands. Neither mode
    /// changes the generic forward or enables a CPU fallback.
    ///
    /// Call `HybridCudaMemoryEstimate::for_resources` with these exact options
    /// before creating the device. The state budget must cover the default state
    /// plus committed/working sequences (at least `2 * max_sequences + 1` copies);
    /// callers wanting additional live forks must budget those copies too.
    pub fn hybrid_cuda(
        resources: BoundDecoderResources,
        tokenizer: crate::tokenizer::TokenizerHandle,
        options: GenericDecoderOptions,
        device: HybridCudaDevice,
    ) -> Result<Self> {
        Self::hybrid_cuda_inner(resources, tokenizer, options, device, None)
    }

    /// External routed-only execution on the same forward/root KV/recurrent path.
    /// Use `for_resources_with_routed_experts` for root admission; runtime owns
    /// separate external-device budgets. No local routed fallback is installed.
    /// The seam receives each packed row's stable sequence topology ID, preserved
    /// across transaction checkouts and changed by logical forks. These are model
    /// sequence identities, not runtime request IDs or transient packed indices.
    pub fn hybrid_cuda_with_routed_experts(
        resources: BoundDecoderResources,
        tokenizer: crate::tokenizer::TokenizerHandle,
        options: GenericDecoderOptions,
        device: HybridCudaDevice,
        experts: HybridCudaRoutedExperts,
    ) -> Result<Self> {
        Self::hybrid_cuda_inner(resources, tokenizer, options, device, Some(experts))
    }

    fn hybrid_cuda_inner(
        resources: BoundDecoderResources,
        tokenizer: crate::tokenizer::TokenizerHandle,
        options: GenericDecoderOptions,
        device: HybridCudaDevice,
        mut experts: Option<HybridCudaRoutedExperts>,
    ) -> Result<Self> {
        if let Some(experts) = &mut experts {
            experts.owner = Some(device.clone());
        }
        options.validate(&resources)?;
        if options.precision() != crate::execution::ExecutionPrecisionPolicy::f32() {
            return Err(unsupported("CUDA hybrid requires F32 execution"));
        }
        let schema = HybridStateSchema::from_spec(
            resources.spec(),
            options
                .active_layers()
                .unwrap_or(resources.spec().layers().len()),
        )?;
        let estimate = if experts.is_some() {
            HybridCudaMemoryEstimate::for_resources_with_routed_experts
        } else {
            HybridCudaMemoryEstimate::for_resources
        }(&resources, &options, device.budget().kv_pages)?;
        let claim = device.admit(estimate, options.capabilities().max_sequences)?;
        // Drop all temporaries before evaluating claim rollback.
        let result = (|| {
            let planes = schema.kv_planes(options.page_size(), options.max_positions())?;
            let mut model_info = super::runner::standard_model_info(&resources, &options);
            model_info.backend = options
                .hybrid_cuda_numeric_fp8_precision()
                .map_or("cuda-hybrid-f32", |precision| {
                    precision.hybrid_backend_name()
                });
            let resources = std::sync::Arc::new(resources);
            let module = crate::transformer::CudaHybridModule::prepare_with_routed_experts(
                resources.clone(),
                device.clone(),
                &options,
                experts,
            )?;
            let pool = device.complete(
                TypedCudaPagedKvPool::<CudaHybridSequenceState>::from_strategy(
                    Rc::clone(device.operators()),
                    &planes,
                    device.budget().kv_pages,
                ),
            )?;
            let mut lifecycle = CudaHybridSequenceLifecycle::new(schema, device.clone())?;
            let default_state = lifecycle.create()?;
            Self::from_runtime(DecoderComponents {
                resources,
                model_info,
                tokenizer,
                capabilities: options.capabilities(),
                page_size: options.page_size(),
                prepared_plan_id: crate::transformer::PreparedDecoderGeneration::take()?.get(),
                forward_executor: crate::transformer::TransformerForwardExecutor::new(module),
                backend: PagedKvBackend::new(pool),
                default_state,
                sequence_lifecycle: lifecycle,
                proposal: NoProposal,
                resource_manager: NoResourceManager,
                observer: StandardDecoderObserver,
                completion_hub: ferrule_common::CompletionHub::new(),
                completion_reactors: Vec::new(),
            })
        })();
        claim.finish(result)
    }
}

/// Unknown completion is terminal for this owner, even if a later CUDA fence succeeds.
#[derive(Debug)]
pub struct HybridCudaCompletionUnknown {
    source: Error,
}
impl std::fmt::Display for HybridCudaCompletionUnknown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "hybrid CUDA completion unknown; quarantine required: {}",
            self.source
        )
    }
}
impl std::error::Error for HybridCudaCompletionUnknown {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
fn unknown(source: Error) -> Error {
    Error::ModelSource {
        source: Box::new(HybridCudaCompletionUnknown { source }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::*;
    fn schema() -> HybridStateSchema {
        let norm = RmsNorm::new(4, 1e-6).unwrap();
        let spec = DecoderModelSpec::new(DecoderModelParts {
            architecture: "state-test".into(),
            hidden_size: 4,
            vocab_size: 8,
            max_sequence_length: Some(8),
            token_embedding: Embedding::new(8, 4, None).unwrap(),
            layers: vec![
                DecoderLayer::new(
                    0,
                    norm.clone(),
                    Attention::GatedDeltaNet(
                        GatedDeltaNetAttention::new(4, 1, 2, 3, 2, 3, 1e-6, false).unwrap(),
                    ),
                    Residual::Add,
                    norm.clone(),
                    FeedForward::SwiGlu(SwiGlu::new(4, 8, false).unwrap()),
                    Residual::Add,
                )
                .unwrap(),
            ],
            final_norm: norm,
            output: Linear::new(4, 8, false).unwrap(),
            tie_word_embeddings: false,
        })
        .unwrap();
        HybridStateSchema::from_spec(&spec, 1).unwrap()
    }
    fn device() -> HybridCudaDevice {
        HybridCudaDevice::new_on_device(
            0,
            HybridCudaMemoryBudget {
                kv_pages: 8,
                state_bytes: 8192,
                weight_bytes: 8192,
                workspace_bytes: 8192,
            },
        )
        .unwrap()
    }
    fn claim(device: &HybridCudaDevice) -> Result<HybridCudaPreparationClaim> {
        device.admit(
            HybridCudaMemoryEstimate {
                per_sequence_state_bytes: state_bytes(&schema())?,
                kv_bytes: 128,
                weight_bytes_upper_bound: 128,
                workspace_bytes_upper_bound: 128,
            },
            1,
        )
    }
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn preparation_clean_state_cleanup_returns_claim() {
        let device = device();
        let preparation = claim(&device).unwrap();
        let result: Result<()> = (|| {
            let mut lifecycle = CudaHybridSequenceLifecycle::new(schema(), device.clone())?;
            let _default_state = lifecycle.create()?;
            Err(error("validation after default-state preparation"))
        })();
        assert!(preparation.finish(result).is_err());
        assert_eq!(device.live_state_bytes(), 0);
        assert!(!device.needs_quarantine());
        assert!(!device.inner.claimed.get());
        drop(claim(&device.clone()).unwrap());
        assert!(!device.inner.claimed.get());
    }
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn preparation_panic_retains_claim() {
        let device = device();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _preparation = claim(&device).unwrap();
            panic!("injected preparation unwind");
        }));
        assert!(panic.is_err());
        assert!(device.needs_quarantine());
        assert!(device.inner.claimed.get());
        assert!(claim(&device.clone()).is_err());
    }
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn preparation_allocation_unknown_retains_claim_and_state_custody() {
        let device = device();
        let preparation = claim(&device).unwrap();
        let result = {
            let mut lifecycle = CudaHybridSequenceLifecycle::new(schema(), device.clone()).unwrap();
            device.operators().failpoints().arm_allocation();
            device.inner.reject_next_fence.set(true);
            lifecycle.create()
        };
        assert!(preparation.finish(result).is_err());
        assert!(device.needs_quarantine());
        assert!(device.inner.claimed.get());
        assert!(
            device.live_state_bytes() > 0,
            "unknown state remains charged"
        );
        assert!(claim(&device.clone()).is_err());
        // Even an independently successful later fence cannot discharge unknown custody.
        device.operators().sync_stream().unwrap();
        device.operators().sync_upload_stream().unwrap();
        assert!(claim(&device.clone()).is_err());
    }
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn preparation_rollback_fence_unknown_permanently_retains_claim() {
        let device = device();
        let preparation = claim(&device).unwrap();
        device.inner.reject_next_fence.set(true);
        assert!(
            preparation
                .finish::<()>(Err(error("clean validation failure")))
                .is_err()
        );
        assert!(device.needs_quarantine());
        assert!(device.inner.claimed.get());
        assert!(claim(&device.clone()).is_err());
    }
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn preparation_drop_returns_only_proven_clean_claim() {
        let device = device();
        drop(claim(&device).unwrap());
        assert!(!device.inner.claimed.get());
        let preparation = claim(&device.clone()).unwrap();
        // Simulate a lower-level destructor retaining custody without an error return.
        let custody = Rc::clone(device.operators());
        assert!(
            preparation
                .finish::<()>(Err(error("retained allocation")))
                .is_err()
        );
        assert!(device.needs_quarantine());
        assert!(device.inner.claimed.get());
        drop(custody);
        assert!(claim(&device).is_err());
    }
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn unknown_fence_retains_state_and_budget_and_permanently_blocks_reuse() {
        let device = device();
        let mut lifecycle = CudaHybridSequenceLifecycle::new(schema(), device.clone()).unwrap();
        let state = lifecycle.create().unwrap();
        let topology = state.topology_id();
        let bytes = device.live_state_bytes();
        device.inner.reject_next_fence.set(true);
        let error = lifecycle.try_release(state).unwrap_err();
        let (source, state) = error.into_parts();
        let Error::ModelSource { source } = source else {
            panic!("expected typed unknown completion")
        };
        assert!(
            source
                .downcast_ref::<HybridCudaCompletionUnknown>()
                .is_some()
        );
        assert_eq!(state.topology_id(), topology);
        assert!(device.needs_quarantine());
        assert!(lifecycle.create().is_err());
        assert!(lifecycle.logical_fork(&state, 0).is_err());
        assert!(lifecycle.try_release(state).is_err());
        assert_eq!(
            device.live_state_bytes(),
            bytes,
            "unknown device memory must remain charged"
        );
    }
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn checkout_allocation_failure_preserves_source_and_returns_budget() {
        let device = device();
        let mut lifecycle = CudaHybridSequenceLifecycle::new(schema(), device.clone()).unwrap();
        let state = lifecycle.create().unwrap();
        let bytes = device.live_state_bytes();
        device.operators().failpoints().arm_allocation();
        let request = DecoderSequenceCheckout::new(
            ferrule_common::execution::ExecutionTransactionId::new(1).unwrap(),
            0,
            0,
        );
        assert!(lifecycle.checkout(request, &state).is_err());
        assert!(!device.needs_quarantine());
        assert_eq!(device.live_state_bytes(), bytes);
        let copy = lifecycle.checkout(request, &state).unwrap();
        assert_eq!(copy.topology_id(), state.topology_id());
        lifecycle.try_release(copy).unwrap();
        lifecycle.try_release(state).unwrap();
        assert_eq!(device.live_state_bytes(), 0);
    }
}
