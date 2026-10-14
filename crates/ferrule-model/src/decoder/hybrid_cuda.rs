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
    fn drain(&mut self) -> Result<WorkProgress>;
    /// Close external owners after drain; the same completion contract applies.
    fn shutdown(&mut self) -> Result<WorkProgress>;
    /// Notify transport cancellation after a forward failure; does not release custody.
    fn on_error(&mut self, _transaction: ferrule_common::execution::ExecutionTransactionId) {}
}

/// Optional external routed-only execution. Router/shared experts remain on root.
/// Construct on the root owner thread; use `HybridCudaDevice::operators()` for
/// the existing CUDA result-combine adapter, not another same-ordinal context.
///
/// The same attachment is consumed by `CudaHybridModule` or
/// [`crate::transformer::CudaHybridDecoderSegment::prepare_with_routed_experts`].
/// A segment uses the caller's packed-sequence session IDs and its global layer
/// range; attachment construction is not child readiness or memory admission.
/// Runtime must establish both before preparing/executing the segment and keep
/// its existing journal alive until segment quiescence proves completion.
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
            matches!(executor.drain(), Ok(WorkProgress::Complete))
                && matches!(executor.shutdown(), Ok(WorkProgress::Complete))
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

/// Immutable Mixed program identity carried by the existing preparation claim.
/// Construct with `HybridMixedRankProgramPlan::cuda_claim_inputs`; metadata is
/// not authority to allocate, complete CUDA work, or publish an image.
#[derive(Debug, Clone)]
pub struct HybridCudaPreparationContext {
    pub(crate) program: crate::transformer::hybrid_parallel::HybridMixedRankProgramPlan,
    pub(crate) parameters: Vec<crate::transformer::BoundParameter>,
    pub(crate) device_ordinal: usize,
    pub(crate) physical_pages: usize,
    pub(crate) max_parameter_bytes: u64,
    pub(crate) estimate: HybridCudaMemoryEstimate,
    pub(crate) projection_limits: [usize; 3],
    pub(crate) auxiliary_grants: usize,
    pub(crate) attention_shapes:
        Vec<ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape>,
    pub(crate) attention_limits: [usize; 2],
}
impl PartialEq for HybridCudaPreparationContext {
    fn eq(&self, other: &Self) -> bool {
        self.program.layout() == other.program.layout()
            && self.program.segment() == other.program.segment()
            && self.program.collective_binding() == other.program.collective_binding()
            && self.program.page_size() == other.program.page_size()
            && self.program.max_positions() == other.program.max_positions()
            && self.program.state_copies() == other.program.state_copies()
            && self.program.decoded_cache == other.program.decoded_cache
            && self.device_ordinal == other.device_ordinal
            && self.physical_pages == other.physical_pages
            && self.max_parameter_bytes == other.max_parameter_bytes
            && self.estimate == other.estimate
            && self.projection_limits == other.projection_limits
            && self.auxiliary_grants == other.auxiliary_grants
            && self.attention_shapes == other.attention_shapes
            && self.attention_limits == other.attention_limits
            && self.parameters.len() == other.parameters.len()
            && self.parameters.iter().zip(&other.parameters).all(|(a, b)| {
                a.shares_storage_with(b) && a.spec() == b.spec() && a.role() == b.role()
            })
    }
}
impl HybridCudaPreparationContext {
    pub(crate) fn validate_for(&self, required: &Self) -> Result<()> {
        if self != required {
            return Err(error(
                "Mixed rank claim context differs from compiled program",
            ));
        }
        Ok(())
    }

    /// Pure preflight; does not construct or query a CUDA owner.
    pub fn validate_device_budget(
        &self,
        ordinal: usize,
        budget: HybridCudaMemoryBudget,
    ) -> Result<()> {
        let state = self
            .estimate
            .per_sequence_state_bytes
            .checked_mul(self.program.state_copies())
            .ok_or_else(|| error("Mixed claim state capacity overflow"))?;
        if ordinal != self.device_ordinal
            || budget.kv_pages != self.physical_pages
            || budget.weight_bytes < self.estimate.weight_bytes_upper_bound
            || budget.workspace_bytes < self.estimate.workspace_bytes_upper_bound
            || budget.state_bytes < state
        {
            return Err(error(
                "Mixed preparation device/configuration budget mismatch",
            ));
        }
        Ok(())
    }

    /// Exact comparison used by the opaque claim before its first image phase.
    pub fn validate_estimate(&self, estimate: HybridCudaMemoryEstimate) -> Result<()> {
        if estimate != self.estimate {
            return Err(error(
                "Mixed preparation estimate differs from sealed context",
            ));
        }
        Ok(())
    }
}

/// CUDA image estimate, including native BF16's live provider validation.
/// Use [`HybridMemoryPlan`] for pure F32-state and numeric FP8 planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridCudaMemoryEstimate {
    pub per_sequence_state_bytes: usize,
    pub kv_bytes: usize,
    pub weight_bytes_upper_bound: usize,
    pub workspace_bytes_upper_bound: usize,
}
impl From<HybridMemoryPlan> for HybridCudaMemoryEstimate {
    fn from(plan: HybridMemoryPlan) -> Self {
        Self {
            per_sequence_state_bytes: plan.per_sequence_state_bytes,
            kv_bytes: plan.kv_bytes,
            weight_bytes_upper_bound: plan.weight_bytes_upper_bound,
            workspace_bytes_upper_bound: plan.workspace_bytes_upper_bound,
        }
    }
}
impl HybridCudaMemoryEstimate {
    pub fn for_resources(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
    ) -> Result<Self> {
        if options.precision() == crate::execution::ExecutionPrecisionPolicy::bf16_compatibility() {
            return Self::bf16_resources(resources, options, kv_pages, false);
        }
        HybridMemoryPlan::for_resources(resources, options, kv_pages).map(Into::into)
    }
    pub fn for_resources_with_routed_experts(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
    ) -> Result<Self> {
        if options.precision() == crate::execution::ExecutionPrecisionPolicy::bf16_compatibility() {
            return Self::bf16_resources(resources, options, kv_pages, true);
        }
        HybridMemoryPlan::for_resources_with_routed_experts(resources, options, kv_pages)
            .map(Into::into)
    }
    pub fn for_resources_with_idle_kv_reclaim(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
        config: IdleKvReclaimConfig,
    ) -> Result<Self> {
        HybridMemoryPlan::for_resources_with_idle_kv_reclaim(resources, options, kv_pages, config)
            .map(Into::into)
    }
    pub fn for_segment(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        plan: &crate::transformer::LayerSegmentPlan,
        kv_pages: usize,
    ) -> Result<Self> {
        HybridMemoryPlan::for_segment(resources, options, plan, kv_pages).map(Into::into)
    }
}
impl GenericDecoderOptions {
    pub(crate) fn validate_hybrid_cuda(&self, resources: &BoundDecoderResources) -> Result<()> {
        if self.precision() == crate::execution::ExecutionPrecisionPolicy::bf16_compatibility() {
            self.validate_bf16_hybrid(resources)
        } else {
            self.validate_hybrid_cuda_profile(resources)
        }
    }
}
#[path = "hybrid_bf16_admission.rs"]
mod bf16_admission;
#[path = "hybrid_bf16_state.rs"]
mod bf16_state;
pub use bf16_state::CudaBf16GatedDeltaState;

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
            Err(CompletionUnknown::wrap(error(
                "hybrid CUDA owner is quarantined",
            )))
        } else {
            Ok(())
        }
    }
    pub(crate) fn quarantine(&self) {
        self.inner.poisoned.set(true);
    }
    /// Reserve the complete CUDA image budget. The returned opaque token must
    /// be kept by the factory until image, pool and initial lifecycle creation
    /// all succeed, then consumed by `HybridCudaPreparationClaim::finish`.
    pub fn admit(
        &self,
        estimate: HybridCudaMemoryEstimate,
        max_sequences: usize,
    ) -> Result<HybridCudaPreparationClaim> {
        self.admit_inner(estimate, max_sequences, None)
    }

    /// Mixed rank admission with the compiled program's non-memory context.
    /// This is the same exclusive claim type used by the F32 rank path; the
    /// context only prevents a claim for one rows/pages/workspace geometry from
    /// being consumed by another compiled image.
    pub fn admit_with_context(
        &self,
        estimate: HybridCudaMemoryEstimate,
        context: HybridCudaPreparationContext,
    ) -> Result<HybridCudaPreparationClaim> {
        context.validate_estimate(estimate)?;
        context.validate_device_budget(self.device_ordinal(), self.budget())?;
        self.admit_inner(
            estimate,
            context.program.collective_binding().max_sequences,
            Some(context),
        )
    }

    fn admit_inner(
        &self,
        estimate: HybridCudaMemoryEstimate,
        max_sequences: usize,
        context: Option<HybridCudaPreparationContext>,
    ) -> Result<HybridCudaPreparationClaim> {
        self.ensure_ready()?;
        if self.inner.claimed.get() || self.live_state_bytes() != 0 {
            return Err(error("hybrid device already owns an image or live state"));
        }
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
            image: ImagePreparation::new(),
            source_retirement: MixedSourceRetirement::default(),
            estimate,
            max_sequences,
            device_owners: Rc::strong_count(&self.inner),
            operator_owners: Rc::strong_count(self.operators()),
            context,
        })
    }
    pub(crate) fn fence(&self) -> Result<()> {
        self.ensure_ready()?;
        #[cfg(test)]
        if self.inner.reject_next_fence.replace(false) {
            self.quarantine();
            return Err(CompletionUnknown::wrap(error(
                "injected missing CUDA completion evidence",
            )));
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
            return Err(CompletionUnknown::wrap(
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImagePreparationPhase {
    Admitted,
    Preparing,
    Prepared,
}
/// Owner-local completion observation. Only finish(Ok) after the final fence
/// grants retirement; this never carries or mutates a shared payload.
#[derive(Clone, Default)]
pub(crate) struct MixedSourceRetirement(Rc<Cell<bool>>);
impl MixedSourceRetirement {
    pub(crate) fn is_complete(&self) -> bool {
        self.0.get()
    }

    #[cfg(test)]
    pub(crate) fn completed_for_test() -> Self {
        Self(Rc::new(Cell::new(true)))
    }
}

struct ImagePreparation(Cell<ImagePreparationPhase>);
impl ImagePreparation {
    fn new() -> Self {
        Self(Cell::new(ImagePreparationPhase::Admitted))
    }
    fn begin(&self) -> Result<()> {
        if self.0.get() != ImagePreparationPhase::Admitted {
            return Err(error("CUDA claim already used for an image"));
        }
        self.0.set(ImagePreparationPhase::Preparing);
        Ok(())
    }
    fn prepared(&self) -> Result<()> {
        if self.0.get() != ImagePreparationPhase::Preparing {
            return Err(error("CUDA image preparation not in progress"));
        }
        self.0.set(ImagePreparationPhase::Prepared);
        Ok(())
    }
    fn is_prepared(&self) -> bool {
        self.0.get() == ImagePreparationPhase::Prepared
    }
}

fn validate_claim_estimate(
    admitted: HybridCudaMemoryEstimate,
    admitted_sequences: usize,
    required: HybridCudaMemoryEstimate,
    sequences: usize,
) -> Result<()> {
    if sequences > admitted_sequences
        || required.per_sequence_state_bytes > admitted.per_sequence_state_bytes
        || required.kv_bytes > admitted.kv_bytes
        || required.weight_bytes_upper_bound > admitted.weight_bytes_upper_bound
        || required.workspace_bytes_upper_bound > admitted.workspace_bytes_upper_bound
    {
        return Err(error("CUDA image exceeds its preparation claim"));
    }
    Ok(())
}

/// Failed preparation returns its claim only after temporary owners have
/// dropped and completion is proven. A successful image consumes the factory:
/// clones, including surviving states, never gain a second image authority.
/// Opaque exclusive image ownership token. Cloning a device never clones
/// this token; dropping it only returns a clean, proven reservation.
/// ```compile_fail
/// use ferrule_model::decoder::HybridCudaPreparationClaim;
/// fn clone_claim(claim: &HybridCudaPreparationClaim) -> HybridCudaPreparationClaim {
///     claim.clone()
/// }
/// ```
/// ```compile_fail
/// use ferrule_model::decoder::HybridCudaPreparationClaim;
/// fn forge_success(claim: &HybridCudaPreparationClaim) {
///     claim.image_prepared().unwrap();
/// }
/// ```
#[must_use = "finish the factory result or drop the token to prove rollback"]
pub struct HybridCudaPreparationClaim {
    device: HybridCudaDevice,
    committed: bool,
    image: ImagePreparation,
    source_retirement: MixedSourceRetirement,
    estimate: HybridCudaMemoryEstimate,
    max_sequences: usize,
    device_owners: usize,
    operator_owners: usize,
    context: Option<HybridCudaPreparationContext>,
}
impl HybridCudaPreparationClaim {
    pub(crate) fn mixed_source_retirement(&self) -> MixedSourceRetirement {
        self.source_retirement.clone()
    }

    /// Begin exactly one image on the admitted owner. Only model preparation
    /// entry points can advance this token; factory code cannot forge success.
    pub(crate) fn begin_image(&self, device: &HybridCudaDevice) -> Result<()> {
        self.validate_device(device)?;
        self.image.begin()
    }

    pub(crate) fn validate_device(&self, device: &HybridCudaDevice) -> Result<()> {
        if !Rc::ptr_eq(&self.device.inner, &device.inner) {
            return Err(error("CUDA preparation claim belongs to another device"));
        }
        device.ensure_ready()
    }

    pub(crate) fn validate_estimate(
        &self,
        required: HybridCudaMemoryEstimate,
        sequences: usize,
    ) -> Result<()> {
        validate_claim_estimate(self.estimate, self.max_sequences, required, sequences)
    }

    pub(crate) fn validate_exact_mixed(
        &self,
        required: HybridCudaMemoryEstimate,
        context: HybridCudaPreparationContext,
    ) -> Result<()> {
        if self.estimate != required
            || self.max_sequences != context.program.collective_binding().max_sequences
        {
            return Err(error(
                "Mixed rank claim estimate differs from compiled program",
            ));
        }
        self.context
            .as_ref()
            .ok_or_else(|| error("Mixed preparation requires a source-bound claim context"))?
            .validate_for(&context)
    }

    pub(crate) fn validate_image_grants(&self, actual: u64) -> Result<()> {
        let limit = self
            .estimate
            .weight_bytes_upper_bound
            .checked_add(self.estimate.workspace_bytes_upper_bound)
            .ok_or_else(|| error("CUDA image grant overflow"))?;
        if actual > limit as u64 {
            return Err(error("CUDA image actual grants exceed preparation claim"));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn image_is_prepared(&self) -> bool {
        self.image.is_prepared()
    }

    pub(crate) fn image_prepared(&self) -> Result<()> {
        self.device.fence()?;
        self.image.prepared()
    }

    /// Consume the exclusive token after image, pool and initial states have
    /// been built. `Ok` is accepted only after a real model preparation marked
    /// the image ready and a final completion proof succeeds. Temporary owners
    /// must be dropped before passing `Err`; unknown custody is never refunded.
    pub fn finish<T>(mut self, result: Result<T>) -> Result<T> {
        let result: Result<()> = match result {
            Ok(value) => {
                // Even a malformed caller result may own submitted buffers.
                // Prove completion before dropping it for missing image authority.
                if let Err(error) = self.device.fence() {
                    std::mem::forget(value);
                    std::mem::forget(self.device.clone());
                    self.committed = true;
                    return Err(error);
                }
                if self.image.is_prepared() {
                    self.committed = true;
                    self.source_retirement.0.set(true);
                    return Ok(value);
                }
                drop(value);
                Err(error("CUDA preparation has no successfully prepared image"))
            }
            Err(source) => Err(source),
        };
        let cleanup = self.rollback_after_failure();
        Err(Error::with_cleanup(
            "hybrid CUDA preparation",
            result.unwrap_err(),
            cleanup,
        ))
    }

    /// Consume a failed preparation without changing the caller's error type.
    /// Shares exactly the same rollback path as finish(Err(...)).
    pub(crate) fn rollback_after_failure(mut self) -> Result<()> {
        let cleanup = self.rollback();
        self.committed = true;
        cleanup
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

/// Fallible lifecycle checkout/fork, never an implicit CUDA buffer clone.
#[doc = include_str!("../../tests/compile/cuda_hybrid_state.md")]
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
    bf16_layers: Option<Vec<Option<CudaBf16GatedDeltaState>>>,
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
        if self.bf16_layers.is_some() {
            return self.validate_bf16(schema, position);
        }
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
            std::mem::forget(self.bf16_layers.take());
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

    fn linear_state(&mut self, layer: usize) -> Result<GatedDeltaStateRef<'_>> {
        self.kv_state().device.ensure_ready()?;
        let slot = self
            .kv_state()
            .schema
            .entry(layer)
            .ok_or_else(|| error("global layer outside CUDA hybrid schema"))?
            .state_slot;
        if self.kv_state().bf16_layers.is_some() {
            return match self
                .kv_state_mut()
                .bf16_layers
                .as_mut()
                .unwrap()
                .get_mut(slot)
            {
                Some(Some(state)) => Ok(GatedDeltaStateRef::CudaBf16(state)),
                _ => Err(error("missing BF16 CUDA linear state")),
            };
        }
        match self.kv_state_mut().layers.get_mut(slot) {
            Some(Some(state)) => Ok(GatedDeltaStateRef::Cuda(state)),
            _ => Err(error("missing CUDA linear state")),
        }
    }
}

pub struct CudaHybridSequenceLifecycle {
    schema: HybridStateSchema,
    device: HybridCudaDevice,
    bf16: bool,
}
impl CudaHybridSequenceLifecycle {
    pub fn new(schema: HybridStateSchema, device: HybridCudaDevice) -> Result<Self> {
        state_bytes(&schema)?;
        Ok(Self {
            schema,
            device,
            bf16: false,
        })
    }
    fn allocate(&self) -> Result<CudaHybridLayerStates> {
        if self.bf16 {
            return self.allocate_bf16();
        }
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
            bf16_layers: None,
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
        if !Rc::ptr_eq(&self.device.inner, &source.kv_state().device.inner)
            || self.bf16 != source.kv_state().bf16_layers.is_some()
        {
            return Err(error("foreign hybrid CUDA lifecycle"));
        }
        source.validate_standard_state(&self.schema)
    }
    fn copy(&self, source: &CudaHybridSequenceState) -> Result<CudaHybridLayerStates> {
        self.validate(source)?;
        source.core().begin_step()?;
        self.device.fence()?;
        let mut destination = self.allocate()?;
        if self.bf16 {
            self.copy_bf16(source, &mut destination)?;
            return Ok(destination);
        }
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
            || self.bf16 != state.kv_state().bf16_layers.is_some()
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
        Self::hybrid_cuda_inner(resources, tokenizer, options, device, None, None)
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
        Self::hybrid_cuda_inner(resources, tokenizer, options, device, Some(experts), None)
    }

    /// Opt-in idle/suspended dense F32 GQA reclamation, not HostKV decode.
    /// Uses the same runner/Entry journal and restores before any forward.
    pub fn hybrid_cuda_with_idle_kv_reclaim(
        resources: BoundDecoderResources,
        tokenizer: crate::tokenizer::TokenizerHandle,
        options: GenericDecoderOptions,
        device: HybridCudaDevice,
        config: IdleKvReclaimConfig,
    ) -> Result<Self> {
        Self::hybrid_cuda_inner(resources, tokenizer, options, device, None, Some(config))
    }

    fn hybrid_cuda_inner(
        resources: BoundDecoderResources,
        tokenizer: crate::tokenizer::TokenizerHandle,
        options: GenericDecoderOptions,
        device: HybridCudaDevice,
        mut experts: Option<HybridCudaRoutedExperts>,
        idle_reclaim: Option<IdleKvReclaimConfig>,
    ) -> Result<Self> {
        if let Some(experts) = &mut experts {
            experts.owner = Some(device.clone());
        }
        options.validate_hybrid_cuda(&resources)?;
        let bf16 =
            options.precision() == crate::execution::ExecutionPrecisionPolicy::bf16_compatibility();
        if bf16 && (experts.is_some() || idle_reclaim.is_some()) {
            return Err(unsupported(
                "native BF16 hybrid requires local resident experts/KV; no HostKV or external adapter",
            ));
        }
        let schema = HybridStateSchema::from_spec(
            resources.spec(),
            options
                .active_layers()
                .unwrap_or(resources.spec().layers().len()),
        )?;
        let mut estimate = if experts.is_some() {
            HybridCudaMemoryEstimate::for_resources_with_routed_experts
        } else {
            HybridCudaMemoryEstimate::for_resources
        }(&resources, &options, device.budget().kv_pages)?;
        if let Some(config) = idle_reclaim {
            estimate = HybridCudaMemoryEstimate::for_resources_with_idle_kv_reclaim(
                &resources,
                &options,
                device.budget().kv_pages,
                config,
            )?;
        }
        let claim = device.admit(estimate, options.capabilities().max_sequences)?;
        // Drop all temporaries before evaluating claim rollback.
        let result = (|| {
            let planes = schema.kv_planes_with_dtype(
                options.page_size(),
                options.max_positions(),
                if bf16 {
                    ferrule_common::execution::KvElementType::Bf16
                } else {
                    ferrule_common::execution::KvElementType::F32
                },
            )?;
            let mut model_info = super::runner::standard_model_info(&resources, &options);
            model_info.backend = options
                .hybrid_cuda_numeric_fp8_precision()
                .map_or("cuda-hybrid-f32", |precision| {
                    precision.hybrid_backend_name()
                });
            if bf16 && options.hybrid_cuda_numeric_fp8().is_none() {
                model_info.backend = "cuda-hybrid-native-bf16-f32-ssm";
            }
            let resources = std::sync::Arc::new(resources);
            let module = crate::transformer::CudaHybridModule::prepare_with_claim(
                resources.clone(),
                device.clone(),
                &options,
                experts,
                &claim,
            )?;
            let pool = device.complete(match idle_reclaim {
                Some(config) => TypedCudaPagedKvPool::<CudaHybridSequenceState>::from_strategy_with_idle_reclaim(
                    Rc::clone(device.operators()), &planes, device.budget().kv_pages, config),
                None => TypedCudaPagedKvPool::<CudaHybridSequenceState>::from_strategy(
                    Rc::clone(device.operators()), &planes, device.budget().kv_pages),
            })?;
            let mut lifecycle = CudaHybridSequenceLifecycle::with_precision(
                schema,
                device.clone(),
                options.precision(),
            )?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::*;
    pub(super) fn schema() -> HybridStateSchema {
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
    pub(super) fn device() -> HybridCudaDevice {
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
    pub(super) fn claim(device: &HybridCudaDevice) -> Result<HybridCudaPreparationClaim> {
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
        assert!(source.downcast_ref::<CompletionUnknown>().is_some());
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

#[cfg(test)]
#[path = "../../tests/unit/cuda_preparation_claim/mod.rs"]
mod preparation_claim_tests;
