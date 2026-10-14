//! Physical state credits follow journal-owned objects, not lifecycle calls.
//! CUDA storage, copies and completion are delegated to the existing lifecycle.
use super::*;
use crate::SequenceStateReleaseError;
use crate::decoder::{
    CudaHybridLayerStates, DecoderSequence, DecoderSequenceCheckout, DecoderSequenceLifecycle,
    GatedDeltaStateRef, StandardSequenceState,
};
use crate::execution::{SequenceStateCore, SequenceTopologyId};
use std::cell::Cell;

#[derive(Debug)]
pub(super) struct StateCredits {
    live: Cell<usize>,
    limit: usize,
    per_state: usize,
    objects: Cell<usize>,
    object_limit: Option<usize>,
}
impl StateCredits {
    pub fn new(limit: usize, per_state: usize) -> Rc<Self> {
        Rc::new(Self {
            live: Cell::new(0),
            limit,
            per_state,
            objects: Cell::new(0),
            object_limit: None,
        })
    }
    pub fn bounded(limit: usize, per_state: usize, copies: usize) -> Rc<Self> {
        Rc::new(Self {
            live: Cell::new(0),
            limit,
            per_state,
            objects: Cell::new(0),
            object_limit: Some(copies),
        })
    }
    fn reserve(&self) -> Result<()> {
        let objects = self
            .objects
            .get()
            .checked_add(1)
            .filter(|n| self.object_limit.is_none_or(|limit| *n <= limit))
            .ok_or_else(|| cuda_error("rank physical state object slots exhausted"))?;
        let next = self
            .live
            .get()
            .checked_add(self.per_state)
            .filter(|&n| n <= self.limit)
            .ok_or_else(|| cuda_error("rank physical state grants exhausted"))?;
        self.live.set(next);
        self.objects.set(objects);
        Ok(())
    }
}
#[derive(Debug)]
struct StateCredit {
    ledger: Rc<StateCredits>,
    device: HybridCudaDevice,
}
impl StateCredit {
    fn acquire(ledger: &Rc<StateCredits>, device: &HybridCudaDevice) -> Result<Self> {
        device.ensure_ready()?;
        ledger.reserve()?;
        Ok(Self {
            ledger: Rc::clone(ledger),
            device: device.clone(),
        })
    }
}
impl Drop for StateCredit {
    fn drop(&mut self) {
        if !self.device.needs_quarantine() && !std::thread::panicking() {
            self.ledger.objects.set(self.ledger.objects.get() - 1);
            self.ledger
                .live
                .set(self.ledger.live.get() - self.ledger.per_state);
        }
    }
}
/// Non-Clone journal state, bound to one prepared rank image by its private credit.
#[derive(Debug)]
pub struct CudaHybridRankState {
    // Drop order is intentional: state completion precedes credit reclamation.
    state: CudaHybridSequenceState,
    credit: StateCredit,
}
impl CudaHybridRankState {
    pub fn kv_state(&self) -> &CudaHybridLayerStates {
        self.state.kv_state()
    }
    pub(super) fn validate_rank(&self, credits: &Rc<StateCredits>) -> Result<()> {
        if !Rc::ptr_eq(&self.credit.ledger, credits) {
            return Err(cuda_error("foreign CUDA rank state image"));
        }
        self.credit.device.ensure_ready()
    }
}
impl DecoderSequence for CudaHybridRankState {
    fn topology_id(&self) -> SequenceTopologyId {
        self.state.topology_id()
    }
    fn core(&self) -> &SequenceStateCore {
        self.state.core()
    }
    fn core_mut(&mut self) -> &mut SequenceStateCore {
        self.state.core_mut()
    }
}
impl StandardSequenceState for CudaHybridRankState {
    fn validate_standard_schema(schema: &HybridStateSchema) -> Result<()> {
        CudaHybridSequenceState::validate_standard_schema(schema)
    }
    fn validate_standard_state_at(
        &self,
        schema: &HybridStateSchema,
        position: usize,
    ) -> Result<()> {
        self.state.validate_standard_state_at(schema, position)
    }

    fn linear_state(&mut self, layer: usize) -> Result<GatedDeltaStateRef<'_>> {
        self.state.linear_state(layer)
    }
}
pub struct CudaHybridRankLifecycle {
    inner: CudaHybridSequenceLifecycle,
    device: HybridCudaDevice,
    credits: Rc<StateCredits>,
}
impl CudaHybridRankLifecycle {
    pub(super) fn new(
        schema: HybridStateSchema,
        device: HybridCudaDevice,
        credits: Rc<StateCredits>,
    ) -> Result<Self> {
        Self::with_precision(schema, device, credits, ExecutionPrecisionPolicy::f32())
    }
    pub(super) fn with_precision(
        schema: HybridStateSchema,
        device: HybridCudaDevice,
        credits: Rc<StateCredits>,
        precision: ExecutionPrecisionPolicy,
    ) -> Result<Self> {
        device.ensure_ready()?;
        Ok(Self {
            inner: CudaHybridSequenceLifecycle::with_precision(schema, device.clone(), precision)?,
            device,
            credits,
        })
    }
    pub fn live_state_grants(&self) -> usize {
        self.credits.live.get()
    }
}
impl DecoderSequenceLifecycle<CudaHybridRankState> for CudaHybridRankLifecycle {
    fn create(&mut self) -> Result<CudaHybridRankState> {
        let credit = StateCredit::acquire(&self.credits, &self.device)?;
        let state = self.inner.create()?;
        Ok(CudaHybridRankState { state, credit })
    }
    fn checkout(
        &mut self,
        request: DecoderSequenceCheckout,
        source: &CudaHybridRankState,
    ) -> Result<CudaHybridRankState> {
        source.validate_rank(&self.credits)?;
        let credit = StateCredit::acquire(&self.credits, &self.device)?;
        let state = self.inner.checkout(request, &source.state)?;
        Ok(CudaHybridRankState { state, credit })
    }
    fn logical_fork(
        &mut self,
        source: &CudaHybridRankState,
        position: usize,
    ) -> Result<CudaHybridRankState> {
        source.validate_rank(&self.credits)?;
        let credit = StateCredit::acquire(&self.credits, &self.device)?;
        let state = self.inner.logical_fork(&source.state, position)?;
        Ok(CudaHybridRankState { state, credit })
    }
    fn reset(&mut self, state: &mut CudaHybridRankState) -> Result<()> {
        state.validate_rank(&self.credits)?;
        // Reset allocates before retiring the original; reserve both copies.
        let _replacement = StateCredit::acquire(&self.credits, &self.device)?;
        self.inner.reset(&mut state.state)
    }
    fn try_release(
        &mut self,
        state: CudaHybridRankState,
    ) -> std::result::Result<(), SequenceStateReleaseError<CudaHybridRankState>> {
        if let Err(error) = state.validate_rank(&self.credits) {
            return Err(SequenceStateReleaseError::new(error, state));
        }
        let CudaHybridRankState { state, credit } = state;
        match self.inner.try_release(state) {
            Ok(()) => {
                drop(credit);
                Ok(())
            }
            Err(error) => {
                let (error, state) = error.into_parts();
                Err(SequenceStateReleaseError::new(
                    error,
                    CudaHybridRankState { state, credit },
                ))
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_state_credits_share_limit_and_reject_overflow() {
        let a = StateCredits::new(1536, 512);
        let b = Rc::clone(&a);
        a.reserve().unwrap();
        b.reserve().unwrap();
        a.reserve().unwrap();
        assert_eq!(b.live.get(), 1536);
        assert!(b.reserve().is_err());
        assert_eq!(a.live.get(), 1536);
        assert!(StateCredits::new(511, 512).reserve().is_err());
        let overflow = StateCredits::new(usize::MAX, 512);
        overflow.live.set(usize::MAX - 511);
        assert!(overflow.reserve().is_err());
    }
}

#[cfg(test)]
mod mixed_state_slots {
    use super::*;
    #[test]
    fn bf16_gqa_zero_byte_states_still_consume_bounded_object_slots() {
        let credits = StateCredits::bounded(0, 0, 3);
        for _ in 0..3 {
            credits.reserve().unwrap();
        }
        assert_eq!(credits.live.get(), 0);
        assert_eq!(credits.objects.get(), 3);
        assert!(credits.reserve().is_err());
        // F32's legacy byte-only admission is deliberately unchanged.
        let legacy = StateCredits::new(0, 0);
        for _ in 0..4 {
            legacy.reserve().unwrap();
        }
    }
}
