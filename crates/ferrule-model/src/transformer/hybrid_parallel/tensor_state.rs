//! Owner binding around the existing hybrid state representation and lifecycle.
use super::{HybridRankLayout, invalid};
use crate::decoder::*;
use crate::runner::SequenceStateReleaseError;
use ferrule_common::Result;
use std::{collections::BTreeMap, sync::Arc};

pub type HybridTensorSequenceState = DecoderSequenceState<(), HybridTensorLayerStates>;
#[derive(Debug, Clone, PartialEq)]
pub struct HybridTensorLayerStates {
    layout: Arc<HybridRankLayout>,
    inner: HybridLayerStates,
}
impl HybridTensorLayerStates {
    pub fn lane(&self) -> usize {
        self.layout.lane()
    }
    pub fn inner(&self) -> &HybridLayerStates {
        &self.inner
    }
    /// Observation only; execution never clones recurrent payloads here.
    pub fn recurrent(&self) -> BTreeMap<usize, GatedDeltaNetState> {
        self.layout
            .state_schema()
            .entries()
            .iter()
            .zip(self.inner.layers())
            .filter_map(|(entry, state)| match state {
                HybridLayerState::GatedDeltaNet(s) => Some((entry.global_layer, s.clone())),
                _ => None,
            })
            .collect()
    }
}
impl StandardSequenceState for HybridTensorSequenceState {
    fn validate_standard_schema(_: &HybridStateSchema) -> Result<()> {
        Ok(())
    }
    fn validate_standard_state_at(
        &self,
        schema: &HybridStateSchema,
        position: usize,
    ) -> Result<()> {
        self.kv_state().inner.validate(schema, position)
    }
    fn linear_state(&mut self, layer: usize) -> Result<GatedDeltaStateRef<'_>> {
        self.kv_state_mut()
            .inner
            .linear_mut(layer)
            .map(GatedDeltaStateRef::Cpu)
    }
}
#[derive(Debug, Clone)]
pub struct HybridTensorSequenceLifecycle {
    pub(super) layout: Arc<HybridRankLayout>,
}
impl HybridTensorSequenceLifecycle {
    pub fn layout(&self) -> &HybridRankLayout {
        &self.layout
    }
    pub fn kv_planes(&self, page_size: usize, max_positions: usize) -> Result<StandardGqaPlanes> {
        self.layout.kv_planes(page_size, max_positions)
    }
    pub(super) fn validate_owner(&self, state: &HybridTensorSequenceState) -> Result<()> {
        if state.kv_state().layout != self.layout {
            return Err(invalid("state belongs to another TP owner/scope/layout"));
        }
        Ok(())
    }
}
impl DecoderSequenceLifecycle<HybridTensorSequenceState> for HybridTensorSequenceLifecycle {
    fn create(&mut self) -> Result<HybridTensorSequenceState> {
        Ok(DecoderSequenceState::new(
            (),
            HybridTensorLayerStates {
                layout: Arc::clone(&self.layout),
                inner: self.layout.state_schema().create()?,
            },
        ))
    }
    fn checkout(
        &mut self,
        _: DecoderSequenceCheckout,
        source: &HybridTensorSequenceState,
    ) -> Result<HybridTensorSequenceState> {
        self.validate_owner(source)?;
        source.validate_standard_state(self.layout.state_schema())?;
        source.transaction_working_copy()
    }
    fn logical_fork(
        &mut self,
        source: &HybridTensorSequenceState,
        position: usize,
    ) -> Result<HybridTensorSequenceState> {
        self.validate_owner(source)?;
        if source.core().position() != position {
            return Err(invalid("TP GDN fork requires exact frontier"));
        }
        source.validate_standard_state(self.layout.state_schema())?;
        source.logical_fork()
    }
    fn reset(&mut self, state: &mut HybridTensorSequenceState) -> Result<()> {
        self.validate_owner(state)?;
        state.validate_standard_state(self.layout.state_schema())?;
        state.kv_state_mut().inner = self.layout.state_schema().create()?;
        state.core_mut().reset();
        Ok(())
    }
    fn try_release(
        &mut self,
        state: HybridTensorSequenceState,
    ) -> std::result::Result<(), SequenceStateReleaseError<HybridTensorSequenceState>> {
        // Failed work may be between layer frontiers; release checks ownership only.
        if let Err(error) = self.validate_owner(&state) {
            return Err(SequenceStateReleaseError::new(error, state));
        }
        Ok(())
    }
}
