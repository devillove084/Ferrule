//! Per-layer state for standard paged GQA / gated-delta decoders.
use super::{
    DecoderSequenceCheckout, DecoderSequenceLifecycle, DecoderSequenceState, StandardGqaPlanes,
};
use crate::runner::SequenceStateReleaseError;
use crate::transformer::{Attention, DecoderModelSpec, UnsupportedOperator};
use ferrule_backend::cpu::gated_delta::GatedDeltaShape;
use ferrule_common::execution::KvElementType;
use ferrule_common::{Error, Result};

pub type HybridDecoderSequenceState = DecoderSequenceState<(), HybridLayerStates>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HybridLayerSchema {
    FullAttention {
        kv_layer: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    GatedDeltaNet(GatedDeltaShape),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridStateSchema {
    layers: Vec<HybridLayerSchema>,
}
impl HybridStateSchema {
    pub fn from_spec(spec: &DecoderModelSpec, active_layers: usize) -> Result<Self> {
        if active_layers == 0 || active_layers > spec.layers().len() {
            return Err(error("invalid hybrid active layer count"));
        }
        let mut full = 0;
        let mut layers = Vec::new();
        for layer in &spec.layers()[..active_layers] {
            layers.push(match layer.attention() {
                Attention::Gqa(gqa) => {
                    let entry = HybridLayerSchema::FullAttention {
                        kv_layer: full,
                        kv_heads: gqa.num_kv_heads(),
                        head_dim: gqa.head_dim(),
                    };
                    full += 1;
                    entry
                }
                Attention::GatedDeltaNet(d) => {
                    let shape = GatedDeltaShape {
                        key_heads: d.num_key_heads(),
                        value_heads: d.num_value_heads(),
                        key_dim: d.key_head_dim(),
                        value_dim: d.value_head_dim(),
                        kernel: d.conv_kernel_dim(),
                    };
                    shape.sizes()?;
                    HybridLayerSchema::GatedDeltaNet(shape)
                }
                _ => {
                    return Err(unsupported(
                        "hybrid state supports GQA and GatedDeltaNet only",
                    ));
                }
            });
        }
        Ok(Self { layers })
    }
    pub fn layers(&self) -> &[HybridLayerSchema] {
        &self.layers
    }
    pub fn kv_planes(&self, page_size: usize, max_positions: usize) -> Result<StandardGqaPlanes> {
        let mut geometry = None;
        let mut count = 0;
        for layer in &self.layers {
            if let HybridLayerSchema::FullAttention {
                kv_heads, head_dim, ..
            } = layer
            {
                if geometry.is_some_and(|g| g != (*kv_heads, *head_dim)) {
                    return Err(unsupported(
                        "hybrid CPU requires uniform full-attention KV geometry",
                    ));
                }
                geometry = Some((*kv_heads, *head_dim));
                count += 1;
            }
        }
        let (heads, dim) = geometry.ok_or_else(|| {
            unsupported("paged hybrid decoder needs at least one full-attention layer")
        })?;
        StandardGqaPlanes::new(
            count,
            heads,
            dim,
            page_size,
            max_positions,
            KvElementType::F32,
        )
    }
    pub fn create(&self) -> Result<HybridLayerStates> {
        Ok(HybridLayerStates {
            schema: self.clone(),
            layers: self
                .layers
                .iter()
                .map(|layer| match layer {
                    HybridLayerSchema::FullAttention { .. } => Ok(HybridLayerState::FullAttention),
                    HybridLayerSchema::GatedDeltaNet(shape) => {
                        let (_, conv, recurrent) = shape.sizes()?;
                        Ok(HybridLayerState::GatedDeltaNet(GatedDeltaNetState {
                            shape: *shape,
                            position: 0,
                            conv_history: vec![0.0; conv],
                            recurrent: vec![0.0; recurrent],
                        }))
                    }
                })
                .collect::<Result<_>>()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatedDeltaNetState {
    pub(crate) shape: GatedDeltaShape,
    pub(crate) position: usize,
    pub(crate) conv_history: Vec<f32>,
    pub(crate) recurrent: Vec<f32>,
}
impl GatedDeltaNetState {
    pub const fn shape(&self) -> GatedDeltaShape {
        self.shape
    }
    pub const fn position(&self) -> usize {
        self.position
    }
    pub fn conv_history(&self) -> &[f32] {
        &self.conv_history
    }
    pub fn recurrent(&self) -> &[f32] {
        &self.recurrent
    }
}
#[derive(Debug, Clone, PartialEq)]
pub enum HybridLayerState {
    FullAttention,
    GatedDeltaNet(GatedDeltaNetState),
}
#[derive(Debug, Clone, PartialEq)]
pub struct HybridLayerStates {
    schema: HybridStateSchema,
    layers: Vec<HybridLayerState>,
}
impl HybridLayerStates {
    pub fn schema(&self) -> &HybridStateSchema {
        &self.schema
    }
    pub fn layers(&self) -> &[HybridLayerState] {
        &self.layers
    }
    pub fn linear_mut(&mut self, layer: usize) -> Result<&mut GatedDeltaNetState> {
        match self.layers.get_mut(layer) {
            Some(HybridLayerState::GatedDeltaNet(state)) => Ok(state),
            _ => Err(error("layer does not have gated delta state")),
        }
    }
    pub fn validate(&self, schema: &HybridStateSchema, position: usize) -> Result<()> {
        if &self.schema != schema || self.layers.len() != schema.layers.len() {
            return Err(error("hybrid state schema mismatch"));
        }
        for (entry, state) in schema.layers.iter().zip(&self.layers) {
            match (entry, state) {
                (HybridLayerSchema::FullAttention { .. }, HybridLayerState::FullAttention) => {}
                (HybridLayerSchema::GatedDeltaNet(shape), HybridLayerState::GatedDeltaNet(s)) => {
                    let (_, c, r) = shape.sizes()?;
                    if s.shape != *shape
                        || s.position != position
                        || s.conv_history.len() != c
                        || s.recurrent.len() != r
                    {
                        return Err(error("hybrid recurrent/conv frontier or shape mismatch"));
                    }
                }
                _ => return Err(error("hybrid layer state kind mismatch")),
            }
        }
        Ok(())
    }
}

/// Borrowed physical state; never converts CUDA state to host storage.
pub enum GatedDeltaStateRef<'a> {
    Cpu(&'a mut GatedDeltaNetState),
    #[cfg(feature = "cuda")]
    Cuda(&'a mut super::CudaGatedDeltaState),
}
impl GatedDeltaStateRef<'_> {
    pub(crate) fn cpu(&mut self) -> Result<&mut GatedDeltaNetState> {
        match self {
            Self::Cpu(state) => Ok(state),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(unsupported(
                "CPU operator cannot access CUDA recurrent state",
            )),
        }
    }
}

/// State access used by the single standard CPU module for both stateless GQA and hybrid layers.
pub trait StandardSequenceState: super::DecoderSequence {
    fn validate_standard_schema(schema: &HybridStateSchema) -> Result<()>
    where
        Self: Sized;
    fn validate_standard_state_at(&self, schema: &HybridStateSchema, position: usize)
    -> Result<()>;
    fn validate_standard_state(&self, schema: &HybridStateSchema) -> Result<()> {
        self.validate_standard_state_at(schema, self.core().position())
    }
    fn linear_state_mut(&mut self, layer: usize) -> Result<&mut GatedDeltaNetState>;
    fn linear_state(&mut self, layer: usize) -> Result<GatedDeltaStateRef<'_>> {
        self.linear_state_mut(layer).map(GatedDeltaStateRef::Cpu)
    }
}
impl StandardSequenceState for super::GenericDecoderSequenceState {
    fn validate_standard_schema(schema: &HybridStateSchema) -> Result<()> {
        if schema
            .layers
            .iter()
            .any(|l| matches!(l, HybridLayerSchema::GatedDeltaNet(_)))
        {
            Err(unsupported(
                "GatedDeltaNet requires hybrid sequence lifecycle",
            ))
        } else {
            Ok(())
        }
    }
    fn validate_standard_state_at(&self, schema: &HybridStateSchema, _: usize) -> Result<()> {
        Self::validate_standard_schema(schema)
    }
    fn linear_state_mut(&mut self, _: usize) -> Result<&mut GatedDeltaNetState> {
        Err(unsupported("sequence has no recurrent state"))
    }
}
impl StandardSequenceState for HybridDecoderSequenceState {
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
    fn linear_state_mut(&mut self, layer: usize) -> Result<&mut GatedDeltaNetState> {
        self.kv_state_mut().linear_mut(layer)
    }
}
#[derive(Debug, Clone)]
pub struct HybridSequenceLifecycle {
    schema: HybridStateSchema,
}
impl HybridSequenceLifecycle {
    pub fn new(schema: HybridStateSchema) -> Self {
        Self { schema }
    }
    pub fn schema(&self) -> &HybridStateSchema {
        &self.schema
    }
}
impl DecoderSequenceLifecycle<HybridDecoderSequenceState> for HybridSequenceLifecycle {
    fn create(&mut self) -> Result<HybridDecoderSequenceState> {
        Ok(DecoderSequenceState::new((), self.schema.create()?))
    }
    fn checkout(
        &mut self,
        _: DecoderSequenceCheckout,
        source: &HybridDecoderSequenceState,
    ) -> Result<HybridDecoderSequenceState> {
        source.validate_standard_state(&self.schema)?;
        source.transaction_working_copy()
    }
    fn logical_fork(
        &mut self,
        source: &HybridDecoderSequenceState,
        expected_position: usize,
    ) -> Result<HybridDecoderSequenceState> {
        source.validate_standard_state(&self.schema)?;
        if source.core().position() != expected_position {
            return Err(unsupported(
                "hybrid fork requires exact committed frontier; partial retain is unsupported",
            ));
        }
        source.logical_fork()
    }
    fn reset(&mut self, state: &mut HybridDecoderSequenceState) -> Result<()> {
        state.validate_standard_state(&self.schema)?;
        let fresh = self.schema.create()?;
        *state.kv_state_mut() = fresh;
        state.core_mut().reset();
        Ok(())
    }
    fn try_release(
        &mut self,
        state: HybridDecoderSequenceState,
    ) -> std::result::Result<(), SequenceStateReleaseError<HybridDecoderSequenceState>> {
        // Aborted working copies can have partially executed layers; release validates ownership/schema, not frontier.
        if state.kv_state().schema() != &self.schema {
            return Err(SequenceStateReleaseError::new(
                error("hybrid release schema mismatch"),
                state,
            ));
        }
        drop(state);
        Ok(())
    }
}
fn error(message: &str) -> Error {
    Error::Execution {
        message: message.into(),
    }
}
fn unsupported(message: &str) -> Error {
    Error::ModelSource {
        source: Box::new(UnsupportedOperator::new("hybrid_decoder", message)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::execution::ExecutionTransactionId;

    fn schema() -> HybridStateSchema {
        HybridStateSchema {
            layers: vec![
                HybridLayerSchema::GatedDeltaNet(GatedDeltaShape {
                    key_heads: 1,
                    value_heads: 2,
                    key_dim: 3,
                    value_dim: 4,
                    kernel: 3,
                }),
                HybridLayerSchema::FullAttention {
                    kv_layer: 0,
                    kv_heads: 1,
                    head_dim: 4,
                },
            ],
        }
    }

    #[test]
    fn failed_release_retains_exact_state_and_partial_work_is_releasable() {
        let mut lifecycle = HybridSequenceLifecycle::new(schema());
        let state = lifecycle.create().unwrap();
        let mut wrong = HybridSequenceLifecycle::new(HybridStateSchema { layers: Vec::new() });
        let original = state.clone();
        let failed = wrong.try_release(state).unwrap_err();
        let (_, state) = failed.into_parts();
        assert_eq!(state, original);
        let mut working = lifecycle
            .checkout(
                DecoderSequenceCheckout::new(ExecutionTransactionId::new(1).unwrap(), 0, 0),
                &state,
            )
            .unwrap();
        working.kv_state_mut().linear_mut(0).unwrap().position = 1;
        assert!(working.validate_standard_state(&schema()).is_err());
        lifecycle.try_release(working).unwrap();
        assert_eq!(state, original);
        lifecycle.try_release(state).unwrap();
    }

    #[test]
    fn checkout_and_fork_validate_recurrent_frontier_before_copying() {
        let mut lifecycle = HybridSequenceLifecycle::new(schema());
        let mut state = lifecycle.create().unwrap();
        state.kv_state_mut().linear_mut(0).unwrap().position = 2;
        assert!(
            lifecycle
                .checkout(
                    DecoderSequenceCheckout::new(ExecutionTransactionId::new(1).unwrap(), 0, 0),
                    &state
                )
                .is_err()
        );
        assert!(lifecycle.logical_fork(&state, 0).is_err());
        assert!(lifecycle.reset(&mut state).is_err());
        lifecycle.try_release(state).unwrap();
    }
}
