//! Per-layer state for standard paged GQA / gated-delta decoders.
use super::{
    DecoderSequenceCheckout, DecoderSequenceLifecycle, DecoderSequenceState, StandardGqaPlanes,
};
use crate::runner::SequenceStateReleaseError;
use crate::transformer::{Attention, DecoderModelSpec, UnsupportedOperator};
use ferrule_backend::cpu::gated_delta::GatedDeltaShape;
use ferrule_common::execution::KvElementType;
use ferrule_common::{Error, Result};
use std::ops::Range;

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
/// State slots retain one entry per decoder layer; only KV planes are compact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridLayerEntry {
    pub global_layer: usize,
    pub local_layer: usize,
    pub state_slot: usize,
    pub global_kv_layer: Option<usize>,
    pub local_kv_layer: Option<usize>,
}
#[derive(Debug, thiserror::Error)]
pub enum HybridSchemaError {
    #[error("invalid hybrid layer range {range:?} for {total_layers} layers")]
    InvalidRange {
        range: Range<usize>,
        total_layers: usize,
    },
    #[error("recurrent-only hybrid segment {range:?} requires a zero-KV backend")]
    RecurrentOnly { range: Range<usize> },
    #[error("hybrid schemas do not form an exact ordered partition at segment {segment}")]
    InvalidPartition { segment: usize },
}
impl From<HybridSchemaError> for Error {
    fn from(source: HybridSchemaError) -> Self {
        Self::ModelSource {
            source: Box::new(source),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridStateSchema {
    layers: Vec<HybridLayerSchema>,
    entries: Vec<HybridLayerEntry>,
    range: Range<usize>,
    total_layers: usize,
}
impl HybridStateSchema {
    pub fn from_spec(spec: &DecoderModelSpec, active_layers: usize) -> Result<Self> {
        Self::build(spec, 0..active_layers, false)
    }
    /// A physical PP segment must contain full attention. The compatible prefix
    /// constructor still permits recurrent-only prefixes, but kv_planes rejects them.
    pub fn for_segment(spec: &DecoderModelSpec, range: Range<usize>) -> Result<Self> {
        Self::build(spec, range, true)
    }
    fn build(spec: &DecoderModelSpec, range: Range<usize>, require_kv: bool) -> Result<Self> {
        if range.start >= range.end || range.end > spec.layers().len() {
            return Err(HybridSchemaError::InvalidRange {
                range,
                total_layers: spec.layers().len(),
            }
            .into());
        }
        let mut global_kv = spec.layers()[..range.start]
            .iter()
            .filter(|l| matches!(l.attention(), Attention::Gqa(_)))
            .count();
        let mut local_kv = 0;
        let mut geometry = None;
        let mut layers = Vec::new();
        let mut entries = Vec::new();
        for (local, layer) in spec.layers()[range.clone()].iter().enumerate() {
            let (description, kv) = match layer.attention() {
                Attention::Gqa(gqa) => {
                    let shape = (gqa.num_kv_heads(), gqa.head_dim());
                    if geometry.is_some_and(|previous| previous != shape) {
                        return Err(unsupported(
                            "hybrid requires uniform full-attention KV geometry",
                        ));
                    }
                    geometry = Some(shape);
                    let description = HybridLayerSchema::FullAttention {
                        kv_layer: local_kv,
                        kv_heads: shape.0,
                        head_dim: shape.1,
                    };
                    let kv = (Some(global_kv), Some(local_kv));
                    global_kv += 1;
                    local_kv += 1;
                    (description, kv)
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
                    (HybridLayerSchema::GatedDeltaNet(shape), (None, None))
                }
                _ => {
                    return Err(unsupported(
                        "hybrid state supports GQA and GatedDeltaNet only",
                    ));
                }
            };
            entries.push(HybridLayerEntry {
                global_layer: range.start + local,
                local_layer: local,
                state_slot: local,
                global_kv_layer: kv.0,
                local_kv_layer: kv.1,
            });
            layers.push(description);
        }
        if require_kv && local_kv == 0 {
            return Err(HybridSchemaError::RecurrentOnly { range }.into());
        }
        let schema = Self {
            layers,
            entries,
            range,
            total_layers: spec.layers().len(),
        };
        schema.per_sequence_state_bytes()?;
        Ok(schema)
    }
    pub fn entries(&self) -> &[HybridLayerEntry] {
        &self.entries
    }
    pub fn entry(&self, global_layer: usize) -> Option<&HybridLayerEntry> {
        global_layer
            .checked_sub(self.range.start)
            .and_then(|i| self.entries.get(i))
    }
    pub fn layer_range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }
    pub fn per_sequence_state_bytes(&self) -> Result<usize> {
        self.layers.iter().try_fold(0usize, |sum, layer| {
            let HybridLayerSchema::GatedDeltaNet(shape) = layer else {
                return Ok(sum);
            };
            let (_, conv, recurrent) = shape.sizes()?;
            for count in [conv, recurrent] {
                if count
                    .checked_mul(std::mem::size_of::<f32>())
                    .is_none_or(|bytes| bytes > isize::MAX as usize)
                {
                    return Err(error("hybrid state buffer exceeds address space"));
                }
            }
            conv.checked_add(recurrent)
                .and_then(|n| n.checked_mul(4))
                .and_then(|n| sum.checked_add(n))
                .ok_or_else(|| error("hybrid state byte overflow"))
        })
    }
    /// Reconstruct each range from the model to check geometry and all mappings.
    pub fn validate_partition(spec: &DecoderModelSpec, schemas: &[Self]) -> Result<()> {
        let mut next = 0;
        for (segment, schema) in schemas.iter().enumerate() {
            if schema.range.start != next
                || schema != &Self::for_segment(spec, schema.range.clone())?
            {
                return Err(HybridSchemaError::InvalidPartition { segment }.into());
            }
            next = schema.range.end;
        }
        if next != spec.layers().len() || schemas.is_empty() {
            return Err(HybridSchemaError::InvalidPartition {
                segment: schemas.len(),
            }
            .into());
        }
        Self::from_spec(spec, spec.layers().len())?;
        Ok(())
    }
    /// Preserve global coordinates and compact slots while specializing physical shapes.
    pub(crate) fn with_local_layers(&self, layers: Vec<HybridLayerSchema>) -> Result<Self> {
        if layers.len() != self.layers.len()
            || layers
                .iter()
                .zip(&self.layers)
                .any(|(a, b)| std::mem::discriminant(a) != std::mem::discriminant(b))
        {
            return Err(error("local hybrid schema changes layer kinds"));
        }
        let mut local = self.clone();
        local.layers = layers;
        local.per_sequence_state_bytes()?;
        Ok(local)
    }
    pub fn layers(&self) -> &[HybridLayerSchema] {
        &self.layers
    }
    pub fn kv_planes(&self, page_size: usize, max_positions: usize) -> Result<StandardGqaPlanes> {
        self.kv_planes_with_dtype(page_size, max_positions, KvElementType::F32)
    }
    /// Explicit typed opt-in; the existing default and rank paths remain F32.
    pub fn kv_planes_with_dtype(
        &self,
        page_size: usize,
        max_positions: usize,
        dtype: KvElementType,
    ) -> Result<StandardGqaPlanes> {
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
        StandardGqaPlanes::new(count, heads, dim, page_size, max_positions, dtype)
    }
    pub fn create(&self) -> Result<HybridLayerStates> {
        self.per_sequence_state_bytes()?;
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
        let slot = self
            .schema
            .entry(layer)
            .ok_or_else(|| error("global layer outside hybrid schema"))?
            .state_slot;
        match self.layers.get_mut(slot) {
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
    #[cfg(feature = "cuda")]
    CudaBf16(&'a mut super::hybrid_cuda::CudaBf16GatedDeltaState),
}
/// Backend-neutral schema, frontier and borrowed recurrent-state contract for
/// the shared standard decoder layer loop.
///
/// The concrete state retains its physical owner and representation. This trait
/// does not require Clone or Send and grants no checkout, release or completion
/// authority; those remain with the sequence lifecycle and owner.
#[doc = include_str!("../../tests/compile/standard_sequence_state.md")]
pub trait StandardSequenceState: super::DecoderSequence {
    fn validate_standard_schema(schema: &HybridStateSchema) -> Result<()>
    where
        Self: Sized;
    fn validate_standard_state_at(&self, schema: &HybridStateSchema, position: usize)
    -> Result<()>;
    fn validate_standard_state(&self, schema: &HybridStateSchema) -> Result<()> {
        self.validate_standard_state_at(schema, self.core().position())
    }
    /// Exclusively borrow the physical state for a global decoder layer.
    ///
    /// Return only the owner's actual storage kind, never a host fallback or a
    /// copied payload. Missing/non-recurrent layers and unavailable owners fail
    /// without transferring custody. The operator validates its expected kind
    /// and owner before use; the borrow cannot outlive this sequence state.
    fn linear_state(&mut self, layer: usize) -> Result<GatedDeltaStateRef<'_>>;
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
    fn linear_state(&mut self, _: usize) -> Result<GatedDeltaStateRef<'_>> {
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
    fn linear_state(&mut self, layer: usize) -> Result<GatedDeltaStateRef<'_>> {
        self.kv_state_mut()
            .linear_mut(layer)
            .map(GatedDeltaStateRef::Cpu)
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
            range: 0..2,
            total_layers: 2,
            entries: vec![
                HybridLayerEntry {
                    global_layer: 0,
                    local_layer: 0,
                    state_slot: 0,
                    global_kv_layer: None,
                    local_kv_layer: None,
                },
                HybridLayerEntry {
                    global_layer: 1,
                    local_layer: 1,
                    state_slot: 1,
                    global_kv_layer: Some(0),
                    local_kv_layer: Some(0),
                },
            ],
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
        let mut wrong = HybridSequenceLifecycle::new(HybridStateSchema {
            layers: Vec::new(),
            entries: Vec::new(),
            range: 0..0,
            total_layers: 0,
        });
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

#[cfg(test)]
#[path = "../../tests/hybrid_parallel/state_bounds.rs"]
mod allocation_bounds;

#[cfg(test)]
#[path = "../../tests/unit/standard_sequence_state.rs"]
mod state_contract_tests;
