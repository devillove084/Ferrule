//! Typed storage inside the existing hybrid lifecycle, never a second authority.
use super::*;
use crate::execution::ExecutionPrecisionPolicy;
use ferrule_backend::cuda::operators::linear::CudaBf16Buffer;

pub struct CudaBf16GatedDeltaState {
    pub(crate) shape: GatedDeltaShape,
    pub(crate) position: usize,
    pub(crate) conv: CudaBf16Buffer,
    pub(crate) recurrent: CudaF32Buffer,
    pub(crate) device: HybridCudaDevice,
}
impl std::fmt::Debug for CudaBf16GatedDeltaState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBf16GatedDeltaState")
            .field("shape", &self.shape)
            .field("position", &self.position)
            .finish()
    }
}
impl CudaBf16GatedDeltaState {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn shape(&self) -> GatedDeltaShape {
        self.shape
    }
    pub fn diagnostic_snapshot(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        self.device.fence()?;
        let result = (|| {
            Ok((
                self.device.operators().download_bf16_buffer(&self.conv)?,
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
            return Err(error("foreign BF16 GDN state"));
        }
        let (_, c, r) = self.shape.sizes()?;
        if self.conv.len() != c || self.recurrent.len() != r {
            return Err(error("BF16 GDN extents"));
        }
        Ok(())
    }
}
impl CudaHybridLayerStates {
    pub fn bf16_layers(&self) -> Option<&[Option<CudaBf16GatedDeltaState>]> {
        self.bf16_layers.as_deref()
    }
    pub(super) fn validate_bf16(&self, schema: &HybridStateSchema, position: usize) -> Result<()> {
        let states = self
            .bf16_layers
            .as_ref()
            .ok_or_else(|| error("missing BF16 state"))?;
        if schema != &self.schema
            || states.len() != schema.layers().len()
            || !self.layers.is_empty()
        {
            return Err(error("BF16 state/schema mismatch"));
        }
        for (entry, state) in schema.layers().iter().zip(states) {
            match (entry, state) {
                (HybridLayerSchema::FullAttention { .. }, None) => {}
                (HybridLayerSchema::GatedDeltaNet(shape), Some(state))
                    if *shape == state.shape && position == state.position =>
                {
                    state.validate_owner(self.device.operators())?;
                }
                _ => return Err(error("BF16 GDN frontier mismatch")),
            }
        }
        Ok(())
    }
}
pub(super) fn state_bytes(schema: &HybridStateSchema) -> Result<usize> {
    schema.layers().iter().try_fold(0usize, |sum, layer| {
        let HybridLayerSchema::GatedDeltaNet(shape) = layer else {
            return Ok(sum);
        };
        let (_, c, r) = shape.sizes()?;
        c.checked_mul(2)
            .and_then(|n| r.checked_mul(4).and_then(|r| n.checked_add(r)))
            .and_then(|n| sum.checked_add(n))
            .ok_or_else(|| error("BF16 state byte overflow"))
    })
}
impl CudaHybridSequenceLifecycle {
    pub fn with_precision(
        schema: HybridStateSchema,
        device: HybridCudaDevice,
        precision: ExecutionPrecisionPolicy,
    ) -> Result<Self> {
        if precision == ExecutionPrecisionPolicy::f32() {
            return Self::new(schema, device);
        }
        if precision != ExecutionPrecisionPolicy::bf16_compatibility() {
            return Err(error("unsupported hybrid state precision"));
        }
        state_bytes(&schema)?;
        Ok(Self {
            schema,
            device,
            bf16: true,
        })
    }
    pub(super) fn allocate_bf16(&self) -> Result<CudaHybridLayerStates> {
        self.device.ensure_ready()?;
        let bytes = state_bytes(&self.schema)?;
        let charged = self
            .device
            .live_state_bytes()
            .checked_add(bytes)
            .filter(|&n| n <= self.device.budget().state_bytes)
            .ok_or_else(|| error("BF16 state budget exhausted (checkouts/forks included)"))?;
        self.device.inner.live_state_bytes.set(charged);
        let mut state = CudaHybridLayerStates {
            schema: self.schema.clone(),
            layers: Vec::new(),
            bf16_layers: Some(Vec::new()),
            device: self.device.clone(),
            charged_bytes: bytes,
            release_proven: false,
        };
        let result = (|| {
            for entry in self.schema.layers() {
                let slot = match entry {
                    HybridLayerSchema::FullAttention { .. } => None,
                    HybridLayerSchema::GatedDeltaNet(shape) => {
                        let (_, c, r) = shape.sizes()?;
                        Some(CudaBf16GatedDeltaState {
                            shape: *shape,
                            position: 0,
                            conv: self.device.operators().zero_bf16_buffer(c)?,
                            recurrent: self.device.operators().zero_f32_buffer(r)?,
                            device: self.device.clone(),
                        })
                    }
                };
                state.bf16_layers.as_mut().unwrap().push(slot);
            }
            Ok(())
        })();
        self.device.complete(result)?;
        Ok(state)
    }
    pub(super) fn copy_bf16(
        &self,
        source: &CudaHybridSequenceState,
        destination: &mut CudaHybridLayerStates,
    ) -> Result<()> {
        let result = (|| {
            for (src, dst) in source
                .kv_state()
                .bf16_layers
                .as_ref()
                .unwrap()
                .iter()
                .zip(destination.bf16_layers.as_mut().unwrap())
            {
                if let (Some(src), Some(dst)) = (src, dst) {
                    // Typed row gather copies raw words, including NaN/signed zero.
                    // Its transient replacement history is charged in workspace.
                    dst.conv =
                        self.device
                            .operators()
                            .embedding_bf16(&src.conv, &[0], src.conv.len())?;
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
        self.device.complete(result)
    }
}
