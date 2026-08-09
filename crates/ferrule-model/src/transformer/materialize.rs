//! Lazy state-dict materialization for standard decoder execution.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ferrule_common::{Error, Result};

use crate::checkpoint::{
    CheckpointDType, CheckpointTensorPayload, CheckpointTensorReader, LinearWeight,
};
use crate::nn::{ParameterId, ParameterResidency};
use crate::support::TensorRole;

use super::{BoundParameter, BoundTensorPart, TensorTransform};
use crate::execution::{PreparedExecutable, PreparedModel};

static NEXT_PREPARED_DECODER_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Process-unique generation assigned when one immutable decoder image is published.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PreparedDecoderGeneration(u64);

impl PreparedDecoderGeneration {
    pub fn take() -> Result<Self> {
        NEXT_PREPARED_DECODER_GENERATION
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
                generation.checked_add(1)
            })
            .map(Self)
            .map_err(|_| model_error("prepared decoder generation space is exhausted"))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Generation-bound slot for one backend-compiled typed decoder payload.
///
/// The slot is created with the prepared image generation and shared with the
/// backend compiler through the image resources. Installation is one-shot, reads
/// are lock-free after publication, and shutdown invalidates all future reads.
#[derive(Debug)]
pub struct PreparedDecoderAttachment<A> {
    inner: Arc<PreparedDecoderAttachmentInner<A>>,
}

#[derive(Debug)]
struct PreparedDecoderAttachmentInner<A> {
    generation: u64,
    state: AtomicU8,
    value: OnceLock<A>,
}

impl<A> Clone for PreparedDecoderAttachment<A> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<A> PreparedDecoderAttachment<A> {
    fn new(generation: u64) -> Self {
        Self {
            inner: Arc::new(PreparedDecoderAttachmentInner {
                generation,
                state: AtomicU8::new(0),
                value: OnceLock::new(),
            }),
        }
    }

    pub fn generation(&self) -> u64 {
        self.inner.generation
    }

    pub fn install(&self, value: A) -> Result<&A> {
        self.inner
            .state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|state| match state {
                3 => model_error("cannot install a shutdown prepared decoder attachment"),
                _ => model_error("prepared decoder attachment is already installed"),
            })?;
        self.inner
            .value
            .set(value)
            .map_err(|_| model_error("prepared decoder attachment storage is already occupied"))?;
        self.inner
            .state
            .compare_exchange(1, 2, Ordering::Release, Ordering::Acquire)
            .map_err(|_| {
                model_error("prepared decoder attachment shut down during installation")
            })?;
        self.get()
    }

    pub fn get(&self) -> Result<&A> {
        self.as_ref()
            .ok_or_else(|| model_error("prepared decoder attachment is unavailable"))
    }

    pub fn as_ref(&self) -> Option<&A> {
        (self.inner.state.load(Ordering::Acquire) == 2)
            .then(|| self.inner.value.get())
            .flatten()
    }

    pub fn is_installed(&self) -> bool {
        self.inner.state.load(Ordering::Acquire) == 2
    }

    pub fn shutdown(&self) {
        self.inner.state.store(3, Ordering::Release);
    }

    pub fn take(self) -> Result<A> {
        let inner = Arc::try_unwrap(self.inner)
            .map_err(|_| model_error("prepared decoder attachment is still shared"))?;
        inner
            .value
            .into_inner()
            .ok_or_else(|| model_error("prepared decoder attachment is not installed"))
    }
}

/// Immutable generation-stamped decoder image shared by all family adapters.
///
/// Binding, executable declaration, and typed backend attachment ownership share
/// one generation. Model runtimes receive only a clone of the generic attachment
/// handle; they do not maintain a second image or generation lifecycle.
#[derive(Debug)]
pub struct PreparedDecoder<R, O, A = ()> {
    prepared: PreparedModel<R, O>,
    attachment: PreparedDecoderAttachment<A>,
}

impl<R, O> PreparedDecoder<R, O> {
    pub fn publish(resources: R, executable: PreparedExecutable<O>) -> Result<Self> {
        Self::publish_with_attachment(executable, |_| Ok(resources))
    }
}

impl<R, O, A> PreparedDecoder<R, O, A> {
    pub fn publish_with_attachment(
        executable: PreparedExecutable<O>,
        resources: impl FnOnce(PreparedDecoderAttachment<A>) -> Result<R>,
    ) -> Result<Self> {
        let generation = PreparedDecoderGeneration::take()?.get();
        let attachment = PreparedDecoderAttachment::new(generation);
        let resources = resources(attachment.clone())?;
        Ok(Self {
            prepared: PreparedModel::new(generation, resources, executable),
            attachment,
        })
    }

    pub const fn generation(&self) -> u64 {
        self.prepared.generation()
    }

    pub const fn resources(&self) -> &R {
        self.prepared.resources()
    }

    pub const fn executable(&self) -> &PreparedExecutable<O> {
        self.prepared.executable()
    }

    pub const fn attachment(&self) -> &PreparedDecoderAttachment<A> {
        &self.attachment
    }

    pub fn shutdown(&self) {
        self.attachment.shutdown();
    }
}

/// Materialized parameter payload retaining its canonical state-dict identity.
#[derive(Debug, Clone)]
pub struct PreparedParameter {
    binding: BoundParameter,
    weight: CheckpointTensorPayload,
    scale: Option<CheckpointTensorPayload>,
}

impl PreparedParameter {
    pub fn binding(&self) -> &BoundParameter {
        &self.binding
    }

    pub const fn canonical_id(&self) -> ParameterId {
        self.binding.canonical_id()
    }

    pub fn residency(&self) -> &ParameterResidency {
        self.binding.residency()
    }

    pub fn role(&self) -> &TensorRole {
        self.binding.role()
    }

    pub fn weight(&self) -> &CheckpointTensorPayload {
        &self.weight
    }

    pub fn scale(&self) -> Option<&CheckpointTensorPayload> {
        self.scale.as_ref()
    }

    pub fn values_f32(&self) -> Result<Vec<f32>> {
        decode_float_payload(&self.weight)
    }

    pub fn values_usize(&self) -> Result<Vec<usize>> {
        let expected = self.weight.slice.element_count()?;
        match self.weight.slice.dtype {
            CheckpointDType::I32 => {
                let expected_bytes = expected.checked_mul(4).ok_or_else(|| {
                    model_error(format!(
                        "I32 parameter '{}' byte length overflows",
                        self.binding.path()
                    ))
                })?;
                if self.weight.bytes.len() != expected_bytes {
                    return Err(model_error(format!(
                        "I32 parameter '{}' has {} bytes, expected {expected_bytes}",
                        self.binding.path(),
                        self.weight.bytes.len()
                    )));
                }
                self.weight
                    .bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| {
                        let value = i32::from_le_bytes(*bytes);
                        usize::try_from(value).map_err(|_| {
                            model_error(format!(
                                "parameter '{}' contains negative integer {value}",
                                self.binding.path()
                            ))
                        })
                    })
                    .collect()
            }
            CheckpointDType::I64 => {
                let expected_bytes = expected.checked_mul(8).ok_or_else(|| {
                    model_error(format!(
                        "I64 parameter '{}' byte length overflows",
                        self.binding.path()
                    ))
                })?;
                if self.weight.bytes.len() != expected_bytes {
                    return Err(model_error(format!(
                        "I64 parameter '{}' has {} bytes, expected {expected_bytes}",
                        self.binding.path(),
                        self.weight.bytes.len()
                    )));
                }
                self.weight
                    .bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|bytes| {
                        let value = i64::from_le_bytes(*bytes);
                        usize::try_from(value).map_err(|_| {
                            model_error(format!(
                                "parameter '{}' contains negative integer {value}",
                                self.binding.path()
                            ))
                        })
                    })
                    .collect()
            }
            ref dtype => Err(model_error(format!(
                "parameter '{}' is {}, not an I32/I64 integer tensor",
                self.binding.path(),
                dtype.as_str()
            ))),
        }
    }

    pub fn values_bf16_words(&self) -> Result<Vec<u16>> {
        if self.weight.slice.dtype != CheckpointDType::Bf16 {
            return Err(model_error(format!(
                "parameter '{}' is {}, not BF16",
                self.binding.path(),
                self.weight.slice.dtype.as_str()
            )));
        }
        Ok(self
            .weight
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
            .collect())
    }

    pub fn into_linear(&self, role: TensorRole) -> Result<PreparedLinear> {
        let weight =
            LinearWeight::from_weight_and_scale(role, self.weight.clone(), self.scale.clone())?;
        Ok(PreparedLinear {
            parameter: Arc::new(self.clone()),
            weight,
            bias: None,
        })
    }
}

/// Prepared matrix and optional additive bias.
#[derive(Debug, Clone)]
pub struct PreparedLinear {
    parameter: Arc<PreparedParameter>,
    weight: LinearWeight,
    bias: Option<Arc<[f32]>>,
}

impl PreparedLinear {
    pub fn from_parameter(parameter: Arc<PreparedParameter>, role: TensorRole) -> Result<Self> {
        let weight = LinearWeight::from_weight_and_scale(
            role,
            parameter.weight.clone(),
            parameter.scale.clone(),
        )?;
        Ok(Self {
            parameter,
            weight,
            bias: None,
        })
    }

    pub fn with_bias(mut self, bias: Arc<PreparedParameter>) -> Result<Self> {
        let values = bias.values_f32()?;
        if values.len() != self.out_features() {
            return Err(model_error(format!(
                "linear bias '{}' has {} values, expected {}",
                bias.binding.path(),
                values.len(),
                self.out_features()
            )));
        }
        self.bias = Some(values.into());
        Ok(self)
    }

    pub fn parameter(&self) -> &PreparedParameter {
        &self.parameter
    }

    pub fn weight(&self) -> &LinearWeight {
        &self.weight
    }

    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    pub fn in_features(&self) -> usize {
        self.weight.format.in_features()
    }

    pub fn out_features(&self) -> usize {
        self.weight.format.out_features()
    }
}

#[cfg(feature = "cuda")]
/// Backend-compiled linear handle with its activation contract.
pub struct PreparedCudaLinear {
    pub(crate) handle: ferrule_backend::cuda::operators::linear::CudaArtifactLinearHandle,
    pub(crate) activation_quantization: Option<crate::checkpoint::weight::ActivationQuantization>,
}

#[cfg(feature = "cuda")]
impl PreparedCudaLinear {
    pub fn new(
        handle: ferrule_backend::cuda::operators::linear::CudaArtifactLinearHandle,
        activation_quantization: Option<crate::checkpoint::weight::ActivationQuantization>,
    ) -> Self {
        Self {
            handle,
            activation_quantization,
        }
    }

    pub const fn handle(
        &self,
    ) -> &ferrule_backend::cuda::operators::linear::CudaArtifactLinearHandle {
        &self.handle
    }

    pub const fn activation_quantization(
        &self,
    ) -> Option<crate::checkpoint::weight::ActivationQuantization> {
        self.activation_quantization
    }
}

/// Prepared affine RMSNorm parameters.
#[derive(Debug, Clone)]
pub struct PreparedNorm {
    parameter: Arc<PreparedParameter>,
    weight: Arc<[f32]>,
    epsilon: f32,
}

impl PreparedNorm {
    pub fn new(parameter: Arc<PreparedParameter>, epsilon: f32) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(model_error(format!("invalid RMSNorm epsilon {epsilon}")));
        }
        let weight: Arc<[f32]> = parameter.values_f32()?.into();
        Ok(Self {
            parameter,
            weight,
            epsilon,
        })
    }

    pub fn parameter(&self) -> &PreparedParameter {
        &self.parameter
    }

    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

/// Prepared table-driven rotary coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRope {
    positions: usize,
    dimensions: usize,
    cosine: Arc<[f32]>,
    sine: Arc<[f32]>,
}

impl PreparedRope {
    pub fn new(
        positions: usize,
        dimensions: usize,
        cosine: impl Into<Arc<[f32]>>,
        sine: impl Into<Arc<[f32]>>,
    ) -> Result<Self> {
        let expected = positions
            .checked_mul(dimensions / 2)
            .ok_or_else(|| model_error("RoPE table size overflow"))?;
        let cosine = cosine.into();
        let sine = sine.into();
        if positions == 0
            || dimensions == 0
            || !dimensions.is_multiple_of(2)
            || cosine.len() != expected
            || sine.len() != expected
        {
            return Err(model_error(format!(
                "invalid RoPE table: positions={positions} dimensions={dimensions} cosine={} sine={} expected={expected}",
                cosine.len(),
                sine.len()
            )));
        }
        Ok(Self {
            positions,
            dimensions,
            cosine,
            sine,
        })
    }

    pub const fn positions(&self) -> usize {
        self.positions
    }

    pub const fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn cosine(&self) -> &[f32] {
        &self.cosine
    }

    pub fn sine(&self) -> &[f32] {
        &self.sine
    }
}

/// Prepared token embedding matrix.
#[derive(Debug, Clone)]
pub struct PreparedEmbedding {
    linear: PreparedLinear,
}

impl PreparedEmbedding {
    pub fn new(linear: PreparedLinear) -> Self {
        Self { linear }
    }

    pub fn linear(&self) -> &PreparedLinear {
        &self.linear
    }

    pub fn vocabulary(&self) -> usize {
        self.linear.out_features()
    }

    pub fn width(&self) -> usize {
        self.linear.in_features()
    }
}

/// Cache interface for lazily materialized layer-resident parameters.
pub trait LayerWeightCache {
    fn get(&self, canonical: ParameterId) -> Option<Arc<PreparedParameter>>;
    fn insert(&mut self, parameter: Arc<PreparedParameter>);
}

/// In-memory reference implementation of [`LayerWeightCache`].
#[derive(Debug, Default)]
pub struct MemoryLayerWeightCache {
    parameters: BTreeMap<ParameterId, Arc<PreparedParameter>>,
}

impl MemoryLayerWeightCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.parameters.clear();
    }
}

impl LayerWeightCache for MemoryLayerWeightCache {
    fn get(&self, canonical: ParameterId) -> Option<Arc<PreparedParameter>> {
        self.parameters.get(&canonical).cloned()
    }

    fn insert(&mut self, parameter: Arc<PreparedParameter>) {
        self.parameters.insert(parameter.canonical_id(), parameter);
    }
}

/// Typed integer payload materialized by the common state-dict materializer.
#[derive(Debug, Clone)]
pub struct PreparedInteger {
    shape: Arc<[usize]>,
    values: Arc<[usize]>,
}

impl PreparedInteger {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn values(&self) -> &[usize] {
        &self.values
    }
}

/// Source-identity-aware lazy materializer for one bound state dict.
#[derive(Debug)]
pub struct StateDictMaterializer {
    max_parameter_bytes: u64,
    reader: CheckpointTensorReader,
    static_parameters: Mutex<BTreeMap<ParameterId, Arc<PreparedParameter>>>,
}

impl StateDictMaterializer {
    pub fn new(max_parameter_bytes: u64) -> Result<Self> {
        if max_parameter_bytes == 0 {
            return Err(model_error("state-dict read limit must be positive"));
        }
        Ok(Self {
            max_parameter_bytes,
            reader: CheckpointTensorReader::new(max_parameter_bytes),
            static_parameters: Mutex::new(BTreeMap::new()),
        })
    }

    pub const fn max_parameter_bytes(&self) -> u64 {
        self.max_parameter_bytes
    }

    /// Materialize one bound parameter as a typed linear weight. Family
    /// execution policies are applied by the caller, not here.
    pub fn linear(&self, binding: &BoundParameter) -> Result<LinearWeight> {
        let parameter = self.parameter(binding)?;
        LinearWeight::from_weight_and_scale(
            binding.role().clone(),
            parameter.weight().clone(),
            parameter.scale().cloned(),
        )
    }

    /// Materialize one bound parameter as a typed integer payload.
    pub fn integer(&self, binding: &BoundParameter) -> Result<PreparedInteger> {
        let parameter = self.parameter(binding)?;
        Ok(PreparedInteger {
            shape: parameter.binding().weight().logical_shape().into(),
            values: parameter.values_usize()?.into(),
        })
    }

    /// Materialize exactly one already-bound parameter.
    ///
    /// Static values use the shared cache. Layer, expert, and attachment values
    /// remain uncached here so callers retain explicit stage/expert ownership.
    pub fn parameter(&self, binding: &BoundParameter) -> Result<Arc<PreparedParameter>> {
        match binding.residency() {
            ParameterResidency::Static => self.static_parameter(binding),
            ParameterResidency::Layer { .. }
            | ParameterResidency::Expert { .. }
            | ParameterResidency::Attachment { .. } => Ok(Arc::new(self.read_parameter(binding)?)),
        }
    }

    pub fn static_parameter(&self, binding: &BoundParameter) -> Result<Arc<PreparedParameter>> {
        if binding.residency() != &ParameterResidency::Static {
            return Err(model_error(format!(
                "parameter '{}' is {:?}, not static",
                binding.path(),
                binding.residency()
            )));
        }
        let canonical = binding.canonical_id();
        if let Some(parameter) = self
            .static_parameters
            .lock()
            .map_err(|_| model_error("static parameter cache is poisoned"))?
            .get(&canonical)
            .cloned()
        {
            return Ok(parameter);
        }
        let parameter = Arc::new(self.read_parameter(binding)?);
        let mut cache = self
            .static_parameters
            .lock()
            .map_err(|_| model_error("static parameter cache is poisoned"))?;
        Ok(cache
            .entry(canonical)
            .or_insert_with(|| Arc::clone(&parameter))
            .clone())
    }

    pub fn layer_parameter(
        &self,
        layer: usize,
        binding: &BoundParameter,
        cache: &mut dyn LayerWeightCache,
    ) -> Result<Arc<PreparedParameter>> {
        if binding.residency() != &(ParameterResidency::Layer { layer }) {
            return Err(model_error(format!(
                "parameter '{}' residency {:?} does not match layer {layer}",
                binding.path(),
                binding.residency()
            )));
        }
        if let Some(parameter) = cache.get(binding.canonical_id()) {
            return Ok(parameter);
        }
        let parameter = Arc::new(self.read_parameter(binding)?);
        cache.insert(Arc::clone(&parameter));
        Ok(parameter)
    }

    pub fn attachment_parameter(
        &self,
        attachment: &crate::nn::ModulePath,
        binding: &BoundParameter,
    ) -> Result<Arc<PreparedParameter>> {
        if binding.residency()
            != &(ParameterResidency::Attachment {
                path: attachment.clone(),
            })
        {
            return Err(model_error(format!(
                "parameter '{}' residency {:?} does not match attachment '{}'",
                binding.path(),
                binding.residency(),
                attachment
            )));
        }
        Ok(Arc::new(self.read_parameter(binding)?))
    }

    /// Expert providers use this method only after selecting an expert. It never
    /// populates the static or layer cache implicitly.
    pub fn expert_parameter(
        &self,
        layer: usize,
        expert: usize,
        binding: &BoundParameter,
    ) -> Result<Arc<PreparedParameter>> {
        if binding.residency() != &(ParameterResidency::Expert { layer, expert }) {
            return Err(model_error(format!(
                "parameter '{}' residency {:?} does not match expert {layer}:{expert}",
                binding.path(),
                binding.residency()
            )));
        }
        Ok(Arc::new(self.read_parameter(binding)?))
    }

    fn read_parameter(&self, binding: &BoundParameter) -> Result<PreparedParameter> {
        let total_bytes = binding
            .weight()
            .slice()
            .bytes
            .checked_add(binding.scale().map_or(0, |scale| scale.slice().bytes))
            .ok_or_else(|| model_error("parameter byte size overflow"))?;
        if total_bytes > self.max_parameter_bytes {
            return Err(model_error(format!(
                "parameter '{}' exceeds bounded read size: {total_bytes} > {} bytes",
                binding.path(),
                self.max_parameter_bytes
            )));
        }
        ensure_current(binding.weight(), binding)?;
        if let Some(scale) = binding.scale() {
            ensure_current(scale, binding)?;
        }
        let mut weight = read_transformed(&self.reader, binding.weight())?;
        weight.slice.role = binding.role().clone();
        let scale = binding
            .scale()
            .map(|scale| {
                let mut payload = read_transformed(&self.reader, scale)?;
                payload.slice.role = binding.role().clone();
                Ok::<_, Error>(payload)
            })
            .transpose()?;
        ensure_current(binding.weight(), binding)?;
        if let Some(scale) = binding.scale() {
            ensure_current(scale, binding)?;
        }
        Ok(PreparedParameter {
            binding: binding.clone(),
            weight,
            scale,
        })
    }
}

fn ensure_current(part: &BoundTensorPart, binding: &BoundParameter) -> Result<()> {
    if part.source_is_current() {
        Ok(())
    } else {
        Err(model_error(format!(
            "checkpoint source identity changed while materializing '{}'",
            binding.path()
        )))
    }
}

fn read_transformed(
    reader: &CheckpointTensorReader,
    part: &BoundTensorPart,
) -> Result<CheckpointTensorPayload> {
    let mut payload = reader.read_slice(part.slice())?;
    match part.transform() {
        TensorTransform::Identity => {}
        TensorTransform::Transpose { axes }
            if axes.as_slice() == [1, 0] && payload.slice.shape.len() == 2 =>
        {
            transpose_2d_payload(&mut payload)?;
        }
        transform => {
            return Err(model_error(format!(
                "materializer does not support bound tensor transform {transform:?}"
            )));
        }
    }
    payload.slice.shape = part.physical_shape().to_vec();
    Ok(payload)
}

fn transpose_2d_payload(payload: &mut CheckpointTensorPayload) -> Result<()> {
    let [source_rows, source_cols]: [usize; 2] = payload
        .slice
        .shape
        .clone()
        .try_into()
        .map_err(|_| model_error("transpose expects a 2D payload"))?;
    let element_bytes = payload
        .slice
        .dtype
        .element_size_bytes()
        .ok_or_else(|| model_error("transpose requires a known fixed-width dtype"))?;
    let expected = source_rows
        .checked_mul(source_cols)
        .and_then(|count| count.checked_mul(element_bytes))
        .ok_or_else(|| model_error("transpose byte size overflow"))?;
    if payload.bytes.len() != expected {
        return Err(model_error(format!(
            "transpose payload byte mismatch: got {}, expected {expected}",
            payload.bytes.len()
        )));
    }
    let mut transposed = vec![0; expected];
    for row in 0..source_rows {
        for column in 0..source_cols {
            let source = (row * source_cols + column) * element_bytes;
            let destination = (column * source_rows + row) * element_bytes;
            transposed[destination..destination + element_bytes]
                .copy_from_slice(&payload.bytes[source..source + element_bytes]);
        }
    }
    payload.bytes = transposed;
    payload.slice.shape = vec![source_cols, source_rows];
    Ok(())
}

fn decode_float_payload(payload: &CheckpointTensorPayload) -> Result<Vec<f32>> {
    match payload.slice.dtype {
        CheckpointDType::F32 => Ok(payload
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect()),
        CheckpointDType::Bf16 => Ok(payload
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes(*bytes)) << 16))
            .collect()),
        ref dtype => Err(model_error(format!(
            "parameter '{}' cannot be decoded as float values from {}",
            payload.slice.name,
            dtype.as_str()
        ))),
    }
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("standard state-dict materialization: {}", message.into()),
    }
}

#[cfg(test)]
mod prepared_tests {
    use super::*;

    #[test]
    fn published_decoder_images_receive_unique_nonzero_generations() {
        let first = PreparedDecoder::publish(
            "first",
            PreparedExecutable::<crate::execution::TransformerStage>::new([], []).unwrap(),
        )
        .unwrap();
        let second = PreparedDecoder::publish(
            "second",
            PreparedExecutable::<crate::execution::TransformerStage>::new([], []).unwrap(),
        )
        .unwrap();

        assert_ne!(first.generation(), 0);
        assert!(second.generation() > first.generation());
        assert_eq!(first.resources(), &"first");
    }

    #[test]
    fn typed_attachment_shares_image_generation_and_has_one_lifecycle() {
        let image = PreparedDecoder::<_, _, String>::publish_with_attachment(
            PreparedExecutable::<crate::execution::TransformerStage>::new([], []).unwrap(),
            |attachment| Ok(("resources", attachment)),
        )
        .unwrap();
        let attachment = image.resources().1.clone();

        assert_eq!(attachment.generation(), image.generation());
        assert!(!attachment.is_installed());
        attachment.install("device-image".into()).unwrap();
        assert_eq!(image.attachment().get().unwrap(), "device-image");
        assert!(attachment.install("duplicate".into()).is_err());

        image.shutdown();
        assert!(attachment.get().is_err());
        assert!(image.attachment().as_ref().is_none());
    }

    #[test]
    fn unshared_attachment_can_be_taken_without_a_second_copy() {
        let attachment = PreparedDecoderAttachment::new(7);
        attachment.install(vec![1, 2, 3]).unwrap();
        assert_eq!(attachment.take().unwrap(), vec![1, 2, 3]);
    }
}
