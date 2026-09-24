//! Model operator composition and adapters over backend semantic providers.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "cuda")]
use std::fmt;
use std::sync::Arc;

use ferrule_backend::cpu::{
    self, CpuExecutionPrecision, CpuOperatorProvider, LinearRef as CpuLinearRef,
    LinearWeight as CpuLinearWeight, NativeCpuProvider, PagedCausalGqa,
    PagedKvHistory as BackendPagedKvHistory, RopeRef as CpuRopeRef,
    RotaryPairing as CpuRotaryPairing, RotaryRegion as CpuRotaryRegion, SwiGluRef as CpuSwiGluRef,
};
#[cfg(feature = "cuda")]
use ferrule_backend::cuda::operators::linear::{CudaBf16Buffer, CudaF32Buffer, CudaOperators};
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, Result};

use crate::checkpoint::LinearWeightFormat;
use crate::execution::ExecutionPrecisionPolicy;

use super::expert_parallel::{ExpertDispatchContext, ExpertResult, ExpertTokenBucket};
use super::{
    MoeRouterSpec, PreparedEmbedding, PreparedLinear, PreparedNorm, PreparedRope, RotaryEmbedding,
    RotaryPairing, RotaryRegion, RotaryScaling, RouterScoreFunction, RouterSelection,
};

pub use ferrule_backend::cpu::{HostRows, RouterRoutes, RowsArenaId, RowsDType, RowsShape};

/// Device that owns row storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowsDevice {
    Host,
    #[cfg(feature = "cuda")]
    Cuda {
        ordinal: u32,
    },
}

/// CUDA-owned dense rows.
#[cfg(feature = "cuda")]
pub struct CudaRows {
    shape: RowsShape,
    dtype: RowsDType,
    arena: Option<RowsArenaId>,
    storage: CudaRowsStorage,
}

#[cfg(feature = "cuda")]
enum CudaRowsStorage {
    F32(CudaF32Buffer),
    Bf16(CudaBf16Buffer),
}

#[cfg(feature = "cuda")]
impl fmt::Debug for CudaRows {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaRows")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("arena", &self.arena)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
impl CudaRows {
    pub fn f32(
        shape: RowsShape,
        arena: Option<RowsArenaId>,
        buffer: CudaF32Buffer,
    ) -> Result<Self> {
        if buffer.len() != shape.elements() {
            return Err(model_error(format!(
                "CUDA F32 rows contain {} values for shape [{}, {}]",
                buffer.len(),
                shape.rows(),
                shape.width()
            )));
        }
        Ok(Self {
            shape,
            dtype: RowsDType::F32,
            arena,
            storage: CudaRowsStorage::F32(buffer),
        })
    }

    pub fn bf16(
        shape: RowsShape,
        arena: Option<RowsArenaId>,
        buffer: CudaBf16Buffer,
    ) -> Result<Self> {
        if buffer.len() != shape.elements() {
            return Err(model_error(format!(
                "CUDA BF16 rows contain {} values for shape [{}, {}]",
                buffer.len(),
                shape.rows(),
                shape.width()
            )));
        }
        Ok(Self {
            shape,
            dtype: RowsDType::Bf16,
            arena,
            storage: CudaRowsStorage::Bf16(buffer),
        })
    }

    pub const fn shape(&self) -> RowsShape {
        self.shape
    }

    pub const fn dtype(&self) -> RowsDType {
        self.dtype
    }

    pub const fn arena(&self) -> Option<RowsArenaId> {
        self.arena
    }

    pub fn device_ordinal(&self) -> u32 {
        match &self.storage {
            CudaRowsStorage::F32(buffer) => buffer.as_device_buffer().context().ordinal() as u32,
            CudaRowsStorage::Bf16(buffer) => buffer.as_device_buffer().context().ordinal() as u32,
        }
    }

    /// Ordinals alone do not identify an allocator/stream owner.
    pub fn validate_owner(&self, operators: &CudaOperators) -> Result<()> {
        let stream = operators.stream_clone();
        let matches = match &self.storage {
            CudaRowsStorage::F32(buffer) => {
                Arc::ptr_eq(buffer.as_device_buffer().context(), stream.context())
            }
            CudaRowsStorage::Bf16(buffer) => {
                Arc::ptr_eq(buffer.as_device_buffer().context(), stream.context())
            }
        };
        if !matches {
            return Err(model_error(
                "CUDA rows belong to a different operator owner",
            ));
        }
        Ok(())
    }

    pub fn f32_buffer(&self) -> Option<&CudaF32Buffer> {
        match &self.storage {
            CudaRowsStorage::F32(buffer) => Some(buffer),
            CudaRowsStorage::Bf16(_) => None,
        }
    }

    pub fn f32_buffer_mut(&mut self) -> Option<&mut CudaF32Buffer> {
        match &mut self.storage {
            CudaRowsStorage::F32(buffer) => Some(buffer),
            CudaRowsStorage::Bf16(_) => None,
        }
    }

    pub fn bf16_buffer(&self) -> Option<&CudaBf16Buffer> {
        match &self.storage {
            CudaRowsStorage::F32(_) => None,
            CudaRowsStorage::Bf16(buffer) => Some(buffer),
        }
    }
}

/// Backend-neutral row ownership retained by model operator composition.
#[derive(Debug)]
pub enum Rows {
    Host(HostRows),
    #[cfg(feature = "cuda")]
    Cuda(CudaRows),
}

impl Rows {
    pub const fn shape(&self) -> RowsShape {
        match self {
            Self::Host(rows) => rows.shape(),
            #[cfg(feature = "cuda")]
            Self::Cuda(rows) => rows.shape(),
        }
    }

    pub const fn dtype(&self) -> RowsDType {
        match self {
            Self::Host(rows) => rows.dtype(),
            #[cfg(feature = "cuda")]
            Self::Cuda(rows) => rows.dtype(),
        }
    }

    pub fn device(&self) -> RowsDevice {
        match self {
            Self::Host(_) => RowsDevice::Host,
            #[cfg(feature = "cuda")]
            Self::Cuda(rows) => RowsDevice::Cuda {
                ordinal: rows.device_ordinal(),
            },
        }
    }

    pub const fn arena(&self) -> Option<RowsArenaId> {
        match self {
            Self::Host(rows) => rows.arena(),
            #[cfg(feature = "cuda")]
            Self::Cuda(rows) => rows.arena(),
        }
    }

    pub fn host(&self) -> Result<&HostRows> {
        match self {
            Self::Host(rows) => Ok(rows),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(model_error("CPU operator received CUDA rows")),
        }
    }

    pub fn into_host(self) -> Result<HostRows> {
        match self {
            Self::Host(rows) => Ok(rows),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(model_error("CPU operator received CUDA rows")),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(&self) -> Result<&CudaRows> {
        match self {
            Self::Cuda(rows) => Ok(rows),
            Self::Host(_) => Err(model_error("CUDA operator received host rows")),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn into_cuda(self) -> Result<CudaRows> {
        match self {
            Self::Cuda(rows) => Ok(rows),
            Self::Host(_) => Err(model_error("CUDA operator received host rows")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorWaiting {
    Experts {
        layer: usize,
        experts: Vec<usize>,
    },
    Backend {
        operator: &'static str,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedOperator {
    pub operator: &'static str,
    pub reason: String,
}

impl UnsupportedOperator {
    pub fn new(operator: &'static str, reason: impl Into<String>) -> Self {
        Self {
            operator,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for UnsupportedOperator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.operator, self.reason)
    }
}

impl std::error::Error for UnsupportedOperator {}

pub(crate) fn standard_rope_unsupported(
    descriptor: &RotaryEmbedding,
) -> Option<UnsupportedOperator> {
    matches!(
        descriptor.scaling(),
        RotaryScaling::YaRN {
            attention_factor: Some(factor),
            ..
        } if *factor != 1.0
    )
    .then(|| {
        UnsupportedOperator::new(
            "rope",
            "standard CPU RoPE does not support YaRN attention_factor other than 1",
        )
    })
}

pub(crate) fn standard_router_unsupported(policy: &MoeRouterSpec) -> Option<UnsupportedOperator> {
    (policy.score_function() != RouterScoreFunction::Softmax
        || policy.selection() != &RouterSelection::TopK
        || !policy.normalize_selected()
        || policy.selection_bias())
        .then(|| {
            UnsupportedOperator::new(
                "router",
                "standard router requires Softmax, TopK, selected renormalization, and no selection bias",
            )
        })
}

#[cfg(test)]
mod capability_tests {
    use super::*;

    fn rows() -> Rows {
        Rows::Host(
            HostRows::new(
                RowsShape::new(1, 2).unwrap(),
                RowsDType::F32,
                None,
                vec![2.0, -1.0],
            )
            .unwrap(),
        )
    }

    #[test]
    fn router_runtime_guard_rejects_unsupported_policies_and_keeps_defaults() {
        for precision in [
            ExecutionPrecisionPolicy::f32(),
            ExecutionPrecisionPolicy::bf16_compatibility(),
        ] {
            let mut operators = CpuStandardDecoderOperators::new(precision);
            for (score, selection, normalize, bias) in [
                (
                    RouterScoreFunction::Sigmoid,
                    RouterSelection::TopK,
                    true,
                    false,
                ),
                (
                    RouterScoreFunction::SqrtSoftplus,
                    RouterSelection::TopK,
                    true,
                    false,
                ),
                (
                    RouterScoreFunction::Softmax,
                    RouterSelection::GroupLimitedTopK {
                        groups: 2,
                        selected_groups: 1,
                    },
                    true,
                    false,
                ),
                (
                    RouterScoreFunction::Softmax,
                    RouterSelection::HashAssistedTopK { hash_layers: 1 },
                    true,
                    false,
                ),
                (
                    RouterScoreFunction::Softmax,
                    RouterSelection::TopK,
                    false,
                    false,
                ),
                (
                    RouterScoreFunction::Softmax,
                    RouterSelection::TopK,
                    true,
                    true,
                ),
            ] {
                let policy = MoeRouterSpec::new(2, 1, score, selection, normalize, 1.0)
                    .unwrap()
                    .with_selection_bias(bias);
                let OperatorProgress::Unsupported(unsupported) =
                    operators.router(&rows(), &policy).unwrap()
                else {
                    panic!("unsupported policy executed: {policy:?}")
                };
                assert_eq!(unsupported.operator, "router");
                assert_eq!(
                    Some(unsupported.clone()),
                    standard_router_unsupported(&policy)
                );
                let error = super::super::standard::ready::<()>(OperatorProgress::Unsupported(
                    unsupported.clone(),
                ))
                .unwrap_err();
                let Error::ModelSource { source } = error else {
                    panic!("lost typed unsupported error: {error:?}")
                };
                assert_eq!(
                    source.downcast_ref::<UnsupportedOperator>(),
                    Some(&unsupported)
                );
            }
            let policy = MoeRouterSpec::new(
                2,
                1,
                RouterScoreFunction::Softmax,
                RouterSelection::TopK,
                true,
                1.0,
            )
            .unwrap();
            let OperatorProgress::Ready(routes) = operators.router(&rows(), &policy).unwrap()
            else {
                panic!("default router rejected")
            };
            assert_eq!(routes.row(0).unwrap(), (&[0][..], &[1.0][..]));
        }
    }

    #[test]
    fn rope_runtime_guard_rejects_only_nonunit_explicit_yarn_attention_factor() {
        let table = PreparedRope::new(1, 2, vec![1.0], vec![0.0]).unwrap();
        for precision in [
            ExecutionPrecisionPolicy::f32(),
            ExecutionPrecisionPolicy::bf16_compatibility(),
        ] {
            let mut operators = CpuStandardDecoderOperators::new(precision);
            for factor in [
                None,
                Some(1.0),
                Some(0.5),
                Some(1.1),
                Some(f32::from_bits(1.0f32.to_bits() + 1)),
            ] {
                let descriptor = RotaryEmbedding::new(
                    2,
                    10000.0,
                    RotaryPairing::Interleaved,
                    RotaryRegion::Prefix { dimensions: 2 },
                    RotaryScaling::YaRN {
                        factor: 2.0,
                        original_max_position_embeddings: 16,
                        beta_fast: 32.0,
                        beta_slow: 1.0,
                        attention_factor: factor,
                    },
                )
                .unwrap();
                let result = operators
                    .rope(&descriptor, &table, rows(), 1, &[0])
                    .unwrap();
                if factor.is_some_and(|factor| factor != 1.0) {
                    let OperatorProgress::Unsupported(unsupported) = result else {
                        panic!("unsupported YaRN factor executed: {factor:?}")
                    };
                    assert_eq!(unsupported.operator, "rope");
                    assert!(unsupported.reason.contains("attention_factor"));
                } else {
                    let OperatorProgress::Ready(output) = result else {
                        panic!("supported YaRN factor rejected: {factor:?}")
                    };
                    assert_eq!(output.host().unwrap().values(), &[2.0, -1.0]);
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum OperatorProgress<T> {
    Ready(T),
    Waiting(OperatorWaiting),
    Unsupported(UnsupportedOperator),
}

impl<T> OperatorProgress<T> {
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> OperatorProgress<U> {
        match self {
            Self::Ready(value) => OperatorProgress::Ready(map(value)),
            Self::Waiting(waiting) => OperatorProgress::Waiting(waiting),
            Self::Unsupported(unsupported) => OperatorProgress::Unsupported(unsupported),
        }
    }
}

/// Packed row metadata used by paged GQA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GqaMetadata {
    sequence_count: usize,
    row_sequence_ids: Vec<usize>,
    row_positions: Vec<usize>,
    row_kv_lens: Vec<usize>,
    block_slots: Vec<i32>,
    block_offsets: Vec<i32>,
}

impl GqaMetadata {
    pub fn new(
        sequence_count: usize,
        row_sequence_ids: Vec<usize>,
        row_positions: Vec<usize>,
        row_kv_lens: Vec<usize>,
    ) -> Result<Self> {
        if sequence_count == 0
            || row_sequence_ids.is_empty()
            || row_positions.len() != row_sequence_ids.len()
            || row_kv_lens.len() != row_sequence_ids.len()
            || row_sequence_ids
                .iter()
                .any(|&sequence| sequence >= sequence_count)
        {
            return Err(model_error(format!(
                "invalid packed GQA metadata: sequences={sequence_count} ids={} positions={} kv_lens={}",
                row_sequence_ids.len(),
                row_positions.len(),
                row_kv_lens.len()
            )));
        }
        Ok(Self {
            sequence_count,
            row_sequence_ids,
            row_positions,
            row_kv_lens,
            block_slots: Vec::new(),
            block_offsets: Vec::new(),
        })
    }

    pub fn with_pages(mut self, block_slots: Vec<i32>, block_offsets: Vec<i32>) -> Result<Self> {
        if block_slots.is_empty() || block_offsets.len() != self.sequence_count + 1 {
            return Err(model_error(format!(
                "invalid paged GQA block table: slots={} offsets={}/{}",
                block_slots.len(),
                block_offsets.len(),
                self.sequence_count + 1
            )));
        }
        if block_offsets.first().copied() != Some(0)
            || block_offsets.last().copied() != i32::try_from(block_slots.len()).ok()
            || block_offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(model_error("paged GQA block offsets are not canonical"));
        }
        self.block_slots = block_slots;
        self.block_offsets = block_offsets;
        Ok(self)
    }

    pub const fn sequence_count(&self) -> usize {
        self.sequence_count
    }

    pub fn row_sequence_ids(&self) -> &[usize] {
        &self.row_sequence_ids
    }

    pub fn row_positions(&self) -> &[usize] {
        &self.row_positions
    }

    pub fn row_kv_lens(&self) -> &[usize] {
        &self.row_kv_lens
    }

    pub fn block_slots(&self) -> &[i32] {
        &self.block_slots
    }

    pub fn block_offsets(&self) -> &[i32] {
        &self.block_offsets
    }
}

pub struct GqaRequest<'a> {
    pub layer: usize,
    pub query: &'a Rows,
    pub key: &'a Rows,
    pub value: &'a Rows,
    pub metadata: &'a GqaMetadata,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub softmax_scale: f32,
    pub arena: Option<RowsArenaId>,
}

pub struct KvAppendRequest<'a> {
    pub layer: usize,
    pub metadata: &'a GqaMetadata,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub key: &'a [f32],
    pub value: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvHistory {
    pub tokens: usize,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

pub trait KvView {
    fn append(&mut self, request: KvAppendRequest<'_>) -> Result<()>;

    fn history(
        &self,
        layer: usize,
        sequence: usize,
        through_position: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<KvHistory>;

    #[cfg(feature = "cuda")]
    fn append_and_attend_cuda(
        &mut self,
        _operators: &CudaOperators,
        _request: GqaRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
            "paged_gqa",
            "KV view does not expose CUDA paged storage",
        )))
    }
}

/// Entered transaction view accepted by device-neutral standard segments.
/// CUDA implementations also implement `KvView::append_and_attend_cuda`; that
/// method owns page translation and must never fall back to host attention.
pub trait StandardDecoderKvView: KvView {
    fn transaction(&self) -> ExecutionTransactionId;
    fn validate_batch(&self, batch: &crate::decoder::PackedDecoderBatch) -> Result<()>;
}

impl StandardDecoderKvView for crate::decoder::CpuKvView {
    fn transaction(&self) -> ExecutionTransactionId {
        self.transaction()
    }
    fn validate_batch(&self, batch: &crate::decoder::PackedDecoderBatch) -> Result<()> {
        self.validate_batch(batch)
    }
}

/// Prepared gate/up/down bundle for one dense or routed expert.
#[derive(Debug, Clone)]
pub struct PreparedSwiGlu {
    gate: PreparedLinear,
    up: PreparedLinear,
    down: PreparedLinear,
    activation_limit: Option<f32>,
}

impl PreparedSwiGlu {
    pub fn new(
        gate: PreparedLinear,
        up: PreparedLinear,
        down: PreparedLinear,
        activation_limit: Option<f32>,
    ) -> Result<Self> {
        if gate.in_features() == 0
            || gate.in_features() != up.in_features()
            || gate.out_features() != up.out_features()
            || down.in_features() != gate.out_features()
            || activation_limit.is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
        {
            return Err(model_error(format!(
                "invalid SwiGLU linears: gate=[{},{}] up=[{},{}] down=[{},{}]",
                gate.out_features(),
                gate.in_features(),
                up.out_features(),
                up.in_features(),
                down.out_features(),
                down.in_features()
            )));
        }
        Ok(Self {
            gate,
            up,
            down,
            activation_limit,
        })
    }

    pub const fn gate(&self) -> &PreparedLinear {
        &self.gate
    }

    pub const fn up(&self) -> &PreparedLinear {
        &self.up
    }

    pub const fn down(&self) -> &PreparedLinear {
        &self.down
    }

    pub const fn activation_limit(&self) -> Option<f32> {
        self.activation_limit
    }

    pub fn input_width(&self) -> usize {
        self.gate.in_features()
    }

    pub fn output_width(&self) -> usize {
        self.down.out_features()
    }
}

pub enum ExpertAvailability {
    Ready(Arc<PreparedSwiGlu>),
    Waiting,
    Unsupported(String),
}

pub trait ExpertProvider {
    fn expert(&mut self, layer: usize, expert: usize) -> Result<ExpertAvailability>;
}

/// Packed projected rows for one GatedDeltaNet layer. States are transaction working copies.
pub struct GatedDeltaNetRequest<'a> {
    pub layer: usize,
    pub shape: cpu::gated_delta::GatedDeltaShape,
    pub qkv: &'a Rows,
    pub z: &'a Rows,
    pub a: &'a Rows,
    pub b: &'a Rows,
    pub conv: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a PreparedNorm,
    pub metadata: &'a GqaMetadata,
    pub states: &'a mut [crate::decoder::GatedDeltaStateRef<'a>],
}

/// Model composition surface. Concrete CPU math belongs to `ferrule_backend::cpu`.
pub trait StandardDecoderOperators {
    fn backend_name(&self) -> &'static str;
    fn precision(&self) -> ExecutionPrecisionPolicy;
    /// Attention executes whole local heads; residual width remains replicated.
    fn attention_heads(&self, query: usize, kv: usize) -> Result<(usize, usize)> {
        Ok((query, kv))
    }
    /// Explicit stage/EP boundary, never called between ordinary layer operators.
    fn bind_rows(&mut self, rows: Rows) -> Result<Rows> {
        rows.into_host().map(Rows::Host)
    }
    fn download_rows(&mut self, rows: Rows) -> Result<HostRows> {
        rows.into_host()
    }

    /// Backends opt in explicitly; never run recurrent state on a host fallback.
    fn gated_delta_net(
        &mut self,
        _request: GatedDeltaNetRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
            "gated_delta_net",
            "backend has no GatedDeltaNet implementation",
        )))
    }
    fn unpack_gated_query(
        &mut self,
        _input: Rows,
        _heads: usize,
        _head_dim: usize,
    ) -> Result<OperatorProgress<(Rows, Rows)>> {
        Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
            "gated_query",
            "backend has no head-interleaved query/gate unpack",
        )))
    }
    fn sigmoid_gate(&mut self, _input: Rows, _gate: &Rows) -> Result<OperatorProgress<Rows>> {
        Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
            "sigmoid_gate",
            "backend has no output gate",
        )))
    }

    fn embedding(
        &mut self,
        embedding: &PreparedEmbedding,
        token_ids: &[u32],
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>>;
    fn linear(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>>;
    fn rms_norm(
        &mut self,
        norm: &PreparedNorm,
        input: &Rows,
        heads: usize,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>>;
    fn rope(
        &mut self,
        descriptor: &RotaryEmbedding,
        table: &PreparedRope,
        input: Rows,
        heads: usize,
        positions: &[usize],
    ) -> Result<OperatorProgress<Rows>>;
    fn paged_gqa(
        &mut self,
        kv: &mut dyn KvView,
        request: GqaRequest<'_>,
    ) -> Result<OperatorProgress<Rows>>;
    fn router(
        &mut self,
        logits: &Rows,
        policy: &MoeRouterSpec,
    ) -> Result<OperatorProgress<RouterRoutes>>;
    fn dense_swiglu(
        &mut self,
        expert: &PreparedSwiGlu,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>>;
    fn routed_swiglu(
        &mut self,
        layer: usize,
        input: &Rows,
        routes: &RouterRoutes,
        experts: &mut dyn ExpertProvider,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>>;
    fn residual(&mut self, residual: Rows, update: &Rows) -> Result<OperatorProgress<Rows>>;
    fn lm_head(
        &mut self,
        head: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>>;
}

/// Synchronous expert-result boundary, independent of decoder forward and
/// weight providers. An owned bucket can be moved to another rank's thread or
/// IPC transport; replies contain only unweighted expert outputs and identity.
///
/// The routed executor calls this once per member, including empty buckets.
/// Implementations must return that owner's complete reply or an error, enforce
/// receive allocation bounds/timeouts, and authenticate the owner. The caller
/// validates replies against its immutable plan before combining. There is no
/// transaction lifecycle, retry or partial-publication state in this trait.
pub trait ExpertResultExecutor {
    fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>>;
}

/// The normalized activations and existing router outputs at a routed FFN seam.
/// Layer coordinates are global, never the segment-local KV layer index.
pub struct RoutedSwiGluRequest<'a> {
    pub context: ExpertDispatchContext,
    pub sequences: &'a [u64],
    pub input: &'a Rows,
    pub routes: &'a RouterRoutes,
    pub arena: Option<RowsArenaId>,
}

/// Injectable replacement for only the routed SwiGLU call in shared layer math.
/// Router, shared-expert, residual and dense-layer math stay with the decoder.
/// The existing transaction authority must reject cancelled/unknown identities
/// and serialize cancellation with publication; this trait owns no such state.
pub trait RoutedSwiGluExecutor {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>>;
}

/// Narrow adapter used by expert-parallel workers to call the existing
/// prepared SwiGLU operator without depending on the rest of the decoder API.
pub trait ExpertSwiGluOperator {
    fn apply_prepared_swiglu(
        &mut self,
        expert: &PreparedSwiGlu,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows>;
}

impl<T> ExpertSwiGluOperator for T
where
    T: StandardDecoderOperators + ?Sized,
{
    fn apply_prepared_swiglu(
        &mut self,
        expert: &PreparedSwiGlu,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        match self.dense_swiglu(expert, input, arena)? {
            OperatorProgress::Ready(rows) => Ok(rows),
            OperatorProgress::Waiting(waiting) => Err(model_error(format!(
                "prepared expert is waiting: {waiting:?}"
            ))),
            OperatorProgress::Unsupported(unsupported) => Err(model_error(format!(
                "prepared expert is unsupported: {}: {}",
                unsupported.operator, unsupported.reason
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuStandardDecoderOperators {
    precision: ExecutionPrecisionPolicy,
    provider: NativeCpuProvider,
}

impl CpuStandardDecoderOperators {
    pub const fn new(precision: ExecutionPrecisionPolicy) -> Self {
        Self {
            precision,
            provider: NativeCpuProvider,
        }
    }

    fn cpu_precision(self) -> CpuExecutionPrecision {
        if self.precision == ExecutionPrecisionPolicy::bf16_compatibility() {
            CpuExecutionPrecision::Bf16
        } else {
            CpuExecutionPrecision::F32
        }
    }
}

impl Default for CpuStandardDecoderOperators {
    fn default() -> Self {
        Self::new(ExecutionPrecisionPolicy::f32())
    }
}

impl StandardDecoderOperators for CpuStandardDecoderOperators {
    fn backend_name(&self) -> &'static str {
        self.provider.name()
    }

    fn precision(&self) -> ExecutionPrecisionPolicy {
        self.precision
    }

    fn unpack_gated_query(
        &mut self,
        input: Rows,
        heads: usize,
        head_dim: usize,
    ) -> Result<OperatorProgress<(Rows, Rows)>> {
        if self.precision != ExecutionPrecisionPolicy::f32() {
            return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                "gated_query",
                "CPU gated attention requires F32",
            )));
        }
        let input = input.into_host()?;
        let width = heads
            .checked_mul(head_dim)
            .filter(|&n| n > 0)
            .ok_or_else(|| model_error("gated query shape overflow"))?;
        if width.checked_mul(2) != Some(input.shape().width()) {
            return Err(model_error("gated query width mismatch"));
        }
        let mut query = Vec::with_capacity(input.shape().rows() * width);
        let mut gate = Vec::with_capacity(query.capacity());
        for head in input.values().chunks_exact(2 * head_dim) {
            query.extend_from_slice(&head[..head_dim]);
            gate.extend_from_slice(&head[head_dim..]);
        }
        let shape = RowsShape::new(input.shape().rows(), width)?;
        Ok(OperatorProgress::Ready((
            Rows::Host(HostRows::new(shape, RowsDType::F32, input.arena(), query)?),
            Rows::Host(HostRows::new(shape, RowsDType::F32, input.arena(), gate)?),
        )))
    }
    fn sigmoid_gate(&mut self, input: Rows, gate: &Rows) -> Result<OperatorProgress<Rows>> {
        if self.precision != ExecutionPrecisionPolicy::f32() {
            return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                "sigmoid_gate",
                "CPU gated attention requires F32",
            )));
        }
        let mut input = input.into_host()?;
        let gate = gate.host()?;
        if input.shape() != gate.shape() {
            return Err(model_error("output gate shape mismatch"));
        }
        for (x, g) in input.values_mut().iter_mut().zip(gate.values()) {
            *x *= cpu::gated_delta::sigmoid(*g);
        }
        Ok(OperatorProgress::Ready(Rows::Host(input)))
    }
    fn gated_delta_net(
        &mut self,
        request: GatedDeltaNetRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        if self.precision != ExecutionPrecisionPolicy::f32() {
            return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                "gated_delta_net",
                "CPU GatedDeltaNet requires F32",
            )));
        }
        let qkv = request.qkv.host()?;
        let z = request.z.host()?;
        let a = request.a.host()?;
        let b = request.b.host()?;
        let (channels, _, _) = request.shape.sizes()?;
        let width = request.shape.value_heads * request.shape.value_dim;
        let rows = request.metadata.row_positions().len();
        if qkv.shape() != RowsShape::new(rows, channels)?
            || z.shape() != RowsShape::new(rows, width)?
            || a.shape() != RowsShape::new(rows, request.shape.value_heads)?
            || a.shape() != b.shape()
            || request.states.len() != request.metadata.sequence_count()
        {
            return Err(model_error(
                "packed GatedDeltaNet projection/state shape mismatch",
            ));
        }
        // Validate every sequence frontier before the first state write.
        let mut positions = request
            .states
            .iter_mut()
            .map(|s| s.cpu().map(|s| s.position))
            .collect::<Result<Vec<_>>>()?;
        for (row, &sequence) in request.metadata.row_sequence_ids().iter().enumerate() {
            if request.states[sequence].cpu()?.shape != request.shape
                || positions[sequence] != request.metadata.row_positions()[row]
            {
                return Err(model_error(
                    "GatedDeltaNet packed sequence frontier mismatch",
                ));
            }
            positions[sequence] = positions[sequence]
                .checked_add(1)
                .ok_or_else(|| model_error("recurrent position overflow"))?;
        }
        let mut output = Vec::with_capacity(rows * width);
        for (row, &sequence) in request.metadata.row_sequence_ids().iter().enumerate() {
            let state = request.states[sequence].cpu()?;
            fn row_slice(input: &HostRows, row: usize) -> &[f32] {
                &input.values()[row * input.shape().width()..(row + 1) * input.shape().width()]
            }
            output.extend(cpu::gated_delta::gated_delta_step(
                request.shape,
                cpu::gated_delta::GatedDeltaWeights {
                    conv: request.conv,
                    a_log: request.a_log,
                    dt_bias: request.dt_bias,
                    norm: request.norm.weight(),
                    norm_epsilon: request.norm.epsilon(),
                },
                cpu::gated_delta::GatedDeltaInput {
                    qkv: row_slice(qkv, row),
                    z: row_slice(z, row),
                    a: row_slice(a, row),
                    b: row_slice(b, row),
                },
                &mut state.conv_history,
                &mut state.recurrent,
            )?);
            state.position += 1;
        }
        Ok(OperatorProgress::Ready(Rows::Host(HostRows::new(
            RowsShape::new(rows, width)?,
            RowsDType::F32,
            None,
            output,
        )?)))
    }

    fn embedding(
        &mut self,
        embedding: &PreparedEmbedding,
        token_ids: &[u32],
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        let linear = CpuLinearAdapter::new(embedding.linear())?;
        Ok(OperatorProgress::Ready(Rows::Host(
            self.provider
                .embedding(linear.as_ref(), token_ids, arena, self.cpu_precision())?,
        )))
    }

    fn linear(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        let linear = CpuLinearAdapter::new(linear)?;
        Ok(OperatorProgress::Ready(Rows::Host(self.provider.linear(
            linear.as_ref(),
            input.host()?,
            arena,
            self.cpu_precision(),
        )?)))
    }

    fn rms_norm(
        &mut self,
        norm: &PreparedNorm,
        input: &Rows,
        heads: usize,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        Ok(OperatorProgress::Ready(Rows::Host(
            self.provider.rms_norm(
                input.host()?,
                norm.weight(),
                norm.epsilon(),
                heads,
                arena,
                self.cpu_precision(),
            )?,
        )))
    }

    fn rope(
        &mut self,
        descriptor: &RotaryEmbedding,
        table: &PreparedRope,
        input: Rows,
        heads: usize,
        positions: &[usize],
    ) -> Result<OperatorProgress<Rows>> {
        if let Some(unsupported) = standard_rope_unsupported(descriptor) {
            return Ok(OperatorProgress::Unsupported(unsupported));
        }
        let pairing = match descriptor.pairing() {
            RotaryPairing::SplitHalf => CpuRotaryPairing::SplitHalf,
            RotaryPairing::Interleaved => CpuRotaryPairing::Interleaved,
        };
        let region = match descriptor.region() {
            RotaryRegion::Prefix { .. } => CpuRotaryRegion::Prefix,
            RotaryRegion::Tail { .. } => CpuRotaryRegion::Tail,
        };
        Ok(OperatorProgress::Ready(Rows::Host(self.provider.rope(
            input.into_host()?,
            CpuRopeRef {
                positions: table.positions(),
                dimensions: table.dimensions(),
                cosine: table.cosine(),
                sine: table.sine(),
            },
            pairing,
            region,
            heads,
            descriptor.head_dim(),
            positions,
            self.cpu_precision(),
        )?)))
    }

    fn paged_gqa(
        &mut self,
        kv: &mut dyn KvView,
        request: GqaRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        let query = request.query.host()?;
        let key = request.key.host()?;
        let value = request.value.host()?;
        let mut key_values = key.values().to_vec();
        let mut value_values = value.values().to_vec();
        self.cpu_precision().apply_slice(&mut key_values);
        self.cpu_precision().apply_slice(&mut value_values);
        kv.append(KvAppendRequest {
            layer: request.layer,
            metadata: request.metadata,
            kv_heads: request.kv_heads,
            head_dim: request.head_dim,
            key: &key_values,
            value: &value_values,
        })?;
        let history = KvHistoryAdapter(kv);
        Ok(OperatorProgress::Ready(Rows::Host(
            self.provider.paged_gqa(
                &history,
                request.layer,
                PagedCausalGqa {
                    query,
                    row_sequence_ids: request.metadata.row_sequence_ids(),
                    row_positions: request.metadata.row_positions(),
                    query_heads: request.query_heads,
                    kv_heads: request.kv_heads,
                    head_dim: request.head_dim,
                    softmax_scale: request.softmax_scale,
                    arena: request.arena,
                },
                self.cpu_precision(),
            )?,
        )))
    }

    fn router(
        &mut self,
        logits: &Rows,
        policy: &MoeRouterSpec,
    ) -> Result<OperatorProgress<RouterRoutes>> {
        if let Some(unsupported) = standard_router_unsupported(policy) {
            return Ok(OperatorProgress::Unsupported(unsupported));
        }
        if logits.shape().width() != policy.num_experts() {
            return Err(model_error(format!(
                "router logits width {} does not match {} experts",
                logits.shape().width(),
                policy.num_experts()
            )));
        }
        Ok(OperatorProgress::Ready(self.provider.router(
            logits.host()?,
            policy.experts_per_token(),
            policy.route_scale(),
            self.cpu_precision(),
        )?))
    }

    fn dense_swiglu(
        &mut self,
        expert: &PreparedSwiGlu,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        let expert = CpuSwiGluAdapter::new(expert)?;
        Ok(OperatorProgress::Ready(Rows::Host(self.provider.swiglu(
            expert.as_ref(),
            input.host()?,
            arena,
            self.cpu_precision(),
        )?)))
    }

    fn routed_swiglu(
        &mut self,
        layer: usize,
        input: &Rows,
        routes: &RouterRoutes,
        experts: &mut dyn ExpertProvider,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        let input = input.host()?;
        if routes.rows() != input.shape().rows() {
            return Err(model_error("routed SwiGLU row count mismatch"));
        }
        let selected = routes.expert_ids().iter().copied().collect::<BTreeSet<_>>();
        let mut prepared = BTreeMap::new();
        let mut waiting = Vec::new();
        for expert in selected {
            match experts.expert(layer, expert)? {
                ExpertAvailability::Ready(value) => {
                    prepared.insert(expert, value);
                }
                ExpertAvailability::Waiting => waiting.push(expert),
                ExpertAvailability::Unsupported(reason) => {
                    return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                        "routed_swiglu",
                        reason,
                    )));
                }
            }
        }
        if !waiting.is_empty() {
            return Ok(OperatorProgress::Waiting(OperatorWaiting::Experts {
                layer,
                experts: waiting,
            }));
        }

        let mut output = vec![0.0; input.values().len()];
        for row in 0..routes.rows() {
            let row_input = HostRows::new(
                RowsShape::new(1, input.shape().width())?,
                input.dtype(),
                arena,
                input.values()[row * input.shape().width()..(row + 1) * input.shape().width()]
                    .to_vec(),
            )?;
            let (expert_ids, weights) = routes.row(row)?;
            for (&expert_id, &route_weight) in expert_ids.iter().zip(weights) {
                let expert = CpuSwiGluAdapter::new(
                    prepared
                        .get(&expert_id)
                        .expect("all selected experts were prepared"),
                )?;
                let expert_output = self.provider.swiglu(
                    expert.as_ref(),
                    &row_input,
                    arena,
                    self.cpu_precision(),
                )?;
                if expert_output.shape().width() != input.shape().width() {
                    return Err(model_error("routed expert output width mismatch"));
                }
                let start = row * input.shape().width();
                self.provider.weighted_reduce(
                    &mut output[start..start + input.shape().width()],
                    expert_output.values(),
                    route_weight,
                    self.cpu_precision(),
                )?;
            }
        }
        Ok(OperatorProgress::Ready(Rows::Host(HostRows::new(
            input.shape(),
            self.cpu_precision().rows_dtype(),
            arena,
            output,
        )?)))
    }

    fn residual(&mut self, residual: Rows, update: &Rows) -> Result<OperatorProgress<Rows>> {
        Ok(OperatorProgress::Ready(Rows::Host(
            self.provider
                .residual(residual.into_host()?, update.host()?, self.cpu_precision())?,
        )))
    }

    fn lm_head(
        &mut self,
        head: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.linear(head, input, arena)
    }
}

struct KvHistoryAdapter<'a>(&'a dyn KvView);

impl BackendPagedKvHistory for KvHistoryAdapter<'_> {
    fn history(
        &self,
        layer: usize,
        sequence: usize,
        through_position: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<cpu::KvHistory> {
        let history = self
            .0
            .history(layer, sequence, through_position, kv_heads, head_dim)?;
        Ok(cpu::KvHistory {
            tokens: history.tokens,
            key: history.key,
            value: history.value,
        })
    }
}

enum CpuWeightStorage<'a> {
    F32(Cow<'a, [f32]>),
    Bf16(&'a [u8]),
}

struct CpuLinearAdapter<'a> {
    storage: CpuWeightStorage<'a>,
    out_features: usize,
    in_features: usize,
    bias: Option<&'a [f32]>,
}

impl<'a> CpuLinearAdapter<'a> {
    fn new(linear: &'a PreparedLinear) -> Result<Self> {
        let weight = linear.weight()?;
        let (storage, out_features, in_features) = match weight.format {
            LinearWeightFormat::Bf16 {
                out_features,
                in_features,
            } => (
                CpuWeightStorage::Bf16(&weight.weight.bytes),
                out_features,
                in_features,
            ),
            _ => (
                CpuWeightStorage::F32(Cow::Owned(weight.reference_weights_f32()?)),
                linear.out_features(),
                linear.in_features(),
            ),
        };
        Ok(Self {
            storage,
            out_features,
            in_features,
            bias: linear.bias(),
        })
    }

    fn as_ref(&self) -> CpuLinearRef<'_> {
        CpuLinearRef {
            weight: match &self.storage {
                CpuWeightStorage::F32(values) => CpuLinearWeight::F32(values),
                CpuWeightStorage::Bf16(bytes) => CpuLinearWeight::Bf16(bytes),
            },
            out_features: self.out_features,
            in_features: self.in_features,
            bias: self.bias,
        }
    }
}

struct CpuSwiGluAdapter<'a> {
    gate: CpuLinearAdapter<'a>,
    up: CpuLinearAdapter<'a>,
    down: CpuLinearAdapter<'a>,
    activation_limit: Option<f32>,
}

impl<'a> CpuSwiGluAdapter<'a> {
    fn new(expert: &'a PreparedSwiGlu) -> Result<Self> {
        Ok(Self {
            gate: CpuLinearAdapter::new(expert.gate())?,
            up: CpuLinearAdapter::new(expert.up())?,
            down: CpuLinearAdapter::new(expert.down())?,
            activation_limit: expert.activation_limit(),
        })
    }

    fn as_ref(&self) -> CpuSwiGluRef<'_> {
        CpuSwiGluRef {
            gate: self.gate.as_ref(),
            up: self.up.as_ref(),
            down: self.down.as_ref(),
            activation_limit: self.activation_limit,
        }
    }
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("standard decoder operators: {}", message.into()),
    }
}
