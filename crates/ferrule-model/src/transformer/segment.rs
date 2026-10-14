//! Immutable standard-decoder segments; transport and KV custody belong to the caller.

use std::ops::Range;

use ferrule_common::ParallelRankId;
use ferrule_common::execution::ExecutionTransactionId;

use super::expert_parallel::RoutedSwiGluExecutor;

use crate::decoder::{CpuKvView, DenseLogits, PackedDecoderBatch};
use crate::execution::ExecutionPrecisionPolicy;
use crate::nn::ParameterResidency;
use crate::support::TensorRole;

use super::standard::{
    LayerOperators, RoutedLayerExecution, execute_standard_layer, execute_standard_output_rows,
    packed_gqa_metadata, prepare_embedding, prepare_layer, prepare_output, ready,
    validate_standard_descriptors,
};
use super::{
    BoundDecoderResources, BoundParameter, CpuStandardDecoderOperators, CpuTransformerHidden,
    HostRows, MemoryLayerWeightCache, PreparedCpuOutput, PreparedEmbedding, PreparedGqaMoeLayer,
    Rows, RowsDType, StandardDecoderKvView, StandardDecoderOperators, StateDictMaterializer,
};

pub type SegmentResult<T> = Result<T, SegmentError>;

/// Whole-layer, non-empty contiguous ownership in global decoder coordinates.
/// Endpoint operations are optional for an individual segment. A complete
/// pipeline must additionally pass [`Self::validate_pipeline`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSegmentPlan {
    total_layers: usize,
    layers: Range<usize>,
    embedding: bool,
    output: bool,
}

impl LayerSegmentPlan {
    pub fn new(
        total_layers: usize,
        layers: Range<usize>,
        embedding: bool,
        output: bool,
    ) -> SegmentResult<Self> {
        if layers.start >= layers.end || layers.end > total_layers {
            return Err(SegmentError::InvalidRange {
                total_layers,
                layers,
            });
        }
        if embedding && layers.start != 0 {
            return Err(SegmentError::EmbeddingOwnership);
        }
        if output && layers.end != total_layers {
            return Err(SegmentError::OutputOwnership);
        }
        Ok(Self {
            total_layers,
            layers,
            embedding,
            output,
        })
    }

    /// Rejects holes, overlap, reordering, mixed layer counts, and missing endpoints.
    /// Each layer must be owned exactly once, never split between segments.
    pub fn validate_pipeline(plans: &[Self]) -> SegmentResult<()> {
        let Some(first) = plans.first() else {
            return Err(SegmentError::InvalidPipeline { segment: 0 });
        };
        let mut next = 0;
        for (index, plan) in plans.iter().enumerate() {
            if plan.total_layers != first.total_layers || plan.layers.start != next {
                return Err(SegmentError::InvalidPipeline { segment: index });
            }
            if plan.embedding != (index == 0) {
                return Err(SegmentError::EmbeddingOwnership);
            }
            if plan.output != (index + 1 == plans.len()) {
                return Err(SegmentError::OutputOwnership);
            }
            next = plan.layers.end;
        }
        if next != first.total_layers {
            return Err(SegmentError::InvalidPipeline {
                segment: plans.len() - 1,
            });
        }
        Ok(())
    }

    pub const fn total_layers(&self) -> usize {
        self.total_layers
    }
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
    pub const fn owns_embedding(&self) -> bool {
        self.embedding
    }
    pub const fn owns_output(&self) -> bool {
        self.output
    }

    /// Local decoder coordinate. Only standard all-GQA segments also use this
    /// as a KV plane index; hybrid segments must use their schema mapping.
    pub fn local_layer(&self, global: usize) -> Option<usize> {
        self.layers
            .contains(&global)
            .then(|| global - self.layers.start)
    }

    pub fn global_layer(&self, local: usize) -> Option<usize> {
        (local < self.layer_count()).then(|| self.layers.start + local)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentStage {
    Prepare,
    Input,
    Embedding,
    Layer { global: usize, local: usize },
    Output,
}

#[derive(Debug, thiserror::Error)]
pub enum SegmentError {
    #[error("segment range {layers:?} is not a non-empty subset of 0..{total_layers}")]
    InvalidRange {
        total_layers: usize,
        layers: Range<usize>,
    },
    #[error("only the first segment may own embedding; a full pipeline must own it")]
    EmbeddingOwnership,
    #[error("only the last segment may own output; a full pipeline must own it")]
    OutputOwnership,
    #[error("pipeline is not an exact ordered layer partition at segment {segment}")]
    InvalidPipeline { segment: usize },
    #[error("segment plan describes {planned} layers, checkpoint describes {actual}")]
    LayerCount { planned: usize, actual: usize },
    #[error("invalid segment position limit {max_positions}")]
    PositionLimit { max_positions: usize },
    #[error("position {position} exceeds the prepared range 0..{max_positions}")]
    Position {
        position: usize,
        max_positions: usize,
    },
    #[error("segment supports only F32 or BF16 compatibility precision")]
    Precision,
    #[error("injected expert execution requires F32 segment precision")]
    ExpertPrecision,
    #[error("expert transaction {actual:?} does not match the KV transaction {expected:?}")]
    ExpertTransaction {
        expected: ExecutionTransactionId,
        actual: ExecutionTransactionId,
    },
    #[error(
        "expert context has {actual} sequence identities, expected {expected} packed sequences"
    )]
    ExpertSequenceCount { expected: usize, actual: usize },
    #[error("hidden activation targets layer {actual}, expected {expected}")]
    HiddenBoundary { expected: usize, actual: usize },
    #[error("hidden rows must have shape [{rows}, {width}] and dtype {dtype:?}")]
    HiddenLayout {
        rows: usize,
        width: usize,
        dtype: RowsDType,
    },
    #[error("segment {stage:?} failed: {source}")]
    Execution {
        stage: SegmentStage,
        #[source]
        source: ferrule_common::Error,
    },
}

impl SegmentError {
    fn at(stage: SegmentStage) -> impl Fn(ferrule_common::Error) -> Self + Copy {
        move |source| Self::Execution { stage, source }
    }
}

/// Tokens come from the validated batch. Hidden rows are in that same packed
/// order; the caller must bind transported activations to the correct batch.
#[derive(Debug)]
pub enum SegmentInput {
    Tokens,
    Hidden { next_layer: usize, rows: HostRows },
}

/// Host-owned values can be moved through an external transport as `Vec<f32>`.
/// BF16 compatibility rows retain their exact rounded F32 values and dtype tag.
#[derive(Debug)]
pub enum SegmentOutput {
    Hidden { next_layer: usize, rows: HostRows },
    Logits(DenseLogits),
}

/// Device-neutral stage input. A same-owner CUDA hidden tensor stays resident.
#[derive(Debug)]
pub enum DeviceSegmentInput {
    Tokens,
    Hidden { next_layer: usize, rows: Rows },
}

impl From<SegmentInput> for DeviceSegmentInput {
    fn from(input: SegmentInput) -> Self {
        match input {
            SegmentInput::Tokens => Self::Tokens,
            SegmentInput::Hidden { next_layer, rows } => Self::Hidden {
                next_layer,
                rows: Rows::Host(rows),
            },
        }
    }
}

#[derive(Debug)]
pub enum DeviceSegmentOutput {
    Hidden { next_layer: usize, rows: Rows },
    Logits(Rows),
}

impl DeviceSegmentOutput {
    /// Explicit PP/logits boundary; never used inside the layer loop.
    pub fn into_host(
        self,
        operators: &mut dyn StandardDecoderOperators,
    ) -> SegmentResult<SegmentOutput> {
        let stage = SegmentError::at(SegmentStage::Output);
        match self {
            Self::Hidden { next_layer, rows } => Ok(SegmentOutput::Hidden {
                next_layer,
                rows: operators.download_rows(rows).map_err(stage)?,
            }),
            Self::Logits(rows) => {
                let rows = operators.download_rows(rows).map_err(stage)?;
                DenseLogits::new(
                    rows.shape().rows(),
                    rows.shape().width(),
                    rows.into_values(),
                )
                .map(SegmentOutput::Logits)
                .map_err(SegmentError::at(SegmentStage::Output))
            }
        }
    }
}

/// Prepared CPU GQA/Add/SwiGLU/MoE segment. No full resource directory, runner,
/// communication object, sequence cursor, or KV transaction is retained.
#[derive(Debug)]
pub struct StandardDecoderSegment {
    plan: LayerSegmentPlan,
    hidden_size: usize,
    max_positions: usize,
    precision: ExecutionPrecisionPolicy,
    materializer: StateDictMaterializer,
    parameters: Box<[BoundParameter]>,
    embedding: Option<PreparedEmbedding>,
    layers: Box<[PreparedGqaMoeLayer]>,
    output: Option<PreparedCpuOutput>,
}

impl StandardDecoderSegment {
    /// Borrows the checkpoint only during preparation. Layer and expert bindings
    /// remain global, but only selected layers are retained/materialized. Routed
    /// expert payloads remain lazy. The private materializer cannot retain another
    /// segment's static cache. A tied output retains its canonical embedding source
    /// without owning the embedding operation. Unsupported output, RoPE and router
    /// descriptors are rejected before reading weights, with an `UnsupportedOperator`
    /// source inside `SegmentError::Execution` at `SegmentStage::Prepare`.
    pub fn prepare(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        precision: ExecutionPrecisionPolicy,
        max_positions: usize,
        max_parameter_bytes: u64,
    ) -> SegmentResult<Self> {
        let materializer = StateDictMaterializer::new(max_parameter_bytes)
            .map_err(SegmentError::at(SegmentStage::Prepare))?;
        Self::prepare_with_materializer(resources, plan, precision, max_positions, materializer)
    }

    /// Prepare the same global decoder graph with rank-local projection payloads.
    /// This is preparation only; CPU execution cannot consume TP linears without
    /// rank-local operators and a collective. CUDA uses this same composition.
    pub fn prepare_tensor(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        tensor: super::StandardTensorPlan,
        rank: ParallelRankId,
        max_positions: usize,
        max_parameter_bytes: u64,
    ) -> SegmentResult<Self> {
        tensor
            .validate_segment(resources.spec(), &plan)
            .and_then(|_| tensor.validate_resources(resources))
            .map_err(SegmentError::at(SegmentStage::Prepare))?;
        let materializer = StateDictMaterializer::for_tensor(max_parameter_bytes, tensor, rank)
            .map_err(SegmentError::at(SegmentStage::Prepare))?;
        Self::prepare_with_materializer(
            resources,
            plan,
            ExecutionPrecisionPolicy::f32(),
            max_positions,
            materializer,
        )
    }

    pub fn tensor_reads(
        &self,
    ) -> ferrule_common::Result<Vec<super::parallel::TensorParallelPreparationRead>> {
        self.materializer.tensor_reads()
    }

    fn prepare_with_materializer(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        precision: ExecutionPrecisionPolicy,
        max_positions: usize,
        materializer: StateDictMaterializer,
    ) -> SegmentResult<Self> {
        let spec = resources.spec();
        if plan.total_layers != spec.layers().len() {
            return Err(SegmentError::LayerCount {
                planned: plan.total_layers,
                actual: spec.layers().len(),
            });
        }
        if max_positions == 0
            || spec
                .max_sequence_length()
                .is_some_and(|max| max_positions > max)
        {
            return Err(SegmentError::PositionLimit { max_positions });
        }
        if precision != ExecutionPrecisionPolicy::f32()
            && precision != ExecutionPrecisionPolicy::bf16_compatibility()
        {
            return Err(SegmentError::Precision);
        }
        crate::moe::catalog::ValidatedExpertCatalog::for_decoder_range(resources, plan.layers())
            .map_err(SegmentError::at(SegmentStage::Prepare))?;
        let descriptors = &spec.layers()[plan.layers()];
        validate_standard_descriptors(spec, descriptors)
            .map_err(SegmentError::at(SegmentStage::Prepare))?;
        let embedding = plan
            .embedding
            .then(|| prepare_embedding(resources, &materializer))
            .transpose()
            .map_err(SegmentError::at(SegmentStage::Embedding))?;
        let output = plan
            .output
            .then(|| prepare_output(resources, &materializer))
            .transpose()
            .map_err(SegmentError::at(SegmentStage::Output))?;
        let layers = descriptors
            .iter()
            .enumerate()
            .map(|(local, descriptor)| {
                prepare_layer(
                    descriptor,
                    resources,
                    &materializer,
                    &mut MemoryLayerWeightCache::new(),
                    max_positions,
                )
                .map_err(SegmentError::at(SegmentStage::Layer {
                    global: descriptor.index(),
                    local,
                }))
            })
            .collect::<SegmentResult<Vec<_>>>()?
            .into_boxed_slice();
        let parameters = resources
            .state_dict()
            .parameters()
            .iter()
            .filter(|parameter| match parameter.residency() {
                ParameterResidency::Layer { layer } | ParameterResidency::Expert { layer, .. } => {
                    plan.layers.contains(layer)
                }
                ParameterResidency::Static => match parameter.role() {
                    TensorRole::TokenEmbedding => plan.embedding,
                    TensorRole::OutputNorm | TensorRole::OutputHead => plan.output,
                    _ => false,
                },
                ParameterResidency::Attachment { .. } => false,
            })
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            plan,
            hidden_size: spec.hidden_size(),
            max_positions,
            precision,
            materializer,
            parameters,
            embedding,
            layers,
            output,
        })
    }

    #[cfg(feature = "cuda")]
    pub(super) fn prepare_cuda_bindings(
        &self,
        operators: &mut super::standard::cuda::CudaStandardDecoderOperators,
    ) -> ferrule_common::Result<()> {
        operators.prepare_image(self.embedding.as_ref(), &self.layers, self.output.as_ref())
    }

    pub const fn plan(&self) -> &LayerSegmentPlan {
        &self.plan
    }

    /// Logical dependencies, including lazy experts and aliases with unchanged
    /// canonical IDs. No parameters from unowned layers or attachments are exposed.
    pub fn parameters(&self) -> &[BoundParameter] {
        &self.parameters
    }

    /// Synchronous, whole-layer execution with segment-local CPU KV planes.
    /// The caller supplies a validated, entered KV view and owns page allocation,
    /// continuation positions, publication, rollback and cancellation between calls.
    /// On a layer/output error KV may already have been written: the caller must
    /// discard/roll back this call before retrying. No partial logits are returned.
    pub fn execute(
        &self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        kv: &mut CpuKvView,
    ) -> SegmentResult<SegmentOutput> {
        let mut operators = CpuStandardDecoderOperators::new(self.precision);
        self.execute_bound(batch, input.into(), kv, &mut operators)?
            .into_host(&mut operators)
    }

    /// Replace only routed SwiGLU with caller-owned expert execution. Dense,
    /// router, shared-expert, normalization and residual math are unchanged.
    ///
    /// `context` is `(transaction, source_rank, sequence_ids)`. The transaction
    /// must be the existing KV operation ID, not a newly allocated EP epoch.
    /// IDs are in `batch.sequences()` order (not state-slot or token-row order);
    /// `row_to_sequence()` expands them to stable per-row identities. The caller
    /// must preserve these identities across PP stages and decode calls and must
    /// not reuse a transaction/layer/source-row identity for different work.
    ///
    /// Currently F32 only: BF16 is rejected before embedding, KV writes or dispatch,
    /// never silently executed locally. Liveness is checked at entry, layer and
    /// output boundaries as well as by the injected executor. Cancellation and
    /// publication must still be serialized by the caller; errors can require KV
    /// rollback. No executor, transport or transaction lifecycle is retained here.
    pub fn execute_with_experts(
        &self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        kv: &mut CpuKvView,
        context: (ExecutionTransactionId, ParallelRankId, &[u64]),
        executor: &mut dyn RoutedSwiGluExecutor,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> ferrule_common::Result<()>,
    ) -> SegmentResult<SegmentOutput> {
        let mut operators = CpuStandardDecoderOperators::new(self.precision);
        self.execute_bound_with_experts(
            batch,
            input.into(),
            kv,
            &mut operators,
            context,
            executor,
            check_active,
        )?
        .into_host(&mut operators)
    }

    pub fn execute_bound(
        &self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        kv: &mut dyn StandardDecoderKvView,
        operators: &mut dyn StandardDecoderOperators,
    ) -> SegmentResult<DeviceSegmentOutput> {
        self.execute_with_layer_operators(batch, input, kv, LayerOperators::Standard(operators))
    }

    pub(super) fn execute_with_layer_operators(
        &self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        kv: &mut dyn StandardDecoderKvView,
        operators: LayerOperators<'_>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        self.execute_inner(batch, input, kv, operators, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_bound_with_experts(
        &self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        kv: &mut dyn StandardDecoderKvView,
        operators: &mut dyn StandardDecoderOperators,
        context: (ExecutionTransactionId, ParallelRankId, &[u64]),
        executor: &mut dyn RoutedSwiGluExecutor,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> ferrule_common::Result<()>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        self.execute_with_experts_and_layer_operators(
            batch,
            input,
            kv,
            LayerOperators::Standard(operators),
            context,
            executor,
            check_active,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_with_experts_and_layer_operators(
        &self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        kv: &mut dyn StandardDecoderKvView,
        operators: LayerOperators<'_>,
        context: (ExecutionTransactionId, ParallelRankId, &[u64]),
        executor: &mut dyn RoutedSwiGluExecutor,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> ferrule_common::Result<()>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        let (transaction, source_rank, sequence_ids) = context;
        if self.precision != ExecutionPrecisionPolicy::f32() {
            return Err(SegmentError::ExpertPrecision);
        }
        if transaction != kv.transaction() {
            return Err(SegmentError::ExpertTransaction {
                expected: kv.transaction(),
                actual: transaction,
            });
        }
        if sequence_ids.len() != batch.sequences().len() {
            return Err(SegmentError::ExpertSequenceCount {
                expected: batch.sequences().len(),
                actual: sequence_ids.len(),
            });
        }
        let row_sequences = batch
            .row_to_sequence()
            .iter()
            .map(|&sequence| sequence_ids[sequence])
            .collect::<Vec<_>>();
        let mut execution = RoutedLayerExecution {
            transaction,
            source_rank,
            sequences: &row_sequences,
            executor,
            check_active,
        };
        self.execute_inner(batch, input, kv, operators, Some(&mut execution))
    }

    fn execute_inner(
        &self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        kv: &mut dyn StandardDecoderKvView,
        mut operators: LayerOperators<'_>,
        mut experts: Option<&mut RoutedLayerExecution<'_>>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        if operators.operators().precision() != self.precision {
            return Err(SegmentError::Precision);
        }
        if let Some(execution) = experts.as_deref_mut() {
            execution
                .check_active()
                .map_err(SegmentError::at(SegmentStage::Input))?;
        }
        kv.validate_batch(batch)
            .map_err(SegmentError::at(SegmentStage::Input))?;
        if let Some(&position) = batch.positions().iter().find(|&&p| p >= self.max_positions) {
            return Err(SegmentError::Position {
                position,
                max_positions: self.max_positions,
            });
        }
        let metadata = packed_gqa_metadata(batch).map_err(SegmentError::at(SegmentStage::Input))?;
        let rows = match (&self.embedding, input) {
            (Some(embedding), DeviceSegmentInput::Tokens) => ready(
                operators
                    .operators()
                    .embedding(embedding, batch.token_ids(), None)
                    .map_err(SegmentError::at(SegmentStage::Embedding))?,
            )
            .map_err(SegmentError::at(SegmentStage::Embedding))?,
            (None, DeviceSegmentInput::Hidden { next_layer, rows }) => {
                if next_layer != self.plan.layers.start {
                    return Err(SegmentError::HiddenBoundary {
                        expected: self.plan.layers.start,
                        actual: next_layer,
                    });
                }
                let dtype = if self.precision == ExecutionPrecisionPolicy::bf16_compatibility() {
                    RowsDType::Bf16
                } else {
                    RowsDType::F32
                };
                if rows.shape().rows() != batch.len()
                    || rows.shape().width() != self.hidden_size
                    || rows.dtype() != dtype
                {
                    return Err(SegmentError::HiddenLayout {
                        rows: batch.len(),
                        width: self.hidden_size,
                        dtype,
                    });
                }
                operators
                    .operators()
                    .bind_rows(rows)
                    .map_err(SegmentError::at(SegmentStage::Input))?
            }
            _ => return Err(SegmentError::EmbeddingOwnership),
        };
        let mut hidden = CpuTransformerHidden {
            rows: Some(rows),
            metadata,
        };
        for (local, layer) in self.layers.iter().enumerate() {
            let stage = SegmentStage::Layer {
                global: layer.index(),
                local,
            };
            if let Some(execution) = experts.as_deref_mut() {
                execution.check_active().map_err(SegmentError::at(stage))?;
            }
            execute_standard_layer(
                &mut operators,
                &self.materializer,
                layer,
                local,
                &mut hidden,
                kv,
                experts.as_deref_mut(),
            )
            .map_err(SegmentError::at(stage))?;
            if let Some(execution) = experts.as_deref_mut() {
                execution.check_active().map_err(SegmentError::at(stage))?;
            }
        }
        if let Some(execution) = experts.as_deref_mut() {
            execution
                .check_active()
                .map_err(SegmentError::at(SegmentStage::Output))?;
        }
        let output = match &self.output {
            Some(output) => execute_standard_output_rows(operators.operators(), output, &hidden)
                .map(DeviceSegmentOutput::Logits)
                .map_err(SegmentError::at(SegmentStage::Output)),
            None => Ok(DeviceSegmentOutput::Hidden {
                next_layer: self.plan.layers.end,
                rows: hidden
                    .take_rows()
                    .map_err(SegmentError::at(SegmentStage::Output))?,
            }),
        }?;
        if let Some(execution) = experts {
            execution
                .check_active()
                .map_err(SegmentError::at(SegmentStage::Output))?;
        }
        Ok(output)
    }
}

/// Stateful F32 hybrid segment. This is a computation image, not a runner or a
/// transaction owner. CPU and CUDA use the same loop and standard layer math.
#[derive(Debug)]
pub struct HybridDecoderSegment {
    plan: LayerSegmentPlan,
    schema: crate::decoder::HybridStateSchema,
    hidden_size: usize,
    max_positions: usize,
    materializer: StateDictMaterializer,
    parameters: Box<[BoundParameter]>,
    embedding: Option<PreparedEmbedding>,
    layers: Box<[super::PreparedStandardLayer]>,
    output: Option<PreparedCpuOutput>,
}

impl HybridDecoderSegment {
    pub fn prepare(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        precision: ExecutionPrecisionPolicy,
        max_positions: usize,
        max_parameter_bytes: u64,
    ) -> SegmentResult<Self> {
        let materializer = StateDictMaterializer::new(max_parameter_bytes)
            .map_err(SegmentError::at(SegmentStage::Prepare))?;
        Self::prepare_with_materializer(
            resources,
            plan,
            precision,
            max_positions,
            materializer,
            None,
        )
    }

    pub(super) fn prepare_with_materializer(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        precision: ExecutionPrecisionPolicy,
        max_positions: usize,
        materializer: StateDictMaterializer,
        local_schema: Option<crate::decoder::HybridStateSchema>,
    ) -> SegmentResult<Self> {
        let spec = resources.spec();
        if plan.total_layers() != spec.layers().len() {
            return Err(SegmentError::LayerCount {
                planned: plan.total_layers(),
                actual: spec.layers().len(),
            });
        }
        if precision != ExecutionPrecisionPolicy::f32() {
            return Err(SegmentError::Precision);
        }
        if max_positions == 0
            || spec
                .max_sequence_length()
                .is_some_and(|max| max_positions > max)
        {
            return Err(SegmentError::PositionLimit { max_positions });
        }
        let prepare = SegmentError::at(SegmentStage::Prepare);
        crate::moe::catalog::ValidatedExpertCatalog::for_decoder_range(resources, plan.layers())
            .map_err(prepare)?;
        let inventory = super::segment_inventory::HybridSegmentInventory::new(resources, &plan)
            .map_err(prepare)?;
        let schema = local_schema.unwrap_or_else(|| inventory.schema().clone());
        let descriptors = inventory.layers();
        super::standard::validate_hybrid_descriptors(spec, descriptors).map_err(prepare)?;
        let embedding = plan
            .owns_embedding()
            .then(|| prepare_embedding(resources, &materializer))
            .transpose()
            .map_err(SegmentError::at(SegmentStage::Embedding))?;
        let output = plan
            .owns_output()
            .then(|| prepare_output(resources, &materializer))
            .transpose()
            .map_err(SegmentError::at(SegmentStage::Output))?;
        let layers = descriptors
            .iter()
            .enumerate()
            .map(|(local, layer)| {
                super::standard::hybrid::prepare_standard_layer(
                    layer,
                    resources,
                    &materializer,
                    &mut MemoryLayerWeightCache::new(),
                    max_positions,
                )
                .map_err(SegmentError::at(SegmentStage::Layer {
                    global: layer.index(),
                    local,
                }))
            })
            .collect::<SegmentResult<Vec<_>>>()?
            .into_boxed_slice();
        let parameters = inventory
            .parameters()
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            plan,
            schema,
            hidden_size: spec.hidden_size(),
            max_positions,
            materializer,
            parameters,
            embedding,
            layers,
            output,
        })
    }
    #[cfg(feature = "cpu-reference")]
    pub(super) fn preparation_reads(
        &self,
    ) -> ferrule_common::Result<Vec<super::parallel::TensorParallelPreparationRead>> {
        self.materializer.tensor_reads()
    }
    #[cfg(feature = "cpu-reference")]
    pub(super) fn checkpoint_read_counters(&self) -> crate::checkpoint::CheckpointReadCounters {
        self.materializer.checkpoint_read_counters()
    }
    pub fn plan(&self) -> &LayerSegmentPlan {
        &self.plan
    }
    pub fn schema(&self) -> &crate::decoder::HybridStateSchema {
        &self.schema
    }
    /// OutputHead retains its canonical storage dependency even when embedding
    /// is unowned. No unowned layers or attachment payloads enter this image.
    pub fn parameters(&self) -> &[BoundParameter] {
        &self.parameters
    }

    pub fn validate_execution<S: crate::decoder::StandardSequenceState>(
        &self,
        batch: &PackedDecoderBatch,
        states: &[S],
    ) -> SegmentResult<()> {
        batch
            .validate_states(states)
            .map_err(SegmentError::at(SegmentStage::Input))?;
        if let Some(&position) = batch.positions().iter().find(|&&p| p >= self.max_positions) {
            return Err(SegmentError::Position {
                position,
                max_positions: self.max_positions,
            });
        }
        for sequence in batch.sequences() {
            states[sequence.state_index()]
                .validate_standard_state_at(&self.schema, sequence.context_len())
                .map_err(SegmentError::at(SegmentStage::Input))?;
        }
        Ok(())
    }

    /// Verify recurrent completion while the logical core is STILL at the input
    /// frontier. Only the outer journal may advance it after completion evidence.
    pub fn validate_frontier<S: crate::decoder::StandardSequenceState>(
        &self,
        batch: &PackedDecoderBatch,
        states: &[S],
    ) -> SegmentResult<()> {
        batch
            .validate_states(states)
            .map_err(SegmentError::at(SegmentStage::Output))?;
        for sequence in batch.sequences() {
            states[sequence.state_index()]
                .validate_standard_state_at(&self.schema, sequence.sequence_len())
                .map_err(SegmentError::at(SegmentStage::Output))?;
        }
        Ok(())
    }
    pub fn execute(
        &self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [crate::decoder::HybridDecoderSequenceState],
        kv: &mut CpuKvView,
    ) -> SegmentResult<SegmentOutput> {
        self.execute_checked(batch, input, states, kv, &mut |_| Ok(()))
    }
    pub fn execute_checked(
        &self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [crate::decoder::HybridDecoderSequenceState],
        kv: &mut CpuKvView,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> ferrule_common::Result<()>,
    ) -> SegmentResult<SegmentOutput> {
        let mut operators = CpuStandardDecoderOperators::new(ExecutionPrecisionPolicy::f32());
        self.execute_bound(
            batch,
            input.into(),
            states,
            kv,
            LayerOperators::Standard(&mut operators),
            check_active,
        )?
        .into_host(&mut operators)
    }

    /// Execute the ordinary hybrid layers with only routed SwiGLU delegated.
    /// `context` uses the KV transaction and sequence IDs in packed-sequence
    /// order, not token-row or state-slot order. No executor or transaction
    /// authority is retained. Failure requires the caller's existing journal
    /// to roll back both recurrent state and KV. This is not TP execution.
    #[cfg(feature = "cpu-reference")]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_experts(
        &self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [crate::decoder::HybridDecoderSequenceState],
        kv: &mut CpuKvView,
        context: (ExecutionTransactionId, ParallelRankId, &[u64]),
        executor: &mut dyn RoutedSwiGluExecutor,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> ferrule_common::Result<()>,
    ) -> SegmentResult<SegmentOutput> {
        let mut operators = CpuStandardDecoderOperators::new(ExecutionPrecisionPolicy::f32());
        self.execute_bound_inner(
            batch,
            input.into(),
            states,
            kv,
            LayerOperators::Standard(&mut operators),
            check_active,
            Some((context, executor)),
        )?
        .into_host(&mut operators)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn prepare_cuda_bindings(
        &self,
        operators: &mut super::standard::cuda::CudaStandardDecoderOperators,
    ) -> ferrule_common::Result<()> {
        operators.prepare_hybrid_segment_image(
            self.embedding.as_ref(),
            &self.layers,
            self.output.as_ref(),
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_bound<S: crate::decoder::StandardSequenceState>(
        &self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        states: &mut [S],
        kv: &mut dyn StandardDecoderKvView,
        operators: LayerOperators<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> ferrule_common::Result<()>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        self.execute_bound_inner(batch, input, states, kv, operators, check_active, None)
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn execute_bound_inner<S: crate::decoder::StandardSequenceState>(
        &self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        states: &mut [S],
        kv: &mut dyn StandardDecoderKvView,
        mut operators: LayerOperators<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> ferrule_common::Result<()>,
        mut experts: Option<(
            (ExecutionTransactionId, ParallelRankId, &[u64]),
            &mut dyn RoutedSwiGluExecutor,
        )>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        let row_sequences = if let Some(((transaction, _, sequences), _)) = &experts {
            if *transaction != kv.transaction() {
                return Err(SegmentError::ExpertTransaction {
                    expected: kv.transaction(),
                    actual: *transaction,
                });
            }
            if sequences.len() != batch.sequences().len() {
                return Err(SegmentError::ExpertSequenceCount {
                    expected: batch.sequences().len(),
                    actual: sequences.len(),
                });
            }
            batch
                .row_to_sequence()
                .iter()
                .map(|&i| sequences[i])
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if operators.operators().precision() != ExecutionPrecisionPolicy::f32() {
            return Err(SegmentError::Precision);
        }
        check_active(kv.transaction()).map_err(SegmentError::at(SegmentStage::Input))?;
        self.validate_execution(batch, states)?;
        kv.validate_batch(batch)
            .map_err(SegmentError::at(SegmentStage::Input))?;
        let metadata = packed_gqa_metadata(batch).map_err(SegmentError::at(SegmentStage::Input))?;
        let rows = match (&self.embedding, input) {
            (Some(embedding), DeviceSegmentInput::Tokens) => ready(
                operators
                    .operators()
                    .embedding(embedding, batch.token_ids(), None)
                    .map_err(SegmentError::at(SegmentStage::Embedding))?,
            )
            .map_err(SegmentError::at(SegmentStage::Embedding))?,
            (None, DeviceSegmentInput::Hidden { next_layer, rows }) => {
                if next_layer != self.plan.layers.start {
                    return Err(SegmentError::HiddenBoundary {
                        expected: self.plan.layers.start,
                        actual: next_layer,
                    });
                }
                if rows.shape().rows() != batch.len()
                    || rows.shape().width() != self.hidden_size
                    || rows.dtype() != RowsDType::F32
                {
                    return Err(SegmentError::HiddenLayout {
                        rows: batch.len(),
                        width: self.hidden_size,
                        dtype: RowsDType::F32,
                    });
                }
                operators
                    .operators()
                    .bind_rows(rows)
                    .map_err(SegmentError::at(SegmentStage::Input))?
            }
            _ => return Err(SegmentError::EmbeddingOwnership),
        };
        let mut hidden = CpuTransformerHidden {
            rows: Some(rows),
            metadata,
        };
        // GDN metadata indexes packed sequences, not arbitrary caller state slots.
        // Taking disjoint mutable references does not require Clone or Send.
        let mut slots = states.iter_mut().map(Some).collect::<Vec<_>>();
        let mut working = batch
            .sequences()
            .iter()
            .map(|sequence| {
                slots[sequence.state_index()]
                    .take()
                    .expect("validated unique state slot")
                    as &mut dyn crate::decoder::StandardSequenceState
            })
            .collect::<Vec<_>>();
        for (local, layer) in self.layers.iter().enumerate() {
            let stage = SegmentStage::Layer {
                global: layer.index(),
                local,
            };
            check_active(kv.transaction()).map_err(SegmentError::at(stage))?;
            let entry = &self.schema.entries()[local];
            let mut execution =
                experts
                    .as_mut()
                    .map(
                        |((transaction, source_rank, _), executor)| RoutedLayerExecution {
                            transaction: *transaction,
                            source_rank: *source_rank,
                            sequences: &row_sequences,
                            executor: &mut **executor,
                            check_active: &mut *check_active,
                        },
                    );
            operators
                .execute_layer(
                    &self.materializer,
                    layer.index(),
                    layer.attention().block().as_ref(),
                    layer.feed_forward().block(),
                    entry.local_kv_layer.unwrap_or(0),
                    &mut hidden,
                    kv,
                    &mut working,
                    execution.as_mut(),
                )
                .map_err(SegmentError::at(stage))?;
            check_active(kv.transaction()).map_err(SegmentError::at(stage))?;
        }
        let output = match &self.output {
            Some(output) => DeviceSegmentOutput::Logits(
                execute_standard_output_rows(operators.operators(), output, &hidden)
                    .map_err(SegmentError::at(SegmentStage::Output))?,
            ),
            None => DeviceSegmentOutput::Hidden {
                next_layer: self.plan.layers.end,
                rows: hidden
                    .take_rows()
                    .map_err(SegmentError::at(SegmentStage::Output))?,
            },
        };
        check_active(kv.transaction()).map_err(SegmentError::at(SegmentStage::Output))?;
        self.validate_frontier(batch, states)?;
        Ok(output)
    }
}
