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
    RoutedLayerExecution, execute_standard_layer, execute_standard_output_rows,
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

    /// Local layer indices are also the segment-local KV plane indices.
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
    fn at(stage: SegmentStage) -> impl FnOnce(ferrule_common::Error) -> Self {
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
        self.execute_inner(batch, input.into(), kv, &mut operators, None)?
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
        operators: &mut dyn StandardDecoderOperators,
        mut experts: Option<&mut RoutedLayerExecution<'_>>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        if operators.precision() != self.precision {
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
                operators,
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
            Some(output) => execute_standard_output_rows(operators, output, &hidden)
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
