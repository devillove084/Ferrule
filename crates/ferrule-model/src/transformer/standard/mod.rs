//! Shared GQA/Add/SwiGLU/MoE transformer composition.

#[cfg(feature = "cuda")]
pub mod cuda;

#[path = "cuda/expert_cache.rs"]
pub mod expert_cache;

mod hybrid;
mod numeric_precision;
#[cfg(any(feature = "cuda", test))]
pub(crate) mod scratch;
pub use hybrid::{PreparedAttentionBlock, PreparedGatedDeltaNetBlock, PreparedStandardLayer};
pub use numeric_precision::NumericFp8Precision;
mod tensor;
pub use tensor::{StandardTensorCollective, StandardTensorPlacement, StandardTensorPlan};

use std::convert::Infallible;
use std::sync::Arc;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, ParallelRankId, Result};

use crate::decoder::{
    CpuKvView, DecoderLogits, DenseLogits, GenericDecoderSequenceState, NoTerminalGuard,
    PackedDecoderBatch,
};
use crate::execution::ExecutionPrecisionPolicy;
use crate::support::TensorRole;

use super::expert_parallel::{ExpertDispatchContext, RoutedSwiGluExecutor, RoutedSwiGluRequest};
use super::operators::{standard_rope_unsupported, standard_router_unsupported};

use super::{
    AddResidual, Attention, BoundDecoderResources, BoundParameter, Connected,
    CpuStandardDecoderOperators, DecoderLayer, DecoderModelSpec, ExpertAvailability,
    ExpertProvider, FeedForward, GqaMetadata, GqaRequest, KvView, LayerRequest, Linear,
    MemoryLayerWeightCache, Moe, NoPostLayerTap, NoReduction, OperatorProgress, OutputPipeline,
    Poll, PreparedEmbedding, PreparedLinear, PreparedNorm, PreparedRope, PreparedSwiGlu,
    PreparedTransformer, Residual, RmsNorm, RotaryEmbedding, RotaryScaling, Rows,
    StandardDecoderOperators, StateDictMaterializer, Step, SwiGlu, TransformerLayer,
    TransformerModule, UnsupportedOperator,
};

/// Prepared GQA block wrapped by an additive connection.
#[derive(Debug, Clone)]
pub struct PreparedGqaBlock {
    descriptor: DecoderLayer,
    input_norm: PreparedNorm,
    query: PreparedLinear,
    key: PreparedLinear,
    value: PreparedLinear,
    output: PreparedLinear,
    query_norm: Option<PreparedNorm>,
    key_norm: Option<PreparedNorm>,
    rope: PreparedRope,
}

/// Prepared dense or routed feed-forward block wrapped by an additive connection.
#[derive(Debug, Clone)]
pub struct PreparedFeedForwardBlock {
    norm: PreparedNorm,
    kind: PreparedFeedForwardKind,
}

#[derive(Debug, Clone)]
enum PreparedFeedForwardKind {
    Dense(PreparedSwiGlu),
    Routed {
        router: PreparedLinear,
        descriptor: Moe,
        shared: Option<PreparedSwiGlu>,
        shared_gate: Option<PreparedLinear>,
        experts: Box<[[BoundParameter; 3]]>,
    },
}

/// Concrete Qwen-compatible layer composition: GQA + Add + MoE + Add.
pub type PreparedGqaMoeLayer = TransformerLayer<
    Connected<AddResidual, PreparedGqaBlock>,
    Connected<AddResidual, PreparedFeedForwardBlock>,
>;

/// Concrete model-independent output pipeline used by one-stream decoders.
pub type PreparedStandardOutput = OutputPipeline<NoReduction, PreparedNorm, PreparedLinear>;
/// Compatibility name for existing CPU runners.
pub type PreparedCpuOutput = PreparedStandardOutput;

#[derive(Debug)]
pub struct StandardTransformerHidden {
    pub(super) rows: Option<Rows>,
    pub(super) metadata: GqaMetadata,
}

/// Compatibility name for existing CPU runners; storage is device-neutral.
pub type CpuTransformerHidden = StandardTransformerHidden;

impl StandardTransformerHidden {
    fn rows(&self) -> Result<&Rows> {
        self.rows
            .as_ref()
            .ok_or_else(|| model_error("standard hidden rows are checked out"))
    }

    pub(super) fn take_rows(&mut self) -> Result<Rows> {
        self.rows
            .take()
            .ok_or_else(|| model_error("standard hidden rows are checked out"))
    }
}

/// Prepared CPU module selected by Qwen and other ordinary GQA/Add decoders.
pub type CpuGqaMoeModule = CpuStandardModule<GenericDecoderSequenceState>;
pub type CpuHybridModule = CpuStandardModule<crate::decoder::HybridDecoderSequenceState>;

#[derive(Debug)]
pub struct CpuStandardModule<S> {
    state: std::marker::PhantomData<fn() -> S>,
    schema: crate::decoder::HybridStateSchema,
    resources: Arc<BoundDecoderResources>,
    materializer: Arc<StateDictMaterializer>,
    operators: CpuStandardDecoderOperators,
    prepared: PreparedTransformer<
        PreparedEmbedding,
        PreparedStandardLayer,
        PreparedCpuOutput,
        NoPostLayerTap,
    >,
}

struct StateDictExpertProvider<'a> {
    layer: usize,
    bindings: &'a [[BoundParameter; 3]],

    activation_limit: Option<f32>,
    materializer: &'a StateDictMaterializer,
}

impl ExpertProvider for StateDictExpertProvider<'_> {
    fn expert_metadata_bindings(
        &self,
        layer: usize,
        expert: usize,
    ) -> Result<Option<super::ExpertMetadataBindings<'_>>> {
        if layer != self.layer {
            return Err(model_error(format!(
                "expert request for unowned layer {layer}"
            )));
        }
        let parameters = self
            .bindings
            .get(expert)
            .ok_or_else(|| model_error(format!("missing expert {layer}:{expert}")))?;
        Ok(Some(super::ExpertMetadataBindings {
            parameters,
            activation_limit: self.activation_limit,
        }))
    }

    fn expert_metadata(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<Option<super::ExpertMetadata>> {
        if layer != self.layer {
            return Err(model_error(format!(
                "expert request for unowned layer {layer}"
            )));
        }
        let bindings = self
            .bindings
            .get(expert)
            .ok_or_else(|| model_error(format!("missing expert {layer}:{expert}")))?;
        super::ExpertMetadata::new(layer, expert, bindings.clone(), self.activation_limit).map(Some)
    }

    fn expert(&mut self, layer: usize, expert: usize) -> Result<ExpertAvailability> {
        if layer != self.layer {
            return Err(model_error(format!(
                "expert request for unowned layer {layer}"
            )));
        }
        let bindings = self
            .bindings
            .get(expert)
            .ok_or_else(|| model_error(format!("missing expert {layer}:{expert}")))?;
        let linear = |binding: &BoundParameter| {
            PreparedLinear::from_parameter(
                self.materializer.expert_parameter(layer, expert, binding)?,
                binding.role().clone(),
            )
        };
        Ok(ExpertAvailability::Ready(Arc::new(PreparedSwiGlu::new(
            linear(&bindings[0])?,
            linear(&bindings[1])?,
            linear(&bindings[2])?,
            self.activation_limit,
        )?)))
    }
}

impl<S: crate::decoder::StandardSequenceState> CpuStandardModule<S> {
    pub fn prepare(
        resources: Arc<BoundDecoderResources>,
        materializer: Arc<StateDictMaterializer>,
        precision: ExecutionPrecisionPolicy,
        max_positions: usize,
    ) -> Result<Self> {
        let active_layers = resources.spec().layers().len();
        Self::prepare_prefix(
            resources,
            materializer,
            precision,
            max_positions,
            active_layers,
        )
    }

    pub fn prepare_prefix(
        resources: Arc<BoundDecoderResources>,
        materializer: Arc<StateDictMaterializer>,
        precision: ExecutionPrecisionPolicy,
        max_positions: usize,
        active_layers: usize,
    ) -> Result<Self> {
        let spec = resources.spec();
        if max_positions == 0
            || spec
                .max_sequence_length()
                .is_some_and(|maximum| max_positions > maximum)
        {
            return Err(model_error(format!(
                "invalid prepared RoPE position count {max_positions}"
            )));
        }
        if active_layers == 0 || active_layers > spec.layers().len() {
            return Err(model_error(format!(
                "active layer count {active_layers} must be in 1..={}",
                spec.layers().len()
            )));
        }
        hybrid::validate_cpu_profile(spec, precision)?;
        validate_hybrid_descriptors(spec, spec.layers())?;
        let schema = crate::decoder::HybridStateSchema::from_spec(spec, active_layers)?;
        S::validate_standard_schema(&schema)?;
        let embedding = prepare_embedding(&resources, &materializer)?;
        let output = prepare_output(&resources, &materializer)?;
        let mut layers = Vec::with_capacity(active_layers);
        for descriptor in &spec.layers()[..active_layers] {
            let mut cache = MemoryLayerWeightCache::new();
            layers.push(hybrid::prepare_standard_layer(
                descriptor,
                &resources,
                &materializer,
                &mut cache,
                max_positions,
            )?);
        }
        Ok(Self {
            state: std::marker::PhantomData,
            schema,
            resources,
            materializer,
            operators: CpuStandardDecoderOperators::new(precision),
            prepared: PreparedTransformer::new(embedding, layers, output, NoPostLayerTap),
        })
    }

    pub fn spec(&self) -> &DecoderModelSpec {
        self.resources.spec()
    }

    fn execute_layer(
        &mut self,
        layer_index: usize,
        hidden: &mut CpuTransformerHidden,
        kv: &mut CpuKvView,
        states: &mut [S],
    ) -> Result<()> {
        let layer = self
            .prepared
            .layers()
            .get(layer_index)
            .ok_or_else(|| model_error("missing prepared standard layer"))?;
        let kv_layer = match &self.schema.layers()[layer_index] {
            crate::decoder::HybridLayerSchema::FullAttention { kv_layer, .. } => *kv_layer,
            _ => 0,
        };
        let mut states = states
            .iter_mut()
            .map(|s| s as &mut dyn crate::decoder::StandardSequenceState)
            .collect::<Vec<_>>();
        execute_standard_layer_inner(
            &mut self.operators,
            &self.materializer,
            layer.index(),
            layer.attention().block().as_ref(),
            layer.feed_forward().block(),
            kv_layer,
            hidden,
            kv,
            &mut states,
            None,
        )
    }
}

/// Call-scoped injection, never retained by a prepared layer or segment.
pub(super) struct RoutedLayerExecution<'a> {
    pub transaction: ExecutionTransactionId,
    pub source_rank: ParallelRankId,
    pub sequences: &'a [u64],
    pub executor: &'a mut dyn RoutedSwiGluExecutor,
    pub check_active: &'a mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
}

impl RoutedLayerExecution<'_> {
    pub fn check_active(&mut self) -> Result<()> {
        (self.check_active)(self.transaction)
    }
}

/// The single device-neutral layer implementation used by runners and segments.
/// Weight/expert coordinates stay global; the caller supplies the KV-local layer.
pub(super) fn execute_standard_layer(
    operators: &mut dyn StandardDecoderOperators,
    materializer: &StateDictMaterializer,
    layer: &PreparedGqaMoeLayer,
    kv_layer: usize,
    hidden: &mut CpuTransformerHidden,
    kv: &mut dyn KvView,
    routed_execution: Option<&mut RoutedLayerExecution<'_>>,
) -> Result<()> {
    execute_standard_layer_inner(
        operators,
        materializer,
        layer.index(),
        hybrid::AttentionBlockRef::Gqa(layer.attention().block()),
        layer.feed_forward().block(),
        kv_layer,
        hidden,
        kv,
        &mut [],
        routed_execution,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_standard_layer_inner(
    operators: &mut dyn StandardDecoderOperators,
    materializer: &StateDictMaterializer,
    layer_index: usize,
    attention: hybrid::AttentionBlockRef<'_>,
    feed_forward: &PreparedFeedForwardBlock,
    kv_layer: usize,
    hidden: &mut CpuTransformerHidden,
    kv: &mut dyn KvView,
    states: &mut [&mut dyn crate::decoder::StandardSequenceState],
    routed_execution: Option<&mut RoutedLayerExecution<'_>>,
) -> Result<()> {
    let arena = None;
    let update = match attention {
        hybrid::AttentionBlockRef::Gqa(block) => {
            execute_gqa(operators, block, kv_layer, hidden, kv)?
        }
        hybrid::AttentionBlockRef::GatedDeltaNet(block) => {
            hybrid::execute_delta(operators, block, layer_index, hidden, states)?
        }
    };
    hidden.rows = Some(ready(operators.residual(hidden.take_rows()?, &update)?)?);
    let normalized = ready(operators.rms_norm(&feed_forward.norm, hidden.rows()?, 1, arena)?)?;
    let update = execute_feed_forward(
        operators,
        materializer,
        layer_index,
        &feed_forward.kind,
        &normalized,
        routed_execution,
    )?;
    hidden.rows = Some(ready(operators.residual(hidden.take_rows()?, &update)?)?);
    Ok(())
}

fn execute_feed_forward(
    operators: &mut dyn StandardDecoderOperators,
    materializer: &StateDictMaterializer,
    layer_index: usize,
    kind: &PreparedFeedForwardKind,
    normalized: &Rows,
    routed_execution: Option<&mut RoutedLayerExecution<'_>>,
) -> Result<Rows> {
    let arena = None;
    let update = match kind {
        PreparedFeedForwardKind::Dense(feed_forward) => {
            ready(operators.dense_swiglu(feed_forward, normalized, arena)?)?
        }
        PreparedFeedForwardKind::Routed {
            router,
            descriptor,
            shared,
            shared_gate,
            experts: bindings,
        } => {
            let logits = ready(operators.linear(router, normalized, arena)?)?;
            let routes = ready(operators.router(&logits, descriptor.router_spec())?)?;
            let mut shared = shared
                .as_ref()
                .map(|shared| operators.dense_swiglu(shared, normalized, arena))
                .transpose()?
                .map(ready)
                .transpose()?;
            if let Some(gate) = shared_gate {
                let gate = ready(operators.linear(gate, normalized, arena)?)?;
                let output = shared
                    .take()
                    .ok_or_else(|| model_error("shared output gate without shared expert"))?;
                shared = Some(ready(operators.shared_expert_gate(output, &gate)?)?);
            }
            let mut routed = if let Some(execution) = routed_execution {
                ready(execution.executor.routed_swiglu(
                    RoutedSwiGluRequest {
                        context: ExpertDispatchContext {
                            transaction: execution.transaction,
                            source_rank: execution.source_rank,
                            layer: layer_index,
                        },
                        sequences: execution.sequences,
                        input: normalized,
                        routes: &routes,
                        arena,
                    },
                    execution.check_active,
                )?)?
            } else {
                let mut experts = StateDictExpertProvider {
                    layer: layer_index,
                    bindings,

                    activation_limit: descriptor.expert().activation_limit(),
                    materializer,
                };
                ready(operators.routed_swiglu(
                    layer_index,
                    normalized,
                    &routes,
                    &mut experts,
                    arena,
                )?)?
            };
            // Injected EP may return staged host rows; bind once at this boundary.
            routed = operators.bind_rows(routed)?;
            if let Some(shared) = &shared {
                routed = ready(operators.residual(routed, shared)?)?;
            }
            routed
        }
    };
    Ok(update)
}

fn execute_gqa(
    operators: &mut dyn StandardDecoderOperators,
    attention: &PreparedGqaBlock,
    kv_layer: usize,
    hidden: &CpuTransformerHidden,
    kv: &mut dyn KvView,
) -> Result<Rows> {
    let Attention::Gqa(gqa) = attention.descriptor.attention() else {
        return Err(model_error("prepared GQA layer lost its descriptor"));
    };
    let (query_heads, kv_heads) = operators.attention_heads(gqa.num_heads(), gqa.num_kv_heads())?;
    let arena = None;
    let normalized = ready(operators.rms_norm(&attention.input_norm, hidden.rows()?, 1, arena)?)?;
    let mut query = ready(operators.linear(&attention.query, &normalized, arena)?)?;
    let gate = if gqa.gated_query() {
        let (q, gate) = ready(operators.unpack_gated_query(query, query_heads, gqa.head_dim())?)?;
        query = q;
        Some(gate)
    } else {
        None
    };
    let mut key = ready(operators.linear(&attention.key, &normalized, arena)?)?;
    let value = ready(operators.linear(&attention.value, &normalized, arena)?)?;
    if let Some(norm) = &attention.query_norm {
        query = ready(operators.rms_norm(norm, &query, query_heads, arena)?)?;
    }
    if let Some(norm) = &attention.key_norm {
        key = ready(operators.rms_norm(norm, &key, kv_heads, arena)?)?;
    }
    query = ready(operators.rope(
        gqa.rotary(),
        &attention.rope,
        query,
        query_heads,
        hidden.metadata.row_positions(),
    )?)?;
    key = ready(operators.rope(
        gqa.rotary(),
        &attention.rope,
        key,
        kv_heads,
        hidden.metadata.row_positions(),
    )?)?;
    let update = ready(operators.paged_gqa(
        kv,
        GqaRequest {
            layer: kv_layer,
            query: &query,
            key: &key,
            value: &value,
            metadata: &hidden.metadata,
            query_heads,
            kv_heads,
            head_dim: gqa.head_dim(),
            softmax_scale: (gqa.head_dim() as f32).sqrt().recip(),
            arena,
        },
    )?)?;
    let update = if let Some(gate) = gate {
        ready(operators.sigmoid_gate(update, &gate)?)?
    } else {
        update
    };
    ready(operators.linear(&attention.output, &update, arena)?)
}

impl<S: crate::decoder::StandardSequenceState> TransformerModule for CpuStandardModule<S> {
    type State = S;
    type KvView = CpuKvView;
    type Hidden = CpuTransformerHidden;
    type ArenaKey = (usize, usize);
    type Arena = ();
    type EmbeddingPending = Infallible;
    type LayerPending = Infallible;
    type OutputPending = Infallible;
    type Quiescence = ();
    type Event = Infallible;
    type TerminalGuard = NoTerminalGuard;

    fn layer_count(&self) -> usize {
        self.prepared.layers().len()
    }

    fn validate(
        &mut self,
        _context: &crate::decoder::DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        kv: &mut Self::KvView,
    ) -> Result<()> {
        for state in states {
            state.validate_standard_state(&self.schema)?;
        }
        kv.validate_batch(batch)
    }

    fn arena_key(&self, batch: &PackedDecoderBatch) -> Result<Self::ArenaKey> {
        Ok((batch.len(), batch.sequences().len()))
    }

    fn build_arena(&mut self, _key: &Self::ArenaKey) -> Result<Self::Arena> {
        Ok(())
    }

    fn submit_embedding(
        &mut self,
        _context: &crate::decoder::DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        _kv: &mut Self::KvView,
        _arena: &mut Self::Arena,
    ) -> Result<Step<Self::Hidden, Self::EmbeddingPending>> {
        let metadata = packed_gqa_metadata(batch)?;
        let rows = ready(self.operators.embedding(
            self.prepared.embedding(),
            batch.token_ids(),
            None,
        )?)?;
        Ok(Step::Complete(CpuTransformerHidden {
            rows: Some(rows),
            metadata,
        }))
    }

    fn poll_embedding(
        &mut self,
        _context: &crate::decoder::DecoderTransactionContext,
        _batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        _kv: &mut Self::KvView,
        _arena: &mut Self::Arena,
        pending: &mut Self::EmbeddingPending,
    ) -> Result<Poll<Self::Hidden, Self::EmbeddingPending>> {
        match *pending {}
    }

    fn start_layer(
        &mut self,
        _context: &crate::decoder::DecoderTransactionContext,
        layer: usize,
        request: LayerRequest<'_, Self::State, Self::KvView>,
        hidden: &mut Self::Hidden,
        _arena: &mut Self::Arena,
    ) -> Result<Step<Vec<Self::Event>, Self::LayerPending>> {
        let LayerRequest::Target { kv, states, .. } = request else {
            return Err(model_error("CPU GQA module does not have proposal stages"));
        };
        self.execute_layer(layer, hidden, kv, states)?;
        Ok(Step::Complete(Vec::new()))
    }

    fn poll_layer(
        &mut self,
        _context: &crate::decoder::DecoderTransactionContext,
        _layer: usize,
        _request: LayerRequest<'_, Self::State, Self::KvView>,
        _hidden: &mut Self::Hidden,
        _arena: &mut Self::Arena,
        pending: &mut Self::LayerPending,
    ) -> Result<Poll<Vec<Self::Event>, Self::LayerPending>> {
        match *pending {}
    }

    fn post_layer(
        &mut self,
        _layer: usize,
        _batch: &PackedDecoderBatch,
        _hidden: &Self::Hidden,
        _arena: &mut Self::Arena,
    ) -> Result<()> {
        Ok(())
    }

    fn submit_output(
        &mut self,
        _context: &crate::decoder::DecoderTransactionContext,
        _batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        _kv: &mut Self::KvView,
        hidden: &Self::Hidden,
        _arena: &mut Self::Arena,
    ) -> Result<Step<DecoderLogits, Self::OutputPending>> {
        Ok(Step::Complete(DecoderLogits::Dense(
            execute_standard_output(&mut self.operators, self.prepared.output(), hidden)?,
        )))
    }

    fn poll_output(
        &mut self,
        _context: &crate::decoder::DecoderTransactionContext,
        _batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        _kv: &mut Self::KvView,
        _hidden: &Self::Hidden,
        _arena: &mut Self::Arena,
        pending: &mut Self::OutputPending,
    ) -> Result<Poll<DecoderLogits, Self::OutputPending>> {
        match *pending {}
    }

    fn poll_embedding_cancel(&mut self, pending: &mut Self::EmbeddingPending) -> Result<bool> {
        match *pending {}
    }

    fn poll_layer_cancel(
        &mut self,
        _layer: usize,
        pending: &mut Self::LayerPending,
    ) -> Result<bool> {
        match *pending {}
    }

    fn poll_output_cancel(&mut self, pending: &mut Self::OutputPending) -> Result<bool> {
        match *pending {}
    }

    fn cancel_embedding(&mut self, pending: Self::EmbeddingPending) -> Result<()> {
        match pending {}
    }

    fn cancel_layer(&mut self, _layer: usize, pending: Self::LayerPending) -> Result<()> {
        match pending {}
    }

    fn cancel_output(&mut self, pending: Self::OutputPending) -> Result<()> {
        match pending {}
    }

    fn begin_quiescence(&mut self) -> Result<Self::Quiescence> {
        Ok(())
    }

    fn poll_quiescence(&mut self, _quiescence: &mut Self::Quiescence) -> Result<bool> {
        Ok(true)
    }

    fn finish(
        &mut self,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        _kv: &mut Self::KvView,
        _arena: &mut Self::Arena,
        events: Vec<Self::Event>,
    ) -> Result<Self::TerminalGuard> {
        for sequence in batch.sequences() {
            let state = states
                .get(sequence.state_index())
                .ok_or_else(|| model_error("missing hybrid sequence at completion"))?;
            state.validate_standard_state_at(&self.schema, sequence.sequence_len())?;
        }
        match events.into_iter().next() {
            Some(event) => match event {},
            None => Ok(NoTerminalGuard),
        }
    }

    fn abort(
        &mut self,
        _batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        _kv: &mut Self::KvView,
        _arena: &mut Self::Arena,
    ) -> Result<()> {
        Ok(())
    }
}

pub(super) fn prepare_embedding(
    resources: &BoundDecoderResources,
    materializer: &StateDictMaterializer,
) -> Result<PreparedEmbedding> {
    Ok(PreparedEmbedding::new(PreparedLinear::from_parameter(
        materializer.static_parameter(resources.require_static_shape(
            TensorRole::TokenEmbedding,
            &resources.spec().token_embedding().weight_shape(),
        )?)?,
        TensorRole::TokenEmbedding,
    )?))
}

pub(super) fn prepare_output(
    resources: &BoundDecoderResources,
    materializer: &StateDictMaterializer,
) -> Result<PreparedCpuOutput> {
    let spec = resources.spec();
    let norm = PreparedNorm::from_descriptor(
        materializer.static_parameter(
            resources
                .require_static_shape(TensorRole::OutputNorm, &spec.final_norm().weight_shape())?,
        )?,
        spec.final_norm(),
    )?;
    // The logical head keeps its canonical embedding dependency even when this
    // stage does not own the embedding operation.
    let head = materializer.prepared_linear(
        resources.require_static_shape(TensorRole::OutputHead, &spec.output().weight_shape())?,
        TensorRole::OutputHead,
    )?;
    Ok(OutputPipeline::new(NoReduction, norm, head))
}

pub(super) fn execute_standard_output(
    operators: &mut dyn StandardDecoderOperators,
    output: &PreparedCpuOutput,
    hidden: &CpuTransformerHidden,
) -> Result<DenseLogits> {
    let rows = execute_standard_output_rows(operators, output, hidden)?;
    let logits = operators.download_rows(rows)?;
    DenseLogits::new(
        logits.shape().rows(),
        logits.shape().width(),
        logits.into_values(),
    )
}

pub(super) fn execute_standard_output_rows(
    operators: &mut dyn StandardDecoderOperators,
    output: &PreparedCpuOutput,
    hidden: &StandardTransformerHidden,
) -> Result<Rows> {
    let normalized = ready(operators.rms_norm(output.norm(), hidden.rows()?, 1, None)?)?;
    ready(operators.lm_head(output.head(), &normalized, None)?)
}

// Validate capabilities before materializing any payload, including embedding.
// Segment callers validate only their layers, but share the output contract.
pub(super) fn validate_standard_descriptors(
    spec: &DecoderModelSpec,
    layers: &[DecoderLayer],
) -> Result<()> {
    for layer in layers {
        if !matches!(layer.attention(), Attention::Gqa(gqa) if !gqa.gated_query() && !gqa.query_norm().is_some_and(RmsNorm::one_plus_weight) && !gqa.key_norm().is_some_and(RmsNorm::one_plus_weight))
            || layer.input_norm().one_plus_weight()
            || layer.post_attention_norm().one_plus_weight()
            || spec.final_norm().one_plus_weight()
        {
            return Err(unsupported_error(UnsupportedOperator::new(
                "standard_segment",
                "hybrid/gated attention requires an explicitly supported backend and sequence state",
            )));
        }
    }
    validate_hybrid_descriptors(spec, layers)
}

fn validate_hybrid_descriptors(spec: &DecoderModelSpec, layers: &[DecoderLayer]) -> Result<()> {
    if spec.output().has_bias() {
        return Err(unsupported_error(UnsupportedOperator::new(
            "lm_head",
            "standard CPU output bias is not supported",
        )));
    }
    if spec.output_hyper_connection().is_some() {
        return Err(unsupported_error(UnsupportedOperator::new(
            "output",
            "standard CPU output_hyper_connection is not supported",
        )));
    }
    for layer in layers {
        match layer.attention() {
            Attention::Gqa(gqa) => {
                if let Some(unsupported) = standard_rope_unsupported(gqa.rotary()) {
                    return Err(unsupported_error(unsupported));
                }
            }
            Attention::GatedDeltaNet(_) => {}
            _ => {
                return Err(unsupported_error(UnsupportedOperator::new(
                    "standard_attention",
                    "only GQA and GatedDeltaNet are supported",
                )));
            }
        }
        if let FeedForward::Moe(moe) = layer.feed_forward()
            && let Some(unsupported) = standard_router_unsupported(moe.router_spec())
        {
            return Err(unsupported_error(unsupported));
        }
        if !matches!(layer.attention_residual(), Residual::Add)
            || !matches!(layer.feed_forward_residual(), Residual::Add)
        {
            return Err(model_error(format!(
                "layer {} requires additive residual connections",
                layer.index()
            )));
        }
    }
    Ok(())
}

pub(super) fn prepare_layer(
    descriptor: &DecoderLayer,
    resources: &BoundDecoderResources,
    materializer: &StateDictMaterializer,
    cache: &mut MemoryLayerWeightCache,
    max_positions: usize,
) -> Result<PreparedGqaMoeLayer> {
    let Attention::Gqa(gqa) = descriptor.attention() else {
        return Err(model_error("GQA module preparation requires GQA"));
    };
    let layer = descriptor.index();
    let input_norm = prepare_layer_norm(
        layer,
        TensorRole::AttentionNorm,
        descriptor.input_norm(),
        resources,
        materializer,
        cache,
    )?;
    let post_attention_norm = prepare_layer_norm(
        layer,
        TensorRole::FeedForwardNorm,
        descriptor.post_attention_norm(),
        resources,
        materializer,
        cache,
    )?;
    let query = prepare_layer_linear(
        layer,
        TensorRole::AttentionQuery,
        gqa.query(),
        resources,
        materializer,
        cache,
    )?;
    let key = prepare_layer_linear(
        layer,
        TensorRole::AttentionKey,
        gqa.key(),
        resources,
        materializer,
        cache,
    )?;
    let value = prepare_layer_linear(
        layer,
        TensorRole::AttentionValue,
        gqa.value(),
        resources,
        materializer,
        cache,
    )?;
    let output = prepare_layer_linear(
        layer,
        TensorRole::AttentionOutput,
        gqa.output(),
        resources,
        materializer,
        cache,
    )?;
    let query_norm = gqa
        .query_norm()
        .map(|norm| {
            prepare_layer_norm(
                layer,
                TensorRole::AttentionQueryNorm,
                norm,
                resources,
                materializer,
                cache,
            )
        })
        .transpose()?;
    let key_norm = gqa
        .key_norm()
        .map(|norm| {
            prepare_layer_norm(
                layer,
                TensorRole::AttentionKeyNorm,
                norm,
                resources,
                materializer,
                cache,
            )
        })
        .transpose()?;
    let feed_forward = prepare_feed_forward(
        descriptor,
        resources,
        materializer,
        cache,
        post_attention_norm,
    )?;
    Ok(TransformerLayer::new(
        layer,
        Connected::new(
            AddResidual,
            PreparedGqaBlock {
                descriptor: descriptor.clone(),
                input_norm,
                query,
                key,
                value,
                output,
                query_norm,
                key_norm,
                rope: prepare_rope(gqa.rotary(), max_positions)?,
            },
        ),
        Connected::new(AddResidual, feed_forward),
    ))
}

fn prepare_feed_forward(
    descriptor: &DecoderLayer,
    resources: &BoundDecoderResources,
    materializer: &StateDictMaterializer,
    cache: &mut MemoryLayerWeightCache,
    norm: PreparedNorm,
) -> Result<PreparedFeedForwardBlock> {
    if let Some(host_experts) = resources.host_experts() {
        materializer.attach_host_experts(host_experts)?;
    }
    let layer = descriptor.index();
    let kind = match descriptor.feed_forward() {
        FeedForward::SwiGlu(feed_forward) => PreparedFeedForwardKind::Dense(prepare_swiglu(
            layer,
            feed_forward,
            resources,
            materializer,
            cache,
            false,
        )?),
        FeedForward::Moe(moe) => PreparedFeedForwardKind::Routed {
            experts: (0..moe.router_spec().num_experts())
                .map(|expert| {
                    let binding = |role, shape: [usize; 2]| {
                        resources
                            .experts()
                            .require_shape(layer, expert, role, &shape)
                            .cloned()
                    };
                    Ok([
                        binding(
                            TensorRole::RoutedExpertGate,
                            moe.expert().gate().weight_shape(),
                        )?,
                        binding(TensorRole::RoutedExpertUp, moe.expert().up().weight_shape())?,
                        binding(
                            TensorRole::RoutedExpertDown,
                            moe.expert().down().weight_shape(),
                        )?,
                    ])
                })
                .collect::<Result<Vec<_>>>()?
                .into_boxed_slice(),

            router: prepare_layer_linear(
                layer,
                TensorRole::RouterLogits,
                moe.router(),
                resources,
                materializer,
                cache,
            )?,
            descriptor: moe.clone(),
            shared_gate: moe
                .shared_expert_gate()
                .map(|gate| {
                    prepare_layer_linear(
                        layer,
                        TensorRole::SharedExpertOutputGate,
                        gate,
                        resources,
                        materializer,
                        cache,
                    )
                })
                .transpose()?,
            shared: moe
                .shared_expert()
                .map(|shared| prepare_swiglu(layer, shared, resources, materializer, cache, true))
                .transpose()?,
        },
    };
    Ok(PreparedFeedForwardBlock { norm, kind })
}

fn prepare_swiglu(
    layer: usize,
    descriptor: &SwiGlu,
    resources: &BoundDecoderResources,
    materializer: &StateDictMaterializer,
    cache: &mut MemoryLayerWeightCache,
    shared: bool,
) -> Result<PreparedSwiGlu> {
    let roles = if shared {
        [
            TensorRole::SharedExpertGate,
            TensorRole::SharedExpertUp,
            TensorRole::SharedExpertDown,
        ]
    } else {
        [
            TensorRole::DenseMlpGate,
            TensorRole::DenseMlpUp,
            TensorRole::DenseMlpDown,
        ]
    };
    PreparedSwiGlu::new(
        prepare_layer_linear(
            layer,
            roles[0].clone(),
            descriptor.gate(),
            resources,
            materializer,
            cache,
        )?,
        prepare_layer_linear(
            layer,
            roles[1].clone(),
            descriptor.up(),
            resources,
            materializer,
            cache,
        )?,
        prepare_layer_linear(
            layer,
            roles[2].clone(),
            descriptor.down(),
            resources,
            materializer,
            cache,
        )?,
        descriptor.activation_limit(),
    )
}

fn prepare_layer_linear(
    layer: usize,
    role: TensorRole,
    descriptor: &Linear,
    resources: &BoundDecoderResources,
    materializer: &StateDictMaterializer,
    cache: &mut MemoryLayerWeightCache,
) -> Result<PreparedLinear> {
    let linear = materializer.prepared_layer_linear(
        layer,
        resources.require_layer_shape(layer, role.clone(), &descriptor.weight_shape())?,
        role.clone(),
        cache,
    )?;
    if descriptor.has_bias() {
        let bias_shape = descriptor
            .bias_shape()
            .expect("descriptor with bias has bias shape");
        let bias = materializer.layer_parameter(
            layer,
            resources.require_layer_shape(layer, role, &bias_shape)?,
            cache,
        )?;
        linear.with_bias(bias)
    } else {
        Ok(linear)
    }
}

fn prepare_layer_norm(
    layer: usize,
    role: TensorRole,
    descriptor: &RmsNorm,
    resources: &BoundDecoderResources,
    materializer: &StateDictMaterializer,
    cache: &mut MemoryLayerWeightCache,
) -> Result<PreparedNorm> {
    PreparedNorm::from_descriptor(
        materializer.layer_parameter(
            layer,
            resources.require_layer_shape(layer, role, &descriptor.weight_shape())?,
            cache,
        )?,
        descriptor,
    )
}

fn prepare_rope(descriptor: &RotaryEmbedding, positions: usize) -> Result<PreparedRope> {
    let dimensions = descriptor.region().dimensions();
    let width = dimensions / 2;
    let mut cosine = Vec::with_capacity(positions * width);
    let mut sine = Vec::with_capacity(positions * width);
    for position in 0..positions {
        for pair in 0..width {
            let frequency = rope_frequency(pair, dimensions, descriptor);
            let angle = position as f32 * frequency;
            if !frequency.is_finite() || !angle.is_finite() {
                return Err(model_error(format!(
                    "RoPE frequency overflow at position={position} pair={pair}"
                )));
            }
            let (sin, cos) = angle.sin_cos();
            cosine.push(cos);
            sine.push(sin);
        }
    }
    PreparedRope::new(positions, dimensions, cosine, sine)
}

fn rope_frequency(pair: usize, dimensions: usize, descriptor: &RotaryEmbedding) -> f32 {
    let base = descriptor
        .theta()
        .powf(-((2 * pair) as f32 / dimensions as f32));
    match descriptor.scaling() {
        RotaryScaling::None => base,
        RotaryScaling::Linear { factor } => base / factor,
        RotaryScaling::YaRN {
            factor,
            original_max_position_embeddings,
            beta_fast,
            beta_slow,
            ..
        } => {
            let low = yarn_correction_dimension(
                *beta_fast,
                dimensions,
                descriptor.theta(),
                *original_max_position_embeddings,
            )
            .floor()
            .max(0.0);
            let high = yarn_correction_dimension(
                *beta_slow,
                dimensions,
                descriptor.theta(),
                *original_max_position_embeddings,
            )
            .ceil()
            .clamp(0.0, dimensions.saturating_sub(1) as f32);
            let maximum = if (low - high).abs() < f32::EPSILON {
                high + 0.001
            } else {
                high
            };
            let smooth = 1.0 - ((pair as f32 - low) / (maximum - low)).clamp(0.0, 1.0);
            base / factor * (1.0 - smooth) + base * smooth
        }
    }
}

fn yarn_correction_dimension(
    rotations: f32,
    dimensions: usize,
    theta: f32,
    original_positions: usize,
) -> f32 {
    dimensions as f32 * (original_positions as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * theta.ln())
}

pub(super) fn packed_gqa_metadata(batch: &PackedDecoderBatch) -> Result<GqaMetadata> {
    let row_kv_lens = batch
        .positions()
        .iter()
        .map(|position| {
            position
                .checked_add(1)
                .ok_or_else(|| model_error("packed GQA history length overflow"))
        })
        .collect::<Result<Vec<_>>>()?;
    let block_slots = batch
        .sequences()
        .iter()
        .flat_map(|sequence| sequence.block_table())
        .map(|page| {
            i32::try_from(page.0)
                .map_err(|_| model_error("physical KV page ID exceeds the GQA i32 ABI"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut block_offsets = Vec::with_capacity(batch.sequences().len() + 1);
    block_offsets.push(0i32);
    for sequence in batch.sequences() {
        let next = block_offsets
            .last()
            .copied()
            .unwrap_or_default()
            .checked_add(
                i32::try_from(sequence.block_table().len())
                    .map_err(|_| model_error("packed GQA block table exceeds the i32 ABI"))?,
            )
            .ok_or_else(|| model_error("packed GQA block offset overflow"))?;
        block_offsets.push(next);
    }
    GqaMetadata::new(
        batch.sequences().len(),
        batch.row_to_sequence().to_vec(),
        batch.positions().to_vec(),
        row_kv_lens,
    )?
    .with_pages(block_slots, block_offsets)
}

pub(super) fn ready<T>(progress: OperatorProgress<T>) -> Result<T> {
    match progress {
        OperatorProgress::Ready(value) => Ok(value),
        OperatorProgress::Waiting(waiting) => Err(model_error(format!(
            "synchronous state-dict operation suspended unexpectedly: {waiting:?}"
        ))),
        OperatorProgress::Unsupported(unsupported) => Err(unsupported_error(unsupported)),
    }
}

fn unsupported_error(unsupported: UnsupportedOperator) -> Error {
    Error::ModelSource {
        source: Box::new(unsupported),
    }
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("standard transformer: {}", message.into()),
    }
}

#[cfg(test)]
mod shared_gate_tests;
