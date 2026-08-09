//! Prepared CPU GQA/Add/MoE transformer composition.

use std::convert::Infallible;
use std::sync::Arc;

use ferrule_common::{Error, Result};

use crate::decoder::{
    CpuKvView, DecoderLogits, DenseLogits, GenericDecoderSequenceState, NoTerminalGuard,
    PackedDecoderBatch,
};
use crate::execution::ExecutionPrecisionPolicy;
use crate::support::TensorRole;

use super::{
    AddResidual, Attention, BoundDecoderResources, Connected, CpuStandardDecoderOperators,
    DecoderLayer, DecoderModelSpec, ExpertAvailability, ExpertProvider, FeedForward, GqaMetadata,
    GqaRequest, LayerRequest, Linear, MemoryLayerWeightCache, Moe, NoPostLayerTap, NoReduction,
    OperatorProgress, OutputPipeline, Poll, PreparedEmbedding, PreparedLinear, PreparedNorm,
    PreparedRope, PreparedSwiGlu, PreparedTransformer, Residual, RmsNorm, RotaryEmbedding,
    RotaryScaling, Rows, StandardDecoderOperators, StateDictMaterializer, Step, SwiGlu,
    TransformerLayer, TransformerModule,
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
    },
}

/// Concrete Qwen-compatible layer composition: GQA + Add + MoE + Add.
pub type PreparedGqaMoeLayer = TransformerLayer<
    Connected<AddResidual, PreparedGqaBlock>,
    Connected<AddResidual, PreparedFeedForwardBlock>,
>;

/// Concrete model-independent output pipeline used by one-stream CPU decoders.
pub type PreparedCpuOutput = OutputPipeline<NoReduction, PreparedNorm, PreparedLinear>;

#[derive(Debug)]
pub struct CpuTransformerHidden {
    rows: Option<Rows>,
    metadata: GqaMetadata,
}

impl CpuTransformerHidden {
    fn rows(&self) -> Result<&Rows> {
        self.rows
            .as_ref()
            .ok_or_else(|| model_error("CPU hidden rows are checked out"))
    }

    fn take_rows(&mut self) -> Result<Rows> {
        self.rows
            .take()
            .ok_or_else(|| model_error("CPU hidden rows are checked out"))
    }
}

/// Prepared CPU module selected by Qwen and other ordinary GQA/Add decoders.
#[derive(Debug)]
pub struct CpuGqaMoeModule {
    resources: Arc<BoundDecoderResources>,
    materializer: Arc<StateDictMaterializer>,
    operators: CpuStandardDecoderOperators,
    prepared: PreparedTransformer<
        PreparedEmbedding,
        PreparedGqaMoeLayer,
        PreparedCpuOutput,
        NoPostLayerTap,
    >,
}

struct StateDictExpertProvider<'a> {
    resources: &'a BoundDecoderResources,
    materializer: &'a StateDictMaterializer,
    layers: &'a [PreparedGqaMoeLayer],
}

impl<'a> StateDictExpertProvider<'a> {
    fn new(
        resources: &'a BoundDecoderResources,
        materializer: &'a StateDictMaterializer,
        layers: &'a [PreparedGqaMoeLayer],
    ) -> Self {
        Self {
            resources,
            materializer,
            layers,
        }
    }

    fn expert_linear(
        &self,
        layer: usize,
        expert: usize,
        role: TensorRole,
        shape: [usize; 2],
    ) -> Result<PreparedLinear> {
        let binding =
            self.resources
                .experts()
                .require_shape(layer, expert, role.clone(), &shape)?;
        let parameter = self.materializer.expert_parameter(layer, expert, binding)?;
        PreparedLinear::from_parameter(parameter, role)
    }
}

impl ExpertProvider for StateDictExpertProvider<'_> {
    fn expert(&mut self, layer: usize, expert: usize) -> Result<ExpertAvailability> {
        let descriptor = self
            .layers
            .get(layer)
            .ok_or_else(|| model_error(format!("missing prepared layer {layer}")))?;
        let PreparedFeedForwardKind::Routed {
            descriptor: moe, ..
        } = &descriptor.feed_forward().block().kind
        else {
            return Ok(ExpertAvailability::Unsupported(format!(
                "layer {layer} is not routed"
            )));
        };
        let expert_descriptor = moe.expert();
        let gate = self.expert_linear(
            layer,
            expert,
            TensorRole::RoutedExpertGate,
            expert_descriptor.gate().weight_shape(),
        )?;
        let up = self.expert_linear(
            layer,
            expert,
            TensorRole::RoutedExpertUp,
            expert_descriptor.up().weight_shape(),
        )?;
        let down = self.expert_linear(
            layer,
            expert,
            TensorRole::RoutedExpertDown,
            expert_descriptor.down().weight_shape(),
        )?;
        Ok(ExpertAvailability::Ready(Arc::new(PreparedSwiGlu::new(
            gate,
            up,
            down,
            expert_descriptor.activation_limit(),
        )?)))
    }
}

impl CpuGqaMoeModule {
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
        validate_gqa_add_descriptors(spec)?;
        let embedding_parameter =
            materializer.static_parameter(resources.require_static_shape(
                TensorRole::TokenEmbedding,
                &spec.token_embedding().weight_shape(),
            )?)?;
        let embedding = PreparedEmbedding::new(PreparedLinear::from_parameter(
            embedding_parameter,
            TensorRole::TokenEmbedding,
        )?);
        let final_norm = PreparedNorm::new(
            materializer.static_parameter(resources.require_static_shape(
                TensorRole::OutputNorm,
                &spec.final_norm().weight_shape(),
            )?)?,
            spec.final_norm().epsilon(),
        )?;
        let output_head = if spec.tie_word_embeddings() {
            embedding.linear().clone()
        } else {
            PreparedLinear::from_parameter(
                materializer.static_parameter(resources.require_static_shape(
                    TensorRole::OutputHead,
                    &spec.output().weight_shape(),
                )?)?,
                TensorRole::OutputHead,
            )?
        };
        let mut layers = Vec::with_capacity(active_layers);
        for descriptor in &spec.layers()[..active_layers] {
            let mut cache = MemoryLayerWeightCache::new();
            layers.push(prepare_layer(
                descriptor,
                &resources,
                &materializer,
                &mut cache,
                max_positions,
            )?);
        }
        Ok(Self {
            resources,
            materializer,
            operators: CpuStandardDecoderOperators::new(precision),
            prepared: PreparedTransformer::new(
                embedding,
                layers,
                OutputPipeline::new(NoReduction, final_norm, output_head),
                NoPostLayerTap,
            ),
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
    ) -> Result<()> {
        let layer = self
            .prepared
            .layers()
            .get(layer_index)
            .ok_or_else(|| model_error(format!("missing prepared layer {layer_index}")))?;
        let attention = layer.attention().block();
        let Attention::Gqa(gqa) = attention.descriptor.attention() else {
            return Err(model_error("prepared GQA layer lost its descriptor"));
        };
        let arena = None;
        let normalized =
            ready(
                self.operators
                    .rms_norm(&attention.input_norm, hidden.rows()?, 1, arena)?,
            )?;
        let mut query = ready(
            self.operators
                .linear(&attention.query, &normalized, arena)?,
        )?;
        let mut key = ready(self.operators.linear(&attention.key, &normalized, arena)?)?;
        let value = ready(
            self.operators
                .linear(&attention.value, &normalized, arena)?,
        )?;
        if let Some(norm) = &attention.query_norm {
            query = ready(
                self.operators
                    .rms_norm(norm, &query, gqa.num_heads(), arena)?,
            )?;
        }
        if let Some(norm) = &attention.key_norm {
            key = ready(
                self.operators
                    .rms_norm(norm, &key, gqa.num_kv_heads(), arena)?,
            )?;
        }
        query = ready(self.operators.rope(
            gqa.rotary(),
            &attention.rope,
            query,
            gqa.num_heads(),
            hidden.metadata.row_positions(),
        )?)?;
        key = ready(self.operators.rope(
            gqa.rotary(),
            &attention.rope,
            key,
            gqa.num_kv_heads(),
            hidden.metadata.row_positions(),
        )?)?;
        let update = ready(self.operators.paged_gqa(
            kv,
            GqaRequest {
                layer: attention.descriptor.index(),
                query: &query,
                key: &key,
                value: &value,
                metadata: &hidden.metadata,
                query_heads: gqa.num_heads(),
                kv_heads: gqa.num_kv_heads(),
                head_dim: gqa.head_dim(),
                softmax_scale: (gqa.head_dim() as f32).sqrt().recip(),
                arena,
            },
        )?)?;
        let update = ready(self.operators.linear(&attention.output, &update, arena)?)?;
        hidden.rows = Some(ready(
            self.operators.residual(hidden.take_rows()?, &update)?,
        )?);

        let feed_forward = layer.feed_forward().block();
        let normalized =
            ready(
                self.operators
                    .rms_norm(&feed_forward.norm, hidden.rows()?, 1, arena)?,
            )?;
        let update = match &feed_forward.kind {
            PreparedFeedForwardKind::Dense(feed_forward) => ready(self.operators.dense_swiglu(
                feed_forward,
                &normalized,
                arena,
            )?)?,
            PreparedFeedForwardKind::Routed {
                router,
                descriptor,
                shared,
            } => {
                let logits = ready(self.operators.linear(router, &normalized, arena)?)?;
                let routes = ready(self.operators.router(&logits, descriptor.router_spec())?)?;
                let shared = shared
                    .as_ref()
                    .map(|shared| self.operators.dense_swiglu(shared, &normalized, arena))
                    .transpose()?
                    .map(ready)
                    .transpose()?;
                let mut experts = StateDictExpertProvider::new(
                    &self.resources,
                    &self.materializer,
                    self.prepared.layers(),
                );
                let mut routed = ready(self.operators.routed_swiglu(
                    attention.descriptor.index(),
                    &normalized,
                    &routes,
                    &mut experts,
                    arena,
                )?)?;
                if let Some(shared) = &shared {
                    routed = ready(self.operators.residual(routed, shared)?)?;
                }
                routed
            }
        };
        hidden.rows = Some(ready(
            self.operators.residual(hidden.take_rows()?, &update)?,
        )?);
        Ok(())
    }
}

impl TransformerModule for CpuGqaMoeModule {
    type State = GenericDecoderSequenceState;
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
        _states: &mut [Self::State],
        kv: &mut Self::KvView,
    ) -> Result<()> {
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
        let LayerRequest::Target { kv, .. } = request else {
            return Err(model_error("CPU GQA module does not have proposal stages"));
        };
        self.execute_layer(layer, hidden, kv)?;
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
        let normalized = ready(self.operators.rms_norm(
            self.prepared.output().norm(),
            hidden.rows()?,
            1,
            None,
        )?)?;
        let logits = ready(self.operators.lm_head(
            self.prepared.output().head(),
            &normalized,
            None,
        )?)?
        .into_host()?;
        Ok(Step::Complete(DecoderLogits::Dense(DenseLogits::new(
            logits.shape().rows(),
            logits.shape().width(),
            logits.values().to_vec(),
        )?)))
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
        _batch: &PackedDecoderBatch,
        _states: &mut [Self::State],
        _kv: &mut Self::KvView,
        _arena: &mut Self::Arena,
        events: Vec<Self::Event>,
    ) -> Result<Self::TerminalGuard> {
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

fn validate_gqa_add_descriptors(spec: &DecoderModelSpec) -> Result<()> {
    for layer in spec.layers() {
        if !matches!(layer.attention(), Attention::Gqa(_)) {
            return Err(model_error(format!(
                "layer {} requires GQA attention",
                layer.index()
            )));
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

fn prepare_layer(
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
    let feed_forward = match descriptor.feed_forward() {
        FeedForward::SwiGlu(feed_forward) => PreparedFeedForwardKind::Dense(prepare_swiglu(
            layer,
            feed_forward,
            resources,
            materializer,
            cache,
            false,
        )?),
        FeedForward::Moe(moe) => PreparedFeedForwardKind::Routed {
            router: prepare_layer_linear(
                layer,
                TensorRole::RouterLogits,
                moe.router(),
                resources,
                materializer,
                cache,
            )?,
            descriptor: moe.clone(),
            shared: moe
                .shared_expert()
                .map(|shared| prepare_swiglu(layer, shared, resources, materializer, cache, true))
                .transpose()?,
        },
    };
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
        Connected::new(
            AddResidual,
            PreparedFeedForwardBlock {
                norm: post_attention_norm,
                kind: feed_forward,
            },
        ),
    ))
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
    let parameter = materializer.layer_parameter(
        layer,
        resources.require_layer_shape(layer, role.clone(), &descriptor.weight_shape())?,
        cache,
    )?;
    let linear = PreparedLinear::from_parameter(parameter, role.clone())?;
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
    PreparedNorm::new(
        materializer.layer_parameter(
            layer,
            resources.require_layer_shape(layer, role, &descriptor.weight_shape())?,
            cache,
        )?,
        descriptor.epsilon(),
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

fn packed_gqa_metadata(batch: &PackedDecoderBatch) -> Result<GqaMetadata> {
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

fn ready<T>(progress: OperatorProgress<T>) -> Result<T> {
    match progress {
        OperatorProgress::Ready(value) => Ok(value),
        OperatorProgress::Waiting(waiting) => Err(model_error(format!(
            "CPU state-dict operation suspended unexpectedly: {waiting:?}"
        ))),
        OperatorProgress::Unsupported(unsupported) => Err(model_error(format!(
            "operator '{}' is unsupported: {}",
            unsupported.operator, unsupported.reason
        ))),
    }
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("CPU GQA/MoE transformer: {}", message.into()),
    }
}
