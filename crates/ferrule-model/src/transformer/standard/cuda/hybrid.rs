//! CUDA protocol adapter over the one standard hybrid layer/output composition.
use super::super::{PreparedStandardLayer, PreparedStandardOutput, StandardTransformerHidden};
use super::*;
use crate::decoder::{
    CudaHybridSequenceState, CudaKvView, DecoderLogits, HybridCudaDevice, HybridLayerSchema,
    HybridStateSchema, NoTerminalGuard, PackedDecoderBatch, StandardSequenceState,
};
use crate::transformer::{
    Attention, BoundDecoderResources, FeedForward, LayerRequest, MemoryLayerWeightCache,
    NoPostLayerTap, Poll, PreparedTransformer, StateDictMaterializer, Step, TransformerModule,
};
use std::convert::Infallible;

#[derive(Debug)]
pub struct CudaHybridModule {
    resources: Arc<BoundDecoderResources>,
    materializer: Arc<StateDictMaterializer>,
    device: HybridCudaDevice,
    schema: HybridStateSchema,
    #[cfg(test)]
    reject_next_finish_completion: std::cell::Cell<bool>,
    operators: CudaStandardDecoderOperators,
    prepared: PreparedTransformer<
        PreparedEmbedding,
        PreparedStandardLayer,
        PreparedStandardOutput,
        NoPostLayerTap,
    >,
}
impl CudaHybridModule {
    pub fn prepare(
        resources: Arc<BoundDecoderResources>,
        device: HybridCudaDevice,
        options: &crate::decoder::GenericDecoderOptions,
    ) -> Result<Self> {
        let spec = resources.spec();
        let active_layers = options.active_layers().unwrap_or(spec.layers().len());
        if options.precision() != ExecutionPrecisionPolicy::f32()
            || options.max_positions() == 0
            || spec
                .max_sequence_length()
                .is_some_and(|max| options.max_positions() > max)
        {
            return Err(unsupported(
                "CUDA hybrid requires F32 and a supported context length",
            ));
        }
        let schema = HybridStateSchema::from_spec(spec, active_layers)?;
        super::super::validate_hybrid_descriptors(spec, spec.layers())?;
        for layer in spec.layers() {
            if !matches!(layer.feed_forward(), FeedForward::SwiGlu(_)) {
                return Err(unsupported(
                    "hybrid CUDA supports dense single-device execution only; MoE/TP/PP unsupported",
                ));
            }
            if let Attention::Gqa(g) = layer.attention() {
                validate_rope(g.rotary())?;
            }
        }
        if !spec.attachments().is_empty() {
            return Err(unsupported("hybrid CUDA attachments/proposals unsupported"));
        }
        for parameter in resources.state_dict().parameters() {
            if !matches!(
                parameter.weight().slice().dtype,
                crate::checkpoint::CheckpointDType::F32 | crate::checkpoint::CheckpointDType::Bf16
            ) || parameter.scale().is_some()
            {
                return Err(unsupported(
                    "hybrid CUDA supports unscaled F32/BF16 checkpoint storage only",
                ));
            }
        }
        device.ensure_ready()?;
        let materializer = Arc::new(StateDictMaterializer::new(options.max_parameter_bytes())?);
        let embedding = super::super::prepare_embedding(&resources, &materializer)?;
        let output = super::super::prepare_output(&resources, &materializer)?;
        let mut layers = Vec::new();
        for layer in &spec.layers()[..active_layers] {
            layers.push(super::super::hybrid::prepare_standard_layer(
                layer,
                &resources,
                &materializer,
                &mut MemoryLayerWeightCache::new(),
                options.max_positions(),
            )?);
        }
        let mut operators = CudaStandardDecoderOperators::new(
            Rc::clone(device.operators()),
            options.precision(),
            resources.state_dict().parameters(),
        )?;
        let result = operators.prepare_hybrid_image(&embedding, &layers, &output);
        if operators.needs_quarantine() {
            device.quarantine();
        }
        result?;
        if operators.resident_parameter_bytes() > device.budget().weight_bytes {
            return Err(unsupported("resident CUDA weights exceed budget"));
        }
        Ok(Self {
            resources,
            materializer,
            device,
            schema,
            #[cfg(test)]
            reject_next_finish_completion: std::cell::Cell::new(false),
            operators,
            prepared: PreparedTransformer::new(embedding, layers, output, NoPostLayerTap),
        })
    }
    #[cfg(test)]
    pub(crate) fn reject_next_finish_completion(&self) {
        self.reject_next_finish_completion.set(true);
    }

    pub fn resident_parameter_bytes(&self) -> usize {
        self.operators.resident_parameter_bytes()
    }
    pub fn spec(&self) -> &crate::transformer::DecoderModelSpec {
        self.resources.spec()
    }
    fn checked<T>(&mut self, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.device.ensure_ready()?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(self)));
        if self.operators.needs_quarantine() {
            self.device.quarantine();
        }
        match result {
            Ok(result) => result,
            Err(panic) => {
                self.device.quarantine();
                std::panic::resume_unwind(panic)
            }
        }
    }
    fn execute_layer(
        &mut self,
        index: usize,
        batch: &PackedDecoderBatch,
        states: &mut [CudaHybridSequenceState],
        kv: &mut CudaKvView,
        hidden: &mut StandardTransformerHidden,
    ) -> Result<()> {
        self.checked(|this| {
            let layer = this
                .prepared
                .layers()
                .get(index)
                .ok_or_else(|| cuda_error("missing CUDA hybrid layer"))?;
            let kv_layer = match this.schema.layers()[index] {
                HybridLayerSchema::FullAttention { kv_layer, .. } => kv_layer,
                _ => 0,
            };
            let mut binding = CudaStandardKvBinding::new(kv, batch)?;
            let mut states = states
                .iter_mut()
                .map(|s| s as &mut dyn StandardSequenceState)
                .collect::<Vec<_>>();
            super::super::execute_standard_layer_inner(
                &mut this.operators,
                &this.materializer,
                layer.index(),
                layer.attention().block().as_ref(),
                layer.feed_forward().block(),
                kv_layer,
                hidden,
                &mut binding,
                &mut states,
                None,
            )
        })
    }
}
fn unsupported(message: &str) -> Error {
    Error::ModelSource {
        source: Box::new(UnsupportedOperator::new("hybrid_cuda", message)),
    }
}

impl TransformerModule for CudaHybridModule {
    type State = CudaHybridSequenceState;
    type KvView = CudaKvView;
    type Hidden = StandardTransformerHidden;
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
        self.device.ensure_ready()?;
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
        self.checked(|this| {
            let metadata = super::super::packed_gqa_metadata(batch)?;
            let rows = super::super::ready(this.operators.embedding(
                this.prepared.embedding(),
                batch.token_ids(),
                None,
            )?)?;
            Ok(Step::Complete(StandardTransformerHidden {
                rows: Some(rows),
                metadata,
            }))
        })
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
        let LayerRequest::Target { batch, kv, states } = request else {
            return Err(cuda_error("CUDA hybrid module has no proposal stages"));
        };
        self.execute_layer(layer, batch, states, kv, hidden)?;
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
        self.checked(|this| {
            Ok(Step::Complete(DecoderLogits::Dense(
                super::super::execute_standard_output(
                    &mut this.operators,
                    this.prepared.output(),
                    hidden,
                )?,
            )))
        })
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
        self.checked(|this| this.operators.quiesce())?;
        self.device.fence()?;
        Ok(())
    }

    fn poll_quiescence(&mut self, _quiescence: &mut Self::Quiescence) -> Result<bool> {
        self.device.ensure_ready()?;
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
        // Simulate lost completion evidence, not an illegal GPU operation. The
        // real fence then returns the owner's permanent typed Unknown result.
        #[cfg(test)]
        if self.reject_next_finish_completion.replace(false) {
            self.device.quarantine();
        }
        self.device.fence()?;
        for sequence in batch.sequences() {
            let state = states
                .get(sequence.state_index())
                .ok_or_else(|| cuda_error("missing hybrid sequence at completion"))?;
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
        self.device.fence()
    }
}
