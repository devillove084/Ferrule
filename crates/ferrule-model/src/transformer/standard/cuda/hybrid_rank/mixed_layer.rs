//! Bound Mixed rank program through the shared standard layer composition.
//! KV append/COW/publication and sequence frontier remain journal-owned.
use super::super::bf16::rows_bf16;
use super::*;
use crate::decoder::StandardSequenceState;
use crate::transformer::hybrid_parallel::mixed_channel::Bf16Channel;
use crate::transformer::{
    MemoryLayerWeightCache, PreparedAttentionBlock, PreparedStandardLayer, StandardDecoderKvView,
    StandardTransformerHidden,
};

pub(in crate::transformer::standard::cuda) struct MixedGqaInvocation {
    plan: HybridMixedRankProgramPlan,
    source_retirement: Option<crate::decoder::MixedSourceRetirement>,
    layers: Vec<PreparedStandardLayer>,
    embedding: Option<PreparedEmbedding>,
    output: Option<super::super::super::PreparedStandardOutput>,
    materializer: StateDictMaterializer,
    linears: BTreeMap<(ParameterId, TensorRole), CudaMixedRankLinear>,
    device: HybridCudaDevice,
    credits: Rc<lifecycle::StateCredits>,
    channel: RefCell<Bf16Channel>,
    attention: Option<ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionPlan>,
}
impl MixedGqaInvocation {
    pub(in crate::transformer::standard::cuda) fn performance_quarantined(&self) -> bool {
        self.device.needs_quarantine()
            || self.channel.try_borrow().is_ok_and(|c| c.completion_unknown())
    }
    pub(in crate::transformer::standard::cuda) fn upload_proof_transaction(
        &self,
    ) -> Option<ExecutionTransactionId> {
        let channel = self.channel.try_borrow().ok()?;
        if self.device.needs_quarantine()
            || channel.failed
            || channel.completion_unknown()
            || self.plan.layout.members().len() <= 1
        {
            return None;
        }
        channel.active.map(|(transaction, _, _)| transaction)
    }
    pub(crate) fn diagnostic_layout(&self) -> &HybridRankLayout {
        &self.plan.layout
    }
    pub(in crate::transformer::standard::cuda) fn abort(&self) {
        if let Ok(mut channel) = self.channel.try_borrow_mut() {
            channel.abort();
        }
    }
}

impl CudaStandardDecoderOperators {
    /// Compatibility entry point; no separate pair preparation or execution.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_mixed_rank_gqa_block(
        &mut self,
        resources: &BoundDecoderResources,
        plan: HybridMixedGqaBlockPlan,
        device: HybridCudaDevice,
        max_parameter_bytes: u64,
        projection_limits: [usize; 3],
        auxiliary_grants: usize,
        transport: Box<dyn HybridBf16Collective>,
        attention_shapes: &[ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape],
        attention_limits: [usize; 2],
    ) -> Result<CudaHybridRankLifecycle> {
        self.prepare_mixed_rank_program(
            resources,
            plan,
            device,
            max_parameter_bytes,
            projection_limits,
            auxiliary_grants,
            transport,
            attention_shapes,
            attention_limits,
        )
    }

    /// Prepare the complete bound program on this same rank operator owner.
    /// Projection limits are [weight grants, GemmEx grants, provider reserve].
    /// `auxiliary_grants` bounds native norm/RoPE/vector roots, local chunk
    /// workspace, and transient state-copy grants. Caller still admits operation
    /// workspace and journal physical KV/state capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_mixed_rank_program(
        &mut self,
        resources: &BoundDecoderResources,
        plan: HybridMixedRankProgramPlan,
        device: HybridCudaDevice,
        max_parameter_bytes: u64,
        projection_limits: [usize; 3],
        auxiliary_grants: usize,
        transport: Box<dyn HybridBf16Collective>,
        attention_shapes: &[ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape],
        attention_limits: [usize; 2],
    ) -> Result<CudaHybridRankLifecycle> {
        self.prepare_mixed_rank_program_inner(
            resources,
            plan,
            device,
            max_parameter_bytes,
            projection_limits,
            auxiliary_grants,
            transport,
            attention_shapes,
            attention_limits,
            None,
        )
    }

    /// Prepare the same Mixed rank program under the caller's existing
    /// `HybridCudaPreparationClaim`. The claim is never created or consumed
    /// here; the owner/factory must call `claim.finish(result)` after pool and
    /// initial-state construction. This keeps F32 claim ownership unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_mixed_rank_program_with_claim(
        &mut self,
        resources: &BoundDecoderResources,
        plan: HybridMixedRankProgramPlan,
        device: HybridCudaDevice,
        max_parameter_bytes: u64,
        transport: Box<dyn HybridBf16Collective>,
        attention_shapes: &[ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape],
        claim: &crate::decoder::HybridCudaPreparationClaim,
    ) -> Result<CudaHybridRankLifecycle> {
        self.validate_mixed_preparation_owner()?;
        plan.validate_collective(transport.as_ref())?;
        claim.validate_device(&device)?;
        let (estimate, context) = plan.cuda_claim_inputs(
            resources,
            device.budget().kv_pages,
            device.device_ordinal(),
            max_parameter_bytes,
            attention_shapes,
        )?;
        context.validate_device_budget(device.device_ordinal(), device.budget())?;
        let projection_limits = context.projection_limits;
        let auxiliary_grants = context.auxiliary_grants;
        let attention_limits = context.attention_limits;
        claim.validate_exact_mixed(estimate, context)?;
        self.prepare_mixed_rank_program_inner(
            resources,
            plan,
            device,
            max_parameter_bytes,
            projection_limits,
            auxiliary_grants,
            transport,
            attention_shapes,
            attention_limits,
            Some(claim),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_mixed_rank_program_inner(
        &mut self,
        resources: &BoundDecoderResources,
        plan: HybridMixedRankProgramPlan,
        device: HybridCudaDevice,
        max_parameter_bytes: u64,
        projection_limits: [usize; 3],
        auxiliary_grants: usize,
        transport: Box<dyn HybridBf16Collective>,
        attention_shapes: &[ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape],
        attention_limits: [usize; 2],
        claim: Option<&crate::decoder::HybridCudaPreparationClaim>,
    ) -> Result<CudaHybridRankLifecycle> {
        self.validate_mixed_preparation_owner()?;
        plan.validate_spec(resources.spec())?;
        plan.validate_collective(transport.as_ref())?;
        if self.mixed_gqa_block.is_some()
            || !Rc::ptr_eq(device.operators(), &self.ops)
            || transport.owner() != plan.layout.owner()
            || transport.members() != plan.layout.members()
        {
            return Err(cuda_error("Mixed block owner/model/transport mismatch"));
        }
        device.ensure_ready()?;
        if plan.decoded_weight_cache_budget().is_some() {
            let (_, context) = plan.cuda_claim_inputs(resources, device.budget().kv_pages,
                device.device_ordinal(), max_parameter_bytes, attention_shapes)?;
            context.validate_device_budget(device.device_ordinal(), device.budget())?;
        }

        // Validate every source binding before materialization, including
        // unselected local experts. Matching sizes alone cannot bind an image.
        for parameter in crate::transformer::segment_inventory::HybridSegmentInventory::new(
            resources,
            &plan.segment,
        )?
        .parameters()
        {
            self.bindings.validate_metadata(parameter)?;
        }
        let attention = plan
            .gqa_layer()
            .map(|_| {
                plan.attention_plan(attention_shapes, attention_limits[0], attention_limits[1])
            })
            .transpose()?;
        if attention.is_none() && !attention_shapes.is_empty() {
            return Err(cuda_error(
                "recurrent-only Mixed program must not reserve attention views",
            ));
        }
        let materializer = StateDictMaterializer::for_hybrid_cuda_rank(
            max_parameter_bytes,
            plan.layout.clone(),
            projection_limits[0]
                .checked_add(auxiliary_grants)
                .ok_or_else(|| cuda_error("Mixed materializer grant overflow"))?,
        )?;
        let layers = plan
            .layers()
            .map(|index| {
                super::super::super::hybrid::prepare_standard_layer(
                    &resources.spec().layers()[index],
                    resources,
                    &materializer,
                    &mut MemoryLayerWeightCache::new(),
                    plan.max_positions,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Every local expert shard is prepared once; no provider or checkpoint
        // access occurs in routed execution, including for unselected experts.
        let experts = layers
            .iter()
            .flat_map(|layer| match &layer.feed_forward().block().kind {
                super::super::super::PreparedFeedForwardKind::Routed { experts, .. } => {
                    experts.iter()
                }
                _ => [].iter(),
            })
            .map(|expert| expert.prepare(&materializer))
            .collect::<Result<Vec<_>>>()?;
        let embedding = plan
            .segment
            .owns_embedding()
            .then(|| super::super::super::prepare_embedding(resources, &materializer))
            .transpose()?;
        let output = plan
            .segment
            .owns_output()
            .then(|| super::super::super::prepare_output(resources, &materializer))
            .transpose()?;
        let mut projections = Vec::new();
        let mut vectors = Vec::new();
        let mut ropes = Vec::new();
        if let Some(output) = &output {
            projections.push(output.head());
            vectors.push(output.norm().parameter());
        }
        if let Some(embedding) = &embedding {
            vectors.push(embedding.linear().parameter());
        }
        for layer in &layers {
            vectors.push(layer.feed_forward().block().norm.parameter());
            match layer.attention().block() {
                PreparedAttentionBlock::Gqa(a) => {
                    validate_rope(match resources.spec().layers()[layer.index()].attention() {
                        crate::transformer::Attention::Gqa(a) => a.rotary(),
                        _ => unreachable!(),
                    })?;
                    projections.extend([&a.query, &a.key, &a.value, &a.output]);
                    vectors.push(a.input_norm.parameter());
                    vectors.extend(
                        [a.query_norm.as_ref(), a.key_norm.as_ref()]
                            .into_iter()
                            .flatten()
                            .map(|n| n.parameter()),
                    );
                    if !ropes.contains(&&a.rope) {
                        ropes.push(&a.rope);
                    }
                }
                PreparedAttentionBlock::GatedDeltaNet(d) => {
                    projections.extend([&d.qkv, &d.z, &d.a, &d.beta, &d.output]);
                    vectors.extend([d.input_norm.parameter(), d.norm.parameter()]);
                    vectors.extend(d.vector_parameters.iter().map(AsRef::as_ref));
                    let HybridLocalAttention::GatedDelta { shape, .. } =
                        plan.layout.attention(layer.index())?
                    else {
                        unreachable!()
                    };
                    let channels = shape.sizes()?.0;
                    for (parameter, words) in d.vector_parameters.iter().zip([
                        channels * shape.kernel,
                        shape.value_heads,
                        shape.value_heads,
                    ]) {
                        let rank = parameter
                            .hybrid_rank()
                            .ok_or_else(|| cuda_error("GDN vector lost rank identity"))?;
                        rank.validate_binding(parameter.binding(), &plan.layout)?;
                        if parameter.mixed_vector_bf16_word_count()? != words {
                            return Err(cuda_error("Mixed recurrent local vector extent"));
                        }
                    }
                }
            }
            match &layer.feed_forward().block().kind {
                super::super::super::PreparedFeedForwardKind::Dense(m) => {
                    projections.extend([m.gate(), m.up(), m.down()])
                }
                super::super::super::PreparedFeedForwardKind::Routed {
                    router,
                    shared,
                    shared_gate,
                    ..
                } => {
                    projections.push(router);
                    projections.extend(shared_gate);
                    if let Some(m) = shared {
                        projections.extend([m.gate(), m.up(), m.down()]);
                    }
                }
            }
        }
        for expert in &experts {
            projections.extend([expert.gate(), expert.up(), expert.down()]);
        }
        let prepared = projections
            .iter()
            .map(|p| {
                p.hybrid_rank()
                    .cloned()
                    .ok_or_else(|| cuda_error("Mixed projection lost rank storage"))
            })
            .collect::<Result<Vec<_>>>()?;
        // Production/claim arithmetic stays native. The explicit local-block
        // routed-down M1 candidate has no full-program acceptance receipt yet.
        let block = HybridMixedBlockPlan::new(
            plan.layout.clone(),
            &prepared,
            plan.max_rows,
            HybridMixedProjectionArithmetic::NativeBf16GemmEx,
            projection_limits[0],
            projection_limits[1],
            projection_limits[2],
        )?;
        let block = match plan.decoded_weight_cache_budget() {
            Some(budget) => block.with_decoded_budget(budget)?, None => block,
        };
        let mut auxiliary = 0usize;
        for parameter in &vectors {
            self.bindings.validate(parameter)?;
            let words = parameter.mixed_vector_bf16_word_count()?;
            auxiliary = auxiliary
                .checked_add(
                    ferrule_backend::cuda_allocation_grant_bytes(words * 2)
                        .ok_or_else(|| cuda_error("Mixed vector grant overflow"))?,
                )
                .ok_or_else(|| cuda_error("Mixed vector total overflow"))?;
        }
        for rope in &ropes {
            for words in [rope.cosine().len(), rope.sine().len()] {
                auxiliary = auxiliary
                    .checked_add(
                        ferrule_backend::cuda_allocation_grant_bytes(words * 2)
                            .ok_or_else(|| cuda_error("Mixed RoPE grant overflow"))?,
                    )
                    .ok_or_else(|| cuda_error("Mixed auxiliary total overflow"))?;
            }
        }
        let chunk_bytes = plan.recurrent_workspace_bytes()?;
        if chunk_bytes != 0 {
            let copy_workspace_grants = plan
                .layout
                .state_budget(plan.state_copies)?
                .copy_workspace_grants;
            auxiliary = auxiliary
                .checked_add(
                    ferrule_backend::cuda_allocation_grant_bytes(chunk_bytes)
                        .ok_or_else(|| cuda_error("Mixed recurrent workspace grant overflow"))?,
                )
                .and_then(|n| n.checked_add(copy_workspace_grants))
                .ok_or_else(|| cuda_error("Mixed recurrent grant overflow"))?;
        }
        if auxiliary > auxiliary_grants {
            return Err(cuda_error("Mixed norm/RoPE grant exhausted before upload"));
        }
        if let Some(claim) = claim {
            claim.begin_image(&device)?;
        }
        let claim_device = device.clone();
        let claim_ops = self.ops.clone();
        let before_image = claim_ops.allocator_metrics().live_granted_bytes;
        self.mixed_decoded_enabled = block.decoded.is_some();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.mixed_preparation_with_completion(
                |this| {
                    let tokens = this.prepare_mixed_rank_block_pending(block, &prepared)?;
                    #[cfg(test)]
                    this.mixed_prepare_checkpoint(super::mixed::PreparationStage::Projections)?;
                    // Only this explicit complete block switches generic layer dispatch.
                    this.bf16 = Some(super::super::bf16::State::default());
                    let before = this.ops.allocator_metrics().granted_bytes;
                    this.synchronous(|this| {
                        for parameter in &vectors {
                            this.bf16_weight(parameter)?;
                        }
                        for rope in &ropes {
                            this.bf16_rope_weight(rope)?;
                        }
                        Ok(())
                    })?;
                    #[cfg(test)]
                    this.mixed_prepare_checkpoint(super::mixed::PreparationStage::Vectors)?;
                    this.prepare_bf16_chunk_workspace(chunk_bytes)?;
                    #[cfg(test)]
                    this.mixed_prepare_checkpoint(super::mixed::PreparationStage::Chunk)?;
                    for layer in &layers {
                        if let PreparedAttentionBlock::GatedDeltaNet(d) = layer.attention().block()
                        {
                            this.bf16_deltas.insert(layer.index(), d.clone());
                        }
                    }
                    if this
                        .ops
                        .allocator_metrics()
                        .granted_bytes
                        .checked_sub(before)
                        .is_none_or(|n| n > auxiliary_grants as u64)
                    {
                        return Err(cuda_error(
                            "Mixed auxiliary physical grants exceed admission",
                        ));
                    }
                    this.prepare_bf16_attention(attention.clone())?;
                    #[cfg(test)]
                    this.mixed_prepare_checkpoint(super::mixed::PreparationStage::Attention)?;
                    let linears = projections
                        .iter()
                        .zip(tokens)
                        .map(|(p, t)| ((p.parameter().binding().id(), p.role().clone()), t))
                        .collect();
                    let state = plan.layout.state_budget(plan.state_copies)?;
                    let credits = lifecycle::StateCredits::bounded(
                        state.total_grants,
                        state.per_sequence_grants,
                        plan.state_copies,
                    );
                    let life = CudaHybridRankLifecycle::with_precision(
                        plan.layout.state_schema().clone(),
                        device.clone(),
                        credits.clone(),
                        ExecutionPrecisionPolicy::bf16_compatibility(),
                    )?;
                    let channel = RefCell::new(Bf16Channel::new(transport, plan.sites()));
                    this.mixed_gqa_block = Some(Rc::new(MixedGqaInvocation {
                        plan,
                        source_retirement: claim.map(|c| c.mixed_source_retirement()),
                        layers: layers.clone(),
                        embedding: embedding.clone(),
                        output: output.clone(),
                        materializer,
                        linears,
                        device,
                        credits,
                        channel,
                        attention,
                    }));
                    #[cfg(test)]
                    this.mixed_prepare_checkpoint(super::mixed::PreparationStage::Lifecycle)?;
                    Ok(life)
                },
                || {
                    if let Some(claim) = claim {
                        let actual = claim_ops
                            .allocator_metrics()
                            .live_granted_bytes
                            .checked_sub(before_image)
                            .ok_or_else(|| {
                                cuda_error("Mixed image live grant accounting underflow")
                            })?;
                        claim.validate_image_grants(
                            u64::try_from(actual)
                                .map_err(|_| cuda_error("Mixed image grant exceeds u64"))?,
                        )?;
                        claim.image_prepared()?;
                    }
                    Ok(())
                },
            )
        }));
        if result.is_err() {
            self.poisoned = true;
            self.cache_quarantine();
        }
        if self.needs_quarantine() {
            claim_device.quarantine();
        }
        match result {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Compatibility entry point; executes the same bound program.
    pub fn execute_mixed_rank_gqa_block(
        &mut self,
        batch: &PackedDecoderBatch,
        input: Rows,
        states: &mut [CudaHybridRankState],
        kv: &mut CudaKvView,
        sessions: &[u64],
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<Rows> {
        self.execute_mixed_rank_program(batch, input, states, kv, sessions, check_active)
    }

    pub fn execute_mixed_rank_program(
        &mut self,
        batch: &PackedDecoderBatch,
        input: Rows,
        states: &mut [CudaHybridRankState],
        kv: &mut CudaKvView,
        sessions: &[u64],
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<Rows> {
        let plan = &self
            .mixed_gqa_block
            .as_ref()
            .ok_or_else(|| cuda_error("Mixed program absent"))?
            .plan;
        if plan.segment.owns_embedding() || plan.segment.owns_output() {
            return Err(cuda_error(
                "hidden-only entry point cannot execute endpoint ownership",
            ));
        }
        let input = DeviceSegmentInput::Hidden {
            next_layer: plan.layers().start,
            rows: input,
        };
        match self.execute_mixed_rank_segment(batch, input, states, kv, sessions, check_active)? {
            DeviceSegmentOutput::Hidden { rows, .. } => Ok(rows),
            _ => Err(cuda_error("unexpected Mixed segment output")),
        }
    }

    /// Runtime calls claim.finish after pool/state construction. Retire only on
    /// the next healthy owner entry, before cloning the invocation Rc. Diagnostic
    /// legacy owners explicitly retain prepared sources; late byte access fails closed
    /// through NumericBinding rather than returning a dangling payload view.
    fn retire_mixed_host_sources(&mut self) -> Result<()> {
        if self.needs_quarantine() {
            return Err(cuda_error(
                "cannot retire Mixed sources on a quarantined owner",
            ));
        }
        let Some(block) = self.mixed_gqa_block.as_mut() else {
            return Ok(());
        };
        let Some(completion) = block.source_retirement.clone() else {
            return Ok(());
        };
        if !completion.is_complete() {
            return Err(cuda_error(
                "Mixed source retirement requires successful final claim fence",
            ));
        }
        if self.diagnostic.requires_prepared_sources() {
            return Ok(());
        }
        block.device.ensure_ready()?;
        if !self
            .mixed_rank_block
            .as_ref()
            .is_some_and(|image| image.is_published())
        {
            return Err(cuda_error(
                "Mixed source retirement requires a published device image",
            ));
        }
        {
            let channel = block
                .channel
                .try_borrow()
                .map_err(|_| cuda_error("Mixed source retirement during collective access"))?;
            if channel.active.is_some() || channel.failed || channel.completion_unknown() {
                return Err(cuda_error(
                    "Mixed source retirement outside idle healthy owner",
                ));
            }
        }
        let block = Rc::get_mut(block).ok_or_else(|| {
            cuda_error("Mixed source retirement requires exclusive invocation custody")
        })?;
        let retired_bytes = self.rank_performance.as_ref().map(|_| source_retirement_logical_bytes(block));
        let runtime = super::super::super::mixed_retirement::MixedRuntimeBindings::prepare(
            &block.layers,
            block.output.as_ref(),
            &completion,
        )?;
        // Everything fallible precedes the replacement. No shared Arc is mutated;
        // external aliases (if any) retain their original immutable payloads.
        block.layers = runtime.layers;
        block.output = runtime.output;
        self.bf16_deltas = runtime.deltas;
        block.source_retirement = None;
        if let Some(bytes) = retired_bytes { self.observe_source_retirement(bytes); }
        Ok(())
    }

    /// One entered transaction for every layer in the admitted rank program.
    /// Endpoint arithmetic follows LayerSegmentPlan; no implicit commit or full-model admission.
    /// Any error/panic permanently aborts this rank collective lifetime.
    pub fn execute_mixed_rank_segment(
        &mut self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        states: &mut [CudaHybridRankState],
        kv: &mut CudaKvView,
        sessions: &[u64],
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<DeviceSegmentOutput> {
        let profiling = self.begin_rank_performance(kv.transaction(), batch);
        let retired = if profiling {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.measure_rank_phase(super::super::rank_timing::Phase::SourceRetirement,
                    |this| this.retire_mixed_host_sources())
            })) {
                Ok(result) => result,
                Err(panic) => {
                    self.end_rank_performance("panic");
                    std::panic::resume_unwind(panic)
                }
            }
        } else {
            self.retire_mixed_host_sources()
        };
        if let Err(error) = retired {
            if profiling { self.end_rank_performance("error"); }
            return Err(error);
        }
        let block = self
            .mixed_gqa_block
            .as_ref()
            .cloned()
            .ok_or_else(|| cuda_error("complete Mixed block not prepared"))?;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block.device.ensure_ready()?;
            block
                .plan
                .validate_collective(block.channel.borrow().transport.as_ref())?;
            match &input {
                DeviceSegmentInput::Tokens if block.plan.segment.owns_embedding() => {
                    if batch.token_ids().iter().any(|&id| {
                        id as usize >= block.embedding.as_ref().unwrap().linear().out_features()
                    }) {
                        return Err(cuda_error("Mixed embedding token outside vocabulary"));
                    }
                    block.plan.validate_input(
                        batch,
                        RowsShape::new(batch.len(), block.plan.width)?,
                        RowsDType::Bf16,
                        sessions,
                    )?;
                }
                DeviceSegmentInput::Hidden { next_layer, rows }
                    if !block.plan.segment.owns_embedding()
                        && *next_layer == block.plan.layers().start =>
                {
                    block
                        .plan
                        .validate_input(batch, rows.shape(), rows.dtype(), sessions)?;
                    self.bf16_buffer(rows)?;
                }
                _ => {
                    return Err(cuda_error(
                        "Mixed segment input/endpoint ownership mismatch",
                    ));
                }
            }
            check_active(kv.transaction())?;
            batch.validate_states(states)?;
            kv.validate_batch(batch)?;
            for s in batch.sequences() {
                let state = &states[s.state_index()];
                state.validate_rank(&block.credits)?;
                if state.kv_state().bf16_layers().is_none() {
                    return Err(cuda_error("Mixed block requires typed BF16 state"));
                }
                state.validate_standard_state_at(
                    block.plan.layout.state_schema(),
                    s.context_len(),
                )?;
            }
            let mut offset = 0;
            let groups = batch
                .sequences()
                .iter()
                .enumerate()
                .map(|(sequence, s)| {
                    let rows = (s.sequence_len() - s.context_len()) as u32;
                    let g = ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionGroup {
                        row_start: offset,
                        rows,
                        sequence: sequence as u32,
                        tokens: s.sequence_len() as u32,
                    };
                    offset += rows;
                    g
                })
                .collect::<Vec<_>>();
            if let Some(attention) = &block.attention {
                attention.view(&groups)?;
            }
            let kv_layers = block
                .plan
                .layout
                .state_schema()
                .layers()
                .iter()
                .filter(|s| matches!(s, HybridLayerSchema::FullAttention { .. }))
                .count();
            for schema in block.plan.layout.state_schema().layers() {
                if let HybridLayerSchema::FullAttention {
                    kv_layer,
                    kv_heads,
                    head_dim,
                } = schema
                {
                    kv.with_bf16_gqa_planes(&self.ops, *kv_layer, batch, |p| {
                        if p.layout().layout.elements_per_token != kv_heads * head_dim
                            || p.layout().layout.layer_count != kv_layers
                            || p.layout().layout.page_tokens != block.plan.page_size
                        {
                            return Err(cuda_error(
                                "Mixed program compact BF16 KV geometry mismatch",
                            ));
                        }
                        Ok(())
                    })?;
                }
            }
            block.channel.borrow_mut().transport.validate_execution(
                kv.transaction(),
                batch,
                sessions,
            )?;
            block.channel.borrow_mut().begin(
                kv.transaction(),
                batch.len(),
                batch.sequences().len(),
            )?;
            let input = match input {
                DeviceSegmentInput::Tokens => super::super::super::ready(self.embedding(
                    block.embedding.as_ref().unwrap(),
                    batch.token_ids(),
                    None,
                )?)?,
                DeviceSegmentInput::Hidden { rows, .. } => rows,
            };
            let mut hidden = StandardTransformerHidden {
                rows: Some(input),
                metadata: super::super::super::packed_gqa_metadata(batch)?,
            };
            self.begin_bf16_attention(kv.transaction())?;
            let mut binding = match &self.bf16_attention {
                Some(attention) => {
                    CudaStandardKvBinding::with_prepared_attention(kv, batch, attention.clone())?
                }
                None => CudaStandardKvBinding::new(kv, batch)?,
            };
            // Packed metadata indexes this cohort, not the caller's sparse state slice.
            let mut remaining = states.iter_mut().enumerate().collect::<BTreeMap<_, _>>();
            let mut local = batch
                .sequences()
                .iter()
                .map(|s| {
                    remaining
                        .remove(&s.state_index())
                        .map(|s| s as &mut dyn StandardSequenceState)
                        .ok_or_else(|| cuda_error("Mixed program duplicate state index"))
                })
                .collect::<Result<Vec<_>>>()?;
            for layer in &block.layers {
                self.begin_rank_layer(layer.index());
                self.set_diagnostic_layer(Some(layer.index()));
                self.trace_bf16(
                    "mixed.layer.input",
                    self.bf16_buffer(hidden.rows()?)?,
                    hidden.rows()?.shape(),
                    None,
                    None,
                )?;
                check_active(binding.transaction())?;
                let entry = &block.plan.layout.state_schema().entries()
                    [layer.index() - block.plan.layers().start];
                super::super::super::execute_standard_layer_inner(
                    self,
                    &block.materializer,
                    layer.index(),
                    layer.attention().block().as_ref(),
                    layer.feed_forward().block(),
                    entry.local_kv_layer.unwrap_or(0),
                    &mut hidden,
                    &mut binding,
                    &mut local,
                    None,
                )?;
                self.trace_bf16(
                    "mixed.layer.output",
                    self.bf16_buffer(hidden.rows()?)?,
                    hidden.rows()?.shape(),
                    None,
                    None,
                )?;
                self.end_rank_layer("ok");
            }
            self.set_diagnostic_layer(None);
            drop(local);
            drop(remaining);
            let output = match &block.output {
                Some(output) => DeviceSegmentOutput::Logits(
                    self.measure_rank_phase(super::super::rank_timing::Phase::Head, |this| super::super::super::execute_standard_output_rows(this, output, &hidden))?,
                ),
                None => DeviceSegmentOutput::Hidden {
                    next_layer: block.plan.layers().end,
                    rows: hidden.take_rows()?,
                },
            };
            self.finish_bf16_attention(binding.transaction(), false)?;
            self.quiesce()?;
            check_active(binding.transaction())?;
            for s in batch.sequences() {
                states[s.state_index()].validate_standard_state_at(
                    block.plan.layout.state_schema(),
                    s.sequence_len(),
                )?;
            }
            block
                .channel
                .borrow_mut()
                .transport
                .complete_owner(binding.transaction())?;
            check_active(binding.transaction())?;
            block.channel.borrow_mut().finish()?;
            Ok(output)
        }));
        match outcome {
            Ok(Ok(output)) => {
                if profiling { self.end_rank_performance("ok"); }
                Ok(output)
            },
            Ok(Err(primary)) => {
                let cleanup = self.finish_bf16_attention(kv.transaction(), true);
                let failure = block.channel.borrow_mut().fail_cleanup(
                    primary,
                    cleanup,
                    self.needs_quarantine() || block.device.needs_quarantine(),
                );
                if block.channel.borrow().completion_unknown() {
                    self.poisoned = true;
                    self.cache_quarantine();
                    block.device.quarantine();
                }
                if profiling { self.end_rank_performance("error"); }
                Err(failure)
            }
            Err(panic) => {
                let cleanup = self.finish_bf16_attention(kv.transaction(), true);
                let outcome = block.channel.borrow_mut().panic_cleanup(
                    panic,
                    cleanup,
                    self.needs_quarantine() || block.device.needs_quarantine(),
                );
                if block.channel.borrow().completion_unknown() {
                    self.poisoned = true;
                    self.cache_quarantine();
                    block.device.quarantine();
                }
                if profiling { self.end_rank_performance("panic"); }
                match outcome {
                    Ok(panic) => std::panic::resume_unwind(panic),
                    Err(unknown) => Err(unknown),
                }
            }
        }
    }

    fn mixed_gather_head(&mut self, head: &PreparedLinear, local: Rows) -> Result<Rows> {
        self.measure_rank_phase(super::super::rank_timing::Phase::Collective, |this| {
        let block = this.mixed_gqa_block.as_ref().unwrap().clone();
        if !block.plan.segment.owns_output() {
            return Err(cuda_error("Mixed output head not owned"));
        }
        let partition = block
            .plan
            .layout
            .parameter_partition(head.parameter().binding())?;
        let plan = partition
            .collective
            .ok_or_else(|| cuda_error("Mixed head lost column collective"))?;
        let rows = local.shape().rows();
        let words = this.mixed_words(&local)?;
        let payload = plan.pack_column_bf16(
            ParallelRankId::new(block.plan.layout.lane() as u32),
            &words,
            rows,
        )?;
        let gathered = block
            .channel
            .borrow_mut()
            .exchange(HybridCollectiveSemantic::OutputHead, payload)?;
        let output = plan.unpack_column_bf16(&gathered, rows)?;
        this.synchronous(|this| {
            let buffer = this.ops.upload_bf16_words(&output)?;
            this.hold_bf16(&buffer)?;
            rows_bf16(
                RowsShape::new(rows, plan.out_features())?,
                local.arena(),
                buffer,
            )
        })
        })
    }

    fn mixed_words(&mut self, rows: &Rows) -> Result<Vec<u16>> {
        self.synchronous(|this| {
            this.hold_bf16(this.bf16_buffer(rows)?)?;
            this.bf16_buffer(rows)?
                .as_device_buffer()
                .to_host_vec(&this.ops.stream_clone())
                .map_err(|e| cuda_error(format!("Mixed BF16 transport DMA: {e}")))
        })
    }
    fn mixed_reduce_rows(
        &mut self,
        rows: Rows,
        semantic: HybridCollectiveSemantic,
    ) -> Result<Rows> {
        self.measure_rank_phase(super::super::rank_timing::Phase::Collective, |this| {
        let block = this
            .mixed_gqa_block
            .as_ref()
            .cloned()
            .ok_or_else(|| cuda_error("Mixed complete block not attached"))?;
        let (before, after) = match semantic {
            HybridCollectiveSemantic::AttentionOutput => {
                ("mixed.attention.partial", "mixed.attention.reduced")
            }
            HybridCollectiveSemantic::RoutedExpertOutput => {
                ("mixed.moe.partial", "mixed.moe.reduced")
            }
            _ => ("mixed.dense.partial", "mixed.dense.reduced"),
        };
        this.trace_bf16(before, this.bf16_buffer(&rows)?, rows.shape(), None, None)?;
        let words = this.mixed_words(&rows)?;
        let result = block.channel.borrow_mut().exchange(semantic, words)?;
        this.synchronous(|this| {
            let output = this.ops.upload_bf16_words(&result)?;
            this.hold_bf16(&output)?;
            this.trace_bf16(after, &output, rows.shape(), None, None)?;
            rows_bf16(rows.shape(), rows.arena(), output)
        })
        })
    }
    pub(in crate::transformer::standard::cuda) fn mixed_block_delta_shape(
        &self,
        layer: usize,
    ) -> Result<ferrule_backend::cpu::gated_delta::GatedDeltaShape> {
        let block = self
            .mixed_gqa_block
            .as_ref()
            .ok_or_else(|| cuda_error("Mixed recurrent block absent"))?;
        if !block.plan.layers().contains(&layer) || block.channel.borrow().active.is_none() {
            return Err(cuda_error("GDN outside admitted Mixed invocation"));
        }
        match block.plan.layout.attention(layer)? {
            HybridLocalAttention::GatedDelta { shape, .. } => Ok(*shape),
            _ => Err(cuda_error("Mixed recurrent geometry absent")),
        }
    }
    pub(in crate::transformer::standard::cuda) fn mixed_block_heads(
        &self,
        query: usize,
        kv: usize,
    ) -> Result<(usize, usize)> {
        let b = self.mixed_gqa_block.as_ref().unwrap();
        let HybridLocalAttention::Gqa {
            query_heads,
            kv_heads,
            ..
        } = b.plan.layout.attention(
            b.plan
                .gqa_layer()
                .ok_or_else(|| cuda_error("Mixed GQA geometry absent"))?,
        )?
        else {
            unreachable!()
        };
        if query_heads.len() * 2 != query || kv_heads.len() != (kv / 2).max(1) {
            return Err(cuda_error("Mixed block head mismatch"));
        }
        Ok((query_heads.len(), kv_heads.len()))
    }
    pub(in crate::transformer::standard::cuda) fn mixed_block_linear(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
    ) -> Result<Rows> {
        let b = self.mixed_gqa_block.as_ref().unwrap();
        if b.channel.borrow().active.is_none() || b.channel.borrow().failed {
            return Err(cuda_error("Mixed projection outside complete invocation"));
        }
        let token = b
            .linears
            .get(&(linear.parameter().binding().id(), linear.role().clone()))
            .cloned()
            .ok_or_else(|| cuda_error("projection outside Mixed block"))?;
        use super::super::rank_timing::Phase;
        let phase = match linear.role() {
            TensorRole::AttentionQuery | TensorRole::AttentionKey | TensorRole::AttentionValue |
            TensorRole::AttentionOutput | TensorRole::LinearAttentionQkv | TensorRole::LinearAttentionZ |
            TensorRole::LinearAttentionA | TensorRole::LinearAttentionBeta => Phase::AttentionProjection,
            TensorRole::RouterLogits => Phase::Router,
            TensorRole::SharedExpertOutputGate => Phase::SharedFfn,
            TensorRole::OutputHead => Phase::Head,
            _ => Phase::Other,
        };
        let output = self.measure_rank_phase(phase, |this| this.mixed_rank_linear_local(&token, input))?;
        if linear.role() == &TensorRole::OutputHead {
            self.mixed_gather_head(linear, output)
        } else if linear.role() == &TensorRole::AttentionOutput {
            self.mixed_reduce_rows(output, HybridCollectiveSemantic::AttentionOutput)
        } else {
            Ok(output)
        }
    }
    pub(in crate::transformer::standard::cuda) fn mixed_block_swiglu(
        &mut self,
        ff: &PreparedSwiGlu,
        input: &Rows,
    ) -> Result<Rows> {
        let b = self.mixed_gqa_block.as_ref().unwrap();
        if b.channel.borrow().active.is_none() || b.channel.borrow().failed {
            return Err(cuda_error("Mixed MLP outside complete invocation"));
        }
        let tokens = [ff.gate(), ff.up(), ff.down()].map(|p| {
            b.linears
                .get(&(p.parameter().binding().id(), p.role().clone()))
                .cloned()
        });
        let [Some(g), Some(u), Some(d)] = tokens else {
            return Err(cuda_error("MLP outside Mixed complete block"));
        };
        let phase = if ff.down().role() == &TensorRole::DenseMlpDown {
            super::super::rank_timing::Phase::DenseFfn
        } else {
            super::super::rank_timing::Phase::SharedFfn
        };
        let output = self.measure_rank_phase(phase,
            |this| this.mixed_rank_swiglu_local([&g, &u, &d], input))?;
        self.trace_bf16(
            "mixed.swiglu.partial",
            self.bf16_buffer(&output)?,
            output.shape(),
            Some(ff.down()),
            None,
        )?;
        match ff.down().role() {
            TensorRole::DenseMlpDown => {
                self.mixed_reduce_rows(output, HybridCollectiveSemantic::DenseMlpDown)
            }
            TensorRole::SharedExpertDown => Ok(output),
            _ => Err(cuda_error("Mixed FFN must be dense or local shared")),
        }
    }

    pub(in crate::transformer::standard::cuda) fn mixed_block_routed(
        &mut self,
        layer: usize,
        input: &Rows,
        routes: &RouterRoutes,
    ) -> Result<Rows> {
        let block = self.mixed_gqa_block.as_ref().unwrap().clone();
        let site = block
            .plan
            .route_sites()
            .into_iter()
            .find(|s| s.layer == layer)
            .ok_or_else(|| cuda_error("routed layer outside Mixed program"))?;
        let agreement = HybridMixedRouteAgreement::new(site.clone(), routes)?;
        self.measure_rank_phase(super::super::rank_timing::Phase::RouteAgreement,
            |_| block.channel.borrow_mut().agree_routes(&agreement))?;
        let output = self.measure_rank_phase(super::super::rank_timing::Phase::RoutedFfn,
            |this| this.mixed_rank_routed_local(layer, input, routes, site.accumulation))?;
        self.trace_bf16(
            "mixed.routed.partial",
            self.bf16_buffer(&output)?,
            output.shape(),
            None,
            None,
        )?;
        Ok(output)
    }

    pub(in crate::transformer::standard::cuda) fn mixed_finish_routed(
        &mut self,
        layer: usize,
        output: Rows,
    ) -> Result<Rows> {
        let block = self.mixed_gqa_block.as_ref().unwrap().clone();
        block.channel.borrow().validate_routed_output(layer)?;
        self.mixed_reduce_rows(output, HybridCollectiveSemantic::RoutedExpertOutput)
    }
}

// Logical source bytes removed from primary layer/head bindings. Recurrent
// descriptor aliases are deliberately not counted twice; this is not OS free.
fn source_retirement_logical_bytes(block: &MixedGqaInvocation) -> u64 {
    let mut bytes = 0u64;
    let mut count = |linear: &PreparedLinear| {
        if let Some(rank) = linear.hybrid_rank()
            && let HybridRankStorage::Numeric(payload) = rank.storage()
        { bytes = bytes.saturating_add(payload.plan().budget().payload_bytes as u64); }
    };
    for layer in &block.layers {
        match layer.attention().block() {
            PreparedAttentionBlock::Gqa(a) => { for l in [&a.query, &a.key, &a.value, &a.output] { count(l); } }
            PreparedAttentionBlock::GatedDeltaNet(a) => { for l in [&a.qkv, &a.z, &a.a, &a.beta, &a.output] { count(l); } }
        }
        let shared = match &layer.feed_forward().block().kind {
            super::super::super::PreparedFeedForwardKind::Dense(s) => Some(s),
            super::super::super::PreparedFeedForwardKind::Routed { router, shared, shared_gate, .. } => {
                count(router); if let Some(g) = shared_gate { count(g); } shared.as_ref()
            }
        };
        if let Some(s) = shared { for l in [s.gate(), s.up(), s.down()] { count(l); } }
    }
    if let Some(output) = &block.output { count(output.head()); }
    bytes
}
