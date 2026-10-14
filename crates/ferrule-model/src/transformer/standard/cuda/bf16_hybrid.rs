//! Native BF16 hybrid semantic operators on the existing standard layer loop.
use super::bf16::{round_rows, rows_bf16};
use super::*;
use crate::decoder::GatedDeltaStateRef;
use crate::transformer::{GatedDeltaNetRequest, PreparedAttentionBlock, PreparedStandardLayer};
use ferrule_backend::cuda::operators::linear::F32ToBf16RowsLayout;
use ferrule_backend::cuda::operators::recurrent::*;
use ferrule_backend::cuda::operators::recurrent_bf16::*;
use ferrule_backend::cuda::operators::standard_bf16::Bf16GateLayout;

impl CudaStandardDecoderOperators {
    pub(super) fn prepare_bf16_gemmex(
        &mut self,
        plans: &[ferrule_backend::cuda::operators::linear::Bf16GemmExPlan],
    ) -> Result<()> {
        if plans.is_empty() {
            return Ok(());
        }
        if self.bf16_gemmex.is_some() {
            return Err(cuda_error("Mixed GemmEx already prepared"));
        }
        if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":0:0") {
            return Err(cuda_error(
                "Mixed GemmEx requires explicit CUBLAS_WORKSPACE_CONFIG=:0:0 before preparation",
            ));
        }
        self.synchronous(|this| {
            let bytes = plans.iter().map(|p| p.storage_grant_bytes()).max().unwrap();
            match this.ops.prepare_bf16_gemmex_views(plans, bytes, plans[0].provider_bytes()) {
                Ok(provider) => this.bf16_gemmex = Some(provider),
                Err(error) => {
                    if ferrule_backend::cuda::operators::linear::CudaBf16GemmEx::preparation_completion_unknown(&this.ops) {
                        this.poisoned = true;
                        this.cache_quarantine();
                    }
                    return Err(error);
                }
            }
            Ok(())
        })
    }

    pub(super) fn prepare_bf16_hybrid_image(
        &mut self,
        embedding: &PreparedEmbedding,
        layers: &[PreparedStandardLayer],
        output: &super::super::PreparedStandardOutput,
        materializer: &crate::transformer::StateDictMaterializer,
    ) -> Result<()> {
        self.synchronous(|this| {
            this.bf16_weight(embedding.linear().parameter())?;
            for layer in layers {
                match layer.attention().block() {
                    PreparedAttentionBlock::Gqa(a) => {
                        this.bf16_weight(a.input_norm.parameter())?;
                        for l in [&a.query, &a.key, &a.value, &a.output] {
                            this.bf16_prepare_linear(l)?;
                        }
                        for n in [&a.query_norm, &a.key_norm].into_iter().flatten() {
                            this.bf16_weight(n.parameter())?;
                        }
                        this.bf16_rope_weight(&a.rope)?;
                    }
                    PreparedAttentionBlock::GatedDeltaNet(a) => {
                        this.bf16_weight(a.input_norm.parameter())?;
                        for l in [&a.qkv, &a.z, &a.beta, &a.a, &a.output] {
                            this.bf16_prepare_linear(l)?;
                        }
                        for p in &a.vector_parameters {
                            this.bf16_weight(p)?;
                        }
                        this.bf16_weight(a.norm.parameter())?;
                        this.bf16_deltas.insert(layer.index(), a.clone());
                    }
                }
                let ff = layer.feed_forward().block();
                this.bf16_weight(ff.norm.parameter())?;
                let shared = match &ff.kind {
                    super::super::PreparedFeedForwardKind::Dense(f) => Some(f),
                    super::super::PreparedFeedForwardKind::Routed {
                        router,
                        shared,
                        shared_gate,
                        descriptor: _,
                        experts,
                    } => {
                        this.bf16_prepare_linear(router)?;
                        if let Some(gate) = shared_gate {
                            this.bf16_prepare_linear(gate)?;
                        }
                        let mut provider = super::super::StateDictExpertProvider {
                            layer: layer.index(),
                            bindings: experts,
                            materializer,
                            reads: None,
                        };
                        for id in 0..if this.bounded_experts.is_some() {
                            0
                        } else {
                            experts.len()
                        } {
                            let ExpertAvailability::Ready(expert) =
                                provider.expert(layer.index(), id)?
                            else {
                                return Err(cuda_error(
                                    "native BF16 preparation requires resident experts",
                                ));
                            };
                            for l in [expert.gate(), expert.up(), expert.down()] {
                                this.bf16_prepare_linear(l)?;
                            }
                            this.experts.insert((layer.index(), id), expert);
                        }
                        shared.as_ref()
                    }
                };
                if let Some(f) = shared {
                    for l in [f.gate(), f.up(), f.down()] {
                        this.bf16_prepare_linear(l)?;
                    }
                }
            }
            this.bf16_weight(output.norm().parameter())?;
            this.bf16_prepare_linear(output.head())?;
            Ok(())
        })
    }

    pub(super) fn bf16_split_query(
        &mut self,
        input: Rows,
        heads: usize,
        head_dim: usize,
    ) -> Result<OperatorProgress<(Rows, Rows)>> {
        self.synchronous(|this| {
            let layout = QueryGateLayout {
                rows: input.shape().rows(),
                heads,
                head_dim,
            };
            if layout.input_elements()? != input.shape().elements() {
                return Err(cuda_error("BF16 interleaved query/gate layout mismatch"));
            }
            let packed = this.ops.bf16_to_f32(this.bf16_buffer(&input)?)?;
            let mut query = this.ops.zero_f32_buffer(layout.output_elements()?)?;
            let mut gate = this.ops.zero_f32_buffer(layout.output_elements()?)?;
            this.ops
                .split_query_gate_f32_into(&packed, &mut query, &mut gate, layout)?;
            let shape = RowsShape::new(layout.rows, heads * head_dim)?;
            Ok(OperatorProgress::Ready((
                round_rows(&this.ops, shape, input.arena(), &query)?,
                round_rows(&this.ops, shape, input.arena(), &gate)?,
            )))
        })
    }

    pub(super) fn bf16_gate(
        &mut self,
        input: Rows,
        gate: &Rows,
        broadcast: bool,
    ) -> Result<OperatorProgress<Rows>> {
        self.measure_rank_phase(
            if broadcast {
                super::rank_timing::Phase::SharedFfn
            } else {
                super::rank_timing::Phase::Gqa
            },
            |this| {
                this.synchronous(|this| {
                    let shape = input.shape();
                    if gate.shape()
                        != RowsShape::new(shape.rows(), if broadcast { 1 } else { shape.width() })?
                    {
                        return Err(cuda_error("BF16 sigmoid gate shape mismatch"));
                    }
                    let mut output = this.ops.zero_bf16_buffer(shape.elements())?;
                    let layout = if broadcast {
                        Bf16GateLayout::RowBroadcast {
                            rows: shape.rows(),
                            width: shape.width(),
                        }
                    } else {
                        Bf16GateLayout::Elementwise {
                            rows: shape.rows(),
                            width: shape.width(),
                        }
                    };
                    this.ops.sigmoid_gate_bf16_into(
                        this.bf16_buffer(&input)?,
                        this.bf16_buffer(gate)?,
                        &mut output,
                        layout,
                    )?;
                    this.trace_bf16(
                        if broadcast {
                            "shared_gate.output"
                        } else {
                            "attention_gate.output"
                        },
                        &output,
                        shape,
                        None,
                        None,
                    )?;
                    rows_bf16(shape, input.arena(), output).map(OperatorProgress::Ready)
                })
            },
        )
    }

    pub(super) fn bf16_router(
        &mut self,
        logits: &Rows,
        policy: &MoeRouterSpec,
    ) -> Result<OperatorProgress<RouterRoutes>> {
        self.measure_rank_phase(super::rank_timing::Phase::Router, |this| {
        this.synchronous(|this| {
            if let Some(unsupported) = standard_router_unsupported(policy) {
                return Ok(OperatorProgress::Unsupported(unsupported));
            }
            if logits.shape().width() != policy.num_experts() {
                return Err(cuda_error("BF16 router width"));
            }
            let layout = SelectedSoftmaxTopKLayout {
                rows: logits.shape().rows(),
                experts: policy.num_experts(),
                top_k: policy.experts_per_token(),
                output_scale: policy.route_scale(),
            };
            let normalization = if this.mixed_gqa_block.is_some() {
                crate::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate
                    .router_normalization()
            } else {
                this.numeric_fp8_precision()
                    .map(|precision| precision.router_normalization())
                    .unwrap_or_default()
            };
            normalization.validate(layout)?;
            let expanded = this.ops.bf16_to_f32(this.bf16_buffer(logits)?)?;
            let mut ids = this.ops.zero_i32_buffer(layout.output_elements()?)?;
            let mut weights = this.ops.zero_f32_buffer(layout.output_elements()?)?;
            let probability_descriptor = if normalization == ferrule_backend::cuda::operators::RouterNormalization::FullSoftmaxThenSelectedRenormalization {
                this.diagnostic_router_descriptor(logits.shape())?
            } else { None };
            let mut probabilities = probability_descriptor.as_ref()
                .map(|_| this.ops.zero_f32_buffer(layout.logit_elements()?)).transpose()?;
            if let Some(buffer) = &probabilities {
                this.expert_hold.push(buffer.as_device_buffer().slice(0, buffer.len())?);
            }
            this.ops.router_softmax_topk_with_probabilities_into(
                &expanded, &mut ids, &mut weights, layout, normalization, probabilities.as_mut(),
            )?;
            if let (Some(descriptor), Some(buffer)) = (&probability_descriptor, &probabilities) {
                let observed = this.trace_router_probabilities(descriptor, buffer);
                // Fence even if a callback declined to download. Only one trace
                // sink may be live; normal synchronous failure custody applies.
                let completed = this.ops.sync_stream();
                if completed.is_err() {
                    this.poisoned = true;
                    this.cache_quarantine();
                }
                match (observed, completed) {
                    (Ok(()), Ok(())) => {},
                    (Err(e), cleanup) => return Err(Error::with_cleanup("router source trace", e, cleanup)),
                    (Ok(()), Err(e)) => return Err(e),
                }
            }
            drop(probabilities);
            let routes = this.measure_rank_phase(super::rank_timing::Phase::RouteHostRead, |this| {
            let ids = this
                .ops
                .download_i32_buffer(&ids)?
                .into_iter()
                .map(|id| {
                    usize::try_from(id)
                        .ok()
                        .filter(|&id| id < policy.num_experts())
                        .ok_or_else(|| cuda_error("invalid BF16 router expert"))
                })
                .collect::<Result<Vec<_>>>()?;
            let routes = RouterRoutes::new(
                layout.rows,
                layout.top_k,
                ids,
                this.ops.download_f32_buffer(&weights)?,
            )?;
                Ok(routes)
            })?;
            this.trace_router_logits(logits, &expanded, &routes)?;
            Ok(OperatorProgress::Ready(routes))
        })
        })
    }

    pub(super) fn bf16_routed(
        &mut self,
        layer: usize,
        input: &Rows,
        routes: &RouterRoutes,
        provider: &mut dyn ExpertProvider,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.synchronous(|this| {
            this.bf16_buffer(input)?;
            if routes.rows() != input.shape().rows()
                || routes.weights().iter().any(|w| !w.is_finite() || *w < 0.0)
            {
                return Err(cuda_error("BF16 routed rows/weights mismatch"));
            }
            let mut buckets = BTreeMap::<usize, Vec<(usize, f32)>>::new();
            for row in 0..routes.rows() {
                let (ids, weights) = routes.row(row)?;
                let mut seen = std::collections::BTreeSet::new();
                for (&id, &weight) in ids.iter().zip(weights) {
                    if !seen.insert(id)
                        || (this.bounded_experts.is_none()
                            && !this.experts.contains_key(&(layer, id)))
                    {
                        return Err(cuda_error("BF16 route requires unique prepared experts"));
                    }
                    buckets.entry(id).or_default().push((row, weight));
                }
            }
            let width = input.shape().width();
            let route_bytes = Self::routed_bucket_scratch_bytes(
                routes.rows(),
                routes.top_k(),
                width,
                routes.rows(),
            )?
            .checked_mul(2)
            .ok_or_else(|| cuda_error("BF16 route scratch overflow"))?;
            let mut metadata = BTreeMap::new();
            if this.bounded_experts.is_some() {
                for (&id, bucket) in &buckets {
                    let bindings = provider
                        .expert_metadata_bindings(layer, id)?
                        .ok_or_else(|| cuda_error("BF16 bounded experts require metadata"))?;
                    let plan = this
                        .bindings
                        .prepared_expert_metadata(layer, id)?
                        .for_bindings(bindings)?;
                    let (_, bytes, scratch) =
                        this.metadata_plan(&plan, layer, id, bucket.len(), width)?;
                    this.preflight_expert_admission(bytes, scratch, route_bytes)?;
                    this.preflight_bucket_numeric(plan.parameters(), bucket.len())?;
                    metadata.insert(id, plan);
                }
                provider.preflight_expert_reads(
                    this.image_generation(),
                    &metadata.values().collect::<Vec<_>>(),
                )?;
                this.cache_scratch(route_bytes)?;
                this.route_scratch_bytes = route_bytes;
            }
            let count = routes.expert_ids().len();
            // A declared, short-lived exact expansion for concatenation only.

            let mut packed = this.ops.zero_f32_buffer(count * width)?;
            this.hold_route_buffer(&packed)?;
            let mut rows = Vec::with_capacity(count);
            let mut weights = Vec::with_capacity(count);
            let mut buckets = buckets.into_iter().peekable();
            while let Some((id, bucket)) = buckets.next() {
                let next = buckets.peek().and_then(|(id, _)| metadata.get(id));
                let ids = bucket
                    .iter()
                    .map(|&(row, _)| row as u32)
                    .collect::<Vec<_>>();
                let gathered = this
                    .ops
                    .embedding_bf16(this.bf16_buffer(input)?, &ids, width)?;
                let gathered = rows_bf16(RowsShape::new(ids.len(), width)?, arena, gathered)?;
                let mut consume = |this: &mut Self, output: Rows| {
                    let expanded = this.ops.bf16_to_f32(this.bf16_buffer(&output)?)?;
                    this.hold_expert_buffer(&expanded)?;
                    this.ops.copy_f32_range(
                        &expanded,
                        0,
                        &mut packed,
                        rows.len() * width,
                        ids.len() * width,
                    )
                };
                if let Some(plan) = metadata.get(&id) {
                    this.metadata_swiglu_consume_lookahead(
                        plan, provider, &gathered, arena, next, consume,
                    )?;
                    // The lease's completion covers the exact expansion/copy too.
                    this.bf16_hold.clear();
                } else {
                    let expert = Arc::clone(&this.experts[&(layer, id)]);
                    let output = super::super::ready(this.bf16_swiglu(&expert, &gathered, arena)?)?;
                    consume(this, output)?;
                }
                for (row, weight) in bucket {
                    rows.push(row as i32);
                    weights.push(weight);
                }
            }
            let packed = this
                .ops
                .f32_rows_to_bf16_rne(&packed, F32ToBf16RowsLayout::packed(1, count, width)?)?;
            this.hold_bf16(&packed)?;
            let ids = this.ops.upload_i32_buffer(&rows)?;
            this.route_ids
                .push(ids.as_device_buffer().slice(0, ids.len())?);
            let weights = this.ops.upload_f32_buffer(&weights)?;
            this.hold_route_buffer(&weights)?;
            let mut output = this.ops.zero_bf16_buffer(input.shape().elements())?;
            this.hold_bf16(&output)?;
            this.ops.weighted_combine_bf16_into(
                &packed,
                &ids,
                &weights,
                &mut output,
                input.shape().rows(),
                width,
            )?;
            this.trace_bf16("routed.output", &output, input.shape(), None, None)?;
            rows_bf16(input.shape(), arena, output).map(OperatorProgress::Ready)
        })
    }

    pub(super) fn bf16_delta(
        &mut self,
        request: GatedDeltaNetRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        self.measure_rank_phase(super::rank_timing::Phase::Gdn, |this| {
            let devices = request
                .states
                .iter()
                .filter_map(|s| match s {
                    GatedDeltaStateRef::CudaBf16(s) => Some(s.device.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let result = this.synchronous(|this| {
                let source = this
                    .bf16_deltas
                    .get(&request.layer)
                    .cloned()
                    .ok_or_else(|| cuda_error("unprepared BF16 GDN"))?;
                let shape = request.shape;
                let channels = shape.sizes()?.0;
                let width = shape.value_heads * shape.value_dim;
                let rows = request.metadata.row_positions().len();
                for (input, width) in [
                    (request.qkv, channels),
                    (request.z, width),
                    (request.a, shape.value_heads),
                    (request.b, shape.value_heads),
                ] {
                    this.bf16_buffer(input)?;
                    if input.shape() != RowsShape::new(rows, width)? {
                        return Err(cuda_error("BF16 GDN projections"));
                    }
                }
                if request.states.len() != request.metadata.sequence_count()
                    || source.conv.as_ref() != request.conv
                    || source.a_log.as_ref() != request.a_log
                    || source.dt_bias.as_ref() != request.dt_bias
                    || request.norm.one_plus_weight()
                    || source.norm.parameter().canonical_id()
                        != request.norm.parameter().canonical_id()
                {
                    return Err(cuda_error("BF16 GDN state/weight identity"));
                }
                let mut cohort = vec![Vec::new(); request.states.len()];
                for (row, &seq) in request.metadata.row_sequence_ids().iter().enumerate() {
                    cohort
                        .get_mut(seq)
                        .ok_or_else(|| cuda_error("BF16 GDN sequence index"))?
                        .push(row as u32);
                }
                for (seq, state) in request.states.iter().enumerate() {
                    let GatedDeltaStateRef::CudaBf16(state) = state else {
                        return Err(cuda_error("BF16 GDN requires typed BF16/F32 state"));
                    };
                    state.validate_owner(&this.ops)?;
                    if state.shape != shape {
                        return Err(cuda_error("BF16 GDN state shape"));
                    }
                    for (offset, &row) in cohort[seq].iter().enumerate() {
                        if state.position.checked_add(offset)
                            != Some(request.metadata.row_positions()[row as usize])
                        {
                            return Err(cuda_error("BF16 GDN frontier"));
                        }
                    }
                    state
                        .position
                        .checked_add(cohort[seq].len())
                        .ok_or_else(|| cuda_error("BF16 GDN position overflow"))?;
                }
                let delta_layout = |rows| GatedDeltaNetLayout {
                    rows,
                    key_heads: shape.key_heads,
                    value_heads: shape.value_heads,
                    key_dim: shape.key_dim,
                    value_dim: shape.value_dim,
                };
                // Validate every cohort against the preallocated image workspace
                // before the first convolution can mutate any working state.
                let chunk_plans = cohort
                    .iter()
                    .zip(request.states.iter())
                    .map(|(ids, state)| {
                        let GatedDeltaStateRef::CudaBf16(state) = state else {
                            return Err(cuda_error("BF16 GDN requires typed BF16/F32 state"));
                        };
                        // HF tests cache history, not the scheduler's phase label.
                        // This frontier is copied/reset with the journal-owned state.
                        this.bf16_chunk_plan(delta_layout(ids.len()), state.position)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let chunked = chunk_plans.iter().any(Option::is_some);
                let conv_weight = this.bf16_weight(&source.vector_parameters[0])?;
                let a_log = this.bf16_weight(&source.vector_parameters[1])?;
                let dt_bias = this.bf16_weight(&source.vector_parameters[2])?;
                let norm = this.bf16_weight(source.norm.parameter())?;
                let mut packed_output = this.ops.zero_f32_buffer(rows * width)?;
                if chunked {
                    this.hold_expert_buffer(&packed_output)?;
                }
                for (seq, ids) in cohort.iter().enumerate() {
                    if ids.is_empty() {
                        continue;
                    }
                    let gather = |input: &Rows| {
                        this.ops.embedding_bf16(
                            this.bf16_buffer(input)?,
                            ids,
                            input.shape().width(),
                        )
                    };
                    let qkv = gather(request.qkv)?;
                    let a = gather(request.a)?;
                    let b = gather(request.b)?;
                    let z = gather(request.z)?;
                    let mut convolved = this.ops.zero_bf16_buffer(ids.len() * channels)?;
                    let mut core = this.ops.zero_bf16_buffer(ids.len() * width)?;
                    let mut output = this.ops.zero_bf16_buffer(ids.len() * width)?;
                    let GatedDeltaStateRef::CudaBf16(state) = &mut request.states[seq] else {
                        unreachable!()
                    };
                    if chunked {
                        for buffer in [&qkv, &a, &b, &z, &convolved, &core, &output, &state.conv] {
                            this.hold_bf16(buffer)?;
                        }
                        this.hold_expert_buffer(&state.recurrent)?;
                    }
                    let _diagnostic_sequence = this.diagnostic.is_enabled().then(|| {
                        this.diagnostic.sequence_scope(CudaDiagnosticSequence {
                            sequence: seq,
                            packed_rows: ids.iter().map(|&r| r as usize).collect(),
                            positions: ids
                                .iter()
                                .map(|&r| request.metadata.row_positions()[r as usize])
                                .collect(),
                            input_frontier: state.position,
                        })
                    });
                    // Opt-in borrowed observations only; no copies when tracing is disabled.
                    for (name, buffer, w) in [
                        ("gdn.input.qkv", &qkv, channels),
                        ("gdn.input.a", &a, shape.value_heads),
                        ("gdn.input.b", &b, shape.value_heads),
                        ("gdn.input.z", &z, width),
                        (
                            "gdn.weight.conv",
                            conv_weight.as_ref(),
                            channels * shape.kernel,
                        ),
                        ("gdn.weight.a_log", a_log.as_ref(), shape.value_heads),
                        ("gdn.weight.dt_bias", dt_bias.as_ref(), shape.value_heads),
                        ("gdn.weight.norm", norm.as_ref(), shape.value_dim),
                        ("gdn.before.conv", &state.conv, state.conv.len()),
                    ] {
                        let binding = match name {
                            "gdn.weight.conv" => Some(source.vector_parameters[0].binding()),
                            "gdn.weight.a_log" => Some(source.vector_parameters[1].binding()),
                            "gdn.weight.dt_bias" => Some(source.vector_parameters[2].binding()),
                            "gdn.weight.norm" => Some(source.norm.parameter().binding()),
                            _ => None,
                        };
                        this.trace_bf16_parameter(
                            name,
                            buffer,
                            RowsShape::new(buffer.len() / w, w)?,
                            binding,
                            None,
                            None,
                        )?;
                    }
                    this.trace_buffer(
                        "gdn.before.ssm",
                        &state.recurrent,
                        RowsShape::new(1, state.recurrent.len())?,
                        None,
                        None,
                        None,
                    )?;
                    this.ops.causal_depthwise_conv1d_silu_bf16_into(
                        Bf16CausalConv1dBuffers {
                            input: &qkv,
                            weight: &conv_weight,
                            bias: None,
                            history: &mut state.conv,
                            output: &mut convolved,
                        },
                        CausalConv1dLayout {
                            rows: ids.len(),
                            channels,
                            kernel_size: shape.kernel,
                        },
                    )?;
                    this.submit_bf16_delta(
                        Bf16GatedDeltaNetBuffers {
                            qkv: &convolved,
                            a: &a,
                            b: &b,
                            a_log: &a_log,
                            dt_bias: &dt_bias,
                            state: &mut state.recurrent,
                            output: &mut core,
                        },
                        delta_layout(ids.len()),
                        chunk_plans[seq],
                    )?;
                    this.ops.gated_rms_norm_bf16_into(
                        &core,
                        &norm,
                        &z,
                        &mut output,
                        F32RowsLayout {
                            rows: ids.len() * shape.value_heads,
                            width: shape.value_dim,
                        },
                        request.norm.epsilon(),
                    )?;
                    for (name, buffer, width) in [
                        ("gdn.convolved", &convolved, channels),
                        ("gdn.recurrent_output", &core, width),
                        ("gdn.gated", &output, width),
                    ] {
                        this.trace_bf16(
                            name,
                            buffer,
                            RowsShape::new(ids.len(), width)?,
                            None,
                            None,
                        )?;
                    }
                    this.trace_bf16(
                        "gdn.conv_state",
                        &state.conv,
                        RowsShape::new(1, state.conv.len())?,
                        None,
                        None,
                    )?;
                    this.trace_buffer(
                        "gdn.recurrent_state",
                        &state.recurrent,
                        RowsShape::new(1, state.recurrent.len())?,
                        None,
                        None,
                        None,
                    )?;
                    let expanded = this.ops.bf16_to_f32(&output)?;
                    if chunked {
                        this.hold_expert_buffer(&expanded)?;
                    }
                    for (local, &row) in ids.iter().enumerate() {
                        this.ops.copy_f32_range(
                            &expanded,
                            local * width,
                            &mut packed_output,
                            row as usize * width,
                            width,
                        )?;
                    }
                }
                round_rows(
                    &this.ops,
                    RowsShape::new(rows, width)?,
                    None,
                    &packed_output,
                )
                .map(OperatorProgress::Ready)
            });
            if this.needs_quarantine() {
                for device in devices {
                    device.quarantine();
                }
            }
            if result.is_ok() {
                for (seq, state) in request.states.iter_mut().enumerate() {
                    if let GatedDeltaStateRef::CudaBf16(state) = state {
                        state.position += request
                            .metadata
                            .row_sequence_ids()
                            .iter()
                            .filter(|&&s| s == seq)
                            .count();
                    }
                }
            }
            result
        })
    }
}

impl CudaStandardDecoderOperators {
    pub(super) fn hold_bf16(
        &mut self,
        buffer: &ferrule_backend::cuda::operators::linear::CudaBf16Buffer,
    ) -> Result<()> {
        self.bf16_hold
            .push(buffer.as_device_buffer().slice(0, buffer.len())?);
        Ok(())
    }

    pub(super) fn bf16_resident_linear(
        &mut self,
        weight: &Rc<binding::ResidentLinear>,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        use ferrule_backend::cuda::operators::linear::Bf16GemmLayout;
        self.bf16_buffer(input)?;
        if weight.bias.is_some() || input.shape().width() != weight.shape.1 {
            return Err(cuda_error("BF16 resident projection shape/bias mismatch"));
        }
        let shape = RowsShape::new(input.shape().rows(), weight.shape.0)?;
        let input_hold = self
            .bf16_buffer(input)?
            .as_device_buffer()
            .slice(0, input.shape().elements())?;
        self.bf16_hold.push(input_hold);
        self.expert_artifacts.push(Rc::clone(weight));
        if let binding::ResidentLinearStorage::Numeric(binding) = &weight.storage {
            binding.source.validate_source_identity()?;
            let precision = self
                .numeric_fp8_precision()
                .ok_or_else(|| cuda_error("missing numeric profile"))?;
            let budget = self.bf16_gemmex.as_ref().map(|p| p.plan());
            if let Some(plan) = precision.gemmex_plan(
                &binding.role,
                binding.artifact.layout(),
                shape.rows(),
                budget.map_or(0, |p| p.storage_grant_bytes()),
                budget.map_or(0, |p| p.provider_bytes()),
            )? {
                let mut output = self.ops.zero_bf16_buffer(shape.elements())?;
                self.hold_bf16(&output)?;
                let input = self.bf16_buffer(input)?;
                self.bf16_gemmex
                    .as_mut()
                    .ok_or_else(|| cuda_error("Mixed GemmEx owner not prepared"))?
                    .execute_view(&self.ops, plan, &binding.artifact, input, &mut output)?;
                return rows_bf16(shape, arena, output);
            }
        }
        if let binding::ResidentLinearStorage::Bf16(resident) = &weight.storage
            && let Some(precision) = self.numeric_fp8_precision()
        {
            let budget = self.bf16_gemmex.as_ref().map(|p| p.plan());
            if let Some(plan) = precision.resident_bf16_gemmex_plan(
                weight.role(),
                weight.shape.0,
                weight.shape.1,
                shape.rows(),
                budget.map_or(0, |p| p.storage_grant_bytes()),
                budget.map_or(0, |p| p.provider_bytes()),
            )? {
                let mut output = self.ops.zero_bf16_buffer(shape.elements())?;
                self.hold_bf16(&output)?;
                let input = self.bf16_buffer(input)?;
                self.bf16_gemmex
                    .as_mut()
                    .ok_or_else(|| cuda_error("resident GemmEx owner not prepared"))?
                    .execute_bf16_view(
                        &self.ops,
                        plan,
                        resident,
                        input,
                        &mut output,
                        Bf16GemmLayout::contiguous(
                            shape.rows(),
                            shape.width(),
                            input.len() / shape.rows(),
                        ),
                    )?;
                return rows_bf16(shape, arena, output);
            }
        }
        let mut output = self.ops.zero_f32_buffer(shape.elements())?;
        self.hold_expert_buffer(&output)?;
        match &weight.storage {
            binding::ResidentLinearStorage::Numeric(binding) => {
                self.numeric_linear_into(binding, input, &mut output)?
            }
            binding::ResidentLinearStorage::Bf16(weight) => self.ops.bf16_linear_into(
                weight,
                self.bf16_buffer(input)?,
                &mut output,
                Bf16GemmLayout::contiguous(shape.rows(), shape.width(), input.shape().width()),
            )?,
            binding::ResidentLinearStorage::Dense(_) => {
                return Err(cuda_error("BF16 image rejects F32 weight binding"));
            }
        }
        let output = self.ops.f32_rows_to_bf16_rne(
            &output,
            F32ToBf16RowsLayout::packed(1, shape.rows(), shape.width())?,
        )?;
        self.hold_bf16(&output)?;
        rows_bf16(shape, arena, output)
    }

    pub(super) fn bf16_metadata_swiglu(
        &mut self,
        metadata: &crate::transformer::ExpertMetadata,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        if metadata.activation_limit().is_some() {
            return Err(cuda_error("clamped BF16 expert unsupported"));
        }
        let linear = |this: &mut Self, index: usize, input: &Rows| {
            let weight = this
                .bindings
                .metadata_linear(&metadata.parameters()[index])?
                .ok_or_else(|| cuda_error("missing BF16 admitted expert binding"))?;
            let parameter = &metadata.parameters()[index];
            if this.diagnostic.is_enabled() {
                this.trace_bf16_parameter(
                    "linear.input",
                    this.bf16_buffer(input)?,
                    input.shape(),
                    Some(parameter),
                    None,
                    None,
                )?;
            }
            let output = this.bf16_resident_linear(&weight, input, arena)?;
            if this.diagnostic.is_enabled() {
                this.trace_bf16_parameter(
                    "linear.output",
                    this.bf16_buffer(&output)?,
                    output.shape(),
                    Some(parameter),
                    None,
                    None,
                )?;
            }
            Ok(output)
        };
        let pair = self.bf16_metadata_gemmex_pair(metadata, input, arena)?;
        let (gate, up) = match pair {
            Some(pair) => pair,
            None => (linear(self, 0, input)?, linear(self, 1, input)?),
        };
        let product = self
            .ops
            .swiglu_bf16(self.bf16_buffer(&gate)?, self.bf16_buffer(&up)?)?;
        self.hold_bf16(&product)?;
        let product = rows_bf16(gate.shape(), arena, product)?;
        linear(self, 2, &product)
    }
    fn bf16_metadata_gemmex_pair(
        &mut self,
        metadata: &crate::transformer::ExpertMetadata,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Option<(Rows, Rows)>> {
        if self.numeric_fp8_precision()
            != Some(crate::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate)
        {
            return Ok(None);
        }
        let parameters = metadata.parameters();
        let weights = [
            self.bindings
                .metadata_linear(&parameters[0])?
                .ok_or_else(|| cuda_error("missing gate binding"))?,
            self.bindings
                .metadata_linear(&parameters[1])?
                .ok_or_else(|| cuda_error("missing up binding"))?,
        ];
        let (
            binding::ResidentLinearStorage::Numeric(gate),
            binding::ResidentLinearStorage::Numeric(up),
        ) = (&weights[0].storage, &weights[1].storage)
        else {
            return Ok(None);
        };
        use crate::support::TensorRole;
        if gate.role != TensorRole::RoutedExpertGate
            || up.role != TensorRole::RoutedExpertUp
            || parameters[0].residency() != parameters[1].residency()
            || weights[0].shape != weights[1].shape
            || weights.iter().any(|w| w.bias.is_some())
        {
            return Err(cuda_error("Mixed GemmEx incompatible expert bindings"));
        }
        gate.source.validate_source_identity()?;
        up.source.validate_source_identity()?;
        let precision = self
            .numeric_fp8_precision()
            .ok_or_else(|| cuda_error("missing numeric profile"))?;
        let budget = self.bf16_gemmex.as_ref().map(|p| p.plan());
        let Some(plan) = precision.gemmex_gate_up_plan(
            gate.artifact.layout(),
            up.artifact.layout(),
            input.shape().rows(),
            budget.map_or(0, |p| p.storage_grant_bytes()),
            budget.map_or(0, |p| p.provider_bytes()),
        )?
        else {
            return Ok(None);
        };
        if input.shape().width() != weights[0].shape.1 {
            return Err(cuda_error("Mixed GemmEx pair input width"));
        }
        let shape = RowsShape::new(input.shape().rows(), weights[0].shape.0)?;
        let mut a = self.ops.zero_bf16_buffer(shape.elements())?;
        let mut b = self.ops.zero_bf16_buffer(shape.elements())?;
        self.hold_bf16(&a)?;
        self.hold_bf16(&b)?;
        let input_buffer = self.bf16_buffer(input)?;
        self.bf16_hold.push(
            input_buffer
                .as_device_buffer()
                .slice(0, input_buffer.len())?,
        );
        self.expert_artifacts.extend(weights.iter().cloned());
        if self.diagnostic.is_enabled() {
            for p in &parameters[..2] {
                self.trace_bf16_parameter(
                    "linear.input",
                    input_buffer,
                    input.shape(),
                    Some(p),
                    None,
                    None,
                )?;
            }
        }
        self.bf16_gemmex
            .as_mut()
            .ok_or_else(|| cuda_error("Mixed GemmEx owner not prepared"))?
            .execute_pair(
                &self.ops,
                plan,
                [&gate.artifact, &up.artifact],
                input_buffer,
                [&mut a, &mut b],
            )?;
        if self.diagnostic.is_enabled() {
            for (buffer, p) in [(&a, &parameters[0]), (&b, &parameters[1])] {
                self.trace_bf16_parameter("linear.output", buffer, shape, Some(p), None, None)?;
            }
        }
        Ok(Some((
            rows_bf16(shape, arena, a)?,
            rows_bf16(shape, arena, b)?,
        )))
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/transformer/bf16_gemmex/gpu.rs"]
mod gemmex_tests;
