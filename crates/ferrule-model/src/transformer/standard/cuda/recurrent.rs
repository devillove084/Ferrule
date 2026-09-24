//! Device-only extensions of the standard operators; all projections stay linear_f32_into.
use super::*;
use crate::decoder::GatedDeltaStateRef;
use crate::transformer::{GatedDeltaNetRequest, PreparedAttentionBlock, PreparedStandardLayer};
use ferrule_backend::cuda::operators::recurrent::*;

pub(super) struct ResidentDelta {
    conv: Rc<CudaF32Buffer>,
    a_log: Rc<CudaF32Buffer>,
    dt_bias: Rc<CudaF32Buffer>,
    norm: Rc<CudaF32Buffer>,
    source: crate::transformer::PreparedGatedDeltaNetBlock,
}
impl CudaStandardDecoderOperators {
    pub(super) fn prepare_hybrid_image(
        &mut self,
        embedding: &PreparedEmbedding,
        layers: &[PreparedStandardLayer],
        output: &super::super::PreparedStandardOutput,
    ) -> Result<()> {
        self.synchronous(|this| {
            if this.bindings.tensor.is_some() {
                return Err(unsupported("hybrid TP/PP unsupported"));
            }
            this.bindings
                .vector(&this.ops, embedding.linear().parameter())?;
            for layer in layers {
                match layer.attention().block() {
                    PreparedAttentionBlock::Gqa(a) => {
                        this.bindings.norm(&this.ops, &a.input_norm)?;
                        for l in [&a.query, &a.key, &a.value, &a.output] {
                            this.bindings.linear(&this.ops, l)?;
                        }
                        for n in [&a.query_norm, &a.key_norm].into_iter().flatten() {
                            this.bindings.norm(&this.ops, n)?;
                        }
                        this.bindings.rope(&this.ops, &a.rope)?;
                    }
                    PreparedAttentionBlock::GatedDeltaNet(a) => {
                        this.bindings.norm(&this.ops, &a.input_norm)?;
                        for l in [&a.qkv, &a.z, &a.beta, &a.a, &a.output] {
                            this.bindings.linear(&this.ops, l)?;
                        }
                        let conv = this.bindings.vector(&this.ops, &a.vector_parameters[0])?;
                        let a_log = this.bindings.vector(&this.ops, &a.vector_parameters[1])?;
                        let dt_bias = this.bindings.vector(&this.ops, &a.vector_parameters[2])?;
                        let norm = this.bindings.norm(&this.ops, &a.norm)?;
                        this.deltas.insert(
                            layer.index(),
                            ResidentDelta {
                                conv,
                                a_log,
                                dt_bias,
                                norm,
                                source: a.clone(),
                            },
                        );
                    }
                }
                let ff = layer.feed_forward().block();
                this.bindings.norm(&this.ops, &ff.norm)?;
                let super::super::PreparedFeedForwardKind::Dense(ff) = &ff.kind else {
                    return Err(unsupported("hybrid CUDA dense FFN only"));
                };
                for l in [ff.gate(), ff.up(), ff.down()] {
                    this.bindings.linear(&this.ops, l)?;
                }
            }
            this.bindings.norm(&this.ops, output.norm())?;
            this.bindings.linear(&this.ops, output.head())?;
            Ok(())
        })
    }
    pub(super) fn split_query_rows(
        &mut self,
        input: Rows,
        heads: usize,
        head_dim: usize,
    ) -> Result<OperatorProgress<(Rows, Rows)>> {
        self.synchronous(|this| {
            if this.bindings.tensor.is_some() {
                return Err(unsupported("hybrid gated attention TP unsupported"));
            }
            let layout = QueryGateLayout {
                rows: input.shape().rows(),
                heads,
                head_dim,
            };
            let mut q = this.ops.zero_f32_buffer(layout.output_elements()?)?;
            let mut gate = this.ops.zero_f32_buffer(layout.output_elements()?)?;
            this.ops
                .split_query_gate_f32_into(this.f32(&input)?, &mut q, &mut gate, layout)?;
            let shape = RowsShape::new(layout.rows, heads * head_dim)?;
            Ok(OperatorProgress::Ready((
                device_rows(shape, input.arena(), q)?,
                device_rows(shape, input.arena(), gate)?,
            )))
        })
    }
    pub(super) fn gate_rows(&mut self, input: Rows, gate: &Rows) -> Result<OperatorProgress<Rows>> {
        self.synchronous(|this| {
            if input.shape() != gate.shape() {
                return Err(cuda_error("CUDA sigmoid gate shape mismatch"));
            }
            let mut output = this.ops.zero_f32_buffer(input.shape().elements())?;
            this.ops.elementwise_gate_f32_into(
                this.f32(&input)?,
                this.f32(gate)?,
                &mut output,
                F32RowsLayout {
                    rows: input.shape().rows(),
                    width: input.shape().width(),
                },
                GateActivation::Sigmoid,
            )?;
            device_rows(input.shape(), input.arena(), output).map(OperatorProgress::Ready)
        })
    }
    pub(super) fn delta_rows(
        &mut self,
        request: GatedDeltaNetRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        // Preserve owner handles outside the launch closure: on an unknown fence every
        // involved sequence is quarantined before the transaction can attempt rollback.
        let devices = request
            .states
            .iter()
            .filter_map(|s| match s {
                GatedDeltaStateRef::Cuda(s) => Some(s.device.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let result = self.synchronous(|this| {
            if this.bindings.tensor.is_some() {
                return Err(unsupported("hybrid CUDA TP unsupported"));
            }
            let resident = this
                .deltas
                .get(&request.layer)
                .ok_or_else(|| cuda_error("unprepared GatedDeltaNet layer"))?;
            let shape = request.shape;
            let channels = shape.sizes()?.0;
            let width = shape.value_heads * shape.value_dim;
            let rows = request.metadata.row_positions().len();
            for (input, expected) in [
                (request.qkv, channels),
                (request.z, width),
                (request.a, shape.value_heads),
                (request.b, shape.value_heads),
            ] {
                this.f32(input)?;
                if input.shape() != RowsShape::new(rows, expected)? {
                    return Err(cuda_error("CUDA GatedDeltaNet packed projection mismatch"));
                }
            }
            if request.states.len() != request.metadata.sequence_count()
                || resident.source.conv.as_ref() != request.conv
                || resident.source.a_log.as_ref() != request.a_log
                || resident.source.dt_bias.as_ref() != request.dt_bias
                || resident.source.norm.parameter().canonical_id()
                    != request.norm.parameter().canonical_id()
                || request.norm.one_plus_weight()
            {
                return Err(cuda_error(
                    "CUDA GatedDeltaNet state/weight identity mismatch",
                ));
            }
            let mut per_sequence = vec![Vec::new(); request.states.len()];
            for (row, &seq) in request.metadata.row_sequence_ids().iter().enumerate() {
                per_sequence[seq].push(row);
            }
            // Validate the whole cohort before any state mutation.
            for (index, state) in request.states.iter().enumerate() {
                let GatedDeltaStateRef::Cuda(state) = state else {
                    return Err(unsupported("CUDA GatedDeltaNet cannot use CPU state"));
                };
                state.validate_owner(&this.ops)?;
                if state.shape != shape {
                    return Err(cuda_error("CUDA recurrent state shape mismatch"));
                }
                for (offset, &row) in per_sequence[index].iter().enumerate() {
                    if state.position.checked_add(offset)
                        != Some(request.metadata.row_positions()[row])
                    {
                        return Err(cuda_error("CUDA recurrent frontier mismatch"));
                    }
                }
                state
                    .position
                    .checked_add(per_sequence[index].len())
                    .ok_or_else(|| cuda_error("CUDA recurrent position overflow"))?;
            }
            let mut output = this.ops.zero_f32_buffer(rows * width)?;
            for (sequence, indices) in per_sequence.iter().enumerate() {
                if indices.is_empty() {
                    continue;
                }
                let GatedDeltaStateRef::Cuda(state) = &mut request.states[sequence] else {
                    unreachable!()
                };
                let copy_rows = |input: &Rows, width: usize| -> Result<CudaF32Buffer> {
                    let mut packed = this.ops.zero_f32_buffer(indices.len() * width)?;
                    for (local, &row) in indices.iter().enumerate() {
                        this.ops.copy_f32_range(
                            this.f32(input)?,
                            row * width,
                            &mut packed,
                            local * width,
                            width,
                        )?;
                    }
                    Ok(packed)
                };
                let qkv = copy_rows(request.qkv, channels)?;
                let a = copy_rows(request.a, shape.value_heads)?;
                let b = copy_rows(request.b, shape.value_heads)?;
                let z = copy_rows(request.z, width)?;
                let mut convolved = this.ops.zero_f32_buffer(indices.len() * channels)?;
                let mut recurrent_output = this.ops.zero_f32_buffer(indices.len() * width)?;
                let mut normalized = this.ops.zero_f32_buffer(indices.len() * width)?;
                let mut gated = this.ops.zero_f32_buffer(indices.len() * width)?;
                this.ops.causal_depthwise_conv1d_silu_into(
                    CausalConv1dBuffers {
                        input: &qkv,
                        weight: &resident.conv,
                        bias: None,
                        history: &mut state.conv,
                        output: &mut convolved,
                    },
                    CausalConv1dLayout {
                        rows: indices.len(),
                        channels,
                        kernel_size: shape.kernel,
                    },
                )?;
                this.ops.gated_delta_net_into(
                    GatedDeltaNetBuffers {
                        qkv: &convolved,
                        a: &a,
                        b: &b,
                        a_log: &resident.a_log,
                        dt_bias: &resident.dt_bias,
                        state: &mut state.recurrent,
                        output: &mut recurrent_output,
                    },
                    GatedDeltaNetLayout {
                        rows: indices.len(),
                        key_heads: shape.key_heads,
                        value_heads: shape.value_heads,
                        key_dim: shape.key_dim,
                        value_dim: shape.value_dim,
                    },
                )?;
                this.ops.rms_norm_f32_into(
                    &recurrent_output,
                    indices.len() * shape.value_heads,
                    &resident.norm,
                    request.norm.epsilon(),
                    &mut normalized,
                )?;
                this.ops.elementwise_gate_f32_into(
                    &normalized,
                    &z,
                    &mut gated,
                    F32RowsLayout {
                        rows: indices.len(),
                        width,
                    },
                    GateActivation::Silu,
                )?;
                for (local, &row) in indices.iter().enumerate() {
                    this.ops.copy_f32_range(
                        &gated,
                        local * width,
                        &mut output,
                        row * width,
                        width,
                    )?;
                }
                // Buffers retire through the backend's stream-fenced allocator.
            }
            device_rows(RowsShape::new(rows, width)?, None, output).map(OperatorProgress::Ready)
        });
        if self.needs_quarantine() {
            for device in &devices {
                device.quarantine();
            }
        }
        if result.is_ok() {
            // Positions become observable only after the operator's completion evidence.
            for (index, state) in request.states.iter_mut().enumerate() {
                if let GatedDeltaStateRef::Cuda(state) = state {
                    state.position += request
                        .metadata
                        .row_sequence_ids()
                        .iter()
                        .filter(|&&s| s == index)
                        .count();
                }
            }
        }
        result
    }
}
fn unsupported(message: &str) -> Error {
    Error::ModelSource {
        source: Box::new(UnsupportedOperator::new("hybrid_cuda", message)),
    }
}
