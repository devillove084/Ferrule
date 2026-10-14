//! Device-only extensions of the standard operators; all projections stay linear_f32_into.
use super::*;
use crate::decoder::GatedDeltaStateRef;
use crate::transformer::segment_inventory::RecurrentScratchPlan;
use crate::transformer::{GatedDeltaNetRequest, PreparedAttentionBlock, PreparedStandardLayer};
use ferrule_backend::cuda::operators::recurrent::*;

/// Operator-private reusable storage, never sequence state or part of fork/COW.
/// Scratch roots and borrowed direct operands survive errors and unwinding.
#[derive(Default)]
pub(super) struct RecurrentScratch {
    buffers: [Option<CudaF32Buffer>; 8],
    retired: Vec<CudaF32Buffer>,
    // Direct operands must outlive the closure even on error/unwind/Unknown.
    direct_hold: Vec<DeviceBuffer<f32>>,
    #[cfg(test)]
    force_general: bool,
    #[cfg(test)]
    direct_batches: std::cell::Cell<usize>,
    #[cfg(test)]
    copy_calls: std::cell::Cell<usize>,
}
impl RecurrentScratch {
    fn reserve(&mut self, ops: &CudaOperators, plan: RecurrentScratchPlan) -> Result<()> {
        // Reserve once for the largest sequence BEFORE launching any sequence.
        // Earlier calls have passed synchronous's fences; this call's replaced
        // roots remain in custody until that same boundary proves completion.
        if !self.retired.is_empty() {
            return Err(cuda_error("recurrent scratch growth before completion"));
        }
        for (buffer, len) in self.buffers.iter_mut().zip(plan.elements) {
            if buffer.as_ref().is_none_or(|buffer| buffer.len() < len) {
                let replacement = ops.zero_f32_buffer(len)?;
                if let Some(previous) = buffer.replace(replacement) {
                    self.retired.push(previous);
                }
            }
        }
        Ok(())
    }

    fn prefixes(&self, plan: RecurrentScratchPlan) -> Result<[CudaF32Buffer; 8]> {
        let mut lengths = plan.elements.into_iter();
        let mut buffers = self.buffers.iter();
        let mut next = || {
            buffers
                .next()
                .expect("eight scratch buffers")
                .as_ref()
                .ok_or_else(|| cuda_error("unreserved recurrent scratch"))?
                .prefix(lengths.next().expect("eight scratch lengths"))
        };
        Ok([
            next()?,
            next()?,
            next()?,
            next()?,
            next()?,
            next()?,
            next()?,
            next()?,
        ])
    }

    pub(super) fn complete(&mut self) {
        self.retired.clear();
        self.direct_hold.clear();
    }
}

// Row indices, not absolute sequence positions: decoding at a nonzero frontier
// is still direct. Extra/empty states and all partial or reordered packs fall back.
fn direct_batch(per_sequence: &[Vec<usize>], rows: usize) -> bool {
    per_sequence.len() == 1
        && rows != 0
        && per_sequence[0].len() == rows
        && per_sequence[0].iter().copied().eq(0..rows)
}

// A later read (e.g. z) may alias an earlier state/scratch write without
// violating either kernel's local guard. Such views retain gather-before-write
// semantics via the general path. This is read-only admission, not a barrier.
fn direct_inputs_disjoint<'a>(
    inputs: &[&CudaF32Buffer],
    writes: impl Iterator<Item = &'a CudaF32Buffer>,
) -> Result<bool> {
    let range = |buffer: &CudaF32Buffer| -> Result<std::ops::Range<u64>> {
        let start = buffer.as_device_buffer().cu_deviceptr();
        let bytes = buffer
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| cuda_error("direct recurrent range overflow"))?;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| cuda_error("direct recurrent range overflow"))?;
        Ok(start..end)
    };
    for write in writes {
        let write = range(write)?;
        for input in inputs {
            let read = range(input)?;
            if read.start < write.end && write.start < read.end {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

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
        materializer: &crate::transformer::StateDictMaterializer,
    ) -> Result<()> {
        if self.bf16.is_some() {
            return self.prepare_bf16_hybrid_image(embedding, layers, output, materializer);
        }
        self.prepare_hybrid_segment_image(Some(embedding), layers, Some(output))
    }
    pub(in crate::transformer) fn prepare_hybrid_segment_image(
        &mut self,
        embedding: Option<&PreparedEmbedding>,
        layers: &[PreparedStandardLayer],
        output: Option<&super::super::PreparedStandardOutput>,
    ) -> Result<()> {
        self.synchronous(|this| {
            if this.bindings.tensor.is_some() {
                return Err(unsupported("hybrid TP unsupported"));
            }
            if let Some(embedding) = embedding {
                this.bindings
                    .vector(&this.ops, embedding.linear().parameter())?;
            }
            for layer in layers {
                match layer.attention().block() {
                    PreparedAttentionBlock::Gqa(a) => {
                        this.root_norm_binding(&a.input_norm)?;
                        for l in [&a.query, &a.key, &a.value, &a.output] {
                            this.prepare_rank_or_standard_linear(l)?;
                        }
                        for n in [&a.query_norm, &a.key_norm].into_iter().flatten() {
                            this.root_norm_binding(n)?;
                        }
                        this.root_rope_binding(&a.rope)?;
                    }
                    PreparedAttentionBlock::GatedDeltaNet(a) => {
                        this.root_norm_binding(&a.input_norm)?;
                        for l in [&a.qkv, &a.z, &a.beta, &a.a, &a.output] {
                            this.prepare_rank_or_standard_linear(l)?;
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
                this.root_norm_binding(&ff.norm)?;
                let dense = match &ff.kind {
                    super::super::PreparedFeedForwardKind::Dense(ff) => Some(ff),
                    super::super::PreparedFeedForwardKind::Routed {
                        router,
                        shared,
                        shared_gate,
                        ..
                    } if this.numeric.is_some() => {
                        this.prepare_rank_or_standard_linear(router)?;
                        if let Some(gate) = shared_gate {
                            this.prepare_rank_or_standard_linear(gate)?;
                        }
                        // Routed weights remain lazy and enter only through the
                        // bounded expert lease in the existing routed operator.
                        shared.as_ref()
                    }
                    _ => {
                        return Err(unsupported(
                            "hybrid CUDA routed FFN requires numeric profile",
                        ));
                    }
                };
                if let Some(ff) = dense {
                    for l in [ff.gate(), ff.up(), ff.down()] {
                        this.prepare_rank_or_standard_linear(l)?;
                    }
                }
            }
            if let Some(output) = output {
                this.root_norm_binding(output.norm())?;
                this.prepare_rank_or_standard_linear(output.head())?;
            }
            Ok(())
        })
    }
    pub(super) fn split_query_rows(
        &mut self,
        input: Rows,
        heads: usize,
        head_dim: usize,
    ) -> Result<OperatorProgress<(Rows, Rows)>> {
        let bytes = root_execution::operand_bytes(input.shape().elements(), 2)?;
        self.root_operation(bytes, 3, |this| {
            this.root_pin_rows(&input)?;
            if this.bindings.tensor.is_some() {
                return Err(unsupported("hybrid gated attention TP unsupported"));
            }
            let layout = QueryGateLayout {
                rows: input.shape().rows(),
                heads,
                head_dim,
            };
            if layout.output_elements()?.checked_mul(2) != Some(input.shape().elements()) {
                return Err(cuda_error("gated query split input shape mismatch"));
            }
            let mut q = this.ops.zero_f32_buffer(layout.output_elements()?)?;
            this.root_pin(&q)?;
            let mut gate = this.ops.zero_f32_buffer(layout.output_elements()?)?;
            this.root_pin(&gate)?;
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
        let bytes = root_execution::operand_bytes(input.shape().elements(), 3)?;
        self.root_operation(bytes, 3, |this| {
            this.root_pin_rows(&input)?;
            this.root_pin_rows(gate)?;
            if input.shape() != gate.shape() {
                return Err(cuda_error("CUDA sigmoid gate shape mismatch"));
            }
            let mut output = this.ops.zero_f32_buffer(input.shape().elements())?;
            this.root_pin(&output)?;
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
        self.root_barrier(BarrierReason::RecurrentState)?;
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
            let largest_sequence = per_sequence.iter().map(Vec::len).max().unwrap_or(0);
            this.recurrent_scratch.reserve(
                &this.ops,
                RecurrentScratchPlan::new(largest_sequence, channels, shape.value_heads, width)?,
            )?;
            let mut output = this.ops.zero_f32_buffer(rows * width)?;
            let direct = if direct_batch(&per_sequence, rows) {
                let GatedDeltaStateRef::Cuda(state) = &request.states[0] else {
                    unreachable!()
                };
                direct_inputs_disjoint(
                    &[
                        this.f32(request.qkv)?,
                        this.f32(request.a)?,
                        this.f32(request.b)?,
                        this.f32(request.z)?,
                    ],
                    [&state.conv, &state.recurrent, &output]
                        .into_iter()
                        .chain(this.recurrent_scratch.buffers.iter().flatten()),
                )?
            } else {
                false
            };
            #[cfg(test)]
            let direct = direct && !this.recurrent_scratch.force_general;
            if direct {
                let holds = [
                    this.f32(request.qkv)?,
                    this.f32(request.a)?,
                    this.f32(request.b)?,
                    this.f32(request.z)?,
                    &output,
                ]
                .map(|buffer| buffer.as_device_buffer().slice(0, buffer.len()));
                for hold in holds {
                    this.recurrent_scratch.direct_hold.push(hold?);
                }
            }
            for (sequence, indices) in per_sequence.iter().enumerate() {
                if indices.is_empty() {
                    continue;
                }
                let GatedDeltaStateRef::Cuda(state) = &mut request.states[sequence] else {
                    unreachable!()
                };
                let [
                    mut qkv,
                    mut a,
                    mut b,
                    mut z,
                    mut convolved,
                    mut recurrent_output,
                    mut normalized,
                    mut gated,
                ] = this.recurrent_scratch.prefixes(RecurrentScratchPlan::new(
                    indices.len(),
                    channels,
                    shape.value_heads,
                    width,
                )?)?;
                let copy_rows =
                    |input: &Rows, width: usize, packed: &mut CudaF32Buffer| -> Result<()> {
                        for (local, &row) in indices.iter().enumerate() {
                            this.ops.copy_f32_range(
                                this.f32(input)?,
                                row * width,
                                packed,
                                local * width,
                                width,
                            )?;
                            #[cfg(test)]
                            this.recurrent_scratch
                                .copy_calls
                                .set(this.recurrent_scratch.copy_calls.get() + 1);
                        }
                        Ok(())
                    };
                if direct {
                    // Owner-checked exact prefixes retain their roots. All writes
                    // still go through the backend's existing overlap guards.
                    qkv = this.f32(request.qkv)?.prefix(indices.len() * channels)?;
                    a = this
                        .f32(request.a)?
                        .prefix(indices.len() * shape.value_heads)?;
                    b = this
                        .f32(request.b)?
                        .prefix(indices.len() * shape.value_heads)?;
                    z = this.f32(request.z)?.prefix(indices.len() * width)?;
                    gated = output.prefix(indices.len() * width)?;
                    #[cfg(test)]
                    this.recurrent_scratch
                        .direct_batches
                        .set(this.recurrent_scratch.direct_batches.get() + 1);
                } else {
                    copy_rows(request.qkv, channels, &mut qkv)?;
                    copy_rows(request.a, shape.value_heads, &mut a)?;
                    copy_rows(request.b, shape.value_heads, &mut b)?;
                    copy_rows(request.z, width, &mut z)?;
                }
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
                for (name, buffer, n, w) in [
                    ("gdn.convolved", &convolved, indices.len(), channels),
                    (
                        "gdn.recurrent_output",
                        &recurrent_output,
                        indices.len(),
                        width,
                    ),
                    ("gdn.conv_state", &state.conv, 1, state.conv.len()),
                    (
                        "gdn.recurrent_state",
                        &state.recurrent,
                        1,
                        state.recurrent.len(),
                    ),
                ] {
                    this.trace_buffer(name, buffer, RowsShape::new(n, w)?, None, None, None)?;
                }
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
                this.trace_buffer(
                    "gdn.normalized",
                    &normalized,
                    RowsShape::new(indices.len(), width)?,
                    None,
                    None,
                    None,
                )?;
                this.trace_buffer(
                    "gdn.gated",
                    &gated,
                    RowsShape::new(indices.len(), width)?,
                    None,
                    None,
                    None,
                )?;
                if !direct {
                    for (local, &row) in indices.iter().enumerate() {
                        this.ops.copy_f32_range(
                            &gated,
                            local * width,
                            &mut output,
                            row * width,
                            width,
                        )?;
                        #[cfg(test)]
                        this.recurrent_scratch
                            .copy_calls
                            .set(this.recurrent_scratch.copy_calls.get() + 1);
                    }
                }
                // Scatter precedes the next sequence's writes on this owner's
                // compute stream. Every logical prefix is fully overwritten;
                // spare capacity is never exposed to a kernel or returned Rows.
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

#[cfg(test)]
#[path = "../../../../tests/unit/transformer/recurrent_cuda.rs"]
mod scratch_tests;
