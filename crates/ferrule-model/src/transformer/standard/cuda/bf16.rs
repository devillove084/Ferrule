//! Real BF16 storage through the ordinary standard layer loop and KV journal.
use super::*;
use crate::execution::bf16_rne_word;
use crate::nn::ParameterId;
use crate::transformer::PreparedParameter;
use ferrule_backend::cuda::operators::linear::{
    Bf16GemmLayout, CudaBf16Buffer, F32ToBf16RowsLayout, bf16_gemm_workspace_requirements,
};

#[derive(Default)]
pub(super) struct State {
    weights: BTreeMap<ParameterId, Rc<CudaBf16Buffer>>,
    ropes: Vec<(PreparedRope, Rc<CudaBf16Buffer>, Rc<CudaBf16Buffer>)>,
    pub bytes: usize,
}
impl CudaStandardDecoderOperators {
    pub(super) fn trace_bf16(
        &self,
        name: &'static str,
        buffer: &CudaBf16Buffer,
        shape: RowsShape,
        linear: Option<&PreparedLinear>,
        norm: Option<&PreparedNorm>,
    ) -> Result<()> {
        self.trace_bf16_parameter(name, buffer, shape,
            linear.map(|p| p.parameter().binding()).or_else(|| norm.map(|p| p.parameter().binding())), linear, norm)
    }
    pub(super) fn bf16_buffer<'a>(&self, rows: &'a Rows) -> Result<&'a CudaBf16Buffer> {
        let rows = rows.cuda()?;
        rows.validate_owner(&self.ops)?;
        rows.bf16_buffer()
            .ok_or_else(|| cuda_error("BF16 profile rejects non-BF16 storage"))
    }
    pub(super) fn bf16_weight(
        &mut self,
        parameter: &PreparedParameter,
    ) -> Result<Rc<CudaBf16Buffer>> {
        self.bindings.validate(parameter)?;
        let mixed_rank = self.mixed_rank_block.is_some();
        let f32_gdn = (self.numeric.is_some() || mixed_rank)
            && parameter.binding().weight().slice().dtype
                == crate::checkpoint::CheckpointDType::F32
            && matches!(
                parameter.role(),
                crate::support::TensorRole::LinearAttentionALog
                    | crate::support::TensorRole::LinearAttentionNorm
            );
        if (parameter.binding().weight().slice().dtype != crate::checkpoint::CheckpointDType::Bf16
            && !f32_gdn)
            || parameter.scale().is_some()
        {
            return Err(cuda_error(
                "BF16 profile requires native unscaled BF16 weights",
            ));
        }
        let state = self
            .bf16
            .as_mut()
            .ok_or_else(|| cuda_error("BF16 profile not selected"))?;
        let id = parameter.canonical_id();
        if let Some(weight) = state.weights.get(&id) {
            return Ok(Rc::clone(weight));
        }
        let weight = Rc::new(if mixed_rank {
            self.ops
                .upload_bf16_words(&parameter.mixed_vector_bf16_words()?)?
        } else if f32_gdn && parameter.hybrid_rank().is_none() {
            let values = parameter.values_f32()?;
            let source = self.ops.upload_f32_buffer(&values)?;
            self.expert_hold
                .push(source.as_device_buffer().slice(0, source.len())?);
            self.ops
                .f32_rows_to_bf16_rne(&source, F32ToBf16RowsLayout::packed(1, 1, values.len())?)?
        } else {
            self.ops
                .upload_bf16_words(&parameter.values_bf16_words()?)?
        });
        state.bytes += weight.len() * 2;
        state.weights.insert(id, Rc::clone(&weight));
        Ok(weight)
    }
    pub(super) fn bf16_linear_weight(
        &mut self,
        linear: &PreparedLinear,
    ) -> Result<Rc<CudaBf16Buffer>> {
        if linear.bias().is_some()
            || linear.tensor_shard().is_some()
            || linear.numeric_fp8().is_some()
            || linear.weight()?.execution.activation_quantization.is_some()
        {
            return Err(cuda_error(
                "BF16 dense profile rejects bias, TP, numeric and activation quantization",
            ));
        }
        bf16_gemm_workspace_requirements(Bf16GemmLayout::contiguous(
            1,
            linear.out_features(),
            linear.in_features(),
        ))?;
        self.bf16_weight(linear.parameter())
    }
    pub(super) fn bf16_prepare_linear(&mut self, linear: &PreparedLinear) -> Result<()> {
        if let Some(state) = &self.numeric {
            let weight = self.bindings.linear(&self.ops, linear)?;
            if let super::binding::ResidentLinearStorage::Numeric(binding) = &weight.storage {
                if state
                    .precision
                    .gemmex_plan(
                        &binding.role,
                        binding.artifact.layout(),
                        1,
                        usize::MAX,
                        usize::MAX,
                    )?
                    .is_some()
                {
                    return Ok(());
                }
                let plan = state.precision.linear_plan(
                    &binding.role,
                    binding.artifact.layout(),
                    1,
                    state.stats.reserved_bytes,
                )?;
                self.ops
                    .prepare_numeric_fp8_provider(plan, plan.provider_bytes())?;
            }
        } else {
            self.bf16_linear_weight(linear)?;
        }
        Ok(())
    }
    pub(super) fn bf16_rope_weight(
        &mut self,
        table: &PreparedRope,
    ) -> Result<(Rc<CudaBf16Buffer>, Rc<CudaBf16Buffer>)> {
        let state = self
            .bf16
            .as_mut()
            .ok_or_else(|| cuda_error("BF16 profile not selected"))?;
        if let Some((_, c, s)) = state.ropes.iter().find(|(t, _, _)| t == table) {
            return Ok((Rc::clone(c), Rc::clone(s)));
        }
        let upload = |v: &[f32]| {
            self.ops
                .upload_bf16_words(&v.iter().copied().map(bf16_rne_word).collect::<Vec<_>>())
                .map(Rc::new)
        };
        let c = upload(table.cosine())?;
        if self.mixed_rank_block.is_some() {
            // Keep the first upload alive if the second upload/fence fails.
            self.bf16_hold.push(c.as_device_buffer().slice(0, c.len())?);
        }
        let s = upload(table.sine())?;
        state.bytes += (c.len() + s.len()) * 2;
        state
            .ropes
            .push((table.clone(), Rc::clone(&c), Rc::clone(&s)));
        Ok((c, s))
    }
    pub(super) fn prepare_bf16_image(
        &mut self,
        embedding: Option<&PreparedEmbedding>,
        layers: &[super::super::PreparedGqaMoeLayer],
        output: Option<&super::super::PreparedCpuOutput>,
    ) -> Result<()> {
        self.synchronous(|this| {
            if let Some(e) = embedding {
                this.bf16_weight(e.linear().parameter())?;
            }
            for layer in layers {
                let a = layer.attention().block();
                for n in [
                    Some(&a.input_norm),
                    a.query_norm.as_ref(),
                    a.key_norm.as_ref(),
                    Some(&layer.feed_forward().block().norm),
                ]
                .into_iter()
                .flatten()
                {
                    if n.one_plus_weight() {
                        return Err(cuda_error("BF16 offset RMSNorm is not admitted"));
                    }
                    this.bf16_weight(n.parameter())?;
                }
                for l in [&a.query, &a.key, &a.value, &a.output] {
                    this.bf16_linear_weight(l)?;
                }
                this.bf16_rope_weight(&a.rope)?;
                let super::super::PreparedFeedForwardKind::Dense(f) =
                    &layer.feed_forward().block().kind
                else {
                    return Err(cuda_error("BF16 profile does not admit MoE"));
                };
                for l in [f.gate(), f.up(), f.down()] {
                    this.bf16_linear_weight(l)?;
                }
            }
            if let Some(o) = output {
                this.bf16_weight(o.norm().parameter())?;
                this.bf16_linear_weight(o.head())?;
            }
            Ok(())
        })
    }
    pub(super) fn bf16_bind(&mut self, rows: Rows) -> Result<Rows> {
        self.synchronous(|this| match rows {
            Rows::Cuda(rows) => {
                rows.validate_owner(&this.ops)?;
                if rows.bf16_buffer().is_none() {
                    return Err(cuda_error("BF16 input storage required"));
                }
                Ok(Rows::Cuda(rows))
            }
            Rows::Host(rows) => {
                if rows.dtype() != RowsDType::Bf16 {
                    return Err(cuda_error("BF16 host boundary required"));
                }
                let buffer = this.boundary.upload_bf16(&this.ops, rows.values())?;
                rows_bf16(rows.shape(), rows.arena(), buffer)
            }
        })
    }
    pub(super) fn bf16_download(&mut self, rows: Rows) -> Result<HostRows> {
        self.synchronous(|this| {
            let buffer = rows.cuda()?;
            buffer.validate_owner(&this.ops)?;
            let buffer = buffer
                .bf16_buffer()
                .ok_or_else(|| cuda_error("BF16 export requires BF16 storage"))?;
            let values = this.boundary.download_bf16(&this.ops, buffer)?;
            HostRows::new(rows.shape(), RowsDType::Bf16, rows.arena(), values)
        })
    }
    pub(super) fn bf16_embedding(
        &mut self,
        e: &PreparedEmbedding,
        ids: &[u32],
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.synchronous(|this| {
            let weight = this.bf16_weight(e.linear().parameter())?;
            let output = this.ops.embedding_bf16(&weight, ids, e.width())?;
            this.trace_bf16(
                "embedding.output",
                &output,
                RowsShape::new(ids.len(), e.width())?,
                Some(e.linear()),
                None,
            )?;
            rows_bf16(RowsShape::new(ids.len(), e.width())?, arena, output)
                .map(OperatorProgress::Ready)
        })
    }
    pub(super) fn bf16_linear_rows(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        if input.shape().width() != linear.in_features() {
            return Err(cuda_error("BF16 linear width mismatch"));
        }
        self.trace_bf16(
            "linear.input",
            self.bf16_buffer(input)?,
            input.shape(),
            Some(linear),
            None,
        )?;
        if self.numeric.is_some() {
            if self.bindings.expert_admission.is_none() {
                let bytes = input
                    .shape()
                    .width()
                    .checked_add(linear.out_features())
                    .and_then(|n| n.checked_mul(input.shape().rows()))
                    .and_then(|n| n.checked_mul(8))
                    .ok_or_else(|| cuda_error("BF16 linear scratch overflow"))?;
                self.reserve_numeric_temporaries(bytes)?;
            }
            let weight = self.bindings.linear(&self.ops, linear)?;
            let rows = self.bf16_resident_linear(&weight, input, arena)?;
            self.trace_bf16(
                "linear.output",
                self.bf16_buffer(&rows)?,
                rows.shape(),
                Some(linear),
                None,
            )?;
            return Ok(rows);
        }
        let weight = self.bf16_linear_weight(linear)?;
        let shape = RowsShape::new(input.shape().rows(), linear.out_features())?;
        let mut output = self.ops.zero_f32_buffer(shape.elements())?;
        self.ops.bf16_linear_into(
            &weight,
            self.bf16_buffer(input)?,
            &mut output,
            Bf16GemmLayout::contiguous(shape.rows(), shape.width(), input.shape().width()),
        )?;
        let rows = round_rows(&self.ops, shape, arena, &output)?;
        self.trace_bf16(
            "linear.output",
            self.bf16_buffer(&rows)?,
            shape,
            Some(linear),
            None,
        )?;
        Ok(rows)
    }
    pub(super) fn bf16_linear(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.synchronous(|this| {
            this.bf16_linear_rows(linear, input, arena)
                .map(OperatorProgress::Ready)
        })
    }
    pub(super) fn bf16_norm(
        &mut self,
        norm: &PreparedNorm,
        input: &Rows,
        heads: usize,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.synchronous(|this| {
            if heads == 0 || norm.weight().len().checked_mul(heads) != Some(input.shape().width()) {
                return Err(cuda_error("BF16 norm layout/profile mismatch"));
            }
            this.trace_bf16(
                "norm.input",
                this.bf16_buffer(input)?,
                input.shape(),
                None,
                Some(norm),
            )?;
            let weight = this.bf16_weight(norm.parameter())?;
            let output = if norm.one_plus_weight() {
                // Qwen3.5 keeps the offset affine in F32 until the sole BF16 cast.
                let mut output = this.ops.zero_bf16_buffer(input.shape().elements())?;
                this.ops.offset_rms_norm_bf16_into(
                    this.bf16_buffer(input)?,
                    &weight,
                    &mut output,
                    ferrule_backend::cuda::operators::recurrent::F32RowsLayout {
                        rows: input.shape().rows() * heads,
                        width: norm.weight().len(),
                    },
                    norm.epsilon(),
                )?;
                output
            } else {
                this.ops.rms_norm_bf16(
                    this.bf16_buffer(input)?,
                    &weight,
                    input.shape().rows() * heads,
                    norm.epsilon(),
                )?
            };
            this.trace_bf16("norm.output", &output, input.shape(), None, Some(norm))?;
            rows_bf16(input.shape(), arena, output).map(OperatorProgress::Ready)
        })
    }
    pub(super) fn bf16_rope(
        &mut self,
        descriptor: &RotaryEmbedding,
        table: &PreparedRope,
        input: Rows,
        heads: usize,
        positions: &[usize],
    ) -> Result<OperatorProgress<Rows>> {
        self.synchronous(|this| {
            validate_rope(descriptor)?;
            if heads.checked_mul(descriptor.head_dim()) != Some(input.shape().width())
                || table.dimensions() != descriptor.region().dimensions()
            {
                return Err(cuda_error("BF16 RoPE shape mismatch"));
            }
            this.trace_bf16(
                "rope.input",
                this.bf16_buffer(&input)?,
                input.shape(),
                None,
                None,
            )?;
            let (c, s) = this.bf16_rope_weight(table)?;
            let output = this.ops.rope_bf16(
                this.bf16_buffer(&input)?,
                &c,
                &s,
                positions,
                SplitHalfRopeLayout {
                    rows: input.shape().rows(),
                    heads,
                    head_dim: descriptor.head_dim(),
                    rope_dim: table.dimensions(),
                    table_positions: table.positions(),
                    restore_bf16_boundary: true,
                },
            )?;
            this.trace_bf16("rope.output", &output, input.shape(), None, None)?;
            rows_bf16(input.shape(), input.arena(), output).map(OperatorProgress::Ready)
        })
    }
    pub(super) fn bf16_swiglu(
        &mut self,
        expert: &PreparedSwiGlu,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.synchronous(|this| {
            if expert.activation_limit().is_some() {
                return Err(cuda_error("clamped BF16 SwiGLU is not admitted"));
            }
            if this.numeric.is_some() {
                let bytes = crate::transformer::SwiGluScratchPlan::for_prepared(
                    expert,
                    input.shape().rows(),
                )?
                .total_bytes()?
                .checked_mul(2)
                .ok_or_else(|| cuda_error("BF16 SwiGLU scratch overflow"))?;
                this.reserve_numeric_temporaries(bytes)?;
            }
            let gate = this.bf16_linear_rows(expert.gate(), input, arena)?;
            let up = this.bf16_linear_rows(expert.up(), input, arena)?;
            let product = this
                .ops
                .swiglu_bf16(this.bf16_buffer(&gate)?, this.bf16_buffer(&up)?)?;
            this.trace_bf16(
                "swiglu.product",
                &product,
                gate.shape(),
                Some(expert.down()),
                None,
            )?;
            let product = rows_bf16(gate.shape(), arena, product)?;
            this.bf16_linear_rows(expert.down(), &product, arena)
                .map(OperatorProgress::Ready)
        })
    }
    pub(super) fn bf16_residual(
        &mut self,
        input: Rows,
        update: &Rows,
    ) -> Result<OperatorProgress<Rows>> {
        self.synchronous(|this| {
            if input.shape() != update.shape() {
                return Err(cuda_error("BF16 residual shape mismatch"));
            }
            this.trace_bf16(
                "residual.input",
                this.bf16_buffer(&input)?,
                input.shape(),
                None,
                None,
            )?;
            this.trace_bf16(
                "residual.update",
                this.bf16_buffer(update)?,
                update.shape(),
                None,
                None,
            )?;
            let output = this
                .ops
                .residual_bf16(this.bf16_buffer(&input)?, this.bf16_buffer(update)?)?;
            this.trace_bf16("residual.output", &output, input.shape(), None, None)?;
            rows_bf16(input.shape(), input.arena(), output).map(OperatorProgress::Ready)
        })
    }
}
pub(super) fn rows_bf16(
    shape: RowsShape,
    arena: Option<RowsArenaId>,
    buffer: CudaBf16Buffer,
) -> Result<Rows> {
    CudaRows::bf16(shape, arena, buffer).map(Rows::Cuda)
}
pub(super) fn round_rows(
    ops: &CudaOperators,
    shape: RowsShape,
    arena: Option<RowsArenaId>,
    input: &CudaF32Buffer,
) -> Result<Rows> {
    let output = ops.f32_rows_to_bf16_rne(
        input,
        F32ToBf16RowsLayout::packed(shape.rows(), 1, shape.width())?,
    )?;
    rows_bf16(shape, arena, output)
}
