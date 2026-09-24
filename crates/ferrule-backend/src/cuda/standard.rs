//! Explicit F32 standard-transformer primitives on the existing CUDA owner.
//!
//! Activations, outputs and K/V storage are F32. Checkpoint BF16-to-F32 weight
//! conversion is the caller's responsibility; no implicit BF16/FP8 boundary is
//! introduced here. These are correctness-first kernels, not a decoder or a
//! performance claim. Host work is limited to shape/address metadata.
//!
//! Every typed input is checked against the exact context owner (not ordinal).
//! Borrowed buffers hold their allocations through submission. The canonical
//! allocator fences retirement on registered streams, including temporary
//! metadata dropped after launch. No raw pointers escape these APIs.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    CudaArtifactLinearHandle, CudaArtifactLinearShape, CudaF32Buffer, CudaI32Buffer, CudaOperators,
    CudaTypedBuffer, checked_u32, cu, exact_element_bytes,
};
use crate::cuda::ffi::core::{
    MOE_SWIGLU_CLAMPED_F32, MOE_SWIGLU_STANDARD_F32, MOE_WEIGHTED_COMBINE_F32, MoeArgs,
    NORM_AFFINE_F32, NormArgs, TRANSFORMER_PAGED_F32_APPEND_CAUSAL_GQA, TransformerArgs,
};
pub use crate::cuda::operators::{SelectedSoftmaxTopKLayout, SplitHalfRopeLayout};
use crate::cuda::providers::cutlass::{F32GemmLayout, f32_gemm_bytes};
use crate::cuda::runtime::DeviceCopy;
use ferrule_common::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("standard CUDA: {}", message.into()),
    }
}

fn product(dimensions: &[usize]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |n, &d| {
        if d == 0 {
            return Err(invalid("dimensions must be positive"));
        }
        n.checked_mul(d).ok_or_else(|| invalid("shape overflow"))
    })
}

fn count(value: usize) -> Result<u32> {
    // Legacy SIMT loops use u32 counters; leave room for block-size increments
    // and keep signed row/expert metadata representable without wraparound.
    if value > i32::MAX as usize {
        return Err(invalid("dimension/count exceeds i32::MAX"));
    }
    checked_u32(value, "standard CUDA", "dimension/count")
}

/// Packed Q/output `[rows, q_heads, head_dim]`, append K/V
/// `[rows, kv_heads, head_dim]`, separate K/V planes
/// `[physical_slots, layer_count, page_tokens, kv_heads, head_dim]`.
/// There is deliberately no hidden-size constraint on `q_heads * head_dim`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedF32GqaLayout {
    pub rows: usize,
    pub sequences: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub page_tokens: usize,
    pub layer_index: usize,
    pub layer_count: usize,
    pub physical_slots: usize,
    pub softmax_scale: f32,
}

impl PagedF32GqaLayout {
    pub fn validate(self) -> Result<()> {
        for d in [
            self.rows,
            self.sequences,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.page_tokens,
            self.layer_count,
            self.physical_slots,
        ] {
            if d == 0 || d > i32::MAX as usize {
                return Err(invalid("dimension outside 1..=i32::MAX"));
            }
        }
        if !self.q_heads.is_multiple_of(self.kv_heads)
            || self.layer_index >= self.layer_count
            || !self.softmax_scale.is_finite()
            || self.softmax_scale <= 0.0
        {
            return Err(invalid("invalid GQA ratio, layer or softmax scale"));
        }
        count(self.query_elements()?)?;
        count(self.append_elements()?)?;
        exact_element_bytes::<f32>(self.cache_elements()?, "standard K/V plane")?;
        Ok(())
    }
    pub fn query_elements(self) -> Result<usize> {
        product(&[self.rows, self.q_heads, self.head_dim])
    }
    pub fn append_elements(self) -> Result<usize> {
        product(&[self.rows, self.kv_heads, self.head_dim])
    }
    pub fn cache_elements(self) -> Result<usize> {
        product(&[
            self.physical_slots,
            self.layer_count,
            self.page_tokens,
            self.kv_heads,
            self.head_dim,
        ])
    }
}

/// CPU control metadata, validated before any cache write and uploaded by the
/// backend. `block_slots` contains actual physical slot indices resolved by the
/// model KV owner, NEVER logical page IDs. CSR `block_offsets` has sequences+1
/// entries. `committed_lengths` is the initialized prefix length before append.
/// Rows may interleave sequences but must extend each sequence consecutively.
/// Shared read-only prefixes are supported; append destinations must be private
/// (the model KV owner must perform copy-on-write first).
#[derive(Debug, Clone, Copy)]
pub struct PagedF32GqaMetadata<'a> {
    pub block_slots: &'a [i32],
    pub block_offsets: &'a [i32],
    pub row_sequence_ids: &'a [i32],
    pub row_positions: &'a [i32],
    pub committed_lengths: &'a [i32],
}

impl PagedF32GqaMetadata<'_> {
    /// Validate the transaction and return per-row visible causal lengths.
    pub fn validate(self, layout: PagedF32GqaLayout) -> Result<Vec<i32>> {
        layout.validate()?;
        if self.block_offsets.len() != layout.sequences + 1
            || self.committed_lengths.len() != layout.sequences
            || self.row_sequence_ids.len() != layout.rows
            || self.row_positions.len() != layout.rows
            || self.block_slots.is_empty()
            || self.block_slots.len() > i32::MAX as usize
            || self.block_offsets.first() != Some(&0)
            || self.block_offsets.last().copied() != Some(self.block_slots.len() as i32)
            || self
                .block_offsets
                .windows(2)
                .any(|w| w[0] < 0 || w[0] > w[1])
            || self
                .block_slots
                .iter()
                .any(|&s| s < 0 || s as usize >= layout.physical_slots)
            || self.committed_lengths.iter().any(|&n| n < 0)
        {
            return Err(invalid("invalid paged GQA metadata lengths/slots/offsets"));
        }
        let address = |sequence: usize, position: usize| -> Result<(i32, usize)> {
            let start = self.block_offsets[sequence] as usize;
            let end = self.block_offsets[sequence + 1] as usize;
            let entry = start
                .checked_add(position / layout.page_tokens)
                .ok_or_else(|| invalid("block index overflow"))?;
            if entry >= end {
                return Err(invalid("page table does not cover visible prefix"));
            }
            Ok((self.block_slots[entry], position % layout.page_tokens))
        };
        let mut lengths = self.committed_lengths.to_vec();
        let mut writes = BTreeSet::new();
        let mut visible = Vec::with_capacity(layout.rows);
        for row in 0..layout.rows {
            let sequence = self.row_sequence_ids[row];
            let position = self.row_positions[row];
            if sequence < 0
                || sequence as usize >= layout.sequences
                || position < 0
                || position != lengths[sequence as usize]
            {
                return Err(invalid(
                    "rows must append consecutive positions after each committed prefix",
                ));
            }
            lengths[sequence as usize] = position
                .checked_add(1)
                .ok_or_else(|| invalid("position exceeds i32"))?;
            if !writes.insert(address(sequence as usize, position as usize)?) {
                return Err(invalid(
                    "aliased append destinations; copy-on-write required",
                ));
            }
            visible.push(position + 1);
        }
        // Count all logical references to written cells: even another sequence's
        // read-only prefix must not observe a concurrent overwrite.
        let mut write_references = BTreeMap::new();
        for (sequence, &length) in lengths.iter().enumerate() {
            let capacity =
                (self.block_offsets[sequence + 1] - self.block_offsets[sequence]) as usize;
            if (length as usize).div_ceil(layout.page_tokens) > capacity {
                return Err(invalid("page table does not cover final sequence length"));
            }
            for position in 0..length as usize {
                let cell = address(sequence, position)?;
                if writes.contains(&cell) {
                    let refs = write_references.entry(cell).or_insert(0usize);
                    *refs += 1;
                    if *refs > 1 {
                        return Err(invalid(
                            "append aliases another visible token; copy-on-write required",
                        ));
                    }
                }
            }
        }
        Ok(visible)
    }
}

pub struct PagedF32GqaBuffers<'a> {
    pub query: &'a CudaF32Buffer,
    pub append_key: &'a CudaF32Buffer,
    pub append_value: &'a CudaF32Buffer,
    pub key_cache: &'a mut CudaF32Buffer,
    pub value_cache: &'a mut CudaF32Buffer,
    pub output: &'a mut CudaF32Buffer,
}

impl CudaOperators {
    fn standard_buffer<T: DeviceCopy>(
        &self,
        buffer: &CudaTypedBuffer<T>,
        len: usize,
    ) -> Result<()> {
        self.check_buffer_owner(&buffer.buffer, "standard F32 operator")?;
        if buffer.len() != len {
            return Err(invalid(format!(
                "buffer length {}, expected {len}",
                buffer.len()
            )));
        }
        Ok(())
    }

    /// Bias-free F32 `[rows,in] * [out,in]^T` through the existing CUTLASS provider.
    /// Uses TF32x3 TensorOp multiplication with F32 accumulation/output (not
    /// bitwise IEEE SGEMM). No implicit SIMT fallback on unsupported capability
    /// or shape. Only F32 artifacts are accepted; conversion is external.
    /// Submission uses this owner's compute stream, events and retirement fences.
    pub fn linear_f32_into(
        &self,
        weight: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let CudaArtifactLinearShape::F32 {
            out_features,
            in_features,
        } = weight.shape
        else {
            return Err(invalid("linear requires F32 weight storage"));
        };
        weight.validate_storage()?;
        self.check_buffer_owner(&weight.weight, "standard F32 linear weight")?;
        self.standard_buffer(input, product(&[rows, in_features])?)?;
        self.standard_buffer(output, product(&[rows, out_features])?)?;
        self.launched(f32_gemm_bytes(
            &self.stream,
            &input.buffer,
            &weight.weight,
            &mut output.buffer,
            F32GemmLayout::contiguous(rows, out_features, in_features),
        ))
    }

    /// Checked embedding/row gather. IDs are CPU metadata; all value copying is GPU-side.
    pub fn embedding_f32(
        &self,
        embedding: &CudaF32Buffer,
        token_ids: &[i32],
        width: usize,
    ) -> Result<CudaF32Buffer> {
        self.check_buffer_owner(&embedding.buffer, "standard F32 embedding")?;
        if width == 0
            || embedding.is_empty()
            || !embedding.len().is_multiple_of(width)
            || token_ids.is_empty()
            || token_ids
                .iter()
                .any(|&id| id < 0 || id as usize >= embedding.len() / width)
        {
            return Err(invalid("invalid embedding shape or token ID"));
        }
        count(product(&[token_ids.len(), width])?)?;
        count(embedding.len() / width)?;
        let ids = self.upload_i32_buffer(token_ids)?;
        self.gather_f32_rows(embedding, &ids, token_ids.len(), width)
    }

    /// `y = x * rsqrt(mean(x*x) + epsilon) * weight`, without BF16 rounding.
    /// Flatten `[rows,heads,head_dim]` into `[rows*heads,head_dim] for Q/K norm.
    pub fn rms_norm_f32_into(
        &self,
        input: &CudaF32Buffer,
        rows: usize,
        weight: &CudaF32Buffer,
        epsilon: f32,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let len = product(&[rows, weight.len()])?;
        self.standard_buffer(input, len)?;
        self.standard_buffer(weight, weight.len())?;
        self.standard_buffer(output, len)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid("RMS epsilon must be finite and positive"));
        }
        let args = NormArgs {
            kind: NORM_AFFINE_F32,
            rows: count(rows)?,
            width: count(weight.len())?,
            epsilon,
            input: input.buffer.cu_deviceptr(),
            weight: weight.buffer.cu_deviceptr(),
            output: output.buffer.cu_deviceptr(),
            ..Default::default()
        };
        self.launched(unsafe { self.module.standard_norm(&self.stream, args) })
    }

    /// In-place split-half RoPE using checked per-row host positions. Tables are
    /// `[table_positions, rope_dim/2]`; non-rotary dimensions remain unchanged.
    /// `restore_bf16_boundary` must be false. Use the existing explicit RoPE API
    /// when a BF16 compatibility boundary is wanted.
    pub fn split_half_rope_f32(
        &self,
        values: &mut CudaF32Buffer,
        cosine: &CudaF32Buffer,
        sine: &CudaF32Buffer,
        positions: &[i32],
        layout: SplitHalfRopeLayout,
    ) -> Result<()> {
        layout.validate()?;
        self.standard_buffer(values, layout.value_elements()?)?;
        self.standard_buffer(cosine, layout.table_elements()?)?;
        self.standard_buffer(sine, layout.table_elements()?)?;
        count(layout.pair_count()?)?;
        if layout.restore_bf16_boundary
            || positions.len() != layout.rows
            || positions
                .iter()
                .any(|&p| p < 0 || p as usize >= layout.table_positions)
        {
            return Err(invalid(
                "F32 RoPE requires valid row positions and no BF16 boundary",
            ));
        }
        let positions = self.upload_i32_buffer(positions)?;
        self.split_half_rope_rows_indexed_from_device(
            values, cosine, sine, &positions, layout, false,
        )
    }

    /// F32 residual addition; no rounding boundary is inserted.
    pub fn residual_add_f32_in_place(
        &self,
        update: &CudaF32Buffer,
        residual: &mut CudaF32Buffer,
    ) -> Result<()> {
        self.standard_buffer(update, residual.len())?;
        self.standard_buffer(residual, update.len())?;
        count(update.len())?;
        if update.is_empty() {
            return Err(invalid("empty residual"));
        }
        self.saxpy_into(1.0, update, residual)
    }

    pub fn residual_add_f32_into(
        &self,
        residual: &CudaF32Buffer,
        update: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        self.standard_buffer(residual, update.len())?;
        self.standard_buffer(update, residual.len())?;
        self.standard_buffer(output, residual.len())?;
        count(residual.len())?;
        if residual.is_empty() {
            return Err(invalid("empty residual"));
        }
        self.copy_f32_range(residual, 0, output, 0, residual.len())?;
        self.residual_add_f32_in_place(update, output)
    }

    pub fn residual_add_f32(
        &self,
        residual: &CudaF32Buffer,
        update: &CudaF32Buffer,
    ) -> Result<CudaF32Buffer> {
        self.standard_buffer(residual, update.len())?;
        self.standard_buffer(update, residual.len())?;
        let mut output = self.zero_f32_buffer(residual.len())?;
        self.residual_add_f32_into(residual, update, &mut output)?;
        Ok(output)
    }

    /// Exact standard SiLU(gate)*up in F32, without the legacy sigmoid cutoff,
    /// clipping, routing weights or BF16/FP8 quantization.
    pub fn swiglu_f32_into(
        &self,
        gate: &CudaF32Buffer,
        up: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        self.standard_buffer(gate, up.len())?;
        self.standard_buffer(up, gate.len())?;
        self.standard_buffer(output, gate.len())?;
        if gate.is_empty() {
            return Err(invalid("empty SwiGLU"));
        }
        let args = MoeArgs {
            kind: MOE_SWIGLU_STANDARD_F32,
            n: count(gate.len())?,
            gate: gate.buffer.cu_deviceptr(),
            up: up.buffer.cu_deviceptr(),
            output: output.buffer.cu_deviceptr(),
            ..Default::default()
        };
        self.launched(unsafe { self.module.standard_moe(&self.stream, args) })
    }

    /// F32 `SiLU(min(gate, limit)) * clamp(up, -limit, limit)`.
    /// `limit` must be finite and nonnegative; zero clamps the up values to zero.
    /// Inputs are preserved. No legacy sigmoid cutoff or BF16/FP8 rounding is used.
    pub fn swiglu_f32_clamped_into(
        &self,
        gate: &CudaF32Buffer,
        up: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
        limit: f32,
    ) -> Result<()> {
        self.standard_buffer(gate, up.len())?;
        self.standard_buffer(up, gate.len())?;
        self.standard_buffer(output, gate.len())?;
        if gate.is_empty() {
            return Err(invalid("empty SwiGLU"));
        }
        if !limit.is_finite() || limit < 0.0 {
            return Err(invalid("SwiGLU limit must be finite and nonnegative"));
        }
        let args = MoeArgs {
            kind: MOE_SWIGLU_CLAMPED_F32,
            n: count(gate.len())?,
            swiglu_limit: limit,
            gate: gate.buffer.cu_deviceptr(),
            up: up.buffer.cu_deviceptr(),
            output: output.buffer.cu_deviceptr(),
            ..Default::default()
        };
        self.launched(unsafe { self.module.standard_moe(&self.stream, args) })
    }

    /// Overwrite `[output_rows,width]` with weighted route sums. Routes are
    /// `[route_rows,width]`; each output element left-folds routes in ascending
    /// route-row order, with separate F32 multiply/add and no atomic additions.
    /// Negative/out-of-range target rows are ignored (masked routes).
    pub fn weighted_combine_f32_into(
        &self,
        values: &CudaF32Buffer,
        route_rows: &CudaI32Buffer,
        weights: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
        output_rows: usize,
        width: usize,
    ) -> Result<()> {
        self.standard_buffer(values, product(&[route_rows.len(), width])?)?;
        self.standard_buffer(route_rows, weights.len())?;
        self.standard_buffer(weights, route_rows.len())?;
        self.standard_buffer(output, product(&[output_rows, width])?)?;
        count(values.len())?;
        count(output.len())?;
        let args = MoeArgs {
            kind: MOE_WEIGHTED_COMBINE_F32,
            batch_columns: count(route_rows.len())?,
            tokens: count(output_rows)?,
            hidden: count(width)?,
            input: values.buffer.cu_deviceptr(),
            route_slots: route_rows.buffer.cu_deviceptr(),
            route_weights: weights.buffer.cu_deviceptr(),
            output: output.buffer.cu_deviceptr(),
            ..Default::default()
        };
        self.launched(unsafe { self.module.standard_moe(&self.stream, args) })
    }

    /// Existing selected-logit softmax top-k, normalized over the selected k.
    /// Ties choose lower expert index. This is NOT unnormalized full-softmax top-k.
    pub fn router_softmax_topk_f32_into(
        &self,
        logits: &CudaF32Buffer,
        indices: &mut CudaI32Buffer,
        weights: &mut CudaF32Buffer,
        layout: SelectedSoftmaxTopKLayout,
    ) -> Result<()> {
        layout.validate()?;
        self.standard_buffer(logits, layout.logit_elements()?)?;
        self.standard_buffer(indices, layout.output_elements()?)?;
        self.standard_buffer(weights, layout.output_elements()?)?;
        self.selected_softmax_topk_from_device_into(logits, indices, weights, layout)
    }

    /// Append all rows, then causal GQA. Host metadata is checked/uploaded; all
    /// QK, softmax and value reduction execute on GPU. This convenience call is
    /// not graph-capture-safe: it allocates metadata and reads one status word.
    /// The caller guarantees committed prefixes contain initialized K/V for this
    /// layer and publishes lengths only after success. Native failure is not a
    /// rollback: partially appended cache data must not be published.
    pub fn append_and_attend_paged_f32(
        &self,
        buffers: PagedF32GqaBuffers<'_>,
        metadata: PagedF32GqaMetadata<'_>,
        layout: PagedF32GqaLayout,
    ) -> Result<()> {
        self.check_capture_safe("standard GQA metadata upload/status download")?;
        let visible = metadata.validate(layout)?;
        let PagedF32GqaBuffers {
            query,
            append_key,
            append_value,
            key_cache,
            value_cache,
            output,
        } = buffers;
        self.standard_buffer(query, layout.query_elements()?)?;
        self.standard_buffer(output, layout.query_elements()?)?;
        self.standard_buffer(append_key, layout.append_elements()?)?;
        self.standard_buffer(append_value, layout.append_elements()?)?;
        self.standard_buffer(key_cache, layout.cache_elements()?)?;
        self.standard_buffer(value_cache, layout.cache_elements()?)?;
        let slots = self.upload_i32_buffer(metadata.block_slots)?;
        let offsets = self.upload_i32_buffer(metadata.block_offsets)?;
        let sequences = self.upload_i32_buffer(metadata.row_sequence_ids)?;
        let positions = self.upload_i32_buffer(metadata.row_positions)?;
        let lengths = self.upload_i32_buffer(&visible)?;
        let status = self.zero_i32_buffer(1)?;
        let head_stride = exact_element_bytes::<f32>(layout.head_dim, "GQA head")? as u64;
        let query_stride = head_stride * layout.q_heads as u64;
        let token_stride = head_stride * layout.kv_heads as u64;
        let layer_stride = token_stride * layout.page_tokens as u64;
        let slot_stride = layer_stride * layout.layer_count as u64;
        let query_bytes = exact_element_bytes::<f32>(query.len(), "GQA query")? as u64;
        let append_bytes = exact_element_bytes::<f32>(append_key.len(), "GQA append")? as u64;
        let cache_bytes = exact_element_bytes::<f32>(key_cache.len(), "GQA cache")? as u64;
        // Historical ABI field names end in _bf16; the separate F32 symbol
        // selects float kernels and validates every stride using four-byte words.
        let args = TransformerArgs {
            kind: TRANSFORMER_PAGED_F32_APPEND_CAUSAL_GQA,
            rows: count(layout.rows)?,
            sequences: count(layout.sequences)?,
            q_heads: count(layout.q_heads)?,
            kv_heads: count(layout.kv_heads)?,
            head_dim: count(layout.head_dim)?,
            page_tokens: count(layout.page_tokens)?,
            layer_index: count(layout.layer_index)?,
            layer_count: count(layout.layer_count)?,
            softmax_scale: layout.softmax_scale,
            query_f32: query.buffer.cu_deviceptr(),
            query_bytes,
            query_row_stride_bytes: query_stride,
            query_head_stride_bytes: head_stride,
            append_key_bf16: append_key.buffer.cu_deviceptr(),
            append_key_bytes: append_bytes,
            append_key_row_stride_bytes: token_stride,
            append_key_head_stride_bytes: head_stride,
            append_value_bf16: append_value.buffer.cu_deviceptr(),
            append_value_bytes: append_bytes,
            append_value_row_stride_bytes: token_stride,
            append_value_head_stride_bytes: head_stride,
            key_cache_bf16: key_cache.buffer.cu_deviceptr(),
            key_cache_bytes: cache_bytes,
            key_slot_stride_bytes: slot_stride,
            key_layer_stride_bytes: layer_stride,
            key_token_stride_bytes: token_stride,
            key_head_stride_bytes: head_stride,
            value_cache_bf16: value_cache.buffer.cu_deviceptr(),
            value_cache_bytes: cache_bytes,
            value_slot_stride_bytes: slot_stride,
            value_layer_stride_bytes: layer_stride,
            value_token_stride_bytes: token_stride,
            value_head_stride_bytes: head_stride,
            block_slots_i32: slots.buffer.cu_deviceptr(),
            block_slots_count: slots.len() as u64,
            block_offsets_i32: offsets.buffer.cu_deviceptr(),
            block_offsets_count: offsets.len() as u64,
            row_sequence_ids_i32: sequences.buffer.cu_deviceptr(),
            row_sequence_ids_count: sequences.len() as u64,
            row_positions_i32: positions.buffer.cu_deviceptr(),
            row_positions_count: positions.len() as u64,
            row_kv_lens_i32: lengths.buffer.cu_deviceptr(),
            row_kv_lens_count: lengths.len() as u64,
            output_f32: output.buffer.cu_deviceptr(),
            output_bytes: query_bytes,
            output_row_stride_bytes: query_stride,
            output_head_stride_bytes: head_stride,
            status_i32: status.buffer.cu_deviceptr(),
            status_count: 1,
            ..Default::default()
        };
        cu(unsafe { self.module.standard_gqa(&self.stream, args) })?;
        self.record_kernel_launches(2);
        match self.download_i32_buffer(&status)?[0] {
            0 => Ok(()),
            code => Err(Error::Execution {
                message: format!("standard F32 GQA device metadata/address status {code}"),
            }),
        }
    }
}
