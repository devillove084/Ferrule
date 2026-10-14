//! BF16 cache binding issued only by an entered transaction view. This is not
//! the F32 HostKV bridge and does not expose mutable allocations or slot tables.
use super::{CudaPagedKvPool, Inner, error, plan};
use crate::cpu::PagedKvBatch;
use crate::cuda::operator_owner::{
    CudaBf16Buffer, CudaF32Buffer, CudaI32Buffer, CudaOperators, CudaTypedBuffer,
};
use crate::cuda::operators::attention::prepared_bf16::{
    Bf16AttentionGroup, Bf16AttentionPlan, Bf16AttentionView, CudaPreparedBf16Attention,
};
use crate::cuda::operators::attention::{
    PagedBf16CausalGqaLayout, PagedBf16GqaBuffers, PagedBf16PlaneLayout,
};
use crate::cuda::operators::kv::{TypedKvHostSnapshot, TypedPagedPlaneLayout};
use crate::cuda::runtime::DeviceCopy;
use ferrule_common::Result;
use ferrule_common::execution::{KvElementType, KvPageId};
use std::mem::size_of;

/// Model-owned row operands. Q/output remain F32 as required by the existing
/// kernel; appended K/V and the persistent cache are actual BF16 words.
pub struct CudaBf16GqaRows<'a> {
    pub query: &'a CudaF32Buffer,
    pub append_key: &'a CudaBf16Buffer,
    pub append_value: &'a CudaBf16Buffer,
    pub output: &'a mut CudaF32Buffer,
}

/// An opaque typed plane guard: metadata is descriptive, never write authority.
/// Only `CudaKvView::with_bf16_gqa_planes` can issue one. The pool remains
/// exclusively borrowed for its lifetime, and leave/commit retain normal fences.
pub struct CudaBf16GqaPlanes<'a> {
    context: &'a CudaOperators,
    key: &'a mut CudaBf16Buffer,
    value: &'a mut CudaBf16Buffer,
    layout: TypedPagedPlaneLayout,
    capacity: usize,
    batch: &'a PagedKvBatch,
    slots: &'a [i32],
    offsets: &'a [i32],
}

impl<'a> CudaBf16GqaPlanes<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        context: &'a CudaOperators,
        key: &'a mut CudaBf16Buffer,
        value: &'a mut CudaBf16Buffer,
        layout: TypedPagedPlaneLayout,
        capacity: usize,
        batch: &'a PagedKvBatch,
        slots: &'a [i32],
        offsets: &'a [i32],
    ) -> Result<Self> {
        layout.validate()?;
        let elements = layout
            .checked_elements_per_page()
            .and_then(|n| n.checked_mul(capacity))
            .ok_or_else(|| error("BF16 cache capacity overflow"))?;
        if layout.element_type != KvElementType::Bf16
            || capacity == 0
            || key.len() != elements
            || value.len() != elements
            || slots.iter().any(|&s| s < 0 || s as usize >= capacity)
        {
            return Err(error("BF16 cache dtype, capacity or slots mismatch"));
        }
        plan::bf16_row_metadata(batch, layout.layout.page_tokens, slots, offsets)?;
        let k = buffer_range(context, key)?;
        let v = buffer_range(context, value)?;
        disjoint(k, v)?;
        Ok(Self {
            context,
            key,
            value,
            layout,
            capacity,
            batch,
            slots,
            offsets,
        })
    }

    pub const fn layout(&self) -> TypedPagedPlaneLayout {
        self.layout
    }
    pub const fn kernel_slot_capacity(&self) -> usize {
        self.capacity
    }

    /// Construct the exact physical cache strides; callers may customize only
    /// query/append/output row strides before submitting `append_and_attend`.
    pub fn attention_layout(
        &self,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        softmax_scale: f32,
    ) -> Result<PagedBf16CausalGqaLayout> {
        let p = self.layout.layout;
        let layout = PagedBf16CausalGqaLayout::packed(
            self.batch.row_positions.len(),
            self.batch.sequences.len(),
            q_heads,
            kv_heads,
            head_dim,
            p.page_tokens,
            p.layer_index,
            p.layer_count,
            self.capacity,
            softmax_scale,
        )?;
        validate_layout(self.layout, self.capacity, self.batch, layout)?;
        Ok(layout)
    }

    /// Append only this transaction's staged rows, then attend its authenticated
    /// history. No caller-provided device metadata or cache pointers are accepted.
    /// This status-checking path is synchronous and does not support graph capture.
    pub fn append_and_attend(
        &mut self,
        rows: CudaBf16GqaRows<'_>,
        layout: PagedBf16CausalGqaLayout,
    ) -> Result<()> {
        self.append_and_attend_profile(rows, layout, false)
    }

    /// Same transaction/storage authority with HF eager BF16 score/probability boundaries.
    pub fn append_and_attend_eager(
        &mut self,
        rows: CudaBf16GqaRows<'_>,
        layout: PagedBf16CausalGqaLayout,
    ) -> Result<()> {
        self.append_and_attend_profile(rows, layout, true)
    }

    /// Compose a checked view from this transaction's authenticated batch.
    /// No private page/slot storage is exposed and no device work is submitted.
    pub fn prepared_attention_view(&self, plan: &Bf16AttentionPlan) -> Result<Bf16AttentionView> {
        prepared_view_for_batch(self.batch, plan)
    }

    /// Prepared arithmetic through the same exclusive plane custody as legacy
    /// append/attend. Caller begins the provider invocation once, shares its
    /// sticky status across layers, and finishes before committing/publishing.
    /// This adapter uploads authenticated metadata; the arithmetic execute call
    /// itself remains allocation/sync-free. It never clears or consumes status.
    pub fn append_and_attend_prepared(
        &mut self,
        rows: CudaBf16GqaRows<'_>,
        layout: PagedBf16CausalGqaLayout,
        prepared: &mut CudaPreparedBf16Attention,
        status: &mut CudaI32Buffer,
    ) -> Result<()> {
        validate_layout(self.layout, self.capacity, self.batch, layout)?;
        let view = self.prepared_attention_view(prepared.plan())?;
        prepared.preflight_view(self.context, status, &view)?;
        let reads = [
            buffer_range(self.context, rows.query)?,
            buffer_range(self.context, rows.append_key)?,
            buffer_range(self.context, rows.append_value)?,
        ];
        let writes = [
            buffer_range(self.context, self.key)?,
            buffer_range(self.context, self.value)?,
            buffer_range(self.context, rows.output)?,
            buffer_range(self.context, status)?,
        ];
        for (i, &w) in writes.iter().enumerate() {
            for &r in reads.iter().chain(writes[..i].iter()) {
                disjoint(w, r)?;
            }
        }
        let (ids, positions, lengths) = plan::bf16_row_metadata(
            self.batch,
            self.layout.layout.page_tokens,
            self.slots,
            self.offsets,
        )?;
        let slots = self.context.upload_i32_buffer(self.slots)?;
        let offsets = self.context.upload_i32_buffer(self.offsets)?;
        let ids = self.context.upload_i32_buffer(&ids)?;
        let positions = self.context.upload_i32_buffer(&positions)?;
        let lengths = self.context.upload_i32_buffer(&lengths)?;
        prepared.execute_view(
            self.context,
            PagedBf16GqaBuffers {
                query: rows.query,
                append_key: rows.append_key,
                append_value: rows.append_value,
                key_cache: self.key,
                value_cache: self.value,
                block_slots: &slots,
                block_offsets: &offsets,
                row_sequence_ids: &ids,
                row_positions: &positions,
                row_kv_lens: &lengths,
                output: rows.output,
                status,
            },
            layout,
            &view,
            true,
        )
    }

    fn append_and_attend_profile(
        &mut self,
        rows: CudaBf16GqaRows<'_>,
        layout: PagedBf16CausalGqaLayout,
        eager: bool,
    ) -> Result<()> {
        validate_layout(self.layout, self.capacity, self.batch, layout)?;
        if self.context.stream_clone().context().is_capturing() {
            return Err(error(
                "transactional BF16 attention does not support graph capture",
            ));
        }
        let reads = [
            buffer_range(self.context, rows.query)?,
            buffer_range(self.context, rows.append_key)?,
            buffer_range(self.context, rows.append_value)?,
        ];
        let writes = [
            buffer_range(self.context, self.key)?,
            buffer_range(self.context, self.value)?,
            buffer_range(self.context, rows.output)?,
        ];
        for (i, &write) in writes.iter().enumerate() {
            for &read in &reads {
                disjoint(write, read)?;
            }
            for &other in &writes[..i] {
                disjoint(write, other)?;
            }
        }
        let (ids, positions, lengths) = plan::bf16_row_metadata(
            self.batch,
            self.layout.layout.page_tokens,
            self.slots,
            self.offsets,
        )?;
        let slots = self.context.upload_i32_buffer(self.slots)?;
        let offsets = self.context.upload_i32_buffer(self.offsets)?;
        let ids = self.context.upload_i32_buffer(&ids)?;
        let positions = self.context.upload_i32_buffer(&positions)?;
        let lengths = self.context.upload_i32_buffer(&lengths)?;
        let mut status = self.context.zero_i32_buffer(1)?;
        let attend = if eager {
            CudaOperators::append_and_attend_paged_bf16_eager
        } else {
            CudaOperators::append_and_attend_paged_bf16
        };
        attend(
            self.context,
            PagedBf16GqaBuffers {
                query: rows.query,
                append_key: rows.append_key,
                append_value: rows.append_value,
                key_cache: self.key,
                value_cache: self.value,
                block_slots: &slots,
                block_offsets: &offsets,
                row_sequence_ids: &ids,
                row_positions: &positions,
                row_kv_lens: &lengths,
                output: rows.output,
                status: &mut status,
            },
            layout,
        )
    }
}

fn prepared_view_for_batch(
    batch: &PagedKvBatch,
    plan: &Bf16AttentionPlan,
) -> Result<Bf16AttentionView> {
    if batch.row_positions.len() != batch.row_sequence_ids.len()
        || batch.row_positions.len() > plan.max_total_rows()
        || batch.sequences.len() > plan.max_sequences()
    {
        return Err(error("prepared batch capacity/metadata mismatch"));
    }
    let mut groups = Vec::with_capacity(batch.sequences.len());
    let mut start = 0usize;
    for (seq, sequence) in batch.sequences.iter().enumerate() {
        let mut end = start;
        while end < batch.row_sequence_ids.len() && batch.row_sequence_ids[end] == seq {
            end += 1;
        }
        let count = end - start;
        let first = sequence
            .sequence_len
            .checked_sub(count)
            .ok_or_else(|| error("prepared cohort exceeds active KV"))?;
        if count == 0
            || batch.row_positions[start..end]
                .iter()
                .enumerate()
                .any(|(i, &p)| p != first + i)
        {
            return Err(error(
                "prepared cohorts must contain contiguous query suffixes in sequence order",
            ));
        }
        groups.push(Bf16AttentionGroup {
            row_start: u32::try_from(start).map_err(|_| error("row offset overflow"))?,
            rows: u32::try_from(count).map_err(|_| error("row count overflow"))?,
            sequence: u32::try_from(seq).map_err(|_| error("sequence overflow"))?,
            tokens: u32::try_from(sequence.sequence_len)
                .map_err(|_| error("KV length overflow"))?,
        });
        start = end;
    }
    if start != batch.row_positions.len() {
        return Err(error("prepared cohorts do not cover all query rows"));
    }
    plan.view(&groups)
}

fn validate_layout(
    plane: TypedPagedPlaneLayout,
    capacity: usize,
    batch: &PagedKvBatch,
    layout: PagedBf16CausalGqaLayout,
) -> Result<()> {
    plane.validate()?;
    layout.validate()?;
    let p = plane.layout;
    if plane.element_type != KvElementType::Bf16
        || capacity == 0
        || layout.rows != batch.row_positions.len()
        || layout.sequences != batch.sequences.len()
        || layout.page_tokens != p.page_tokens
        || layout.layer_index != p.layer_index
        || layout.layer_count != p.layer_count
        || layout.kv_heads.checked_mul(layout.head_dim) != Some(p.elements_per_token)
    {
        return Err(error(
            "BF16 attention layout disagrees with transaction plane/batch",
        ));
    }
    for n in [
        layout.rows,
        layout.sequences,
        layout.q_heads,
        layout.kv_heads,
        layout.head_dim,
    ] {
        u32::try_from(n).map_err(|_| error("BF16 attention shape exceeds u32 ABI"))?;
    }
    let expected = PagedBf16PlaneLayout::packed(
        capacity,
        p.layer_count,
        p.page_tokens,
        layout.kv_heads,
        layout.head_dim,
    )?;
    if layout.key_cache != expected || layout.value_cache != expected {
        return Err(error(
            "BF16 attention cache strides/capacity do not match owned storage",
        ));
    }
    Ok(())
}

fn byte_range(start: u64, elements: usize, element_bytes: usize) -> Result<(u64, u64)> {
    if !start.is_multiple_of(element_bytes as u64) {
        return Err(error("unaligned BF16 attention operand"));
    }
    let bytes = elements
        .checked_mul(element_bytes)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| error("BF16 attention operand byte overflow"))?;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| error("BF16 attention address overflow"))?;
    Ok((start, end))
}
fn buffer_range<T: DeviceCopy>(
    context: &CudaOperators,
    buffer: &CudaTypedBuffer<T>,
) -> Result<(u64, u64)> {
    let device = buffer.as_device_buffer();
    device
        .check_context(
            context.stream_clone().context(),
            "transactional BF16 operand",
        )
        .map_err(|e| error(e.to_string()))?;
    byte_range(device.cu_deviceptr(), buffer.len(), size_of::<T>())
}
fn disjoint(a: (u64, u64), b: (u64, u64)) -> Result<()> {
    if a.0 < a.1 && b.0 < b.1 && a.0 < b.1 && b.0 < a.1 {
        return Err(error("BF16 attention writable buffers overlap"));
    }
    Ok(())
}

impl CudaPagedKvPool {
    /// Raw typed payload produced by this pool's completed preemption. This is
    /// an observation, not a restore/import or logical ownership capability.
    pub fn typed_host_snapshot(&self, page: KvPageId) -> Result<TypedKvHostSnapshot> {
        let inner = self
            .inner
            .try_borrow()
            .map_err(|_| error("CUDA KV storage already borrowed"))?;
        inner.available()?;
        inner
            .storage()
            .typed_snapshot(page)
            .cloned()
            .ok_or_else(|| error("page has no complete typed host snapshot"))
    }
}

pub(super) fn clear_transaction_slot(
    inner: &mut Inner,
    context: &CudaOperators,
    slot: u32,
) -> Result<()> {
    if inner.storage().planes()[0].element_type == ferrule_common::execution::KvElementType::Bf16 {
        let planes = inner.storage().planes().len();
        for plane in 0..planes {
            let elements = inner
                .storage()
                .page_elements(plane)
                .ok_or_else(|| error("BF16 transaction plane layout is absent"))?;
            let offset = (slot as usize)
                .checked_mul(elements)
                .ok_or_else(|| error("BF16 transaction slot offset overflow"))?;
            let zeroes = vec![0u16; elements];
            inner
                .storage_mut()
                .overwrite_bf16_words_range(context, plane, offset, &zeroes)?;
        }
        Ok(())
    } else {
        inner.storage_mut().clear_transaction_slot(context, slot)
    }
}

#[cfg(test)]
#[path = "bf16_tests.rs"]
mod tests;
