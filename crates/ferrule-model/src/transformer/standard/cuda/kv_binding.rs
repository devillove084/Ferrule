//! Call-scoped computation binding over the independent CUDA KV custodian.
//! Slot allocation, rollback, COW and commit stay entirely in decoder/KV.

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_backend::cuda::operators::standard::{
    PagedF32GqaBuffers, PagedF32GqaLayout, PagedF32GqaMetadata,
};
use ferrule_common::Result;
use ferrule_common::execution::ExecutionTransactionId;

use crate::decoder::{CudaKvView, PackedDecoderBatch};
use crate::transformer::{
    GqaRequest, KvAppendRequest, KvHistory, KvView, OperatorProgress, Rows, StandardDecoderKvView,
};

use super::{cuda_error, device_rows};
#[path = "append_gqa.rs"]
mod append_gqa;

/// Borrows an entered view and its exact batch for one standard forward call.
/// It neither allocates pages nor acknowledges transaction completion.
pub struct CudaStandardKvBinding<'a> {
    view: &'a mut CudaKvView,
    batch: &'a PackedDecoderBatch,
    attention: Option<std::rc::Rc<std::cell::RefCell<super::bf16_attention::PreparedAttention>>>,
}

impl StandardDecoderKvView for CudaKvView {
    fn transaction(&self) -> ExecutionTransactionId {
        self.transaction()
    }
    fn validate_batch(&self, batch: &PackedDecoderBatch) -> Result<()> {
        self.validate_batch(batch)
    }
}

impl<'a> CudaStandardKvBinding<'a> {
    /// Call-scoped selection in the normal layer implementation. Neither branch
    /// can retain the context borrow or output guard in a continuation.
    pub(super) fn with_context<T>(
        view: &mut CudaKvView,
        batch: &PackedDecoderBatch,
        context: &crate::decoder::DecoderTransactionContext<'_>,
        layer: usize,
        attention: Option<
            std::rc::Rc<std::cell::RefCell<super::bf16_attention::PreparedAttention>>,
        >,
        run: impl FnOnce(&mut dyn KvView) -> Result<T>,
    ) -> Result<T> {
        if let Some(invocation) = context.invocation() {
            let append = invocation.cuda_host_kv()?;
            invocation.begin_layer(layer)?;
            let mut binding = append_gqa::CudaStandardAppendKvBinding::new(
                view,
                batch,
                append.layer(layer)?,
                append.budget(),
                append.block_tokens(),
            )?;
            let result = run(&mut binding)?;
            binding.ensure_retired()?;
            invocation.retire_layer(layer)?;
            Ok(result)
        } else {
            let mut binding = CudaStandardKvBinding::new(view, batch)?;
            binding.attention = attention;
            run(&mut binding)
        }
    }

    /// Explicit runtime-authorized route; default binding remains Drop-free.
    pub fn with_append_prefix(
        view: &'a mut CudaKvView,
        batch: &'a PackedDecoderBatch,
        authority: &'a dyn ferrule_backend::cuda::operators::kv::transaction::append_bridge::AppendPrefixAuthority,
        budget: &'a ferrule_common::host_kv::placement::PlacementBudget,
        block_tokens: usize,
    ) -> Result<append_gqa::CudaStandardAppendKvBinding<'a>> {
        append_gqa::CudaStandardAppendKvBinding::new(view, batch, authority, budget, block_tokens)
    }

    pub(super) fn with_prepared_attention(
        view: &'a mut CudaKvView,
        batch: &'a PackedDecoderBatch,
        attention: std::rc::Rc<std::cell::RefCell<super::bf16_attention::PreparedAttention>>,
    ) -> Result<Self> {
        let mut result = Self::new(view, batch)?;
        result.attention = Some(attention);
        Ok(result)
    }

    pub fn new(view: &'a mut CudaKvView, batch: &'a PackedDecoderBatch) -> Result<Self> {
        view.validate_batch(batch)?;
        Ok(Self {
            view,
            batch,
            attention: None,
        })
    }
}

impl StandardDecoderKvView for CudaStandardKvBinding<'_> {
    fn transaction(&self) -> ExecutionTransactionId {
        self.view.transaction()
    }
    fn validate_batch(&self, batch: &PackedDecoderBatch) -> Result<()> {
        if batch != self.batch {
            return Err(cuda_error(
                "CUDA computation binding received a foreign batch",
            ));
        }
        self.view.validate_batch(batch)
    }
}

impl KvView for CudaStandardKvBinding<'_> {
    fn append(&mut self, _request: KvAppendRequest<'_>) -> Result<()> {
        Err(cuda_error("CUDA binding does not expose host KV writes"))
    }
    fn history(
        &self,
        _layer: usize,
        _sequence: usize,
        _through_position: usize,
        _kv_heads: usize,
        _head_dim: usize,
    ) -> Result<KvHistory> {
        Err(cuda_error(
            "CUDA binding does not expose host attention history",
        ))
    }

    fn append_and_attend_cuda(
        &mut self,
        ops: &CudaOperators,
        request: GqaRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        if request.metadata.row_positions() != self.batch.positions()
            || request.metadata.row_sequence_ids() != self.batch.row_to_sequence()
            || request.metadata.sequence_count() != self.batch.sequences().len()
            || request
                .metadata
                .row_kv_lens()
                .iter()
                .zip(self.batch.positions())
                .any(|(&len, &p)| Some(len) != p.checked_add(1))
        {
            return Err(cuda_error(
                "attention metadata does not match the entered CUDA KV batch",
            ));
        }
        let query = request.query.cuda()?;
        let key = request.key.cuda()?;
        let value = request.value.cuda()?;
        for rows in [query, key, value] {
            rows.validate_owner(ops)?;
        }
        let q_width = request
            .query_heads
            .checked_mul(request.head_dim)
            .ok_or_else(|| cuda_error("query shape overflow"))?;
        let kv_width = request
            .kv_heads
            .checked_mul(request.head_dim)
            .ok_or_else(|| cuda_error("KV shape overflow"))?;
        if query.shape().rows() != self.batch.len()
            || query.shape().width() != q_width
            || key.shape().rows() != self.batch.len()
            || key.shape().width() != kv_width
            || value.shape() != key.shape()
        {
            return Err(cuda_error("GQA row shape mismatch"));
        }
        if query.dtype() == crate::transformer::RowsDType::Bf16 {
            let query_buffer = ops.bf16_to_f32(
                query
                    .bf16_buffer()
                    .ok_or_else(|| cuda_error("BF16 Q required"))?,
            )?;
            let key_buffer = key
                .bf16_buffer()
                .ok_or_else(|| cuda_error("BF16 K required"))?;
            let value_buffer = value
                .bf16_buffer()
                .ok_or_else(|| cuda_error("BF16 V required"))?;
            let mut output = ops.zero_f32_buffer(query.shape().elements())?;
            let transaction = self.view.transaction();
            self.view
                .with_bf16_gqa_planes(ops, request.layer, self.batch, |mut planes| {
                    let layout = planes.attention_layout(
                        request.query_heads,
                        request.kv_heads,
                        request.head_dim,
                        request.softmax_scale,
                    )?;
                    let rows = ferrule_backend::cuda::operators::kv::CudaBf16GqaRows {
                        query: &query_buffer,
                        append_key: key_buffer,
                        append_value: value_buffer,
                        output: &mut output,
                    };
                    if let Some(owner) = &self.attention {
                        let mut owner = owner.borrow_mut();
                        owner.validate_transaction(transaction)?;
                        let super::bf16_attention::PreparedAttention {
                            provider, status, ..
                        } = &mut *owner;
                        planes.append_and_attend_prepared(rows, layout, provider, status)
                    } else {
                        planes.append_and_attend_eager(rows, layout)
                    }
                })?;
            // Q/output are declared F32 scratch, never activation or cache storage.
            return super::bf16::round_rows(ops, query.shape(), request.arena, &output)
                .map(OperatorProgress::Ready);
        }
        let query_buffer = query
            .f32_buffer()
            .ok_or_else(|| cuda_error("query must be F32"))?;
        let key_buffer = key
            .f32_buffer()
            .ok_or_else(|| cuda_error("key must be F32"))?;
        let value_buffer = value
            .f32_buffer()
            .ok_or_else(|| cuda_error("value must be F32"))?;
        let sequences = checked_i32(self.batch.row_to_sequence().iter().copied())?;
        let positions = checked_i32(self.batch.positions().iter().copied())?;
        let committed = checked_i32(self.batch.sequences().iter().map(|s| s.context_len()))?;
        let mut output = ops.zero_f32_buffer(query.shape().elements())?;
        self.view.try_with_f32_gqa_planes_mut(
            ops,
            request.layer,
            self.batch,
            self.batch.protected_pages(),
            |planes| {
                // `request.layer` is the segment-local layer index produced by
                // execute_standard_layer. A PP stage never addresses its global
                // checkpoint layer directly in its local KV pool.
                if request.layer >= planes.key_layout.layer_count {
                    return Err(cuda_error(
                        "CUDA KV layer index is outside the segment-local schema",
                    ));
                }
                if planes.key_layout != planes.value_layout
                    || planes.key_layout.elements_per_token != kv_width
                {
                    return Err(cuda_error(
                        "CUDA KV planes do not match the attention descriptor",
                    ));
                }
                let slot_elements = planes
                    .key_layout
                    .checked_elements_per_page()
                    .ok_or_else(|| cuda_error("CUDA KV slot extent overflow"))?;
                if slot_elements == 0 || !planes.key.len().is_multiple_of(slot_elements) {
                    return Err(cuda_error("invalid CUDA KV plane extent"));
                }
                let layout = PagedF32GqaLayout {
                    rows: self.batch.len(),
                    sequences: self.batch.sequences().len(),
                    q_heads: request.query_heads,
                    kv_heads: request.kv_heads,
                    head_dim: request.head_dim,
                    page_tokens: planes.key_layout.page_tokens,
                    layer_index: request.layer,
                    layer_count: planes.key_layout.layer_count,
                    physical_slots: planes.key.len() / slot_elements,
                    softmax_scale: request.softmax_scale,
                };
                ops.append_and_attend_paged_f32(
                    PagedF32GqaBuffers {
                        query: query_buffer,
                        append_key: key_buffer,
                        append_value: value_buffer,
                        key_cache: planes.key,
                        value_cache: planes.value,
                        output: &mut output,
                    },
                    PagedF32GqaMetadata {
                        block_slots: planes.block_slots,
                        block_offsets: planes.block_offsets,
                        row_sequence_ids: &sequences,
                        row_positions: &positions,
                        committed_lengths: &committed,
                    },
                    layout,
                )
            },
        )?;
        ops.sync_stream()?;
        device_rows(query.shape(), request.arena, output).map(OperatorProgress::Ready)
    }
}

fn checked_i32(values: impl Iterator<Item = usize>) -> Result<Vec<i32>> {
    values
        .map(|v| i32::try_from(v).map_err(|_| cuda_error("attention metadata exceeds i32 ABI")))
        .collect()
}
