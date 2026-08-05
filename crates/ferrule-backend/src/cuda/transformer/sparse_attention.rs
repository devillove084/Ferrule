//! CUDA sparse-attention backend contracts.
//!
//! Selected latent attention over contiguous or paged KV storage with an
//! optional attention sink. Execution is delegated to the semantic CUTLASS
//! provider; unsupported shapes fail closed without a scalar fallback.

use crate::cuda::cutlass::{
    HybridMlaExplicitSelectionBuffers, HybridMlaExplicitSelectionLayout, HybridMlaKvStorageKind,
    hybrid_mla_explicit_selection_launch,
};
use crate::cuda::runtime::{CudaStream, DeviceBuffer};
use ferrule_common::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaSparseAttentionShape {
    pub batch_size: usize,
    pub tokens_per_batch: usize,
    pub kv_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    pub softmax_scale: f32,
}

impl CudaSparseAttentionShape {
    pub fn tokens(&self) -> usize {
        self.batch_size * self.tokens_per_batch
    }

    pub fn output_elements(&self) -> usize {
        self.tokens() * self.heads * self.head_dim
    }

    pub fn q_elements(&self) -> usize {
        self.output_elements()
    }

    pub fn kv_elements(&self) -> usize {
        self.batch_size * self.kv_len * self.head_dim
    }

    pub fn topk_elements(&self) -> usize {
        self.tokens() * self.topk
    }

    pub fn explicit_selection_layout(&self) -> HybridMlaExplicitSelectionLayout {
        HybridMlaExplicitSelectionLayout {
            kind: HybridMlaKvStorageKind::Contiguous,
            rows: self.tokens(),
            tokens_per_sequence: 0,
            kv_len: self.kv_len,
            heads: self.heads,
            head_dim: self.head_dim,
            selected_width: self.topk,
            page_tokens: 0,
            first_elements_per_token: 0,
            second_elements_per_token: 0,
            layer_index: 0,
            layer_count: 0,
            row_sequence_ids: false,
            row_kv_lens: false,
            softmax_scale: self.softmax_scale,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0
            || self.tokens_per_batch == 0
            || self.kv_len == 0
            || self.heads == 0
            || self.head_dim == 0
            || self.topk == 0
        {
            return Err(Error::Internal {
                message: format!(
                    "invalid sparse attention shape: batch={} tokens_per_batch={} kv_len={} heads={} head_dim={} topk={}",
                    self.batch_size,
                    self.tokens_per_batch,
                    self.kv_len,
                    self.heads,
                    self.head_dim,
                    self.topk
                ),
            });
        }
        if !self.softmax_scale.is_finite() || self.softmax_scale <= 0.0 {
            return Err(Error::Internal {
                message: format!(
                    "invalid sparse attention softmax_scale {}",
                    self.softmax_scale
                ),
            });
        }
        checked_u32(
            self.output_elements(),
            "sparse attention",
            "output elements",
        )?;
        checked_u32(
            self.tokens_per_batch,
            "sparse attention",
            "tokens_per_batch",
        )?;
        checked_u32(self.kv_len, "sparse attention", "kv_len")?;
        checked_u32(self.heads, "sparse attention", "heads")?;
        checked_u32(self.head_dim, "sparse attention", "head_dim")?;
        checked_u32(self.topk, "sparse attention", "topk")?;
        Ok(())
    }
}

/// Addressing and shape contract for sparse attention over a paged KV plane.
///
/// Query tokens are laid out as a fixed `tokens_per_sequence` for each batch
/// entry. KV lengths and packed block-table ranges are independently ragged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedSparseAttentionLayout {
    pub batch_size: usize,
    pub tokens_per_sequence: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    pub page_tokens: usize,
    pub elements_per_token: usize,
    pub layer_index: usize,
    pub layer_count: usize,
    pub softmax_scale: f32,
}

impl PagedSparseAttentionLayout {
    pub fn tokens(&self) -> Result<usize> {
        self.batch_size
            .checked_mul(self.tokens_per_sequence)
            .ok_or_else(|| Error::Internal {
                message: "paged sparse attention token count overflow".into(),
            })
    }

    pub fn output_elements(&self) -> Result<usize> {
        self.tokens()?
            .checked_mul(self.heads)
            .and_then(|elements| elements.checked_mul(self.head_dim))
            .ok_or_else(|| Error::Internal {
                message: "paged sparse attention output size overflow".into(),
            })
    }

    pub fn topk_elements(&self) -> Result<usize> {
        self.tokens()?
            .checked_mul(self.topk)
            .ok_or_else(|| Error::Internal {
                message: "paged sparse attention top-k size overflow".into(),
            })
    }

    pub fn explicit_selection_layout(
        &self,
        row_metadata: bool,
    ) -> Result<HybridMlaExplicitSelectionLayout> {
        Ok(HybridMlaExplicitSelectionLayout {
            kind: HybridMlaKvStorageKind::Paged,
            rows: self.tokens()?,
            tokens_per_sequence: if row_metadata {
                0
            } else {
                self.tokens_per_sequence
            },
            kv_len: 0,
            heads: self.heads,
            head_dim: self.head_dim,
            selected_width: self.topk,
            page_tokens: self.page_tokens,
            first_elements_per_token: self.elements_per_token,
            second_elements_per_token: 0,
            layer_index: self.layer_index,
            layer_count: self.layer_count,
            row_sequence_ids: row_metadata,
            row_kv_lens: row_metadata,
            softmax_scale: self.softmax_scale,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0
            || self.tokens_per_sequence == 0
            || self.heads == 0
            || self.head_dim == 0
            || self.topk == 0
            || self.page_tokens == 0
            || self.elements_per_token < self.head_dim
            || self.layer_count == 0
            || self.layer_index >= self.layer_count
        {
            return Err(Error::Internal {
                message: format!(
                    "invalid paged sparse attention layout: batch={} tokens_per_sequence={} heads={} head_dim={} topk={} page_tokens={} elements_per_token={} layer={}/{}",
                    self.batch_size,
                    self.tokens_per_sequence,
                    self.heads,
                    self.head_dim,
                    self.topk,
                    self.page_tokens,
                    self.elements_per_token,
                    self.layer_index,
                    self.layer_count
                ),
            });
        }
        if !self.softmax_scale.is_finite() || self.softmax_scale <= 0.0 {
            return Err(Error::Internal {
                message: format!(
                    "invalid paged sparse attention softmax_scale {}",
                    self.softmax_scale
                ),
            });
        }
        checked_u32(self.tokens()?, "paged sparse attention", "tokens")?;
        checked_u32(
            self.tokens()?
                .checked_mul(self.heads)
                .ok_or_else(|| Error::Internal {
                    message: "paged sparse attention pair count overflow".into(),
                })?,
            "paged sparse attention",
            "token/head pairs",
        )?;
        for (field, value) in [
            ("tokens_per_sequence", self.tokens_per_sequence),
            ("heads", self.heads),
            ("head_dim", self.head_dim),
            ("topk", self.topk),
            ("page_tokens", self.page_tokens),
            ("elements_per_token", self.elements_per_token),
            ("layer_index", self.layer_index),
            ("layer_count", self.layer_count),
        ] {
            checked_u32(value, "paged sparse attention", field)?;
        }
        self.output_elements()?;
        self.topk_elements()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_buffer_lengths(
        &self,
        q_elements: usize,
        plane_elements: usize,
        block_slots: usize,
        block_offsets: usize,
        kv_lens: usize,
        topk_elements: usize,
        sink_elements: usize,
        output_elements: usize,
    ) -> Result<()> {
        self.validate()?;
        let expected_output = self.output_elements()?;
        if q_elements != expected_output
            || block_offsets != self.batch_size + 1
            || kv_lens != self.batch_size
            || topk_elements != self.topk_elements()?
            || sink_elements != self.heads
            || output_elements != expected_output
        {
            return Err(Error::Internal {
                message: format!(
                    "paged sparse attention buffer length mismatch: q={q_elements}/{expected_output} plane={plane_elements} block_slots={block_slots} block_offsets={block_offsets}/{} kv_lens={kv_lens}/{} topk={topk_elements}/{} sink={sink_elements}/{} output={output_elements}/{expected_output}",
                    self.batch_size + 1,
                    self.batch_size,
                    self.topk_elements()?,
                    self.heads,
                ),
            });
        }
        if plane_elements == 0 || block_slots == 0 {
            return Err(Error::Internal {
                message: "paged sparse attention plane and block table must not be empty".into(),
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_selected_buffer_lengths(
        &self,
        q_elements: usize,
        plane_elements: usize,
        block_slots: usize,
        block_offsets: usize,
        sequence_kv_lens: usize,
        row_sequence_ids: usize,
        row_kv_lens: usize,
        topk_elements: usize,
        sink_elements: usize,
        output_elements: usize,
    ) -> Result<()> {
        self.validate()?;
        let rows = self.tokens()?;
        let expected_output = self.output_elements()?;
        if q_elements != expected_output
            || block_offsets < 2
            || sequence_kv_lens + 1 != block_offsets
            || row_sequence_ids != rows
            || row_kv_lens != rows
            || topk_elements != self.topk_elements()?
            || sink_elements != self.heads
            || output_elements != expected_output
            || plane_elements == 0
            || block_slots == 0
        {
            return Err(Error::Internal {
                message: format!(
                    "selected paged sparse attention buffer length mismatch: rows={rows} q={q_elements}/{expected_output} plane={plane_elements} block_slots={block_slots} block_offsets={block_offsets} sequence_kv_lens={sequence_kv_lens} row_sequence_ids={row_sequence_ids} row_kv_lens={row_kv_lens} topk={topk_elements}/{} sink={sink_elements}/{} output={output_elements}/{expected_output}",
                    self.topk_elements()?,
                    self.heads,
                ),
            });
        }
        Ok(())
    }

    pub fn resolve_selected_kv_offset(
        &self,
        row: usize,
        logical_token: i32,
        plane_elements: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
        sequence_kv_lens: &[i32],
        row_sequence_ids: &[i32],
        row_kv_lens: &[i32],
    ) -> Option<usize> {
        let sequence = usize::try_from(*row_sequence_ids.get(row)?).ok()?;
        let visible_len = *row_kv_lens.get(row)?;
        if logical_token < 0 || visible_len < 0 || logical_token >= visible_len {
            return None;
        }
        self.resolve_kv_offset(
            sequence,
            logical_token,
            plane_elements,
            block_slots,
            block_offsets,
            sequence_kv_lens,
        )
    }

    pub fn validate_metadata(
        &self,
        plane_elements: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
        kv_lens: &[i32],
    ) -> Result<()> {
        self.validate_buffer_lengths(
            self.output_elements()?,
            plane_elements,
            block_slots.len(),
            block_offsets.len(),
            kv_lens.len(),
            self.topk_elements()?,
            self.heads,
            self.output_elements()?,
        )?;
        if block_offsets.first().copied() != Some(0) {
            return Err(Error::Internal {
                message: "paged sparse attention block offsets must start at zero".into(),
            });
        }
        if nonnegative_usize(
            *block_offsets.last().expect("validated length"),
            "block offset",
        )? != block_slots.len()
        {
            return Err(Error::Internal {
                message:
                    "paged sparse attention final block offset must equal packed block slot count"
                        .into(),
            });
        }
        for sequence in 0..self.batch_size {
            let start = nonnegative_usize(block_offsets[sequence], "block offset")?;
            let end = nonnegative_usize(block_offsets[sequence + 1], "block offset")?;
            let kv_len = nonnegative_usize(kv_lens[sequence], "KV length")?;
            if start > end || end > block_slots.len() {
                return Err(Error::Internal {
                    message: format!(
                        "paged sparse attention invalid block range for sequence {sequence}: {start}..{end} of {}",
                        block_slots.len()
                    ),
                });
            }
            let required_pages = kv_len.div_ceil(self.page_tokens);
            if required_pages > end - start {
                return Err(Error::Internal {
                    message: format!(
                        "paged sparse attention sequence {sequence} needs {required_pages} pages but block range has {}",
                        end - start
                    ),
                });
            }
            for slot in &block_slots[start..end] {
                if *slot < 0 {
                    continue;
                }
                let slot = *slot as usize;
                let end = slot
                    .checked_add(1)
                    .and_then(|slots| slots.checked_mul(self.layer_count))
                    .and_then(|layers| layers.checked_mul(self.page_tokens))
                    .and_then(|tokens| tokens.checked_mul(self.elements_per_token))
                    .ok_or_else(|| Error::Internal {
                        message: "paged sparse attention slot range overflow".into(),
                    })?;
                if end > plane_elements {
                    return Err(Error::Internal {
                        message: format!(
                            "paged sparse attention physical slot {slot} exceeds plane storage {plane_elements}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn resolve_kv_offset(
        &self,
        sequence: usize,
        logical_token: i32,
        plane_elements: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
        kv_lens: &[i32],
    ) -> Option<usize> {
        if sequence >= self.batch_size || logical_token < 0 {
            return None;
        }
        let logical_token = logical_token as usize;
        let kv_len = usize::try_from(*kv_lens.get(sequence)?).ok()?;
        if logical_token >= kv_len {
            return None;
        }
        let start = usize::try_from(*block_offsets.get(sequence)?).ok()?;
        let end = usize::try_from(*block_offsets.get(sequence + 1)?).ok()?;
        let entry = start.checked_add(logical_token / self.page_tokens)?;
        if entry >= end {
            return None;
        }
        let physical_slot = usize::try_from(*block_slots.get(entry)?).ok()?;
        let slot_stride = self
            .layer_count
            .checked_mul(self.page_tokens)?
            .checked_mul(self.elements_per_token)?;
        let layer_stride = self.page_tokens.checked_mul(self.elements_per_token)?;
        let offset = physical_slot
            .checked_mul(slot_stride)?
            .checked_add(self.layer_index.checked_mul(layer_stride)?)?
            .checked_add(
                (logical_token % self.page_tokens).checked_mul(self.elements_per_token)?,
            )?;
        (offset.checked_add(self.head_dim)? <= plane_elements).then_some(offset)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualPlanePagedSparseAttentionLayout {
    pub base: PagedSparseAttentionLayout,
    pub second_elements_per_token: usize,
}

impl DualPlanePagedSparseAttentionLayout {
    fn second_layout(&self) -> PagedSparseAttentionLayout {
        PagedSparseAttentionLayout {
            elements_per_token: self.second_elements_per_token,
            ..self.base
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.base.validate()?;
        self.second_layout().validate()
    }

    pub fn explicit_selection_layout(
        &self,
        row_metadata: bool,
    ) -> Result<HybridMlaExplicitSelectionLayout> {
        Ok(HybridMlaExplicitSelectionLayout {
            kind: HybridMlaKvStorageKind::DualPaged,
            second_elements_per_token: self.second_elements_per_token,
            ..self.base.explicit_selection_layout(row_metadata)?
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_buffer_lengths(
        &self,
        q_elements: usize,
        first_plane_elements: usize,
        second_plane_elements: usize,
        block_slots: usize,
        block_offsets: usize,
        kv_lens: usize,
        topk_elements: usize,
        selector_elements: usize,
        sink_elements: usize,
        output_elements: usize,
    ) -> Result<()> {
        self.base.validate_buffer_lengths(
            q_elements,
            first_plane_elements,
            block_slots,
            block_offsets,
            kv_lens,
            topk_elements,
            sink_elements,
            output_elements,
        )?;
        if second_plane_elements == 0 || selector_elements != self.base.topk_elements()? {
            return Err(Error::Internal {
                message: format!(
                    "dual-plane paged sparse attention length mismatch: second_plane={second_plane_elements} selectors={selector_elements}/{}",
                    self.base.topk_elements()?
                ),
            });
        }
        self.second_layout().validate()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_selected_buffer_lengths(
        &self,
        q_elements: usize,
        first_plane_elements: usize,
        second_plane_elements: usize,
        block_slots: usize,
        block_offsets: usize,
        sequence_kv_lens: usize,
        row_sequence_ids: usize,
        row_kv_lens: usize,
        topk_elements: usize,
        selector_elements: usize,
        sink_elements: usize,
        output_elements: usize,
    ) -> Result<()> {
        self.base.validate_selected_buffer_lengths(
            q_elements,
            first_plane_elements,
            block_slots,
            block_offsets,
            sequence_kv_lens,
            row_sequence_ids,
            row_kv_lens,
            topk_elements,
            sink_elements,
            output_elements,
        )?;
        if second_plane_elements == 0 || selector_elements != self.base.topk_elements()? {
            return Err(Error::Internal {
                message: format!(
                    "selected dual-plane paged sparse attention length mismatch: second_plane={second_plane_elements} selectors={selector_elements}/{}",
                    self.base.topk_elements()?
                ),
            });
        }
        self.second_layout().validate()
    }

    pub fn validate_metadata(
        &self,
        first_plane_elements: usize,
        second_plane_elements: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
        kv_lens: &[i32],
    ) -> Result<()> {
        self.base
            .validate_metadata(first_plane_elements, block_slots, block_offsets, kv_lens)?;
        self.second_layout().validate_metadata(
            second_plane_elements,
            block_slots,
            block_offsets,
            kv_lens,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resolve_kv_offset(
        &self,
        selector: i32,
        sequence: usize,
        logical_token: i32,
        first_plane_elements: usize,
        second_plane_elements: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
        kv_lens: &[i32],
    ) -> Option<(usize, usize)> {
        match selector {
            0 => self
                .base
                .resolve_kv_offset(
                    sequence,
                    logical_token,
                    first_plane_elements,
                    block_slots,
                    block_offsets,
                    kv_lens,
                )
                .map(|offset| (0, offset)),
            1 => self
                .second_layout()
                .resolve_kv_offset(
                    sequence,
                    logical_token,
                    second_plane_elements,
                    block_slots,
                    block_offsets,
                    kv_lens,
                )
                .map(|offset| (1, offset)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct CudaSparseAttentionExecutor<'a> {
    pub stream: &'a CudaStream,
}

impl<'a> CudaSparseAttentionExecutor<'a> {
    pub fn new(stream: &'a CudaStream) -> Self {
        Self { stream }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dual_plane_paged_sparse_attention_sink_f32(
        &self,
        q: &DeviceBuffer<f32>,
        first_plane: &DeviceBuffer<f32>,
        second_plane: &DeviceBuffer<f32>,
        block_slots: &DeviceBuffer<i32>,
        sequence_block_offsets: &DeviceBuffer<i32>,
        sequence_kv_lens: &DeviceBuffer<i32>,
        row_metadata: Option<(&DeviceBuffer<i32>, &DeviceBuffer<i32>)>,
        topk: &DeviceBuffer<i32>,
        selectors: &DeviceBuffer<i32>,
        sink: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        workspace: &mut DeviceBuffer<u8>,
        status: &mut DeviceBuffer<i32>,
        #[cfg(ferrule_cuda_test_oracle)] oracle_output: &mut DeviceBuffer<f32>,
        layout: DualPlanePagedSparseAttentionLayout,
    ) -> Result<()> {
        if let Some((row_sequence_ids, row_kv_lens)) = row_metadata {
            layout.validate_selected_buffer_lengths(
                q.len(),
                first_plane.len(),
                second_plane.len(),
                block_slots.len(),
                sequence_block_offsets.len(),
                sequence_kv_lens.len(),
                row_sequence_ids.len(),
                row_kv_lens.len(),
                topk.len(),
                selectors.len(),
                sink.len(),
                output.len(),
            )?;
        } else {
            layout.validate_buffer_lengths(
                q.len(),
                first_plane.len(),
                second_plane.len(),
                block_slots.len(),
                sequence_block_offsets.len(),
                sequence_kv_lens.len(),
                topk.len(),
                selectors.len(),
                sink.len(),
                output.len(),
            )?;
        }

        let (row_sequence_ids, row_kv_lens) = match row_metadata {
            Some((ids, lens)) => (Some(ids), Some(lens)),
            None => (None, None),
        };
        let cutlass_layout = layout.explicit_selection_layout(row_metadata.is_some())?;
        let mut buffers = HybridMlaExplicitSelectionBuffers {
            query: q,
            #[cfg(ferrule_cuda_test_oracle)]
            oracle_output,
            first_plane,
            second_plane: Some(second_plane),
            block_slots: Some(block_slots),
            block_offsets: Some(sequence_block_offsets),
            sequence_kv_lens: Some(sequence_kv_lens),
            row_sequence_ids,
            row_kv_lens,
            selected_indices: topk,
            selectors: Some(selectors),
            attention_sink: sink,
            workspace,
            output,
            status,
        };
        hybrid_mla_explicit_selection_launch(self.stream, &mut buffers, cutlass_layout)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_sparse_attention_sink_f32(
        &self,
        q: &DeviceBuffer<f32>,
        plane: &DeviceBuffer<f32>,
        block_slots: &DeviceBuffer<i32>,
        sequence_block_offsets: &DeviceBuffer<i32>,
        sequence_kv_lens: &DeviceBuffer<i32>,
        row_metadata: Option<(&DeviceBuffer<i32>, &DeviceBuffer<i32>)>,
        topk: &DeviceBuffer<i32>,
        sink: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        workspace: &mut DeviceBuffer<u8>,
        status: &mut DeviceBuffer<i32>,
        #[cfg(ferrule_cuda_test_oracle)] oracle_output: &mut DeviceBuffer<f32>,
        layout: PagedSparseAttentionLayout,
    ) -> Result<()> {
        if let Some((row_sequence_ids, row_kv_lens)) = row_metadata {
            layout.validate_selected_buffer_lengths(
                q.len(),
                plane.len(),
                block_slots.len(),
                sequence_block_offsets.len(),
                sequence_kv_lens.len(),
                row_sequence_ids.len(),
                row_kv_lens.len(),
                topk.len(),
                sink.len(),
                output.len(),
            )?;
        } else {
            layout.validate_buffer_lengths(
                q.len(),
                plane.len(),
                block_slots.len(),
                sequence_block_offsets.len(),
                sequence_kv_lens.len(),
                topk.len(),
                sink.len(),
                output.len(),
            )?;
        }

        let (row_sequence_ids, row_kv_lens) = match row_metadata {
            Some((ids, lens)) => (Some(ids), Some(lens)),
            None => (None, None),
        };
        let cutlass_layout = layout.explicit_selection_layout(row_metadata.is_some())?;
        let mut buffers = HybridMlaExplicitSelectionBuffers {
            query: q,
            #[cfg(ferrule_cuda_test_oracle)]
            oracle_output,
            first_plane: plane,
            second_plane: None,
            block_slots: Some(block_slots),
            block_offsets: Some(sequence_block_offsets),
            sequence_kv_lens: Some(sequence_kv_lens),
            row_sequence_ids,
            row_kv_lens,
            selected_indices: topk,
            selectors: None,
            attention_sink: sink,
            workspace,
            output,
            status,
        };
        hybrid_mla_explicit_selection_launch(self.stream, &mut buffers, cutlass_layout)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attention_sink_f32(
        &self,
        q: &DeviceBuffer<f32>,
        kv: &DeviceBuffer<f32>,
        topk: &DeviceBuffer<i32>,
        sink: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        workspace: &mut DeviceBuffer<u8>,
        status: &mut DeviceBuffer<i32>,
        #[cfg(ferrule_cuda_test_oracle)] oracle_output: &mut DeviceBuffer<f32>,
        shape: CudaSparseAttentionShape,
    ) -> Result<()> {
        shape.validate()?;
        if shape.batch_size != 1 {
            return Err(Error::Internal {
                message: format!(
                    "CUTLASS contiguous sparse attention requires batch_size=1, got {}",
                    shape.batch_size
                ),
            });
        }

        let cutlass_layout = shape.explicit_selection_layout();
        let mut buffers = HybridMlaExplicitSelectionBuffers {
            query: q,
            #[cfg(ferrule_cuda_test_oracle)]
            oracle_output,
            first_plane: kv,
            second_plane: None,
            block_slots: None,
            block_offsets: None,
            sequence_kv_lens: None,
            row_sequence_ids: None,
            row_kv_lens: None,
            selected_indices: topk,
            selectors: None,
            attention_sink: sink,
            workspace,
            output,
            status,
        };
        hybrid_mla_explicit_selection_launch(self.stream, &mut buffers, cutlass_layout)
    }
}

fn nonnegative_usize(value: i32, field: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::Internal {
        message: format!("paged sparse attention {field} must be nonnegative, got {value}"),
    })
}

fn checked_u32(value: usize, label: &str, field: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Internal {
        message: format!("{label} {field} exceeds CUDA u32 launch ABI: {value}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paged_layout() -> PagedSparseAttentionLayout {
        PagedSparseAttentionLayout {
            batch_size: 2,
            tokens_per_sequence: 1,
            heads: 1,
            head_dim: 4,
            topk: 3,
            page_tokens: 2,
            elements_per_token: 4,
            layer_index: 1,
            layer_count: 3,
            softmax_scale: 0.5,
        }
    }

    #[test]
    fn paged_address_mapping_handles_boundaries_layers_and_invalid_slots() {
        let layout = paged_layout();
        let block_slots = [2, 5, -1];
        let block_offsets = [0, 2, 3];
        let kv_lens = [4, 1];
        layout
            .validate_metadata(144, &block_slots, &block_offsets, &kv_lens)
            .unwrap();

        assert_eq!(
            layout.resolve_kv_offset(0, 0, 144, &block_slots, &block_offsets, &kv_lens),
            Some(56)
        );
        assert_eq!(
            layout.resolve_kv_offset(0, 1, 144, &block_slots, &block_offsets, &kv_lens),
            Some(60)
        );
        assert_eq!(
            layout.resolve_kv_offset(0, 2, 144, &block_slots, &block_offsets, &kv_lens),
            Some(128)
        );
        assert_eq!(
            layout.resolve_kv_offset(0, -1, 144, &block_slots, &block_offsets, &kv_lens),
            None
        );
        assert_eq!(
            layout.resolve_kv_offset(0, 4, 144, &block_slots, &block_offsets, &kv_lens),
            None
        );
        assert_eq!(
            layout.resolve_kv_offset(1, 0, 144, &block_slots, &block_offsets, &kv_lens),
            None
        );
    }

    #[test]
    fn selected_row_mapping_reorders_sequences_and_enforces_row_visibility() {
        let layout = PagedSparseAttentionLayout {
            batch_size: 3,
            tokens_per_sequence: 1,
            ..paged_layout()
        };
        let block_slots = [2, 5, 1];
        let block_offsets = [0, 2, 3];
        let sequence_kv_lens = [4, 2];
        let row_sequence_ids = [1, 0, 0];
        let row_kv_lens = [2, 2, 4];

        assert_eq!(
            layout.resolve_selected_kv_offset(
                0,
                1,
                144,
                &block_slots,
                &block_offsets,
                &sequence_kv_lens,
                &row_sequence_ids,
                &row_kv_lens,
            ),
            Some(36)
        );
        assert_eq!(
            layout.resolve_selected_kv_offset(
                1,
                2,
                144,
                &block_slots,
                &block_offsets,
                &sequence_kv_lens,
                &row_sequence_ids,
                &row_kv_lens,
            ),
            None
        );
        assert_eq!(
            layout.resolve_selected_kv_offset(
                2,
                2,
                144,
                &block_slots,
                &block_offsets,
                &sequence_kv_lens,
                &row_sequence_ids,
                &row_kv_lens,
            ),
            Some(128)
        );
    }

    #[test]
    fn dual_plane_mapping_uses_selector_specific_stride() {
        let layout = DualPlanePagedSparseAttentionLayout {
            base: paged_layout(),
            second_elements_per_token: 8,
        };
        let block_slots = [2, 5, -1];
        let block_offsets = [0, 2, 3];
        let kv_lens = [4, 1];
        layout
            .validate_metadata(144, 288, &block_slots, &block_offsets, &kv_lens)
            .unwrap();
        assert_eq!(
            layout.resolve_kv_offset(0, 0, 0, 144, 288, &block_slots, &block_offsets, &kv_lens),
            Some((0, 56))
        );
        assert_eq!(
            layout.resolve_kv_offset(1, 0, 0, 144, 288, &block_slots, &block_offsets, &kv_lens),
            Some((1, 112))
        );
        assert_eq!(
            layout.resolve_kv_offset(-1, 0, 0, 144, 288, &block_slots, &block_offsets, &kv_lens),
            None
        );
        assert_eq!(
            layout.resolve_kv_offset(2, 0, 0, 144, 288, &block_slots, &block_offsets, &kv_lens),
            None
        );
    }

    #[test]
    fn paged_shape_validation_rejects_short_or_out_of_range_tables() {
        let layout = paged_layout();
        assert!(
            layout
                .validate_metadata(144, &[0], &[0, 1, 1], &[4, 0])
                .is_err()
        );
        assert!(
            layout
                .validate_metadata(32, &[2, 5, -1], &[0, 2, 3], &[4, 1])
                .is_err()
        );
        let invalid_layer = PagedSparseAttentionLayout {
            layer_index: 3,
            ..layout
        };
        assert!(invalid_layer.validate().is_err());
    }

    #[test]
    fn sparse_attention_shape_validates_large_decode() {
        let shape = CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: 1,
            kv_len: 128 + 1024,
            heads: 64,
            head_dim: 512,
            topk: 128 + 512,
            softmax_scale: 512.0f32.powf(-0.5),
        };
        shape.validate().unwrap();
        assert_eq!(shape.output_elements(), 64 * 512);
    }

    #[test]
    fn sparse_attention_shape_rejects_zero_topk() {
        let shape = CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: 1,
            kv_len: 1,
            heads: 1,
            head_dim: 1,
            topk: 0,
            softmax_scale: 1.0,
        };
        assert!(
            shape
                .validate()
                .unwrap_err()
                .to_string()
                .contains("invalid sparse attention")
        );
    }
}
