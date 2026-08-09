//! Provider-neutral CUDA semantic operator façade.
//!
//! Backend execution code dispatches operations through this module. The
//! concrete provider implementation remains private so callers do not depend on
//! its implementation technology.

pub use crate::cuda::providers::cutlass::{
    GroupedFp4MoeBuffers, GroupedFp4MoeLayout, HYBRID_MLA_ATTENTION_HEAD_DIM,
    HYBRID_MLA_ATTENTION_HEADS, HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILE,
    HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILES, HYBRID_MLA_ATTENTION_PAGE_TOKENS,
    HYBRID_MLA_ATTENTION_TOKEN_CAPACITY, HYBRID_MLA_ATTENTION_WINDOW,
    HYBRID_MLA_EXPLICIT_SELECTION_MAXIMUM_WIDTH, HybridMlaAttentionLayout,
    HybridMlaExplicitSelectionBuffers, HybridMlaExplicitSelectionLayout, HybridMlaKvStorageKind,
    PROPOSAL_ROWS, ProposalHeadLayout, bf16_compressor, fp8_projection, fp8_query_a_kv,
    grouped_fp4_moe_can_implement, grouped_fp4_moe_launch, grouped_fp4_moe_workspace_size,
    hc_producer, hybrid_mla_attention, hybrid_mla_explicit_selection_can_implement,
    hybrid_mla_explicit_selection_launch, hybrid_mla_explicit_selection_workspace_requirements,
    main_project_norm, mla_output, mxfp4_sfb_storage_bytes, prepare_mxfp4_sfb, proposal_head,
    shared_ffn,
};

#[cfg(ferrule_cuda_test_oracle)]
pub use crate::cuda::providers::cutlass::HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE_RESULT_WORDS;

use ferrule_common::{Error, Result};

/// Provider-neutral workspace size and alignment requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatorWorkspaceRequirements {
    pub bytes: u64,
    pub alignment: u32,
}

/// Native status indicating that row or page-table metadata is invalid.
pub const PAGED_BF16_TRANSFORMER_METADATA_ERROR: i32 = 1;
/// Native status indicating that a computed K/V byte address exceeds capacity.
pub const PAGED_BF16_TRANSFORMER_ADDRESS_ERROR: i32 = 2;

/// Interpret the device status produced by a paged BF16 transformer operation.
pub fn validate_paged_bf16_transformer_status(status: i32) -> Result<()> {
    match status {
        0 => Ok(()),
        PAGED_BF16_TRANSFORMER_METADATA_ERROR => Err(Error::Execution {
            message: "paged BF16 transformer rejected row or page-table metadata".into(),
        }),
        PAGED_BF16_TRANSFORMER_ADDRESS_ERROR => Err(Error::Execution {
            message: "paged BF16 transformer computed a cache address outside its plane".into(),
        }),
        status => Err(Error::Execution {
            message: format!("paged BF16 transformer returned unknown device status {status}"),
        }),
    }
}

/// Dense `[rows, heads, dimensions]` F32 row layout for core operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StridedF32RowsLayout {
    pub rows: usize,
    pub heads: usize,
    pub dimensions: usize,
    pub row_stride_bytes: usize,
    pub head_stride_bytes: usize,
}

impl StridedF32RowsLayout {
    pub fn packed(rows: usize, heads: usize, dimensions: usize) -> Result<Self> {
        let head_stride_bytes = checked_bytes(dimensions, size_of::<f32>(), "F32 head")?;
        let row_stride_bytes = checked_bytes(heads, head_stride_bytes, "F32 row")?;
        let layout = Self {
            rows,
            heads,
            dimensions,
            row_stride_bytes,
            head_stride_bytes,
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn required_bytes(self) -> Result<usize> {
        required_strided_bytes(
            self.rows,
            self.row_stride_bytes,
            self.heads,
            self.head_stride_bytes,
            checked_bytes(self.dimensions, size_of::<f32>(), "F32 head")?,
            size_of::<f32>(),
            "F32 rows",
        )
    }

    pub fn validate(self) -> Result<()> {
        validate_strided_rows(
            self.rows,
            self.heads,
            checked_bytes(self.dimensions, size_of::<f32>(), "F32 head")?,
            self.row_stride_bytes,
            self.head_stride_bytes,
            size_of::<f32>(),
            "F32 rows",
        )
    }
}

/// Dense `[rows, heads, dimensions]` BF16 row layout for core operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StridedBf16RowsLayout {
    pub rows: usize,
    pub heads: usize,
    pub dimensions: usize,
    pub row_stride_bytes: usize,
    pub head_stride_bytes: usize,
}

impl StridedBf16RowsLayout {
    pub fn packed(rows: usize, heads: usize, dimensions: usize) -> Result<Self> {
        let head_stride_bytes = checked_bytes(dimensions, size_of::<u16>(), "BF16 head")?;
        let row_stride_bytes = checked_bytes(heads, head_stride_bytes, "BF16 row")?;
        let layout = Self {
            rows,
            heads,
            dimensions,
            row_stride_bytes,
            head_stride_bytes,
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn required_bytes(self) -> Result<usize> {
        required_strided_bytes(
            self.rows,
            self.row_stride_bytes,
            self.heads,
            self.head_stride_bytes,
            checked_bytes(self.dimensions, size_of::<u16>(), "BF16 head")?,
            size_of::<u16>(),
            "BF16 rows",
        )
    }

    pub fn validate(self) -> Result<()> {
        validate_strided_rows(
            self.rows,
            self.heads,
            checked_bytes(self.dimensions, size_of::<u16>(), "BF16 head")?,
            self.row_stride_bytes,
            self.head_stride_bytes,
            size_of::<u16>(),
            "BF16 rows",
        )
    }
}

/// Provider-neutral F32-to-BF16 RNE conversion over one logical row tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32ToBf16RowsLayout {
    pub input: StridedF32RowsLayout,
    pub output: StridedBf16RowsLayout,
}

impl F32ToBf16RowsLayout {
    pub fn packed(rows: usize, heads: usize, dimensions: usize) -> Result<Self> {
        let layout = Self {
            input: StridedF32RowsLayout::packed(rows, heads, dimensions)?,
            output: StridedBf16RowsLayout::packed(rows, heads, dimensions)?,
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn validate(self) -> Result<()> {
        self.input.validate()?;
        self.output.validate()?;
        if self.input.rows != self.output.rows
            || self.input.heads != self.output.heads
            || self.input.dimensions != self.output.dimensions
        {
            return Err(Error::Internal {
                message: format!(
                    "F32-to-BF16 row layouts disagree: input={:?} output={:?}",
                    self.input, self.output
                ),
            });
        }
        Ok(())
    }

    pub fn element_count(self) -> Result<usize> {
        self.validate()?;
        checked_product(
            &[self.input.rows, self.input.heads, self.input.dimensions],
            "F32-to-BF16 rows",
        )
    }
}

/// One independently allocated paged BF16 K or V cache plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedBf16PlaneLayout {
    pub capacity_bytes: usize,
    pub slot_stride_bytes: usize,
    pub layer_stride_bytes: usize,
    pub token_stride_bytes: usize,
    pub head_stride_bytes: usize,
}

impl PagedBf16PlaneLayout {
    pub fn packed(
        physical_slots: usize,
        layer_count: usize,
        page_tokens: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let head_stride_bytes = checked_bytes(head_dim, size_of::<u16>(), "paged BF16 head")?;
        let token_stride_bytes = checked_bytes(kv_heads, head_stride_bytes, "paged BF16 token")?;
        let layer_stride_bytes =
            checked_bytes(page_tokens, token_stride_bytes, "paged BF16 layer")?;
        let slot_stride_bytes = checked_bytes(layer_count, layer_stride_bytes, "paged BF16 slot")?;
        let capacity_bytes = checked_bytes(physical_slots, slot_stride_bytes, "paged BF16 plane")?;
        Ok(Self {
            capacity_bytes,
            slot_stride_bytes,
            layer_stride_bytes,
            token_stride_bytes,
            head_stride_bytes,
        })
    }

    fn validate(
        self,
        layer_count: usize,
        page_tokens: usize,
        kv_heads: usize,
        head_dim: usize,
        label: &str,
    ) -> Result<()> {
        let head_bytes = checked_bytes(head_dim, size_of::<u16>(), label)?;
        let token_bytes = (kv_heads.saturating_sub(1))
            .checked_mul(self.head_stride_bytes)
            .and_then(|bytes| bytes.checked_add(head_bytes))
            .ok_or_else(|| Error::Internal {
                message: format!("{label} token extent overflow"),
            })?;
        let layer_bytes = (page_tokens.saturating_sub(1))
            .checked_mul(self.token_stride_bytes)
            .and_then(|bytes| bytes.checked_add(token_bytes))
            .ok_or_else(|| Error::Internal {
                message: format!("{label} layer extent overflow"),
            })?;
        let slot_bytes = (layer_count.saturating_sub(1))
            .checked_mul(self.layer_stride_bytes)
            .and_then(|bytes| bytes.checked_add(layer_bytes))
            .ok_or_else(|| Error::Internal {
                message: format!("{label} slot extent overflow"),
            })?;
        if self.capacity_bytes < self.slot_stride_bytes
            || self.head_stride_bytes < head_bytes
            || self.token_stride_bytes < token_bytes
            || self.layer_stride_bytes < layer_bytes
            || self.slot_stride_bytes < slot_bytes
            || !self.capacity_bytes.is_multiple_of(size_of::<u16>())
            || !self.slot_stride_bytes.is_multiple_of(size_of::<u16>())
            || !self.layer_stride_bytes.is_multiple_of(size_of::<u16>())
            || !self.token_stride_bytes.is_multiple_of(size_of::<u16>())
            || !self.head_stride_bytes.is_multiple_of(size_of::<u16>())
        {
            return Err(Error::Internal {
                message: format!(
                    "invalid {label} byte strides: capacity={} slot={} layer={} token={} head={} required_head={head_bytes}",
                    self.capacity_bytes,
                    self.slot_stride_bytes,
                    self.layer_stride_bytes,
                    self.token_stride_bytes,
                    self.head_stride_bytes,
                ),
            });
        }
        Ok(())
    }

    pub fn byte_offset(
        self,
        physical_slot: usize,
        layer: usize,
        token_in_page: usize,
        head: usize,
        dimension: usize,
    ) -> Option<usize> {
        let offset = physical_slot
            .checked_mul(self.slot_stride_bytes)?
            .checked_add(layer.checked_mul(self.layer_stride_bytes)?)?
            .checked_add(token_in_page.checked_mul(self.token_stride_bytes)?)?
            .checked_add(head.checked_mul(self.head_stride_bytes)?)?
            .checked_add(dimension.checked_mul(size_of::<u16>())?)?;
        (offset.checked_add(size_of::<u16>())? <= self.capacity_bytes).then_some(offset)
    }
}

/// Provider-neutral standard causal GQA and paged BF16 K/V cache layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedBf16CausalGqaLayout {
    pub rows: usize,
    pub sequences: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub page_tokens: usize,
    pub layer_index: usize,
    pub layer_count: usize,
    pub softmax_scale: f32,
    pub query: StridedF32RowsLayout,
    pub append_key: StridedBf16RowsLayout,
    pub append_value: StridedBf16RowsLayout,
    pub key_cache: PagedBf16PlaneLayout,
    pub value_cache: PagedBf16PlaneLayout,
    pub output: StridedF32RowsLayout,
}

impl PagedBf16CausalGqaLayout {
    pub fn packed(
        rows: usize,
        sequences: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        page_tokens: usize,
        layer_index: usize,
        layer_count: usize,
        physical_slots: usize,
        softmax_scale: f32,
    ) -> Result<Self> {
        let layout = Self {
            rows,
            sequences,
            q_heads,
            kv_heads,
            head_dim,
            page_tokens,
            layer_index,
            layer_count,
            softmax_scale,
            query: StridedF32RowsLayout::packed(rows, q_heads, head_dim)?,
            append_key: StridedBf16RowsLayout::packed(rows, kv_heads, head_dim)?,
            append_value: StridedBf16RowsLayout::packed(rows, kv_heads, head_dim)?,
            key_cache: PagedBf16PlaneLayout::packed(
                physical_slots,
                layer_count,
                page_tokens,
                kv_heads,
                head_dim,
            )?,
            value_cache: PagedBf16PlaneLayout::packed(
                physical_slots,
                layer_count,
                page_tokens,
                kv_heads,
                head_dim,
            )?,
            output: StridedF32RowsLayout::packed(rows, q_heads, head_dim)?,
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn validate(self) -> Result<()> {
        if self.rows == 0
            || self.sequences == 0
            || self.q_heads == 0
            || self.kv_heads == 0
            || !self.q_heads.is_multiple_of(self.kv_heads)
            || self.head_dim == 0
            || self.page_tokens == 0
            || self.layer_count == 0
            || self.layer_index >= self.layer_count
            || !self.softmax_scale.is_finite()
            || self.softmax_scale <= 0.0
        {
            return Err(Error::Internal {
                message: format!(
                    "invalid paged BF16 causal GQA layout: rows={} sequences={} q_heads={} kv_heads={} head_dim={} page_tokens={} layer={}/{} scale={}",
                    self.rows,
                    self.sequences,
                    self.q_heads,
                    self.kv_heads,
                    self.head_dim,
                    self.page_tokens,
                    self.layer_index,
                    self.layer_count,
                    self.softmax_scale,
                ),
            });
        }
        if self.query.rows != self.rows
            || self.query.heads != self.q_heads
            || self.query.dimensions != self.head_dim
            || self.output.rows != self.rows
            || self.output.heads != self.q_heads
            || self.output.dimensions != self.head_dim
            || self.append_key.rows != self.rows
            || self.append_key.heads != self.kv_heads
            || self.append_key.dimensions != self.head_dim
            || self.append_value.rows != self.rows
            || self.append_value.heads != self.kv_heads
            || self.append_value.dimensions != self.head_dim
        {
            return Err(Error::Internal {
                message: "paged BF16 causal GQA row layouts disagree with the logical shape".into(),
            });
        }
        self.query.validate()?;
        self.output.validate()?;
        self.append_key.validate()?;
        self.append_value.validate()?;
        self.key_cache.validate(
            self.layer_count,
            self.page_tokens,
            self.kv_heads,
            self.head_dim,
            "key cache",
        )?;
        self.value_cache.validate(
            self.layer_count,
            self.page_tokens,
            self.kv_heads,
            self.head_dim,
            "value cache",
        )?;
        Ok(())
    }

    pub fn validate_metadata_lengths(
        self,
        block_slots: usize,
        block_offsets: usize,
        row_sequence_ids: usize,
        row_positions: usize,
        row_kv_lens: usize,
    ) -> Result<()> {
        self.validate()?;
        if block_slots == 0
            || block_offsets != self.sequences + 1
            || row_sequence_ids != self.rows
            || row_positions != self.rows
            || row_kv_lens != self.rows
        {
            return Err(Error::Internal {
                message: format!(
                    "paged BF16 causal GQA metadata mismatch: slots={block_slots} offsets={block_offsets}/{} sequence_ids={row_sequence_ids}/{} positions={row_positions}/{} kv_lens={row_kv_lens}/{}",
                    self.sequences + 1,
                    self.rows,
                    self.rows,
                    self.rows,
                ),
            });
        }
        Ok(())
    }

    pub fn resolve_cache_byte_offset(
        self,
        value_cache: bool,
        sequence: usize,
        logical_token: usize,
        head: usize,
        dimension: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
    ) -> Option<usize> {
        if sequence >= self.sequences
            || head >= self.kv_heads
            || dimension >= self.head_dim
            || block_offsets.len() != self.sequences + 1
        {
            return None;
        }
        let start = usize::try_from(block_offsets[sequence]).ok()?;
        let end = usize::try_from(block_offsets[sequence + 1]).ok()?;
        let entry = start.checked_add(logical_token / self.page_tokens)?;
        if start > end || entry >= end {
            return None;
        }
        let physical_slot = usize::try_from(*block_slots.get(entry)?).ok()?;
        let plane = if value_cache {
            self.value_cache
        } else {
            self.key_cache
        };
        plane.byte_offset(
            physical_slot,
            self.layer_index,
            logical_token % self.page_tokens,
            head,
            dimension,
        )
    }
}

/// Indexed split-half RoPE shape. The first half is paired with the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitHalfRopeLayout {
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub table_positions: usize,
    pub restore_bf16_boundary: bool,
}

impl SplitHalfRopeLayout {
    pub const fn table_width(self) -> usize {
        self.rope_dim / 2
    }

    pub fn table_index(self, position: usize, pair: usize) -> Option<usize> {
        (position < self.table_positions && pair < self.table_width())
            .then(|| position.checked_mul(self.table_width())?.checked_add(pair))
            .flatten()
    }

    pub fn value_pair_indices(
        self,
        row: usize,
        head: usize,
        pair: usize,
    ) -> Option<(usize, usize)> {
        if row >= self.rows || head >= self.heads || pair >= self.table_width() {
            return None;
        }
        let base = row
            .checked_mul(self.heads)?
            .checked_add(head)?
            .checked_mul(self.head_dim)?;
        Some((
            base.checked_add(pair)?,
            base.checked_add(self.table_width())?.checked_add(pair)?,
        ))
    }

    pub fn validate(self) -> Result<()> {
        if self.rows == 0
            || self.heads == 0
            || self.head_dim == 0
            || self.rope_dim == 0
            || !self.rope_dim.is_multiple_of(2)
            || self.rope_dim > self.head_dim
            || self.table_positions == 0
        {
            return Err(Error::Internal {
                message: format!("invalid split-half RoPE layout: {self:?}"),
            });
        }
        Ok(())
    }

    pub fn value_elements(self) -> Result<usize> {
        checked_product(
            &[self.rows, self.heads, self.head_dim],
            "split-half RoPE values",
        )
    }

    pub fn table_elements(self) -> Result<usize> {
        checked_product(
            &[self.table_positions, self.rope_dim / 2],
            "split-half RoPE table",
        )
    }

    pub fn pair_count(self) -> Result<usize> {
        checked_product(
            &[self.rows, self.heads, self.rope_dim / 2],
            "split-half RoPE pairs",
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedSoftmaxTopKLayout {
    pub rows: usize,
    pub experts: usize,
    pub top_k: usize,
    pub output_scale: f32,
}

impl SelectedSoftmaxTopKLayout {
    pub fn validate(self) -> Result<()> {
        if self.rows == 0
            || self.experts == 0
            || self.top_k == 0
            || self.top_k > self.experts
            || !self.output_scale.is_finite()
        {
            return Err(Error::Internal {
                message: format!("invalid selected-softmax top-k layout: {self:?}"),
            });
        }
        Ok(())
    }

    pub fn logit_elements(self) -> Result<usize> {
        checked_product(&[self.rows, self.experts], "router logits")
    }

    pub fn output_elements(self) -> Result<usize> {
        checked_product(&[self.rows, self.top_k], "router output")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16MoeRowsLayout {
    pub source_rows: usize,
    pub route_rows: usize,
    pub output_rows: usize,
    pub row_width: usize,
}

impl Bf16MoeRowsLayout {
    pub fn validate(self) -> Result<()> {
        if self.source_rows == 0
            || self.route_rows == 0
            || self.output_rows == 0
            || self.row_width == 0
        {
            return Err(Error::Internal {
                message: format!("invalid BF16 MoE rows layout: {self:?}"),
            });
        }
        Ok(())
    }

    pub fn source_elements(self) -> Result<usize> {
        checked_product(&[self.source_rows, self.row_width], "BF16 MoE source")
    }

    pub fn route_elements(self) -> Result<usize> {
        checked_product(&[self.route_rows, self.row_width], "BF16 MoE routes")
    }

    pub fn output_elements(self) -> Result<usize> {
        checked_product(&[self.output_rows, self.row_width], "BF16 MoE output")
    }
}

fn validate_strided_rows(
    rows: usize,
    heads: usize,
    head_bytes: usize,
    row_stride_bytes: usize,
    head_stride_bytes: usize,
    element_alignment: usize,
    label: &str,
) -> Result<()> {
    let row_extent = (heads.saturating_sub(1))
        .checked_mul(head_stride_bytes)
        .and_then(|bytes| bytes.checked_add(head_bytes))
        .ok_or_else(|| Error::Internal {
            message: format!("{label} row extent overflow"),
        })?;
    if rows == 0
        || heads == 0
        || head_bytes == 0
        || head_stride_bytes < head_bytes
        || row_stride_bytes < row_extent
        || !head_stride_bytes.is_multiple_of(element_alignment)
        || !row_stride_bytes.is_multiple_of(element_alignment)
    {
        return Err(Error::Internal {
            message: format!(
                "invalid {label} strides: rows={rows} heads={heads} head_bytes={head_bytes} row_stride={row_stride_bytes} head_stride={head_stride_bytes}"
            ),
        });
    }
    Ok(())
}

fn required_strided_bytes(
    rows: usize,
    row_stride_bytes: usize,
    heads: usize,
    head_stride_bytes: usize,
    head_bytes: usize,
    element_alignment: usize,
    label: &str,
) -> Result<usize> {
    validate_strided_rows(
        rows,
        heads,
        head_bytes,
        row_stride_bytes,
        head_stride_bytes,
        element_alignment,
        label,
    )?;
    (rows - 1)
        .checked_mul(row_stride_bytes)
        .and_then(|bytes| bytes.checked_add((heads - 1).checked_mul(head_stride_bytes)?))
        .and_then(|bytes| bytes.checked_add(head_bytes))
        .ok_or_else(|| Error::Internal {
            message: format!("{label} byte extent overflow"),
        })
}

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product.checked_mul(*value).ok_or_else(|| Error::Internal {
            message: format!("{label} size overflow"),
        })
    })
}

fn checked_bytes(count: usize, stride: usize, label: &str) -> Result<usize> {
    count.checked_mul(stride).ok_or_else(|| Error::Internal {
        message: format!("{label} byte size overflow"),
    })
}

#[cfg(test)]
mod core_layout_tests {
    use super::*;

    #[test]
    fn paged_transformer_status_values_are_checked() {
        assert!(validate_paged_bf16_transformer_status(0).is_ok());
        assert!(
            validate_paged_bf16_transformer_status(PAGED_BF16_TRANSFORMER_METADATA_ERROR).is_err()
        );
        assert!(
            validate_paged_bf16_transformer_status(PAGED_BF16_TRANSFORMER_ADDRESS_ERROR).is_err()
        );
        assert!(validate_paged_bf16_transformer_status(17).is_err());
    }

    #[test]
    fn packed_gqa_layout_maps_independent_kv_planes() {
        let mut layout = PagedBf16CausalGqaLayout::packed(3, 2, 4, 2, 8, 4, 1, 2, 3, 0.5)
            .expect("packed GQA layout");
        layout.value_cache.slot_stride_bytes += 64;
        layout.value_cache.capacity_bytes += 3 * 64;
        layout.validate().expect("independent value stride");
        let slots = [2, 0, 1];
        let offsets = [0, 2, 3];
        let key = layout
            .resolve_cache_byte_offset(false, 0, 1, 1, 3, &slots, &offsets)
            .expect("key address");
        let value = layout
            .resolve_cache_byte_offset(true, 0, 1, 1, 3, &slots, &offsets)
            .expect("value address");
        assert_ne!(key, value);
    }

    #[test]
    fn gqa_requires_query_heads_divisible_by_kv_heads() {
        assert!(PagedBf16CausalGqaLayout::packed(1, 1, 6, 4, 8, 4, 0, 1, 1, 0.5).is_err());
    }

    #[test]
    fn f32_to_bf16_layout_keeps_independent_byte_strides() {
        let layout = F32ToBf16RowsLayout {
            input: StridedF32RowsLayout {
                rows: 2,
                heads: 3,
                dimensions: 5,
                row_stride_bytes: 96,
                head_stride_bytes: 28,
            },
            output: StridedBf16RowsLayout {
                rows: 2,
                heads: 3,
                dimensions: 5,
                row_stride_bytes: 52,
                head_stride_bytes: 16,
            },
        };
        layout.validate().unwrap();
        assert_eq!(layout.element_count().unwrap(), 30);
        assert_eq!(layout.input.required_bytes().unwrap(), 172);
        assert_eq!(layout.output.required_bytes().unwrap(), 94);

        let mismatched = F32ToBf16RowsLayout {
            output: StridedBf16RowsLayout::packed(2, 2, 5).unwrap(),
            ..layout
        };
        assert!(mismatched.validate().is_err());
    }

    #[test]
    fn split_half_and_router_shapes_are_overflow_checked() {
        let rope = SplitHalfRopeLayout {
            rows: 2,
            heads: 3,
            head_dim: 8,
            rope_dim: 6,
            table_positions: 5,
            restore_bf16_boundary: false,
        };
        assert_eq!(rope.value_elements().unwrap(), 48);
        assert_eq!(rope.table_width(), 3);
        assert_eq!(rope.table_elements().unwrap(), 15);
        assert_eq!(rope.pair_count().unwrap(), 18);
        assert_eq!(rope.table_index(4, 2), Some(14));
        assert_eq!(rope.table_index(5, 0), None);
        assert_eq!(rope.value_pair_indices(1, 2, 2), Some((42, 45)));
        assert_eq!(rope.value_pair_indices(2, 0, 0), None);
        assert!(
            SelectedSoftmaxTopKLayout {
                rows: 1,
                experts: 4,
                top_k: 5,
                output_scale: 1.0,
            }
            .validate()
            .is_err()
        );
    }
}
