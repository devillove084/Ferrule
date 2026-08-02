//! Free helper functions: RMSNorm, RoPE, YaRN, top-k, cache keys.
//!
//! Family-neutral math, artifact loading, shape validation, and JSON keys live
//! in [`crate::models::common`]; this module preserves DeepSeek-specific error
//! context and keeps the MLA/indexer/compressor helpers local.

use std::path::Path;

use crate::checkpoint::encoding::{
    normalized_hadamard_transform_rows_in_place, simulate_fp4_e2m1_e8m0_activation_quant_in_place,
    simulate_fp8_e4m3fn_e8m0_activation_quant_in_place,
};
use crate::checkpoint::tensor::{
    CheckpointTensorPayload, CheckpointTensorReader, CheckpointTensorSlice,
};
use crate::checkpoint::weight::LinearWeight;
use crate::{HfSafetensorsInventory, TensorRole};
use ferrule_common::{Error, Result};

pub(crate) use crate::models::common::config_json::{f32_key, usize_key};
pub(crate) use crate::models::common::math::{dot, rms_norm};
pub(crate) use crate::models::common::rope::apply_rotary_tail;

/// DeepSeek wrapper preserving the layer-indexed error context.
pub(crate) fn rms_norm_heads_in_place(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    eps: f32,
    layer: usize,
) -> Result<()> {
    crate::models::common::math::rms_norm_heads_in_place(
        values,
        heads,
        head_dim,
        eps,
        &format!("DeepSeek-V4 layer {layer}"),
    )
}

use super::attention::DeepSeekV4IndexerPayload;
use super::config::{
    DeepSeekV4AttentionConfig, DeepSeekV4RopeParams, with_deepseek_v4_linear_execution_policy,
};
use super::operators::DeepSeekV4OperatorContext;

pub(crate) fn bind_aux_linear(
    auxiliary: &[CheckpointTensorSlice],
    reader: &CheckpointTensorReader,
    role: TensorRole,
    weight_name: &str,
    scale_name: Option<&str>,
) -> Result<LinearWeight> {
    let weight = read_aux_tensor(auxiliary, reader, weight_name)?;
    let scale = scale_name
        .map(|name| read_aux_tensor(auxiliary, reader, name))
        .transpose()?;
    LinearWeight::from_weight_and_scale(role, weight, scale)
        .map(with_deepseek_v4_linear_execution_policy)
}

pub(crate) fn read_aux_tensor(
    auxiliary: &[CheckpointTensorSlice],
    reader: &CheckpointTensorReader,
    name: &str,
) -> Result<CheckpointTensorPayload> {
    let slice = auxiliary
        .iter()
        .find(|slice| slice.name == name)
        .ok_or_else(|| Error::Model {
            message: format!("DeepSeek-V4 missing auxiliary tensor '{name}'"),
        })?;
    reader.read_slice(slice)
}

pub(crate) fn read_aux_tensor_f32(
    auxiliary: &[CheckpointTensorSlice],
    reader: &CheckpointTensorReader,
    name: &str,
) -> Result<CheckpointTensorPayload> {
    let payload = read_aux_tensor(auxiliary, reader, name)?;
    let _ = decode_tensor_f32(&payload)?;
    Ok(payload)
}

pub(crate) fn two_dim_shape_from_payload(
    payload: &CheckpointTensorPayload,
    label: &str,
) -> Result<(usize, usize)> {
    crate::models::common::shape::two_dim_shape_from_payload(payload, label, "DeepSeek-V4")
}

pub(crate) fn check_linear(
    layer: usize,
    label: &str,
    linear: &LinearWeight,
    out: usize,
    input: usize,
) -> Result<()> {
    crate::models::common::shape::check_linear(
        linear,
        out,
        input,
        label,
        &format!("DeepSeek-V4 layer {layer}"),
    )
}

pub(crate) fn check_len(layer: usize, label: &str, got: usize, expected: usize) -> Result<()> {
    crate::models::common::shape::check_len(
        got,
        expected,
        label,
        &format!("DeepSeek-V4 layer {layer}"),
    )
}

pub(crate) fn rms_norm_rows_with_operators(
    operators: &mut DeepSeekV4OperatorContext,
    input: &[f32],
    tokens: usize,
    weight: &[f32],
    eps: f32,
    label: &str,
) -> Result<Vec<f32>> {
    if tokens == 0 || weight.is_empty() || input.len() != tokens * weight.len() {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 {label} batched RMS length mismatch: tokens={tokens} input={} weight={}",
                input.len(),
                weight.len()
            ),
        });
    }
    operators.rms_norm_rows(input, tokens, weight, eps, label)
}

pub(crate) fn quantize_attention_kv_for_qat_in_place(
    values: &mut [f32],
    head_dim: usize,
    rope_dim: usize,
) -> Result<()> {
    quantize_non_rope_fp8_for_qat_in_place(values, head_dim, rope_dim, 64)
}

pub(crate) fn quantize_compressed_kv_for_qat_in_place(
    values: &mut [f32],
    head_dim: usize,
    rope_dim: usize,
    rotate_for_indexer: bool,
) -> Result<()> {
    if rotate_for_indexer {
        quantize_indexer_activation_for_qat_in_place(values, head_dim)
    } else {
        quantize_non_rope_fp8_for_qat_in_place(values, head_dim, rope_dim, 64)
    }
}

pub(crate) fn quantize_non_rope_fp8_for_qat_in_place(
    values: &mut [f32],
    head_dim: usize,
    rope_dim: usize,
    block_size: usize,
) -> Result<()> {
    if head_dim == 0 || rope_dim > head_dim || !values.len().is_multiple_of(head_dim) {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 QAT FP8 shape mismatch: values={} head_dim={head_dim} rope_dim={rope_dim}",
                values.len()
            ),
        });
    }
    let non_rope = head_dim - rope_dim;
    if non_rope == 0 {
        return Ok(());
    }
    let effective_block_size = if non_rope.is_multiple_of(block_size) {
        block_size
    } else {
        non_rope
    };
    for row in values.chunks_exact_mut(head_dim) {
        simulate_fp8_e4m3fn_e8m0_activation_quant_in_place(
            &mut row[..non_rope],
            non_rope,
            effective_block_size,
        )?;
    }
    Ok(())
}

pub(crate) fn quantize_indexer_activation_for_qat_in_place(
    values: &mut [f32],
    row_width: usize,
) -> Result<()> {
    normalized_hadamard_transform_rows_in_place(values, row_width)?;
    simulate_fp4_e2m1_e8m0_activation_quant_in_place(values, row_width, 32)
}

pub(crate) fn window_topk_indices_prefill(window_size: usize, tokens: usize) -> Vec<isize> {
    let cols = tokens.min(window_size);
    let mut out = vec![-1; tokens * cols];
    for token in 0..tokens {
        let first = (token + 1).saturating_sub(window_size);
        for col in 0..cols {
            let idx = first + col;
            if idx <= token {
                out[token * cols + col] = idx as isize;
            }
        }
    }
    out
}

pub(crate) fn compress_topk_indices_prefill(
    ratio: usize,
    tokens: usize,
    offset: usize,
) -> (Vec<isize>, usize) {
    if ratio == 0 {
        return (Vec::new(), 0);
    }
    let cols = tokens / ratio;
    if cols == 0 {
        return (Vec::new(), 0);
    }
    let mut out = vec![-1; tokens * cols];
    for token in 0..tokens {
        let visible = (token + 1) / ratio;
        for idx in 0..cols {
            if idx < visible {
                out[token * cols + idx] = (offset + idx) as isize;
            }
        }
    }
    (out, cols)
}

pub(crate) fn concat_topk_rows(
    left: &[isize],
    left_cols: usize,
    right: &[isize],
    right_cols: usize,
    tokens: usize,
) -> Result<Vec<isize>> {
    if left.len() != tokens * left_cols || right.len() != tokens * right_cols {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 top-k concat shape mismatch: tokens={tokens} left={} left_cols={left_cols} right={} right_cols={right_cols}",
                left.len(),
                right.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(tokens * (left_cols + right_cols));
    for token in 0..tokens {
        out.extend_from_slice(&left[token * left_cols..(token + 1) * left_cols]);
        out.extend_from_slice(&right[token * right_cols..(token + 1) * right_cols]);
    }
    Ok(out)
}

pub(crate) fn indexer_topk_indices_prefill(
    indexer: &DeepSeekV4IndexerPayload,
    cfg: DeepSeekV4AttentionConfig,
    q_latents: &[f32],
    hidden: &[f32],
    indexer_compressed: &[f32],
    offset: usize,
    operators: &mut DeepSeekV4OperatorContext,
) -> Result<(Vec<isize>, usize)> {
    let tokens = hidden.len() / cfg.hidden_size;
    let compressed_len = indexer_compressed.len() / cfg.index_head_dim;
    let cols = cfg.index_topk.min(compressed_len);
    if cols == 0 {
        return Ok((Vec::new(), 0));
    }
    if hidden.len() != tokens * cfg.hidden_size
        || q_latents.len() != tokens * cfg.q_lora_rank
        || indexer_compressed.len() != compressed_len * cfg.index_head_dim
    {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 indexer prefill shape mismatch: tokens={tokens} hidden={} q_latents={} compressed={}",
                hidden.len(),
                q_latents.len(),
                indexer_compressed.len()
            ),
        });
    }

    let mut out = vec![-1; tokens * cols];
    for token in 0..tokens {
        let q_latent = &q_latents[token * cfg.q_lora_rank..(token + 1) * cfg.q_lora_rank];
        let mut query = operators.linear_matvec(&indexer.wq_b, q_latent)?;
        apply_rotary_tail_scaled(
            &mut query,
            cfg.index_n_heads,
            cfg.index_head_dim,
            cfg.rope_head_dim.min(cfg.index_head_dim),
            token,
            cfg.rope_params(),
            false,
        )?;
        quantize_indexer_activation_for_qat_in_place(&mut query, cfg.index_head_dim)?;
        let hidden_row = &hidden[token * cfg.hidden_size..(token + 1) * cfg.hidden_size];
        let mut weights = operators.linear_matvec(&indexer.weights_proj, hidden_row)?;
        let scale = (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
        for weight in &mut weights {
            *weight *= scale;
        }

        let visible = (token + 1) / cfg.compress_ratio;
        if visible == 0 {
            continue;
        }
        let mut scores = vec![f32::NEG_INFINITY; compressed_len];
        for idx in 0..compressed_len.min(visible) {
            let kv = &indexer_compressed[idx * cfg.index_head_dim..(idx + 1) * cfg.index_head_dim];
            let mut score = 0.0f32;
            for head in 0..cfg.index_n_heads {
                let q = &query[head * cfg.index_head_dim..(head + 1) * cfg.index_head_dim];
                score += dot(q, kv).max(0.0) * weights[head];
            }
            scores[idx] = score;
        }
        let mut order = (0..compressed_len.min(visible)).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        for (slot, idx) in order.into_iter().take(cols).enumerate() {
            if scores[idx].is_finite() {
                out[token * cols + slot] = (offset + idx) as isize;
            }
        }
    }
    Ok((out, cols))
}

pub(crate) fn compress_rows_softmax(
    kv_rows: &[f32],
    score_rows: &[f32],
    rows: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if rows == 0
        || head_dim == 0
        || kv_rows.len() != rows * head_dim
        || score_rows.len() != rows * head_dim
    {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 compressor row shape mismatch: rows={rows} head_dim={head_dim} kv={} score={}",
                kv_rows.len(),
                score_rows.len()
            ),
        });
    }
    let mut out = vec![0.0f32; head_dim];
    for dim in 0..head_dim {
        let mut max_score = f32::NEG_INFINITY;
        for row in 0..rows {
            max_score = max_score.max(score_rows[row * head_dim + dim]);
        }
        if !max_score.is_finite() {
            continue;
        }
        let mut denom = 0.0f32;
        for row in 0..rows {
            let score = score_rows[row * head_dim + dim];
            if score.is_finite() {
                denom += (score - max_score).exp();
            }
        }
        if denom == 0.0 || !denom.is_finite() {
            return Err(Error::Model {
                message: "DeepSeek-V4 compressor softmax denominator is invalid".into(),
            });
        }
        for row in 0..rows {
            let score = score_rows[row * head_dim + dim];
            if score.is_finite() {
                let weight = (score - max_score).exp() / denom;
                out[dim] += weight * kv_rows[row * head_dim + dim];
            }
        }
    }
    Ok(out)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the indexer kernel keeps model inputs, cache coordinates, and operator state explicit"
)]
pub(crate) fn indexer_topk_indices(
    indexer: &DeepSeekV4IndexerPayload,
    cfg: DeepSeekV4AttentionConfig,
    q_latent: &[f32],
    hidden: &[f32],
    position: usize,
    indexer_compressed: &[f32],
    offset: usize,
    operators: &mut DeepSeekV4OperatorContext,
) -> Result<Vec<isize>> {
    let compressed_len = indexer_compressed.len() / cfg.index_head_dim;
    if compressed_len == 0 {
        return Ok(Vec::new());
    }
    if indexer_compressed.len() != compressed_len * cfg.index_head_dim {
        return Err(Error::Model {
            message:
                "DeepSeek-V4 indexer compressed cache length is not divisible by index_head_dim"
                    .into(),
        });
    }
    let mut query = operators.linear_matvec(&indexer.wq_b, q_latent)?;
    apply_rotary_tail_scaled(
        &mut query,
        cfg.index_n_heads,
        cfg.index_head_dim,
        cfg.rope_head_dim.min(cfg.index_head_dim),
        position,
        cfg.rope_params(),
        false,
    )?;
    quantize_indexer_activation_for_qat_in_place(&mut query, cfg.index_head_dim)?;
    let mut weights = operators.linear_matvec(&indexer.weights_proj, hidden)?;
    let scale = (cfg.index_head_dim as f32).powf(-0.5) * (cfg.index_n_heads as f32).powf(-0.5);
    for weight in &mut weights {
        *weight *= scale;
    }
    let mut scores = vec![0.0f32; compressed_len];
    for token in 0..compressed_len {
        let kv = &indexer_compressed[token * cfg.index_head_dim..(token + 1) * cfg.index_head_dim];
        let mut score = 0.0f32;
        for head in 0..cfg.index_n_heads {
            let q = &query[head * cfg.index_head_dim..(head + 1) * cfg.index_head_dim];
            score += dot(q, kv).max(0.0) * weights[head];
        }
        scores[token] = score;
    }
    let take = cfg.index_topk.min(compressed_len);
    let mut order = (0..compressed_len).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    Ok(order
        .into_iter()
        .take(take)
        .map(|idx| (offset + idx) as isize)
        .collect())
}

pub(crate) fn grouped_output_a(
    output_a: &LinearWeight,
    context: &[f32],
    cfg: DeepSeekV4AttentionConfig,
    layer: usize,
) -> Result<Vec<f32>> {
    if context.len() != cfg.q_full_dim() {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 layer {layer} context length mismatch: expected {}, got {}",
                cfg.q_full_dim(),
                context.len()
            ),
        });
    }
    check_linear(
        layer,
        "wo_a",
        output_a,
        cfg.output_latent_dim(),
        cfg.output_group_input_dim(),
    )?;
    let weights = output_a.reference_weights_f32()?;
    let group_in = cfg.output_group_input_dim();
    let mut out = vec![0.0; cfg.output_latent_dim()];
    for group in 0..cfg.o_groups {
        let context_start = group * group_in;
        let context_group = &context[context_start..context_start + group_in];
        for rank in 0..cfg.o_lora_rank {
            let row = group * cfg.o_lora_rank + rank;
            let weight_row = &weights[row * group_in..(row + 1) * group_in];
            out[row] = dot(weight_row, context_group);
        }
    }
    Ok(out)
}

pub(crate) fn apply_rotary_tail_scaled(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    rope: DeepSeekV4RopeParams,
    inverse: bool,
) -> Result<()> {
    crate::models::common::rope::apply_rotary_tail_scaled(
        values,
        heads,
        head_dim,
        rope_dim,
        position,
        rope.into(),
        inverse,
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn yarn_frequency(pair: usize, rope_dim: usize, rope: DeepSeekV4RopeParams) -> f32 {
    crate::models::common::rope::yarn_frequency(pair, rope_dim, rope.into())
}

pub(crate) fn unique_top_level_slice(
    model_dir: &Path,
    inventory: &HfSafetensorsInventory,
    role: TensorRole,
) -> Result<CheckpointTensorSlice> {
    crate::models::common::checkpoint::unique_top_level_slice(
        model_dir,
        inventory,
        role,
        "DeepSeek-V4",
    )
}

pub(crate) fn read_named_vector_f32(
    model_dir: &Path,
    inventory: &HfSafetensorsInventory,
    reader: &CheckpointTensorReader,
    name: &str,
    role: TensorRole,
) -> Result<Vec<f32>> {
    crate::models::common::checkpoint::read_named_vector_f32(
        model_dir,
        inventory,
        reader,
        name,
        role,
        "DeepSeek-V4",
    )
}

pub(crate) fn decode_vector_f32(payload: &CheckpointTensorPayload) -> Result<Vec<f32>> {
    crate::models::common::checkpoint::decode_vector_f32(payload, "DeepSeek-V4")
}

pub(crate) fn decode_tensor_f32(payload: &CheckpointTensorPayload) -> Result<Vec<f32>> {
    crate::models::common::checkpoint::decode_tensor_f32(payload, "DeepSeek-V4")
}
