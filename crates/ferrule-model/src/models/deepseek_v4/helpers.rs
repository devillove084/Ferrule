//! Free helper functions: RMSNorm, RoPE, YaRN, top-k, cache keys.

use std::cmp::Ordering;
use std::path::Path;

use crate::artifact::format::{
    normalized_hadamard_transform_rows_in_place, simulate_fp4_e2m1_e8m0_activation_quant_in_place,
    simulate_fp8_e4m3fn_e8m0_activation_quant_in_place,
};
use crate::artifact::linear::ArtifactLinearPayload;
use crate::artifact::tensor::{
    ArtifactDType, ArtifactTensorPayload, ArtifactTensorReader, ArtifactTensorSlice,
};
use crate::{HfSafetensorsInventory, HfSafetensorsTensorInfo, TensorRole};
use ferrule_common::{Error, Result};

use super::attention::DeepSeekV4IndexerPayload;
use super::config::{
    with_deepseek_v4_linear_execution_policy, DeepSeekV4AttentionConfig, DeepSeekV4RopeParams,
};
use super::operators::DeepSeekV4OperatorContext;
use crate::runner::TokenLogit;

pub(crate) fn bind_aux_linear(
    auxiliary: &[ArtifactTensorSlice],
    reader: &ArtifactTensorReader,
    role: TensorRole,
    weight_name: &str,
    scale_name: Option<&str>,
) -> Result<ArtifactLinearPayload> {
    let weight = read_aux_tensor(auxiliary, reader, weight_name)?;
    let scale = scale_name
        .map(|name| read_aux_tensor(auxiliary, reader, name))
        .transpose()?;
    ArtifactLinearPayload::from_weight_and_scale(role, weight, scale)
        .map(with_deepseek_v4_linear_execution_policy)
}

pub(crate) fn read_aux_tensor(
    auxiliary: &[ArtifactTensorSlice],
    reader: &ArtifactTensorReader,
    name: &str,
) -> Result<ArtifactTensorPayload> {
    let slice = auxiliary
        .iter()
        .find(|slice| slice.name == name)
        .ok_or_else(|| Error::Model(format!("DeepSeek-V4 missing auxiliary tensor '{name}'")))?;
    reader.read_slice(slice)
}

pub(crate) fn read_aux_tensor_f32(
    auxiliary: &[ArtifactTensorSlice],
    reader: &ArtifactTensorReader,
    name: &str,
) -> Result<ArtifactTensorPayload> {
    let payload = read_aux_tensor(auxiliary, reader, name)?;
    let _ = decode_tensor_f32(&payload)?;
    Ok(payload)
}

pub(crate) fn two_dim_shape_from_payload(
    payload: &ArtifactTensorPayload,
    label: &str,
) -> Result<(usize, usize)> {
    let [rows, cols]: [usize; 2] =
        payload
            .slice
            .shape
            .clone()
            .try_into()
            .map_err(|shape: Vec<usize>| {
                Error::Model(format!(
                    "DeepSeek-V4 {label} '{}' expects 2D shape, got {:?}",
                    payload.slice.name, shape
                ))
            })?;
    Ok((rows, cols))
}

pub(crate) fn check_linear(
    layer: usize,
    label: &str,
    linear: &ArtifactLinearPayload,
    out: usize,
    input: usize,
) -> Result<()> {
    if linear.format.out_features() != out || linear.format.in_features() != input {
        return Err(Error::Model(format!(
            "DeepSeek-V4 layer {layer} {label} shape mismatch: got [{}, {}], expected [{out}, {input}]",
            linear.format.out_features(),
            linear.format.in_features()
        )));
    }
    Ok(())
}

pub(crate) fn check_len(layer: usize, label: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(Error::Model(format!(
            "DeepSeek-V4 layer {layer} {label} length mismatch: got {got}, expected {expected}"
        )));
    }
    Ok(())
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
        return Err(Error::Model(format!(
            "DeepSeek-V4 {label} batched RMS length mismatch: tokens={tokens} input={} weight={}",
            input.len(),
            weight.len()
        )));
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
        return Err(Error::Model(format!(
            "DeepSeek-V4 QAT FP8 shape mismatch: values={} head_dim={head_dim} rope_dim={rope_dim}",
            values.len()
        )));
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
        return Err(Error::Model(format!(
            "DeepSeek-V4 top-k concat shape mismatch: tokens={tokens} left={} left_cols={left_cols} right={} right_cols={right_cols}",
            left.len(),
            right.len()
        )));
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
        return Err(Error::Model(format!(
            "DeepSeek-V4 indexer prefill shape mismatch: tokens={tokens} hidden={} q_latents={} compressed={}",
            hidden.len(),
            q_latents.len(),
            indexer_compressed.len()
        )));
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
        return Err(Error::Model(format!(
            "DeepSeek-V4 compressor row shape mismatch: rows={rows} head_dim={head_dim} kv={} score={}",
            kv_rows.len(),
            score_rows.len()
        )));
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
            return Err(Error::Model(
                "DeepSeek-V4 compressor softmax denominator is invalid".into(),
            ));
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
        return Err(Error::Model(
            "DeepSeek-V4 indexer compressed cache length is not divisible by index_head_dim".into(),
        ));
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
    output_a: &ArtifactLinearPayload,
    context: &[f32],
    cfg: DeepSeekV4AttentionConfig,
    layer: usize,
) -> Result<Vec<f32>> {
    if context.len() != cfg.q_full_dim() {
        return Err(Error::Model(format!(
            "DeepSeek-V4 layer {layer} context length mismatch: expected {}, got {}",
            cfg.q_full_dim(),
            context.len()
        )));
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

pub(crate) fn rms_norm(input: &[f32], weight: &[f32], eps: f32, label: &str) -> Result<Vec<f32>> {
    if input.len() != weight.len() || input.is_empty() {
        return Err(Error::Model(format!(
            "DeepSeek-V4 {label} RMS length mismatch: input={}, weight={}",
            input.len(),
            weight.len()
        )));
    }
    let scale = (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32 + eps)
        .sqrt()
        .recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * weight)
        .collect())
}

pub(crate) fn rms_norm_heads_in_place(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    eps: f32,
    layer: usize,
) -> Result<()> {
    if values.len() != heads * head_dim {
        return Err(Error::Model(format!(
            "DeepSeek-V4 layer {layer} query length mismatch: expected {}, got {}",
            heads * head_dim,
            values.len()
        )));
    }
    for head in 0..heads {
        let row = &mut values[head * head_dim..(head + 1) * head_dim];
        let scale = (row.iter().map(|value| value * value).sum::<f32>() / head_dim as f32 + eps)
            .sqrt()
            .recip();
        for value in row {
            *value *= scale;
        }
    }
    Ok(())
}

pub(crate) fn apply_rotary_tail(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    theta: f32,
    inverse: bool,
) -> Result<()> {
    apply_rotary_tail_scaled(
        values,
        heads,
        head_dim,
        rope_dim,
        position,
        DeepSeekV4RopeParams::plain(theta),
        inverse,
    )
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
    if rope_dim == 0 {
        return Ok(());
    }
    if rope_dim > head_dim || !rope_dim.is_multiple_of(2) || values.len() != heads * head_dim {
        return Err(Error::Model(format!(
            "DeepSeek-V4 rotary shape mismatch: values={}, heads={heads}, head_dim={head_dim}, rope_dim={rope_dim}",
            values.len()
        )));
    }
    let tail_start = head_dim - rope_dim;
    for head in 0..heads {
        let base = head * head_dim + tail_start;
        for pair in 0..rope_dim / 2 {
            let freq = yarn_frequency(pair, rope_dim, rope);
            let angle = position as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let sin = if inverse { -sin } else { sin };
            let x0 = values[base + 2 * pair];
            let x1 = values[base + 2 * pair + 1];
            values[base + 2 * pair] = x0 * cos - x1 * sin;
            values[base + 2 * pair + 1] = x0 * sin + x1 * cos;
        }
    }
    Ok(())
}

pub(crate) fn yarn_frequency(pair: usize, rope_dim: usize, rope: DeepSeekV4RopeParams) -> f32 {
    let base_freq = 1.0 / rope.theta.powf((2 * pair) as f32 / rope_dim as f32);
    if rope.original_seq_len == 0 || rope.factor == 1.0 {
        return base_freq;
    }
    let (low, high) = yarn_correction_range(
        rope.beta_fast as f32,
        rope.beta_slow as f32,
        rope_dim,
        rope.theta,
        rope.original_seq_len as f32,
    );
    let ramp = yarn_linear_ramp(pair as f32, low as f32, high as f32);
    let smooth = 1.0 - ramp;
    base_freq / rope.factor * (1.0 - smooth) + base_freq * smooth
}

pub(crate) fn yarn_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: usize,
    base: f32,
    max_position: f32,
) -> (usize, usize) {
    let low = yarn_correction_dim(low_rot, dim, base, max_position).floor() as isize;
    let high = yarn_correction_dim(high_rot, dim, base, max_position).ceil() as isize;
    (
        low.max(0) as usize,
        high.min(dim as isize - 1).max(0) as usize,
    )
}

pub(crate) fn yarn_correction_dim(
    num_rotations: f32,
    dim: usize,
    base: f32,
    max_position: f32,
) -> f32 {
    dim as f32 * (max_position / (num_rotations * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

pub(crate) fn yarn_linear_ramp(value: f32, min: f32, mut max: f32) -> f32 {
    if (min - max).abs() < f32::EPSILON {
        max += 0.001;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

pub(crate) fn unique_top_level_slice(
    model_dir: &Path,
    inventory: &HfSafetensorsInventory,
    role: TensorRole,
) -> Result<ArtifactTensorSlice> {
    let tensors = inventory
        .tensors
        .iter()
        .filter(|tensor| tensor.role == role)
        .collect::<Vec<_>>();
    match tensors.as_slice() {
        [tensor] => Ok(ArtifactTensorSlice::from_hf_inventory(model_dir, tensor)),
        [] => Err(Error::Model(format!(
            "DeepSeek-V4 missing top-level tensor role {role}"
        ))),
        _ => Err(Error::Model(format!(
            "DeepSeek-V4 expected exactly one top-level tensor role {role}, got {}",
            tensors.len()
        ))),
    }
}

pub(crate) fn read_named_vector_f32(
    model_dir: &Path,
    inventory: &HfSafetensorsInventory,
    reader: &ArtifactTensorReader,
    name: &str,
    role: TensorRole,
) -> Result<Vec<f32>> {
    let tensor = inventory_tensor(inventory, name)?;
    let mut slice = ArtifactTensorSlice::from_hf_inventory(model_dir, tensor);
    slice.role = role;
    decode_vector_f32(&reader.read_slice(&slice)?)
}

pub(crate) fn inventory_tensor<'a>(
    inventory: &'a HfSafetensorsInventory,
    name: &str,
) -> Result<&'a HfSafetensorsTensorInfo> {
    inventory
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::Model(format!("DeepSeek-V4 missing tensor '{name}'")))
}

pub(crate) fn decode_vector_f32(payload: &ArtifactTensorPayload) -> Result<Vec<f32>> {
    if payload.slice.shape.len() != 1 {
        return Err(Error::Model(format!(
            "DeepSeek-V4 artifact vector '{}' expects 1D shape, got {:?}",
            payload.slice.name, payload.slice.shape
        )));
    }
    decode_tensor_f32(payload)
}

pub(crate) fn decode_tensor_f32(payload: &ArtifactTensorPayload) -> Result<Vec<f32>> {
    let expected = payload.slice.element_count()?;
    match payload.slice.dtype {
        ArtifactDType::F32 => {
            if payload.bytes.len() != expected * 4 {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 F32 tensor '{}' byte length mismatch",
                    payload.slice.name
                )));
            }
            Ok(payload
                .bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        ArtifactDType::Bf16 => {
            if payload.bytes.len() != expected * 2 {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 BF16 tensor '{}' byte length mismatch",
                    payload.slice.name
                )));
            }
            Ok(payload
                .bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    f32::from_bits(bits << 16)
                })
                .collect())
        }
        _ => Err(Error::Model(format!(
            "DeepSeek-V4 artifact tensor '{}' has unsupported vector dtype {}",
            payload.slice.name,
            payload.slice.dtype.as_str()
        ))),
    }
}

pub(crate) fn usize_key(json: &serde_json::Value, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|key| {
        json.get(*key)
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
    })
}

pub(crate) fn f32_key(json: &serde_json::Value, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| {
        json.get(*key)
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
    })
}

pub(crate) fn rank_logits_desc(left: &TokenLogit, right: &TokenLogit) -> Ordering {
    right
        .logit
        .total_cmp(&left.logit)
        .then_with(|| left.token_id.cmp(&right.token_id))
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
