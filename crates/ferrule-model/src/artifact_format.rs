//! Artifact weight-format helpers for reference tests and tiny fixtures.
//!
//! These routines are intentionally correctness/debug utilities, not the final
//! high-throughput path. Production execution should consume packed FP4 + E8M0
//! scales directly in backend kernels instead of expanding experts to F32.

use ferrule_common::{Error, Result};

/// Decode one NVIDIA/OCP-style finite E2M1 FP4 nibble into f32.
///
/// The unsigned magnitude table is {0, 0.5, 1, 1.5, 2, 3, 4, 6}; bit 3 is sign.
/// This matches the `float4_e2m1fn` value range used by the reference kernels
/// (`fp4_max = 6.0`).
pub fn decode_fp4_e2m1_nibble(nibble: u8) -> f32 {
    let magnitude = match nibble & 0x07 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    if nibble & 0x08 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Decode packed FP4 bytes into f32 values, low nibble first.
///
/// This is a debug/reference convention. If a future reference comparison proves
/// torch's `float4_e2m1fn_x2` exposes the nibbles in the opposite logical order,
/// the high-performance kernels should be adjusted with an explicit layout flag.
pub fn decode_fp4_e2m1_packed_low_first(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(decode_fp4_e2m1_nibble(byte & 0x0f));
        out.push(decode_fp4_e2m1_nibble(byte >> 4));
    }
    out
}

/// Decode unsigned E8M0 power-of-two scale into f32.
///
/// Decode FP4/FP8 block scales stored as `float8_e8m0fnu`. The operational
/// behavior needed by kernels is a power-of-two scale. This reference helper uses
/// the standard exponent-bias interpretation where byte 127 maps to 1.0.
pub fn decode_e8m0_scale(byte: u8) -> f32 {
    2.0f32.powi(byte as i32 - 127)
}

/// Decode one finite FP8 E4M3FN byte into f32.
///
/// This follows the common OCP/NVIDIA E4M3 finite encoding used by
/// `torch.float8_e4m3fn`: 1 sign bit, 4 exponent bits, 3 mantissa bits, exponent
/// bias 7, no infinities, and byte patterns with exponent=15/mantissa=7 treated
/// as NaN. The maximum finite magnitude is 448.0 (`0x7e`).
pub fn decode_fp8_e4m3fn_byte(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        if mantissa == 0 {
            return sign * 0.0;
        }
        return sign * (mantissa as f32) * 2.0f32.powi(-9);
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    sign * 2.0f32.powi(exponent as i32 - 7) * (1.0 + mantissa as f32 / 8.0)
}

/// Dequantize one FP4 E2M1 packed matrix with E8M0 per-block scales.
///
/// `weight` is laid out as `[out_features, in_features / 2]` packed FP4 bytes.
/// `scales` is `[out_features, in_features / block_size]` with one scale per
/// logical K block. This is for tiny fixtures and CPU reference checks only.
pub fn dequantize_fp4_e2m1_with_e8m0_scales(
    weight: &[u8],
    scales: &[u8],
    out_features: usize,
    in_features: usize,
    block_size: usize,
) -> Result<Vec<f32>> {
    if block_size == 0 || !in_features.is_multiple_of(block_size) || !in_features.is_multiple_of(2)
    {
        return Err(Error::Model(format!(
            "invalid FP4 shape: in_features={in_features}, block_size={block_size}"
        )));
    }
    let packed_cols = in_features / 2;
    let scale_cols = in_features / block_size;
    let expected_weight = out_features
        .checked_mul(packed_cols)
        .ok_or_else(|| Error::Model("FP4 weight size overflow".into()))?;
    let expected_scales = out_features
        .checked_mul(scale_cols)
        .ok_or_else(|| Error::Model("FP4 scale size overflow".into()))?;
    if weight.len() != expected_weight {
        return Err(Error::Model(format!(
            "FP4 weight length mismatch: expected {expected_weight}, got {}",
            weight.len()
        )));
    }
    if scales.len() != expected_scales {
        return Err(Error::Model(format!(
            "FP4 scale length mismatch: expected {expected_scales}, got {}",
            scales.len()
        )));
    }

    let mut out = vec![0.0f32; out_features * in_features];
    for row in 0..out_features {
        for k in 0..in_features {
            let packed = weight[row * packed_cols + k / 2];
            let nibble = if k % 2 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let scale = decode_e8m0_scale(scales[row * scale_cols + k / block_size]);
            out[row * in_features + k] = decode_fp4_e2m1_nibble(nibble) * scale;
        }
    }
    Ok(out)
}

/// Dequantize one FP8 E4M3FN matrix with E8M0 2D block scales.
///
/// `weight` is `[out_features, in_features]` FP8 bytes. `scales` follows the
/// official reference linear layout for FP8 weights: `[ceil(out / block_m),
/// ceil(in / block_k)]`, with one scale per output/input tile.
pub fn dequantize_fp8_e4m3fn_with_e8m0_scales(
    weight: &[u8],
    scales: &[u8],
    out_features: usize,
    in_features: usize,
    block_m: usize,
    block_k: usize,
) -> Result<Vec<f32>> {
    if block_m == 0 || block_k == 0 {
        return Err(Error::Model(format!(
            "invalid FP8 block shape: block_m={block_m}, block_k={block_k}"
        )));
    }
    let expected_weight = out_features
        .checked_mul(in_features)
        .ok_or_else(|| Error::Model("FP8 weight size overflow".into()))?;
    let scale_rows = out_features.div_ceil(block_m);
    let scale_cols = in_features.div_ceil(block_k);
    let expected_scales = scale_rows
        .checked_mul(scale_cols)
        .ok_or_else(|| Error::Model("FP8 scale size overflow".into()))?;
    if weight.len() != expected_weight {
        return Err(Error::Model(format!(
            "FP8 weight length mismatch: expected {expected_weight}, got {}",
            weight.len()
        )));
    }
    if scales.len() != expected_scales {
        return Err(Error::Model(format!(
            "FP8 scale length mismatch: expected {expected_scales}, got {}",
            scales.len()
        )));
    }

    let mut out = vec![0.0f32; expected_weight];
    for row in 0..out_features {
        for col in 0..in_features {
            let scale = decode_e8m0_scale(scales[(row / block_m) * scale_cols + col / block_k]);
            out[row * in_features + col] =
                decode_fp8_e4m3fn_byte(weight[row * in_features + col]) * scale;
        }
    }
    Ok(out)
}

/// Simulate official block-wise FP8 activation quantization in-place.
///
/// This matches the reference `act_quant(..., scale_dtype=float8_e8m0fnu,
/// scale_fmt="ue8m0", inplace=True)` contract: each row/block uses a power-of-two
/// scale computed from `ceil(log2(absmax / 448))`, casts through E4M3FN, and
/// dequantizes back to f32. It is intentionally a reference helper for semantic
/// parity; production backends should fuse this into kernels.
pub fn simulate_fp8_e4m3fn_e8m0_activation_quant_in_place(
    values: &mut [f32],
    row_width: usize,
    block_size: usize,
) -> Result<()> {
    if row_width == 0 || block_size == 0 || !row_width.is_multiple_of(block_size) {
        return Err(Error::Model(format!(
            "invalid FP8 activation quant shape: row_width={row_width}, block_size={block_size}"
        )));
    }
    if !values.len().is_multiple_of(row_width) {
        return Err(Error::Model(format!(
            "FP8 activation quant length {} is not a multiple of row_width {row_width}",
            values.len()
        )));
    }

    for row in values.chunks_exact_mut(row_width) {
        for block in row.chunks_exact_mut(block_size) {
            let amax = block
                .iter()
                .fold(0.0f32, |acc, value| acc.max(value.abs()))
                .max(1e-4);
            let scale = rounded_power_of_two_scale(amax, 448.0);
            for value in block {
                let quantized = quantize_fp8_e4m3fn_to_f32((*value / scale).clamp(-448.0, 448.0));
                *value = quantized * scale;
            }
        }
    }
    Ok(())
}

/// Simulate official block-wise FP4 E2M1 activation quantization in-place.
///
/// This corresponds to `fp4_act_quant(..., inplace=True)`: each row/block uses a
/// power-of-two scale from `ceil(log2(absmax / 6))`, casts through E2M1, and
/// dequantizes back to f32.
pub fn simulate_fp4_e2m1_e8m0_activation_quant_in_place(
    values: &mut [f32],
    row_width: usize,
    block_size: usize,
) -> Result<()> {
    if row_width == 0 || block_size == 0 || !row_width.is_multiple_of(block_size) {
        return Err(Error::Model(format!(
            "invalid FP4 activation quant shape: row_width={row_width}, block_size={block_size}"
        )));
    }
    if !values.len().is_multiple_of(row_width) {
        return Err(Error::Model(format!(
            "FP4 activation quant length {} is not a multiple of row_width {row_width}",
            values.len()
        )));
    }

    for row in values.chunks_exact_mut(row_width) {
        for block in row.chunks_exact_mut(block_size) {
            let amax = block
                .iter()
                .fold(0.0f32, |acc, value| acc.max(value.abs()))
                .max(6.0 * 2.0f32.powi(-126));
            let scale = rounded_power_of_two_scale(amax, 6.0);
            for value in block {
                let quantized = quantize_fp4_e2m1_to_f32((*value / scale).clamp(-6.0, 6.0));
                *value = quantized * scale;
            }
        }
    }
    Ok(())
}

/// Apply a normalized Walsh-Hadamard transform independently to each row.
///
/// The official DSV4 indexer uses `fast_hadamard_transform(x, scale=n^-0.5)`
/// before FP4 activation quantization. This helper keeps that transform generic
/// and CPU-reference-only.
pub fn normalized_hadamard_transform_rows_in_place(
    values: &mut [f32],
    row_width: usize,
) -> Result<()> {
    if row_width == 0 || !row_width.is_power_of_two() {
        return Err(Error::Model(format!(
            "Hadamard row_width must be a non-zero power of two, got {row_width}"
        )));
    }
    if !values.len().is_multiple_of(row_width) {
        return Err(Error::Model(format!(
            "Hadamard input length {} is not a multiple of row_width {row_width}",
            values.len()
        )));
    }

    let scale = (row_width as f32).sqrt().recip();
    for row in values.chunks_exact_mut(row_width) {
        let mut span = 1usize;
        while span < row_width {
            let step = span * 2;
            let mut start = 0usize;
            while start < row_width {
                for offset in 0..span {
                    let left = start + offset;
                    let right = left + span;
                    let a = row[left];
                    let b = row[right];
                    row[left] = a + b;
                    row[right] = a - b;
                }
                start += step;
            }
            span = step;
        }
        for value in row {
            *value *= scale;
        }
    }
    Ok(())
}

fn rounded_power_of_two_scale(amax: f32, quant_max: f32) -> f32 {
    2.0f32.powf((amax / quant_max).log2().ceil())
}

fn quantize_fp8_e4m3fn_to_f32(value: f32) -> f32 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    let sign = if value.is_sign_negative() { -1.0 } else { 1.0 };
    let magnitude = value.abs().min(448.0);
    sign * nearest_fp8_e4m3fn_positive(magnitude)
}

fn nearest_fp8_e4m3fn_positive(magnitude: f32) -> f32 {
    let mut best = nearest_fp8_subnormal_positive(magnitude);
    let mut best_err = (best - magnitude).abs();
    let exp_floor = magnitude.log2().floor() as i32;
    for exp in exp_floor - 1..=exp_floor + 1 {
        if !(-6..=8).contains(&exp) {
            continue;
        }
        let scale = 2.0f32.powi(exp);
        let mut mantissa = ((magnitude / scale - 1.0) * 8.0).round() as i32;
        let mut candidate_exp = exp;
        if mantissa < 0 {
            continue;
        }
        if mantissa > 7 {
            candidate_exp += 1;
            mantissa = 0;
        }
        if candidate_exp > 8 {
            candidate_exp = 8;
            mantissa = 6;
        }
        if candidate_exp == 8 && mantissa > 6 {
            mantissa = 6;
        }
        let candidate = 2.0f32.powi(candidate_exp) * (1.0 + mantissa as f32 / 8.0);
        let err = (candidate - magnitude).abs();
        if err < best_err {
            best = candidate;
            best_err = err;
        }
    }
    best
}

fn nearest_fp8_subnormal_positive(magnitude: f32) -> f32 {
    let step = 2.0f32.powi(-9);
    let mantissa = (magnitude / step).round().clamp(0.0, 7.0);
    mantissa * step
}

fn quantize_fp4_e2m1_to_f32(value: f32) -> f32 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    let sign = if value.is_sign_negative() { -1.0 } else { 1.0 };
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = value.abs().min(6.0);
    let mut best = MAGNITUDES[0];
    let mut best_err = (magnitude - best).abs();
    for candidate in MAGNITUDES.into_iter().skip(1) {
        let err = (magnitude - candidate).abs();
        if err < best_err {
            best = candidate;
            best_err = err;
        }
    }
    sign * best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_e2m1_nibble_table_matches_expected_range() {
        let values = (0u8..16).map(decode_fp4_e2m1_nibble).collect::<Vec<_>>();
        assert_eq!(
            values,
            vec![
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0
            ]
        );
    }

    #[test]
    fn e8m0_scale_uses_power_of_two_bias() {
        assert_eq!(decode_e8m0_scale(127), 1.0);
        assert_eq!(decode_e8m0_scale(128), 2.0);
        assert_eq!(decode_e8m0_scale(126), 0.5);
    }

    #[test]
    fn fp8_e4m3fn_decodes_core_values() {
        assert_eq!(decode_fp8_e4m3fn_byte(0x00), 0.0);
        assert_eq!(decode_fp8_e4m3fn_byte(0x80), -0.0);
        assert_eq!(decode_fp8_e4m3fn_byte(0x38), 1.0);
        assert_eq!(decode_fp8_e4m3fn_byte(0xb8), -1.0);
        assert_eq!(decode_fp8_e4m3fn_byte(0x40), 2.0);
        assert_eq!(decode_fp8_e4m3fn_byte(0x7e), 448.0);
        assert!(decode_fp8_e4m3fn_byte(0x7f).is_nan());
    }

    #[test]
    fn dequantizes_tiny_fp4_matrix_with_one_scale_block() {
        let mut weight = vec![0u8; 16];
        weight[0] = 0x21; // low=0.5, high=1.0
        weight[1] = 0x87; // low=6.0, high=-0.0
        let scales = vec![127u8]; // scale 1.0
        let out = dequantize_fp4_e2m1_with_e8m0_scales(&weight, &scales, 1, 32, 32).unwrap();
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 6.0);
        assert_eq!(out[3], -0.0);
    }

    #[test]
    fn dequantizes_tiny_fp8_matrix_with_two_dimensional_scales() {
        let weight = vec![0x38, 0x40, 0xb8, 0x00, 0x38, 0x38];
        // out=2, in=3, block_m=1, block_k=2 => scales [2, 2]
        let scales = vec![127, 128, 126, 127];
        let out = dequantize_fp8_e4m3fn_with_e8m0_scales(&weight, &scales, 2, 3, 1, 2).unwrap();
        assert_eq!(out, vec![1.0, 2.0, -2.0, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn dequant_respects_e8m0_scale() {
        let mut weight = vec![0u8; 16];
        weight[0] = 0x22; // 1.0, 1.0
        let scales = vec![128u8]; // scale 2.0
        let out = dequantize_fp4_e2m1_with_e8m0_scales(&weight, &scales, 1, 32, 32).unwrap();
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 2.0);
    }

    #[test]
    fn fp8_activation_quant_uses_power_of_two_scale() {
        let mut values = vec![0.0f32; 128];
        values[0] = 448.0;
        values[1] = 224.0;
        values[2] = 500.0;
        values[3] = 1.3;
        simulate_fp8_e4m3fn_e8m0_activation_quant_in_place(&mut values, 128, 128).unwrap();
        assert_eq!(values[0], 448.0);
        assert_eq!(values[1], 224.0);
        assert_eq!(values[2], 512.0);
        assert!((values[3] - 1.25).abs() < 1e-6, "{}", values[3]);
    }

    #[test]
    fn fp4_activation_quant_uses_e2m1_grid() {
        let mut values = vec![0.0f32; 32];
        values[0] = 6.0;
        values[1] = 5.1;
        values[2] = 2.6;
        values[3] = -0.7;
        simulate_fp4_e2m1_e8m0_activation_quant_in_place(&mut values, 32, 32).unwrap();
        assert_eq!(values[0], 6.0);
        assert_eq!(values[1], 6.0);
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], -0.5);
    }

    #[test]
    fn normalized_hadamard_transform_is_row_local() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        normalized_hadamard_transform_rows_in_place(&mut values, 4).unwrap();
        assert_eq!(values, vec![5.0, -1.0, -2.0, 0.0, 5.0, 1.0, 2.0, 0.0]);
    }
}
