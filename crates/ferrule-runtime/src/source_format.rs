//! Source weight-format helpers for reference tests and tiny fixtures.
//!
//! These routines are intentionally correctness/debug utilities, not the final
//! high-throughput path. Production execution should consume packed FP4 + E8M0
//! scales directly in backend kernels instead of expanding experts to F32.

use ferrule_core::{Error, Result};

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
    if block_size == 0 || in_features % block_size != 0 || in_features % 2 != 0 {
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
}
