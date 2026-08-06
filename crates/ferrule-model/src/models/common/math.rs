//! Family-neutral tensor math: RMSNorm and small vector helpers.

use ferrule_common::{Error, Result};

/// Round one FP32 value to BF16 with round-to-nearest, ties-to-even, while
/// retaining the rounded value in FP32 storage.
pub(crate) fn round_to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return f32::from_bits((((bits >> 16) | 0x0040) & 0xffff) << 16);
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding_bias) & 0xffff_0000)
}

pub(crate) fn round_to_bf16_in_place(values: &mut [f32]) {
    values
        .iter_mut()
        .for_each(|value| *value = round_to_bf16(*value));
}

/// Row-wise RMSNorm: `out[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]`.
pub(crate) fn rms_norm(input: &[f32], weight: &[f32], eps: f32, label: &str) -> Result<Vec<f32>> {
    if input.len() != weight.len() || input.is_empty() {
        return Err(Error::Model {
            message: format!(
                "{label} RMS length mismatch: input={}, weight={}",
                input.len(),
                weight.len()
            ),
        });
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

/// In-place per-head RMSNorm over one packed `[heads, head_dim]` buffer.
pub(crate) fn rms_norm_heads_in_place(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    eps: f32,
    label: &str,
) -> Result<()> {
    if values.len() != heads * head_dim {
        return Err(Error::Model {
            message: format!(
                "{label} per-head RMS length mismatch: expected {}, got {}",
                heads * head_dim,
                values.len()
            ),
        });
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

/// Dot product of two equal-length slices. Callers own shape validation.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_uses_ties_to_even_and_quiets_nan() {
        assert_eq!(
            round_to_bf16(f32::from_bits(0x3f80_7fff)).to_bits(),
            0x3f80_0000
        );
        assert_eq!(
            round_to_bf16(f32::from_bits(0x3f80_8000)).to_bits(),
            0x3f80_0000
        );
        assert_eq!(
            round_to_bf16(f32::from_bits(0x3f81_8000)).to_bits(),
            0x3f82_0000
        );
        assert_eq!(round_to_bf16(f32::INFINITY), f32::INFINITY);
        assert_eq!(round_to_bf16(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(round_to_bf16(f32::from_bits(0x7f80_0001)).is_nan());
        assert_ne!(
            round_to_bf16(f32::from_bits(0x7f80_0001)).to_bits() & 0x0040_0000,
            0
        );
    }
}
