//! Family-neutral tensor math: RMSNorm and small vector helpers.

use ferrule_common::{Error, Result};

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
