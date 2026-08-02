//! Family-neutral RoPE/YaRN frequency geometry and rotary application.
//!
//! Works on any packed `[heads, head_dim]` buffer where the rotary pairs live
//! in the trailing `rope_dim` channels of each head row. Model families only
//! contribute their parameter values through the crate-private `RopeParams`.

use ferrule_common::{Error, Result};

/// Rotary embedding parameters. `original_seq_len == 0` or `factor == 1.0`
/// selects the plain (unscaled) frequency schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RopeParams {
    pub theta: f32,
    pub original_seq_len: usize,
    pub factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
}

impl RopeParams {
    pub fn plain(theta: f32) -> Self {
        Self {
            theta,
            original_seq_len: 0,
            factor: 1.0,
            beta_fast: 32,
            beta_slow: 1,
        }
    }
}

/// Apply the plain (unscaled) rotary embedding to the trailing `rope_dim`
/// channels of every head row.
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
        RopeParams::plain(theta),
        inverse,
    )
}

/// Apply the (optionally YaRN-scaled) rotary embedding to the trailing
/// `rope_dim` channels of every head row.
pub(crate) fn apply_rotary_tail_scaled(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    rope: RopeParams,
    inverse: bool,
) -> Result<()> {
    if rope_dim == 0 {
        return Ok(());
    }
    if rope_dim > head_dim || !rope_dim.is_multiple_of(2) || values.len() != heads * head_dim {
        return Err(Error::Model {
            message: format!(
                "rotary shape mismatch: values={}, heads={heads}, head_dim={head_dim}, rope_dim={rope_dim}",
                values.len()
            ),
        });
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

/// Frequency of one rotary pair, with YaRN range correction when the
/// parameters select a scaled schedule.
pub(crate) fn yarn_frequency(pair: usize, rope_dim: usize, rope: RopeParams) -> f32 {
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
