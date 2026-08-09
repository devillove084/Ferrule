//! Compatibility adapters for family code using backend-owned CPU rotary math.

use ferrule_common::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RopeParams {
    pub theta: f32,
    pub original_seq_len: usize,
    pub factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
}

impl From<RopeParams> for ferrule_backend::cpu::RotaryFrequencyParams {
    fn from(parameters: RopeParams) -> Self {
        Self {
            theta: parameters.theta,
            original_sequence_length: parameters.original_seq_len,
            factor: parameters.factor,
            beta_fast: parameters.beta_fast,
            beta_slow: parameters.beta_slow,
        }
    }
}

pub fn apply_rotary_split_half(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    theta: f32,
) -> Result<()> {
    apply_rotary_split_half_indexed(values, 1, heads, head_dim, rotary_dim, &[position], theta)
}

pub fn apply_rotary_split_half_indexed(
    values: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    positions: &[usize],
    theta: f32,
) -> Result<()> {
    ferrule_backend::cpu::apply_rotary_split_half_indexed(
        values, rows, heads, head_dim, rotary_dim, positions, theta,
    )
}

pub(crate) fn yarn_frequency(pair: usize, rope_dim: usize, rope: RopeParams) -> f32 {
    ferrule_backend::cpu::rotary_frequency(pair, rope_dim, rope.into())
}
