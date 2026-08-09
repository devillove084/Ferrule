//! Observable execution precision boundaries for inference graphs.
//!
//! A boundary is a tensor value that can be consumed by a later semantic
//! operator. Arithmetic internal to an operator remains its implementation
//! detail; only values crossing one of these boundaries are rounded. This is
//! important for reproducing Hugging Face BF16 eager execution without turning
//! every F32 accumulator operation into BF16 arithmetic.

/// Representation applied when a tensor crosses an execution boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryPrecision {
    /// Keep the F32 value produced by the operator.
    Preserve,
    /// Round to BF16 using round-to-nearest, ties-to-even, then expose it as F32.
    Bf16Rne,
}

/// Observable boundaries in a standard decoder execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecutionPrecisionBoundary {
    EmbeddingOutput,
    LinearInput,
    LinearOutput,
    NormInput,
    NormNormalized,
    NormOutput,
    RopeTable,
    RopeInput,
    RopeOutput,
    KvWrite,
    AttentionDot,
    AttentionScore,
    AttentionProbability,
    AttentionOutput,
    RouterInput,
    RouterLogits,
    RouterWeight,
    SwiGluInput,
    SwiGluGate,
    SwiGluUp,
    SwiGluActivation,
    SwiGluProduct,
    SwiGluOutput,
    ResidualInput,
    ResidualOutput,
    LmHeadOutput,
}

/// Model-independent policy for all observable execution precision boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionPrecisionPolicy {
    preset: ExecutionPrecisionPreset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ExecutionPrecisionPreset {
    F32,
    Bf16Compatibility,
}

impl ExecutionPrecisionPolicy {
    /// Preserve all values as F32 between operators.
    pub const fn f32() -> Self {
        Self {
            preset: ExecutionPrecisionPreset::F32,
        }
    }

    /// Match the observable BF16 tensor boundaries used by Hugging Face eager
    /// decoder execution while retaining F32 reductions and accumulators inside
    /// semantic operators.
    pub const fn bf16_compatibility() -> Self {
        Self {
            preset: ExecutionPrecisionPreset::Bf16Compatibility,
        }
    }

    pub const fn at(self, _boundary: ExecutionPrecisionBoundary) -> BoundaryPrecision {
        match self.preset {
            ExecutionPrecisionPreset::F32 => BoundaryPrecision::Preserve,
            ExecutionPrecisionPreset::Bf16Compatibility => BoundaryPrecision::Bf16Rne,
        }
    }

    /// Apply this policy at exactly one observable scalar boundary.
    pub fn apply(self, boundary: ExecutionPrecisionBoundary, value: f32) -> f32 {
        match self.at(boundary) {
            BoundaryPrecision::Preserve => value,
            BoundaryPrecision::Bf16Rne => bf16_rne(value),
        }
    }

    /// Apply this policy at exactly one observable tensor boundary.
    pub fn apply_slice(self, boundary: ExecutionPrecisionBoundary, values: &mut [f32]) {
        if self.at(boundary) == BoundaryPrecision::Bf16Rne {
            values
                .iter_mut()
                .for_each(|value| *value = bf16_rne(*value));
        }
    }
}

impl Default for ExecutionPrecisionPolicy {
    fn default() -> Self {
        Self::f32()
    }
}

/// IEEE-754 BF16 round-to-nearest, ties-to-even with quiet-NaN preservation.
pub fn bf16_rne(value: f32) -> f32 {
    f32::from_bits(u32::from(bf16_rne_word(value)) << 16)
}

/// Raw BF16 word for [`bf16_rne`].
pub fn bf16_rne_word(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let tie_to_even_bias = 0x7fff + ((bits >> 16) & 1);
    bits.wrapping_add(tie_to_even_bias).wrapping_shr(16) as u16
}

/// Expand one raw BF16 word into an exactly represented F32 value.
pub fn bf16_word_value(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_cover_every_boundary_consistently() {
        let boundaries = [
            ExecutionPrecisionBoundary::EmbeddingOutput,
            ExecutionPrecisionBoundary::LinearInput,
            ExecutionPrecisionBoundary::LinearOutput,
            ExecutionPrecisionBoundary::NormInput,
            ExecutionPrecisionBoundary::NormNormalized,
            ExecutionPrecisionBoundary::NormOutput,
            ExecutionPrecisionBoundary::RopeTable,
            ExecutionPrecisionBoundary::RopeInput,
            ExecutionPrecisionBoundary::RopeOutput,
            ExecutionPrecisionBoundary::KvWrite,
            ExecutionPrecisionBoundary::AttentionDot,
            ExecutionPrecisionBoundary::AttentionScore,
            ExecutionPrecisionBoundary::AttentionProbability,
            ExecutionPrecisionBoundary::AttentionOutput,
            ExecutionPrecisionBoundary::RouterInput,
            ExecutionPrecisionBoundary::RouterLogits,
            ExecutionPrecisionBoundary::RouterWeight,
            ExecutionPrecisionBoundary::SwiGluInput,
            ExecutionPrecisionBoundary::SwiGluGate,
            ExecutionPrecisionBoundary::SwiGluUp,
            ExecutionPrecisionBoundary::SwiGluActivation,
            ExecutionPrecisionBoundary::SwiGluProduct,
            ExecutionPrecisionBoundary::SwiGluOutput,
            ExecutionPrecisionBoundary::ResidualInput,
            ExecutionPrecisionBoundary::ResidualOutput,
            ExecutionPrecisionBoundary::LmHeadOutput,
        ];
        for boundary in boundaries {
            assert_eq!(
                ExecutionPrecisionPolicy::f32().at(boundary),
                BoundaryPrecision::Preserve
            );
            assert_eq!(
                ExecutionPrecisionPolicy::bf16_compatibility().at(boundary),
                BoundaryPrecision::Bf16Rne
            );
        }
    }

    #[test]
    fn bf16_uses_ties_to_even_and_quiets_nan() {
        assert_eq!(bf16_rne_word(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(bf16_rne_word(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(bf16_rne_word(f32::NAN) & 0x0040, 0x0040);
    }
}
