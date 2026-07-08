//! Feed-forward network reference helpers.
//!
//! This module is the generic dense/shared FFN counterpart to routed expert
//! execution. Shared experts that use the same SwiGLU structure as routed experts
//! should be represented as ordinary semantic artifact linears rather than fake
//! routed expert ids.

use ferrule_common::{Error, Result};

use crate::artifact_linear::ArtifactLinearPayload;

#[derive(Debug, Clone, PartialEq)]
pub struct SwiGluFfnPayload {
    pub gate: ArtifactLinearPayload,
    pub up: ArtifactLinearPayload,
    pub down: ArtifactLinearPayload,
    /// `0.0` disables clipping; positive values apply model-family-specific clipping.
    pub swiglu_limit: f32,
}

impl SwiGluFfnPayload {
    pub fn reference_execute(&self, input: &[f32], output_scale: f32) -> Result<Vec<f32>> {
        let mut gate = self.gate.reference_matvec(input)?;
        let mut up = self.up.reference_matvec(input)?;
        if gate.len() != up.len() {
            return Err(Error::Model(format!(
                "SwiGLU gate/up mismatch: {} vs {}",
                gate.len(),
                up.len()
            )));
        }

        if self.swiglu_limit > 0.0 {
            for value in &mut gate {
                *value = value.min(self.swiglu_limit);
            }
            for value in &mut up {
                *value = value.clamp(-self.swiglu_limit, self.swiglu_limit);
            }
        }

        let hidden = gate
            .iter()
            .zip(up.iter())
            .map(|(&gate, &up)| silu(gate) * up * output_scale)
            .collect::<Vec<_>>();
        self.down.reference_matvec(&hidden)
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::TensorRole;

    use super::*;
    use crate::artifact_linear::ArtifactLinearPayload;
    use crate::artifact_tensor::{ArtifactDType, ArtifactTensorPayload, ArtifactTensorSlice};

    #[test]
    fn swiglu_ffn_reference_executes_f32_linears() {
        let ffn = SwiGluFfnPayload {
            gate: f32_linear(
                TensorRole::SharedExpertGate,
                "gate",
                2,
                3,
                &[1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
            up: f32_linear(
                TensorRole::SharedExpertUp,
                "up",
                2,
                3,
                &[0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ),
            down: f32_linear(
                TensorRole::SharedExpertDown,
                "down",
                3,
                2,
                &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            swiglu_limit: 0.0,
        };
        let out = ffn.reference_execute(&[1.0, 2.0, 3.0], 0.5).unwrap();
        let gate0 = 4.0f32;
        let up0 = 5.0f32;
        let expected0 = gate0 / (1.0 + (-gate0).exp()) * up0 * 0.5;
        assert!(
            (out[0] - expected0).abs() < 1e-6,
            "{} vs {}",
            out[0],
            expected0
        );
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn swiglu_ffn_applies_limit_before_down_projection() {
        let ffn = SwiGluFfnPayload {
            gate: f32_linear(TensorRole::SharedExpertGate, "gate", 1, 1, &[20.0]),
            up: f32_linear(TensorRole::SharedExpertUp, "up", 1, 1, &[20.0]),
            down: f32_linear(TensorRole::SharedExpertDown, "down", 1, 1, &[1.0]),
            swiglu_limit: 10.0,
        };
        let out = ffn.reference_execute(&[1.0], 1.0).unwrap();
        let expected = 10.0f32 / (1.0 + (-10.0f32).exp()) * 10.0;
        assert!(
            (out[0] - expected).abs() < 1e-4,
            "{} vs {}",
            out[0],
            expected
        );
    }

    fn f32_linear(
        role: TensorRole,
        name: &str,
        out: usize,
        input: usize,
        values: &[f32],
    ) -> ArtifactLinearPayload {
        assert_eq!(values.len(), out * input);
        ArtifactLinearPayload::from_weight_and_scale(
            role,
            ArtifactTensorPayload {
                slice: ArtifactTensorSlice {
                    name: format!("{name}.weight"),
                    role: TensorRole::Unknown,
                    path: PathBuf::from("synthetic.safetensors"),
                    offset: 0,
                    bytes: (values.len() * 4) as u64,
                    dtype: ArtifactDType::F32,
                    shape: vec![out, input],
                },
                bytes: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            },
            None,
        )
        .unwrap()
    }
}
