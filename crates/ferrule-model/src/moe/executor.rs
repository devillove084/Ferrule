//! Expert execution interfaces and reference implementations.
//!
//! This module is deliberately separate from expert residency/streaming. Streaming
//! decides *where bytes come from* and yields an `ExpertComputeBundle`; executors
//! decide *how to compute* with that bundle. The first implementation is a small
//! CPU reference path for packed FP4 + E8M0 expert fixtures. Production execution
//! should add CUDA executors that consume packed payloads directly instead of
//! dequantizing whole experts to f32.

use ferrule_common::{Error, Result};

use crate::artifact::format::dequantize_fp4_e2m1_with_e8m0_scales;
use crate::artifact::linear::ArtifactActivationQuantization;
use crate::moe::streaming::{ExpertComputeBundle, ExpertLinearFormat, ExpertLinearPayload};

/// Executes a single routed expert for one activation vector.
///
/// The interface is intentionally narrow: router/scheduler code owns selected
/// expert ids and weights, while this trait owns only the expert-local FFN math:
///
/// `down(silu(gate(x)) * up(x) * route_weight)`
///
/// Batch, top-k aggregation, shared-expert fusion, and CUDA residency handles can
/// be layered above this trait without making model-family names part of the
/// runtime API.
pub trait ExpertExecutor {
    fn execute(
        &self,
        bundle: &ExpertComputeBundle,
        input: &[f32],
        route_weight: f32,
    ) -> Result<Vec<f32>>;
}

/// CPU reference executor for correctness tests and tiny fixtures.
///
/// This is not a performance path. It expands packed FP4 matrices into f32 and is
/// only meant for small synthetic tensors or debugging slices. Full-size packed
/// experts are too large to run through this path in production.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuReferenceExpertExecutor {
    /// `0.0` disables clipping and matches plain SwiGLU experts; positive values
    /// apply model-family-specific SwiGLU clipping.
    pub swiglu_limit: f32,
    /// Optional activation quantization to apply before quantized expert linears.
    /// This is disabled by default for tiny fixtures and enabled by model-family
    /// boundaries whose official artifact contract requires quantized activations.
    pub activation_quantization: Option<ArtifactActivationQuantization>,
}

impl CpuReferenceExpertExecutor {
    pub fn new(swiglu_limit: f32) -> Self {
        Self {
            swiglu_limit,
            activation_quantization: None,
        }
    }

    pub fn with_activation_quantization(
        mut self,
        activation_quantization: ArtifactActivationQuantization,
    ) -> Self {
        self.activation_quantization = Some(activation_quantization);
        self
    }

    fn quantized_input(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut quantized = input.to_vec();
        if let Some(activation_quantization) = self.activation_quantization {
            activation_quantization.apply_in_place(&mut quantized, input.len())?;
        }
        Ok(quantized)
    }
}

impl Default for CpuReferenceExpertExecutor {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl ExpertExecutor for CpuReferenceExpertExecutor {
    fn execute(
        &self,
        bundle: &ExpertComputeBundle,
        input: &[f32],
        route_weight: f32,
    ) -> Result<Vec<f32>> {
        let quantized_input = self.quantized_input(input)?;
        let mut gate = reference_linear(&bundle.gate, &quantized_input)?;
        let mut up = reference_linear(&bundle.up, &quantized_input)?;
        if gate.len() != up.len() {
            return Err(Error::Model(format!(
                "expert layer {} expert {} gate/up mismatch: {} vs {}",
                bundle.expert.layer,
                bundle.expert.expert,
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

        let mut hidden = gate
            .iter()
            .zip(up.iter())
            .map(|(&gate, &up)| silu(gate) * up * route_weight)
            .collect::<Vec<_>>();
        if let Some(activation_quantization) = self.activation_quantization {
            let row_width = hidden.len();
            activation_quantization.apply_in_place(&mut hidden, row_width)?;
        }
        reference_linear(&bundle.down, &hidden)
    }
}

pub fn reference_linear(linear: &ExpertLinearPayload, input: &[f32]) -> Result<Vec<f32>> {
    match linear.format {
        ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features,
            in_features,
            block_size,
        } => {
            if input.len() != in_features {
                return Err(Error::Model(format!(
                    "expert {:?} input length mismatch: expected {in_features}, got {}",
                    linear.matrix,
                    input.len()
                )));
            }
            let scale = linear.scale.as_ref().ok_or_else(|| {
                Error::Model(format!(
                    "expert {:?} FP4 linear is missing E8M0 scale payload",
                    linear.matrix
                ))
            })?;
            let weights = dequantize_fp4_e2m1_with_e8m0_scales(
                &linear.weight.bytes,
                &scale.bytes,
                out_features,
                in_features,
                block_size,
            )?;
            Ok(matvec_row_major(&weights, out_features, in_features, input))
        }
        ExpertLinearFormat::Opaque => Err(Error::Model(format!(
            "expert {:?} linear format is opaque; no reference executor is available",
            linear.matrix
        ))),
    }
}

fn matvec_row_major(weights: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; rows];
    for row in 0..rows {
        let mut acc = 0.0f32;
        let offset = row * cols;
        for col in 0..cols {
            acc += weights[offset + col] * input[col];
        }
        output[row] = acc;
    }
    output
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::moe::streaming::{
        ExpertArtifactPayload, ExpertId, ExpertMatrixKind, ExpertTensorComponent, ExpertTensorKey,
        ExpertTensorPayload, ExpertTensorSlice,
    };

    #[test]
    fn cpu_reference_executor_runs_tiny_fp4_swiglu_expert() {
        let expert = ExpertId::new(2, 7);
        let bundle = ExpertComputeBundle::from_artifact_payload(ExpertArtifactPayload {
            expert,
            tensors: vec![
                tiny_fp4_payload(expert, ExpertMatrixKind::Gate, &[(0, 0, 4)]),
                tiny_scale_payload(expert, ExpertMatrixKind::Gate),
                tiny_fp4_payload(expert, ExpertMatrixKind::Up, &[(0, 1, 5)]),
                tiny_scale_payload(expert, ExpertMatrixKind::Up),
                tiny_fp4_payload(expert, ExpertMatrixKind::Down, &[(0, 0, 2)]),
                tiny_scale_payload(expert, ExpertMatrixKind::Down),
            ],
        })
        .unwrap();

        let mut input = vec![0.0f32; 32];
        input[0] = 1.0;
        input[1] = 1.0;

        let out = CpuReferenceExpertExecutor::default()
            .execute(&bundle, &input, 0.5)
            .unwrap();
        let expected = (2.0f32 / (1.0 + (-2.0f32).exp())) * 3.0 * 0.5;
        assert_eq!(out.len(), 32);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "{} vs {}",
            out[0],
            expected
        );
        assert!(out[1..].iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn cpu_reference_executor_applies_swiglu_limit() {
        let expert = ExpertId::new(0, 0);
        let bundle = ExpertComputeBundle::from_artifact_payload(ExpertArtifactPayload {
            expert,
            tensors: vec![
                tiny_fp4_payload(expert, ExpertMatrixKind::Gate, &[(0, 0, 7)]),
                tiny_scale_payload(expert, ExpertMatrixKind::Gate),
                tiny_fp4_payload(expert, ExpertMatrixKind::Up, &[(0, 1, 7)]),
                tiny_scale_payload(expert, ExpertMatrixKind::Up),
                tiny_fp4_payload(expert, ExpertMatrixKind::Down, &[(0, 0, 2)]),
                tiny_scale_payload(expert, ExpertMatrixKind::Down),
            ],
        })
        .unwrap();
        let mut input = vec![0.0f32; 32];
        input[0] = 2.0;
        input[1] = 2.0;

        let out = CpuReferenceExpertExecutor::new(10.0)
            .execute(&bundle, &input, 1.0)
            .unwrap();
        let expected = (10.0f32 / (1.0 + (-10.0f32).exp())) * 10.0;
        assert!(
            (out[0] - expected).abs() < 1e-4,
            "{} vs {}",
            out[0],
            expected
        );
    }

    #[test]
    fn cpu_reference_executor_rejects_wrong_activation_size() {
        let expert = ExpertId::new(0, 0);
        let bundle = ExpertComputeBundle::from_artifact_payload(ExpertArtifactPayload {
            expert,
            tensors: vec![
                tiny_fp4_payload(expert, ExpertMatrixKind::Gate, &[]),
                tiny_scale_payload(expert, ExpertMatrixKind::Gate),
                tiny_fp4_payload(expert, ExpertMatrixKind::Up, &[]),
                tiny_scale_payload(expert, ExpertMatrixKind::Up),
                tiny_fp4_payload(expert, ExpertMatrixKind::Down, &[]),
                tiny_scale_payload(expert, ExpertMatrixKind::Down),
            ],
        })
        .unwrap();
        let err = CpuReferenceExpertExecutor::default()
            .execute(&bundle, &[1.0, 2.0], 1.0)
            .unwrap_err();
        assert!(err.to_string().contains("input length mismatch"));
    }

    fn tiny_fp4_payload(
        expert: ExpertId,
        matrix: ExpertMatrixKind,
        nonzero_nibbles: &[(usize, usize, u8)],
    ) -> ExpertTensorPayload {
        let mut bytes = vec![0u8; 32 * 16];
        for &(row, col, nibble) in nonzero_nibbles {
            assert!(row < 32);
            assert!(col < 32);
            assert!(nibble < 16);
            let offset = row * 16 + col / 2;
            if col % 2 == 0 {
                bytes[offset] = (bytes[offset] & 0xf0) | nibble;
            } else {
                bytes[offset] = (bytes[offset] & 0x0f) | (nibble << 4);
            }
        }
        payload(
            expert,
            matrix,
            ExpertTensorComponent::Weight,
            "I8",
            vec![32, 16],
            bytes,
        )
    }

    fn tiny_scale_payload(expert: ExpertId, matrix: ExpertMatrixKind) -> ExpertTensorPayload {
        payload(
            expert,
            matrix,
            ExpertTensorComponent::Scale,
            "F8_E8M0",
            vec![32, 1],
            vec![127u8; 32],
        )
    }

    fn payload(
        expert: ExpertId,
        matrix: ExpertMatrixKind,
        component: ExpertTensorComponent,
        dtype: &str,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) -> ExpertTensorPayload {
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component,
                path: PathBuf::from("synthetic.safetensors"),
                offset: 0,
                bytes: bytes.len() as u64,
                dtype: dtype.into(),
                shape,
            },
            bytes,
        }
    }
}
