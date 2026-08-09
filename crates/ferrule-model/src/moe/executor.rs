//! Expert execution interfaces and model-to-backend adapters.
//!
//! Streaming decides where artifact bytes come from. Concrete expert matrix,
//! SwiGLU, and down-projection math belongs to `ferrule_backend::cpu`.

use ferrule_backend::cpu::{
    ExpertLinearRef, ExpertSwiGluRef, execute_reference_expert_with_hidden_transform, expert_linear,
};
use ferrule_common::{Error, Result};

use crate::checkpoint::weight::ActivationQuantization;
use crate::moe::streaming::{ExpertComputeBundle, ExpertLinearFormat, ExpertLinearPayload};

/// Executes a single routed expert for one activation vector.
pub trait ExpertExecutor {
    fn execute(
        &self,
        bundle: &ExpertComputeBundle,
        input: &[f32],
        route_weight: f32,
    ) -> Result<Vec<f32>>;
}

/// CPU reference executor adapter for correctness tests and tiny fixtures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuReferenceExpertExecutor {
    pub swiglu_limit: f32,
    pub activation_quantization: Option<ActivationQuantization>,
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
        activation_quantization: ActivationQuantization,
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
        let activation_quantization = self.activation_quantization;
        execute_reference_expert_with_hidden_transform(
            ExpertSwiGluRef {
                gate: backend_linear(&bundle.gate)?,
                up: backend_linear(&bundle.up)?,
                down: backend_linear(&bundle.down)?,
                activation_limit: (self.swiglu_limit > 0.0).then_some(self.swiglu_limit),
            },
            &quantized_input,
            route_weight,
            |hidden| {
                if let Some(activation_quantization) = activation_quantization {
                    activation_quantization.apply_in_place(hidden, hidden.len())?;
                }
                Ok(())
            },
        )
    }
}

/// Compatibility adapter for tests and callers that inspect one expert linear.
pub fn reference_linear(linear: &ExpertLinearPayload, input: &[f32]) -> Result<Vec<f32>> {
    expert_linear(backend_linear(linear)?, input)
}

fn backend_linear(linear: &ExpertLinearPayload) -> Result<ExpertLinearRef<'_>> {
    match linear.format {
        ExpertLinearFormat::Bf16 {
            out_features,
            in_features,
        } => {
            if linear.scale.is_some() {
                return Err(model_error(format!(
                    "expert {:?} BF16 linear has an unexpected scale payload",
                    linear.matrix
                )));
            }
            if linear.weight.slice.dtype != "BF16"
                || linear.weight.slice.shape.as_slice() != [out_features, in_features]
            {
                return Err(model_error(format!(
                    "expert {:?} BF16 payload mismatch: dtype={} shape={:?}, expected BF16 [{out_features}, {in_features}]",
                    linear.matrix, linear.weight.slice.dtype, linear.weight.slice.shape
                )));
            }
            Ok(ExpertLinearRef::Bf16 {
                weight: &linear.weight.bytes,
                out_features,
                in_features,
            })
        }
        ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features,
            in_features,
            block_size,
        } => {
            let scale = linear.scale.as_ref().ok_or_else(|| {
                model_error(format!(
                    "expert {:?} FP4 linear is missing E8M0 scale payload",
                    linear.matrix
                ))
            })?;
            Ok(ExpertLinearRef::Fp4E2M1E8M0 {
                weight: &linear.weight.bytes,
                scales: &scale.bytes,
                out_features,
                in_features,
                block_size,
            })
        }
        ExpertLinearFormat::Opaque => Err(model_error(format!(
            "expert {:?} linear format is opaque; no reference provider is available",
            linear.matrix
        ))),
    }
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: message.into(),
    }
}
