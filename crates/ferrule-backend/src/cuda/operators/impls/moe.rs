//! Semantic moe execution using borrowed owner resources.

use crate::cuda::operators::OperatorOwner;
use crate::cuda::operators::linear::{
    CudaArtifactLinearHandle, CudaArtifactLinearShape, CudaF32Buffer, CudaFp8ActivationPack,
    CudaOperators, CudaPreparedFp8Activation,
};
use ferrule_common::{Error, Result};

impl CudaOperators {
    /// Execute the complete fused shared-expert gate/up -> SwiGLU -> down
    /// bundle and write directly to the caller-owned destination.
    #[allow(clippy::too_many_arguments)]
    pub fn artifact_shared_ffn_into(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &CudaPreparedFp8Activation<'_>,
        hidden_f32: &mut CudaF32Buffer,
        hidden: &mut CudaFp8ActivationPack,
        rows: usize,
        output_scale: f32,
        swiglu_limit: f32,
        output: &mut CudaF32Buffer,
        accumulate_output: bool,
    ) -> Result<()> {
        let gate = gate.operator_view();
        let up = up.operator_view();
        let down = down.operator_view();
        let input = input.operator_view();
        let hidden = hidden.operator_storage_mut();
        let (
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features: intermediate,
                in_features: input_size,
                block_m: gate_block_m,
                block_k: gate_block_k,
            },
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features: up_out,
                in_features: up_in,
                block_m: up_block_m,
                block_k: up_block_k,
            },
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features: output_size,
                in_features: down_in,
                block_m: down_block_m,
                block_k: down_block_k,
            },
        ) = (gate.shape, up.shape, down.shape)
        else {
            return Err(Error::Internal {
                message: format!(
                    "fused shared FFN requires FP8 weights: gate={:?} up={:?} down={:?}",
                    gate.shape, up.shape, down.shape
                ),
            });
        };
        if rows == 0
            || input_size != up_in
            || intermediate != up_out
            || intermediate != down_in
            || input.rows != rows
            || input.row_width != input_size
            || hidden_f32.len() < rows * intermediate
            || hidden.value_capacity != rows * intermediate
            || hidden.scale_capacity != rows * intermediate.div_ceil(128)
            || output.len() != rows * output_size
        {
            return Err(Error::Internal {
                message: format!(
                    "fused shared FFN shape mismatch: rows={rows} input=[{},{}] hidden_f32={} hidden=[{},{}] output={} gate={:?} up={:?} down={:?}",
                    input.rows,
                    input.row_width,
                    hidden_f32.len(),
                    hidden.value_capacity,
                    hidden.scale_capacity,
                    output.len(),
                    gate.shape,
                    up.shape,
                    down.shape
                ),
            });
        }
        let gate_scales = gate.scale.ok_or_else(|| Error::Internal {
            message: "fused shared gate scales are missing".into(),
        })?;
        let up_scales = up.scale.ok_or_else(|| Error::Internal {
            message: "fused shared up scales are missing".into(),
        })?;
        let down_scales = down.scale.ok_or_else(|| Error::Internal {
            message: "fused shared down scales are missing".into(),
        })?;
        self.submit_operator(1, |stream| {
            crate::cuda::providers::cutlass::submit_shared_ffn(
                stream,
                input.x_packed,
                input.x_scales,
                gate.weight,
                gate_scales,
                up.weight,
                up_scales,
                down.weight,
                down_scales,
                hidden_f32.as_device_buffer_mut(),
                hidden.x_packed,
                hidden.x_scales,
                output.as_device_buffer_mut(),
                rows,
                input_size,
                intermediate,
                output_size,
                (gate_block_m, gate_block_k),
                (up_block_m, up_block_k),
                (down_block_m, down_block_k),
                output_scale,
                swiglu_limit,
                accumulate_output,
            )
        })?;
        Ok(())
    }
}
