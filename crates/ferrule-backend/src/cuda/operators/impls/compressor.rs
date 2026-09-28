//! Semantic compressor execution using borrowed owner resources.

use crate::cuda::operators::OperatorOwner;
use crate::cuda::operators::linear::{
    CudaArtifactLinearHandle, CudaArtifactLinearShape, CudaF32Buffer, CudaOperators,
};
use ferrule_common::{Error, Result};

impl CudaOperators {
    /// Execute the BF16 compressor dual projection semantic operator.
    pub fn artifact_bf16_compressor_into(
        &self,
        projection1: &CudaArtifactLinearHandle,
        projection2: &CudaArtifactLinearHandle,
        activation: &CudaF32Buffer,
        rows: usize,
        projection1_output: &mut CudaF32Buffer,
        projection2_output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let projection1 = projection1.operator_view();
        let projection2 = projection2.operator_view();
        let (
            CudaArtifactLinearShape::Bf16Bytes {
                out_features: n1,
                in_features: k1,
            },
            CudaArtifactLinearShape::Bf16Bytes {
                out_features: n2,
                in_features: k2,
            },
        ) = (projection1.shape, projection2.shape)
        else {
            return Err(Error::Internal {
                message: format!(
                    "BF16 compressor requires BF16 weights, got first={:?} second={:?}",
                    projection1.shape, projection2.shape
                ),
            });
        };
        if k1 != k2 || activation.len() != rows * k1 {
            return Err(Error::Internal {
                message: format!(
                    "BF16 compressor input mismatch: rows={rows} first_k={k1} second_k={k2} input={}",
                    activation.len()
                ),
            });
        }
        if projection1_output.len() != rows * n1 || projection2_output.len() != rows * n2 {
            return Err(Error::Internal {
                message: format!(
                    "BF16 compressor output mismatch: first={}/{} second={}/{}",
                    projection1_output.len(),
                    rows * n1,
                    projection2_output.len(),
                    rows * n2
                ),
            });
        }
        self.submit_operator(1, |stream| {
            crate::cuda::providers::cutlass::submit_bf16_compressor(
                stream,
                activation.as_device_buffer(),
                &projection1.weight,
                &projection2.weight,
                projection1_output.as_device_buffer_mut(),
                projection2_output.as_device_buffer_mut(),
                rows,
                n1,
                n2,
                k1,
            )
        })?;
        Ok(())
    }
}
