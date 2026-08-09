//! Provider-neutral linear and representation-conversion operations.

use ferrule_common::Result;

pub use crate::cuda::context::{
    CudaArtifactLinearHandle, CudaArtifactLinearShape, CudaArtifactLinearWorkspace, CudaBf16Buffer,
    CudaF32Buffer, CudaFp8ActivationPack, CudaOperators, CudaPreparedFp8Activation, cuda_gemv,
    cuda_gemv_fp8_e4m3fn_e8m0_2d,
};
pub use crate::cuda::operators::contracts::{
    F32ToBf16RowsLayout, StridedBf16RowsLayout, StridedF32RowsLayout,
};
pub use crate::cuda::providers::cutlass::{PROPOSAL_ROWS, ProposalHeadLayout};

impl CudaOperators {
    /// Convert strided F32 rows to BF16 using round-to-nearest-even.
    pub fn f32_rows_to_bf16_rne(
        &self,
        values: &CudaF32Buffer,
        layout: F32ToBf16RowsLayout,
    ) -> Result<CudaBf16Buffer> {
        self.f32_to_bf16_rne_rows_from_device(values, layout)
    }

    /// Convert strided F32 rows into an existing BF16 buffer.
    pub fn f32_rows_to_bf16_rne_into(
        &self,
        values: &CudaF32Buffer,
        output: &mut CudaBf16Buffer,
        layout: F32ToBf16RowsLayout,
    ) -> Result<()> {
        self.f32_to_bf16_rne_rows_from_device_into(values, output, layout)
    }
}
