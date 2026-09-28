//! Numeric FP8 storage bridge on the existing CUDA context owner. Kept separate
//! from E8M0 artifacts. Precision is explicitly selected in the budget plan;
//! existing BF16 callers keep their original conversion boundary.
use super::CudaOperators;
use crate::cuda::operators::linear::{
    CudaNumericFp8Artifact, CudaNumericFp8Workspace, NumericFp8Layout, NumericFp8LinearPlan,
};
use ferrule_common::Result;
use ferrule_common::numeric_fp8::ImmutableValidatedNumericFp8Payload;

impl CudaOperators {
    /// Validate immutable packed FP8 and numeric scales before any H2D/allocation.
    /// Scales stay in their original BF16/F32 representation on device.
    pub fn upload_numeric_fp8_linear(
        &self,
        layout: NumericFp8Layout,
        weight: &[u8],
        scales: &[u8],
    ) -> Result<CudaNumericFp8Artifact> {
        layout.validate_payload(weight, scales)?;
        self.upload_numeric_fp8_validated_bytes(layout, weight, scales)
    }

    /// Reuse immutable CPU payload validation without rescanning its contents.
    /// This proof contains no device/source identity. Exact CUDA owner, plan,
    /// spans, overlap and workspace budget are still checked before launch.
    pub fn upload_validated_numeric_fp8_linear(
        &self,
        payload: &ImmutableValidatedNumericFp8Payload,
    ) -> Result<CudaNumericFp8Artifact> {
        let layout = payload.layout();
        layout.validate_lengths(payload.weight_bytes().len(), payload.scale_bytes().len())?;
        self.upload_numeric_fp8_validated_bytes(
            layout,
            payload.weight_bytes(),
            payload.scale_bytes(),
        )
    }

    fn upload_numeric_fp8_validated_bytes(
        &self,
        layout: NumericFp8Layout,
        weight: &[u8],
        scales: &[u8],
    ) -> Result<CudaNumericFp8Artifact> {
        let artifact = CudaNumericFp8Artifact::from_device_buffers(
            layout,
            self.upload_u8(weight)?,
            self.upload_u8(scales)?,
        );
        self.counters
            .add_artifact_upload(artifact.storage_bytes() as u64);
        Ok(artifact)
    }

    /// Allocate exactly the plan's workspace view, within its explicit budget.
    /// Workspace precision is fixed at construction; cache/reuse keys must include
    /// `plan.precision()`. F32Tf32x3 needs only `4*K*tile_rows` scratch bytes:
    /// its caller-owned F32 activation is read directly, without a staging copy.
    /// The existing allocator may reserve larger pooled segments; that allocator
    /// reserve is distinct from live operator scratch and is not hidden here.
    pub fn numeric_fp8_linear_workspace(
        &self,
        plan: NumericFp8LinearPlan,
    ) -> Result<CudaNumericFp8Workspace> {
        Ok(CudaNumericFp8Workspace::from_buffer_with_precision(
            self.uninitialized_device_buffer::<u8>(plan.workspace_requirements().bytes as usize)?,
            plan.precision(),
        ))
    }
}
