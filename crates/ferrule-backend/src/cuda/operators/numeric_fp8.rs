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
    /// `plan.precision()`. W8A16 uses zero global scratch for every M. F32 M1
    /// needs zero bytes, while F32 rows > 1 need
    /// `4*K*tile_rows` scratch bytes:
    /// its caller-owned F32 activation is read directly, without a staging copy.
    /// The existing allocator may reserve larger pooled segments; that allocator
    /// reserve is distinct from live operator scratch and is not hidden here.
    pub fn numeric_fp8_linear_workspace(
        &self,
        plan: NumericFp8LinearPlan,
    ) -> Result<CudaNumericFp8Workspace> {
        let gemv = if plan.provider_bytes() != 0 {
            Some(self.numeric_gemv.borrow().clone().ok_or_else(||
                ferrule_common::Error::Internal { message:
                    "BF16 GEMV requires explicit provider budget preparation before scratch allocation".into() })?)
        } else {
            None
        };
        let mut workspace = CudaNumericFp8Workspace::from_buffer_with_precision(
            self.uninitialized_device_buffer::<u8>(plan.workspace_requirements().bytes as usize)?,
            plan.precision(),
        );
        workspace.gemv = gemv;
        Ok(workspace)
    }
    /// Prepare persistent provider resources once on this owner, outside capture.
    /// The caller must admit `plan.provider_bytes()` IN ADDITION to numeric scratch.
    /// Fixed warmup scratch uses the plan's grant and is retired before return.
    pub fn prepare_numeric_fp8_provider(
        &self,
        plan: NumericFp8LinearPlan,
        provider_budget_bytes: usize,
    ) -> Result<()> {
        let required = plan.provider_bytes();
        if required == 0 {
            return Ok(());
        }
        plan.validate_provider_budget(provider_budget_bytes)?;
        self.check_capture_safe("BF16 GEMV provider initialization")?;
        let mut provider = self.numeric_gemv.borrow_mut();
        if provider.is_none() {
            let storage = self
                .uninitialized_device_buffer::<u8>(plan.workspace_requirements().bytes as usize)?;
            match crate::cuda::providers::cutlass::NumericBf16Gemv::new(&self.stream, &storage) {
                Ok(value) => *provider = Some(value),
                Err(error) => {
                    if self._ctx.graph_completion_unknown() {
                        std::mem::forget(storage);
                    }
                    return Err(error);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/numeric_bf16_gemv/gpu.rs"]
mod gemv_tests;

impl CudaOperators {
    /// Allocate only after both caller storage and external provider budgets
    /// have been admitted by the explicit plan. No existing profile selects it.
    pub fn prepare_bf16_gemmex(
        &self,
        plan: crate::cuda_plan::bf16_gemmex::Bf16GemmExPlan,
    ) -> Result<crate::cuda::operators::linear::CudaBf16GemmEx> {
        self.prepare_bf16_gemmex_plans(&[plan])
    }

    pub(crate) fn prepare_bf16_gemmex_plans(
        &self,
        plans: &[crate::cuda_plan::bf16_gemmex::Bf16GemmExPlan],
    ) -> Result<crate::cuda::operators::linear::CudaBf16GemmEx> {
        use crate::cuda::providers::bf16_gemmex::{self, NativeBf16GemmEx};
        self.check_capture_safe("BF16 GemmEx preparation")?;
        bf16_gemmex::available()?;
        let plan = *plans
            .iter()
            .max_by_key(|p| p.storage_bytes())
            .ok_or_else(|| bf16_gemmex::invalid("empty shape set"))?;
        bf16_gemmex::Admission::prepare(&self.stream, plan, |admission| {
            let storage = self.uninitialized_device_buffer::<u8>(plan.storage_bytes())?;
            let native = NativeBf16GemmEx::new(&self.stream, storage, plan, admission)?;
            let native = NativeBf16GemmEx::prepare_views(&self.stream, native, plans)?;
            Ok(crate::cuda::operators::linear::CudaBf16GemmEx { native })
        })
    }
}
