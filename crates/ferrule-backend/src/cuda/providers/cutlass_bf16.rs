//! Tiled BF16/F32, fused M1 F32 and fused multirow W8A16 on the existing provider.
//! Numeric FP8 is storage only: GPU decode/multiply, not native FP8 MMA.

use crate::cuda::operator_owner::cu;
use crate::cuda::operators::OperatorWorkspaceRequirements;
use crate::cuda::runtime::{CudaStream, DeviceBuffer, DeviceCopy};
use ferrule_common::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("CUTLASS BF16/numeric FP8: {}", message.into()),
    }
}
fn checked(value: Option<usize>) -> Result<usize> {
    value
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or_else(|| invalid("shape/size overflow"))
}
fn status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(invalid("invalid native layout")),
        2 => Err(invalid(
            "requested numeric kernel unavailable; no implicit fallback",
        )),
        3 => Err(invalid("unsupported TensorOp shape")),
        _ => Err(invalid(format!(
            "native submission failed ({code}); completion unknown"
        ))),
    }
}
pub(super) const fn kernel_for_precision(precision: NumericFp8Precision) -> super::CutlassKernelId {
    match precision {
        NumericFp8Precision::Bf16M1GemvF32Accumulate => super::CutlassKernelId::NumericBf16M1Gemv,
        NumericFp8Precision::Bf16RneF32Accumulate
        | NumericFp8Precision::Bf16SlicedK2F32Accumulate => super::CutlassKernelId::Bf16Gemm,
        NumericFp8Precision::F32Tf32x3 => super::CutlassKernelId::F32Gemm,
        NumericFp8Precision::W8A16Bf16F32Accumulate => super::CutlassKernelId::NumericFp8W8A16Fused,
    }
}

fn available(precision: NumericFp8Precision) -> Result<()> {
    if super::discover_provider()?.supports(kernel_for_precision(precision)) {
        Ok(())
    } else {
        status(2)
    }
}

use crate::cuda::ffi::cutlass::{
    Bf16Args, NumericArgs, ferrule_cutlass_bf16_can_implement, ferrule_cutlass_bf16_launch,
    ferrule_cutlass_numeric_fp8_bf16_can_implement, ferrule_cutlass_numeric_fp8_bf16_launch,
    ferrule_cutlass_numeric_fp8_can_implement, ferrule_cutlass_numeric_fp8_f32_can_implement,
    ferrule_cutlass_numeric_fp8_f32_launch, ferrule_cutlass_numeric_fp8_launch,
    ferrule_cutlass_numeric_fp8_m1_can_implement, ferrule_cutlass_numeric_fp8_m1_launch,
    ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement,
    ferrule_cutlass_numeric_fp8_sliced_k2_bf16_launch,
    ferrule_cutlass_numeric_fp8_sliced_k2_can_implement,
    ferrule_cutlass_numeric_fp8_sliced_k2_launch,
    ferrule_cutlass_numeric_fp8_w8a16_bf16_can_implement,
    ferrule_cutlass_numeric_fp8_w8a16_bf16_launch, ferrule_cutlass_numeric_fp8_w8a16_can_implement,
    ferrule_cutlass_numeric_fp8_w8a16_launch,
};

use crate::cuda::operators::linear::{
    Bf16GemmLayout, CudaNumericFp8Artifact, CudaNumericFp8Workspace, NumericFp8LinearPlan,
    NumericFp8Precision, NumericFp8SelectedKernel,
};

/// Compiled implementation capability, not a device-owner or operand proof.
/// The legacy F32Gemm manifest bit describes TF32x3 only. Fused numeric M1 is
/// reported by its own bit so callers cannot mistake it for that implementation.
/// The W8A16 family bit covers both shape-selected BF16 vector and MMA kernels;
/// it never admits a legacy BF16 or F32 precision fallback.
pub fn numeric_fp8_kernel_available(kernel: NumericFp8SelectedKernel) -> Result<bool> {
    let id = match kernel {
        NumericFp8SelectedKernel::F32FusedM1 => super::CutlassKernelId::NumericFp8FusedM1,
        NumericFp8SelectedKernel::F32TiledTf32x3 => super::CutlassKernelId::F32Gemm,
        NumericFp8SelectedKernel::Bf16M1Gemv => super::CutlassKernelId::NumericBf16M1Gemv,
        NumericFp8SelectedKernel::Bf16TiledGemm
        | NumericFp8SelectedKernel::Bf16TiledSlicedK2Gemm => super::CutlassKernelId::Bf16Gemm,
        NumericFp8SelectedKernel::W8A16FusedBf16Mma
        | NumericFp8SelectedKernel::W8A16FusedBf16Vector => {
            super::CutlassKernelId::NumericFp8W8A16Fused
        }
    };
    Ok(super::discover_provider()?.supports(id))
}

fn extent(rows: usize, width: usize, stride: usize, element: usize) -> Result<usize> {
    if rows == 0 || width == 0 || stride < width || stride > i32::MAX as usize {
        return Err(invalid("invalid row stride/extent"));
    }
    checked(
        (rows - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(width))
            .and_then(|v| v.checked_mul(element)),
    )
}
fn buffer<T: DeviceCopy>(
    stream: &CudaStream,
    b: &DeviceBuffer<T>,
    bytes: usize,
    align: u64,
) -> Result<(u64, u64)> {
    cu(b.check_context(stream.context(), "CUTLASS BF16/numeric FP8"))?;
    if b.num_bytes() < bytes || !b.cu_deviceptr().is_multiple_of(align) {
        return Err(invalid("buffer too small or misaligned"));
    }
    let start = b.cu_deviceptr();
    Ok((
        start,
        start
            .checked_add(bytes as u64)
            .ok_or_else(|| invalid("address overflow"))?,
    ))
}
fn no_overlap(reads: &[(u64, u64)], writes: &[(u64, u64)]) -> Result<()> {
    for (i, &(start, end)) in writes.iter().enumerate() {
        if reads
            .iter()
            .chain(&writes[..i])
            .any(|&(a, b)| start < end && a < b && start < b && a < end)
        {
            return Err(invalid("overlapping input/output/workspace ranges"));
        }
    }
    Ok(())
}
pub fn bf16_gemm_workspace_requirements(
    layout: Bf16GemmLayout,
) -> Result<OperatorWorkspaceRequirements> {
    layout.validate()?;
    available(NumericFp8Precision::Bf16RneF32Accumulate)?;
    Ok(OperatorWorkspaceRequirements {
        bytes: 0,
        alignment: 1,
    })
}
fn bf16_args(
    stream: &CudaStream,
    activation: &DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    output: &DeviceBuffer<f32>,
    l: Bf16GemmLayout,
) -> Result<Bf16Args> {
    bf16_gemm_workspace_requirements(l)?;
    let a = buffer(
        stream,
        activation,
        extent(l.rows, l.k, l.activation_stride, 2)?,
        16,
    )?;
    let b = buffer(stream, weight, extent(l.n, l.k, l.weight_stride, 2)?, 16)?;
    let d = buffer(stream, output, extent(l.rows, l.n, l.output_stride, 4)?, 4)?;
    no_overlap(&[a, b], &[d])?;
    Ok(Bf16Args {
        m: l.rows as u32,
        n: l.n as u32,
        k: l.k as u32,
        lda: l.activation_stride as u32,
        ldb: l.weight_stride as u32,
        ldd: l.output_stride as u32,
        activation: a.0,
        weight: b.0,
        output: d.0,
        stream: stream.cu_stream() as usize as u64,
    })
}
pub fn bf16_gemm_can_implement(
    stream: &CudaStream,
    activation: &DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    output: &DeviceBuffer<f32>,
    layout: Bf16GemmLayout,
) -> Result<()> {
    let args = bf16_args(stream, activation, weight, output, layout)?;
    status(unsafe { ferrule_cutlass_bf16_can_implement(&args) })
}
/// Enqueue on the supplied owner stream. Retain operands through completion;
/// cross-stream consumers must wait for a completion event recorded afterwards.
pub fn bf16_gemm(
    stream: &CudaStream,
    activation: &DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    layout: Bf16GemmLayout,
) -> Result<()> {
    let args = bf16_args(stream, activation, weight, output, layout)?;
    status(unsafe { ferrule_cutlass_bf16_can_implement(&args) })?;
    cu(stream.context().bind_to_thread())?;
    status(unsafe { ferrule_cutlass_bf16_launch(&args) })
}

/// Strides are F32 elements; slice input/output buffers to supply base offsets.
/// Each output tile writes D[:,first..first+count] with the SAME full D stride.
/// One completion event recorded after this call covers every decode/GEMM tile.
/// Retain operands and scratch until completion; before cross-stream scratch
/// reuse or output consumption, wait for that exact event on the next stream.
#[allow(clippy::too_many_arguments)]
pub fn numeric_fp8_linear(
    stream: &CudaStream,
    artifact: &CudaNumericFp8Artifact,
    activation: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    workspace: &mut CudaNumericFp8Workspace,
    plan: NumericFp8LinearPlan,
    activation_stride: usize,
    output_stride: usize,
) -> Result<()> {
    numeric_fp8_submit(
        stream,
        artifact,
        activation,
        output,
        workspace,
        plan,
        activation_stride,
        output_stride,
        false,
    )
}

/// Explicit BF16-input owner path for fused W8A16 or bounded tiled BF16 GEMM.
/// No F32 input conversion, allocation, fallback, or synchronization.
#[allow(clippy::too_many_arguments)]
pub(crate) fn numeric_fp8_bf16_linear(
    stream: &CudaStream,
    artifact: &CudaNumericFp8Artifact,
    activation: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    workspace: &mut CudaNumericFp8Workspace,
    plan: NumericFp8LinearPlan,
    activation_stride: usize,
    output_stride: usize,
) -> Result<()> {
    if !matches!(
        plan.precision(),
        NumericFp8Precision::W8A16Bf16F32Accumulate
            | NumericFp8Precision::Bf16RneF32Accumulate
            | NumericFp8Precision::Bf16SlicedK2F32Accumulate
            | NumericFp8Precision::Bf16M1GemvF32Accumulate
    ) {
        return Err(invalid(
            "BF16 input requires explicit W8A16 or tiled BF16 profile",
        ));
    }
    numeric_fp8_submit(
        stream,
        artifact,
        activation,
        output,
        workspace,
        plan,
        activation_stride,
        output_stride,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn numeric_fp8_submit<T: DeviceCopy>(
    stream: &CudaStream,
    artifact: &CudaNumericFp8Artifact,
    activation: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<f32>,
    workspace: &mut CudaNumericFp8Workspace,
    plan: NumericFp8LinearPlan,
    activation_stride: usize,
    output_stride: usize,
    bf16_input: bool,
) -> Result<()> {
    if workspace.is_poisoned() {
        return Err(invalid("workspace quarantined after unknown submission"));
    }
    if artifact.layout() != plan.layout() {
        return Err(invalid("artifact/plan layout mismatch"));
    }
    if workspace.precision() != plan.precision() {
        return Err(invalid("workspace/plan precision profile mismatch"));
    }
    if !numeric_fp8_kernel_available(plan.selected_kernel())? {
        return status(2);
    }
    let l = plan.layout();
    let (weight, scales) = artifact.buffers();
    l.validate_lengths(weight.num_bytes(), scales.num_bytes())?;
    let (w, s) = l.storage_lengths()?;
    let a = buffer(
        stream,
        activation,
        extent(
            plan.rows(),
            l.k,
            activation_stride,
            std::mem::size_of::<T>(),
        )?,
        std::mem::size_of::<T>() as u64,
    )?;
    let w = buffer(stream, weight, w, 1)?;
    let s = buffer(stream, scales, s, l.scale_type.element_bytes() as u64)?;
    let d = buffer(
        stream,
        output,
        extent(plan.rows(), l.n, output_stride, 4)?,
        4,
    )?;
    // Oversized supplied storage must not make the declared budget fictitious.
    if workspace.allocated_bytes() > plan.scratch_budget_bytes() {
        return Err(invalid("workspace allocation exceeds explicit budget"));
    }
    // Retained compatibility storage still has owner/alignment/alias obligations
    // even when a fused kernel does not touch it. Empty spans never alias operands.
    let scratch_bytes = if matches!(
        plan.selected_kernel(),
        NumericFp8SelectedKernel::F32FusedM1
            | NumericFp8SelectedKernel::W8A16FusedBf16Mma
            | NumericFp8SelectedKernel::W8A16FusedBf16Vector
    ) {
        workspace.allocated_bytes()
    } else {
        plan.workspace_requirements().bytes as usize
    };
    let scratch = buffer(stream, workspace.storage(), scratch_bytes, 16)?;
    no_overlap(&[a, w, s], &[d, scratch])?;
    let args = NumericArgs {
        m: plan.rows() as u32,
        n: l.n as u32,
        k: l.k as u32,
        padded_k: plan.padded_k() as u32,
        tile_rows: plan.tile_rows() as u32,
        scale_cols: l.scale_shape()?[1] as u32,
        row_origin: (l.row_origin % 128) as u32,
        column_origin: (l.column_origin % 128) as u32,
        scale_bytes: l.scale_type.element_bytes() as u32,
        lda: activation_stride as u32,
        ldd: output_stride as u32,
        reserved: 0,
        activation: a.0,
        weight: w.0,
        scales: s.0,
        output: d.0,
        workspace: scratch.0,
        workspace_bytes: plan.workspace_requirements().bytes,
        stream: stream.cu_stream() as usize as u64,
    };
    if plan.selected_kernel() == NumericFp8SelectedKernel::Bf16M1Gemv {
        cu(stream.context().check_graph_ready())?;
        if !bf16_input {
            return Err(invalid(
                "BF16 GEMV requires typed BF16 input; no implicit F32 conversion",
            ));
        }
        let provider = workspace
            .gemv
            .as_ref()
            .ok_or_else(|| invalid("BF16 GEMV requires owner-prepared provider resources"))?;
        if !std::ptr::eq(provider.stream.as_ref(), stream) {
            return Err(invalid("BF16 GEMV provider stream owner mismatch"));
        }
        status(unsafe {
            crate::cuda::ffi::cutlass::ferrule_numeric_bf16_gemv_can_implement(&args)
        })?;
        cu(stream.context().bind_to_thread())?;
        let result = status(unsafe {
            crate::cuda::ffi::cutlass::ferrule_numeric_bf16_gemv_launch(&args, provider.raw)
        });
        if result.is_err() {
            workspace.poison();
        }
        return result;
    }
    let (preflight, launch): (
        unsafe extern "C" fn(*const NumericArgs) -> i32,
        unsafe extern "C" fn(*const NumericArgs) -> i32,
    ) = match plan.selected_kernel() {
        NumericFp8SelectedKernel::Bf16M1Gemv => unreachable!(),
        NumericFp8SelectedKernel::Bf16TiledSlicedK2Gemm if bf16_input => (
            ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement,
            ferrule_cutlass_numeric_fp8_sliced_k2_bf16_launch,
        ),
        NumericFp8SelectedKernel::Bf16TiledSlicedK2Gemm => (
            ferrule_cutlass_numeric_fp8_sliced_k2_can_implement,
            ferrule_cutlass_numeric_fp8_sliced_k2_launch,
        ),
        NumericFp8SelectedKernel::Bf16TiledGemm if bf16_input => (
            ferrule_cutlass_numeric_fp8_bf16_can_implement,
            ferrule_cutlass_numeric_fp8_bf16_launch,
        ),
        NumericFp8SelectedKernel::Bf16TiledGemm => (
            ferrule_cutlass_numeric_fp8_can_implement,
            ferrule_cutlass_numeric_fp8_launch,
        ),
        NumericFp8SelectedKernel::F32TiledTf32x3 => (
            ferrule_cutlass_numeric_fp8_f32_can_implement,
            ferrule_cutlass_numeric_fp8_f32_launch,
        ),
        NumericFp8SelectedKernel::F32FusedM1 => (
            ferrule_cutlass_numeric_fp8_m1_can_implement,
            ferrule_cutlass_numeric_fp8_m1_launch,
        ),
        NumericFp8SelectedKernel::W8A16FusedBf16Mma
        | NumericFp8SelectedKernel::W8A16FusedBf16Vector
            if bf16_input =>
        {
            (
                ferrule_cutlass_numeric_fp8_w8a16_bf16_can_implement,
                ferrule_cutlass_numeric_fp8_w8a16_bf16_launch,
            )
        }
        NumericFp8SelectedKernel::W8A16FusedBf16Mma
        | NumericFp8SelectedKernel::W8A16FusedBf16Vector => (
            ferrule_cutlass_numeric_fp8_w8a16_can_implement,
            ferrule_cutlass_numeric_fp8_w8a16_launch,
        ),
    };
    status(unsafe { preflight(&args) })?;
    cu(stream.context().bind_to_thread())?;
    let result = status(unsafe { launch(&args) });
    if result.is_err() {
        workspace.poison();
    }
    result
}

/// One handle per existing CUDA owner; workspace rebinding only clones its lease.
#[derive(Debug)]
pub(crate) struct NumericBf16Gemv {
    raw: u64,
    stream: std::sync::Arc<CudaStream>,
    pub(crate) growth_bytes: u64,
    pub(crate) version: i32,
}
impl NumericBf16Gemv {
    pub(crate) fn new(
        stream: &CudaStream,
        storage: &DeviceBuffer<u8>,
    ) -> Result<std::sync::Arc<Self>> {
        available(NumericFp8Precision::Bf16M1GemvF32Accumulate)?;
        cu(stream.context().check_graph_ready())?;
        cu(stream.context().bind_to_thread())?;
        cu(storage.check_context(stream.context(), "BF16 GEMV initialization"))?;
        let mut provider = Self {
            raw: 0,
            stream: cu(stream.owner())?,
            growth_bytes: 0,
            version: 0,
        };
        let code = unsafe {
            crate::cuda::ffi::cutlass::ferrule_numeric_bf16_gemv_create(
                stream.cu_stream() as usize as u64,
                storage.cu_deviceptr(),
                storage.num_bytes() as u64,
                crate::cuda_plan::NUMERIC_BF16_GEMV_PROVIDER_BYTES as u64,
                &mut provider.raw,
                &mut provider.growth_bytes,
                &mut provider.version,
            )
        };
        // Even failed warmups may have submitted work. Never release their storage
        // or context without completion proof; the owner quarantines on this bit.
        if let Err(error) = stream.synchronize() {
            stream.context().mark_graph_unknown();
            return Err(invalid(format!(
                "BF16 GEMV initialization completion unknown: {error}"
            )));
        }
        if code == 7 {
            return Err(invalid(format!(
                "BF16 GEMV requires validated cuBLAS 120405; loaded {} (no fallback)",
                provider.version
            )));
        }
        if code == 5 || code == 6 {
            return Err(invalid(format!(
                "BF16 GEMV provider admission failed: status={code}, measured={}, reserve={}, cuBLAS={}",
                provider.growth_bytes,
                crate::cuda_plan::NUMERIC_BF16_GEMV_PROVIDER_BYTES,
                provider.version
            )));
        }
        status(code)?;
        Ok(std::sync::Arc::new(provider))
    }
}
impl Drop for NumericBf16Gemv {
    fn drop(&mut self) {
        if self.raw == 0 {
            return;
        }
        if self.stream.context().is_capturing()
            || self.stream.context().bind_to_thread().is_err()
            || self.stream.synchronize().is_err()
        {
            self.stream.context().mark_graph_unknown();
            std::mem::forget(std::sync::Arc::clone(&self.stream));
            return;
        }
        if unsafe { crate::cuda::ffi::cutlass::ferrule_numeric_bf16_gemv_destroy(self.raw) } != 0 {
            self.stream.context().mark_graph_unknown();
            std::mem::forget(std::sync::Arc::clone(&self.stream));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::operators::linear::{NumericFp8Layout, NumericFp8ScaleType};
    #[test]
    fn precision_uses_independent_existing_manifest_bits() {
        let only_f32 = super::super::CutlassProviderManifest {
            kernel_mask: super::super::CutlassKernelId::F32Gemm.mask(),
        };
        assert!(only_f32.supports(kernel_for_precision(NumericFp8Precision::F32Tf32x3)));
        assert!(!only_f32.supports(super::super::CutlassKernelId::NumericFp8FusedM1));
        let only_m1 = super::super::CutlassProviderManifest {
            kernel_mask: super::super::CutlassKernelId::NumericFp8FusedM1.mask(),
        };
        assert!(!only_m1.supports(kernel_for_precision(NumericFp8Precision::F32Tf32x3)));
        assert!(only_m1.supports(super::super::CutlassKernelId::NumericFp8FusedM1));
        assert!(!only_f32.supports(kernel_for_precision(
            NumericFp8Precision::Bf16RneF32Accumulate
        )));
        let only_bf16 = super::super::CutlassProviderManifest {
            kernel_mask: super::super::CutlassKernelId::Bf16Gemm.mask(),
        };
        assert!(!only_bf16.supports(kernel_for_precision(NumericFp8Precision::F32Tf32x3)));
        assert!(only_bf16.supports(kernel_for_precision(
            NumericFp8Precision::Bf16RneF32Accumulate
        )));
    }

    #[test]
    #[ignore = "requires CUDA; poisoned workspace cannot be reused under either profile"]
    fn poisoned_workspace_stays_quarantined_for_both_profiles() {
        let ctx = crate::cuda::runtime::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let layout = NumericFp8Layout {
            n: 1,
            k: 1,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::F32,
        };
        let artifact =
            CudaNumericFp8Artifact::upload(&stream, layout, &[0x38], &1.0f32.to_le_bytes())
                .unwrap();
        let a = DeviceBuffer::from_host(&stream, &[1.0f32]).unwrap();
        let mut d = DeviceBuffer::from_host(&stream, &[73.0f32]).unwrap();
        for precision in [
            NumericFp8Precision::Bf16RneF32Accumulate,
            NumericFp8Precision::F32Tf32x3,
        ] {
            let plan = NumericFp8LinearPlan::new(layout, 1, 32, precision).unwrap();
            let mut workspace = CudaNumericFp8Workspace::from_buffer_with_precision(
                DeviceBuffer::<u8>::zeroed(&stream, plan.workspace_requirements().bytes as usize)
                    .unwrap(),
                precision,
            );
            // Model the existing unknown-submission state without injecting an
            // illegal access into the shared GPU context.
            workspace.poison();
            assert!(
                crate::cuda::operators::linear::numeric_fp8_linear(
                    &stream,
                    &artifact,
                    &a,
                    &mut d,
                    &mut workspace,
                    plan,
                    1,
                    1,
                )
                .is_err()
            );
            assert!(workspace.is_poisoned());
            assert_eq!(d.to_host_vec(&stream).unwrap(), [73.0]);
        }
    }

    #[test]
    fn m1_native_preflight_is_host_only_and_bounded() {
        fn args() -> NumericArgs {
            NumericArgs {
                m: 1,
                n: 7,
                k: 129,
                padded_k: 129,
                tile_rows: 0,
                scale_cols: 2,
                row_origin: 127,
                column_origin: 125,
                scale_bytes: 4,
                lda: 132,
                ldd: 12,
                reserved: 0,
                activation: 0x1000,
                weight: 0x2000,
                scales: 0x3000,
                output: 0x4000,
                workspace: 0,
                workspace_bytes: 0,
                stream: 0,
            }
        }
        let preflight =
            |a: &NumericArgs| unsafe { ferrule_cutlass_numeric_fp8_m1_can_implement(a) };
        assert_eq!(preflight(&args()), 0);
        let mut a = args();
        a.scale_bytes = 2;
        a.weight += 1; // Scalar byte loads permit an unaligned compressed origin.
        a.activation += 4;
        assert_eq!(preflight(&a), 0);
        a.n = i32::MAX as u32 - 128;
        a.ldd = a.n;
        a.k = i32::MAX as u32 - 32;
        a.padded_k = a.k;
        a.lda = a.k;
        a.scale_cols = (a.column_origin + a.k).div_ceil(128);
        assert_eq!(preflight(&a), 0); // Bounded grid-stride; no allocation/launch.
        for change in [
            |a: &mut NumericArgs| a.m = 2,
            |a: &mut NumericArgs| a.n = 0,
            |a: &mut NumericArgs| a.n = i32::MAX as u32,
            |a: &mut NumericArgs| a.k = 0,
            |a: &mut NumericArgs| a.k = i32::MAX as u32,
            |a: &mut NumericArgs| a.padded_k += 1,
            |a: &mut NumericArgs| a.tile_rows = 1,
            |a: &mut NumericArgs| a.workspace_bytes = 1,
            |a: &mut NumericArgs| a.reserved = 1,
            |a: &mut NumericArgs| a.scale_cols += 1,
            |a: &mut NumericArgs| a.row_origin = 128,
            |a: &mut NumericArgs| a.column_origin = 128,
            |a: &mut NumericArgs| a.scale_bytes = 1,
            |a: &mut NumericArgs| a.lda = 128,
            |a: &mut NumericArgs| a.ldd = 6,
            |a: &mut NumericArgs| a.lda = u32::MAX,
            |a: &mut NumericArgs| a.ldd = u32::MAX,
            |a: &mut NumericArgs| a.activation = 0,
            |a: &mut NumericArgs| a.weight = 0,
            |a: &mut NumericArgs| a.scales = 0,
            |a: &mut NumericArgs| a.output = 0,
            |a: &mut NumericArgs| a.activation += 1,
            |a: &mut NumericArgs| a.output += 1,
            |a: &mut NumericArgs| a.scales += 1,
            |a: &mut NumericArgs| a.activation = u64::MAX - 3,
            |a: &mut NumericArgs| a.weight = u64::MAX,
            |a: &mut NumericArgs| a.scales = u64::MAX - 3,
            |a: &mut NumericArgs| a.output = u64::MAX - 3,
        ] {
            let mut a = args();
            change(&mut a);
            assert_eq!(preflight(&a), 1);
        }
        assert_eq!(
            unsafe { ferrule_cutlass_numeric_fp8_m1_can_implement(std::ptr::null()) },
            1
        );
    }

    #[test]
    fn zero_scratch_span_does_not_weaken_operand_overlap_checks() {
        assert!(no_overlap(&[(16, 48)], &[(64, 80), (0, 0)]).is_ok());
        assert!(no_overlap(&[(16, 48)], &[(64, 80), (32, 32)]).is_ok());
        assert!(no_overlap(&[(16, 48)], &[(32, 48), (0, 0)]).is_err());
        assert!(no_overlap(&[(16, 48)], &[(64, 80), (32, 64)]).is_err());
        assert!(no_overlap(&[(16, 48)], &[(64, 80), (64, 96)]).is_err());
    }

    #[test]
    fn abi() {
        assert_eq!(std::mem::size_of::<Bf16Args>(), 56);
        assert_eq!(std::mem::size_of::<NumericArgs>(), 104);
        assert_eq!(std::mem::offset_of!(NumericArgs, activation), 48);
        assert_eq!(std::mem::offset_of!(NumericArgs, stream), 96);
    }
}

#[cfg(test)]
#[path = "../../../tests/numeric_bf16_tiled/preflight.rs"]
mod typed_tiled_tests;

#[cfg(test)]
#[path = "../../../tests/numeric_bf16_gemv/preflight.rs"]
mod gemv_preflight_tests;
