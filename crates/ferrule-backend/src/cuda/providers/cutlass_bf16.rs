//! Explicit BF16 or F32 TF32x3 arithmetic on the existing CUTLASS provider.
//! Numeric FP8 is storage only: GPU decode/multiply, not native FP8 MMA.

use crate::cuda::context::cu;
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
            "requested TensorOp profile unavailable; no SIMT fallback",
        )),
        3 => Err(invalid("unsupported TensorOp shape")),
        _ => Err(invalid(format!(
            "native submission failed ({code}); completion unknown"
        ))),
    }
}
pub(super) const fn kernel_for_precision(precision: NumericFp8Precision) -> super::CutlassKernelId {
    match precision {
        NumericFp8Precision::Bf16RneF32Accumulate => super::CutlassKernelId::Bf16Gemm,
        NumericFp8Precision::F32Tf32x3 => super::CutlassKernelId::F32Gemm,
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
    ferrule_cutlass_numeric_fp8_can_implement, ferrule_cutlass_numeric_fp8_f32_can_implement,
    ferrule_cutlass_numeric_fp8_f32_launch, ferrule_cutlass_numeric_fp8_launch,
};

use crate::cuda::operators::linear::{
    Bf16GemmLayout, CudaNumericFp8Artifact, CudaNumericFp8Workspace, NumericFp8LinearPlan,
    NumericFp8Precision,
};

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
            .any(|&(a, b)| start < b && a < end)
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
    if workspace.is_poisoned() {
        return Err(invalid("workspace quarantined after unknown submission"));
    }
    if artifact.layout() != plan.layout() {
        return Err(invalid("artifact/plan layout mismatch"));
    }
    if workspace.precision() != plan.precision() {
        return Err(invalid("workspace/plan precision profile mismatch"));
    }
    available(plan.precision())?;
    let l = plan.layout();
    let (weight, scales) = artifact.buffers();
    l.validate_lengths(weight.num_bytes(), scales.num_bytes())?;
    let (w, s) = l.storage_lengths()?;
    let a = buffer(
        stream,
        activation,
        extent(plan.rows(), l.k, activation_stride, 4)?,
        4,
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
    let scratch = buffer(
        stream,
        workspace.storage(),
        plan.workspace_requirements().bytes as usize,
        16,
    )?;
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
    let (preflight, launch): (
        unsafe extern "C" fn(*const NumericArgs) -> i32,
        unsafe extern "C" fn(*const NumericArgs) -> i32,
    ) = match plan.precision() {
        NumericFp8Precision::Bf16RneF32Accumulate => (
            ferrule_cutlass_numeric_fp8_can_implement,
            ferrule_cutlass_numeric_fp8_launch,
        ),
        NumericFp8Precision::F32Tf32x3 => (
            ferrule_cutlass_numeric_fp8_f32_can_implement,
            ferrule_cutlass_numeric_fp8_f32_launch,
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
    fn abi() {
        assert_eq!(std::mem::size_of::<Bf16Args>(), 56);
        assert_eq!(std::mem::size_of::<NumericArgs>(), 104);
        assert_eq!(std::mem::offset_of!(NumericArgs, activation), 48);
        assert_eq!(std::mem::offset_of!(NumericArgs, stream), 96);
    }
}
