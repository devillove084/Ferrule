//! Generic F32 extension of the existing CUTLASS provider, not a separate provider.

use crate::cuda::operators::OperatorWorkspaceRequirements;
use crate::cuda::runtime::{CudaError, CudaStream, DeviceBuffer, DeviceCopy};
use snafu::Snafu;

#[derive(Debug, Snafu)]
pub enum F32GemmError {
    #[snafu(display(
        "CUTLASS F32 GEMM: dimensions must be positive, representable, and not overflow"
    ))]
    InvalidShape,
    #[snafu(display("CUTLASS F32 GEMM: invalid {operand} leading stride"))]
    InvalidStride { operand: &'static str },
    #[snafu(display("CUTLASS F32 GEMM: {operand} buffer too small ({actual} < {required} bytes)"))]
    BufferTooSmall {
        operand: &'static str,
        actual: usize,
        required: usize,
    },
    #[snafu(display("CUTLASS F32 GEMM: {operand} belongs to another context owner"))]
    OwnerMismatch { operand: &'static str },
    #[snafu(display("CUTLASS F32 GEMM: {operand} is not F32 aligned"))]
    Misaligned { operand: &'static str },
    #[snafu(display("CUTLASS F32 GEMM: output overlaps an input"))]
    Aliasing,
    #[snafu(display("CUTLASS F32 TensorOp capability unavailable; no SIMT fallback"))]
    UnsupportedCapability,
    #[snafu(display("CUTLASS F32 TensorOp does not support this shape; no implicit fallback"))]
    UnsupportedShape,
    #[snafu(display("CUTLASS F32 GEMM context binding failed: {source}"))]
    Context { source: CudaError },
    #[snafu(display("CUTLASS F32 GEMM native failure: {status}"))]
    Native { status: i32 },
}

type Result<T> = std::result::Result<T, F32GemmError>;

/// Bias-free `D = A * W^T`. A `[rows,k]`, W `[n,k]`, D `[rows,n]`
/// are row-major, with leading strides measured in F32 elements. Padding is
/// allowed; inner strides are always one. No BF16 conversion or bias is hidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32GemmLayout {
    pub rows: usize,
    pub n: usize,
    pub k: usize,
    pub activation_stride: usize,
    pub weight_stride: usize,
    pub output_stride: usize,
}

impl F32GemmLayout {
    pub const fn contiguous(rows: usize, n: usize, k: usize) -> Self {
        Self {
            rows,
            n,
            k,
            activation_stride: k,
            weight_stride: k,
            output_stride: n,
        }
    }

    pub fn validate(self) -> Result<()> {
        self.required_bytes().map(|_| ())
    }

    fn required_bytes(self) -> Result<[usize; 3]> {
        // Leave room for CUTLASS tile rounding and signed iterator increments.
        if self.rows == 0
            || self.n == 0
            || self.k == 0
            || self.rows > i32::MAX as usize - 64
            || self.n > i32::MAX as usize - 64
            || self.k > i32::MAX as usize - 16
        {
            return Err(F32GemmError::InvalidShape);
        }
        let mut bytes = [0; 3];
        for (index, (operand, rows, width, stride)) in [
            ("activation", self.rows, self.k, self.activation_stride),
            ("weight", self.n, self.k, self.weight_stride),
            ("output", self.rows, self.n, self.output_stride),
        ]
        .into_iter()
        .enumerate()
        {
            if stride < width || stride > i32::MAX as usize {
                return Err(F32GemmError::InvalidStride { operand });
            }
            bytes[index] = (rows - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(width))
                .and_then(|n| n.checked_mul(4))
                .filter(|&n| n <= isize::MAX as usize)
                .ok_or(F32GemmError::InvalidShape)?;
        }
        Ok(bytes)
    }
}

use crate::cuda::ffi::cutlass::{
    CutlassF32Args as Args, ferrule_cutlass_f32_can_implement, ferrule_cutlass_f32_launch,
    ferrule_cutlass_provider_manifest,
};

fn available() -> bool {
    unsafe { ferrule_cutlass_provider_manifest() }.supports(super::CutlassKernelId::F32Gemm)
}

fn check(status: i32) -> Result<()> {
    match status {
        0 => Ok(()),
        1 => Err(F32GemmError::InvalidShape),
        2 => Err(F32GemmError::UnsupportedCapability),
        3 => Err(F32GemmError::UnsupportedShape),
        status => Err(F32GemmError::Native { status }),
    }
}

/// This non-split-K TensorOp uses no scratch. No allocation or synchronization
/// occurs during validation/submission. Unsupported capability is never SIMT.
pub fn f32_gemm_workspace_requirements(
    layout: F32GemmLayout,
) -> Result<OperatorWorkspaceRequirements> {
    layout.validate()?;
    if !available() {
        return Err(F32GemmError::UnsupportedCapability);
    }
    if layout.n > 64 * 65535 {
        return Err(F32GemmError::UnsupportedShape);
    }
    Ok(OperatorWorkspaceRequirements {
        bytes: 0,
        alignment: 1,
    })
}

fn buffer<T: DeviceCopy>(
    stream: &CudaStream,
    value: &DeviceBuffer<T>,
    operand: &'static str,
    required: usize,
) -> Result<()> {
    value
        .check_context(stream.context(), "CUTLASS F32 GEMM")
        .map_err(|_| F32GemmError::OwnerMismatch { operand })?;
    if value.num_bytes() < required {
        return Err(F32GemmError::BufferTooSmall {
            operand,
            actual: value.num_bytes(),
            required,
        });
    }
    if value.cu_deviceptr() % 4 != 0 {
        return Err(F32GemmError::Misaligned { operand });
    }
    Ok(())
}

fn args<W: DeviceCopy>(
    stream: &CudaStream,
    activation: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<W>,
    output: &DeviceBuffer<f32>,
    layout: F32GemmLayout,
) -> Result<Args> {
    let bytes = layout.required_bytes()?;
    buffer(stream, activation, "activation", bytes[0])?;
    buffer(stream, weight, "weight", bytes[1])?;
    buffer(stream, output, "output", bytes[2])?;
    let addresses = [
        activation.cu_deviceptr(),
        weight.cu_deviceptr(),
        output.cu_deviceptr(),
    ];
    let mut ends = [0; 3];
    for i in 0..3 {
        ends[i] = addresses[i]
            .checked_add(bytes[i] as u64)
            .ok_or(F32GemmError::InvalidShape)?;
    }
    if (0..2).any(|i| addresses[i] < ends[2] && addresses[2] < ends[i]) {
        return Err(F32GemmError::Aliasing);
    }
    f32_gemm_workspace_requirements(layout)?;
    Ok(Args {
        m: layout.rows as u32,
        n: layout.n as u32,
        k: layout.k as u32,
        lda: layout.activation_stride as u32,
        ldb: layout.weight_stride as u32,
        ldd: layout.output_stride as u32,
        activation: addresses[0],
        weight: addresses[1],
        output: addresses[2],
        stream: stream.cu_stream() as usize as u64,
    })
}

/// Preflight typed owners, extents, strides, overlap and native support without
/// launching. The exact context owner must match, even on the same ordinal.
pub fn f32_gemm_can_implement(
    stream: &CudaStream,
    activation: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    layout: F32GemmLayout,
) -> Result<()> {
    let args = args(stream, activation, weight, output, layout)?;
    check(unsafe { ferrule_cutlass_f32_can_implement(&args) })
}

fn launch<W: DeviceCopy>(
    stream: &CudaStream,
    activation: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<W>,
    output: &mut DeviceBuffer<f32>,
    layout: F32GemmLayout,
) -> Result<()> {
    let args = args(stream, activation, weight, output, layout)?;
    check(unsafe { ferrule_cutlass_f32_can_implement(&args) })?;
    stream
        .context()
        .bind_to_thread()
        .map_err(|source| F32GemmError::Context { source })?;
    check(unsafe { ferrule_cutlass_f32_launch(&args) })
}

/// Enqueue CUTLASS TF32x3 TensorOp GEMM with F32 accumulation/output. This
/// approximates F32 multiplication, not bitwise IEEE SGEMM. No fallback exists.
/// Uses the caller's stream and allocator retirement fences. Cross-stream
/// consumers must wait for the caller-recorded completion event; inputs must
/// not be overwritten until those consumers have quiesced.
pub fn f32_gemm(
    stream: &CudaStream,
    activation: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    layout: F32GemmLayout,
) -> Result<()> {
    launch(stream, activation, weight, output, layout)
}

pub(crate) fn f32_gemm_bytes(
    stream: &CudaStream,
    activation: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    layout: F32GemmLayout,
) -> Result<()> {
    launch(stream, activation, weight, output, layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_and_shape_errors_are_distinct() {
        assert!(matches!(check(2), Err(F32GemmError::UnsupportedCapability)));
        assert!(matches!(check(3), Err(F32GemmError::UnsupportedShape)));
    }

    #[test]
    #[ignore = "requires native CUDA GPU; byte-backed weight alignment rejection"]
    fn byte_weights_reject_misalignment_before_launch() {
        let context = crate::cuda::runtime::CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let input = DeviceBuffer::from_host(&stream, &[1.0f32]).unwrap();
        let weight_root = DeviceBuffer::from_host(&stream, &[0u8; 5]).unwrap();
        let weight = weight_root.slice(1, 4).unwrap();
        let mut output = DeviceBuffer::from_host(&stream, &[73.0f32]).unwrap();
        assert!(matches!(
            f32_gemm_bytes(
                &stream,
                &input,
                &weight,
                &mut output,
                F32GemmLayout::contiguous(1, 1, 1)
            ),
            Err(F32GemmError::Misaligned { operand: "weight" })
        ));
        assert_eq!(output.to_host_vec(&stream).unwrap(), [73.0]);
    }

    #[test]
    fn native_f32_abi() {
        assert_eq!(std::mem::size_of::<Args>(), 56);
        assert_eq!(std::mem::align_of::<Args>(), 8);
        assert_eq!(std::mem::offset_of!(Args, activation), 24);
        assert_eq!(std::mem::offset_of!(Args, stream), 48);
    }
}
