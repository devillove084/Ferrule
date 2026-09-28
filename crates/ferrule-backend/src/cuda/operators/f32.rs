//! F32 linear contracts. TF32x3 arithmetic and failure variants are unchanged.
//! Diagnostic strings retain the legacy provider name for source/error compatibility.

use crate::cuda::runtime::CudaError;
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

    pub(crate) fn required_bytes(self) -> Result<[usize; 3]> {
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

/// Byte-backed artifact entry used by the existing owner-facing F32 operator.
/// Keep the native validation order and error variants unchanged.
pub(crate) fn f32_gemm_bytes(
    stream: &crate::cuda::runtime::CudaStream,
    activation: &crate::cuda::runtime::DeviceBuffer<f32>,
    weight: &crate::cuda::runtime::DeviceBuffer<u8>,
    output: &mut crate::cuda::runtime::DeviceBuffer<f32>,
    layout: F32GemmLayout,
) -> Result<()> {
    crate::cuda::providers::cutlass::f32_gemm_bytes_native(
        stream, activation, weight, output, layout,
    )
}
