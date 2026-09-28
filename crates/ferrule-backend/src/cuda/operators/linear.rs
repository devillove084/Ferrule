//! Provider-neutral linear and representation-conversion operations.
//!
//! Owner resource submission and mutable storage views are backend-private,
//! including through the public family facade.
//! ```compile_fail,E0603
//! use ferrule_backend::cuda::operators::OperatorOwner;
//! ```
//! ```compile_fail,E0624
//! use ferrule_backend::cuda::operators::linear::CudaF32Buffer;
//! fn expose(buffer: &mut CudaF32Buffer) { let _ = buffer.as_device_buffer_mut(); }
//! ```

//! Resource borrows are not a new public raw-storage API.
//! ```compile_fail,E0624
//! use ferrule_backend::cuda::operators::linear::CudaArtifactLinearHandle;
//! fn expose(weight: &CudaArtifactLinearHandle) { let _ = weight.operator_view(); }
//! ```
//! ```compile_fail,E0624
//! use ferrule_backend::cuda::operators::linear::CudaFp8ActivationPack;
//! fn expose(pack: &mut CudaFp8ActivationPack) { let _ = pack.operator_storage_mut(); }
//! ```
//! A prepared activation keeps its producer pack borrowed until its last use.
//! ```compile_fail,E0499
//! use ferrule_backend::cuda::operators::linear::{CudaOperators, CudaF32Buffer, CudaFp8ActivationPack};
//! fn overwrite(op: &CudaOperators, input: &CudaF32Buffer, pack: &mut CudaFp8ActivationPack) {
//!     let prepared = op.prepare_fp8_activation_from_device(input, 1, 128, pack).unwrap();
//!     let _next = op.prepare_fp8_activation_from_device(input, 1, 128, pack).unwrap();
//!     drop(prepared);
//! }
//! ```
//! ```compile_fail,E0515
//! use ferrule_backend::cuda::operators::linear::{CudaOperators, CudaPreparedFp8Activation};
//! fn escape(op: &CudaOperators) -> CudaPreparedFp8Activation<'static> {
//!     let pack = op.fp8_activation_pack(1, 128).unwrap();
//!     op.prepared_fp8_activation_from_storage(&pack, 1, 128).unwrap()
//! }
//! ```

use ferrule_common::Result;

use crate::cuda::operators::{OperatorOwner, OperatorWorkspaceRequirements};
use crate::cuda::providers::cutlass;
use crate::cuda::runtime::{CudaStream, DeviceBuffer};

pub(crate) use crate::cuda::context::ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE;
pub use crate::cuda::context::{
    CudaArtifactLinearHandle, CudaArtifactLinearShape, CudaArtifactLinearWorkspace, CudaBf16Buffer,
    CudaF32Buffer, CudaFp8ActivationPack, CudaOperators, CudaPreparedFp8Activation, cuda_gemv,
    cuda_gemv_fp8_e4m3fn_e8m0_2d,
};
pub use crate::cuda::operators::contracts::{
    F32ToBf16RowsLayout, StridedBf16RowsLayout, StridedF32RowsLayout,
};

/// Compatibility boundary for proposal-head constants and layout.
///
/// Definitions now belong to `operators::proposal`. Keep this historical
/// linear-family import path as a same-type alias for downstream callers.
pub use crate::cuda::operators::proposal::{PROPOSAL_ROWS, ProposalHeadLayout};

/// Provider-neutral BF16 GEMM layout.
///
/// D[M,N] = BF16 A[M,K] * BF16 W[N,K]^T with F32 accumulation/output.
/// Strides are elements; K and BF16 strides must be multiples of 8.
/// Arithmetic, validation, and native capability decisions remain unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16GemmLayout {
    pub rows: usize,
    pub n: usize,
    pub k: usize,
    pub activation_stride: usize,
    pub weight_stride: usize,
    pub output_stride: usize,
}

impl Bf16GemmLayout {
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
        validate_bf16_gemm_layout(self)
    }
}

fn invalid_bf16(message: impl Into<String>) -> ferrule_common::Error {
    ferrule_common::Error::Internal {
        message: format!("BF16 GEMM: {}", message.into()),
    }
}

fn checked_bf16(value: Option<usize>) -> Result<usize> {
    value
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or_else(|| invalid_bf16("shape/size overflow"))
}

fn validate_bf16_gemm_layout(layout: Bf16GemmLayout) -> Result<()> {
    if layout.rows == 0
        || layout.rows > i32::MAX as usize - 64
        || layout.n == 0
        || layout.n > 64 * 65535
        || layout.k == 0
        || layout.k > i32::MAX as usize - 32
        || !layout.k.is_multiple_of(8)
        || !layout.activation_stride.is_multiple_of(8)
        || !layout.weight_stride.is_multiple_of(8)
    {
        return Err(invalid_bf16(
            "BF16 GEMM requires positive M/N, K8 and representable dimensions",
        ));
    }
    extent(layout.rows, layout.k, layout.activation_stride, 2)?;
    extent(layout.n, layout.k, layout.weight_stride, 2)?;
    extent(layout.rows, layout.n, layout.output_stride, 4)?;
    Ok(())
}
fn extent(rows: usize, width: usize, stride: usize, element: usize) -> Result<usize> {
    if rows == 0 || width == 0 || stride < width || stride > i32::MAX as usize {
        return Err(invalid_bf16("invalid row stride/extent"));
    }
    checked_bf16(
        (rows - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(width))
            .and_then(|v| v.checked_mul(element)),
    )
}

/// Query BF16 GEMM workspace without allocation or launch.
pub fn bf16_gemm_workspace_requirements(
    layout: Bf16GemmLayout,
) -> Result<OperatorWorkspaceRequirements> {
    cutlass::bf16_gemm_workspace_requirements(layout)
}

/// Validate exact owner, extent, aliasing, and native support without launch.
pub fn bf16_gemm_can_implement(
    stream: &CudaStream,
    activation: &DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    output: &DeviceBuffer<f32>,
    layout: Bf16GemmLayout,
) -> Result<()> {
    cutlass::bf16_gemm_can_implement(stream, activation, weight, output, layout)
}

/// Submit BF16 GEMM with F32 accumulation/output on the supplied stream.
///
/// This wrapper adds no allocation, synchronization, fallback, or completion
/// claim. Existing event/lifetime/quarantine obligations remain unchanged.
pub fn bf16_gemm(
    stream: &CudaStream,
    activation: &DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    layout: Bf16GemmLayout,
) -> Result<()> {
    cutlass::bf16_gemm(stream, activation, weight, output, layout)
}

#[path = "numeric.rs"]
mod numeric;

pub use numeric::{
    CudaNumericFp8Artifact, CudaNumericFp8Workspace, NumericFp8Layout, NumericFp8LinearPlan,
    NumericFp8Precision, NumericFp8ScaleType,
};

/// Submit a numeric-FP8 plan through the semantic linear boundary.
///
/// The provider implementation is deliberately delegated unchanged: preflight
/// failures do not poison the workspace, native submission failures do, and a
/// successful return still means submission rather than completion.
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
    cutlass::numeric_fp8_linear(
        stream,
        artifact,
        activation,
        output,
        workspace,
        plan,
        activation_stride,
        output_stride,
    )
}

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

#[path = "f32.rs"]
mod f32;
pub(crate) use f32::f32_gemm_bytes;
pub use f32::{F32GemmError, F32GemmLayout};

/// Low-level F32 submission compatibility API. Contracts are operator-owned;
/// native preflight/submission stay in the selected provider. Owner-facing
/// callers should use the existing standard linear methods on `CudaOperators`.
pub use crate::cuda::providers::cutlass::{
    f32_gemm, f32_gemm_can_implement, f32_gemm_workspace_requirements,
};

impl CudaOperators {
    /// Allocation/synchronization-free submission on this owner's compute stream.
    /// Input/output may be padded; strides are F32 elements. Record the existing
    /// compute event AFTER this call to cover all output chunks. On asynchronous
    /// completion failure retain/quarantine operands under the existing command
    /// owner protocol; success here is submission, not proof of completion.
    #[allow(clippy::too_many_arguments)]
    pub fn numeric_fp8_linear_into(
        &self,
        weight: &CudaNumericFp8Artifact,
        input: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
        workspace: &mut CudaNumericFp8Workspace,
        plan: NumericFp8LinearPlan,
        activation_stride: usize,
        output_stride: usize,
    ) -> Result<()> {
        self.submit_operator(plan.kernel_launches() as u64, |stream| {
            numeric_fp8_linear(
                stream,
                weight,
                input.as_device_buffer(),
                output.as_device_buffer_mut(),
                workspace,
                plan,
                activation_stride,
                output_stride,
            )
        })?;
        Ok(())
    }
}
