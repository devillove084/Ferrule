//! Provider-neutral numeric FP8 storage and linear execution contracts.
//!
//! Numeric FP8 is a storage format. Arithmetic precision is selected explicitly
//! by [`NumericFp8Precision`], while provider capability and native entry-point
//! selection remain private to the provider module.

use crate::cuda::context::cu;
use crate::cuda::operators::OperatorWorkspaceRequirements;
use crate::cuda::runtime::{CudaStream, DeviceBuffer};
use ferrule_common::numeric_fp8::ImmutableValidatedNumericFp8Payload;
use ferrule_common::{Error, Result};

pub use ferrule_common::numeric_fp8::{NumericFp8Layout, NumericFp8ScaleType};

fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("numeric FP8: {}", message.into()),
    }
}

fn checked(value: Option<usize>) -> Result<usize> {
    value
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or_else(|| invalid("shape/size overflow"))
}

/// Immutable encoded device storage. It never creates a full BF16 or F32
/// weight copy. Device ownership and retirement remain those of the stream
/// used for upload.
pub struct CudaNumericFp8Artifact {
    layout: NumericFp8Layout,
    weight: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
}

impl CudaNumericFp8Artifact {
    // The context bridge has already validated payload bytes before uploading.
    pub(crate) fn from_device_buffers(
        layout: NumericFp8Layout,
        weight: DeviceBuffer<u8>,
        scales: DeviceBuffer<u8>,
    ) -> Self {
        Self {
            layout,
            weight,
            scales,
        }
    }

    pub(crate) fn buffers(&self) -> (&DeviceBuffer<u8>, &DeviceBuffer<u8>) {
        (&self.weight, &self.scales)
    }

    pub fn layout(&self) -> NumericFp8Layout {
        self.layout
    }

    pub fn storage_bytes(&self) -> usize {
        self.weight.num_bytes() + self.scales.num_bytes()
    }

    /// Compatibility raw upload API. Prefer
    /// `CudaOperators::upload_numeric_fp8_linear` when allocator accounting and
    /// the owning operator context are available.
    pub fn upload(
        stream: &CudaStream,
        layout: NumericFp8Layout,
        weight: &[u8],
        scales: &[u8],
    ) -> Result<Self> {
        layout.validate_payload(weight, scales)?;
        Self::upload_validated_bytes(stream, layout, weight, scales)
    }

    /// Reuse CPU payload validation, not CUDA ownership. Device storage is
    /// still created on this exact stream owner; launch preflight remains
    /// mandatory.
    pub fn upload_validated(
        stream: &CudaStream,
        payload: &ImmutableValidatedNumericFp8Payload,
    ) -> Result<Self> {
        let layout = payload.layout();
        layout.validate_lengths(payload.weight_bytes().len(), payload.scale_bytes().len())?;
        Self::upload_validated_bytes(
            stream,
            layout,
            payload.weight_bytes(),
            payload.scale_bytes(),
        )
    }

    fn upload_validated_bytes(
        stream: &CudaStream,
        layout: NumericFp8Layout,
        weight: &[u8],
        scales: &[u8],
    ) -> Result<Self> {
        Ok(Self {
            layout,
            weight: cu(DeviceBuffer::from_host(stream, weight))?,
            scales: cu(DeviceBuffer::from_host(stream, scales))?,
        })
    }
}

/// Explicit arithmetic profile, independent of numeric FP8 storage encoding.
/// Both profiles accumulate and output F32. TF32x3 approximates IEEE F32
/// products; it is not bitwise SGEMM and does not promise a reduction order.
/// A deprecated inherent `kernel()` method remains in the provider's isolated
/// legacy bridge for source/const compatibility. It is a temporary vendor leak,
/// not part of the semantic contract; new callers should submit a plan instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericFp8Precision {
    /// F32 -> BF16 RNE activation and decoded BF16 weights, then BF16 TensorOp.
    Bf16RneF32Accumulate,
    /// F32 activation and decoded F32 weights, then TF32x3 TensorOp.
    /// No BF16 rounding or CPU fallback. F32 accumulation can still lose
    /// accuracy for ill-conditioned/cancelling dot products.
    F32Tf32x3,
}

/// Validated budget plan. BF16 retains a K8-padded activation plus one BF16
/// weight tile. F32 reads caller-owned strided F32 activation and retains only
/// one unpadded F32 weight tile. Precision participates in plan identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericFp8LinearPlan {
    layout: NumericFp8Layout,
    rows: usize,
    padded_k: usize,
    tile_rows: usize,
    bytes: usize,
    budget: usize,
    precision: NumericFp8Precision,
}

impl NumericFp8LinearPlan {
    pub fn new(
        layout: NumericFp8Layout,
        rows: usize,
        scratch_budget_bytes: usize,
        precision: NumericFp8Precision,
    ) -> Result<Self> {
        layout.storage_lengths()?;
        if rows == 0 || rows > i32::MAX as usize - 64 {
            return Err(invalid("invalid activation rows"));
        }
        let (padded_k, row_bytes, activation_bytes) = match precision {
            NumericFp8Precision::Bf16RneF32Accumulate => {
                let padded_k = layout.k.div_ceil(8) * 8;
                let row_bytes = checked(padded_k.checked_mul(2))?;
                let activation = checked(rows.checked_mul(row_bytes))?;
                (padded_k, row_bytes, activation)
            }
            NumericFp8Precision::F32Tf32x3 => {
                let row_bytes = checked(layout.k.checked_mul(4))?;
                (layout.k, row_bytes, 0)
            }
        };
        let available = scratch_budget_bytes
            .checked_sub(activation_bytes)
            .ok_or_else(|| invalid("scratch budget below activation bytes"))?;
        // Retain the established maximum output-row tile contract. Changing
        // this bound would change budget partitioning and launch counts.
        let tile_rows = (available / row_bytes).min(layout.n).min(64 * 65535);
        if tile_rows == 0 {
            return Err(invalid(
                "scratch budget cannot hold one weight row plus required conversion storage",
            ));
        }
        let bytes = checked(match precision {
            NumericFp8Precision::Bf16RneF32Accumulate => rows
                .checked_add(tile_rows)
                .and_then(|r| r.checked_mul(row_bytes)),
            NumericFp8Precision::F32Tf32x3 => tile_rows.checked_mul(row_bytes),
        })?;
        Ok(Self {
            layout,
            rows,
            padded_k,
            tile_rows,
            bytes,
            budget: scratch_budget_bytes,
            precision,
        })
    }

    pub(crate) fn rows(self) -> usize {
        self.rows
    }

    pub fn layout(self) -> NumericFp8Layout {
        self.layout
    }

    pub fn workspace_requirements(self) -> OperatorWorkspaceRequirements {
        OperatorWorkspaceRequirements {
            bytes: self.bytes as u64,
            alignment: 16,
        }
    }

    pub const fn precision(self) -> NumericFp8Precision {
        self.precision
    }

    pub fn scratch_budget_bytes(self) -> usize {
        self.budget
    }

    /// Additional activation scratch, not caller-owned input storage. F32 uses zero.
    pub fn activation_bytes(self) -> usize {
        match self.precision {
            NumericFp8Precision::Bf16RneF32Accumulate => self.rows * self.padded_k * 2,
            NumericFp8Precision::F32Tf32x3 => 0,
        }
    }

    pub fn weight_tile_bytes(self) -> usize {
        match self.precision {
            NumericFp8Precision::Bf16RneF32Accumulate => self.tile_rows * self.padded_k * 2,
            NumericFp8Precision::F32Tf32x3 => self.tile_rows * self.padded_k * 4,
        }
    }

    pub fn tile_rows(self) -> usize {
        self.tile_rows
    }

    pub fn tile_count(self) -> usize {
        self.layout.n.div_ceil(self.tile_rows)
    }

    pub fn kernel_launches(self) -> usize {
        match self.precision {
            NumericFp8Precision::Bf16RneF32Accumulate => 1 + 2 * self.tile_count(),
            NumericFp8Precision::F32Tf32x3 => 2 * self.tile_count(),
        }
    }

    pub fn padded_k(self) -> usize {
        self.padded_k
    }
}

/// Caller-owned, graph-stable scratch. After a native submission error it is
/// poisoned and follows existing retirement/quarantine rules. There is no
/// public storage accessor or poison reset, including through the legacy path:
///
/// ```compile_fail,E0624
/// use ferrule_backend::cuda::operators::linear::CudaNumericFp8Workspace;
/// fn raw_storage(workspace: &CudaNumericFp8Workspace) {
///     let _ = workspace.storage();
/// }
/// ```
///
/// ```compile_fail,E0616
/// use ferrule_backend::cuda::operators::linear::CudaNumericFp8Workspace;
/// fn raw_storage(workspace: &CudaNumericFp8Workspace) {
///     let _ = &workspace.storage;
/// }
/// ```
///
/// ```compile_fail,E0616
/// use ferrule_backend::cuda::providers::cutlass::CudaNumericFp8Workspace;
/// fn reset(workspace: &mut CudaNumericFp8Workspace) {
///     workspace.poisoned = false;
/// }
/// ```
pub struct CudaNumericFp8Workspace {
    storage: DeviceBuffer<u8>,
    poisoned: bool,
    precision: NumericFp8Precision,
}

impl CudaNumericFp8Workspace {
    pub(crate) fn storage(&self) -> &DeviceBuffer<u8> {
        &self.storage
    }

    pub(crate) fn poison(&mut self) {
        self.poisoned = true;
    }

    /// Compatibility raw-storage constructor. Prefer
    /// `CudaOperators::numeric_fp8_linear_workspace` for budget-sized storage.
    pub fn from_buffer(storage: DeviceBuffer<u8>) -> Self {
        Self::from_buffer_with_precision(storage, NumericFp8Precision::Bf16RneF32Accumulate)
    }

    /// Compatibility raw-storage constructor with an explicit profile.
    /// Takes ownership without allocating or synchronizing. Aliases of supplied
    /// storage still obey the caller's completion/lifetime obligations; owner,
    /// alignment, overlap, and exact budget checks remain launch preflight.
    /// Prefer `CudaOperators::numeric_fp8_linear_workspace` for new callers.
    pub fn from_buffer_with_precision(
        storage: DeviceBuffer<u8>,
        precision: NumericFp8Precision,
    ) -> Self {
        Self {
            storage,
            poisoned: false,
            precision,
        }
    }

    pub const fn precision(&self) -> NumericFp8Precision {
        self.precision
    }

    pub fn allocated_bytes(&self) -> usize {
        self.storage.num_bytes()
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

#[cfg(test)]
mod owner_tests {
    use super::*;
    use crate::cuda::operators::linear::CudaOperators;

    #[test]
    #[ignore = "actual CUDA sm80+; public owner API rejects poisoned scratch during real capture"]
    fn poisoned_workspace_owner_api_preserves_quarantine_and_capture_scope() {
        let op = CudaOperators::new_on_device(0).expect("required CUDA owner");
        let layout = NumericFp8Layout {
            n: 1,
            k: 8,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::F32,
        };
        let artifact = op
            .upload_numeric_fp8_linear(layout, &[0x38; 8], &1.0f32.to_le_bytes())
            .unwrap();
        let input = op.upload_f32_buffer(&[1.0; 8]).unwrap();
        for precision in [
            NumericFp8Precision::Bf16RneF32Accumulate,
            NumericFp8Precision::F32Tf32x3,
        ] {
            let plan = NumericFp8LinearPlan::new(layout, 1, 64, precision).unwrap();
            let mut scratch = op.numeric_fp8_linear_workspace(plan).unwrap();
            let mut output = op.upload_f32_buffer(&[73.0]).unwrap();
            // Inject the existing terminal state, not a device fault. No test
            // claims to reproduce native asynchronous failure on this GPU.
            scratch.poison();
            op.reset_counters();
            for _ in 0..2 {
                let error = op
                    .numeric_fp8_linear_into(
                        &artifact,
                        &input,
                        &mut output,
                        &mut scratch,
                        plan,
                        8,
                        1,
                    )
                    .unwrap_err();
                assert!(error.to_string().contains("quarantined"));
            }
            let error = op
                .capture_decode_graph(|| {
                    op.numeric_fp8_linear_into(
                        &artifact,
                        &input,
                        &mut output,
                        &mut scratch,
                        plan,
                        8,
                        1,
                    )
                })
                .err()
                .expect("poison remains fatal inside real capture");
            assert!(error.to_string().contains("quarantined"));
            assert!(!op.is_capture_safe());
            assert!(scratch.is_poisoned());
            assert_eq!(op.counters().compute_kernel_launches, 0);
            assert_eq!(op.counters().device_allocation_attempts, 0);
            assert_eq!(op.counters().stream_wide_syncs, 0);
            assert_eq!(op.download_f32_buffer(&output).unwrap(), [73.0]);
            let mut fresh = op.numeric_fp8_linear_workspace(plan).unwrap();
            let graph = op
                .capture_decode_graph(|| {
                    op.numeric_fp8_linear_into(
                        &artifact,
                        &input,
                        &mut output,
                        &mut fresh,
                        plan,
                        8,
                        1,
                    )
                })
                .unwrap();
            op.launch_graph(&graph).unwrap();
            op.record_compute_event().unwrap().synchronize().unwrap();
            assert_eq!(op.download_f32_buffer(&output).unwrap(), [8.0]);
            assert!(scratch.is_poisoned());
            assert!(!fresh.is_poisoned());
        }
    }
}
