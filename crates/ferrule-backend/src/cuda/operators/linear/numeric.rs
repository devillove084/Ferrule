//! Provider-neutral numeric FP8 storage and linear execution contracts.
//!
//! Numeric FP8 is a storage format. Arithmetic precision is selected explicitly
//! by [`NumericFp8Precision`], while selected-kernel metadata makes the M1 specialization explicit.
//! Native entry points remain private to the provider module.

use crate::cuda::operator_owner::cu;
use crate::cuda::runtime::{CudaStream, DeviceBuffer};
use ferrule_common::Result;
use ferrule_common::numeric_fp8::ImmutableValidatedNumericFp8Payload;

pub use ferrule_common::numeric_fp8::{NumericFp8Layout, NumericFp8ScaleType};

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

pub use crate::cuda_plan::{NumericFp8LinearPlan, NumericFp8Precision, NumericFp8SelectedKernel};

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
    pub(crate) gemv: Option<std::sync::Arc<crate::cuda::providers::cutlass::NumericBf16Gemv>>,
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
            gemv: None,
        }
    }

    pub const fn precision(&self) -> NumericFp8Precision {
        self.precision
    }

    pub fn allocated_bytes(&self) -> usize {
        self.storage.num_bytes()
    }

    /// Measured provider growth, separate from allocator-owned numeric scratch.
    pub fn provider_observed_growth_bytes(&self) -> u64 {
        self.gemv.as_ref().map_or(0, |p| p.growth_bytes)
    }

    pub fn provider_version(&self) -> Option<i32> {
        self.gemv.as_ref().map(|p| p.version)
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

#[cfg(test)]
#[path = "../../../../tests/numeric_w8a16/poison.rs"]
mod w8a16_poison_tests;
