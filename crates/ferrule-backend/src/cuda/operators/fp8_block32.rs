//! SM80+ correctness baseline for row-major E4M3 + per-32 UE8M0 linear.
//!
//! This is intentionally a two-stage path: a clean-room scalar decode writes a
//! packed BF16 temporary weight, then the existing BF16 TensorOp provider
//! computes F32 output. It is not a fused FP8 kernel, Marlin path, or
//! performance implementation. The temporary weight and optional activation
//! padding are caller-owned workspace and must remain alive until the owner's
//! completion event proves the compute stream is finished.

use std::{cell::Cell, mem::size_of};

use crate::cuda::architecture::{COMPILED_TARGET, compiled_capabilities};
use crate::cuda::ffi::core::{
    Bf16Block32PackArgs, Fp8Block32DecodeArgs, ferrule_core_fp8_block32_device_capability,
};
use crate::cuda::operator_owner::{CudaBf16Buffer, CudaF32Buffer, CudaOperators, cu};
use crate::cuda::operators::{OperatorOwner, linear::Bf16GemmLayout};
use crate::cuda::providers::cutlass::{bf16_gemm, bf16_gemm_can_implement};
use crate::cuda::runtime::{DeviceBuffer, DeviceCopy};
use crate::operator_contract::Fp8E4m3Ue8m0LinearPlan;
use crate::plan::OperatorWorkspaceRequirements;
use ferrule_common::{Error, Result};

/// Stable observability label for the deliberately unfused baseline.
pub const FP8_BLOCK32_BASELINE_PATH: &str = "fp8-e4m3-ue8m0-block32-decode-bf16-gemm-baseline";

/// BF16 activation is the first supported activation boundary. F32 activation
/// conversion is deliberately not hidden here because it would add a second
/// temporary and a third semantic dtype boundary.
pub const FP8_BLOCK32_ACTIVATION_DTYPE: &str = "BF16";

const _: () = {
    assert!(size_of::<Fp8Block32DecodeArgs>() == 88);
    assert!(std::mem::offset_of!(Fp8Block32DecodeArgs, weight) == 56);
    assert!(std::mem::offset_of!(Fp8Block32DecodeArgs, stream) == 80);
    assert!(size_of::<Bf16Block32PackArgs>() == 48);
    assert!(std::mem::offset_of!(Bf16Block32PackArgs, input_bf16) == 24);
    assert!(std::mem::offset_of!(Bf16Block32PackArgs, stream) == 40);
};

fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("FP8 block-32 baseline: {}", message.into()),
    }
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{field} exceeds CUDA u32 ABI")))
}

fn checked_u64(value: usize, field: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid(format!("{field} exceeds CUDA u64 ABI")))
}

fn gemm_k_for_plan(plan: Fp8E4m3Ue8m0LinearPlan) -> Result<usize> {
    plan.validate()?;
    let physical_k = plan.layout.padding.padded_k.max(plan.k);
    physical_k
        .checked_add(7)
        .map(|value| value / 8 * 8)
        .ok_or_else(|| invalid("padded BF16 GEMM K overflows"))
}

fn ensure_supported(plan: Fp8E4m3Ue8m0LinearPlan) -> Result<usize> {
    plan.validate()?;
    let capabilities = compiled_capabilities();
    if !capabilities.baseline_simt || !capabilities.bf16_mma_sync {
        return Err(crate::operator_contract::OperatorContractError::Unsupported {
            capability: "SM80+ BF16 decode baseline",
            reason: format!(
                "compiled CUDA target {COMPILED_TARGET:?} does not provide the required SM80+ BF16 capability"
            ),
        }
        .into());
    }
    let gemm_k = gemm_k_for_plan(plan)?;
    Bf16GemmLayout::contiguous(plan.m, plan.n, gemm_k)
        .validate()
        .map_err(|error| invalid(format!("BF16 provider shape admission failed: {error}")))?;
    for (field, value) in [
        ("M", plan.m),
        ("N", plan.n),
        ("K", plan.k),
        ("padded K", gemm_k),
        ("padded N", plan.layout.padding.padded_n),
        ("weight row stride", plan.layout.weight_row_stride_bytes),
        ("scale rows", plan.layout.scale_shape[0]),
        ("scale columns", plan.layout.scale_shape[1]),
        ("scale row stride", plan.layout.scale_stride_bytes[0]),
        ("scale column stride", plan.layout.scale_stride_bytes[1]),
        ("row origin", plan.layout.row_origin),
        ("column origin", plan.layout.column_origin),
    ] {
        checked_u32(value, field)?;
    }
    let decode_elements = plan
        .n
        .checked_mul(gemm_k)
        .ok_or_else(|| invalid("decoded BF16 weight size overflows"))?;
    if decode_elements > 256usize.saturating_mul(65_535) {
        return Err(invalid(
            "decoded BF16 weight launch grid exceeds CUDA limit",
        ));
    }
    if plan.m.checked_mul(gemm_k).is_none() || plan.m.checked_mul(plan.n).is_none() {
        return Err(invalid("activation/output extent overflows"));
    }
    Ok(gemm_k)
}

fn workspace_bytes_for_plan(plan: Fp8E4m3Ue8m0LinearPlan, gemm_k: usize) -> Result<usize> {
    let decoded = plan
        .n
        .checked_mul(gemm_k)
        .and_then(|elements| elements.checked_mul(size_of::<u16>()))
        .ok_or_else(|| invalid("decoded BF16 workspace size overflows"))?;
    let activation = if gemm_k > plan.k {
        plan.m
            .checked_mul(gemm_k)
            .and_then(|elements| elements.checked_mul(size_of::<u16>()))
            .ok_or_else(|| invalid("padded BF16 activation workspace size overflows"))?
    } else {
        0
    };
    decoded
        .checked_add(activation)
        .ok_or_else(|| invalid("FP8 block-32 workspace size overflows"))
}

fn device_span<T: DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    elements: usize,
    field: &str,
) -> Result<(u64, u64)> {
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or_else(|| invalid(format!("{field} byte extent overflows")))?;
    if buffer.num_bytes() < bytes {
        return Err(invalid(format!(
            "{field} buffer is too small: {} < {bytes} bytes",
            buffer.num_bytes()
        )));
    }
    let start = buffer.cu_deviceptr();
    let end = start
        .checked_add(checked_u64(bytes, field)?)
        .ok_or_else(|| invalid(format!("{field} device span overflows")))?;
    Ok((start, end))
}

fn spans_overlap(first: (u64, u64), second: (u64, u64)) -> bool {
    first.0 < second.1 && second.0 < first.1
}

/// Uploaded, immutable E4M3/UE8M0 payload associated with one verified plan.
pub struct CudaFp8Block32Artifact {
    plan: Fp8E4m3Ue8m0LinearPlan,
    weight: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
}

impl CudaFp8Block32Artifact {
    pub fn plan(&self) -> Fp8E4m3Ue8m0LinearPlan {
        self.plan
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight.len()
    }

    pub fn scale_bytes(&self) -> usize {
        self.scales.len()
    }
}

/// Caller-owned storage for the decoded BF16 weight and, only for ragged K,
/// a BF16 activation copy padded to the existing provider's K8 requirement.
pub struct CudaFp8Block32Workspace {
    plan: Fp8E4m3Ue8m0LinearPlan,
    gemm_k: usize,
    decoded_weight: DeviceBuffer<u16>,
    activation_padded: Option<DeviceBuffer<u16>>,
    poisoned: Cell<bool>,
    workspace_bytes: usize,
}

impl CudaFp8Block32Workspace {
    pub fn plan(&self) -> Fp8E4m3Ue8m0LinearPlan {
        self.plan
    }

    pub fn gemm_k(&self) -> usize {
        self.gemm_k
    }

    pub fn decoded_weight_bytes(&self) -> usize {
        self.decoded_weight.num_bytes()
    }

    pub fn activation_padding_bytes(&self) -> usize {
        self.activation_padded
            .as_ref()
            .map(DeviceBuffer::num_bytes)
            .unwrap_or(0)
    }
}

impl CudaFp8Block32Workspace {
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned.get()
    }

    fn poison(&self) {
        self.poisoned.set(true);
    }
}

/// Query the exact logical workspace payload. Allocator grant rounding remains
/// owner-local and is not part of this provider-neutral byte count.
pub fn fp8_block32_workspace_requirements(
    plan: Fp8E4m3Ue8m0LinearPlan,
) -> Result<OperatorWorkspaceRequirements> {
    let gemm_k = ensure_supported(plan)?;
    Ok(OperatorWorkspaceRequirements {
        bytes: checked_u64(workspace_bytes_for_plan(plan, gemm_k)?, "workspace bytes")?,
        alignment: 16,
    })
}

impl CudaOperators {
    fn fp8_block32_graph_preflight(&self) -> Result<()> {
        let stream = self.stream_clone();
        cu(stream.context().check_graph_ready())?;
        cu(stream.context().bind_to_thread())?;
        Ok(())
    }

    fn fp8_block32_runtime_capability(&self) -> Result<()> {
        let expected_ordinal = checked_u32(self.device_ordinal(), "CUDA device ordinal")?;
        let mut actual_sm = 0u32;
        let status =
            unsafe { ferrule_core_fp8_block32_device_capability(expected_ordinal, &mut actual_sm) };
        if status != 0 {
            return Err(invalid(format!(
                "runtime CUDA capability rejected: ordinal={} actual_sm={} compiled_target={COMPILED_TARGET:?} native_status={status}",
                self.device_ordinal(),
                actual_sm
            )));
        }
        Ok(())
    }

    fn fp8_block32_context_preflight(&self) -> Result<()> {
        self.fp8_block32_graph_preflight()?;
        self.fp8_block32_runtime_capability()
    }

    fn check_fp8_block32_owners(
        &self,
        artifact: &CudaFp8Block32Artifact,
        input: &CudaBf16Buffer,
        output: &CudaF32Buffer,
        workspace: &CudaFp8Block32Workspace,
    ) -> Result<()> {
        self.check_buffer_owner(&artifact.weight, "FP8 block-32 artifact weight")?;
        self.check_buffer_owner(&artifact.scales, "FP8 block-32 artifact scales")?;
        self.check_buffer_owner(input.as_device_buffer(), "FP8 block-32 activation")?;
        self.check_buffer_owner(output.as_device_buffer(), "FP8 block-32 output")?;
        self.check_buffer_owner(&workspace.decoded_weight, "FP8 block-32 decoded workspace")?;
        if let Some(padded) = workspace.activation_padded.as_ref() {
            self.check_buffer_owner(padded, "FP8 block-32 activation workspace")?;
        }
        Ok(())
    }

    /// Upload and host-validate one E4M3/UE8M0 block-32 artifact. The upload is
    /// ordered on this owner's compute stream, so no global stream or event is
    /// introduced. Payload validation applies the CPU oracle's NaN/0xff and
    /// padding policy before any CUDA allocation or copy.
    pub fn upload_fp8_e4m3_ue8m0_block32(
        &self,
        plan: Fp8E4m3Ue8m0LinearPlan,
        weight: &[u8],
        scales: &[u8],
    ) -> Result<CudaFp8Block32Artifact> {
        self.fp8_block32_context_preflight()?;
        ensure_supported(plan)?;
        plan.validate_payload(weight, scales)?;
        let (weight, scales) = self.upload_fp8_block32_payload(weight, scales)?;
        Ok(CudaFp8Block32Artifact {
            plan,
            weight,
            scales,
        })
    }

    /// Allocate the owner-local decode workspace before submission/capture.
    /// No kernel, stream synchronization, or provider handle is created here.
    pub fn fp8_block32_workspace(
        &self,
        plan: Fp8E4m3Ue8m0LinearPlan,
    ) -> Result<CudaFp8Block32Workspace> {
        self.fp8_block32_context_preflight()?;
        let gemm_k = ensure_supported(plan)?;
        let workspace_bytes = workspace_bytes_for_plan(plan, gemm_k)?;
        let decoded_elements = plan
            .n
            .checked_mul(gemm_k)
            .ok_or_else(|| invalid("decoded BF16 workspace size overflows"))?;
        let activation_elements = (gemm_k > plan.k).then(|| {
            plan.m
                .checked_mul(gemm_k)
                .ok_or_else(|| invalid("padded BF16 activation workspace size overflows"))
        });
        let activation_elements = activation_elements.transpose()?;
        let decoded_bytes = decoded_elements
            .checked_mul(size_of::<u16>())
            .ok_or_else(|| invalid("decoded BF16 workspace byte size overflows"))?;
        let activation_bytes = activation_elements
            .map(|elements| {
                elements
                    .checked_mul(size_of::<u16>())
                    .ok_or_else(|| invalid("padded BF16 activation byte size overflows"))
            })
            .transpose()?;
        let allocated_bytes = decoded_bytes
            .checked_add(activation_bytes.unwrap_or(0))
            .ok_or_else(|| invalid("FP8 block-32 workspace byte size overflows"))?;
        if allocated_bytes != workspace_bytes {
            return Err(invalid("FP8 block-32 workspace size calculation mismatch"));
        }
        let (decoded_weight, activation_padded) =
            self.allocate_fp8_block32_workspace_buffers(decoded_elements, activation_elements)?;
        Ok(CudaFp8Block32Workspace {
            plan,
            gemm_k,
            decoded_weight,
            activation_padded,
            poisoned: Cell::new(false),
            workspace_bytes,
        })
    }

    /// Submit the unfused correctness baseline using BF16 activation storage:
    /// decode E4M3+UE8M0 into BF16, optionally pad activation K, then call the
    /// existing BF16 GEMM provider for F32 output. This method allocates and
    /// synchronizes nothing. After it returns successfully, retain artifact,
    /// input, output, and workspace until `record_compute_event()` proves the
    /// owner compute stream complete; on an error after a native submission the
    /// workspace is conservatively poisoned and must be rebuilt. Call
    /// [`Self::fp8_block32_linear_into_with_completion`] when the caller wants
    /// that event recorded as part of the same owner-local submission API.
    pub fn fp8_block32_linear_into(
        &self,
        artifact: &CudaFp8Block32Artifact,
        input: &CudaBf16Buffer,
        output: &mut CudaF32Buffer,
        workspace: &mut CudaFp8Block32Workspace,
        plan: Fp8E4m3Ue8m0LinearPlan,
    ) -> Result<()> {
        self.fp8_block32_graph_preflight()?;
        let gemm_k = ensure_supported(plan)?;
        self.check_fp8_block32_owners(artifact, input, output, workspace)?;
        self.fp8_block32_runtime_capability()?;
        if artifact.plan != plan || workspace.plan != plan || workspace.gemm_k != gemm_k {
            return Err(invalid("artifact/workspace/plan mismatch"));
        }
        if workspace.poisoned.get() {
            return Err(invalid(
                "workspace was quarantined after unknown submission",
            ));
        }

        let input_elements = plan
            .m
            .checked_mul(plan.k)
            .ok_or_else(|| invalid("BF16 activation extent overflows"))?;
        let output_elements = plan
            .m
            .checked_mul(plan.n)
            .ok_or_else(|| invalid("F32 output extent overflows"))?;
        let input_span = device_span(input.as_device_buffer(), input_elements, "activation")?;
        let output_span = device_span(output.as_device_buffer(), output_elements, "output")?;
        if spans_overlap(input_span, output_span) {
            return Err(invalid("activation and output ranges overlap"));
        }

        let layout = Bf16GemmLayout::contiguous(plan.m, plan.n, gemm_k);
        let needs_activation_padding = workspace.activation_padded.is_some();
        if let Some(padded) = workspace.activation_padded.as_ref() {
            bf16_gemm_can_implement(
                &self.stream_clone(),
                padded,
                &workspace.decoded_weight,
                output.as_device_buffer(),
                layout,
            )?;
        } else {
            bf16_gemm_can_implement(
                &self.stream_clone(),
                input.as_device_buffer(),
                &workspace.decoded_weight,
                output.as_device_buffer(),
                layout,
            )?;
        }

        let launches = if needs_activation_padding { 3 } else { 2 };
        self.submit_operator(launches, |stream| {
            let decode_args = Fp8Block32DecodeArgs {
                n: checked_u32(plan.n, "N")?,
                k: checked_u32(plan.k, "K")?,
                padded_k: checked_u32(gemm_k, "padded K")?,
                weight_padded_k: checked_u32(plan.layout.padding.padded_k, "physical padded K")?,
                padded_n: checked_u32(plan.layout.padding.padded_n, "padded N")?,
                weight_row_stride_bytes: checked_u32(
                    plan.layout.weight_row_stride_bytes,
                    "weight row stride",
                )?,
                scale_rows: checked_u32(plan.layout.scale_shape[0], "scale rows")?,
                scale_cols: checked_u32(plan.layout.scale_shape[1], "scale columns")?,
                scale_row_stride_bytes: checked_u32(
                    plan.layout.scale_stride_bytes[0],
                    "scale row stride",
                )?,
                scale_col_stride_bytes: checked_u32(
                    plan.layout.scale_stride_bytes[1],
                    "scale column stride",
                )?,
                row_origin: checked_u32(plan.layout.row_origin, "row origin")?,
                column_origin: checked_u32(plan.layout.column_origin, "column origin")?,
                reserved: 0,
                weight: artifact.weight.cu_deviceptr(),
                scales: artifact.scales.cu_deviceptr(),
                decoded_weight_bf16: workspace.decoded_weight.cu_deviceptr(),
                stream: 0,
            };
            if let Err(error) = self.launch_fp8_block32_decode(stream, decode_args) {
                workspace.poison();
                return Err(error);
            }

            if let Some(padded) = workspace.activation_padded.as_ref() {
                let pack_args = Bf16Block32PackArgs {
                    rows: checked_u32(plan.m, "M")?,
                    k: checked_u32(plan.k, "K")?,
                    padded_k: checked_u32(gemm_k, "padded K")?,
                    input_stride: checked_u32(plan.k, "activation stride")?,
                    reserved: 0,
                    input_bf16: input.as_device_buffer().cu_deviceptr(),
                    output_bf16: padded.cu_deviceptr(),
                    stream: 0,
                };
                if let Err(error) = self.launch_bf16_block32_pack(stream, pack_args) {
                    workspace.poison();
                    return Err(error);
                }
            }

            let gemm_result = if let Some(padded) = workspace.activation_padded.as_ref() {
                bf16_gemm(
                    stream,
                    padded,
                    &workspace.decoded_weight,
                    output.as_device_buffer_mut(),
                    layout,
                )
            } else {
                bf16_gemm(
                    stream,
                    input.as_device_buffer(),
                    &workspace.decoded_weight,
                    output.as_device_buffer_mut(),
                    layout,
                )
            };
            if let Err(error) = gemm_result {
                workspace.poison();
                return Err(error);
            }
            Ok(())
        })
    }

    /// Submit the baseline and record the existing owner compute completion
    /// event immediately after it. The returned event does not own operands;
    /// callers retain artifact, input, output, and workspace until it completes.
    pub fn fp8_block32_linear_into_with_completion(
        &self,
        artifact: &CudaFp8Block32Artifact,
        input: &CudaBf16Buffer,
        output: &mut CudaF32Buffer,
        workspace: &mut CudaFp8Block32Workspace,
        plan: Fp8E4m3Ue8m0LinearPlan,
    ) -> Result<crate::cuda::operator_owner::CudaComputeEvent> {
        self.fp8_block32_linear_into(artifact, input, output, workspace, plan)?;
        match self.record_compute_event() {
            Ok(event) => Ok(event),
            Err(error) => {
                workspace.poison();
                Err(error)
            }
        }
    }
}
