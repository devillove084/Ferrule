//! SM80+ correctness baseline for row-major E4M3 + per-32 UE8M0 linear.
//!
//! This is intentionally a two-stage path: a clean-room scalar decode writes a
//! packed BF16 temporary weight, then the existing BF16 TensorOp provider
//! computes F32 output. It is not a fused FP8 kernel, Marlin path, or
//! performance implementation. The temporary weight and optional activation
//! padding are caller-owned workspace and must remain alive until the owner's
//! completion event proves the compute stream is finished.

use std::{cell::Cell, mem::size_of, sync::Arc};

use crate::cuda::architecture::{COMPILED_TARGET, compiled_capabilities};
use crate::cuda::ffi::core::{
    Bf16Block32PackArgs, Fp8Block32DecodeArgs, ferrule_core_fp8_block32_device_capability,
};
use crate::cuda::operator_owner::{
    CudaBf16Buffer, CudaComputeEvent, CudaF32Buffer, CudaOperators, cu,
};
use crate::cuda::operators::{OperatorOwner, linear::Bf16GemmLayout};
use crate::cuda::providers::cutlass::{bf16_gemm, bf16_gemm_can_implement};
use crate::cuda::runtime::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
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
    let activation_elements = plan
        .m
        .checked_mul(gemm_k)
        .ok_or_else(|| invalid("activation extent overflows"))?;
    if plan.m.checked_mul(plan.n).is_none() {
        return Err(invalid("output extent overflows"));
    }
    if gemm_k > plan.k && activation_elements > 256usize.saturating_mul(65_535) {
        return Err(invalid(
            "padded BF16 activation launch grid exceeds CUDA limit",
        ));
    }
    Ok(gemm_k)
}

fn decoded_weight_bytes_for_plan(plan: Fp8E4m3Ue8m0LinearPlan, gemm_k: usize) -> Result<usize> {
    plan.n
        .checked_mul(gemm_k)
        .and_then(|elements| elements.checked_mul(size_of::<u16>()))
        .ok_or_else(|| invalid("decoded BF16 weight size overflows"))
}

fn activation_workspace_bytes_for_plan(
    plan: Fp8E4m3Ue8m0LinearPlan,
    gemm_k: usize,
) -> Result<usize> {
    if gemm_k > plan.k {
        plan.m
            .checked_mul(gemm_k)
            .and_then(|elements| elements.checked_mul(size_of::<u16>()))
            .ok_or_else(|| invalid("padded BF16 activation workspace size overflows"))
    } else {
        Ok(0)
    }
}

fn workspace_bytes_for_plan(plan: Fp8E4m3Ue8m0LinearPlan, gemm_k: usize) -> Result<usize> {
    decoded_weight_bytes_for_plan(plan, gemm_k)?
        .checked_add(activation_workspace_bytes_for_plan(plan, gemm_k)?)
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

/// Explicit, owner-local persistent BF16 storage prepared from one encoded
/// artifact. The encoded source is retained as provenance and as an
/// asynchronous decode lifetime lease; it is not a global residency entry.
///
/// The caller owns the artifact lifetime and the extra
/// [`Self::decoded_weight_bytes`] BF16 bytes. Retain it until the final
/// consumer completion event has completed, and retire it explicitly when the
/// caller's memory budget permits. No implicit cache lookup or eviction is
/// performed by this type.
pub struct CudaDecodedFp8Block32Artifact {
    source: CudaFp8Block32Artifact,
    plan: Fp8E4m3Ue8m0LinearPlan,
    gemm_k: usize,
    decoded_weight: DeviceBuffer<u16>,
    ready_event: CudaComputeEvent,
    poisoned: Cell<bool>,
}

/// Short name for callers that do not need the CUDA prefix used by the older
/// encoded artifact type.
pub type DecodedFp8Block32Artifact = CudaDecodedFp8Block32Artifact;

impl CudaDecodedFp8Block32Artifact {
    pub fn plan(&self) -> Fp8E4m3Ue8m0LinearPlan {
        self.plan
    }

    pub fn gemm_k(&self) -> usize {
        self.gemm_k
    }

    /// Exact persistent BF16 storage charged to the caller's cache budget.
    pub fn decoded_weight_bytes(&self) -> usize {
        self.decoded_weight.num_bytes()
    }

    /// The encoded source remains owned by this artifact for provenance and
    /// until decode completion, even though steady GEMM does not read it.
    pub fn source_artifact(&self) -> &CudaFp8Block32Artifact {
        &self.source
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned.get()
    }

    /// Explicitly quarantine this artifact after an external completion/event
    /// failure. A quarantined artifact cannot be submitted again.
    pub fn quarantine(&self) {
        self.poison();
    }

    /// Query the owner-local decode event without synchronizing the stream.
    /// This standalone query has no activation workspace to quarantine; the
    /// decoded submission path passes its workspace to the same classifier.
    pub fn is_ready(&self) -> Result<bool> {
        if self.poisoned.get() {
            return Err(invalid("decoded artifact is quarantined"));
        }
        match self.ready_event.is_complete() {
            Ok(ready) => Ok(ready),
            Err(error) => {
                let (_, error) = classify_fp8_block32_event_failure(
                    self.decoded_weight.context(),
                    "query decode readiness",
                    error,
                    self,
                    None,
                );
                Err(error)
            }
        }
    }

    /// Wait for the one-time decode completion. Capture-active synchronization
    /// failure remains retryable; Unknown or an actual event failure quarantines
    /// this artifact. The steady same-owner-stream entry does not need to wait
    /// because the GEMM is ordered after the recorded event on that stream.
    pub fn wait_until_ready(&self) -> Result<()> {
        if self.poisoned.get() {
            return Err(invalid("decoded artifact is quarantined"));
        }
        match self.ready_event.synchronize() {
            Ok(()) => Ok(()),
            Err(error) => {
                let (_, error) = classify_fp8_block32_event_failure(
                    self.decoded_weight.context(),
                    "wait for decode readiness",
                    error,
                    self,
                    None,
                );
                Err(error)
            }
        }
    }

    fn poison(&self) {
        self.poisoned.set(true);
    }
}

/// Caller-owned activation-only storage used by the decoded path when logical
/// K is not already a provider-compatible multiple of eight. It intentionally
/// contains no decoded weight, so its byte budget cannot double-count the
/// persistent artifact.
pub struct CudaFp8Block32ActivationWorkspace {
    plan: Fp8E4m3Ue8m0LinearPlan,
    gemm_k: usize,
    /// Stored even for K8-compatible plans where no activation allocation is
    /// needed, so zero-byte workspaces remain owner/context-bound.
    owner_context: Arc<CudaContext>,
    activation_padded: Option<DeviceBuffer<u16>>,
    poisoned: Cell<bool>,
    workspace_bytes: usize,
}

impl CudaFp8Block32ActivationWorkspace {
    pub fn plan(&self) -> Fp8E4m3Ue8m0LinearPlan {
        self.plan
    }

    pub fn gemm_k(&self) -> usize {
        self.gemm_k
    }

    pub fn activation_padding_bytes(&self) -> usize {
        self.activation_padded
            .as_ref()
            .map(DeviceBuffer::num_bytes)
            .unwrap_or(0)
    }

    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned.get()
    }

    /// Explicitly quarantine transient activation storage when a caller-owned
    /// completion/event operation fails.
    pub fn quarantine(&self) {
        self.poison();
    }

    fn poison(&self) {
        self.poisoned.set(true);
    }
}

/// A one-shot completion proof bound to the exact decoded artifact and
/// activation workspace borrowed for one submission. The event is not
/// interchangeable with another event from the same owner or an older call.
pub struct Fp8Block32Completion<'a> {
    artifact: &'a CudaDecodedFp8Block32Artifact,
    workspace: &'a CudaFp8Block32ActivationWorkspace,
    plan: Fp8E4m3Ue8m0LinearPlan,
    generation: u64,
    owner_context: Arc<CudaContext>,
    owner_stream: Arc<CudaStream>,
    event: Option<CudaComputeEvent>,
    finalized: bool,
}

impl Fp8Block32Completion<'_> {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    fn consume_after_terminal_failure(&mut self) {
        self.event.take();
        self.finalized = true;
    }

    /// Consume the completion event exactly once. Capture-active is a
    /// retryable preflight state and leaves the token/resources usable;
    /// completion Unknown or an actual event synchronization failure
    /// quarantines both resources and consumes the token.
    pub fn finalize(&mut self, owner: &CudaOperators) -> Result<()> {
        if self.finalized || self.event.is_none() {
            return Err(invalid("FP8 block-32 completion was already finalized"));
        }
        let owner_context = owner.stream_clone();
        if !Arc::ptr_eq(&self.owner_context, owner_context.context())
            || !Arc::ptr_eq(&self.owner_stream, &owner_context)
        {
            return Err(invalid("FP8 block-32 completion owner/stream mismatch"));
        }
        if self.generation == 0
            || self.artifact.plan != self.plan
            || self.workspace.plan != self.plan
        {
            return Err(invalid("FP8 block-32 completion token metadata mismatch"));
        }
        if self.artifact.poisoned.get() || self.workspace.poisoned.get() {
            return Err(invalid(
                "decoded artifact or activation workspace was quarantined",
            ));
        }
        owner.check_buffer_owner(
            &self.artifact.source.weight,
            "FP8 block-32 completion source weight",
        )?;
        owner.check_buffer_owner(
            &self.artifact.source.scales,
            "FP8 block-32 completion source scales",
        )?;
        owner.check_buffer_owner(
            &self.artifact.decoded_weight,
            "FP8 block-32 completion decoded weight",
        )?;
        if !Arc::ptr_eq(&self.workspace.owner_context, owner_context.context()) {
            return Err(invalid("FP8 block-32 completion workspace mismatch"));
        }
        if let Some(padded) = self.workspace.activation_padded.as_ref() {
            owner.check_buffer_owner(padded, "FP8 block-32 completion activation")?;
        }
        let event = self.event.as_ref().expect("completion event checked above");
        cu(event
            .event()
            .check_context(owner_context.context(), "FP8 block-32 completion event"))?;

        let graph_state = match owner.fp8_block32_completion_graph_state() {
            Ok(state) => state,
            Err(error) => {
                let (state, error) = classify_fp8_block32_event_failure(
                    owner_context.context(),
                    "check FP8 block-32 completion readiness",
                    error,
                    self.artifact,
                    Some(self.workspace),
                );
                if !matches!(state, Fp8Block32GraphState::CaptureActive) {
                    self.consume_after_terminal_failure();
                }
                return Err(error);
            }
        };
        match graph_state {
            Fp8Block32GraphState::CaptureActive => {
                return Err(invalid(
                    "FP8 block-32 completion finalize deferred: CUDA capture active",
                ));
            }
            Fp8Block32GraphState::CompletionUnknown => {
                let (_, error) = classify_fp8_block32_event_failure(
                    owner_context.context(),
                    "finalize FP8 block-32 completion",
                    invalid("FP8 block-32 completion reached sticky completion Unknown"),
                    self.artifact,
                    Some(self.workspace),
                );
                self.consume_after_terminal_failure();
                return Err(error);
            }
            Fp8Block32GraphState::Ready => {}
        }

        match event.synchronize() {
            Ok(()) => {
                self.event.take();
                self.finalized = true;
                Ok(())
            }
            Err(error) => {
                let (state, error) = classify_fp8_block32_event_failure(
                    owner_context.context(),
                    "finalize FP8 block-32 completion",
                    error,
                    self.artifact,
                    Some(self.workspace),
                );
                if !matches!(state, Fp8Block32GraphState::CaptureActive) {
                    self.consume_after_terminal_failure();
                }
                Err(error)
            }
        }
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

/// Query only the transient activation-padding budget for the explicit decoded
/// path. Persistent decoded weight bytes are reported by the artifact instead.
pub fn fp8_block32_activation_workspace_requirements(
    plan: Fp8E4m3Ue8m0LinearPlan,
) -> Result<OperatorWorkspaceRequirements> {
    let gemm_k = ensure_supported(plan)?;
    Ok(OperatorWorkspaceRequirements {
        bytes: checked_u64(
            activation_workspace_bytes_for_plan(plan, gemm_k)?,
            "activation workspace bytes",
        )?,
        alignment: 16,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fp8Block32GraphState {
    Ready,
    CaptureActive,
    CompletionUnknown,
}

fn fp8_block32_context_graph_state(context: &CudaContext) -> Fp8Block32GraphState {
    if context.graph_completion_unknown() {
        Fp8Block32GraphState::CompletionUnknown
    } else if context.is_capturing() {
        Fp8Block32GraphState::CaptureActive
    } else {
        Fp8Block32GraphState::Ready
    }
}

/// Classify a failure from a decoded artifact or bound completion event.
/// Capture-active is the only retryable class. A sticky Unknown or an actual
/// event failure quarantines every resource supplied by the caller; standalone
/// artifact readiness intentionally supplies no workspace because no decoded
/// submission has associated one yet.
fn classify_fp8_block32_event_failure(
    context: &CudaContext,
    operation: &str,
    error: Error,
    artifact: &CudaDecodedFp8Block32Artifact,
    workspace: Option<&CudaFp8Block32ActivationWorkspace>,
) -> (Fp8Block32GraphState, Error) {
    let state = fp8_block32_context_graph_state(context);
    let error = match state {
        Fp8Block32GraphState::CaptureActive => invalid(format!(
            "{operation} deferred: CUDA capture active; retry after capture completes"
        )),
        Fp8Block32GraphState::CompletionUnknown => {
            artifact.poison();
            if let Some(workspace) = workspace {
                workspace.poison();
            }
            let resources = if workspace.is_some() {
                "artifact and activation workspace"
            } else {
                "decoded artifact"
            };
            invalid(format!(
                "{operation} reached sticky completion Unknown; {resources} quarantined"
            ))
        }
        Fp8Block32GraphState::Ready => {
            artifact.poison();
            if let Some(workspace) = workspace {
                workspace.poison();
            }
            error
        }
    };
    (state, error)
}

impl CudaOperators {
    fn fp8_block32_completion_graph_state(&self) -> Result<Fp8Block32GraphState> {
        let stream = self.stream_clone();
        let context = stream.context();
        if context.graph_completion_unknown() {
            return Ok(Fp8Block32GraphState::CompletionUnknown);
        }
        if context.is_capturing() {
            return Ok(Fp8Block32GraphState::CaptureActive);
        }
        match context.check_graph_ready() {
            Ok(()) => {}
            Err(error) => {
                if context.graph_completion_unknown() {
                    return Ok(Fp8Block32GraphState::CompletionUnknown);
                }
                if context.is_capturing() {
                    return Ok(Fp8Block32GraphState::CaptureActive);
                }
                return Err(error.into());
            }
        }
        cu(context.bind_to_thread())?;
        Ok(Fp8Block32GraphState::Ready)
    }

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

    fn check_fp8_block32_decoded_owners(
        &self,
        artifact: &CudaDecodedFp8Block32Artifact,
        input: &CudaBf16Buffer,
        output: &CudaF32Buffer,
        workspace: &CudaFp8Block32ActivationWorkspace,
    ) -> Result<()> {
        self.check_buffer_owner(&artifact.source.weight, "FP8 block-32 source weight")?;
        self.check_buffer_owner(&artifact.source.scales, "FP8 block-32 source scales")?;
        self.check_buffer_owner(
            &artifact.decoded_weight,
            "FP8 block-32 decoded artifact weight",
        )?;
        self.check_buffer_owner(input.as_device_buffer(), "FP8 block-32 activation")?;
        self.check_buffer_owner(output.as_device_buffer(), "FP8 block-32 output")?;
        if !Arc::ptr_eq(&workspace.owner_context, self.stream_clone().context()) {
            return Err(invalid("FP8 block-32 activation workspace owner mismatch"));
        }
        if let Some(padded) = workspace.activation_padded.as_ref() {
            self.check_buffer_owner(padded, "FP8 block-32 activation workspace")?;
        }
        cu(artifact.ready_event.event().check_context(
            self.stream_clone().context(),
            "FP8 block-32 decoded artifact ready event",
        ))?;
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

    /// Allocate only the transient activation padding required by the decoded
    /// path. For K8-compatible plans this owns no storage and reports zero
    /// bytes; decoded weight storage is owned by [`CudaDecodedFp8Block32Artifact`].
    pub fn fp8_block32_activation_workspace(
        &self,
        plan: Fp8E4m3Ue8m0LinearPlan,
    ) -> Result<CudaFp8Block32ActivationWorkspace> {
        self.fp8_block32_context_preflight()?;
        let gemm_k = ensure_supported(plan)?;
        let activation_bytes = activation_workspace_bytes_for_plan(plan, gemm_k)?;
        let activation_elements = if gemm_k > plan.k {
            Some(
                plan.m
                    .checked_mul(gemm_k)
                    .ok_or_else(|| invalid("padded BF16 activation workspace size overflows"))?,
            )
        } else {
            None
        };
        let activation_padded =
            self.allocate_fp8_block32_activation_workspace_buffer(activation_elements)?;
        if activation_padded
            .as_ref()
            .map(DeviceBuffer::num_bytes)
            .unwrap_or(0)
            != activation_bytes
        {
            return Err(invalid("FP8 block-32 activation workspace size mismatch"));
        }
        Ok(CudaFp8Block32ActivationWorkspace {
            plan,
            gemm_k,
            owner_context: Arc::clone(self.stream_clone().context()),
            activation_padded,
            poisoned: Cell::new(false),
            workspace_bytes: activation_bytes,
        })
    }

    /// Prepare one explicit persistent BF16 decoded-weight artifact. The
    /// encoded source is moved into the returned artifact and remains owned
    /// until the caller retires that artifact after its completion event.
    /// Preparation submits exactly one decode launch and no GEMM.
    pub fn prepare_fp8_block32_decoded_weight(
        &self,
        source: CudaFp8Block32Artifact,
        plan: Fp8E4m3Ue8m0LinearPlan,
    ) -> Result<CudaDecodedFp8Block32Artifact> {
        self.fp8_block32_graph_preflight()?;
        let gemm_k = ensure_supported(plan)?;
        self.check_buffer_owner(&source.weight, "FP8 block-32 source weight")?;
        self.check_buffer_owner(&source.scales, "FP8 block-32 source scales")?;
        if source.plan != plan {
            return Err(invalid("encoded source/prepare plan mismatch"));
        }
        self.fp8_block32_runtime_capability()?;
        let decoded_elements = plan
            .n
            .checked_mul(gemm_k)
            .ok_or_else(|| invalid("decoded BF16 artifact size overflows"))?;
        let decoded_weight = self.allocate_fp8_block32_decoded_weight_buffer(decoded_elements)?;
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
            weight: source.weight.cu_deviceptr(),
            scales: source.scales.cu_deviceptr(),
            decoded_weight_bf16: decoded_weight.cu_deviceptr(),
            stream: 0,
        };
        self.submit_operator(1, |stream| {
            self.launch_fp8_block32_decode(stream, decode_args)
        })?;
        let ready_event = self.record_compute_event()?;
        Ok(CudaDecodedFp8Block32Artifact {
            source,
            plan,
            gemm_k,
            decoded_weight,
            ready_event,
            poisoned: Cell::new(false),
        })
    }

    /// Submit the explicit decoded path. After preparation this performs only
    /// owner/graph/plan/readiness checks and the existing BF16 GEMM, plus the
    /// required activation pack for ragged K. The decoded weight is never
    /// allocated or written here. The launch counter is committed only after
    /// the semantic submission closure returns successfully; a partial native
    /// submit is quarantined and is not a completion or partial-launch ledger.
    pub fn fp8_block32_linear_from_decoded(
        &self,
        artifact: &CudaDecodedFp8Block32Artifact,
        input: &CudaBf16Buffer,
        output: &mut CudaF32Buffer,
        workspace: &mut CudaFp8Block32ActivationWorkspace,
        plan: Fp8E4m3Ue8m0LinearPlan,
    ) -> Result<()> {
        if let Err(error) = self.fp8_block32_graph_preflight() {
            if self.stream_clone().context().graph_completion_unknown() {
                artifact.poison();
                workspace.poison();
            }
            return Err(error);
        }
        let gemm_k = ensure_supported(plan)?;
        self.check_fp8_block32_decoded_owners(artifact, input, output, workspace)?;
        self.fp8_block32_runtime_capability()?;
        if artifact.plan != plan
            || artifact.gemm_k != gemm_k
            || workspace.plan != plan
            || workspace.gemm_k != gemm_k
        {
            return Err(invalid("decoded artifact/workspace/plan mismatch"));
        }
        if artifact.poisoned.get() || workspace.poisoned.get() {
            return Err(invalid(
                "decoded artifact or activation workspace was quarantined",
            ));
        }
        if let Err(error) = artifact.ready_event.is_complete() {
            let (_, error) = classify_fp8_block32_event_failure(
                artifact.decoded_weight.context(),
                "query decode readiness",
                error,
                artifact,
                Some(workspace),
            );
            return Err(error);
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
        if let Some(padded) = workspace.activation_padded.as_ref() {
            bf16_gemm_can_implement(
                &self.stream_clone(),
                padded,
                &artifact.decoded_weight,
                output.as_device_buffer(),
                layout,
            )?;
        } else {
            bf16_gemm_can_implement(
                &self.stream_clone(),
                input.as_device_buffer(),
                &artifact.decoded_weight,
                output.as_device_buffer(),
                layout,
            )?;
        }

        let needs_activation_padding = workspace.activation_padded.is_some();
        self.submit_operator(if needs_activation_padding { 2 } else { 1 }, |stream| {
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
                    artifact.poison();
                    workspace.poison();
                    return Err(error);
                }
            }

            let gemm_result = if let Some(padded) = workspace.activation_padded.as_ref() {
                bf16_gemm(
                    stream,
                    padded,
                    &artifact.decoded_weight,
                    output.as_device_buffer_mut(),
                    layout,
                )
            } else {
                bf16_gemm(
                    stream,
                    input.as_device_buffer(),
                    &artifact.decoded_weight,
                    output.as_device_buffer_mut(),
                    layout,
                )
            };
            if let Err(error) = gemm_result {
                artifact.poison();
                workspace.poison();
                return Err(error);
            }
            Ok(())
        })
    }

    /// Submit the explicit decoded path and return a one-shot completion token
    /// bound to this exact artifact/workspace/plan submission. The token keeps
    /// those borrows alive until its finalize attempt completes.
    pub fn fp8_block32_linear_from_decoded_with_completion<'a>(
        &self,
        artifact: &'a CudaDecodedFp8Block32Artifact,
        input: &CudaBf16Buffer,
        output: &mut CudaF32Buffer,
        workspace: &'a mut CudaFp8Block32ActivationWorkspace,
        plan: Fp8E4m3Ue8m0LinearPlan,
    ) -> Result<Fp8Block32Completion<'a>> {
        self.fp8_block32_linear_from_decoded(artifact, input, output, workspace, plan)?;
        let event = match self.record_compute_event() {
            Ok(event) => event,
            Err(error) => {
                artifact.poison();
                workspace.poison();
                return Err(error);
            }
        };
        Ok(Fp8Block32Completion {
            artifact,
            workspace,
            plan,
            generation: self.next_fp8_block32_completion_generation(),
            owner_context: Arc::clone(self.stream_clone().context()),
            owner_stream: self.stream_clone(),
            event: Some(event),
            finalized: false,
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
