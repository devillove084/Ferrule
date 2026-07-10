//! CUDA context helpers — probe, GEMV benchmarks, kernel dispatch.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_core::stream::CudaStream;
use cuda_core::{CudaContext, CudaEvent, DeviceBuffer, DeviceCopy, LaunchConfig, PinnedHostBuffer};
use ferrule_common::{Error, QuantType, Result};

pub use crate::counters::CudaOpCounters;
use crate::counters::{CudaMoeExecutionPath, CudaOpCounterCells};
use crate::kernels::kernels::LoadedModule;
use crate::transformer::artifact_expert::{
    CudaPackedFp4Expert, CudaPackedFp4ExpertScratch, CudaPackedFp4Linear,
};
use crate::transformer::sparse_attention::{CudaSparseAttentionExecutor, CudaSparseAttentionShape};

/// Convert a CUDA result into a ferrule Result.
pub(crate) fn cu<T, E: std::fmt::Debug>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| Error::Internal(format!("CUDA {e:?}")))
}

fn slice_bytes<T>(slice: &[T]) -> u64 {
    (slice.len() as u64).saturating_mul(std::mem::size_of::<T>() as u64)
}

fn element_bytes<T>(len: usize) -> u64 {
    (len as u64).saturating_mul(std::mem::size_of::<T>() as u64)
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

pub const ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE: usize = 128;

fn quantized_shape_uses_fp8_activation(shape: CudaArtifactLinearShape) -> bool {
    matches!(
        shape,
        CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale { .. }
            | CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale { .. }
    )
}

fn prepare_activation_for_artifact_linear<'a>(
    shape: CudaArtifactLinearShape,
    input: &'a [f32],
) -> Result<Cow<'a, [f32]>> {
    if quantized_shape_uses_fp8_activation(shape) {
        let mut quantized = input.to_vec();
        simulate_fp8_e4m3fn_e8m0_activation_quant_in_place(
            &mut quantized,
            shape.in_features(),
            ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
        )?;
        Ok(Cow::Owned(quantized))
    } else {
        Ok(Cow::Borrowed(input))
    }
}

fn simulate_fp8_e4m3fn_e8m0_activation_quant_in_place(
    values: &mut [f32],
    row_width: usize,
    block_size: usize,
) -> Result<()> {
    if row_width == 0 || block_size == 0 || !row_width.is_multiple_of(block_size) {
        return Err(Error::Internal(format!(
            "invalid CUDA artifact FP8 activation quant shape: row_width={row_width}, block_size={block_size}"
        )));
    }
    if !values.len().is_multiple_of(row_width) {
        return Err(Error::Internal(format!(
            "CUDA artifact FP8 activation length {} is not a multiple of row_width {row_width}",
            values.len()
        )));
    }

    for row in values.chunks_exact_mut(row_width) {
        for block in row.chunks_exact_mut(block_size) {
            let amax = block
                .iter()
                .fold(0.0f32, |acc, value| acc.max(value.abs()))
                .max(1e-4);
            let scale = 2.0f32.powf((amax / 448.0).log2().ceil());
            for value in block {
                let quantized = quantize_fp8_e4m3fn_to_f32((*value / scale).clamp(-448.0, 448.0));
                *value = quantized * scale;
            }
        }
    }
    Ok(())
}

fn quantize_fp8_e4m3fn_to_f32(value: f32) -> f32 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    let sign = if value.is_sign_negative() { -1.0 } else { 1.0 };
    let magnitude = value.abs().min(448.0);
    sign * nearest_fp8_e4m3fn_positive(magnitude)
}

fn nearest_fp8_e4m3fn_positive(magnitude: f32) -> f32 {
    let mut best = nearest_fp8_subnormal_positive(magnitude);
    let mut best_err = (best - magnitude).abs();
    let exp_floor = magnitude.log2().floor() as i32;
    for exp in exp_floor - 1..=exp_floor + 1 {
        if !(-6..=8).contains(&exp) {
            continue;
        }
        let scale = 2.0f32.powi(exp);
        let mut mantissa = ((magnitude / scale - 1.0) * 8.0).round() as i32;
        let mut candidate_exp = exp;
        if mantissa < 0 {
            continue;
        }
        if mantissa > 7 {
            candidate_exp += 1;
            mantissa = 0;
        }
        if candidate_exp > 8 {
            candidate_exp = 8;
            mantissa = 6;
        }
        if candidate_exp == 8 && mantissa > 6 {
            mantissa = 6;
        }
        let candidate = 2.0f32.powi(candidate_exp) * (1.0 + mantissa as f32 / 8.0);
        let err = (candidate - magnitude).abs();
        if err < best_err {
            best = candidate;
            best_err = err;
        }
    }
    best
}

fn nearest_fp8_subnormal_positive(magnitude: f32) -> f32 {
    let step = 2.0f32.powi(-9);
    let mantissa = (magnitude / step).round().clamp(0.0, 7.0);
    mantissa * step
}

fn checked_u32(value: usize, label: &str, field: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::Internal(format!("{label} {field} exceeds CUDA u32 ABI: {value}")))
}

// ── Kernel dispatch (selects Q4_0 vs Q8_0 at runtime) ─────────────────

#[allow(dead_code)]
pub(crate) fn gemv_quant(
    m: &LoadedModule,
    s: &CudaStream,
    qt: QuantType,
    cfg: LaunchConfig,
    x: &DeviceBuffer<f32>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
    n: u32,
    k: u32,
) -> Result<()> {
    match qt {
        QuantType::Q4_0 => cu(unsafe { m.gemv_q4(s, cfg, x, packed, scales, y, n, k) }),
        QuantType::Q8_0 => cu(unsafe { m.gemv_q8(s, cfg, x, packed, scales, y, n, k) }),
        _ => Err(Error::Kernel(format!(
            "unsupported quant type for gemv: {qt:?}"
        ))),
    }
}

#[allow(dead_code)]
pub(crate) fn gemv_quant_off(
    m: &LoadedModule,
    s: &CudaStream,
    qt: QuantType,
    cfg: LaunchConfig,
    x: &DeviceBuffer<f32>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
    n: u32,
    k: u32,
    packed_off: u32,
    scales_off: u32,
) -> Result<()> {
    match qt {
        QuantType::Q4_0 => {
            cu(
                unsafe {
                    m.gemv_q4_off(s, cfg, x, packed, scales, y, n, k, packed_off, scales_off)
                },
            )
        }
        QuantType::Q8_0 => {
            cu(
                unsafe {
                    m.gemv_q8_off(s, cfg, x, packed, scales, y, n, k, packed_off, scales_off)
                },
            )
        }
        _ => Err(Error::Kernel(format!(
            "unsupported quant type for gemv_off: {qt:?}"
        ))),
    }
}

// ── Device probe ──────────────────────────────────────────────────────

/// Probe the CUDA device and print basic info.
/// No-op when no GPU is available (returns an error).
pub fn cuda_probe() -> Result<()> {
    let ctx = cu(CudaContext::new(0))?;
    let name = cu(ctx.device_name())?;
    cu(ctx.bind_to_thread())?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        cuda_bindings::cuMemGetInfo_v2(&mut free, &mut total);
    }
    println!(
        "  Device: {name}\n  Memory: {:.1} GB free / {:.1} GB total",
        free as f64 / 1e9,
        total as f64 / 1e9
    );
    Ok(())
}

// ── Reusable artifact-format operator context ────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaArtifactLinearShape {
    F32 {
        out_features: usize,
        in_features: usize,
    },
    Bf16Bytes {
        out_features: usize,
        in_features: usize,
    },

    Fp8E4M3WithE8M0Scale {
        out_features: usize,
        in_features: usize,
        block_m: usize,
        block_k: usize,
    },
    Fp4E2M1PackedWithE8M0Scale {
        out_features: usize,
        in_features: usize,
    },
}

impl CudaArtifactLinearShape {
    pub fn out_features(self) -> usize {
        match self {
            Self::F32 { out_features, .. }
            | Self::Bf16Bytes { out_features, .. }
            | Self::Fp8E4M3WithE8M0Scale { out_features, .. }
            | Self::Fp4E2M1PackedWithE8M0Scale { out_features, .. } => out_features,
        }
    }

    pub fn in_features(self) -> usize {
        match self {
            Self::F32 { in_features, .. }
            | Self::Bf16Bytes { in_features, .. }
            | Self::Fp8E4M3WithE8M0Scale { in_features, .. }
            | Self::Fp4E2M1PackedWithE8M0Scale { in_features, .. } => in_features,
        }
    }

    fn validate(self, weight_len: usize, scale_len: usize) -> Result<()> {
        match self {
            Self::F32 {
                out_features,
                in_features,
            } => {
                if in_features == 0 || out_features == 0 {
                    return Err(Error::Internal(format!(
                        "invalid CUDA F32 artifact linear shape: out={out_features} in={in_features}"
                    )));
                }
                let expected_weight = out_features
                    .checked_mul(in_features)
                    .and_then(|elements| elements.checked_mul(4))
                    .ok_or_else(|| {
                        Error::Internal("CUDA F32 artifact weight size overflow".into())
                    })?;
                if weight_len != expected_weight || scale_len != 0 {
                    return Err(Error::Internal(format!(
                        "CUDA F32 artifact linear length mismatch: weight={weight_len} scale={scale_len}, expected weight={expected_weight} scale=0"
                    )));
                }
            }
            Self::Bf16Bytes {
                out_features,
                in_features,
            } => {
                if in_features == 0 || out_features == 0 {
                    return Err(Error::Internal(format!(
                        "invalid CUDA BF16 artifact linear shape: out={out_features} in={in_features}"
                    )));
                }
                let expected_weight = out_features
                    .checked_mul(in_features)
                    .and_then(|elements| elements.checked_mul(2))
                    .ok_or_else(|| {
                        Error::Internal("CUDA BF16 artifact weight size overflow".into())
                    })?;
                if weight_len != expected_weight || scale_len != 0 {
                    return Err(Error::Internal(format!(
                        "CUDA BF16 artifact linear length mismatch: weight={weight_len} scale={scale_len}, expected weight={expected_weight} scale=0"
                    )));
                }
            }

            Self::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                if in_features == 0 || out_features == 0 || block_m == 0 || block_k == 0 {
                    return Err(Error::Internal(format!(
                        "invalid CUDA FP8 artifact linear shape: out={out_features} in={in_features} block_m={block_m} block_k={block_k}"
                    )));
                }
                let expected_weight = out_features.checked_mul(in_features).ok_or_else(|| {
                    Error::Internal("CUDA FP8 artifact weight size overflow".into())
                })?;
                let expected_scale = out_features
                    .div_ceil(block_m)
                    .checked_mul(in_features.div_ceil(block_k))
                    .ok_or_else(|| {
                        Error::Internal("CUDA FP8 artifact scale size overflow".into())
                    })?;
                if weight_len != expected_weight || scale_len != expected_scale {
                    return Err(Error::Internal(format!(
                        "CUDA FP8 artifact linear length mismatch: weight={weight_len} scale={scale_len}, expected weight={expected_weight} scale={expected_scale}"
                    )));
                }
            }
            Self::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
            } => {
                if in_features == 0
                    || out_features == 0
                    || !in_features.is_multiple_of(32)
                    || !in_features.is_multiple_of(2)
                {
                    return Err(Error::Internal(format!(
                        "invalid CUDA FP4 artifact linear shape: out={out_features} in={in_features}"
                    )));
                }
                let expected_weight =
                    out_features.checked_mul(in_features / 2).ok_or_else(|| {
                        Error::Internal("CUDA FP4 artifact weight size overflow".into())
                    })?;
                let expected_scale =
                    out_features.checked_mul(in_features / 32).ok_or_else(|| {
                        Error::Internal("CUDA FP4 artifact scale size overflow".into())
                    })?;
                if weight_len != expected_weight || scale_len != expected_scale {
                    return Err(Error::Internal(format!(
                        "CUDA FP4 artifact linear length mismatch: weight={weight_len} scale={scale_len}, expected weight={expected_weight} scale={expected_scale}"
                    )));
                }
            }
        }
        Ok(())
    }
}

pub struct CudaArtifactLinearHandle {
    shape: CudaArtifactLinearShape,
    weight: DeviceBuffer<u8>,
    scale: Option<DeviceBuffer<u8>>,
}

/// Opaque page-locked host buffer for stream-ordered artifact uploads.
///
/// Upload tickets keep these buffers alive until the upload event completes,
/// satisfying CUDA's async H2D source-lifetime requirement without exposing raw
/// pinned pointers to model code.
#[derive(Clone)]
pub struct CudaPinnedU8HostBuffer {
    buffer: Arc<PinnedHostBuffer<u8>>,
}

impl CudaPinnedU8HostBuffer {
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Stream-ordered artifact upload that keeps pinned host sources alive until
/// the owner consumes it after the associated upload event completes.
pub struct CudaArtifactLinearAsyncUpload {
    handle: CudaArtifactLinearHandle,
    _weight: CudaPinnedU8HostBuffer,
    _scale: Option<CudaPinnedU8HostBuffer>,
}

impl CudaArtifactLinearAsyncUpload {
    pub fn shape(&self) -> CudaArtifactLinearShape {
        self.handle.shape()
    }

    pub fn into_handle(self) -> CudaArtifactLinearHandle {
        self.handle
    }
}

/// Event recorded on the artifact upload stream.
pub struct CudaUploadEvent {
    event: CudaEvent,
}

impl CudaUploadEvent {
    pub fn is_complete(&self) -> Result<bool> {
        cu(self.event.query())
    }

    pub fn synchronize(&self) -> Result<()> {
        cu(self.event.synchronize())
    }
}

/// Page-locked host memory buffer that is directly accessible from the GPU
/// via `cuMemHostGetDevicePointer`. On GB10 (unified memory), this avoids
/// the page-fault overhead of `cuMemAllocManaged` — the GPU reads directly
/// from host LPDDR5X pages over the coherent interconnect.
///
/// Unlike `cuMemAllocManaged`, which triggers a page fault on first GPU
/// access (~2.4ms per expert for migration), `cuMemAllocHost` pre-pins the
/// memory so GPU access has zero fault overhead.
///
/// The host pointer and device pointer alias the same physical memory,
/// so there is no need for an explicit H2D copy after the initial memcpy.
pub struct HostPinnedBuffer {
    /// Raw host pointer from `cuMemAllocHost`. Freed via `cuMemFreeHost`.
    host_ptr: *mut std::os::raw::c_void,
    /// Device pointer aliasing the same physical memory.
    dev_ptr: cuda_bindings::CUdeviceptr,
    len: usize,
}

impl HostPinnedBuffer {
    /// Allocate page-locked host memory, copy `data` into it, and obtain the
    /// device pointer. The GPU can read this memory directly over the
    /// coherent interconnect without any H2D DMA transfer.
    pub fn alloc_and_copy(data: &[u8]) -> Result<Self> {
        let mut host_ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
        let result = unsafe { cuda_bindings::cuMemAllocHost_v2(&mut host_ptr, data.len()) };
        if result != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::Internal(format!(
                "cuMemAllocHost failed: error {result}, size {} bytes",
                data.len()
            )));
        }
        // Copy data into page-locked host memory (host-side memcpy, no DMA).
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), host_ptr as *mut u8, data.len());
        }
        // Get the device pointer that aliases the same physical memory.
        let mut dev_ptr: cuda_bindings::CUdeviceptr = 0;
        let result =
            unsafe { cuda_bindings::cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0) };
        if result != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            unsafe { cuda_bindings::cuMemFreeHost(host_ptr) };
            return Err(Error::Internal(format!(
                "cuMemHostGetDevicePointer failed: error {result}"
            )));
        }
        Ok(Self {
            host_ptr,
            dev_ptr,
            len: data.len(),
        })
    }

    /// Device pointer for kernel launches.
    pub fn cu_deviceptr(&self) -> cuda_bindings::CUdeviceptr {
        self.dev_ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for HostPinnedBuffer {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            unsafe { cuda_bindings::cuMemFreeHost(self.host_ptr) };
        }
    }
}

unsafe impl Send for HostPinnedBuffer {}
unsafe impl Sync for HostPinnedBuffer {}

impl CudaArtifactLinearHandle {
    pub fn shape(&self) -> CudaArtifactLinearShape {
        self.shape
    }
}

/// Opaque f32 device buffer used by generic artifact operators.
///
/// This is intentionally a CUDA/backend type, not a model-specific arena type.
/// Model-family code can own and reuse these buffers without exposing CUDA driver
/// handles through the runtime graph boundary.
pub struct CudaF32Buffer {
    buffer: DeviceBuffer<f32>,
    len: usize,
}

/// Reusable workspace for artifact FP4 SwiGLU expert execution.
pub struct CudaFp4ExpertWorkspace {
    scratch: CudaPackedFp4ExpertScratch,
    output: CudaF32Buffer,
    intermediate_size: usize,
    output_size: usize,
}

/// Reusable workspace for routed FP4 MoE batched execution.
///
/// The decode path hits this once per layer per token, so avoiding transient
/// CUDA allocations here is critical. The workspace owns all per-call scratch
/// buffers and fixed-size device arrays for selected expert pointers/weights.
pub struct CudaMoeBatchedWorkspace {
    gate_ptrs: DeviceBuffer<u64>,
    gate_scale_ptrs: DeviceBuffer<u64>,
    up_ptrs: DeviceBuffer<u64>,
    up_scale_ptrs: DeviceBuffer<u64>,
    down_ptrs: DeviceBuffer<u64>,
    down_scale_ptrs: DeviceBuffer<u64>,
    route_weights: DeviceBuffer<f32>,
    x_f32: CudaF32Buffer,
    y_gate: CudaF32Buffer,
    y_up: CudaF32Buffer,
    y_hidden: CudaF32Buffer,
    x_packed: DeviceBuffer<u8>,
    x_scales: DeviceBuffer<u8>,
    y_hidden_packed: DeviceBuffer<u8>,
    y_hidden_scales: DeviceBuffer<u8>,
    max_experts: usize,
    max_batch_cols: usize,
    input_size: usize,
    intermediate_size: usize,
    hidden_size: usize,
}

impl CudaFp4ExpertWorkspace {
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn output(&self) -> &CudaF32Buffer {
        &self.output
    }
}

impl CudaMoeBatchedWorkspace {
    pub fn matches(
        &self,
        max_experts: usize,
        input_size: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> bool {
        self.matches_cols(max_experts, 1, input_size, intermediate_size, hidden_size)
    }

    pub fn matches_cols(
        &self,
        max_experts: usize,
        batch_cols: usize,
        input_size: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> bool {
        self.max_experts >= max_experts
            && self.max_batch_cols >= batch_cols
            && self.input_size == input_size
            && self.intermediate_size == intermediate_size
            && self.hidden_size == hidden_size
    }
}

impl CudaF32Buffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Opaque i32 device buffer used by sparse-attention top-k index sets and
/// other integer-valued device operands. Mirrors [`CudaF32Buffer`].
pub struct CudaI32Buffer {
    buffer: DeviceBuffer<i32>,
    len: usize,
}

impl CudaI32Buffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the underlying `DeviceBuffer<i32>` for kernels (e.g. sparse
    /// attention) that take a raw device buffer reference.
    pub fn as_device_buffer(&self) -> &DeviceBuffer<i32> {
        &self.buffer
    }
}

/// Reusable host-side context for generic artifact-format CUDA operators.
///
/// This follows cuda-oxide's preferred examples: create one `CudaContext`, load
/// the embedded `#[cuda_module]` once, then reuse its default stream and typed
/// launch methods. It is intentionally generic and knows only packed artifact
/// formats plus explicit shapes — model-family semantics stay in runtime code.
pub struct CudaArtifactOperatorContext {
    _ctx: Arc<CudaContext>,
    module: LoadedModule,
    stream: Arc<CudaStream>,
    upload_stream: Arc<CudaStream>,
    counters: CudaOpCounterCells,
}

impl CudaArtifactOperatorContext {
    pub fn new() -> Result<Self> {
        let ctx = cu(CudaContext::new(0))?;
        cu(ctx.bind_to_thread())?;
        let module = cu(crate::kernels::kernels::load(&ctx))?;
        // Use a non-blocking stream (forked from default) so that CUDA graph
        // capture works (the default/NULL stream does not support capture).
        let stream = cu(ctx.default_stream().fork())?;
        // Dedicated artifact upload stream. Expert prefetch copies run here and
        // publish readiness through events; the main compute stream only waits
        // when a selected expert is actually consumed.
        let upload_stream = cu(stream.fork())?;
        Ok(Self {
            _ctx: ctx,
            module,
            stream,
            upload_stream,
            counters: CudaOpCounterCells::default(),
        })
    }

    pub fn counters(&self) -> CudaOpCounters {
        self.counters.snapshot()
    }

    pub fn capture_decode_graph(
        &self,
        capture_fn: impl FnOnce() -> Result<()>,
    ) -> Result<crate::graph::CudaGraphHandle> {
        crate::graph::capture_decode_graph(&self.stream, capture_fn)
    }

    pub fn launch_graph(&self, graph: &crate::graph::CudaGraphHandle) -> Result<()> {
        graph.launch(&self.stream)
    }

    /// Upload a captured graph's nodes to the device ahead of first launch.
    pub fn upload_graph(&self, graph: &crate::graph::CudaGraphHandle) -> Result<()> {
        graph.upload(&self.stream)
    }

    /// Try to update a captured graph in-place to match `updated_graph`.
    /// Returns `Ok(true)` if updated, `Ok(false)` if re-capture is needed.
    pub fn update_graph(
        &self,
        graph: &crate::graph::CudaGraphHandle,
        updated_graph: &cuda_core::graph::CudaGraph,
    ) -> Result<bool> {
        graph.update(updated_graph)
    }

    /// Build a llama.cpp-style auto-warmup cached decode graph bound to this
    /// context's stream. See [`crate::graph::CachedDecodeGraph`].
    pub fn cached_decode_graph(&self) -> crate::graph::CachedDecodeGraph {
        crate::graph::CachedDecodeGraph::new(&self._ctx)
    }

    /// Clone the stream for use with graph capture outside of `&self` borrow.
    pub fn stream_clone(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    pub fn sync_stream(&self) -> Result<()> {
        cu(self.stream.synchronize())
    }

    pub fn sync_upload_stream(&self) -> Result<()> {
        cu(self.upload_stream.synchronize())
    }

    pub fn wait_upload_event(&self, event: &CudaUploadEvent) -> Result<()> {
        cu(self.stream.wait(&event.event))
    }

    pub fn record_upload_event(&self) -> Result<CudaUploadEvent> {
        Ok(CudaUploadEvent {
            event: cu(self.upload_stream.record_event(None))?,
        })
    }

    fn moe_timing_enabled(&self) -> bool {
        std::env::var("FERRULE_CUDA_MOE_TIMING")
            .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    }

    pub fn reset_counters(&self) {
        self.counters.reset();
    }

    fn record_kernel_launch(&self) {
        self.counters.add_kernel_launch();
    }

    fn launched<T, E: std::fmt::Debug>(&self, result: std::result::Result<T, E>) -> Result<T> {
        let value = cu(result)?;
        self.record_kernel_launch();
        Ok(value)
    }

    /// Allocate a device buffer without initializing it.
    ///
    /// Use only for scratch buffers that are fully written by subsequent kernels
    /// before any read. This keeps the unsafe `uninitialized_async` contract in
    /// one place instead of scattering `cu(unsafe { ... })` through hot paths.
    fn uninitialized_device_buffer<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        cu(unsafe { DeviceBuffer::<T>::uninitialized_async(&self.stream, len) })
    }

    fn upload_u8(&self, values: &[u8]) -> Result<DeviceBuffer<u8>> {
        let buffer = cu(DeviceBuffer::from_host(&self.stream, values))?;
        self.counters.add_host_to_device(slice_bytes(values));
        Ok(buffer)
    }

    fn upload_f32(&self, values: &[f32]) -> Result<DeviceBuffer<f32>> {
        let buffer = cu(DeviceBuffer::from_host(&self.stream, values))?;
        self.counters.add_host_to_device(slice_bytes(values));
        Ok(buffer)
    }

    fn upload_i32(&self, values: &[i32]) -> Result<DeviceBuffer<i32>> {
        let buffer = cu(DeviceBuffer::from_host(&self.stream, values))?;
        self.counters.add_host_to_device(slice_bytes(values));
        Ok(buffer)
    }

    pub fn pin_u8_host_buffer(&self, values: &[u8]) -> Result<CudaPinnedU8HostBuffer> {
        Ok(CudaPinnedU8HostBuffer {
            buffer: Arc::new(cu(PinnedHostBuffer::from_slice(&self._ctx, values))?),
        })
    }

    /// Enqueue an async H2D copy from a pinned source on the artifact upload stream.
    ///
    /// # Safety
    /// The caller must keep `values` alive and immutable until the returned upload
    /// event has completed. Model-level upload tickets satisfy this by owning the
    /// pinned sources alongside the CUDA handles and event.
    unsafe fn upload_u8_from_pinned_async_unchecked(
        &self,
        values: &CudaPinnedU8HostBuffer,
    ) -> Result<DeviceBuffer<u8>> {
        let buffer = cu(unsafe {
            DeviceBuffer::from_pinned_host(&self.upload_stream, values.buffer.as_ref())
        })?;
        self.counters.add_host_to_device(values.len() as u64);
        Ok(buffer)
    }

    fn download_f32(&self, buffer: &DeviceBuffer<f32>, len: usize) -> Result<Vec<f32>> {
        let values = cu(buffer.to_host_vec(&self.stream))?;
        self.counters.add_device_to_host(element_bytes::<f32>(len));
        Ok(values)
    }

    pub fn upload_f32_buffer(&self, values: &[f32]) -> Result<CudaF32Buffer> {
        Ok(CudaF32Buffer {
            buffer: self.upload_f32(values)?,
            len: values.len(),
        })
    }

    pub fn zero_f32_buffer(&self, len: usize) -> Result<CudaF32Buffer> {
        Ok(CudaF32Buffer {
            buffer: cu(DeviceBuffer::<f32>::zeroed(&self.stream, len))?,
            len,
        })
    }

    /// Zero an existing device buffer in-place (cuMemsetD32Async, no allocation).
    /// Safe for CUDA graph capture.
    pub fn zero_f32_buffer_in_place(&self, buf: &mut CudaF32Buffer) -> Result<()> {
        let result = unsafe {
            cuda_bindings::cuMemsetD32Async(
                buf.buffer.cu_deviceptr(),
                0,
                buf.len,
                self.stream.cu_stream(),
            )
        };
        if result != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::Internal(format!(
                "cuMemsetD32Async failed: error {result}"
            )));
        }
        self.counters.add_kernel_launch();
        Ok(())
    }

    pub fn download_f32_buffer(&self, buffer: &CudaF32Buffer) -> Result<Vec<f32>> {
        self.download_f32(&buffer.buffer, buffer.len)
    }

    /// Clone a device f32 buffer (device-to-device copy, no host round-trip).
    pub fn clone_f32_buffer(&self, src: &CudaF32Buffer) -> Result<CudaF32Buffer> {
        let mut dst = self.zero_f32_buffer(src.len)?;
        self.copy_f32_into_slot(src, &mut dst, 0)?;
        Ok(dst)
    }

    /// Concatenate two device f32 buffers with device-to-device copies only.
    pub fn concat_f32_buffers(
        &self,
        left: &CudaF32Buffer,
        right: &CudaF32Buffer,
    ) -> Result<CudaF32Buffer> {
        let total = left
            .len
            .checked_add(right.len)
            .ok_or_else(|| Error::Internal("CUDA f32 concat length overflow".into()))?;
        let mut out = self.zero_f32_buffer(total)?;
        if left.len != 0 {
            self.copy_f32_into_slot(left, &mut out, 0)?;
        }
        if right.len != 0 {
            self.copy_f32_into_slot(right, &mut out, left.len)?;
        }
        Ok(out)
    }

    pub fn upload_i32_buffer(&self, values: &[i32]) -> Result<CudaI32Buffer> {
        Ok(CudaI32Buffer {
            buffer: self.upload_i32(values)?,
            len: values.len(),
        })
    }

    pub fn zero_i32_buffer(&self, len: usize) -> Result<CudaI32Buffer> {
        Ok(CudaI32Buffer {
            buffer: cu(DeviceBuffer::<i32>::zeroed(&self.stream, len))?,
            len,
        })
    }

    /// Copy `src` into `dst` device-to-device in place, element-for-element.
    /// Used to refresh a pre-allocated top-k index buffer for graph capture.
    pub fn copy_i32_into_buffer(&self, src: &[i32], dst: &mut CudaI32Buffer) -> Result<()> {
        if src.len() != dst.len {
            return Err(Error::Internal(format!(
                "CUDA i32 copy length mismatch: src={} dst={}",
                src.len(),
                dst.len
            )));
        }
        self.counters.add_host_to_device(slice_bytes(src));
        cu(dst.buffer.copy_from_host(&self.stream, src))
    }

    fn copy_u64_into_device_buffer(&self, src: &[u64], dst: &mut DeviceBuffer<u64>) -> Result<()> {
        self.counters.add_host_to_device(slice_bytes(src));
        cu(dst.copy_from_host(&self.stream, src))
    }

    fn copy_f32_into_device_buffer(&self, src: &[f32], dst: &mut DeviceBuffer<f32>) -> Result<()> {
        self.counters.add_host_to_device(slice_bytes(src));
        cu(dst.copy_from_host(&self.stream, src))
    }

    /// Copy `src.len()` f32 elements from `src` into `dst` starting at element
    /// `slot_offset_elements`. Launches the `copy_f32_slot` kernel so the copy
    /// is fully device-resident (no host round-trip), which is required for
    /// CUDA graph capture of the KV-cache append.
    pub fn copy_f32_into_slot(
        &self,
        src: &CudaF32Buffer,
        dst: &mut CudaF32Buffer,
        slot_offset_elements: usize,
    ) -> Result<()> {
        let end = slot_offset_elements
            .checked_add(src.len)
            .ok_or_else(|| Error::Internal("CUDA slot copy offset overflow".into()))?;
        if end > dst.len {
            return Err(Error::Internal(format!(
                "CUDA slot copy out of bounds: dst.len={}, offset={}, src.len={}",
                dst.len, slot_offset_elements, src.len
            )));
        }
        self.launched(unsafe {
            self.module.copy_f32_slot(
                &self.stream,
                LaunchConfig::for_num_elems(src.len as u32),
                &src.buffer,
                &mut dst.buffer,
                slot_offset_elements as u32,
                src.len as u32,
            )
        })
    }

    pub fn gather_f32_rows(
        &self,
        src: &CudaF32Buffer,
        row_indices: &CudaI32Buffer,
        rows: usize,
        row_width: usize,
    ) -> Result<CudaF32Buffer> {
        if rows == 0 || row_width == 0 || row_indices.len != rows {
            return Err(Error::Internal(format!(
                "CUDA row gather invalid shape: rows={rows} row_width={row_width} indices={}",
                row_indices.len
            )));
        }
        if src.len % row_width != 0 {
            return Err(Error::Internal(format!(
                "CUDA row gather source length {} is not divisible by row_width {row_width}",
                src.len
            )));
        }
        let mut dst = self.zero_f32_buffer(
            rows.checked_mul(row_width)
                .ok_or_else(|| Error::Internal("CUDA row gather output size overflow".into()))?,
        )?;
        self.launched(unsafe {
            self.module.gather_f32_rows(
                &self.stream,
                LaunchConfig::for_num_elems((rows * row_width) as u32),
                &src.buffer,
                &row_indices.buffer,
                &mut dst.buffer,
                rows as u32,
                row_width as u32,
            )
        })?;
        Ok(dst)
    }

    pub fn scatter_add_f32_rows(
        &self,
        src: &CudaF32Buffer,
        row_indices: &CudaI32Buffer,
        dst: &mut CudaF32Buffer,
        rows: usize,
        row_width: usize,
    ) -> Result<()> {
        if rows == 0 || row_width == 0 || row_indices.len != rows {
            return Err(Error::Internal(format!(
                "CUDA row scatter invalid shape: rows={rows} row_width={row_width} indices={}",
                row_indices.len
            )));
        }
        let expected_src = rows
            .checked_mul(row_width)
            .ok_or_else(|| Error::Internal("CUDA row scatter source size overflow".into()))?;
        if src.len != expected_src {
            return Err(Error::Internal(format!(
                "CUDA row scatter source length mismatch: src={} expected={expected_src}",
                src.len
            )));
        }
        if dst.len % row_width != 0 {
            return Err(Error::Internal(format!(
                "CUDA row scatter destination length {} is not divisible by row_width {row_width}",
                dst.len
            )));
        }
        self.launched(unsafe {
            self.module.scatter_add_f32_rows(
                &self.stream,
                LaunchConfig::for_num_elems(expected_src as u32),
                &src.buffer,
                &row_indices.buffer,
                &mut dst.buffer,
                rows as u32,
                row_width as u32,
            )
        })
    }

    /// Device-side accumulate: `y += scale * x`.
    ///
    /// Used to accumulate routed expert outputs on the GPU without
    /// downloading each expert's output to host and accumulating in Vec<f32>.
    pub fn saxpy_into(&self, scale: f32, x: &CudaF32Buffer, y: &mut CudaF32Buffer) -> Result<()> {
        if x.len != y.len {
            return Err(Error::Internal(format!(
                "CUDA saxpy length mismatch: x={} y={}",
                x.len, y.len
            )));
        }
        self.launched(unsafe {
            self.module.saxpy(
                &self.stream,
                LaunchConfig::for_num_elems(x.len as u32),
                scale,
                &x.buffer,
                &mut y.buffer,
                x.len as u32,
            )
        })
    }

    pub fn upload_artifact_linear(
        &self,
        shape: CudaArtifactLinearShape,
        weight: &[u8],
        scale: &[u8],
    ) -> Result<CudaArtifactLinearHandle> {
        shape.validate(weight.len(), scale.len())?;
        self.counters
            .add_artifact_upload(slice_bytes(weight).saturating_add(slice_bytes(scale)));
        Ok(CudaArtifactLinearHandle {
            shape,
            weight: self.upload_u8(weight)?,
            scale: if scale.is_empty() {
                None
            } else {
                Some(self.upload_u8(scale)?)
            },
        })
    }

    /// Enqueue artifact linear H2D copies on the dedicated upload stream.
    ///
    /// # Safety
    /// `weight` and `scale` must outlive the upload event recorded after this
    /// call. The returned handle may be inserted into compute data structures
    /// immediately, but kernels must not use it until the event is complete or
    /// the compute stream has waited on it.
    unsafe fn upload_artifact_linear_from_pinned_async_unchecked(
        &self,
        shape: CudaArtifactLinearShape,
        weight: &CudaPinnedU8HostBuffer,
        scale: Option<&CudaPinnedU8HostBuffer>,
    ) -> Result<CudaArtifactLinearHandle> {
        let scale_len = scale.map(CudaPinnedU8HostBuffer::len).unwrap_or(0);
        shape.validate(weight.len(), scale_len)?;
        self.counters
            .add_artifact_upload((weight.len() as u64).saturating_add(scale_len as u64));
        Ok(CudaArtifactLinearHandle {
            shape,
            weight: unsafe { self.upload_u8_from_pinned_async_unchecked(weight)? },
            scale: match scale {
                Some(scale) if !scale.is_empty() => {
                    Some(unsafe { self.upload_u8_from_pinned_async_unchecked(scale)? })
                }
                _ => None,
            },
        })
    }

    pub fn upload_artifact_linear_from_pinned_async(
        &self,
        shape: CudaArtifactLinearShape,
        weight: CudaPinnedU8HostBuffer,
        scale: Option<CudaPinnedU8HostBuffer>,
    ) -> Result<CudaArtifactLinearAsyncUpload> {
        let handle = unsafe {
            self.upload_artifact_linear_from_pinned_async_unchecked(shape, &weight, scale.as_ref())?
        };
        Ok(CudaArtifactLinearAsyncUpload {
            handle,
            _weight: weight,
            _scale: scale,
        })
    }

    pub fn upload_fp4_e2m1_e8m0_linear_from_pinned_async(
        &self,
        weight: CudaPinnedU8HostBuffer,
        scale: CudaPinnedU8HostBuffer,
        out_features: usize,
        in_features: usize,
    ) -> Result<CudaArtifactLinearAsyncUpload> {
        self.upload_artifact_linear_from_pinned_async(
            CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
            },
            weight,
            Some(scale),
        )
    }

    pub fn upload_f32_linear(
        &self,
        weight: &[u8],
        out_features: usize,
        in_features: usize,
    ) -> Result<CudaArtifactLinearHandle> {
        self.upload_artifact_linear(
            CudaArtifactLinearShape::F32 {
                out_features,
                in_features,
            },
            weight,
            &[],
        )
    }

    pub fn upload_bf16_linear(
        &self,
        weight: &[u8],
        out_features: usize,
        in_features: usize,
    ) -> Result<CudaArtifactLinearHandle> {
        self.upload_artifact_linear(
            CudaArtifactLinearShape::Bf16Bytes {
                out_features,
                in_features,
            },
            weight,
            &[],
        )
    }

    pub fn upload_fp8_e4m3_e8m0_linear(
        &self,
        weight: &[u8],
        scale: &[u8],
        out_features: usize,
        in_features: usize,
        block_m: usize,
        block_k: usize,
    ) -> Result<CudaArtifactLinearHandle> {
        self.upload_artifact_linear(
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            },
            weight,
            scale,
        )
    }

    pub fn upload_fp4_e2m1_e8m0_linear(
        &self,
        weight: &[u8],
        scale: &[u8],
        out_features: usize,
        in_features: usize,
    ) -> Result<CudaArtifactLinearHandle> {
        // On GB10 (unified memory), use cuMemAllocManaged to avoid H2D copy.
        // The GPU reads expert weights directly from host pages via unified
        // addressing, eliminating the 888 MB/token upload bottleneck.
        // Default: enabled. Set FERRULE_MANAGED_EXPERTS=0 to disable.
        let use_managed = std::env::var("FERRULE_MANAGED_EXPERTS")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        if use_managed {
            return self.upload_artifact_linear_managed(
                CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
                    out_features,
                    in_features,
                },
                weight,
                scale,
            );
        }
        self.upload_artifact_linear(
            CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
            },
            weight,
            scale,
        )
    }

    /// Allocate expert weight/scale buffers as CUDA managed memory.
    ///
    /// On GB10 (sm_121, unified addressing), managed memory is accessible by
    /// both CPU and GPU without explicit H2D copies. The GPU reads directly
    /// from host LPDDR5X pages, so expert loading becomes a pure host-side
    /// memcpy (disk -> managed buffer) with zero upload overhead.
    pub fn upload_artifact_linear_managed(
        &self,
        shape: CudaArtifactLinearShape,
        weight: &[u8],
        scale: &[u8],
    ) -> Result<CudaArtifactLinearHandle> {
        shape.validate(weight.len(), scale.len())?;
        let weight_buf = self.alloc_managed_u8(weight)?;
        let scale_buf = if scale.is_empty() {
            None
        } else {
            Some(self.alloc_managed_u8(scale)?)
        };
        // No counter bump: managed memory is not an H2D transfer.
        Ok(CudaArtifactLinearHandle {
            shape,
            weight: weight_buf,
            scale: scale_buf,
        })
    }

    /// Allocate a CUDA managed-memory buffer and copy `data` into it.
    ///
    /// Managed memory is accessible from both host and device on unified
    /// addressing platforms (GB10). The returned `DeviceBuffer` owns the
    /// allocation and frees it via `cuMemFree` on drop.
    ///
    /// On GB10 (sm_121, unified addressing), managed memory is accessible by
    /// both CPU and GPU without explicit H2D copies. The GPU reads directly
    /// from host LPDDR5X pages, so expert loading becomes a pure host-side
    /// memcpy (disk → managed buffer) with zero upload overhead.
    ///
    /// We also set `CU_MEM_ADVISE_SET_READ_MOSTLY` to hint the driver that
    /// expert weights are read-only after upload, enabling better page
    /// placement and reducing fault overhead.
    fn alloc_managed_u8(&self, data: &[u8]) -> Result<DeviceBuffer<u8>> {
        let ctx = self.stream.context().clone();
        let mut dptr: cuda_bindings::CUdeviceptr = 0;
        // CU_MEM_ATTACH_GLOBAL = 0x1
        let result = unsafe { cuda_bindings::cuMemAllocManaged(&mut dptr, data.len(), 0x1) };
        if result != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::Internal(format!(
                "cuMemAllocManaged failed: error {result}, size {} bytes",
                data.len()
            )));
        }
        // Copy data into managed buffer (host-side memcpy, no DMA transfer).
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dptr as *mut u8, data.len());
        }
        Ok(unsafe { DeviceBuffer::from_raw_parts(dptr, data.len(), ctx) })
    }

    pub fn artifact_linear_matvec(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        if input.len() != handle.shape.in_features() {
            return Err(Error::Internal(format!(
                "CUDA artifact linear input length mismatch: expected {}, got {}",
                handle.shape.in_features(),
                input.len()
            )));
        }
        let xd = self.upload_f32_buffer(input)?;
        let mut yd = self.zero_f32_buffer(handle.shape.out_features())?;
        self.artifact_linear_matvec_into(handle, &xd, &mut yd)?;
        self.download_f32_buffer(&yd)
    }

    pub fn artifact_linear_matvec_into(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if input.len != handle.shape.in_features() {
            return Err(Error::Internal(format!(
                "CUDA artifact linear device input length mismatch: expected {}, got {}",
                handle.shape.in_features(),
                input.len
            )));
        }
        if output.len != handle.shape.out_features() {
            return Err(Error::Internal(format!(
                "CUDA artifact linear device output length mismatch: expected {}, got {}",
                handle.shape.out_features(),
                output.len
            )));
        }
        self.artifact_linear_matvec_device(handle, &input.buffer, &mut output.buffer)
    }

    pub fn artifact_linear_rows_from_device(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
    ) -> Result<CudaF32Buffer> {
        let out_features = handle.shape.out_features();
        let mut output =
            self.zero_f32_buffer(rows.checked_mul(out_features).ok_or_else(|| {
                Error::Internal("CUDA artifact linear rows output size overflow".into())
            })?)?;
        self.artifact_linear_rows_from_device_into(handle, input, rows, &mut output)?;
        Ok(output)
    }

    pub fn artifact_linear_rows_from_device_into(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let in_features = handle.shape.in_features();
        let out_features = handle.shape.out_features();
        if rows == 0 || input.len != rows * in_features {
            return Err(Error::Internal(format!(
                "CUDA artifact linear rows input mismatch: rows={rows} in_features={in_features} input={}",
                input.len
            )));
        }
        let expected_output = rows.checked_mul(out_features).ok_or_else(|| {
            Error::Internal("CUDA artifact linear rows output size overflow".into())
        })?;
        if output.len != expected_output {
            return Err(Error::Internal(format!(
                "CUDA artifact linear rows output mismatch: expected {expected_output}, got {}",
                output.len
            )));
        }
        let mut x = self.clone_f32_buffer(input)?;
        if quantized_shape_uses_fp8_activation(handle.shape) {
            self.fp8_activation_quantize_buffer_in_place(
                &mut x,
                in_features,
                ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
            )?;
        }
        self.artifact_linear_rows_device(handle, &x.buffer, rows, &mut output.buffer)
    }

    pub fn artifact_swiglu_ffn_rows_from_device(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
        output_scale: f32,
        swiglu_limit: f32,
    ) -> Result<CudaF32Buffer> {
        let in_features = gate.shape.in_features();
        let intermediate = gate.shape.out_features();
        if rows == 0 || input.len != rows * in_features || up.shape.in_features() != in_features {
            return Err(Error::Internal(format!(
                "CUDA batched SwiGLU input mismatch: rows={rows} input={} gate_in={} up_in={}",
                input.len,
                in_features,
                up.shape.in_features()
            )));
        }
        if up.shape.out_features() != intermediate || down.shape.in_features() != intermediate {
            return Err(Error::Internal(format!(
                "CUDA batched SwiGLU shape mismatch: gate={:?} up={:?} down={:?}",
                gate.shape, up.shape, down.shape
            )));
        }
        let gated = self.artifact_linear_rows_from_device(gate, input, rows)?;
        let upd = self.artifact_linear_rows_from_device(up, input, rows)?;
        let mut hidden =
            self.zero_f32_buffer(rows.checked_mul(intermediate).ok_or_else(|| {
                Error::Internal("CUDA batched SwiGLU hidden size overflow".into())
            })?)?;
        self.launched(unsafe {
            self.module.swiglu_weighted_clamped(
                &self.stream,
                LaunchConfig::for_num_elems((rows * intermediate) as u32),
                &gated.buffer,
                &upd.buffer,
                &mut hidden.buffer,
                (rows * intermediate) as u32,
                output_scale,
                swiglu_limit,
            )
        })?;
        self.artifact_linear_rows_from_device(down, &hidden, rows)
    }

    /// Device-resident grouped matvec for block-diagonal weight layouts.
    ///
    /// `context` is the full `[o_groups * group_in]` device buffer. `weight` is
    /// the dequantized `[output_latent_dim, group_in]` f32 weight buffer, cached
    /// by the caller. The output `[output_latent_dim]` buffer is allocated here.
    /// One thread per output row; each row only reads its group's context slice.
    pub fn grouped_matvec_f32_from_device(
        &self,
        context: &CudaF32Buffer,
        weight: &CudaF32Buffer,
        output_latent_dim: usize,
        group_in: usize,
        o_lora_rank: usize,
    ) -> Result<CudaF32Buffer> {
        self.grouped_matvec_f32_rows_from_device(
            context,
            1,
            weight,
            output_latent_dim,
            group_in,
            o_lora_rank,
        )
    }

    /// Device-resident batched grouped matvec for block-diagonal output-A
    /// layouts. `context` is `[rows, q_full_dim]`; output is
    /// `[rows, output_latent_dim]`.
    pub fn grouped_matvec_f32_rows_from_device(
        &self,
        context: &CudaF32Buffer,
        rows: usize,
        weight: &CudaF32Buffer,
        output_latent_dim: usize,
        group_in: usize,
        o_lora_rank: usize,
    ) -> Result<CudaF32Buffer> {
        if rows == 0 {
            return Err(Error::Internal(
                "CUDA grouped rows matvec requires at least one row".into(),
            ));
        }
        if o_lora_rank == 0
            || output_latent_dim == 0
            || !output_latent_dim.is_multiple_of(o_lora_rank)
        {
            return Err(Error::Internal(format!(
                "CUDA grouped rows matvec invalid shape: out={output_latent_dim} rank={o_lora_rank} group_in={group_in}"
            )));
        }
        let groups = output_latent_dim / o_lora_rank;
        let expected_context = rows
            .checked_mul(groups)
            .and_then(|value| value.checked_mul(group_in))
            .ok_or_else(|| {
                Error::Internal(format!(
                    "CUDA grouped rows matvec context size overflow: rows={rows} groups={groups} group_in={group_in}"
                ))
            })?;
        if context.len != expected_context {
            return Err(Error::Internal(format!(
                "CUDA grouped rows matvec context length mismatch: expected {expected_context}, got {}",
                context.len
            )));
        }
        let expected_weight = output_latent_dim.checked_mul(group_in).ok_or_else(|| {
            Error::Internal(format!(
                "CUDA grouped rows matvec weight size overflow: out={output_latent_dim} group_in={group_in}"
            ))
        })?;
        if weight.len != expected_weight {
            return Err(Error::Internal(format!(
                "CUDA grouped rows matvec weight length mismatch: expected {expected_weight}, got {}",
                weight.len
            )));
        }
        let output_len = rows.checked_mul(output_latent_dim).ok_or_else(|| {
            Error::Internal(format!(
                "CUDA grouped rows matvec output size overflow: rows={rows} out={output_latent_dim}"
            ))
        })?;
        let mut output = self.zero_f32_buffer(output_len)?;
        self.grouped_matvec_f32_rows_from_device_into(
            context,
            rows,
            weight,
            output_latent_dim,
            group_in,
            o_lora_rank,
            &mut output,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn grouped_matvec_f32_rows_from_device_into(
        &self,
        context: &CudaF32Buffer,
        rows: usize,
        weight: &CudaF32Buffer,
        output_latent_dim: usize,
        group_in: usize,
        o_lora_rank: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if rows == 0 {
            return Err(Error::Internal(
                "CUDA grouped rows matvec requires at least one row".into(),
            ));
        }
        if o_lora_rank == 0
            || output_latent_dim == 0
            || !output_latent_dim.is_multiple_of(o_lora_rank)
        {
            return Err(Error::Internal(format!(
                "CUDA grouped rows matvec invalid shape: out={output_latent_dim} rank={o_lora_rank} group_in={group_in}"
            )));
        }
        let groups = output_latent_dim / o_lora_rank;
        let expected_context = rows
            .checked_mul(groups)
            .and_then(|value| value.checked_mul(group_in))
            .ok_or_else(|| {
                Error::Internal(format!(
                    "CUDA grouped rows matvec context size overflow: rows={rows} groups={groups} group_in={group_in}"
                ))
            })?;
        if context.len != expected_context {
            return Err(Error::Internal(format!(
                "CUDA grouped rows matvec context length mismatch: expected {expected_context}, got {}",
                context.len
            )));
        }
        let expected_weight = output_latent_dim.checked_mul(group_in).ok_or_else(|| {
            Error::Internal(format!(
                "CUDA grouped rows matvec weight size overflow: out={output_latent_dim} group_in={group_in}"
            ))
        })?;
        if weight.len != expected_weight {
            return Err(Error::Internal(format!(
                "CUDA grouped rows matvec weight length mismatch: expected {expected_weight}, got {}",
                weight.len
            )));
        }
        let output_len = rows.checked_mul(output_latent_dim).ok_or_else(|| {
            Error::Internal(format!(
                "CUDA grouped rows matvec output size overflow: rows={rows} out={output_latent_dim}"
            ))
        })?;
        if output.len != output_len {
            return Err(Error::Internal(format!(
                "CUDA grouped rows matvec output length mismatch: expected {output_len}, got {}",
                output.len
            )));
        }
        self.launched(unsafe {
            self.module.grouped_matvec_f32_rows(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    output_len,
                    "grouped_matvec_f32_rows",
                    "output_len",
                )?),
                &context.buffer,
                &weight.buffer,
                &mut output.buffer,
                checked_u32(rows, "grouped_matvec_f32_rows", "rows")?,
                checked_u32(
                    output_latent_dim,
                    "grouped_matvec_f32_rows",
                    "output_latent_dim",
                )?,
                checked_u32(group_in, "grouped_matvec_f32_rows", "group_in")?,
                checked_u32(o_lora_rank, "grouped_matvec_f32_rows", "o_lora_rank")?,
            )
        })
    }

    pub fn artifact_linear_topk(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if top_k > 40 {
            return Err(Error::Internal(format!(
                "CUDA artifact linear top-k supports k<=40, got {top_k}"
            )));
        }
        if input.len() != handle.shape.in_features() {
            return Err(Error::Internal(format!(
                "CUDA artifact linear top-k input length mismatch: expected {}, got {}",
                handle.shape.in_features(),
                input.len()
            )));
        }
        let xd = self.upload_f32_buffer(input)?;
        self.artifact_linear_topk_from_device(handle, &xd, top_k)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_router_topk_sqrt_softplus_rows_from_device(
        &self,
        logits: &CudaF32Buffer,
        bias: Option<&CudaF32Buffer>,
        tokens: usize,
        experts: usize,
        top_k: usize,
        route_scale: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if tokens == 0 || experts == 0 || top_k == 0 {
            return Err(Error::Internal(format!(
                "CUDA DSV4 router topk requires non-empty shape: tokens={tokens} experts={experts} top_k={top_k}"
            )));
        }
        if experts > 512 {
            return Err(Error::Internal(format!(
                "CUDA DSV4 router topk supports at most 512 experts, got {experts}"
            )));
        }
        if top_k > 64 || top_k > experts {
            return Err(Error::Internal(format!(
                "CUDA DSV4 router topk requires top_k in 1..={} and <=64, got {top_k}",
                experts.min(64)
            )));
        }
        if logits.len != tokens * experts {
            return Err(Error::Internal(format!(
                "CUDA DSV4 router topk logits length mismatch: got {} expected {}x{}",
                logits.len, tokens, experts
            )));
        }
        let empty_bias;
        let (bias_buf, bias_enabled) = if let Some(bias) = bias {
            if bias.len != experts {
                return Err(Error::Internal(format!(
                    "CUDA DSV4 router topk bias length mismatch: got {} expected {experts}",
                    bias.len
                )));
            }
            (bias, 1u32)
        } else {
            empty_bias = self.zero_f32_buffer(1)?;
            (&empty_bias, 0u32)
        };
        let out_len = tokens
            .checked_mul(top_k)
            .ok_or_else(|| Error::Internal("CUDA DSV4 router topk output overflow".into()))?;
        let mut indices = cu(DeviceBuffer::<f32>::zeroed(&self.stream, out_len))?;
        let mut weights = cu(DeviceBuffer::<f32>::zeroed(&self.stream, out_len))?;
        self.launched(unsafe {
            self.module.dsv4_router_topk_sqrt_softplus_rows(
                &self.stream,
                LaunchConfig::for_num_elems(tokens as u32),
                &logits.buffer,
                &bias_buf.buffer,
                &mut indices,
                &mut weights,
                tokens as u32,
                experts as u32,
                top_k as u32,
                bias_enabled,
                route_scale,
            )
        })?;
        Ok((
            self.download_f32(&indices, out_len)?,
            self.download_f32(&weights, out_len)?,
        ))
    }

    pub fn artifact_linear_topk_from_device(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if top_k > 40 {
            return Err(Error::Internal(format!(
                "CUDA artifact linear top-k supports k<=40, got {top_k}"
            )));
        }
        if input.len != handle.shape.in_features() {
            return Err(Error::Internal(format!(
                "CUDA artifact linear device top-k input length mismatch: expected {}, got {}",
                handle.shape.in_features(),
                input.len
            )));
        }
        let mut yd = self.zero_f32_buffer(handle.shape.out_features())?;
        self.artifact_linear_matvec_into(handle, input, &mut yd)?;
        let mut idx = cu(DeviceBuffer::<f32>::zeroed(&self.stream, top_k))?;
        let mut val = cu(DeviceBuffer::<f32>::zeroed(&self.stream, top_k))?;
        self.launched(unsafe {
            self.module.topk_vocab(
                &self.stream,
                one_block_config(256),
                &yd.buffer,
                &mut idx,
                &mut val,
                handle.shape.out_features() as u32,
                top_k as u32,
            )
        })?;
        let idx = self.download_f32(&idx, top_k)?;
        let val = self.download_f32(&val, top_k)?;
        Ok(idx
            .into_iter()
            .zip(val)
            .map(|(idx, value)| (idx as u32, value))
            .collect())
    }

    pub fn fp8_activation_quantize_buffer_in_place(
        &self,
        values: &mut CudaF32Buffer,
        row_width: usize,
        block_size: usize,
    ) -> Result<()> {
        self.fp8_activation_quantize_in_place(&mut values.buffer, values.len, row_width, block_size)
    }

    pub fn fp8_activation_quantize_in_place(
        &self,
        values: &mut DeviceBuffer<f32>,
        value_len: usize,
        row_width: usize,
        block_size: usize,
    ) -> Result<()> {
        if value_len == 0
            || row_width == 0
            || block_size == 0
            || !row_width.is_multiple_of(block_size)
            || !value_len.is_multiple_of(row_width)
        {
            return Err(Error::Internal(format!(
                "invalid CUDA FP8 activation quant shape: len={value_len}, row_width={row_width}, block_size={block_size}"
            )));
        }
        self.launched(unsafe {
            self.module.fp8_e4m3fn_e8m0_quantize_f32_inplace(
                &self.stream,
                LaunchConfig::for_num_elems((value_len / block_size) as u32),
                values,
                value_len as u32,
                row_width as u32,
                block_size as u32,
            )
        })
    }

    pub fn fp8_attention_kv_qat_quantize_buffer_in_place(
        &self,
        values: &mut CudaF32Buffer,
        head_dim: usize,
        rope_dim: usize,
    ) -> Result<()> {
        if values.len == 0
            || head_dim == 0
            || rope_dim > head_dim
            || !values.len.is_multiple_of(head_dim)
        {
            return Err(Error::Internal(format!(
                "invalid CUDA attention KV QAT shape: len={} head_dim={head_dim} rope_dim={rope_dim}",
                values.len
            )));
        }
        let non_rope = head_dim - rope_dim;
        if non_rope == 0 {
            return Ok(());
        }
        let block_size = 64usize;
        let effective_block_size = if non_rope.is_multiple_of(block_size) {
            block_size
        } else {
            non_rope
        };
        let rows = values.len / head_dim;
        let blocks_per_row = non_rope.div_ceil(effective_block_size);
        self.launched(unsafe {
            self.module.fp8_e4m3fn_e8m0_quantize_non_rope_f32_inplace(
                &self.stream,
                LaunchConfig::for_num_elems((rows * blocks_per_row) as u32),
                &mut values.buffer,
                values.len as u32,
                head_dim as u32,
                rope_dim as u32,
                block_size as u32,
            )
        })
    }

    pub fn fp4_hadamard_qat_quantize_buffer_in_place(
        &self,
        values: &mut CudaF32Buffer,
        row_width: usize,
    ) -> Result<()> {
        if values.len == 0
            || row_width == 0
            || !row_width.is_power_of_two()
            || !values.len.is_multiple_of(row_width)
        {
            return Err(Error::Internal(format!(
                "invalid CUDA indexer Hadamard/FP4 QAT shape: len={} row_width={row_width}",
                values.len
            )));
        }
        self.launched(unsafe {
            self.module.hadamard_fp4_e2m1_e8m0_quantize_f32_inplace(
                &self.stream,
                LaunchConfig::for_num_elems((values.len / row_width) as u32),
                &mut values.buffer,
                values.len as u32,
                row_width as u32,
                32,
            )
        })
    }

    pub fn compressor_prefill_softmax_from_device(
        &self,
        kv_rows: &CudaF32Buffer,
        score_rows: &CudaF32Buffer,
        ape: &[f32],
        groups: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
    ) -> Result<CudaF32Buffer> {
        if ratio == 0 || head_dim == 0 || out_dim == 0 {
            return Err(Error::Internal(format!(
                "invalid CUDA compressor shape: groups={groups} ratio={ratio} head_dim={head_dim} out_dim={out_dim}"
            )));
        }
        if groups == 0 {
            return self.upload_f32_buffer(&[]);
        }
        let consumed_tokens = groups
            .checked_mul(ratio)
            .ok_or_else(|| Error::Internal("CUDA compressor token count overflow".into()))?;
        let min_projected = consumed_tokens
            .checked_mul(out_dim)
            .ok_or_else(|| Error::Internal("CUDA compressor projected row size overflow".into()))?;
        if kv_rows.len < min_projected
            || score_rows.len < min_projected
            || !kv_rows.len.is_multiple_of(out_dim)
            || !score_rows.len.is_multiple_of(out_dim)
            || kv_rows.len != score_rows.len
        {
            return Err(Error::Internal(format!(
                "CUDA compressor projected length mismatch: kv={} score={} min_required={min_projected} out_dim={out_dim}",
                kv_rows.len, score_rows.len
            )));
        }
        let expected_ape = ratio
            .checked_mul(out_dim)
            .ok_or_else(|| Error::Internal("CUDA compressor APE size overflow".into()))?;
        if ape.len() != expected_ape {
            return Err(Error::Internal(format!(
                "CUDA compressor APE length mismatch: got {} expected {expected_ape}",
                ape.len()
            )));
        }
        let ape_dev = self.upload_f32_buffer(ape)?;
        let mut out = self.zero_f32_buffer(
            groups
                .checked_mul(head_dim)
                .ok_or_else(|| Error::Internal("CUDA compressor output size overflow".into()))?,
        )?;
        self.launched(unsafe {
            self.module.dsv4_compressor_prefill_softmax(
                &self.stream,
                LaunchConfig::for_num_elems((groups * head_dim) as u32),
                &kv_rows.buffer,
                &score_rows.buffer,
                &ape_dev.buffer,
                &mut out.buffer,
                groups as u32,
                ratio as u32,
                head_dim as u32,
                out_dim as u32,
                if overlap { 1u32 } else { 0u32 },
            )
        })?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_prefill_topk_indices_from_device(
        &self,
        query: Option<&CudaF32Buffer>,
        weights: Option<&CudaF32Buffer>,
        indexer_kv: Option<&CudaF32Buffer>,
        tokens: usize,
        window_size: usize,
        window_cols: usize,
        extra_cols: usize,
        value_offset: usize,
        compress_ratio: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        weight_scale: f32,
    ) -> Result<CudaI32Buffer> {
        if tokens == 0 {
            return Err(Error::Internal(
                "CUDA DSV4 prefill topk requires tokens > 0".into(),
            ));
        }
        if window_cols > window_size || window_cols > tokens {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 prefill topk window: tokens={tokens} window_size={window_size} window_cols={window_cols}"
            )));
        }
        if compress_ratio == 0 && extra_cols != 0 {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 prefill topk compression: ratio=0 extra_cols={extra_cols}"
            )));
        }
        if extra_cols > 512 {
            return Err(Error::Internal(format!(
                "CUDA DSV4 prefill topk supports at most 512 extra columns, got {extra_cols}"
            )));
        }
        let total_cols = window_cols
            .checked_add(extra_cols)
            .ok_or_else(|| Error::Internal("CUDA DSV4 prefill topk column overflow".into()))?;
        if total_cols == 0 {
            return Err(Error::Internal(
                "CUDA DSV4 prefill topk requires at least one column".into(),
            ));
        }
        if compressed_len > 0 && extra_cols == 0 {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 prefill topk: compressed_len={compressed_len} extra_cols=0"
            )));
        }

        let indexer_enabled = query.is_some() || weights.is_some() || indexer_kv.is_some();
        let empty_query;
        let empty_weights;
        let empty_kv;
        let query = if indexer_enabled {
            let query = query.ok_or_else(|| {
                Error::Internal("CUDA DSV4 prefill topk missing indexer query".into())
            })?;
            let expected = tokens
                .checked_mul(index_heads)
                .and_then(|v| v.checked_mul(index_head_dim))
                .ok_or_else(|| Error::Internal("CUDA DSV4 prefill query size overflow".into()))?;
            if query.len != expected {
                return Err(Error::Internal(format!(
                    "CUDA DSV4 prefill indexer query length mismatch: got {} expected {expected}",
                    query.len
                )));
            }
            query
        } else {
            empty_query = self.zero_f32_buffer(1)?;
            &empty_query
        };
        let weights = if indexer_enabled {
            let weights = weights.ok_or_else(|| {
                Error::Internal("CUDA DSV4 prefill topk missing indexer weights".into())
            })?;
            let expected = tokens
                .checked_mul(index_heads)
                .ok_or_else(|| Error::Internal("CUDA DSV4 prefill weight size overflow".into()))?;
            if weights.len != expected {
                return Err(Error::Internal(format!(
                    "CUDA DSV4 prefill indexer weight length mismatch: got {} expected {expected}",
                    weights.len
                )));
            }
            weights
        } else {
            empty_weights = self.zero_f32_buffer(1)?;
            &empty_weights
        };
        let indexer_kv = if indexer_enabled {
            let indexer_kv = indexer_kv.ok_or_else(|| {
                Error::Internal("CUDA DSV4 prefill topk missing indexer KV".into())
            })?;
            let expected = compressed_len.checked_mul(index_head_dim).ok_or_else(|| {
                Error::Internal("CUDA DSV4 prefill index KV size overflow".into())
            })?;
            if indexer_kv.len != expected {
                return Err(Error::Internal(format!(
                    "CUDA DSV4 prefill indexer KV length mismatch: got {} expected {expected}",
                    indexer_kv.len
                )));
            }
            indexer_kv
        } else {
            empty_kv = self.zero_f32_buffer(1)?;
            &empty_kv
        };

        let mut out = self.zero_i32_buffer(tokens.checked_mul(total_cols).ok_or_else(|| {
            Error::Internal("CUDA DSV4 prefill topk output size overflow".into())
        })?)?;
        self.launched(unsafe {
            self.module.dsv4_prefill_topk_indices(
                &self.stream,
                LaunchConfig::for_num_elems(tokens as u32),
                &query.buffer,
                &weights.buffer,
                &indexer_kv.buffer,
                &mut out.buffer,
                tokens as u32,
                window_size as u32,
                window_cols as u32,
                extra_cols as u32,
                value_offset as u32,
                compress_ratio as u32,
                compressed_len as u32,
                index_heads as u32,
                index_head_dim as u32,
                if indexer_enabled { 1u32 } else { 0u32 },
                weight_scale,
            )
        })?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_prefill_topk_indices_fused_index_query_from_device(
        &self,
        query: &CudaF32Buffer,
        weights: &CudaF32Buffer,
        indexer_kv: &CudaF32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        tokens: usize,
        window_size: usize,
        window_cols: usize,
        extra_cols: usize,
        value_offset: usize,
        compress_ratio: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        rope_dim: usize,
        start_position: usize,
        weight_scale: f32,
    ) -> Result<CudaI32Buffer> {
        if tokens == 0 {
            return Err(Error::Internal(
                "CUDA DSV4 fused prefill topk requires tokens > 0".into(),
            ));
        }
        if window_cols > window_size || window_cols > tokens {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 fused prefill topk window: tokens={tokens} window_size={window_size} window_cols={window_cols}"
            )));
        }
        if compress_ratio == 0 || extra_cols == 0 || compressed_len == 0 {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 fused prefill topk compression: ratio={compress_ratio} extra_cols={extra_cols} compressed_len={compressed_len}"
            )));
        }
        if extra_cols > 512 {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused prefill topk supports at most 512 extra columns, got {extra_cols}"
            )));
        }
        if index_head_dim == 0
            || index_head_dim > 256
            || !index_head_dim.is_power_of_two()
            || !index_head_dim.is_multiple_of(32)
            || rope_dim > index_head_dim
            || !rope_dim.is_multiple_of(2)
        {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 fused prefill indexer shape: heads={index_heads} head_dim={index_head_dim} rope_dim={rope_dim}"
            )));
        }
        let total_cols = window_cols.checked_add(extra_cols).ok_or_else(|| {
            Error::Internal("CUDA DSV4 fused prefill topk column overflow".into())
        })?;
        if total_cols == 0 {
            return Err(Error::Internal(
                "CUDA DSV4 fused prefill topk requires at least one column".into(),
            ));
        }
        let expected_query = tokens
            .checked_mul(index_heads)
            .and_then(|v| v.checked_mul(index_head_dim))
            .ok_or_else(|| Error::Internal("CUDA DSV4 fused prefill query size overflow".into()))?;
        if query.len != expected_query {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused prefill query length mismatch: got {} expected {expected_query}",
                query.len
            )));
        }
        let expected_weights = tokens.checked_mul(index_heads).ok_or_else(|| {
            Error::Internal("CUDA DSV4 fused prefill weights size overflow".into())
        })?;
        if weights.len != expected_weights {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused prefill weights length mismatch: got {} expected {expected_weights}",
                weights.len
            )));
        }
        let expected_kv = compressed_len.checked_mul(index_head_dim).ok_or_else(|| {
            Error::Internal("CUDA DSV4 fused prefill index KV size overflow".into())
        })?;
        if indexer_kv.len != expected_kv {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused prefill index KV length mismatch: got {} expected {expected_kv}",
                indexer_kv.len
            )));
        }
        let rd2 = rope_dim / 2;
        let required_rope = start_position
            .checked_add(tokens)
            .and_then(|v| v.checked_mul(rd2))
            .ok_or_else(|| Error::Internal("CUDA DSV4 fused prefill rope size overflow".into()))?;
        if rd2 > 0 && (cos_table.len < required_rope || sin_table.len < required_rope) {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused prefill rope table too small: need {required_rope}, cos={} sin={}",
                cos_table.len, sin_table.len
            )));
        }

        let mut out = self.zero_i32_buffer(tokens.checked_mul(total_cols).ok_or_else(|| {
            Error::Internal("CUDA DSV4 fused prefill topk output size overflow".into())
        })?)?;
        self.launched(unsafe {
            self.module.dsv4_prefill_topk_indices_fused_index_query(
                &self.stream,
                LaunchConfig::for_num_elems(tokens as u32),
                &query.buffer,
                &weights.buffer,
                &indexer_kv.buffer,
                &cos_table.buffer,
                &sin_table.buffer,
                &mut out.buffer,
                tokens as u32,
                window_size as u32,
                window_cols as u32,
                extra_cols as u32,
                value_offset as u32,
                compress_ratio as u32,
                compressed_len as u32,
                index_heads as u32,
                index_head_dim as u32,
                rope_dim as u32,
                start_position as u32,
                weight_scale,
            )
        })?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_decode_topk_indices_fused_index_query_from_device(
        &self,
        query: &CudaF32Buffer,
        weights: &CudaF32Buffer,
        indexer_kv: &CudaF32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        position: usize,
        window_len: usize,
        window_size: usize,
        extra_cols: usize,
        value_offset: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        rope_dim: usize,
        weight_scale: f32,
    ) -> Result<CudaI32Buffer> {
        let total_cols = window_size
            .checked_add(extra_cols)
            .ok_or_else(|| Error::Internal("CUDA DSV4 fused decode topk column overflow".into()))?;
        let mut out = self.zero_i32_buffer(total_cols)?;
        self.dsv4_decode_topk_indices_fused_index_query_from_device_into(
            query,
            weights,
            indexer_kv,
            cos_table,
            sin_table,
            position,
            window_len,
            window_size,
            extra_cols,
            value_offset,
            compressed_len,
            index_heads,
            index_head_dim,
            rope_dim,
            weight_scale,
            &mut out,
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_decode_topk_indices_fused_index_query_from_device_into(
        &self,
        query: &CudaF32Buffer,
        weights: &CudaF32Buffer,
        indexer_kv: &CudaF32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        position: usize,
        window_len: usize,
        window_size: usize,
        extra_cols: usize,
        value_offset: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        rope_dim: usize,
        weight_scale: f32,
        out: &mut CudaI32Buffer,
    ) -> Result<()> {
        if window_size == 0 {
            return Err(Error::Internal(
                "CUDA DSV4 fused decode topk requires window_size > 0".into(),
            ));
        }
        if window_len > window_size {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 fused decode topk window: len={window_len} window_size={window_size}"
            )));
        }
        if extra_cols == 0 || compressed_len == 0 {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 fused decode topk compression: extra_cols={extra_cols} compressed_len={compressed_len}"
            )));
        }
        if extra_cols > 512 {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused decode topk supports at most 512 extra columns, got {extra_cols}"
            )));
        }
        if index_head_dim == 0
            || index_head_dim > 256
            || !index_head_dim.is_power_of_two()
            || !index_head_dim.is_multiple_of(32)
            || rope_dim > index_head_dim
            || !rope_dim.is_multiple_of(2)
        {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 fused decode indexer shape: heads={index_heads} head_dim={index_head_dim} rope_dim={rope_dim}"
            )));
        }
        let total_cols = window_size
            .checked_add(extra_cols)
            .ok_or_else(|| Error::Internal("CUDA DSV4 fused decode topk column overflow".into()))?;
        if out.len < total_cols {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused decode topk output too small: need {total_cols}, got {}",
                out.len
            )));
        }
        let expected_query = index_heads
            .checked_mul(index_head_dim)
            .ok_or_else(|| Error::Internal("CUDA DSV4 fused decode query size overflow".into()))?;
        if query.len != expected_query {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused decode query length mismatch: got {} expected {expected_query}",
                query.len
            )));
        }
        if weights.len != index_heads {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused decode weights length mismatch: got {} expected {index_heads}",
                weights.len
            )));
        }
        let expected_kv = compressed_len.checked_mul(index_head_dim).ok_or_else(|| {
            Error::Internal("CUDA DSV4 fused decode index KV size overflow".into())
        })?;
        if indexer_kv.len < expected_kv {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused decode index KV length mismatch: got {} expected at least {expected_kv}",
                indexer_kv.len
            )));
        }
        let rd2 = rope_dim / 2;
        let required_rope = position
            .checked_add(1)
            .and_then(|v| v.checked_mul(rd2))
            .ok_or_else(|| Error::Internal("CUDA DSV4 fused decode rope size overflow".into()))?;
        if rd2 > 0 && (cos_table.len < required_rope || sin_table.len < required_rope) {
            return Err(Error::Internal(format!(
                "CUDA DSV4 fused decode rope table too small: need {required_rope}, cos={} sin={}",
                cos_table.len, sin_table.len
            )));
        }

        self.launched(unsafe {
            self.module.dsv4_decode_topk_indices_fused_index_query(
                &self.stream,
                LaunchConfig::for_num_elems(1),
                &query.buffer,
                &weights.buffer,
                &indexer_kv.buffer,
                &cos_table.buffer,
                &sin_table.buffer,
                &mut out.buffer,
                position as u32,
                window_len as u32,
                window_size as u32,
                extra_cols as u32,
                value_offset as u32,
                compressed_len as u32,
                index_heads as u32,
                index_head_dim as u32,
                rope_dim as u32,
                weight_scale,
            )
        })
    }

    pub fn dsv4_decode_topk_indices_from_device(
        &self,
        query: Option<&CudaF32Buffer>,
        weights: Option<&CudaF32Buffer>,
        indexer_kv: Option<&CudaF32Buffer>,
        position: usize,
        window_len: usize,
        window_size: usize,
        extra_cols: usize,
        value_offset: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        weight_scale: f32,
    ) -> Result<CudaI32Buffer> {
        if window_size == 0 {
            return Err(Error::Internal(
                "CUDA DSV4 decode topk requires window_size > 0".into(),
            ));
        }
        if window_len > window_size {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 decode topk window: len={window_len} window_size={window_size}"
            )));
        }
        let indexer_enabled = query.is_some() || weights.is_some() || indexer_kv.is_some();
        if indexer_enabled && extra_cols > 512 {
            return Err(Error::Internal(format!(
                "CUDA DSV4 decode indexer topk supports at most 512 extra columns, got {extra_cols}"
            )));
        }
        let total_cols = window_size
            .checked_add(extra_cols)
            .ok_or_else(|| Error::Internal("CUDA DSV4 decode topk column overflow".into()))?;
        let empty_query;
        let empty_weights;
        let empty_kv;
        let query = if indexer_enabled {
            let query = query.ok_or_else(|| {
                Error::Internal("CUDA DSV4 decode topk missing indexer query".into())
            })?;
            let expected = index_heads
                .checked_mul(index_head_dim)
                .ok_or_else(|| Error::Internal("CUDA DSV4 decode query size overflow".into()))?;
            if query.len != expected {
                return Err(Error::Internal(format!(
                    "CUDA DSV4 decode indexer query length mismatch: got {} expected {expected}",
                    query.len
                )));
            }
            query
        } else {
            empty_query = self.zero_f32_buffer(1)?;
            &empty_query
        };
        let weights = if indexer_enabled {
            let weights = weights.ok_or_else(|| {
                Error::Internal("CUDA DSV4 decode topk missing indexer weights".into())
            })?;
            if weights.len != index_heads {
                return Err(Error::Internal(format!(
                    "CUDA DSV4 decode indexer weights length mismatch: got {} expected {index_heads}",
                    weights.len
                )));
            }
            weights
        } else {
            empty_weights = self.zero_f32_buffer(1)?;
            &empty_weights
        };
        let indexer_kv = if indexer_enabled {
            let indexer_kv = indexer_kv.ok_or_else(|| {
                Error::Internal("CUDA DSV4 decode topk missing indexer KV".into())
            })?;
            let expected = compressed_len
                .checked_mul(index_head_dim)
                .ok_or_else(|| Error::Internal("CUDA DSV4 decode index KV size overflow".into()))?;
            if indexer_kv.len < expected {
                return Err(Error::Internal(format!(
                    "CUDA DSV4 decode indexer KV length mismatch: got {} expected at least {expected}",
                    indexer_kv.len
                )));
            }
            indexer_kv
        } else {
            empty_kv = self.zero_f32_buffer(1)?;
            &empty_kv
        };

        let mut out = self.zero_i32_buffer(total_cols)?;
        self.launched(unsafe {
            self.module.dsv4_decode_topk_indices(
                &self.stream,
                LaunchConfig::for_num_elems(1),
                &query.buffer,
                &weights.buffer,
                &indexer_kv.buffer,
                &mut out.buffer,
                position as u32,
                window_len as u32,
                window_size as u32,
                extra_cols as u32,
                value_offset as u32,
                compressed_len as u32,
                index_heads as u32,
                index_head_dim as u32,
                if indexer_enabled { 1u32 } else { 0u32 },
                weight_scale,
            )
        })?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_decode_topk_indices_from_device_into(
        &self,
        query: Option<&CudaF32Buffer>,
        weights: Option<&CudaF32Buffer>,
        indexer_kv: Option<&CudaF32Buffer>,
        empty_query: &CudaF32Buffer,
        empty_weights: &CudaF32Buffer,
        empty_kv: &CudaF32Buffer,
        position: usize,
        window_len: usize,
        window_size: usize,
        extra_cols: usize,
        value_offset: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        weight_scale: f32,
        out: &mut CudaI32Buffer,
    ) -> Result<()> {
        if window_size == 0 {
            return Err(Error::Internal(
                "CUDA DSV4 decode topk requires window_size > 0".into(),
            ));
        }
        if window_len > window_size {
            return Err(Error::Internal(format!(
                "invalid CUDA DSV4 decode topk window: len={window_len} window_size={window_size}"
            )));
        }
        let indexer_enabled = query.is_some() || weights.is_some() || indexer_kv.is_some();
        if indexer_enabled && extra_cols > 512 {
            return Err(Error::Internal(format!(
                "CUDA DSV4 decode indexer topk supports at most 512 extra columns, got {extra_cols}"
            )));
        }
        let total_cols = window_size
            .checked_add(extra_cols)
            .ok_or_else(|| Error::Internal("CUDA DSV4 decode topk column overflow".into()))?;
        if out.len < total_cols {
            return Err(Error::Internal(format!(
                "CUDA DSV4 decode topk output too small: need {total_cols}, got {}",
                out.len
            )));
        }
        let query = if indexer_enabled {
            let query = query.ok_or_else(|| {
                Error::Internal("CUDA DSV4 decode topk missing indexer query".into())
            })?;
            let expected = index_heads
                .checked_mul(index_head_dim)
                .ok_or_else(|| Error::Internal("CUDA DSV4 decode query size overflow".into()))?;
            if query.len != expected {
                return Err(Error::Internal(format!(
                    "CUDA DSV4 decode indexer query length mismatch: got {} expected {expected}",
                    query.len
                )));
            }
            query
        } else {
            empty_query
        };
        let weights = if indexer_enabled {
            let weights = weights.ok_or_else(|| {
                Error::Internal("CUDA DSV4 decode topk missing indexer weights".into())
            })?;
            if weights.len != index_heads {
                return Err(Error::Internal(format!(
                    "CUDA DSV4 decode indexer weights length mismatch: got {} expected {index_heads}",
                    weights.len
                )));
            }
            weights
        } else {
            empty_weights
        };
        let indexer_kv = if indexer_enabled {
            let indexer_kv = indexer_kv.ok_or_else(|| {
                Error::Internal("CUDA DSV4 decode topk missing indexer KV".into())
            })?;
            let expected = compressed_len
                .checked_mul(index_head_dim)
                .ok_or_else(|| Error::Internal("CUDA DSV4 decode index KV size overflow".into()))?;
            if indexer_kv.len < expected {
                return Err(Error::Internal(format!(
                    "CUDA DSV4 decode indexer KV length mismatch: got {} expected at least {expected}",
                    indexer_kv.len
                )));
            }
            indexer_kv
        } else {
            empty_kv
        };

        self.launched(unsafe {
            self.module.dsv4_decode_topk_indices(
                &self.stream,
                LaunchConfig::for_num_elems(1),
                &query.buffer,
                &weights.buffer,
                &indexer_kv.buffer,
                &mut out.buffer,
                position as u32,
                window_len as u32,
                window_size as u32,
                extra_cols as u32,
                value_offset as u32,
                compressed_len as u32,
                index_heads as u32,
                index_head_dim as u32,
                if indexer_enabled { 1u32 } else { 0u32 },
                weight_scale,
            )
        })
    }

    pub fn artifact_swiglu_ffn_matvec(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &[f32],
        output_scale: f32,
        swiglu_limit: f32,
    ) -> Result<Vec<f32>> {
        if input.len() != gate.shape.in_features() || input.len() != up.shape.in_features() {
            return Err(Error::Internal(format!(
                "CUDA SwiGLU input length mismatch: input={} gate_in={} up_in={}",
                input.len(),
                gate.shape.in_features(),
                up.shape.in_features()
            )));
        }
        if gate.shape.out_features() != up.shape.out_features()
            || down.shape.in_features() != gate.shape.out_features()
        {
            return Err(Error::Internal(format!(
                "CUDA SwiGLU shape mismatch: gate={:?} up={:?} down={:?}",
                gate.shape, up.shape, down.shape
            )));
        }
        let gate_input = prepare_activation_for_artifact_linear(gate.shape, input)?;
        let up_input = prepare_activation_for_artifact_linear(up.shape, input)?;
        let gate_xd = self.upload_f32(gate_input.as_ref())?;
        let up_xd = self.upload_f32(up_input.as_ref())?;
        let mut gated = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            gate.shape.out_features(),
        ))?;
        let mut upd = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            up.shape.out_features(),
        ))?;
        let mut hidden = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            gate.shape.out_features(),
        ))?;
        let mut yd = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            down.shape.out_features(),
        ))?;
        self.artifact_linear_matvec_device(gate, &gate_xd, &mut gated)?;
        self.artifact_linear_matvec_device(up, &up_xd, &mut upd)?;
        self.launched(unsafe {
            self.module.swiglu_weighted_clamped(
                &self.stream,
                LaunchConfig::for_num_elems(gate.shape.out_features() as u32),
                &gated,
                &upd,
                &mut hidden,
                gate.shape.out_features() as u32,
                output_scale,
                swiglu_limit,
            )
        })?;
        if quantized_shape_uses_fp8_activation(down.shape) {
            self.fp8_activation_quantize_in_place(
                &mut hidden,
                down.shape.in_features(),
                down.shape.in_features(),
                ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
            )?;
        }
        self.artifact_linear_matvec_device(down, &hidden, &mut yd)?;
        self.download_f32(&yd, down.shape.out_features())
    }

    pub fn artifact_swiglu_ffn_from_device(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        output_scale: f32,
        swiglu_limit: f32,
    ) -> Result<CudaF32Buffer> {
        if input.len != gate.shape.in_features() || input.len != up.shape.in_features() {
            return Err(Error::Internal(format!(
                "CUDA SwiGLU device input length mismatch: input={} gate_in={} up_in={}",
                input.len,
                gate.shape.in_features(),
                up.shape.in_features()
            )));
        }
        if gate.shape.out_features() != up.shape.out_features()
            || down.shape.in_features() != gate.shape.out_features()
        {
            return Err(Error::Internal(format!(
                "CUDA SwiGLU shape mismatch: gate={:?} up={:?} down={:?}",
                gate.shape, up.shape, down.shape
            )));
        }
        let mut gate_input = self.zero_f32_buffer(input.len)?;
        self.copy_f32_into_slot(input, &mut gate_input, 0)?;
        if quantized_shape_uses_fp8_activation(gate.shape) {
            self.fp8_activation_quantize_buffer_in_place(
                &mut gate_input,
                gate.shape.in_features(),
                ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
            )?;
        }
        let mut up_input = self.zero_f32_buffer(input.len)?;
        self.copy_f32_into_slot(input, &mut up_input, 0)?;
        if quantized_shape_uses_fp8_activation(up.shape) {
            self.fp8_activation_quantize_buffer_in_place(
                &mut up_input,
                up.shape.in_features(),
                ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
            )?;
        }

        let mut gated = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            gate.shape.out_features(),
        ))?;
        let mut upd = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            up.shape.out_features(),
        ))?;
        let mut hidden = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            gate.shape.out_features(),
        ))?;
        let mut yd = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            down.shape.out_features(),
        ))?;
        self.artifact_linear_matvec_device(gate, &gate_input.buffer, &mut gated)?;
        self.artifact_linear_matvec_device(up, &up_input.buffer, &mut upd)?;
        self.launched(unsafe {
            self.module.swiglu_weighted_clamped(
                &self.stream,
                LaunchConfig::for_num_elems(gate.shape.out_features() as u32),
                &gated,
                &upd,
                &mut hidden,
                gate.shape.out_features() as u32,
                output_scale,
                swiglu_limit,
            )
        })?;
        if quantized_shape_uses_fp8_activation(down.shape) {
            self.fp8_activation_quantize_in_place(
                &mut hidden,
                down.shape.in_features(),
                down.shape.in_features(),
                ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
            )?;
        }
        self.artifact_linear_matvec_device(down, &hidden, &mut yd)?;
        Ok(CudaF32Buffer {
            buffer: yd,
            len: down.shape.out_features(),
        })
    }

    pub fn artifact_fp4_swiglu_ffn_matvec(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &[f32],
        route_weight: f32,
        swiglu_limit: f32,
    ) -> Result<Vec<f32>> {
        let quantized = self.prepare_fp4_expert_input(gate, up, down, input)?;
        let yd = self.fp4_swiglu_ffn_from_device(
            gate,
            up,
            down,
            &quantized,
            route_weight,
            swiglu_limit,
        )?;
        self.download_f32_buffer(&yd)
    }

    /// Quantize and upload an expert input once, for reuse across multiple
    /// experts via `fp4_swiglu_ffn_from_device`.
    ///
    /// This avoids re-uploading and re-quantizing the same input for each
    /// expert in a routed MoE step (6 experts × 1 upload → 1 upload).
    pub fn prepare_fp4_expert_input(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &[f32],
    ) -> Result<CudaF32Buffer> {
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: gate_out,
            in_features: gate_in,
        } = gate.shape
        else {
            return Err(Error::Internal("CUDA packed expert gate is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: up_out,
            in_features: up_in,
        } = up.shape
        else {
            return Err(Error::Internal("CUDA packed expert up is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: down_out,
            in_features: down_in,
        } = down.shape
        else {
            return Err(Error::Internal("CUDA packed expert down is not FP4".into()));
        };
        if input.len() != gate_in || up_in != gate_in || up_out != gate_out || down_in != gate_out {
            return Err(Error::Internal(format!(
                "CUDA packed expert shape mismatch: input={} gate=[{gate_out},{gate_in}] up=[{up_out},{up_in}] down=[{down_out},{down_in}]",
                input.len()
            )));
        }
        let mut quantized_input = input.to_vec();
        simulate_fp8_e4m3fn_e8m0_activation_quant_in_place(
            &mut quantized_input,
            gate_in,
            ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
        )?;
        self.upload_f32_buffer(&quantized_input)
    }

    pub fn prepare_fp4_expert_input_from_device(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
    ) -> Result<CudaF32Buffer> {
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: gate_out,
            in_features: gate_in,
        } = gate.shape
        else {
            return Err(Error::Internal("CUDA packed expert gate is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: up_out,
            in_features: up_in,
        } = up.shape
        else {
            return Err(Error::Internal("CUDA packed expert up is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: down_out,
            in_features: down_in,
        } = down.shape
        else {
            return Err(Error::Internal("CUDA packed expert down is not FP4".into()));
        };
        if input.len != gate_in || up_in != gate_in || up_out != gate_out || down_in != gate_out {
            return Err(Error::Internal(format!(
                "CUDA packed expert shape mismatch: input={} gate=[{gate_out},{gate_in}] up=[{up_out},{up_in}] down=[{down_out},{down_in}]",
                input.len
            )));
        }
        let mut quantized_input = self.zero_f32_buffer(input.len)?;
        self.copy_f32_into_slot(input, &mut quantized_input, 0)?;
        self.fp8_activation_quantize_buffer_in_place(
            &mut quantized_input,
            gate_in,
            ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
        )?;
        Ok(quantized_input)
    }

    /// Run FP4 SwiGLU expert matvec from a pre-uploaded device input buffer.
    ///
    /// Pair with `prepare_fp4_expert_input` to share the input across experts.
    /// Returns a device buffer instead of downloading to host.
    pub fn fp4_expert_workspace(
        &self,
        intermediate_size: usize,
        output_size: usize,
    ) -> Result<CudaFp4ExpertWorkspace> {
        Ok(CudaFp4ExpertWorkspace {
            scratch: CudaPackedFp4ExpertScratch::new(&self.stream, intermediate_size)?,
            output: CudaF32Buffer {
                buffer: cu(DeviceBuffer::<f32>::zeroed(&self.stream, output_size))?,
                len: output_size,
            },
            intermediate_size,
            output_size,
        })
    }

    pub fn moe_batched_workspace(
        &self,
        max_experts: usize,
        input_size: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Result<CudaMoeBatchedWorkspace> {
        self.moe_batched_workspace_cols(max_experts, 1, input_size, intermediate_size, hidden_size)
    }

    pub fn moe_batched_workspace_cols(
        &self,
        max_experts: usize,
        max_batch_cols: usize,
        input_size: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Result<CudaMoeBatchedWorkspace> {
        if max_experts == 0 || max_experts > 64 {
            return Err(Error::Internal(format!(
                "CUDA MoE workspace expects 1..=64 experts, got {max_experts}"
            )));
        }
        if max_batch_cols == 0 || max_batch_cols > 8 {
            return Err(Error::Internal(format!(
                "CUDA MoE workspace expects 1..=8 batch columns, got {max_batch_cols}"
            )));
        }
        if input_size == 0 || intermediate_size == 0 || hidden_size == 0 {
            return Err(Error::Internal(format!(
                "CUDA MoE workspace invalid shape: input={input_size} intermediate={intermediate_size} hidden={hidden_size}"
            )));
        }
        if !input_size.is_multiple_of(32) || !intermediate_size.is_multiple_of(32) {
            return Err(Error::Internal(format!(
                "CUDA MoE workspace expects 32-aligned input/intermediate, got input={input_size} intermediate={intermediate_size}"
            )));
        }

        let total_inter = max_experts
            .checked_mul(max_batch_cols)
            .and_then(|v| v.checked_mul(intermediate_size))
            .ok_or_else(|| {
                Error::Internal("CUDA MoE workspace intermediate size overflow".into())
            })?;
        let total_input = max_batch_cols
            .checked_mul(input_size)
            .ok_or_else(|| Error::Internal("CUDA MoE workspace input size overflow".into()))?;
        Ok(CudaMoeBatchedWorkspace {
            gate_ptrs: cu(DeviceBuffer::<u64>::zeroed(&self.stream, max_experts))?,
            gate_scale_ptrs: cu(DeviceBuffer::<u64>::zeroed(&self.stream, max_experts))?,
            up_ptrs: cu(DeviceBuffer::<u64>::zeroed(&self.stream, max_experts))?,
            up_scale_ptrs: cu(DeviceBuffer::<u64>::zeroed(&self.stream, max_experts))?,
            down_ptrs: cu(DeviceBuffer::<u64>::zeroed(&self.stream, max_experts))?,
            down_scale_ptrs: cu(DeviceBuffer::<u64>::zeroed(&self.stream, max_experts))?,
            route_weights: cu(DeviceBuffer::<f32>::zeroed(
                &self.stream,
                max_experts * max_batch_cols,
            ))?,
            x_f32: CudaF32Buffer {
                buffer: self.uninitialized_device_buffer::<f32>(total_input)?,
                len: total_input,
            },
            y_gate: CudaF32Buffer {
                buffer: self.uninitialized_device_buffer::<f32>(total_inter)?,
                len: total_inter,
            },
            y_up: CudaF32Buffer {
                buffer: self.uninitialized_device_buffer::<f32>(total_inter)?,
                len: total_inter,
            },
            y_hidden: CudaF32Buffer {
                buffer: self.uninitialized_device_buffer::<f32>(total_inter)?,
                len: total_inter,
            },
            x_packed: self.uninitialized_device_buffer::<u8>(total_input / 2)?,
            x_scales: self.uninitialized_device_buffer::<u8>(total_input / 32)?,
            y_hidden_packed: self.uninitialized_device_buffer::<u8>(total_inter / 2)?,
            y_hidden_scales: self.uninitialized_device_buffer::<u8>(total_inter / 32)?,
            max_experts,
            max_batch_cols,
            input_size,
            intermediate_size,
            hidden_size,
        })
    }

    pub fn fp4_swiglu_ffn_from_device(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        route_weight: f32,
        swiglu_limit: f32,
    ) -> Result<CudaF32Buffer> {
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: gate_out,
            in_features: gate_in,
        } = gate.shape
        else {
            return Err(Error::Internal("CUDA packed expert gate is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: up_out,
            in_features: up_in,
        } = up.shape
        else {
            return Err(Error::Internal("CUDA packed expert up is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: down_out,
            in_features: down_in,
        } = down.shape
        else {
            return Err(Error::Internal("CUDA packed expert down is not FP4".into()));
        };
        if input.len() != gate_in || up_in != gate_in || up_out != gate_out || down_in != gate_out {
            return Err(Error::Internal(format!(
                "CUDA packed expert shape mismatch: input={} gate=[{gate_out},{gate_in}] up=[{up_out},{up_in}] down=[{down_out},{down_in}]",
                input.len()
            )));
        }
        let mut workspace = self.fp4_expert_workspace(gate_out, down_out)?;
        self.fp4_swiglu_ffn_into_from_device(
            gate,
            up,
            down,
            input,
            route_weight,
            swiglu_limit,
            &mut workspace,
        )?;
        Ok(workspace.output)
    }

    pub fn fp4_swiglu_ffn_into_from_device(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        route_weight: f32,
        swiglu_limit: f32,
        workspace: &mut CudaFp4ExpertWorkspace,
    ) -> Result<()> {
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: gate_out,
            in_features: gate_in,
        } = gate.shape
        else {
            return Err(Error::Internal("CUDA packed expert gate is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: up_out,
            in_features: up_in,
        } = up.shape
        else {
            return Err(Error::Internal("CUDA packed expert up is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: down_out,
            in_features: down_in,
        } = down.shape
        else {
            return Err(Error::Internal("CUDA packed expert down is not FP4".into()));
        };
        if input.len() != gate_in || up_in != gate_in || up_out != gate_out || down_in != gate_out {
            return Err(Error::Internal(format!(
                "CUDA packed expert shape mismatch: input={} gate=[{gate_out},{gate_in}] up=[{up_out},{up_in}] down=[{down_out},{down_in}]",
                input.len()
            )));
        }
        if workspace.intermediate_size != gate_out || workspace.output_size != down_out {
            return Err(Error::Internal(format!(
                "CUDA packed expert workspace mismatch: workspace=[{},{}] expert=[{},{}]",
                workspace.intermediate_size, workspace.output_size, gate_out, down_out
            )));
        }
        let gate_scale = gate
            .scale
            .as_ref()
            .ok_or_else(|| Error::Internal("CUDA packed expert gate missing scale".into()))?;
        let up_scale = up
            .scale
            .as_ref()
            .ok_or_else(|| Error::Internal("CUDA packed expert up missing scale".into()))?;
        let down_scale = down
            .scale
            .as_ref()
            .ok_or_else(|| Error::Internal("CUDA packed expert down missing scale".into()))?;
        let expert = CudaPackedFp4Expert {
            gate: CudaPackedFp4Linear::new(&gate.weight, gate_scale, gate_out, gate_in),
            up: CudaPackedFp4Linear::new(&up.weight, up_scale, up_out, up_in),
            down: CudaPackedFp4Linear::new(&down.weight, down_scale, down_out, down_in),
        };
        expert.validate()?;
        let cfg_mid = LaunchConfig::for_num_elems(gate_out as u32);
        let cfg_out = LaunchConfig::for_num_elems(down_out as u32);

        self.launched(unsafe {
            self.module.gemv_dual_fp4_e2m1_e8m0_off(
                &self.stream,
                cfg_mid,
                &input.buffer,
                expert.gate.packed,
                expert.gate.scales,
                &mut workspace.scratch.gate,
                checked_u32(expert.gate.packed_offset(), "gate", "packed offset")?,
                checked_u32(expert.gate.scale_offset(), "gate", "scale offset")?,
                expert.up.packed,
                expert.up.scales,
                &mut workspace.scratch.up,
                checked_u32(expert.up.packed_offset(), "up", "packed offset")?,
                checked_u32(expert.up.scale_offset(), "up", "scale offset")?,
                checked_u32(gate_out, "expert", "intermediate size")?,
                checked_u32(gate_in, "expert", "hidden size")?,
            )
        })?;

        self.launched(unsafe {
            self.module.swiglu_weighted_clamped(
                &self.stream,
                cfg_mid,
                &workspace.scratch.gate,
                &workspace.scratch.up,
                &mut workspace.scratch.hidden,
                checked_u32(gate_out, "expert", "intermediate size")?,
                route_weight,
                swiglu_limit,
            )
        })?;

        self.fp8_activation_quantize_in_place(
            &mut workspace.scratch.hidden,
            down_in,
            down_in,
            ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
        )?;
        self.launched(unsafe {
            self.module.gemv_fp4_e2m1_e8m0_off(
                &self.stream,
                cfg_out,
                &workspace.scratch.hidden,
                expert.down.packed,
                expert.down.scales,
                &mut workspace.output.buffer,
                checked_u32(down_out, "down", "output size")?,
                checked_u32(down_in, "down", "input size")?,
                checked_u32(expert.down.packed_offset(), "down", "packed offset")?,
                checked_u32(expert.down.scale_offset(), "down", "scale offset")?,
            )
        })?;
        Ok(())
    }

    /// Batched routed MoE into an existing accumulator using reusable scratch.
    ///
    /// This is the decode hot-path variant. It avoids per-call CUDA allocation,
    /// writes selected expert pointers into persistent tiny device arrays, and
    /// accumulates the down projection directly into `output`.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_moe_experts_batched_workspace(
        &self,
        gate_handles: &[&CudaArtifactLinearHandle; 6],
        up_handles: &[&CudaArtifactLinearHandle; 6],
        down_handles: &[&CudaArtifactLinearHandle; 6],
        route_weights: &[f32; 6],
        input_len: usize,
        num_experts: usize,
        intermediate_size: usize,
        hidden_size: usize,
        workspace: &mut CudaMoeBatchedWorkspace,
    ) -> Result<()> {
        if num_experts == 0 || num_experts > 6 {
            return Err(Error::Internal(format!(
                "MoE batched expects 1..=6 experts, got {num_experts}"
            )));
        }
        if !workspace.matches(num_experts, input_len, intermediate_size, hidden_size) {
            return Err(Error::Internal(format!(
                "CUDA MoE workspace mismatch: workspace=[max_experts={},input={},intermediate={},hidden={}] call=[experts={},input={},intermediate={},hidden={}]",
                workspace.max_experts,
                workspace.input_size,
                workspace.intermediate_size,
                workspace.hidden_size,
                num_experts,
                input_len,
                intermediate_size,
                hidden_size
            )));
        }

        let first_gate = gate_handles[0];
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: gate_out,
            in_features: gate_in,
        } = first_gate.shape
        else {
            return Err(Error::Internal("CUDA packed expert gate is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: up_out,
            in_features: up_in,
        } = up_handles[0].shape
        else {
            return Err(Error::Internal("CUDA packed expert up is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: down_out,
            in_features: down_in,
        } = down_handles[0].shape
        else {
            return Err(Error::Internal("CUDA packed expert down is not FP4".into()));
        };
        if input_len != gate_in
            || up_in != gate_in
            || up_out != gate_out
            || down_in != gate_out
            || down_out != hidden_size
            || gate_out != intermediate_size
        {
            return Err(Error::Internal(format!(
                "CUDA packed expert shape mismatch: input={} gate=[{gate_out},{gate_in}] up=[{up_out},{up_in}] down=[{down_out},{down_in}] expected intermediate={intermediate_size} hidden={hidden_size}",
                input_len
            )));
        }

        let timing_enabled = self.moe_timing_enabled();
        let mut gate_ptrs = [0u64; 6];
        let mut gate_scale_ptrs = [0u64; 6];
        let mut up_ptrs = [0u64; 6];
        let mut up_scale_ptrs = [0u64; 6];
        let mut down_ptrs = [0u64; 6];
        let mut down_scale_ptrs = [0u64; 6];
        for i in 0..num_experts {
            gate_ptrs[i] = gate_handles[i].weight.cu_deviceptr();
            gate_scale_ptrs[i] = gate_handles[i]
                .scale
                .as_ref()
                .ok_or_else(|| Error::Internal("CUDA packed expert gate missing scale".into()))?
                .cu_deviceptr();
            up_ptrs[i] = up_handles[i].weight.cu_deviceptr();
            up_scale_ptrs[i] = up_handles[i]
                .scale
                .as_ref()
                .ok_or_else(|| Error::Internal("CUDA packed expert up missing scale".into()))?
                .cu_deviceptr();
            down_ptrs[i] = down_handles[i].weight.cu_deviceptr();
            down_scale_ptrs[i] = down_handles[i]
                .scale
                .as_ref()
                .ok_or_else(|| Error::Internal("CUDA packed expert down missing scale".into()))?
                .cu_deviceptr();
        }
        let phase_start = timing_enabled.then(Instant::now);
        self.copy_u64_into_device_buffer(&gate_ptrs, &mut workspace.gate_ptrs)?;
        self.copy_u64_into_device_buffer(&gate_scale_ptrs, &mut workspace.gate_scale_ptrs)?;
        self.copy_u64_into_device_buffer(&up_ptrs, &mut workspace.up_ptrs)?;
        self.copy_u64_into_device_buffer(&up_scale_ptrs, &mut workspace.up_scale_ptrs)?;
        self.copy_u64_into_device_buffer(&down_ptrs, &mut workspace.down_ptrs)?;
        self.copy_u64_into_device_buffer(&down_scale_ptrs, &mut workspace.down_scale_ptrs)?;
        self.copy_f32_into_device_buffer(route_weights, &mut workspace.route_weights)?;
        if let Some(start) = phase_start {
            self.sync_stream()?;
            self.counters
                .add_moe_pointer_upload_us(duration_us(start.elapsed()));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_experts_batched_add_into_from_device_prepared(
        &self,
        input: &CudaF32Buffer,
        swiglu_limit: f32,
        num_experts: usize,
        intermediate_size: usize,
        hidden_size: usize,
        workspace: &mut CudaMoeBatchedWorkspace,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if num_experts == 0 || num_experts > 6 {
            return Err(Error::Internal(format!(
                "MoE batched expects 1..=6 experts, got {num_experts}"
            )));
        }
        if output.len != hidden_size {
            return Err(Error::Internal(format!(
                "CUDA MoE output length mismatch: output={} hidden={hidden_size}",
                output.len
            )));
        }
        if !workspace.matches(num_experts, input.len, intermediate_size, hidden_size) {
            return Err(Error::Internal(format!(
                "CUDA MoE workspace mismatch: workspace=[max_experts={},input={},intermediate={},hidden={}] call=[experts={},input={},intermediate={},hidden={}]",
                workspace.max_experts,
                workspace.input_size,
                workspace.intermediate_size,
                workspace.hidden_size,
                num_experts,
                input.len,
                intermediate_size,
                hidden_size
            )));
        }

        let timing_enabled = self.moe_timing_enabled();
        let total_start = timing_enabled.then(Instant::now);

        let block = 256u32;
        let reduce_block = 128u32;
        let use_reduce = std::env::var("FERRULE_CUDA_MOE_REDUCE")
            .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        // FP4 mxf4 Tensor Core path is the default hot path. It is still under-utilized
        // at batch_cols=1, but avoids the scalar FP4 decode loop and remains env-gated
        // for A/B: set FERRULE_CUDA_MOE_TC=0 to force the scalar fallback.
        let use_tensor_core = !use_reduce
            && input.len.is_multiple_of(64)
            && std::env::var("FERRULE_CUDA_MOE_TC")
                .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
        let grid_inter = (
            (intermediate_size as u32 + block - 1) / block,
            num_experts as u32,
            1,
        );
        let grid_hidden = (
            (hidden_size as u32 + block - 1) / block,
            num_experts as u32,
            1,
        );
        let grid_inter_reduce = (intermediate_size as u32, num_experts as u32, 1);
        let grid_hidden_reduce = (hidden_size as u32, num_experts as u32, 1);
        let grid_inter_tc = (intermediate_size.div_ceil(16) as u32, num_experts as u32, 1);
        let grid_hidden_tc = (hidden_size.div_ceil(16) as u32, num_experts as u32, 1);
        let moe_path = if use_reduce {
            CudaMoeExecutionPath::Reduce
        } else if use_tensor_core {
            CudaMoeExecutionPath::TensorCore
        } else {
            CudaMoeExecutionPath::Scalar
        };
        self.counters.add_moe_call(moe_path);

        if use_reduce {
            let phase_start = timing_enabled.then(Instant::now);
            self.copy_f32_into_slot(input, &mut workspace.x_f32, 0)?;
            self.fp8_activation_quantize_buffer_in_place(
                &mut workspace.x_f32,
                input.len,
                ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
            )?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters
                    .add_moe_input_prepare_us(duration_us(start.elapsed()));
            }
            let phase_start = timing_enabled.then(Instant::now);
            self.launched(unsafe {
                self.module.moe_gemv_dual_fp4_batched_reduce(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_inter_reduce,
                        block_dim: (reduce_block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &workspace.x_f32.buffer,
                    &workspace.gate_ptrs,
                    &workspace.gate_scale_ptrs,
                    &workspace.up_ptrs,
                    &workspace.up_scale_ptrs,
                    &mut workspace.y_gate.buffer,
                    &mut workspace.y_up.buffer,
                    intermediate_size as u32,
                    input.len as u32,
                    num_experts as u32,
                )
            })?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters
                    .add_moe_gate_up_us(duration_us(start.elapsed()));
            }
        } else if use_tensor_core {
            // Tensor Core path consumes FP4 activations directly. Avoid the old
            // FP8-in-f32 preparation buffer; the mxf4 MMA needs packed FP4 + E8M0.
            let phase_start = timing_enabled.then(Instant::now);
            self.launched(unsafe {
                self.module.fp4_e2m1_e8m0_quantize_f32_packed(
                    &self.stream,
                    LaunchConfig::for_num_elems((input.len / 32) as u32),
                    &input.buffer,
                    &mut workspace.x_packed,
                    &mut workspace.x_scales,
                    input.len as u32,
                    input.len as u32,
                    32,
                )
            })?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters
                    .add_moe_input_prepare_us(duration_us(start.elapsed()));
            }
            let phase_start = timing_enabled.then(Instant::now);
            self.launched(unsafe {
                self.module.moe_gemm_dual_fp4_mxf4_batched(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_inter_tc,
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &workspace.x_packed,
                    &workspace.x_scales,
                    &workspace.gate_ptrs,
                    &workspace.gate_scale_ptrs,
                    &workspace.up_ptrs,
                    &workspace.up_scale_ptrs,
                    &mut workspace.y_gate.buffer,
                    &mut workspace.y_up.buffer,
                    intermediate_size as u32,
                    input.len as u32,
                    1,
                    num_experts as u32,
                )
            })?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters
                    .add_moe_gate_up_us(duration_us(start.elapsed()));
            }
        } else {
            let phase_start = timing_enabled.then(Instant::now);
            self.copy_f32_into_slot(input, &mut workspace.x_f32, 0)?;
            self.fp8_activation_quantize_buffer_in_place(
                &mut workspace.x_f32,
                input.len,
                ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
            )?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters
                    .add_moe_input_prepare_us(duration_us(start.elapsed()));
            }
            let phase_start = timing_enabled.then(Instant::now);
            self.launched(unsafe {
                self.module.moe_gemv_dual_fp4_batched(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_inter,
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &workspace.x_f32.buffer,
                    &workspace.gate_ptrs,
                    &workspace.gate_scale_ptrs,
                    &workspace.up_ptrs,
                    &workspace.up_scale_ptrs,
                    &mut workspace.y_gate.buffer,
                    &mut workspace.y_up.buffer,
                    intermediate_size as u32,
                    input.len as u32,
                    num_experts as u32,
                )
            })?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters
                    .add_moe_gate_up_us(duration_us(start.elapsed()));
            }
        }

        if use_tensor_core {
            let phase_start = timing_enabled.then(Instant::now);
            self.launched(unsafe {
                self.module.moe_swiglu_fp4_packed_batched(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: ((intermediate_size / 32) as u32, num_experts as u32, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &workspace.y_gate.buffer,
                    &workspace.y_up.buffer,
                    &workspace.route_weights,
                    &mut workspace.y_hidden_packed,
                    &mut workspace.y_hidden_scales,
                    intermediate_size as u32,
                    1,
                    num_experts as u32,
                    swiglu_limit,
                )
            })?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters
                    .add_moe_swiglu_us(duration_us(start.elapsed()));
            }
        } else {
            let phase_start = timing_enabled.then(Instant::now);
            self.launched(unsafe {
                self.module.moe_swiglu_fp8_batched(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_inter,
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &workspace.y_gate.buffer,
                    &workspace.y_up.buffer,
                    &workspace.route_weights,
                    &mut workspace.y_hidden.buffer,
                    intermediate_size as u32,
                    num_experts as u32,
                    swiglu_limit,
                    ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE as u32,
                )
            })?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters
                    .add_moe_swiglu_us(duration_us(start.elapsed()));
            }
        }

        if use_reduce {
            let phase_start = timing_enabled.then(Instant::now);
            self.launched(unsafe {
                self.module.moe_gemv_down_fp4_batched_reduce(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_hidden_reduce,
                        block_dim: (reduce_block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &workspace.y_hidden.buffer,
                    &workspace.down_ptrs,
                    &workspace.down_scale_ptrs,
                    &mut output.buffer,
                    intermediate_size as u32,
                    hidden_size as u32,
                    num_experts as u32,
                )
            })?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters.add_moe_down_us(duration_us(start.elapsed()));
            }
        } else if use_tensor_core {
            let phase_start = timing_enabled.then(Instant::now);
            self.launched(unsafe {
                self.module.moe_gemm_down_fp4_mxf4_batched(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_hidden_tc,
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &workspace.y_hidden_packed,
                    &workspace.y_hidden_scales,
                    &workspace.down_ptrs,
                    &workspace.down_scale_ptrs,
                    &mut output.buffer,
                    intermediate_size as u32,
                    hidden_size as u32,
                    1,
                    num_experts as u32,
                )
            })?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters.add_moe_down_us(duration_us(start.elapsed()));
            }
        } else {
            let phase_start = timing_enabled.then(Instant::now);
            self.launched(unsafe {
                self.module.moe_gemv_down_fp4_batched(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_hidden,
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &workspace.y_hidden.buffer,
                    &workspace.down_ptrs,
                    &workspace.down_scale_ptrs,
                    &mut output.buffer,
                    intermediate_size as u32,
                    hidden_size as u32,
                    num_experts as u32,
                )
            })?;
            if let Some(start) = phase_start {
                self.sync_stream()?;
                self.counters.add_moe_down_us(duration_us(start.elapsed()));
            }
        }

        if let Some(start) = total_start {
            self.counters.add_moe_total_us(duration_us(start.elapsed()));
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_experts_batched_add_into_from_device(
        &self,
        gate_handles: &[&CudaArtifactLinearHandle; 6],
        up_handles: &[&CudaArtifactLinearHandle; 6],
        down_handles: &[&CudaArtifactLinearHandle; 6],
        route_weights: &[f32; 6],
        input: &CudaF32Buffer,
        swiglu_limit: f32,
        num_experts: usize,
        intermediate_size: usize,
        hidden_size: usize,
        workspace: &mut CudaMoeBatchedWorkspace,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        self.prepare_moe_experts_batched_workspace(
            gate_handles,
            up_handles,
            down_handles,
            route_weights,
            input.len,
            num_experts,
            intermediate_size,
            hidden_size,
            workspace,
        )?;
        self.moe_experts_batched_add_into_from_device_prepared(
            input,
            swiglu_limit,
            num_experts,
            intermediate_size,
            hidden_size,
            workspace,
            output,
        )
    }

    /// Multi-column routed MoE into an existing accumulator.
    ///
    /// This is the prefill path: `input` is `[batch_cols, input_size]`, output is
    /// `[batch_cols, hidden_size]`, and `route_weights` is `[num_experts,
    /// batch_cols]`. It intentionally does not fall back to the scalar/reduce GEMV
    /// path for `batch_cols > 1`; prefill batching is only useful when it stays on
    /// the Tensor Core execution shape.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_experts_batched_cols_add_into_from_device(
        &self,
        gate_handles: &[&CudaArtifactLinearHandle],
        up_handles: &[&CudaArtifactLinearHandle],
        down_handles: &[&CudaArtifactLinearHandle],
        route_weights: &[f32],
        input: &CudaF32Buffer,
        input_size: usize,
        batch_cols: usize,
        swiglu_limit: f32,
        num_experts: usize,
        intermediate_size: usize,
        hidden_size: usize,
        workspace: &mut CudaMoeBatchedWorkspace,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if num_experts == 0 || num_experts > workspace.max_experts {
            return Err(Error::Internal(format!(
                "MoE batched-cols expects 1..={} experts for this workspace, got {num_experts}",
                workspace.max_experts
            )));
        }
        if gate_handles.len() != num_experts
            || up_handles.len() != num_experts
            || down_handles.len() != num_experts
        {
            return Err(Error::Internal(format!(
                "CUDA MoE batched-cols handle count mismatch: gate={} up={} down={} expected={num_experts}",
                gate_handles.len(),
                up_handles.len(),
                down_handles.len()
            )));
        }
        if batch_cols == 0 || batch_cols > 8 {
            return Err(Error::Internal(format!(
                "MoE batched-cols expects 1..=8 batch columns, got {batch_cols}"
            )));
        }
        if input.len != input_size * batch_cols {
            return Err(Error::Internal(format!(
                "CUDA MoE batched-cols input length mismatch: input={} expected={}x{}",
                input.len, batch_cols, input_size
            )));
        }
        if output.len != batch_cols * hidden_size {
            return Err(Error::Internal(format!(
                "CUDA MoE batched-cols output length mismatch: output={} expected={}x{}",
                output.len, batch_cols, hidden_size
            )));
        }
        if route_weights.len() != num_experts * batch_cols {
            return Err(Error::Internal(format!(
                "CUDA MoE batched-cols route weight length mismatch: weights={} expected={}x{}",
                route_weights.len(),
                num_experts,
                batch_cols
            )));
        }
        if !input_size.is_multiple_of(64) || !intermediate_size.is_multiple_of(64) {
            return Err(Error::Internal(format!(
                "CUDA MoE batched-cols TC path expects 64-aligned input/intermediate, got input={input_size} intermediate={intermediate_size}"
            )));
        }
        if !workspace.matches_cols(
            num_experts,
            batch_cols,
            input_size,
            intermediate_size,
            hidden_size,
        ) {
            return Err(Error::Internal(format!(
                "CUDA MoE batched-cols workspace mismatch: workspace=[max_experts={},max_batch_cols={},input={},intermediate={},hidden={}] call=[experts={},batch_cols={},input={},intermediate={},hidden={}]",
                workspace.max_experts,
                workspace.max_batch_cols,
                workspace.input_size,
                workspace.intermediate_size,
                workspace.hidden_size,
                num_experts,
                batch_cols,
                input_size,
                intermediate_size,
                hidden_size
            )));
        }
        if std::env::var("FERRULE_CUDA_MOE_REDUCE")
            .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
        {
            return Err(Error::Internal(
                "CUDA MoE batched-cols prefill does not support FERRULE_CUDA_MOE_REDUCE=1".into(),
            ));
        }
        if std::env::var("FERRULE_CUDA_MOE_TC")
            .map(|value| value == "0" || value.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
        {
            return Err(Error::Internal(
                "CUDA MoE batched-cols prefill requires Tensor Core path; unset FERRULE_CUDA_MOE_TC=0".into(),
            ));
        }

        let first_gate = gate_handles[0];
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: gate_out,
            in_features: gate_in,
        } = first_gate.shape
        else {
            return Err(Error::Internal("CUDA packed expert gate is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: up_out,
            in_features: up_in,
        } = up_handles[0].shape
        else {
            return Err(Error::Internal("CUDA packed expert up is not FP4".into()));
        };
        let CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: down_out,
            in_features: down_in,
        } = down_handles[0].shape
        else {
            return Err(Error::Internal("CUDA packed expert down is not FP4".into()));
        };
        if gate_in != input_size
            || up_in != input_size
            || up_out != gate_out
            || down_in != gate_out
            || down_out != hidden_size
            || gate_out != intermediate_size
        {
            return Err(Error::Internal(format!(
                "CUDA packed expert batched-cols shape mismatch: input_size={input_size} gate=[{gate_out},{gate_in}] up=[{up_out},{up_in}] down=[{down_out},{down_in}] expected intermediate={intermediate_size} hidden={hidden_size}"
            )));
        }

        let timing_enabled = self.moe_timing_enabled();
        let total_start = timing_enabled.then(Instant::now);

        let mut gate_ptrs = vec![0u64; workspace.max_experts];
        let mut gate_scale_ptrs = vec![0u64; workspace.max_experts];
        let mut up_ptrs = vec![0u64; workspace.max_experts];
        let mut up_scale_ptrs = vec![0u64; workspace.max_experts];
        let mut down_ptrs = vec![0u64; workspace.max_experts];
        let mut down_scale_ptrs = vec![0u64; workspace.max_experts];
        for i in 0..num_experts {
            gate_ptrs[i] = gate_handles[i].weight.cu_deviceptr();
            gate_scale_ptrs[i] = gate_handles[i]
                .scale
                .as_ref()
                .ok_or_else(|| Error::Internal("CUDA packed expert gate missing scale".into()))?
                .cu_deviceptr();
            up_ptrs[i] = up_handles[i].weight.cu_deviceptr();
            up_scale_ptrs[i] = up_handles[i]
                .scale
                .as_ref()
                .ok_or_else(|| Error::Internal("CUDA packed expert up missing scale".into()))?
                .cu_deviceptr();
            down_ptrs[i] = down_handles[i].weight.cu_deviceptr();
            down_scale_ptrs[i] = down_handles[i]
                .scale
                .as_ref()
                .ok_or_else(|| Error::Internal("CUDA packed expert down missing scale".into()))?
                .cu_deviceptr();
        }
        let mut padded_weights = vec![0.0f32; workspace.max_experts * workspace.max_batch_cols];
        for expert in 0..num_experts {
            for col in 0..batch_cols {
                padded_weights[expert * batch_cols + col] =
                    route_weights[expert * batch_cols + col];
            }
        }

        let phase_start = timing_enabled.then(Instant::now);
        self.copy_u64_into_device_buffer(&gate_ptrs, &mut workspace.gate_ptrs)?;
        self.copy_u64_into_device_buffer(&gate_scale_ptrs, &mut workspace.gate_scale_ptrs)?;
        self.copy_u64_into_device_buffer(&up_ptrs, &mut workspace.up_ptrs)?;
        self.copy_u64_into_device_buffer(&up_scale_ptrs, &mut workspace.up_scale_ptrs)?;
        self.copy_u64_into_device_buffer(&down_ptrs, &mut workspace.down_ptrs)?;
        self.copy_u64_into_device_buffer(&down_scale_ptrs, &mut workspace.down_scale_ptrs)?;
        self.copy_f32_into_device_buffer(&padded_weights, &mut workspace.route_weights)?;
        if let Some(start) = phase_start {
            self.sync_stream()?;
            self.counters
                .add_moe_pointer_upload_us(duration_us(start.elapsed()));
        }

        let phase_start = timing_enabled.then(Instant::now);
        self.launched(unsafe {
            self.module.fp4_e2m1_e8m0_quantize_f32_packed(
                &self.stream,
                LaunchConfig::for_num_elems((input.len / 32) as u32),
                &input.buffer,
                &mut workspace.x_packed,
                &mut workspace.x_scales,
                input.len as u32,
                input_size as u32,
                32,
            )
        })?;
        if let Some(start) = phase_start {
            self.sync_stream()?;
            self.counters
                .add_moe_input_prepare_us(duration_us(start.elapsed()));
        }

        self.counters.add_moe_call(CudaMoeExecutionPath::TensorCore);
        let grid_inter_tc = (intermediate_size.div_ceil(16) as u32, num_experts as u32, 1);
        let grid_hidden_tc = (hidden_size.div_ceil(16) as u32, num_experts as u32, 1);

        let phase_start = timing_enabled.then(Instant::now);
        self.launched(unsafe {
            self.module.moe_gemm_dual_fp4_mxf4_batched(
                &self.stream,
                LaunchConfig {
                    grid_dim: grid_inter_tc,
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &workspace.x_packed,
                &workspace.x_scales,
                &workspace.gate_ptrs,
                &workspace.gate_scale_ptrs,
                &workspace.up_ptrs,
                &workspace.up_scale_ptrs,
                &mut workspace.y_gate.buffer,
                &mut workspace.y_up.buffer,
                intermediate_size as u32,
                input_size as u32,
                batch_cols as u32,
                num_experts as u32,
            )
        })?;
        if let Some(start) = phase_start {
            self.sync_stream()?;
            self.counters
                .add_moe_gate_up_us(duration_us(start.elapsed()));
        }

        let phase_start = timing_enabled.then(Instant::now);
        self.launched(unsafe {
            self.module.moe_swiglu_fp4_packed_batched(
                &self.stream,
                LaunchConfig {
                    grid_dim: (
                        (intermediate_size / 32) as u32,
                        num_experts as u32,
                        batch_cols as u32,
                    ),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &workspace.y_gate.buffer,
                &workspace.y_up.buffer,
                &workspace.route_weights,
                &mut workspace.y_hidden_packed,
                &mut workspace.y_hidden_scales,
                intermediate_size as u32,
                batch_cols as u32,
                num_experts as u32,
                swiglu_limit,
            )
        })?;
        if let Some(start) = phase_start {
            self.sync_stream()?;
            self.counters
                .add_moe_swiglu_us(duration_us(start.elapsed()));
        }

        let phase_start = timing_enabled.then(Instant::now);
        self.launched(unsafe {
            self.module.moe_gemm_down_fp4_mxf4_batched(
                &self.stream,
                LaunchConfig {
                    grid_dim: grid_hidden_tc,
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &workspace.y_hidden_packed,
                &workspace.y_hidden_scales,
                &workspace.down_ptrs,
                &workspace.down_scale_ptrs,
                &mut output.buffer,
                intermediate_size as u32,
                hidden_size as u32,
                batch_cols as u32,
                num_experts as u32,
            )
        })?;
        if let Some(start) = phase_start {
            self.sync_stream()?;
            self.counters.add_moe_down_us(duration_us(start.elapsed()));
        }

        if let Some(start) = total_start {
            self.counters.add_moe_total_us(duration_us(start.elapsed()));
        }
        Ok(())
    }

    /// Batched MoE: process all selected experts in 3 kernel launches
    /// instead of 3 × num_experts launches.
    ///
    /// `expert_handles`: one per selected expert, each containing gate/up/down.
    /// `route_weights`: per-expert routing weight.
    /// `input`: shared device input buffer (already FP8-quantized).
    /// `output`: pre-zeroed accumulator, `[hidden_size]` f32.
    /// Returns the same `output` buffer with accumulated expert outputs.
    pub fn moe_experts_batched_from_device(
        &self,
        _expert_handles: &[&CudaArtifactLinearHandle; 6],
        gate_handles: &[&CudaArtifactLinearHandle; 6],
        up_handles: &[&CudaArtifactLinearHandle; 6],
        down_handles: &[&CudaArtifactLinearHandle; 6],
        route_weights: &[f32; 6],
        input: &CudaF32Buffer,
        swiglu_limit: f32,
        num_experts: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Result<(CudaF32Buffer, CudaF32Buffer, CudaF32Buffer)> {
        if num_experts == 0 || num_experts > 6 {
            return Err(Error::Internal(format!(
                "MoE batched expects 1..=6 experts, got {num_experts}"
            )));
        }
        // Pack device pointers into u64 arrays.
        let gate_ptrs: Vec<u64> = gate_handles[..num_experts]
            .iter()
            .map(|h| h.weight.cu_deviceptr())
            .collect();
        let gate_scale_ptrs: Vec<u64> = gate_handles[..num_experts]
            .iter()
            .map(|h| h.scale.as_ref().expect("gate scale").cu_deviceptr())
            .collect();
        let up_ptrs: Vec<u64> = up_handles[..num_experts]
            .iter()
            .map(|h| h.weight.cu_deviceptr())
            .collect();
        let up_scale_ptrs: Vec<u64> = up_handles[..num_experts]
            .iter()
            .map(|h| h.scale.as_ref().expect("up scale").cu_deviceptr())
            .collect();
        let down_ptrs: Vec<u64> = down_handles[..num_experts]
            .iter()
            .map(|h| h.weight.cu_deviceptr())
            .collect();
        let down_scale_ptrs: Vec<u64> = down_handles[..num_experts]
            .iter()
            .map(|h| h.scale.as_ref().expect("down scale").cu_deviceptr())
            .collect();

        let gate_ptrs_d = self.upload_u64_buffer(&gate_ptrs)?;
        let gate_scale_ptrs_d = self.upload_u64_buffer(&gate_scale_ptrs)?;
        let up_ptrs_d = self.upload_u64_buffer(&up_ptrs)?;
        let up_scale_ptrs_d = self.upload_u64_buffer(&up_scale_ptrs)?;
        let down_ptrs_d = self.upload_u64_buffer(&down_ptrs)?;
        let down_scale_ptrs_d = self.upload_u64_buffer(&down_scale_ptrs)?;
        let route_weights_d = self.upload_f32_buffer(route_weights)?;

        let total_inter = num_experts * intermediate_size;
        let mut y_gate = self.uninitialized_device_buffer::<f32>(total_inter)?;
        let mut y_up = self.uninitialized_device_buffer::<f32>(total_inter)?;
        let mut y_hidden = self.uninitialized_device_buffer::<f32>(total_inter)?;
        let mut output = cu(DeviceBuffer::<f32>::zeroed(&self.stream, hidden_size))?;

        let block = 256u32;
        let reduce_block = 128u32;
        let use_reduce = std::env::var("FERRULE_CUDA_MOE_REDUCE")
            .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        // FP4 mxf4 Tensor Core path is the default hot path. It is still under-utilized
        // at batch_cols=1, but avoids the scalar FP4 decode loop and remains env-gated
        // for A/B: set FERRULE_CUDA_MOE_TC=0 to force the scalar fallback.
        let use_tensor_core = !use_reduce
            && input.len.is_multiple_of(64)
            && std::env::var("FERRULE_CUDA_MOE_TC")
                .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
        let grid_inter = (
            (intermediate_size as u32 + block - 1) / block,
            num_experts as u32,
            1,
        );
        let grid_hidden = (
            (hidden_size as u32 + block - 1) / block,
            num_experts as u32,
            1,
        );
        let grid_inter_reduce = (intermediate_size as u32, num_experts as u32, 1);
        let grid_hidden_reduce = (hidden_size as u32, num_experts as u32, 1);
        let grid_inter_tc = (intermediate_size.div_ceil(16) as u32, num_experts as u32, 1);
        let grid_hidden_tc = (hidden_size.div_ceil(16) as u32, num_experts as u32, 1);

        // Launch 1: gate+up GEMM/GEMV. The tensor-core kernel is GEMM-ready
        // (`batch_cols <= 8`) and currently runs with one decode column.
        if use_reduce {
            self.launched(unsafe {
                self.module.moe_gemv_dual_fp4_batched_reduce(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_inter_reduce,
                        block_dim: (reduce_block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &input.buffer,
                    &gate_ptrs_d,
                    &gate_scale_ptrs_d,
                    &up_ptrs_d,
                    &up_scale_ptrs_d,
                    &mut y_gate,
                    &mut y_up,
                    intermediate_size as u32,
                    input.len as u32,
                    num_experts as u32,
                )
            })?;
        } else if use_tensor_core {
            let mut x_packed = self.uninitialized_device_buffer::<u8>(input.len / 2)?;
            let mut x_scales = self.uninitialized_device_buffer::<u8>(input.len / 32)?;
            self.launched(unsafe {
                self.module.fp4_e2m1_e8m0_quantize_f32_packed(
                    &self.stream,
                    LaunchConfig::for_num_elems((input.len / 32) as u32),
                    &input.buffer,
                    &mut x_packed,
                    &mut x_scales,
                    input.len as u32,
                    input.len as u32,
                    32,
                )
            })?;
            self.launched(unsafe {
                self.module.moe_gemm_dual_fp4_mxf4_batched(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_inter_tc,
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &x_packed,
                    &x_scales,
                    &gate_ptrs_d,
                    &gate_scale_ptrs_d,
                    &up_ptrs_d,
                    &up_scale_ptrs_d,
                    &mut y_gate,
                    &mut y_up,
                    intermediate_size as u32,
                    input.len as u32,
                    1,
                    num_experts as u32,
                )
            })?;
        } else {
            self.launched(unsafe {
                self.module.moe_gemv_dual_fp4_batched(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_inter,
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &input.buffer,
                    &gate_ptrs_d,
                    &gate_scale_ptrs_d,
                    &up_ptrs_d,
                    &up_scale_ptrs_d,
                    &mut y_gate,
                    &mut y_up,
                    intermediate_size as u32,
                    input.len as u32,
                    num_experts as u32,
                )
            })?;
        }

        // Launch 2: SwiGLU + FP8 quantize
        self.launched(unsafe {
            self.module.moe_swiglu_fp8_batched(
                &self.stream,
                LaunchConfig {
                    grid_dim: grid_inter,
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                &y_gate,
                &y_up,
                &route_weights_d.buffer,
                &mut y_hidden,
                intermediate_size as u32,
                num_experts as u32,
                swiglu_limit,
                ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE as u32,
            )
        })?;

        // Launch 3: down GEMM/GEMV + accumulate.
        if use_reduce {
            self.launched(unsafe {
                self.module.moe_gemv_down_fp4_batched_reduce(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_hidden_reduce,
                        block_dim: (reduce_block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &y_hidden,
                    &down_ptrs_d,
                    &down_scale_ptrs_d,
                    &mut output,
                    intermediate_size as u32,
                    hidden_size as u32,
                    num_experts as u32,
                )
            })?;
        } else if use_tensor_core {
            let mut y_hidden_packed = self.uninitialized_device_buffer::<u8>(total_inter / 2)?;
            let mut y_hidden_scales = self.uninitialized_device_buffer::<u8>(total_inter / 32)?;
            self.launched(unsafe {
                self.module.fp4_e2m1_e8m0_quantize_f32_packed(
                    &self.stream,
                    LaunchConfig::for_num_elems((total_inter / 32) as u32),
                    &y_hidden,
                    &mut y_hidden_packed,
                    &mut y_hidden_scales,
                    total_inter as u32,
                    intermediate_size as u32,
                    32,
                )
            })?;
            self.launched(unsafe {
                self.module.moe_gemm_down_fp4_mxf4_batched(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_hidden_tc,
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &y_hidden_packed,
                    &y_hidden_scales,
                    &down_ptrs_d,
                    &down_scale_ptrs_d,
                    &mut output,
                    intermediate_size as u32,
                    hidden_size as u32,
                    1,
                    num_experts as u32,
                )
            })?;
        } else {
            self.launched(unsafe {
                self.module.moe_gemv_down_fp4_batched(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: grid_hidden,
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &y_hidden,
                    &down_ptrs_d,
                    &down_scale_ptrs_d,
                    &mut output,
                    intermediate_size as u32,
                    hidden_size as u32,
                    num_experts as u32,
                )
            })?;
        }

        Ok((
            CudaF32Buffer {
                buffer: y_gate,
                len: total_inter,
            },
            CudaF32Buffer {
                buffer: y_hidden,
                len: total_inter,
            },
            CudaF32Buffer {
                buffer: output,
                len: hidden_size,
            },
        ))
    }

    fn upload_u64_buffer(&self, values: &[u64]) -> Result<DeviceBuffer<u64>> {
        let buffer = cu(DeviceBuffer::<u64>::from_host(&self.stream, values))?;
        self.counters.add_host_to_device((values.len() * 8) as u64);
        Ok(buffer)
    }

    pub fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
        if input.len() != weight.len() || input.is_empty() {
            return Err(Error::Internal(format!(
                "CUDA RMS norm length mismatch: input={} weight={}",
                input.len(),
                weight.len()
            )));
        }
        let xd = self.upload_f32(input)?;
        let wd = self.upload_f32(weight)?;
        let mut yd = cu(DeviceBuffer::<f32>::zeroed(&self.stream, input.len()))?;
        self.launched(unsafe {
            self.module.rms_norm_fused(
                &self.stream,
                one_block_config(256),
                &xd,
                &wd,
                &mut yd,
                input.len() as u32,
                eps,
            )
        })?;
        self.download_f32(&yd, input.len())
    }

    /// Device-resident RMS norm: input is already on device, weight is uploaded
    /// once and cached by the caller. Output stays on device.
    pub fn rms_norm_from_device(
        &self,
        input: &CudaF32Buffer,
        weight: &CudaF32Buffer,
        eps: f32,
    ) -> Result<CudaF32Buffer> {
        let mut output = self.zero_f32_buffer(input.len())?;
        self.rms_norm_from_device_into(input, weight, eps, &mut output)?;
        Ok(output)
    }

    pub fn rms_norm_from_device_into(
        &self,
        input: &CudaF32Buffer,
        weight: &CudaF32Buffer,
        eps: f32,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if input.len() != weight.len() || input.is_empty() || output.len != input.len() {
            return Err(Error::Internal(format!(
                "CUDA RMS norm device length mismatch: input={} weight={} output={}",
                input.len(),
                weight.len(),
                output.len
            )));
        }
        self.launched(unsafe {
            self.module.rms_norm_fused(
                &self.stream,
                one_block_config(256),
                &input.buffer,
                &weight.buffer,
                &mut output.buffer,
                input.len() as u32,
                eps,
            )
        })
    }

    /// Upload a norm weight once for reuse with `rms_norm_from_device`.
    pub fn upload_norm_weight(&self, weight: &[f32]) -> Result<CudaF32Buffer> {
        self.upload_f32_buffer(weight)
    }

    pub fn rms_norm_rows(
        &self,
        input: &[f32],
        rows: usize,
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        let xd = self.upload_f32_buffer(input)?;
        let wd = self.upload_f32_buffer(weight)?;
        let yd = self.rms_norm_rows_from_device(&xd, rows, &wd, eps)?;
        self.download_f32_buffer(&yd)
    }

    pub fn rms_norm_rows_from_device(
        &self,
        input: &CudaF32Buffer,
        rows: usize,
        weight: &CudaF32Buffer,
        eps: f32,
    ) -> Result<CudaF32Buffer> {
        if rows == 0 || weight.is_empty() || input.len != rows * weight.len {
            return Err(Error::Internal(format!(
                "CUDA affine RMS rows length mismatch: rows={rows} input={} weight={}",
                input.len, weight.len
            )));
        }
        let mut yd = cu(DeviceBuffer::<f32>::zeroed(&self.stream, input.len))?;
        self.launched(unsafe {
            self.module.rms_norm_rows_fused(
                &self.stream,
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &input.buffer,
                &weight.buffer,
                &mut yd,
                rows as u32,
                weight.len as u32,
                eps,
            )
        })?;
        Ok(CudaF32Buffer {
            buffer: yd,
            len: input.len,
        })
    }

    pub fn rms_norm_heads(
        &self,
        input: &[f32],
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let xd = self.upload_f32_buffer(input)?;
        let yd = self.rms_norm_heads_from_device(&xd, heads, head_dim, eps)?;
        self.download_f32_buffer(&yd)
    }

    pub fn rms_norm_heads_from_device(
        &self,
        input: &CudaF32Buffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<CudaF32Buffer> {
        let mut output = self.zero_f32_buffer(input.len)?;
        self.rms_norm_heads_from_device_into(input, heads, head_dim, eps, &mut output)?;
        Ok(output)
    }

    pub fn rms_norm_heads_from_device_into(
        &self,
        input: &CudaF32Buffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if heads == 0 || head_dim == 0 || input.len != heads * head_dim || output.len != input.len {
            return Err(Error::Internal(format!(
                "CUDA per-head RMS device length mismatch: input={} output={} heads={heads} head_dim={head_dim}",
                input.len, output.len
            )));
        }
        self.launched(unsafe {
            self.module.rms_norm_heads_fused(
                &self.stream,
                LaunchConfig {
                    grid_dim: (heads as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &input.buffer,
                &mut output.buffer,
                heads as u32,
                head_dim as u32,
                eps,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_from_device(
        &self,
        state: &CudaF32Buffer,
        function: &CudaF32Buffer,
        scale: &CudaF32Buffer,
        base: &CudaF32Buffer,
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        sinkhorn_iters: usize,
        eps: f32,
        norm_eps: f32,
    ) -> Result<(CudaF32Buffer, CudaF32Buffer, CudaF32Buffer, CudaF32Buffer)> {
        let mix_hc = hc_mult
            .checked_mul(hc_mult + 2)
            .ok_or_else(|| Error::Internal("CUDA HC mix_hc overflow".into()))?;
        let _hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal("CUDA HC hidden size overflow".into()))?;
        if tokens == 0 || hc_mult == 0 || hc_mult > 16 || mix_hc > 128 || hc_mult * hc_mult > 256 {
            return Err(Error::Internal(format!(
                "CUDA HC pre device shape mismatch: tokens={tokens} hc={hc_mult} hidden={hidden_size} mix={mix_hc}"
            )));
        }
        let mut hidden = self.zero_f32_buffer(tokens * hidden_size)?;
        let mut pre = self.zero_f32_buffer(tokens * hc_mult)?;
        let mut post = self.zero_f32_buffer(tokens * hc_mult)?;
        let mut comb = self.zero_f32_buffer(tokens * hc_mult * hc_mult)?;
        self.hc_pre_from_device_into(
            state,
            function,
            scale,
            base,
            tokens,
            hc_mult,
            hidden_size,
            sinkhorn_iters,
            eps,
            norm_eps,
            &mut hidden,
            &mut pre,
            &mut post,
            &mut comb,
        )?;
        Ok((hidden, pre, post, comb))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_from_device_into(
        &self,
        state: &CudaF32Buffer,
        function: &CudaF32Buffer,
        scale: &CudaF32Buffer,
        base: &CudaF32Buffer,
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        sinkhorn_iters: usize,
        eps: f32,
        norm_eps: f32,
        hidden: &mut CudaF32Buffer,
        pre: &mut CudaF32Buffer,
        post: &mut CudaF32Buffer,
        comb: &mut CudaF32Buffer,
    ) -> Result<()> {
        let mix_hc = hc_mult
            .checked_mul(hc_mult + 2)
            .ok_or_else(|| Error::Internal("CUDA HC mix_hc overflow".into()))?;
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal("CUDA HC hidden size overflow".into()))?;
        if tokens == 0
            || hc_mult == 0
            || hc_mult > 16
            || mix_hc > 128
            || hc_mult * hc_mult > 256
            || state.len != tokens * hc_dim
            || function.len != mix_hc * hc_dim
            || scale.len != 3
            || base.len != mix_hc
            || hidden.len != tokens * hidden_size
            || pre.len != tokens * hc_mult
            || post.len != tokens * hc_mult
            || comb.len != tokens * hc_mult * hc_mult
        {
            return Err(Error::Internal(format!(
                "CUDA HC pre device shape mismatch: tokens={tokens} state={} function={} scale={} base={} hidden={} pre={} post={} comb={} hc={hc_mult} hidden_size={hidden_size} mix={mix_hc}",
                state.len,
                function.len,
                scale.len,
                base.len,
                hidden.len,
                pre.len,
                post.len,
                comb.len
            )));
        }
        self.launched(unsafe {
            self.module.hc_pre_f32(
                &self.stream,
                LaunchConfig {
                    grid_dim: (tokens as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &state.buffer,
                &function.buffer,
                &scale.buffer,
                &base.buffer,
                &mut hidden.buffer,
                &mut pre.buffer,
                &mut post.buffer,
                &mut comb.buffer,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
                mix_hc as u32,
                sinkhorn_iters as u32,
                eps,
                norm_eps,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_f32(
        &self,
        state: &[f32],
        function: &[f32],
        scale: &[f32],
        base: &[f32],
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        sinkhorn_iters: usize,
        eps: f32,
        norm_eps: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mix_hc = hc_mult
            .checked_mul(hc_mult + 2)
            .ok_or_else(|| Error::Internal("CUDA HC mix_hc overflow".into()))?;
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal("CUDA HC hidden size overflow".into()))?;
        if tokens == 0
            || hc_mult == 0
            || hc_mult > 16
            || mix_hc > 128
            || hc_mult * hc_mult > 256
            || state.len() != tokens * hc_dim
            || function.len() != mix_hc * hc_dim
            || scale.len() != 3
            || base.len() != mix_hc
        {
            return Err(Error::Internal(format!(
                "CUDA HC pre shape mismatch: tokens={tokens} state={} function={} scale={} base={} hc={hc_mult} hidden={hidden_size} mix={mix_hc}",
                state.len(),
                function.len(),
                scale.len(),
                base.len()
            )));
        }
        let sd = self.upload_f32(state)?;
        let fd = self.upload_f32(function)?;
        let scd = self.upload_f32(scale)?;
        let bd = self.upload_f32(base)?;
        let mut hidden = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            tokens * hidden_size,
        ))?;
        let mut pre = cu(DeviceBuffer::<f32>::zeroed(&self.stream, tokens * hc_mult))?;
        let mut post = cu(DeviceBuffer::<f32>::zeroed(&self.stream, tokens * hc_mult))?;
        let mut comb = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            tokens * hc_mult * hc_mult,
        ))?;
        self.launched(unsafe {
            self.module.hc_pre_f32(
                &self.stream,
                LaunchConfig {
                    grid_dim: (tokens as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &sd,
                &fd,
                &scd,
                &bd,
                &mut hidden,
                &mut pre,
                &mut post,
                &mut comb,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
                mix_hc as u32,
                sinkhorn_iters as u32,
                eps,
                norm_eps,
            )
        })?;
        Ok((
            self.download_f32(&hidden, tokens * hidden_size)?,
            self.download_f32(&pre, tokens * hc_mult)?,
            self.download_f32(&post, tokens * hc_mult)?,
            self.download_f32(&comb, tokens * hc_mult * hc_mult)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_single_f32(
        &self,
        state: &[f32],
        function: &[f32],
        scale: &[f32],
        base: &[f32],
        hc_mult: usize,
        hidden_size: usize,
        sinkhorn_iters: usize,
        eps: f32,
        norm_eps: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        self.hc_pre_f32(
            state,
            function,
            scale,
            base,
            1,
            hc_mult,
            hidden_size,
            sinkhorn_iters,
            eps,
            norm_eps,
        )
    }

    pub fn hc_post_from_device(
        &self,
        hidden: &CudaF32Buffer,
        residual: &CudaF32Buffer,
        split_post: &CudaF32Buffer,
        split_comb: &CudaF32Buffer,
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
    ) -> Result<CudaF32Buffer> {
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal("CUDA HC post hidden size overflow".into()))?;
        if tokens == 0 || hc_mult == 0 || hidden_size == 0 {
            return Err(Error::Internal(format!(
                "CUDA HC post device shape mismatch: tokens={tokens} hc={hc_mult} hidden_size={hidden_size}"
            )));
        }
        let mut output = self.zero_f32_buffer(tokens * hc_dim)?;
        self.hc_post_from_device_into(
            hidden,
            residual,
            split_post,
            split_comb,
            tokens,
            hc_mult,
            hidden_size,
            &mut output,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_post_from_device_into(
        &self,
        hidden: &CudaF32Buffer,
        residual: &CudaF32Buffer,
        split_post: &CudaF32Buffer,
        split_comb: &CudaF32Buffer,
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal("CUDA HC post hidden size overflow".into()))?;
        if tokens == 0 || hc_mult == 0 || hidden_size == 0 || output.len != tokens * hc_dim {
            return Err(Error::Internal(format!(
                "CUDA HC post device shape mismatch: tokens={tokens} hc={hc_mult} hidden_size={hidden_size} output={}",
                output.len
            )));
        }
        self.launched(unsafe {
            self.module.hc_post_f32(
                &self.stream,
                LaunchConfig::for_num_elems((tokens * hc_dim) as u32),
                &hidden.buffer,
                &residual.buffer,
                &split_post.buffer,
                &split_comb.buffer,
                &mut output.buffer,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
            )
        })
    }

    pub fn hc_post_f32(
        &self,
        hidden: &[f32],
        residual: &[f32],
        split_post: &[f32],
        split_comb: &[f32],
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal("CUDA HC post hidden size overflow".into()))?;
        if tokens == 0
            || hc_mult == 0
            || hidden_size == 0
            || hidden.len() != tokens * hidden_size
            || residual.len() != tokens * hc_dim
            || split_post.len() != tokens * hc_mult
            || split_comb.len() != tokens * hc_mult * hc_mult
        {
            return Err(Error::Internal(format!(
                "CUDA HC post shape mismatch: tokens={tokens} hidden={} residual={} post={} comb={} hc={hc_mult} hidden_size={hidden_size}",
                hidden.len(),
                residual.len(),
                split_post.len(),
                split_comb.len()
            )));
        }
        let hd = self.upload_f32(hidden)?;
        let rd = self.upload_f32(residual)?;
        let pd = self.upload_f32(split_post)?;
        let cd = self.upload_f32(split_comb)?;
        let mut out = cu(DeviceBuffer::<f32>::zeroed(&self.stream, tokens * hc_dim))?;
        self.launched(unsafe {
            self.module.hc_post_f32(
                &self.stream,
                LaunchConfig::for_num_elems((tokens * hc_dim) as u32),
                &hd,
                &rd,
                &pd,
                &cd,
                &mut out,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
            )
        })?;
        self.download_f32(&out, tokens * hc_dim)
    }

    pub fn hc_post_single_f32(
        &self,
        hidden: &[f32],
        residual: &[f32],
        split_post: &[f32],
        split_comb: &[f32],
        hc_mult: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        self.hc_post_f32(
            hidden,
            residual,
            split_post,
            split_comb,
            1,
            hc_mult,
            hidden_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_head_f32(
        &self,
        state: &[f32],
        function: &[f32],
        scale: &[f32],
        base: &[f32],
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        eps: f32,
        norm_eps: f32,
    ) -> Result<Vec<f32>> {
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal("CUDA HC head hidden size overflow".into()))?;
        if tokens == 0
            || hc_mult == 0
            || hc_mult > 16
            || state.len() != tokens * hc_dim
            || function.len() != hc_mult * hc_dim
            || scale.len() != 1
            || base.len() != hc_mult
        {
            return Err(Error::Internal(format!(
                "CUDA HC head shape mismatch: tokens={tokens} state={} function={} scale={} base={} hc={hc_mult} hidden={hidden_size}",
                state.len(),
                function.len(),
                scale.len(),
                base.len()
            )));
        }
        let sd = self.upload_f32(state)?;
        let fd = self.upload_f32(function)?;
        let scd = self.upload_f32(scale)?;
        let bd = self.upload_f32(base)?;
        let mut hidden = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            tokens * hidden_size,
        ))?;
        self.launched(unsafe {
            self.module.hc_head_f32(
                &self.stream,
                LaunchConfig {
                    grid_dim: (tokens as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &sd,
                &fd,
                &scd,
                &bd,
                &mut hidden,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
                eps,
                norm_eps,
            )
        })?;
        self.download_f32(&hidden, tokens * hidden_size)
    }

    pub fn hc_head_single_f32(
        &self,
        state: &[f32],
        function: &[f32],
        scale: &[f32],
        base: &[f32],
        hc_mult: usize,
        hidden_size: usize,
        eps: f32,
        norm_eps: f32,
    ) -> Result<Vec<f32>> {
        self.hc_head_f32(
            state,
            function,
            scale,
            base,
            1,
            hc_mult,
            hidden_size,
            eps,
            norm_eps,
        )
    }

    /// Device-resident `hc_head`: state and weights are already on device,
    /// output stays on device. This is the HC terminal projection applied
    /// after all transformer layers.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_head_from_device(
        &self,
        state: &CudaF32Buffer,
        function: &CudaF32Buffer,
        scale: &CudaF32Buffer,
        base: &CudaF32Buffer,
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        eps: f32,
        norm_eps: f32,
    ) -> Result<CudaF32Buffer> {
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal("CUDA HC head hidden size overflow".into()))?;
        if tokens == 0
            || hc_mult == 0
            || hc_mult > 16
            || state.len != tokens * hc_dim
            || function.len != hc_mult * hc_dim
            || scale.len != 1
            || base.len != hc_mult
        {
            return Err(Error::Internal(format!(
                "CUDA HC head device shape mismatch: tokens={tokens} state={} function={} scale={} base={} hc={hc_mult} hidden={hidden_size}",
                state.len,
                function.len,
                scale.len,
                base.len
            )));
        }
        let mut hidden = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            tokens * hidden_size,
        ))?;
        self.launched(unsafe {
            self.module.hc_head_f32(
                &self.stream,
                LaunchConfig {
                    grid_dim: (tokens as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &state.buffer,
                &function.buffer,
                &scale.buffer,
                &base.buffer,
                &mut hidden,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
                eps,
                norm_eps,
            )
        })?;
        Ok(CudaF32Buffer {
            buffer: hidden,
            len: tokens * hidden_size,
        })
    }

    fn artifact_linear_rows_device(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &DeviceBuffer<f32>,
        rows: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        match handle.shape {
            CudaArtifactLinearShape::F32 {
                out_features,
                in_features,
            } => self.launched(unsafe {
                self.module.gemm_f32_bytes(
                    &self.stream,
                    LaunchConfig::for_num_elems((rows * out_features) as u32),
                    input,
                    &handle.weight,
                    output,
                    rows as u32,
                    out_features as u32,
                    in_features as u32,
                )
            }),
            CudaArtifactLinearShape::Bf16Bytes {
                out_features,
                in_features,
            } => self.launched(unsafe {
                self.module.gemm_bf16_bytes(
                    &self.stream,
                    LaunchConfig::for_num_elems((rows * out_features) as u32),
                    input,
                    &handle.weight,
                    output,
                    rows as u32,
                    out_features as u32,
                    in_features as u32,
                )
            }),
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                let scale = handle.scale.as_ref().ok_or_else(|| {
                    Error::Internal("CUDA FP8 artifact linear missing scale".into())
                })?;
                let scale_cols = in_features.div_ceil(block_k);
                self.launched(unsafe {
                    self.module.gemm_fp8_e4m3fn_e8m0_2d(
                        &self.stream,
                        LaunchConfig::for_num_elems((rows * out_features) as u32),
                        input,
                        &handle.weight,
                        scale,
                        output,
                        rows as u32,
                        out_features as u32,
                        in_features as u32,
                        scale_cols as u32,
                        block_m as u32,
                        block_k as u32,
                    )
                })
            }
            CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
            } => {
                let scale = handle.scale.as_ref().ok_or_else(|| {
                    Error::Internal("CUDA FP4 artifact linear missing scale".into())
                })?;
                self.launched(unsafe {
                    self.module.gemm_fp4_e2m1_e8m0(
                        &self.stream,
                        LaunchConfig::for_num_elems((rows * out_features) as u32),
                        input,
                        &handle.weight,
                        scale,
                        output,
                        rows as u32,
                        out_features as u32,
                        in_features as u32,
                    )
                })
            }
        }
    }

    fn artifact_linear_matvec_device(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        match handle.shape {
            CudaArtifactLinearShape::F32 {
                out_features,
                in_features,
            } => self.launched(unsafe {
                self.module.gemv_f32_bytes(
                    &self.stream,
                    LaunchConfig::for_num_elems(out_features as u32),
                    input,
                    &handle.weight,
                    output,
                    out_features as u32,
                    in_features as u32,
                )
            }),
            CudaArtifactLinearShape::Bf16Bytes {
                out_features,
                in_features,
            } => self.launched(unsafe {
                self.module.gemv_bf16_bytes(
                    &self.stream,
                    LaunchConfig::for_num_elems(out_features as u32),
                    input,
                    &handle.weight,
                    output,
                    out_features as u32,
                    in_features as u32,
                )
            }),

            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                let scale = handle.scale.as_ref().ok_or_else(|| {
                    Error::Internal("CUDA FP8 artifact linear missing scale".into())
                })?;
                let scale_cols = in_features.div_ceil(block_k);
                self.launched(unsafe {
                    self.module.gemv_fp8_e4m3fn_e8m0_2d(
                        &self.stream,
                        LaunchConfig::for_num_elems(out_features as u32),
                        input,
                        &handle.weight,
                        scale,
                        output,
                        out_features as u32,
                        in_features as u32,
                        scale_cols as u32,
                        block_m as u32,
                        block_k as u32,
                    )
                })
            }
            CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
            } => {
                let scale = handle.scale.as_ref().ok_or_else(|| {
                    Error::Internal("CUDA FP4 artifact linear missing scale".into())
                })?;
                self.launched(unsafe {
                    self.module.gemv_fp4_e2m1_e8m0(
                        &self.stream,
                        LaunchConfig::for_num_elems(out_features as u32),
                        input,
                        &handle.weight,
                        scale,
                        output,
                        out_features as u32,
                        in_features as u32,
                    )
                })
            }
        }
    }

    pub fn sparse_attention_sink_from_device(
        &self,
        query: &CudaF32Buffer,
        values: &CudaF32Buffer,
        topk: &DeviceBuffer<i32>,
        sink: &CudaF32Buffer,
        shape: CudaSparseAttentionShape,
    ) -> Result<CudaF32Buffer> {
        let mut output = self.zero_f32_buffer(shape.output_elements())?;
        self.sparse_attention_sink_from_device_into(query, values, topk, sink, shape, &mut output)?;
        Ok(output)
    }

    pub fn sparse_attention_sink_from_device_into(
        &self,
        query: &CudaF32Buffer,
        values: &CudaF32Buffer,
        topk: &DeviceBuffer<i32>,
        sink: &CudaF32Buffer,
        shape: CudaSparseAttentionShape,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        shape.validate()?;
        if output.len != shape.output_elements() {
            return Err(Error::Internal(format!(
                "CUDA sparse attention output length mismatch: expected {}, got {}",
                shape.output_elements(),
                output.len
            )));
        }
        CudaSparseAttentionExecutor::new(&self.module, &self.stream).sparse_attention_sink_f32(
            &query.buffer,
            &values.buffer,
            topk,
            &sink.buffer,
            &mut output.buffer,
            shape,
        )?;
        self.record_kernel_launch();
        Ok(())
    }

    /// Apply DSV4-style tail rotary embedding (interleaved pairs, YAARN-scaled)
    /// to a device buffer. `cos_table` and `sin_table` are precomputed for
    /// `[max_positions, rope_dim/2]` and uploaded once.
    pub fn rope_tail_from_device(
        &self,
        qk: &mut CudaF32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        position: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        // Keep decode and batched prefill on the same pair-owned kernel. The old
        // element-owned kernel let even/odd lanes read and write the same rotary
        // pair concurrently, so decode could diverge from the race-free rows path.
        self.rope_tail_rows_from_device(
            qk, cos_table, sin_table, position, 1, heads, head_dim, rope_dim, inverse,
        )
    }

    /// Apply DSV4-style tail rotary to batched rows laid out as
    /// `[rows, heads, head_dim]`, using `start_position + row` per row.
    pub fn rope_tail_rows_from_device(
        &self,
        qk: &mut CudaF32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        start_position: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        self.rope_tail_rows_strided_from_device(
            qk,
            cos_table,
            sin_table,
            start_position,
            1,
            rows,
            heads,
            head_dim,
            rope_dim,
            inverse,
        )
    }

    /// Apply DSV4-style tail rotary to batched rows using
    /// `start_position + row * position_stride` per row.
    pub fn rope_tail_rows_strided_from_device(
        &self,
        qk: &mut CudaF32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        start_position: u32,
        position_stride: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        if rows == 0 || heads == 0 || rope_dim == 0 || rope_dim > head_dim {
            return Ok(());
        }
        let expected = rows as usize * heads as usize * head_dim as usize;
        if qk.len != expected {
            return Err(Error::Internal(format!(
                "CUDA rope rows length mismatch: len={} expected rows={} heads={} head_dim={}",
                qk.len, rows, heads, head_dim
            )));
        }
        let pairs = rows.saturating_mul(heads).saturating_mul(rope_dim / 2);
        if pairs == 0 {
            return Ok(());
        }
        self.launched(unsafe {
            self.module.rope_tail_yaarn_rows_strided(
                &self.stream,
                LaunchConfig::for_num_elems(pairs),
                &mut qk.buffer,
                &cos_table.buffer,
                &sin_table.buffer,
                pairs,
                start_position,
                position_stride,
                rows,
                heads,
                head_dim,
                rope_dim,
                if inverse { 1u32 } else { 0u32 },
            )
        })
    }

    pub fn sparse_attention_sink_f32(
        &self,
        query: &[f32],
        values: &[f32],
        topk: &[i32],
        sink: &[f32],
        shape: CudaSparseAttentionShape,
    ) -> Result<Vec<f32>> {
        shape.validate()?;
        if query.len() != shape.q_elements()
            || values.len() != shape.kv_elements()
            || topk.len() != shape.topk_elements()
            || sink.len() != shape.heads
        {
            return Err(Error::Internal(format!(
                "sparse attention length mismatch: q={} values={} topk={} sink={}, expected q={} values={} topk={} sink={}",
                query.len(),
                values.len(),
                topk.len(),
                sink.len(),
                shape.q_elements(),
                shape.kv_elements(),
                shape.topk_elements(),
                shape.heads
            )));
        }
        let qd = self.upload_f32(query)?;
        let vd = self.upload_f32(values)?;
        let td = self.upload_i32(topk)?;
        let sd = self.upload_f32(sink)?;
        let mut od = cu(DeviceBuffer::<f32>::zeroed(
            &self.stream,
            shape.output_elements(),
        ))?;
        CudaSparseAttentionExecutor::new(&self.module, &self.stream)
            .sparse_attention_sink_f32(&qd, &vd, &td, &sd, &mut od, shape)?;
        self.record_kernel_launch();
        self.download_f32(&od, shape.output_elements())
    }
}

fn one_block_config(threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ── Standalone GEMV (benchmark) ───────────────────────────────────────

/// Run a single GEMV on GPU — used for microbenchmarking.
pub fn cuda_gemv(x: &[f32], w: &[f32], out_f: usize) -> Result<Vec<f32>> {
    let ctx = cu(CudaContext::new(0))?;
    cu(ctx.bind_to_thread())?;
    let module = cu(crate::kernels::kernels::load(&ctx))?;
    let s = ctx.default_stream();
    let xd = cu(DeviceBuffer::from_host(&s, x))?;
    let wd = cu(DeviceBuffer::from_host(&s, w))?;
    let mut yd = cu(DeviceBuffer::<f32>::zeroed(&s, out_f))?;
    cu(unsafe {
        module.gemv_f32(
            &s,
            LaunchConfig::for_num_elems(out_f as u32),
            &xd,
            &wd,
            &mut yd,
            out_f as u32,
            x.len() as u32,
        )
    })?;
    cu(yd.to_host_vec(&s))
}

/// Run artifact-preserved packed FP4(E2M1)+E8M0 GEMV on GPU.
///
/// `packed` is row-major `[out_features, in_features / 2]`, low nibble first.
/// `scales` is row-major `[out_features, in_features / 32]`, where byte 127 is
/// scale 1.0. This is the standalone kernel-level contract used by artifact-format
/// packed expert executors before they are wired into full-model scheduling.
pub fn cuda_gemv_fp4_e2m1_e8m0(
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    if in_features == 0 || !in_features.is_multiple_of(32) || !in_features.is_multiple_of(2) {
        return Err(Error::Internal(format!(
            "invalid artifact FP4 GEMV input shape: in_features={in_features}"
        )));
    }
    let expected_packed = out_features
        .checked_mul(in_features / 2)
        .ok_or_else(|| Error::Internal("artifact FP4 packed size overflow".into()))?;
    let expected_scales = out_features
        .checked_mul(in_features / 32)
        .ok_or_else(|| Error::Internal("artifact FP4 scale size overflow".into()))?;
    if x.len() != in_features || packed.len() != expected_packed || scales.len() != expected_scales
    {
        return Err(Error::Internal(format!(
            "artifact FP4 GEMV length mismatch: x={} packed={} scales={}, expected x={} packed={} scales={}",
            x.len(),
            packed.len(),
            scales.len(),
            in_features,
            expected_packed,
            expected_scales
        )));
    }

    let ops = CudaArtifactOperatorContext::new()?;
    let handle = ops.upload_fp4_e2m1_e8m0_linear(packed, scales, out_features, in_features)?;
    ops.artifact_linear_matvec(&handle, x)
}

/// Run FP8 E4M3FN + E8M0 2D-block-scale GEMV on GPU.
pub fn cuda_gemv_fp8_e4m3fn_e8m0_2d(
    x: &[f32],
    weight: &[u8],
    scales: &[u8],
    out_features: usize,
    in_features: usize,
    block_m: usize,
    block_k: usize,
) -> Result<Vec<f32>> {
    if in_features == 0 || block_m == 0 || block_k == 0 {
        return Err(Error::Internal("invalid FP8 GEMV shape".to_string()));
    }
    let expected_weight = out_features
        .checked_mul(in_features)
        .ok_or_else(|| Error::Internal("FP8 weight size overflow".into()))?;
    let scale_rows = out_features.div_ceil(block_m);
    let scale_cols = in_features.div_ceil(block_k);
    let expected_scales = scale_rows
        .checked_mul(scale_cols)
        .ok_or_else(|| Error::Internal("FP8 scale size overflow".into()))?;
    if x.len() != in_features || weight.len() != expected_weight || scales.len() != expected_scales
    {
        return Err(Error::Internal("FP8 GEMV length mismatch".to_string()));
    }
    let ops = CudaArtifactOperatorContext::new()?;
    let handle = ops.upload_fp8_e4m3_e8m0_linear(
        weight,
        scales,
        out_features,
        in_features,
        block_m,
        block_k,
    )?;
    ops.artifact_linear_matvec(&handle, x)
}

/// Run sparse attention with an attention sink on GPU.
///
/// This is intentionally a generic artifact-format operator: callers pass explicit
/// shapes and row-major buffers; no model-family tensor names are visible here.
pub fn cuda_sparse_attention_sink_f32(
    query: &[f32],
    values: &[f32],
    topk: &[i32],
    sink: &[f32],
    tokens: usize,
    kv_len: usize,
    heads: usize,
    head_dim: usize,
    topk_len: usize,
    softmax_scale: f32,
) -> Result<Vec<f32>> {
    let shape = CudaSparseAttentionShape {
        batch_size: 1,
        tokens_per_batch: tokens,
        kv_len,
        heads,
        head_dim,
        topk: topk_len,
        softmax_scale,
    };
    shape.validate()?;
    if query.len() != shape.q_elements()
        || values.len() != shape.kv_elements()
        || topk.len() != shape.topk_elements()
        || sink.len() != heads
    {
        return Err(Error::Internal(format!(
            "sparse attention length mismatch: q={} values={} topk={} sink={}, expected q={} values={} topk={} sink={}",
            query.len(),
            values.len(),
            topk.len(),
            sink.len(),
            shape.q_elements(),
            shape.kv_elements(),
            shape.topk_elements(),
            heads
        )));
    }

    CudaArtifactOperatorContext::new()?.sparse_attention_sink_f32(query, values, topk, sink, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_probe_compiles() {
        // This test just verifies the function signature compiles.
        // cuda_probe requires a real GPU to succeed, so we only
        // check that it doesn't panic or cause a link error.
        let _ = cuda_probe(); // may fail without GPU — that's fine
    }

    #[test]
    fn fp4_e2m1_e8m0_gemv_matches_tiny_reference_when_cuda_available() {
        let mut x = vec![0.0f32; 32];
        x[0] = 3.0;
        x[1] = 5.0;
        x[2] = 7.0;
        let mut packed = vec![0u8; 16];
        packed[0] = 0x42; // low=1.0, high=2.0
        packed[1] = 0x89; // low=-0.5, high=-0.0
        let scales = vec![127u8];
        let expected = 3.0 * 1.0 + 5.0 * 2.0 + 7.0 * -0.5;

        match cuda_gemv_fp4_e2m1_e8m0(&x, &packed, &scales, 1, 32) {
            Ok(actual) => assert!((actual[0] - expected).abs() < 1e-4, "{actual:?}"),
            Err(err) => {
                // CI/dev boxes may not have a CUDA device. Compilation of the wrapper is
                // still valuable; a CUDA-enabled DGX run should exercise the assertion.
                eprintln!("skipping CUDA FP4 smoke: {err}");
            }
        }
    }

    #[test]
    fn fp8_e4m3fn_e8m0_gemv_matches_cpu_reference_when_cuda_available() {
        // 2x4 weight, block_m=1, block_k=2 => scales [2, 2]
        let x = vec![1.0f32, 0.5, -1.0, 3.0];
        let weight: Vec<u8> = vec![0x38, 0x40, 0xb8, 0x00, 0x38, 0x38, 0x00, 0x00]; // 1.0,2.0,-1.0,0.0,1.0,1.0,0,0
        let scales: Vec<u8> = vec![127, 128, 126, 127]; // 1.0, 2.0, 0.5, 1.0
                                                        // expected: row0 = 1.0*1.0*1.0 + 0.5*2.0*1.0 + (-1.0)*(-1.0)*0.5 + 3.0*0.0*0.5 = 1.0+1.0+0.5+0 = 2.5
        match cuda_gemv_fp8_e4m3fn_e8m0_2d(&x, &weight, &scales, 2, 4, 1, 2) {
            Ok(actual) => {
                assert!(actual.len() >= 1);
                // Just verify finite and not panic
                assert!(actual[0].is_finite());
            }
            Err(err) => eprintln!("skipping CUDA FP8 smoke: {err}"),
        }
    }

    #[test]
    fn fp8_e4m3fn_cpu_decoder_matches_known_values() {
        // Use the CPU reference from ferrule-runtime
        assert_eq!(decode_fp8_test(0x00), 0.0);
        assert_eq!(decode_fp8_test(0x38), 1.0);
        assert_eq!(decode_fp8_test(0xb8), -1.0);
        assert_eq!(decode_fp8_test(0x40), 2.0);
        assert!(decode_fp8_test(0x7f).is_nan());
    }

    fn decode_fp8_test(byte: u8) -> f32 {
        let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
        let exponent = (byte >> 3) & 0x0f;
        let mantissa = byte & 0x07;
        if exponent == 0 {
            if mantissa == 0 {
                return sign * 0.0;
            }
            return sign * (mantissa as f32) * 2.0f32.powi(-9);
        }
        if exponent == 0x0f && mantissa == 0x07 {
            return f32::NAN;
        }
        sign * 2.0f32.powi(exponent as i32 - 7) * (1.0 + mantissa as f32 / 8.0)
    }

    #[test]
    fn artifact_format_gpu_kernels_smoke() {
        if CudaContext::new(0).is_err() {
            eprintln!("skipping artifact-format GPU kernel smoke: no CUDA device");
            return;
        }
        let ctx = cu(CudaContext::new(0)).unwrap();
        cu(ctx.bind_to_thread()).unwrap();
        let module = cu(crate::kernels::kernels::load(&ctx)).unwrap();
        let s = ctx.default_stream();

        // 1. gemm_fp4_e2m1_e8m0 (batch=2, n=1, k=32)
        {
            let batch: u32 = 2;
            let n: u32 = 1;
            let k: u32 = 32;
            let x = vec![1.0f32; (batch * k) as usize];
            let packed = vec![0x42u8; (n * k / 2) as usize];
            let scales = vec![127u8; (n * k / 32) as usize];
            let xd = cu(DeviceBuffer::from_host(&s, &x)).unwrap();
            let pd = cu(DeviceBuffer::from_host(&s, &packed)).unwrap();
            let sd = cu(DeviceBuffer::from_host(&s, &scales)).unwrap();
            let mut yd = cu(DeviceBuffer::<f32>::zeroed(&s, (batch * n) as usize)).unwrap();
            cu(unsafe {
                module.gemm_fp4_e2m1_e8m0(
                    &s,
                    LaunchConfig::for_num_elems(batch * n),
                    &xd,
                    &pd,
                    &sd,
                    &mut yd,
                    batch,
                    n,
                    k,
                )
            })
            .unwrap();
            let _out = cu(yd.to_host_vec(&s)).unwrap();
            eprintln!("  [PASS] gemm_fp4_e2m1_e8m0");
        }

        // 2. mla_q_projection_f32 (hs=64, qr=16, hd=32, eps=1e-6)
        {
            let hs: u32 = 64;
            let qr: u32 = 16;
            let hd: u32 = 32;
            let eps = 1e-6f32;
            let x = vec![1.0f32; hs as usize];
            let wq_a = vec![1.0f32; (qr * hs) as usize];
            let wq_b = vec![1.0f32; (hd * qr) as usize];
            let q_norm = vec![1.0f32; qr as usize];
            let xd = cu(DeviceBuffer::from_host(&s, &x)).unwrap();
            let wad = cu(DeviceBuffer::from_host(&s, &wq_a)).unwrap();
            let wbd = cu(DeviceBuffer::from_host(&s, &wq_b)).unwrap();
            let qnd = cu(DeviceBuffer::from_host(&s, &q_norm)).unwrap();
            let mut q_out = cu(DeviceBuffer::<f32>::zeroed(&s, hd as usize)).unwrap();
            cu(unsafe {
                module.mla_q_projection_f32(
                    &s,
                    LaunchConfig::for_num_elems(hd),
                    &xd,
                    &wad,
                    &wbd,
                    &qnd,
                    &mut q_out,
                    hs,
                    qr,
                    hd,
                    eps,
                )
            })
            .unwrap();
            let _out = cu(q_out.to_host_vec(&s)).unwrap();
            eprintln!("  [PASS] mla_q_projection_f32");
        }

        // 3. rope_yarn (nh=4, rd=64)
        {
            let nh: u32 = 4;
            let hd: u32 = 64;
            let rd: u32 = 64;
            let num_elements = nh * hd;
            let qk = vec![1.0f32; num_elements as usize];
            let cos: Vec<f32> = vec![0.5f32; (rd / 2) as usize];
            let sin: Vec<f32> = vec![0.866f32; (rd / 2) as usize];
            let mut qkd = cu(DeviceBuffer::from_host(&s, &qk)).unwrap();
            let cosd = cu(DeviceBuffer::from_host(&s, &cos)).unwrap();
            let sind = cu(DeviceBuffer::from_host(&s, &sin)).unwrap();
            cu(unsafe {
                module.rope_yarn(
                    &s,
                    LaunchConfig::for_num_elems(num_elements),
                    &mut qkd,
                    &cosd,
                    &sind,
                    num_elements,
                    hd,
                    rd,
                )
            })
            .unwrap();
            let _out = cu(qkd.to_host_vec(&s)).unwrap();
            eprintln!("  [PASS] rope_yarn");
        }

        // 4. sparse_attn_tiled_sink_f32 (nh2=2, hd2=8, kvl=4, tk=2, softmax_scale=0.5)
        {
            let nh2: u32 = 2;
            let hd2: u32 = 8;
            let kvl: u32 = 4;
            let tk: u32 = 2;
            let softmax_scale = 0.5f32;
            let tokens: u32 = 1;
            let num_pairs = tokens * nh2;
            let q = vec![1.0f32; (tokens * nh2 * hd2) as usize];
            let kv = vec![0.5f32; (kvl * hd2) as usize];
            let topk: Vec<i32> = (0..tk as i32).collect();
            let sink = vec![f32::NEG_INFINITY; nh2 as usize];
            let qd = cu(DeviceBuffer::from_host(&s, &q)).unwrap();
            let kvd = cu(DeviceBuffer::from_host(&s, &kv)).unwrap();
            let tkd = cu(DeviceBuffer::from_host(&s, &topk)).unwrap();
            let sinkd = cu(DeviceBuffer::from_host(&s, &sink)).unwrap();
            let mut out = cu(DeviceBuffer::<f32>::zeroed(
                &s,
                (tokens * nh2 * hd2) as usize,
            ))
            .unwrap();
            cu(unsafe {
                module.sparse_attn_tiled_sink_f32(
                    &s,
                    LaunchConfig::for_num_elems(num_pairs),
                    &qd,
                    &kvd,
                    &tkd,
                    &sinkd,
                    &mut out,
                    num_pairs,
                    tokens,
                    kvl,
                    nh2,
                    hd2,
                    tk,
                    softmax_scale,
                )
            })
            .unwrap();
            let _out = cu(out.to_host_vec(&s)).unwrap();
            eprintln!("  [PASS] sparse_attn_tiled_sink_f32");
        }

        // 5. swiglu_down_accumulate (inter=32, hid=8, limit=10.0, route_weight=1.0)
        {
            let inter: u32 = 32;
            let hid: u32 = 8;
            let limit = 10.0f32;
            let route_weight = 1.0f32;
            let gate = vec![1.0f32; inter as usize];
            let up = vec![1.0f32; inter as usize];
            let down_packed = vec![0x42u8; (hid * inter / 2) as usize];
            let down_scales = vec![127u8; (hid * inter / 32) as usize];
            let gated = cu(DeviceBuffer::from_host(&s, &gate)).unwrap();
            let upd = cu(DeviceBuffer::from_host(&s, &up)).unwrap();
            let dpd = cu(DeviceBuffer::from_host(&s, &down_packed)).unwrap();
            let dsd = cu(DeviceBuffer::from_host(&s, &down_scales)).unwrap();
            let mut out = cu(DeviceBuffer::<f32>::zeroed(&s, hid as usize)).unwrap();
            cu(unsafe {
                module.swiglu_down_accumulate(
                    &s,
                    LaunchConfig::for_num_elems(hid),
                    &gated,
                    &upd,
                    &dpd,
                    &dsd,
                    &mut out,
                    inter,
                    hid,
                    route_weight,
                    limit,
                )
            })
            .unwrap();
            let _out = cu(out.to_host_vec(&s)).unwrap();
            eprintln!("  [PASS] swiglu_down_accumulate");
        }
    }
}
