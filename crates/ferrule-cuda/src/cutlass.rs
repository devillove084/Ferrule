//! GB10/SM121a semantic superkernel provider.
//!
//! Ferrule owns CUDA contexts, streams, allocations, execution plans, and
//! tensor lifetimes. The native boundary contains only versioned POD arguments
//! for complete semantic bundles; no C++ object or generic GEMM fallback
//! crosses the FFI.

#[cfg(feature = "cutlass")]
use cuda_core::{DeviceBuffer, stream::CudaStream};
use ferrule_common::{Error, Result};

pub const CUTLASS_ABI_VERSION: u32 = 5;
const PINNED_CUTLASS_VERSION: u32 = 461;
const GB10_SM: u32 = 121;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CutlassKernelId {
    Fp8QueryAKvSm121 = 1,
    Bf16CompressorSm121 = 2,
    HcProducerSm121 = 3,
    SharedFfnSm121 = 4,
    StableFrameFp4MoeSm121 = 5,
    MlaOutputSm121 = 6,
}

impl CutlassKernelId {
    pub const fn mask(self) -> u64 {
        1u64 << (self as u32 - 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassProviderManifest {
    pub abi_version: u32,
    pub cutlass_version: u32,
    pub target_sm: u32,
    pub kernel_count: u32,
    pub kernel_mask: u64,
}

impl CutlassProviderManifest {
    pub const fn supports(self, kernel: CutlassKernelId) -> bool {
        self.kernel_mask & kernel.mask() != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutlassProvider {
    manifest: CutlassProviderManifest,
}

impl CutlassProvider {
    pub const fn manifest(self) -> CutlassProviderManifest {
        self.manifest
    }

    pub const fn supports(self, kernel: CutlassKernelId) -> bool {
        self.manifest.supports(kernel)
    }

    pub fn execution_manifest(self) -> Result<ferrule_common::kernel_plan::ProviderManifest> {
        Ok(
            ferrule_common::kernel_plan::ProviderManifest::cutlass_cubin(
                u16::try_from(self.manifest.abi_version)
                    .map_err(|_| Error::Internal("CUTLASS ABI version exceeds u16".into()))?,
                self.manifest.kernel_count as usize,
            ),
        )
    }
}

/// Discover the required GB10 provider. Absence or any target/ABI mismatch is
/// fatal; Ferrule does not route production execution to another architecture.
pub fn discover_provider() -> Result<CutlassProvider> {
    #[cfg(not(feature = "cutlass"))]
    {
        Err(Error::Internal(
            "GB10 execution requires the `cutlass` feature and SM121a provider".into(),
        ))
    }
    #[cfg(feature = "cutlass")]
    {
        let manifest = unsafe { ffi::ferrule_cutlass_provider_manifest() };
        if manifest.abi_version != CUTLASS_ABI_VERSION {
            return Err(Error::Internal(format!(
                "CUTLASS ABI mismatch: native={} rust={CUTLASS_ABI_VERSION}",
                manifest.abi_version
            )));
        }
        if manifest.cutlass_version != PINNED_CUTLASS_VERSION {
            return Err(Error::Internal(format!(
                "CUTLASS version mismatch: native={} expected={PINNED_CUTLASS_VERSION}",
                manifest.cutlass_version
            )));
        }
        if manifest.target_sm != GB10_SM {
            return Err(Error::Internal(format!(
                "Ferrule requires the GB10 SM121a provider, got sm_{}",
                manifest.target_sm
            )));
        }
        let target = crate::architecture::CudaTarget::parse(crate::architecture::COMPILED_TARGET)
            .ok_or_else(|| {
            Error::Internal(format!(
                "invalid compiled CUDA target '{}'",
                crate::architecture::COMPILED_TARGET
            ))
        })?;
        if target.compute_capability() != GB10_SM || !target.has_accelerated_target() {
            return Err(Error::Internal(format!(
                "Ferrule GB10 provider requires sm_121a, compiled for '{}'",
                crate::architecture::COMPILED_TARGET
            )));
        }
        Ok(CutlassProvider { manifest })
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CutlassBf16CompressorArgs {
    abi_version: u32,
    rows: u32,
    n1: u32,
    n2: u32,
    k: u32,
    reserved0: u32,
    activation_f32: u64,
    projection1_weight_bf16: u64,
    projection2_weight_bf16: u64,
    projection1_output_f32: u64,
    projection2_output_f32: u64,
    stream: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CutlassFp8QueryAKvArgs {
    abi_version: u32,
    rows: u32,
    n1: u32,
    n2: u32,
    k: u32,
    scale_cols: u32,
    activation_fp8: u64,
    activation_ue8m0: u64,
    query_a_weight_fp8: u64,
    query_a_weight_ue8m0: u64,
    kv_weight_fp8: u64,
    kv_weight_ue8m0: u64,
    query_a_output_f32: u64,
    kv_output_f32: u64,
    stream: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CutlassHcProducerArgs {
    abi_version: u32,
    rows: u32,
    hc: u32,
    hidden: u32,
    mix: u32,
    sinkhorn_iters: u32,
    hc_eps: f32,
    hc_norm_eps: f32,
    layer_rms_eps: f32,
    reserved: u32,
    state_f32: u64,
    function_col_major_f32: u64,
    hc_scale_f32: u64,
    hc_base_f32: u64,
    layer_rms_weight_f32: u64,
    hidden_f32: u64,
    normalized_f32: u64,
    packed_e4m3: u64,
    scales_ue8m0: u64,
    split_pre_f32: u64,
    split_post_f32: u64,
    split_comb_f32: u64,
    stream: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CutlassSharedFfnArgs {
    abi_version: u32,
    input_fp8: u64,
    input_ue8m0: u64,
    gate_weight_fp8: u64,
    gate_weight_ue8m0: u64,
    up_weight_fp8: u64,
    up_weight_ue8m0: u64,
    down_weight_fp8: u64,
    down_weight_ue8m0: u64,
    hidden_f32: u64,
    hidden_fp8: u64,
    hidden_ue8m0: u64,
    output_f32: u64,
    rows: u32,
    input_size: u32,
    intermediate_size: u32,
    output_size: u32,
    gate_block_m: u32,
    gate_block_k: u32,
    up_block_m: u32,
    up_block_k: u32,
    down_block_m: u32,
    down_block_k: u32,
    output_scale: f32,
    swiglu_limit: f32,
    flags: u32,
    stream: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CutlassMlaOutputArgs {
    abi_version: u32,
    rows: u32,
    context_size: u32,
    groups: u32,
    group_input_size: u32,
    rank: u32,
    latent_size: u32,
    hidden_size: u32,
    output_a_scale_cols: u32,
    reserved0: u32,
    context_f32: u64,
    output_a_weight_fp8: u64,
    output_a_weight_ue8m0: u64,
    output_b_weight_fp8: u64,
    output_b_weight_ue8m0: u64,
    latent_f32: u64,
    latent_fp8: u64,
    latent_ue8m0: u64,
    output_f32: u64,
    stream: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CutlassStableFrameFp4MoeArgs {
    abi_version: u32,
    reserved0: u32,
    input_size: u32,
    intermediate_size: u32,
    hidden_size: u32,
    num_tokens: u32,
    num_routes: u32,
    slot_capacity: u32,
    num_segments: u32,
    swiglu_limit: f32,
    x_packed: u64,
    x_scales: u64,
    gate_ptrs: u64,
    gate_scale_ptrs: u64,
    up_ptrs: u64,
    up_scale_ptrs: u64,
    down_ptrs: u64,
    down_scale_ptrs: u64,
    slot_generations: u64,
    segment_expert_slots: u64,
    segment_generations: u64,
    segment_token_indices: u64,
    segment_route_indices: u64,
    segment_route_weights: u64,
    segment_states: u64,
    segment_bindings: u64,
    hidden_f32: u64,
    hidden_packed: u64,
    hidden_scales: u64,
    route_written: u64,
    route_error: u64,
    route_output: u64,
    stream: u64,
}

#[cfg(feature = "cutlass")]
impl CutlassBf16CompressorArgs {
    #[allow(clippy::too_many_arguments)]
    fn from_buffers(
        stream: &CudaStream,
        activation: &DeviceBuffer<f32>,
        projection1_weight: &DeviceBuffer<u8>,
        projection2_weight: &DeviceBuffer<u8>,
        projection1_output: &mut DeviceBuffer<f32>,
        projection2_output: &mut DeviceBuffer<f32>,
        rows: usize,
        n1: usize,
        n2: usize,
        k: usize,
    ) -> Result<Self> {
        validate_bf16_problem(
            activation,
            projection1_weight,
            projection2_weight,
            projection1_output,
            projection2_output,
            rows,
            n1,
            n2,
            k,
        )?;
        Ok(Self {
            abi_version: CUTLASS_ABI_VERSION,
            rows: checked_u32(rows, "rows")?,
            n1: checked_u32(n1, "n1")?,
            n2: checked_u32(n2, "n2")?,
            k: checked_u32(k, "k")?,
            reserved0: 0,
            activation_f32: activation.cu_deviceptr(),
            projection1_weight_bf16: projection1_weight.cu_deviceptr(),
            projection2_weight_bf16: projection2_weight.cu_deviceptr(),
            projection1_output_f32: projection1_output.cu_deviceptr(),
            projection2_output_f32: projection2_output.cu_deviceptr(),
            stream: stream.cu_stream() as usize as u64,
        })
    }
}

#[cfg(feature = "cutlass")]
impl CutlassFp8QueryAKvArgs {
    #[allow(clippy::too_many_arguments)]
    fn from_buffers(
        stream: &CudaStream,
        activation: &DeviceBuffer<u8>,
        activation_scales: &DeviceBuffer<u8>,
        query_a_weight: &DeviceBuffer<u8>,
        query_a_weight_scales: &DeviceBuffer<u8>,
        kv_weight: &DeviceBuffer<u8>,
        kv_weight_scales: &DeviceBuffer<u8>,
        query_a_output: &mut DeviceBuffer<f32>,
        kv_output: &mut DeviceBuffer<f32>,
        rows: usize,
        n1: usize,
        n2: usize,
        k: usize,
    ) -> Result<Self> {
        validate_fp8_problem(
            activation,
            activation_scales,
            query_a_weight,
            query_a_weight_scales,
            kv_weight,
            kv_weight_scales,
            query_a_output,
            kv_output,
            rows,
            n1,
            n2,
            k,
        )?;
        let scale_cols = k / 128;
        Ok(Self {
            abi_version: CUTLASS_ABI_VERSION,
            rows: checked_u32(rows, "rows")?,
            n1: checked_u32(n1, "n1")?,
            n2: checked_u32(n2, "n2")?,
            k: checked_u32(k, "k")?,
            scale_cols: checked_u32(scale_cols, "scale_cols")?,
            activation_fp8: activation.cu_deviceptr(),
            activation_ue8m0: activation_scales.cu_deviceptr(),
            query_a_weight_fp8: query_a_weight.cu_deviceptr(),
            query_a_weight_ue8m0: query_a_weight_scales.cu_deviceptr(),
            kv_weight_fp8: kv_weight.cu_deviceptr(),
            kv_weight_ue8m0: kv_weight_scales.cu_deviceptr(),
            query_a_output_f32: query_a_output.cu_deviceptr(),
            kv_output_f32: kv_output.cu_deviceptr(),
            stream: stream.cu_stream() as usize as u64,
        })
    }
}

#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
fn validate_bf16_problem(
    activation: &DeviceBuffer<f32>,
    projection1_weight: &DeviceBuffer<u8>,
    projection2_weight: &DeviceBuffer<u8>,
    projection1_output: &DeviceBuffer<f32>,
    projection2_output: &DeviceBuffer<f32>,
    rows: usize,
    n1: usize,
    n2: usize,
    k: usize,
) -> Result<()> {
    if rows == 0 || n1 == 0 || n2 == 0 || k == 0 || !k.is_multiple_of(16) {
        return Err(Error::Internal(format!(
            "invalid SM121 BF16 compressor shape: rows={rows} n1={n1} n2={n2} k={k}"
        )));
    }
    let required = [
        (
            "activation",
            activation.len(),
            checked_mul(rows, k, "BF16 activation")?,
        ),
        (
            "projection1 weight",
            projection1_weight.len(),
            checked_mul(
                checked_mul(n1, k, "BF16 projection1 weight")?,
                2,
                "BF16 projection1 bytes",
            )?,
        ),
        (
            "projection2 weight",
            projection2_weight.len(),
            checked_mul(
                checked_mul(n2, k, "BF16 projection2 weight")?,
                2,
                "BF16 projection2 bytes",
            )?,
        ),
        (
            "projection1 output",
            projection1_output.len(),
            checked_mul(rows, n1, "BF16 projection1 output")?,
        ),
        (
            "projection2 output",
            projection2_output.len(),
            checked_mul(rows, n2, "BF16 projection2 output")?,
        ),
    ];
    for (name, actual, expected) in required {
        if actual != expected {
            return Err(Error::Internal(format!(
                "SM121 BF16 compressor {name} length mismatch: actual={actual} expected={expected}"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
fn validate_fp8_problem(
    activation: &DeviceBuffer<u8>,
    activation_scales: &DeviceBuffer<u8>,
    query_a_weight: &DeviceBuffer<u8>,
    query_a_weight_scales: &DeviceBuffer<u8>,
    kv_weight: &DeviceBuffer<u8>,
    kv_weight_scales: &DeviceBuffer<u8>,
    query_a_output: &DeviceBuffer<f32>,
    kv_output: &DeviceBuffer<f32>,
    rows: usize,
    n1: usize,
    n2: usize,
    k: usize,
) -> Result<()> {
    if rows == 0 || n1 == 0 || n2 == 0 || k == 0 || !k.is_multiple_of(128) {
        return Err(Error::Internal(format!(
            "invalid SM121 FP8 QueryA+KV shape: rows={rows} n1={n1} n2={n2} k={k}"
        )));
    }
    let scale_cols = k / 128;
    let required = [
        (
            "activation",
            activation.len(),
            checked_mul(rows, k, "activation")?,
        ),
        (
            "activation scales",
            activation_scales.len(),
            checked_mul(rows, scale_cols, "activation scales")?,
        ),
        (
            "QueryA weight",
            query_a_weight.len(),
            checked_mul(n1, k, "QueryA weight")?,
        ),
        (
            "QueryA weight scales",
            query_a_weight_scales.len(),
            checked_mul(n1.div_ceil(128), scale_cols, "QueryA weight scales")?,
        ),
        (
            "KV weight",
            kv_weight.len(),
            checked_mul(n2, k, "KV weight")?,
        ),
        (
            "KV weight scales",
            kv_weight_scales.len(),
            checked_mul(n2.div_ceil(128), scale_cols, "KV weight scales")?,
        ),
        (
            "QueryA output",
            query_a_output.len(),
            checked_mul(rows, n1, "QueryA output")?,
        ),
        (
            "KV output",
            kv_output.len(),
            checked_mul(rows, n2, "KV output")?,
        ),
    ];
    for (name, actual, expected) in required {
        if actual != expected {
            return Err(Error::Internal(format!(
                "SM121 FP8 QueryA+KV {name} length mismatch: actual={actual} expected={expected}"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "cutlass")]
fn checked_mul(lhs: usize, rhs: usize, name: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::Internal(format!("SM121 FP8 {name} size overflow")))
}

#[cfg(feature = "cutlass")]
fn checked_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Internal(format!("SM121 FP8 {name} exceeds u32")))
}

/// Launch one semantic BF16 compressor bundle. The native GB10 provider owns
/// small-M versus tiled schedule selection.
#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
pub fn bf16_compressor(
    stream: &CudaStream,
    activation: &DeviceBuffer<f32>,
    projection1_weight: &DeviceBuffer<u8>,
    projection2_weight: &DeviceBuffer<u8>,
    projection1_output: &mut DeviceBuffer<f32>,
    projection2_output: &mut DeviceBuffer<f32>,
    rows: usize,
    n1: usize,
    n2: usize,
    k: usize,
) -> Result<()> {
    let args = CutlassBf16CompressorArgs::from_buffers(
        stream,
        activation,
        projection1_weight,
        projection2_weight,
        projection1_output,
        projection2_output,
        rows,
        n1,
        n2,
        k,
    )?;
    let can_implement = unsafe { ffi::ferrule_cutlass_bf16_compressor_can_implement(&args) };
    if can_implement != status::SUCCESS {
        return Err(native_error(
            "validate SM121 BF16 compressor",
            can_implement,
        ));
    }
    let status = unsafe { ffi::ferrule_cutlass_bf16_compressor_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch SM121 BF16 compressor", status))
    }
}

/// Launch one semantic FP8 QueryA+KV bundle. The executable plan does not bind
/// an M bucket or expose a native schedule variant.
#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
pub fn fp8_query_a_kv(
    stream: &CudaStream,
    activation: &DeviceBuffer<u8>,
    activation_scales: &DeviceBuffer<u8>,
    query_a_weight: &DeviceBuffer<u8>,
    query_a_weight_scales: &DeviceBuffer<u8>,
    kv_weight: &DeviceBuffer<u8>,
    kv_weight_scales: &DeviceBuffer<u8>,
    query_a_output: &mut DeviceBuffer<f32>,
    kv_output: &mut DeviceBuffer<f32>,
    rows: usize,
    n1: usize,
    n2: usize,
    k: usize,
) -> Result<()> {
    let args = CutlassFp8QueryAKvArgs::from_buffers(
        stream,
        activation,
        activation_scales,
        query_a_weight,
        query_a_weight_scales,
        kv_weight,
        kv_weight_scales,
        query_a_output,
        kv_output,
        rows,
        n1,
        n2,
        k,
    )?;
    let can_implement = unsafe { ffi::ferrule_cutlass_fp8_query_a_kv_can_implement(&args) };
    if can_implement != status::SUCCESS {
        return Err(native_error("validate SM121 FP8 QueryA+KV", can_implement));
    }
    let status = unsafe { ffi::ferrule_cutlass_fp8_query_a_kv_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch SM121 FP8 QueryA+KV", status))
    }
}

/// Launch the complete HC-pre + layer RMSNorm + FP8 producer bundle.
#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
pub fn hc_producer(
    stream: &CudaStream,
    state: &DeviceBuffer<f32>,
    function_col_major: &DeviceBuffer<f32>,
    hc_scale: &DeviceBuffer<f32>,
    hc_base: &DeviceBuffer<f32>,
    layer_rms_weight: &DeviceBuffer<f32>,
    hidden_output: &mut DeviceBuffer<f32>,
    normalized_output: &mut DeviceBuffer<f32>,
    packed_output: &mut DeviceBuffer<u8>,
    scale_output: &mut DeviceBuffer<u8>,
    split_pre: &mut DeviceBuffer<f32>,
    split_post: &mut DeviceBuffer<f32>,
    split_comb: &mut DeviceBuffer<f32>,
    rows: usize,
    hc: usize,
    hidden: usize,
    sinkhorn_iters: usize,
    hc_eps: f32,
    hc_norm_eps: f32,
    layer_rms_eps: f32,
) -> Result<()> {
    let mix = checked_mul(hc, hc + 2, "HC mix")?;
    let hc_hidden = checked_mul(hc, hidden, "HC state width")?;
    let scale_cols = hidden.div_ceil(128);
    let required = [
        (
            "state",
            state.len(),
            checked_mul(rows, hc_hidden, "HC state")?,
        ),
        (
            "function",
            function_col_major.len(),
            checked_mul(hc_hidden, mix, "HC function")?,
        ),
        ("HC scale", hc_scale.len(), 3),
        ("HC base", hc_base.len(), mix),
        ("layer RMS weight", layer_rms_weight.len(), hidden),
        (
            "hidden output",
            hidden_output.len(),
            checked_mul(rows, hidden, "HC hidden output")?,
        ),
        (
            "normalized output",
            normalized_output.len(),
            checked_mul(rows, hidden, "HC normalized output")?,
        ),
        (
            "packed output",
            packed_output.len(),
            checked_mul(rows, hidden, "HC packed output")?,
        ),
        (
            "scale output",
            scale_output.len(),
            checked_mul(rows, scale_cols, "HC scale output")?,
        ),
        (
            "split pre",
            split_pre.len(),
            checked_mul(rows, hc, "HC split pre")?,
        ),
        (
            "split post",
            split_post.len(),
            checked_mul(rows, hc, "HC split post")?,
        ),
        (
            "split comb",
            split_comb.len(),
            checked_mul(
                checked_mul(rows, hc, "HC split comb rows")?,
                hc,
                "HC split comb",
            )?,
        ),
    ];
    if rows == 0
        || sinkhorn_iters == 0
        || !hc_eps.is_finite()
        || !hc_norm_eps.is_finite()
        || !layer_rms_eps.is_finite()
    {
        return Err(Error::Internal(format!(
            "invalid SM121 HC producer parameters: rows={rows} hc={hc} hidden={hidden} sinkhorn={sinkhorn_iters}"
        )));
    }
    validate_lengths("SM121 HC producer", &required)?;
    let args = CutlassHcProducerArgs {
        abi_version: CUTLASS_ABI_VERSION,
        rows: checked_u32(rows, "rows")?,
        hc: checked_u32(hc, "hc")?,
        hidden: checked_u32(hidden, "hidden")?,
        mix: checked_u32(mix, "mix")?,
        sinkhorn_iters: checked_u32(sinkhorn_iters, "sinkhorn_iters")?,
        hc_eps,
        hc_norm_eps,
        layer_rms_eps,
        reserved: 0,
        state_f32: state.cu_deviceptr(),
        function_col_major_f32: function_col_major.cu_deviceptr(),
        hc_scale_f32: hc_scale.cu_deviceptr(),
        hc_base_f32: hc_base.cu_deviceptr(),
        layer_rms_weight_f32: layer_rms_weight.cu_deviceptr(),
        hidden_f32: hidden_output.cu_deviceptr(),
        normalized_f32: normalized_output.cu_deviceptr(),
        packed_e4m3: packed_output.cu_deviceptr(),
        scales_ue8m0: scale_output.cu_deviceptr(),
        split_pre_f32: split_pre.cu_deviceptr(),
        split_post_f32: split_post.cu_deviceptr(),
        split_comb_f32: split_comb.cu_deviceptr(),
        stream: stream.cu_stream() as usize as u64,
    };
    let can_implement = unsafe { ffi::ferrule_cutlass_hc_producer_can_implement(&args) };
    if can_implement != status::SUCCESS {
        return Err(native_error("validate SM121 HC producer", can_implement));
    }
    let status = unsafe { ffi::ferrule_cutlass_hc_producer_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch SM121 HC producer", status))
    }
}

/// Launch the complete shared gate/up -> SwiGLU -> down bundle.
#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
pub fn shared_ffn(
    stream: &CudaStream,
    input_fp8: &DeviceBuffer<u8>,
    input_scales: &DeviceBuffer<u8>,
    gate_weight: &DeviceBuffer<u8>,
    gate_scales: &DeviceBuffer<u8>,
    up_weight: &DeviceBuffer<u8>,
    up_scales: &DeviceBuffer<u8>,
    down_weight: &DeviceBuffer<u8>,
    down_scales: &DeviceBuffer<u8>,
    hidden_f32: &mut DeviceBuffer<f32>,
    hidden_fp8: &mut DeviceBuffer<u8>,
    hidden_scales: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    input_size: usize,
    intermediate_size: usize,
    output_size: usize,
    gate_blocks: (usize, usize),
    up_blocks: (usize, usize),
    down_blocks: (usize, usize),
    output_scale: f32,
    swiglu_limit: f32,
    accumulate_output: bool,
) -> Result<()> {
    let input_scale_cols = input_size.div_ceil(128);
    let hidden_scale_cols = intermediate_size.div_ceil(128);
    let required = [
        (
            "input FP8",
            input_fp8.len(),
            checked_mul(rows, input_size, "shared FFN input")?,
        ),
        (
            "input scales",
            input_scales.len(),
            checked_mul(rows, input_scale_cols, "shared FFN input scales")?,
        ),
        (
            "gate weight",
            gate_weight.len(),
            checked_mul(intermediate_size, input_size, "shared FFN gate")?,
        ),
        (
            "gate scales",
            gate_scales.len(),
            checked_mul(
                intermediate_size.div_ceil(128),
                input_scale_cols,
                "shared FFN gate scales",
            )?,
        ),
        (
            "up weight",
            up_weight.len(),
            checked_mul(intermediate_size, input_size, "shared FFN up")?,
        ),
        (
            "up scales",
            up_scales.len(),
            checked_mul(
                intermediate_size.div_ceil(128),
                input_scale_cols,
                "shared FFN up scales",
            )?,
        ),
        (
            "down weight",
            down_weight.len(),
            checked_mul(output_size, intermediate_size, "shared FFN down")?,
        ),
        (
            "down scales",
            down_scales.len(),
            checked_mul(
                output_size.div_ceil(128),
                hidden_scale_cols,
                "shared FFN down scales",
            )?,
        ),
        (
            "hidden FP8",
            hidden_fp8.len(),
            checked_mul(rows, intermediate_size, "shared FFN hidden")?,
        ),
        (
            "hidden scales",
            hidden_scales.len(),
            checked_mul(rows, hidden_scale_cols, "shared FFN hidden scales")?,
        ),
        (
            "output",
            output.len(),
            checked_mul(rows, output_size, "shared FFN output")?,
        ),
    ];
    if rows == 0 || !output_scale.is_finite() || !swiglu_limit.is_finite() {
        return Err(Error::Internal(format!(
            "invalid SM121 shared FFN parameters: rows={rows} output_scale={output_scale} swiglu_limit={swiglu_limit}"
        )));
    }
    validate_lengths("SM121 shared FFN", &required)?;
    let hidden_values = checked_mul(rows, intermediate_size, "shared FFN hidden F32")?;
    if hidden_f32.len() < hidden_values {
        return Err(Error::Internal(format!(
            "SM121 shared FFN hidden F32 capacity is too small: actual={} required={hidden_values}",
            hidden_f32.len()
        )));
    }
    let args = CutlassSharedFfnArgs {
        abi_version: CUTLASS_ABI_VERSION,
        input_fp8: input_fp8.cu_deviceptr(),
        input_ue8m0: input_scales.cu_deviceptr(),
        gate_weight_fp8: gate_weight.cu_deviceptr(),
        gate_weight_ue8m0: gate_scales.cu_deviceptr(),
        up_weight_fp8: up_weight.cu_deviceptr(),
        up_weight_ue8m0: up_scales.cu_deviceptr(),
        down_weight_fp8: down_weight.cu_deviceptr(),
        down_weight_ue8m0: down_scales.cu_deviceptr(),
        hidden_f32: hidden_f32.cu_deviceptr(),
        hidden_fp8: hidden_fp8.cu_deviceptr(),
        hidden_ue8m0: hidden_scales.cu_deviceptr(),
        output_f32: output.cu_deviceptr(),
        rows: checked_u32(rows, "rows")?,
        input_size: checked_u32(input_size, "input_size")?,
        intermediate_size: checked_u32(intermediate_size, "intermediate_size")?,
        output_size: checked_u32(output_size, "output_size")?,
        gate_block_m: checked_u32(gate_blocks.0, "gate_block_m")?,
        gate_block_k: checked_u32(gate_blocks.1, "gate_block_k")?,
        up_block_m: checked_u32(up_blocks.0, "up_block_m")?,
        up_block_k: checked_u32(up_blocks.1, "up_block_k")?,
        down_block_m: checked_u32(down_blocks.0, "down_block_m")?,
        down_block_k: checked_u32(down_blocks.1, "down_block_k")?,
        output_scale,
        swiglu_limit,
        flags: u32::from(accumulate_output),
        stream: stream.cu_stream() as usize as u64,
    };
    let can_implement = unsafe { ffi::ferrule_cutlass_shared_ffn_can_implement(&args) };
    if can_implement != status::SUCCESS {
        return Err(native_error("validate SM121 shared FFN", can_implement));
    }
    let status = unsafe { ffi::ferrule_cutlass_shared_ffn_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch SM121 shared FFN", status))
    }
}

/// Launch grouped output-A -> BF16 boundary -> FP8 pack -> output-B as one MLA bundle.
#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
pub fn mla_output(
    stream: &CudaStream,
    context: &DeviceBuffer<f32>,
    output_a_weight: &DeviceBuffer<u8>,
    output_a_scales: &DeviceBuffer<u8>,
    output_b_weight: &DeviceBuffer<u8>,
    output_b_scales: &DeviceBuffer<u8>,
    latent: &mut DeviceBuffer<f32>,
    latent_fp8: &mut DeviceBuffer<u8>,
    latent_scales: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    context_size: usize,
    groups: usize,
    group_input_size: usize,
    rank: usize,
    latent_size: usize,
    hidden_size: usize,
) -> Result<()> {
    let scale_cols = group_input_size / 128;
    let required = [
        (
            "context",
            context.len(),
            checked_mul(rows, context_size, "MLA output context")?,
        ),
        (
            "output-A weight",
            output_a_weight.len(),
            checked_mul(latent_size, group_input_size, "MLA output-A weight")?,
        ),
        (
            "output-A scales",
            output_a_scales.len(),
            checked_mul(latent_size.div_ceil(128), scale_cols, "MLA output-A scales")?,
        ),
        (
            "output-B FP8 weight",
            output_b_weight.len(),
            checked_mul(hidden_size, latent_size, "MLA output-B weight")?,
        ),
        (
            "output-B scales",
            output_b_scales.len(),
            checked_mul(
                hidden_size.div_ceil(128),
                latent_size / 128,
                "MLA output-B scales",
            )?,
        ),
        (
            "latent",
            latent.len(),
            checked_mul(rows, latent_size, "MLA output latent")?,
        ),
        (
            "output",
            output.len(),
            checked_mul(rows, hidden_size, "MLA output")?,
        ),
    ];
    if rows == 0
        || groups == 0
        || group_input_size == 0
        || rank == 0
        || context_size != groups * group_input_size
        || latent_size != groups * rank
        || !group_input_size.is_multiple_of(128)
        || !rank.is_multiple_of(16)
        || !latent_size.is_multiple_of(128)
    {
        return Err(Error::Internal(format!(
            "invalid SM121 MLA output shape: rows={rows} context={context_size} groups={groups} group_input={group_input_size} rank={rank} latent={latent_size} hidden={hidden_size}"
        )));
    }
    validate_lengths("SM121 MLA output", &required)?;
    let latent_values = checked_mul(rows, latent_size, "MLA output latent FP8")?;
    let latent_scale_values = checked_mul(rows, latent_size / 128, "MLA output latent scales")?;
    if latent_fp8.len() < latent_values || latent_scales.len() < latent_scale_values {
        return Err(Error::Internal(format!(
            "SM121 MLA output scratch is too small: latent_fp8={}/{} latent_scales={}/{}",
            latent_fp8.len(),
            latent_values,
            latent_scales.len(),
            latent_scale_values
        )));
    }
    let args = CutlassMlaOutputArgs {
        abi_version: CUTLASS_ABI_VERSION,
        rows: checked_u32(rows, "rows")?,
        context_size: checked_u32(context_size, "context_size")?,
        groups: checked_u32(groups, "groups")?,
        group_input_size: checked_u32(group_input_size, "group_input_size")?,
        rank: checked_u32(rank, "rank")?,
        latent_size: checked_u32(latent_size, "latent_size")?,
        hidden_size: checked_u32(hidden_size, "hidden_size")?,
        output_a_scale_cols: checked_u32(scale_cols, "output_a_scale_cols")?,
        reserved0: 0,
        context_f32: context.cu_deviceptr(),
        output_a_weight_fp8: output_a_weight.cu_deviceptr(),
        output_a_weight_ue8m0: output_a_scales.cu_deviceptr(),
        output_b_weight_fp8: output_b_weight.cu_deviceptr(),
        output_b_weight_ue8m0: output_b_scales.cu_deviceptr(),
        latent_f32: latent.cu_deviceptr(),
        latent_fp8: latent_fp8.cu_deviceptr(),
        latent_ue8m0: latent_scales.cu_deviceptr(),
        output_f32: output.cu_deviceptr(),
        stream: stream.cu_stream() as usize as u64,
    };
    let can_implement = unsafe { ffi::ferrule_cutlass_mla_output_can_implement(&args) };
    if can_implement != status::SUCCESS {
        return Err(native_error("validate SM121 MLA output", can_implement));
    }
    let status = unsafe { ffi::ferrule_cutlass_mla_output_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch SM121 MLA output", status))
    }
}

/// Launch the complete stable-frame routed MXFP4 expert bundle.
#[cfg(feature = "cutlass")]
#[allow(clippy::too_many_arguments)]
pub fn stable_frame_fp4_moe(
    stream: &CudaStream,
    x_packed: &DeviceBuffer<u8>,
    x_scales: &DeviceBuffer<u8>,
    gate_ptrs: &DeviceBuffer<u64>,
    gate_scale_ptrs: &DeviceBuffer<u64>,
    up_ptrs: &DeviceBuffer<u64>,
    up_scale_ptrs: &DeviceBuffer<u64>,
    down_ptrs: &DeviceBuffer<u64>,
    down_scale_ptrs: &DeviceBuffer<u64>,
    slot_generations: &DeviceBuffer<i32>,
    segment_expert_slots: &DeviceBuffer<i32>,
    segment_generations: &DeviceBuffer<i32>,
    segment_token_indices: &DeviceBuffer<i32>,
    segment_route_indices: &DeviceBuffer<i32>,
    segment_route_weights: &DeviceBuffer<f32>,
    segment_states: &mut DeviceBuffer<i32>,
    segment_bindings: &mut DeviceBuffer<u64>,
    hidden_f32: &mut DeviceBuffer<f32>,
    hidden_packed: &mut DeviceBuffer<u8>,
    hidden_scales: &mut DeviceBuffer<u8>,
    route_written: &mut DeviceBuffer<i32>,
    route_error: &mut DeviceBuffer<i32>,
    route_output: &mut DeviceBuffer<f32>,
    input_size: usize,
    intermediate_size: usize,
    hidden_size: usize,
    num_tokens: usize,
    num_routes: usize,
    slot_capacity: usize,
    num_segments: usize,
    swiglu_limit: f32,
) -> Result<()> {
    let segment_columns = checked_mul(num_segments, 8, "FP4 MoE segment columns")?;
    let required = [
        (
            "packed input",
            x_packed.len(),
            checked_mul(num_tokens, input_size / 2, "FP4 MoE input")?,
        ),
        (
            "input scales",
            x_scales.len(),
            checked_mul(num_tokens, input_size / 32, "FP4 MoE input scales")?,
        ),
        ("gate pointers", gate_ptrs.len(), slot_capacity),
        ("gate scale pointers", gate_scale_ptrs.len(), slot_capacity),
        ("up pointers", up_ptrs.len(), slot_capacity),
        ("up scale pointers", up_scale_ptrs.len(), slot_capacity),
        ("down pointers", down_ptrs.len(), slot_capacity),
        ("down scale pointers", down_scale_ptrs.len(), slot_capacity),
        ("slot generations", slot_generations.len(), slot_capacity),
        ("segment slots", segment_expert_slots.len(), num_segments),
        (
            "segment generations",
            segment_generations.len(),
            num_segments,
        ),
        (
            "segment token indices",
            segment_token_indices.len(),
            segment_columns,
        ),
        (
            "segment route indices",
            segment_route_indices.len(),
            segment_columns,
        ),
        (
            "segment route weights",
            segment_route_weights.len(),
            segment_columns,
        ),
        ("segment states", segment_states.len(), num_segments),
        (
            "segment bindings",
            segment_bindings.len(),
            checked_mul(num_segments, 6, "FP4 MoE segment bindings")?,
        ),
        (
            "hidden F32 scratch",
            hidden_f32.len(),
            checked_mul(segment_columns, intermediate_size, "FP4 MoE hidden F32")?,
        ),
        (
            "hidden packed scratch",
            hidden_packed.len(),
            checked_mul(
                segment_columns,
                intermediate_size / 2,
                "FP4 MoE hidden packed",
            )?,
        ),
        (
            "hidden scale scratch",
            hidden_scales.len(),
            checked_mul(
                segment_columns,
                intermediate_size / 32,
                "FP4 MoE hidden scales",
            )?,
        ),
        ("route error", route_error.len(), 1),
        (
            "route output",
            route_output.len(),
            checked_mul(num_routes, hidden_size, "FP4 MoE route output")?,
        ),
    ];
    if input_size == 0
        || intermediate_size == 0
        || hidden_size == 0
        || num_tokens == 0
        || num_routes == 0
        || slot_capacity == 0
        || num_segments == 0
        || !swiglu_limit.is_finite()
    {
        return Err(Error::Internal(
            "invalid SM121 stable-frame FP4 MoE parameters".into(),
        ));
    }
    validate_lengths("SM121 stable-frame FP4 MoE", &required)?;
    if route_written.len() < num_routes {
        return Err(Error::Internal(format!(
            "SM121 stable-frame FP4 MoE route written capacity is too small: actual={} required={num_routes}",
            route_written.len()
        )));
    }
    let args = CutlassStableFrameFp4MoeArgs {
        abi_version: CUTLASS_ABI_VERSION,
        reserved0: 0,
        input_size: checked_u32(input_size, "input_size")?,
        intermediate_size: checked_u32(intermediate_size, "intermediate_size")?,
        hidden_size: checked_u32(hidden_size, "hidden_size")?,
        num_tokens: checked_u32(num_tokens, "num_tokens")?,
        num_routes: checked_u32(num_routes, "num_routes")?,
        slot_capacity: checked_u32(slot_capacity, "slot_capacity")?,
        num_segments: checked_u32(num_segments, "num_segments")?,
        swiglu_limit,
        x_packed: x_packed.cu_deviceptr(),
        x_scales: x_scales.cu_deviceptr(),
        gate_ptrs: gate_ptrs.cu_deviceptr(),
        gate_scale_ptrs: gate_scale_ptrs.cu_deviceptr(),
        up_ptrs: up_ptrs.cu_deviceptr(),
        up_scale_ptrs: up_scale_ptrs.cu_deviceptr(),
        down_ptrs: down_ptrs.cu_deviceptr(),
        down_scale_ptrs: down_scale_ptrs.cu_deviceptr(),
        slot_generations: slot_generations.cu_deviceptr(),
        segment_expert_slots: segment_expert_slots.cu_deviceptr(),
        segment_generations: segment_generations.cu_deviceptr(),
        segment_token_indices: segment_token_indices.cu_deviceptr(),
        segment_route_indices: segment_route_indices.cu_deviceptr(),
        segment_route_weights: segment_route_weights.cu_deviceptr(),
        segment_states: segment_states.cu_deviceptr(),
        segment_bindings: segment_bindings.cu_deviceptr(),
        hidden_f32: hidden_f32.cu_deviceptr(),
        hidden_packed: hidden_packed.cu_deviceptr(),
        hidden_scales: hidden_scales.cu_deviceptr(),
        route_written: route_written.cu_deviceptr(),
        route_error: route_error.cu_deviceptr(),
        route_output: route_output.cu_deviceptr(),
        stream: stream.cu_stream() as usize as u64,
    };
    let can_implement = unsafe { ffi::ferrule_cutlass_stable_frame_fp4_moe_can_implement(&args) };
    if can_implement != status::SUCCESS {
        return Err(native_error(
            "validate SM121 stable-frame FP4 MoE",
            can_implement,
        ));
    }
    let status = unsafe { ffi::ferrule_cutlass_stable_frame_fp4_moe_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch SM121 stable-frame FP4 MoE", status))
    }
}

#[cfg(feature = "cutlass")]
fn validate_lengths(scope: &str, required: &[(&str, usize, usize)]) -> Result<()> {
    for &(name, actual, expected) in required {
        if actual != expected {
            return Err(Error::Internal(format!(
                "{scope} {name} length mismatch: actual={actual} expected={expected}"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "cutlass")]
fn native_error(operation: &str, code: i32) -> Error {
    let reason = match code {
        status::INVALID_ABI => "ABI mismatch",
        status::INVALID_ARGUMENT => "invalid or unsupported problem",
        status::LAUNCH_FAILED => "kernel launch failed",
        _ => "unknown native status",
    };
    Error::Internal(format!("CUTLASS {operation} failed: {reason} ({code})"))
}

#[cfg(feature = "cutlass")]
mod status {
    pub const SUCCESS: i32 = 0;
    pub const INVALID_ABI: i32 = 1;
    pub const INVALID_ARGUMENT: i32 = 2;
    pub const LAUNCH_FAILED: i32 = 3;
}

#[cfg(feature = "cutlass")]
mod ffi {
    use super::{
        CutlassBf16CompressorArgs, CutlassFp8QueryAKvArgs, CutlassHcProducerArgs,
        CutlassMlaOutputArgs, CutlassProviderManifest, CutlassSharedFfnArgs,
        CutlassStableFrameFp4MoeArgs,
    };

    unsafe extern "C" {
        pub fn ferrule_cutlass_provider_manifest() -> CutlassProviderManifest;
        pub fn ferrule_cutlass_bf16_compressor_can_implement(
            args: *const CutlassBf16CompressorArgs,
        ) -> i32;
        pub fn ferrule_cutlass_bf16_compressor_launch(
            args: *const CutlassBf16CompressorArgs,
        ) -> i32;
        pub fn ferrule_cutlass_fp8_query_a_kv_can_implement(
            args: *const CutlassFp8QueryAKvArgs,
        ) -> i32;
        pub fn ferrule_cutlass_fp8_query_a_kv_launch(args: *const CutlassFp8QueryAKvArgs) -> i32;
        pub fn ferrule_cutlass_hc_producer_can_implement(args: *const CutlassHcProducerArgs)
        -> i32;
        pub fn ferrule_cutlass_hc_producer_launch(args: *const CutlassHcProducerArgs) -> i32;
        pub fn ferrule_cutlass_shared_ffn_can_implement(args: *const CutlassSharedFfnArgs) -> i32;
        pub fn ferrule_cutlass_shared_ffn_launch(args: *const CutlassSharedFfnArgs) -> i32;
        pub fn ferrule_cutlass_mla_output_can_implement(args: *const CutlassMlaOutputArgs) -> i32;
        pub fn ferrule_cutlass_mla_output_launch(args: *const CutlassMlaOutputArgs) -> i32;
        pub fn ferrule_cutlass_stable_frame_fp4_moe_can_implement(
            args: *const CutlassStableFrameFp4MoeArgs,
        ) -> i32;
        pub fn ferrule_cutlass_stable_frame_fp4_moe_launch(
            args: *const CutlassStableFrameFp4MoeArgs,
        ) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pod_layout_matches_native_contract() {
        assert_eq!(std::mem::size_of::<CutlassProviderManifest>(), 24);
        assert_eq!(std::mem::size_of::<CutlassFp8QueryAKvArgs>(), 96);
        assert_eq!(std::mem::size_of::<CutlassBf16CompressorArgs>(), 72);
        assert_eq!(std::mem::size_of::<CutlassHcProducerArgs>(), 144);
        assert_eq!(std::mem::size_of::<CutlassSharedFfnArgs>(), 168);
        assert_eq!(std::mem::size_of::<CutlassMlaOutputArgs>(), 120);
        assert_eq!(std::mem::size_of::<CutlassStableFrameFp4MoeArgs>(), 224);
    }

    #[cfg(feature = "cutlass")]
    #[test]
    fn native_provider_is_exactly_the_sm121_bundle_catalog() {
        let provider = discover_provider().expect("SM121 provider");
        let manifest = provider.manifest();
        assert_eq!(manifest.abi_version, CUTLASS_ABI_VERSION);
        assert_eq!(manifest.cutlass_version, PINNED_CUTLASS_VERSION);
        assert_eq!(manifest.target_sm, GB10_SM);
        assert_eq!(manifest.kernel_count, 6);
        assert!(manifest.supports(CutlassKernelId::Fp8QueryAKvSm121));
        assert!(manifest.supports(CutlassKernelId::Bf16CompressorSm121));
        assert!(manifest.supports(CutlassKernelId::HcProducerSm121));
        assert!(manifest.supports(CutlassKernelId::SharedFfnSm121));
        assert!(manifest.supports(CutlassKernelId::StableFrameFp4MoeSm121));
        assert!(manifest.supports(CutlassKernelId::MlaOutputSm121));
    }
}
