//! Source-compatible raw entry points, not the production owner dispatch path.
//! Keep signature and const compatibility while native lowering stays private.
use super::{CudaStream, CutlassKernelId, DeviceBuffer, NumericFp8Precision, Result, bf16_linear};

/// Temporary source-compatibility bridge. This inherent const method cannot be
/// hidden from the semantic type without breaking old callers (including const
/// expressions). Only this legacy impl mentions the vendor capability; operator
/// contracts and production dispatch must not call it.
impl NumericFp8Precision {
    #[deprecated(note = "legacy provider bridge; submit a semantic numeric FP8 plan instead")]
    pub const fn kernel(self) -> CutlassKernelId {
        bf16_linear::kernel_for_precision(self)
    }
}

/// Transitional extension from the previous migration slice, retained so its
/// imports and UFCS calls also compile. The deprecated inherent const bridge
/// above restores the original source contract.
#[deprecated(note = "provider introspection only; submit a semantic numeric FP8 plan instead")]
pub trait NumericFp8PrecisionExt {
    fn kernel(self) -> CutlassKernelId;
}

#[allow(deprecated)]
impl NumericFp8PrecisionExt for NumericFp8Precision {
    fn kernel(self) -> CutlassKernelId {
        bf16_linear::kernel_for_precision(self)
    }
}

/// Compatibility-only raw launch; prefer the semantic owner method.
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
    super::submit_bf16_compressor(
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
    )
}

/// Compatibility-only raw launch; prefer the semantic owner method.
#[allow(clippy::too_many_arguments)]
pub fn fp8_projection(
    stream: &CudaStream,
    activation: &DeviceBuffer<u8>,
    activation_scales: &DeviceBuffer<u8>,
    weight: &DeviceBuffer<u8>,
    weight_scales: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    super::submit_fp8_projection(
        stream,
        activation,
        activation_scales,
        weight,
        weight_scales,
        output,
        rows,
        n,
        k,
    )
}

/// Compatibility-only raw launch; prefer the semantic owner method.
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
    super::submit_fp8_query_a_kv(
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
    )
}

/// Compatibility-only raw launch; prefer the semantic owner method.
#[allow(clippy::too_many_arguments)]
pub fn hc_producer(
    stream: &CudaStream,
    state: &DeviceBuffer<f32>,
    function_row_major: &DeviceBuffer<f32>,
    hc_scale: &DeviceBuffer<f32>,
    hc_base: &DeviceBuffer<f32>,
    layer_rms_weight: &DeviceBuffer<f32>,
    mix_output: &mut DeviceBuffer<f32>,
    workspace: &mut DeviceBuffer<f32>,
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
    super::submit_hc_producer(
        stream,
        state,
        function_row_major,
        hc_scale,
        hc_base,
        layer_rms_weight,
        mix_output,
        workspace,
        hidden_output,
        normalized_output,
        packed_output,
        scale_output,
        split_pre,
        split_post,
        split_comb,
        rows,
        hc,
        hidden,
        sinkhorn_iters,
        hc_eps,
        hc_norm_eps,
        layer_rms_eps,
    )
}

/// Compatibility-only raw launch; prefer the semantic owner method.
#[allow(clippy::too_many_arguments)]
pub fn main_project_norm(
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    activation: &mut DeviceBuffer<u8>,
    activation_scales: &mut DeviceBuffer<u8>,
    weight: &DeviceBuffer<u8>,
    weight_scales: &DeviceBuffer<u8>,
    norm_weight: &DeviceBuffer<f32>,
    inv_rms: &mut DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    rms_eps: f32,
) -> Result<()> {
    super::submit_main_project_norm(
        stream,
        input,
        activation,
        activation_scales,
        weight,
        weight_scales,
        norm_weight,
        inv_rms,
        output,
        rows,
        input_size,
        output_size,
        rms_eps,
    )
}

/// Compatibility-only raw launch; prefer the semantic owner method.
#[allow(clippy::too_many_arguments)]
pub fn mla_output(
    stream: &CudaStream,
    context: &DeviceBuffer<f32>,
    output_a_weight: &DeviceBuffer<u8>,
    output_a_scales: &DeviceBuffer<u8>,
    output_b_weight: &DeviceBuffer<u8>,
    output_b_scales: &DeviceBuffer<u8>,
    latent: &mut DeviceBuffer<u16>,
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
    super::submit_mla_output(
        stream,
        context,
        output_a_weight,
        output_a_scales,
        output_b_weight,
        output_b_scales,
        latent,
        latent_fp8,
        latent_scales,
        output,
        rows,
        context_size,
        groups,
        group_input_size,
        rank,
        latent_size,
        hidden_size,
    )
}

/// Compatibility-only raw launch; prefer the semantic owner method.
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
    super::submit_shared_ffn(
        stream,
        input_fp8,
        input_scales,
        gate_weight,
        gate_scales,
        up_weight,
        up_scales,
        down_weight,
        down_scales,
        hidden_f32,
        hidden_fp8,
        hidden_scales,
        output,
        rows,
        input_size,
        intermediate_size,
        output_size,
        gate_blocks,
        up_blocks,
        down_blocks,
        output_scale,
        swiglu_limit,
        accumulate_output,
    )
}
