//! Native CUDA core-kernel facade.
//!
//! Ferrule owns the context, streams, and device allocations. This module only
//! translates the existing host-facing calls into plain POD arguments for the
//! native core provider.

use std::ffi::c_void;
use std::sync::Arc;

use snafu::Snafu;

use crate::cuda::runtime::{
    CudaContext, CudaError, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig,
};

pub(crate) const DSV4_DECODE_INDEX_QUERY_SHARED_ELEMENTS: usize = 8192;

const LINEAR_F32: u32 = 1;
const LINEAR_F32_BYTES: u32 = 2;
const LINEAR_BF16_BYTES: u32 = 3;
const LINEAR_FP8: u32 = 4;
const LINEAR_FP8_PACKED: u32 = 5;
const LINEAR_BF16_ROUNDED_INPUT: u32 = 7;
const LINEAR_FP8_FROM_F32: u32 = 8;

const GROUPED_F32: u32 = 1;
const GROUPED_FP8_TO_BF16: u32 = 2;

const QUANT_FP8_IN_PLACE: u32 = 1;
const QUANT_FP8_NON_ROPE: u32 = 2;
const QUANT_HADAMARD_FP4: u32 = 3;
const QUANT_FP4_PACKED: u32 = 4;
const QUANT_FP8_PACKED: u32 = 5;

const DATA_FILL_I32: u32 = 1;
const DATA_PACK_I32_F32: u32 = 2;
const DATA_PACK_PROPOSAL_HEAD: u32 = 3;
const DATA_FILL_PAGED_WINDOW: u32 = 4;
const DATA_FILL_DECODE_TOPK: u32 = 5;
const DATA_COPY_F32: u32 = 6;
const DATA_GATHER_F32_ROWS: u32 = 7;
const DATA_SCATTER_ADD_F32_ROWS: u32 = 8;
const DATA_SAXPY: u32 = 9;
const DATA_CONVERT_COMBINED_RING: u32 = 10;
const DATA_PAGED_PLANE_SCATTER: u32 = 11;
const DATA_FILL_RECENT_ROWS: u32 = 12;

const EMBED_RESIDENT_HC_BF16: u32 = 1;
const EMBED_PROPOSAL_HC_BF16: u32 = 2;

const NORM_COMPUTE_RMS: u32 = 1;
const NORM_AFFINE_ROW: u32 = 2;
const NORM_AFFINE_ROWS: u32 = 3;
const NORM_HEAD_ROWS: u32 = 4;

const ROPE_YARN: u32 = 1;
const ROPE_TAIL_STRIDED: u32 = 2;
const ROPE_TAIL_INDEXED: u32 = 3;

const ROUTER_TOPK: u32 = 1;
const ROUTER_HASH: u32 = 2;
const VOCAB_TOPK_F32: u32 = 3;
const VOCAB_TOPK_I32: u32 = 4;

const COMPRESSOR_PREFILL: u32 = 1;
const COMPRESSOR_RESET: u32 = 2;
const COMPRESSOR_APPEND: u32 = 3;
const COMPRESSOR_SEED: u32 = 4;
const COMPRESSOR_SOFTMAX: u32 = 5;

const EXPERT_INSTALL: u32 = 1;
const EXPERT_EVICT: u32 = 2;
const EXPERT_INITIALIZE_RESOLVE: u32 = 3;
const EXPERT_RESOLVE: u32 = 4;
const EXPERT_GATHER_DISPATCH: u32 = 5;

const EXPERT_GROUP_ROUTE_INIT_INVOCATION: u32 = 1;
const EXPERT_GROUP_ROUTE_INIT_PLAN: u32 = 2;
const EXPERT_GROUP_ROUTE_COUNT: u32 = 3;
const EXPERT_GROUP_ROUTE_COMPACT: u32 = 4;
const EXPERT_GROUP_ROUTE_SCATTER: u32 = 5;

const MOE_WEIGHTED_SWIGLU_F32: u32 = 2;
const MOE_REDUCE_EXPERT: u32 = 5;
const MOE_REDUCE_SPLIT_EXPERT: u32 = 6;
const MOE_REDUCE_ROUTES: u32 = 7;
const MOE_REDUCE_EXPERT_GROUP_ROUTES: u32 = 8;

const HC_PRE: u32 = 1;
const HC_POST: u32 = 2;
const HC_MEAN_SCATTER: u32 = 3;
const HC_HEAD: u32 = 4;

#[derive(Debug, Snafu)]
pub enum NativeKernelError {
    #[snafu(display("native CUDA operation `{operation}` failed with status {status}"))]
    Launch {
        operation: &'static str,
        status: i32,
    },

    #[snafu(display("native CUDA operation `{operation}` could not bind its context: {source}"))]
    Context {
        operation: &'static str,
        source: CudaError,
    },
}

type KernelResult<T> = std::result::Result<T, NativeKernelError>;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LinearArgs {
    kind: u32,
    batch: u32,
    n: u32,
    k: u32,
    scale_cols: u32,
    block_m: u32,
    block_k: u32,
    packed_offset: u32,
    scale_offset: u32,
    x: u64,
    x_scales: u64,
    weight: u64,
    weight_scales: u64,
    output: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DualLinearArgs {
    kind: u32,
    first_n: u32,
    second_n: u32,
    k: u32,
    first_packed_offset: u32,
    first_scale_offset: u32,
    second_packed_offset: u32,
    second_scale_offset: u32,
    reserved: u32,
    x: u64,
    first_weight: u64,
    first_scales: u64,
    first_output: u64,
    second_weight: u64,
    second_scales: u64,
    second_output: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GroupedLinearArgs {
    kind: u32,
    rows: u32,
    output_dim: u32,
    group_input: u32,
    rank: u32,
    scale_cols: u32,
    reserved: u32,
    input: u64,
    weight: u64,
    weight_scales: u64,
    output: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct QuantizeArgs {
    kind: u32,
    value_offset: u32,
    value_len: u32,
    row_width: u32,
    block_size: u32,
    rope_dim: u32,
    reserved: u32,
    values: u64,
    packed: u64,
    scales: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DataArgs {
    kind: u32,
    count: u32,
    rows: u32,
    width: u32,
    offset: u32,
    start: u32,
    value0: u32,
    value1: u32,
    value2: u32,
    value3: u32,
    flags: u32,
    scale: f32,
    reserved: u32,
    input0: u64,
    input1: u64,
    input2: u64,
    input3: u64,
    input4: u64,
    input5: u64,
    output0: u64,
    output1: u64,
    stream: u64,
    output_elements: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct EmbeddingArgs {
    kind: u32,
    rows: u32,
    vocab: u32,
    hc: u32,
    hidden: u32,
    anchor_token: u32,
    noise_token: u32,
    embedding: u64,
    token_ids: u64,
    output: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NormArgs {
    kind: u32,
    rows: u32,
    width: u32,
    epsilon: f32,
    reserved: u32,
    input: u64,
    weight: u64,
    output: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RopeArgs {
    kind: u32,
    rows: u32,
    heads: u32,
    head_dim: u32,
    rope_dim: u32,
    pair_count: u32,
    start_position: u32,
    position_stride: u32,
    inverse: u32,
    restore_bf16_boundary: u32,
    values: u64,
    cosine: u64,
    sine: u64,
    positions: u64,
    stream: u64,
}

impl RopeArgs {
    fn with_inverse(mut self, inverse: u32) -> Self {
        self.inverse = inverse;
        self.restore_bf16_boundary = inverse;
        self
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RouterArgs {
    kind: u32,
    rows: u32,
    columns: u32,
    k: u32,
    hash_rows: u32,
    hash_columns: u32,
    flags: u32,
    route_scale: f32,
    reserved: u32,
    logits: u64,
    bias: u64,
    token_ids: u64,
    hash_table: u64,
    indices: u64,
    weights: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CompressorArgs {
    kind: u32,
    tokens: u32,
    groups: u32,
    ratio: u32,
    head_dim: u32,
    output_dim: u32,
    overlap: u32,
    position: u32,
    state_elements: u32,
    kv_input: u64,
    score_input: u64,
    ape: u64,
    kv_state: u64,
    score_state: u64,
    output: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IndexerArgs {
    rows: u32,
    prefill: u32,
    window_size: u32,
    window_columns: u32,
    topk: u32,
    value_offset: u32,
    compress_ratio: u32,
    compressed_len: u32,
    heads: u32,
    head_dim: u32,
    page_tokens: u32,
    layer_index: u32,
    layer_count: u32,
    position: u32,
    window_len: u32,
    rope_dim: u32,
    start_position: u32,
    weight_scale: f32,
    flags: u32,
    query: u64,
    weights: u64,
    cosine: u64,
    sine: u64,
    plane: u64,
    plane_elements: u64,
    block_slots: u64,
    block_offsets: u64,
    row_sequence_ids: u64,
    positions: u64,
    window_lens: u64,
    compressed_lens: u64,
    indices: u64,
    selectors: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ExpertTableArgs {
    kind: u32,
    route_count: u32,
    expert_capacity: u32,
    slot_capacity: u32,
    miss_capacity: u32,
    route_capacity: u32,
    expert: u32,
    slot: u32,
    generation: i32,
    active_value: i32,
    reserved: u32,
    gate_weights: u64,
    gate_scales: u64,
    up_weights: u64,
    up_scales: u64,
    down_weights: u64,
    down_scales: u64,
    expert_to_slot: u64,
    expert_generations: u64,
    slot_generations: u64,
    expert_ids: u64,
    route_slots: u64,
    route_generations: u64,
    miss_markers: u64,
    miss_control: u64,
    router_weights: u64,
    active_markers: u64,
    output_gate_weights: u64,
    output_gate_scales: u64,
    output_up_weights: u64,
    output_up_scales: u64,
    output_down_weights: u64,
    output_down_scales: u64,
    output_route_weights: u64,
    dispatch_error: u64,
    gate_weight_value: u64,
    gate_scale_value: u64,
    up_weight_value: u64,
    up_scale_value: u64,
    down_weight_value: u64,
    down_scale_value: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ExpertGroupRoutePlanArgs {
    kind: u32,
    route_count: u32,
    routes_per_token: u32,
    slot_capacity: u32,
    route_capacity: u32,
    output_elements: u32,
    small_group_row_limit: u32,
    route_slots: u64,
    route_generations: u64,
    router_weights: u64,
    slot_generations: u64,
    slot_counts: u64,
    slot_route_offsets: u64,
    slot_cursors: u64,
    active_expert_slots: u64,
    active_group_generations: u64,
    expert_route_indptr: u64,
    expert_route_counts: u64,
    route_token_indices: u64,
    route_indices: u64,
    route_weights: u64,
    host_scalars: u64,
    route_output: u64,
    route_written: u64,
    route_error: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MoeArgs {
    kind: u32,
    n: u32,
    k: u32,
    batch_columns: u32,
    experts: u32,
    tokens: u32,
    routes_per_token: u32,
    output_offset: u32,
    hidden: u32,
    route_weight: f32,
    swiglu_limit: f32,
    input: u64,
    input_scales: u64,
    gate_ptrs: u64,
    gate_scale_ptrs: u64,
    up_ptrs: u64,
    up_scale_ptrs: u64,
    down_ptrs: u64,
    down_scale_ptrs: u64,
    gate: u64,
    up: u64,
    route_weights: u64,
    hidden_values: u64,
    hidden_packed: u64,
    hidden_scales: u64,
    expert_output: u64,
    resident_output: u64,
    materialized_output: u64,
    route_slots: u64,
    materialized_route_slots: u64,
    miss_markers: u64,
    route_output: u64,
    route_written: u64,
    route_error: u64,
    output: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HcArgs {
    kind: u32,
    tokens: u32,
    hc: u32,
    hidden_size: u32,
    mix: u32,
    sinkhorn_iters: u32,
    tap_slot: u32,
    tap_count: u32,
    reserved: u32,
    epsilon: f32,
    norm_epsilon: f32,
    state: u64,
    function: u64,
    scale: u64,
    base: u64,
    hidden: u64,
    residual: u64,
    split_pre: u64,
    split_post: u64,
    split_comb: u64,
    output: u64,
    stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(
    dead_code,
    reason = "layout mirror for the statically linked Core MLA symbol"
)]
struct MlaArgs {
    hidden_size: u32,
    rank: u32,
    output_size: u32,
    epsilon: f32,
    reserved: u32,
    input: u64,
    weight_a: u64,
    weight_b: u64,
    norm_weight: u64,
    output: u64,
    stream: u64,
}

unsafe extern "C" {
    fn ferrule_core_linear_launch(args: *const LinearArgs) -> i32;
    fn ferrule_core_dual_linear_launch(args: *const DualLinearArgs) -> i32;
    fn ferrule_core_grouped_linear_launch(args: *const GroupedLinearArgs) -> i32;
    fn ferrule_core_quantize_launch(args: *const QuantizeArgs) -> i32;
    fn ferrule_core_data_launch(args: *const DataArgs) -> i32;
    fn ferrule_core_embedding_launch(args: *const EmbeddingArgs) -> i32;
    fn ferrule_core_norm_launch(args: *const NormArgs) -> i32;
    fn ferrule_core_rope_launch(args: *const RopeArgs) -> i32;
    fn ferrule_core_router_launch(args: *const RouterArgs) -> i32;
    fn ferrule_core_compressor_launch(args: *const CompressorArgs) -> i32;
    fn ferrule_core_indexer_launch(args: *const IndexerArgs) -> i32;

    fn ferrule_core_expert_table_launch(args: *const ExpertTableArgs) -> i32;
    fn ferrule_core_expert_group_route_plan_launch(args: *const ExpertGroupRoutePlanArgs) -> i32;
    fn ferrule_core_moe_launch(args: *const MoeArgs) -> i32;
    fn ferrule_core_hc_launch(args: *const HcArgs) -> i32;
    #[allow(
        dead_code,
        reason = "keeps the Core MLA link symbol declared and layout-tested"
    )]
    fn ferrule_core_mla_launch(args: *const MlaArgs) -> i32;
}

fn device_ptr<T: DeviceCopy>(buffer: &DeviceBuffer<T>) -> u64 {
    buffer.cu_deviceptr()
}

fn stream_ptr(stream: &CudaStream) -> u64 {
    stream.cu_stream() as *mut c_void as usize as u64
}

fn bind(stream: &CudaStream, operation: &'static str) -> KernelResult<()> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|source| NativeKernelError::Context { operation, source })
}

fn check(operation: &'static str, status: i32) -> KernelResult<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(NativeKernelError::Launch { operation, status })
    }
}

macro_rules! invoke {
    ($stream:expr, $args:ident, $field:ident, $operation:literal) => {{
        bind($stream, $operation)?;
        $args.stream = stream_ptr($stream);
        check($operation, unsafe { $field(&$args) })
    }};
}

pub mod kernels {
    use super::*;

    #[derive(Debug)]
    pub struct LoadedModule {
        _context: Arc<CudaContext>,
    }

    pub fn load(context: &Arc<CudaContext>) -> KernelResult<LoadedModule> {
        context
            .bind_to_thread()
            .map_err(|source| NativeKernelError::Context {
                operation: "load native core provider",
                source,
            })?;
        Ok(LoadedModule {
            _context: Arc::clone(context),
        })
    }

    impl LoadedModule {
        #[allow(clippy::too_many_arguments)]
        unsafe fn linear(
            &self,
            stream: &CudaStream,
            kind: u32,
            x: u64,
            x_scales: u64,
            weight: u64,
            weight_scales: u64,
            output: u64,
            batch: u32,
            n: u32,
            k: u32,
            scale_cols: u32,
            block_m: u32,
            block_k: u32,
            packed_offset: u32,
            scale_offset: u32,
        ) -> KernelResult<()> {
            let mut args = LinearArgs {
                kind,
                batch,
                n,
                k,
                scale_cols,
                block_m,
                block_k,
                packed_offset,
                scale_offset,
                x,
                x_scales,
                weight,
                weight_scales,
                output,
                stream: 0,
            };
            invoke!(stream, args, ferrule_core_linear_launch, "core linear")
        }

        pub unsafe fn gemv_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            w: &DeviceBuffer<f32>,
            y: &mut DeviceBuffer<f32>,
            n: u32,
            k: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.linear(
                    stream,
                    LINEAR_F32,
                    device_ptr(x),
                    0,
                    device_ptr(w),
                    0,
                    device_ptr(y),
                    1,
                    n,
                    k,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            }
        }

        pub unsafe fn gemv_f32_bytes(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            w: &DeviceBuffer<u8>,
            y: &mut DeviceBuffer<f32>,
            n: u32,
            k: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.linear(
                    stream,
                    LINEAR_F32_BYTES,
                    device_ptr(x),
                    0,
                    device_ptr(w),
                    0,
                    device_ptr(y),
                    1,
                    n,
                    k,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            }
        }

        pub unsafe fn gemm_f32_bytes(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            w: &DeviceBuffer<u8>,
            y: &mut DeviceBuffer<f32>,
            batch: u32,
            n: u32,
            k: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.linear(
                    stream,
                    LINEAR_F32_BYTES,
                    device_ptr(x),
                    0,
                    device_ptr(w),
                    0,
                    device_ptr(y),
                    batch,
                    n,
                    k,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            }
        }

        pub unsafe fn linear_bf16_from_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            w: &DeviceBuffer<u8>,
            y: &mut DeviceBuffer<f32>,
            n: u32,
            k: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.linear(
                    stream,
                    LINEAR_BF16_BYTES,
                    device_ptr(x),
                    0,
                    device_ptr(w),
                    0,
                    device_ptr(y),
                    1,
                    n,
                    k,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            }
        }

        pub unsafe fn linear_rows_bf16_from_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            w: &DeviceBuffer<u8>,
            y: &mut DeviceBuffer<f32>,
            batch: u32,
            n: u32,
            k: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.linear(
                    stream,
                    LINEAR_BF16_ROUNDED_INPUT,
                    device_ptr(x),
                    0,
                    device_ptr(w),
                    0,
                    device_ptr(y),
                    batch,
                    n,
                    k,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        unsafe fn dual_linear(
            &self,
            stream: &CudaStream,
            kind: u32,
            x: u64,
            first_weight: u64,
            first_scales: u64,
            first_output: u64,
            second_weight: u64,
            second_scales: u64,
            second_output: u64,
            first_n: u32,
            second_n: u32,
            k: u32,
            first_packed_offset: u32,
            first_scale_offset: u32,
            second_packed_offset: u32,
            second_scale_offset: u32,
        ) -> KernelResult<()> {
            let mut args = DualLinearArgs {
                kind,
                first_n,
                second_n,
                k,
                first_packed_offset,
                first_scale_offset,
                second_packed_offset,
                second_scale_offset,
                x,
                first_weight,
                first_scales,
                first_output,
                second_weight,
                second_scales,
                second_output,
                ..Default::default()
            };
            invoke!(
                stream,
                args,
                ferrule_core_dual_linear_launch,
                "core dual linear"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn dual_linear_bf16_from_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            first_w: &DeviceBuffer<u8>,
            second_w: &DeviceBuffer<u8>,
            first_y: &mut DeviceBuffer<f32>,
            second_y: &mut DeviceBuffer<f32>,
            first_n: u32,
            second_n: u32,
            k: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.dual_linear(
                    stream,
                    LINEAR_BF16_BYTES,
                    device_ptr(x),
                    device_ptr(first_w),
                    0,
                    device_ptr(first_y),
                    device_ptr(second_w),
                    0,
                    device_ptr(second_y),
                    first_n,
                    second_n,
                    k,
                    0,
                    0,
                    0,
                    0,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn gemv_fp8_e4m3fn_e8m0_2d(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            weight: &DeviceBuffer<u8>,
            scales: &DeviceBuffer<u8>,
            y: &mut DeviceBuffer<f32>,
            n: u32,
            k: u32,
            scale_cols: u32,
            block_m: u32,
            block_k: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.linear(
                    stream,
                    LINEAR_FP8,
                    device_ptr(x),
                    0,
                    device_ptr(weight),
                    device_ptr(scales),
                    device_ptr(y),
                    1,
                    n,
                    k,
                    scale_cols,
                    block_m,
                    block_k,
                    0,
                    0,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn gemm_fp8_e4m3fn_e8m0_2d(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            weight: &DeviceBuffer<u8>,
            scales: &DeviceBuffer<u8>,
            y: &mut DeviceBuffer<f32>,
            batch: u32,
            n: u32,
            k: u32,
            scale_cols: u32,
            block_m: u32,
            block_k: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.linear(
                    stream,
                    LINEAR_FP8,
                    device_ptr(x),
                    0,
                    device_ptr(weight),
                    device_ptr(scales),
                    device_ptr(y),
                    batch,
                    n,
                    k,
                    scale_cols,
                    block_m,
                    block_k,
                    0,
                    0,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn gemm_fp8_e4m3fn_e8m0_prepacked(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x_packed: &DeviceBuffer<u8>,
            x_scales: &DeviceBuffer<u8>,
            weight: &DeviceBuffer<u8>,
            weight_scales: &DeviceBuffer<u8>,
            y: &mut DeviceBuffer<f32>,
            batch: u32,
            n: u32,
            k: u32,
            scale_cols: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.linear(
                    stream,
                    LINEAR_FP8_PACKED,
                    device_ptr(x_packed),
                    device_ptr(x_scales),
                    device_ptr(weight),
                    device_ptr(weight_scales),
                    device_ptr(y),
                    batch,
                    n,
                    k,
                    scale_cols,
                    128,
                    128,
                    0,
                    0,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn gemv_fp8_e4m3fn_e8m0_from_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            weight: &DeviceBuffer<u8>,
            weight_scales: &DeviceBuffer<u8>,
            y: &mut DeviceBuffer<f32>,
            n: u32,
            k: u32,
            scale_cols: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.linear(
                    stream,
                    LINEAR_FP8_FROM_F32,
                    device_ptr(x),
                    0,
                    device_ptr(weight),
                    device_ptr(weight_scales),
                    device_ptr(y),
                    1,
                    n,
                    k,
                    scale_cols,
                    128,
                    128,
                    0,
                    0,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        unsafe fn grouped_linear(
            &self,
            stream: &CudaStream,
            kind: u32,
            input: u64,
            weight: u64,
            scales: u64,
            output: u64,
            rows: u32,
            output_dim: u32,
            group_input: u32,
            rank: u32,
            scale_cols: u32,
        ) -> KernelResult<()> {
            let mut args = GroupedLinearArgs {
                kind,
                rows,
                output_dim,
                group_input,
                rank,
                scale_cols,
                input,
                weight,
                weight_scales: scales,
                output,
                ..Default::default()
            };
            invoke!(
                stream,
                args,
                ferrule_core_grouped_linear_launch,
                "core grouped linear"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn grouped_matvec_f32_rows(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            context: &DeviceBuffer<f32>,
            weight: &DeviceBuffer<f32>,
            output: &mut DeviceBuffer<f32>,
            rows: u32,
            output_latent_dim: u32,
            group_in: u32,
            o_lora_rank: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.grouped_linear(
                    stream,
                    GROUPED_F32,
                    device_ptr(context),
                    device_ptr(weight),
                    0,
                    device_ptr(output),
                    rows,
                    output_latent_dim,
                    group_in,
                    o_lora_rank,
                    0,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn grouped_output_a_bf16_from_fp8(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            context: &DeviceBuffer<f32>,
            weight: &DeviceBuffer<u8>,
            weight_scales: &DeviceBuffer<u8>,
            output: &mut DeviceBuffer<f32>,
            rows: u32,
            output_latent_dim: u32,
            group_in: u32,
            o_lora_rank: u32,
            scale_cols: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.grouped_linear(
                    stream,
                    GROUPED_FP8_TO_BF16,
                    device_ptr(context),
                    device_ptr(weight),
                    device_ptr(weight_scales),
                    device_ptr(output),
                    rows,
                    output_latent_dim,
                    group_in,
                    o_lora_rank,
                    scale_cols,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        unsafe fn quantize(
            &self,
            stream: &CudaStream,
            kind: u32,
            values: u64,
            packed: u64,
            scales: u64,
            value_offset: u32,
            value_len: u32,
            row_width: u32,
            block_size: u32,
            rope_dim: u32,
        ) -> KernelResult<()> {
            let mut args = QuantizeArgs {
                kind,
                value_offset,
                value_len,
                row_width,
                block_size,
                rope_dim,
                values,
                packed,
                scales,
                ..Default::default()
            };
            invoke!(stream, args, ferrule_core_quantize_launch, "core quantize")
        }

        pub unsafe fn fp8_e4m3fn_e8m0_quantize_f32_inplace(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            values: &mut DeviceBuffer<f32>,
            value_len: u32,
            row_width: u32,
            block_size: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.quantize(
                    stream,
                    QUANT_FP8_IN_PLACE,
                    device_ptr(values),
                    0,
                    0,
                    0,
                    value_len,
                    row_width,
                    block_size,
                    0,
                )
            }
        }

        pub unsafe fn fp8_e4m3fn_e8m0_quantize_non_rope_f32_inplace(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            values: &mut DeviceBuffer<f32>,
            value_len: u32,
            head_dim: u32,
            rope_dim: u32,
            block_size: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.quantize(
                    stream,
                    QUANT_FP8_NON_ROPE,
                    device_ptr(values),
                    0,
                    0,
                    0,
                    value_len,
                    head_dim,
                    block_size,
                    rope_dim,
                )
            }
        }

        pub unsafe fn hadamard_fp4_e2m1_e8m0_quantize_f32_inplace(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            values: &mut DeviceBuffer<f32>,
            value_len: u32,
            row_width: u32,
            block_size: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.quantize(
                    stream,
                    QUANT_HADAMARD_FP4,
                    device_ptr(values),
                    0,
                    0,
                    0,
                    value_len,
                    row_width,
                    block_size,
                    0,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn fp4_e2m1_e8m0_quantize_f32_packed(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            values: &DeviceBuffer<f32>,
            packed: &mut DeviceBuffer<u8>,
            scales: &mut DeviceBuffer<u8>,
            value_offset: u32,
            value_len: u32,
            row_width: u32,
            block_size: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.quantize(
                    stream,
                    QUANT_FP4_PACKED,
                    device_ptr(values),
                    device_ptr(packed),
                    device_ptr(scales),
                    value_offset,
                    value_len,
                    row_width,
                    block_size,
                    0,
                )
            }
        }

        pub unsafe fn fp8_e4m3fn_e8m0_quantize_f32_packed(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            values: &DeviceBuffer<f32>,
            packed: &mut DeviceBuffer<u8>,
            scales: &mut DeviceBuffer<u8>,
            value_len: u32,
            row_width: u32,
            block_size: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.quantize(
                    stream,
                    QUANT_FP8_PACKED,
                    device_ptr(values),
                    device_ptr(packed),
                    device_ptr(scales),
                    0,
                    value_len,
                    row_width,
                    block_size,
                    0,
                )
            }
        }

        unsafe fn data(&self, stream: &CudaStream, mut args: DataArgs) -> KernelResult<()> {
            invoke!(
                stream,
                args,
                ferrule_core_data_launch,
                "core data operation"
            )
        }

        pub unsafe fn fill_i32_sequence(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            output: &mut DeviceBuffer<i32>,
            start: i32,
            len: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_FILL_I32,
                        count: len,
                        start: start as u32,
                        output0: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        pub unsafe fn pack_i32_f32_pairs(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            indices: &DeviceBuffer<i32>,
            weights: &DeviceBuffer<f32>,
            output: &mut DeviceBuffer<i32>,
            pair_count: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_PACK_I32_F32,
                        count: pair_count.saturating_mul(2),
                        input0: device_ptr(indices),
                        input1: device_ptr(weights),
                        output0: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        pub unsafe fn pack_proposal_head_result(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            status: &DeviceBuffer<i32>,
            token_ids: &DeviceBuffer<i32>,
            confidence: &DeviceBuffer<f32>,
            output: &mut DeviceBuffer<i32>,
            rows: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_PACK_PROPOSAL_HEAD,
                        count: rows.saturating_mul(2).saturating_add(1),
                        rows,
                        input0: device_ptr(status),
                        input1: device_ptr(token_ids),
                        input2: device_ptr(confidence),
                        output0: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        pub unsafe fn fill_dsv4_paged_window_topk(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            output: &mut DeviceBuffer<i32>,
            start: u32,
            valid_len: u32,
            output_len: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_FILL_PAGED_WINDOW,
                        count: output_len,
                        start,
                        value0: valid_len,
                        output0: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn fill_dsv4_decode_attention_topk(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            output: &mut DeviceBuffer<i32>,
            position: u32,
            window_size: u32,
            window_len: u32,
            compressed_len: u32,
            output_len: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_FILL_DECODE_TOPK,
                        count: output_len,
                        width: window_size,
                        start: position,
                        value0: window_len,
                        value1: compressed_len,
                        output0: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        pub unsafe fn fill_recent_rows(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            visible_lens: &DeviceBuffer<i32>,
            output: &mut DeviceBuffer<i32>,
            rows: u32,
            width: u32,
            output_len: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_FILL_RECENT_ROWS,
                        count: output_len,
                        rows,
                        width,
                        input0: device_ptr(visible_lens),
                        output0: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        pub unsafe fn copy_f32_slot(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            src: &DeviceBuffer<f32>,
            dst: &mut DeviceBuffer<f32>,
            dst_offset: u32,
            n: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_COPY_F32,
                        count: n,
                        offset: dst_offset,
                        input0: device_ptr(src),
                        output0: device_ptr(dst),
                        ..Default::default()
                    },
                )
            }
        }

        pub unsafe fn gather_f32_rows(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            src: &DeviceBuffer<f32>,
            row_indices: &DeviceBuffer<i32>,
            dst: &mut DeviceBuffer<f32>,
            rows: u32,
            row_width: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_GATHER_F32_ROWS,
                        count: rows.saturating_mul(row_width),
                        rows,
                        width: row_width,
                        input0: device_ptr(src),
                        input1: device_ptr(row_indices),
                        output0: device_ptr(dst),
                        ..Default::default()
                    },
                )
            }
        }

        pub unsafe fn scatter_add_f32_rows(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            src: &DeviceBuffer<f32>,
            row_indices: &DeviceBuffer<i32>,
            dst: &mut DeviceBuffer<f32>,
            rows: u32,
            row_width: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_SCATTER_ADD_F32_ROWS,
                        count: rows.saturating_mul(row_width),
                        rows,
                        width: row_width,
                        input0: device_ptr(src),
                        input1: device_ptr(row_indices),
                        output0: device_ptr(dst),
                        ..Default::default()
                    },
                )
            }
        }

        pub unsafe fn saxpy(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            scale: f32,
            x: &DeviceBuffer<f32>,
            y: &mut DeviceBuffer<f32>,
            n: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_SAXPY,
                        count: n,
                        scale,
                        input0: device_ptr(x),
                        output0: device_ptr(y),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn convert_combined_ring_topk_indices(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            combined: &DeviceBuffer<i32>,
            row_window_lens: &DeviceBuffer<i32>,
            logical_indices: &mut DeviceBuffer<i32>,
            plane_selectors: &mut DeviceBuffer<i32>,
            elements: u32,
            rows: u32,
            topk: u32,
            start_position: u32,
            position_stride: u32,
            window_size: u32,
            explicit_window_lens: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_CONVERT_COMBINED_RING,
                        count: elements,
                        rows,
                        width: topk,
                        start: start_position,
                        value0: position_stride,
                        value1: window_size,
                        flags: explicit_window_lens,
                        input0: device_ptr(combined),
                        input1: device_ptr(row_window_lens),
                        output0: device_ptr(logical_indices),
                        output1: device_ptr(plane_selectors),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn paged_plane_scatter_rows_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            values: &DeviceBuffer<f32>,
            positions: &DeviceBuffer<i32>,
            block_slots: &DeviceBuffer<i32>,
            block_offsets: &DeviceBuffer<i32>,
            row_sequence_ids: &DeviceBuffer<i32>,
            mask: &DeviceBuffer<i32>,
            plane: &mut DeviceBuffer<f32>,
            num_elements: u32,
            plane_elements: u32,
            rows: u32,
            row_dim: u32,
            page_tokens: u32,
            layer_index: u32,
            layer_count: u32,
            use_row_sequence_ids: u32,
            use_mask: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.data(
                    stream,
                    DataArgs {
                        kind: DATA_PAGED_PLANE_SCATTER,
                        count: num_elements,
                        rows,
                        width: row_dim,
                        value0: page_tokens,
                        value1: layer_index,
                        value2: layer_count,
                        flags: use_row_sequence_ids | (use_mask << 1),
                        input0: device_ptr(values),
                        input1: device_ptr(positions),
                        input2: device_ptr(block_slots),
                        input3: device_ptr(block_offsets),
                        input4: device_ptr(row_sequence_ids),
                        input5: device_ptr(mask),
                        output0: device_ptr(plane),
                        output_elements: u64::from(plane_elements),
                        ..Default::default()
                    },
                )
            }
        }

        pub unsafe fn resident_embedding_hc_bf16(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            embedding: &DeviceBuffer<u8>,
            token_ids: &DeviceBuffer<i32>,
            output: &mut DeviceBuffer<f32>,
            rows: u32,
            vocab: u32,
            hc_mult: u32,
            hidden: u32,
        ) -> KernelResult<()> {
            let mut args = EmbeddingArgs {
                kind: EMBED_RESIDENT_HC_BF16,
                rows,
                vocab,
                hc: hc_mult,
                hidden,
                embedding: device_ptr(embedding),
                token_ids: device_ptr(token_ids),
                output: device_ptr(output),
                ..Default::default()
            };
            invoke!(
                stream,
                args,
                ferrule_core_embedding_launch,
                "resident BF16 embedding"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn proposal_embedding_hc_bf16(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            embedding: &DeviceBuffer<u8>,
            output: &mut DeviceBuffer<f32>,
            anchor_token: u32,
            noise_token: u32,
            rows: u32,
            hc_mult: u32,
            hidden: u32,
        ) -> KernelResult<()> {
            let mut args = EmbeddingArgs {
                kind: EMBED_PROPOSAL_HC_BF16,
                rows,
                hc: hc_mult,
                hidden,
                anchor_token,
                noise_token,
                embedding: device_ptr(embedding),
                output: device_ptr(output),
                vocab: u32::MAX,
                ..Default::default()
            };
            invoke!(
                stream,
                args,
                ferrule_core_embedding_launch,
                "proposal BF16 embedding"
            )
        }

        unsafe fn norm(
            &self,
            stream: &CudaStream,
            kind: u32,
            input: u64,
            weight: u64,
            output: u64,
            rows: u32,
            width: u32,
            epsilon: f32,
        ) -> KernelResult<()> {
            let mut args = NormArgs {
                kind,
                rows,
                width,
                epsilon,
                input,
                weight,
                output,
                ..Default::default()
            };
            invoke!(stream, args, ferrule_core_norm_launch, "core normalization")
        }

        pub unsafe fn compute_rms(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            rms_out: &mut DeviceBuffer<f32>,
            n: u32,
            eps: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.norm(
                    stream,
                    NORM_COMPUTE_RMS,
                    device_ptr(x),
                    0,
                    device_ptr(rms_out),
                    1,
                    n,
                    eps,
                )
            }
        }

        pub unsafe fn rms_norm_fused(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            w: &DeviceBuffer<f32>,
            y: &mut DeviceBuffer<f32>,
            n: u32,
            eps: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.norm(
                    stream,
                    NORM_AFFINE_ROW,
                    device_ptr(x),
                    device_ptr(w),
                    device_ptr(y),
                    1,
                    n,
                    eps,
                )
            }
        }

        pub unsafe fn rms_norm_rows_fused(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            w: &DeviceBuffer<f32>,
            y: &mut DeviceBuffer<f32>,
            rows: u32,
            row_dim: u32,
            eps: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.norm(
                    stream,
                    NORM_AFFINE_ROWS,
                    device_ptr(x),
                    device_ptr(w),
                    device_ptr(y),
                    rows,
                    row_dim,
                    eps,
                )
            }
        }

        pub unsafe fn rms_norm_heads_fused(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            x: &DeviceBuffer<f32>,
            y: &mut DeviceBuffer<f32>,
            heads: u32,
            head_dim: u32,
            eps: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.norm(
                    stream,
                    NORM_HEAD_ROWS,
                    device_ptr(x),
                    0,
                    device_ptr(y),
                    heads,
                    head_dim,
                    eps,
                )
            }
        }

        pub unsafe fn swiglu_weighted_clamped(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            gate: &DeviceBuffer<f32>,
            up: &DeviceBuffer<f32>,
            y: &mut DeviceBuffer<f32>,
            n: u32,
            route_weight: f32,
            limit: f32,
        ) -> KernelResult<()> {
            let mut args = MoeArgs {
                kind: MOE_WEIGHTED_SWIGLU_F32,
                n,
                batch_columns: 1,
                experts: 1,
                route_weight,
                swiglu_limit: limit,
                gate: device_ptr(gate),
                up: device_ptr(up),
                hidden_values: device_ptr(y),
                ..Default::default()
            };
            invoke!(stream, args, ferrule_core_moe_launch, "SwiGLU")
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn rope_yarn(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            qk: &mut DeviceBuffer<f32>,
            freqs_cos: &DeviceBuffer<f32>,
            freqs_sin: &DeviceBuffer<f32>,
            num_elements: u32,
            head_dim: u32,
            rope_dim: u32,
        ) -> KernelResult<()> {
            let heads = if head_dim == 0 {
                0
            } else {
                num_elements / head_dim
            };
            let mut args = RopeArgs {
                kind: ROPE_YARN,
                rows: 1,
                heads,
                head_dim,
                rope_dim,
                pair_count: heads.saturating_mul(rope_dim / 2),
                values: device_ptr(qk),
                cosine: device_ptr(freqs_cos),
                sine: device_ptr(freqs_sin),
                ..Default::default()
            }
            .with_inverse(0);
            invoke!(stream, args, ferrule_core_rope_launch, "YAARN rotary")
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn rope_tail_yaarn_rows_strided(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            qk: &mut DeviceBuffer<f32>,
            cos_table: &DeviceBuffer<f32>,
            sin_table: &DeviceBuffer<f32>,
            num_pairs: u32,
            start_position: u32,
            position_stride: u32,
            rows: u32,
            heads: u32,
            head_dim: u32,
            rope_dim: u32,
            inverse: u32,
        ) -> KernelResult<()> {
            let mut args = RopeArgs {
                kind: ROPE_TAIL_STRIDED,
                rows,
                heads,
                head_dim,
                rope_dim,
                pair_count: num_pairs,
                start_position,
                position_stride,
                values: device_ptr(qk),
                cosine: device_ptr(cos_table),
                sine: device_ptr(sin_table),
                ..Default::default()
            }
            .with_inverse(inverse);
            invoke!(
                stream,
                args,
                ferrule_core_rope_launch,
                "strided tail rotary"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn rope_tail_yaarn_rows_indexed(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            qk: &mut DeviceBuffer<f32>,
            cos_table: &DeviceBuffer<f32>,
            sin_table: &DeviceBuffer<f32>,
            positions: &DeviceBuffer<i32>,
            num_pairs: u32,
            rows: u32,
            heads: u32,
            head_dim: u32,
            rope_dim: u32,
            inverse: u32,
        ) -> KernelResult<()> {
            let mut args = RopeArgs {
                kind: ROPE_TAIL_INDEXED,
                rows,
                heads,
                head_dim,
                rope_dim,
                pair_count: num_pairs,
                values: device_ptr(qk),
                cosine: device_ptr(cos_table),
                sine: device_ptr(sin_table),
                positions: device_ptr(positions),
                ..Default::default()
            }
            .with_inverse(inverse);
            invoke!(
                stream,
                args,
                ferrule_core_rope_launch,
                "indexed tail rotary"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn dsv4_router_topk_sqrt_softplus_rows(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            logits: &DeviceBuffer<f32>,
            bias: &DeviceBuffer<f32>,
            indices: &mut DeviceBuffer<i32>,
            weights: &mut DeviceBuffer<f32>,
            tokens: u32,
            experts: u32,
            k: u32,
            bias_enabled: u32,
            route_scale: f32,
        ) -> KernelResult<()> {
            let mut args = RouterArgs {
                kind: ROUTER_TOPK,
                rows: tokens,
                columns: experts,
                k,
                flags: bias_enabled,
                route_scale,
                logits: device_ptr(logits),
                bias: device_ptr(bias),
                indices: device_ptr(indices),
                weights: device_ptr(weights),
                ..Default::default()
            };
            invoke!(
                stream,
                args,
                ferrule_core_router_launch,
                "DSV4 router top-k"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn dsv4_router_hash_sqrt_softplus_rows(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            logits: &DeviceBuffer<f32>,
            token_ids: &DeviceBuffer<i32>,
            hash_table: &DeviceBuffer<i32>,
            indices: &mut DeviceBuffer<i32>,
            weights: &mut DeviceBuffer<f32>,
            tokens: u32,
            experts: u32,
            hash_rows: u32,
            hash_cols: u32,
            k: u32,
            route_scale: f32,
        ) -> KernelResult<()> {
            let mut args = RouterArgs {
                kind: ROUTER_HASH,
                rows: tokens,
                columns: experts,
                k,
                hash_rows,
                hash_columns: hash_cols,
                route_scale,
                logits: device_ptr(logits),
                token_ids: device_ptr(token_ids),
                hash_table: device_ptr(hash_table),
                indices: device_ptr(indices),
                weights: device_ptr(weights),
                ..Default::default()
            };
            invoke!(stream, args, ferrule_core_router_launch, "DSV4 hash router")
        }

        pub unsafe fn topk_vocab(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            logits: &DeviceBuffer<f32>,
            out_idx: &mut DeviceBuffer<f32>,
            out_val: &mut DeviceBuffer<f32>,
            vocab: u32,
            k: u32,
        ) -> KernelResult<()> {
            let mut args = RouterArgs {
                kind: VOCAB_TOPK_F32,
                rows: 1,
                columns: vocab,
                k,
                logits: device_ptr(logits),
                indices: device_ptr(out_idx),
                weights: device_ptr(out_val),
                ..Default::default()
            };
            invoke!(stream, args, ferrule_core_router_launch, "vocabulary top-k")
        }

        pub unsafe fn topk_vocab_rows(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            logits: &DeviceBuffer<f32>,
            out_idx: &mut DeviceBuffer<i32>,
            out_val: &mut DeviceBuffer<f32>,
            rows: u32,
            vocab: u32,
            k: u32,
        ) -> KernelResult<()> {
            let mut args = RouterArgs {
                kind: VOCAB_TOPK_I32,
                rows,
                columns: vocab,
                k,
                logits: device_ptr(logits),
                indices: device_ptr(out_idx),
                weights: device_ptr(out_val),
                ..Default::default()
            };
            invoke!(
                stream,
                args,
                ferrule_core_router_launch,
                "row vocabulary top-k"
            )
        }

        unsafe fn compressor(
            &self,
            stream: &CudaStream,
            mut args: CompressorArgs,
        ) -> KernelResult<()> {
            invoke!(
                stream,
                args,
                ferrule_core_compressor_launch,
                "recurrent compressor"
            )
        }

        pub unsafe fn compressor_recurrent_reset_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            kv_state: &mut DeviceBuffer<f32>,
            score_state: &mut DeviceBuffer<f32>,
            state_elements: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.compressor(
                    stream,
                    CompressorArgs {
                        kind: COMPRESSOR_RESET,
                        state_elements,
                        kv_state: device_ptr(kv_state),
                        score_state: device_ptr(score_state),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn compressor_recurrent_append_projected_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            projected_kv: &DeviceBuffer<f32>,
            projected_score: &DeviceBuffer<f32>,
            ape: &DeviceBuffer<f32>,
            kv_state: &mut DeviceBuffer<f32>,
            score_state: &mut DeviceBuffer<f32>,
            position: u32,
            ratio: u32,
            out_dim: u32,
            overlap: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.compressor(
                    stream,
                    CompressorArgs {
                        kind: COMPRESSOR_APPEND,
                        ratio,
                        output_dim: out_dim,
                        overlap,
                        position,
                        kv_input: device_ptr(projected_kv),
                        score_input: device_ptr(projected_score),
                        ape: device_ptr(ape),
                        kv_state: device_ptr(kv_state),
                        score_state: device_ptr(score_state),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn compressor_recurrent_seed_prefill_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            projected_kv_rows: &DeviceBuffer<f32>,
            projected_score_rows: &DeviceBuffer<f32>,
            ape: &DeviceBuffer<f32>,
            kv_state: &mut DeviceBuffer<f32>,
            score_state: &mut DeviceBuffer<f32>,
            tokens: u32,
            ratio: u32,
            out_dim: u32,
            overlap: u32,
            state_elements: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.compressor(
                    stream,
                    CompressorArgs {
                        kind: COMPRESSOR_SEED,
                        tokens,
                        ratio,
                        output_dim: out_dim,
                        overlap,
                        state_elements,
                        kv_input: device_ptr(projected_kv_rows),
                        score_input: device_ptr(projected_score_rows),
                        ape: device_ptr(ape),
                        kv_state: device_ptr(kv_state),
                        score_state: device_ptr(score_state),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn compressor_recurrent_softmax_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            kv_state: &DeviceBuffer<f32>,
            score_state: &DeviceBuffer<f32>,
            output: &mut DeviceBuffer<f32>,
            ratio: u32,
            head_dim: u32,
            out_dim: u32,
            overlap: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.compressor(
                    stream,
                    CompressorArgs {
                        kind: COMPRESSOR_SOFTMAX,
                        ratio,
                        head_dim,
                        output_dim: out_dim,
                        overlap,
                        kv_state: device_ptr(kv_state),
                        score_state: device_ptr(score_state),
                        output: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn dsv4_compressor_prefill_softmax(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            kv_rows: &DeviceBuffer<f32>,
            score_rows: &DeviceBuffer<f32>,
            ape: &DeviceBuffer<f32>,
            output: &mut DeviceBuffer<f32>,
            groups: u32,
            ratio: u32,
            head_dim: u32,
            out_dim: u32,
            overlap: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.compressor(
                    stream,
                    CompressorArgs {
                        kind: COMPRESSOR_PREFILL,
                        groups,
                        ratio,
                        head_dim,
                        output_dim: out_dim,
                        overlap,
                        kv_input: device_ptr(kv_rows),
                        score_input: device_ptr(score_rows),
                        ape: device_ptr(ape),
                        output: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        unsafe fn indexer(&self, stream: &CudaStream, mut args: IndexerArgs) -> KernelResult<()> {
            invoke!(stream, args, ferrule_core_indexer_launch, "DSV4 indexer")
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn dsv4_prefill_topk_indices_paged_indexer(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            query: &DeviceBuffer<f32>,
            weights: &DeviceBuffer<f32>,
            indexer_plane: &DeviceBuffer<f32>,
            block_slots: &DeviceBuffer<i32>,
            block_offsets: &DeviceBuffer<i32>,
            out: &mut DeviceBuffer<i32>,
            tokens: u32,
            window_size: u32,
            window_cols: u32,
            extra_cols: u32,
            value_offset: u32,
            compress_ratio: u32,
            compressed_len: u32,
            index_heads: u32,
            index_head_dim: u32,
            page_tokens: u32,
            layer_index: u32,
            layer_count: u32,
            weight_scale: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.indexer(
                    stream,
                    IndexerArgs {
                        rows: tokens,
                        prefill: 1,
                        window_size,
                        window_columns: window_cols,
                        topk: extra_cols,
                        value_offset,
                        compress_ratio,
                        compressed_len,
                        heads: index_heads,
                        head_dim: index_head_dim,
                        page_tokens,
                        layer_index,
                        layer_count,
                        weight_scale,
                        query: device_ptr(query),
                        weights: device_ptr(weights),
                        plane: device_ptr(indexer_plane),
                        plane_elements: indexer_plane.len() as u64,
                        block_slots: device_ptr(block_slots),
                        block_offsets: device_ptr(block_offsets),
                        indices: device_ptr(out),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn dsv4_prefill_topk_indices_fused_index_query_paged_indexer(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            query: &DeviceBuffer<f32>,
            weights: &DeviceBuffer<f32>,
            indexer_plane: &DeviceBuffer<f32>,
            block_slots: &DeviceBuffer<i32>,
            block_offsets: &DeviceBuffer<i32>,
            cosine: &DeviceBuffer<f32>,
            sine: &DeviceBuffer<f32>,
            out: &mut DeviceBuffer<i32>,
            tokens: u32,
            window_size: u32,
            window_cols: u32,
            extra_cols: u32,
            value_offset: u32,
            compress_ratio: u32,
            compressed_len: u32,
            index_heads: u32,
            index_head_dim: u32,
            rope_dim: u32,
            start_position: u32,
            page_tokens: u32,
            layer_index: u32,
            layer_count: u32,
            weight_scale: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.indexer(
                    stream,
                    IndexerArgs {
                        rows: tokens,
                        prefill: 1,
                        window_size,
                        window_columns: window_cols,
                        topk: extra_cols,
                        value_offset,
                        compress_ratio,
                        compressed_len,
                        heads: index_heads,
                        head_dim: index_head_dim,
                        page_tokens,
                        layer_index,
                        layer_count,
                        rope_dim,
                        start_position,
                        weight_scale,
                        flags: 1,
                        query: device_ptr(query),
                        weights: device_ptr(weights),
                        cosine: device_ptr(cosine),
                        sine: device_ptr(sine),
                        plane: device_ptr(indexer_plane),
                        plane_elements: indexer_plane.len() as u64,
                        block_slots: device_ptr(block_slots),
                        block_offsets: device_ptr(block_offsets),
                        indices: device_ptr(out),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn dsv4_decode_topk_indices_paged_indexer(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            query: &DeviceBuffer<f32>,
            weights: &DeviceBuffer<f32>,
            indexer_plane: &DeviceBuffer<f32>,
            block_slots: &DeviceBuffer<i32>,
            block_offsets: &DeviceBuffer<i32>,
            out: &mut DeviceBuffer<i32>,
            position: u32,
            window_len: u32,
            window_size: u32,
            extra_cols: u32,
            value_offset: u32,
            compressed_len: u32,
            index_heads: u32,
            index_head_dim: u32,
            page_tokens: u32,
            layer_index: u32,
            layer_count: u32,
            weight_scale: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.indexer(
                    stream,
                    IndexerArgs {
                        rows: 1,
                        window_size,
                        topk: extra_cols,
                        value_offset,
                        compressed_len,
                        heads: index_heads,
                        head_dim: index_head_dim,
                        page_tokens,
                        layer_index,
                        layer_count,
                        position,
                        window_len,
                        weight_scale,
                        query: device_ptr(query),
                        weights: device_ptr(weights),
                        plane: device_ptr(indexer_plane),
                        plane_elements: indexer_plane.len() as u64,
                        block_slots: device_ptr(block_slots),
                        block_offsets: device_ptr(block_offsets),
                        indices: device_ptr(out),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn dsv4_decode_topk_indices_fused_index_query_paged_indexer(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            query: &DeviceBuffer<f32>,
            weights: &DeviceBuffer<f32>,
            indexer_plane: &DeviceBuffer<f32>,
            block_slots: &DeviceBuffer<i32>,
            block_offsets: &DeviceBuffer<i32>,
            cosine: &DeviceBuffer<f32>,
            sine: &DeviceBuffer<f32>,
            out: &mut DeviceBuffer<i32>,
            position: u32,
            window_len: u32,
            window_size: u32,
            extra_cols: u32,
            value_offset: u32,
            compressed_len: u32,
            index_heads: u32,
            index_head_dim: u32,
            rope_dim: u32,
            page_tokens: u32,
            layer_index: u32,
            layer_count: u32,
            weight_scale: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.indexer(
                    stream,
                    IndexerArgs {
                        rows: 1,
                        window_size,
                        topk: extra_cols,
                        value_offset,
                        compressed_len,
                        heads: index_heads,
                        head_dim: index_head_dim,
                        page_tokens,
                        layer_index,
                        layer_count,
                        position,
                        window_len,
                        rope_dim,
                        weight_scale,
                        flags: 1,
                        query: device_ptr(query),
                        weights: device_ptr(weights),
                        cosine: device_ptr(cosine),
                        sine: device_ptr(sine),
                        plane: device_ptr(indexer_plane),
                        plane_elements: indexer_plane.len() as u64,
                        block_slots: device_ptr(block_slots),
                        block_offsets: device_ptr(block_offsets),
                        indices: device_ptr(out),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn dsv4_decode_topk_indices_paged_indexer_rows(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            query: &DeviceBuffer<f32>,
            weights: &DeviceBuffer<f32>,
            indexer_plane: &DeviceBuffer<f32>,
            block_slots: &DeviceBuffer<i32>,
            block_offsets: &DeviceBuffer<i32>,
            row_sequence_ids: &DeviceBuffer<i32>,
            positions: &DeviceBuffer<i32>,
            window_lens: &DeviceBuffer<i32>,
            compressed_lens: &DeviceBuffer<i32>,
            logical_indices: &mut DeviceBuffer<i32>,
            plane_selectors: &mut DeviceBuffer<i32>,
            rows: u32,
            window_size: u32,
            index_topk: u32,
            index_heads: u32,
            index_head_dim: u32,
            page_tokens: u32,
            layer_index: u32,
            layer_count: u32,
            weight_scale: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.indexer(
                    stream,
                    IndexerArgs {
                        rows,
                        window_size,
                        topk: index_topk,
                        heads: index_heads,
                        head_dim: index_head_dim,
                        page_tokens,
                        layer_index,
                        layer_count,
                        weight_scale,
                        query: device_ptr(query),
                        weights: device_ptr(weights),
                        plane: device_ptr(indexer_plane),
                        plane_elements: indexer_plane.len() as u64,
                        block_slots: device_ptr(block_slots),
                        block_offsets: device_ptr(block_offsets),
                        row_sequence_ids: device_ptr(row_sequence_ids),
                        positions: device_ptr(positions),
                        window_lens: device_ptr(window_lens),
                        compressed_lens: device_ptr(compressed_lens),
                        indices: device_ptr(logical_indices),
                        selectors: device_ptr(plane_selectors),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn install_expert_slot_binding(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            gate_weight: &mut DeviceBuffer<u64>,
            gate_scale: &mut DeviceBuffer<u64>,
            up_weight: &mut DeviceBuffer<u64>,
            up_scale: &mut DeviceBuffer<u64>,
            down_weight: &mut DeviceBuffer<u64>,
            down_scale: &mut DeviceBuffer<u64>,
            expert_to_slot: &mut DeviceBuffer<i32>,
            expert_generation: &mut DeviceBuffer<i32>,
            slot_generation: &mut DeviceBuffer<i32>,
            expert: u32,
            slot: u32,
            generation: i32,
            gate_weight_ptr: u64,
            gate_scale_ptr: u64,
            up_weight_ptr: u64,
            up_scale_ptr: u64,
            down_weight_ptr: u64,
            down_scale_ptr: u64,
        ) -> KernelResult<()> {
            let mut args = ExpertTableArgs {
                kind: EXPERT_INSTALL,
                expert,
                slot,
                generation,
                gate_weights: device_ptr(gate_weight),
                gate_scales: device_ptr(gate_scale),
                up_weights: device_ptr(up_weight),
                up_scales: device_ptr(up_scale),
                down_weights: device_ptr(down_weight),
                down_scales: device_ptr(down_scale),
                expert_to_slot: device_ptr(expert_to_slot),
                expert_generations: device_ptr(expert_generation),
                slot_generations: device_ptr(slot_generation),
                gate_weight_value: gate_weight_ptr,
                gate_scale_value: gate_scale_ptr,
                up_weight_value: up_weight_ptr,
                up_scale_value: up_scale_ptr,
                down_weight_value: down_weight_ptr,
                down_scale_value: down_scale_ptr,
                ..Default::default()
            };
            invoke!(
                stream,
                args,
                ferrule_core_expert_table_launch,
                "install expert slot"
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn evict_expert_slot_binding(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            gate_weight: &mut DeviceBuffer<u64>,
            gate_scale: &mut DeviceBuffer<u64>,
            up_weight: &mut DeviceBuffer<u64>,
            up_scale: &mut DeviceBuffer<u64>,
            down_weight: &mut DeviceBuffer<u64>,
            down_scale: &mut DeviceBuffer<u64>,
            expert_to_slot: &mut DeviceBuffer<i32>,
            expert_generation: &mut DeviceBuffer<i32>,
            slot_generation: &mut DeviceBuffer<i32>,
            expert: u32,
            slot: u32,
            next_generation: i32,
        ) -> KernelResult<()> {
            let mut args = ExpertTableArgs {
                kind: EXPERT_EVICT,
                expert,
                slot,
                generation: next_generation,
                gate_weights: device_ptr(gate_weight),
                gate_scales: device_ptr(gate_scale),
                up_weights: device_ptr(up_weight),
                up_scales: device_ptr(up_scale),
                down_weights: device_ptr(down_weight),
                down_scales: device_ptr(down_scale),
                expert_to_slot: device_ptr(expert_to_slot),
                expert_generations: device_ptr(expert_generation),
                slot_generations: device_ptr(slot_generation),
                ..Default::default()
            };
            invoke!(
                stream,
                args,
                ferrule_core_expert_table_launch,
                "evict expert slot"
            )
        }

        unsafe fn expert_table(
            &self,
            stream: &CudaStream,
            mut args: ExpertTableArgs,
        ) -> KernelResult<()> {
            invoke!(
                stream,
                args,
                ferrule_core_expert_table_launch,
                "expert slot table"
            )
        }

        pub unsafe fn initialize_expert_slot_resolve(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            miss_control: &mut DeviceBuffer<i32>,
            miss_capacity: u32,
            route_capacity: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.expert_table(
                    stream,
                    ExpertTableArgs {
                        kind: EXPERT_INITIALIZE_RESOLVE,
                        miss_capacity,
                        route_capacity,
                        miss_control: device_ptr(miss_control),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn resolve_expert_slots(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            expert_ids: &DeviceBuffer<i32>,
            expert_to_slot: &DeviceBuffer<i32>,
            expert_generations: &DeviceBuffer<i32>,
            slot_generations: &DeviceBuffer<i32>,
            route_slots: &mut DeviceBuffer<i32>,
            route_generations: &mut DeviceBuffer<i32>,
            miss_markers: &mut DeviceBuffer<i32>,
            miss_control: &mut DeviceBuffer<i32>,
            route_count: u32,
            expert_capacity: u32,
            slot_capacity: u32,
            miss_capacity: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.expert_table(
                    stream,
                    ExpertTableArgs {
                        kind: EXPERT_RESOLVE,
                        route_count,
                        expert_capacity,
                        slot_capacity,
                        miss_capacity,
                        expert_to_slot: device_ptr(expert_to_slot),
                        expert_generations: device_ptr(expert_generations),
                        slot_generations: device_ptr(slot_generations),
                        expert_ids: device_ptr(expert_ids),
                        route_slots: device_ptr(route_slots),
                        route_generations: device_ptr(route_generations),
                        miss_markers: device_ptr(miss_markers),
                        miss_control: device_ptr(miss_control),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn gather_stable_moe_dispatch(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            table_gate_ptrs: &DeviceBuffer<u64>,
            table_gate_scale_ptrs: &DeviceBuffer<u64>,
            table_up_ptrs: &DeviceBuffer<u64>,
            table_up_scale_ptrs: &DeviceBuffer<u64>,
            table_down_ptrs: &DeviceBuffer<u64>,
            table_down_scale_ptrs: &DeviceBuffer<u64>,
            slot_generations: &DeviceBuffer<i32>,
            resolved_slots: &DeviceBuffer<i32>,
            resolved_generations: &DeviceBuffer<i32>,
            router_weights: &DeviceBuffer<f32>,
            active_markers: &DeviceBuffer<i32>,
            active_value: i32,
            gate_ptrs: &mut DeviceBuffer<u64>,
            gate_scale_ptrs: &mut DeviceBuffer<u64>,
            up_ptrs: &mut DeviceBuffer<u64>,
            up_scale_ptrs: &mut DeviceBuffer<u64>,
            down_ptrs: &mut DeviceBuffer<u64>,
            down_scale_ptrs: &mut DeviceBuffer<u64>,
            route_weights: &mut DeviceBuffer<f32>,
            route_slots: &mut DeviceBuffer<i32>,
            dispatch_error: &mut DeviceBuffer<i32>,
            route_count: u32,
            slot_capacity: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.expert_table(
                    stream,
                    ExpertTableArgs {
                        kind: EXPERT_GATHER_DISPATCH,
                        route_count,
                        slot_capacity,
                        active_value,
                        gate_weights: device_ptr(table_gate_ptrs),
                        gate_scales: device_ptr(table_gate_scale_ptrs),
                        up_weights: device_ptr(table_up_ptrs),
                        up_scales: device_ptr(table_up_scale_ptrs),
                        down_weights: device_ptr(table_down_ptrs),
                        down_scales: device_ptr(table_down_scale_ptrs),
                        slot_generations: device_ptr(slot_generations),
                        expert_ids: device_ptr(route_slots),
                        route_slots: device_ptr(resolved_slots),
                        route_generations: device_ptr(resolved_generations),
                        router_weights: device_ptr(router_weights),
                        active_markers: device_ptr(active_markers),
                        output_gate_weights: device_ptr(gate_ptrs),
                        output_gate_scales: device_ptr(gate_scale_ptrs),
                        output_up_weights: device_ptr(up_ptrs),
                        output_up_scales: device_ptr(up_scale_ptrs),
                        output_down_weights: device_ptr(down_ptrs),
                        output_down_scales: device_ptr(down_scale_ptrs),
                        output_route_weights: device_ptr(route_weights),
                        dispatch_error: device_ptr(dispatch_error),
                        ..Default::default()
                    },
                )
            }
        }

        unsafe fn expert_group_route_plan(
            &self,
            stream: &CudaStream,
            mut args: ExpertGroupRoutePlanArgs,
        ) -> KernelResult<()> {
            invoke!(
                stream,
                args,
                ferrule_core_expert_group_route_plan_launch,
                "expert group route plan"
            )
        }

        pub unsafe fn initialize_expert_group_route_invocation(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            route_output: &mut DeviceBuffer<f32>,
            route_written: &mut DeviceBuffer<i32>,
            route_error: &mut DeviceBuffer<i32>,
            output_elements: u32,
            route_count: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.expert_group_route_plan(
                    stream,
                    ExpertGroupRoutePlanArgs {
                        kind: EXPERT_GROUP_ROUTE_INIT_INVOCATION,
                        route_count,
                        output_elements,
                        route_output: device_ptr(route_output),
                        route_written: device_ptr(route_written),
                        route_error: device_ptr(route_error),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn initialize_expert_group_route_plan(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            slot_counts: &mut DeviceBuffer<i32>,
            slot_route_offsets: &mut DeviceBuffer<i32>,
            slot_cursors: &mut DeviceBuffer<i32>,
            active_expert_slots: &mut DeviceBuffer<i32>,
            active_group_generations: &mut DeviceBuffer<i32>,
            expert_route_indptr: &mut DeviceBuffer<i32>,
            expert_route_counts: &mut DeviceBuffer<i32>,
            route_token_indices: &mut DeviceBuffer<i32>,
            route_indices: &mut DeviceBuffer<i32>,
            route_weights: &mut DeviceBuffer<f32>,
            host_scalars: &mut DeviceBuffer<i32>,
            slot_capacity: u32,
            route_capacity: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.expert_group_route_plan(
                    stream,
                    ExpertGroupRoutePlanArgs {
                        kind: EXPERT_GROUP_ROUTE_INIT_PLAN,
                        slot_capacity,
                        route_capacity,
                        slot_counts: device_ptr(slot_counts),
                        slot_route_offsets: device_ptr(slot_route_offsets),
                        slot_cursors: device_ptr(slot_cursors),
                        active_expert_slots: device_ptr(active_expert_slots),
                        active_group_generations: device_ptr(active_group_generations),
                        expert_route_indptr: device_ptr(expert_route_indptr),
                        expert_route_counts: device_ptr(expert_route_counts),
                        route_token_indices: device_ptr(route_token_indices),
                        route_indices: device_ptr(route_indices),
                        route_weights: device_ptr(route_weights),
                        host_scalars: device_ptr(host_scalars),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn count_expert_group_routes(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            route_slots: &DeviceBuffer<i32>,
            route_generations: &DeviceBuffer<i32>,
            slot_generations: &DeviceBuffer<i32>,
            slot_counts: &mut DeviceBuffer<i32>,
            route_count: u32,
            slot_capacity: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.expert_group_route_plan(
                    stream,
                    ExpertGroupRoutePlanArgs {
                        kind: EXPERT_GROUP_ROUTE_COUNT,
                        route_count,
                        slot_capacity,
                        route_slots: device_ptr(route_slots),
                        route_generations: device_ptr(route_generations),
                        slot_generations: device_ptr(slot_generations),
                        slot_counts: device_ptr(slot_counts),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn compact_expert_group_routes(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            slot_counts: &DeviceBuffer<i32>,
            slot_generations: &DeviceBuffer<i32>,
            slot_route_offsets: &mut DeviceBuffer<i32>,
            active_expert_slots: &mut DeviceBuffer<i32>,
            active_group_generations: &mut DeviceBuffer<i32>,
            expert_route_indptr: &mut DeviceBuffer<i32>,
            expert_route_counts: &mut DeviceBuffer<i32>,
            host_scalars: &mut DeviceBuffer<i32>,
            route_error: &mut DeviceBuffer<i32>,
            slot_capacity: u32,
            route_capacity: u32,
            small_group_row_limit: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.expert_group_route_plan(
                    stream,
                    ExpertGroupRoutePlanArgs {
                        kind: EXPERT_GROUP_ROUTE_COMPACT,
                        slot_capacity,
                        route_capacity,
                        small_group_row_limit,
                        slot_generations: device_ptr(slot_generations),
                        slot_counts: device_ptr(slot_counts),
                        slot_route_offsets: device_ptr(slot_route_offsets),
                        active_expert_slots: device_ptr(active_expert_slots),
                        active_group_generations: device_ptr(active_group_generations),
                        expert_route_indptr: device_ptr(expert_route_indptr),
                        expert_route_counts: device_ptr(expert_route_counts),
                        host_scalars: device_ptr(host_scalars),
                        route_error: device_ptr(route_error),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn scatter_expert_group_routes(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            route_slots: &DeviceBuffer<i32>,
            route_generations: &DeviceBuffer<i32>,
            router_weights: &DeviceBuffer<f32>,
            slot_generations: &DeviceBuffer<i32>,
            slot_route_offsets: &DeviceBuffer<i32>,
            slot_cursors: &mut DeviceBuffer<i32>,
            route_token_indices: &mut DeviceBuffer<i32>,
            route_indices: &mut DeviceBuffer<i32>,
            route_weights: &mut DeviceBuffer<f32>,
            route_error: &mut DeviceBuffer<i32>,
            route_count: u32,
            routes_per_token: u32,
            slot_capacity: u32,
            route_capacity: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.expert_group_route_plan(
                    stream,
                    ExpertGroupRoutePlanArgs {
                        kind: EXPERT_GROUP_ROUTE_SCATTER,
                        route_count,
                        routes_per_token,
                        slot_capacity,
                        route_capacity,
                        route_slots: device_ptr(route_slots),
                        route_generations: device_ptr(route_generations),
                        router_weights: device_ptr(router_weights),
                        slot_generations: device_ptr(slot_generations),
                        slot_route_offsets: device_ptr(slot_route_offsets),
                        slot_cursors: device_ptr(slot_cursors),
                        route_token_indices: device_ptr(route_token_indices),
                        route_indices: device_ptr(route_indices),
                        route_weights: device_ptr(route_weights),
                        route_error: device_ptr(route_error),
                        ..Default::default()
                    },
                )
            }
        }

        unsafe fn moe(&self, stream: &CudaStream, mut args: MoeArgs) -> KernelResult<()> {
            invoke!(stream, args, ferrule_core_moe_launch, "MoE operator")
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn moe_reduce_expert_outputs_ranked(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            expert_output: &DeviceBuffer<f32>,
            route_slots: &DeviceBuffer<i32>,
            output: &mut DeviceBuffer<f32>,
            output_offset: u32,
            hidden_size: u32,
            batch_columns: u32,
            routes_per_column: u32,
            experts: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.moe(
                    stream,
                    MoeArgs {
                        kind: MOE_REDUCE_EXPERT,
                        batch_columns,
                        experts,
                        routes_per_token: routes_per_column,
                        output_offset,
                        hidden: hidden_size,
                        expert_output: device_ptr(expert_output),
                        route_slots: device_ptr(route_slots),
                        output: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn moe_reduce_split_expert_outputs_ranked(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            resident_output: &DeviceBuffer<f32>,
            materialized_output: &DeviceBuffer<f32>,
            resident_route_slots: &DeviceBuffer<i32>,
            materialized_route_slots: &DeviceBuffer<i32>,
            miss_markers: &DeviceBuffer<i32>,
            output: &mut DeviceBuffer<f32>,
            output_offset: u32,
            hidden_size: u32,
            batch_columns: u32,
            routes_per_column: u32,
            experts: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.moe(
                    stream,
                    MoeArgs {
                        kind: MOE_REDUCE_SPLIT_EXPERT,
                        batch_columns,
                        experts,
                        routes_per_token: routes_per_column,
                        output_offset,
                        hidden: hidden_size,
                        resident_output: device_ptr(resident_output),
                        materialized_output: device_ptr(materialized_output),
                        route_slots: device_ptr(resident_route_slots),
                        materialized_route_slots: device_ptr(materialized_route_slots),
                        miss_markers: device_ptr(miss_markers),
                        output: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        pub unsafe fn moe_reduce_route_outputs_ranked(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            route_output: &DeviceBuffer<f32>,
            output: &mut DeviceBuffer<f32>,
            tokens: u32,
            routes_per_token: u32,
            hidden_size: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.moe(
                    stream,
                    MoeArgs {
                        kind: MOE_REDUCE_ROUTES,
                        tokens,
                        routes_per_token,
                        hidden: hidden_size,
                        route_output: device_ptr(route_output),
                        output: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn moe_reduce_expert_group_route_outputs_ranked(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            route_output: &DeviceBuffer<f32>,
            route_written: &DeviceBuffer<i32>,
            route_error: &DeviceBuffer<i32>,
            output: &mut DeviceBuffer<f32>,
            tokens: u32,
            routes_per_token: u32,
            hidden_size: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.moe(
                    stream,
                    MoeArgs {
                        kind: MOE_REDUCE_EXPERT_GROUP_ROUTES,
                        tokens,
                        routes_per_token,
                        hidden: hidden_size,
                        route_output: device_ptr(route_output),
                        route_written: device_ptr(route_written),
                        route_error: device_ptr(route_error),
                        output: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        unsafe fn hc(&self, stream: &CudaStream, mut args: HcArgs) -> KernelResult<()> {
            invoke!(stream, args, ferrule_core_hc_launch, "hyper-connection")
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn hc_pre_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            state: &DeviceBuffer<f32>,
            function: &DeviceBuffer<f32>,
            scale: &DeviceBuffer<f32>,
            base: &DeviceBuffer<f32>,
            hidden: &mut DeviceBuffer<f32>,
            split_pre: &mut DeviceBuffer<f32>,
            split_post: &mut DeviceBuffer<f32>,
            split_comb: &mut DeviceBuffer<f32>,
            tokens: u32,
            hc: u32,
            hidden_size: u32,
            mix: u32,
            sinkhorn_iters: u32,
            epsilon: f32,
            norm_epsilon: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.hc(
                    stream,
                    HcArgs {
                        kind: HC_PRE,
                        tokens,
                        hc,
                        hidden_size,
                        mix,
                        sinkhorn_iters,
                        epsilon,
                        norm_epsilon,
                        state: device_ptr(state),
                        function: device_ptr(function),
                        scale: device_ptr(scale),
                        base: device_ptr(base),
                        hidden: device_ptr(hidden),
                        split_pre: device_ptr(split_pre),
                        split_post: device_ptr(split_post),
                        split_comb: device_ptr(split_comb),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn hc_post_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            hidden: &DeviceBuffer<f32>,
            residual: &DeviceBuffer<f32>,
            split_post: &DeviceBuffer<f32>,
            split_comb: &DeviceBuffer<f32>,
            output: &mut DeviceBuffer<f32>,
            tokens: u32,
            hc: u32,
            hidden_size: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.hc(
                    stream,
                    HcArgs {
                        kind: HC_POST,
                        tokens,
                        hc,
                        hidden_size,
                        hidden: device_ptr(hidden),
                        residual: device_ptr(residual),
                        split_post: device_ptr(split_post),
                        split_comb: device_ptr(split_comb),
                        output: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn hc_mean_scatter_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            state: &DeviceBuffer<f32>,
            output: &mut DeviceBuffer<f32>,
            rows: u32,
            hc: u32,
            hidden_size: u32,
            tap_slot: u32,
            tap_count: u32,
        ) -> KernelResult<()> {
            unsafe {
                self.hc(
                    stream,
                    HcArgs {
                        kind: HC_MEAN_SCATTER,
                        tokens: rows,
                        hc,
                        hidden_size,
                        tap_slot,
                        tap_count,
                        state: device_ptr(state),
                        output: device_ptr(output),
                        ..Default::default()
                    },
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn hc_head_f32(
            &self,
            stream: &CudaStream,
            _config: LaunchConfig,
            state: &DeviceBuffer<f32>,
            function: &DeviceBuffer<f32>,
            scale: &DeviceBuffer<f32>,
            base: &DeviceBuffer<f32>,
            hidden: &mut DeviceBuffer<f32>,
            tokens: u32,
            hc: u32,
            hidden_size: u32,
            epsilon: f32,
            norm_epsilon: f32,
        ) -> KernelResult<()> {
            unsafe {
                self.hc(
                    stream,
                    HcArgs {
                        kind: HC_HEAD,
                        tokens,
                        hc,
                        hidden_size,
                        epsilon,
                        norm_epsilon,
                        state: device_ptr(state),
                        function: device_ptr(function),
                        scale: device_ptr(scale),
                        base: device_ptr(base),
                        hidden: device_ptr(hidden),
                        ..Default::default()
                    },
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    macro_rules! assert_pod_layout {
        ($type:ty, $size:expr, $first:ident, $first_u64:ident => $first_u64_offset:expr, $stream_offset:expr) => {{
            assert_eq!(size_of::<$type>(), $size);
            assert_eq!(align_of::<$type>(), 8);
            assert_eq!(offset_of!($type, $first), 0);
            assert_eq!(offset_of!($type, $first_u64), $first_u64_offset);
            assert_eq!(offset_of!($type, stream), $stream_offset);
        }};
    }

    #[test]
    fn core_pod_layouts_match_native_abi() {
        assert_pod_layout!(LinearArgs, 88, kind, x => 40, 80);
        assert_pod_layout!(DualLinearArgs, 104, kind, x => 40, 96);
        assert_pod_layout!(GroupedLinearArgs, 72, kind, input => 32, 64);
        assert_pod_layout!(QuantizeArgs, 64, kind, values => 32, 56);
        assert_pod_layout!(DataArgs, 136, kind, input0 => 56, 120);
        assert_pod_layout!(EmbeddingArgs, 64, kind, embedding => 32, 56);
        assert_pod_layout!(NormArgs, 56, kind, input => 24, 48);
        assert_pod_layout!(RopeArgs, 80, kind, values => 40, 72);
        assert_eq!(offset_of!(RopeArgs, inverse), 32);
        assert_eq!(offset_of!(RopeArgs, restore_bf16_boundary), 36);
        assert_pod_layout!(RouterArgs, 96, kind, logits => 40, 88);
        assert_pod_layout!(CompressorArgs, 96, kind, kv_input => 40, 88);
        assert_pod_layout!(IndexerArgs, 200, rows, query => 80, 192);

        assert_pod_layout!(ExpertTableArgs, 296, kind, gate_weights => 48, 288);
        assert_pod_layout!(ExpertGroupRoutePlanArgs, 184, kind, route_slots => 32, 176);
        assert_pod_layout!(MoeArgs, 248, kind, input => 48, 240);
        assert_pod_layout!(HcArgs, 136, kind, state => 48, 128);
        assert_pod_layout!(MlaArgs, 72, hidden_size, input => 24, 64);

        assert_eq!(offset_of!(DataArgs, output_elements), 128);
    }

    #[test]
    fn rope_inverse_restores_bf16_boundary() {
        let forward = RopeArgs::default().with_inverse(0);
        assert_eq!(forward.inverse, 0);
        assert_eq!(forward.restore_bf16_boundary, 0);

        let inverse = RopeArgs::default().with_inverse(1);
        assert_eq!(inverse.inverse, 1);
        assert_eq!(inverse.restore_bf16_boundary, 1);
    }

    #[test]
    fn expert_group_route_plan_layout_is_stable() {
        assert_eq!(offset_of!(ExpertTableArgs, expert_ids), 120);
        assert_eq!(offset_of!(ExpertTableArgs, route_slots), 128);
        assert_eq!(offset_of!(ExpertTableArgs, route_generations), 136);
        assert_eq!(offset_of!(ExpertTableArgs, miss_markers), 144);
        assert_eq!(offset_of!(ExpertTableArgs, output_gate_weights), 176);
        assert_eq!(offset_of!(ExpertTableArgs, gate_weight_value), 240);

        assert_eq!(offset_of!(ExpertGroupRoutePlanArgs, slot_route_offsets), 72);
        assert_eq!(
            offset_of!(ExpertGroupRoutePlanArgs, active_expert_slots),
            88
        );
        assert_eq!(
            offset_of!(ExpertGroupRoutePlanArgs, expert_route_indptr),
            104
        );
        assert_eq!(
            offset_of!(ExpertGroupRoutePlanArgs, route_token_indices),
            120
        );
        assert_eq!(offset_of!(ExpertGroupRoutePlanArgs, route_indices), 128);
        assert_eq!(offset_of!(ExpertGroupRoutePlanArgs, host_scalars), 144);
        assert_eq!(offset_of!(ExpertGroupRoutePlanArgs, route_output), 152);
        assert_eq!(offset_of!(ExpertGroupRoutePlanArgs, route_written), 160);
        assert_eq!(offset_of!(ExpertGroupRoutePlanArgs, route_error), 168);
    }
}
