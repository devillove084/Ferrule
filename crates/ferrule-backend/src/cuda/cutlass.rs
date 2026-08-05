//! CUDA CUTLASS semantic operator provider.
//!
//! Ferrule owns CUDA contexts, streams, allocations, execution plans, and
//! tensor lifetimes. The native boundary contains only POD arguments for
//! complete semantic operators; architecture and launch-policy selection remain
//! private to the native provider.

use crate::cuda::runtime::{CudaStream, DeviceBuffer};
use ferrule_common::{Error, Result};
#[cfg(ferrule_cuda_test_oracle)]
use std::sync::atomic::{AtomicU64, Ordering};

pub const PROPOSAL_ROWS: usize = 5;
pub const HYBRID_MLA_ATTENTION_HEADS: usize = 64;
pub const HYBRID_MLA_ATTENTION_HEAD_DIM: usize = 512;
pub const HYBRID_MLA_ATTENTION_WINDOW: usize = 128;
pub const HYBRID_MLA_ATTENTION_PAGE_TOKENS: usize = 16;
pub const HYBRID_MLA_ATTENTION_TOKEN_CAPACITY: usize = HYBRID_MLA_ATTENTION_WINDOW + PROPOSAL_ROWS;
pub const HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILE: usize = 64;
pub const HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILES: usize =
    HYBRID_MLA_ATTENTION_TOKEN_CAPACITY.div_ceil(HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILE);
pub const HYBRID_MLA_EXPLICIT_SELECTION_MAXIMUM_WIDTH: usize = 640;
#[cfg(ferrule_cuda_test_oracle)]
pub const HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE_RESULT_WORDS: usize = 5;
#[cfg(ferrule_cuda_test_oracle)]
static HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE_CALL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CutlassKernelId {
    Fp8QueryAKv = 1,
    Bf16Compressor = 2,
    HyperConnectionProducer = 3,
    SharedFfn = 4,
    GroupedFp4Moe = 5,
    MlaOutput = 6,
    MainProjectNorm = 7,
    HybridMlaAttention = 8,
    ProposalHead = 9,
    Fp8Projection = 10,
}

impl CutlassKernelId {
    pub const fn mask(self) -> u64 {
        1u64 << (self as u32 - 1)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Fp8QueryAKv => "fp8-query-a-kv",
            Self::Bf16Compressor => "bf16-compressor",
            Self::HyperConnectionProducer => "hyper-connection-producer",
            Self::SharedFfn => "shared-ffn",
            Self::GroupedFp4Moe => "grouped_fp4_moe",
            Self::MlaOutput => "mla-output",
            Self::MainProjectNorm => "main-project-norm",
            Self::HybridMlaAttention => "hybrid-mla-attention",
            Self::ProposalHead => "proposal-head",
            Self::Fp8Projection => "fp8-projection",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CutlassProviderManifest {
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

    pub fn execution_manifest(self) -> Result<crate::plan::ProviderManifest> {
        use crate::plan::{
            ExecutionModeSet, KernelOperation, KernelProviderId, OperationCapability,
            ProviderManifest,
        };

        let mut operations = [
            (KernelOperation::MlaQueryAKv, CutlassKernelId::Fp8QueryAKv),
            (KernelOperation::MlaQueryB, CutlassKernelId::Fp8Projection),
            (
                KernelOperation::MainCompressorProjection,
                CutlassKernelId::Bf16Compressor,
            ),
            (
                KernelOperation::IndexerCompressorProjection,
                CutlassKernelId::Bf16Compressor,
            ),
            (
                KernelOperation::AttentionHcPre,
                CutlassKernelId::HyperConnectionProducer,
            ),
            (
                KernelOperation::FeedForwardHcPre,
                CutlassKernelId::HyperConnectionProducer,
            ),
            (KernelOperation::MlaOutput, CutlassKernelId::MlaOutput),
            (KernelOperation::SharedFfn, CutlassKernelId::SharedFfn),
            (
                KernelOperation::MainProjectNorm,
                CutlassKernelId::MainProjectNorm,
            ),
            (
                KernelOperation::HybridMlaAttention,
                CutlassKernelId::HybridMlaAttention,
            ),
            (KernelOperation::ProposalHead, CutlassKernelId::ProposalHead),
        ]
        .into_iter()
        .filter_map(|(operation, kernel)| {
            self.supports(kernel).then_some(OperationCapability::new(
                operation,
                ExecutionModeSet::INFERENCE,
            ))
        })
        .collect::<Vec<_>>();
        if self.supports(CutlassKernelId::GroupedFp4Moe) {
            operations.push(OperationCapability::new(
                KernelOperation::GroupedFp4Moe,
                ExecutionModeSet::INFERENCE,
            ));
        }

        Ok(ProviderManifest::new(
            KernelProviderId::CUDA_CUTLASS,
            "cuda-cutlass",
            operations,
        ))
    }
}

/// Discover the semantic capabilities published by the native CUTLASS provider.
pub fn discover_provider() -> Result<CutlassProvider> {
    let manifest = unsafe { ffi::ferrule_cutlass_provider_manifest() };
    Ok(CutlassProvider { manifest })
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CutlassBf16CompressorArgs {
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
struct CutlassMainProjectNormArgs {
    rows: u32,
    input_size: u32,
    output_size: u32,
    scale_cols: u32,
    reserved0: u32,
    rms_eps: f32,
    reserved1: u32,
    input_f32: u64,
    activation_fp8: u64,
    activation_ue8m0: u64,
    weight_fp8: u64,
    weight_ue8m0: u64,
    norm_weight_f32: u64,
    inv_rms_f32: u64,
    output_f32: u64,
    stream: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CutlassHybridMlaAttentionArgs {
    block_rows: u32,
    heads: u32,
    head_dim: u32,
    sequence_tokens: u32,
    window_size: u32,
    page_tokens: u32,
    elements_per_token: u32,
    layer_index: u32,
    layer_count: u32,
    block_slot_offset: u32,
    block_slot_count: u32,
    softmax_scale: f32,
    reserved0: u32,
    context_plane_elements: u64,
    query_f32: u64,
    context_plane_f32: u64,
    block_kv_f32: u64,
    block_slots_i32: u64,
    attention_sink_f32: u64,
    query_bf16: u64,
    gathered_kv_bf16: u64,
    scores_f32: u64,
    probabilities_bf16: u64,
    online_rescales_f32: u64,
    denominators_f32: u64,
    output_f32: u64,
    status_i32: u64,
    stream: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct FerruleCutlassHybridMlaExplicitSelectionArgs {
    kind: u32,
    rows: u32,
    tokens_per_sequence: u32,
    kv_len: u32,
    heads: u32,
    head_dim: u32,
    selected_width: u32,
    page_tokens: u32,
    first_elements_per_token: u32,
    second_elements_per_token: u32,
    layer_index: u32,
    layer_count: u32,
    flags: u32,
    softmax_scale: f32,
    reserved0: u32,
    first_plane_elements: u64,
    second_plane_elements: u64,
    query_f32: u64,
    first_plane_f32: u64,
    second_plane_f32: u64,
    block_slots_i32: u64,
    block_offsets_i32: u64,
    sequence_kv_lens_i32: u64,
    second_sequence_kv_lens_i32: u64,
    row_sequence_ids_i32: u64,
    row_kv_lens_i32: u64,
    row_second_kv_lens_i32: u64,
    selected_indices_i32: u64,
    selectors_i32: u64,
    attention_sink_f32: u64,
    workspace: u64,
    workspace_bytes: u64,
    output_f32: u64,
    status_i32: u64,
    stream: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FerruleCutlassWorkspaceRequirements {
    pub bytes: u64,
    pub alignment: u32,
    pub reserved: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CutlassProposalHeadArgs {
    rows: u32,
    hc: u32,
    hidden: u32,
    vocab: u32,
    markov_rank: u32,
    partial_capacity: u32,
    reserved0: u32,
    hc_eps: f32,
    norm_eps: f32,
    hc_state_f32: u64,
    hc_function_f32: u64,
    hc_scale_f32: u64,
    hc_base_f32: u64,
    norm_weight_f32: u64,
    lm_head_bf16: u64,
    markov_w1_bf16: u64,
    markov_w2_bf16: u64,
    confidence_weight_bf16: u64,
    hidden_f32: u64,
    normalized_f32: u64,
    base_logits_f32: u64,
    partial_values_f32: u64,
    partial_indices_i32: u64,
    token_ids_i32: u64,
    confidence_f32: u64,
    status_i32: u64,
    stream: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProposalHeadLayout {
    pub rows: usize,
    pub hc: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub markov_rank: usize,
    pub partial_capacity: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridMlaAttentionLayout {
    pub sequence_tokens: usize,
    pub page_tokens: usize,
    pub elements_per_token: usize,
    pub layer_index: usize,
    pub layer_count: usize,
    pub block_slot_offset: usize,
    pub block_slot_count: usize,
    pub softmax_scale: f32,
}

/// KV storage topology for hybrid MLA with explicit selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum HybridMlaKvStorageKind {
    Contiguous = 1,
    Paged = 2,
    DualPaged = 3,
}

/// Dimensions and paging metadata for hybrid MLA with explicit selection.
/// Fields unused by a storage topology must be zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridMlaExplicitSelectionLayout {
    pub kind: HybridMlaKvStorageKind,
    pub rows: usize,
    pub tokens_per_sequence: usize,
    pub kv_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub selected_width: usize,
    pub page_tokens: usize,
    pub first_elements_per_token: usize,
    pub second_elements_per_token: usize,
    pub layer_index: usize,
    pub layer_count: usize,
    pub row_sequence_ids: bool,
    pub row_kv_lens: bool,
    pub softmax_scale: f32,
}

/// Caller-owned inputs, outputs, metadata, and opaque provider workspace for
/// hybrid MLA with explicit selection. Paged metadata is present only for paged
/// topologies; the second plane and selectors are present only for `DualPaged`.
pub struct HybridMlaExplicitSelectionBuffers<'a> {
    pub query: &'a DeviceBuffer<f32>,
    #[cfg(ferrule_cuda_test_oracle)]
    pub oracle_output: &'a mut DeviceBuffer<f32>,
    pub first_plane: &'a DeviceBuffer<f32>,
    pub second_plane: Option<&'a DeviceBuffer<f32>>,
    pub block_slots: Option<&'a DeviceBuffer<i32>>,
    pub block_offsets: Option<&'a DeviceBuffer<i32>>,
    pub sequence_kv_lens: Option<&'a DeviceBuffer<i32>>,
    pub second_sequence_kv_lens: Option<&'a DeviceBuffer<i32>>,
    pub row_sequence_ids: Option<&'a DeviceBuffer<i32>>,
    pub row_kv_lens: Option<&'a DeviceBuffer<i32>>,
    pub row_second_kv_lens: Option<&'a DeviceBuffer<i32>>,
    pub selected_indices: &'a DeviceBuffer<i32>,
    pub selectors: Option<&'a DeviceBuffer<i32>>,
    pub attention_sink: &'a DeviceBuffer<f32>,
    pub workspace: &'a mut DeviceBuffer<u8>,
    pub output: &'a mut DeviceBuffer<f32>,
    pub status: &'a mut DeviceBuffer<i32>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CutlassHcProducerArgs {
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
struct FerruleCutlassGroupedFp4MoeArgs {
    active_group_count: u32,
    small_group_count: u32,
    slot_capacity: u32,
    max_group_rows: u32,
    total_routed_rows: u32,
    num_tokens: u32,
    num_routes: u32,
    input_size: u32,
    intermediate_size: u32,
    hidden_size: u32,
    swiglu_limit: f32,
    active_expert_slots: u64,
    active_group_generations: u64,
    expert_route_indptr: u64,
    expert_route_counts: u64,
    route_token_indices: u64,
    route_indices: u64,
    route_weights: u64,
    slot_generations: u64,
    gate_ptrs: u64,
    gate_scale_ptrs: u64,
    up_ptrs: u64,
    up_scale_ptrs: u64,
    down_ptrs: u64,
    down_scale_ptrs: u64,
    input_packed: u64,
    input_scales: u64,
    route_output: u64,
    route_written: u64,
    route_error: u64,
    workspace: u64,
    workspace_bytes: u64,
    stream: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PrepareMxfp4SfbArgs {
    n: u32,
    k: u32,
    reserved0: u32,
    linear_source: u64,
    prepared_destination: u64,
    stream: u64,
}

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

    fn from_single_buffers(
        stream: &CudaStream,
        activation: &DeviceBuffer<u8>,
        activation_scales: &DeviceBuffer<u8>,
        weight: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        validate_fp8_projection_problem(
            activation,
            activation_scales,
            weight,
            weight_scales,
            output,
            rows,
            n,
            k,
        )?;
        let scale_cols = k / 128;
        Ok(Self {
            rows: checked_u32(rows, "rows")?,
            n1: checked_u32(n, "n")?,
            n2: 0,
            k: checked_u32(k, "k")?,
            scale_cols: checked_u32(scale_cols, "scale_cols")?,
            activation_fp8: activation.cu_deviceptr(),
            activation_ue8m0: activation_scales.cu_deviceptr(),
            query_a_weight_fp8: weight.cu_deviceptr(),
            query_a_weight_ue8m0: weight_scales.cu_deviceptr(),
            kv_weight_fp8: 0,
            kv_weight_ue8m0: 0,
            query_a_output_f32: output.cu_deviceptr(),
            kv_output_f32: 0,
            stream: stream.cu_stream() as usize as u64,
        })
    }
}

impl CutlassMainProjectNormArgs {
    #[allow(clippy::too_many_arguments)]
    fn from_buffers(
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
    ) -> Result<Self> {
        validate_main_project_norm_problem(
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
        )?;
        Ok(Self {
            rows: checked_u32(rows, "proposal rows")?,
            input_size: checked_u32(input_size, "proposal input size")?,
            output_size: checked_u32(output_size, "proposal output size")?,
            scale_cols: checked_u32(input_size / 128, "proposal scale columns")?,
            reserved0: 0,
            rms_eps,
            reserved1: 0,
            input_f32: input.cu_deviceptr(),
            activation_fp8: activation.cu_deviceptr(),
            activation_ue8m0: activation_scales.cu_deviceptr(),
            weight_fp8: weight.cu_deviceptr(),
            weight_ue8m0: weight_scales.cu_deviceptr(),
            norm_weight_f32: norm_weight.cu_deviceptr(),
            inv_rms_f32: inv_rms.cu_deviceptr(),
            output_f32: output.cu_deviceptr(),
            stream: stream.cu_stream() as usize as u64,
        })
    }
}

impl CutlassHybridMlaAttentionArgs {
    #[allow(clippy::too_many_arguments)]
    fn from_buffers(
        stream: &CudaStream,
        query: &DeviceBuffer<f32>,
        context_plane: &DeviceBuffer<f32>,
        block_kv: &DeviceBuffer<f32>,
        block_slots: &DeviceBuffer<i32>,
        attention_sink: &DeviceBuffer<f32>,
        query_bf16: &mut DeviceBuffer<u16>,
        gathered_kv_bf16: &mut DeviceBuffer<u16>,
        scores: &mut DeviceBuffer<f32>,
        probabilities_bf16: &mut DeviceBuffer<u16>,
        online_rescales: &mut DeviceBuffer<f32>,
        denominators: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        status: &mut DeviceBuffer<i32>,
        layout: HybridMlaAttentionLayout,
    ) -> Result<Self> {
        validate_hybrid_mla_attention_problem(
            query,
            context_plane,
            block_kv,
            block_slots,
            attention_sink,
            query_bf16,
            gathered_kv_bf16,
            scores,
            probabilities_bf16,
            online_rescales,
            denominators,
            output,
            status,
            layout,
        )?;
        Ok(Self {
            block_rows: PROPOSAL_ROWS as u32,
            heads: HYBRID_MLA_ATTENTION_HEADS as u32,
            head_dim: HYBRID_MLA_ATTENTION_HEAD_DIM as u32,
            sequence_tokens: checked_u32(layout.sequence_tokens, "proposal sequence tokens")?,
            window_size: HYBRID_MLA_ATTENTION_WINDOW as u32,
            page_tokens: checked_u32(layout.page_tokens, "proposal page tokens")?,
            elements_per_token: checked_u32(
                layout.elements_per_token,
                "proposal elements per token",
            )?,
            layer_index: checked_u32(layout.layer_index, "proposal layer index")?,
            layer_count: checked_u32(layout.layer_count, "proposal layer count")?,
            block_slot_offset: checked_u32(layout.block_slot_offset, "proposal block-slot offset")?,
            block_slot_count: checked_u32(layout.block_slot_count, "proposal block-slot count")?,
            softmax_scale: layout.softmax_scale,
            reserved0: 0,
            context_plane_elements: u64::try_from(context_plane.len()).map_err(|_| {
                Error::Internal {
                    message: "proposal context plane exceeds u64 ABI".into(),
                }
            })?,
            query_f32: query.cu_deviceptr(),
            context_plane_f32: context_plane.cu_deviceptr(),
            block_kv_f32: block_kv.cu_deviceptr(),
            block_slots_i32: block_slots.cu_deviceptr(),
            attention_sink_f32: attention_sink.cu_deviceptr(),
            query_bf16: query_bf16.cu_deviceptr(),
            gathered_kv_bf16: gathered_kv_bf16.cu_deviceptr(),
            scores_f32: scores.cu_deviceptr(),
            probabilities_bf16: probabilities_bf16.cu_deviceptr(),
            online_rescales_f32: online_rescales.cu_deviceptr(),
            denominators_f32: denominators.cu_deviceptr(),
            output_f32: output.cu_deviceptr(),
            status_i32: status.cu_deviceptr(),
            stream: stream.cu_stream() as usize as u64,
        })
    }
}

impl FerruleCutlassHybridMlaExplicitSelectionArgs {
    fn shape_only(layout: HybridMlaExplicitSelectionLayout) -> Result<Self> {
        Ok(Self {
            kind: layout.kind as u32,
            rows: checked_u32(layout.rows, "hybrid MLA explicit selection rows")?,
            tokens_per_sequence: checked_u32(
                layout.tokens_per_sequence,
                "hybrid MLA explicit selection tokens per sequence",
            )?,
            kv_len: checked_u32(layout.kv_len, "hybrid MLA explicit selection KV length")?,
            heads: checked_u32(layout.heads, "hybrid MLA explicit selection heads")?,
            head_dim: checked_u32(
                layout.head_dim,
                "hybrid MLA explicit selection head dimension",
            )?,
            selected_width: checked_u32(
                layout.selected_width,
                "hybrid MLA explicit selection width",
            )?,
            page_tokens: checked_u32(
                layout.page_tokens,
                "hybrid MLA explicit selection page tokens",
            )?,
            first_elements_per_token: checked_u32(
                layout.first_elements_per_token,
                "hybrid MLA explicit selection first elements per token",
            )?,
            second_elements_per_token: checked_u32(
                layout.second_elements_per_token,
                "hybrid MLA explicit selection second elements per token",
            )?,
            layer_index: checked_u32(
                layout.layer_index,
                "hybrid MLA explicit selection layer index",
            )?,
            layer_count: checked_u32(
                layout.layer_count,
                "hybrid MLA explicit selection layer count",
            )?,
            flags: u32::from(layout.row_sequence_ids) | (u32::from(layout.row_kv_lens) << 1),
            softmax_scale: layout.softmax_scale,
            reserved0: 0,
            first_plane_elements: 0,
            second_plane_elements: 0,
            query_f32: 0,
            first_plane_f32: 0,
            second_plane_f32: 0,
            block_slots_i32: 0,
            block_offsets_i32: 0,
            sequence_kv_lens_i32: 0,
            second_sequence_kv_lens_i32: 0,
            row_sequence_ids_i32: 0,
            row_kv_lens_i32: 0,
            row_second_kv_lens_i32: 0,
            selected_indices_i32: 0,
            selectors_i32: 0,
            attention_sink_f32: 0,
            workspace: 0,
            workspace_bytes: 0,
            output_f32: 0,
            status_i32: 0,
            stream: 0,
        })
    }

    fn from_buffers(
        stream: &CudaStream,
        buffers: &HybridMlaExplicitSelectionBuffers<'_>,
        layout: HybridMlaExplicitSelectionLayout,
    ) -> Result<Self> {
        validate_hybrid_mla_explicit_selection_contract(buffers, layout)?;

        let mut args = Self::shape_only(layout)?;
        args.flags = u32::from(buffers.row_sequence_ids.is_some())
            | (u32::from(buffers.row_kv_lens.is_some()) << 1);
        args.first_plane_elements = checked_u64(
            buffers.first_plane.len(),
            "hybrid MLA explicit selection first plane elements",
        )?;
        args.second_plane_elements = buffers.second_plane.map_or(Ok(0), |buffer| {
            checked_u64(
                buffer.len(),
                "hybrid MLA explicit selection second plane elements",
            )
        })?;
        args.query_f32 = buffers.query.cu_deviceptr();
        args.first_plane_f32 = buffers.first_plane.cu_deviceptr();
        args.second_plane_f32 = buffers.second_plane.map_or(0, DeviceBuffer::cu_deviceptr);
        args.block_slots_i32 = buffers.block_slots.map_or(0, DeviceBuffer::cu_deviceptr);
        args.block_offsets_i32 = buffers.block_offsets.map_or(0, DeviceBuffer::cu_deviceptr);
        args.sequence_kv_lens_i32 = buffers
            .sequence_kv_lens
            .map_or(0, DeviceBuffer::cu_deviceptr);
        args.second_sequence_kv_lens_i32 = buffers
            .second_sequence_kv_lens
            .map_or(0, DeviceBuffer::cu_deviceptr);
        args.row_sequence_ids_i32 = buffers
            .row_sequence_ids
            .map_or(0, DeviceBuffer::cu_deviceptr);
        args.row_kv_lens_i32 = buffers.row_kv_lens.map_or(0, DeviceBuffer::cu_deviceptr);
        args.row_second_kv_lens_i32 = buffers
            .row_second_kv_lens
            .map_or(0, DeviceBuffer::cu_deviceptr);
        args.selected_indices_i32 = buffers.selected_indices.cu_deviceptr();
        args.selectors_i32 = buffers.selectors.map_or(0, DeviceBuffer::cu_deviceptr);
        args.attention_sink_f32 = buffers.attention_sink.cu_deviceptr();
        args.workspace = buffers.workspace.cu_deviceptr();
        args.workspace_bytes = checked_u64(
            buffers.workspace.len(),
            "hybrid MLA explicit selection workspace bytes",
        )?;
        args.output_f32 = buffers.output.cu_deviceptr();
        args.status_i32 = buffers.status.cu_deviceptr();
        args.stream = stream.cu_stream() as usize as u64;
        Ok(args)
    }
}

impl CutlassProposalHeadArgs {
    #[allow(clippy::too_many_arguments)]
    fn from_buffers(
        stream: &CudaStream,
        hc_state: &DeviceBuffer<f32>,
        hc_function: &DeviceBuffer<f32>,
        hc_scale: &DeviceBuffer<f32>,
        hc_base: &DeviceBuffer<f32>,
        norm_weight: &DeviceBuffer<f32>,
        lm_head_bf16: &DeviceBuffer<u8>,
        markov_w1_bf16: &DeviceBuffer<u8>,
        markov_w2_bf16: &DeviceBuffer<u8>,
        confidence_weight_bf16: &DeviceBuffer<u8>,
        hidden: &mut DeviceBuffer<f32>,
        normalized: &mut DeviceBuffer<f32>,
        base_logits: &mut DeviceBuffer<f32>,
        partial_values: &mut DeviceBuffer<f32>,
        partial_indices: &mut DeviceBuffer<i32>,
        token_ids: &mut DeviceBuffer<i32>,
        confidence: &mut DeviceBuffer<f32>,
        status: &mut DeviceBuffer<i32>,
        layout: ProposalHeadLayout,
    ) -> Result<Self> {
        let hc_hidden = checked_mul(layout.hc, layout.hidden, "proposal proposal HC width")?;
        let hidden_values = checked_mul(layout.rows, layout.hidden, "proposal proposal hidden")?;
        let logits_values = checked_mul(layout.rows, layout.vocab, "proposal proposal logits")?;
        let markov_values =
            checked_mul(layout.vocab, layout.markov_rank, "proposal Markov weights")?;
        let required = [
            (
                "HC state",
                hc_state.len(),
                checked_mul(layout.rows, hc_hidden, "proposal HC state")?,
            ),
            (
                "HC function",
                hc_function.len(),
                checked_mul(layout.hc, hc_hidden, "proposal HC function")?,
            ),
            ("HC scale", hc_scale.len(), 1),
            ("HC base", hc_base.len(), layout.hc),
            ("norm weight", norm_weight.len(), layout.hidden),
            (
                "LM head bytes",
                lm_head_bf16.len(),
                checked_mul(
                    checked_mul(layout.vocab, layout.hidden, "proposal LM elements")?,
                    2,
                    "proposal LM bytes",
                )?,
            ),
            (
                "Markov W1 bytes",
                markov_w1_bf16.len(),
                checked_mul(markov_values, 2, "proposal W1 bytes")?,
            ),
            (
                "Markov W2 bytes",
                markov_w2_bf16.len(),
                checked_mul(markov_values, 2, "proposal W2 bytes")?,
            ),
            (
                "confidence bytes",
                confidence_weight_bf16.len(),
                checked_mul(
                    layout
                        .hidden
                        .checked_add(layout.markov_rank)
                        .ok_or_else(|| Error::Internal {
                            message: "proposal confidence width overflow".into(),
                        })?,
                    2,
                    "proposal confidence bytes",
                )?,
            ),
            ("hidden", hidden.len(), hidden_values),
            ("normalized", normalized.len(), hidden_values),
            ("base logits", base_logits.len(), logits_values),
            (
                "partial values",
                partial_values.len(),
                layout.partial_capacity,
            ),
            (
                "partial indices",
                partial_indices.len(),
                layout.partial_capacity,
            ),
            ("token ids", token_ids.len(), layout.rows + 1),
            ("confidence", confidence.len(), layout.rows),
            ("status", status.len(), 1),
        ];
        for (name, actual, expected) in required {
            if actual != expected {
                return Err(Error::Internal {
                    message: format!(
                        "proposal proposal-head {name} length mismatch: actual={actual} expected={expected}"
                    ),
                });
            }
        }
        if layout.rows != PROPOSAL_ROWS
            || layout.hc == 0
            || layout.hidden == 0
            || layout.vocab == 0
            || layout.markov_rank == 0
            || layout.partial_capacity == 0
            || !layout.hc_eps.is_finite()
            || !layout.norm_eps.is_finite()
            || layout.hc_eps <= 0.0
            || layout.norm_eps <= 0.0
        {
            return Err(Error::Internal {
                message: format!("invalid proposal proposal-head layout: {layout:?}"),
            });
        }
        Ok(Self {
            rows: checked_u32(layout.rows, "proposal rows")?,
            hc: checked_u32(layout.hc, "proposal proposal HC")?,
            hidden: checked_u32(layout.hidden, "proposal proposal hidden")?,
            vocab: checked_u32(layout.vocab, "proposal proposal vocab")?,
            markov_rank: checked_u32(layout.markov_rank, "proposal Markov rank")?,
            partial_capacity: checked_u32(layout.partial_capacity, "proposal partial capacity")?,
            reserved0: 0,
            hc_eps: layout.hc_eps,
            norm_eps: layout.norm_eps,
            hc_state_f32: hc_state.cu_deviceptr(),
            hc_function_f32: hc_function.cu_deviceptr(),
            hc_scale_f32: hc_scale.cu_deviceptr(),
            hc_base_f32: hc_base.cu_deviceptr(),
            norm_weight_f32: norm_weight.cu_deviceptr(),
            lm_head_bf16: lm_head_bf16.cu_deviceptr(),
            markov_w1_bf16: markov_w1_bf16.cu_deviceptr(),
            markov_w2_bf16: markov_w2_bf16.cu_deviceptr(),
            confidence_weight_bf16: confidence_weight_bf16.cu_deviceptr(),
            hidden_f32: hidden.cu_deviceptr(),
            normalized_f32: normalized.cu_deviceptr(),
            base_logits_f32: base_logits.cu_deviceptr(),
            partial_values_f32: partial_values.cu_deviceptr(),
            partial_indices_i32: partial_indices.cu_deviceptr(),
            token_ids_i32: token_ids.cu_deviceptr(),
            confidence_f32: confidence.cu_deviceptr(),
            status_i32: status.cu_deviceptr(),
            stream: stream.cu_stream() as usize as u64,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_hybrid_mla_attention_problem(
    query: &DeviceBuffer<f32>,
    context_plane: &DeviceBuffer<f32>,
    block_kv: &DeviceBuffer<f32>,
    block_slots: &DeviceBuffer<i32>,
    attention_sink: &DeviceBuffer<f32>,
    query_bf16: &DeviceBuffer<u16>,
    gathered_kv_bf16: &DeviceBuffer<u16>,
    scores: &DeviceBuffer<f32>,
    probabilities_bf16: &DeviceBuffer<u16>,
    online_rescales: &DeviceBuffer<f32>,
    denominators: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    status: &DeviceBuffer<i32>,
    layout: HybridMlaAttentionLayout,
) -> Result<()> {
    let output_values = checked_mul(
        checked_mul(
            PROPOSAL_ROWS,
            HYBRID_MLA_ATTENTION_HEADS,
            "proposal query rows/heads",
        )?,
        HYBRID_MLA_ATTENTION_HEAD_DIM,
        "proposal query values",
    )?;
    let score_values = checked_mul(
        checked_mul(
            PROPOSAL_ROWS,
            HYBRID_MLA_ATTENTION_HEADS,
            "proposal score rows/heads",
        )?,
        HYBRID_MLA_ATTENTION_TOKEN_CAPACITY,
        "proposal score values",
    )?;
    let pair_values = checked_mul(
        PROPOSAL_ROWS,
        HYBRID_MLA_ATTENTION_HEADS,
        "proposal row/head pairs",
    )?;
    let rescale_values = checked_mul(
        pair_values,
        HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILES,
        "proposal online-softmax rescales",
    )?;
    let block_values = checked_mul(
        PROPOSAL_ROWS,
        HYBRID_MLA_ATTENTION_HEAD_DIM,
        "proposal block KV",
    )?;
    let slot_end = layout
        .block_slot_offset
        .checked_add(layout.block_slot_count)
        .ok_or_else(|| Error::Internal {
            message: "proposal block-slot range overflow".into(),
        })?;
    let required_slots = layout
        .sequence_tokens
        .div_ceil(HYBRID_MLA_ATTENTION_PAGE_TOKENS);
    if layout.sequence_tokens == 0
        || layout.page_tokens != HYBRID_MLA_ATTENTION_PAGE_TOKENS
        || layout.elements_per_token != HYBRID_MLA_ATTENTION_HEAD_DIM
        || layout.layer_count == 0
        || layout.layer_index >= layout.layer_count
        || layout.block_slot_count < required_slots
        || slot_end > block_slots.len()
        || context_plane.is_empty()
        || !layout.softmax_scale.is_finite()
        || layout.softmax_scale <= 0.0
    {
        return Err(Error::Internal {
            message: format!(
                "invalid proposal hybrid-attention layout: {layout:?} slots={} plane={}",
                block_slots.len(),
                context_plane.len()
            ),
        });
    }
    let required = [
        ("query", query.len(), output_values),
        ("block KV", block_kv.len(), block_values),
        (
            "attention sink",
            attention_sink.len(),
            HYBRID_MLA_ATTENTION_HEADS,
        ),
        ("query BF16 scratch", query_bf16.len(), output_values),
        (
            "gathered KV BF16 scratch",
            gathered_kv_bf16.len(),
            HYBRID_MLA_ATTENTION_TOKEN_CAPACITY * HYBRID_MLA_ATTENTION_HEAD_DIM,
        ),
        ("score scratch", scores.len(), score_values),
        (
            "probability scratch",
            probabilities_bf16.len(),
            score_values,
        ),
        (
            "online-softmax rescale scratch",
            online_rescales.len(),
            rescale_values,
        ),
        (
            "softmax denominator scratch",
            denominators.len(),
            pair_values,
        ),
        ("output", output.len(), output_values),
        ("device status", status.len(), 1),
    ];
    for (name, actual, expected) in required {
        if actual != expected {
            return Err(Error::Internal {
                message: format!(
                    "proposal hybrid-attention {name} length mismatch: actual={actual} expected={expected}"
                ),
            });
        }
    }
    Ok(())
}

fn validate_hybrid_mla_explicit_selection_contract(
    buffers: &HybridMlaExplicitSelectionBuffers<'_>,
    layout: HybridMlaExplicitSelectionLayout,
) -> Result<()> {
    const HEADS: usize = 64;
    const HEAD_DIM: usize = 512;
    const MAXIMUM_SELECTED_WIDTH: usize = HYBRID_MLA_EXPLICIT_SELECTION_MAXIMUM_WIDTH;

    if layout.rows == 0
        || layout.heads != HEADS
        || layout.head_dim != HEAD_DIM
        || layout.selected_width == 0
        || layout.selected_width > MAXIMUM_SELECTED_WIDTH
        || !layout.softmax_scale.is_finite()
        || layout.softmax_scale <= 0.0
    {
        return Err(Error::Internal {
            message: format!("unsupported hybrid MLA explicit selection layout: {layout:?}"),
        });
    }

    if layout.row_sequence_ids != buffers.row_sequence_ids.is_some()
        || layout.row_kv_lens != buffers.row_kv_lens.is_some()
    {
        return Err(Error::Internal {
            message: "hybrid MLA explicit selection row metadata flags do not match buffers".into(),
        });
    }

    let paged_metadata = (
        buffers.block_slots,
        buffers.block_offsets,
        buffers.sequence_kv_lens,
    );
    match layout.kind {
        HybridMlaKvStorageKind::Contiguous => {
            if layout.tokens_per_sequence != 0
                || layout.kv_len == 0
                || layout.page_tokens != 0
                || layout.first_elements_per_token != 0
                || layout.second_elements_per_token != 0
                || layout.layer_index != 0
                || layout.layer_count != 0
                || buffers.second_plane.is_some()
                || paged_metadata.0.is_some()
                || paged_metadata.1.is_some()
                || paged_metadata.2.is_some()
                || buffers.second_sequence_kv_lens.is_some()
                || buffers.row_sequence_ids.is_some()
                || buffers.row_second_kv_lens.is_some()
                || buffers.selectors.is_some()
            {
                return Err(Error::Internal {
                    message: "invalid contiguous hybrid MLA explicit selection contract".into(),
                });
            }
            let first_plane_values = checked_mul(
                layout.kv_len,
                layout.head_dim,
                "hybrid MLA explicit selection contiguous plane",
            )?;
            validate_capacity(
                "hybrid MLA explicit selection",
                "first plane",
                buffers.first_plane.len(),
                first_plane_values,
            )?;
        }
        HybridMlaKvStorageKind::Paged | HybridMlaKvStorageKind::DualPaged => {
            if layout.kv_len != 0
                || layout.page_tokens == 0
                || layout.first_elements_per_token < layout.head_dim
                || layout.layer_count == 0
                || layout.layer_index >= layout.layer_count
                || buffers.first_plane.is_empty()
            {
                return Err(Error::Internal {
                    message: format!(
                        "invalid paged hybrid MLA explicit selection layout: {layout:?}"
                    ),
                });
            }
            let (Some(block_slots), Some(block_offsets), Some(sequence_kv_lens)) = paged_metadata
            else {
                return Err(Error::Internal {
                    message:
                        "paged hybrid MLA explicit selection requires complete paging metadata"
                            .into(),
                });
            };
            let expected_block_offsets =
                sequence_kv_lens
                    .len()
                    .checked_add(1)
                    .ok_or_else(|| Error::Internal {
                        message: "hybrid MLA explicit selection sequence count overflow".into(),
                    })?;
            if block_slots.is_empty()
                || sequence_kv_lens.is_empty()
                || block_offsets.len() != expected_block_offsets
                || (buffers.row_sequence_ids.is_some() && layout.tokens_per_sequence != 0)
                || (buffers.row_sequence_ids.is_none()
                    && (layout.tokens_per_sequence == 0
                        || sequence_kv_lens.len()
                            != layout.rows.div_ceil(layout.tokens_per_sequence)))
            {
                return Err(Error::Internal {
                    message: "invalid paged hybrid MLA explicit selection metadata lengths".into(),
                });
            }

            match layout.kind {
                HybridMlaKvStorageKind::Paged => {
                    if layout.second_elements_per_token != 0
                        || buffers.second_plane.is_some()
                        || buffers.second_sequence_kv_lens.is_some()
                        || buffers.row_second_kv_lens.is_some()
                        || buffers.selectors.is_some()
                    {
                        return Err(Error::Internal {
                            message:
                                "single-paged hybrid MLA explicit selection received dual-paged buffers"
                                    .into(),
                        });
                    }
                }
                HybridMlaKvStorageKind::DualPaged => {
                    if layout.second_elements_per_token < layout.head_dim
                        || buffers.second_plane.is_none_or(DeviceBuffer::is_empty)
                        || buffers.second_sequence_kv_lens.is_none()
                        || (layout.row_kv_lens && buffers.row_second_kv_lens.is_none())
                        || (!layout.row_kv_lens && buffers.row_second_kv_lens.is_some())
                        || buffers.selectors.is_none()
                    {
                        return Err(Error::Internal {
                            message: "dual-paged hybrid MLA explicit selection requires its second plane and selectors"
                                .into(),
                        });
                    }
                    let second_sequence_kv_lens = buffers
                        .second_sequence_kv_lens
                        .expect("validated dual-paged second sequence lengths");
                    if second_sequence_kv_lens.len() != sequence_kv_lens.len() {
                        return Err(Error::Internal {
                            message: "dual-paged hybrid MLA explicit selection sequence length metadata mismatch".into(),
                        });
                    }
                }
                HybridMlaKvStorageKind::Contiguous => unreachable!(),
            }
        }
    }

    let query_values = checked_mul(
        checked_mul(
            layout.rows,
            layout.heads,
            "hybrid MLA explicit selection query rows/heads",
        )?,
        layout.head_dim,
        "hybrid MLA explicit selection query values",
    )?;
    let selection_values = checked_mul(
        layout.rows,
        layout.selected_width,
        "hybrid MLA explicit selection selections",
    )?;

    validate_lengths(
        "hybrid MLA explicit selection",
        &[
            ("query", buffers.query.len(), query_values),
            (
                "selected indices",
                buffers.selected_indices.len(),
                selection_values,
            ),
            ("attention sink", buffers.attention_sink.len(), layout.heads),
            ("output", buffers.output.len(), query_values),
        ],
    )?;
    if let Some(row_sequence_ids) = buffers.row_sequence_ids {
        validate_lengths(
            "hybrid MLA explicit selection",
            &[("row sequence IDs", row_sequence_ids.len(), layout.rows)],
        )?;
    }
    if let Some(row_kv_lens) = buffers.row_kv_lens {
        validate_lengths(
            "hybrid MLA explicit selection",
            &[("row KV lengths", row_kv_lens.len(), layout.rows)],
        )?;
    }
    if let Some(row_second_kv_lens) = buffers.row_second_kv_lens {
        validate_lengths(
            "hybrid MLA explicit selection",
            &[(
                "row second-plane KV lengths",
                row_second_kv_lens.len(),
                layout.rows,
            )],
        )?;
    }
    if let Some(selectors) = buffers.selectors {
        validate_lengths(
            "hybrid MLA explicit selection",
            &[("selectors", selectors.len(), selection_values)],
        )?;
    }

    validate_capacity(
        "hybrid MLA explicit selection",
        "device status",
        buffers.status.len(),
        1,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_main_project_norm_problem(
    input: &DeviceBuffer<f32>,
    activation: &DeviceBuffer<u8>,
    activation_scales: &DeviceBuffer<u8>,
    weight: &DeviceBuffer<u8>,
    weight_scales: &DeviceBuffer<u8>,
    norm_weight: &DeviceBuffer<f32>,
    inv_rms: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    rms_eps: f32,
) -> Result<()> {
    if rows == 0
        || input_size == 0
        || output_size == 0
        || !input_size.is_multiple_of(128)
        || !output_size.is_multiple_of(128)
        || !rms_eps.is_finite()
        || rms_eps <= 0.0
    {
        return Err(Error::Internal {
            message: format!(
                "invalid proposal main-project/norm shape: rows={rows} input={input_size} output={output_size} eps={rms_eps}"
            ),
        });
    }
    let scale_cols = input_size / 128;
    let required = [
        (
            "input",
            input.len(),
            checked_mul(rows, input_size, "proposal input")?,
        ),
        (
            "activation",
            activation.len(),
            checked_mul(rows, input_size, "proposal activation")?,
        ),
        (
            "activation scales",
            activation_scales.len(),
            checked_mul(rows, scale_cols, "proposal activation scales")?,
        ),
        (
            "weight",
            weight.len(),
            checked_mul(output_size, input_size, "proposal weight")?,
        ),
        (
            "weight scales",
            weight_scales.len(),
            checked_mul(
                output_size.div_ceil(128),
                scale_cols,
                "proposal weight scales",
            )?,
        ),
        ("norm weight", norm_weight.len(), output_size),
        ("inverse RMS", inv_rms.len(), rows),
        (
            "output",
            output.len(),
            checked_mul(rows, output_size, "proposal output")?,
        ),
    ];
    for (name, actual, expected) in required {
        if actual != expected {
            return Err(Error::Internal {
                message: format!(
                    "proposal main-project/norm {name} length mismatch: actual={actual} expected={expected}"
                ),
            });
        }
    }
    Ok(())
}

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
        return Err(Error::Internal {
            message: format!("invalid BF16 compressor shape: rows={rows} n1={n1} n2={n2} k={k}"),
        });
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
            return Err(Error::Internal {
                message: format!(
                    "BF16 compressor {name} length mismatch: actual={actual} expected={expected}"
                ),
            });
        }
    }
    Ok(())
}

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
        return Err(Error::Internal {
            message: format!("invalid FP8 QueryA+KV shape: rows={rows} n1={n1} n2={n2} k={k}"),
        });
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
            return Err(Error::Internal {
                message: format!(
                    "FP8 QueryA+KV {name} length mismatch: actual={actual} expected={expected}"
                ),
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_fp8_projection_problem(
    activation: &DeviceBuffer<u8>,
    activation_scales: &DeviceBuffer<u8>,
    weight: &DeviceBuffer<u8>,
    weight_scales: &DeviceBuffer<u8>,
    output: &DeviceBuffer<f32>,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if rows == 0 || n == 0 || k == 0 || !k.is_multiple_of(128) {
        return Err(Error::Internal {
            message: format!("invalid FP8 projection shape: rows={rows} n={n} k={k}"),
        });
    }
    let scale_cols = k / 128;
    let activation_required = checked_mul(rows, k, "activation")?;
    let activation_scales_required = checked_mul(rows, scale_cols, "activation scales")?;
    if activation.len() < activation_required
        || activation_scales.len() < activation_scales_required
    {
        return Err(Error::Internal {
            message: format!(
                "FP8 projection scratch too small: activation={}/{} scales={}/{}",
                activation.len(),
                activation_required,
                activation_scales.len(),
                activation_scales_required
            ),
        });
    }
    let required = [
        ("weight", weight.len(), checked_mul(n, k, "weight")?),
        (
            "weight scales",
            weight_scales.len(),
            checked_mul(n.div_ceil(128), scale_cols, "weight scales")?,
        ),
        ("output", output.len(), checked_mul(rows, n, "output")?),
    ];
    for (name, actual, expected) in required {
        if actual != expected {
            return Err(Error::Internal {
                message: format!(
                    "FP8 projection {name} length mismatch: actual={actual} expected={expected}"
                ),
            });
        }
    }
    Ok(())
}

fn checked_mul(lhs: usize, rhs: usize, name: &str) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| Error::Internal {
        message: format!("FP8 {name} size overflow"),
    })
}

fn checked_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Internal {
        message: format!("FP8 {name} exceeds u32"),
    })
}

fn checked_u64(value: usize, name: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::Internal {
        message: format!("{name} exceeds u64"),
    })
}

/// Launch one semantic BF16 compressor bundle. The native CUDA provider owns
/// small-M versus tiled schedule selection.
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
        return Err(native_error("validate BF16 compressor", can_implement));
    }
    let status = unsafe { ffi::ferrule_cutlass_bf16_compressor_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch BF16 compressor", status))
    }
}

/// Launch one semantic FP8 QueryA+KV bundle. The executable plan does not bind
/// an M bucket or expose a native schedule variant.
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
        return Err(native_error("validate FP8 QueryA+KV", can_implement));
    }
    let status = unsafe { ffi::ferrule_cutlass_fp8_query_a_kv_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch FP8 QueryA+KV", status))
    }
}

/// Launch one small-M FP8 projection through the native pipeline.
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
    let args = CutlassFp8QueryAKvArgs::from_single_buffers(
        stream,
        activation,
        activation_scales,
        weight,
        weight_scales,
        output,
        rows,
        n,
        k,
    )?;
    let can_implement = unsafe { ffi::ferrule_cutlass_fp8_projection_can_implement(&args) };
    if can_implement != status::SUCCESS {
        return Err(native_error("validate FP8 projection", can_implement));
    }
    let status = unsafe { ffi::ferrule_cutlass_fp8_projection_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch FP8 projection", status))
    }
}

/// Launch the checkpoint-native proposal stage-zero target-tap projection and
/// RMSNorm as one cooperative semantic operation.
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
    let args = CutlassMainProjectNormArgs::from_buffers(
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
    )?;
    let can_implement = unsafe { ffi::ferrule_cutlass_main_project_norm_can_implement(&args) };
    if can_implement != status::SUCCESS {
        return Err(native_error(
            "validate proposal main-project/norm",
            can_implement,
        ));
    }
    let status = unsafe { ffi::ferrule_cutlass_main_project_norm_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch proposal main-project/norm", status))
    }
}

/// Launch checkpoint-native proposal attention over committed paged context plus
/// the complete read-only five-row proposal block.
#[allow(clippy::too_many_arguments)]
pub fn hybrid_mla_attention(
    stream: &CudaStream,
    query: &DeviceBuffer<f32>,
    context_plane: &DeviceBuffer<f32>,
    block_kv: &DeviceBuffer<f32>,
    block_slots: &DeviceBuffer<i32>,
    attention_sink: &DeviceBuffer<f32>,
    query_bf16: &mut DeviceBuffer<u16>,
    gathered_kv_bf16: &mut DeviceBuffer<u16>,
    scores: &mut DeviceBuffer<f32>,
    probabilities_bf16: &mut DeviceBuffer<u16>,
    online_rescales: &mut DeviceBuffer<f32>,
    denominators: &mut DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    status: &mut DeviceBuffer<i32>,
    layout: HybridMlaAttentionLayout,
) -> Result<()> {
    let args = CutlassHybridMlaAttentionArgs::from_buffers(
        stream,
        query,
        context_plane,
        block_kv,
        block_slots,
        attention_sink,
        query_bf16,
        gathered_kv_bf16,
        scores,
        probabilities_bf16,
        online_rescales,
        denominators,
        output,
        status,
        layout,
    )?;
    let can_implement = unsafe { ffi::ferrule_cutlass_hybrid_mla_attention_can_implement(&args) };
    if can_implement != status::SUCCESS {
        return Err(native_error(
            "validate proposal hybrid MLA attention",
            can_implement,
        ));
    }
    let launch = unsafe { ffi::ferrule_cutlass_hybrid_mla_attention_launch(&args) };
    if launch == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch proposal hybrid MLA attention", launch))
    }
}

/// Query the opaque workspace required by a hybrid MLA explicit-selection layout.
pub fn hybrid_mla_explicit_selection_workspace_requirements(
    layout: HybridMlaExplicitSelectionLayout,
) -> Result<FerruleCutlassWorkspaceRequirements> {
    let args = FerruleCutlassHybridMlaExplicitSelectionArgs::shape_only(layout)?;
    let mut requirements = FerruleCutlassWorkspaceRequirements {
        bytes: 0,
        alignment: 0,
        reserved: 0,
    };
    let status = unsafe {
        ffi::ferrule_cutlass_hybrid_mla_explicit_selection_workspace_requirements(
            &args,
            &mut requirements,
        )
    };
    native_result(
        "query hybrid MLA explicit selection workspace requirements",
        status,
    )?;
    if requirements.alignment == 0 || !requirements.alignment.is_power_of_two() {
        return Err(Error::Internal {
            message: format!(
                "hybrid MLA explicit selection workspace returned invalid alignment {}",
                requirements.alignment
            ),
        });
    }
    Ok(requirements)
}

/// Validate hybrid MLA explicit selection without launching. Unsupported geometry
/// fails closed rather than selecting another attention implementation.
pub fn hybrid_mla_explicit_selection_can_implement(
    stream: &CudaStream,
    buffers: &HybridMlaExplicitSelectionBuffers<'_>,
    layout: HybridMlaExplicitSelectionLayout,
) -> Result<()> {
    let args = FerruleCutlassHybridMlaExplicitSelectionArgs::from_buffers(stream, buffers, layout)?;
    let status = unsafe { ffi::ferrule_cutlass_hybrid_mla_explicit_selection_can_implement(&args) };
    native_result("validate hybrid MLA explicit selection", status)
}

/// Launch hybrid MLA explicit selection. No fallback is attempted.
pub fn hybrid_mla_explicit_selection_launch(
    stream: &CudaStream,
    buffers: &mut HybridMlaExplicitSelectionBuffers<'_>,
    layout: HybridMlaExplicitSelectionLayout,
) -> Result<()> {
    let requirements = hybrid_mla_explicit_selection_workspace_requirements(layout)?;
    let workspace_bytes = checked_u64(
        buffers.workspace.len(),
        "hybrid MLA explicit selection workspace bytes",
    )?;
    if workspace_bytes < requirements.bytes
        || !buffers
            .workspace
            .cu_deviceptr()
            .is_multiple_of(u64::from(requirements.alignment))
    {
        return Err(Error::Internal {
            message: format!(
                "hybrid MLA explicit selection workspace mismatch: bytes={workspace_bytes}/{} pointer=0x{:x} alignment={}",
                requirements.bytes,
                buffers.workspace.cu_deviceptr(),
                requirements.alignment,
            ),
        });
    }
    let args = FerruleCutlassHybridMlaExplicitSelectionArgs::from_buffers(stream, buffers, layout)?;
    #[cfg(ferrule_cuda_test_oracle)]
    let compare = std::env::var("FERRULE_CUDA_HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE")
        .is_ok_and(|value| value == "1");
    #[cfg(ferrule_cuda_test_oracle)]
    if !compare
        && std::env::var("FERRULE_CUDA_HYBRID_MLA_EXPLICIT_SELECTION_TEST_ORACLE")
            .is_ok_and(|value| value == "1")
    {
        let status =
            unsafe { ffi::ferrule_cutlass_test_hybrid_mla_explicit_selection_scalar_launch(&args) };
        return native_result("launch hybrid MLA explicit selection test oracle", status);
    }
    let can_implement =
        unsafe { ffi::ferrule_cutlass_hybrid_mla_explicit_selection_can_implement(&args) };
    native_result("validate hybrid MLA explicit selection", can_implement)?;
    let status = unsafe { ffi::ferrule_cutlass_hybrid_mla_explicit_selection_launch(&args) };
    native_result("launch hybrid MLA explicit selection", status)?;

    #[cfg(ferrule_cuda_test_oracle)]
    if compare {
        let call = HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE_CALL_SEQUENCE
            .fetch_add(1, Ordering::Relaxed)
            + 1;

        let output_values = layout
            .rows
            .checked_mul(layout.heads)
            .and_then(|value| value.checked_mul(layout.head_dim))
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "hybrid MLA explicit selection test compare call {call} layer {} output size overflow",
                    layout.layer_index
                ),
            })?;
        if buffers.oracle_output.len() < output_values {
            return Err(Error::Internal {
                message: format!(
                    "hybrid MLA explicit selection test compare call {call} layer {} oracle output capacity mismatch: oracle_output={} output={output_values} selected_width={} heads={}",
                    layout.layer_index,
                    buffers.oracle_output.len(),
                    layout.selected_width,
                    layout.heads,
                ),
            });
        }
        let temporary_result = (buffers.status.len()
            < HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE_RESULT_WORDS)
            .then(|| {
                DeviceBuffer::<i32>::zeroed(
                    stream,
                    HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE_RESULT_WORDS,
                )
                .map_err(|source| Error::Backend {
                    source: Box::new(source),
                })
            })
            .transpose()?;
        let compare_result = temporary_result.as_ref().unwrap_or(&*buffers.status);
        let compare_status = unsafe {
            ffi::ferrule_cutlass_test_hybrid_mla_explicit_selection_compare_launch(
                &args,
                buffers.oracle_output.cu_deviceptr(),
                compare_result.cu_deviceptr(),
            )
        };
        native_result(
            "launch hybrid MLA explicit selection test comparator",
            compare_status,
        )?;
        let result = compare_result
            .to_host_vec(stream)
            .map_err(|source| Error::Backend {
                source: Box::new(source),
            })?;
        let mismatch_count = result[0] as u32;
        if mismatch_count != 0 {
            let first_index = result[1] as u32;
            let max_abs = f32::from_bits(result[2] as u32);
            let first_actual = f32::from_bits(result[3] as u32);
            let first_expected = f32::from_bits(result[4] as u32);
            let message = format!(
                "hybrid MLA explicit selection test compare mismatch: call={call} layer={} kind={:?} rows={} width={} mismatches={mismatch_count} first_index={first_index} max_abs={max_abs:e} first_actual={first_actual:e} (0x{:08x}) first_expected={first_expected:e} (0x{:08x})",
                layout.layer_index,
                layout.kind,
                layout.rows,
                layout.selected_width,
                result[3] as u32,
                result[4] as u32,
            );
            eprintln!("{message}");
            return Err(Error::Internal { message });
        }
    }
    Ok(())
}

/// Launch the checkpoint-native proposal HC/LM/Markov/confidence proposal head.
#[allow(clippy::too_many_arguments)]
pub fn proposal_head(
    stream: &CudaStream,
    hc_state: &DeviceBuffer<f32>,
    hc_function: &DeviceBuffer<f32>,
    hc_scale: &DeviceBuffer<f32>,
    hc_base: &DeviceBuffer<f32>,
    norm_weight: &DeviceBuffer<f32>,
    lm_head_bf16: &DeviceBuffer<u8>,
    markov_w1_bf16: &DeviceBuffer<u8>,
    markov_w2_bf16: &DeviceBuffer<u8>,
    confidence_weight_bf16: &DeviceBuffer<u8>,
    hidden: &mut DeviceBuffer<f32>,
    normalized: &mut DeviceBuffer<f32>,
    base_logits: &mut DeviceBuffer<f32>,
    partial_values: &mut DeviceBuffer<f32>,
    partial_indices: &mut DeviceBuffer<i32>,
    token_ids: &mut DeviceBuffer<i32>,
    confidence: &mut DeviceBuffer<f32>,
    status: &mut DeviceBuffer<i32>,
    layout: ProposalHeadLayout,
) -> Result<()> {
    let args = CutlassProposalHeadArgs::from_buffers(
        stream,
        hc_state,
        hc_function,
        hc_scale,
        hc_base,
        norm_weight,
        lm_head_bf16,
        markov_w1_bf16,
        markov_w2_bf16,
        confidence_weight_bf16,
        hidden,
        normalized,
        base_logits,
        partial_values,
        partial_indices,
        token_ids,
        confidence,
        status,
        layout,
    )?;
    let can_implement = unsafe { ffi::ferrule_cutlass_proposal_head_can_implement(&args) };
    if can_implement != status::SUCCESS {
        return Err(native_error("validate proposal head", can_implement));
    }
    let launch = unsafe { ffi::ferrule_cutlass_proposal_head_launch(&args) };
    if launch == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch proposal head", launch))
    }
}

/// Launch the complete HC-pre + layer RMSNorm + FP8 producer bundle.
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
        return Err(Error::Internal {
            message: format!(
                "invalid HC producer parameters: rows={rows} hc={hc} hidden={hidden} sinkhorn={sinkhorn_iters}"
            ),
        });
    }
    validate_lengths("HC producer", &required)?;
    let args = CutlassHcProducerArgs {
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
        return Err(native_error("validate HC producer", can_implement));
    }
    let status = unsafe { ffi::ferrule_cutlass_hc_producer_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch HC producer", status))
    }
}

/// Launch the complete shared gate/up -> SwiGLU -> down bundle.
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
        return Err(Error::Internal {
            message: format!(
                "invalid shared FFN parameters: rows={rows} output_scale={output_scale} swiglu_limit={swiglu_limit}"
            ),
        });
    }
    validate_lengths("shared FFN", &required)?;
    let hidden_values = checked_mul(rows, intermediate_size, "shared FFN hidden F32")?;
    if hidden_f32.len() < hidden_values {
        return Err(Error::Internal {
            message: format!(
                "shared FFN hidden F32 capacity is too small: actual={} required={hidden_values}",
                hidden_f32.len()
            ),
        });
    }
    let args = CutlassSharedFfnArgs {
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
        return Err(native_error("validate shared FFN", can_implement));
    }
    let status = unsafe { ffi::ferrule_cutlass_shared_ffn_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch shared FFN", status))
    }
}

/// Launch grouped output-A -> BF16 boundary -> FP8 pack -> output-B as one MLA bundle.
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
        return Err(Error::Internal {
            message: format!(
                "invalid MLA output shape: rows={rows} context={context_size} groups={groups} group_input={group_input_size} rank={rank} latent={latent_size} hidden={hidden_size}"
            ),
        });
    }
    validate_lengths("MLA output", &required)?;
    let latent_values = checked_mul(rows, latent_size, "MLA output latent FP8")?;
    let latent_scale_values = checked_mul(rows, latent_size / 128, "MLA output latent scales")?;

    if latent_fp8.len() < latent_values || latent_scales.len() < latent_scale_values {
        return Err(Error::Internal {
            message: format!(
                "MLA output scratch is too small: latent_fp8={}/{} latent_scales={}/{}",
                latent_fp8.len(),
                latent_values,
                latent_scales.len(),
                latent_scale_values
            ),
        });
    }

    let args = CutlassMlaOutputArgs {
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
        return Err(native_error("validate MLA output", can_implement));
    }
    let status = unsafe { ffi::ferrule_cutlass_mla_output_launch(&args) };
    if status == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error("launch MLA output", status))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupedFp4MoeLayout {
    pub active_group_count: usize,
    pub small_group_count: usize,
    pub slot_capacity: usize,
    pub max_group_rows: usize,
    pub total_routed_rows: usize,
    pub num_tokens: usize,
    pub num_routes: usize,
    pub input_size: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub swiglu_limit: f32,
}

pub struct GroupedFp4MoeBuffers<'a> {
    pub active_expert_slots: &'a DeviceBuffer<i32>,
    pub active_group_generations: &'a DeviceBuffer<i32>,
    pub expert_route_indptr: &'a DeviceBuffer<i32>,
    pub expert_route_counts: &'a DeviceBuffer<i32>,
    pub route_token_indices: &'a DeviceBuffer<i32>,
    pub route_indices: &'a DeviceBuffer<i32>,
    pub route_weights: &'a DeviceBuffer<f32>,
    pub slot_generations: &'a DeviceBuffer<i32>,
    pub gate_ptrs: &'a DeviceBuffer<u64>,
    pub gate_scale_ptrs: &'a DeviceBuffer<u64>,
    pub up_ptrs: &'a DeviceBuffer<u64>,
    pub up_scale_ptrs: &'a DeviceBuffer<u64>,
    pub down_ptrs: &'a DeviceBuffer<u64>,
    pub down_scale_ptrs: &'a DeviceBuffer<u64>,
    pub input_packed: &'a DeviceBuffer<u8>,
    pub input_scales: &'a DeviceBuffer<u8>,
    pub route_output: &'a mut DeviceBuffer<f32>,
    pub route_written: &'a mut DeviceBuffer<i32>,
    pub route_error: &'a mut DeviceBuffer<i32>,
    pub workspace: &'a mut DeviceBuffer<u8>,
}

impl FerruleCutlassGroupedFp4MoeArgs {
    fn for_workspace_query(layout: GroupedFp4MoeLayout) -> Result<Self> {
        validate_grouped_fp4_moe_layout(layout)?;
        Ok(Self {
            active_group_count: checked_u32(layout.active_group_count, "active_group_count")?,
            small_group_count: checked_u32(layout.small_group_count, "small_group_count")?,
            slot_capacity: checked_u32(layout.slot_capacity, "slot_capacity")?,
            max_group_rows: checked_u32(layout.max_group_rows, "max_group_rows")?,
            total_routed_rows: checked_u32(layout.total_routed_rows, "total_routed_rows")?,
            num_tokens: checked_u32(layout.num_tokens, "num_tokens")?,
            num_routes: checked_u32(layout.num_routes, "num_routes")?,
            input_size: checked_u32(layout.input_size, "input_size")?,
            intermediate_size: checked_u32(layout.intermediate_size, "intermediate_size")?,
            hidden_size: checked_u32(layout.hidden_size, "hidden_size")?,
            swiglu_limit: layout.swiglu_limit,
            active_expert_slots: 0,
            active_group_generations: 0,
            expert_route_indptr: 0,
            expert_route_counts: 0,
            route_token_indices: 0,
            route_indices: 0,
            route_weights: 0,
            slot_generations: 0,
            gate_ptrs: 0,
            gate_scale_ptrs: 0,
            up_ptrs: 0,
            up_scale_ptrs: 0,
            down_ptrs: 0,
            down_scale_ptrs: 0,
            input_packed: 0,
            input_scales: 0,
            route_output: 0,
            route_written: 0,
            route_error: 0,
            workspace: 0,
            workspace_bytes: 0,
            stream: 0,
        })
    }

    fn from_buffers(
        stream: &CudaStream,
        buffers: &GroupedFp4MoeBuffers<'_>,
        layout: GroupedFp4MoeLayout,
    ) -> Result<Self> {
        let mut args = Self::for_workspace_query(layout)?;
        let active_indptr =
            layout
                .active_group_count
                .checked_add(1)
                .ok_or_else(|| Error::Internal {
                    message: "grouped FP4 MoE active indptr length overflow".into(),
                })?;
        let required = [
            (
                "active expert slots",
                buffers.active_expert_slots.len(),
                layout.active_group_count,
            ),
            (
                "active group generations",
                buffers.active_group_generations.len(),
                layout.active_group_count,
            ),
            (
                "expert route indptr",
                buffers.expert_route_indptr.len(),
                active_indptr,
            ),
            (
                "expert route counts",
                buffers.expert_route_counts.len(),
                layout.active_group_count,
            ),
            (
                "route token indices",
                buffers.route_token_indices.len(),
                layout.total_routed_rows,
            ),
            (
                "route indices",
                buffers.route_indices.len(),
                layout.total_routed_rows,
            ),
            (
                "route weights",
                buffers.route_weights.len(),
                layout.total_routed_rows,
            ),
            (
                "slot generations",
                buffers.slot_generations.len(),
                layout.slot_capacity,
            ),
            (
                "gate pointers",
                buffers.gate_ptrs.len(),
                layout.slot_capacity,
            ),
            (
                "gate scale pointers",
                buffers.gate_scale_ptrs.len(),
                layout.slot_capacity,
            ),
            ("up pointers", buffers.up_ptrs.len(), layout.slot_capacity),
            (
                "up scale pointers",
                buffers.up_scale_ptrs.len(),
                layout.slot_capacity,
            ),
            (
                "down pointers",
                buffers.down_ptrs.len(),
                layout.slot_capacity,
            ),
            (
                "down scale pointers",
                buffers.down_scale_ptrs.len(),
                layout.slot_capacity,
            ),
            (
                "packed input",
                buffers.input_packed.len(),
                checked_mul(
                    layout.num_tokens,
                    layout.input_size / 2,
                    "grouped FP4 input",
                )?,
            ),
            (
                "input scales",
                buffers.input_scales.len(),
                checked_mul(
                    layout.num_tokens,
                    layout.input_size / 32,
                    "grouped FP4 input scales",
                )?,
            ),
            (
                "route output",
                buffers.route_output.len(),
                checked_mul(
                    layout.num_routes,
                    layout.hidden_size,
                    "grouped FP4 route output",
                )?,
            ),
            (
                "route written",
                buffers.route_written.len(),
                layout.num_routes,
            ),
            ("route error", buffers.route_error.len(), 1),
        ];
        validate_lengths("grouped FP4 MoE", &required)?;

        let workspace_required = grouped_fp4_moe_workspace_size(layout)?;
        if buffers.workspace.len() < workspace_required {
            return Err(Error::Internal {
                message: format!(
                    "grouped FP4 MoE workspace is too small: actual={} required={workspace_required}",
                    buffers.workspace.len()
                ),
            });
        }

        args.active_expert_slots = buffers.active_expert_slots.cu_deviceptr();
        args.active_group_generations = buffers.active_group_generations.cu_deviceptr();
        args.expert_route_indptr = buffers.expert_route_indptr.cu_deviceptr();
        args.expert_route_counts = buffers.expert_route_counts.cu_deviceptr();
        args.route_token_indices = buffers.route_token_indices.cu_deviceptr();
        args.route_indices = buffers.route_indices.cu_deviceptr();
        args.route_weights = buffers.route_weights.cu_deviceptr();
        args.slot_generations = buffers.slot_generations.cu_deviceptr();
        args.gate_ptrs = buffers.gate_ptrs.cu_deviceptr();
        args.gate_scale_ptrs = buffers.gate_scale_ptrs.cu_deviceptr();
        args.up_ptrs = buffers.up_ptrs.cu_deviceptr();
        args.up_scale_ptrs = buffers.up_scale_ptrs.cu_deviceptr();
        args.down_ptrs = buffers.down_ptrs.cu_deviceptr();
        args.down_scale_ptrs = buffers.down_scale_ptrs.cu_deviceptr();
        args.input_packed = buffers.input_packed.cu_deviceptr();
        args.input_scales = buffers.input_scales.cu_deviceptr();
        args.route_output = buffers.route_output.cu_deviceptr();
        args.route_written = buffers.route_written.cu_deviceptr();
        args.route_error = buffers.route_error.cu_deviceptr();
        args.workspace = buffers.workspace.cu_deviceptr();
        args.workspace_bytes =
            u64::try_from(buffers.workspace.len()).map_err(|_| Error::Internal {
                message: "grouped FP4 MoE workspace length exceeds u64".into(),
            })?;
        args.stream = stream.cu_stream() as usize as u64;
        Ok(args)
    }
}

fn validate_grouped_fp4_moe_layout(layout: GroupedFp4MoeLayout) -> Result<()> {
    if layout.active_group_count == 0
        || layout.active_group_count > layout.slot_capacity
        || layout.small_group_count > layout.active_group_count
        || layout.max_group_rows == 0
        || layout.max_group_rows > layout.total_routed_rows
        || layout.total_routed_rows == 0
        || layout.total_routed_rows > layout.num_routes
        || layout.num_tokens == 0
        || layout.num_routes == 0
        || layout.input_size == 0
        || !layout.input_size.is_multiple_of(32)
        || layout.intermediate_size == 0
        || !layout.intermediate_size.is_multiple_of(32)
        || layout.hidden_size == 0
        || !layout.swiglu_limit.is_finite()
    {
        return Err(Error::Internal {
            message: format!("invalid grouped FP4 MoE layout: {layout:?}"),
        });
    }
    Ok(())
}

/// Return the caller-owned workspace required by a grouped FP4 MoE launch.
pub fn grouped_fp4_moe_workspace_size(layout: GroupedFp4MoeLayout) -> Result<usize> {
    let args = FerruleCutlassGroupedFp4MoeArgs::for_workspace_query(layout)?;
    let bytes = unsafe { ffi::ferrule_cutlass_grouped_fp4_moe_workspace_size(&args) };
    usize::try_from(bytes).map_err(|_| Error::Internal {
        message: format!("grouped FP4 MoE workspace size exceeds usize: {bytes}"),
    })
}

/// Validate a grouped FP4 MoE problem without launching it.
pub fn grouped_fp4_moe_can_implement(
    stream: &CudaStream,
    buffers: &GroupedFp4MoeBuffers<'_>,
    layout: GroupedFp4MoeLayout,
) -> Result<()> {
    let args = FerruleCutlassGroupedFp4MoeArgs::from_buffers(stream, buffers, layout)?;
    let status = unsafe { ffi::ferrule_cutlass_grouped_fp4_moe_can_implement(&args) };
    native_result("validate grouped FP4 MoE", status)
}

/// Launch the complete compact grouped FP4 MoE expert pipeline.
pub fn grouped_fp4_moe_launch(
    stream: &CudaStream,
    buffers: &mut GroupedFp4MoeBuffers<'_>,
    layout: GroupedFp4MoeLayout,
) -> Result<()> {
    let args = FerruleCutlassGroupedFp4MoeArgs::from_buffers(stream, buffers, layout)?;
    let can_implement = unsafe { ffi::ferrule_cutlass_grouped_fp4_moe_can_implement(&args) };
    native_result("validate grouped FP4 MoE", can_implement)?;
    let status = unsafe { ffi::ferrule_cutlass_grouped_fp4_moe_launch(&args) };
    native_result("launch grouped FP4 MoE", status)
}

/// Return the storage required for one prepared CUTLASS MXFP4 SFB scale tensor.
pub fn mxfp4_sfb_storage_bytes(n: usize, k: usize) -> Result<usize> {
    if n == 0 || k == 0 || !k.is_multiple_of(32) {
        return Err(Error::Internal {
            message: format!("invalid MXFP4 SFB shape: n={n} k={k}"),
        });
    }
    let bytes = unsafe {
        ffi::ferrule_cutlass_mxfp4_sfb_storage_bytes(
            checked_u32(n, "MXFP4 SFB n")?,
            checked_u32(k, "MXFP4 SFB k")?,
        )
    };
    usize::try_from(bytes).map_err(|_| Error::Internal {
        message: format!("MXFP4 SFB storage size exceeds usize: {bytes}"),
    })
}

/// Prepare linear UE8M0 scales into CUTLASS MXFP4 SFB layout asynchronously.
pub fn prepare_mxfp4_sfb(
    stream: &CudaStream,
    linear_source: &DeviceBuffer<u8>,
    prepared_destination: &mut DeviceBuffer<u8>,
    n: usize,
    k: usize,
) -> Result<()> {
    let source_bytes = checked_mul(n, k / 32, "MXFP4 linear scale storage")?;
    let destination_bytes = mxfp4_sfb_storage_bytes(n, k)?;
    validate_lengths(
        "MXFP4 SFB preparation",
        &[
            ("linear source", linear_source.len(), source_bytes),
            (
                "prepared destination",
                prepared_destination.len(),
                destination_bytes,
            ),
        ],
    )?;
    let args = PrepareMxfp4SfbArgs {
        n: checked_u32(n, "MXFP4 SFB n")?,
        k: checked_u32(k, "MXFP4 SFB k")?,
        reserved0: 0,
        linear_source: linear_source.cu_deviceptr(),
        prepared_destination: prepared_destination.cu_deviceptr(),
        stream: stream.cu_stream() as usize as u64,
    };
    let status = unsafe { ffi::ferrule_cutlass_prepare_mxfp4_sfb(&args) };
    native_result("prepare MXFP4 SFB scales", status)
}

fn validate_capacity(scope: &str, name: &str, actual: usize, required: usize) -> Result<()> {
    if actual < required {
        return Err(Error::Internal {
            message: format!(
                "{scope} {name} capacity is too small: actual={actual} required={required}"
            ),
        });
    }
    Ok(())
}

fn validate_lengths(scope: &str, required: &[(&str, usize, usize)]) -> Result<()> {
    for &(name, actual, expected) in required {
        if actual != expected {
            return Err(Error::Internal {
                message: format!(
                    "{scope} {name} length mismatch: actual={actual} expected={expected}"
                ),
            });
        }
    }
    Ok(())
}

fn native_result(operation: &str, code: i32) -> Result<()> {
    if code == status::SUCCESS {
        Ok(())
    } else {
        Err(native_error(operation, code))
    }
}

fn native_error(operation: &str, code: i32) -> Error {
    let reason = match code {
        status::INVALID_ARGUMENT => "invalid arguments",
        status::LAUNCH_FAILED => "kernel launch failed",
        status::UNSUPPORTED => "unsupported capability",
        _ => "unknown native status",
    };
    Error::Internal {
        message: format!("CUTLASS {operation} failed: {reason} ({code})"),
    }
}

mod status {
    pub const SUCCESS: i32 = 0;
    pub const INVALID_ARGUMENT: i32 = 2;
    pub const LAUNCH_FAILED: i32 = 3;
    pub const UNSUPPORTED: i32 = 4;
}

mod ffi {
    use super::{
        CutlassBf16CompressorArgs, CutlassFp8QueryAKvArgs, CutlassHcProducerArgs,
        CutlassHybridMlaAttentionArgs, CutlassMainProjectNormArgs, CutlassMlaOutputArgs,
        CutlassProposalHeadArgs, CutlassProviderManifest, CutlassSharedFfnArgs,
        FerruleCutlassGroupedFp4MoeArgs, FerruleCutlassHybridMlaExplicitSelectionArgs,
        FerruleCutlassWorkspaceRequirements, PrepareMxfp4SfbArgs,
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
        pub fn ferrule_cutlass_fp8_projection_can_implement(
            args: *const CutlassFp8QueryAKvArgs,
        ) -> i32;
        pub fn ferrule_cutlass_fp8_projection_launch(args: *const CutlassFp8QueryAKvArgs) -> i32;
        pub fn ferrule_cutlass_main_project_norm_can_implement(
            args: *const CutlassMainProjectNormArgs,
        ) -> i32;
        pub fn ferrule_cutlass_main_project_norm_launch(
            args: *const CutlassMainProjectNormArgs,
        ) -> i32;
        pub fn ferrule_cutlass_hybrid_mla_attention_can_implement(
            args: *const CutlassHybridMlaAttentionArgs,
        ) -> i32;
        pub fn ferrule_cutlass_hybrid_mla_attention_launch(
            args: *const CutlassHybridMlaAttentionArgs,
        ) -> i32;
        pub fn ferrule_cutlass_hybrid_mla_explicit_selection_workspace_requirements(
            args: *const FerruleCutlassHybridMlaExplicitSelectionArgs,
            requirements: *mut FerruleCutlassWorkspaceRequirements,
        ) -> i32;
        pub fn ferrule_cutlass_hybrid_mla_explicit_selection_can_implement(
            args: *const FerruleCutlassHybridMlaExplicitSelectionArgs,
        ) -> i32;
        pub fn ferrule_cutlass_hybrid_mla_explicit_selection_launch(
            args: *const FerruleCutlassHybridMlaExplicitSelectionArgs,
        ) -> i32;
        #[cfg(ferrule_cuda_test_oracle)]
        pub fn ferrule_cutlass_test_hybrid_mla_explicit_selection_scalar_launch(
            args: *const FerruleCutlassHybridMlaExplicitSelectionArgs,
        ) -> i32;
        #[cfg(ferrule_cuda_test_oracle)]
        pub fn ferrule_cutlass_test_hybrid_mla_explicit_selection_compare_launch(
            args: *const FerruleCutlassHybridMlaExplicitSelectionArgs,
            oracle_output_f32: u64,
            compare_result_i32: u64,
        ) -> i32;
        pub fn ferrule_cutlass_proposal_head_can_implement(
            args: *const CutlassProposalHeadArgs,
        ) -> i32;
        pub fn ferrule_cutlass_proposal_head_launch(args: *const CutlassProposalHeadArgs) -> i32;
        pub fn ferrule_cutlass_hc_producer_can_implement(args: *const CutlassHcProducerArgs)
        -> i32;
        pub fn ferrule_cutlass_hc_producer_launch(args: *const CutlassHcProducerArgs) -> i32;
        pub fn ferrule_cutlass_shared_ffn_can_implement(args: *const CutlassSharedFfnArgs) -> i32;
        pub fn ferrule_cutlass_shared_ffn_launch(args: *const CutlassSharedFfnArgs) -> i32;
        pub fn ferrule_cutlass_mla_output_can_implement(args: *const CutlassMlaOutputArgs) -> i32;
        pub fn ferrule_cutlass_mla_output_launch(args: *const CutlassMlaOutputArgs) -> i32;
        pub fn ferrule_cutlass_grouped_fp4_moe_workspace_size(
            args: *const FerruleCutlassGroupedFp4MoeArgs,
        ) -> u64;
        pub fn ferrule_cutlass_grouped_fp4_moe_can_implement(
            args: *const FerruleCutlassGroupedFp4MoeArgs,
        ) -> i32;
        pub fn ferrule_cutlass_grouped_fp4_moe_launch(
            args: *const FerruleCutlassGroupedFp4MoeArgs,
        ) -> i32;
        pub fn ferrule_cutlass_mxfp4_sfb_storage_bytes(n: u32, k: u32) -> u64;
        pub fn ferrule_cutlass_prepare_mxfp4_sfb(args: *const PrepareMxfp4SfbArgs) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    macro_rules! assert_pod_layout {
        ($ty:ty, size = $size:expr, align = $align:expr, $($field:ident = $offset:expr),+ $(,)?) => {{
            assert_eq!(size_of::<$ty>(), $size, "{} size", stringify!($ty));
            assert_eq!(align_of::<$ty>(), $align, "{} alignment", stringify!($ty));
            $(
                assert_eq!(
                    offset_of!($ty, $field),
                    $offset,
                    "{}.{} offset",
                    stringify!($ty),
                    stringify!($field),
                );
            )+
        }};
    }

    #[test]
    fn ffi_pod_layouts_match_native_contract() {
        assert_pod_layout!(
            CutlassProviderManifest,
            size = 8,
            align = 8,
            kernel_mask = 0,
        );
        assert_pod_layout!(
            CutlassFp8QueryAKvArgs,
            size = 96,
            align = 8,
            rows = 0,
            n1 = 4,
            n2 = 8,
            k = 12,
            scale_cols = 16,
            activation_fp8 = 24,
            activation_ue8m0 = 32,
            query_a_weight_fp8 = 40,
            query_a_weight_ue8m0 = 48,
            kv_weight_fp8 = 56,
            kv_weight_ue8m0 = 64,
            query_a_output_f32 = 72,
            kv_output_f32 = 80,
            stream = 88,
        );
        assert_pod_layout!(
            CutlassBf16CompressorArgs,
            size = 72,
            align = 8,
            rows = 0,
            n1 = 4,
            n2 = 8,
            k = 12,
            reserved0 = 16,
            activation_f32 = 24,
            projection1_weight_bf16 = 32,
            projection2_weight_bf16 = 40,
            projection1_output_f32 = 48,
            projection2_output_f32 = 56,
            stream = 64,
        );
        assert_pod_layout!(
            CutlassHcProducerArgs,
            size = 144,
            align = 8,
            rows = 0,
            hc = 4,
            hidden = 8,
            mix = 12,
            sinkhorn_iters = 16,
            hc_eps = 20,
            hc_norm_eps = 24,
            layer_rms_eps = 28,
            reserved = 32,
            state_f32 = 40,
            function_col_major_f32 = 48,
            hc_scale_f32 = 56,
            hc_base_f32 = 64,
            layer_rms_weight_f32 = 72,
            hidden_f32 = 80,
            normalized_f32 = 88,
            packed_e4m3 = 96,
            scales_ue8m0 = 104,
            split_pre_f32 = 112,
            split_post_f32 = 120,
            split_comb_f32 = 128,
            stream = 136,
        );
        assert_pod_layout!(
            CutlassSharedFfnArgs,
            size = 160,
            align = 8,
            input_fp8 = 0,
            input_ue8m0 = 8,
            gate_weight_fp8 = 16,
            gate_weight_ue8m0 = 24,
            up_weight_fp8 = 32,
            up_weight_ue8m0 = 40,
            down_weight_fp8 = 48,
            down_weight_ue8m0 = 56,
            hidden_f32 = 64,
            hidden_fp8 = 72,
            hidden_ue8m0 = 80,
            output_f32 = 88,
            rows = 96,
            input_size = 100,
            intermediate_size = 104,
            output_size = 108,
            gate_block_m = 112,
            gate_block_k = 116,
            up_block_m = 120,
            up_block_k = 124,
            down_block_m = 128,
            down_block_k = 132,
            output_scale = 136,
            swiglu_limit = 140,
            flags = 144,
            stream = 152,
        );
        assert_pod_layout!(
            CutlassMlaOutputArgs,
            size = 120,
            align = 8,
            rows = 0,
            context_size = 4,
            groups = 8,
            group_input_size = 12,
            rank = 16,
            latent_size = 20,
            hidden_size = 24,
            output_a_scale_cols = 28,
            reserved0 = 32,
            context_f32 = 40,
            output_a_weight_fp8 = 48,
            output_a_weight_ue8m0 = 56,
            output_b_weight_fp8 = 64,
            output_b_weight_ue8m0 = 72,
            latent_f32 = 80,
            latent_fp8 = 88,
            latent_ue8m0 = 96,
            output_f32 = 104,
            stream = 112,
        );
        assert_pod_layout!(
            CutlassMainProjectNormArgs,
            size = 104,
            align = 8,
            rows = 0,
            input_size = 4,
            output_size = 8,
            scale_cols = 12,
            reserved0 = 16,
            rms_eps = 20,
            reserved1 = 24,
            input_f32 = 32,
            activation_fp8 = 40,
            activation_ue8m0 = 48,
            weight_fp8 = 56,
            weight_ue8m0 = 64,
            norm_weight_f32 = 72,
            inv_rms_f32 = 80,
            output_f32 = 88,
            stream = 96,
        );
        assert_pod_layout!(
            CutlassHybridMlaAttentionArgs,
            size = 176,
            align = 8,
            block_rows = 0,
            heads = 4,
            head_dim = 8,
            sequence_tokens = 12,
            window_size = 16,
            page_tokens = 20,
            elements_per_token = 24,
            layer_index = 28,
            layer_count = 32,
            block_slot_offset = 36,
            block_slot_count = 40,
            softmax_scale = 44,
            reserved0 = 48,
            context_plane_elements = 56,
            query_f32 = 64,
            context_plane_f32 = 72,
            block_kv_f32 = 80,
            block_slots_i32 = 88,
            attention_sink_f32 = 96,
            query_bf16 = 104,
            gathered_kv_bf16 = 112,
            scores_f32 = 120,
            probabilities_bf16 = 128,
            online_rescales_f32 = 136,
            denominators_f32 = 144,
            output_f32 = 152,
            status_i32 = 160,
            stream = 168,
        );
        assert_pod_layout!(
            FerruleCutlassHybridMlaExplicitSelectionArgs,
            size = 224,
            align = 8,
            kind = 0,
            rows = 4,
            tokens_per_sequence = 8,
            kv_len = 12,
            heads = 16,
            head_dim = 20,
            selected_width = 24,
            page_tokens = 28,
            first_elements_per_token = 32,
            second_elements_per_token = 36,
            layer_index = 40,
            layer_count = 44,
            flags = 48,
            softmax_scale = 52,
            reserved0 = 56,
            first_plane_elements = 64,
            second_plane_elements = 72,
            query_f32 = 80,
            first_plane_f32 = 88,
            second_plane_f32 = 96,
            block_slots_i32 = 104,
            block_offsets_i32 = 112,
            sequence_kv_lens_i32 = 120,
            second_sequence_kv_lens_i32 = 128,
            row_sequence_ids_i32 = 136,
            row_kv_lens_i32 = 144,
            row_second_kv_lens_i32 = 152,
            selected_indices_i32 = 160,
            selectors_i32 = 168,
            attention_sink_f32 = 176,
            workspace = 184,
            workspace_bytes = 192,
            output_f32 = 200,
            status_i32 = 208,
            stream = 216,
        );
        assert_pod_layout!(
            FerruleCutlassWorkspaceRequirements,
            size = 16,
            align = 8,
            bytes = 0,
            alignment = 8,
            reserved = 12,
        );
        assert_pod_layout!(
            CutlassProposalHeadArgs,
            size = 184,
            align = 8,
            rows = 0,
            hc = 4,
            hidden = 8,
            vocab = 12,
            markov_rank = 16,
            partial_capacity = 20,
            reserved0 = 24,
            hc_eps = 28,
            norm_eps = 32,
            hc_state_f32 = 40,
            hc_function_f32 = 48,
            hc_scale_f32 = 56,
            hc_base_f32 = 64,
            norm_weight_f32 = 72,
            lm_head_bf16 = 80,
            markov_w1_bf16 = 88,
            markov_w2_bf16 = 96,
            confidence_weight_bf16 = 104,
            hidden_f32 = 112,
            normalized_f32 = 120,
            base_logits_f32 = 128,
            partial_values_f32 = 136,
            partial_indices_i32 = 144,
            token_ids_i32 = 152,
            confidence_f32 = 160,
            status_i32 = 168,
            stream = 176,
        );
        assert_pod_layout!(
            FerruleCutlassGroupedFp4MoeArgs,
            size = 224,
            align = 8,
            active_group_count = 0,
            small_group_count = 4,
            slot_capacity = 8,
            max_group_rows = 12,
            total_routed_rows = 16,
            num_tokens = 20,
            num_routes = 24,
            input_size = 28,
            intermediate_size = 32,
            hidden_size = 36,
            swiglu_limit = 40,
            active_expert_slots = 48,
            active_group_generations = 56,
            expert_route_indptr = 64,
            expert_route_counts = 72,
            route_token_indices = 80,
            route_indices = 88,
            route_weights = 96,
            slot_generations = 104,
            gate_ptrs = 112,
            gate_scale_ptrs = 120,
            up_ptrs = 128,
            up_scale_ptrs = 136,
            down_ptrs = 144,
            down_scale_ptrs = 152,
            input_packed = 160,
            input_scales = 168,
            route_output = 176,
            route_written = 184,
            route_error = 192,
            workspace = 200,
            workspace_bytes = 208,
            stream = 216,
        );
        assert_pod_layout!(
            PrepareMxfp4SfbArgs,
            size = 40,
            align = 8,
            n = 0,
            k = 4,
            reserved0 = 8,
            linear_source = 16,
            prepared_destination = 24,
            stream = 32,
        );
    }

    #[test]
    fn native_provider_publishes_semantic_capabilities() {
        let manifest = discover_provider().expect("provider").manifest();
        let target = crate::cuda::architecture::CudaTarget::parse(
            crate::cuda::architecture::COMPILED_TARGET,
        )
        .expect("compiled CUDA target");
        let capabilities = target.capabilities();
        assert_eq!(
            manifest.supports(CutlassKernelId::Fp8QueryAKv),
            capabilities.fp8_mma_sync
        );
        assert!(manifest.supports(CutlassKernelId::Bf16Compressor));
        assert!(manifest.supports(CutlassKernelId::HyperConnectionProducer));
        assert_eq!(
            manifest.supports(CutlassKernelId::SharedFfn),
            capabilities.fp8_mma_sync
        );
        assert_eq!(
            manifest.supports(CutlassKernelId::GroupedFp4Moe),
            capabilities.sm103_block_scaled_fp4 && !cfg!(debug_assertions)
        );
        assert_eq!(
            manifest.supports(CutlassKernelId::MlaOutput),
            capabilities.fp8_mma_sync
        );
        assert_eq!(
            manifest.supports(CutlassKernelId::MainProjectNorm),
            capabilities.fp8_mma_sync
        );
        assert!(manifest.supports(CutlassKernelId::HybridMlaAttention));
        assert!(manifest.supports(CutlassKernelId::ProposalHead));
        assert_eq!(
            manifest.supports(CutlassKernelId::Fp8Projection),
            capabilities.fp8_mma_sync
        );
    }
}
