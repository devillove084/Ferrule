//! Private POD contracts and symbols for the core CUDA provider.

pub(crate) const DSV4_DECODE_INDEX_QUERY_SHARED_ELEMENTS: usize = 8192;

pub(crate) const LINEAR_F32: u32 = 1;
pub(crate) const LINEAR_F32_BYTES: u32 = 2;
pub(crate) const LINEAR_BF16_BYTES: u32 = 3;
pub(crate) const LINEAR_FP8: u32 = 4;
pub(crate) const LINEAR_FP8_PACKED: u32 = 5;
pub(crate) const LINEAR_BF16_ROUNDED_INPUT: u32 = 7;
pub(crate) const LINEAR_FP8_FROM_F32: u32 = 8;

pub(crate) const GROUPED_F32: u32 = 1;
pub(crate) const GROUPED_FP8_TO_BF16: u32 = 2;

pub(crate) const QUANT_FP8_IN_PLACE: u32 = 1;
pub(crate) const QUANT_FP8_NON_ROPE: u32 = 2;
pub(crate) const QUANT_HADAMARD_FP4: u32 = 3;
pub(crate) const QUANT_FP4_PACKED: u32 = 4;
pub(crate) const QUANT_FP8_PACKED: u32 = 5;

pub(crate) const DATA_FILL_I32: u32 = 1;
pub(crate) const DATA_PACK_I32_F32: u32 = 2;
pub(crate) const DATA_PACK_PROPOSAL_HEAD: u32 = 3;
pub(crate) const DATA_FILL_PAGED_WINDOW: u32 = 4;
pub(crate) const DATA_FILL_DECODE_TOPK: u32 = 5;
pub(crate) const DATA_COPY_F32: u32 = 6;
pub(crate) const DATA_GATHER_F32_ROWS: u32 = 7;
pub(crate) const DATA_SCATTER_ADD_F32_ROWS: u32 = 8;
pub(crate) const DATA_SAXPY: u32 = 9;
pub(crate) const DATA_CONVERT_COMBINED_RING: u32 = 10;
pub(crate) const DATA_PAGED_PLANE_SCATTER: u32 = 11;
pub(crate) const DATA_FILL_RECENT_ROWS: u32 = 12;

pub(crate) const ROWS_F32_TO_BF16_RNE: u32 = 1;

pub(crate) const EMBED_RESIDENT_HC_BF16: u32 = 1;
pub(crate) const EMBED_PROPOSAL_HC_BF16: u32 = 2;

pub(crate) const NORM_COMPUTE_RMS: u32 = 1;
pub(crate) const NORM_AFFINE_ROW: u32 = 2;
pub(crate) const NORM_AFFINE_ROWS: u32 = 3;
pub(crate) const NORM_HEAD_ROWS: u32 = 4;

pub(crate) const ROPE_YARN: u32 = 1;
pub(crate) const ROPE_TAIL_STRIDED: u32 = 2;
pub(crate) const ROPE_TAIL_INDEXED: u32 = 3;
pub(crate) const ROPE_SPLIT_HALF_INDEXED: u32 = 4;

pub(crate) const ROUTER_TOPK: u32 = 1;
pub(crate) const ROUTER_HASH: u32 = 2;
pub(crate) const VOCAB_TOPK_F32: u32 = 3;
pub(crate) const VOCAB_TOPK_I32: u32 = 4;
pub(crate) const ROUTER_SELECTED_SOFTMAX_TOPK: u32 = 5;

pub(crate) const COMPRESSOR_PREFILL: u32 = 1;
pub(crate) const COMPRESSOR_RESET: u32 = 2;
pub(crate) const COMPRESSOR_APPEND: u32 = 3;
pub(crate) const COMPRESSOR_SEED: u32 = 4;
pub(crate) const COMPRESSOR_SOFTMAX: u32 = 5;

pub(crate) const INDEXER_FUSED_QUERY: u32 = 1 << 0;
pub(crate) const INDEXER_ROW_METADATA: u32 = 1 << 1;
pub(crate) const INDEXER_DIRECT_COMPRESSED: u32 = 1 << 2;

pub(crate) const EXPERT_INSTALL: u32 = 1;
pub(crate) const EXPERT_EVICT: u32 = 2;
pub(crate) const EXPERT_INITIALIZE_RESOLVE: u32 = 3;
pub(crate) const EXPERT_RESOLVE: u32 = 4;
pub(crate) const EXPERT_GATHER_DISPATCH: u32 = 5;

pub(crate) const EXPERT_GROUP_ROUTE_INIT_INVOCATION: u32 = 1;
pub(crate) const EXPERT_GROUP_ROUTE_INIT_PLAN: u32 = 2;
pub(crate) const EXPERT_GROUP_ROUTE_COUNT: u32 = 3;
pub(crate) const EXPERT_GROUP_ROUTE_COMPACT: u32 = 4;
pub(crate) const EXPERT_GROUP_ROUTE_SCATTER: u32 = 5;

pub(crate) const MOE_WEIGHTED_SWIGLU_F32: u32 = 2;
pub(crate) const MOE_REDUCE_EXPERT: u32 = 5;
pub(crate) const MOE_REDUCE_SPLIT_EXPERT: u32 = 6;
pub(crate) const MOE_REDUCE_ROUTES: u32 = 7;
pub(crate) const MOE_REDUCE_EXPERT_GROUP_ROUTES: u32 = 8;
pub(crate) const MOE_GATHER_BF16_ROWS: u32 = 9;
pub(crate) const MOE_WEIGHTED_SCATTER_ADD_BF16_ROWS: u32 = 10;

pub(crate) const HC_PRE: u32 = 1;
pub(crate) const HC_POST: u32 = 2;
pub(crate) const HC_MEAN_SCATTER: u32 = 3;
pub(crate) const HC_HEAD: u32 = 4;

pub(crate) const TRANSFORMER_PAGED_BF16_KV_APPEND: u32 = 1;
pub(crate) const TRANSFORMER_PAGED_BF16_CAUSAL_GQA: u32 = 2;
pub(crate) const TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct LinearArgs {
    pub(crate) kind: u32,
    pub(crate) batch: u32,
    pub(crate) n: u32,
    pub(crate) k: u32,
    pub(crate) scale_cols: u32,
    pub(crate) block_m: u32,
    pub(crate) block_k: u32,
    pub(crate) packed_offset: u32,
    pub(crate) scale_offset: u32,
    pub(crate) x: u64,
    pub(crate) x_scales: u64,
    pub(crate) weight: u64,
    pub(crate) weight_scales: u64,
    pub(crate) output: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct DualLinearArgs {
    pub(crate) kind: u32,
    pub(crate) first_n: u32,
    pub(crate) second_n: u32,
    pub(crate) k: u32,
    pub(crate) first_packed_offset: u32,
    pub(crate) first_scale_offset: u32,
    pub(crate) second_packed_offset: u32,
    pub(crate) second_scale_offset: u32,
    pub(crate) reserved: u32,
    pub(crate) x: u64,
    pub(crate) first_weight: u64,
    pub(crate) first_scales: u64,
    pub(crate) first_output: u64,
    pub(crate) second_weight: u64,
    pub(crate) second_scales: u64,
    pub(crate) second_output: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct GroupedLinearArgs {
    pub(crate) kind: u32,
    pub(crate) rows: u32,
    pub(crate) output_dim: u32,
    pub(crate) group_input: u32,
    pub(crate) rank: u32,
    pub(crate) scale_cols: u32,
    pub(crate) reserved: u32,
    pub(crate) input: u64,
    pub(crate) weight: u64,
    pub(crate) weight_scales: u64,
    pub(crate) output: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct QuantizeArgs {
    pub(crate) kind: u32,
    pub(crate) value_offset: u32,
    pub(crate) value_len: u32,
    pub(crate) row_width: u32,
    pub(crate) block_size: u32,
    pub(crate) rope_dim: u32,
    pub(crate) reserved: u32,
    pub(crate) values: u64,
    pub(crate) packed: u64,
    pub(crate) scales: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct DataArgs {
    pub(crate) kind: u32,
    pub(crate) count: u32,
    pub(crate) rows: u32,
    pub(crate) width: u32,
    pub(crate) offset: u32,
    pub(crate) start: u32,
    pub(crate) value0: u32,
    pub(crate) value1: u32,
    pub(crate) value2: u32,
    pub(crate) value3: u32,
    pub(crate) flags: u32,
    pub(crate) scale: f32,
    pub(crate) reserved: u32,
    pub(crate) input0: u64,
    pub(crate) input1: u64,
    pub(crate) input2: u64,
    pub(crate) input3: u64,
    pub(crate) input4: u64,
    pub(crate) input5: u64,
    pub(crate) output0: u64,
    pub(crate) output1: u64,
    pub(crate) stream: u64,
    pub(crate) output_elements: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RowsArgs {
    pub(crate) kind: u32,
    pub(crate) rows: u32,
    pub(crate) heads: u32,
    pub(crate) dimensions: u32,
    pub(crate) input_f32: u64,
    pub(crate) input_bytes: u64,
    pub(crate) input_row_stride_bytes: u64,
    pub(crate) input_head_stride_bytes: u64,
    pub(crate) output_bf16: u64,
    pub(crate) output_bytes: u64,
    pub(crate) output_row_stride_bytes: u64,
    pub(crate) output_head_stride_bytes: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct EmbeddingArgs {
    pub(crate) kind: u32,
    pub(crate) rows: u32,
    pub(crate) vocab: u32,
    pub(crate) hc: u32,
    pub(crate) hidden: u32,
    pub(crate) anchor_token: u32,
    pub(crate) noise_token: u32,
    pub(crate) embedding: u64,
    pub(crate) token_ids: u64,
    pub(crate) output: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct NormArgs {
    pub(crate) kind: u32,
    pub(crate) rows: u32,
    pub(crate) width: u32,
    pub(crate) epsilon: f32,
    pub(crate) reserved: u32,
    pub(crate) input: u64,
    pub(crate) weight: u64,
    pub(crate) output: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RopeArgs {
    pub(crate) kind: u32,
    pub(crate) rows: u32,
    pub(crate) heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) rope_dim: u32,
    pub(crate) pair_count: u32,
    pub(crate) start_position: u32,
    pub(crate) position_stride: u32,
    pub(crate) inverse: u32,
    pub(crate) restore_bf16_boundary: u32,
    pub(crate) values: u64,
    pub(crate) cosine: u64,
    pub(crate) sine: u64,
    pub(crate) positions: u64,
    pub(crate) stream: u64,
}

impl RopeArgs {
    pub(crate) fn with_inverse(mut self, inverse: u32) -> Self {
        self.inverse = inverse;
        self.restore_bf16_boundary = 1;
        self
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RouterArgs {
    pub(crate) kind: u32,
    pub(crate) rows: u32,
    pub(crate) columns: u32,
    pub(crate) k: u32,
    pub(crate) hash_rows: u32,
    pub(crate) hash_columns: u32,
    pub(crate) flags: u32,
    pub(crate) route_scale: f32,
    pub(crate) reserved: u32,
    pub(crate) logits: u64,
    pub(crate) bias: u64,
    pub(crate) token_ids: u64,
    pub(crate) hash_table: u64,
    pub(crate) indices: u64,
    pub(crate) weights: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct CompressorArgs {
    pub(crate) kind: u32,
    pub(crate) tokens: u32,
    pub(crate) groups: u32,
    pub(crate) ratio: u32,
    pub(crate) head_dim: u32,
    pub(crate) output_dim: u32,
    pub(crate) overlap: u32,
    pub(crate) position: u32,
    pub(crate) state_elements: u32,
    pub(crate) kv_input: u64,
    pub(crate) score_input: u64,
    pub(crate) ape: u64,
    pub(crate) kv_state: u64,
    pub(crate) score_state: u64,
    pub(crate) output: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct IndexerArgs {
    pub(crate) rows: u32,
    pub(crate) prefill: u32,
    pub(crate) window_size: u32,
    pub(crate) window_columns: u32,
    pub(crate) topk: u32,
    pub(crate) value_offset: u32,
    pub(crate) compress_ratio: u32,
    pub(crate) compressed_len: u32,
    pub(crate) heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) page_tokens: u32,
    pub(crate) layer_index: u32,
    pub(crate) layer_count: u32,
    pub(crate) position: u32,
    pub(crate) window_len: u32,
    pub(crate) rope_dim: u32,
    pub(crate) start_position: u32,
    pub(crate) weight_scale: f32,
    pub(crate) flags: u32,
    pub(crate) query: u64,
    pub(crate) weights: u64,
    pub(crate) cosine: u64,
    pub(crate) sine: u64,
    pub(crate) plane: u64,
    pub(crate) plane_elements: u64,
    pub(crate) block_slots: u64,
    pub(crate) block_offsets: u64,
    pub(crate) row_sequence_ids: u64,
    pub(crate) positions: u64,
    pub(crate) window_lens: u64,
    pub(crate) compressed_lens: u64,
    pub(crate) indices: u64,
    pub(crate) selectors: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct ExpertTableArgs {
    pub(crate) kind: u32,
    pub(crate) route_count: u32,
    pub(crate) expert_capacity: u32,
    pub(crate) slot_capacity: u32,
    pub(crate) miss_capacity: u32,
    pub(crate) route_capacity: u32,
    pub(crate) expert: u32,
    pub(crate) slot: u32,
    pub(crate) generation: i32,
    pub(crate) active_value: i32,
    pub(crate) reserved: u32,
    pub(crate) gate_weights: u64,
    pub(crate) gate_scales: u64,
    pub(crate) up_weights: u64,
    pub(crate) up_scales: u64,
    pub(crate) down_weights: u64,
    pub(crate) down_scales: u64,
    pub(crate) expert_to_slot: u64,
    pub(crate) expert_generations: u64,
    pub(crate) slot_generations: u64,
    pub(crate) expert_ids: u64,
    pub(crate) route_slots: u64,
    pub(crate) route_generations: u64,
    pub(crate) miss_markers: u64,
    pub(crate) miss_control: u64,
    pub(crate) router_weights: u64,
    pub(crate) active_markers: u64,
    pub(crate) output_gate_weights: u64,
    pub(crate) output_gate_scales: u64,
    pub(crate) output_up_weights: u64,
    pub(crate) output_up_scales: u64,
    pub(crate) output_down_weights: u64,
    pub(crate) output_down_scales: u64,
    pub(crate) output_route_weights: u64,
    pub(crate) dispatch_error: u64,
    pub(crate) gate_weight_value: u64,
    pub(crate) gate_scale_value: u64,
    pub(crate) up_weight_value: u64,
    pub(crate) up_scale_value: u64,
    pub(crate) down_weight_value: u64,
    pub(crate) down_scale_value: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct ExpertGroupRoutePlanArgs {
    pub(crate) kind: u32,
    pub(crate) route_count: u32,
    pub(crate) routes_per_token: u32,
    pub(crate) slot_capacity: u32,
    pub(crate) route_capacity: u32,
    pub(crate) output_elements: u32,
    pub(crate) small_group_row_limit: u32,
    pub(crate) route_slots: u64,
    pub(crate) route_generations: u64,
    pub(crate) router_weights: u64,
    pub(crate) slot_generations: u64,
    pub(crate) slot_counts: u64,
    pub(crate) slot_route_offsets: u64,
    pub(crate) slot_cursors: u64,
    pub(crate) active_expert_slots: u64,
    pub(crate) active_group_generations: u64,
    pub(crate) expert_route_indptr: u64,
    pub(crate) expert_route_counts: u64,
    pub(crate) route_token_indices: u64,
    pub(crate) route_indices: u64,
    pub(crate) route_weights: u64,
    pub(crate) host_scalars: u64,
    pub(crate) route_output: u64,
    pub(crate) route_written: u64,
    pub(crate) route_error: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct MoeArgs {
    pub(crate) kind: u32,
    pub(crate) n: u32,
    pub(crate) k: u32,
    pub(crate) batch_columns: u32,
    pub(crate) experts: u32,
    pub(crate) tokens: u32,
    pub(crate) routes_per_token: u32,
    pub(crate) output_offset: u32,
    pub(crate) hidden: u32,
    pub(crate) route_weight: f32,
    pub(crate) swiglu_limit: f32,
    pub(crate) input: u64,
    pub(crate) input_scales: u64,
    pub(crate) gate_ptrs: u64,
    pub(crate) gate_scale_ptrs: u64,
    pub(crate) up_ptrs: u64,
    pub(crate) up_scale_ptrs: u64,
    pub(crate) down_ptrs: u64,
    pub(crate) down_scale_ptrs: u64,
    pub(crate) gate: u64,
    pub(crate) up: u64,
    pub(crate) route_weights: u64,
    pub(crate) hidden_values: u64,
    pub(crate) hidden_packed: u64,
    pub(crate) hidden_scales: u64,
    pub(crate) expert_output: u64,
    pub(crate) resident_output: u64,
    pub(crate) materialized_output: u64,
    pub(crate) route_slots: u64,
    pub(crate) materialized_route_slots: u64,
    pub(crate) miss_markers: u64,
    pub(crate) route_output: u64,
    pub(crate) route_written: u64,
    pub(crate) route_error: u64,
    pub(crate) output: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct HcArgs {
    pub(crate) kind: u32,
    pub(crate) tokens: u32,
    pub(crate) hc: u32,
    pub(crate) hidden_size: u32,
    pub(crate) mix: u32,
    pub(crate) sinkhorn_iters: u32,
    pub(crate) tap_slot: u32,
    pub(crate) tap_count: u32,
    pub(crate) reserved: u32,
    pub(crate) epsilon: f32,
    pub(crate) norm_epsilon: f32,
    pub(crate) state: u64,
    pub(crate) function: u64,
    pub(crate) scale: u64,
    pub(crate) base: u64,
    pub(crate) hidden: u64,
    pub(crate) residual: u64,
    pub(crate) split_pre: u64,
    pub(crate) split_post: u64,
    pub(crate) split_comb: u64,
    pub(crate) output: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(
    dead_code,
    reason = "layout mirror for the statically linked Core MLA symbol"
)]
pub(crate) struct MlaArgs {
    pub(crate) hidden_size: u32,
    pub(crate) rank: u32,
    pub(crate) output_size: u32,
    pub(crate) epsilon: f32,
    pub(crate) reserved: u32,
    pub(crate) input: u64,
    pub(crate) weight_a: u64,
    pub(crate) weight_b: u64,
    pub(crate) norm_weight: u64,
    pub(crate) output: u64,
    pub(crate) stream: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct TransformerArgs {
    pub(crate) kind: u32,
    pub(crate) rows: u32,
    pub(crate) sequences: u32,
    pub(crate) q_heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) page_tokens: u32,
    pub(crate) layer_index: u32,
    pub(crate) layer_count: u32,
    pub(crate) softmax_scale: f32,
    pub(crate) flags: u32,
    pub(crate) reserved: u32,
    pub(crate) query_f32: u64,
    pub(crate) query_bytes: u64,
    pub(crate) query_row_stride_bytes: u64,
    pub(crate) query_head_stride_bytes: u64,
    pub(crate) append_key_bf16: u64,
    pub(crate) append_key_bytes: u64,
    pub(crate) append_key_row_stride_bytes: u64,
    pub(crate) append_key_head_stride_bytes: u64,
    pub(crate) append_value_bf16: u64,
    pub(crate) append_value_bytes: u64,
    pub(crate) append_value_row_stride_bytes: u64,
    pub(crate) append_value_head_stride_bytes: u64,
    pub(crate) key_cache_bf16: u64,
    pub(crate) key_cache_bytes: u64,
    pub(crate) key_slot_stride_bytes: u64,
    pub(crate) key_layer_stride_bytes: u64,
    pub(crate) key_token_stride_bytes: u64,
    pub(crate) key_head_stride_bytes: u64,
    pub(crate) value_cache_bf16: u64,
    pub(crate) value_cache_bytes: u64,
    pub(crate) value_slot_stride_bytes: u64,
    pub(crate) value_layer_stride_bytes: u64,
    pub(crate) value_token_stride_bytes: u64,
    pub(crate) value_head_stride_bytes: u64,
    pub(crate) block_slots_i32: u64,
    pub(crate) block_slots_count: u64,
    pub(crate) block_offsets_i32: u64,
    pub(crate) block_offsets_count: u64,
    pub(crate) row_sequence_ids_i32: u64,
    pub(crate) row_sequence_ids_count: u64,
    pub(crate) row_positions_i32: u64,
    pub(crate) row_positions_count: u64,
    pub(crate) row_kv_lens_i32: u64,
    pub(crate) row_kv_lens_count: u64,
    pub(crate) output_f32: u64,
    pub(crate) output_bytes: u64,
    pub(crate) output_row_stride_bytes: u64,
    pub(crate) output_head_stride_bytes: u64,
    pub(crate) status_i32: u64,
    pub(crate) status_count: u64,
    pub(crate) stream: u64,
}

unsafe extern "C" {
    pub(crate) fn ferrule_core_linear_launch(args: *const LinearArgs) -> i32;
    pub(crate) fn ferrule_core_dual_linear_launch(args: *const DualLinearArgs) -> i32;
    pub(crate) fn ferrule_core_grouped_linear_launch(args: *const GroupedLinearArgs) -> i32;
    pub(crate) fn ferrule_core_quantize_launch(args: *const QuantizeArgs) -> i32;
    pub(crate) fn ferrule_core_data_launch(args: *const DataArgs) -> i32;
    pub(crate) fn ferrule_core_rows_launch(args: *const RowsArgs) -> i32;
    pub(crate) fn ferrule_core_embedding_launch(args: *const EmbeddingArgs) -> i32;
    pub(crate) fn ferrule_core_norm_launch(args: *const NormArgs) -> i32;
    pub(crate) fn ferrule_core_rope_launch(args: *const RopeArgs) -> i32;
    pub(crate) fn ferrule_core_router_launch(args: *const RouterArgs) -> i32;
    pub(crate) fn ferrule_core_compressor_launch(args: *const CompressorArgs) -> i32;
    pub(crate) fn ferrule_core_indexer_launch(args: *const IndexerArgs) -> i32;

    pub(crate) fn ferrule_core_expert_table_launch(args: *const ExpertTableArgs) -> i32;
    pub(crate) fn ferrule_core_expert_group_route_plan_launch(
        args: *const ExpertGroupRoutePlanArgs,
    ) -> i32;
    pub(crate) fn ferrule_core_moe_launch(args: *const MoeArgs) -> i32;
    pub(crate) fn ferrule_core_hc_launch(args: *const HcArgs) -> i32;
    #[allow(
        dead_code,
        reason = "keeps the Core MLA link symbol declared and layout-tested"
    )]
    pub(crate) fn ferrule_core_mla_launch(args: *const MlaArgs) -> i32;
    pub(crate) fn ferrule_core_transformer_launch(args: *const TransformerArgs) -> i32;
}
