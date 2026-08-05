#ifndef FERRULE_CUDA_ABI_CORE_PROVIDER_H_
#define FERRULE_CUDA_ABI_CORE_PROVIDER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FerruleCoreLinearKind {
  FERRULE_CORE_LINEAR_F32 = 1,
  FERRULE_CORE_LINEAR_F32_BYTES = 2,
  FERRULE_CORE_LINEAR_BF16_BYTES = 3,
  FERRULE_CORE_LINEAR_FP8_E4M3_E8M0 = 4,
  FERRULE_CORE_LINEAR_FP8_E4M3_E8M0_PACKED = 5,
  FERRULE_CORE_LINEAR_BF16_BYTES_ROUNDED_INPUT = 7,
  FERRULE_CORE_LINEAR_FP8_E4M3_E8M0_FROM_F32 = 8,
} FerruleCoreLinearKind;

typedef struct FerruleCoreLinearArgs {
  uint32_t kind;
  uint32_t batch;
  uint32_t n;
  uint32_t k;
  uint32_t scale_cols;
  uint32_t block_m;
  uint32_t block_k;
  uint32_t packed_offset;
  uint32_t scale_offset;
  uint64_t x;
  uint64_t x_scales;
  uint64_t weight;
  uint64_t weight_scales;
  uint64_t output;
  uint64_t stream;
} FerruleCoreLinearArgs;

typedef struct FerruleCoreDualLinearArgs {
  uint32_t kind;
  uint32_t first_n;
  uint32_t second_n;
  uint32_t k;
  uint32_t first_packed_offset;
  uint32_t first_scale_offset;
  uint32_t second_packed_offset;
  uint32_t second_scale_offset;
  uint32_t reserved;
  uint64_t x;
  uint64_t first_weight;
  uint64_t first_scales;
  uint64_t first_output;
  uint64_t second_weight;
  uint64_t second_scales;
  uint64_t second_output;
  uint64_t stream;
} FerruleCoreDualLinearArgs;

typedef enum FerruleCoreGroupedLinearKind {
  FERRULE_CORE_GROUPED_LINEAR_F32 = 1,
  FERRULE_CORE_GROUPED_LINEAR_FP8_TO_BF16 = 2,
} FerruleCoreGroupedLinearKind;

typedef struct FerruleCoreGroupedLinearArgs {
  uint32_t kind;
  uint32_t rows;
  uint32_t output_dim;
  uint32_t group_input;
  uint32_t rank;
  uint32_t scale_cols;
  uint32_t reserved;
  uint64_t input;
  uint64_t weight;
  uint64_t weight_scales;
  uint64_t output;
  uint64_t stream;
} FerruleCoreGroupedLinearArgs;

typedef enum FerruleCoreQuantizeKind {
  FERRULE_CORE_QUANTIZE_FP8_IN_PLACE = 1,
  FERRULE_CORE_QUANTIZE_FP8_NON_ROPE_IN_PLACE = 2,
  FERRULE_CORE_QUANTIZE_HADAMARD_FP4_IN_PLACE = 3,
  FERRULE_CORE_QUANTIZE_FP4_PACKED = 4,
  FERRULE_CORE_QUANTIZE_FP8_PACKED = 5,
} FerruleCoreQuantizeKind;

typedef struct FerruleCoreQuantizeArgs {
  uint32_t kind;
  uint32_t value_offset;
  uint32_t value_len;
  uint32_t row_width;
  uint32_t block_size;
  uint32_t rope_dim;
  uint32_t reserved;
  uint64_t values;
  uint64_t packed;
  uint64_t scales;
  uint64_t stream;
} FerruleCoreQuantizeArgs;

typedef enum FerruleCoreDataKind {
  FERRULE_CORE_DATA_FILL_I32 = 1,
  FERRULE_CORE_DATA_PACK_I32_F32 = 2,
  FERRULE_CORE_DATA_PACK_PROPOSAL_HEAD = 3,
  FERRULE_CORE_DATA_FILL_PAGED_WINDOW = 4,
  FERRULE_CORE_DATA_FILL_DECODE_TOPK = 5,
  FERRULE_CORE_DATA_COPY_F32 = 6,
  FERRULE_CORE_DATA_GATHER_F32_ROWS = 7,
  FERRULE_CORE_DATA_SCATTER_ADD_F32_ROWS = 8,
  FERRULE_CORE_DATA_SAXPY = 9,
  FERRULE_CORE_DATA_CONVERT_COMBINED_RING = 10,
  FERRULE_CORE_DATA_PAGED_PLANE_SCATTER = 11,
  FERRULE_CORE_DATA_FILL_RECENT_ROWS = 12,
} FerruleCoreDataKind;

typedef struct FerruleCoreDataArgs {
  uint32_t kind;
  uint32_t count;
  uint32_t rows;
  uint32_t width;
  uint32_t offset;
  uint32_t start;
  uint32_t value0;
  uint32_t value1;
  uint32_t value2;
  uint32_t value3;
  uint32_t flags;
  float scale;
  uint32_t reserved;
  uint64_t input0;
  uint64_t input1;
  uint64_t input2;
  uint64_t input3;
  uint64_t input4;
  uint64_t input5;
  uint64_t output0;
  uint64_t output1;
  uint64_t stream;
  uint64_t output_elements;
} FerruleCoreDataArgs;

typedef enum FerruleCoreEmbeddingKind {
  FERRULE_CORE_EMBED_RESIDENT_HC_BF16 = 1,
  FERRULE_CORE_EMBED_PROPOSAL_HC_BF16 = 2,
} FerruleCoreEmbeddingKind;

typedef struct FerruleCoreEmbeddingArgs {
  uint32_t kind;
  uint32_t rows;
  uint32_t vocab;
  uint32_t hc;
  uint32_t hidden;
  uint32_t anchor_token;
  uint32_t noise_token;
  uint64_t embedding;
  uint64_t token_ids;
  uint64_t output;
  uint64_t stream;
} FerruleCoreEmbeddingArgs;

typedef enum FerruleCoreNormKind {
  FERRULE_CORE_NORM_COMPUTE_RMS = 1,
  FERRULE_CORE_NORM_AFFINE_ROW = 2,
  FERRULE_CORE_NORM_AFFINE_ROWS = 3,
  FERRULE_CORE_NORM_HEAD_ROWS = 4,
} FerruleCoreNormKind;

typedef struct FerruleCoreNormArgs {
  uint32_t kind;
  uint32_t rows;
  uint32_t width;
  float epsilon;
  uint32_t reserved;
  uint64_t input;
  uint64_t weight;
  uint64_t output;
  uint64_t stream;
} FerruleCoreNormArgs;

typedef enum FerruleCoreRopeKind {
  FERRULE_CORE_ROPE_YARN = 1,
  FERRULE_CORE_ROPE_TAIL_STRIDED = 2,
  FERRULE_CORE_ROPE_TAIL_INDEXED = 3,
} FerruleCoreRopeKind;

typedef struct FerruleCoreRopeArgs {
  uint32_t kind;
  uint32_t rows;
  uint32_t heads;
  uint32_t head_dim;
  uint32_t rope_dim;
  uint32_t pair_count;
  uint32_t start_position;
  uint32_t position_stride;
  uint32_t inverse;
  uint64_t values;
  uint64_t cosine;
  uint64_t sine;
  uint64_t positions;
  uint64_t stream;
} FerruleCoreRopeArgs;

typedef enum FerruleCoreRouterKind {
  FERRULE_CORE_ROUTER_TOPK = 1,
  FERRULE_CORE_ROUTER_HASH = 2,
  FERRULE_CORE_VOCAB_TOPK_F32_INDEX = 3,
  FERRULE_CORE_VOCAB_TOPK_I32_INDEX = 4,
} FerruleCoreRouterKind;

typedef struct FerruleCoreRouterArgs {
  uint32_t kind;
  uint32_t rows;
  uint32_t columns;
  uint32_t k;
  uint32_t hash_rows;
  uint32_t hash_columns;
  uint32_t flags;
  float route_scale;
  uint32_t reserved;
  uint64_t logits;
  uint64_t bias;
  uint64_t token_ids;
  uint64_t hash_table;
  uint64_t indices;
  uint64_t weights;
  uint64_t stream;
} FerruleCoreRouterArgs;

typedef enum FerruleCoreCompressorKind {
  FERRULE_CORE_COMPRESSOR_PREFILL = 1,
  FERRULE_CORE_COMPRESSOR_RESET = 2,
  FERRULE_CORE_COMPRESSOR_APPEND = 3,
  FERRULE_CORE_COMPRESSOR_SEED = 4,
  FERRULE_CORE_COMPRESSOR_SOFTMAX = 5,
} FerruleCoreCompressorKind;

typedef struct FerruleCoreCompressorArgs {
  uint32_t kind;
  uint32_t tokens;
  uint32_t groups;
  uint32_t ratio;
  uint32_t head_dim;
  uint32_t output_dim;
  uint32_t overlap;
  uint32_t position;
  uint32_t state_elements;
  uint64_t kv_input;
  uint64_t score_input;
  uint64_t ape;
  uint64_t kv_state;
  uint64_t score_state;
  uint64_t output;
  uint64_t stream;
} FerruleCoreCompressorArgs;

typedef struct FerruleCoreIndexerArgs {
  uint32_t rows;
  uint32_t prefill;
  uint32_t window_size;
  uint32_t window_columns;
  uint32_t topk;
  uint32_t value_offset;
  uint32_t compress_ratio;
  uint32_t compressed_len;
  uint32_t heads;
  uint32_t head_dim;
  uint32_t page_tokens;
  uint32_t layer_index;
  uint32_t layer_count;
  uint32_t position;
  uint32_t window_len;
  uint32_t rope_dim;
  uint32_t start_position;
  float weight_scale;
  uint32_t flags;
  uint64_t query;
  uint64_t weights;
  uint64_t cosine;
  uint64_t sine;
  uint64_t plane;
  uint64_t plane_elements;
  uint64_t block_slots;
  uint64_t block_offsets;
  uint64_t row_sequence_ids;
  uint64_t positions;
  uint64_t window_lens;
  uint64_t compressed_lens;
  uint64_t indices;
  uint64_t selectors;
  uint64_t stream;
} FerruleCoreIndexerArgs;

typedef enum FerruleCoreExpertTableKind {
  FERRULE_CORE_EXPERT_INSTALL = 1,
  FERRULE_CORE_EXPERT_EVICT = 2,
  FERRULE_CORE_EXPERT_INITIALIZE_RESOLVE = 3,
  FERRULE_CORE_EXPERT_RESOLVE = 4,
  FERRULE_CORE_EXPERT_GATHER_DISPATCH = 5,
} FerruleCoreExpertTableKind;

typedef struct FerruleCoreExpertTableArgs {
  uint32_t kind;
  uint32_t route_count;
  uint32_t expert_capacity;
  uint32_t slot_capacity;
  uint32_t miss_capacity;
  uint32_t route_capacity;
  uint32_t expert;
  uint32_t slot;
  int32_t generation;
  int32_t active_value;
  uint32_t reserved;
  uint64_t gate_weights;
  uint64_t gate_scales;
  uint64_t up_weights;
  uint64_t up_scales;
  uint64_t down_weights;
  uint64_t down_scales;
  uint64_t expert_to_slot;
  uint64_t expert_generations;
  uint64_t slot_generations;
  uint64_t expert_ids;
  uint64_t route_slots;
  uint64_t route_generations;
  uint64_t miss_markers;
  uint64_t miss_control;
  uint64_t router_weights;
  uint64_t active_markers;
  uint64_t output_gate_weights;
  uint64_t output_gate_scales;
  uint64_t output_up_weights;
  uint64_t output_up_scales;
  uint64_t output_down_weights;
  uint64_t output_down_scales;
  uint64_t output_route_weights;
  uint64_t dispatch_error;
  uint64_t gate_weight_value;
  uint64_t gate_scale_value;
  uint64_t up_weight_value;
  uint64_t up_scale_value;
  uint64_t down_weight_value;
  uint64_t down_scale_value;
  uint64_t stream;
} FerruleCoreExpertTableArgs;

typedef enum FerruleCoreExpertGroupRoutePlanKind {
  FERRULE_CORE_EXPERT_GROUP_ROUTE_INIT_INVOCATION = 1,
  FERRULE_CORE_EXPERT_GROUP_ROUTE_INIT_PLAN = 2,
  FERRULE_CORE_EXPERT_GROUP_ROUTE_COUNT = 3,
  FERRULE_CORE_EXPERT_GROUP_ROUTE_COMPACT = 4,
  FERRULE_CORE_EXPERT_GROUP_ROUTE_SCATTER = 5,
} FerruleCoreExpertGroupRoutePlanKind;

typedef struct FerruleCoreExpertGroupRoutePlanArgs {
  uint32_t kind;
  uint32_t route_count;
  uint32_t routes_per_token;
  uint32_t slot_capacity;
  uint32_t route_capacity;
  uint32_t output_elements;
  uint32_t small_group_row_limit;
  uint64_t route_slots;
  uint64_t route_generations;
  uint64_t router_weights;
  uint64_t slot_generations;
  uint64_t slot_counts;
  uint64_t slot_route_offsets;
  uint64_t slot_cursors;
  uint64_t active_expert_slots;
  uint64_t active_group_generations;
  uint64_t expert_route_indptr;
  uint64_t expert_route_counts;
  uint64_t route_token_indices;
  uint64_t route_indices;
  uint64_t route_weights;
  uint64_t host_scalars;
  uint64_t route_output;
  uint64_t route_written;
  uint64_t route_error;
  uint64_t stream;
} FerruleCoreExpertGroupRoutePlanArgs;

typedef enum FerruleCoreMoeKind {
  FERRULE_CORE_MOE_WEIGHTED_SWIGLU_F32 = 2,
  FERRULE_CORE_MOE_REDUCE_EXPERT = 5,
  FERRULE_CORE_MOE_REDUCE_SPLIT_EXPERT = 6,
  FERRULE_CORE_MOE_REDUCE_ROUTES = 7,
  FERRULE_CORE_MOE_REDUCE_EXPERT_GROUP_ROUTES = 8,
} FerruleCoreMoeKind;

typedef struct FerruleCoreMoeArgs {
  uint32_t kind;
  uint32_t n;
  uint32_t k;
  uint32_t batch_columns;
  uint32_t experts;
  uint32_t tokens;
  uint32_t routes_per_token;
  uint32_t output_offset;
  uint32_t hidden;
  float route_weight;
  float swiglu_limit;
  uint64_t input;
  uint64_t input_scales;
  uint64_t gate_ptrs;
  uint64_t gate_scale_ptrs;
  uint64_t up_ptrs;
  uint64_t up_scale_ptrs;
  uint64_t down_ptrs;
  uint64_t down_scale_ptrs;
  uint64_t gate;
  uint64_t up;
  uint64_t route_weights;
  uint64_t hidden_values;
  uint64_t hidden_packed;
  uint64_t hidden_scales;
  uint64_t expert_output;
  uint64_t resident_output;
  uint64_t materialized_output;
  uint64_t route_slots;
  uint64_t materialized_route_slots;
  uint64_t miss_markers;
  uint64_t route_output;
  uint64_t route_written;
  uint64_t route_error;
  uint64_t output;
  uint64_t stream;
} FerruleCoreMoeArgs;

typedef enum FerruleCoreHcKind {
  FERRULE_CORE_HC_PRE = 1,
  FERRULE_CORE_HC_POST = 2,
  FERRULE_CORE_HC_MEAN_SCATTER = 3,
  FERRULE_CORE_HC_HEAD = 4,
} FerruleCoreHcKind;

typedef struct FerruleCoreHcArgs {
  uint32_t kind;
  uint32_t tokens;
  uint32_t hc;
  uint32_t hidden_size;
  uint32_t mix;
  uint32_t sinkhorn_iters;
  uint32_t tap_slot;
  uint32_t tap_count;
  uint32_t reserved;
  float epsilon;
  float norm_epsilon;
  uint64_t state;
  uint64_t function;
  uint64_t scale;
  uint64_t base;
  uint64_t hidden;
  uint64_t residual;
  uint64_t split_pre;
  uint64_t split_post;
  uint64_t split_comb;
  uint64_t output;
  uint64_t stream;
} FerruleCoreHcArgs;

typedef struct FerruleCoreMlaArgs {
  uint32_t hidden_size;
  uint32_t rank;
  uint32_t output_size;
  float epsilon;
  uint32_t reserved;
  uint64_t input;
  uint64_t weight_a;
  uint64_t weight_b;
  uint64_t norm_weight;
  uint64_t output;
  uint64_t stream;
} FerruleCoreMlaArgs;

int32_t ferrule_core_linear_launch(const FerruleCoreLinearArgs *args);
int32_t ferrule_core_dual_linear_launch(const FerruleCoreDualLinearArgs *args);
int32_t
ferrule_core_grouped_linear_launch(const FerruleCoreGroupedLinearArgs *args);
int32_t ferrule_core_quantize_launch(const FerruleCoreQuantizeArgs *args);
int32_t ferrule_core_data_launch(const FerruleCoreDataArgs *args);
int32_t ferrule_core_embedding_launch(const FerruleCoreEmbeddingArgs *args);
int32_t ferrule_core_norm_launch(const FerruleCoreNormArgs *args);
int32_t ferrule_core_rope_launch(const FerruleCoreRopeArgs *args);
int32_t ferrule_core_router_launch(const FerruleCoreRouterArgs *args);
int32_t ferrule_core_compressor_launch(const FerruleCoreCompressorArgs *args);
int32_t ferrule_core_indexer_launch(const FerruleCoreIndexerArgs *args);
int32_t
ferrule_core_expert_table_launch(const FerruleCoreExpertTableArgs *args);
int32_t ferrule_core_expert_group_route_plan_launch(
    const FerruleCoreExpertGroupRoutePlanArgs *args);
int32_t ferrule_core_moe_launch(const FerruleCoreMoeArgs *args);
int32_t ferrule_core_hc_launch(const FerruleCoreHcArgs *args);
int32_t ferrule_core_mla_launch(const FerruleCoreMlaArgs *args);

#ifdef __cplusplus
}

#define FERRULE_CORE_ASSERT_LAYOUT(type, size, first_member, first_u64,        \
                                   first_u64_offset, stream_offset)            \
  static_assert(sizeof(type) == size, #type " size mismatch");                 \
  static_assert(alignof(type) == alignof(uint64_t),                            \
                #type " alignment mismatch");                                  \
  static_assert(offsetof(type, first_member) == 0,                             \
                #type " first member offset mismatch");                        \
  static_assert(offsetof(type, first_u64) == first_u64_offset,                 \
                #type " first uint64_t offset mismatch");                      \
  static_assert(offsetof(type, stream) == stream_offset,                       \
                #type " stream offset mismatch")

FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreLinearArgs, 88, kind, x, 40, 80);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreDualLinearArgs, 104, kind, x, 40, 96);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreGroupedLinearArgs, 72, kind, input, 32,
                           64);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreQuantizeArgs, 64, kind, values, 32, 56);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreDataArgs, 136, kind, input0, 56, 120);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreEmbeddingArgs, 64, kind, embedding, 32,
                           56);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreNormArgs, 56, kind, input, 24, 48);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreRopeArgs, 80, kind, values, 40, 72);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreRouterArgs, 96, kind, logits, 40, 88);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreCompressorArgs, 96, kind, kv_input, 40,
                           88);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreIndexerArgs, 200, rows, query, 80, 192);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreExpertTableArgs, 296, kind, gate_weights,
                           48, 288);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreExpertGroupRoutePlanArgs, 184, kind,
                           route_slots, 32, 176);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreMoeArgs, 248, kind, input, 48, 240);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreHcArgs, 136, kind, state, 48, 128);
FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreMlaArgs, 72, hidden_size, input, 24, 64);

static_assert(offsetof(FerruleCoreDataArgs, output_elements) == 128,
              "FerruleCoreDataArgs output_elements offset mismatch");
static_assert(offsetof(FerruleCoreExpertTableArgs, route_slots) == 128,
              "FerruleCoreExpertTableArgs route_slots offset mismatch");
static_assert(offsetof(FerruleCoreExpertTableArgs, output_gate_weights) == 176,
              "FerruleCoreExpertTableArgs output gate offset mismatch");
static_assert(offsetof(FerruleCoreExpertTableArgs, gate_weight_value) == 240,
              "FerruleCoreExpertTableArgs value offset mismatch");
static_assert(
    offsetof(FerruleCoreExpertGroupRoutePlanArgs, active_expert_slots) == 88,
    "FerruleCoreExpertGroupRoutePlanArgs active_expert_slots offset mismatch");
static_assert(
    offsetof(FerruleCoreExpertGroupRoutePlanArgs, expert_route_indptr) == 104,
    "FerruleCoreExpertGroupRoutePlanArgs expert_route_indptr offset mismatch");
static_assert(
    offsetof(FerruleCoreExpertGroupRoutePlanArgs, host_scalars) == 144,
    "FerruleCoreExpertGroupRoutePlanArgs host_scalars offset mismatch");
static_assert(
    offsetof(FerruleCoreExpertGroupRoutePlanArgs, route_output) == 152,
    "FerruleCoreExpertGroupRoutePlanArgs route_output offset mismatch");

#undef FERRULE_CORE_ASSERT_LAYOUT
#endif

#endif
