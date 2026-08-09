#ifndef FERRULE_CUDA_ABI_CUTLASS_PROVIDER_H_
#define FERRULE_CUDA_ABI_CUTLASS_PROVIDER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef FERRULE_CUDA_TARGET_SM
#define FERRULE_CUDA_TARGET_SM 0u
#endif

#define FERRULE_CUTLASS_KERNEL_FP8_QUERY_A_KV 1u
#define FERRULE_CUTLASS_KERNEL_BF16_COMPRESSOR 2u
#define FERRULE_CUTLASS_KERNEL_HYPER_CONNECTION_PRODUCER 3u
#define FERRULE_CUTLASS_KERNEL_SHARED_FFN 4u
#define FERRULE_CUTLASS_KERNEL_GROUPED_FP4_MOE 5u
#define FERRULE_CUTLASS_KERNEL_MLA_OUTPUT 6u
#define FERRULE_CUTLASS_KERNEL_MAIN_PROJECT_NORM 7u
#define FERRULE_CUTLASS_KERNEL_HYBRID_MLA_ATTENTION 8u
#define FERRULE_CUTLASS_KERNEL_PROPOSAL_HEAD 9u
#define FERRULE_CUTLASS_KERNEL_FP8_PROJECTION 10u
#define FERRULE_CUTLASS_KERNEL_BIT(id) (1ull << ((id) - 1u))

typedef enum FerruleCutlassStatus {
  FERRULE_CUTLASS_SUCCESS = 0,
  FERRULE_CUTLASS_INVALID_ARGUMENT = 2,
  FERRULE_CUTLASS_LAUNCH_FAILED = 3,
  FERRULE_CUTLASS_UNSUPPORTED = 4,
} FerruleCutlassStatus;

typedef struct FerruleCutlassProviderManifest {
  uint64_t kernel_mask;
} FerruleCutlassProviderManifest;

// Semantic one-launch QueryA+KV FP8 projection bundle. All tensors are
// contiguous: activation_fp8 is E4M3 [rows, k], each weight is E4M3 [n, k],
// and each output is F32 [rows, n]. UE8M0 scales cover K128 blocks;
// activation scales are [rows, scale_cols], while weight scales are
// [ceil(n / 128), scale_cols]. Ferrule owns all storage and the CUDA stream.
typedef struct FerruleCutlassFp8QueryAKvArgs {
  uint32_t rows;
  uint32_t n1;
  uint32_t n2;
  uint32_t k;
  uint32_t scale_cols;
  uint64_t activation_fp8;
  uint64_t activation_ue8m0;
  uint64_t query_a_weight_fp8;
  uint64_t query_a_weight_ue8m0;
  uint64_t kv_weight_fp8;
  uint64_t kv_weight_ue8m0;
  uint64_t query_a_output_f32;
  uint64_t kv_output_f32;
  uint64_t stream;
} FerruleCutlassFp8QueryAKvArgs;

// Semantic one-launch BF16 compressor bundle. activation_f32 is contiguous
// [rows, k], weights are BF16 [n1, k] and [n2, k], and outputs are F32
// [rows, n1] and [rows, n2]. The provider owns no storage.
typedef struct FerruleCutlassBf16CompressorArgs {
  uint32_t rows;
  uint32_t n1;
  uint32_t n2;
  uint32_t k;
  uint32_t reserved0;
  uint64_t activation_f32;
  uint64_t projection1_weight_bf16;
  uint64_t projection2_weight_bf16;
  uint64_t projection1_output_f32;
  uint64_t projection2_output_f32;
  uint64_t stream;
} FerruleCutlassBf16CompressorArgs;

// Semantic one-launch HC producer: HC mix/split, pre-RMSNorm, and packed FP8
// production remain one operation. Tensor layouts and fixed dimensions match
// hyper-connection operator contract. Ferrule owns every address and stream.
typedef struct FerruleCutlassHcProducerArgs {
  uint32_t rows;
  uint32_t hc;
  uint32_t hidden;
  uint32_t mix;
  uint32_t sinkhorn_iters;
  float hc_eps;
  float hc_norm_eps;
  float layer_rms_eps;
  uint32_t reserved;

  uint64_t state_f32;
  uint64_t function_row_major_f32;
  uint64_t hc_scale_f32;
  uint64_t hc_base_f32;
  uint64_t layer_rms_weight_f32;
  uint64_t mix_f32;
  uint64_t workspace;
  uint64_t workspace_bytes;
  uint64_t hidden_f32;
  uint64_t normalized_f32;
  uint64_t packed_e4m3;
  uint64_t scales_ue8m0;
  uint64_t split_pre_f32;
  uint64_t split_post_f32;
  uint64_t split_comb_f32;
  uint64_t stream;
} FerruleCutlassHcProducerArgs;

// Semantic one-launch shared FFN. The fields from input_fp8 through flags are
// the exact shared-FFN Args POD in its native order. The operation is
// the complete gate/up -> SwiGLU -> down chain; Ferrule supplies graph-stable
// compact intermediate storage and the provider performs no allocation or host
// synchronization.
typedef struct FerruleCutlassSharedFfnArgs {
  uint64_t input_fp8;
  uint64_t input_ue8m0;
  uint64_t gate_weight_fp8;
  uint64_t gate_weight_ue8m0;
  uint64_t up_weight_fp8;
  uint64_t up_weight_ue8m0;
  uint64_t down_weight_fp8;
  uint64_t down_weight_ue8m0;
  uint64_t hidden_f32;
  uint64_t hidden_fp8;
  uint64_t hidden_ue8m0;
  uint64_t output_f32;

  uint32_t rows;
  uint32_t input_size;
  uint32_t intermediate_size;
  uint32_t output_size;

  uint32_t gate_block_m;
  uint32_t gate_block_k;
  uint32_t up_block_m;
  uint32_t up_block_k;
  uint32_t down_block_m;
  uint32_t down_block_k;

  float output_scale;
  float swiglu_limit;
  uint32_t flags;
  uint64_t stream;
} FerruleCutlassSharedFfnArgs;

// Semantic one-launch MLA output bundle. Grouped FP8/E8M0 output-A writes the
// BF16 latent boundary, which is packed once and consumed by FP8/E8M0 output-B
// after device-wide barriers. No projection sub-kernel crosses FFI.
typedef struct FerruleCutlassMlaOutputArgs {
  uint32_t rows;
  uint32_t context_size;
  uint32_t groups;
  uint32_t group_input_size;
  uint32_t rank;
  uint32_t latent_size;
  uint32_t hidden_size;
  uint32_t output_a_scale_cols;
  uint32_t reserved0;

  uint64_t context_f32;
  uint64_t output_a_weight_fp8;
  uint64_t output_a_weight_ue8m0;
  uint64_t output_b_weight_fp8;
  uint64_t output_b_weight_ue8m0;
  uint64_t latent_bf16;
  uint64_t latent_fp8;
  uint64_t latent_ue8m0;
  uint64_t output_f32;
  uint64_t stream;
} FerruleCutlassMlaOutputArgs;

// Stage-zero target-tap projection and normalization. The input and output are
// F32 storage carrying the checkpoint's BF16 numerical boundaries. Activation
// FP8/E8M0 and inverse-RMS storage are graph-stable Ferrule-owned scratch. The
// provider performs one cooperative launch.
typedef struct FerruleCutlassMainProjectNormArgs {
  uint32_t rows;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t scale_cols;
  uint32_t reserved0;
  float rms_eps;
  uint32_t reserved1;

  uint64_t input_f32;
  uint64_t activation_fp8;
  uint64_t activation_ue8m0;
  uint64_t weight_fp8;
  uint64_t weight_ue8m0;
  uint64_t norm_weight_f32;
  uint64_t inv_rms_f32;
  uint64_t output_f32;
  uint64_t stream;
} FerruleCutlassMainProjectNormArgs;

// Checkpoint-native proposal attention. The release shape is fixed at
// five proposal rows, 64 heads, D=512, a 128-token committed window, and
// 16-token pages. All heads share each latent K/V row. Every proposal query
// sees the complete ephemeral five-row block; only committed context is
// page-backed. Scores, BF16 probabilities, output, and device status are
// Ferrule-owned.
typedef struct FerruleCutlassHybridMlaAttentionArgs {
  uint32_t block_rows;
  uint32_t heads;
  uint32_t head_dim;
  uint32_t sequence_tokens;
  uint32_t window_size;
  uint32_t page_tokens;
  uint32_t elements_per_token;
  uint32_t layer_index;
  uint32_t layer_count;
  uint32_t block_slot_offset;
  uint32_t block_slot_count;
  float softmax_scale;
  uint32_t reserved0;
  uint64_t context_plane_elements;

  uint64_t query_f32;
  uint64_t context_plane_f32;
  uint64_t block_kv_f32;
  uint64_t block_slots_i32;
  uint64_t attention_sink_f32;
  uint64_t query_bf16;
  uint64_t gathered_kv_bf16;
  uint64_t scores_f32;
  uint64_t probabilities_bf16;
  uint64_t online_rescales_f32;
  uint64_t denominators_f32;
  uint64_t output_f32;
  uint64_t status_i32;
  uint64_t stream;
} FerruleCutlassHybridMlaAttentionArgs;

typedef enum FerruleCutlassHybridMlaExplicitSelectionKind {
  FERRULE_CUTLASS_HYBRID_MLA_EXPLICIT_SELECTION_CONTIGUOUS = 1,
  FERRULE_CUTLASS_HYBRID_MLA_EXPLICIT_SELECTION_PAGED = 2,
  FERRULE_CUTLASS_HYBRID_MLA_EXPLICIT_SELECTION_DUAL_PAGED = 3,
} FerruleCutlassHybridMlaExplicitSelectionKind;

// Explicit-selection latent attention is one semantic hybrid-MLA operation.
// Selected indices are [rows, selected_width]; all 64 heads share each selected
// D=512 latent K/V row, and invalid selections are masked from softmax. Native
// code derives the private scratch layout inside opaque caller-owned workspace.
// A workspace query may use a null query pointer, but all scalar shape fields
// must still describe the complete layout.
typedef struct FerruleCutlassHybridMlaExplicitSelectionArgs {
  uint32_t kind;
  uint32_t rows;
  uint32_t tokens_per_sequence;
  uint32_t kv_len;
  uint32_t heads;
  uint32_t head_dim;
  uint32_t selected_width;
  uint32_t page_tokens;
  uint32_t first_elements_per_token;
  uint32_t second_elements_per_token;
  uint32_t layer_index;
  uint32_t layer_count;
  uint32_t flags;
  float softmax_scale;
  uint32_t reserved0;
  uint64_t first_plane_elements;
  uint64_t second_plane_elements;

  uint64_t query_f32;
  uint64_t first_plane_f32;
  uint64_t second_plane_f32;
  uint64_t block_slots_i32;
  uint64_t block_offsets_i32;
  uint64_t sequence_kv_lens_i32;
  uint64_t second_sequence_kv_lens_i32;
  uint64_t row_sequence_ids_i32;
  uint64_t row_kv_lens_i32;
  uint64_t row_second_kv_lens_i32;
  uint64_t selected_indices_i32;
  uint64_t selectors_i32;
  uint64_t attention_sink_f32;
  uint64_t workspace;
  uint64_t workspace_bytes;
  uint64_t output_f32;
  uint64_t status_i32;
  uint64_t stream;
} FerruleCutlassHybridMlaExplicitSelectionArgs;

// Generic caller-owned workspace description returned by semantic native
// operators. reserved is zero and alignment is a power of two.
typedef struct FerruleCutlassWorkspaceRequirements {
  uint64_t bytes;
  uint32_t alignment;
  uint32_t reserved;
} FerruleCutlassWorkspaceRequirements;

// Checkpoint-native proposal head. One semantic launch performs the
// five-row HC head and final norm, one tensor-core BF16 base-LM projection,
// then sequential device-only Markov bias/argmax and confidence. Token
// dependency never crosses the host boundary.
typedef struct FerruleCutlassProposalHeadArgs {
  uint32_t rows;
  uint32_t hc;
  uint32_t hidden;
  uint32_t vocab;
  uint32_t markov_rank;
  uint32_t partial_capacity;
  uint32_t reserved0;
  float hc_eps;
  float norm_eps;

  uint64_t hc_state_f32;
  uint64_t hc_function_f32;
  uint64_t hc_scale_f32;
  uint64_t hc_base_f32;
  uint64_t norm_weight_f32;
  uint64_t lm_head_bf16;
  uint64_t markov_w1_bf16;
  uint64_t markov_w2_bf16;
  uint64_t confidence_weight_bf16;

  uint64_t hidden_f32;
  uint64_t normalized_f32;
  uint64_t base_logits_f32;
  uint64_t partial_values_f32;
  uint64_t partial_indices_i32;
  uint64_t token_ids_i32;
  uint64_t confidence_f32;
  uint64_t status_i32;
  uint64_t stream;
} FerruleCutlassProposalHeadArgs;

// Complete variable-M grouped FP4 MoE pipeline. The active groups
// are compact, every group has M > 0, and groups are ordered as
// [small_group_count small groups, remaining large groups]. FC1 descriptor
// order within each bucket is [gate..., up...]; FC2 uses one descriptor per
// group. Device metadata is produced by Ferrule core and remains valid through
// the asynchronous launch. Active/slot generations are part of that trusted
// lifetime contract; expert-major route indices are unique, and the provider
// does not fabricate fallback descriptors for stale bindings. Weight scale
// pointers address prepared SFB storage.
typedef struct FerruleCutlassGroupedFp4MoeArgs {
  uint32_t active_group_count;
  uint32_t small_group_count;
  uint32_t slot_capacity;
  uint32_t max_group_rows;
  uint32_t total_routed_rows;
  uint32_t num_tokens;
  uint32_t num_routes;
  uint32_t input_size;
  uint32_t intermediate_size;
  uint32_t hidden_size;
  float swiglu_limit;

  uint64_t active_expert_slots;
  uint64_t active_group_generations;
  uint64_t expert_route_indptr;
  uint64_t expert_route_counts;
  uint64_t route_token_indices;
  uint64_t route_indices;
  uint64_t route_weights;

  uint64_t slot_generations;
  uint64_t gate_ptrs;
  uint64_t gate_scale_ptrs;
  uint64_t up_ptrs;
  uint64_t up_scale_ptrs;
  uint64_t down_ptrs;
  uint64_t down_scale_ptrs;

  uint64_t input_fp8;
  uint64_t input_ue8m0;

  uint64_t route_output;
  uint64_t route_written;
  uint64_t route_error;

  uint64_t workspace;
  uint64_t workspace_bytes;
  uint64_t stream;
} FerruleCutlassGroupedFp4MoeArgs;

// Converts linear UE8M0 weight scales [n, k / 32] into the CUTLASS SFB
// layout. Source and destination are device-accessible, and preparation is
// enqueued asynchronously on stream.
typedef struct FerruleCutlassPrepareMxfp4SfbArgs {
  uint32_t n;
  uint32_t k;
  uint32_t reserved0;
  uint64_t linear_source;
  uint64_t prepared_destination;
  uint64_t stream;
} FerruleCutlassPrepareMxfp4SfbArgs;

FerruleCutlassProviderManifest ferrule_cutlass_provider_manifest(void);
int32_t ferrule_cutlass_fp8_query_a_kv_can_implement(
    const FerruleCutlassFp8QueryAKvArgs *args);
int32_t ferrule_cutlass_fp8_query_a_kv_launch(
    const FerruleCutlassFp8QueryAKvArgs *args);
int32_t ferrule_cutlass_fp8_projection_can_implement(
    const FerruleCutlassFp8QueryAKvArgs *args);
int32_t ferrule_cutlass_fp8_projection_launch(
    const FerruleCutlassFp8QueryAKvArgs *args);
int32_t ferrule_cutlass_bf16_compressor_can_implement(
    const FerruleCutlassBf16CompressorArgs *args);
int32_t ferrule_cutlass_bf16_compressor_launch(
    const FerruleCutlassBf16CompressorArgs *args);
int32_t ferrule_cutlass_hc_producer_can_implement(
    const FerruleCutlassHcProducerArgs *args);
int32_t
ferrule_cutlass_hc_producer_launch(const FerruleCutlassHcProducerArgs *args);
int32_t ferrule_cutlass_shared_ffn_can_implement(
    const FerruleCutlassSharedFfnArgs *args);
int32_t
ferrule_cutlass_shared_ffn_launch(const FerruleCutlassSharedFfnArgs *args);
int32_t ferrule_cutlass_mla_output_can_implement(
    const FerruleCutlassMlaOutputArgs *args);
int32_t
ferrule_cutlass_mla_output_launch(const FerruleCutlassMlaOutputArgs *args);
int32_t ferrule_cutlass_main_project_norm_can_implement(
    const FerruleCutlassMainProjectNormArgs *args);
int32_t ferrule_cutlass_main_project_norm_launch(
    const FerruleCutlassMainProjectNormArgs *args);
int32_t ferrule_cutlass_hybrid_mla_attention_can_implement(
    const FerruleCutlassHybridMlaAttentionArgs *args);
int32_t ferrule_cutlass_hybrid_mla_attention_launch(
    const FerruleCutlassHybridMlaAttentionArgs *args);
int32_t ferrule_cutlass_hybrid_mla_explicit_selection_workspace_requirements(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args,
    FerruleCutlassWorkspaceRequirements *requirements);
int32_t ferrule_cutlass_hybrid_mla_explicit_selection_can_implement(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args);
int32_t ferrule_cutlass_hybrid_mla_explicit_selection_launch(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args);
#ifdef FERRULE_CUDA_TEST_ORACLE
int32_t ferrule_cutlass_test_hybrid_mla_explicit_selection_scalar_launch(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args);
// Enqueues the direct-source scalar oracle into oracle_output_f32, then
// compares it with output_f32. compare_result_i32 addresses five device i32
// words: [mismatch_count, first_index, max_abs_bits, first_actual_bits,
// first_expected_bits]. Both scratch buffers are test-build-only caller
// storage.
int32_t ferrule_cutlass_test_hybrid_mla_explicit_selection_compare_launch(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args,
    uint64_t oracle_output_f32, uint64_t compare_result_i32);
#endif
int32_t ferrule_cutlass_proposal_head_can_implement(
    const FerruleCutlassProposalHeadArgs *args);
int32_t ferrule_cutlass_proposal_head_launch(
    const FerruleCutlassProposalHeadArgs *args);
// Reads scalar count/shape fields only; pointer, workspace, and stream fields
// may be zero while querying caller-owned storage requirements.
uint64_t ferrule_cutlass_grouped_fp4_moe_workspace_size(
    const FerruleCutlassGroupedFp4MoeArgs *args);
int32_t ferrule_cutlass_grouped_fp4_moe_can_implement(
    const FerruleCutlassGroupedFp4MoeArgs *args);
int32_t ferrule_cutlass_grouped_fp4_moe_launch(
    const FerruleCutlassGroupedFp4MoeArgs *args);
uint64_t ferrule_cutlass_mxfp4_sfb_storage_bytes(uint32_t n, uint32_t k);
int32_t ferrule_cutlass_prepare_mxfp4_sfb(
    const FerruleCutlassPrepareMxfp4SfbArgs *args);

#ifdef __cplusplus
}
#endif

#endif // FERRULE_CUDA_ABI_CUTLASS_PROVIDER_H_
