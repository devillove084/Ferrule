#ifndef FERRULE_CUTLASS_H_
#define FERRULE_CUTLASS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FERRULE_CUTLASS_ABI_VERSION 8u

#ifndef FERRULE_CUTLASS_TARGET_SM
#define FERRULE_CUTLASS_TARGET_SM 0u
#endif

#define FERRULE_CUTLASS_KERNEL_FP8_QUERY_A_KV_SM121 1u
#define FERRULE_CUTLASS_KERNEL_BF16_COMPRESSOR_SM121 2u
#define FERRULE_CUTLASS_KERNEL_HC_PRODUCER_SM121 3u
#define FERRULE_CUTLASS_KERNEL_SHARED_FFN_SM121 4u
#define FERRULE_CUTLASS_KERNEL_STABLE_FRAME_FP4_MOE_SM121 5u
#define FERRULE_CUTLASS_KERNEL_MLA_OUTPUT_SM121 6u
#define FERRULE_CUTLASS_KERNEL_DSPARK_MAIN_PROJECT_NORM_SM121 7u
#define FERRULE_CUTLASS_KERNEL_DSPARK_HYBRID_MLA_ATTENTION_SM121 8u
#define FERRULE_CUTLASS_KERNEL_DSPARK_PROPOSAL_HEAD_SM121 9u
#define FERRULE_CUTLASS_KERNEL_BIT(id) (1ull << ((id) - 1u))

typedef enum FerruleCutlassStatus {
  FERRULE_CUTLASS_SUCCESS = 0,
  FERRULE_CUTLASS_INVALID_ABI = 1,
  FERRULE_CUTLASS_INVALID_ARGUMENT = 2,
  FERRULE_CUTLASS_LAUNCH_FAILED = 3,
} FerruleCutlassStatus;

typedef struct FerruleCutlassProviderManifest {
  uint32_t abi_version;
  uint32_t cutlass_version;
  uint32_t target_sm;
  uint32_t kernel_count;
  uint64_t kernel_mask;
} FerruleCutlassProviderManifest;

// Semantic one-launch QueryA+KV FP8 projection bundle. All tensors are
// contiguous: activation_fp8 is E4M3 [rows, k], each weight is E4M3 [n, k],
// and each output is F32 [rows, n]. UE8M0 scales cover K128 blocks;
// activation scales are [rows, scale_cols], while weight scales are
// [ceil(n / 128), scale_cols]. Ferrule owns all storage and the CUDA stream.
typedef struct FerruleCutlassFp8QueryAKvArgs {
  uint32_t abi_version;
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
  uint32_t abi_version;
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
// sm121_hc_producer.cuh exactly. Ferrule owns every address and the stream.
typedef struct FerruleCutlassHcProducerArgs {
  uint32_t abi_version;
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
  uint64_t function_col_major_f32;
  uint64_t hc_scale_f32;
  uint64_t hc_base_f32;
  uint64_t layer_rms_weight_f32;
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
// the exact sm121_shared_ffn.cuh Args POD in its native order. The operation is
// the complete gate/up -> SwiGLU -> down chain; Ferrule supplies graph-stable
// compact intermediate storage and the provider performs no allocation or host
// synchronization.
typedef struct FerruleCutlassSharedFfnArgs {
  uint32_t abi_version;

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

// Semantic one-launch stable-frame routed MXFP4 MoE. This is the complete
// routed gate/up -> SwiGLU -> down/rank-output operation. All fields after
// abi_version map one-for-one to sm121_fp4_moe.cuh Args; Ferrule owns route
// metadata, outputs, expert bindings, and the stream.
// Semantic one-launch MLA output bundle. Grouped FP8/E8M0 output-A writes the
// BF16-rounded latent boundary, which is packed once and consumed by FP8/E8M0
// output-B after device-wide barriers. No projection sub-kernel crosses FFI.
typedef struct FerruleCutlassMlaOutputArgs {
  uint32_t abi_version;
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
  uint64_t latent_f32;
  uint64_t latent_fp8;
  uint64_t latent_ue8m0;
  uint64_t output_f32;
  uint64_t stream;
} FerruleCutlassMlaOutputArgs;

// DSpark stage-zero target-tap projection and normalization. The input and
// output are F32 storage carrying the checkpoint's BF16 numerical boundaries.
// Activation FP8/E8M0 and inverse-RMS storage are graph-stable Ferrule-owned
// scratch. The provider performs one cooperative launch.
typedef struct FerruleCutlassDsparkMainProjectNormArgs {
  uint32_t abi_version;
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
} FerruleCutlassDsparkMainProjectNormArgs;

// Checkpoint-native DSpark proposal attention. The release shape is fixed at
// five proposal rows, 64 heads, D=512, a 128-token committed window, and
// 16-token pages. All heads share each latent K/V row. Every proposal query sees
// the complete ephemeral five-row block; only committed context is page-backed.
// Scores, BF16 probabilities, output, and device status are Ferrule-owned.
typedef struct FerruleCutlassDsparkHybridMlaAttentionArgs {
  uint32_t abi_version;
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
  uint64_t output_f32;
  uint64_t status_i32;
  uint64_t stream;
} FerruleCutlassDsparkHybridMlaAttentionArgs;

// Checkpoint-native DSpark proposal head. One semantic launch performs the
// five-row HC head and final norm, one tensor-core BF16 base-LM projection, then
// sequential device-only Markov bias/argmax and confidence. Token dependency
// never crosses the host boundary.
typedef struct FerruleCutlassDsparkProposalHeadArgs {
  uint32_t abi_version;
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
} FerruleCutlassDsparkProposalHeadArgs;

typedef struct FerruleCutlassStableFrameFp4MoeArgs {
  uint32_t abi_version;
  uint32_t reserved0;
  uint32_t input_size;
  uint32_t intermediate_size;
  uint32_t hidden_size;
  uint32_t num_tokens;
  uint32_t num_routes;
  uint32_t slot_capacity;
  uint32_t num_segments;
  float swiglu_limit;

  uint64_t x_packed;
  uint64_t x_scales;

  uint64_t gate_ptrs;
  uint64_t gate_scale_ptrs;
  uint64_t up_ptrs;
  uint64_t up_scale_ptrs;
  uint64_t down_ptrs;
  uint64_t down_scale_ptrs;
  uint64_t slot_generations;

  uint64_t segment_expert_slots;
  uint64_t segment_generations;
  uint64_t segment_token_indices;
  uint64_t segment_route_indices;
  uint64_t segment_route_weights;

  uint64_t segment_states;
  uint64_t segment_bindings;
  uint64_t hidden_f32;
  uint64_t hidden_packed;
  uint64_t hidden_scales;
  uint64_t route_written;
  uint64_t route_error;
  uint64_t route_output;
  uint64_t stream;
} FerruleCutlassStableFrameFp4MoeArgs;

FerruleCutlassProviderManifest ferrule_cutlass_provider_manifest(void);
int32_t ferrule_cutlass_fp8_query_a_kv_can_implement(
    const FerruleCutlassFp8QueryAKvArgs *args);
int32_t ferrule_cutlass_fp8_query_a_kv_launch(
    const FerruleCutlassFp8QueryAKvArgs *args);
int32_t ferrule_cutlass_bf16_compressor_can_implement(
    const FerruleCutlassBf16CompressorArgs *args);
int32_t ferrule_cutlass_bf16_compressor_launch(
    const FerruleCutlassBf16CompressorArgs *args);
int32_t ferrule_cutlass_hc_producer_can_implement(
    const FerruleCutlassHcProducerArgs *args);
int32_t ferrule_cutlass_hc_producer_launch(
    const FerruleCutlassHcProducerArgs *args);
int32_t ferrule_cutlass_shared_ffn_can_implement(
    const FerruleCutlassSharedFfnArgs *args);
int32_t ferrule_cutlass_shared_ffn_launch(
    const FerruleCutlassSharedFfnArgs *args);
int32_t ferrule_cutlass_mla_output_can_implement(
    const FerruleCutlassMlaOutputArgs *args);
int32_t ferrule_cutlass_mla_output_launch(
    const FerruleCutlassMlaOutputArgs *args);
int32_t ferrule_cutlass_dspark_main_project_norm_can_implement(
    const FerruleCutlassDsparkMainProjectNormArgs *args);
int32_t ferrule_cutlass_dspark_main_project_norm_launch(
    const FerruleCutlassDsparkMainProjectNormArgs *args);
int32_t ferrule_cutlass_dspark_hybrid_mla_attention_can_implement(
    const FerruleCutlassDsparkHybridMlaAttentionArgs *args);
int32_t ferrule_cutlass_dspark_hybrid_mla_attention_launch(
    const FerruleCutlassDsparkHybridMlaAttentionArgs *args);
int32_t ferrule_cutlass_dspark_proposal_head_can_implement(
    const FerruleCutlassDsparkProposalHeadArgs *args);
int32_t ferrule_cutlass_dspark_proposal_head_launch(
    const FerruleCutlassDsparkProposalHeadArgs *args);
int32_t ferrule_cutlass_stable_frame_fp4_moe_can_implement(
    const FerruleCutlassStableFrameFp4MoeArgs *args);
int32_t ferrule_cutlass_stable_frame_fp4_moe_launch(
    const FerruleCutlassStableFrameFp4MoeArgs *args);

#ifdef __cplusplus
}
#endif

#endif // FERRULE_CUTLASS_H_
