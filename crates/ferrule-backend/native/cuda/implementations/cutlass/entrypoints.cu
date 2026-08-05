#include "../../abi/cutlass_provider.h"
#include "architectures/profile.cuh"
#include "capabilities/bf16_compressor.cuh"
#include "capabilities/fp8_projection.cuh"
#include "capabilities/hybrid_mla_attention.cuh"
#include "capabilities/hybrid_mla_attention/router.cuh"
#include "capabilities/hyper_connection.cuh"
#include "capabilities/main_project_norm.cuh"
#include "capabilities/mla_output.cuh"
#include "capabilities/proposal_head.cuh"
#include "capabilities/shared_ffn.cuh"

// Device-debug disables the optimizer required to compile CUTLASS's SM103
// grouped kernels without unreliable debug-register metadata. Treat that mode
// as an incomplete native pipeline and do not publish grouped FP4 MoE or SFB
// preparation; release SM103 builds retain the complete implementation.
#if FERRULE_CUDA_HAS_NATIVE_GROUPED_FP4_MOE &&                                 \
    FERRULE_CUDA_TARGET_SM == 103 && !defined(__CUDACC_DEBUG__)
#define FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE 1
#include "architectures/sm103/grouped_fp4_moe.cuh"
#else
#define FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE 0
#endif

#include <cuda_runtime_api.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cutlass/version.h>

static_assert(CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 6 && CUTLASS_PATCH == 1,
              "Ferrule's CUTLASS provider is pinned to CUTLASS 4.6.1");
static_assert(sizeof(FerruleCutlassProviderManifest) == 8,
              "Ferrule CUTLASS manifest ABI layout changed");
static_assert(sizeof(FerruleCutlassFp8QueryAKvArgs) == 96,
              "Ferrule CUTLASS FP8 QueryA+KV ABI layout changed");
static_assert(sizeof(FerruleCutlassBf16CompressorArgs) == 72,
              "Ferrule CUTLASS BF16 compressor ABI layout changed");
static_assert(sizeof(FerruleCutlassHcProducerArgs) == 144,
              "Ferrule CUTLASS HC producer ABI layout changed");
static_assert(sizeof(FerruleCutlassSharedFfnArgs) == 160,
              "Ferrule CUTLASS shared FFN ABI layout changed");
static_assert(sizeof(FerruleCutlassMlaOutputArgs) == 120,
              "Ferrule CUTLASS MLA output ABI layout changed");
static_assert(sizeof(FerruleCutlassMainProjectNormArgs) == 104,
              "Ferrule CUTLASS main-project/norm ABI layout changed");
static_assert(sizeof(FerruleCutlassHybridMlaAttentionArgs) == 160,
              "Ferrule CUTLASS hybrid-attention ABI layout changed");
static_assert(sizeof(FerruleCutlassHybridMlaExplicitSelectionArgs) == 208,
              "Ferrule CUTLASS explicit-selection ABI layout changed");
static_assert(sizeof(FerruleCutlassWorkspaceRequirements) == 16,
              "Ferrule CUTLASS workspace requirements ABI layout changed");
static_assert(sizeof(FerruleCutlassProposalHeadArgs) == 184,
              "Ferrule CUTLASS proposal-head ABI layout changed");
static_assert(sizeof(FerruleCutlassGroupedFp4MoeArgs) == 224,
              "Ferrule CUTLASS grouped FP4 MoE ABI layout changed");
static_assert(sizeof(FerruleCutlassPrepareMxfp4SfbArgs) == 40,
              "Ferrule CUTLASS MXFP4 SFB prepare ABI layout changed");

#define FERRULE_CUTLASS_ASSERT_ALIGNMENT(type)                                 \
  static_assert(alignof(type) == 8, #type " ABI alignment changed")
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassProviderManifest);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassFp8QueryAKvArgs);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassBf16CompressorArgs);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassHcProducerArgs);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassSharedFfnArgs);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassMlaOutputArgs);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassMainProjectNormArgs);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassHybridMlaAttentionArgs);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassHybridMlaExplicitSelectionArgs);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassWorkspaceRequirements);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassProposalHeadArgs);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassGroupedFp4MoeArgs);
FERRULE_CUTLASS_ASSERT_ALIGNMENT(FerruleCutlassPrepareMxfp4SfbArgs);
#undef FERRULE_CUTLASS_ASSERT_ALIGNMENT

#define FERRULE_CUTLASS_ASSERT_OFFSET(type, field, expected)                   \
  static_assert(offsetof(type, field) == expected,                             \
                #type "." #field " ABI offset changed")

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProviderManifest, kernel_mask, 0);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, rows, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, n1, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, n2, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, k, 12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, scale_cols, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, activation_fp8,
                              24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, activation_ue8m0,
                              32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, query_a_weight_fp8,
                              40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs,
                              query_a_weight_ue8m0, 48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, kv_weight_fp8, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, kv_weight_ue8m0,
                              64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, query_a_output_f32,
                              72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, kv_output_f32, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassFp8QueryAKvArgs, stream, 88);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs, rows, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs, n1, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs, n2, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs, k, 12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs, reserved0, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs, activation_f32,
                              24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs,
                              projection1_weight_bf16, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs,
                              projection2_weight_bf16, 40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs,
                              projection1_output_f32, 48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs,
                              projection2_output_f32, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassBf16CompressorArgs, stream, 64);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, rows, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hc, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hidden, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, mix, 12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, sinkhorn_iters, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hc_eps, 20);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hc_norm_eps, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, layer_rms_eps, 28);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, reserved, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, state_f32, 40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs,
                              function_col_major_f32, 48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hc_scale_f32, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hc_base_f32, 64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs,
                              layer_rms_weight_f32, 72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hidden_f32, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, normalized_f32, 88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, packed_e4m3, 96);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, scales_ue8m0, 104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, split_pre_f32, 112);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, split_post_f32,
                              120);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, split_comb_f32,
                              128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, stream, 136);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, input_fp8, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, input_ue8m0, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, gate_weight_fp8, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, gate_weight_ue8m0,
                              24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, up_weight_fp8, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, up_weight_ue8m0, 40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, down_weight_fp8, 48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, down_weight_ue8m0,
                              56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, hidden_f32, 64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, hidden_fp8, 72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, hidden_ue8m0, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, output_f32, 88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, rows, 96);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, input_size, 100);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, intermediate_size,
                              104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, output_size, 108);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, gate_block_m, 112);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, gate_block_k, 116);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, up_block_m, 120);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, up_block_k, 124);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, down_block_m, 128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, down_block_k, 132);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, output_scale, 136);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, swiglu_limit, 140);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, flags, 144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, stream, 152);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, rows, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, context_size, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, groups, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, group_input_size,
                              12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, rank, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, latent_size, 20);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, hidden_size, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, output_a_scale_cols,
                              28);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, reserved0, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, context_f32, 40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, output_a_weight_fp8,
                              48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs,
                              output_a_weight_ue8m0, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, output_b_weight_fp8,
                              64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs,
                              output_b_weight_ue8m0, 72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, latent_f32, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, latent_fp8, 88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, latent_ue8m0, 96);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, output_f32, 104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, stream, 112);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, rows, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, input_size, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, output_size,
                              8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, scale_cols,
                              12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, reserved0, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, rms_eps, 20);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, reserved1, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, input_f32, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, activation_fp8,
                              40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs,
                              activation_ue8m0, 48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, weight_fp8,
                              56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, weight_ue8m0,
                              64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs,
                              norm_weight_f32, 72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, inv_rms_f32,
                              80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, output_f32,
                              88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMainProjectNormArgs, stream, 96);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, block_rows,
                              0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, heads, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, head_dim,
                              8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              sequence_tokens, 12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, window_size,
                              16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, page_tokens,
                              20);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              elements_per_token, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, layer_index,
                              28);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, layer_count,
                              32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              block_slot_offset, 36);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              block_slot_count, 40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              softmax_scale, 44);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, reserved0,
                              48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              context_plane_elements, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, query_f32,
                              64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              context_plane_f32, 72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              block_kv_f32, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              block_slots_i32, 88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              attention_sink_f32, 96);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, query_bf16,
                              104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              gathered_kv_bf16, 112);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, scores_f32,
                              120);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              probabilities_bf16, 128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, output_f32,
                              136);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, status_i32,
                              144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, stream,
                              152);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              kind, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              rows, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              tokens_per_sequence, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              kv_len, 12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              heads, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              head_dim, 20);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              selected_width, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              page_tokens, 28);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              first_elements_per_token, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              second_elements_per_token, 36);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              layer_index, 40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              layer_count, 44);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              flags, 48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              softmax_scale, 52);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              reserved0, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              first_plane_elements, 64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              second_plane_elements, 72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              query_f32, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              first_plane_f32, 88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              second_plane_f32, 96);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              block_slots_i32, 104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              block_offsets_i32, 112);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              sequence_kv_lens_i32, 120);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              row_sequence_ids_i32, 128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              row_kv_lens_i32, 136);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              selected_indices_i32, 144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              selectors_i32, 152);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              attention_sink_f32, 160);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              workspace, 168);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              workspace_bytes, 176);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              output_f32, 184);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              status_i32, 192);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              stream, 200);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassWorkspaceRequirements, bytes, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassWorkspaceRequirements, alignment,
                              8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassWorkspaceRequirements, reserved,
                              12);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, rows, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, hc, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, hidden, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, vocab, 12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, markov_rank, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, partial_capacity,
                              20);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, reserved0, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, hc_eps, 28);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, norm_eps, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, hc_state_f32, 40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, hc_function_f32,
                              48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, hc_scale_f32, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, hc_base_f32, 64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, norm_weight_f32,
                              72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, lm_head_bf16, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, markov_w1_bf16,
                              88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, markov_w2_bf16,
                              96);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs,
                              confidence_weight_bf16, 104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, hidden_f32, 112);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, normalized_f32,
                              120);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, base_logits_f32,
                              128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs,
                              partial_values_f32, 136);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs,
                              partial_indices_i32, 144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, token_ids_i32,
                              152);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, confidence_f32,
                              160);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, status_i32, 168);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassProposalHeadArgs, stream, 176);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                              active_group_count, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                              small_group_count, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, slot_capacity,
                              8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, max_group_rows,
                              12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                              total_routed_rows, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, num_tokens, 20);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, num_routes, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, input_size, 28);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                              intermediate_size, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, hidden_size, 36);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, swiglu_limit,
                              40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                              active_expert_slots, 48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                              active_group_generations, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                              expert_route_indptr, 64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                              expert_route_counts, 72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                              route_token_indices, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, route_indices,
                              88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, route_weights,
                              96);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, slot_generations,
                              104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, gate_ptrs, 112);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, gate_scale_ptrs,
                              120);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, up_ptrs, 128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, up_scale_ptrs,
                              136);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, down_ptrs, 144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, down_scale_ptrs,
                              152);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, input_packed,
                              160);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, input_scales,
                              168);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, route_output,
                              176);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, route_written,
                              184);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, route_error,
                              192);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, workspace, 200);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, workspace_bytes,
                              208);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, stream, 216);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassPrepareMxfp4SfbArgs, n, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassPrepareMxfp4SfbArgs, k, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassPrepareMxfp4SfbArgs, reserved0, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassPrepareMxfp4SfbArgs, linear_source,
                              16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassPrepareMxfp4SfbArgs,
                              prepared_destination, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassPrepareMxfp4SfbArgs, stream, 32);

#undef FERRULE_CUTLASS_ASSERT_OFFSET

namespace {

namespace fp8_prefill = ferrule::cuda::cutlass::capabilities::fp8_projection;
namespace bf16_prefill = ferrule::cuda::cutlass::capabilities::bf16_compressor;
namespace main_project_norm =
    ferrule::cuda::cutlass::capabilities::main_project_norm;
namespace hybrid_mla_attention =
    ferrule::cuda::cutlass::capabilities::hybrid_mla_attention;

namespace proposal_head = ferrule::cuda::cutlass::capabilities::proposal_head;
namespace hc_producer = ferrule::cuda::cutlass::capabilities::hyper_connection;
namespace mla_output = ferrule::cuda::cutlass::capabilities::mla_output;
namespace shared_ffn = ferrule::cuda::cutlass::capabilities::shared_ffn;
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
namespace grouped_fp4_moe =
    ferrule::cuda::cutlass::architectures::sm103::grouped_fp4_moe;
#endif

static_assert(sizeof(bf16_prefill::Args) ==
              sizeof(FerruleCutlassBf16CompressorArgs));
static_assert(sizeof(hc_producer::HcPreRmsNormFp8Args) ==
              sizeof(FerruleCutlassHcProducerArgs));
static_assert(sizeof(shared_ffn::Args) == 152);
static_assert(sizeof(mla_output::Args) == sizeof(FerruleCutlassMlaOutputArgs));
static_assert(sizeof(main_project_norm::Args) ==
              sizeof(FerruleCutlassMainProjectNormArgs));
static_assert(sizeof(hybrid_mla_attention::Args) ==
              sizeof(FerruleCutlassHybridMlaAttentionArgs));

static_assert(sizeof(proposal_head::Args) ==
              sizeof(FerruleCutlassProposalHeadArgs));
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
static_assert(sizeof(grouped_fp4_moe::GroupedFp4MoeArgs) ==
              offsetof(FerruleCutlassGroupedFp4MoeArgs, workspace));
static_assert(alignof(grouped_fp4_moe::GroupedFp4MoeArgs) ==
              alignof(FerruleCutlassGroupedFp4MoeArgs));
#endif

#define FERRULE_CUTLASS_ASSERT_SAME_OFFSET(c_type, native_type, field)         \
  static_assert(offsetof(c_type, field) == offsetof(native_type, field),       \
                #c_type "." #field " no longer matches native Args")
static_assert(offsetof(FerruleCutlassBf16CompressorArgs, rows) ==
              offsetof(bf16_prefill::Args, m));
static_assert(offsetof(FerruleCutlassBf16CompressorArgs, reserved0) ==
              offsetof(bf16_prefill::Args, reserved));
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_prefill::Args, n1);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_prefill::Args, n2);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_prefill::Args, k);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_prefill::Args, activation_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_prefill::Args, projection1_weight_bf16);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_prefill::Args, projection2_weight_bf16);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_prefill::Args, projection1_output_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_prefill::Args, projection2_output_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_prefill::Args, stream);

FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassHcProducerArgs,
                                   hc_producer::HcPreRmsNormFp8Args, rows);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassHcProducerArgs,
                                   hc_producer::HcPreRmsNormFp8Args, state_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassHcProducerArgs,
                                   hc_producer::HcPreRmsNormFp8Args, stream);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassSharedFfnArgs,
                                   shared_ffn::Args, input_fp8);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassSharedFfnArgs,
                                   shared_ffn::Args, hidden_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassSharedFfnArgs,
                                   shared_ffn::Args, hidden_fp8);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassSharedFfnArgs,
                                   shared_ffn::Args, output_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassSharedFfnArgs,
                                   shared_ffn::Args, rows);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassSharedFfnArgs,
                                   shared_ffn::Args, flags);

static_assert(offsetof(FerruleCutlassMlaOutputArgs, reserved0) ==
              offsetof(mla_output::Args, reserved));
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, rows);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, context_size);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, groups);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, group_input_size);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, rank);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, latent_size);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, hidden_size);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, output_a_scale_cols);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, context_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, output_a_weight_fp8);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, output_a_weight_ue8m0);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, output_b_weight_fp8);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, output_b_weight_ue8m0);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, latent_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, latent_fp8);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, latent_ue8m0);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, output_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassMlaOutputArgs,
                                   mla_output::Args, stream);

#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   active_group_count);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   small_group_count);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   slot_capacity);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   max_group_rows);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   total_routed_rows);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   num_tokens);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   num_routes);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   input_size);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   intermediate_size);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   hidden_size);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   swiglu_limit);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   active_expert_slots);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   active_group_generations);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   expert_route_indptr);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   expert_route_counts);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   route_token_indices);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   route_indices);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   route_weights);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   slot_generations);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   gate_ptrs);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   gate_scale_ptrs);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs, up_ptrs);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   up_scale_ptrs);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   down_ptrs);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   down_scale_ptrs);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   input_packed);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   input_scales);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   route_output);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   route_written);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   route_error);
#endif

#undef FERRULE_CUTLASS_ASSERT_SAME_OFFSET

using Bf16Mma = cute::SM80_16x8x16_F32BF16BF16F32_TN;

struct alignas(16) Bf16CompressorSharedStorage {
  alignas(16) uint16_t activation[128];
  alignas(16) uint16_t projection1_weight[256];
  alignas(16) uint16_t projection2_weight[256];
};

__device__ __forceinline__ uint16_t f32_to_bf16_rne(float value) {
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  }
  uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

__device__ __forceinline__ void
load_a_fragment_16x32_bytes(const uint8_t *shared, uint32_t lane,
                            uint32_t (&fragment)[4]) {
  uint32_t quad = lane >> 3;
  uint32_t row = (lane & 7u) + ((quad & 1u) != 0 ? 8u : 0u);
  uint32_t column_bytes = quad >= 2 ? 16u : 0u;
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + row * 32u + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1], fragment[2],
                                fragment[3]);
}

__device__ __forceinline__ void
load_b_fragment_16_byte_rows(const uint8_t *shared, uint32_t lane,
                             uint32_t (&fragment)[2]) {
  auto const &source =
      *reinterpret_cast<const cute::uint128_t *>(shared + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void mma_bf16(float (&accumulator)[4],
                                         const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
  Bf16Mma::fma(accumulator[0], accumulator[1], accumulator[2], accumulator[3],
               a[0], a[1], a[2], a[3], b[0], b[1], accumulator[0],
               accumulator[1], accumulator[2], accumulator[3]);
}

__global__ void
ferrule_bf16_compressor_single_row(FerruleCutlassBf16CompressorArgs args) {
  __shared__ Bf16CompressorSharedStorage shared;

  uint32_t lane = threadIdx.x;
  uint32_t channel_base = blockIdx.x * 16u;
  auto *activation = reinterpret_cast<const float *>(
      static_cast<uintptr_t>(args.activation_f32));
  auto *projection1_weight = reinterpret_cast<const uint16_t *>(
      static_cast<uintptr_t>(args.projection1_weight_bf16));
  auto *projection2_weight = reinterpret_cast<const uint16_t *>(
      static_cast<uintptr_t>(args.projection2_weight_bf16));
  auto *projection1_output = reinterpret_cast<float *>(
      static_cast<uintptr_t>(args.projection1_output_f32));
  auto *projection2_output = reinterpret_cast<float *>(
      static_cast<uintptr_t>(args.projection2_output_f32));

  float projection1_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float projection2_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t k_base = 0; k_base < args.k; k_base += 16u) {
    if (lane < 16u) {
      const uint32_t channel = channel_base + lane;
      auto *projection1_destination =
          reinterpret_cast<uint4 *>(shared.projection1_weight + lane * 16u);
      auto *projection2_destination =
          reinterpret_cast<uint4 *>(shared.projection2_weight + lane * 16u);
      const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
      if (channel < args.n1) {
        auto *source = reinterpret_cast<const uint4 *>(
            projection1_weight + static_cast<uint64_t>(channel) * args.k +
            k_base);
        projection1_destination[0] = source[0];
        projection1_destination[1] = source[1];
      } else {
        projection1_destination[0] = zero;
        projection1_destination[1] = zero;
      }
      if (channel < args.n2) {
        auto *source = reinterpret_cast<const uint4 *>(
            projection2_weight + static_cast<uint64_t>(channel) * args.k +
            k_base);
        projection2_destination[0] = source[0];
        projection2_destination[1] = source[1];
      } else {
        projection2_destination[0] = zero;
        projection2_destination[1] = zero;
      }
    }

    for (uint32_t linear = lane; linear < 128u; linear += 32u) {
      uint32_t k_local = linear >> 3;
      uint32_t row = linear & 7u;
      float value = row < args.rows
                        ? activation[static_cast<uint64_t>(row) * args.k +
                                     k_base + k_local]
                        : 0.0f;
      shared.activation[linear] = f32_to_bf16_rne(value);
    }
    __syncthreads();

    uint32_t projection1_fragment[4];
    uint32_t projection2_fragment[4];
    uint32_t activation_fragment[2];
    load_a_fragment_16x32_bytes(
        reinterpret_cast<const uint8_t *>(shared.projection1_weight), lane,
        projection1_fragment);
    load_a_fragment_16x32_bytes(
        reinterpret_cast<const uint8_t *>(shared.projection2_weight), lane,
        projection2_fragment);
    load_b_fragment_16_byte_rows(
        reinterpret_cast<const uint8_t *>(shared.activation), lane,
        activation_fragment);
    mma_bf16(projection1_accumulator, projection1_fragment,
             activation_fragment);
    mma_bf16(projection2_accumulator, projection2_fragment,
             activation_fragment);
    __syncthreads();
  }

  uint32_t channel_group = lane >> 2;
  uint32_t row_pair = lane & 3u;
#pragma unroll
  for (uint32_t element = 0; element < 4u; ++element) {
    uint32_t channel = channel_base + channel_group + (element >= 2u ? 8u : 0u);
    uint32_t row = row_pair * 2u + (element & 1u);
    if (row < args.rows && channel < args.n1) {
      projection1_output[static_cast<uint64_t>(row) * args.n1 + channel] =
          projection1_accumulator[element];
    }
    if (row < args.rows && channel < args.n2) {
      projection2_output[static_cast<uint64_t>(row) * args.n2 + channel] =
          projection2_accumulator[element];
    }
  }
}

// The BF16 compressor retains its latency-tuned single-tile schedule for
// verification-width inputs; QueryA+KV always uses its pipelined provider.
inline constexpr uint32_t kTinyLinearRows = 8u;

fp8_prefill::Args make_prefill_args(const FerruleCutlassFp8QueryAKvArgs &args) {
  return fp8_prefill::Args{
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.activation_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.activation_ue8m0)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.query_a_weight_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.query_a_weight_ue8m0)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.kv_weight_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.kv_weight_ue8m0)),
      reinterpret_cast<float *>(
          static_cast<uintptr_t>(args.query_a_output_f32)),
      reinterpret_cast<float *>(static_cast<uintptr_t>(args.kv_output_f32)),
      args.rows,
      args.n1,
      args.n2,
      args.k,
  };
}

bf16_prefill::Args
make_prefill_args(const FerruleCutlassBf16CompressorArgs &args) {
  return bf16_prefill::Args{
      args.rows,
      args.n1,
      args.n2,
      args.k,
      args.reserved0,
      args.activation_f32,
      args.projection1_weight_bf16,
      args.projection2_weight_bf16,
      args.projection1_output_f32,
      args.projection2_output_f32,
      args.stream,
  };
}

hc_producer::HcPreRmsNormFp8Args
make_hc_producer_args(const FerruleCutlassHcProducerArgs &args) {
  return hc_producer::HcPreRmsNormFp8Args{
      args.rows,
      args.hc,
      args.hidden,
      args.mix,
      args.sinkhorn_iters,
      args.hc_eps,
      args.hc_norm_eps,
      args.layer_rms_eps,
      args.reserved,
      args.state_f32,
      args.function_col_major_f32,
      args.hc_scale_f32,
      args.hc_base_f32,
      args.layer_rms_weight_f32,
      args.hidden_f32,
      args.normalized_f32,
      args.packed_e4m3,
      args.scales_ue8m0,
      args.split_pre_f32,
      args.split_post_f32,
      args.split_comb_f32,
      args.stream,
  };
}

shared_ffn::Args make_shared_ffn_args(const FerruleCutlassSharedFfnArgs &args) {
  return shared_ffn::Args{
      reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(args.input_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.input_ue8m0)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.gate_weight_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.gate_weight_ue8m0)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.up_weight_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.up_weight_ue8m0)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.down_weight_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.down_weight_ue8m0)),
      reinterpret_cast<float *>(static_cast<uintptr_t>(args.hidden_f32)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.hidden_fp8)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.hidden_ue8m0)),
      reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_f32)),
      args.rows,
      args.input_size,
      args.intermediate_size,
      args.output_size,
      args.gate_block_m,
      args.gate_block_k,
      args.up_block_m,
      args.up_block_k,
      args.down_block_m,
      args.down_block_k,
      args.output_scale,
      args.swiglu_limit,
      args.flags,
  };
}

mla_output::Args make_mla_output_args(const FerruleCutlassMlaOutputArgs &args) {
  return mla_output::Args{
      args.rows,
      args.context_size,
      args.groups,
      args.group_input_size,
      args.rank,
      args.latent_size,
      args.hidden_size,
      args.output_a_scale_cols,
      args.reserved0,
      args.context_f32,
      args.output_a_weight_fp8,
      args.output_a_weight_ue8m0,
      args.output_b_weight_fp8,
      args.output_b_weight_ue8m0,
      args.latent_f32,
      args.latent_fp8,
      args.latent_ue8m0,
      args.output_f32,
      args.stream,
  };
}

main_project_norm::Args
make_main_project_norm_args(const FerruleCutlassMainProjectNormArgs &args) {
  return main_project_norm::Args{
      args.rows,
      args.input_size,
      args.output_size,
      args.scale_cols,
      args.reserved0,
      args.rms_eps,
      args.reserved1,
      args.input_f32,
      args.activation_fp8,
      args.activation_ue8m0,
      args.weight_fp8,
      args.weight_ue8m0,
      args.norm_weight_f32,
      args.inv_rms_f32,
      args.output_f32,
      args.stream,
  };
}

hybrid_mla_attention::Args make_hybrid_mla_attention_args(
    const FerruleCutlassHybridMlaAttentionArgs &args) {
  return hybrid_mla_attention::Args{
      args.block_rows,         args.heads,
      args.head_dim,           args.sequence_tokens,
      args.window_size,        args.page_tokens,
      args.elements_per_token, args.layer_index,
      args.layer_count,        args.block_slot_offset,
      args.block_slot_count,   args.softmax_scale,
      args.reserved0,          args.context_plane_elements,
      args.query_f32,          args.context_plane_f32,
      args.block_kv_f32,       args.block_slots_i32,
      args.attention_sink_f32, args.query_bf16,
      args.gathered_kv_bf16,   args.scores_f32,
      args.probabilities_bf16, args.output_f32,
      args.status_i32,         args.stream,
  };
}

template <class T> T *native_pointer(uint64_t address) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(address));
}

hybrid_mla_attention::ExplicitSelectionArgs
make_hybrid_mla_explicit_selection_args(
    const FerruleCutlassHybridMlaExplicitSelectionArgs &args) {
  return hybrid_mla_attention::ExplicitSelectionArgs{
      args.kind,
      args.rows,
      args.tokens_per_sequence,
      args.kv_len,
      args.heads,
      args.head_dim,
      args.selected_width,
      args.page_tokens,
      args.first_elements_per_token,
      args.second_elements_per_token,
      args.layer_index,
      args.layer_count,
      args.flags,
      args.softmax_scale,
      args.reserved0,
      args.first_plane_elements,
      args.second_plane_elements,
      native_pointer<const float>(args.query_f32),
      native_pointer<const float>(args.first_plane_f32),
      native_pointer<const float>(args.second_plane_f32),
      native_pointer<const int32_t>(args.block_slots_i32),
      native_pointer<const int32_t>(args.block_offsets_i32),
      native_pointer<const int32_t>(args.sequence_kv_lens_i32),
      native_pointer<const int32_t>(args.row_sequence_ids_i32),
      native_pointer<const int32_t>(args.row_kv_lens_i32),
      native_pointer<const int32_t>(args.selected_indices_i32),
      native_pointer<const int32_t>(args.selectors_i32),
      native_pointer<const float>(args.attention_sink_f32),
      native_pointer<void>(args.workspace),
      args.workspace_bytes,
      native_pointer<float>(args.output_f32),
      native_pointer<int32_t>(args.status_i32),
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args.stream)),
  };
}

proposal_head::Args
make_proposal_head_args(const FerruleCutlassProposalHeadArgs &args) {
  return proposal_head::Args{
      args.rows,
      args.hc,
      args.hidden,
      args.vocab,
      args.markov_rank,
      args.partial_capacity,
      args.reserved0,
      args.hc_eps,
      args.norm_eps,
      args.hc_state_f32,
      args.hc_function_f32,
      args.hc_scale_f32,
      args.hc_base_f32,
      args.norm_weight_f32,
      args.lm_head_bf16,
      args.markov_w1_bf16,
      args.markov_w2_bf16,
      args.confidence_weight_bf16,
      args.hidden_f32,
      args.normalized_f32,
      args.base_logits_f32,
      args.partial_values_f32,
      args.partial_indices_i32,
      args.token_ids_i32,
      args.confidence_f32,
      args.status_i32,
      args.stream,
  };
}

#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
bool grouped_fp4_moe_options(grouped_fp4_moe::LaunchOptions &options) {
  int device_id = -1;
  int sm_count = 0;
  if (cudaGetDevice(&device_id) != cudaSuccess ||
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                             device_id) != cudaSuccess ||
      device_id < 0 || sm_count <= 0) {
    return false;
  }
  options.device_id = device_id;
  options.sm_count = sm_count;
  options.two_sm_min_rows = grouped_fp4_moe::kDefault2SmMinRows;
  return true;
}

grouped_fp4_moe::GroupedFp4MoeArgs
make_grouped_fp4_moe_args(const FerruleCutlassGroupedFp4MoeArgs &args) {
  return grouped_fp4_moe::GroupedFp4MoeArgs{
      args.active_group_count,
      args.small_group_count,
      args.slot_capacity,
      args.max_group_rows,
      args.total_routed_rows,
      args.num_tokens,
      args.num_routes,
      args.input_size,
      args.intermediate_size,
      args.hidden_size,
      args.swiglu_limit,
      args.active_expert_slots,
      args.active_group_generations,
      args.expert_route_indptr,
      args.expert_route_counts,
      args.route_token_indices,
      args.route_indices,
      args.route_weights,
      args.slot_generations,
      args.gate_ptrs,
      args.gate_scale_ptrs,
      args.up_ptrs,
      args.up_scale_ptrs,
      args.down_ptrs,
      args.down_scale_ptrs,
      args.input_packed,
      args.input_scales,
      args.route_output,
      args.route_written,
      args.route_error,
  };
}

int32_t grouped_fp4_moe_status(grouped_fp4_moe::Status status) {
  switch (status) {
  case grouped_fp4_moe::Status::kSuccess:
    return FERRULE_CUTLASS_SUCCESS;
  case grouped_fp4_moe::Status::kInvalidArgument:
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  case grouped_fp4_moe::Status::kUnsupportedResources:
    return FERRULE_CUTLASS_UNSUPPORTED;
  case grouped_fp4_moe::Status::kLaunchFailed:
    return FERRULE_CUTLASS_LAUNCH_FAILED;
  }
  return FERRULE_CUTLASS_LAUNCH_FAILED;
}
#endif

int32_t helper_launch_status(cudaError_t status) {
  if (status == cudaSuccess) {
    return FERRULE_CUTLASS_SUCCESS;
  }
  return status == cudaErrorInvalidValue ? FERRULE_CUTLASS_INVALID_ARGUMENT
                                         : FERRULE_CUTLASS_LAUNCH_FAILED;
}

} // namespace

extern "C" FerruleCutlassProviderManifest
ferrule_cutlass_provider_manifest(void) {
  return FerruleCutlassProviderManifest{
      FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_FP8_QUERY_A_KV) |
          FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_BF16_COMPRESSOR) |
          FERRULE_CUTLASS_KERNEL_BIT(
              FERRULE_CUTLASS_KERNEL_HYPER_CONNECTION_PRODUCER) |
          FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_SHARED_FFN) |
          (FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
               ? FERRULE_CUTLASS_KERNEL_BIT(
                     FERRULE_CUTLASS_KERNEL_GROUPED_FP4_MOE)
               : 0ull) |
          FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_MLA_OUTPUT) |
          FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_MAIN_PROJECT_NORM) |
          FERRULE_CUTLASS_KERNEL_BIT(
              FERRULE_CUTLASS_KERNEL_HYBRID_MLA_ATTENTION) |
          FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_PROPOSAL_HEAD) |
          FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_FP8_PROJECTION),
  };
}

extern "C" int32_t ferrule_cutlass_fp8_query_a_kv_can_implement(
    const FerruleCutlassFp8QueryAKvArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  if (args->scale_cols != args->k / 128u) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  return fp8_prefill::validate(make_prefill_args(*args)) ==
                 fp8_prefill::ValidationResult::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
}

extern "C" int32_t ferrule_cutlass_fp8_query_a_kv_launch(
    const FerruleCutlassFp8QueryAKvArgs *args) {
  int32_t status = ferrule_cutlass_fp8_query_a_kv_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }

  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return helper_launch_status(
      fp8_prefill::launch(make_prefill_args(*args), stream));
}

extern "C" int32_t ferrule_cutlass_fp8_projection_can_implement(
    const FerruleCutlassFp8QueryAKvArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  if (args->n2 != 0u || args->kv_weight_fp8 != 0u ||
      args->kv_weight_ue8m0 != 0u || args->kv_output_f32 != 0u ||
      args->scale_cols != args->k / 128u) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  return fp8_prefill::validate_single(make_prefill_args(*args)) ==
                 fp8_prefill::ValidationResult::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
}

extern "C" int32_t ferrule_cutlass_fp8_projection_launch(
    const FerruleCutlassFp8QueryAKvArgs *args) {
  const int32_t status = ferrule_cutlass_fp8_projection_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return helper_launch_status(
      fp8_prefill::launch_single(make_prefill_args(*args), stream));
}

extern "C" int32_t ferrule_cutlass_bf16_compressor_can_implement(
    const FerruleCutlassBf16CompressorArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  return bf16_prefill::validate(make_prefill_args(*args)) ==
                 bf16_prefill::ValidationResult::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
}

extern "C" int32_t ferrule_cutlass_bf16_compressor_launch(
    const FerruleCutlassBf16CompressorArgs *args) {
  int32_t status = ferrule_cutlass_bf16_compressor_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }

  if (args->rows > kTinyLinearRows) {
    return helper_launch_status(bf16_prefill::launch(make_prefill_args(*args)));
  }
  uint32_t max_n = args->n1 > args->n2 ? args->n1 : args->n2;
  uint32_t blocks =
      static_cast<uint32_t>((static_cast<uint64_t>(max_n) + 15u) / 16u);
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  ferrule_bf16_compressor_single_row<<<blocks, 32, 0, stream>>>(*args);
  return cudaGetLastError() == cudaSuccess ? FERRULE_CUTLASS_SUCCESS
                                           : FERRULE_CUTLASS_LAUNCH_FAILED;
}

extern "C" int32_t ferrule_cutlass_hc_producer_can_implement(
    const FerruleCutlassHcProducerArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hc_producer_args(*args);
  return hc_producer::validate_hc_pre_rmsnorm_fp8(native_args)
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
}

extern "C" int32_t
ferrule_cutlass_hc_producer_launch(const FerruleCutlassHcProducerArgs *args) {
  const int32_t status = ferrule_cutlass_hc_producer_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  return helper_launch_status(
      hc_producer::launch_hc_pre_rmsnorm_fp8(make_hc_producer_args(*args)));
}

extern "C" int32_t ferrule_cutlass_shared_ffn_can_implement(
    const FerruleCutlassSharedFfnArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  return shared_ffn::validate(make_shared_ffn_args(*args)) ==
                 shared_ffn::ValidationResult::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
}

extern "C" int32_t
ferrule_cutlass_shared_ffn_launch(const FerruleCutlassSharedFfnArgs *args) {
  const int32_t status = ferrule_cutlass_shared_ffn_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return helper_launch_status(
      shared_ffn::launch(make_shared_ffn_args(*args), stream));
}

extern "C" int32_t ferrule_cutlass_mla_output_can_implement(
    const FerruleCutlassMlaOutputArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_mla_output_args(*args);
  return static_cast<int32_t>(mla_output::validate(&native_args));
}

extern "C" int32_t
ferrule_cutlass_mla_output_launch(const FerruleCutlassMlaOutputArgs *args) {
  const int32_t status = ferrule_cutlass_mla_output_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_mla_output_args(*args);
  return static_cast<int32_t>(mla_output::launch(&native_args));
}

extern "C" int32_t ferrule_cutlass_main_project_norm_can_implement(
    const FerruleCutlassMainProjectNormArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_main_project_norm_args(*args);
  return static_cast<int32_t>(main_project_norm::validate(&native_args));
}

extern "C" int32_t ferrule_cutlass_main_project_norm_launch(
    const FerruleCutlassMainProjectNormArgs *args) {
  const int32_t status = ferrule_cutlass_main_project_norm_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_main_project_norm_args(*args);
  return static_cast<int32_t>(main_project_norm::launch(&native_args));
}

extern "C" int32_t ferrule_cutlass_hybrid_mla_attention_can_implement(
    const FerruleCutlassHybridMlaAttentionArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_attention_args(*args);
  return static_cast<int32_t>(hybrid_mla_attention::validate(&native_args));
}

extern "C" int32_t ferrule_cutlass_hybrid_mla_attention_launch(
    const FerruleCutlassHybridMlaAttentionArgs *args) {
  const int32_t status =
      ferrule_cutlass_hybrid_mla_attention_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_hybrid_mla_attention_args(*args);
  return static_cast<int32_t>(hybrid_mla_attention::launch(&native_args));
}

extern "C" int32_t
ferrule_cutlass_hybrid_mla_explicit_selection_workspace_requirements(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args,
    FerruleCutlassWorkspaceRequirements *requirements) {
  if (args == nullptr || requirements == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_explicit_selection_args(*args);
  hybrid_mla_attention::explicit_selection_workspace::Requirements
      native_requirements{};
  const auto status =
      hybrid_mla_attention::explicit_selection_workspace_requirements(
          &native_args, &native_requirements);
  if (status != hybrid_mla_attention::ExplicitSelectionStatus::kSuccess) {
    return static_cast<int32_t>(status);
  }
  *requirements = FerruleCutlassWorkspaceRequirements{
      native_requirements.bytes,
      native_requirements.alignment,
      0u,
  };
  return FERRULE_CUTLASS_SUCCESS;
}

extern "C" int32_t ferrule_cutlass_hybrid_mla_explicit_selection_can_implement(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_explicit_selection_args(*args);
  return static_cast<int32_t>(
      hybrid_mla_attention::explicit_selection_can_implement(&native_args));
}

extern "C" int32_t ferrule_cutlass_hybrid_mla_explicit_selection_launch(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_explicit_selection_args(*args);
  return static_cast<int32_t>(
      hybrid_mla_attention::explicit_selection_launch(&native_args));
}

#ifdef FERRULE_CUDA_TEST_ORACLE
extern "C" int32_t
ferrule_cutlass_test_hybrid_mla_explicit_selection_scalar_launch(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_explicit_selection_args(*args);
  return static_cast<int32_t>(
      hybrid_mla_attention::test_oracle::scalar_launch(&native_args));
}

extern "C" int32_t
ferrule_cutlass_test_hybrid_mla_explicit_selection_compare_launch(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args,
    uint64_t oracle_output_f32, uint64_t compare_result_i32) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_explicit_selection_args(*args);
  return static_cast<int32_t>(hybrid_mla_attention::test_oracle::compare_launch(
      &native_args, oracle_output_f32, compare_result_i32));
}
#endif

extern "C" int32_t ferrule_cutlass_proposal_head_can_implement(
    const FerruleCutlassProposalHeadArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_proposal_head_args(*args);
  return static_cast<int32_t>(proposal_head::validate(&native_args));
}

extern "C" int32_t ferrule_cutlass_proposal_head_launch(
    const FerruleCutlassProposalHeadArgs *args) {
  const int32_t status = ferrule_cutlass_proposal_head_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_proposal_head_args(*args);
  return static_cast<int32_t>(proposal_head::launch(&native_args));
}

extern "C" uint64_t ferrule_cutlass_grouped_fp4_moe_workspace_size(
    const FerruleCutlassGroupedFp4MoeArgs *args) {
  if (args == nullptr) {
    return 0;
  }
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
  grouped_fp4_moe::LaunchOptions options{};
  if (!grouped_fp4_moe_options(options)) {
    return 0;
  }
  const auto native_args = make_grouped_fp4_moe_args(*args);
  return static_cast<uint64_t>(
      grouped_fp4_moe::workspace_bytes(native_args, options));
#else
  return 0;
#endif
}

extern "C" int32_t ferrule_cutlass_grouped_fp4_moe_can_implement(
    const FerruleCutlassGroupedFp4MoeArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
  grouped_fp4_moe::LaunchOptions options{};
  if (!grouped_fp4_moe_options(options)) {
    return FERRULE_CUTLASS_UNSUPPORTED;
  }
  const auto native_args = make_grouped_fp4_moe_args(*args);
  void *workspace =
      reinterpret_cast<void *>(static_cast<uintptr_t>(args->workspace));
  return grouped_fp4_moe_status(grouped_fp4_moe::can_implement(
      &native_args, workspace, static_cast<size_t>(args->workspace_bytes),
      options));
#else
  return FERRULE_CUTLASS_UNSUPPORTED;
#endif
}

extern "C" int32_t ferrule_cutlass_grouped_fp4_moe_launch(
    const FerruleCutlassGroupedFp4MoeArgs *args) {
  const int32_t status = ferrule_cutlass_grouped_fp4_moe_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
  grouped_fp4_moe::LaunchOptions options{};
  if (!grouped_fp4_moe_options(options)) {
    return FERRULE_CUTLASS_UNSUPPORTED;
  }
  const auto native_args = make_grouped_fp4_moe_args(*args);
  void *workspace =
      reinterpret_cast<void *>(static_cast<uintptr_t>(args->workspace));
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return grouped_fp4_moe_status(grouped_fp4_moe::launch(
      &native_args, workspace, static_cast<size_t>(args->workspace_bytes),
      stream, options));
#else
  return FERRULE_CUTLASS_UNSUPPORTED;
#endif
}

extern "C" uint64_t ferrule_cutlass_mxfp4_sfb_storage_bytes(uint32_t n,
                                                            uint32_t k) {
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
  if (n > 0x7fffffffu || k > 0x7fffffffu) {
    return 0;
  }
  return static_cast<uint64_t>(grouped_fp4_moe::prepared_sfb_bytes(
      static_cast<int>(n), static_cast<int>(k)));
#else
  static_cast<void>(n);
  static_cast<void>(k);
  return 0;
#endif
}

extern "C" int32_t ferrule_cutlass_prepare_mxfp4_sfb(
    const FerruleCutlassPrepareMxfp4SfbArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  if (args->reserved0 != 0u) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
  if (args->n > 0x7fffffffu || args->k > 0x7fffffffu) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  auto *destination = reinterpret_cast<uint8_t *>(
      static_cast<uintptr_t>(args->prepared_destination));
  const auto *source = reinterpret_cast<const uint8_t *>(
      static_cast<uintptr_t>(args->linear_source));
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return helper_launch_status(grouped_fp4_moe::launch_prepare_sfb(
      destination, source, static_cast<int32_t>(args->n),
      static_cast<int32_t>(args->k), stream));
#else
  return FERRULE_CUTLASS_UNSUPPORTED;
#endif
}
