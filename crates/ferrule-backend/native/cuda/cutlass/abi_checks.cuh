#pragma once

#include "cutlass/abi.h"
#include "cutlass/target.cuh"
#include "cutlass/attention.cuh"
#include "cutlass/decoder_ops.cuh"
#include "cutlass/moe.cuh"
#include "cutlass/projections.cuh"

#include <cstddef>
#include <cutlass/version.h>

namespace {
namespace fp8_prefill = ferrule::cuda::cutlass::operators::fp8_projection;
namespace bf16_contract = ferrule::cuda::cutlass::operators::bf16_compressor;
namespace bf16_schedule = ferrule::cuda::cutlass::operators::bf16_compressor;
namespace main_project_norm =
    ferrule::cuda::cutlass::operators::main_project_norm;
namespace hybrid_full_block =
    ferrule::cuda::cutlass::operators::hybrid_mla_attention::full_block;
namespace hybrid_explicit = ferrule::cuda::cutlass::operators::
    hybrid_mla_attention::explicit_selection;
namespace proposal_head = ferrule::cuda::cutlass::operators::proposal_head;
namespace hc_producer =
    ferrule::cuda::cutlass::operators::hyper_connection_producer;
namespace mla_output = ferrule::cuda::cutlass::operators::mla_output;
namespace shared_ffn = ferrule::cuda::cutlass::operators::shared_ffn;
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
namespace grouped_fp4_moe =
    ferrule::cuda::cutlass::operators::grouped_fp4_moe;
#endif
} // namespace

static_assert(CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 6 && CUTLASS_PATCH == 1,
              "Ferrule's CUTLASS provider is pinned to CUTLASS 4.6.1");
static_assert(sizeof(FerruleCutlassProviderManifest) == 8,
              "Ferrule CUTLASS manifest ABI layout changed");
static_assert(sizeof(FerruleCutlassFp8QueryAKvArgs) == 96,
              "Ferrule CUTLASS FP8 QueryA+KV ABI layout changed");
static_assert(sizeof(FerruleCutlassBf16CompressorArgs) == 72,
              "Ferrule CUTLASS BF16 compressor ABI layout changed");
static_assert(sizeof(FerruleCutlassHcProducerArgs) == 168,
              "Ferrule CUTLASS HC producer ABI layout changed");
static_assert(sizeof(FerruleCutlassSharedFfnArgs) == 160,
              "Ferrule CUTLASS shared FFN ABI layout changed");
static_assert(sizeof(FerruleCutlassMlaOutputArgs) == 120,
              "Ferrule CUTLASS MLA output ABI layout changed");
static_assert(sizeof(FerruleCutlassMainProjectNormArgs) == 104,
              "Ferrule CUTLASS main-project/norm ABI layout changed");
static_assert(sizeof(FerruleCutlassHybridMlaAttentionArgs) == 176,
              "Ferrule CUTLASS hybrid-attention ABI layout changed");
static_assert(sizeof(FerruleCutlassHybridMlaExplicitSelectionArgs) == 224,
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
                              function_row_major_f32, 48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hc_scale_f32, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hc_base_f32, 64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs,
                              layer_rms_weight_f32, 72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, mix_f32, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, workspace, 88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, workspace_bytes,
                              96);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hidden_f32, 104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, normalized_f32,
                              112);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, packed_e4m3, 120);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, scales_ue8m0, 128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, split_pre_f32, 136);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, split_post_f32,
                              144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, split_comb_f32,
                              152);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, stream, 160);

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
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassMlaOutputArgs, latent_bf16, 80);
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
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              online_rescales_f32, 136);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs,
                              denominators_f32, 144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, output_f32,
                              152);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, status_i32,
                              160);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaAttentionArgs, stream,
                              168);

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
                              second_sequence_kv_lens_i32, 128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              row_sequence_ids_i32, 136);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              row_kv_lens_i32, 144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              row_second_kv_lens_i32, 152);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              selected_indices_i32, 160);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              selectors_i32, 168);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              attention_sink_f32, 176);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              workspace, 184);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              workspace_bytes, 192);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              output_f32, 200);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              status_i32, 208);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHybridMlaExplicitSelectionArgs,
                              stream, 216);

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
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, input_fp8, 160);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassGroupedFp4MoeArgs, input_ue8m0,
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
static_assert(sizeof(bf16_contract::Args) ==
              sizeof(FerruleCutlassBf16CompressorArgs));
static_assert(sizeof(hc_producer::HcPreRmsNormFp8Args) ==
              sizeof(FerruleCutlassHcProducerArgs));
static_assert(sizeof(shared_ffn::Args) == 152);
static_assert(sizeof(mla_output::Args) == sizeof(FerruleCutlassMlaOutputArgs));
static_assert(sizeof(main_project_norm::Args) ==
              sizeof(FerruleCutlassMainProjectNormArgs));
static_assert(sizeof(hybrid_full_block::Args) ==
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
              offsetof(bf16_contract::Args, m));
static_assert(offsetof(FerruleCutlassBf16CompressorArgs, reserved0) ==
              offsetof(bf16_contract::Args, reserved));
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_contract::Args, n1);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_contract::Args, n2);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_contract::Args, k);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_contract::Args, activation_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_contract::Args, projection1_weight_bf16);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_contract::Args, projection2_weight_bf16);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_contract::Args, projection1_output_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_contract::Args, projection2_output_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassBf16CompressorArgs,
                                   bf16_contract::Args, stream);

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
                                   mla_output::Args, latent_bf16);
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
                                   input_fp8);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassGroupedFp4MoeArgs,
                                   grouped_fp4_moe::GroupedFp4MoeArgs,
                                   input_ue8m0);
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

} // namespace
