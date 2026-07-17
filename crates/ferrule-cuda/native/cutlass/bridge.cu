#include "ferrule_cutlass.h"
#include "sm121_bf16_compressor_prefill.cuh"
#include "sm121_dspark_main_project_norm.cuh"
#include "sm121_dspark_hybrid_attention.cuh"
#include "sm121_dspark_proposal_head.cuh"
#include "sm121_fp4_moe.cuh"},{
#include "sm121_fp8_query_kv_prefill.cuh"
#include "sm121_hc_producer.cuh"
#include "sm121_mla_output.cuh"
#include "sm121_shared_ffn.cuh"

#include <cuda_runtime_api.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cutlass/version.h>

#if FERRULE_CUTLASS_TARGET_SM != 121
#error "Ferrule CUTLASS is a GB10-only provider and must target SM121a"
#endif

static_assert(FERRULE_CUTLASS_TARGET_SM == 121,
              "Ferrule CUTLASS target must be compute capability 121");
static_assert(CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 6 && CUTLASS_PATCH == 1,
              "Ferrule's CUTLASS provider is pinned to CUTLASS 4.6.1");
static_assert(sizeof(FerruleCutlassProviderManifest) == 24,
              "Ferrule CUTLASS manifest ABI layout changed");
static_assert(sizeof(FerruleCutlassFp8QueryAKvArgs) == 96,
              "Ferrule CUTLASS FP8 QueryA+KV ABI layout changed");
static_assert(sizeof(FerruleCutlassBf16CompressorArgs) == 72,
              "Ferrule CUTLASS BF16 compressor ABI layout changed");
static_assert(sizeof(FerruleCutlassHcProducerArgs) == 144,
              "Ferrule CUTLASS HC producer ABI layout changed");
static_assert(sizeof(FerruleCutlassSharedFfnArgs) == 168,
              "Ferrule CUTLASS shared FFN ABI layout changed");
static_assert(sizeof(FerruleCutlassMlaOutputArgs) == 120,
              "Ferrule CUTLASS MLA output ABI layout changed");
static_assert(sizeof(FerruleCutlassDsparkMainProjectNormArgs) == 104,
              "Ferrule CUTLASS DSpark main-project/norm ABI layout changed");
static_assert(sizeof(FerruleCutlassDsparkHybridMlaAttentionArgs) == 160,
              "Ferrule CUTLASS DSpark hybrid-attention ABI layout changed");
static_assert(sizeof(FerruleCutlassDsparkProposalHeadArgs) == 184,
              "Ferrule CUTLASS DSpark proposal-head ABI layout changed");
static_assert(sizeof(FerruleCutlassStableFrameFp4MoeArgs) == 224,
              "Ferrule CUTLASS stable-frame FP4 MoE ABI layout changed");

#define FERRULE_CUTLASS_ASSERT_OFFSET(type, field, expected)                  \
  static_assert(offsetof(type, field) == expected,                            \
                #type "." #field " ABI offset changed")

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, abi_version, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, rows, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hc, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hidden, 12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, mix, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, sinkhorn_iters, 20);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hc_eps, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, hc_norm_eps, 28);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, layer_rms_eps, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassHcProducerArgs, reserved, 36);
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

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, abi_version, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, input_fp8, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, input_ue8m0, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, gate_weight_fp8, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, gate_weight_ue8m0,
                              32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, up_weight_fp8, 40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, up_weight_ue8m0, 48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, down_weight_fp8, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, down_weight_ue8m0,
                              64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, hidden_f32, 72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, hidden_fp8, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, hidden_ue8m0, 88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, output_f32, 96);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, rows, 104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, input_size, 108);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, intermediate_size,
                              112);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, output_size, 116);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, gate_block_m, 120);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, gate_block_k, 124);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, up_block_m, 128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, up_block_k, 132);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, down_block_m, 136);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, down_block_k, 140);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, output_scale, 144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, swiglu_limit, 148);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, flags, 152);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassSharedFfnArgs, stream, 160);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkMainProjectNormArgs,
                              abi_version, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkMainProjectNormArgs, rows, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkMainProjectNormArgs,
                              input_size, 8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkMainProjectNormArgs,
                              output_size, 12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkMainProjectNormArgs,
                              rms_eps, 24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkMainProjectNormArgs,
                              input_f32, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkMainProjectNormArgs,
                              output_f32, 88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkMainProjectNormArgs, stream,
                              96);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              abi_version, 0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              block_rows, 4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              sequence_tokens, 16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              layer_index, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              softmax_scale, 48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              context_plane_elements, 56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              query_f32, 64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              query_bf16, 104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              gathered_kv_bf16, 112);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              probabilities_bf16, 128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              status_i32, 144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkHybridMlaAttentionArgs,
                              stream, 152);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkProposalHeadArgs, abi_version,
                              0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkProposalHeadArgs, hc_eps, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkProposalHeadArgs, hc_state_f32,
                              40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassDsparkProposalHeadArgs, stream, 176);

FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, abi_version,
                              0);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, reserved0,
                              4);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, input_size,
                              8);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              intermediate_size, 12);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, hidden_size,
                              16);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, num_tokens,
                              20);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, num_routes,
                              24);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              slot_capacity, 28);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              num_segments, 32);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              swiglu_limit, 36);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, x_packed,
                              40);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, x_scales,
                              48);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, gate_ptrs,
                              56);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              gate_scale_ptrs, 64);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, up_ptrs,
                              72);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              up_scale_ptrs, 80);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, down_ptrs,
                              88);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              down_scale_ptrs, 96);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              slot_generations, 104);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              segment_expert_slots, 112);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              segment_generations, 120);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              segment_token_indices, 128);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              segment_route_indices, 136);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              segment_route_weights, 144);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              segment_states, 152);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              segment_bindings, 160);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, hidden_f32,
                              168);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              hidden_packed, 176);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              hidden_scales, 184);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                              route_written, 192);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, route_error,
                              200);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, route_output,
                              208);
FERRULE_CUTLASS_ASSERT_OFFSET(FerruleCutlassStableFrameFp4MoeArgs, stream, 216);

#undef FERRULE_CUTLASS_ASSERT_OFFSET

namespace {

namespace fp8_prefill = ferrule::sm121_fp8_query_kv_prefill;
namespace bf16_prefill = ferrule::cutlass::sm121_bf16_compressor_prefill;
namespace dspark_main = ferrule::cutlass::sm121_dspark_main_project_norm;
namespace dspark_attention = ferrule::cutlass::sm121_dspark_hybrid_attention;
namespace dspark_head = ferrule::cutlass::sm121_dspark_proposal_head;
namespace hc_producer = ferrule::sm121;
namespace mla_output = ferrule::cutlass::sm121_mla_output;
namespace shared_ffn = ferrule::cutlass::sm121_shared_ffn;
namespace fp4_moe = ferrule::cutlass::sm121_fp4_moe;

static_assert(sizeof(hc_producer::HcPreRmsNormFp8Args) ==
              sizeof(FerruleCutlassHcProducerArgs));
static_assert(sizeof(shared_ffn::Args) == 152);
static_assert(sizeof(mla_output::Args) ==
              sizeof(FerruleCutlassMlaOutputArgs));
static_assert(sizeof(dspark_main::Args) ==
              sizeof(FerruleCutlassDsparkMainProjectNormArgs));
static_assert(sizeof(dspark_attention::Args) ==
              sizeof(FerruleCutlassDsparkHybridMlaAttentionArgs));
static_assert(sizeof(dspark_head::Args) ==
              sizeof(FerruleCutlassDsparkProposalHeadArgs));
static_assert(sizeof(fp4_moe::Args) ==
              sizeof(FerruleCutlassStableFrameFp4MoeArgs));

#define FERRULE_CUTLASS_ASSERT_SAME_OFFSET(c_type, native_type, field)        \
  static_assert(offsetof(c_type, field) == offsetof(native_type, field),      \
                #c_type "." #field " no longer matches native Args")
#define FERRULE_CUTLASS_ASSERT_WRAPPED_OFFSET(c_type, native_type, field)     \
  static_assert(offsetof(c_type, field) == offsetof(native_type, field) + 8,  \
                #c_type "." #field " no longer wraps native Args")

FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassHcProducerArgs,
                                   hc_producer::HcPreRmsNormFp8Args,
                                   abi_version);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassHcProducerArgs,
                                   hc_producer::HcPreRmsNormFp8Args, rows);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassHcProducerArgs,
                                   hc_producer::HcPreRmsNormFp8Args, state_f32);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassHcProducerArgs,
                                   hc_producer::HcPreRmsNormFp8Args, stream);
FERRULE_CUTLASS_ASSERT_WRAPPED_OFFSET(FerruleCutlassSharedFfnArgs,
                                      shared_ffn::Args, input_fp8);
FERRULE_CUTLASS_ASSERT_WRAPPED_OFFSET(FerruleCutlassSharedFfnArgs,
                                      shared_ffn::Args, hidden_f32);
FERRULE_CUTLASS_ASSERT_WRAPPED_OFFSET(FerruleCutlassSharedFfnArgs,
                                      shared_ffn::Args, hidden_fp8);
FERRULE_CUTLASS_ASSERT_WRAPPED_OFFSET(FerruleCutlassSharedFfnArgs,
                                      shared_ffn::Args, output_f32);
FERRULE_CUTLASS_ASSERT_WRAPPED_OFFSET(FerruleCutlassSharedFfnArgs,
                                      shared_ffn::Args, rows);
FERRULE_CUTLASS_ASSERT_WRAPPED_OFFSET(FerruleCutlassSharedFfnArgs,
                                      shared_ffn::Args, flags);
static_assert(offsetof(fp4_moe::Args, rows) == 4,
              "SM121 FP4 MoE internal segment width offset changed");
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                                   fp4_moe::Args, x_packed);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                                   fp4_moe::Args, segment_expert_slots);
FERRULE_CUTLASS_ASSERT_SAME_OFFSET(FerruleCutlassStableFrameFp4MoeArgs,
                                   fp4_moe::Args, stream);

#undef FERRULE_CUTLASS_ASSERT_WRAPPED_OFFSET
#undef FERRULE_CUTLASS_ASSERT_SAME_OFFSET

using Sm121Fp8Mma =
    cute::SM120_16x8x32_TN<cute::float_e4m3_t, cute::float_e4m3_t,
                           float>;
using Sm121Bf16Mma = cute::SM80_16x8x16_F32BF16BF16F32_TN;

struct alignas(16) Fp8QueryAKvSharedStorage {
  alignas(16) uint8_t activation[256];
  alignas(16) uint8_t query_a_weight[512];
  alignas(16) uint8_t kv_weight[512];
  alignas(16) uint8_t activation_scales[8];
};

struct alignas(16) Bf16CompressorSharedStorage {
  alignas(16) uint16_t activation[128];
  alignas(16) uint16_t projection1_weight[256];
  alignas(16) uint16_t projection2_weight[256];
};

__device__ __forceinline__ float ue8m0_to_float(uint8_t value) {
  // Match Ferrule's artifact semantics for the UE8M0 zero encoding.
  uint32_t bits = value == 0 ? (1u << 22) : (static_cast<uint32_t>(value) << 23);
  return __uint_as_float(bits);
}

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
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1],
                                fragment[2], fragment[3]);
}

__device__ __forceinline__ void
load_b_fragment_16_byte_rows(const uint8_t *shared, uint32_t lane,
                             uint32_t (&fragment)[2]) {
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void
mma_fp8_e4m3(float (&accumulator)[4], const uint32_t (&a)[4],
             const uint32_t (&b)[2]) {
  Sm121Fp8Mma::fma(accumulator[0], accumulator[1], accumulator[2],
                   accumulator[3], a[0], a[1], a[2], a[3], b[0], b[1],
                   accumulator[0], accumulator[1], accumulator[2],
                   accumulator[3]);
}

__device__ __forceinline__ void
mma_bf16(float (&accumulator)[4], const uint32_t (&a)[4],
         const uint32_t (&b)[2]) {
  Sm121Bf16Mma::fma(accumulator[0], accumulator[1], accumulator[2],
                    accumulator[3], a[0], a[1], a[2], a[3], b[0], b[1],
                    accumulator[0], accumulator[1], accumulator[2],
                    accumulator[3]);
}

__global__ void
ferrule_fp8_query_a_kv_sm121(FerruleCutlassFp8QueryAKvArgs args) {
  __shared__ Fp8QueryAKvSharedStorage shared;

  uint32_t lane = threadIdx.x;
  uint32_t channel_base = blockIdx.x * 16u;
  auto *activation = reinterpret_cast<const uint8_t *>(
      static_cast<uintptr_t>(args.activation_fp8));
  auto *activation_scales = reinterpret_cast<const uint8_t *>(
      static_cast<uintptr_t>(args.activation_ue8m0));
  auto *query_a_weight = reinterpret_cast<const uint8_t *>(
      static_cast<uintptr_t>(args.query_a_weight_fp8));
  auto *query_a_weight_scales = reinterpret_cast<const uint8_t *>(
      static_cast<uintptr_t>(args.query_a_weight_ue8m0));
  auto *kv_weight = reinterpret_cast<const uint8_t *>(
      static_cast<uintptr_t>(args.kv_weight_fp8));
  auto *kv_weight_scales = reinterpret_cast<const uint8_t *>(
      static_cast<uintptr_t>(args.kv_weight_ue8m0));
  auto *query_a_output = reinterpret_cast<float *>(
      static_cast<uintptr_t>(args.query_a_output_f32));
  auto *kv_output = reinterpret_cast<float *>(
      static_cast<uintptr_t>(args.kv_output_f32));

  float query_a_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float kv_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t scale_block = 0; scale_block < args.scale_cols;
       ++scale_block) {
    if (lane < args.rows) {
      shared.activation_scales[lane] =
          activation_scales[static_cast<uint64_t>(lane) * args.scale_cols +
                            scale_block];
    }
    __syncthreads();

    float query_a_block_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float kv_block_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t k_block = scale_block * 128u;

#pragma unroll
    for (uint32_t k_sub = 0; k_sub < 128u; k_sub += 32u) {
      if (lane < 16u) {
        const uint32_t channel = channel_base + lane;
        auto *query_a_destination =
            reinterpret_cast<uint4 *>(shared.query_a_weight + lane * 32u);
        auto *kv_destination =
            reinterpret_cast<uint4 *>(shared.kv_weight + lane * 32u);
        const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
        if (channel < args.n1) {
          auto *source = reinterpret_cast<const uint4 *>(
              query_a_weight + static_cast<uint64_t>(channel) * args.k +
              k_block + k_sub);
          query_a_destination[0] = source[0];
          query_a_destination[1] = source[1];
        } else {
          query_a_destination[0] = zero;
          query_a_destination[1] = zero;
        }
        if (channel < args.n2) {
          auto *source = reinterpret_cast<const uint4 *>(
              kv_weight + static_cast<uint64_t>(channel) * args.k + k_block +
              k_sub);
          kv_destination[0] = source[0];
          kv_destination[1] = source[1];
        } else {
          kv_destination[0] = zero;
          kv_destination[1] = zero;
        }
      }

      for (uint32_t linear = lane; linear < 256u; linear += 32u) {
        uint32_t k_pair = linear >> 4;
        uint32_t pair_byte = linear & 15u;
        uint32_t row = pair_byte >> 1;
        uint32_t byte = pair_byte & 1u;
        uint64_t k_index = static_cast<uint64_t>(k_block) + k_sub +
                           k_pair * 2u + byte;
        shared.activation[linear] =
            row < args.rows
                ? activation[static_cast<uint64_t>(row) * args.k + k_index]
                : 0;
      }
      __syncthreads();

      uint32_t query_a_fragment[4];
      uint32_t kv_fragment[4];
      uint32_t activation_fragment[2];
      load_a_fragment_16x32_bytes(shared.query_a_weight, lane,
                                  query_a_fragment);
      load_a_fragment_16x32_bytes(shared.kv_weight, lane, kv_fragment);
      load_b_fragment_16_byte_rows(shared.activation, lane,
                                   activation_fragment);
      mma_fp8_e4m3(query_a_block_accumulator, query_a_fragment,
                   activation_fragment);
      mma_fp8_e4m3(kv_block_accumulator, kv_fragment,
                   activation_fragment);
      __syncthreads();
    }

    float query_a_weight_scale =
        channel_base < args.n1
            ? ue8m0_to_float(query_a_weight_scales[
                  static_cast<uint64_t>(channel_base / 128u) *
                      args.scale_cols +
                  scale_block])
            : 0.0f;
    float kv_weight_scale =
        channel_base < args.n2
            ? ue8m0_to_float(kv_weight_scales[
                  static_cast<uint64_t>(channel_base / 128u) *
                      args.scale_cols +
                  scale_block])
            : 0.0f;
    uint32_t row_pair = lane & 3u;

#pragma unroll
    for (uint32_t element = 0; element < 4u; ++element) {
      uint32_t row = row_pair * 2u + (element & 1u);
      if (row < args.rows) {
        float activation_scale =
            ue8m0_to_float(shared.activation_scales[row]);
        query_a_accumulator[element] += query_a_block_accumulator[element] *
                                        query_a_weight_scale *
                                        activation_scale;
        kv_accumulator[element] += kv_block_accumulator[element] *
                                   kv_weight_scale * activation_scale;
      }
    }
    __syncthreads();
  }

  uint32_t channel_group = lane >> 2;
  uint32_t row_pair = lane & 3u;
#pragma unroll
  for (uint32_t element = 0; element < 4u; ++element) {
    uint32_t channel = channel_base + channel_group +
                       (element >= 2u ? 8u : 0u);
    uint32_t row = row_pair * 2u + (element & 1u);
    if (row < args.rows && channel < args.n1) {
      query_a_output[static_cast<uint64_t>(row) * args.n1 + channel] =
          query_a_accumulator[element];
    }
    if (row < args.rows && channel < args.n2) {
      kv_output[static_cast<uint64_t>(row) * args.n2 + channel] =
          kv_accumulator[element];
    }
  }
}

__global__ void
ferrule_bf16_compressor_sm121(FerruleCutlassBf16CompressorArgs args) {
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
      auto *projection1_destination = reinterpret_cast<uint4 *>(
          shared.projection1_weight + lane * 16u);
      auto *projection2_destination = reinterpret_cast<uint4 *>(
          shared.projection2_weight + lane * 16u);
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
      float value =
          row < args.rows
              ? activation[static_cast<uint64_t>(row) * args.k + k_base +
                           k_local]
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
    uint32_t channel = channel_base + channel_group +
                       (element >= 2u ? 8u : 0u);
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

// Row scheduling is provider-private. The semantic ABI accepts the complete
// grid-backed M range; only the launch implementation decides whether the
// latency-tuned single-tile schedule is preferable.
inline constexpr uint32_t kTinyLinearRows = 8u;

fp8_prefill::Args make_prefill_args(
    const FerruleCutlassFp8QueryAKvArgs &args) {
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

bf16_prefill::Args make_prefill_args(
    const FerruleCutlassBf16CompressorArgs &args) {
  return bf16_prefill::Args{
      bf16_prefill::kArgsVersion,
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

hc_producer::HcPreRmsNormFp8Args make_hc_producer_args(
    const FerruleCutlassHcProducerArgs &args) {
  return hc_producer::HcPreRmsNormFp8Args{
      hc_producer::kHcPreRmsNormFp8AbiVersion,
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

shared_ffn::Args make_shared_ffn_args(
    const FerruleCutlassSharedFfnArgs &args) {
  return shared_ffn::Args{
      reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(args.input_fp8)),
      reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(args.input_ue8m0)),
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

mla_output::Args make_mla_output_args(
    const FerruleCutlassMlaOutputArgs &args) {
  return mla_output::Args{
      mla_output::kArgsVersion,
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

dspark_main::Args make_dspark_main_args(
    const FerruleCutlassDsparkMainProjectNormArgs &args) {
  return dspark_main::Args{
      dspark_main::kArgsVersion,
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

dspark_attention::Args make_dspark_attention_args(
    const FerruleCutlassDsparkHybridMlaAttentionArgs &args) {
  return dspark_attention::Args{
      dspark_attention::kArgsVersion,
      args.block_rows,
      args.heads,
      args.head_dim,
      args.sequence_tokens,
      args.window_size,
      args.page_tokens,
      args.elements_per_token,
      args.layer_index,
      args.layer_count,
      args.block_slot_offset,
      args.block_slot_count,
      args.softmax_scale,
      args.reserved0,
      args.context_plane_elements,
      args.query_f32,
      args.context_plane_f32,
      args.block_kv_f32,
      args.block_slots_i32,
      args.attention_sink_f32,
      args.query_bf16,
      args.gathered_kv_bf16,
      args.scores_f32,
      args.probabilities_bf16,
      args.output_f32,
      args.status_i32,
      args.stream,
  };
}

dspark_head::Args make_dspark_head_args(
    const FerruleCutlassDsparkProposalHeadArgs &args) {
  return dspark_head::Args{
      dspark_head::kArgsVersion,
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

fp4_moe::Args make_fp4_moe_args(
    const FerruleCutlassStableFrameFp4MoeArgs &args) {
  return fp4_moe::Args{
      fp4_moe::kArgsVersion,
      8u,
      args.input_size,
      args.intermediate_size,
      args.hidden_size,
      args.num_tokens,
      args.num_routes,
      args.slot_capacity,
      args.num_segments,
      args.swiglu_limit,
      args.x_packed,
      args.x_scales,
      args.gate_ptrs,
      args.gate_scale_ptrs,
      args.up_ptrs,
      args.up_scale_ptrs,
      args.down_ptrs,
      args.down_scale_ptrs,
      args.slot_generations,
      args.segment_expert_slots,
      args.segment_generations,
      args.segment_token_indices,
      args.segment_route_indices,
      args.segment_route_weights,
      args.segment_states,
      args.segment_bindings,
      args.hidden_f32,
      args.hidden_packed,
      args.hidden_scales,
      args.route_written,
      args.route_error,
      args.route_output,
      args.stream,
  };
}

int32_t helper_launch_status(cudaError_t status) {
  if (status == cudaSuccess) {
    return FERRULE_CUTLASS_SUCCESS;
  }
  return status == cudaErrorInvalidValue ? FERRULE_CUTLASS_INVALID_ARGUMENT
                                         : FERRULE_CUTLASS_LAUNCH_FAILED;
}

int32_t fp4_moe_launch_status(fp4_moe::Status status) {
  switch (status) {
  case fp4_moe::Status::kSuccess:
    return FERRULE_CUTLASS_SUCCESS;
  case fp4_moe::Status::kInvalidAbi:
    return FERRULE_CUTLASS_INVALID_ABI;
  case fp4_moe::Status::kInvalidArgument:
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  case fp4_moe::Status::kUnsupportedResources:
  case fp4_moe::Status::kLaunchFailed:
    return FERRULE_CUTLASS_LAUNCH_FAILED;
  }
  return FERRULE_CUTLASS_LAUNCH_FAILED;
}

} // namespace

extern "C" FerruleCutlassProviderManifest
ferrule_cutlass_provider_manifest(void) {
  return FerruleCutlassProviderManifest{
      FERRULE_CUTLASS_ABI_VERSION,
      CUTLASS_VERSION,
      FERRULE_CUTLASS_TARGET_SM,
      9u,
      FERRULE_CUTLASS_KERNEL_BIT(
          FERRULE_CUTLASS_KERNEL_FP8_QUERY_A_KV_SM121) |
          FERRULE_CUTLASS_KERNEL_BIT(
              FERRULE_CUTLASS_KERNEL_BF16_COMPRESSOR_SM121) |
          FERRULE_CUTLASS_KERNEL_BIT(
              FERRULE_CUTLASS_KERNEL_HC_PRODUCER_SM121) |
          FERRULE_CUTLASS_KERNEL_BIT(
              FERRULE_CUTLASS_KERNEL_SHARED_FFN_SM121) |
          FERRULE_CUTLASS_KERNEL_BIT(
              FERRULE_CUTLASS_KERNEL_STABLE_FRAME_FP4_MOE_SM121) |
          FERRULE_CUTLASS_KERNEL_BIT(
              FERRULE_CUTLASS_KERNEL_MLA_OUTPUT_SM121) |
          FERRULE_CUTLASS_KERNEL_BIT(
              FERRULE_CUTLASS_KERNEL_DSPARK_MAIN_PROJECT_NORM_SM121) |
          FERRULE_CUTLASS_KERNEL_BIT(
              FERRULE_CUTLASS_KERNEL_DSPARK_HYBRID_MLA_ATTENTION_SM121) |
          FERRULE_CUTLASS_KERNEL_BIT(
              FERRULE_CUTLASS_KERNEL_DSPARK_PROPOSAL_HEAD_SM121),
  };
}

extern "C" int32_t ferrule_cutlass_fp8_query_a_kv_can_implement(
    const FerruleCutlassFp8QueryAKvArgs *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
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
  if (args->rows > kTinyLinearRows) {
    return helper_launch_status(
        fp8_prefill::launch(make_prefill_args(*args), stream));
  }
  uint32_t max_n = args->n1 > args->n2 ? args->n1 : args->n2;
  uint32_t blocks = static_cast<uint32_t>(
      (static_cast<uint64_t>(max_n) + 15u) / 16u);
  ferrule_fp8_query_a_kv_sm121<<<blocks, 32, 0, stream>>>(*args);
  return cudaGetLastError() == cudaSuccess ? FERRULE_CUTLASS_SUCCESS
                                           : FERRULE_CUTLASS_LAUNCH_FAILED;
}

extern "C" int32_t ferrule_cutlass_bf16_compressor_can_implement(
    const FerruleCutlassBf16CompressorArgs *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
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
  uint32_t blocks = static_cast<uint32_t>(
      (static_cast<uint64_t>(max_n) + 15u) / 16u);
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  ferrule_bf16_compressor_sm121<<<blocks, 32, 0, stream>>>(*args);
  return cudaGetLastError() == cudaSuccess ? FERRULE_CUTLASS_SUCCESS
                                           : FERRULE_CUTLASS_LAUNCH_FAILED;
}

extern "C" int32_t ferrule_cutlass_hc_producer_can_implement(
    const FerruleCutlassHcProducerArgs *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
  }
  const auto native_args = make_hc_producer_args(*args);
  return hc_producer::validate_hc_pre_rmsnorm_fp8(native_args)
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
}

extern "C" int32_t ferrule_cutlass_hc_producer_launch(
    const FerruleCutlassHcProducerArgs *args) {
  const int32_t status = ferrule_cutlass_hc_producer_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  return helper_launch_status(hc_producer::launch_hc_pre_rmsnorm_fp8(
      make_hc_producer_args(*args)));
}

extern "C" int32_t ferrule_cutlass_shared_ffn_can_implement(
    const FerruleCutlassSharedFfnArgs *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
  }
  return shared_ffn::validate(make_shared_ffn_args(*args)) ==
                 shared_ffn::ValidationResult::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
}

extern "C" int32_t ferrule_cutlass_shared_ffn_launch(
    const FerruleCutlassSharedFfnArgs *args) {
  const int32_t status = ferrule_cutlass_shared_ffn_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return helper_launch_status(shared_ffn::launch(make_shared_ffn_args(*args),
                                                  stream));
}

extern "C" int32_t ferrule_cutlass_mla_output_can_implement(
    const FerruleCutlassMlaOutputArgs *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
  }
  const auto native_args = make_mla_output_args(*args);
  return static_cast<int32_t>(mla_output::validate(&native_args));
}

extern "C" int32_t ferrule_cutlass_mla_output_launch(
    const FerruleCutlassMlaOutputArgs *args) {
  const int32_t status = ferrule_cutlass_mla_output_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_mla_output_args(*args);
  return static_cast<int32_t>(mla_output::launch(&native_args));
}

extern "C" int32_t ferrule_cutlass_dspark_main_project_norm_can_implement(
    const FerruleCutlassDsparkMainProjectNormArgs *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
  }
  const auto native_args = make_dspark_main_args(*args);
  return static_cast<int32_t>(dspark_main::validate(&native_args));
}

extern "C" int32_t ferrule_cutlass_dspark_main_project_norm_launch(
    const FerruleCutlassDsparkMainProjectNormArgs *args) {
  const int32_t status =
      ferrule_cutlass_dspark_main_project_norm_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_dspark_main_args(*args);
  return static_cast<int32_t>(dspark_main::launch(&native_args));
}

extern "C" int32_t ferrule_cutlass_dspark_hybrid_mla_attention_can_implement(
    const FerruleCutlassDsparkHybridMlaAttentionArgs *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
  }
  const auto native_args = make_dspark_attention_args(*args);
  return static_cast<int32_t>(dspark_attention::validate(&native_args));
}

extern "C" int32_t ferrule_cutlass_dspark_hybrid_mla_attention_launch(
    const FerruleCutlassDsparkHybridMlaAttentionArgs *args) {
  const int32_t status =
      ferrule_cutlass_dspark_hybrid_mla_attention_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_dspark_attention_args(*args);
  return static_cast<int32_t>(dspark_attention::launch(&native_args));
}

extern "C" int32_t ferrule_cutlass_dspark_proposal_head_can_implement(
    const FerruleCutlassDsparkProposalHeadArgs *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
  }
  const auto native_args = make_dspark_head_args(*args);
  return static_cast<int32_t>(dspark_head::validate(&native_args));
}

extern "C" int32_t ferrule_cutlass_dspark_proposal_head_launch(
    const FerruleCutlassDsparkProposalHeadArgs *args) {
  const int32_t status = ferrule_cutlass_dspark_proposal_head_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_dspark_head_args(*args);
  return static_cast<int32_t>(dspark_head::launch(&native_args));
}

extern "C" int32_t ferrule_cutlass_stable_frame_fp4_moe_can_implement(
    const FerruleCutlassStableFrameFp4MoeArgs *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
  }
  if (args->reserved0 != 0u) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_fp4_moe_args(*args);
  return fp4_moe::validate(&native_args) == fp4_moe::Status::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
}

extern "C" int32_t ferrule_cutlass_stable_frame_fp4_moe_launch(
    const FerruleCutlassStableFrameFp4MoeArgs *args) {
  const int32_t status =
      ferrule_cutlass_stable_frame_fp4_moe_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_fp4_moe_args(*args);
  return fp4_moe_launch_status(fp4_moe::launch(&native_args));
}
