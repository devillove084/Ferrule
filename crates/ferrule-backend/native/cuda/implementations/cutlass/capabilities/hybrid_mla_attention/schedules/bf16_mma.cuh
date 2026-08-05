#ifndef FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_SCHEDULES_BF16_MMA_CUH_
#define FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_SCHEDULES_BF16_MMA_CUH_

#include "../../hybrid_mla_attention.cuh"
#include "../explicit_selection_contract.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::
    schedules::bf16_mma {

inline constexpr std::uint32_t kContiguous = kExplicitSelectionContiguous;
inline constexpr std::uint32_t kPaged = kExplicitSelectionPaged;
inline constexpr std::uint32_t kDualPaged = kExplicitSelectionDualPaged;
inline constexpr std::uint32_t kHeads = kExplicitSelectionHeads;
inline constexpr std::uint32_t kHeadDim = kExplicitSelectionHeadDim;
inline constexpr std::uint32_t kMaximumSelectedWidth =
    kExplicitSelectionMaximumWidth;
inline constexpr std::uint32_t kOnlineSoftmaxTile = 64u;
inline constexpr std::uint32_t kMmaRows = 8u;
inline constexpr std::uint32_t kMmaColumns = 16u;
inline constexpr std::uint32_t kKTile = 16u;
inline constexpr std::uint32_t kWarpSize = 32u;
inline constexpr std::uint32_t kWarps = 4u;
inline constexpr std::uint32_t kMmaThreads = kWarpSize * kWarps;
inline constexpr std::uint32_t kGatherThreads = 256u;
inline constexpr std::uint32_t kHeadsPerWarp = kMmaRows;
inline constexpr std::uint32_t kHeadsPerBlock = kWarps * kHeadsPerWarp;
inline constexpr std::uint32_t kHeadGroups = kHeads / kHeadsPerBlock;
inline constexpr std::uint32_t kChannelTiles = kHeadDim / kMmaColumns;
inline constexpr std::uint32_t kLaunchCount = 4u;

static_assert(kHeads % kHeadsPerBlock == 0u);
static_assert(kHeadDim % kMmaColumns == 0u);
static_assert(kHeadDim % kKTile == 0u);
static_assert(sizeof(std::uintptr_t) == sizeof(std::uint64_t));

// Private schedule ABI. Input tensors, latent planes, and output remain F32;
// the intermediate addresses are supplied from
// explicit_selection_workspace::Binding by the caller.
struct Args {
  std::uint32_t kind;
  std::uint32_t rows;
  std::uint32_t tokens_per_sequence;
  std::uint32_t kv_len;
  std::uint32_t heads;
  std::uint32_t head_dim;
  std::uint32_t selected_width;
  std::uint32_t page_tokens;
  std::uint32_t first_elements_per_token;
  std::uint32_t second_elements_per_token;
  std::uint32_t layer_index;
  std::uint32_t layer_count;
  std::uint32_t flags;
  float softmax_scale;
  std::uint32_t reserved0;
  std::uint64_t first_plane_elements;
  std::uint64_t second_plane_elements;

  std::uint64_t query_f32;
  std::uint64_t first_plane_f32;
  std::uint64_t second_plane_f32;
  std::uint64_t block_slots_i32;
  std::uint64_t block_offsets_i32;
  std::uint64_t sequence_kv_lens_i32;
  std::uint64_t row_sequence_ids_i32;
  std::uint64_t row_kv_lens_i32;
  std::uint64_t selected_indices_i32;
  std::uint64_t selectors_i32;
  std::uint64_t attention_sink_f32;
  std::uint64_t source_addresses_u64;
  std::uint64_t query_bf16;
  std::uint64_t gathered_kv_bf16;
  std::uint64_t raw_scores_f32;
  std::uint64_t probabilities_bf16;
  std::uint64_t online_rescales_f32;
  std::uint64_t denominators_f32;
  std::uint64_t output_f32;
  std::uint64_t status_i32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 248u, "BF16 selected-attention POD changed");
static_assert(alignof(Args) == 8u);
static_assert(offsetof(Args, kind) == 0u);
static_assert(offsetof(Args, rows) == 4u);
static_assert(offsetof(Args, tokens_per_sequence) == 8u);
static_assert(offsetof(Args, kv_len) == 12u);
static_assert(offsetof(Args, heads) == 16u);
static_assert(offsetof(Args, head_dim) == 20u);
static_assert(offsetof(Args, selected_width) == 24u);
static_assert(offsetof(Args, page_tokens) == 28u);
static_assert(offsetof(Args, first_elements_per_token) == 32u);
static_assert(offsetof(Args, second_elements_per_token) == 36u);
static_assert(offsetof(Args, layer_index) == 40u);
static_assert(offsetof(Args, layer_count) == 44u);
static_assert(offsetof(Args, flags) == 48u);
static_assert(offsetof(Args, softmax_scale) == 52u);
static_assert(offsetof(Args, reserved0) == 56u);
static_assert(offsetof(Args, first_plane_elements) == 64u);
static_assert(offsetof(Args, second_plane_elements) == 72u);
static_assert(offsetof(Args, query_f32) == 80u);
static_assert(offsetof(Args, first_plane_f32) == 88u);
static_assert(offsetof(Args, second_plane_f32) == 96u);
static_assert(offsetof(Args, block_slots_i32) == 104u);
static_assert(offsetof(Args, block_offsets_i32) == 112u);
static_assert(offsetof(Args, sequence_kv_lens_i32) == 120u);
static_assert(offsetof(Args, row_sequence_ids_i32) == 128u);
static_assert(offsetof(Args, row_kv_lens_i32) == 136u);
static_assert(offsetof(Args, selected_indices_i32) == 144u);
static_assert(offsetof(Args, selectors_i32) == 152u);
static_assert(offsetof(Args, attention_sink_f32) == 160u);
static_assert(offsetof(Args, source_addresses_u64) == 168u);
static_assert(offsetof(Args, query_bf16) == 176u);
static_assert(offsetof(Args, gathered_kv_bf16) == 184u);
static_assert(offsetof(Args, raw_scores_f32) == 192u);
static_assert(offsetof(Args, probabilities_bf16) == 200u);
static_assert(offsetof(Args, online_rescales_f32) == 208u);
static_assert(offsetof(Args, denominators_f32) == 216u);
static_assert(offsetof(Args, output_f32) == 224u);
static_assert(offsetof(Args, status_i32) == 232u);
static_assert(offsetof(Args, stream) == 240u);

using Status = ExplicitSelectionStatus;

namespace detail {

struct Binding {
  const float *query;
  const float *first_plane;
  const float *second_plane;
  const std::int32_t *block_slots;
  const std::int32_t *block_offsets;
  const std::int32_t *sequence_kv_lens;
  const std::int32_t *row_sequence_ids;
  const std::int32_t *row_kv_lens;
  const std::int32_t *selected_indices;
  const std::int32_t *selectors;
  const float *attention_sink;
  std::uint64_t *source_addresses;
  std::uint16_t *query_bf16;
  std::uint16_t *gathered_kv_bf16;
  float *raw_scores;
  std::uint16_t *probabilities_bf16;
  float *online_rescales;
  float *denominators;
  float *output;
  std::int32_t *status;
};

struct ValueLocation {
  const float *values;
  std::uint64_t base;
  bool valid;
};

struct alignas(16) MmaSharedStorage {
  alignas(16) std::uint16_t common[kMmaColumns * kKTile];
  alignas(16) std::uint16_t activation[kWarps][kKTile * kMmaRows];
};

static_assert(sizeof(MmaSharedStorage) == 1536u);
static_assert(std::is_trivially_copyable_v<Binding>);

inline constexpr bool aligned(std::uint64_t address, std::uint64_t alignment) {
  return address != 0u && (address & (alignment - 1u)) == 0u;
}

inline constexpr bool optional_aligned(std::uint64_t address,
                                       std::uint64_t alignment) {
  return address == 0u || (address & (alignment - 1u)) == 0u;
}

template <class T>
__host__ __device__ __forceinline__ T *device_pointer(std::uint64_t address) {
  return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address));
}

__device__ __forceinline__ bool checked_multiply(std::uint64_t left,
                                                 std::uint64_t right,
                                                 std::uint64_t &result) {
  if (left != 0u && right > UINT64_MAX / left) {
    return false;
  }
  result = left * right;
  return true;
}

__device__ __forceinline__ bool
checked_add(std::uint64_t left, std::uint64_t right, std::uint64_t &result) {
  if (right > UINT64_MAX - left) {
    return false;
  }
  result = left + right;
  return true;
}

__device__ __forceinline__ std::uint32_t
sequence_for_row(const Args &args, const Binding &binding, std::uint32_t row) {
  if ((args.flags & 1u) != 0u) {
    const std::int32_t sequence = binding.row_sequence_ids[row];
    return sequence >= 0 ? static_cast<std::uint32_t>(sequence) : UINT32_MAX;
  }
  return args.tokens_per_sequence == 0u ? 0u : row / args.tokens_per_sequence;
}

__device__ __forceinline__ ValueLocation
paged_location(const Args &args, const Binding &binding, const float *plane,
               std::uint64_t plane_elements, std::uint32_t elements_per_token,
               std::uint32_t sequence, std::int32_t logical) {
  if (logical < 0 || sequence == UINT32_MAX) {
    return {nullptr, 0u, false};
  }

  const std::int32_t sequence_length = binding.sequence_kv_lens[sequence];
  if (sequence_length < 0 || logical >= sequence_length) {
    return {nullptr, 0u, false};
  }

  const std::int32_t begin = binding.block_offsets[sequence];
  const std::int32_t end = binding.block_offsets[sequence + 1u];
  if (begin < 0 || end < begin) {
    return {nullptr, 0u, false};
  }
  const std::uint64_t page =
      static_cast<std::uint32_t>(logical) / args.page_tokens;
  std::uint64_t entry = 0u;
  if (!checked_add(static_cast<std::uint64_t>(begin), page, entry) ||
      entry >= static_cast<std::uint64_t>(end)) {
    return {nullptr, 0u, false};
  }
  const std::int32_t slot = binding.block_slots[entry];
  if (slot < 0) {
    return {nullptr, 0u, false};
  }

  std::uint64_t layer_stride = 0u;
  std::uint64_t slot_stride = 0u;
  std::uint64_t base = 0u;
  std::uint64_t part = 0u;
  if (!checked_multiply(args.page_tokens, elements_per_token, layer_stride) ||
      !checked_multiply(args.layer_count, layer_stride, slot_stride) ||
      !checked_multiply(static_cast<std::uint32_t>(slot), slot_stride, base) ||
      !checked_multiply(args.layer_index, layer_stride, part) ||
      !checked_add(base, part, base) ||
      !checked_multiply(static_cast<std::uint32_t>(logical) % args.page_tokens,
                        elements_per_token, part) ||
      !checked_add(base, part, base) || base > plane_elements ||
      args.head_dim > plane_elements - base) {
    return {nullptr, 0u, false};
  }
  return {plane, base, true};
}

__device__ __forceinline__ ValueLocation
selected_location(const Args &args, const Binding &binding, std::uint32_t row,
                  std::uint32_t selected) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(row) * args.selected_width + selected;
  std::int32_t logical = binding.selected_indices[index];
  if ((args.flags & 2u) != 0u) {
    const std::int32_t visible = binding.row_kv_lens[row];
    if (logical < 0 || logical >= visible) {
      logical = -1;
    }
  }
  if (logical < 0) {
    return {nullptr, 0u, false};
  }

  if (args.kind == kContiguous) {
    if (static_cast<std::uint32_t>(logical) >= args.kv_len) {
      return {nullptr, 0u, false};
    }
    std::uint64_t base = 0u;
    if (!checked_multiply(static_cast<std::uint32_t>(logical), args.head_dim,
                          base) ||
        base > args.first_plane_elements ||
        args.head_dim > args.first_plane_elements - base) {
      return {nullptr, 0u, false};
    }
    return {binding.first_plane, base, true};
  }

  const std::uint32_t sequence = sequence_for_row(args, binding, row);
  const std::int32_t selector =
      args.kind == kDualPaged ? binding.selectors[index] : 0;
  if (selector == 0) {
    return paged_location(args, binding, binding.first_plane,
                          args.first_plane_elements,
                          args.first_elements_per_token, sequence, logical);
  }
  if (args.kind == kDualPaged && selector == 1) {
    return paged_location(args, binding, binding.second_plane,
                          args.second_plane_elements,
                          args.second_elements_per_token, sequence, logical);
  }
  return {nullptr, 0u, false};
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (std::uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (std::uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__global__
__launch_bounds__(kGatherThreads) void gather_pack_kernel(Args args,
                                                          Binding binding) {
  __shared__ std::uint64_t shared_source_address;

  const std::uint64_t selection_index = blockIdx.x;
  const std::uint32_t row =
      static_cast<std::uint32_t>(selection_index / args.selected_width);
  const std::uint32_t selected =
      static_cast<std::uint32_t>(selection_index % args.selected_width);

  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *binding.status = 0;
  }
  if (threadIdx.x == 0u) {
    const ValueLocation location =
        selected_location(args, binding, row, selected);
    const float *source =
        location.valid ? location.values + location.base : nullptr;
    shared_source_address =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(source));
    binding.source_addresses[selection_index] = shared_source_address;
  }
  __syncthreads();

  const auto *source = reinterpret_cast<const float *>(
      static_cast<std::uintptr_t>(shared_source_address));
  const std::uint64_t gathered_base = selection_index * args.head_dim;
  for (std::uint32_t dimension = threadIdx.x; dimension < kHeadDim;
       dimension += blockDim.x) {
    binding.gathered_kv_bf16[gathered_base + dimension] =
        source == nullptr
            ? 0u
            : ::ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::
                  detail::f32_to_bf16_rne(source[dimension]);
  }

  const std::uint64_t global_thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t global_stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const std::uint64_t query_elements =
      static_cast<std::uint64_t>(args.rows) * kHeads * kHeadDim;
  for (std::uint64_t index = global_thread; index < query_elements;
       index += global_stride) {
    binding.query_bf16[index] = ::ferrule::cuda::cutlass::capabilities::
        hybrid_mla_attention::detail::f32_to_bf16_rne(binding.query[index]);
  }
}

__device__ __forceinline__ void
stage_qk_kv(const Args &args, const Binding &binding, MmaSharedStorage &shared,
            std::uint32_t row, std::uint32_t selected_base,
            std::uint32_t dimension_base) {
  for (std::uint32_t linear = threadIdx.x; linear < kMmaColumns * kKTile;
       linear += blockDim.x) {
    const std::uint32_t selected_local = linear / kKTile;
    const std::uint32_t dimension_local = linear % kKTile;
    const std::uint32_t selected = selected_base + selected_local;
    shared.common[linear] =
        selected < args.selected_width
            ? binding.gathered_kv_bf16[(static_cast<std::uint64_t>(row) *
                                            args.selected_width +
                                        selected) *
                                           kHeadDim +
                                       dimension_base + dimension_local]
            : 0u;
  }
}

__device__ __forceinline__ void
stage_query(const Binding &binding, MmaSharedStorage &shared,
            std::uint32_t warp, std::uint32_t lane, std::uint32_t row,
            std::uint32_t first_head, std::uint32_t dimension_base) {
  for (std::uint32_t linear = lane; linear < kKTile * kMmaRows;
       linear += kWarpSize) {
    const std::uint32_t dimension_local = linear / kMmaRows;
    const std::uint32_t mma_row = linear % kMmaRows;
    const std::uint32_t head = first_head + mma_row;
    shared.activation[warp][linear] =
        binding.query_bf16[(static_cast<std::uint64_t>(row) * kHeads + head) *
                               kHeadDim +
                           dimension_base + dimension_local];
  }
}

__device__ __forceinline__ void
store_qk_scores(const Args &args, const Binding &binding, std::uint32_t row,
                std::uint32_t first_head, std::uint32_t selected_base,
                std::uint32_t lane, const float (&accumulator)[4]) {
  const std::uint32_t selected_group = lane >> 2;
  const std::uint32_t mma_row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t selected =
        selected_base + selected_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t head = first_head + mma_row_pair * 2u + (element & 1u);
    if (selected < args.selected_width) {
      binding.raw_scores[(static_cast<std::uint64_t>(row) * kHeads + head) *
                             args.selected_width +
                         selected] = accumulator[element];
    }
  }
}

__global__ __launch_bounds__(kMmaThreads, 1) void qk_kernel(Args args,
                                                            Binding binding) {
  __shared__ MmaSharedStorage shared;
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t selection_tiles =
      (args.selected_width + kMmaColumns - 1u) / kMmaColumns;
  const std::uint32_t selection_tile = blockIdx.x % selection_tiles;
  const std::uint32_t row_head_group = blockIdx.x / selection_tiles;
  const std::uint32_t head_group = row_head_group % kHeadGroups;
  const std::uint32_t row = row_head_group / kHeadGroups;
  const std::uint32_t first_head =
      head_group * kHeadsPerBlock + warp * kHeadsPerWarp;
  const std::uint32_t selected_base = selection_tile * kMmaColumns;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t dimension_base = 0u; dimension_base < kHeadDim;
       dimension_base += kKTile) {
    stage_qk_kv(args, binding, shared, row, selected_base, dimension_base);
    stage_query(binding, shared, warp, lane, row, first_head, dimension_base);
    __syncthreads();
    std::uint32_t kv_fragment[4];
    std::uint32_t query_fragment[2];
    ::ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::detail::
        load_weight_fragment(shared.common, lane, kv_fragment);
    ::ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::detail::
        load_activation_fragment(shared.activation[warp], lane, query_fragment);
    ::ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::detail::
        mma_bf16(accumulator, kv_fragment, query_fragment);
    __syncthreads();
  }
  store_qk_scores(args, binding, row, first_head, selected_base, lane,
                  accumulator);
}

__global__ __launch_bounds__(kMmaThreads) void softmax_kernel(Args args,
                                                              Binding binding) {
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint64_t pair =
      static_cast<std::uint64_t>(blockIdx.x) * kWarps + warp;
  const std::uint64_t pair_count =
      static_cast<std::uint64_t>(args.rows) * kHeads;
  if (pair >= pair_count) {
    return;
  }

  const std::uint32_t row = static_cast<std::uint32_t>(pair / kHeads);
  const std::uint32_t head = static_cast<std::uint32_t>(pair % kHeads);
  const std::uint64_t selected_base =
      static_cast<std::uint64_t>(row) * args.selected_width;
  const std::uint64_t score_base = pair * args.selected_width;

  const std::uint32_t online_tiles =
      (args.selected_width + kOnlineSoftmaxTile - 1u) / kOnlineSoftmaxTile;
  float running_maximum = -INFINITY;
  float denominator = 0.0f;

  for (std::uint32_t online_tile = 0u; online_tile < online_tiles;
       ++online_tile) {
    const std::uint32_t tile_base = online_tile * kOnlineSoftmaxTile;
    float tile_maximum = -INFINITY;
#pragma unroll
    for (std::uint32_t iteration = 0u; iteration < 2u; ++iteration) {
      const std::uint32_t selected = tile_base + lane + iteration * kWarpSize;
      if (selected < args.selected_width &&
          binding.source_addresses[selected_base + selected] != 0u) {
        tile_maximum = fmaxf(
            tile_maximum, __fmul_rn(binding.raw_scores[score_base + selected],
                                    args.softmax_scale));
      }
    }
    tile_maximum = warp_max(tile_maximum);
    const float next_maximum = fmaxf(running_maximum, tile_maximum);
    const float rescale = running_maximum == -INFINITY
                              ? 0.0f
                              : expf(__fsub_rn(running_maximum, next_maximum));
    float tile_sum = 0.0f;

#pragma unroll
    for (std::uint32_t iteration = 0u; iteration < 2u; ++iteration) {
      const std::uint32_t selected = tile_base + lane + iteration * kWarpSize;
      if (selected >= args.selected_width) {
        continue;
      }
      std::uint16_t weight = 0u;
      if (binding.source_addresses[selected_base + selected] != 0u) {
        const float value =
            expf(__fmaf_rn(binding.raw_scores[score_base + selected],
                           args.softmax_scale, -next_maximum));
        tile_sum = __fadd_rn(tile_sum, value);
        weight = ::ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::
            detail::f32_to_bf16_rne(value);
      }
      binding.probabilities_bf16[score_base + selected] = weight;
    }
    tile_sum = warp_sum(tile_sum);
    denominator = __fmaf_rn(denominator, rescale, tile_sum);
    if (lane == 0u) {
      binding.online_rescales[pair * online_tiles + online_tile] = rescale;
    }
    running_maximum = next_maximum;
  }

  if (lane == 0u) {
    binding.denominators[pair] =
        running_maximum == -INFINITY
            ? 1.0f
            : __fadd_rn(denominator,
                        expf(__fsub_rn(binding.attention_sink[head],
                                       running_maximum)));
  }
}

__device__ __forceinline__ void
stage_pv_values(const Args &args, const Binding &binding,
                MmaSharedStorage &shared, std::uint32_t row,
                std::uint32_t channel_base, std::uint32_t selected_base) {
  for (std::uint32_t linear = threadIdx.x; linear < kMmaColumns * kKTile;
       linear += blockDim.x) {
    const std::uint32_t channel_local = linear / kKTile;
    const std::uint32_t selected_local = linear % kKTile;
    const std::uint32_t selected = selected_base + selected_local;
    shared.common[linear] =
        selected < args.selected_width
            ? binding.gathered_kv_bf16[(static_cast<std::uint64_t>(row) *
                                            args.selected_width +
                                        selected) *
                                           kHeadDim +
                                       channel_base + channel_local]
            : 0u;
  }
}

__device__ __forceinline__ void
stage_probabilities(const Args &args, const Binding &binding,
                    MmaSharedStorage &shared, std::uint32_t warp,
                    std::uint32_t lane, std::uint32_t row,
                    std::uint32_t first_head, std::uint32_t selected_base) {
  for (std::uint32_t linear = lane; linear < kKTile * kMmaRows;
       linear += kWarpSize) {
    const std::uint32_t selected_local = linear / kMmaRows;
    const std::uint32_t mma_row = linear % kMmaRows;
    const std::uint32_t selected = selected_base + selected_local;
    const std::uint32_t head = first_head + mma_row;
    const std::uint64_t probability_base =
        (static_cast<std::uint64_t>(row) * kHeads + head) * args.selected_width;
    shared.activation[warp][linear] =
        selected < args.selected_width
            ? binding.probabilities_bf16[probability_base + selected]
            : 0u;
  }
}

__device__ __forceinline__ void
rescale_pv_accumulator(const Args &args, const Binding &binding,
                       std::uint32_t row, std::uint32_t first_head,
                       std::uint32_t online_tile, std::uint32_t lane,
                       float (&accumulator)[4]) {
  const std::uint32_t online_tiles =
      (args.selected_width + kOnlineSoftmaxTile - 1u) / kOnlineSoftmaxTile;
  const std::uint32_t mma_row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t head = first_head + mma_row_pair * 2u + (element & 1u);
    const std::uint64_t pair = static_cast<std::uint64_t>(row) * kHeads + head;
    accumulator[element] =
        __fmul_rn(accumulator[element],
                  binding.online_rescales[pair * online_tiles + online_tile]);
  }
}

__device__ __forceinline__ void
store_pv_output(const Binding &binding, std::uint32_t row,
                std::uint32_t first_head, std::uint32_t channel_base,
                std::uint32_t lane, const float (&accumulator)[4]) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t mma_row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t head = first_head + mma_row_pair * 2u + (element & 1u);
    const std::uint64_t pair = static_cast<std::uint64_t>(row) * kHeads + head;
    const float normalized =
        __fdiv_rn(accumulator[element], binding.denominators[pair]);
    const std::uint16_t output_bf16 = ::ferrule::cuda::cutlass::capabilities::
        hybrid_mla_attention::detail::f32_to_bf16_rne(normalized);
    binding.output[pair * kHeadDim + channel] = ::ferrule::cuda::cutlass::
        capabilities::hybrid_mla_attention::detail::bf16_to_f32(output_bf16);
  }
}

__global__ __launch_bounds__(kMmaThreads, 1) void pv_kernel(Args args,
                                                            Binding binding) {
  __shared__ MmaSharedStorage shared;
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t channel_tile = blockIdx.x % kChannelTiles;
  const std::uint32_t row_head_group = blockIdx.x / kChannelTiles;
  const std::uint32_t head_group = row_head_group % kHeadGroups;
  const std::uint32_t row = row_head_group / kHeadGroups;
  const std::uint32_t first_head =
      head_group * kHeadsPerBlock + warp * kHeadsPerWarp;
  const std::uint32_t channel_base = channel_tile * kMmaColumns;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t selected_base = 0u; selected_base < args.selected_width;
       selected_base += kKTile) {
    if (selected_base % kOnlineSoftmaxTile == 0u) {
      rescale_pv_accumulator(args, binding, row, first_head,
                             selected_base / kOnlineSoftmaxTile, lane,
                             accumulator);
    }
    stage_pv_values(args, binding, shared, row, channel_base, selected_base);
    stage_probabilities(args, binding, shared, warp, lane, row, first_head,
                        selected_base);
    __syncthreads();
    std::uint32_t value_fragment[4];
    std::uint32_t probability_fragment[2];
    ::ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::detail::
        load_weight_fragment(shared.common, lane, value_fragment);
    ::ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::detail::
        load_activation_fragment(shared.activation[warp], lane,
                                 probability_fragment);
    ::ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::detail::
        mma_bf16(accumulator, value_fragment, probability_fragment);
    __syncthreads();
  }
  store_pv_output(binding, row, first_head, channel_base, lane, accumulator);
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr) {
    return Status::kInvalidArgument;
  }
#if !FERRULE_CUDA_HAS_BF16_MMA_SYNC
  return Status::kUnsupported;
#else
  if (args->heads != kHeads || args->head_dim != kHeadDim ||
      args->selected_width > kMaximumSelectedWidth) {
    return Status::kUnsupported;
  }
  if (args->kind < kContiguous || args->kind > kDualPaged || args->rows == 0u ||
      args->selected_width == 0u || (args->flags & ~3u) != 0u ||
      args->reserved0 != 0u || !std::isfinite(args->softmax_scale) ||
      args->softmax_scale <= 0.0f || args->first_plane_elements == 0u) {
    return Status::kInvalidArgument;
  }
  if (args->kind != kContiguous &&
      (args->page_tokens == 0u || args->layer_count == 0u ||
       args->layer_index >= args->layer_count ||
       args->first_elements_per_token < kHeadDim ||
       (((args->flags & 1u) == 0u) && args->tokens_per_sequence == 0u))) {
    return Status::kInvalidArgument;
  }
  if (args->kind == kDualPaged && (args->second_elements_per_token < kHeadDim ||
                                   args->second_plane_elements == 0u)) {
    return Status::kInvalidArgument;
  }

  const bool common_pointers =
      detail::aligned(args->query_f32, 16u) &&
      detail::aligned(args->first_plane_f32, 16u) &&
      detail::aligned(args->selected_indices_i32, 4u) &&
      detail::aligned(args->attention_sink_f32, 16u) &&
      detail::aligned(args->source_addresses_u64, 16u) &&
      detail::aligned(args->query_bf16, 16u) &&
      detail::aligned(args->gathered_kv_bf16, 16u) &&
      detail::aligned(args->raw_scores_f32, 16u) &&
      detail::aligned(args->probabilities_bf16, 16u) &&
      detail::aligned(args->online_rescales_f32, 16u) &&
      detail::aligned(args->denominators_f32, 16u) &&
      detail::aligned(args->output_f32, 16u) &&
      detail::aligned(args->status_i32, 4u) &&
      (((args->flags & 1u) == 0u) ||
       detail::aligned(args->row_sequence_ids_i32, 4u)) &&
      (((args->flags & 2u) == 0u) ||
       detail::aligned(args->row_kv_lens_i32, 4u));
  if (!common_pointers) {
    return Status::kInvalidArgument;
  }
  if (args->kind != kContiguous &&
      (!detail::aligned(args->block_slots_i32, 4u) ||
       !detail::aligned(args->block_offsets_i32, 4u) ||
       !detail::aligned(args->sequence_kv_lens_i32, 4u))) {
    return Status::kInvalidArgument;
  }
  if (args->kind == kDualPaged &&
      (!detail::aligned(args->second_plane_f32, 16u) ||
       !detail::aligned(args->selectors_i32, 4u))) {
    return Status::kInvalidArgument;
  }
  if (!detail::optional_aligned(args->second_plane_f32, 16u) ||
      !detail::optional_aligned(args->block_slots_i32, 4u) ||
      !detail::optional_aligned(args->block_offsets_i32, 4u) ||
      !detail::optional_aligned(args->sequence_kv_lens_i32, 4u) ||
      !detail::optional_aligned(args->row_sequence_ids_i32, 4u) ||
      !detail::optional_aligned(args->row_kv_lens_i32, 4u) ||
      !detail::optional_aligned(args->selectors_i32, 4u)) {
    return Status::kInvalidArgument;
  }
  return Status::kSuccess;
#endif
}

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }

  const std::uint64_t selections =
      static_cast<std::uint64_t>(args->rows) * args->selected_width;
  const std::uint64_t selection_tiles =
      (args->selected_width + kMmaColumns - 1u) / kMmaColumns;
  const std::uint64_t qk_blocks =
      static_cast<std::uint64_t>(args->rows) * kHeadGroups * selection_tiles;
  const std::uint64_t pairs = static_cast<std::uint64_t>(args->rows) * kHeads;
  const std::uint64_t softmax_blocks = (pairs + kWarps - 1u) / kWarps;
  const std::uint64_t pv_blocks =
      static_cast<std::uint64_t>(args->rows) * kHeadGroups * kChannelTiles;
  if (selections == 0u || selections > UINT32_MAX || qk_blocks == 0u ||
      qk_blocks > UINT32_MAX || softmax_blocks == 0u ||
      softmax_blocks > UINT32_MAX || pv_blocks == 0u ||
      pv_blocks > UINT32_MAX) {
    return Status::kLaunchFailed;
  }

  const detail::Binding binding{
      detail::device_pointer<const float>(args->query_f32),
      detail::device_pointer<const float>(args->first_plane_f32),
      detail::device_pointer<const float>(args->second_plane_f32),
      detail::device_pointer<const std::int32_t>(args->block_slots_i32),
      detail::device_pointer<const std::int32_t>(args->block_offsets_i32),
      detail::device_pointer<const std::int32_t>(args->sequence_kv_lens_i32),
      detail::device_pointer<const std::int32_t>(args->row_sequence_ids_i32),
      detail::device_pointer<const std::int32_t>(args->row_kv_lens_i32),
      detail::device_pointer<const std::int32_t>(args->selected_indices_i32),
      detail::device_pointer<const std::int32_t>(args->selectors_i32),
      detail::device_pointer<const float>(args->attention_sink_f32),
      detail::device_pointer<std::uint64_t>(args->source_addresses_u64),
      detail::device_pointer<std::uint16_t>(args->query_bf16),
      detail::device_pointer<std::uint16_t>(args->gathered_kv_bf16),
      detail::device_pointer<float>(args->raw_scores_f32),
      detail::device_pointer<std::uint16_t>(args->probabilities_bf16),
      detail::device_pointer<float>(args->online_rescales_f32),
      detail::device_pointer<float>(args->denominators_f32),
      detail::device_pointer<float>(args->output_f32),
      detail::device_pointer<std::int32_t>(args->status_i32),
  };
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));

  detail::gather_pack_kernel<<<static_cast<std::uint32_t>(selections),
                               kGatherThreads, 0u, stream>>>(*args, binding);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  detail::qk_kernel<<<static_cast<std::uint32_t>(qk_blocks), kMmaThreads, 0u,
                      stream>>>(*args, binding);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  detail::softmax_kernel<<<static_cast<std::uint32_t>(softmax_blocks),
                           kMmaThreads, 0u, stream>>>(*args, binding);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  detail::pv_kernel<<<static_cast<std::uint32_t>(pv_blocks), kMmaThreads, 0u,
                      stream>>>(*args, binding);
  return cudaPeekAtLastError() == cudaSuccess ? Status::kSuccess
                                              : Status::kLaunchFailed;
}

} // namespace
  // ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::schedules::bf16_mma
  // schedules::bf16_mma

#endif // FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_SCHEDULES_BF16_MMA_CUH_
