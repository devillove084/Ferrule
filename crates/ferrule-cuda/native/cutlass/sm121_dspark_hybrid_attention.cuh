#ifndef FERRULE_CUTLASS_SM121_DSPARK_HYBRID_ATTENTION_CUH_
#define FERRULE_CUTLASS_SM121_DSPARK_HYBRID_ATTENTION_CUH_

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(FERRULE_CUTLASS_TARGET_SM) && FERRULE_CUTLASS_TARGET_SM != 121
#error "sm121_dspark_hybrid_attention.cuh requires Ferrule's SM121a target"
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1210
#error "sm121_dspark_hybrid_attention.cuh device code must target SM121"
#endif

namespace ferrule::cutlass::sm121_dspark_hybrid_attention {

inline constexpr std::uint32_t kArgsVersion = 1u;
inline constexpr std::uint32_t kBlockRows = 5u;
inline constexpr std::uint32_t kHeads = 64u;
inline constexpr std::uint32_t kHeadDim = 512u;
inline constexpr std::uint32_t kWindowSize = 128u;
inline constexpr std::uint32_t kPageTokens = 16u;
inline constexpr std::uint32_t kTokenCapacity = kWindowSize + kBlockRows;
inline constexpr std::uint32_t kMmaRows = 8u;
inline constexpr std::uint32_t kMmaColumns = 16u;
inline constexpr std::uint32_t kKTile = 16u;
inline constexpr std::uint32_t kWarpSize = 32u;
inline constexpr std::uint32_t kWarps = 4u;
inline constexpr std::uint32_t kThreads = kWarpSize * kWarps;
inline constexpr std::uint32_t kHeadGroups = kHeads / kWarps;
inline constexpr std::uint32_t kTokenTiles =
    (kTokenCapacity + kMmaColumns - 1u) / kMmaColumns;
inline constexpr std::uint32_t kCooperativeBlocks = kHeadGroups * kTokenTiles;

// One checkpoint-native DSpark proposal-attention transaction:
//
//   Q F32/BF16-boundary            [5, 64, 512]
//   committed paged latent KV      [last min(sequence, 128), 512]
//   ephemeral proposal-block KV    [5, 512]
//   per-head attention sink        [64]
//       -> QK BF16 MMA
//       -> sink-aware softmax
//       -> PV BF16 MMA
//   output F32                     [5, 64, 512]
//
// Every proposal query sees the complete five-row proposal block. The block KV
// operand is read-only and is never published through the page table. Scores,
// BF16 probabilities, and device status are graph-stable Ferrule-owned scratch.
struct Args {
  std::uint32_t args_version;
  std::uint32_t block_rows;
  std::uint32_t heads;
  std::uint32_t head_dim;
  std::uint32_t sequence_tokens;
  std::uint32_t window_size;
  std::uint32_t page_tokens;
  std::uint32_t elements_per_token;
  std::uint32_t layer_index;
  std::uint32_t layer_count;
  std::uint32_t block_slot_offset;
  std::uint32_t block_slot_count;
  float softmax_scale;
  std::uint32_t reserved;
  std::uint64_t context_plane_elements;

  std::uint64_t query_f32;
  std::uint64_t context_plane_f32;
  std::uint64_t block_kv_f32;
  std::uint64_t block_slots_i32;
  std::uint64_t attention_sink_f32;
  std::uint64_t query_bf16;
  std::uint64_t gathered_kv_bf16;
  std::uint64_t scores_f32;
  std::uint64_t probabilities_bf16;
  std::uint64_t output_f32;
  std::uint64_t status_i32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 160u,
              "SM121 DSpark hybrid-attention POD ABI changed");

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidAbi = 1,
  kInvalidArgument = 2,
  kLaunchFailed = 3,
};

namespace detail {

using Bf16Mma = cute::SM80_16x8x16_F32BF16BF16F32_TN;

struct alignas(16) SharedStorage {
  // Shared by four heads: one latent K/V tile is read from global memory once.
  alignas(16) std::uint16_t common[kMmaColumns * kKTile];
  // One query/probability tile per head warp.
  alignas(16) std::uint16_t activation[kWarps][kKTile * kMmaRows];
};

static_assert(sizeof(SharedStorage) == 1536u);

struct Binding {
  const float *query;
  const float *context_plane;
  const float *block_kv;
  const std::int32_t *block_slots;
  const float *attention_sink;
  std::uint16_t *query_bf16;
  std::uint16_t *gathered_kv_bf16;
  float *scores;
  std::uint16_t *probabilities;
  float *output;
  std::int32_t *status;
};

static_assert(std::is_trivially_copyable_v<Binding>);

inline constexpr bool aligned(std::uint64_t address,
                              std::uint64_t alignment) {
  return address != 0u && (address & (alignment - 1u)) == 0u;
}

template <class T>
__host__ __device__ __forceinline__ T *device_pointer(std::uint64_t address) {
  return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address));
}

__device__ __forceinline__ std::uint16_t f32_to_bf16_rne(float value) {
  std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  }
  const std::uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>((bits + bias) >> 16);
}

__device__ __forceinline__ float bf16_to_f32(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16);
}

__device__ __forceinline__ std::uint32_t context_tokens(const Args &args) {
  return args.sequence_tokens < args.window_size ? args.sequence_tokens
                                                  : args.window_size;
}

__device__ __forceinline__ float latent_value(
    const Args &args, const Binding &binding, std::uint32_t concatenated_token,
    std::uint32_t dimension) {
  const std::uint32_t committed = context_tokens(args);
  if (concatenated_token >= committed) {
    const std::uint32_t block_row = concatenated_token - committed;
    return block_row < args.block_rows
               ? binding.block_kv[static_cast<std::uint64_t>(block_row) *
                                      args.head_dim +
                                  dimension]
               : 0.0f;
  }

  const std::uint32_t logical =
      args.sequence_tokens - committed + concatenated_token;
  const std::uint32_t relative_page = logical / args.page_tokens;
  if (relative_page >= args.block_slot_count) {
    atomicExch(binding.status, 1);
    return 0.0f;
  }
  const std::int32_t slot =
      binding.block_slots[args.block_slot_offset + relative_page];
  if (slot < 0) {
    atomicExch(binding.status, 1);
    return 0.0f;
  }

  const std::uint64_t layer_stride =
      static_cast<std::uint64_t>(args.page_tokens) * args.elements_per_token;
  const std::uint64_t slot_stride =
      static_cast<std::uint64_t>(args.layer_count) * layer_stride;
  const std::uint64_t offset = static_cast<std::uint64_t>(slot) * slot_stride +
                               static_cast<std::uint64_t>(args.layer_index) *
                                   layer_stride +
                               static_cast<std::uint64_t>(logical %
                                                          args.page_tokens) *
                                   args.elements_per_token +
                               dimension;
  if (offset >= args.context_plane_elements) {
    atomicExch(binding.status, 1);
    return 0.0f;
  }
  return binding.context_plane[offset];
}

__device__ __forceinline__ void pack_inputs(const Args &args,
                                            const Binding &binding) {
  const std::uint64_t thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const std::uint64_t query_values =
      static_cast<std::uint64_t>(args.block_rows) * args.heads * args.head_dim;
  for (std::uint64_t index = thread; index < query_values; index += stride) {
    binding.query_bf16[index] = f32_to_bf16_rne(binding.query[index]);
  }

  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
  const std::uint64_t kv_values =
      static_cast<std::uint64_t>(total_tokens) * args.head_dim;
  for (std::uint64_t index = thread; index < kv_values; index += stride) {
    const std::uint32_t token = static_cast<std::uint32_t>(index / args.head_dim);
    const std::uint32_t dimension =
        static_cast<std::uint32_t>(index % args.head_dim);
    binding.gathered_kv_bf16[index] =
        f32_to_bf16_rne(latent_value(args, binding, token, dimension));
  }
}

__device__ __forceinline__ void
load_weight_fragment(const std::uint16_t *shared, std::uint32_t lane,
                     std::uint32_t (&fragment)[4]) {
  const std::uint32_t quad = lane >> 3;
  const std::uint32_t row =
      (lane & 7u) + ((quad & 1u) != 0u ? 8u : 0u);
  const std::uint32_t column_bytes = quad >= 2u ? 16u : 0u;
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(shared);
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      bytes + row * 32u + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1],
                                fragment[2], fragment[3]);
}

__device__ __forceinline__ void
load_activation_fragment(const std::uint16_t *shared, std::uint32_t lane,
                         std::uint32_t (&fragment)[2]) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(shared);
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      bytes + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void
mma_bf16(float (&accumulator)[4], const std::uint32_t (&weight)[4],
         const std::uint32_t (&activation)[2]) {
  Bf16Mma::fma(accumulator[0], accumulator[1], accumulator[2],
               accumulator[3], weight[0], weight[1], weight[2], weight[3],
               activation[0], activation[1], accumulator[0], accumulator[1],
               accumulator[2], accumulator[3]);
}

__device__ __forceinline__ void stage_qk_latent(
    const Args &args, const Binding &binding, SharedStorage &shared,
    std::uint32_t token_base, std::uint32_t dimension_base) {
  for (std::uint32_t linear = threadIdx.x;
       linear < kMmaColumns * kKTile; linear += blockDim.x) {
    const std::uint32_t token_local = linear / kKTile;
    const std::uint32_t dimension_local = linear % kKTile;
    const std::uint32_t token = token_base + token_local;
    const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
    shared.common[linear] =
        token < total_tokens
            ? binding.gathered_kv_bf16[
                  static_cast<std::uint64_t>(token) * args.head_dim +
                  dimension_base + dimension_local]
            : 0u;
  }
}

__device__ __forceinline__ void stage_query(
    const Args &args, const Binding &binding, SharedStorage &shared,
    std::uint32_t warp, std::uint32_t lane, std::uint32_t head,
    std::uint32_t dimension_base) {
  for (std::uint32_t linear = lane; linear < kKTile * kMmaRows;
       linear += kWarpSize) {
    const std::uint32_t dimension_local = linear / kMmaRows;
    const std::uint32_t row = linear % kMmaRows;
    shared.activation[warp][linear] =
        row < args.block_rows
            ? binding.query_bf16
                  [(static_cast<std::uint64_t>(row) * args.heads + head) *
                       args.head_dim +
                   dimension_base + dimension_local]
            : 0u;
  }
}

__device__ __forceinline__ void store_scores(
    const Args &args, const Binding &binding, std::uint32_t head,
    std::uint32_t token_base, std::uint32_t lane,
    const float (&accumulator)[4]) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t token =
        token_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row = row_pair * 2u + (element & 1u);
    if (row < args.block_rows && token < total_tokens) {
      binding.scores[(static_cast<std::uint64_t>(row) * args.heads + head) *
                         kTokenCapacity +
                     token] = accumulator[element] * args.softmax_scale;
    }
  }
}

__device__ __forceinline__ void qk_task(
    const Args &args, const Binding &binding, SharedStorage &shared,
    std::uint32_t task, std::uint32_t warp, std::uint32_t lane) {
  const std::uint32_t token_tile = task % kTokenTiles;
  const std::uint32_t head_group = task / kTokenTiles;
  const std::uint32_t token_base = token_tile * kMmaColumns;
  const std::uint32_t head = head_group * kWarps + warp;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t dimension_base = 0u; dimension_base < kHeadDim;
       dimension_base += kKTile) {
    stage_qk_latent(args, binding, shared, token_base, dimension_base);
    stage_query(args, binding, shared, warp, lane, head, dimension_base);
    __syncthreads();
    std::uint32_t latent_fragment[4];
    std::uint32_t query_fragment[2];
    load_weight_fragment(shared.common, lane, latent_fragment);
    load_activation_fragment(shared.activation[warp], lane, query_fragment);
    mma_bf16(accumulator, latent_fragment, query_fragment);
    __syncthreads();
  }
  store_scores(args, binding, head, token_base, lane, accumulator);
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

__device__ __forceinline__ void softmax_task(
    const Args &args, const Binding &binding, std::uint32_t task,
    std::uint32_t lane) {
  const std::uint32_t head = task % args.heads;
  const std::uint32_t row = task / args.heads;
  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
  const std::uint64_t base =
      (static_cast<std::uint64_t>(row) * args.heads + head) * kTokenCapacity;

  float maximum = lane == 0u ? binding.attention_sink[head] : -INFINITY;
  for (std::uint32_t token = lane; token < total_tokens;
       token += kWarpSize) {
    maximum = fmaxf(maximum, binding.scores[base + token]);
  }
  maximum = warp_max(maximum);

  float denominator =
      lane == 0u ? __expf(binding.attention_sink[head] - maximum) : 0.0f;
  for (std::uint32_t token = lane; token < total_tokens;
       token += kWarpSize) {
    denominator += __expf(binding.scores[base + token] - maximum);
  }
  denominator = warp_sum(denominator);

  for (std::uint32_t token = lane; token < total_tokens;
       token += kWarpSize) {
    const float probability =
        __expf(binding.scores[base + token] - maximum) / denominator;
    binding.probabilities[base + token] = f32_to_bf16_rne(probability);
  }
}

__device__ __forceinline__ void stage_pv_latent(
    const Args &args, const Binding &binding, SharedStorage &shared,
    std::uint32_t channel_base, std::uint32_t token_base) {
  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
  for (std::uint32_t linear = threadIdx.x;
       linear < kMmaColumns * kKTile; linear += blockDim.x) {
    const std::uint32_t channel_local = linear / kKTile;
    const std::uint32_t token_local = linear % kKTile;
    const std::uint32_t token = token_base + token_local;
    shared.common[linear] =
        token < total_tokens
            ? binding.gathered_kv_bf16[
                  static_cast<std::uint64_t>(token) * args.head_dim +
                  channel_base + channel_local]
            : 0u;
  }
}

__device__ __forceinline__ void stage_probabilities(
    const Args &args, const Binding &binding, SharedStorage &shared,
    std::uint32_t warp, std::uint32_t lane, std::uint32_t head,
    std::uint32_t token_base) {
  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
  for (std::uint32_t linear = lane; linear < kKTile * kMmaRows;
       linear += kWarpSize) {
    const std::uint32_t token_local = linear / kMmaRows;
    const std::uint32_t row = linear % kMmaRows;
    const std::uint32_t token = token_base + token_local;
    shared.activation[warp][linear] =
        row < args.block_rows && token < total_tokens
            ? binding.probabilities
                  [(static_cast<std::uint64_t>(row) * args.heads + head) *
                       kTokenCapacity +
                   token]
            : 0u;
  }
}

__device__ __forceinline__ void store_output(
    const Args &args, const Binding &binding, std::uint32_t head,
    std::uint32_t channel_base, std::uint32_t lane,
    const float (&accumulator)[4]) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row = row_pair * 2u + (element & 1u);
    if (row < args.block_rows && channel < args.head_dim) {
      binding.output[(static_cast<std::uint64_t>(row) * args.heads + head) *
                         args.head_dim +
                     channel] = accumulator[element];
    }
  }
}

__device__ __forceinline__ void pv_task(
    const Args &args, const Binding &binding, SharedStorage &shared,
    std::uint32_t task, std::uint32_t warp, std::uint32_t lane) {
  constexpr std::uint32_t kChannelTiles = kHeadDim / kMmaColumns;
  const std::uint32_t channel_tile = task % kChannelTiles;
  const std::uint32_t head_group = task / kChannelTiles;
  const std::uint32_t channel_base = channel_tile * kMmaColumns;
  const std::uint32_t head = head_group * kWarps + warp;
  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t token_base = 0u; token_base < total_tokens;
       token_base += kKTile) {
    stage_pv_latent(args, binding, shared, channel_base, token_base);
    stage_probabilities(args, binding, shared, warp, lane, head, token_base);
    __syncthreads();
    std::uint32_t latent_fragment[4];
    std::uint32_t probability_fragment[2];
    load_weight_fragment(shared.common, lane, latent_fragment);
    load_activation_fragment(shared.activation[warp], lane,
                             probability_fragment);
    mma_bf16(accumulator, latent_fragment, probability_fragment);
    __syncthreads();
  }
  store_output(args, binding, head, channel_base, lane, accumulator);
}

__global__ __launch_bounds__(kThreads, 1) void kernel(Args args,
                                                      Binding binding) {
  __shared__ SharedStorage shared;
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp;
  const std::uint32_t warp_stride = gridDim.x * kWarps;

  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *binding.status = 0;
  }
  grid.sync();

  pack_inputs(args, binding);
  grid.sync();

  for (std::uint32_t task = blockIdx.x; task < kCooperativeBlocks;
       task += gridDim.x) {
    qk_task(args, binding, shared, task, warp, lane);
  }
  grid.sync();

  const std::uint32_t softmax_tasks = args.block_rows * args.heads;
  for (std::uint32_t task = global_warp; task < softmax_tasks;
       task += warp_stride) {
    softmax_task(args, binding, task, lane);
  }
  grid.sync();

  constexpr std::uint32_t kChannelTiles = kHeadDim / kMmaColumns;
  constexpr std::uint32_t kPvTasks = kHeadGroups * kChannelTiles;
  for (std::uint32_t task = blockIdx.x; task < kPvTasks;
       task += gridDim.x) {
    pv_task(args, binding, shared, task, warp, lane);
  }
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr || args->args_version != kArgsVersion) {
    return Status::kInvalidAbi;
  }
  if (args->page_tokens != kPageTokens) {
    return Status::kInvalidArgument;
  }
  const std::uint64_t required_slots =
      (static_cast<std::uint64_t>(args->sequence_tokens) +
       args->page_tokens - 1u) /
      args->page_tokens;
  if (args->block_rows != kBlockRows || args->heads != kHeads ||
      args->head_dim != kHeadDim || args->window_size != kWindowSize ||
      args->elements_per_token != kHeadDim || args->sequence_tokens == 0u ||
      args->layer_count == 0u || args->layer_index >= args->layer_count ||
      args->block_slot_count < required_slots || args->reserved != 0u ||
      args->context_plane_elements == 0u ||
      !std::isfinite(args->softmax_scale) || args->softmax_scale <= 0.0f) {
    return Status::kInvalidArgument;
  }
  const bool pointers_valid =
      detail::aligned(args->query_f32, 16u) &&
      detail::aligned(args->context_plane_f32, 16u) &&
      detail::aligned(args->block_kv_f32, 16u) &&
      detail::aligned(args->block_slots_i32, 4u) &&
      detail::aligned(args->attention_sink_f32, 16u) &&
      detail::aligned(args->query_bf16, 16u) &&
      detail::aligned(args->gathered_kv_bf16, 16u) &&
      detail::aligned(args->scores_f32, 16u) &&
      detail::aligned(args->probabilities_bf16, 16u) &&
      detail::aligned(args->output_f32, 16u) &&
      detail::aligned(args->status_i32, 4u);
  return pointers_valid ? Status::kSuccess : Status::kInvalidArgument;
}

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));
  const detail::Binding binding{
      detail::device_pointer<const float>(args->query_f32),
      detail::device_pointer<const float>(args->context_plane_f32),
      detail::device_pointer<const float>(args->block_kv_f32),
      detail::device_pointer<const std::int32_t>(args->block_slots_i32),
      detail::device_pointer<const float>(args->attention_sink_f32),
      detail::device_pointer<std::uint16_t>(args->query_bf16),
      detail::device_pointer<std::uint16_t>(args->gathered_kv_bf16),
      detail::device_pointer<float>(args->scores_f32),
      detail::device_pointer<std::uint16_t>(args->probabilities_bf16),
      detail::device_pointer<float>(args->output_f32),
      detail::device_pointer<std::int32_t>(args->status_i32),
  };
  int blocks_per_sm = 0;
  int device = 0;
  int multiprocessors = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm, detail::kernel, kThreads, 0u) != cudaSuccess ||
      cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess ||
      blocks_per_sm <= 0 || multiprocessors <= 0) {
    return Status::kLaunchFailed;
  }
  const std::uint32_t resident_blocks =
      static_cast<std::uint32_t>(blocks_per_sm) *
      static_cast<std::uint32_t>(multiprocessors);
  const std::uint32_t blocks = resident_blocks < kCooperativeBlocks
                                   ? resident_blocks
                                   : kCooperativeBlocks;
  void *kernel_args[] = {const_cast<Args *>(args),
                         const_cast<detail::Binding *>(&binding)};
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::kernel), dim3(blocks, 1u, 1u),
      dim3(kThreads, 1u, 1u), kernel_args, 0u, stream);
  return status == cudaSuccess ? Status::kSuccess : Status::kLaunchFailed;
}

} // namespace ferrule::cutlass::sm121_dspark_hybrid_attention

#endif // FERRULE_CUTLASS_SM121_DSPARK_HYBRID_ATTENTION_CUH_
