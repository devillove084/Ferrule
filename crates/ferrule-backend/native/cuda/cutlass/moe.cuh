#pragma once

#include "cutlass/fp4.cuh"
#include "cutlass/fp8.cuh"
#include "cutlass/target.cuh"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cooperative_groups.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>


// One-launch shared-FFN provider for the DeepSeek-V4 artifact contract.
//
// The kernel is a cooperative, two-stage semantic pipeline rather than three
// host-launched GEMMs:
//
//   1. CTAs partition [M, intermediate] into [8, 128] tiles. Eight warps
//   compute
//      gate/up FP8 MMA tiles, apply SwiGLU, and quantize the tile directly into
//      the compact global FP8 intermediate. No gate/up F32 tensor is published.
//   2. A grid barrier makes the compact intermediate visible. CTAs then
//      partition [M, output] into [8, 128] tiles and compute the down
//      projection.
//
// The compact intermediate is caller-owned graph-stable storage. It is the only
// cross-CTA dependency and is part of this semantic launch; gate/up/down are
// not independently callable provider entries.

namespace ferrule::cuda::cutlass::operators::shared_ffn {

constexpr std::uint32_t kArtifactBlock = 128;
constexpr std::uint32_t kMmaChannels = 16;
constexpr std::uint32_t kMmaRows = 8;
constexpr std::uint32_t kWarpSize = 32;
constexpr std::uint32_t kWarpCount = 8;
constexpr std::uint32_t kThreads = kWarpSize * kWarpCount;
constexpr std::uint32_t kChannelsPerCta = kWarpCount * kMmaChannels;
// The fixed tile uses about 10 KiB shared memory. Keep a conservative
// cooperative-grid cap; architecture-specific occupancy tuning stays here.
constexpr std::uint32_t kCooperativeBlockLimit = 40;

constexpr std::uint32_t kAccumulateOutput = 1u << 0;
constexpr std::uint32_t kKnownFlags = kAccumulateOutput;

struct Args {
  const std::uint8_t *input_fp8;
  const std::uint8_t *input_ue8m0;
  const std::uint8_t *gate_weight_fp8;
  const std::uint8_t *gate_weight_ue8m0;
  const std::uint8_t *up_weight_fp8;
  const std::uint8_t *up_weight_ue8m0;
  const std::uint8_t *down_weight_fp8;
  const std::uint8_t *down_weight_ue8m0;
  float *hidden_f32;
  std::uint8_t *hidden_fp8;
  std::uint8_t *hidden_ue8m0;
  float *output_f32;

  std::uint32_t rows;
  std::uint32_t input_size;
  std::uint32_t intermediate_size;
  std::uint32_t output_size;

  std::uint32_t gate_block_m;
  std::uint32_t gate_block_k;
  std::uint32_t up_block_m;
  std::uint32_t up_block_k;
  std::uint32_t down_block_m;
  std::uint32_t down_block_k;

  float output_scale;
  float swiglu_limit;
  std::uint32_t flags;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);

enum class ValidationResult : std::uint32_t {
  kSuccess = 0,
  kNullPointer,
  kMisalignedPointer,
  kUnsupportedShape,
  kUnsupportedArtifactBlocks,
  kInvalidScalar,
  kInvalidFlags,
};

namespace detail {

constexpr std::size_t kWeightStageBytes = 16u * 32u;
constexpr std::size_t kActivationStageBytes = 8u * 32u;
constexpr std::size_t kWarpStageBytes =
    kWeightStageBytes + kActivationStageBytes;

__host__ __device__ constexpr std::size_t align_up(std::size_t value,
                                                   std::size_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

inline bool aligned_16(const void *pointer) {
  return pointer != nullptr &&
         (reinterpret_cast<std::uintptr_t>(pointer) & 15u) == 0u;
}

inline bool finite_scalar(float value) {
  return value == value && value <= FLT_MAX && value >= -FLT_MAX;
}

__device__ __forceinline__ float ue8m0_to_float(std::uint8_t value) {
  std::uint32_t bits =
      value == 0 ? (1u << 22) : (static_cast<std::uint32_t>(value) << 23);
  return __uint_as_float(bits);
}

__device__ __forceinline__ float nearest_fp8_subnormal(float magnitude) {
  float mantissa = roundf(magnitude * 512.0f);
  mantissa = fminf(7.0f, fmaxf(0.0f, mantissa));
  return mantissa * (1.0f / 512.0f);
}

__device__ __forceinline__ float nearest_fp8_e4m3fn_positive(float magnitude) {
  float best = nearest_fp8_subnormal(magnitude);
  float best_error = fabsf(best - magnitude);
  int exponent_floor = static_cast<int>(floorf(log2f(magnitude)));
#pragma unroll
  for (int exponent = exponent_floor - 1; exponent <= exponent_floor + 1;
       ++exponent) {
    if (exponent < -6 || exponent > 8) {
      continue;
    }
    float scale = exp2f(static_cast<float>(exponent));
    int mantissa = static_cast<int>(roundf((magnitude / scale - 1.0f) * 8.0f));
    int candidate_exponent = exponent;
    if (mantissa < 0) {
      continue;
    }
    if (mantissa > 7) {
      ++candidate_exponent;
      mantissa = 0;
    }
    if (candidate_exponent > 8) {
      candidate_exponent = 8;
      mantissa = 6;
    }
    if (candidate_exponent == 8 && mantissa > 6) {
      mantissa = 6;
    }
    float candidate = exp2f(static_cast<float>(candidate_exponent)) *
                      (1.0f + static_cast<float>(mantissa) * 0.125f);
    float error = fabsf(candidate - magnitude);
    if (error < best_error) {
      best = candidate;
      best_error = error;
    }
  }
  return best;
}

__device__ __forceinline__ std::uint8_t quantize_fp8_e4m3fn_byte(float value) {
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ std::uint16_t f32_to_bf16_rne(float value) {
  std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  }
  const std::uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>((bits + bias) >> 16);
}

__device__ __forceinline__ float bf16_round(float value) {
  return __uint_as_float(static_cast<std::uint32_t>(f32_to_bf16_rne(value))
                         << 16);
}

__device__ __forceinline__ std::uint8_t scale_byte_for_amax(float amax) {
  if (!isfinite(amax) || amax <= 0.0f) {
    return 127u;
  }
  int byte = static_cast<int>(ceilf(log2f(amax / 448.0f))) + 127;
  return static_cast<std::uint8_t>(byte < 0 ? 0 : (byte > 255 ? 255 : byte));
}

__device__ __forceinline__ float clamp_value(float value, float lower,
                                             float upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

__device__ __forceinline__ float sigmoid(float value) {
  if (value < -16.0f) {
    return 0.0f;
  }
  if (value > 16.0f) {
    return 1.0f;
  }
  if (value >= 0.0f) {
    return 1.0f / (1.0f + expf(-value));
  }
  float exponential = expf(value);
  return exponential / (1.0f + exponential);
}

__device__ __forceinline__ float
swiglu(float gate, float up, float output_scale, float swiglu_limit) {
  if (swiglu_limit > 0.0f) {
    gate = gate > swiglu_limit ? swiglu_limit : gate;
    up = clamp_value(up, -swiglu_limit, swiglu_limit);
  }
  return gate * sigmoid(gate) * up * output_scale;
}

__device__ __forceinline__ void
load_weight_fragment(const std::uint8_t *shared, std::uint32_t lane,
                     std::uint32_t (&fragment)[4]) {
  std::uint32_t quad = lane >> 3;
  std::uint32_t row = (lane & 7u) + ((quad & 1u) != 0 ? 8u : 0u);
  std::uint32_t column_bytes = quad >= 2 ? 16u : 0u;
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + row * 32u + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1], fragment[2],
                                fragment[3]);
}

__device__ __forceinline__ void
load_activation_fragment(const std::uint8_t *shared, std::uint32_t lane,
                         std::uint32_t (&fragment)[2]) {
  auto const &source =
      *reinterpret_cast<const cute::uint128_t *>(shared + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void
mma_fp8_e4m3(float (&accumulator)[4], const std::uint32_t (&weight)[4],
             const std::uint32_t (&activation)[2]) {
  architectures::sm89::mma_sync_f32_e4m3_e4m3_m16n8k32(accumulator, weight, activation);
}

struct SharedLayout {
  std::uint8_t *warp_stages;
};

__device__ __forceinline__ SharedLayout
make_shared_layout(std::uint8_t *storage) {
  return SharedLayout{storage};
}

__device__ __forceinline__ void
stage_weight_k32(std::uint8_t *stage, const std::uint8_t *weight,
                 std::uint32_t channel_base, std::uint32_t channel_count,
                 std::uint32_t row_width, std::uint32_t k_base,
                 std::uint32_t lane) {
  if (lane >= kMmaChannels) {
    return;
  }
  auto *destination =
      reinterpret_cast<uint4 *>(stage + static_cast<std::uint64_t>(lane) * 32u);
  const std::uint32_t channel = channel_base + lane;
  if (channel < channel_count) {
    auto *source = reinterpret_cast<const uint4 *>(
        weight + static_cast<std::uint64_t>(channel) * row_width + k_base);
    destination[0] = source[0];
    destination[1] = source[1];
  } else {
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    destination[0] = zero;
    destination[1] = zero;
  }
}

__device__ __forceinline__ void
stage_activation_k32(std::uint8_t *stage, const std::uint8_t *values,
                     std::uint32_t row_base, std::uint32_t active_rows,
                     std::uint32_t row_width, std::uint32_t k_base,
                     std::uint32_t lane) {
  for (std::uint32_t linear = lane; linear < kActivationStageBytes;
       linear += kWarpSize) {
    std::uint32_t k_pair = linear >> 4;
    std::uint32_t pair_byte = linear & 15u;
    std::uint32_t row = pair_byte >> 1;
    std::uint32_t byte = pair_byte & 1u;
    stage[linear] =
        row < active_rows
            ? values[static_cast<std::uint64_t>(row_base + row) * row_width +
                     k_base + k_pair * 2u + byte]
            : 0u;
  }
}

__device__ __forceinline__ void
gate_up_tile(const Args &args, const SharedLayout &shared,
             std::uint32_t row_base, std::uint32_t active_rows,
             std::uint32_t channel_block, std::uint32_t warp,
             std::uint32_t lane) {
  std::uint32_t channel_base = channel_block * kMmaChannels;
  std::uint32_t input_scale_cols = args.input_size / kArtifactBlock;
  std::uint8_t *warp_stage = shared.warp_stages + warp * kWarpStageBytes;
  std::uint8_t *weight_stage = warp_stage;
  std::uint8_t *activation_stage = warp_stage + kWeightStageBytes;
  float gate_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t scale_block = 0; scale_block < input_scale_cols;
       ++scale_block) {
    float gate_block[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_block[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (std::uint32_t k_sub = 0; k_sub < kArtifactBlock; k_sub += 32u) {
      std::uint32_t k_base = scale_block * kArtifactBlock + k_sub;
      stage_weight_k32(weight_stage, args.gate_weight_fp8, channel_base,
                       args.intermediate_size, args.input_size, k_base, lane);
      stage_activation_k32(activation_stage, args.input_fp8, row_base,
                           active_rows, args.input_size, k_base, lane);
      __syncwarp();
      std::uint32_t gate_fragment[4];
      std::uint32_t activation_fragment[2];
      load_weight_fragment(weight_stage, lane, gate_fragment);
      load_activation_fragment(activation_stage, lane, activation_fragment);
      __syncwarp();
      stage_weight_k32(weight_stage, args.up_weight_fp8, channel_base,
                       args.intermediate_size, args.input_size, k_base, lane);
      __syncwarp();
      std::uint32_t up_fragment[4];
      load_weight_fragment(weight_stage, lane, up_fragment);
      mma_fp8_e4m3(gate_block, gate_fragment, activation_fragment);
      mma_fp8_e4m3(up_block, up_fragment, activation_fragment);
      __syncwarp();
    }

    std::uint32_t weight_scale_row = channel_base / kArtifactBlock;
    float gate_weight_scale = ue8m0_to_float(
        args.gate_weight_ue8m0[weight_scale_row * input_scale_cols +
                               scale_block]);
    float up_weight_scale = ue8m0_to_float(
        args.up_weight_ue8m0[weight_scale_row * input_scale_cols +
                             scale_block]);
    std::uint32_t row_pair = lane & 3u;
#pragma unroll
    for (std::uint32_t element = 0; element < 4u; ++element) {
      std::uint32_t row = row_pair * 2u + (element & 1u);
      if (row < active_rows) {
        float activation_scale = ue8m0_to_float(
            args.input_ue8m0[(row_base + row) * input_scale_cols +
                             scale_block]);
        gate_accumulator[element] +=
            gate_block[element] * gate_weight_scale * activation_scale;
        up_accumulator[element] +=
            up_block[element] * up_weight_scale * activation_scale;
      }
    }
  }

  std::uint32_t channel_group = lane >> 2;
  std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0; element < 4u; ++element) {
    std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    std::uint32_t row = row_pair * 2u + (element & 1u);
    if (row < active_rows && channel < args.intermediate_size) {
      const float gate = bf16_round(gate_accumulator[element]);
      const float up = bf16_round(up_accumulator[element]);
      args.hidden_f32[static_cast<std::uint64_t>(row_base + row) *
                          args.intermediate_size +
                      channel] =
          bf16_round(swiglu(gate, up, args.output_scale, args.swiglu_limit));
    }
  }
}

__device__ __forceinline__ void pack_hidden_block(const Args &args,
                                                  std::uint32_t row,
                                                  std::uint32_t scale_block,
                                                  std::uint32_t lane) {
  const std::uint64_t start =
      static_cast<std::uint64_t>(row) * args.intermediate_size +
      scale_block * kArtifactBlock;
  float values[4];
  float amax = 1.0e-4f;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const float value = args.hidden_f32[start + lane + element * kWarpSize];
    values[element] = value;
    amax = fmaxf(amax, fabsf(value));
  }
#pragma unroll
  for (std::uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta));
  }
  std::uint32_t scale_byte = scale_byte_for_amax(amax);
  scale_byte = __shfl_sync(0xffffffffu, scale_byte, 0);
  if (lane == 0u) {
    args.hidden_ue8m0[row * (args.intermediate_size / kArtifactBlock) +
                      scale_block] = static_cast<std::uint8_t>(scale_byte);
  }
  const float scale = ue8m0_to_float(static_cast<std::uint8_t>(scale_byte));
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const float value = values[element] / scale;
    args.hidden_fp8[start + lane + element * kWarpSize] =
        quantize_fp8_e4m3fn_byte(clamp_value(value, -448.0f, 448.0f));
  }
}

__device__ __forceinline__ void
down_tile(const Args &args, const SharedLayout &shared, std::uint32_t row_base,
          std::uint32_t active_rows, std::uint32_t channel_block,
          std::uint32_t warp, std::uint32_t lane) {
  std::uint32_t channel_base = channel_block * kMmaChannels;
  std::uint32_t scale_cols = args.intermediate_size / kArtifactBlock;
  std::uint8_t *warp_stage = shared.warp_stages + warp * kWarpStageBytes;
  std::uint8_t *weight_stage = warp_stage;
  std::uint8_t *activation_stage = warp_stage + kWeightStageBytes;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t scale_block = 0; scale_block < scale_cols; ++scale_block) {
    float block_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (std::uint32_t k_sub = 0; k_sub < kArtifactBlock; k_sub += 32u) {
      std::uint32_t k_base = scale_block * kArtifactBlock + k_sub;
      stage_weight_k32(weight_stage, args.down_weight_fp8, channel_base,
                       args.output_size, args.intermediate_size, k_base, lane);
      stage_activation_k32(activation_stage, args.hidden_fp8, row_base,
                           active_rows, args.intermediate_size, k_base, lane);
      __syncwarp();
      std::uint32_t weight_fragment[4];
      std::uint32_t activation_fragment[2];
      load_weight_fragment(weight_stage, lane, weight_fragment);
      load_activation_fragment(activation_stage, lane, activation_fragment);
      mma_fp8_e4m3(block_accumulator, weight_fragment, activation_fragment);
      __syncwarp();
    }

    float weight_scale = ue8m0_to_float(
        args.down_weight_ue8m0[(channel_base / kArtifactBlock) * scale_cols +
                               scale_block]);
    std::uint32_t row_pair = lane & 3u;
#pragma unroll
    for (std::uint32_t element = 0; element < 4u; ++element) {
      std::uint32_t row = row_pair * 2u + (element & 1u);
      if (row < active_rows) {
        float activation_scale = ue8m0_to_float(
            args.hidden_ue8m0[(row_base + row) * scale_cols + scale_block]);
        accumulator[element] +=
            block_accumulator[element] * weight_scale * activation_scale;
      }
    }
  }

  std::uint32_t channel_group = lane >> 2;
  std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0; element < 4u; ++element) {
    std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    std::uint32_t row = row_pair * 2u + (element & 1u);
    if (row < active_rows && channel < args.output_size) {
      std::uint64_t output_index =
          static_cast<std::uint64_t>(row_base + row) * args.output_size +
          channel;
      const float down = bf16_round(accumulator[element]);
      if ((args.flags & kAccumulateOutput) != 0u) {
        args.output_f32[output_index] =
            bf16_round(args.output_f32[output_index] + down);
      } else {
        args.output_f32[output_index] = down;
      }
    }
  }
}

__global__ __launch_bounds__(kThreads, 1) void kernel(Args args) {
  extern __shared__ __align__(16) std::uint8_t storage[];
  SharedLayout shared = make_shared_layout(storage);
  std::uint32_t warp = threadIdx.x / kWarpSize;
  std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  std::uint32_t global_warp = blockIdx.x * kWarpCount + warp;
  std::uint32_t warp_stride = gridDim.x * kWarpCount;
  std::uint32_t row_tiles = (args.rows + kMmaRows - 1u) / kMmaRows;
  std::uint32_t intermediate_channel_blocks =
      (args.intermediate_size + kMmaChannels - 1u) / kMmaChannels;
  std::uint32_t gate_tasks = row_tiles * intermediate_channel_blocks;

  for (std::uint32_t task = global_warp; task < gate_tasks;
       task += warp_stride) {
    std::uint32_t row_tile = task / intermediate_channel_blocks;
    std::uint32_t channel_block = task - row_tile * intermediate_channel_blocks;
    std::uint32_t row_base = row_tile * kMmaRows;
    std::uint32_t active_rows = min(kMmaRows, args.rows - row_base);
    gate_up_tile(args, shared, row_base, active_rows, channel_block, warp,
                 lane);
  }

  cooperative_groups::this_grid().sync();

  std::uint32_t hidden_scale_cols = args.intermediate_size / kArtifactBlock;
  std::uint32_t pack_tasks = args.rows * hidden_scale_cols;
  for (std::uint32_t task = global_warp; task < pack_tasks;
       task += warp_stride) {
    std::uint32_t row = task / hidden_scale_cols;
    std::uint32_t scale_block = task - row * hidden_scale_cols;
    pack_hidden_block(args, row, scale_block, lane);
  }

  cooperative_groups::this_grid().sync();

  std::uint32_t output_channel_blocks =
      (args.output_size + kMmaChannels - 1u) / kMmaChannels;
  std::uint32_t down_tasks = row_tiles * output_channel_blocks;
  for (std::uint32_t task = global_warp; task < down_tasks;
       task += warp_stride) {
    std::uint32_t row_tile = task / output_channel_blocks;
    std::uint32_t channel_block = task - row_tile * output_channel_blocks;
    std::uint32_t row_base = row_tile * kMmaRows;
    std::uint32_t active_rows = min(kMmaRows, args.rows - row_base);
    down_tile(args, shared, row_base, active_rows, channel_block, warp, lane);
  }
}

} // namespace detail

inline std::size_t dynamic_shared_memory_bytes() {
  return static_cast<std::size_t>(kWarpCount) * detail::kWarpStageBytes;
}

inline ValidationResult validate(const Args &args) {
  if (args.input_fp8 == nullptr || args.input_ue8m0 == nullptr ||
      args.gate_weight_fp8 == nullptr || args.gate_weight_ue8m0 == nullptr ||
      args.up_weight_fp8 == nullptr || args.up_weight_ue8m0 == nullptr ||
      args.down_weight_fp8 == nullptr || args.down_weight_ue8m0 == nullptr ||
      args.hidden_f32 == nullptr || args.hidden_fp8 == nullptr ||
      args.hidden_ue8m0 == nullptr || args.output_f32 == nullptr) {
    return ValidationResult::kNullPointer;
  }
  if (!detail::aligned_16(args.input_fp8) ||
      !detail::aligned_16(args.input_ue8m0) ||
      !detail::aligned_16(args.gate_weight_fp8) ||
      !detail::aligned_16(args.gate_weight_ue8m0) ||
      !detail::aligned_16(args.up_weight_fp8) ||
      !detail::aligned_16(args.up_weight_ue8m0) ||
      !detail::aligned_16(args.down_weight_fp8) ||
      !detail::aligned_16(args.down_weight_ue8m0) ||
      !detail::aligned_16(args.hidden_f32) ||
      !detail::aligned_16(args.hidden_fp8) ||
      !detail::aligned_16(args.hidden_ue8m0) ||
      !detail::aligned_16(args.output_f32)) {
    return ValidationResult::kMisalignedPointer;
  }
  if (args.rows == 0u || args.input_size == 0u ||
      args.intermediate_size == 0u || args.output_size == 0u ||
      (args.input_size % kArtifactBlock) != 0u ||
      (args.intermediate_size % kArtifactBlock) != 0u ||
      (args.output_size % kMmaChannels) != 0u) {
    return ValidationResult::kUnsupportedShape;
  }
  if (args.gate_block_m != kArtifactBlock ||
      args.gate_block_k != kArtifactBlock ||
      args.up_block_m != kArtifactBlock || args.up_block_k != kArtifactBlock ||
      args.down_block_m != kArtifactBlock ||
      args.down_block_k != kArtifactBlock) {
    return ValidationResult::kUnsupportedArtifactBlocks;
  }
  if (!detail::finite_scalar(args.output_scale) ||
      !detail::finite_scalar(args.swiglu_limit)) {
    return ValidationResult::kInvalidScalar;
  }
  if ((args.flags & ~kKnownFlags) != 0u) {
    return ValidationResult::kInvalidFlags;
  }
  return ValidationResult::kSuccess;
}

inline cudaError_t launch(const Args &args, cudaStream_t stream) {
  if (validate(args) != ValidationResult::kSuccess) {
    return cudaErrorInvalidValue;
  }
  std::uint32_t row_tiles = (args.rows + kMmaRows - 1u) / kMmaRows;
  std::uint32_t gate_tasks =
      row_tiles * ((args.intermediate_size + kMmaChannels - 1u) / kMmaChannels);
  std::uint32_t down_tasks =
      row_tiles * ((args.output_size + kMmaChannels - 1u) / kMmaChannels);
  std::uint32_t warp_tasks = max(gate_tasks, down_tasks);
  std::uint32_t blocks =
      min(kCooperativeBlockLimit, (warp_tasks + kWarpCount - 1u) / kWarpCount);
  void *kernel_args[] = {const_cast<Args *>(&args)};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::kernel), dim3(blocks, 1, 1),
      dim3(kThreads, 1, 1), kernel_args, dynamic_shared_memory_bytes(), stream);
}

} // namespace ferrule::cuda::cutlass::operators::shared_ffn


#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE

#if !defined(__CUDACC__)
#error "SM103 grouped FP4 MoE support must be compiled with nvcc"
#endif


#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace ferrule::cuda::cutlass::operators::grouped_fp4_moe {

using namespace ferrule::cuda::cutlass::architectures::sm103::grouped_fp4_moe;

inline constexpr std::uint32_t kPrepareThreads = 128;
inline constexpr std::uint32_t kQuantThreads = 128;
inline constexpr std::uint32_t kScatterThreads = 256;
// The 1SM and 2SM reference kernels use M tiles of 128 and 256 respectively.
// Their midpoint is a neutral default; providers may tune this crossover.
inline constexpr std::uint32_t kDefault2SmMinRows = 192;
inline constexpr std::size_t kWorkspaceAlignment = kCutlassWorkspaceAlignment;

// Backend-private POD. All addresses refer to device-accessible memory. Ferrule
// core is the trusted producer of compact device metadata and keeps slot
// generations and bindings stable through the asynchronous launch. Active
// groups are ordered by M: groups [0, small_group_count) have
// route_counts[g] < options.two_sm_min_rows and the remainder have
// route_counts[g] >= options.two_sm_min_rows. Every active group has M > 0.
struct GroupedFp4MoeArgs {
  std::uint32_t active_group_count{};
  std::uint32_t small_group_count{};
  std::uint32_t slot_capacity{};
  std::uint32_t max_group_rows{};
  std::uint32_t total_routed_rows{};
  std::uint32_t num_tokens{};
  std::uint32_t num_routes{};
  std::uint32_t input_size{};
  std::uint32_t intermediate_size{};
  std::uint32_t hidden_size{};
  float swiglu_limit{};

  // int32[active_group_count] and int32[active_group_count].
  std::uint64_t active_expert_slots{};
  std::uint64_t active_group_generations{};
  // uint32[active_group_count + 1] and uint32[active_group_count].
  std::uint64_t expert_route_indptr{};
  std::uint64_t expert_route_counts{};
  // Expert-contiguous arrays of length total_routed_rows.
  std::uint64_t route_token_indices{}; // int32
  std::uint64_t route_indices{};       // int32
  std::uint64_t route_weights{};       // float

  // Slot-indexed arrays. Scale pointers address prepared CUTLASS SFB layouts.
  std::uint64_t slot_generations{}; // int32[slot_capacity]
  std::uint64_t gate_ptrs{};        // uint64[slot_capacity]
  std::uint64_t gate_scale_ptrs{};  // uint64[slot_capacity]
  std::uint64_t up_ptrs{};          // uint64[slot_capacity]
  std::uint64_t up_scale_ptrs{};    // uint64[slot_capacity]
  std::uint64_t down_ptrs{};        // uint64[slot_capacity]
  std::uint64_t down_scale_ptrs{};  // uint64[slot_capacity]

  // Full-token FP8 activation: E4M3 [num_tokens,input_size], UE8M0 scales
  // [num_tokens,input_size/128].
  std::uint64_t input_fp8{};
  std::uint64_t input_ue8m0{};

  // Route-major output and status arrays.
  std::uint64_t route_output{};  // float[num_routes,hidden_size]
  std::uint64_t route_written{}; // int32[num_routes]
  std::uint64_t route_error{};   // int32[1]
};

struct LaunchOptions {
  std::int32_t device_id{-1};
  std::int32_t sm_count{};
  std::uint32_t two_sm_min_rows{kDefault2SmMinRows};
};

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidArgument = 1,
  kUnsupportedResources = 2,
  kLaunchFailed = 3,
};

struct WorkspacePlan {
  std::int32_t descriptor_capacity{};
  std::size_t descriptor_offset{};
  std::size_t descriptor_bytes{};
  std::size_t group_bindings_offset{};
  std::size_t group_bindings_bytes{};
  std::size_t route_groups_offset{};
  std::size_t route_groups_bytes{};
  std::size_t gathered_x_offset{};
  std::size_t gathered_x_bytes{};
  std::size_t gathered_x_sfa_offset{};
  std::size_t gathered_x_sfa_bytes{};
  std::size_t gathered_x_sfa_group_stride{};
  std::size_t gate_up_bf16_offset{};
  std::size_t gate_up_bf16_bytes{};
  std::size_t hidden_fp8_offset{};
  std::size_t hidden_fp8_bytes{};
  std::size_t hidden_sfa_offset{};
  std::size_t hidden_sfa_bytes{};
  std::size_t hidden_sfa_group_stride{};
  std::size_t down_bf16_offset{};
  std::size_t down_bf16_bytes{};
  std::size_t cutlass_offset{};
  std::size_t cutlass_bytes{};
  std::size_t total_bytes{};

  bool valid() const noexcept { return total_bytes != 0; }
};

static_assert(std::is_standard_layout<GroupedFp4MoeArgs>::value,
              "GroupedFp4MoeArgs POD");
static_assert(std::is_trivially_copyable<GroupedFp4MoeArgs>::value,
              "GroupedFp4MoeArgs must be trivially copyable");
static_assert(sizeof(GroupedFp4MoeArgs) == 200u);
static_assert(alignof(GroupedFp4MoeArgs) == 8u);
static_assert(offsetof(GroupedFp4MoeArgs, active_group_count) == 0u);
static_assert(offsetof(GroupedFp4MoeArgs, swiglu_limit) == 40u);
static_assert(offsetof(GroupedFp4MoeArgs, active_expert_slots) == 48u);
static_assert(offsetof(GroupedFp4MoeArgs, route_error) == 192u);
static_assert(std::is_standard_layout<LaunchOptions>::value,
              "LaunchOptions POD");
static_assert(std::is_trivially_copyable<WorkspacePlan>::value,
              "WorkspacePlan must be trivially copyable");

namespace moe_detail {

constexpr bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

constexpr bool aligned_address(std::uint64_t address,
                               std::size_t alignment) noexcept {
  return address != 0 && is_power_of_two(alignment) &&
         (address & (alignment - 1)) == 0;
}

inline bool aligned_pointer(void const *pointer,
                            std::size_t alignment) noexcept {
  return pointer != nullptr && is_power_of_two(alignment) &&
         (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

inline bool checked_add(std::size_t &result, std::size_t left,
                        std::size_t right) noexcept {
  if (left > (std::numeric_limits<std::size_t>::max)() - right) {
    return false;
  }
  result = left + right;
  return true;
}

inline bool checked_product(std::size_t &result, std::size_t left,
                            std::size_t right) noexcept {
  if (left != 0 && right > (std::numeric_limits<std::size_t>::max)() / left) {
    return false;
  }
  result = left * right;
  return true;
}

inline bool checked_product(std::size_t &result, std::size_t first,
                            std::size_t second, std::size_t third) noexcept {
  std::size_t partial = 0;
  return checked_product(partial, first, second) &&
         checked_product(result, partial, third);
}

inline bool checked_product(std::size_t &result, std::size_t first,
                            std::size_t second, std::size_t third,
                            std::size_t fourth) noexcept {
  std::size_t partial = 0;
  return checked_product(partial, first, second, third) &&
         checked_product(result, partial, fourth);
}

inline bool append_region(std::size_t &cursor, std::size_t bytes,
                          std::size_t alignment, std::size_t &offset) noexcept {
  if (!is_power_of_two(alignment) ||
      cursor > (std::numeric_limits<std::size_t>::max)() - (alignment - 1)) {
    return false;
  }
  cursor = detail::align_up(cursor, alignment);
  offset = cursor;
  return checked_add(cursor, cursor, bytes);
}

inline bool scalar_args_valid(GroupedFp4MoeArgs const &args,
                              LaunchOptions const &options) noexcept {
  return args.active_group_count != 0 &&
         args.active_group_count <= 0x3fffffffu &&
         args.small_group_count <= args.active_group_count &&
         args.slot_capacity != 0 && args.max_group_rows != 0 &&
         args.max_group_rows <= 0x7fffffffu && args.total_routed_rows != 0 &&
         args.total_routed_rows <= args.num_routes &&
         static_cast<std::uint64_t>(args.total_routed_rows) >=
             args.active_group_count &&
         static_cast<std::uint64_t>(args.total_routed_rows) <=
             static_cast<std::uint64_t>(args.active_group_count) *
                 args.max_group_rows &&
         args.num_tokens != 0 && args.num_routes != 0 && args.input_size != 0 &&
         args.input_size <= 0x7fffffffu && args.intermediate_size != 0 &&
         args.intermediate_size <= 0x7fffffffu && args.hidden_size != 0 &&
         args.hidden_size <= 0x7fffffffu && (args.input_size % 128u) == 0 &&
         (args.intermediate_size % 128u) == 0 &&
         (args.hidden_size % 64u) == 0 && std::isfinite(args.swiglu_limit) &&
         options.device_id >= 0 && options.sm_count > 0 &&
         options.two_sm_min_rows != 0 &&
         (args.small_group_count == args.active_group_count ||
          options.two_sm_min_rows <= args.max_group_rows);
}

inline bool pointer_args_valid(GroupedFp4MoeArgs const &args) noexcept {
  return aligned_address(args.active_expert_slots, alignof(std::int32_t)) &&
         aligned_address(args.active_group_generations,
                         alignof(std::int32_t)) &&
         aligned_address(args.expert_route_indptr, alignof(std::uint32_t)) &&
         aligned_address(args.expert_route_counts, alignof(std::uint32_t)) &&
         aligned_address(args.route_token_indices, alignof(std::int32_t)) &&
         aligned_address(args.route_indices, alignof(std::int32_t)) &&
         aligned_address(args.route_weights, alignof(float)) &&
         aligned_address(args.slot_generations, alignof(std::int32_t)) &&
         aligned_address(args.gate_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.gate_scale_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.up_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.up_scale_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.down_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.down_scale_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.input_fp8, 16) &&
         aligned_address(args.input_ue8m0, 16) &&
         aligned_address(args.route_output, 16) &&
         aligned_address(args.route_written, alignof(std::int32_t)) &&
         aligned_address(args.route_error, alignof(std::int32_t));
}

// These are real C++ objects, not fabricated addresses. CUTLASS workspace
// queries only consume num_groups; the non-null sentinels satisfy the grouped
// argument shape without pointer arithmetic or dereferencing device metadata.
struct QuerySentinels {
  ProblemShapeValue problem_shape{};
  ElementA const *a{};
  ElementB const *b{};
  ElementD *d{};
  ElementScale const *sfa{};
  ElementScale const *sfb{};
  StrideA stride_a{};
  StrideB stride_b{};
  StrideD stride_d{};
  LayoutSFA layout_sfa{};
  LayoutSFB layout_sfb{};
};

inline QuerySentinels query_sentinels{};

inline DeviceDescriptorView
query_descriptor_view(std::int32_t groups) noexcept {
  DeviceDescriptorView view{};
  view.groups = groups;
  view.problem_shapes = &query_sentinels.problem_shape;
  view.a = &query_sentinels.a;
  view.b = &query_sentinels.b;
  view.d = &query_sentinels.d;
  view.sfa = &query_sentinels.sfa;
  view.sfb = &query_sentinels.sfb;
  view.stride_a = &query_sentinels.stride_a;
  view.stride_b = &query_sentinels.stride_b;
  view.stride_d = &query_sentinels.stride_d;
  view.layout_sfa = &query_sentinels.layout_sfa;
  view.layout_sfb = &query_sentinels.layout_sfb;
  return view;
}

inline std::size_t cutlass_bytes_for(MmaMode mode, std::uint32_t groups,
                                     LaunchOptions const &options) noexcept {
  if (groups == 0 || groups > 0x7fffffffu) {
    return 0;
  }
  GroupedProblem problem{};
  problem.descriptors =
      query_descriptor_view(static_cast<std::int32_t>(groups));
  problem.device_id = options.device_id;
  problem.sm_count = options.sm_count;
  return cutlass_workspace_bytes(mode, problem);
}

inline DeviceDescriptorView slice_descriptors(DeviceDescriptorView view,
                                              std::int32_t offset,
                                              std::int32_t groups) noexcept {
  view.groups = groups;
  view.problem_shapes += offset;
  view.a += offset;
  view.b += offset;
  view.d += offset;
  view.sfa += offset;
  view.sfb += offset;
  view.stride_a += offset;
  view.stride_b += offset;
  view.stride_d += offset;
  view.layout_sfa += offset;
  view.layout_sfb += offset;
  return view;
}

template <class T>
__host__ __device__ __forceinline__ T *device_pointer(std::uint64_t address) {
  return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address));
}

__device__ __forceinline__ void set_route_error(GroupedFp4MoeArgs const &args) {
  atomicOr(reinterpret_cast<unsigned int *>(
               device_pointer<std::int32_t>(args.route_error)),
           1u);
}

struct WorkspaceView {
  DeviceDescriptorView descriptors{};
  std::uint64_t *group_bindings{};
  std::uint32_t *route_groups{};
  std::uint8_t *gathered_x{};
  ElementScale *gathered_x_sfa{};
  ElementD *gate_up_bf16{};
  std::uint8_t *hidden_fp8{};
  ElementScale *hidden_sfa{};
  ElementD *down_bf16{};
  std::size_t gathered_x_sfa_group_stride{};
  std::size_t hidden_sfa_group_stride{};
};

static_assert(std::is_trivially_copyable<WorkspaceView>::value,
              "WorkspaceView must be trivially copyable");

__device__ __forceinline__ void
write_descriptor(DeviceDescriptorView const &descriptors, std::int32_t index,
                 std::int32_t m, std::int32_t n, std::int32_t k,
                 ElementA const *a, ElementB const *b, ElementD *d,
                 ElementScale const *sfa, ElementScale const *sfb) {
  descriptors.problem_shapes[index] = ProblemShapeValue{m, n, k};
  descriptors.a[index] = a;
  descriptors.b[index] = b;
  descriptors.d[index] = d;
  descriptors.sfa[index] = sfa;
  descriptors.sfb[index] = sfb;
  descriptors.stride_a[index] =
      ::cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  descriptors.stride_b[index] =
      ::cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  descriptors.stride_d[index] =
      ::cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  descriptors.layout_sfa[index] =
      BlockScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  descriptors.layout_sfb[index] =
      BlockScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
}

__global__ __launch_bounds__(kPrepareThreads) void prepare_gate_up_kernel(
    GroupedFp4MoeArgs args, LaunchOptions options, WorkspaceView workspace) {
  const std::uint32_t group = blockIdx.x;
  if (group >= args.active_group_count) {
    return;
  }

  __shared__ std::int32_t metadata_valid;
  __shared__ std::int32_t expert_slot;
  __shared__ std::uint32_t route_begin;
  __shared__ std::uint32_t route_count;
  __shared__ std::uint64_t bindings[6];

  const auto *slots =
      device_pointer<std::int32_t const>(args.active_expert_slots);
  const auto *active_generations =
      device_pointer<std::int32_t const>(args.active_group_generations);
  const auto *slot_generations =
      device_pointer<std::int32_t const>(args.slot_generations);
  const auto *indptr =
      device_pointer<std::uint32_t const>(args.expert_route_indptr);
  const auto *counts =
      device_pointer<std::uint32_t const>(args.expert_route_counts);
  if (threadIdx.x == 0) {
    route_begin = indptr[group];
    const std::uint32_t route_end = indptr[group + 1];
    route_count = counts[group];
    expert_slot = slots[group];
    const bool expected_small = group < args.small_group_count;
    const bool actual_small = route_count < options.two_sm_min_rows;
    const bool slot_valid =
        expert_slot >= 0 &&
        static_cast<std::uint32_t>(expert_slot) < args.slot_capacity;
    const bool generation_valid =
        slot_valid &&
        active_generations[group] == slot_generations[expert_slot];
    metadata_valid = route_count != 0 && route_count <= args.max_group_rows &&
                             route_begin <= route_end &&
                             route_end <= args.total_routed_rows &&
                             route_end - route_begin == route_count &&
                             (group != 0 || route_begin == 0) &&
                             (group + 1 != args.active_group_count ||
                              route_end == args.total_routed_rows) &&
                             expected_small == actual_small && generation_valid
                         ? 1
                         : 0;
  }
  __syncthreads();

  const auto *tokens =
      device_pointer<std::int32_t const>(args.route_token_indices);
  const auto *routes = device_pointer<std::int32_t const>(args.route_indices);
  const auto *weights = device_pointer<float const>(args.route_weights);
  if (metadata_valid != 0) {
    for (std::uint32_t row = threadIdx.x; row < route_count;
         row += blockDim.x) {
      const std::uint32_t ordinal = route_begin + row;
      const std::int32_t token = tokens[ordinal];
      const std::int32_t route = routes[ordinal];
      if (token < 0 || static_cast<std::uint32_t>(token) >= args.num_tokens ||
          route < 0 || static_cast<std::uint32_t>(route) >= args.num_routes ||
          !isfinite(weights[ordinal])) {
        atomicExch(&metadata_valid, 0);
      }
    }
  }
  __syncthreads();

  // Invalid device metadata violates the trusted-producer contract. Trap the
  // stream before CUTLASS can observe an invalid descriptor; in particular,
  // this path never fabricates an M=0 grouped problem.
  if (metadata_valid == 0) {
    if (threadIdx.x == 0) {
      set_route_error(args);
    }
    __syncthreads();
    __trap();
    return;
  }

  if (threadIdx.x == 0) {
    const std::uint32_t expert = static_cast<std::uint32_t>(expert_slot);
    const auto *gate_ptrs = device_pointer<std::uint64_t const>(args.gate_ptrs);
    const auto *gate_scale_ptrs =
        device_pointer<std::uint64_t const>(args.gate_scale_ptrs);
    const auto *up_ptrs = device_pointer<std::uint64_t const>(args.up_ptrs);
    const auto *up_scale_ptrs =
        device_pointer<std::uint64_t const>(args.up_scale_ptrs);
    const auto *down_ptrs = device_pointer<std::uint64_t const>(args.down_ptrs);
    const auto *down_scale_ptrs =
        device_pointer<std::uint64_t const>(args.down_scale_ptrs);
    bindings[0] = gate_ptrs[expert];
    bindings[1] = gate_scale_ptrs[expert];
    bindings[2] = up_ptrs[expert];
    bindings[3] = up_scale_ptrs[expert];
    bindings[4] = down_ptrs[expert];
    bindings[5] = down_scale_ptrs[expert];

    const std::size_t binding_base = static_cast<std::size_t>(group) * 6;
#pragma unroll
    for (int pointer = 0; pointer < 6; ++pointer) {
      workspace
          .group_bindings[binding_base + static_cast<std::size_t>(pointer)] =
          bindings[pointer];
    }
  }
  __syncthreads();

  const std::size_t input_row_bytes = args.input_size;
  const std::uint32_t input_sfa_columns = args.input_size / kScaleVectorSize;
  const std::uint32_t input_scale_columns = args.input_size / 128;
  auto *group_sfa_bytes =
      reinterpret_cast<std::uint8_t *>(workspace.gathered_x_sfa) +
      static_cast<std::size_t>(group) * workspace.gathered_x_sfa_group_stride;
  const auto *input = device_pointer<std::uint8_t const>(args.input_fp8);
  const auto *input_scales =
      device_pointer<std::uint8_t const>(args.input_ue8m0);
  const std::size_t input_vector_count =
      static_cast<std::size_t>(route_count) * (args.input_size / 16);
  for (std::size_t vector = threadIdx.x; vector < input_vector_count;
       vector += blockDim.x) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(vector / (args.input_size / 16));
    const std::uint32_t vector_column =
        static_cast<std::uint32_t>(vector % (args.input_size / 16));
    const std::uint32_t ordinal = route_begin + row;
    const std::int32_t token = tokens[ordinal];
    const auto *source = reinterpret_cast<uint4 const *>(
        input + static_cast<std::size_t>(token) * input_row_bytes +
        static_cast<std::size_t>(vector_column) * 16);
    auto *destination = reinterpret_cast<uint4 *>(
        workspace.gathered_x +
        static_cast<std::size_t>(ordinal) * input_row_bytes +
        static_cast<std::size_t>(vector_column) * 16);
    *destination = *source;
  }

  const auto sfa_layout = BlockScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(static_cast<int>(route_count), 1,
                       static_cast<int>(args.input_size), 1));
  const std::size_t sfa_count =
      static_cast<std::size_t>(route_count) * input_sfa_columns;
  for (std::size_t scale = threadIdx.x; scale < sfa_count;
       scale += blockDim.x) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(scale / input_sfa_columns);
    const std::uint32_t scale_column =
        static_cast<std::uint32_t>(scale % input_sfa_columns);
    const std::uint32_t ordinal = route_begin + row;
    const std::int32_t token = tokens[ordinal];
    const auto destination = sfa_layout(
        cute::make_coord(static_cast<int>(row),
                         static_cast<int>(scale_column * kScaleVectorSize), 0));
    group_sfa_bytes[static_cast<std::size_t>(destination)] =
        input_scales[static_cast<std::size_t>(token) * input_scale_columns +
                     scale_column / 4];
    workspace.route_groups[ordinal] = group;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const std::int32_t m = static_cast<std::int32_t>(route_count);
    auto *a_bytes = workspace.gathered_x +
                    static_cast<std::size_t>(route_begin) * input_row_bytes;
    auto *a = reinterpret_cast<ElementA const *>(a_bytes);
    auto *sfa = reinterpret_cast<ElementScale const *>(group_sfa_bytes);
    auto *gate_d =
        workspace.gate_up_bf16 +
        static_cast<std::size_t>(route_begin) * args.intermediate_size;
    auto *up_d =
        workspace.gate_up_bf16 +
        (static_cast<std::size_t>(args.total_routed_rows) + route_begin) *
            args.intermediate_size;
    const auto *gate_b = reinterpret_cast<ElementB const *>(
        static_cast<std::uintptr_t>(bindings[0]));
    const auto *gate_sfb = reinterpret_cast<ElementScale const *>(
        static_cast<std::uintptr_t>(bindings[1]));
    const auto *up_b = reinterpret_cast<ElementB const *>(
        static_cast<std::uintptr_t>(bindings[2]));
    const auto *up_sfb = reinterpret_cast<ElementScale const *>(
        static_cast<std::uintptr_t>(bindings[3]));

    const std::uint32_t bucket_groups =
        group < args.small_group_count
            ? args.small_group_count
            : args.active_group_count - args.small_group_count;
    const std::uint32_t local_group =
        group < args.small_group_count ? group : group - args.small_group_count;
    const std::uint32_t bucket_base =
        group < args.small_group_count ? 0 : args.small_group_count * 2;
    const std::int32_t gate_index =
        static_cast<std::int32_t>(bucket_base + local_group);
    const std::int32_t up_index =
        static_cast<std::int32_t>(bucket_base + bucket_groups + local_group);
    write_descriptor(workspace.descriptors, gate_index, m,
                     static_cast<std::int32_t>(args.intermediate_size),
                     static_cast<std::int32_t>(args.input_size), a, gate_b,
                     gate_d, sfa, gate_sfb);
    write_descriptor(workspace.descriptors, up_index, m,
                     static_cast<std::int32_t>(args.intermediate_size),
                     static_cast<std::int32_t>(args.input_size), a, up_b, up_d,
                     sfa, up_sfb);
  }
}

__device__ __forceinline__ float e8m0_value(std::uint8_t encoded) {
  const std::uint32_t bits =
      encoded == 0 ? (1u << 22) : (static_cast<std::uint32_t>(encoded) << 23);
  return __uint_as_float(bits);
}

__device__ __forceinline__ std::uint8_t e8m0_for_fp8_amax(float amax) {
  if (!isfinite(amax) || amax <= 0.0f) {
    return kScalePadding;
  }
  const float exponent_value = ceilf(log2f(fmaxf(amax, 1.0e-4f) / 448.0f));
  if (!isfinite(exponent_value) || exponent_value < -127.0f) {
    return 0;
  }
  const int encoded = static_cast<int>(exponent_value) + 127;
  return static_cast<std::uint8_t>(
      encoded < 0 ? 0 : (encoded > 255 ? 255 : encoded));
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

__device__ __forceinline__ std::uint8_t quantize_fp8(float value) {
  value = fminf(fmaxf(value, -448.0f), 448.0f);
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float swiglu(float gate, float up, float limit) {
  if (limit > 0.0f) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
  }
  return gate * (1.0f / (1.0f + __expf(-gate))) * up;
}

__global__ __launch_bounds__(kQuantThreads) void swiglu_requant_kernel(
    GroupedFp4MoeArgs args, WorkspaceView workspace, float *debug_down_input,
    std::uint32_t debug_first_row, std::uint32_t debug_second_row) {
  const std::size_t scale_columns = args.intermediate_size / 128;
  const std::size_t row_scale = blockIdx.x;
  const std::size_t routed_row = row_scale / scale_columns;
  if (routed_row >= args.total_routed_rows) {
    return;
  }
  const std::uint32_t scale_column =
      static_cast<std::uint32_t>(row_scale % scale_columns);
  const std::uint32_t group = workspace.route_groups[routed_row];
  const auto *indptr =
      device_pointer<std::uint32_t const>(args.expert_route_indptr);
  const auto *counts =
      device_pointer<std::uint32_t const>(args.expert_route_counts);
  const std::uint32_t route_begin = indptr[group];
  const std::uint32_t route_count = counts[group];
  const std::uint32_t row =
      static_cast<std::uint32_t>(routed_row) - route_begin;
  const std::uint32_t channel = scale_column * 128 + threadIdx.x;
  const std::size_t value_index = routed_row * args.intermediate_size + channel;
  const std::size_t up_base =
      static_cast<std::size_t>(args.total_routed_rows) * args.intermediate_size;
  const auto *gate_up = workspace.gate_up_bf16;
  const float route_weight =
      device_pointer<float const>(args.route_weights)[routed_row];
  const float gate = static_cast<float>(gate_up[value_index]);
  const float up = static_cast<float>(gate_up[up_base + value_index]);
  float value = swiglu(gate, up, args.swiglu_limit) * route_weight;
  if (!isfinite(value)) {
    value = 0.0f;
    set_route_error(args);
  }
  value = bf16_to_f32(f32_to_bf16_rne(value));
  if (debug_down_input != nullptr &&
      (routed_row == debug_first_row || routed_row == debug_second_row)) {
    const std::size_t debug_row = routed_row == debug_first_row ? 0u : 1u;
    debug_down_input[debug_row * args.intermediate_size + channel] = value;
  }

  __shared__ float warp_amax[4];
  __shared__ std::uint32_t scale_byte;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5;
  float amax = fabsf(value);
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta));
  }
  if (lane == 0) {
    warp_amax[warp] = amax;
  }
  __syncthreads();
  if (warp == 0) {
    float block_amax = lane < 4 ? warp_amax[lane] : 0.0f;
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      block_amax =
          fmaxf(block_amax, __shfl_down_sync(0xffffffffu, block_amax, delta));
    }
    if (lane == 0) {
      scale_byte = e8m0_for_fp8_amax(block_amax);
    }
  }
  __syncthreads();

  const float reciprocal_scale =
      1.0f / e8m0_value(static_cast<std::uint8_t>(scale_byte));
  workspace.hidden_fp8[value_index] = quantize_fp8(value * reciprocal_scale);
  if (threadIdx.x == 0) {
    auto *sfa_group =
        reinterpret_cast<std::uint8_t *>(workspace.hidden_sfa) +
        static_cast<std::size_t>(group) * workspace.hidden_sfa_group_stride;
    const auto layout =
        BlockScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(
            static_cast<int>(route_count), static_cast<int>(args.hidden_size),
            static_cast<int>(args.intermediate_size), 1));
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      const auto destination = layout(cute::make_coord(
          static_cast<int>(row),
          static_cast<int>(scale_column * 128 + sub * kScaleVectorSize), 0));
      sfa_group[static_cast<std::size_t>(destination)] =
          static_cast<std::uint8_t>(scale_byte);
    }
  }
}

__global__ __launch_bounds__(kPrepareThreads) void prepare_down_kernel(
    GroupedFp4MoeArgs args, WorkspaceView workspace) {
  const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= args.active_group_count) {
    return;
  }

  const auto *indptr =
      device_pointer<std::uint32_t const>(args.expert_route_indptr);
  const auto *counts =
      device_pointer<std::uint32_t const>(args.expert_route_counts);
  const std::uint32_t route_begin = indptr[group];
  const std::uint32_t route_count = counts[group];
  const std::int32_t m = static_cast<std::int32_t>(route_count);
  const std::size_t hidden_row_bytes = args.intermediate_size;
  auto *a_bytes = workspace.hidden_fp8 +
                  static_cast<std::size_t>(route_begin) * hidden_row_bytes;
  auto *sfa_bytes =
      reinterpret_cast<std::uint8_t *>(workspace.hidden_sfa) +
      static_cast<std::size_t>(group) * workspace.hidden_sfa_group_stride;
  auto *a = reinterpret_cast<ElementA const *>(a_bytes);
  auto *sfa = reinterpret_cast<ElementScale const *>(sfa_bytes);
  auto *d = workspace.down_bf16 +
            static_cast<std::size_t>(route_begin) * args.hidden_size;
  const std::size_t binding_base = static_cast<std::size_t>(group) * 6;
  const auto *b = reinterpret_cast<ElementB const *>(
      static_cast<std::uintptr_t>(workspace.group_bindings[binding_base + 4]));
  const auto *sfb = reinterpret_cast<ElementScale const *>(
      static_cast<std::uintptr_t>(workspace.group_bindings[binding_base + 5]));
  write_descriptor(workspace.descriptors, static_cast<std::int32_t>(group), m,
                   static_cast<std::int32_t>(args.hidden_size),
                   static_cast<std::int32_t>(args.intermediate_size), a, b, d,
                   sfa, sfb);
}

__global__ __launch_bounds__(kScatterThreads) void scatter_kernel(
    GroupedFp4MoeArgs args, WorkspaceView workspace) {
  const std::size_t routed_row = blockIdx.x;
  if (routed_row >= args.total_routed_rows) {
    return;
  }
  const auto *routes = device_pointer<std::int32_t const>(args.route_indices);
  const std::int32_t route = routes[routed_row];
  if (route < 0 || static_cast<std::uint32_t>(route) >= args.num_routes) {
    if (threadIdx.x == 0) {
      set_route_error(args);
    }
    return;
  }

  const auto *source = workspace.down_bf16 + routed_row * args.hidden_size;
  auto *destination = device_pointer<float>(args.route_output) +
                      static_cast<std::size_t>(route) * args.hidden_size;
  for (std::size_t channel = threadIdx.x; channel < args.hidden_size;
       channel += blockDim.x) {
    destination[channel] = static_cast<float>(source[channel]);
  }
  if (threadIdx.x == 0) {
    auto *written = device_pointer<std::int32_t>(args.route_written);
    atomicOr(reinterpret_cast<unsigned int *>(written + route), 1u);
  }
}

inline WorkspaceView
make_workspace_view(void *workspace, WorkspacePlan const &plan,
                    GroupedFp4MoeArgs const &args) noexcept {
  auto *base = static_cast<std::uint8_t *>(workspace);
  DescriptorStorage descriptor_storage{};
  descriptor_storage.data = base + plan.descriptor_offset;
  descriptor_storage.bytes = plan.descriptor_bytes;
  descriptor_storage.groups = plan.descriptor_capacity;

  WorkspaceView view{};
  view.descriptors = descriptor_storage.view();
  view.group_bindings =
      reinterpret_cast<std::uint64_t *>(base + plan.group_bindings_offset);
  view.route_groups =
      reinterpret_cast<std::uint32_t *>(base + plan.route_groups_offset);
  view.gathered_x = base + plan.gathered_x_offset;
  view.gathered_x_sfa =
      reinterpret_cast<ElementScale *>(base + plan.gathered_x_sfa_offset);
  view.gate_up_bf16 =
      reinterpret_cast<ElementD *>(base + plan.gate_up_bf16_offset);
  view.hidden_fp8 = base + plan.hidden_fp8_offset;
  view.hidden_sfa =
      reinterpret_cast<ElementScale *>(base + plan.hidden_sfa_offset);
  view.down_bf16 = reinterpret_cast<ElementD *>(base + plan.down_bf16_offset);
  view.gathered_x_sfa_group_stride = plan.gathered_x_sfa_group_stride;
  view.hidden_sfa_group_stride = plan.hidden_sfa_group_stride;
  static_cast<void>(args);
  return view;
}

struct DebugTap {
  bool enabled{};
  std::string directory{};
  std::uint32_t rows[2]{};
  std::uint32_t groups[2]{};
  float *down_input{};
};

inline bool debug_copy(void *host, const void *device, std::size_t bytes,
                       cudaStream_t stream) {
  return cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream) ==
             cudaSuccess &&
         cudaStreamSynchronize(stream) == cudaSuccess;
}

inline bool prepare_debug_tap(GroupedFp4MoeArgs const &args,
                              cudaStream_t stream, DebugTap &tap) {
  const char *directory = std::getenv("FERRULE_DEBUG_GROUPED_FP4_TAP_DIR");
  if (directory == nullptr || *directory == '\0' || args.num_routes != 6u ||
      args.active_group_count != 6u || args.total_routed_rows != 6u) {
    return true;
  }
  std::vector<std::int32_t> slots(args.active_group_count);
  std::vector<std::uint32_t> indptr(args.active_group_count + 1u);
  std::vector<std::int32_t> routes(args.total_routed_rows);
  if (!debug_copy(slots.data(),
                  device_pointer<void const>(args.active_expert_slots),
                  slots.size() * sizeof(slots[0]), stream) ||
      !debug_copy(indptr.data(),
                  device_pointer<void const>(args.expert_route_indptr),
                  indptr.size() * sizeof(indptr[0]), stream) ||
      !debug_copy(routes.data(), device_pointer<void const>(args.route_indices),
                  routes.size() * sizeof(routes[0]), stream)) {
    return false;
  }
  std::int32_t route_experts[6] = {-1, -1, -1, -1, -1, -1};
  std::uint32_t route_rows[6]{};
  std::uint32_t route_groups[6]{};
  for (std::uint32_t group = 0; group < args.active_group_count; ++group) {
    for (std::uint32_t ordinal = indptr[group]; ordinal < indptr[group + 1u];
         ++ordinal) {
      const std::int32_t route = routes[ordinal];
      if (route < 0 || route >= 6) {
        return true;
      }
      route_experts[route] = slots[group];
      route_rows[route] = ordinal;
      route_groups[route] = group;
    }
  }
  constexpr std::int32_t expected[6] = {127, 182, 103, 65, 246, 154};
  for (int route = 0; route < 6; ++route) {
    if (route_experts[route] != expected[route]) {
      return true;
    }
  }
  tap.enabled = true;
  tap.directory = directory;
  tap.rows[0] = route_rows[0];
  tap.rows[1] = route_rows[1];
  tap.groups[0] = route_groups[0];
  tap.groups[1] = route_groups[1];
  return cudaMalloc(reinterpret_cast<void **>(&tap.down_input),
                    2u * args.intermediate_size * sizeof(float)) == cudaSuccess;
}

inline bool write_debug_file(std::string const &path, const void *data,
                             std::size_t bytes) {
  std::FILE *file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return false;
  }
  const bool written = std::fwrite(data, 1u, bytes, file) == bytes;
  return std::fclose(file) == 0 && written;
}

inline float debug_bf16_round(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  bits &= 0xffff0000u;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline bool dump_debug_rows(DebugTap const &tap, const char *stage,
                            const float *source, std::size_t row_width,
                            bool round_bf16, cudaStream_t stream) {
  if (!tap.enabled) {
    return true;
  }
  std::vector<float> values(row_width);
  constexpr int experts[2] = {127, 182};
  for (int target = 0; target < 2; ++target) {
    if (!debug_copy(values.data(), source + tap.rows[target] * row_width,
                    row_width * sizeof(float), stream)) {
      return false;
    }
    if (round_bf16) {
      for (float &value : values) {
        value = debug_bf16_round(value);
      }
    }
    const std::string path = tap.directory + "/expert" +
                             std::to_string(experts[target]) + "." + stage +
                             ".device.f32";
    if (!write_debug_file(path, values.data(), values.size() * sizeof(float))) {
      return false;
    }
  }
  return true;
}

inline bool dump_debug_launch(DebugTap const &tap,
                              WorkspaceView const &workspace,
                              GroupedFp4MoeArgs const &args,
                              cudaStream_t stream) {
  if (!tap.enabled) {
    return true;
  }
  std::uint64_t bindings[12]{};
  std::int32_t active_generations[2]{};
  std::int32_t slot_generations[2]{};
  const auto *active =
      device_pointer<std::int32_t const>(args.active_group_generations);
  const auto *slots =
      device_pointer<std::int32_t const>(args.active_expert_slots);
  std::int32_t target_slots[2]{};
  for (int target = 0; target < 2; ++target) {
    if (!debug_copy(bindings + target * 6,
                    workspace.group_bindings + tap.groups[target] * 6,
                    6u * sizeof(std::uint64_t), stream) ||
        !debug_copy(active_generations + target, active + tap.groups[target],
                    sizeof(std::int32_t), stream) ||
        !debug_copy(target_slots + target, slots + tap.groups[target],
                    sizeof(std::int32_t), stream)) {
      return false;
    }
    const auto *slot_generation =
        device_pointer<std::int32_t const>(args.slot_generations) +
        target_slots[target];
    if (!debug_copy(slot_generations + target, slot_generation,
                    sizeof(std::int32_t), stream)) {
      return false;
    }
  }
  const std::string path = tap.directory + "/launch-tuples.txt";
  std::FILE *file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return false;
  }
  constexpr int experts[2] = {127, 182};
  bool ok = true;
  for (int target = 0; target < 2; ++target) {
    ok &= std::fprintf(
              file,
              "expert=%d group=%u row=%u slot=%d active_generation=%d "
              "slot_generation=%d gate_weight=%#llx gate_scale=%#llx "
              "up_weight=%#llx up_scale=%#llx down_weight=%#llx "
              "down_scale=%#llx\n",
              experts[target], tap.groups[target], tap.rows[target],
              target_slots[target], active_generations[target],
              slot_generations[target],
              static_cast<unsigned long long>(bindings[target * 6]),
              static_cast<unsigned long long>(bindings[target * 6 + 1]),
              static_cast<unsigned long long>(bindings[target * 6 + 2]),
              static_cast<unsigned long long>(bindings[target * 6 + 3]),
              static_cast<unsigned long long>(bindings[target * 6 + 4]),
              static_cast<unsigned long long>(bindings[target * 6 + 5])) > 0;
  }
  return std::fclose(file) == 0 && ok;
}

inline ::cutlass::Status
run_grouped(MmaMode mode, DeviceDescriptorView descriptors,
            LaunchOptions const &options, void *workspace,
            std::size_t workspace_bytes, cudaStream_t stream) noexcept {
  if (descriptors.groups == 0) {
    return ::cutlass::Status::kSuccess;
  }
  GroupedProblem problem{};
  problem.descriptors = descriptors;
  problem.device_id = options.device_id;
  problem.sm_count = options.sm_count;
  return mode == MmaMode::k1Sm
             ? run_1sm(problem, workspace, workspace_bytes, stream)
             : run_2sm(problem, workspace, workspace_bytes, stream);
}

inline std::size_t
required_cutlass_bytes(GroupedFp4MoeArgs const &args,
                       LaunchOptions const &options) noexcept {
  const std::uint32_t small = args.small_group_count;
  const std::uint32_t large = args.active_group_count - small;
  std::size_t result = 0;
  const std::size_t sizes[] = {
      cutlass_bytes_for(MmaMode::k1Sm, small * 2, options),
      cutlass_bytes_for(MmaMode::k1Sm, small, options),
      cutlass_bytes_for(MmaMode::k2Sm, large * 2, options),
      cutlass_bytes_for(MmaMode::k2Sm, large, options)};
  for (std::size_t size : sizes) {
    result = size > result ? size : result;
  }
  return result;
}

} // namespace moe_detail

inline WorkspacePlan workspace_plan(GroupedFp4MoeArgs const &args,
                                    LaunchOptions const &options) noexcept {
  WorkspacePlan plan{};
  if (!moe_detail::scalar_args_valid(args, options)) {
    return plan;
  }

  const std::size_t groups = args.active_group_count;
  const std::size_t rows = args.total_routed_rows;
  const std::size_t descriptor_capacity = groups * 2;
  if (descriptor_capacity > 0x7fffffffu) {
    return {};
  }
  plan.descriptor_capacity = static_cast<std::int32_t>(descriptor_capacity);
  plan.descriptor_bytes = descriptor_bytes(plan.descriptor_capacity);
  plan.gathered_x_sfa_group_stride = prepared_sfa_bytes(
      static_cast<int>(args.max_group_rows), static_cast<int>(args.input_size));
  plan.hidden_sfa_group_stride =
      prepared_sfa_bytes(static_cast<int>(args.max_group_rows),
                         static_cast<int>(args.intermediate_size));
  if (plan.descriptor_bytes == 0 || plan.gathered_x_sfa_group_stride == 0 ||
      plan.hidden_sfa_group_stride == 0 ||
      !moe_detail::checked_product(plan.group_bindings_bytes, groups, 6,
                                   sizeof(std::uint64_t)) ||
      !moe_detail::checked_product(plan.route_groups_bytes, rows,
                                   sizeof(std::uint32_t)) ||
      !moe_detail::checked_product(plan.gathered_x_bytes, rows,
                                   args.input_size) ||
      !moe_detail::checked_product(plan.gathered_x_sfa_bytes, groups,
                                   plan.gathered_x_sfa_group_stride) ||
      !moe_detail::checked_product(plan.gate_up_bf16_bytes, 2, rows,
                                   args.intermediate_size, sizeof(ElementD)) ||
      !moe_detail::checked_product(plan.hidden_fp8_bytes, rows,
                                   args.intermediate_size) ||
      !moe_detail::checked_product(plan.hidden_sfa_bytes, groups,
                                   plan.hidden_sfa_group_stride) ||
      !moe_detail::checked_product(plan.down_bf16_bytes, rows, args.hidden_size,
                                   sizeof(ElementD))) {
    return {};
  }
  plan.cutlass_bytes = moe_detail::required_cutlass_bytes(args, options);

  std::size_t cursor = 0;
  if (!moe_detail::append_region(cursor, plan.descriptor_bytes,
                                 kDescriptorAlignment,
                                 plan.descriptor_offset) ||
      !moe_detail::append_region(cursor, plan.group_bindings_bytes,
                                 alignof(std::uint64_t),
                                 plan.group_bindings_offset) ||
      !moe_detail::append_region(cursor, plan.route_groups_bytes,
                                 alignof(std::uint32_t),
                                 plan.route_groups_offset) ||
      !moe_detail::append_region(cursor, plan.gathered_x_bytes, 16,
                                 plan.gathered_x_offset) ||
      !moe_detail::append_region(cursor, plan.gathered_x_sfa_bytes, 16,
                                 plan.gathered_x_sfa_offset) ||
      !moe_detail::append_region(cursor, plan.gate_up_bf16_bytes, 16,
                                 plan.gate_up_bf16_offset) ||
      !moe_detail::append_region(cursor, plan.hidden_fp8_bytes, 16,
                                 plan.hidden_fp8_offset) ||
      !moe_detail::append_region(cursor, plan.hidden_sfa_bytes, 16,
                                 plan.hidden_sfa_offset) ||
      !moe_detail::append_region(cursor, plan.down_bf16_bytes, 16,
                                 plan.down_bf16_offset) ||
      !moe_detail::append_region(cursor, plan.cutlass_bytes,
                                 kCutlassWorkspaceAlignment,
                                 plan.cutlass_offset)) {
    return {};
  }
  if (cursor >
      (std::numeric_limits<std::size_t>::max)() - (kWorkspaceAlignment - 1)) {
    return {};
  }
  plan.total_bytes = detail::align_up(cursor, kWorkspaceAlignment);
  return plan;
}

inline std::size_t workspace_bytes(GroupedFp4MoeArgs const &args,
                                   LaunchOptions const &options) noexcept {
  return workspace_plan(args, options).total_bytes;
}

inline Status can_implement(GroupedFp4MoeArgs const *args, void *workspace,
                            std::size_t workspace_bytes,
                            LaunchOptions const &options) noexcept {
  if (args == nullptr || !moe_detail::scalar_args_valid(*args, options) ||
      !moe_detail::pointer_args_valid(*args)) {
    return Status::kInvalidArgument;
  }
  const WorkspacePlan plan = workspace_plan(*args, options);
  if (!plan.valid() || workspace_bytes < plan.total_bytes ||
      !moe_detail::aligned_pointer(workspace, kWorkspaceAlignment)) {
    return Status::kUnsupportedResources;
  }
  return Status::kSuccess;
}

// The caller owns workspace and stream. Weight scale pointers must reference
// grouped_fp4_moe::launch_prepare_sfb output for gate/up [intermediate,K]
// and down [hidden,intermediate]. route_written and route_error are not cleared
// by this helper, allowing composition with a larger routed operation.
inline Status launch(GroupedFp4MoeArgs const *args, void *workspace,
                     std::size_t workspace_bytes, cudaStream_t stream,
                     LaunchOptions const &options) noexcept {
  const Status validation =
      can_implement(args, workspace, workspace_bytes, options);
  if (validation != Status::kSuccess) {
    return validation;
  }
  const WorkspacePlan plan = workspace_plan(*args, options);

  moe_detail::WorkspaceView view =
      moe_detail::make_workspace_view(workspace, plan, *args);
  if (view.descriptors.groups != plan.descriptor_capacity) {
    return Status::kUnsupportedResources;
  }
  auto *base = static_cast<std::uint8_t *>(workspace);
  void *cutlass_workspace = base + plan.cutlass_offset;

  cudaError_t cuda_status =
      cudaMemsetAsync(view.route_groups, 0xff, plan.route_groups_bytes, stream);
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemsetAsync(view.gathered_x_sfa, kScalePadding,
                                  plan.gathered_x_sfa_bytes, stream);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemsetAsync(view.hidden_sfa, kScalePadding,
                                  plan.hidden_sfa_bytes, stream);
  }
  if (cuda_status != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  moe_detail::prepare_gate_up_kernel<<<args->active_group_count,
                                       kPrepareThreads, 0, stream>>>(
      *args, options, view);
  if (cudaGetLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  const std::uint32_t small = args->small_group_count;
  const std::uint32_t large = args->active_group_count - small;
  ::cutlass::Status cutlass_status = ::cutlass::Status::kSuccess;
  if (small != 0) {
    DeviceDescriptorView small_gate_up = moe_detail::slice_descriptors(
        view.descriptors, 0, static_cast<std::int32_t>(small * 2));
    cutlass_status =
        moe_detail::run_grouped(MmaMode::k1Sm, small_gate_up, options,
                                cutlass_workspace, plan.cutlass_bytes, stream);
  }
  if (cutlass_status == ::cutlass::Status::kSuccess && large != 0) {
    DeviceDescriptorView large_gate_up = moe_detail::slice_descriptors(
        view.descriptors, static_cast<std::int32_t>(small * 2),
        static_cast<std::int32_t>(large * 2));
    cutlass_status =
        moe_detail::run_grouped(MmaMode::k2Sm, large_gate_up, options,
                                cutlass_workspace, plan.cutlass_bytes, stream);
  }
  if (cutlass_status != ::cutlass::Status::kSuccess) {
    return Status::kLaunchFailed;
  }

  std::size_t quant_blocks = 0;
  if (!moe_detail::checked_product(quant_blocks, args->total_routed_rows,
                                   args->intermediate_size / 128) ||
      quant_blocks == 0 || quant_blocks > 0x7fffffffu) {
    return Status::kUnsupportedResources;
  }
  moe_detail::swiglu_requant_kernel<<<static_cast<unsigned>(quant_blocks),
                                      kQuantThreads, 0, stream>>>(
      *args, view, nullptr, UINT32_MAX, UINT32_MAX);
  if (cudaGetLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  const std::uint32_t down_blocks =
      (args->active_group_count + kPrepareThreads - 1) / kPrepareThreads;
  moe_detail::prepare_down_kernel<<<down_blocks, kPrepareThreads, 0, stream>>>(
      *args, view);
  if (cudaGetLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  cutlass_status = ::cutlass::Status::kSuccess;
  if (small != 0) {
    DeviceDescriptorView small_down = moe_detail::slice_descriptors(
        view.descriptors, 0, static_cast<std::int32_t>(small));
    cutlass_status =
        moe_detail::run_grouped(MmaMode::k1Sm, small_down, options,
                                cutlass_workspace, plan.cutlass_bytes, stream);
  }
  if (cutlass_status == ::cutlass::Status::kSuccess && large != 0) {
    DeviceDescriptorView large_down = moe_detail::slice_descriptors(
        view.descriptors, static_cast<std::int32_t>(small),
        static_cast<std::int32_t>(large));
    cutlass_status =
        moe_detail::run_grouped(MmaMode::k2Sm, large_down, options,
                                cutlass_workspace, plan.cutlass_bytes, stream);
  }
  if (cutlass_status != ::cutlass::Status::kSuccess) {
    return Status::kLaunchFailed;
  }

  moe_detail::
      scatter_kernel<<<args->total_routed_rows, kScatterThreads, 0, stream>>>(
          *args, view);
  return cudaGetLastError() == cudaSuccess ? Status::kSuccess
                                           : Status::kLaunchFailed;
}

} // namespace ferrule::cuda::cutlass::operators::grouped_fp4_moe

#endif // FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
