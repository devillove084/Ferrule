#pragma once

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cooperative_groups.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>

#if defined(FERRULE_CUTLASS_TARGET_SM) && FERRULE_CUTLASS_TARGET_SM != 121
#error "sm121_shared_ffn.cuh is only valid for Ferrule's SM121a provider"
#endif

// One-launch shared-FFN provider for the DeepSeek-V4 artifact contract.
//
// The kernel is a cooperative, two-stage semantic pipeline rather than three
// host-launched GEMMs:
//
//   1. CTAs partition [M, intermediate] into [8, 128] tiles. Eight warps compute
//      gate/up FP8 MMA tiles, apply SwiGLU, and quantize the tile directly into
//      the compact global FP8 intermediate. No gate/up F32 tensor is published.
//   2. A grid barrier makes the compact intermediate visible. CTAs then
//      partition [M, output] into [8, 128] tiles and compute the down projection.
//
// The compact intermediate is caller-owned graph-stable storage. It is the only
// cross-CTA dependency and is part of this semantic launch; gate/up/down are not
// independently callable provider entries.

namespace ferrule::cutlass::sm121_shared_ffn {

constexpr std::uint32_t kArtifactBlock = 128;
constexpr std::uint32_t kMmaChannels = 16;
constexpr std::uint32_t kMmaRows = 8;
constexpr std::uint32_t kWarpSize = 32;
constexpr std::uint32_t kWarpCount = 8;
constexpr std::uint32_t kThreads = kWarpSize * kWarpCount;
constexpr std::uint32_t kChannelsPerCta = kWarpCount * kMmaChannels;
// GB10 has 20 SMs. The fixed tile uses about 10 KiB shared memory and supports
// two resident CTAs per SM in the release provider, which covers all 32 formal
// down-projection tiles in one cooperative wave.
constexpr std::uint32_t kGb10CooperativeBlocks = 40;

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

__device__ __forceinline__ float nearest_fp8_e4m3fn_positive(
    float magnitude) {
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

__device__ __forceinline__ std::uint8_t quantize_fp8_e4m3fn_byte(
    float value) {
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
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

__device__ __forceinline__ float swiglu(float gate, float up,
                                        float output_scale,
                                        float swiglu_limit) {
  if (swiglu_limit > 0.0f) {
    gate = gate > swiglu_limit ? swiglu_limit : gate;
    up = clamp_value(up, -swiglu_limit, swiglu_limit);
  }
  return gate * sigmoid(gate) * up * output_scale;
}

__device__ __forceinline__ void load_weight_fragment(
    const std::uint8_t *shared, std::uint32_t lane,
    std::uint32_t (&fragment)[4]) {
  std::uint32_t quad = lane >> 3;
  std::uint32_t row = (lane & 7u) + ((quad & 1u) != 0 ? 8u : 0u);
  std::uint32_t column_bytes = quad >= 2 ? 16u : 0u;
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + row * 32u + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1],
                                fragment[2], fragment[3]);
}

__device__ __forceinline__ void load_activation_fragment(
    const std::uint8_t *shared, std::uint32_t lane,
    std::uint32_t (&fragment)[2]) {
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void mma_fp8_e4m3(
    float (&accumulator)[4], const std::uint32_t (&weight)[4],
    const std::uint32_t (&activation)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator[0]), "+f"(accumulator[1]),
        "+f"(accumulator[2]), "+f"(accumulator[3])
      : "r"(weight[0]), "r"(weight[1]), "r"(weight[2]),
        "r"(weight[3]), "r"(activation[0]), "r"(activation[1]));
}

struct SharedLayout {
  std::uint8_t *warp_stages;
};

__device__ __forceinline__ SharedLayout make_shared_layout(
    std::uint8_t *storage) {
  return SharedLayout{storage};
}

__device__ __forceinline__ void stage_weight_k32(
    std::uint8_t *stage, const std::uint8_t *weight,
    std::uint32_t channel_base, std::uint32_t channel_count,
    std::uint32_t row_width, std::uint32_t k_base, std::uint32_t lane) {
  if (lane >= kMmaChannels) {
    return;
  }
  auto *destination = reinterpret_cast<uint4 *>(
      stage + static_cast<std::uint64_t>(lane) * 32u);
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

__device__ __forceinline__ void stage_activation_k32(
    std::uint8_t *stage, const std::uint8_t *values,
    std::uint32_t row_base, std::uint32_t active_rows,
    std::uint32_t row_width, std::uint32_t k_base, std::uint32_t lane) {
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

__device__ __forceinline__ void gate_up_tile(
    const Args &args, const SharedLayout &shared, std::uint32_t row_base,
    std::uint32_t active_rows, std::uint32_t channel_block,
    std::uint32_t warp, std::uint32_t lane) {
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
    std::uint32_t channel = channel_base + channel_group +
                            (element >= 2u ? 8u : 0u);
    std::uint32_t row = row_pair * 2u + (element & 1u);
    if (row < active_rows && channel < args.intermediate_size) {
      args.hidden_f32[static_cast<std::uint64_t>(row_base + row) *
                          args.intermediate_size +
                      channel] =
          swiglu(gate_accumulator[element], up_accumulator[element],
                  args.output_scale, args.swiglu_limit);
    }
  }
}

__device__ __forceinline__ void pack_hidden_block(
    const Args &args, std::uint32_t row, std::uint32_t scale_block,
    std::uint32_t lane) {
  const std::uint64_t start = static_cast<std::uint64_t>(row) *
                                  args.intermediate_size +
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
  const float scale =
      ue8m0_to_float(static_cast<std::uint8_t>(scale_byte));
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const float value = values[element] / scale;
    args.hidden_fp8[start + lane + element * kWarpSize] =
        quantize_fp8_e4m3fn_byte(clamp_value(value, -448.0f, 448.0f));
  }
}

__device__ __forceinline__ void down_tile(
    const Args &args, const SharedLayout &shared, std::uint32_t row_base,
    std::uint32_t active_rows, std::uint32_t channel_block,
    std::uint32_t warp, std::uint32_t lane) {
  std::uint32_t channel_base = channel_block * kMmaChannels;
  std::uint32_t scale_cols = args.intermediate_size / kArtifactBlock;
  std::uint8_t *warp_stage = shared.warp_stages + warp * kWarpStageBytes;
  std::uint8_t *weight_stage = warp_stage;
  std::uint8_t *activation_stage = warp_stage + kWeightStageBytes;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t scale_block = 0; scale_block < scale_cols;
       ++scale_block) {
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
    std::uint32_t channel = channel_base + channel_group +
                            (element >= 2u ? 8u : 0u);
    std::uint32_t row = row_pair * 2u + (element & 1u);
    if (row < active_rows && channel < args.output_size) {
      std::uint64_t output_index =
          static_cast<std::uint64_t>(row_base + row) * args.output_size +
          channel;
      if ((args.flags & kAccumulateOutput) != 0u) {
        args.output_f32[output_index] += accumulator[element];
      } else {
        args.output_f32[output_index] = accumulator[element];
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
    std::uint32_t channel_block =
        task - row_tile * intermediate_channel_blocks;
    std::uint32_t row_base = row_tile * kMmaRows;
    std::uint32_t active_rows = min(kMmaRows, args.rows - row_base);
    gate_up_tile(args, shared, row_base, active_rows, channel_block, warp,
                 lane);
  }

  cooperative_groups::this_grid().sync();

  std::uint32_t hidden_scale_cols =
      args.intermediate_size / kArtifactBlock;
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
      args.gate_weight_fp8 == nullptr ||
      args.gate_weight_ue8m0 == nullptr || args.up_weight_fp8 == nullptr ||
      args.up_weight_ue8m0 == nullptr || args.down_weight_fp8 == nullptr ||
      args.down_weight_ue8m0 == nullptr || args.hidden_f32 == nullptr ||
      args.hidden_fp8 == nullptr || args.hidden_ue8m0 == nullptr ||
      args.output_f32 == nullptr) {
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
      args.up_block_m != kArtifactBlock ||
      args.up_block_k != kArtifactBlock ||
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
      row_tiles * ((args.intermediate_size + kMmaChannels - 1u) /
                   kMmaChannels);
  std::uint32_t down_tasks =
      row_tiles * ((args.output_size + kMmaChannels - 1u) /
                   kMmaChannels);
  std::uint32_t warp_tasks = max(gate_tasks, down_tasks);
  std::uint32_t blocks = min(
      kGb10CooperativeBlocks,
      (warp_tasks + kWarpCount - 1u) / kWarpCount);
  void *kernel_args[] = {const_cast<Args *>(&args)};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::kernel), dim3(blocks, 1, 1),
      dim3(kThreads, 1, 1), kernel_args, dynamic_shared_memory_bytes(), stream);
}

} // namespace ferrule::cutlass::sm121_shared_ffn
