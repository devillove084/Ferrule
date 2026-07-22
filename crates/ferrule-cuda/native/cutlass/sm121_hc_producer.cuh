#ifndef FERRULE_CUDA_NATIVE_CUTLASS_SM121_HC_PRODUCER_CUH_
#define FERRULE_CUDA_NATIVE_CUTLASS_SM121_HC_PRODUCER_CUH_

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/numeric/int.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(FERRULE_CUTLASS_TARGET_SM) && FERRULE_CUTLASS_TARGET_SM != 121
#error "sm121_hc_producer.cuh is only valid for Ferrule's GB10 SM121 target"
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1210
#error "sm121_hc_producer.cuh device code must be compiled for SM121"
#endif

namespace ferrule {
namespace sm121 {

inline constexpr std::uint32_t kHcPreRmsNormFp8AbiVersion = 1;
inline constexpr std::uint32_t kThreads = 256;
inline constexpr std::uint32_t kHc = 4;
inline constexpr std::uint32_t kHidden = 4096;
inline constexpr std::uint32_t kMix = 24;
inline constexpr std::uint32_t kHcHidden = kHc * kHidden;
inline constexpr std::uint32_t kFp8Block = 128;
inline constexpr std::uint32_t kScaleColumns = kHidden / kFp8Block;
inline constexpr std::uint32_t kFunctionTileColumns = 128;
inline constexpr std::uint32_t kSingleRowFunctionTileColumns = 256;

static_assert(sizeof(cute::uint128_t) == 16);

// Fixed-specialization tensor contract:
//
//   state_f32:               [rows, 4, 4096]
//   function_col_major_f32:  [4 * 4096, 24], indexed [column, mix_row]
//   hc_scale_f32:            [3]
//   hc_base_f32:             [24]
//   layer_rms_weight_f32:    [4096]
//   hidden_f32:              [rows, 4096]
//   normalized_f32:          [rows, 4096]
//   packed_e4m3:             [rows, 4096]
//   scales_ue8m0:            [rows, 32], one scale per K128 block
//   split_pre_f32:           [rows, 4]
//   split_post_f32:          [rows, 4]
//   split_comb_f32:          [rows, 4, 4]
//
// Every address names Ferrule-owned device storage. The kernel allocates no
// memory, and launch_hc_pre_rmsnorm_fp8 never synchronizes the host.
struct alignas(16) HcPreRmsNormFp8Args {
  std::uint32_t abi_version;
  std::uint32_t rows;
  std::uint32_t hc;
  std::uint32_t hidden;
  std::uint32_t mix;
  std::uint32_t sinkhorn_iters;
  float hc_eps;
  float hc_norm_eps;
  float layer_rms_eps;
  std::uint32_t reserved;

  std::uint64_t state_f32;
  std::uint64_t function_col_major_f32;
  std::uint64_t hc_scale_f32;
  std::uint64_t hc_base_f32;
  std::uint64_t layer_rms_weight_f32;
  std::uint64_t hidden_f32;
  std::uint64_t normalized_f32;
  std::uint64_t packed_e4m3;
  std::uint64_t scales_ue8m0;
  std::uint64_t split_pre_f32;
  std::uint64_t split_post_f32;
  std::uint64_t split_comb_f32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<HcPreRmsNormFp8Args>);
static_assert(std::is_trivially_copyable_v<HcPreRmsNormFp8Args>);
static_assert(sizeof(HcPreRmsNormFp8Args) == 144);
static_assert(offsetof(HcPreRmsNormFp8Args, state_f32) == 40);
static_assert(offsetof(HcPreRmsNormFp8Args, stream) == 136);

namespace detail {



inline bool aligned_device_address(std::uint64_t address,
                                   std::uint64_t alignment) noexcept {
  return address != 0 && (address & (alignment - 1)) == 0;
}

__device__ __forceinline__ bool finite_f32(float value) {
  return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ float rsqrt_approx(float value) {
  float result;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float fast_exp(float value) {
  return expf(value);
}

__device__ __forceinline__ float fast_sigmoid(float value) {
  if (value < -16.0f) {
    return 0.0f;
  }
  if (value > 16.0f) {
    return 1.0f;
  }
  if (value >= 0.0f) {
    return 1.0f / (1.0f + fast_exp(-value));
  }
  float exponential = fast_exp(value);
  return exponential / (1.0f + exponential);
}

__device__ __forceinline__ float block_sum_256(float value,
                                                float *reduction) {
  std::uint32_t tid = threadIdx.x;
  reduction[tid] = value;
  __syncthreads();

#pragma unroll
  for (std::uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  return reduction[0];
}

__device__ __forceinline__ float pow2_via_exp(float exponent) {
  // Exact f32 spelling of core::f32::consts::LN_2 used by kernels.rs.
  constexpr float kLn2 = 0x1.62e43p-1f;
  return fast_exp(exponent * kLn2);
}

__device__ __forceinline__ float nearest_fp8_subnormal_positive(
    float magnitude) {
  float step = pow2_via_exp(-9.0f);
  float mantissa = roundf(magnitude / step);
  mantissa = mantissa < 0.0f ? 0.0f : mantissa;
  mantissa = mantissa > 7.0f ? 7.0f : mantissa;
  return mantissa * step;
}

__device__ __forceinline__ float nearest_fp8_e4m3fn_positive(
    float magnitude) {
  float best = nearest_fp8_subnormal_positive(magnitude);
  float best_error = best > magnitude ? best - magnitude : magnitude - best;
  int exponent_floor = static_cast<int>(floorf(log2f(magnitude)));

#pragma unroll
  for (int exponent = exponent_floor - 1; exponent <= exponent_floor + 1;
       ++exponent) {
    if (exponent < -6 || exponent > 8) {
      continue;
    }

    float scale = pow2_via_exp(static_cast<float>(exponent));
    int mantissa =
        static_cast<int>(roundf((magnitude / scale - 1.0f) * 8.0f));
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

    float candidate =
        pow2_via_exp(static_cast<float>(candidate_exponent)) *
        (1.0f + static_cast<float>(mantissa) / 8.0f);
    float error = candidate > magnitude ? candidate - magnitude
                                        : magnitude - candidate;
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

__device__ __forceinline__ std::uint8_t e8m0_scale_byte_for_amax(
    float amax) {
  if (!finite_f32(amax) || amax <= 0.0f) {
    return 127;
  }
  int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
  int byte = exponent + 127;
  byte = byte < 0 ? 0 : byte;
  byte = byte > 255 ? 255 : byte;
  return static_cast<std::uint8_t>(byte);
}

__device__ __forceinline__ float ue8m0_to_float(std::uint8_t byte) {
  std::uint32_t bits =
      byte == 0 ? (1u << 22) : (static_cast<std::uint32_t>(byte) << 23);
  return __uint_as_float(bits);
}

__device__ __forceinline__ float clamp_fp8_input(float value) {
  if (value < -448.0f) {
    return -448.0f;
  }
  if (value > 448.0f) {
    return 448.0f;
  }
  return value;
}

struct alignas(16) SharedStorage {
  float reduction[kThreads];
  float mix[kMix];
  float pre[kHc];
  float comb[kHc * kHc];
  alignas(16) float function_tile[kSingleRowFunctionTileColumns * kMix];
  alignas(16) float state_tile[kSingleRowFunctionTileColumns];
};

static_assert((kFunctionTileColumns * kMix) % 4 == 0);
static_assert(kFunctionTileColumns % 4 == 0);
static_assert((kSingleRowFunctionTileColumns * kMix) % 4 == 0);
static_assert(kSingleRowFunctionTileColumns % 4 == 0);


template <std::uint32_t TileColumns>
__device__ __forceinline__ float mix_state_function(
    SharedStorage &shared, const float *state, const float *function_col_major,
    std::uint32_t state_base, std::uint32_t tid) {
  float accumulator = 0.0f;
  for (std::uint32_t tile_base = 0; tile_base < kHcHidden;
       tile_base += TileColumns) {
    auto const *function_vectors = reinterpret_cast<cute::uint128_t const *>(
        function_col_major + tile_base * kMix);
    auto *shared_function_vectors =
        reinterpret_cast<cute::uint128_t *>(shared.function_tile);
    constexpr std::uint32_t kFunctionVectors = TileColumns * kMix / 4;
    for (std::uint32_t vector = tid; vector < kFunctionVectors;
         vector += kThreads) {
      shared_function_vectors[vector] = function_vectors[vector];
    }

    if (tid < TileColumns / 4) {
      auto const *state_vectors = reinterpret_cast<cute::uint128_t const *>(
          state + state_base + tile_base);
      auto *shared_state_vectors =
          reinterpret_cast<cute::uint128_t *>(shared.state_tile);
      shared_state_vectors[tid] = state_vectors[tid];
    }
    __syncthreads();

    if (tid < kMix) {
#pragma unroll
      for (std::uint32_t column = 0; column < TileColumns; ++column) {
        accumulator += shared.function_tile[column * kMix + tid] *
                       shared.state_tile[column];
      }
    }
    __syncthreads();
  }
  return accumulator;
}

} // namespace detail

inline bool validate_hc_pre_rmsnorm_fp8(
    HcPreRmsNormFp8Args const &args) noexcept {
  if (args.abi_version != kHcPreRmsNormFp8AbiVersion ||
      args.rows == 0 || args.hc != kHc || args.hidden != kHidden ||
      args.mix != kMix || args.reserved != 0) {
    return false;
  }

  if (!std::isfinite(args.hc_eps) || !std::isfinite(args.hc_norm_eps) ||
      !std::isfinite(args.layer_rms_eps)) {
    return false;
  }

  // Ferrule device allocations satisfy this naturally. The stronger alignment
  // permits the CuTe uint128_t transport below without changing arithmetic.
  return detail::aligned_device_address(args.state_f32, 16) &&
         detail::aligned_device_address(args.function_col_major_f32, 16) &&
         detail::aligned_device_address(args.hc_scale_f32, 16) &&
         detail::aligned_device_address(args.hc_base_f32, 16) &&
         detail::aligned_device_address(args.layer_rms_weight_f32, 16) &&
         detail::aligned_device_address(args.hidden_f32, 16) &&
         detail::aligned_device_address(args.normalized_f32, 16) &&
         detail::aligned_device_address(args.packed_e4m3, 16) &&
         detail::aligned_device_address(args.scales_ue8m0, 16) &&
         detail::aligned_device_address(args.split_pre_f32, 16) &&
         detail::aligned_device_address(args.split_post_f32, 16) &&
         detail::aligned_device_address(args.split_comb_f32, 16);
}

__global__ __launch_bounds__(kThreads) void hc_pre_rmsnorm_fp8_kernel(
    HcPreRmsNormFp8Args args) {
  __shared__ detail::SharedStorage shared;

  std::uint32_t row = blockIdx.x;
  std::uint32_t tid = threadIdx.x;
  if (row >= args.rows) {
    return;
  }

  auto const *state = reinterpret_cast<float const *>(
      static_cast<std::uintptr_t>(args.state_f32));
  auto const *function_col_major = reinterpret_cast<float const *>(
      static_cast<std::uintptr_t>(args.function_col_major_f32));
  auto const *hc_scale = reinterpret_cast<float const *>(
      static_cast<std::uintptr_t>(args.hc_scale_f32));
  auto const *hc_base = reinterpret_cast<float const *>(
      static_cast<std::uintptr_t>(args.hc_base_f32));
  auto const *layer_rms_weight = reinterpret_cast<float const *>(
      static_cast<std::uintptr_t>(args.layer_rms_weight_f32));
  auto *hidden = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(args.hidden_f32));
  auto *normalized = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(args.normalized_f32));
  auto *packed = reinterpret_cast<std::uint8_t *>(
      static_cast<std::uintptr_t>(args.packed_e4m3));
  auto *scales = reinterpret_cast<std::uint8_t *>(
      static_cast<std::uintptr_t>(args.scales_ue8m0));
  auto *split_pre = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(args.split_pre_f32));
  auto *split_post = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(args.split_post_f32));
  auto *split_comb = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(args.split_comb_f32));

  std::uint32_t state_base = row * kHcHidden;

  // HC input RMS: identical 256-thread striding and tree shape to hc_pre_f32.
  float state_square_sum = 0.0f;
  for (std::uint32_t column = tid; column < kHcHidden;
       column += kThreads) {
    float value = state[state_base + column];
    state_square_sum += value * value;
  }
  float state_total = detail::block_sum_256(state_square_sum, shared.reduction);
  if (tid == 0) {
    shared.reduction[0] = detail::rsqrt_approx(
        state_total / static_cast<float>(kHcHidden) + args.hc_norm_eps);
  }
  __syncthreads();
  float state_rms = shared.reduction[0];

  // CuTe's uint128_t is transport only; each of the 24 arithmetic lanes still
  // consumes columns in increasing order. A single row uses a wider tile to
  // halve CTA barriers; wider inputs preserve the original 128-column path.
  float mix_accumulator = 0.0f;
  if (args.rows == 1u) {
    mix_accumulator = detail::mix_state_function<kSingleRowFunctionTileColumns>(
        shared, state, function_col_major, state_base, tid);
  } else {
    mix_accumulator = detail::mix_state_function<kFunctionTileColumns>(
        shared, state, function_col_major, state_base, tid);
  }

  if (tid < kMix) {
    shared.mix[tid] = mix_accumulator * state_rms;
  }
  __syncthreads();

  if (tid == 0) {
    std::uint32_t split_base = row * kHc;
    std::uint32_t comb_base = row * kHc * kHc;

#pragma unroll
    for (std::uint32_t copy = 0; copy < kHc; ++copy) {
      float pre = detail::fast_sigmoid(shared.mix[copy] * hc_scale[0] +
                                       hc_base[copy]) +
                  args.hc_eps;
      float post =
          2.0f * detail::fast_sigmoid(shared.mix[kHc + copy] * hc_scale[1] +
                                      hc_base[kHc + copy]);
      shared.pre[copy] = pre;
      split_pre[split_base + copy] = pre;
      split_post[split_base + copy] = post;
    }

#pragma unroll
    for (std::uint32_t comb_row = 0; comb_row < kHc; ++comb_row) {
      float row_max = __int_as_float(0xff800000u);
#pragma unroll
      for (std::uint32_t column = 0; column < kHc; ++column) {
        std::uint32_t index = comb_row * kHc + column;
        float value = shared.mix[2 * kHc + index] * hc_scale[2] +
                      hc_base[2 * kHc + index];
        shared.comb[index] = value;
        row_max = value > row_max ? value : row_max;
      }

      float row_sum = 0.0f;
#pragma unroll
      for (std::uint32_t column = 0; column < kHc; ++column) {
        std::uint32_t index = comb_row * kHc + column;
        float value = detail::fast_exp(shared.comb[index] - row_max);
        shared.comb[index] = value;
        row_sum += value;
      }
#pragma unroll
      for (std::uint32_t column = 0; column < kHc; ++column) {
        std::uint32_t index = comb_row * kHc + column;
        shared.comb[index] /= row_sum;
        shared.comb[index] += args.hc_eps;
      }
    }

#pragma unroll
    for (std::uint32_t column = 0; column < kHc; ++column) {
      float column_sum = 0.0f;
#pragma unroll
      for (std::uint32_t comb_row = 0; comb_row < kHc; ++comb_row) {
        column_sum += shared.comb[comb_row * kHc + column];
      }
#pragma unroll
      for (std::uint32_t comb_row = 0; comb_row < kHc; ++comb_row) {
        shared.comb[comb_row * kHc + column] /= column_sum + args.hc_eps;
      }
    }

    for (std::uint32_t iteration = 1; iteration < args.sinkhorn_iters;
         ++iteration) {
#pragma unroll
      for (std::uint32_t comb_row = 0; comb_row < kHc; ++comb_row) {
        float row_sum = 0.0f;
#pragma unroll
        for (std::uint32_t column = 0; column < kHc; ++column) {
          row_sum += shared.comb[comb_row * kHc + column];
        }
#pragma unroll
        for (std::uint32_t column = 0; column < kHc; ++column) {
          shared.comb[comb_row * kHc + column] /= row_sum + args.hc_eps;
        }
      }

#pragma unroll
      for (std::uint32_t column = 0; column < kHc; ++column) {
        float column_sum = 0.0f;
#pragma unroll
        for (std::uint32_t comb_row = 0; comb_row < kHc; ++comb_row) {
          column_sum += shared.comb[comb_row * kHc + column];
        }
#pragma unroll
        for (std::uint32_t comb_row = 0; comb_row < kHc; ++comb_row) {
          shared.comb[comb_row * kHc + column] /=
              column_sum + args.hc_eps;
        }
      }
    }

#pragma unroll
    for (std::uint32_t index = 0; index < kHc * kHc; ++index) {
      split_comb[comb_base + index] = shared.comb[index];
    }
  }
  __syncthreads();

  std::uint32_t hidden_base = row * kHidden;
  for (std::uint32_t dimension = tid; dimension < kHidden;
       dimension += kThreads) {
    float output = 0.0f;
#pragma unroll
    for (std::uint32_t copy = 0; copy < kHc; ++copy) {
      output += shared.pre[copy] *
                state[state_base + copy * kHidden + dimension];
    }
    hidden[hidden_base + dimension] = output;
  }
  __syncthreads();

  // Affine layer RMSNorm, preserving rms_norm_rows_fused's reduction and
  // x * rsqrt * weight evaluation order.
  float hidden_square_sum = 0.0f;
  for (std::uint32_t dimension = tid; dimension < kHidden;
       dimension += kThreads) {
    float value = hidden[hidden_base + dimension];
    hidden_square_sum += value * value;
  }
  float hidden_total =
      detail::block_sum_256(hidden_square_sum, shared.reduction);
  if (tid == 0) {
    shared.reduction[0] = detail::rsqrt_approx(
        hidden_total / static_cast<float>(kHidden) + args.layer_rms_eps);
  }
  __syncthreads();
  float hidden_rms = shared.reduction[0];

  for (std::uint32_t dimension = tid; dimension < kHidden;
       dimension += kThreads) {
    normalized[hidden_base + dimension] =
        hidden[hidden_base + dimension] * hidden_rms *
        layer_rms_weight[dimension];
  }
  __syncthreads();

  // Eight-thread subgroups cover all 32 K128 blocks concurrently. Each lane
  // owns 16 values; lane zero preserves the exact UE8M0 scale contract.
  constexpr std::uint32_t kPackLanes = 8u;
  const std::uint32_t scale_block = tid / kPackLanes;
  const std::uint32_t pack_lane = tid & (kPackLanes - 1u);
  const std::uint32_t block_start =
      hidden_base + scale_block * kFp8Block;
  float amax = 1.0e-4f;
#pragma unroll
  for (std::uint32_t index = pack_lane; index < kFp8Block;
       index += kPackLanes) {
    amax = fmaxf(amax, fabsf(normalized[block_start + index]));
  }
#pragma unroll
  for (std::uint32_t delta = kPackLanes / 2u; delta > 0u; delta >>= 1u) {
    const float other =
        __shfl_down_sync(0xffffffffu, amax, delta, kPackLanes);
    if (pack_lane < delta) {
      amax = fmaxf(amax, other);
    }
  }
  std::uint32_t scale_byte = detail::e8m0_scale_byte_for_amax(amax);
  scale_byte =
      __shfl_sync(0xffffffffu, scale_byte, 0u, kPackLanes);
  if (pack_lane == 0u) {
    scales[row * kScaleColumns + scale_block] =
        static_cast<std::uint8_t>(scale_byte);
  }
  const float scale =
      detail::ue8m0_to_float(static_cast<std::uint8_t>(scale_byte));
#pragma unroll
  for (std::uint32_t index = pack_lane; index < kFp8Block;
       index += kPackLanes) {
    const float scaled = detail::clamp_fp8_input(
        normalized[block_start + index] / scale);
    packed[block_start + index] =
        detail::quantize_fp8_e4m3fn_byte(scaled);
  }
}

inline cudaError_t launch_hc_pre_rmsnorm_fp8(
    HcPreRmsNormFp8Args const &args) noexcept {
  if (!validate_hc_pre_rmsnorm_fp8(args)) {
    return cudaErrorInvalidValue;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(
      static_cast<std::uintptr_t>(args.stream));
  hc_pre_rmsnorm_fp8_kernel<<<args.rows, kThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

} // namespace sm121
} // namespace ferrule

#endif // FERRULE_CUDA_NATIVE_CUTLASS_SM121_HC_PRODUCER_CUH_
