#pragma once

#include "cutlass/fp8.cuh"
#include "cutlass/target.cuh"

#include <cstdint>

namespace ferrule::cuda::cutlass::operators::cooperative {

// Conservative provider-wide ceiling. Schedules with tighter resource bounds
// may impose a lower limit or query occupancy at launch time.
inline constexpr std::uint32_t kDefaultBlockLimit = 160u;

} // namespace ferrule::cuda::cutlass::operators::cooperative


#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/numeric/int.hpp>


#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ferrule::cuda::cutlass::operators::hyper_connection_producer {

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
struct HcPreRmsNormFp8Args {
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
  std::uint64_t function_row_major_f32;
  std::uint64_t hc_scale_f32;
  std::uint64_t hc_base_f32;
  std::uint64_t layer_rms_weight_f32;
  std::uint64_t mix_f32;
  std::uint64_t workspace;
  std::uint64_t workspace_bytes;
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
static_assert(sizeof(HcPreRmsNormFp8Args) == 168);
static_assert(offsetof(HcPreRmsNormFp8Args, state_f32) == 40);
static_assert(offsetof(HcPreRmsNormFp8Args, mix_f32) == 80);
static_assert(offsetof(HcPreRmsNormFp8Args, stream) == 160);

namespace detail {

inline bool aligned_device_address(std::uint64_t address,
                                   std::uint64_t alignment) noexcept {
  return address != 0 && (address & (alignment - 1)) == 0;
}

__device__ __forceinline__ bool finite_f32(float value) {
  return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ float rsqrt_approx(float value) {
  return rsqrtf(value);
}

__device__ __forceinline__ float fast_exp(float value) { return expf(value); }

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

__device__ __forceinline__ float fast_sigmoid(float value) {
  return 1.0f / (1.0f + fast_exp(-value));
}

__device__ __forceinline__ float block_sum_256(float value, float *reduction) {
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

__device__ __forceinline__ float
nearest_fp8_subnormal_positive(float magnitude) {
  float step = pow2_via_exp(-9.0f);
  float mantissa = roundf(magnitude / step);
  mantissa = mantissa < 0.0f ? 0.0f : mantissa;
  mantissa = mantissa > 7.0f ? 7.0f : mantissa;
  return mantissa * step;
}

__device__ __forceinline__ float nearest_fp8_e4m3fn_positive(float magnitude) {
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

    float candidate = pow2_via_exp(static_cast<float>(candidate_exponent)) *
                      (1.0f + static_cast<float>(mantissa) / 8.0f);
    float error =
        candidate > magnitude ? candidate - magnitude : magnitude - candidate;
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

__device__ __forceinline__ std::uint8_t e8m0_scale_byte_for_amax(float amax) {
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

struct HcLinearPlan {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t input_layout = nullptr;
  cublasLtMatrixLayout_t weight_layout = nullptr;
  cublasLtMatrixLayout_t output_layout = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  std::uint32_t rows = 0u;
  std::size_t workspace_bytes = 0u;

  ~HcLinearPlan() {
    if (input_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(input_layout);
    }
    if (weight_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(weight_layout);
    }
    if (output_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(output_layout);
    }
    if (operation != nullptr) {
      cublasLtMatmulDescDestroy(operation);
    }
    if (handle != nullptr) {
      cublasLtDestroy(handle);
    }
  }

  bool prepare(std::uint32_t requested_rows,
               std::size_t available_workspace_bytes) {
    if (rows == requested_rows &&
        workspace_bytes <= available_workspace_bytes && handle != nullptr) {
      return true;
    }
    if (input_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(input_layout);
      input_layout = nullptr;
    }
    if (weight_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(weight_layout);
      weight_layout = nullptr;
    }
    if (output_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(output_layout);
      output_layout = nullptr;
    }
    if (operation != nullptr) {
      cublasLtMatmulDescDestroy(operation);
      operation = nullptr;
    }
    rows = 0u;
    workspace_bytes = 0u;

    if (handle == nullptr && cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    if (cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
        CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasOperation_t op_t = CUBLAS_OP_T;
    if (cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &op_n,
                                       sizeof(op_n)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &op_t,
                                       sizeof(op_t)) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    if (cublasLtMatrixLayoutCreate(&input_layout, CUDA_R_32F, requested_rows,
                                   kHcHidden,
                                   kHcHidden) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&weight_layout, CUDA_R_32F, kMix, kHcHidden,
                                   kHcHidden) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&output_layout, CUDA_R_32F, requested_rows,
                                   kMix, kMix) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    if (cublasLtMatrixLayoutSetAttribute(
            input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutSetAttribute(
            weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutSetAttribute(
            output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major)) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }

    cublasLtMatmulPreference_t preference = nullptr;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    const cublasStatus_t preference_status =
        cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &available_workspace_bytes, sizeof(available_workspace_bytes));
    cublasLtMatmulHeuristicResult_t candidates[32];
    int candidate_count = 0;
    const cublasStatus_t heuristic_status =
        preference_status == CUBLAS_STATUS_SUCCESS
            ? cublasLtMatmulAlgoGetHeuristic(
                  handle, operation, input_layout, weight_layout, output_layout,
                  output_layout, preference, 32, candidates, &candidate_count)
            : preference_status;
    cublasLtMatmulPreferenceDestroy(preference);
    if (heuristic_status != CUBLAS_STATUS_SUCCESS) {
      return false;
    }

    for (int candidate = 0; candidate < candidate_count; ++candidate) {
      int split_k = 0;
      int reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
      std::size_t written = 0u;
      if (cublasLtMatmulAlgoConfigGetAttribute(
              &candidates[candidate].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
              &split_k, sizeof(split_k), &written) != CUBLAS_STATUS_SUCCESS ||
          cublasLtMatmulAlgoConfigGetAttribute(
              &candidates[candidate].algo,
              CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction,
              sizeof(reduction), &written) != CUBLAS_STATUS_SUCCESS) {
        continue;
      }
      // Split-K changes the FP32 reduction tree and its legal split count is
      // architecture- and CUDA-version-dependent. Keep one K reduction per
      // output so decode and wider batches share the same numerical boundary.
      if (candidates[candidate].state == CUBLAS_STATUS_SUCCESS &&
          split_k == 1 && reduction == CUBLASLT_REDUCTION_SCHEME_NONE &&
          candidates[candidate].workspaceSize <= available_workspace_bytes) {
        algorithm = candidates[candidate].algo;
        rows = requested_rows;
        workspace_bytes = candidates[candidate].workspaceSize;
        return true;
      }
    }
    return false;
  }
};

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
__device__ __forceinline__ void
mix_state_function(SharedStorage &shared, const float *state,
                   const float *function_col_major, std::uint32_t state_base,
                   std::uint32_t tid) {
  constexpr std::uint32_t kWarpSize = 32u;
  constexpr std::uint32_t kWarps = kThreads / kWarpSize;
  constexpr std::uint32_t kOutputsPerWarp = kMix / kWarps;
  const std::uint32_t warp = tid / kWarpSize;
  const std::uint32_t lane = tid & (kWarpSize - 1u);
  float accumulator[kOutputsPerWarp] = {0.0f, 0.0f, 0.0f};

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

#pragma unroll
    for (std::uint32_t output_slot = 0; output_slot < kOutputsPerWarp;
         ++output_slot) {
      const std::uint32_t output = warp + output_slot * kWarps;
#pragma unroll
      for (std::uint32_t column = lane; column < TileColumns;
           column += kWarpSize) {
        accumulator[output_slot] =
            __fmaf_rn(shared.function_tile[column * kMix + output],
                      shared.state_tile[column], accumulator[output_slot]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (std::uint32_t offset = kWarpSize / 2u; offset > 0u; offset >>= 1u) {
#pragma unroll
    for (std::uint32_t output_slot = 0; output_slot < kOutputsPerWarp;
         ++output_slot) {
      accumulator[output_slot] = __fadd_rn(
          accumulator[output_slot],
          __shfl_down_sync(0xffffffffu, accumulator[output_slot], offset));
    }
  }
  if (lane == 0u) {
#pragma unroll
    for (std::uint32_t output_slot = 0; output_slot < kOutputsPerWarp;
         ++output_slot) {
      shared.mix[warp + output_slot * kWarps] = accumulator[output_slot];
    }
  }
  __syncthreads();
}

} // namespace detail

inline bool
validate_hc_pre_rmsnorm_fp8(HcPreRmsNormFp8Args const &args) noexcept {
  if (args.rows == 0 || args.hc != kHc || args.hidden != kHidden ||
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
         detail::aligned_device_address(args.function_row_major_f32, 16) &&
         detail::aligned_device_address(args.hc_scale_f32, 16) &&
         detail::aligned_device_address(args.hc_base_f32, 16) &&
         detail::aligned_device_address(args.layer_rms_weight_f32, 16) &&
         detail::aligned_device_address(args.mix_f32, 16) &&
         detail::aligned_device_address(args.workspace, 16) &&
         args.workspace_bytes >= static_cast<std::uint64_t>(args.rows) *
                                         args.mix * 64u * sizeof(float) +
                                     3u &&
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
  auto const *hc_scale = reinterpret_cast<float const *>(
      static_cast<std::uintptr_t>(args.hc_scale_f32));
  auto const *hc_base = reinterpret_cast<float const *>(
      static_cast<std::uintptr_t>(args.hc_base_f32));
  auto const *layer_rms_weight = reinterpret_cast<float const *>(
      static_cast<std::uintptr_t>(args.layer_rms_weight_f32));
  auto const *mix = reinterpret_cast<float const *>(
      static_cast<std::uintptr_t>(args.mix_f32));
  auto *hidden =
      reinterpret_cast<float *>(static_cast<std::uintptr_t>(args.hidden_f32));
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
  for (std::uint32_t column = tid; column < kHcHidden; column += kThreads) {
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

  if (tid < kMix) {
    shared.mix[tid] =
        mix[static_cast<std::uint64_t>(row) * kMix + tid] * state_rms;
  }
  __syncthreads();

  if (tid == 0) {
    std::uint32_t split_base = row * kHc;
    std::uint32_t comb_base = row * kHc * kHc;

#pragma unroll
    for (std::uint32_t copy = 0; copy < kHc; ++copy) {
      float pre =
          detail::fast_sigmoid(shared.mix[copy] * hc_scale[0] + hc_base[copy]) +
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
          shared.comb[comb_row * kHc + column] /= column_sum + args.hc_eps;
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
      output +=
          shared.pre[copy] * state[state_base + copy * kHidden + dimension];
    }
    hidden[hidden_base + dimension] = detail::bf16_round(output);
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
        detail::bf16_round(hidden[hidden_base + dimension] * hidden_rms *
                           layer_rms_weight[dimension]);
  }
  __syncthreads();

  // Eight-thread subgroups cover all 32 K128 blocks concurrently. Each lane
  // owns 16 values; lane zero preserves the exact UE8M0 scale contract.
  constexpr std::uint32_t kPackLanes = 8u;
  const std::uint32_t scale_block = tid / kPackLanes;
  const std::uint32_t pack_lane = tid & (kPackLanes - 1u);
  const std::uint32_t block_start = hidden_base + scale_block * kFp8Block;
  float amax = 1.0e-4f;
#pragma unroll
  for (std::uint32_t index = pack_lane; index < kFp8Block;
       index += kPackLanes) {
    amax = fmaxf(amax, fabsf(normalized[block_start + index]));
  }
#pragma unroll
  for (std::uint32_t delta = kPackLanes / 2u; delta > 0u; delta >>= 1u) {
    const float other = __shfl_down_sync(0xffffffffu, amax, delta, kPackLanes);
    if (pack_lane < delta) {
      amax = fmaxf(amax, other);
    }
  }
  std::uint32_t scale_byte = detail::e8m0_scale_byte_for_amax(amax);
  scale_byte = __shfl_sync(0xffffffffu, scale_byte, 0u, kPackLanes);
  if (pack_lane == 0u) {
    scales[row * kScaleColumns + scale_block] =
        static_cast<std::uint8_t>(scale_byte);
  }
  const float scale =
      detail::ue8m0_to_float(static_cast<std::uint8_t>(scale_byte));
#pragma unroll
  for (std::uint32_t index = pack_lane; index < kFp8Block;
       index += kPackLanes) {
    const float scaled =
        detail::clamp_fp8_input(normalized[block_start + index] / scale);
    packed[block_start + index] = detail::quantize_fp8_e4m3fn_byte(scaled);
  }
}

inline cudaError_t
launch_hc_pre_rmsnorm_fp8(HcPreRmsNormFp8Args const &args) noexcept {
  if (!validate_hc_pre_rmsnorm_fp8(args)) {
    return cudaErrorInvalidValue;
  }

  cudaStream_t stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args.stream));
  thread_local detail::HcLinearPlan plan;
  if (!plan.prepare(args.rows,
                    static_cast<std::size_t>(args.workspace_bytes))) {
    return cudaErrorNotSupported;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const auto *state = reinterpret_cast<const float *>(
      static_cast<std::uintptr_t>(args.state_f32));
  const auto *weight = reinterpret_cast<const float *>(
      static_cast<std::uintptr_t>(args.function_row_major_f32));
  auto *mix =
      reinterpret_cast<float *>(static_cast<std::uintptr_t>(args.mix_f32));
  void *workspace =
      reinterpret_cast<void *>(static_cast<std::uintptr_t>(args.workspace));
  const cublasStatus_t matmul_status =
      cublasLtMatmul(plan.handle, plan.operation, &alpha, state,
                     plan.input_layout, weight, plan.weight_layout, &beta, mix,
                     plan.output_layout, mix, plan.output_layout,
                     &plan.algorithm, workspace, plan.workspace_bytes, stream);
  if (matmul_status != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorLaunchFailure;
  }
  hc_pre_rmsnorm_fp8_kernel<<<args.rows, kThreads, 0, stream>>>(args);
  return cudaGetLastError();
}

} // namespace ferrule::cuda::cutlass::operators::hyper_connection_producer


#include <cooperative_groups.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>


#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ferrule::cuda::cutlass::operators::mla_output {

inline constexpr std::uint32_t kMmaRows = 8u;
inline constexpr std::uint32_t kMmaColumns = 16u;
inline constexpr std::uint32_t kKTile = 16u;
inline constexpr std::uint32_t kWarpSize = 32u;
inline constexpr std::uint32_t kWarps = 4u;
inline constexpr std::uint32_t kThreads = kWarpSize * kWarps;
inline constexpr std::uint32_t kCooperativeBlockLimit =
    operators::cooperative::kDefaultBlockLimit;
inline constexpr std::uint32_t kMaxRows = 65535u * kMmaRows;

// One semantic MLA output transaction:
//
//   context_f32 [rows, context_size]
//     -> grouped output-A FP8/E8M0 [latent_size, group_input_size]
//     -> latent_bf16 [rows, latent_size]
//     -> latent FP8/E8M0 pack [rows, latent_size]
//     -> output-B FP8/E8M0 [hidden_size, latent_size]
//     -> output_f32 [rows, hidden_size]
//
// The caller-owned latent tensor is both the numerical BF16 boundary and the
// warmed cross-kernel scratch. The provider uses a three-phase same-stream path
// for one row and one cooperative launch for wider inputs. Neither path uses a
// host-side GEMM, allocation, synchronization, or fallback.
struct Args {
  std::uint32_t rows;
  std::uint32_t context_size;
  std::uint32_t groups;
  std::uint32_t group_input_size;
  std::uint32_t rank;
  std::uint32_t latent_size;
  std::uint32_t hidden_size;
  std::uint32_t output_a_scale_cols;
  std::uint32_t reserved;

  std::uint64_t context_f32;
  std::uint64_t output_a_weight_fp8;
  std::uint64_t output_a_weight_ue8m0;
  std::uint64_t output_b_weight_fp8;
  std::uint64_t output_b_weight_ue8m0;
  std::uint64_t latent_bf16;
  std::uint64_t latent_fp8;
  std::uint64_t latent_ue8m0;
  std::uint64_t output_f32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 120u, "MLA output POD ABI changed");
static_assert(alignof(Args) == 8u);
static_assert(offsetof(Args, rows) == 0u);
static_assert(offsetof(Args, reserved) == 32u);
static_assert(offsetof(Args, context_f32) == 40u);
static_assert(offsetof(Args, stream) == 112u);

struct Binding {
  const float *context;
  const std::uint8_t *output_a_weight;
  const std::uint8_t *output_a_scales;
  const std::uint8_t *output_b_weight;
  const std::uint8_t *output_b_scales;
  std::uint16_t *latent;
  std::uint8_t *latent_fp8;
  std::uint8_t *latent_scales;
  float *output;
};

static_assert(std::is_trivially_copyable_v<Binding>);

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidArgument = 2,
  kLaunchFailed = 3,
};

namespace detail {

using Bf16Mma = cute::SM80_16x8x16_F32BF16BF16F32_TN;

struct alignas(16) WarpStage {
  alignas(16) std::uint16_t weight[kMmaColumns * kKTile];
  alignas(16) std::uint16_t activation[kKTile * kMmaRows];
};

static_assert(sizeof(WarpStage) == 768u);

inline constexpr bool aligned(std::uint64_t address, std::uint64_t alignment) {
  return address != 0u && (address & (alignment - 1u)) == 0u;
}

template <class T>
__device__ __forceinline__ T *device_pointer(std::uint64_t address) {
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

__device__ __forceinline__ float ue8m0_to_float(std::uint8_t value) {
  const std::uint32_t bits =
      value == 0u ? (1u << 22) : (static_cast<std::uint32_t>(value) << 23);
  return __uint_as_float(bits);
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(std::uint8_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x80u) << 24;
  const std::uint32_t exponent = (value >> 3) & 0x0fu;
  const std::uint32_t mantissa = value & 0x07u;
  if (exponent == 0u) {
    if (mantissa == 0u) {
      return __uint_as_float(sign);
    }
    const float magnitude = static_cast<float>(mantissa) * (1.0f / 512.0f);
    return sign != 0u ? -magnitude : magnitude;
  }
  if (exponent == 0x0fu && mantissa == 0x07u) {
    return __int_as_float(0x7fffffffu);
  }
  return __uint_as_float(sign | ((exponent + 120u) << 23) | (mantissa << 20));
}

__device__ __forceinline__ void
load_weight_fragment(const std::uint16_t *shared, std::uint32_t lane,
                     std::uint32_t (&fragment)[4]) {
  const std::uint32_t quad = lane >> 3;
  const std::uint32_t row = (lane & 7u) + ((quad & 1u) != 0u ? 8u : 0u);
  const std::uint32_t column_bytes = quad >= 2u ? 16u : 0u;
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(shared);
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      bytes + row * 32u + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1], fragment[2],
                                fragment[3]);
}

__device__ __forceinline__ void
load_activation_fragment(const std::uint16_t *shared, std::uint32_t lane,
                         std::uint32_t (&fragment)[2]) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(shared);
  auto const &source =
      *reinterpret_cast<const cute::uint128_t *>(bytes + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void mma_bf16(float (&accumulator)[4],
                                         const std::uint32_t (&weight)[4],
                                         const std::uint32_t (&activation)[2]) {
  Bf16Mma::fma(accumulator[0], accumulator[1], accumulator[2], accumulator[3],
               weight[0], weight[1], weight[2], weight[3], activation[0],
               activation[1], accumulator[0], accumulator[1], accumulator[2],
               accumulator[3]);
}

__device__ __forceinline__ void mma_fp8(float (&accumulator)[4],
                                        const std::uint32_t (&weight)[4],
                                        const std::uint32_t (&activation)[2]) {
  architectures::sm89::mma_sync_f32_e4m3_e4m3_m16n8k32(accumulator, weight, activation);
}

__device__ __forceinline__ std::uint8_t ue8m0_scale_byte_for_amax(float amax) {
  if (!isfinite(amax) || amax <= 0.0f) {
    return 127u;
  }
  const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
  const int encoded = exponent + 127;
  return static_cast<std::uint8_t>(
      encoded < 0 ? 0 : (encoded > 255 ? 255 : encoded));
}

__device__ __forceinline__ std::uint8_t quantize_fp8(float value) {
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ void
stage_f32_activation(std::uint16_t *stage, const float *activation,
                     std::uint32_t row_base, std::uint32_t rows,
                     std::uint32_t row_stride, std::uint32_t column_base,
                     std::uint32_t lane) {
  for (std::uint32_t linear = lane; linear < kKTile * kMmaRows;
       linear += kWarpSize) {
    const std::uint32_t k_local = linear >> 3;
    const std::uint32_t row_local = linear & 7u;
    const std::uint32_t row = row_base + row_local;
    const float value =
        row < rows ? activation[static_cast<std::uint64_t>(row) * row_stride +
                                column_base + k_local]
                   : 0.0f;
    stage[linear] = f32_to_bf16_rne(value);
  }
}

__device__ __forceinline__ void
stage_output_a_weight(std::uint16_t *stage, const Binding &binding,
                      const Args &args, std::uint32_t channel_base,
                      std::uint32_t k_base, std::uint32_t lane) {
  const std::uint32_t local_channel = lane >> 1;
  const std::uint32_t half = lane & 1u;
  const std::uint32_t channel = channel_base + local_channel;
  auto *destination = stage + local_channel * kKTile + half * 8u;
  if (channel >= args.latent_size) {
    *reinterpret_cast<uint4 *>(destination) = make_uint4(0u, 0u, 0u, 0u);
    return;
  }
  const std::uint64_t source_offset =
      static_cast<std::uint64_t>(channel) * args.group_input_size + k_base +
      half * 8u;
  const std::uint64_t packed = *reinterpret_cast<const std::uint64_t *>(
      binding.output_a_weight + source_offset);
  const std::uint64_t scale_offset =
      static_cast<std::uint64_t>(channel / 128u) * args.output_a_scale_cols +
      k_base / 128u;
  const float scale = ue8m0_to_float(binding.output_a_scales[scale_offset]);
#pragma unroll
  for (std::uint32_t element = 0u; element < 8u; ++element) {
    const std::uint8_t value =
        static_cast<std::uint8_t>(packed >> (element * 8u));
    destination[element] = f32_to_bf16_rne(fp8_e4m3fn_to_float(value) * scale);
  }
}

__device__ __forceinline__ void
stage_output_b_weight(std::uint16_t *stage, const std::uint8_t *weight,
                      const Args &args, std::uint32_t channel_base,
                      std::uint32_t k_base, std::uint32_t lane) {
  if (lane >= kMmaColumns) {
    return;
  }
  auto *destination =
      reinterpret_cast<uint4 *>(reinterpret_cast<std::uint8_t *>(stage) +
                                static_cast<std::uint64_t>(lane) * 32u);
  const std::uint32_t channel = channel_base + lane;
  if (channel < args.hidden_size) {
    auto *source = reinterpret_cast<const uint4 *>(
        weight + static_cast<std::uint64_t>(channel) * args.latent_size +
        k_base);
    destination[0] = source[0];
    destination[1] = source[1];
  } else {
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    destination[0] = zero;
    destination[1] = zero;
  }
}

__device__ __forceinline__ void
stage_output_b_activation(std::uint16_t *stage, const Binding &binding,
                          const Args &args, std::uint32_t row_base,
                          std::uint32_t k_base, std::uint32_t lane) {
  auto *bytes = reinterpret_cast<std::uint8_t *>(stage);
  for (std::uint32_t linear = lane; linear < 128u; linear += kWarpSize) {
    const std::uint32_t k_pair = linear >> 3;
    const std::uint32_t row_local = linear & 7u;
    const std::uint32_t destination = k_pair * 16u + row_local * 2u;
    const std::uint32_t row = row_base + row_local;
    if (row < args.rows) {
      const std::uint64_t source =
          static_cast<std::uint64_t>(row) * args.latent_size + k_base +
          k_pair * 2u;
      bytes[destination] = binding.latent_fp8[source];
      bytes[destination + 1u] = binding.latent_fp8[source + 1u];
    } else {
      bytes[destination] = 0u;
      bytes[destination + 1u] = 0u;
    }
  }
}

__device__ __forceinline__ void
pack_latent_task(const Args &args, const Binding &binding, std::uint32_t row,
                 std::uint32_t scale_block, std::uint32_t lane) {
  const std::uint64_t base =
      static_cast<std::uint64_t>(row) * args.latent_size + scale_block * 128u;
  float values[4];
  float amax = 1.0e-4f;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const float value =
        bf16_to_f32(binding.latent[base + lane + element * kWarpSize]);
    values[element] = value;
    amax = fmaxf(amax, fabsf(value));
  }
#pragma unroll
  for (std::uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta));
  }
  std::uint32_t scale_byte = ue8m0_scale_byte_for_amax(amax);
  scale_byte = __shfl_sync(0xffffffffu, scale_byte, 0);
  if (lane == 0u) {
    binding.latent_scales[static_cast<std::uint64_t>(row) *
                              (args.latent_size / 128u) +
                          scale_block] = static_cast<std::uint8_t>(scale_byte);
  }
  const float scale = ue8m0_to_float(static_cast<std::uint8_t>(scale_byte));
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    binding.latent_fp8[base + lane + element * kWarpSize] =
        quantize_fp8(values[element] / scale);
  }
}

__device__ __forceinline__ void
store_bf16_tile(std::uint16_t *output, std::uint32_t output_columns,
                std::uint32_t rows, std::uint32_t row_base,
                std::uint32_t channel_base, std::uint32_t lane,
                const float (&accumulator)[4]) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row = row_base + row_pair * 2u + (element & 1u);
    if (row < rows && channel < output_columns) {
      output[static_cast<std::uint64_t>(row) * output_columns + channel] =
          f32_to_bf16_rne(accumulator[element]);
    }
  }
}

__device__ __forceinline__ void
store_f32_bf16_tile(float *output, std::uint32_t output_columns,
                    std::uint32_t rows, std::uint32_t row_base,
                    std::uint32_t channel_base, std::uint32_t lane,
                    const float (&accumulator)[4]) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row = row_base + row_pair * 2u + (element & 1u);
    if (row < rows && channel < output_columns) {
      output[static_cast<std::uint64_t>(row) * output_columns + channel] =
          bf16_to_f32(f32_to_bf16_rne(accumulator[element]));
    }
  }
}

__device__ __forceinline__ void
output_a_task(const Args &args, const Binding &binding, WarpStage &stage,
              std::uint32_t row_base, std::uint32_t channel_base,
              std::uint32_t lane) {
  const std::uint32_t group = channel_base / args.rank;
  const std::uint32_t context_group_base = group * args.group_input_size;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (std::uint32_t k_base = 0u; k_base < args.group_input_size;
       k_base += kKTile) {
    stage_output_a_weight(stage.weight, binding, args, channel_base, k_base,
                          lane);
    stage_f32_activation(stage.activation, binding.context, row_base, args.rows,
                         args.context_size, context_group_base + k_base, lane);
    __syncwarp();
    std::uint32_t weight_fragment[4];
    std::uint32_t activation_fragment[2];
    load_weight_fragment(stage.weight, lane, weight_fragment);
    load_activation_fragment(stage.activation, lane, activation_fragment);
    mma_bf16(accumulator, weight_fragment, activation_fragment);
    __syncwarp();
  }
  store_bf16_tile(binding.latent, args.latent_size, args.rows, row_base,
                  channel_base, lane, accumulator);
}

__device__ __forceinline__ void
output_b_task(const Args &args, const Binding &binding, WarpStage &stage,
              std::uint32_t row_base, std::uint32_t channel_base,
              std::uint32_t lane) {
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const std::uint32_t scale_cols = args.latent_size / 128u;
  for (std::uint32_t scale_block = 0u; scale_block < scale_cols;
       ++scale_block) {
    float block_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const std::uint32_t block_base = scale_block * 128u;
#pragma unroll
    for (std::uint32_t k_sub = 0u; k_sub < 128u; k_sub += 32u) {
      stage_output_b_weight(stage.weight, binding.output_b_weight, args,
                            channel_base, block_base + k_sub, lane);
      stage_output_b_activation(stage.activation, binding, args, row_base,
                                block_base + k_sub, lane);
      __syncwarp();
      std::uint32_t weight_fragment[4];
      std::uint32_t activation_fragment[2];
      load_weight_fragment(stage.weight, lane, weight_fragment);
      load_activation_fragment(stage.activation, lane, activation_fragment);
      mma_fp8(block_accumulator, weight_fragment, activation_fragment);
      __syncwarp();
    }
    const float weight_scale = ue8m0_to_float(
        binding
            .output_b_scales[static_cast<std::uint64_t>(channel_base / 128u) *
                                 scale_cols +
                             scale_block]);
    const std::uint32_t row_pair = lane & 3u;
#pragma unroll
    for (std::uint32_t element = 0u; element < 4u; ++element) {
      const std::uint32_t row = row_base + row_pair * 2u + (element & 1u);
      if (row < args.rows) {
        const float activation_scale = ue8m0_to_float(
            binding.latent_scales[static_cast<std::uint64_t>(row) * scale_cols +
                                  scale_block]);
        accumulator[element] +=
            block_accumulator[element] * weight_scale * activation_scale;
      }
    }
  }
  store_f32_bf16_tile(binding.output, args.hidden_size, args.rows, row_base,
                      channel_base, lane, accumulator);
}

__global__ __launch_bounds__(kThreads,
                             1) void cooperative_kernel(Args args,
                                                        Binding binding) {
  extern __shared__ __align__(16) std::uint8_t storage[];
  auto *stages = reinterpret_cast<WarpStage *>(storage);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp;
  const std::uint32_t warp_stride = gridDim.x * kWarps;
  const std::uint32_t row_tiles = (args.rows + kMmaRows - 1u) / kMmaRows;

  const std::uint32_t output_a_channel_tiles =
      (args.latent_size + kMmaColumns - 1u) / kMmaColumns;
  const std::uint32_t output_a_tasks = row_tiles * output_a_channel_tiles;
  for (std::uint32_t task = global_warp; task < output_a_tasks;
       task += warp_stride) {
    const std::uint32_t row_tile = task / output_a_channel_tiles;
    const std::uint32_t channel_tile = task - row_tile * output_a_channel_tiles;
    output_a_task(args, binding, stages[warp], row_tile * kMmaRows,
                  channel_tile * kMmaColumns, lane);
  }

  cooperative_groups::this_grid().sync();

  const std::uint32_t latent_scale_cols = args.latent_size / 128u;
  const std::uint32_t pack_tasks = args.rows * latent_scale_cols;
  for (std::uint32_t task = global_warp; task < pack_tasks;
       task += warp_stride) {
    const std::uint32_t row = task / latent_scale_cols;
    const std::uint32_t scale_block = task - row * latent_scale_cols;
    pack_latent_task(args, binding, row, scale_block, lane);
  }

  cooperative_groups::this_grid().sync();

  const std::uint32_t output_b_channel_tiles =
      (args.hidden_size + kMmaColumns - 1u) / kMmaColumns;
  const std::uint32_t output_b_tasks = row_tiles * output_b_channel_tiles;
  for (std::uint32_t task = global_warp; task < output_b_tasks;
       task += warp_stride) {
    const std::uint32_t row_tile = task / output_b_channel_tiles;
    const std::uint32_t channel_tile = task - row_tile * output_b_channel_tiles;
    output_b_task(args, binding, stages[warp], row_tile * kMmaRows,
                  channel_tile * kMmaColumns, lane);
  }
}

__global__ __launch_bounds__(kThreads, 1) void output_a_single_row_kernel(
    Args args, Binding binding) {
  extern __shared__ __align__(16) std::uint8_t storage[];
  auto *stages = reinterpret_cast<WarpStage *>(storage);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp;
  const std::uint32_t warp_stride = gridDim.x * kWarps;
  const std::uint32_t tasks =
      (args.latent_size + kMmaColumns - 1u) / kMmaColumns;
  for (std::uint32_t task = global_warp; task < tasks; task += warp_stride) {
    output_a_task(args, binding, stages[warp], 0u, task * kMmaColumns, lane);
  }
}

__global__ __launch_bounds__(kWarpSize, 1) void pack_latent_single_row_kernel(
    Args args, Binding binding) {
  const std::uint32_t task = blockIdx.x;
  const std::uint32_t tasks = args.latent_size / 128u;
  if (task < tasks) {
    pack_latent_task(args, binding, 0u, task, threadIdx.x);
  }
}

__global__ __launch_bounds__(kThreads, 1) void output_b_single_row_kernel(
    Args args, Binding binding) {
  extern __shared__ __align__(16) std::uint8_t storage[];
  auto *stages = reinterpret_cast<WarpStage *>(storage);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp;
  const std::uint32_t warp_stride = gridDim.x * kWarps;
  const std::uint32_t tasks =
      (args.hidden_size + kMmaColumns - 1u) / kMmaColumns;
  for (std::uint32_t task = global_warp; task < tasks; task += warp_stride) {
    output_b_task(args, binding, stages[warp], 0u, task * kMmaColumns, lane);
  }
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr) {
    return Status::kInvalidArgument;
  }
  if (args->rows == 0u || args->rows > kMaxRows || args->groups == 0u ||
      args->group_input_size == 0u || args->rank == 0u ||
      args->latent_size == 0u || args->hidden_size == 0u ||
      args->reserved != 0u ||
      args->context_size != args->groups * args->group_input_size ||
      args->latent_size != args->groups * args->rank ||
      args->output_a_scale_cols != args->group_input_size / 128u ||
      (args->group_input_size % 128u) != 0u || (args->rank % 128u) != 0u ||
      (args->latent_size % 128u) != 0u) {
    return Status::kInvalidArgument;
  }
  const bool pointers_valid =
      detail::aligned(args->context_f32, 16u) &&
      detail::aligned(args->output_a_weight_fp8, 16u) &&
      detail::aligned(args->output_a_weight_ue8m0, 16u) &&
      detail::aligned(args->output_b_weight_fp8, 16u) &&
      detail::aligned(args->output_b_weight_ue8m0, 16u) &&
      detail::aligned(args->latent_bf16, 16u) &&
      detail::aligned(args->latent_fp8, 16u) &&
      detail::aligned(args->latent_ue8m0, 16u) &&
      detail::aligned(args->output_f32, 16u) && args->stream != 0u;
  return pointers_valid ? Status::kSuccess : Status::kInvalidArgument;
}

inline std::size_t required_shared_storage_bytes() {
  return sizeof(detail::WarpStage) * kWarps;
}

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));
  const Binding binding{
      reinterpret_cast<const float *>(
          static_cast<std::uintptr_t>(args->context_f32)),
      reinterpret_cast<const std::uint8_t *>(
          static_cast<std::uintptr_t>(args->output_a_weight_fp8)),
      reinterpret_cast<const std::uint8_t *>(
          static_cast<std::uintptr_t>(args->output_a_weight_ue8m0)),
      reinterpret_cast<const std::uint8_t *>(
          static_cast<std::uintptr_t>(args->output_b_weight_fp8)),
      reinterpret_cast<const std::uint8_t *>(
          static_cast<std::uintptr_t>(args->output_b_weight_ue8m0)),
      reinterpret_cast<std::uint16_t *>(
          static_cast<std::uintptr_t>(args->latent_bf16)),
      reinterpret_cast<std::uint8_t *>(
          static_cast<std::uintptr_t>(args->latent_fp8)),
      reinterpret_cast<std::uint8_t *>(
          static_cast<std::uintptr_t>(args->latent_ue8m0)),
      reinterpret_cast<float *>(static_cast<std::uintptr_t>(args->output_f32)),
  };

  if (args->rows == 1u) {
    const std::uint32_t output_a_tasks =
        (args->latent_size + kMmaColumns - 1u) / kMmaColumns;
    const std::uint32_t output_a_blocks =
        (output_a_tasks + kWarps - 1u) / kWarps;
    detail::output_a_single_row_kernel<<<
        output_a_blocks, kThreads, required_shared_storage_bytes(), stream>>>(
        *args, binding);
    if (cudaGetLastError() != cudaSuccess) {
      return Status::kLaunchFailed;
    }

    const std::uint32_t pack_blocks = args->latent_size / 128u;
    detail::
        pack_latent_single_row_kernel<<<pack_blocks, kWarpSize, 0u, stream>>>(
            *args, binding);
    if (cudaGetLastError() != cudaSuccess) {
      return Status::kLaunchFailed;
    }

    const std::uint32_t output_b_tasks =
        (args->hidden_size + kMmaColumns - 1u) / kMmaColumns;
    const std::uint32_t output_b_blocks =
        (output_b_tasks + kWarps - 1u) / kWarps;
    detail::output_b_single_row_kernel<<<
        output_b_blocks, kThreads, required_shared_storage_bytes(), stream>>>(
        *args, binding);
    return cudaGetLastError() == cudaSuccess ? Status::kSuccess
                                             : Status::kLaunchFailed;
  }

  const std::uint32_t row_tiles = (args->rows + kMmaRows - 1u) / kMmaRows;
  const std::uint32_t output_a_tasks =
      row_tiles * ((args->latent_size + kMmaColumns - 1u) / kMmaColumns);
  const std::uint32_t pack_tasks = args->rows * (args->latent_size / 128u);
  const std::uint32_t output_b_tasks =
      row_tiles * ((args->hidden_size + kMmaColumns - 1u) / kMmaColumns);
  const std::uint32_t warp_tasks =
      max(output_a_tasks, max(pack_tasks, output_b_tasks));
  const std::uint32_t blocks =
      min(kCooperativeBlockLimit, (warp_tasks + kWarps - 1u) / kWarps);
  void *kernel_args[] = {const_cast<Args *>(args),
                         const_cast<Binding *>(&binding)};
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::cooperative_kernel),
      dim3(blocks, 1u, 1u), dim3(kThreads, 1u, 1u), kernel_args,
      required_shared_storage_bytes(), stream);
  return status == cudaSuccess ? Status::kSuccess : Status::kLaunchFailed;
}

} // namespace ferrule::cuda::cutlass::operators::mla_output


#include <cooperative_groups.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm120.hpp>


#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ferrule::cuda::cutlass::operators::main_project_norm {
inline constexpr std::uint32_t kMmaRows = 8u;
inline constexpr std::uint32_t kMmaColumns = 16u;
inline constexpr std::uint32_t kMmaK = 32u;
inline constexpr std::uint32_t kScaleK = 128u;
inline constexpr std::uint32_t kWarpSize = 32u;
inline constexpr std::uint32_t kWarps = 4u;
inline constexpr std::uint32_t kThreads = kWarpSize * kWarps;
inline constexpr std::uint32_t kCooperativeBlockLimit =
    operators::cooperative::kDefaultBlockLimit;
inline constexpr std::uint32_t kMaxRows = 65535u;

// One semantic stage-zero transaction:
//
//   target_taps_f32 [rows, input_size]
//     -> K128 FP8/E8M0 activation pack
//     -> FP8/E8M0 main projection [output_size, input_size]
//     -> BF16-rounded projected boundary [rows, output_size]
//     -> RMSNorm with F32-exposed BF16 checkpoint weights
//     -> BF16-rounded output_f32 [rows, output_size]
//
// The provider issues one cooperative launch. Ferrule owns the stream, weights,
// output, and graph-stable activation/inverse-RMS scratch. The provider
// performs no allocation, host synchronization, or fallback dispatch.
struct Args {
  std::uint32_t rows;
  std::uint32_t input_size;
  std::uint32_t output_size;
  std::uint32_t scale_cols;
  std::uint32_t reserved0;
  float rms_eps;
  std::uint32_t reserved1;

  std::uint64_t input_f32;
  std::uint64_t activation_fp8;
  std::uint64_t activation_ue8m0;
  std::uint64_t weight_fp8;
  std::uint64_t weight_ue8m0;
  std::uint64_t norm_weight_f32;
  std::uint64_t inv_rms_f32;
  std::uint64_t output_f32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 104u, "main-project/norm POD ABI changed");
static_assert(alignof(Args) == 8u);
static_assert(offsetof(Args, rows) == 0u);
static_assert(offsetof(Args, input_f32) == 32u);
static_assert(offsetof(Args, stream) == 96u);

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidArgument = 2,
  kLaunchFailed = 3,
};

namespace detail {

struct alignas(16) WarpStage {
  alignas(16) std::uint8_t weight[kMmaColumns * kMmaK];
  alignas(16) std::uint8_t activation[kMmaK * kMmaRows];
};

static_assert(sizeof(WarpStage) == 768u);

struct Binding {
  const float *input;
  std::uint8_t *activation;
  std::uint8_t *activation_scales;
  const std::uint8_t *weight;
  const std::uint8_t *weight_scales;
  const float *norm_weight;
  float *inv_rms;
  float *output;
};

static_assert(std::is_trivially_copyable_v<Binding>);

inline constexpr bool aligned(std::uint64_t address, std::uint64_t alignment) {
  return address != 0u && (address & (alignment - 1u)) == 0u;
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

__device__ __forceinline__ float ue8m0_to_float(std::uint8_t value) {
  const std::uint32_t bits =
      value == 0u ? (1u << 22) : (static_cast<std::uint32_t>(value) << 23);
  return __uint_as_float(bits);
}

__device__ __forceinline__ std::uint8_t ue8m0_scale_byte_for_amax(float amax) {
  // Match the reference act_quant path: clamp amax to 1e-4 and round
  // amax / 448 upward to a power of two represented as UE8M0.
  amax = fmaxf(amax, 1.0e-4f);
  const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
  const int encoded = exponent + 127;
  return static_cast<std::uint8_t>(
      encoded < 0 ? 0 : (encoded > 255 ? 255 : encoded));
}

__device__ __forceinline__ std::uint8_t quantize_fp8(float value) {
  value = fminf(fmaxf(value, -448.0f), 448.0f);
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ void
load_weight_fragment(const std::uint8_t *shared, std::uint32_t lane,
                     std::uint32_t (&fragment)[4]) {
  const std::uint32_t quad = lane >> 3;
  const std::uint32_t row = (lane & 7u) + ((quad & 1u) != 0u ? 8u : 0u);
  const std::uint32_t column_bytes = quad >= 2u ? 16u : 0u;
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + row * kMmaK + column_bytes);
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

__device__ __forceinline__ void mma_fp8(float (&accumulator)[4],
                                        const std::uint32_t (&weight)[4],
                                        const std::uint32_t (&activation)[2]) {
  architectures::sm89::mma_sync_f32_e4m3_e4m3_m16n8k32(accumulator, weight, activation);
}

__device__ __forceinline__ void pack_activation_task(const Args &args,
                                                     const Binding &binding,
                                                     std::uint32_t row,
                                                     std::uint32_t scale_block,
                                                     std::uint32_t lane) {
  const std::uint64_t base =
      static_cast<std::uint64_t>(row) * args.input_size + scale_block * kScaleK;
  float values[4];
  float amax = 0.0f;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const float value = binding.input[base + lane + element * kWarpSize];
    values[element] = value;
    amax = fmaxf(amax, fabsf(value));
  }
#pragma unroll
  for (std::uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta));
  }
  std::uint32_t scale_byte = ue8m0_scale_byte_for_amax(amax);
  scale_byte = __shfl_sync(0xffffffffu, scale_byte, 0);
  if (lane == 0u) {
    binding
        .activation_scales[static_cast<std::uint64_t>(row) * args.scale_cols +
                           scale_block] = static_cast<std::uint8_t>(scale_byte);
  }
  const float scale = ue8m0_to_float(static_cast<std::uint8_t>(scale_byte));
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    binding.activation[base + lane + element * kWarpSize] =
        quantize_fp8(values[element] / scale);
  }
}

__device__ __forceinline__ void
stage_weight(WarpStage &stage, const Binding &binding, const Args &args,
             std::uint32_t channel_base, std::uint32_t k_base,
             std::uint32_t lane) {
  const std::uint32_t local_channel = lane >> 1;
  const std::uint32_t half = lane & 1u;
  const std::uint32_t channel = channel_base + local_channel;
  auto *destination = reinterpret_cast<uint4 *>(
      stage.weight + local_channel * kMmaK + half * 16u);
  if (channel < args.output_size) {
    auto const *source = reinterpret_cast<const uint4 *>(
        binding.weight + static_cast<std::uint64_t>(channel) * args.input_size +
        k_base + half * 16u);
    *destination = *source;
  } else {
    *destination = make_uint4(0u, 0u, 0u, 0u);
  }
}

__device__ __forceinline__ void
stage_activation(WarpStage &stage, const Binding &binding, const Args &args,
                 std::uint32_t row_base, std::uint32_t k_base,
                 std::uint32_t lane) {
  if (lane >= kMmaK / 2u) {
    return;
  }
  auto *destination = reinterpret_cast<std::uint16_t *>(stage.activation +
                                                        lane * kMmaRows * 2u);
#pragma unroll
  for (std::uint32_t row_local = 0u; row_local < kMmaRows; ++row_local) {
    const std::uint32_t row = row_base + row_local;
    destination[row_local] =
        row < args.rows
            ? *reinterpret_cast<const std::uint16_t *>(
                  binding.activation +
                  static_cast<std::uint64_t>(row) * args.input_size + k_base +
                  lane * 2u)
            : 0u;
  }
}

__device__ __forceinline__ void
projection_task(const Args &args, const Binding &binding, WarpStage &stage,
                std::uint32_t row_base, std::uint32_t channel_base,
                std::uint32_t lane) {
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (std::uint32_t scale_block = 0u; scale_block < args.scale_cols;
       ++scale_block) {
    float block_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const std::uint32_t block_base = scale_block * kScaleK;
#pragma unroll
    for (std::uint32_t k_sub = 0u; k_sub < kScaleK; k_sub += kMmaK) {
      stage_weight(stage, binding, args, channel_base, block_base + k_sub,
                   lane);
      stage_activation(stage, binding, args, row_base, block_base + k_sub,
                       lane);
      __syncwarp();
      std::uint32_t weight_fragment[4];
      std::uint32_t activation_fragment[2];
      load_weight_fragment(stage.weight, lane, weight_fragment);
      load_activation_fragment(stage.activation, lane, activation_fragment);
      mma_fp8(block_accumulator, weight_fragment, activation_fragment);
      __syncwarp();
    }

    const float weight_scale =
        channel_base < args.output_size
            ? ue8m0_to_float(binding.weight_scales[static_cast<std::uint64_t>(
                                                       channel_base / kScaleK) *
                                                       args.scale_cols +
                                                   scale_block])
            : 0.0f;
    const std::uint32_t row_pair = lane & 3u;
#pragma unroll
    for (std::uint32_t element = 0u; element < 4u; ++element) {
      const std::uint32_t row = row_base + row_pair * 2u + (element & 1u);
      if (row < args.rows) {
        const float activation_scale = ue8m0_to_float(
            binding.activation_scales[static_cast<std::uint64_t>(row) *
                                          args.scale_cols +
                                      scale_block]);
        accumulator[element] +=
            block_accumulator[element] * weight_scale * activation_scale;
      }
    }
  }

  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row = row_base + row_pair * 2u + (element & 1u);
    if (row < args.rows && channel < args.output_size) {
      binding.output[static_cast<std::uint64_t>(row) * args.output_size +
                     channel] =
          bf16_to_f32(f32_to_bf16_rne(accumulator[element]));
    }
  }
}

__device__ __forceinline__ void inverse_rms_task(const Args &args,
                                                 const Binding &binding,
                                                 std::uint32_t row,
                                                 std::uint32_t lane) {
  float sum = 0.0f;
  const std::uint64_t base = static_cast<std::uint64_t>(row) * args.output_size;
  for (std::uint32_t channel = lane; channel < args.output_size;
       channel += kWarpSize) {
    const float value = binding.output[base + channel];
    sum = fmaf(value, value, sum);
  }
#pragma unroll
  for (std::uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    sum += __shfl_down_sync(0xffffffffu, sum, delta);
  }
  if (lane == 0u) {
    binding.inv_rms[row] =
        rsqrtf(sum / static_cast<float>(args.output_size) + args.rms_eps);
  }
}

__device__ __forceinline__ void
normalize_task(const Args &args, const Binding &binding, std::uint32_t row,
               std::uint32_t channel_base, std::uint32_t lane) {
  const std::uint64_t row_base =
      static_cast<std::uint64_t>(row) * args.output_size;
  const float inv_rms = binding.inv_rms[row];
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel = channel_base + lane + element * kWarpSize;
    if (channel < args.output_size) {
      const float normalized = binding.output[row_base + channel] * inv_rms *
                               binding.norm_weight[channel];
      binding.output[row_base + channel] =
          bf16_to_f32(f32_to_bf16_rne(normalized));
    }
  }
}

__global__ __launch_bounds__(kThreads, 1) void kernel(Args args,
                                                      Binding binding) {
  extern __shared__ __align__(16) std::uint8_t storage[];
  auto *stages = reinterpret_cast<WarpStage *>(storage);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp;
  const std::uint32_t warp_stride = gridDim.x * kWarps;

  const std::uint32_t pack_tasks = args.rows * args.scale_cols;
  for (std::uint32_t task = global_warp; task < pack_tasks;
       task += warp_stride) {
    const std::uint32_t row = task / args.scale_cols;
    const std::uint32_t scale_block = task - row * args.scale_cols;
    pack_activation_task(args, binding, row, scale_block, lane);
  }

  cooperative_groups::this_grid().sync();

  const std::uint32_t row_tiles = (args.rows + kMmaRows - 1u) / kMmaRows;
  const std::uint32_t channel_tiles =
      (args.output_size + kMmaColumns - 1u) / kMmaColumns;
  const std::uint32_t projection_tasks = row_tiles * channel_tiles;
  for (std::uint32_t task = global_warp; task < projection_tasks;
       task += warp_stride) {
    const std::uint32_t row_tile = task / channel_tiles;
    const std::uint32_t channel_tile = task - row_tile * channel_tiles;
    projection_task(args, binding, stages[warp], row_tile * kMmaRows,
                    channel_tile * kMmaColumns, lane);
  }

  cooperative_groups::this_grid().sync();

  for (std::uint32_t row = global_warp; row < args.rows; row += warp_stride) {
    inverse_rms_task(args, binding, row, lane);
  }

  cooperative_groups::this_grid().sync();

  const std::uint32_t norm_channel_tiles = args.output_size / kScaleK;
  const std::uint32_t norm_tasks = args.rows * norm_channel_tiles;
  for (std::uint32_t task = global_warp; task < norm_tasks;
       task += warp_stride) {
    const std::uint32_t row = task / norm_channel_tiles;
    const std::uint32_t channel_tile = task - row * norm_channel_tiles;
    normalize_task(args, binding, row, channel_tile * kScaleK, lane);
  }
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr) {
    return Status::kInvalidArgument;
  }
  if (args->rows == 0u || args->rows > kMaxRows || args->input_size == 0u ||
      args->output_size == 0u ||
      args->scale_cols != args->input_size / kScaleK ||
      (args->input_size % kScaleK) != 0u ||
      (args->output_size % kScaleK) != 0u || args->reserved0 != 0u ||
      args->reserved1 != 0u || !(args->rms_eps > 0.0f)) {
    return Status::kInvalidArgument;
  }
  const bool pointers_valid = detail::aligned(args->input_f32, 16u) &&
                              detail::aligned(args->activation_fp8, 16u) &&
                              detail::aligned(args->activation_ue8m0, 16u) &&
                              detail::aligned(args->weight_fp8, 16u) &&
                              detail::aligned(args->weight_ue8m0, 16u) &&
                              detail::aligned(args->norm_weight_f32, 16u) &&
                              detail::aligned(args->inv_rms_f32, 16u) &&
                              detail::aligned(args->output_f32, 16u);
  return pointers_valid ? Status::kSuccess : Status::kInvalidArgument;
}

inline std::size_t required_shared_storage_bytes() {
  return sizeof(detail::WarpStage) * kWarps;
}

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }
  const std::uint32_t pack_tasks = args->rows * args->scale_cols;
  const std::uint32_t row_tiles = (args->rows + kMmaRows - 1u) / kMmaRows;
  const std::uint32_t projection_tasks =
      row_tiles * ((args->output_size + kMmaColumns - 1u) / kMmaColumns);
  const std::uint32_t norm_tasks = args->rows * (args->output_size / kScaleK);
  std::uint32_t warp_tasks =
      pack_tasks > projection_tasks ? pack_tasks : projection_tasks;
  warp_tasks = warp_tasks > args->rows ? warp_tasks : args->rows;
  warp_tasks = warp_tasks > norm_tasks ? warp_tasks : norm_tasks;
  std::uint32_t blocks = (warp_tasks + kWarps - 1u) / kWarps;
  blocks = blocks < kCooperativeBlockLimit ? blocks : kCooperativeBlockLimit;

  const detail::Binding binding{
      reinterpret_cast<const float *>(
          static_cast<std::uintptr_t>(args->input_f32)),
      reinterpret_cast<std::uint8_t *>(
          static_cast<std::uintptr_t>(args->activation_fp8)),
      reinterpret_cast<std::uint8_t *>(
          static_cast<std::uintptr_t>(args->activation_ue8m0)),
      reinterpret_cast<const std::uint8_t *>(
          static_cast<std::uintptr_t>(args->weight_fp8)),
      reinterpret_cast<const std::uint8_t *>(
          static_cast<std::uintptr_t>(args->weight_ue8m0)),
      reinterpret_cast<const float *>(
          static_cast<std::uintptr_t>(args->norm_weight_f32)),
      reinterpret_cast<float *>(static_cast<std::uintptr_t>(args->inv_rms_f32)),
      reinterpret_cast<float *>(static_cast<std::uintptr_t>(args->output_f32)),
  };
  void *kernel_args[] = {const_cast<Args *>(args),
                         const_cast<detail::Binding *>(&binding)};
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::kernel), dim3(blocks, 1u, 1u),
      dim3(kThreads, 1u, 1u), kernel_args, required_shared_storage_bytes(),
      stream);
  return status == cudaSuccess ? Status::kSuccess : Status::kLaunchFailed;
}

} // namespace ferrule::cuda::cutlass::operators::main_project_norm
