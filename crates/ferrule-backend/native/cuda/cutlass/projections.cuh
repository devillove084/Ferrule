#pragma once

#include "cutlass/fp8.cuh"
#include "cutlass/target.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ferrule::cuda::cutlass::operators::bf16_compressor {

// Private native contract. The provider verifies this POD against the public C
// ABI before exposing any binding.
struct Args {
  std::uint32_t m;
  std::uint32_t n1;
  std::uint32_t n2;
  std::uint32_t k;
  std::uint32_t reserved;

  std::uint64_t activation_f32;
  std::uint64_t projection1_weight_bf16;
  std::uint64_t projection2_weight_bf16;
  std::uint64_t projection1_output_f32;
  std::uint64_t projection2_output_f32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 72u, "BF16 compressor POD ABI changed");
static_assert(alignof(Args) == 8u);
static_assert(offsetof(Args, m) == 0u);
static_assert(offsetof(Args, reserved) == 16u);
static_assert(offsetof(Args, activation_f32) == 24u);
static_assert(offsetof(Args, stream) == 64u);

enum class ValidationResult : std::uint32_t {
  kSuccess = 0u,
  kNullPointer,
  kMisalignedPointer,
  kUnsupportedM,
  kUnsupportedShape,
  kInvalidReserved,
};

inline constexpr std::uint32_t kKTile = 16u;
inline constexpr std::uint32_t kCtaRows = 8u;
inline constexpr std::uint32_t kMaximumRows = kCtaRows * 65535u;

inline constexpr bool aligned_device_address(std::uint64_t address,
                                             std::uint64_t alignment) {
  return address != 0u && (address & (alignment - 1u)) == 0u;
}

inline ValidationResult validate(const Args &args) noexcept {
  if (args.activation_f32 == 0u || args.projection1_weight_bf16 == 0u ||
      args.projection2_weight_bf16 == 0u || args.projection1_output_f32 == 0u ||
      args.projection2_output_f32 == 0u) {
    return ValidationResult::kNullPointer;
  }
  if (!aligned_device_address(args.activation_f32, 16u) ||
      !aligned_device_address(args.projection1_weight_bf16, 16u) ||
      !aligned_device_address(args.projection2_weight_bf16, 16u) ||
      !aligned_device_address(args.projection1_output_f32, 16u) ||
      !aligned_device_address(args.projection2_output_f32, 16u)) {
    return ValidationResult::kMisalignedPointer;
  }
  if (args.m == 0u || args.m > kMaximumRows) {
    return ValidationResult::kUnsupportedM;
  }
  if (args.n1 == 0u || args.n2 == 0u || args.k == 0u ||
      (args.k % kKTile) != 0u) {
    return ValidationResult::kUnsupportedShape;
  }
  if (args.reserved != 0u) {
    return ValidationResult::kInvalidReserved;
  }
  return ValidationResult::kSuccess;
}

} // namespace ferrule::cuda::cutlass::operators::bf16_compressor


#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>


#include <cstddef>
#include <cstdint>
#include <type_traits>

// One-launch BF16 compressor prefill for CUDA. Tensors are contiguous:
//
//   activation_f32:          [m, k]
//   projection1_weight_bf16: [n1, k]
//   projection2_weight_bf16: [n2, k]
//   projection1_output_f32:  [m, n1]
//   projection2_output_f32:  [m, n2]
//
// A 128-thread CTA owns an 8x64 output tile. Its four warps compute disjoint
// 8x16 N strips. For every K16 step, the CTA converts exactly one 8x16 F32
// activation tile to BF16 in shared memory and all four warps reuse it for both
// projections. The grid tiles both M and max(n1, n2), so prefill M contributes
// independent CTAs instead of being serialized inside a single block.
//
// launch() performs exactly one asynchronous kernel launch on the supplied
// stream. It does not allocate, synchronize, invoke host-side GEMMs, or select
// a fallback path. Ferrule owns every pointer and the stream.
namespace ferrule::cuda::cutlass::operators::bf16_compressor::projection {

inline constexpr std::uint32_t kMmaRows = 8u;
inline constexpr std::uint32_t kMmaColumns = 16u;
inline constexpr std::uint32_t kKTile = 16u;
inline constexpr std::uint32_t kWarpSize = 32u;
inline constexpr std::uint32_t kWarps = 4u;
inline constexpr std::uint32_t kThreads = kWarpSize * kWarps;
inline constexpr std::uint32_t kCtaRows = kMmaRows;
inline constexpr std::uint32_t kCtaColumns = kMmaColumns * kWarps;
inline constexpr std::uint32_t kMaxGridY = 65535u;
inline constexpr std::uint32_t kMaxRows = kCtaRows * kMaxGridY;

namespace semantic = operators::bf16_compressor;

namespace detail {

using Bf16Mma = cute::SM80_16x8x16_F32BF16BF16F32_TN;

struct alignas(16) SharedStorage {
  // K-major [16, 8]. Each K row is 16 bytes, matching ldmatrix.trans.
  alignas(16) std::uint16_t activation[kKTile * kCtaRows];
  // A private [16, 16] BF16 stage for each warp. A warp reuses the same stage
  // sequentially for projection 1 and projection 2.
  alignas(16) std::uint16_t weight[kWarps][kMmaColumns * kKTile];
};

static_assert(sizeof(SharedStorage) == 2304u);
static_assert(alignof(SharedStorage) >= 16u);

inline constexpr bool aligned_device_address(std::uint64_t address,
                                             std::uint64_t alignment) {
  return address != 0u && (address & (alignment - 1u)) == 0u;
}

template <class T>
__device__ __forceinline__ T *device_pointer(std::uint64_t address) {
  return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address));
}

__device__ __forceinline__ std::uint16_t f32_to_bf16_rne(float value) {
  std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    // Preserve NaN while forcing a non-zero BF16 payload.
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  }
  const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
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

__device__ __forceinline__ void
stage_activation(SharedStorage &shared, const float *activation,
                 const semantic::Args &args, std::uint32_t m_base, std::uint32_t k_base) {
  const std::uint32_t linear = threadIdx.x;
  const std::uint32_t k_local = linear >> 3;
  const std::uint32_t row_local = linear & 7u;
  const std::uint32_t row = m_base + row_local;
  const float value =
      row < args.m ? activation[static_cast<std::uint64_t>(row) * args.k +
                                k_base + k_local]
                   : 0.0f;
  shared.activation[linear] = f32_to_bf16_rne(value);
}

__device__ __forceinline__ void
stage_weight(std::uint16_t *stage, const std::uint16_t *weight,
             std::uint32_t channel_base, std::uint32_t channel_count,
             std::uint32_t k_extent, std::uint32_t k_base, std::uint32_t lane) {
  for (std::uint32_t linear = lane; linear < kMmaColumns * kKTile;
       linear += kWarpSize) {
    const std::uint32_t local_channel = linear >> 4;
    const std::uint32_t k_local = linear & 15u;
    const std::uint32_t channel = channel_base + local_channel;
    stage[linear] =
        channel < channel_count
            ? weight[static_cast<std::uint64_t>(channel) * k_extent + k_base +
                     k_local]
            : 0u;
  }
}

__device__ __forceinline__ void
store_projection(float *output, const semantic::Args &args, std::uint32_t channel_count,
                 std::uint32_t m_base, std::uint32_t channel_base,
                 std::uint32_t lane, const float (&accumulator)[4]) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row = m_base + row_pair * 2u + (element & 1u);
    if (row < args.m && channel < channel_count) {
      output[static_cast<std::uint64_t>(row) * channel_count + channel] =
          accumulator[element];
    }
  }
}

} // namespace detail

inline semantic::ValidationResult validate(const semantic::Args &args) noexcept {
  if (args.activation_f32 == 0u || args.projection1_weight_bf16 == 0u ||
      args.projection2_weight_bf16 == 0u || args.projection1_output_f32 == 0u ||
      args.projection2_output_f32 == 0u) {
    return semantic::ValidationResult::kNullPointer;
  }
  if (!detail::aligned_device_address(args.activation_f32, 16u) ||
      !detail::aligned_device_address(args.projection1_weight_bf16, 16u) ||
      !detail::aligned_device_address(args.projection2_weight_bf16, 16u) ||
      !detail::aligned_device_address(args.projection1_output_f32, 16u) ||
      !detail::aligned_device_address(args.projection2_output_f32, 16u)) {
    return semantic::ValidationResult::kMisalignedPointer;
  }
  if (args.m == 0u || args.m > kMaxRows) {
    return semantic::ValidationResult::kUnsupportedM;
  }
  if (args.n1 == 0u || args.n2 == 0u || args.k == 0u ||
      (args.k % kKTile) != 0u) {
    return semantic::ValidationResult::kUnsupportedShape;
  }
  if (args.reserved != 0u) {
    return semantic::ValidationResult::kInvalidReserved;
  }
  return semantic::ValidationResult::kSuccess;
}

__global__ __launch_bounds__(kThreads, 4) void kernel(semantic::Args args) {
  __shared__ detail::SharedStorage shared;

  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t m_base = blockIdx.y * kCtaRows;
  const std::uint32_t n_tile_base = blockIdx.x * kCtaColumns;
  const std::uint32_t channel_base = n_tile_base + warp * kMmaColumns;

  auto const *activation =
      detail::device_pointer<const float>(args.activation_f32);
  auto const *projection1_weight =
      detail::device_pointer<const std::uint16_t>(args.projection1_weight_bf16);
  auto const *projection2_weight =
      detail::device_pointer<const std::uint16_t>(args.projection2_weight_bf16);
  auto *projection1_output =
      detail::device_pointer<float>(args.projection1_output_f32);
  auto *projection2_output =
      detail::device_pointer<float>(args.projection2_output_f32);

  const bool projection1_live = channel_base < args.n1;
  const bool projection2_live = channel_base < args.n2;
  const bool warp_live = projection1_live || projection2_live;
  float projection1_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float projection2_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t k_base = 0u; k_base < args.k; k_base += kKTile) {
    // Exactly one CTA-wide F32->BF16 conversion for this MxK tile. Both
    // projections and every live N warp consume the same shared values.
    detail::stage_activation(shared, activation, args, m_base, k_base);
    __syncthreads();

    if (warp_live) {
      std::uint32_t activation_fragment[2];
      std::uint32_t weight_fragment[4];
      detail::load_activation_fragment(shared.activation, lane,
                                       activation_fragment);

      std::uint16_t *weight_stage = shared.weight[warp];
      if (projection1_live) {
        detail::stage_weight(weight_stage, projection1_weight, channel_base,
                             args.n1, args.k, k_base, lane);
        __syncwarp();
        detail::load_weight_fragment(weight_stage, lane, weight_fragment);
        __syncwarp();
        detail::mma_bf16(projection1_accumulator, weight_fragment,
                         activation_fragment);
      }

      if (projection2_live) {
        detail::stage_weight(weight_stage, projection2_weight, channel_base,
                             args.n2, args.k, k_base, lane);
        __syncwarp();
        detail::load_weight_fragment(weight_stage, lane, weight_fragment);
        __syncwarp();
        detail::mma_bf16(projection2_accumulator, weight_fragment,
                         activation_fragment);
      }
    }

    // No warp may let the next K tile overwrite the shared activation until all
    // consumers have issued both MMA operations.
    __syncthreads();
  }

  if (projection1_live) {
    detail::store_projection(projection1_output, args, args.n1, m_base,
                             channel_base, lane, projection1_accumulator);
  }
  if (projection2_live) {
    detail::store_projection(projection2_output, args, args.n2, m_base,
                             channel_base, lane, projection2_accumulator);
  }
}

inline cudaError_t launch(const semantic::Args &args) noexcept {
  if (projection::validate(args) != semantic::ValidationResult::kSuccess) {
    return cudaErrorInvalidValue;
  }

  const std::uint32_t max_n = args.n1 > args.n2 ? args.n1 : args.n2;
  const std::uint32_t grid_x = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(max_n) + kCtaColumns - 1u) / kCtaColumns);
  const std::uint32_t grid_y = (args.m + kCtaRows - 1u) / kCtaRows;
  const dim3 grid(grid_x, grid_y, 1u);
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args.stream));

  kernel<<<grid, kThreads, 0u, stream>>>(args);
  return cudaGetLastError();
}

} // namespace ferrule::cuda::cutlass::operators::bf16_compressor::projection


#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>

#include <cstdint>

namespace ferrule::cuda::cutlass::operators::bf16_compressor::tiny_rows {

namespace semantic = operators::bf16_compressor;
using Bf16Mma = cute::SM80_16x8x16_F32BF16BF16F32_TN;

inline constexpr std::uint32_t kMaximumRows = 8u;
inline constexpr std::uint32_t kThreads = 32u;
inline constexpr std::uint32_t kColumns = 16u;

struct alignas(16) SharedStorage {
  alignas(16) std::uint16_t activation[128];
  alignas(16) std::uint16_t projection1_weight[256];
  alignas(16) std::uint16_t projection2_weight[256];
};

__device__ __forceinline__ std::uint16_t f32_to_bf16_rne(float value) {
  std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  }
  const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

__device__ __forceinline__ void
load_a_fragment_16x32_bytes(const std::uint8_t *shared, std::uint32_t lane,
                            std::uint32_t (&fragment)[4]) {
  const std::uint32_t quad = lane >> 3;
  const std::uint32_t row = (lane & 7u) + ((quad & 1u) != 0 ? 8u : 0u);
  const std::uint32_t column_bytes = quad >= 2 ? 16u : 0u;
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + row * 32u + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1], fragment[2],
                                fragment[3]);
}

__device__ __forceinline__ void
load_b_fragment_16_byte_rows(const std::uint8_t *shared, std::uint32_t lane,
                             std::uint32_t (&fragment)[2]) {
  auto const &source =
      *reinterpret_cast<const cute::uint128_t *>(shared + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void mma_bf16(float (&accumulator)[4],
                                         const std::uint32_t (&a)[4],
                                         const std::uint32_t (&b)[2]) {
  Bf16Mma::fma(accumulator[0], accumulator[1], accumulator[2], accumulator[3],
               a[0], a[1], a[2], a[3], b[0], b[1], accumulator[0],
               accumulator[1], accumulator[2], accumulator[3]);
}

__global__ void kernel(semantic::Args args) {
  __shared__ SharedStorage shared;

  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t channel_base = blockIdx.x * kColumns;
  auto *activation = reinterpret_cast<const float *>(
      static_cast<std::uintptr_t>(args.activation_f32));
  auto *projection1_weight = reinterpret_cast<const std::uint16_t *>(
      static_cast<std::uintptr_t>(args.projection1_weight_bf16));
  auto *projection2_weight = reinterpret_cast<const std::uint16_t *>(
      static_cast<std::uintptr_t>(args.projection2_weight_bf16));
  auto *projection1_output = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(args.projection1_output_f32));
  auto *projection2_output = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(args.projection2_output_f32));

  float projection1_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float projection2_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t k_base = 0; k_base < args.k; k_base += 16u) {
    if (lane < 16u) {
      const std::uint32_t channel = channel_base + lane;
      auto *projection1_destination =
          reinterpret_cast<uint4 *>(shared.projection1_weight + lane * 16u);
      auto *projection2_destination =
          reinterpret_cast<uint4 *>(shared.projection2_weight + lane * 16u);
      const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
      if (channel < args.n1) {
        auto *source = reinterpret_cast<const uint4 *>(
            projection1_weight + static_cast<std::uint64_t>(channel) * args.k +
            k_base);
        projection1_destination[0] = source[0];
        projection1_destination[1] = source[1];
      } else {
        projection1_destination[0] = zero;
        projection1_destination[1] = zero;
      }
      if (channel < args.n2) {
        auto *source = reinterpret_cast<const uint4 *>(
            projection2_weight + static_cast<std::uint64_t>(channel) * args.k +
            k_base);
        projection2_destination[0] = source[0];
        projection2_destination[1] = source[1];
      } else {
        projection2_destination[0] = zero;
        projection2_destination[1] = zero;
      }
    }

    for (std::uint32_t linear = lane; linear < 128u; linear += 32u) {
      const std::uint32_t k_local = linear >> 3;
      const std::uint32_t row = linear & 7u;
      const float value =
          row < args.m ? activation[static_cast<std::uint64_t>(row) * args.k +
                                    k_base + k_local]
                       : 0.0f;
      shared.activation[linear] = f32_to_bf16_rne(value);
    }
    __syncthreads();

    std::uint32_t projection1_fragment[4];
    std::uint32_t projection2_fragment[4];
    std::uint32_t activation_fragment[2];
    load_a_fragment_16x32_bytes(
        reinterpret_cast<const std::uint8_t *>(shared.projection1_weight), lane,
        projection1_fragment);
    load_a_fragment_16x32_bytes(
        reinterpret_cast<const std::uint8_t *>(shared.projection2_weight), lane,
        projection2_fragment);
    load_b_fragment_16_byte_rows(
        reinterpret_cast<const std::uint8_t *>(shared.activation), lane,
        activation_fragment);
    mma_bf16(projection1_accumulator, projection1_fragment,
             activation_fragment);
    mma_bf16(projection2_accumulator, projection2_fragment,
             activation_fragment);
    __syncthreads();
  }

  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row = row_pair * 2u + (element & 1u);
    if (row < args.m && channel < args.n1) {
      projection1_output[static_cast<std::uint64_t>(row) * args.n1 + channel] =
          projection1_accumulator[element];
    }
    if (row < args.m && channel < args.n2) {
      projection2_output[static_cast<std::uint64_t>(row) * args.n2 + channel] =
          projection2_accumulator[element];
    }
  }
}

inline cudaError_t launch(const semantic::Args &args) noexcept {
  const std::uint32_t max_n = args.n1 > args.n2 ? args.n1 : args.n2;
  const std::uint32_t blocks = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(max_n) + kColumns - 1u) / kColumns);
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args.stream));
  kernel<<<blocks, kThreads, 0u, stream>>>(args);
  return cudaGetLastError();
}

} // namespace ferrule::cuda::cutlass::operators::bf16_compressor::tiny_rows


namespace ferrule::cuda::cutlass::operators::bf16_compressor {

inline cudaError_t
launch(const operators::bf16_compressor::Args &args) noexcept {
  if (operators::bf16_compressor::validate(args) !=
      operators::bf16_compressor::ValidationResult::kSuccess) {
    return cudaErrorInvalidValue;
  }
  return args.m <= tiny_rows::kMaximumRows ? tiny_rows::launch(args)
                                           : projection::launch(args);
}

} // namespace ferrule::cuda::cutlass::operators::bf16_compressor


#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cutlass/version.h>


#if CUTLASS_MAJOR != 4 || CUTLASS_MINOR != 6 || CUTLASS_PATCH != 1
#error "Ferrule's FP8 projection operator requires CUTLASS 4.6.1"
#endif

namespace ferrule::cuda::cutlass::operators::fp8_projection {

// Packed tensor layouts:
//   activation_fp8:         [m, k] row-major E4M3
//   activation_ue8m0:       [m, k / 128] row-major UE8M0
//   *_weight_fp8:           [n, k] row-major E4M3
//   *_weight_ue8m0:         [ceil(n / 128), k / 128] row-major UE8M0
//   *_output_f32:           [m, n] row-major F32
//
// Weight scales intentionally match Ferrule's K128 packing: one scale covers a
// 128-channel by 128-K weight block. Activation scales cover one row by 128-K.
struct Args {
  const std::uint8_t *activation_fp8;
  const std::uint8_t *activation_ue8m0;
  const std::uint8_t *query_a_weight_fp8;
  const std::uint8_t *query_a_weight_ue8m0;
  const std::uint8_t *kv_weight_fp8;
  const std::uint8_t *kv_weight_ue8m0;
  float *query_a_output_f32;
  float *kv_output_f32;
  std::uint32_t m;
  std::uint32_t n_query_a;
  std::uint32_t n_kv;
  std::uint32_t k;
};

static_assert(std::is_standard_layout_v<Args> &&
                  std::is_trivially_copyable_v<Args>,
              "Args must remain a POD launch ABI");
static_assert(sizeof(Args) == 80, "Args ABI layout changed");

enum class ValidationResult : std::uint32_t {
  kSuccess = 0,
  kNullPointer,
  kMisalignedPointer,
  kUnsupportedM,
  kInvalidN,
  kInvalidK,
};

inline constexpr std::uint32_t kCtaRows = 8u;
inline constexpr std::uint32_t kMaxGridY = 65535u;
inline constexpr std::uint32_t kMaxRows = kCtaRows * kMaxGridY;

inline bool is_aligned_16(const void *pointer) noexcept {
  return pointer != nullptr &&
         (reinterpret_cast<std::uintptr_t>(pointer) & 15u) == 0;
}

inline ValidationResult validate(const Args &args) noexcept {
  if (args.activation_fp8 == nullptr || args.activation_ue8m0 == nullptr ||
      args.query_a_weight_fp8 == nullptr ||
      args.query_a_weight_ue8m0 == nullptr || args.kv_weight_fp8 == nullptr ||
      args.kv_weight_ue8m0 == nullptr || args.query_a_output_f32 == nullptr ||
      args.kv_output_f32 == nullptr) {
    return ValidationResult::kNullPointer;
  }
  if (!is_aligned_16(args.activation_fp8) ||
      !is_aligned_16(args.activation_ue8m0) ||
      !is_aligned_16(args.query_a_weight_fp8) ||
      !is_aligned_16(args.query_a_weight_ue8m0) ||
      !is_aligned_16(args.kv_weight_fp8) ||
      !is_aligned_16(args.kv_weight_ue8m0) ||
      !is_aligned_16(args.query_a_output_f32) ||
      !is_aligned_16(args.kv_output_f32)) {
    return ValidationResult::kMisalignedPointer;
  }
  if (args.m == 0u || args.m > kMaxRows) {
    return ValidationResult::kUnsupportedM;
  }
  if (args.n_query_a == 0u || args.n_kv == 0u) {
    return ValidationResult::kInvalidN;
  }
  if (args.k == 0u || (args.k & 127u) != 0u) {
    return ValidationResult::kInvalidK;
  }
  return ValidationResult::kSuccess;
}

namespace detail {

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

constexpr std::uint32_t kTileM = kCtaRows;
constexpr std::uint32_t kTileN = 16;
constexpr std::uint32_t kScaleK = 128;
constexpr std::uint32_t kMmaK = 32;
constexpr std::uint32_t kMmaKSteps = kScaleK / kMmaK;
constexpr std::uint32_t kThreads = 64;
constexpr std::uint32_t kStages = 2;

template <bool DualProjection> struct alignas(16) StageStorage;

template <> struct alignas(16) StageStorage<true> {
  // Each K32 sub-tile has the exact ldmatrix layout consumed by one MMA.
  alignas(16) std::uint8_t query_a_weight[kMmaKSteps][kTileN * kMmaK];
  alignas(16) std::uint8_t kv_weight[kMmaKSteps][kTileN * kMmaK];
  alignas(16) std::uint8_t activation[kMmaKSteps][kTileM * kMmaK];
};

template <> struct alignas(16) StageStorage<false> {
  alignas(16) std::uint8_t query_a_weight[kMmaKSteps][kTileN * kMmaK];
  alignas(16) std::uint8_t activation[kMmaKSteps][kTileM * kMmaK];
};

template <bool DualProjection> struct alignas(16) SharedStorage {
  StageStorage<DualProjection> stage[kStages];
  volatile std::int32_t ready[kStages];
  volatile std::int32_t done[kStages];
};

static_assert(sizeof(StageStorage<true>) == 5120,
              "Unexpected dual-projection stage footprint");
static_assert(sizeof(SharedStorage<true>) == 10256,
              "Unexpected dual-projection shared-memory footprint");
static_assert(sizeof(StageStorage<false>) == 3072,
              "Unexpected single-projection stage footprint");
static_assert(sizeof(SharedStorage<false>) == 6160,
              "Unexpected single-projection shared-memory footprint");

template <class T> __device__ __forceinline__ T device_load(const T *pointer) {
  return *pointer;
}

__device__ __forceinline__ void cp_async_16(void *destination,
                                            const void *source) {
#if defined(__CUDA_ARCH__)
  std::uint32_t shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_address),
      "l"(source)
      : "memory");
#else
  (void)destination;
  (void)source;
#endif
}

__device__ __forceinline__ void cp_async_commit_and_wait() {
#if defined(__CUDA_ARCH__)
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void clear_16(void *destination) {
  auto *words = static_cast<std::uint32_t *>(destination);
  words[0] = 0u;
  words[1] = 0u;
  words[2] = 0u;
  words[3] = 0u;
}

__device__ __forceinline__ void wait_until(const volatile std::int32_t *state,
                                           std::int32_t expected) {
  while (*state != expected) {
#if defined(__CUDA_ARCH__)
    __nanosleep(32);
#endif
  }
}

__device__ __forceinline__ void load_a_fragment(const std::uint8_t *shared,
                                                std::uint32_t lane,
                                                std::uint32_t (&fragment)[4]) {
  std::uint32_t quad = lane >> 3;
  std::uint32_t row = (lane & 7u) + ((quad & 1u) != 0u ? 8u : 0u);
  std::uint32_t column_bytes = quad >= 2u ? 16u : 0u;
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + row * kMmaK + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1], fragment[2],
                                fragment[3]);
}

__device__ __forceinline__ void load_b_fragment(const std::uint8_t *shared,
                                                std::uint32_t lane,
                                                std::uint32_t (&fragment)[2]) {
  auto const &source =
      *reinterpret_cast<const cute::uint128_t *>(shared + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ float ue8m0_to_float(std::uint8_t value) {
  // UE8M0's zero encoding represents 2^-127 in Ferrule's packed artifacts.
  const std::uint32_t bits =
      value == 0u ? (1u << 22) : (static_cast<std::uint32_t>(value) << 23);
  return __uint_as_float(bits);
}

__device__ __forceinline__ void mma(float (&accumulator)[4],
                                    const std::uint32_t (&a)[4],
                                    const std::uint32_t (&b)[2]) {
  architectures::sm89::mma_sync_f32_e4m3_e4m3_m16n8k32(accumulator, a, b);
}

struct QueryAKvCollective {
  template <bool DualProjection>
  __device__ static void producer(const Args &args,
                                  SharedStorage<DualProjection> &shared,
                                  std::uint32_t lane, std::uint32_t row_base,
                                  std::uint32_t channel_base) {
    const std::uint32_t scale_columns = args.k / kScaleK;

    for (std::uint32_t scale_block = 0; scale_block < scale_columns;
         ++scale_block) {
      const std::uint32_t stage_index = scale_block & (kStages - 1u);
      if (scale_block >= kStages) {
        wait_until(&shared.done[stage_index],
                   static_cast<std::int32_t>(scale_block - kStages));
      }

      StageStorage<DualProjection> &stage = shared.stage[stage_index];
      const std::uint64_t k_base =
          static_cast<std::uint64_t>(scale_block) * kScaleK;

      // A 16x128 weight tile is split into four contiguous 16x32 ldmatrix
      // tiles. Each producer lane moves four 16-byte vectors per projection.
      for (std::uint32_t chunk = lane; chunk < (kTileN * kScaleK) / 16u;
           chunk += 32u) {
        const std::uint32_t k_step = chunk / 32u;
        const std::uint32_t tile_chunk = chunk & 31u;
        const std::uint32_t local_channel = tile_chunk >> 1;
        const std::uint32_t k_half = tile_chunk & 1u;
        const std::uint32_t channel = channel_base + local_channel;
        const std::uint64_t k_offset =
            k_base + static_cast<std::uint64_t>(k_step) * kMmaK + k_half * 16u;

        void *query_destination =
            stage.query_a_weight[k_step] + local_channel * kMmaK + k_half * 16u;
        if (channel < args.n_query_a) {
          cp_async_16(query_destination,
                      args.query_a_weight_fp8 +
                          static_cast<std::uint64_t>(channel) * args.k +
                          k_offset);
        } else {
          clear_16(query_destination);
        }

        if constexpr (DualProjection) {
          void *kv_destination =
              stage.kv_weight[k_step] + local_channel * kMmaK + k_half * 16u;
          if (channel < args.n_kv) {
            cp_async_16(kv_destination,
                        args.kv_weight_fp8 +
                            static_cast<std::uint64_t>(channel) * args.k +
                            k_offset);
          } else {
            clear_16(kv_destination);
          }
        }
      }

      // ldmatrix.trans consumes each 16-byte segment as eight FP8 pairs from
      // eight rows. Build that layout while the two weight copies are in
      // flight.
      for (std::uint32_t segment = lane; segment < kMmaKSteps * (kMmaK / 2u);
           segment += 32u) {
        const std::uint32_t k_step = segment / (kMmaK / 2u);
        const std::uint32_t k_pair = segment & (kMmaK / 2u - 1u);
        auto *destination = reinterpret_cast<std::uint16_t *>(
            stage.activation[k_step] + k_pair * 16u);
        const std::uint64_t k_offset =
            k_base + static_cast<std::uint64_t>(k_step) * kMmaK + k_pair * 2u;

#pragma unroll
        for (std::uint32_t row = 0; row < kTileM; ++row) {
          const std::uint32_t global_row = row_base + row;
          destination[row] =
              global_row < args.m
                  ? device_load(reinterpret_cast<const std::uint16_t *>(
                        args.activation_fp8 +
                        static_cast<std::uint64_t>(global_row) * args.k +
                        k_offset))
                  : 0u;
        }
      }

      cp_async_commit_and_wait();
      __syncwarp();
      if (lane == 0u) {
        __threadfence_block();
        shared.ready[stage_index] = static_cast<std::int32_t>(scale_block);
      }
    }
  }

  template <bool DualProjection>
  __device__ static void consumer(const Args &args,
                                  SharedStorage<DualProjection> &shared,
                                  std::uint32_t lane, std::uint32_t row_base,
                                  std::uint32_t channel_base) {
    float query_a_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float kv_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const std::uint32_t scale_columns = args.k / kScaleK;

    for (std::uint32_t scale_block = 0; scale_block < scale_columns;
         ++scale_block) {
      const std::uint32_t stage_index = scale_block & (kStages - 1u);
      wait_until(&shared.ready[stage_index],
                 static_cast<std::int32_t>(scale_block));
      __threadfence_block();

      float query_a_block_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float kv_block_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const StageStorage<DualProjection> &stage = shared.stage[stage_index];
#pragma unroll
      for (std::uint32_t k_step = 0; k_step < kMmaKSteps; ++k_step) {
        std::uint32_t query_a_fragment[4];
        std::uint32_t activation_fragment[2];
        load_a_fragment(stage.query_a_weight[k_step], lane, query_a_fragment);
        load_b_fragment(stage.activation[k_step], lane, activation_fragment);
        mma(query_a_block_accumulator, query_a_fragment, activation_fragment);
        if constexpr (DualProjection) {
          std::uint32_t kv_fragment[4];
          load_a_fragment(stage.kv_weight[k_step], lane, kv_fragment);
          mma(kv_block_accumulator, kv_fragment, activation_fragment);
        }
      }

      const float query_weight_scale =
          channel_base < args.n_query_a
              ? ue8m0_to_float(
                    args.query_a_weight_ue8m0[static_cast<std::uint64_t>(
                                                  channel_base / 128u) *
                                                  scale_columns +
                                              scale_block])
              : 0.0f;
      float kv_weight_scale = 0.0f;
      if constexpr (DualProjection) {
        kv_weight_scale =
            channel_base < args.n_kv
                ? ue8m0_to_float(
                      args.kv_weight_ue8m0[static_cast<std::uint64_t>(
                                               channel_base / 128u) *
                                               scale_columns +
                                           scale_block])
                : 0.0f;
      }

      const std::uint32_t row_pair = lane & 3u;
#pragma unroll
      for (std::uint32_t element = 0; element < 4u; ++element) {
        const std::uint32_t row = row_base + row_pair * 2u + (element & 1u);
        if (row < args.m) {
          const float activation_scale = ue8m0_to_float(
              args.activation_ue8m0[static_cast<std::uint64_t>(row) *
                                        scale_columns +
                                    scale_block]);
          query_a_accumulator[element] += query_a_block_accumulator[element] *
                                          query_weight_scale * activation_scale;
          if constexpr (DualProjection) {
            kv_accumulator[element] += kv_block_accumulator[element] *
                                       kv_weight_scale * activation_scale;
          }
        }
      }

      __syncwarp();
      if (lane == 0u) {
        __threadfence_block();
        shared.done[stage_index] = static_cast<std::int32_t>(scale_block);
      }
    }

    const std::uint32_t channel_group = lane >> 2;
    const std::uint32_t row_pair = lane & 3u;
#pragma unroll
    for (std::uint32_t element = 0; element < 4u; ++element) {
      const std::uint32_t channel =
          channel_base + channel_group + (element >= 2u ? 8u : 0u);
      const std::uint32_t row = row_base + row_pair * 2u + (element & 1u);
      if (row < args.m && channel < args.n_query_a) {
        args.query_a_output_f32[static_cast<std::uint64_t>(row) *
                                    args.n_query_a +
                                channel] =
            bf16_round(query_a_accumulator[element]);
      }
      if constexpr (DualProjection) {
        if (row < args.m && channel < args.n_kv) {
          args.kv_output_f32[static_cast<std::uint64_t>(row) * args.n_kv +
                             channel] = bf16_round(kv_accumulator[element]);
        }
      }
    }
  }
};

} // namespace detail

// One CTA owns one 8x16 output tile. Warp 0 is the double-buffered
// global-to-shared producer and warp 1 is the K128-scaled FP8 MMA consumer.
// The specialization owns either one projection or the fused QueryA+KV pair.
template <bool DualProjection>
__global__ __launch_bounds__(detail::kThreads, 2) void kernel(Args args) {
  __shared__ detail::SharedStorage<DualProjection> shared;

  const std::uint32_t warp = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row_base = blockIdx.y * detail::kTileM;
  const std::uint32_t channel_base = blockIdx.x * detail::kTileN;

  if (threadIdx.x < detail::kStages) {
    shared.ready[threadIdx.x] = -1;
    shared.done[threadIdx.x] = -1;
  }
  __syncthreads();

  if (warp == 0u) {
    detail::QueryAKvCollective::producer<DualProjection>(
        args, shared, lane, row_base, channel_base);
  } else {
    detail::QueryAKvCollective::consumer<DualProjection>(
        args, shared, lane, row_base, channel_base);
  }
}

inline cudaError_t launch(const Args &args, cudaStream_t stream) noexcept {
  if (validate(args) != ValidationResult::kSuccess) {
    return cudaErrorInvalidValue;
  }

  const std::uint32_t max_n =
      args.n_query_a > args.n_kv ? args.n_query_a : args.n_kv;
  const std::uint32_t grid_n = 1u + (max_n - 1u) / detail::kTileN;
  const dim3 grid(grid_n, (args.m + detail::kTileM - 1u) / detail::kTileM, 1u);
  kernel<true><<<grid, detail::kThreads, 0, stream>>>(args);
  return cudaPeekAtLastError();
}

inline ValidationResult validate_single(const Args &args) noexcept {
  if (args.activation_fp8 == nullptr || args.activation_ue8m0 == nullptr ||
      args.query_a_weight_fp8 == nullptr ||
      args.query_a_weight_ue8m0 == nullptr ||
      args.query_a_output_f32 == nullptr) {
    return ValidationResult::kNullPointer;
  }
  if (!is_aligned_16(args.activation_fp8) ||
      !is_aligned_16(args.activation_ue8m0) ||
      !is_aligned_16(args.query_a_weight_fp8) ||
      !is_aligned_16(args.query_a_weight_ue8m0) ||
      !is_aligned_16(args.query_a_output_f32)) {
    return ValidationResult::kMisalignedPointer;
  }
  if (args.m == 0u || args.m > kMaxRows) {
    return ValidationResult::kUnsupportedM;
  }
  if (args.n_query_a == 0u) {
    return ValidationResult::kInvalidN;
  }
  if (args.k == 0u || (args.k & 127u) != 0u) {
    return ValidationResult::kInvalidK;
  }
  return ValidationResult::kSuccess;
}

inline cudaError_t launch_single(const Args &args,
                                 cudaStream_t stream) noexcept {
  if (validate_single(args) != ValidationResult::kSuccess) {
    return cudaErrorInvalidValue;
  }

  const std::uint32_t grid_n = 1u + (args.n_query_a - 1u) / detail::kTileN;
  const dim3 grid(grid_n, (args.m + detail::kTileM - 1u) / detail::kTileM, 1u);
  kernel<false><<<grid, detail::kThreads, 0, stream>>>(args);
  return cudaPeekAtLastError();
}

} // namespace ferrule::cuda::cutlass::operators::fp8_projection


#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>


namespace ferrule::cuda::cutlass::operators::proposal_head {

namespace bf16_contract =
    ferrule::cuda::cutlass::operators::bf16_compressor;
namespace bf16_projection =
    ferrule::cuda::cutlass::operators::bf16_compressor::projection;

inline constexpr std::uint32_t kProposalRows = 5u;
inline constexpr std::uint32_t kThreads = 256u;
inline constexpr std::uint32_t kMaximumCooperativeBlocks = 64u;

struct Args {
  std::uint32_t rows;
  std::uint32_t hc;
  std::uint32_t hidden;
  std::uint32_t vocab;
  std::uint32_t markov_rank;
  std::uint32_t partial_capacity;
  std::uint32_t reserved;
  float hc_eps;
  float norm_eps;

  std::uint64_t hc_state_f32;
  std::uint64_t hc_function_f32;
  std::uint64_t hc_scale_f32;
  std::uint64_t hc_base_f32;
  std::uint64_t norm_weight_f32;
  std::uint64_t lm_head_bf16;
  std::uint64_t markov_w1_bf16;
  std::uint64_t markov_w2_bf16;
  std::uint64_t confidence_weight_bf16;

  std::uint64_t hidden_f32;
  std::uint64_t normalized_f32;
  std::uint64_t base_logits_f32;
  std::uint64_t partial_values_f32;
  std::uint64_t partial_indices_i32;
  std::uint64_t token_ids_i32;
  std::uint64_t confidence_f32;
  std::uint64_t status_i32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 184u, "proposal-head POD ABI changed");
static_assert(alignof(Args) == 8u);
static_assert(offsetof(Args, rows) == 0u);
static_assert(offsetof(Args, hc_state_f32) == 40u);
static_assert(offsetof(Args, stream) == 176u);

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidArgument = 2,
  kLaunchFailed = 3,
};

namespace detail {

inline constexpr bool aligned(std::uint64_t address, std::uint64_t alignment) {
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

__device__ __forceinline__ float bf16_boundary(float value) {
  return bf16_to_f32(f32_to_bf16_rne(value));
}

__device__ __forceinline__ float block_sum(float value,
                                           float (&scratch)[kThreads]) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (std::uint32_t stride = kThreads / 2u; stride != 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

struct HcShared {
  float reduction[kThreads];
  float coefficient[8];
  float hidden_inv_rms;
};

__global__ __launch_bounds__(kThreads, 1) void hc_head_norm_kernel(Args args) {
  __shared__ HcShared shared;
  const std::uint32_t row = blockIdx.x;
  if (row >= args.rows) {
    return;
  }
  const auto *state = device_pointer<const float>(args.hc_state_f32) +
                      static_cast<std::uint64_t>(row) * args.hc * args.hidden;
  const auto *function = device_pointer<const float>(args.hc_function_f32);
  const auto *scale = device_pointer<const float>(args.hc_scale_f32);
  const auto *base = device_pointer<const float>(args.hc_base_f32);
  const auto *norm = device_pointer<const float>(args.norm_weight_f32);
  auto *hidden = device_pointer<float>(args.hidden_f32) +
                 static_cast<std::uint64_t>(row) * args.hidden;
  auto *normalized = device_pointer<float>(args.normalized_f32) +
                     static_cast<std::uint64_t>(row) * args.hidden;

  float local_square = 0.0f;
  const std::uint32_t hc_hidden = args.hc * args.hidden;
  for (std::uint32_t index = threadIdx.x; index < hc_hidden;
       index += blockDim.x) {
    const float value = state[index];
    local_square += value * value;
  }
  const float state_inv_rms = rsqrtf(block_sum(local_square, shared.reduction) /
                                         static_cast<float>(hc_hidden) +
                                     args.norm_eps);

  for (std::uint32_t output_hc = 0u; output_hc < args.hc; ++output_hc) {
    float local_dot = 0.0f;
    const auto *function_row =
        function + static_cast<std::uint64_t>(output_hc) * hc_hidden;
    for (std::uint32_t index = threadIdx.x; index < hc_hidden;
         index += blockDim.x) {
      local_dot += state[index] * function_row[index];
    }
    const float mix = block_sum(local_dot, shared.reduction) * state_inv_rms;
    if (threadIdx.x == 0u) {
      shared.coefficient[output_hc] =
          1.0f / (1.0f + expf(-(mix * scale[0] + base[output_hc]))) +
          args.hc_eps;
    }
    __syncthreads();
  }

  float local_hidden_square = 0.0f;
  for (std::uint32_t dimension = threadIdx.x; dimension < args.hidden;
       dimension += blockDim.x) {
    float value = 0.0f;
    for (std::uint32_t input_hc = 0u; input_hc < args.hc; ++input_hc) {
      value +=
          shared.coefficient[input_hc] *
          state[static_cast<std::uint64_t>(input_hc) * args.hidden + dimension];
    }
    value = bf16_boundary(value);
    hidden[dimension] = value;
    local_hidden_square += value * value;
  }
  const float hidden_inv_rms =
      rsqrtf(block_sum(local_hidden_square, shared.reduction) /
                 static_cast<float>(args.hidden) +
             args.norm_eps);
  if (threadIdx.x == 0u) {
    shared.hidden_inv_rms = hidden_inv_rms;
  }
  __syncthreads();
  for (std::uint32_t dimension = threadIdx.x; dimension < args.hidden;
       dimension += blockDim.x) {
    normalized[dimension] = bf16_boundary(
        hidden[dimension] * shared.hidden_inv_rms * norm[dimension]);
  }
}

struct ProposalShared {
  float markov[1024];
  float values[kThreads];
  std::int32_t indices[kThreads];
};

__device__ __forceinline__ bool better(float candidate_value,
                                       std::int32_t candidate_index,
                                       float current_value,
                                       std::int32_t current_index) {
  return candidate_value > current_value ||
         (candidate_value == current_value && candidate_index < current_index);
}

__device__ __forceinline__ void block_argmax(float &value, std::int32_t &index,
                                             ProposalShared &shared) {
  shared.values[threadIdx.x] = value;
  shared.indices[threadIdx.x] = index;
  __syncthreads();
  for (std::uint32_t stride = kThreads / 2u; stride != 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      const float candidate_value = shared.values[threadIdx.x + stride];
      const std::int32_t candidate_index = shared.indices[threadIdx.x + stride];
      if (better(candidate_value, candidate_index, shared.values[threadIdx.x],
                 shared.indices[threadIdx.x])) {
        shared.values[threadIdx.x] = candidate_value;
        shared.indices[threadIdx.x] = candidate_index;
      }
    }
    __syncthreads();
  }
  value = shared.values[0];
  index = shared.indices[0];
}

__global__ __launch_bounds__(kThreads, 1) void proposal_kernel(Args args) {
  __shared__ ProposalShared shared;
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const auto *w1 = device_pointer<const std::uint16_t>(args.markov_w1_bf16);
  const auto *w2 = device_pointer<const std::uint16_t>(args.markov_w2_bf16);
  const auto *confidence_weight =
      device_pointer<const std::uint16_t>(args.confidence_weight_bf16);
  const auto *hidden = device_pointer<const float>(args.hidden_f32);
  const auto *base_logits = device_pointer<const float>(args.base_logits_f32);
  auto *partial_values = device_pointer<float>(args.partial_values_f32);
  auto *partial_indices =
      device_pointer<std::int32_t>(args.partial_indices_i32);
  auto *token_ids = device_pointer<std::int32_t>(args.token_ids_i32);
  auto *confidence = device_pointer<float>(args.confidence_f32);
  auto *status = device_pointer<std::int32_t>(args.status_i32);

  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *status = 0;
  }
  grid.sync();

  for (std::uint32_t position = 0u; position < args.rows; ++position) {
    const std::int32_t previous_token = token_ids[position];
    if (previous_token < 0 ||
        static_cast<std::uint32_t>(previous_token) >= args.vocab) {
      if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *status = 1;
      }
      return;
    }
    for (std::uint32_t rank = threadIdx.x; rank < args.markov_rank;
         rank += blockDim.x) {
      shared.markov[rank] = bf16_to_f32(
          w1[static_cast<std::uint64_t>(previous_token) * args.markov_rank +
             rank]);
    }
    __syncthreads();

    float local_value = -std::numeric_limits<float>::infinity();
    std::int32_t local_index = std::numeric_limits<std::int32_t>::max();
    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t global_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t token = global_thread; token < args.vocab;
         token += global_stride) {
      float markov_bias = 0.0f;
      const auto *w2_row = w2 + token * args.markov_rank;
      for (std::uint32_t rank = 0u; rank < args.markov_rank; ++rank) {
        markov_bias += bf16_to_f32(w2_row[rank]) * shared.markov[rank];
      }
      const float score =
          base_logits[static_cast<std::uint64_t>(position) * args.vocab +
                      token] +
          markov_bias;
      const auto token_i32 = static_cast<std::int32_t>(token);
      if (better(score, token_i32, local_value, local_index)) {
        local_value = score;
        local_index = token_i32;
      }
    }
    block_argmax(local_value, local_index, shared);
    if (threadIdx.x == 0u) {
      partial_values[blockIdx.x] = local_value;
      partial_indices[blockIdx.x] = local_index;
    }
    grid.sync();

    if (blockIdx.x == 0u) {
      float global_value = -std::numeric_limits<float>::infinity();
      std::int32_t global_index = std::numeric_limits<std::int32_t>::max();
      for (std::uint32_t block = threadIdx.x; block < gridDim.x;
           block += blockDim.x) {
        if (better(partial_values[block], partial_indices[block], global_value,
                   global_index)) {
          global_value = partial_values[block];
          global_index = partial_indices[block];
        }
      }
      block_argmax(global_value, global_index, shared);
      if (threadIdx.x == 0u) {
        token_ids[position + 1u] = global_index;
      }

      float local_confidence = 0.0f;
      for (std::uint32_t dimension = threadIdx.x; dimension < args.hidden;
           dimension += blockDim.x) {
        local_confidence +=
            hidden[static_cast<std::uint64_t>(position) * args.hidden +
                   dimension] *
            bf16_to_f32(confidence_weight[dimension]);
      }
      for (std::uint32_t rank = threadIdx.x; rank < args.markov_rank;
           rank += blockDim.x) {
        local_confidence += shared.markov[rank] *
                            bf16_to_f32(confidence_weight[args.hidden + rank]);
      }
      shared.values[threadIdx.x] = local_confidence;
      __syncthreads();
      for (std::uint32_t stride = kThreads / 2u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
          shared.values[threadIdx.x] += shared.values[threadIdx.x + stride];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0u) {
        confidence[position] = shared.values[0];
      }
    }
    grid.sync();
  }
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr) {
    return Status::kInvalidArgument;
  }
  if (args->rows != kProposalRows || args->hc == 0u || args->hc > 8u ||
      args->hidden == 0u || (args->hidden % 16u) != 0u || args->vocab == 0u ||
      args->markov_rank == 0u || args->markov_rank > 1024u ||
      (args->markov_rank % 16u) != 0u || args->partial_capacity == 0u ||
      args->partial_capacity > kMaximumCooperativeBlocks ||
      args->reserved != 0u || !std::isfinite(args->hc_eps) ||
      !std::isfinite(args->norm_eps) || args->hc_eps <= 0.0f ||
      args->norm_eps <= 0.0f) {
    return Status::kInvalidArgument;
  }
  const bool pointers_valid =
      detail::aligned(args->hc_state_f32, 16u) &&
      detail::aligned(args->hc_function_f32, 16u) &&
      detail::aligned(args->hc_scale_f32, 4u) &&
      detail::aligned(args->hc_base_f32, 4u) &&
      detail::aligned(args->norm_weight_f32, 16u) &&
      detail::aligned(args->lm_head_bf16, 16u) &&
      detail::aligned(args->markov_w1_bf16, 16u) &&
      detail::aligned(args->markov_w2_bf16, 16u) &&
      detail::aligned(args->confidence_weight_bf16, 16u) &&
      detail::aligned(args->hidden_f32, 16u) &&
      detail::aligned(args->normalized_f32, 16u) &&
      detail::aligned(args->base_logits_f32, 16u) &&
      detail::aligned(args->partial_values_f32, 16u) &&
      detail::aligned(args->partial_indices_i32, 16u) &&
      detail::aligned(args->token_ids_i32, 4u) &&
      detail::aligned(args->confidence_f32, 16u) &&
      detail::aligned(args->status_i32, 4u);
  return pointers_valid ? Status::kSuccess : Status::kInvalidArgument;
}

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));

  detail::hc_head_norm_kernel<<<args->rows, kThreads, 0u, stream>>>(*args);
  if (cudaGetLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  const bf16_contract::Args base_args{
      args->rows,
      args->vocab,
      0u,
      args->hidden,
      0u,
      args->normalized_f32,
      args->lm_head_bf16,
      args->lm_head_bf16,
      args->base_logits_f32,
      args->base_logits_f32,
      args->stream,
  };
  const dim3 base_grid((args->vocab + bf16_projection::kCtaColumns - 1u) /
                           bf16_projection::kCtaColumns,
                       (args->rows + bf16_projection::kCtaRows - 1u) / bf16_projection::kCtaRows, 1u);
  bf16_projection::kernel<<<base_grid, bf16_projection::kThreads, 0u, stream>>>(base_args);
  if (cudaGetLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  int blocks_per_sm = 0;
  int device = 0;
  int multiprocessors = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm, detail::proposal_kernel, kThreads, 0u) !=
          cudaSuccess ||
      cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess ||
      blocks_per_sm <= 0 || multiprocessors <= 0) {
    return Status::kLaunchFailed;
  }
  std::uint32_t blocks = static_cast<std::uint32_t>(blocks_per_sm) *
                         static_cast<std::uint32_t>(multiprocessors);
  blocks = blocks < args->partial_capacity ? blocks : args->partial_capacity;
  blocks =
      blocks < kMaximumCooperativeBlocks ? blocks : kMaximumCooperativeBlocks;
  if (blocks == 0u) {
    return Status::kLaunchFailed;
  }
  void *kernel_args[] = {const_cast<Args *>(args)};
  const cudaError_t launch_status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::proposal_kernel), dim3(blocks, 1u, 1u),
      dim3(kThreads, 1u, 1u), kernel_args, 0u, stream);
  return launch_status == cudaSuccess ? Status::kSuccess
                                      : Status::kLaunchFailed;
}

} // namespace ferrule::cuda::cutlass::operators::proposal_head
