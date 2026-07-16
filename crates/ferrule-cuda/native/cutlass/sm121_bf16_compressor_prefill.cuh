#ifndef FERRULE_CUTLASS_SM121_BF16_COMPRESSOR_PREFILL_CUH_
#define FERRULE_CUTLASS_SM121_BF16_COMPRESSOR_PREFILL_CUH_

#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(FERRULE_CUTLASS_TARGET_SM) && FERRULE_CUTLASS_TARGET_SM != 121
#error "sm121_bf16_compressor_prefill.cuh requires Ferrule's SM121a target"
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1210
#error "sm121_bf16_compressor_prefill.cuh device code must target SM121"
#endif

// One-launch BF16 compressor prefill for GB10. Tensors are contiguous:
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
// stream. It does not allocate, synchronize, invoke host-side GEMMs, or select a
// fallback path. Ferrule owns every pointer and the stream.
namespace ferrule::cutlass::sm121_bf16_compressor_prefill {

inline constexpr std::uint32_t kArgsVersion = 1u;
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

// Device pointers and cudaStream_t are encoded as integers to keep this bridge
// POD independent of host CUDA pointer typedefs. BF16 values use their native
// 16-bit storage representation.
struct Args {
  std::uint32_t args_version;
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
static_assert(sizeof(Args) == 72u, "SM121 BF16 compressor POD ABI changed");
static_assert(offsetof(Args, activation_f32) == 24u);
static_assert(offsetof(Args, stream) == 64u);

enum class ValidationResult : std::uint32_t {
  kSuccess = 0u,
  kInvalidAbi,
  kNullPointer,
  kMisalignedPointer,
  kUnsupportedM,
  kUnsupportedShape,
  kInvalidReserved,
};

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

__device__ __forceinline__ void load_weight_fragment(
    const std::uint16_t *shared, std::uint32_t lane,
    std::uint32_t (&fragment)[4]) {
  const std::uint32_t quad = lane >> 3;
  const std::uint32_t row =
      (lane & 7u) + ((quad & 1u) != 0u ? 8u : 0u);
  const std::uint32_t column_bytes = quad >= 2u ? 16u : 0u;
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(shared);
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      bytes + row * 32u + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1], fragment[2],
                                fragment[3]);
}

__device__ __forceinline__ void load_activation_fragment(
    const std::uint16_t *shared, std::uint32_t lane,
    std::uint32_t (&fragment)[2]) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(shared);
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      bytes + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void mma_bf16(
    float (&accumulator)[4], const std::uint32_t (&weight)[4],
    const std::uint32_t (&activation)[2]) {
  Bf16Mma::fma(accumulator[0], accumulator[1], accumulator[2],
               accumulator[3], weight[0], weight[1], weight[2], weight[3],
               activation[0], activation[1], accumulator[0], accumulator[1],
               accumulator[2], accumulator[3]);
}

__device__ __forceinline__ void stage_activation(
    SharedStorage &shared, const float *activation, const Args &args,
    std::uint32_t m_base, std::uint32_t k_base) {
  const std::uint32_t linear = threadIdx.x;
  const std::uint32_t k_local = linear >> 3;
  const std::uint32_t row_local = linear & 7u;
  const std::uint32_t row = m_base + row_local;
  const float value =
      row < args.m
          ? activation[static_cast<std::uint64_t>(row) * args.k + k_base +
                       k_local]
          : 0.0f;
  shared.activation[linear] = f32_to_bf16_rne(value);
}

__device__ __forceinline__ void stage_weight(
    std::uint16_t *stage, const std::uint16_t *weight,
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

__device__ __forceinline__ void store_projection(
    float *output, const Args &args, std::uint32_t channel_count,
    std::uint32_t m_base, std::uint32_t channel_base, std::uint32_t lane,
    const float (&accumulator)[4]) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row =
        m_base + row_pair * 2u + (element & 1u);
    if (row < args.m && channel < channel_count) {
      output[static_cast<std::uint64_t>(row) * channel_count + channel] =
          accumulator[element];
    }
  }
}

} // namespace detail

inline ValidationResult validate(const Args &args) noexcept {
  if (args.args_version != kArgsVersion) {
    return ValidationResult::kInvalidAbi;
  }
  if (args.activation_f32 == 0u || args.projection1_weight_bf16 == 0u ||
      args.projection2_weight_bf16 == 0u ||
      args.projection1_output_f32 == 0u ||
      args.projection2_output_f32 == 0u) {
    return ValidationResult::kNullPointer;
  }
  if (!detail::aligned_device_address(args.activation_f32, 16u) ||
      !detail::aligned_device_address(args.projection1_weight_bf16, 16u) ||
      !detail::aligned_device_address(args.projection2_weight_bf16, 16u) ||
      !detail::aligned_device_address(args.projection1_output_f32, 16u) ||
      !detail::aligned_device_address(args.projection2_output_f32, 16u)) {
    return ValidationResult::kMisalignedPointer;
  }
  if (args.m == 0u || args.m > kMaxRows) {
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

__global__ __launch_bounds__(kThreads, 4) void kernel(Args args) {
  __shared__ detail::SharedStorage shared;

  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t m_base = blockIdx.y * kCtaRows;
  const std::uint32_t n_tile_base = blockIdx.x * kCtaColumns;
  const std::uint32_t channel_base = n_tile_base + warp * kMmaColumns;

  auto const *activation =
      detail::device_pointer<const float>(args.activation_f32);
  auto const *projection1_weight = detail::device_pointer<const std::uint16_t>(
      args.projection1_weight_bf16);
  auto const *projection2_weight = detail::device_pointer<const std::uint16_t>(
      args.projection2_weight_bf16);
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

inline cudaError_t launch(const Args &args) noexcept {
  if (validate(args) != ValidationResult::kSuccess) {
    return cudaErrorInvalidValue;
  }

  const std::uint32_t max_n = args.n1 > args.n2 ? args.n1 : args.n2;
  const std::uint32_t grid_x = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(max_n) + kCtaColumns - 1u) / kCtaColumns);
  const std::uint32_t grid_y = (args.m + kCtaRows - 1u) / kCtaRows;
  const dim3 grid(grid_x, grid_y, 1u);
  const auto stream = reinterpret_cast<cudaStream_t>(
      static_cast<std::uintptr_t>(args.stream));

  kernel<<<grid, kThreads, 0u, stream>>>(args);
  return cudaGetLastError();
}

} // namespace ferrule::cutlass::sm121_bf16_compressor_prefill

#endif // FERRULE_CUTLASS_SM121_BF16_COMPRESSOR_PREFILL_CUH_
