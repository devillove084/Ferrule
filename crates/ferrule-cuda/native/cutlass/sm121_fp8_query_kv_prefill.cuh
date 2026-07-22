#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cutlass/version.h>

#if CUTLASS_MAJOR != 4 || CUTLASS_MINOR != 6 || CUTLASS_PATCH != 1
#error "sm121_fp8_query_kv_prefill.cuh requires CUTLASS 4.6.1"
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1210
#error "sm121_fp8_query_kv_prefill.cuh is only valid for SM121a device code"
#endif

namespace ferrule::sm121_fp8_query_kv_prefill {

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
      args.kv_weight_ue8m0 == nullptr ||
      args.query_a_output_f32 == nullptr || args.kv_output_f32 == nullptr) {
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

constexpr std::uint32_t kTileM = kCtaRows;
constexpr std::uint32_t kTileN = 16;
constexpr std::uint32_t kScaleK = 128;
constexpr std::uint32_t kMmaK = 32;
constexpr std::uint32_t kMmaKSteps = kScaleK / kMmaK;
constexpr std::uint32_t kThreads = 64;
constexpr std::uint32_t kStages = 2;

// GB10 executes the forward-compatible dense FP8 encoding. CUDA 13.0 ptxas
// rejects SM120's .kind::f8f6f4 and block_scale encodings for sm_121a.
using Fp8Mma = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;

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

template <class T>
__device__ __forceinline__ T device_load(const T *pointer) {
  return *pointer;
}

__device__ __forceinline__ void cp_async_16(void *destination,
                                            const void *source) {
#if defined(__CUDA_ARCH__)
  std::uint32_t shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                   "r"(shared_address),
               "l"(source)
               : "memory");
#else
  (void)destination;
  (void)source;
#endif
}

__device__ __forceinline__ void cp_async_commit_and_wait() {
#if defined(__CUDA_ARCH__)
  asm volatile("cp.async.commit_group;\n" :: : "memory");
  asm volatile("cp.async.wait_group 0;\n" :: : "memory");
#endif
}

__device__ __forceinline__ void clear_16(void *destination) {
  auto *words = static_cast<std::uint32_t *>(destination);
  words[0] = 0u;
  words[1] = 0u;
  words[2] = 0u;
  words[3] = 0u;
}

__device__ __forceinline__ void wait_until(
    const volatile std::int32_t *state, std::int32_t expected) {
  while (*state != expected) {
#if defined(__CUDA_ARCH__)
    __nanosleep(32);
#endif
  }
}

__device__ __forceinline__ void load_a_fragment(
    const std::uint8_t *shared, std::uint32_t lane,
    std::uint32_t (&fragment)[4]) {
  std::uint32_t quad = lane >> 3;
  std::uint32_t row = (lane & 7u) + ((quad & 1u) != 0u ? 8u : 0u);
  std::uint32_t column_bytes = quad >= 2u ? 16u : 0u;
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + row * kMmaK + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1],
                                fragment[2], fragment[3]);
}

__device__ __forceinline__ void load_b_fragment(
    const std::uint8_t *shared, std::uint32_t lane,
    std::uint32_t (&fragment)[2]) {
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + (lane & 15u) * 16u);
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
  Fp8Mma::fma(accumulator[0], accumulator[1], accumulator[2],
              accumulator[3], a[0], a[1], a[2], a[3], b[0], b[1],
              accumulator[0], accumulator[1], accumulator[2],
              accumulator[3]);
}

struct QueryAKvCollective {
  template <bool DualProjection>
  __device__ static void producer(const Args &args,
                                  SharedStorage<DualProjection> &shared,
                                  std::uint32_t lane,
                                  std::uint32_t row_base,
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
      for (std::uint32_t chunk = lane;
           chunk < (kTileN * kScaleK) / 16u; chunk += 32u) {
        const std::uint32_t k_step = chunk / 32u;
        const std::uint32_t tile_chunk = chunk & 31u;
        const std::uint32_t local_channel = tile_chunk >> 1;
        const std::uint32_t k_half = tile_chunk & 1u;
        const std::uint32_t channel = channel_base + local_channel;
        const std::uint64_t k_offset =
            k_base + static_cast<std::uint64_t>(k_step) * kMmaK +
            k_half * 16u;

        void *query_destination =
            stage.query_a_weight[k_step] + local_channel * kMmaK +
            k_half * 16u;
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
              stage.kv_weight[k_step] + local_channel * kMmaK +
              k_half * 16u;
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
      // eight rows. Build that layout while the two weight copies are in flight.
      for (std::uint32_t segment = lane;
           segment < kMmaKSteps * (kMmaK / 2u); segment += 32u) {
        const std::uint32_t k_step = segment / (kMmaK / 2u);
        const std::uint32_t k_pair = segment & (kMmaK / 2u - 1u);
        auto *destination = reinterpret_cast<std::uint16_t *>(
            stage.activation[k_step] + k_pair * 16u);
        const std::uint64_t k_offset =
            k_base + static_cast<std::uint64_t>(k_step) * kMmaK +
            k_pair * 2u;

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
                                  std::uint32_t lane,
                                  std::uint32_t row_base,
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
        mma(query_a_block_accumulator, query_a_fragment,
            activation_fragment);
        if constexpr (DualProjection) {
          std::uint32_t kv_fragment[4];
          load_a_fragment(stage.kv_weight[k_step], lane, kv_fragment);
          mma(kv_block_accumulator, kv_fragment, activation_fragment);
        }
      }

      const float query_weight_scale =
          channel_base < args.n_query_a
              ? ue8m0_to_float(args.query_a_weight_ue8m0[
                    static_cast<std::uint64_t>(channel_base / 128u) *
                        scale_columns +
                    scale_block])
              : 0.0f;
      float kv_weight_scale = 0.0f;
      if constexpr (DualProjection) {
        kv_weight_scale =
            channel_base < args.n_kv
                ? ue8m0_to_float(args.kv_weight_ue8m0[
                      static_cast<std::uint64_t>(channel_base / 128u) *
                          scale_columns +
                      scale_block])
                : 0.0f;
      }

      const std::uint32_t row_pair = lane & 3u;
#pragma unroll
      for (std::uint32_t element = 0; element < 4u; ++element) {
        const std::uint32_t row =
            row_base + row_pair * 2u + (element & 1u);
        if (row < args.m) {
          const float activation_scale =
              ue8m0_to_float(args.activation_ue8m0[
                  static_cast<std::uint64_t>(row) * scale_columns +
                  scale_block]);
          query_a_accumulator[element] +=
              query_a_block_accumulator[element] * query_weight_scale *
              activation_scale;
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
      const std::uint32_t row =
          row_base + row_pair * 2u + (element & 1u);
      if (row < args.m && channel < args.n_query_a) {
        args.query_a_output_f32[
            static_cast<std::uint64_t>(row) * args.n_query_a + channel] =
            query_a_accumulator[element];
      }
      if constexpr (DualProjection) {
        if (row < args.m && channel < args.n_kv) {
          args.kv_output_f32[static_cast<std::uint64_t>(row) * args.n_kv +
                             channel] = kv_accumulator[element];
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
  const std::uint32_t grid_n =
      1u + (max_n - 1u) / detail::kTileN;
  const dim3 grid(grid_n,
                  (args.m + detail::kTileM - 1u) / detail::kTileM, 1u);
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

  const std::uint32_t grid_n =
      1u + (args.n_query_a - 1u) / detail::kTileN;
  const dim3 grid(grid_n,
                  (args.m + detail::kTileM - 1u) / detail::kTileM, 1u);
  kernel<false><<<grid, detail::kThreads, 0, stream>>>(args);
  return cudaPeekAtLastError();
}

} // namespace ferrule::sm121_fp8_query_kv_prefill
