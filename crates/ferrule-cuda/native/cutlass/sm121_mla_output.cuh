#ifndef FERRULE_CUTLASS_SM121_MLA_OUTPUT_CUH_
#define FERRULE_CUTLASS_SM121_MLA_OUTPUT_CUH_

#include <cooperative_groups.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm120.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(FERRULE_CUTLASS_TARGET_SM) && FERRULE_CUTLASS_TARGET_SM != 121
#error "sm121_mla_output.cuh requires Ferrule's SM121a target"
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1210
#error "sm121_mla_output.cuh device code must target SM121"
#endif

namespace ferrule::cutlass::sm121_mla_output {

inline constexpr std::uint32_t kArgsVersion = 1u;
inline constexpr std::uint32_t kMmaRows = 8u;
inline constexpr std::uint32_t kMmaColumns = 16u;
inline constexpr std::uint32_t kKTile = 16u;
inline constexpr std::uint32_t kWarpSize = 32u;
inline constexpr std::uint32_t kWarps = 4u;
inline constexpr std::uint32_t kThreads = kWarpSize * kWarps;
inline constexpr std::uint32_t kGb10CooperativeBlocks = 160u;
inline constexpr std::uint32_t kMaxRows = 65535u * kMmaRows;

// One semantic MLA output transaction:
//
//   context_f32 [rows, context_size]
//     -> grouped output-A FP8/E8M0 [latent_size, group_input_size]
//     -> BF16-rounded latent_f32 [rows, latent_size]
//     -> latent FP8/E8M0 pack [rows, latent_size]
//     -> output-B FP8/E8M0 [hidden_size, latent_size]
//     -> output_f32 [rows, hidden_size]
//
// The caller-owned latent tensor is both the numerical BF16 boundary and the
// warmed cross-grid scratch. The provider issues one cooperative launch and no
// host-side GEMM, allocation, synchronization, or fallback.
struct Args {
  std::uint32_t args_version;
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
  std::uint64_t latent_f32;
  std::uint64_t latent_fp8;
  std::uint64_t latent_ue8m0;
  std::uint64_t output_f32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 120u, "SM121 MLA output POD ABI changed");

struct Binding {
  const float *context;
  const std::uint8_t *output_a_weight;
  const std::uint8_t *output_a_scales;
  const std::uint8_t *output_b_weight;
  const std::uint8_t *output_b_scales;
  float *latent;
  std::uint8_t *latent_fp8;
  std::uint8_t *latent_scales;
  float *output;
};

static_assert(std::is_trivially_copyable_v<Binding>);

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidAbi = 1,
  kInvalidArgument = 2,
  kLaunchFailed = 3,
};

namespace detail {

using Bf16Mma = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using Fp8Mma =
    cute::SM120_16x8x32_TN<cute::float_e4m3_t, cute::float_e4m3_t,
                           float>;

struct alignas(16) WarpStage {
  alignas(16) std::uint16_t weight[kMmaColumns * kKTile];
  alignas(16) std::uint16_t activation[kKTile * kMmaRows];
};

static_assert(sizeof(WarpStage) == 768u);

inline constexpr bool aligned(std::uint64_t address,
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
  const std::uint32_t sign =
      static_cast<std::uint32_t>(value & 0x80u) << 24;
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
  return __uint_as_float(sign | ((exponent + 120u) << 23) |
                         (mantissa << 20));
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

__device__ __forceinline__ void
mma_fp8(float (&accumulator)[4], const std::uint32_t (&weight)[4],
        const std::uint32_t (&activation)[2]) {
  Fp8Mma::fma(accumulator[0], accumulator[1], accumulator[2],
              accumulator[3], weight[0], weight[1], weight[2], weight[3],
              activation[0], activation[1], accumulator[0], accumulator[1],
              accumulator[2], accumulator[3]);
}

__device__ __forceinline__ std::uint8_t
ue8m0_scale_byte_for_amax(float amax) {
  if (!isfinite(amax) || amax <= 0.0f) {
    return 127u;
  }
  const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
  const int encoded = exponent + 127;
  return static_cast<std::uint8_t>(encoded < 0 ? 0 : (encoded > 255 ? 255 : encoded));
}

__device__ __forceinline__ std::uint8_t quantize_fp8(float value) {
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ void stage_f32_activation(
    std::uint16_t *stage, const float *activation, std::uint32_t row_base,
    std::uint32_t rows, std::uint32_t row_stride,
    std::uint32_t column_base, std::uint32_t lane) {
  for (std::uint32_t linear = lane; linear < kKTile * kMmaRows;
       linear += kWarpSize) {
    const std::uint32_t k_local = linear >> 3;
    const std::uint32_t row_local = linear & 7u;
    const std::uint32_t row = row_base + row_local;
    const float value =
        row < rows
            ? activation[static_cast<std::uint64_t>(row) * row_stride +
                         column_base + k_local]
            : 0.0f;
    stage[linear] = f32_to_bf16_rne(value);
  }
}

__device__ __forceinline__ void stage_output_a_weight(
    std::uint16_t *stage, const Binding &binding, const Args &args,
    std::uint32_t channel_base, std::uint32_t k_base,
    std::uint32_t lane) {
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
      static_cast<std::uint64_t>(channel / 128u) *
          args.output_a_scale_cols +
      k_base / 128u;
  const float scale =
      ue8m0_to_float(binding.output_a_scales[scale_offset]);
#pragma unroll
  for (std::uint32_t element = 0u; element < 8u; ++element) {
    const std::uint8_t value =
        static_cast<std::uint8_t>(packed >> (element * 8u));
    destination[element] =
        f32_to_bf16_rne(fp8_e4m3fn_to_float(value) * scale);
  }
}

__device__ __forceinline__ void stage_output_b_weight(
    std::uint16_t *stage, const std::uint8_t *weight, const Args &args,
    std::uint32_t channel_base, std::uint32_t k_base,
    std::uint32_t lane) {
  if (lane >= kMmaColumns) {
    return;
  }
  auto *destination = reinterpret_cast<uint4 *>(
      reinterpret_cast<std::uint8_t *>(stage) +
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

__device__ __forceinline__ void stage_output_b_activation(
    std::uint16_t *stage, const Binding &binding, const Args &args,
    std::uint32_t row_base, std::uint32_t k_base,
    std::uint32_t lane) {
  auto *bytes = reinterpret_cast<std::uint8_t *>(stage);
  for (std::uint32_t linear = lane; linear < 128u;
       linear += kWarpSize) {
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

__device__ __forceinline__ void pack_latent_task(
    const Args &args, const Binding &binding, std::uint32_t row,
    std::uint32_t scale_block, std::uint32_t lane) {
  const std::uint64_t base =
      static_cast<std::uint64_t>(row) * args.latent_size +
      scale_block * 128u;
  float values[4];
  float amax = 1.0e-4f;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const float value = binding.latent[base + lane + element * kWarpSize];
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

__device__ __forceinline__ void store_tile(
    float *output, std::uint32_t output_columns, std::uint32_t rows,
    std::uint32_t row_base, std::uint32_t channel_base, std::uint32_t lane,
    const float (&accumulator)[4], bool round_to_bf16) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row =
        row_base + row_pair * 2u + (element & 1u);
    if (row < rows && channel < output_columns) {
      const float value = round_to_bf16
                              ? bf16_to_f32(f32_to_bf16_rne(
                                    accumulator[element]))
                              : accumulator[element];
      output[static_cast<std::uint64_t>(row) * output_columns + channel] =
          value;
    }
  }
}

__device__ __forceinline__ void output_a_task(
    const Args &args, const Binding &binding, WarpStage &stage,
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
  store_tile(binding.latent, args.latent_size, args.rows, row_base,
             channel_base, lane, accumulator, true);
}

__device__ __forceinline__ void output_b_task(
    const Args &args, const Binding &binding, WarpStage &stage,
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
        binding.output_b_scales[static_cast<std::uint64_t>(channel_base / 128u) *
                                    scale_cols +
                                scale_block]);
    const std::uint32_t row_pair = lane & 3u;
#pragma unroll
    for (std::uint32_t element = 0u; element < 4u; ++element) {
      const std::uint32_t row =
          row_base + row_pair * 2u + (element & 1u);
      if (row < args.rows) {
        const float activation_scale = ue8m0_to_float(
            binding.latent_scales[static_cast<std::uint64_t>(row) * scale_cols +
                                  scale_block]);
        accumulator[element] +=
            block_accumulator[element] * weight_scale * activation_scale;
      }
    }
  }
  store_tile(binding.output, args.hidden_size, args.rows, row_base,
             channel_base, lane, accumulator, false);
}

__global__ __launch_bounds__(kThreads, 1) void kernel(Args args,
                                                      Binding binding) {
  extern __shared__ __align__(16) std::uint8_t storage[];
  auto *stages = reinterpret_cast<WarpStage *>(storage);
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp;
  const std::uint32_t warp_stride = gridDim.x * kWarps;
  const std::uint32_t row_tiles =
      (args.rows + kMmaRows - 1u) / kMmaRows;

  const std::uint32_t output_a_channel_tiles =
      (args.latent_size + kMmaColumns - 1u) / kMmaColumns;
  const std::uint32_t output_a_tasks =
      row_tiles * output_a_channel_tiles;
  for (std::uint32_t task = global_warp; task < output_a_tasks;
       task += warp_stride) {
    const std::uint32_t row_tile = task / output_a_channel_tiles;
    const std::uint32_t channel_tile =
        task - row_tile * output_a_channel_tiles;
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
  const std::uint32_t output_b_tasks =
      row_tiles * output_b_channel_tiles;
  for (std::uint32_t task = global_warp; task < output_b_tasks;
       task += warp_stride) {
    const std::uint32_t row_tile = task / output_b_channel_tiles;
    const std::uint32_t channel_tile =
        task - row_tile * output_b_channel_tiles;
    output_b_task(args, binding, stages[warp], row_tile * kMmaRows,
                  channel_tile * kMmaColumns, lane);
  }
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr || args->args_version != kArgsVersion) {
    return Status::kInvalidAbi;
  }
  if (args->rows == 0u || args->rows > kMaxRows || args->groups == 0u ||
      args->group_input_size == 0u || args->rank == 0u ||
      args->latent_size == 0u || args->hidden_size == 0u ||
      args->reserved != 0u ||
      args->context_size != args->groups * args->group_input_size ||
      args->latent_size != args->groups * args->rank ||
      args->output_a_scale_cols != args->group_input_size / 128u ||
      (args->group_input_size % 128u) != 0u ||
      (args->rank % kMmaColumns) != 0u ||
      (args->latent_size % 128u) != 0u) {
    return Status::kInvalidArgument;
  }
  const bool pointers_valid =
      detail::aligned(args->context_f32, 16u) &&
      detail::aligned(args->output_a_weight_fp8, 16u) &&
      detail::aligned(args->output_a_weight_ue8m0, 16u) &&
      detail::aligned(args->output_b_weight_fp8, 16u) &&
      detail::aligned(args->output_b_weight_ue8m0, 16u) &&
      detail::aligned(args->latent_f32, 16u) &&
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
  const std::uint32_t row_tiles =
      (args->rows + kMmaRows - 1u) / kMmaRows;
  const std::uint32_t output_a_tasks =
      row_tiles * ((args->latent_size + kMmaColumns - 1u) / kMmaColumns);
  const std::uint32_t pack_tasks =
      args->rows * (args->latent_size / 128u);
  const std::uint32_t output_b_tasks =
      row_tiles * ((args->hidden_size + kMmaColumns - 1u) / kMmaColumns);
  const std::uint32_t warp_tasks = max(output_a_tasks, max(pack_tasks, output_b_tasks));
  const std::uint32_t blocks =
      min(kGb10CooperativeBlocks, (warp_tasks + kWarps - 1u) / kWarps);
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
      reinterpret_cast<float *>(static_cast<std::uintptr_t>(args->latent_f32)),
      reinterpret_cast<std::uint8_t *>(
          static_cast<std::uintptr_t>(args->latent_fp8)),
      reinterpret_cast<std::uint8_t *>(
          static_cast<std::uintptr_t>(args->latent_ue8m0)),
      reinterpret_cast<float *>(static_cast<std::uintptr_t>(args->output_f32)),
  };
  void *kernel_args[] = {const_cast<Args *>(args),
                         const_cast<Binding *>(&binding)};
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::kernel), dim3(blocks, 1u, 1u),
      dim3(kThreads, 1u, 1u), kernel_args, required_shared_storage_bytes(),
      stream);
  return status == cudaSuccess ? Status::kSuccess : Status::kLaunchFailed;
}

} // namespace ferrule::cutlass::sm121_mla_output

#endif // FERRULE_CUTLASS_SM121_MLA_OUTPUT_CUH_
