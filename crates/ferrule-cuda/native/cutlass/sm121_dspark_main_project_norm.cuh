#ifndef FERRULE_CUTLASS_SM121_DSPARK_MAIN_PROJECT_NORM_CUH_
#define FERRULE_CUTLASS_SM121_DSPARK_MAIN_PROJECT_NORM_CUH_

#include <cooperative_groups.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm120.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(FERRULE_CUTLASS_TARGET_SM) && FERRULE_CUTLASS_TARGET_SM != 121
#error "sm121_dspark_main_project_norm.cuh requires Ferrule's SM121a target"
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1210
#error "sm121_dspark_main_project_norm.cuh device code must target SM121"
#endif

namespace ferrule::cutlass::sm121_dspark_main_project_norm {

inline constexpr std::uint32_t kArgsVersion = 1u;
inline constexpr std::uint32_t kMmaRows = 8u;
inline constexpr std::uint32_t kMmaColumns = 16u;
inline constexpr std::uint32_t kMmaK = 32u;
inline constexpr std::uint32_t kScaleK = 128u;
inline constexpr std::uint32_t kWarpSize = 32u;
inline constexpr std::uint32_t kWarps = 4u;
inline constexpr std::uint32_t kThreads = kWarpSize * kWarps;
inline constexpr std::uint32_t kGb10CooperativeBlocks = 160u;
inline constexpr std::uint32_t kMaxRows = 65535u;

// One semantic DSpark stage-zero transaction:
//
//   target_taps_f32 [rows, input_size]
//     -> K128 FP8/E8M0 activation pack
//     -> FP8/E8M0 main projection [output_size, input_size]
//     -> BF16-rounded projected boundary [rows, output_size]
//     -> RMSNorm with F32-exposed BF16 checkpoint weights
//     -> BF16-rounded output_f32 [rows, output_size]
//
// The provider issues one cooperative launch. Ferrule owns the stream, weights,
// output, and graph-stable activation/inverse-RMS scratch. The provider performs
// no allocation, host synchronization, or fallback dispatch.
struct Args {
  std::uint32_t args_version;
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
static_assert(sizeof(Args) == 104u,
              "SM121 DSpark main-project/norm POD ABI changed");
static_assert(offsetof(Args, input_f32) == 32u);
static_assert(offsetof(Args, stream) == 96u);

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidAbi = 1,
  kInvalidArgument = 2,
  kLaunchFailed = 3,
};

namespace detail {

using Fp8Mma =
    cute::SM120_16x8x32_TN<cute::float_e4m3_t, cute::float_e4m3_t,
                           float>;

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

inline constexpr bool aligned(std::uint64_t address,
                              std::uint64_t alignment) {
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

__device__ __forceinline__ std::uint8_t
ue8m0_scale_byte_for_amax(float amax) {
  // Match the official DSpark act_quant path: clamp amax to 1e-4 and round
  // amax / 448 upward to a power of two represented as UE8M0.
  amax = fmaxf(amax, 1.0e-4f);
  const int exponent = static_cast<int>(ceilf(log2f(amax / 448.0f)));
  const int encoded = exponent + 127;
  return static_cast<std::uint8_t>(encoded < 0 ? 0 : (encoded > 255 ? 255 : encoded));
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
  const std::uint32_t row =
      (lane & 7u) + ((quad & 1u) != 0u ? 8u : 0u);
  const std::uint32_t column_bytes = quad >= 2u ? 16u : 0u;
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + row * kMmaK + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1],
                                fragment[2], fragment[3]);
}

__device__ __forceinline__ void
load_activation_fragment(const std::uint8_t *shared, std::uint32_t lane,
                         std::uint32_t (&fragment)[2]) {
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void
mma_fp8(float (&accumulator)[4], const std::uint32_t (&weight)[4],
        const std::uint32_t (&activation)[2]) {
  Fp8Mma::fma(accumulator[0], accumulator[1], accumulator[2],
              accumulator[3], weight[0], weight[1], weight[2], weight[3],
              activation[0], activation[1], accumulator[0], accumulator[1],
              accumulator[2], accumulator[3]);
}

__device__ __forceinline__ void pack_activation_task(
    const Args &args, const Binding &binding, std::uint32_t row,
    std::uint32_t scale_block, std::uint32_t lane) {
  const std::uint64_t base =
      static_cast<std::uint64_t>(row) * args.input_size +
      scale_block * kScaleK;
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
    binding.activation_scales[static_cast<std::uint64_t>(row) *
                                  args.scale_cols +
                              scale_block] =
        static_cast<std::uint8_t>(scale_byte);
  }
  const float scale = ue8m0_to_float(static_cast<std::uint8_t>(scale_byte));
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    binding.activation[base + lane + element * kWarpSize] =
        quantize_fp8(values[element] / scale);
  }
}

__device__ __forceinline__ void stage_weight(
    WarpStage &stage, const Binding &binding, const Args &args,
    std::uint32_t channel_base, std::uint32_t k_base,
    std::uint32_t lane) {
  const std::uint32_t local_channel = lane >> 1;
  const std::uint32_t half = lane & 1u;
  const std::uint32_t channel = channel_base + local_channel;
  auto *destination = reinterpret_cast<uint4 *>(
      stage.weight + local_channel * kMmaK + half * 16u);
  if (channel < args.output_size) {
    auto const *source = reinterpret_cast<const uint4 *>(
        binding.weight + static_cast<std::uint64_t>(channel) *
                             args.input_size +
                         k_base + half * 16u);
    *destination = *source;
  } else {
    *destination = make_uint4(0u, 0u, 0u, 0u);
  }
}

__device__ __forceinline__ void stage_activation(
    WarpStage &stage, const Binding &binding, const Args &args,
    std::uint32_t row_base, std::uint32_t k_base,
    std::uint32_t lane) {
  if (lane >= kMmaK / 2u) {
    return;
  }
  auto *destination = reinterpret_cast<std::uint16_t *>(
      stage.activation + lane * kMmaRows * 2u);
#pragma unroll
  for (std::uint32_t row_local = 0u; row_local < kMmaRows; ++row_local) {
    const std::uint32_t row = row_base + row_local;
    destination[row_local] =
        row < args.rows
            ? *reinterpret_cast<const std::uint16_t *>(
                  binding.activation +
                  static_cast<std::uint64_t>(row) * args.input_size +
                  k_base + lane * 2u)
            : 0u;
  }
}

__device__ __forceinline__ void projection_task(
    const Args &args, const Binding &binding, WarpStage &stage,
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
            ? ue8m0_to_float(binding.weight_scales[
                  static_cast<std::uint64_t>(channel_base / kScaleK) *
                      args.scale_cols +
                  scale_block])
            : 0.0f;
    const std::uint32_t row_pair = lane & 3u;
#pragma unroll
    for (std::uint32_t element = 0u; element < 4u; ++element) {
      const std::uint32_t row =
          row_base + row_pair * 2u + (element & 1u);
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
    const std::uint32_t row =
        row_base + row_pair * 2u + (element & 1u);
    if (row < args.rows && channel < args.output_size) {
      binding.output[static_cast<std::uint64_t>(row) * args.output_size +
                     channel] =
          bf16_to_f32(f32_to_bf16_rne(accumulator[element]));
    }
  }
}

__device__ __forceinline__ void inverse_rms_task(
    const Args &args, const Binding &binding, std::uint32_t row,
    std::uint32_t lane) {
  float sum = 0.0f;
  const std::uint64_t base =
      static_cast<std::uint64_t>(row) * args.output_size;
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

__device__ __forceinline__ void normalize_task(
    const Args &args, const Binding &binding, std::uint32_t row,
    std::uint32_t channel_base, std::uint32_t lane) {
  const std::uint64_t row_base =
      static_cast<std::uint64_t>(row) * args.output_size;
  const float inv_rms = binding.inv_rms[row];
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + lane + element * kWarpSize;
    if (channel < args.output_size) {
      const float normalized =
          binding.output[row_base + channel] * inv_rms *
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

  const std::uint32_t row_tiles =
      (args.rows + kMmaRows - 1u) / kMmaRows;
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

  for (std::uint32_t row = global_warp; row < args.rows;
       row += warp_stride) {
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
  if (args == nullptr || args->args_version != kArgsVersion) {
    return Status::kInvalidAbi;
  }
  if (args->rows == 0u || args->rows > kMaxRows ||
      args->input_size == 0u || args->output_size == 0u ||
      args->scale_cols != args->input_size / kScaleK ||
      (args->input_size % kScaleK) != 0u ||
      (args->output_size % kScaleK) != 0u || args->reserved0 != 0u ||
      args->reserved1 != 0u || !(args->rms_eps > 0.0f)) {
    return Status::kInvalidArgument;
  }
  const bool pointers_valid =
      detail::aligned(args->input_f32, 16u) &&
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
  const std::uint32_t row_tiles =
      (args->rows + kMmaRows - 1u) / kMmaRows;
  const std::uint32_t projection_tasks =
      row_tiles * ((args->output_size + kMmaColumns - 1u) / kMmaColumns);
  const std::uint32_t norm_tasks =
      args->rows * (args->output_size / kScaleK);
  std::uint32_t warp_tasks = pack_tasks > projection_tasks
                                 ? pack_tasks
                                 : projection_tasks;
  warp_tasks = warp_tasks > args->rows ? warp_tasks : args->rows;
  warp_tasks = warp_tasks > norm_tasks ? warp_tasks : norm_tasks;
  std::uint32_t blocks = (warp_tasks + kWarps - 1u) / kWarps;
  blocks = blocks < kGb10CooperativeBlocks ? blocks
                                           : kGb10CooperativeBlocks;

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
      reinterpret_cast<float *>(
          static_cast<std::uintptr_t>(args->inv_rms_f32)),
      reinterpret_cast<float *>(
          static_cast<std::uintptr_t>(args->output_f32)),
  };
  void *kernel_args[] = {const_cast<Args *>(args),
                         const_cast<detail::Binding *>(&binding)};
  const auto stream = reinterpret_cast<cudaStream_t>(
      static_cast<std::uintptr_t>(args->stream));
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::kernel), dim3(blocks, 1u, 1u),
      dim3(kThreads, 1u, 1u), kernel_args, required_shared_storage_bytes(),
      stream);
  return status == cudaSuccess ? Status::kSuccess : Status::kLaunchFailed;
}

} // namespace ferrule::cutlass::sm121_dspark_main_project_norm

#endif // FERRULE_CUTLASS_SM121_DSPARK_MAIN_PROJECT_NORM_CUH_
