#ifndef FERRULE_CUTLASS_SM121_FP4_MOE_CUH_
#define FERRULE_CUTLASS_SM121_FP4_MOE_CUH_

#include <cooperative_groups.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm120.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ferrule::cutlass::sm121_fp4_moe {

inline constexpr uint32_t kArgsVersion = 1u;
inline constexpr uint32_t kSegmentColumns = 8u;
inline constexpr uint32_t kThreads = 128u;
inline constexpr uint32_t kWarps = 4u;
inline constexpr uint32_t kMmaChannels = 16u;
// 128 threads and 3 KiB shared memory per CTA permit eight resident CTAs per
// GB10 SM for this specialization. A larger cooperative grid reduces the
// number of channel-task waves for V=8's many active expert segments.
inline constexpr uint32_t kGb10CooperativeBlocks = 160u;

// Stable-frame semantic POD. The three hidden scratch addresses are internal
// cross-CTA storage owned by Ferrule's warmed segment workspace.
struct Args {
  uint32_t args_version;
  uint32_t rows;
  uint32_t input_size;
  uint32_t intermediate_size;
  uint32_t hidden_size;
  uint32_t num_tokens;
  uint32_t num_routes;
  uint32_t slot_capacity;
  uint32_t num_segments;
  float swiglu_limit;

  uint64_t x_packed;
  uint64_t x_scales;

  uint64_t gate_ptrs;
  uint64_t gate_scale_ptrs;
  uint64_t up_ptrs;
  uint64_t up_scale_ptrs;
  uint64_t down_ptrs;
  uint64_t down_scale_ptrs;
  uint64_t slot_generations;

  uint64_t segment_expert_slots;
  uint64_t segment_generations;
  uint64_t segment_token_indices;
  uint64_t segment_route_indices;
  uint64_t segment_route_weights;

  uint64_t segment_states;
  uint64_t segment_bindings;
  uint64_t hidden_f32;
  uint64_t hidden_packed;
  uint64_t hidden_scales;
  uint64_t route_written;
  uint64_t route_error;
  uint64_t route_output;
  uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 224u, "SM121 routed MXFP4 POD ABI changed");

enum class Status : int32_t {
  kSuccess = 0,
  kInvalidAbi = 1,
  kInvalidArgument = 2,
  kUnsupportedResources = 3,
  kLaunchFailed = 4,
};

namespace detail {

using Mxfp4Atom =
    cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
        cute::float_e2m1_t, cute::float_e2m1_t, float,
        cute::float_ue8m0_t, 32>;
using ScaleRegister = typename Mxfp4Atom::RegTypeSF;

struct alignas(16) WarpStage {
  alignas(16) uint8_t a[512];
  alignas(16) uint8_t b[256];
};

static_assert(sizeof(WarpStage) == 768u);

inline constexpr bool aligned(uint64_t address, uint64_t alignment) {
  return address != 0u && (address & (alignment - 1u)) == 0u;
}

template <class T>
__device__ __forceinline__ T *device_pointer(uint64_t address) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(address));
}

__device__ __forceinline__ void set_route_error(int32_t *route_error) {
  atomicOr(reinterpret_cast<unsigned int *>(route_error), 1u);
}

__device__ __forceinline__ void
load_a_fragment(const uint8_t *shared, uint32_t lane,
                uint32_t (&fragment)[4]) {
  const uint32_t quad = lane >> 3;
  const uint32_t row = (lane & 7u) + ((quad & 1u) != 0u ? 8u : 0u);
  const uint32_t column_bytes = quad >= 2u ? 16u : 0u;
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + row * 32u + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1],
                                fragment[2], fragment[3]);
}

__device__ __forceinline__ void
load_b_fragment(const uint8_t *shared, uint32_t lane,
                uint32_t (&fragment)[2]) {
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      shared + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void
stage_weight_tile(uint8_t *shared, const uint8_t *weight,
                  uint32_t channel_base, uint32_t channel_limit,
                  uint32_t packed_columns, uint32_t packed_k,
                  uint32_t lane) {
  if (lane >= kMmaChannels) {
    return;
  }
  auto *destination =
      reinterpret_cast<uint4 *>(shared + static_cast<uint64_t>(lane) * 32u);
  const uint32_t channel = channel_base + lane;
  if (channel < channel_limit) {
    auto *source = reinterpret_cast<const uint4 *>(
        weight + static_cast<uint64_t>(channel) * packed_columns + packed_k);
    destination[0] = source[0];
    destination[1] = source[1];
  } else {
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    destination[0] = zero;
    destination[1] = zero;
  }
}

__device__ __forceinline__ void
mma_mxfp4(float (&accumulator)[4], const uint32_t (&a)[4],
          const uint32_t (&b)[2], uint16_t scale_a, uint16_t scale_b) {
  const ScaleRegister sfa = static_cast<ScaleRegister>(scale_a);
  const ScaleRegister sfb = static_cast<ScaleRegister>(scale_b);
  Mxfp4Atom::fma(accumulator[0], accumulator[1], accumulator[2],
                  accumulator[3], a[0], a[1], a[2], a[3], b[0], b[1],
                  accumulator[0], accumulator[1], accumulator[2],
                  accumulator[3], sfa, sfb);
}

__device__ __forceinline__ float e8m0_scale(uint8_t value) {
  const uint32_t bits =
      value == 0u ? (1u << 22) : (static_cast<uint32_t>(value) << 23);
  return __uint_as_float(bits);
}

__device__ __forceinline__ float quantize_fp8_e4m3fn(float value) {
  return static_cast<float>(__nv_fp8_e4m3(value));
}

__device__ __forceinline__ uint8_t e8m0_scale_byte_for_amax(float amax) {
  if (!isfinite(amax) || amax <= 0.0f) {
    return 127u;
  }
  const int exponent = static_cast<int>(ceilf(log2f(amax / 6.0f)));
  const int encoded = exponent + 127;
  return static_cast<uint8_t>(encoded < 0 ? 0 : (encoded > 255 ? 255 : encoded));
}

__device__ __forceinline__ float fp4_e2m1_value(uint8_t value) {
  constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                2.0f, 3.0f, 4.0f, 6.0f};
  return values[value & 7u];
}

__device__ __forceinline__ uint8_t quantize_fp4_e2m1(float value) {
  if (!isfinite(value) || value == 0.0f) {
    return 0u;
  }
  const uint8_t sign = value < 0.0f ? 8u : 0u;
  const float magnitude = fminf(fabsf(value), 6.0f);
  uint8_t best = 0u;
  float best_error = magnitude;
#pragma unroll
  for (uint8_t candidate = 1u; candidate < 8u; ++candidate) {
    const float error = fabsf(fp4_e2m1_value(candidate) - magnitude);
    if (error < best_error) {
      best = candidate;
      best_error = error;
    }
  }
  return sign | best;
}

__device__ __forceinline__ float swiglu(float gate, float up,
                                        float route_weight,
                                        float limit) {
  if (limit > 0.0f) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
  }
  const float sigmoid = 1.0f / (1.0f + __expf(-gate));
  return quantize_fp8_e4m3fn(gate * sigmoid * up * route_weight);
}

struct Bindings {
  const uint8_t *gate;
  const uint8_t *gate_scales;
  const uint8_t *up;
  const uint8_t *up_scales;
  const uint8_t *down;
  const uint8_t *down_scales;
};

__device__ __forceinline__ void prepare_segment(const Args &args,
                                                uint32_t segment) {
  auto *states = device_pointer<int32_t>(args.segment_states);
  auto *cached = device_pointer<uint64_t>(args.segment_bindings);
  auto *segment_slots =
      device_pointer<const int32_t>(args.segment_expert_slots);
  const int32_t slot = segment_slots[segment];
  if (slot == -1) {
    states[segment] = 0;
    return;
  }

  bool valid = slot >= 0 && static_cast<uint32_t>(slot) < args.slot_capacity;
  const uint32_t expert = valid ? static_cast<uint32_t>(slot) : 0u;
  uint64_t bindings[6] = {};
  if (valid) {
    auto *gate_ptrs = device_pointer<const uint64_t>(args.gate_ptrs);
    auto *gate_scale_ptrs =
        device_pointer<const uint64_t>(args.gate_scale_ptrs);
    auto *up_ptrs = device_pointer<const uint64_t>(args.up_ptrs);
    auto *up_scale_ptrs =
        device_pointer<const uint64_t>(args.up_scale_ptrs);
    auto *down_ptrs = device_pointer<const uint64_t>(args.down_ptrs);
    auto *down_scale_ptrs =
        device_pointer<const uint64_t>(args.down_scale_ptrs);
    bindings[0] = gate_ptrs[expert];
    bindings[1] = gate_scale_ptrs[expert];
    bindings[2] = up_ptrs[expert];
    bindings[3] = up_scale_ptrs[expert];
    bindings[4] = down_ptrs[expert];
    bindings[5] = down_scale_ptrs[expert];
    auto *slot_generations =
        device_pointer<const int32_t>(args.slot_generations);
    auto *segment_generations =
        device_pointer<const int32_t>(args.segment_generations);
    valid = segment_generations[segment] > 0 &&
            slot_generations[expert] == segment_generations[segment];
#pragma unroll
    for (uint32_t pointer = 0; pointer < 6u; ++pointer) {
      valid = valid && bindings[pointer] != 0u;
    }
    valid = valid && (bindings[0] & 15u) == 0u &&
            (bindings[2] & 15u) == 0u && (bindings[4] & 15u) == 0u;
  }

  auto *segment_tokens =
      device_pointer<const int32_t>(args.segment_token_indices);
  auto *segment_routes =
      device_pointer<const int32_t>(args.segment_route_indices);
  const uint64_t metadata =
      static_cast<uint64_t>(segment) * kSegmentColumns;
#pragma unroll
  for (uint32_t column = 0u; column < kSegmentColumns; ++column) {
    const int32_t token = segment_tokens[metadata + column];
    const int32_t route = segment_routes[metadata + column];
    const bool padding = token == -1 && route == -1;
    const bool populated =
        token >= 0 && static_cast<uint32_t>(token) < args.num_tokens &&
        route >= 0 && static_cast<uint32_t>(route) < args.num_routes;
    valid = valid && (padding || populated);
  }
  if (!valid) {
    states[segment] = -1;
    set_route_error(device_pointer<int32_t>(args.route_error));
    return;
  }
#pragma unroll
  for (uint32_t pointer = 0; pointer < 6u; ++pointer) {
    cached[static_cast<uint64_t>(segment) * 6u + pointer] =
        bindings[pointer];
  }
  states[segment] = 1;
}

__device__ __forceinline__ bool load_segment_bindings(
    const Args &args, uint32_t segment, Bindings &bindings) {
  auto *states = device_pointer<const int32_t>(args.segment_states);
  if (states[segment] != 1) {
    return false;
  }
  auto *cached = device_pointer<const uint64_t>(args.segment_bindings);
  const uint64_t base = static_cast<uint64_t>(segment) * 6u;
  bindings.gate = device_pointer<const uint8_t>(cached[base]);
  bindings.gate_scales = device_pointer<const uint8_t>(cached[base + 1u]);
  bindings.up = device_pointer<const uint8_t>(cached[base + 2u]);
  bindings.up_scales = device_pointer<const uint8_t>(cached[base + 3u]);
  bindings.down = device_pointer<const uint8_t>(cached[base + 4u]);
  bindings.down_scales = device_pointer<const uint8_t>(cached[base + 5u]);
  return true;
}

__device__ __forceinline__ void gate_up_task(
    const Args &args, WarpStage &stage, uint32_t segment,
    uint32_t channel_block, uint32_t lane, const Bindings &bindings) {
  const uint32_t channel_base = channel_block * kMmaChannels;
  const uint32_t input_packed_cols = args.input_size / 2u;
  const uint32_t input_scale_cols = args.input_size / 32u;
  const uint64_t segment_metadata =
      static_cast<uint64_t>(segment) * kSegmentColumns;
  auto *segment_tokens =
      device_pointer<const int32_t>(args.segment_token_indices);
  auto *segment_weights =
      device_pointer<const float>(args.segment_route_weights);
  auto *x_packed = device_pointer<const uint8_t>(args.x_packed);
  auto *x_scales = device_pointer<const uint8_t>(args.x_scales);
  float gate_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up_accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t k = 0u; k < args.input_size; k += 64u) {
    for (uint32_t linear = lane; linear < 128u; linear += 32u) {
      const uint32_t k4 = linear >> 3;
      const uint32_t column = linear & 7u;
      const uint32_t destination = k4 * 16u + column * 2u;
      const int32_t token = segment_tokens[segment_metadata + column];
      if (token >= 0) {
        const uint64_t source =
            static_cast<uint64_t>(token) * input_packed_cols + k / 2u +
            k4 * 2u;
        stage.b[destination] = x_packed[source];
        stage.b[destination + 1u] = x_packed[source + 1u];
      } else {
        stage.b[destination] = 0u;
        stage.b[destination + 1u] = 0u;
      }
    }
    stage_weight_tile(stage.a, bindings.gate, channel_base,
                      args.intermediate_size, input_packed_cols, k / 2u, lane);
    __syncwarp();

    uint32_t a_fragment[4];
    uint32_t b_fragment[2];
    load_a_fragment(stage.a, lane, a_fragment);
    load_b_fragment(stage.b, lane, b_fragment);
    const uint32_t group = lane >> 2;
    const uint32_t scale_row = group + ((lane & 1u) != 0u ? 8u : 0u);
    const uint32_t logical_channel = channel_base + scale_row;
    const uint32_t scale_block = k / 32u;
    const uint64_t gate_scale_offset =
        static_cast<uint64_t>(logical_channel) * input_scale_cols +
        scale_block;
    const uint16_t gate_scale = static_cast<uint16_t>(
        static_cast<uint16_t>(bindings.gate_scales[gate_scale_offset]) |
        (static_cast<uint16_t>(
             bindings.gate_scales[gate_scale_offset + 1u])
         << 8));
    const int32_t scale_token =
        segment_tokens[segment_metadata + group];
    uint16_t activation_scale = static_cast<uint16_t>(127u | (127u << 8));
    if (scale_token >= 0) {
      const uint64_t activation_scale_offset =
          static_cast<uint64_t>(scale_token) * input_scale_cols + scale_block;
      activation_scale = static_cast<uint16_t>(
          static_cast<uint16_t>(x_scales[activation_scale_offset]) |
          (static_cast<uint16_t>(x_scales[activation_scale_offset + 1u])
           << 8));
    }
    mma_mxfp4(gate_accumulator, a_fragment, b_fragment, gate_scale,
               activation_scale);
    __syncwarp();

    stage_weight_tile(stage.a, bindings.up, channel_base,
                      args.intermediate_size, input_packed_cols, k / 2u, lane);
    __syncwarp();
    load_a_fragment(stage.a, lane, a_fragment);
    const uint64_t up_scale_offset =
        static_cast<uint64_t>(logical_channel) * input_scale_cols +
        scale_block;
    const uint16_t up_scale = static_cast<uint16_t>(
        static_cast<uint16_t>(bindings.up_scales[up_scale_offset]) |
        (static_cast<uint16_t>(bindings.up_scales[up_scale_offset + 1u])
         << 8));
    mma_mxfp4(up_accumulator, a_fragment, b_fragment, up_scale,
               activation_scale);
    __syncwarp();
  }

  auto *hidden_f32 = device_pointer<float>(args.hidden_f32);
  const uint32_t channel_group = lane >> 2;
  const uint32_t row_pair = lane & 3u;
#pragma unroll
  for (uint32_t element = 0u; element < 4u; ++element) {
    const uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const uint32_t row = row_pair * 2u + (element & 1u);
    if (channel < args.intermediate_size) {
      hidden_f32[(static_cast<uint64_t>(segment) * kSegmentColumns + row) *
                     args.intermediate_size +
                 channel] =
          swiglu(gate_accumulator[element], up_accumulator[element],
                  segment_weights[segment_metadata + row],
                  args.swiglu_limit);
    }
  }
}

__device__ __forceinline__ void pack_task(const Args &args,
                                          uint32_t segment,
                                          uint32_t row,
                                          uint32_t scale_block,
                                          uint32_t lane) {
  auto *hidden_f32 = device_pointer<const float>(args.hidden_f32);
  auto *hidden_packed = device_pointer<uint8_t>(args.hidden_packed);
  auto *hidden_scales = device_pointer<uint8_t>(args.hidden_scales);
  const uint32_t scale_cols = args.intermediate_size / 32u;
  const uint32_t packed_cols = args.intermediate_size / 2u;
  const uint64_t f32_base =
      (static_cast<uint64_t>(segment) * kSegmentColumns + row) *
          args.intermediate_size +
      scale_block * 32u;
  const float value = hidden_f32[f32_base + lane];
  float amax = fabsf(value);
#pragma unroll
  for (uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta));
  }
  uint32_t scale_byte = e8m0_scale_byte_for_amax(amax);
  scale_byte = __shfl_sync(0xffffffffu, scale_byte, 0);
  const uint32_t nibble = quantize_fp4_e2m1(
      value / e8m0_scale(static_cast<uint8_t>(scale_byte)));
  const uint32_t low =
      __shfl_sync(0xffffffffu, nibble, (lane * 2u) & 31u);
  const uint32_t high =
      __shfl_sync(0xffffffffu, nibble, (lane * 2u + 1u) & 31u);
  if (lane == 0u) {
    hidden_scales[(static_cast<uint64_t>(segment) * kSegmentColumns + row) *
                      scale_cols +
                  scale_block] = static_cast<uint8_t>(scale_byte);
  }
  if (lane < 16u) {
    hidden_packed[(static_cast<uint64_t>(segment) * kSegmentColumns + row) *
                      packed_cols +
                  scale_block * 16u + lane] =
        static_cast<uint8_t>(low | (high << 4));
  }
}

__device__ __forceinline__ void down_task(
    const Args &args, WarpStage &stage, uint32_t segment,
    uint32_t channel_block, uint32_t lane, const Bindings &bindings) {
  const uint32_t channel_base = channel_block * kMmaChannels;
  const uint32_t intermediate_packed_cols = args.intermediate_size / 2u;
  const uint32_t intermediate_scale_cols = args.intermediate_size / 32u;
  auto *hidden_packed = device_pointer<const uint8_t>(args.hidden_packed);
  auto *hidden_scales = device_pointer<const uint8_t>(args.hidden_scales);
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t k = 0u; k < args.intermediate_size; k += 64u) {
    for (uint32_t linear = lane; linear < 128u; linear += 32u) {
      const uint32_t k4 = linear >> 3;
      const uint32_t column = linear & 7u;
      const uint32_t destination = k4 * 16u + column * 2u;
      const uint64_t source =
          (static_cast<uint64_t>(segment) * kSegmentColumns + column) *
              intermediate_packed_cols +
          k / 2u + k4 * 2u;
      stage.b[destination] = hidden_packed[source];
      stage.b[destination + 1u] = hidden_packed[source + 1u];
    }
    stage_weight_tile(stage.a, bindings.down, channel_base, args.hidden_size,
                      intermediate_packed_cols, k / 2u, lane);
    __syncwarp();

    uint32_t a_fragment[4];
    uint32_t b_fragment[2];
    load_a_fragment(stage.a, lane, a_fragment);
    load_b_fragment(stage.b, lane, b_fragment);
    const uint32_t group = lane >> 2;
    const uint32_t scale_row = group + ((lane & 1u) != 0u ? 8u : 0u);
    const uint32_t logical_channel = channel_base + scale_row;
    const uint32_t scale_block = k / 32u;
    uint16_t down_scale = static_cast<uint16_t>(127u | (127u << 8));
    if (logical_channel < args.hidden_size) {
      const uint64_t scale_offset =
          static_cast<uint64_t>(logical_channel) * intermediate_scale_cols +
          scale_block;
      down_scale = static_cast<uint16_t>(
          static_cast<uint16_t>(bindings.down_scales[scale_offset]) |
          (static_cast<uint16_t>(bindings.down_scales[scale_offset + 1u])
           << 8));
    }
    const uint64_t activation_scale_offset =
        (static_cast<uint64_t>(segment) * kSegmentColumns + group) *
            intermediate_scale_cols +
        scale_block;
    const uint16_t activation_scale = static_cast<uint16_t>(
        static_cast<uint16_t>(hidden_scales[activation_scale_offset]) |
        (static_cast<uint16_t>(hidden_scales[activation_scale_offset + 1u])
         << 8));
    mma_mxfp4(accumulator, a_fragment, b_fragment, down_scale,
               activation_scale);
    __syncwarp();
  }

  auto *segment_routes =
      device_pointer<const int32_t>(args.segment_route_indices);
  auto *route_output = device_pointer<float>(args.route_output);
  const uint64_t segment_metadata =
      static_cast<uint64_t>(segment) * kSegmentColumns;
  const uint32_t channel_group = lane >> 2;
  const uint32_t row_pair = lane & 3u;
#pragma unroll
  for (uint32_t element = 0u; element < 4u; ++element) {
    const uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const uint32_t row = row_pair * 2u + (element & 1u);
    const int32_t route = segment_routes[segment_metadata + row];
    if (channel < args.hidden_size && route >= 0) {
      route_output[static_cast<uint64_t>(route) * args.hidden_size + channel] =
          accumulator[element];
    }
  }

  if (channel_block == 0u && lane < kSegmentColumns) {
    const int32_t route = segment_routes[segment_metadata + lane];
    if (route >= 0) {
      auto *route_written = device_pointer<int32_t>(args.route_written);
      atomicOr(reinterpret_cast<unsigned int *>(route_written + route), 1u);
    }
  }
}

__global__ __launch_bounds__(kThreads, 1) void routed_expert_bundle(Args args) {
  extern __shared__ __align__(16) uint8_t storage[];
  auto *stages = reinterpret_cast<WarpStage *>(storage);
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t global_warp = blockIdx.x * kWarps + warp;
  const uint32_t warp_stride = gridDim.x * kWarps;

  for (uint32_t segment = blockIdx.x * blockDim.x + threadIdx.x;
       segment < args.num_segments; segment += gridDim.x * blockDim.x) {
    prepare_segment(args, segment);
  }
  cooperative_groups::this_grid().sync();

  const uint32_t intermediate_channel_blocks =
      (args.intermediate_size + kMmaChannels - 1u) / kMmaChannels;
  const uint32_t gate_tasks = args.num_segments * intermediate_channel_blocks;
  for (uint32_t task = global_warp; task < gate_tasks;
       task += warp_stride) {
    const uint32_t segment = task / intermediate_channel_blocks;
    const uint32_t channel_block =
        task - segment * intermediate_channel_blocks;
    Bindings bindings{};
    if (load_segment_bindings(args, segment, bindings)) {
      gate_up_task(args, stages[warp], segment, channel_block, lane, bindings);
    }
  }

  cooperative_groups::this_grid().sync();

  const uint32_t intermediate_scale_blocks = args.intermediate_size / 32u;
  const uint32_t pack_tasks =
      args.num_segments * kSegmentColumns * intermediate_scale_blocks;
  for (uint32_t task = global_warp; task < pack_tasks;
       task += warp_stride) {
    const uint32_t segment_stride =
        kSegmentColumns * intermediate_scale_blocks;
    const uint32_t segment = task / segment_stride;
    const uint32_t within_segment = task - segment * segment_stride;
    const uint32_t row = within_segment / intermediate_scale_blocks;
    const uint32_t scale_block =
        within_segment - row * intermediate_scale_blocks;
    auto *states = device_pointer<const int32_t>(args.segment_states);
    if (states[segment] == 1) {
      pack_task(args, segment, row, scale_block, lane);
    }
  }

  cooperative_groups::this_grid().sync();

  const uint32_t hidden_channel_blocks =
      (args.hidden_size + kMmaChannels - 1u) / kMmaChannels;
  const uint32_t down_tasks = args.num_segments * hidden_channel_blocks;
  for (uint32_t task = global_warp; task < down_tasks;
       task += warp_stride) {
    const uint32_t segment = task / hidden_channel_blocks;
    const uint32_t channel_block = task - segment * hidden_channel_blocks;
    Bindings bindings{};
    if (load_segment_bindings(args, segment, bindings)) {
      down_task(args, stages[warp], segment, channel_block, lane, bindings);
    }
  }
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr || args->args_version != kArgsVersion) {
    return Status::kInvalidAbi;
  }
  if (args->rows != kSegmentColumns || args->input_size == 0u ||
      args->intermediate_size == 0u || args->hidden_size == 0u ||
      args->num_tokens == 0u || args->num_routes == 0u ||
      args->slot_capacity == 0u || args->num_segments == 0u ||
      args->num_segments > 0x7fffffffu ||
      (args->input_size % 64u) != 0u ||
      (args->intermediate_size % 64u) != 0u ||
      !std::isfinite(args->swiglu_limit)) {
    return Status::kInvalidArgument;
  }

  const bool pointers_valid =
      detail::aligned(args->x_packed, 16u) &&
      detail::aligned(args->x_scales, 16u) &&
      detail::aligned(args->gate_ptrs, alignof(uint64_t)) &&
      detail::aligned(args->gate_scale_ptrs, alignof(uint64_t)) &&
      detail::aligned(args->up_ptrs, alignof(uint64_t)) &&
      detail::aligned(args->up_scale_ptrs, alignof(uint64_t)) &&
      detail::aligned(args->down_ptrs, alignof(uint64_t)) &&
      detail::aligned(args->down_scale_ptrs, alignof(uint64_t)) &&
      detail::aligned(args->slot_generations, alignof(int32_t)) &&
      detail::aligned(args->segment_expert_slots, alignof(int32_t)) &&
      detail::aligned(args->segment_generations, alignof(int32_t)) &&
      detail::aligned(args->segment_token_indices, alignof(int32_t)) &&
      detail::aligned(args->segment_route_indices, alignof(int32_t)) &&
      detail::aligned(args->segment_route_weights, alignof(float)) &&
      detail::aligned(args->segment_states, alignof(int32_t)) &&
      detail::aligned(args->segment_bindings, alignof(uint64_t)) &&
      detail::aligned(args->hidden_f32, 16u) &&
      detail::aligned(args->hidden_packed, 16u) &&
      detail::aligned(args->hidden_scales, 16u) &&
      detail::aligned(args->route_written, alignof(int32_t)) &&
      detail::aligned(args->route_error, alignof(int32_t)) &&
      detail::aligned(args->route_output, alignof(float));
  return pointers_valid ? Status::kSuccess : Status::kInvalidArgument;
}

inline size_t required_shared_storage_bytes(const Args &) {
  return sizeof(detail::WarpStage) * kWarps;
}

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }
  const uint32_t gate_tasks =
      args->num_segments *
      ((args->intermediate_size + kMmaChannels - 1u) / kMmaChannels);
  const uint32_t pack_tasks =
      args->num_segments * kSegmentColumns * (args->intermediate_size / 32u);
  const uint32_t down_tasks =
      args->num_segments *
      ((args->hidden_size + kMmaChannels - 1u) / kMmaChannels);
  const uint32_t warp_tasks = max(gate_tasks, max(pack_tasks, down_tasks));
  const uint32_t blocks = min(
      kGb10CooperativeBlocks,
      (warp_tasks + kWarps - 1u) / kWarps);
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  void *kernel_args[] = {const_cast<Args *>(args)};
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::routed_expert_bundle),
      dim3(blocks, 1, 1), dim3(kThreads, 1, 1), kernel_args,
      required_shared_storage_bytes(*args), stream);
  return status == cudaSuccess ? Status::kSuccess : Status::kLaunchFailed;
}

} // namespace ferrule::cutlass::sm121_fp4_moe

#endif // FERRULE_CUTLASS_SM121_FP4_MOE_CUH_
