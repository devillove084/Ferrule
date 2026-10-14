#pragma once
#include "core/bf16_rms_reduction.cuh"
#include "core/recurrent_bf16_abi.h"
#include "core/device.cuh"
#include "core/validation.cuh"

namespace ferrule::cuda::core::hf_bf16 {
__device__ inline float load(uint64_t address, uint64_t index) {
  return bf16_value(const_pointer<uint16_t>(address)[index]);
}

// Each channel owns its BF16 history throughout the chronological scan.
__global__ void conv(FerruleCoreBf16ConvArgs a) {
  const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= a.channels) return;
  auto *history = pointer<uint16_t>(a.history_bf16) + uint64_t(c) * a.kernel_size;
  for (uint32_t t = 0; t < a.rows; ++t) {
    const uint64_t i = uint64_t(t) * a.channels + c;
    for (uint32_t k = 0; k + 1 < a.kernel_size; ++k) history[k] = history[k + 1];
    history[a.kernel_size - 1] = const_pointer<uint16_t>(a.input_bf16)[i];
    float sum = 0.0f;
    for (uint32_t k = 0; k < a.kernel_size; ++k)
      sum = __fadd_rn(sum, __fmul_rn(bf16_value(history[k]),
                                   load(a.weight_bf16, uint64_t(c) * a.kernel_size + k)));
    if (a.bias_bf16) sum = __fadd_rn(sum, load(a.bias_bf16, c));
    // HF Conv1d materializes BF16 before the separate SiLU op.
    const float x = bf16_round(sum);
    pointer<uint16_t>(a.output_bf16)[i] = bf16_rne(x / (1.0f + expf(-x)));
  }
}

__device__ inline float inverse_l2(uint64_t qkv, uint64_t base, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t d = 0; d < dim; ++d) {
    const float x = load(qkv, base + d);
    sum = __fadd_rn(sum, bf16_round(__fmul_rn(x, x)));
  }
  // l2norm runs BEFORE HF's .to(float32), including the reduction output.
  return bf16_round(rsqrtf(bf16_round(__fadd_rn(bf16_round(sum), 1.0e-6f))));
}

// HF eager materializes each F32 product before reducing [Hv,Dk,Dv] over
// Dk. For vectorized columns (Dv % 4 == 0, Hv*Dv >= 128), Torch Reduce.cuh
// uses four y lanes with four accumulators each when 64 <= Dk <= 512.
// Emulate that order within the column owner: no inter-thread state sharing,
// temporary product tensor, FMA contraction, or extra launch is needed.
__device__ inline float column_sum4(const float *state, uint64_t qkv,
                                   uint64_t base, float inv, float scale,
                                   uint32_t dim, uint32_t stride) {
  float partial[4];
#pragma unroll
  for (uint32_t lane = 0; lane < 4; ++lane) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint32_t d = lane; d < dim; d += 16) {
      const float x0 = __fmul_rn(bf16_round(__fmul_rn(load(qkv, base + d), inv)), scale);
      acc0 = __fadd_rn(acc0, __fmul_rn(state[uint64_t(d) * stride], x0));
      if (d + 4 < dim) {
        const float x1 = __fmul_rn(bf16_round(__fmul_rn(load(qkv, base + d + 4), inv)), scale);
        acc1 = __fadd_rn(acc1, __fmul_rn(state[uint64_t(d + 4) * stride], x1));
      }
      if (d + 8 < dim) {
        const float x2 = __fmul_rn(bf16_round(__fmul_rn(load(qkv, base + d + 8), inv)), scale);
        acc2 = __fadd_rn(acc2, __fmul_rn(state[uint64_t(d + 8) * stride], x2));
      }
      if (d + 12 < dim) {
        const float x3 = __fmul_rn(bf16_round(__fmul_rn(load(qkv, base + d + 12), inv)), scale);
        acc3 = __fadd_rn(acc3, __fmul_rn(state[uint64_t(d + 12) * stride], x3));
      }
    }
    partial[lane] = __fadd_rn(__fadd_rn(__fadd_rn(acc0, acc1), acc2), acc3);
  }
  return __fadd_rn(__fadd_rn(partial[0], partial[2]),
                   __fadd_rn(partial[1], partial[3]));
}

// Fuse each independent column update with its following eager reduction.
// A dimension belongs to exactly one (lane, accumulator, iteration), so its
// F32 store can move before that product without changing any dependency.
// Keep the four accumulator chains and both final merge trees unchanged.
template <bool Update>
__device__ inline float column_update_sum4(float *state, uint64_t qkv,
                                          uint64_t k, uint64_t q,
                                          float kn, float qn, float scale,
                                          float factor, uint32_t dim,
                                          uint32_t stride) {
  const auto product = [&](uint32_t d) {
    const uint64_t off = uint64_t(d) * stride;
    const float kd = bf16_round(__fmul_rn(load(qkv, k + d), kn));
    float x;
    if constexpr (Update) {
      state[off] = __fadd_rn(state[off], __fmul_rn(kd, factor));
      x = __fmul_rn(bf16_round(__fmul_rn(load(qkv, q + d), qn)), scale);
    } else {
      state[off] = __fmul_rn(state[off], factor);
      x = __fmul_rn(kd, 1.0f);
    }
    return __fmul_rn(state[off], x);
  };
  float partial[4];
#pragma unroll
  for (uint32_t lane = 0; lane < 4; ++lane) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint32_t d = lane; d < dim; d += 16) {
      acc0 = __fadd_rn(acc0, product(d));
      if (d + 4 < dim) acc1 = __fadd_rn(acc1, product(d + 4));
      if (d + 8 < dim) acc2 = __fadd_rn(acc2, product(d + 8));
      if (d + 12 < dim) acc3 = __fadd_rn(acc3, product(d + 12));
    }
    partial[lane] = __fadd_rn(__fadd_rn(__fadd_rn(acc0, acc1), acc2), acc3);
  }
  return __fadd_rn(__fadd_rn(partial[0], partial[2]),
                   __fadd_rn(partial[1], partial[3]));
}

// Separate entry points prevent the eager accumulator live ranges from
// increasing register pressure in chronological multirow/excluded scans.
__global__ void delta_scan(FerruleCoreBf16DeltaArgs a) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t vw = uint64_t(a.value_heads) * a.value_dim;
  if (i >= vw) return;
  const uint32_t h = i / a.value_dim, column = i % a.value_dim;
  const uint32_t kh = h / (a.value_heads / a.key_heads);
  const uint64_t kw = uint64_t(a.key_heads) * a.key_dim;
  float *s = pointer<float>(a.state_f32) + uint64_t(h) * a.key_dim * a.value_dim + column;
  const float scale = 1.0f / sqrtf(float(a.key_dim));
  const float A = expf(load(a.a_log_bf16, h));
  for (uint32_t t = 0; t < a.rows; ++t) {
    const uint64_t q = uint64_t(t) * (2 * kw + vw) + uint64_t(kh) * a.key_dim;
    const uint64_t k = q + kw;
    const float qn = inverse_l2(a.qkv_bf16, q, a.key_dim);
    const float kn = inverse_l2(a.qkv_bf16, k, a.key_dim);
    const uint64_t g = uint64_t(t) * a.value_heads + h;
    const float x = __fadd_rn(load(a.a_bf16, g), load(a.dt_bias_bf16, h));
    const float softplus = x > 20.0f ? x : log1pf(expf(x));
    const float decay = expf(__fmul_rn(-A, softplus));
    const float beta = bf16_round(1.0f / (1.0f + expf(-load(a.b_bf16, g))));
    float memory = 0.0f;
    for (uint32_t d = 0; d < a.key_dim; ++d) {
      const uint64_t off = uint64_t(d) * a.value_dim;
      s[off] = __fmul_rn(s[off], decay);
      const float kd = bf16_round(__fmul_rn(load(a.qkv_bf16, k + d), kn));
      memory = __fadd_rn(memory, __fmul_rn(kd, s[off]));
    }
    const float v = load(a.qkv_bf16, uint64_t(t) * (2 * kw + vw) + 2 * kw + i);
    const float correction = __fmul_rn(__fsub_rn(v, memory), beta);
    float result = 0.0f;
    for (uint32_t d = 0; d < a.key_dim; ++d) {
      const uint64_t off = uint64_t(d) * a.value_dim;
      const float kd = bf16_round(__fmul_rn(load(a.qkv_bf16, k + d), kn));
      const float qd = __fmul_rn(bf16_round(__fmul_rn(load(a.qkv_bf16, q + d), qn)), scale);
      s[off] = __fadd_rn(s[off], __fmul_rn(kd, correction));
      result = __fadd_rn(result, __fmul_rn(s[off], qd));
    }
    pointer<uint16_t>(a.output_bf16)[uint64_t(t) * vw + i] = bf16_rne(result);
  }
}

__global__ void delta(FerruleCoreBf16DeltaArgs a) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t vw = uint64_t(a.value_heads) * a.value_dim;
  if (i >= vw) return;
  const uint32_t h = i / a.value_dim, column = i % a.value_dim;
  const uint32_t kh = h / (a.value_heads / a.key_heads);
  const uint64_t kw = uint64_t(a.key_heads) * a.key_dim;
  float *s = pointer<float>(a.state_f32) + uint64_t(h) * a.key_dim * a.value_dim + column;
  const float scale = 1.0f / sqrtf(float(a.key_dim));
  const float A = expf(load(a.a_log_bf16, h));
  const uint64_t q = uint64_t(kh) * a.key_dim;
  const uint64_t k = q + kw;
  const float qn = inverse_l2(a.qkv_bf16, q, a.key_dim);
  const float kn = inverse_l2(a.qkv_bf16, k, a.key_dim);
  const uint64_t g = h;
  const float x = __fadd_rn(load(a.a_bf16, g), load(a.dt_bias_bf16, h));
  const float softplus = x > 20.0f ? x : log1pf(expf(x));
  const float decay = expf(__fmul_rn(-A, softplus));
  const float beta = bf16_round(1.0f / (1.0f + expf(-load(a.b_bf16, g))));
  const float memory = column_update_sum4<false>(
      s, a.qkv_bf16, k, q, kn, qn, scale, decay, a.key_dim, a.value_dim);
  const float v = load(a.qkv_bf16, 2 * kw + i);
  const float correction = __fmul_rn(__fsub_rn(v, memory), beta);
  const float result = column_update_sum4<true>(
      s, a.qkv_bf16, k, q, kn, qn, scale, correction, a.key_dim, a.value_dim);
  pointer<uint16_t>(a.output_bf16)[i] = bf16_rne(result);
}

// Shared pinned F32 mean tree; gated and ordinary norms keep distinct epilogues.
inline dim3 gated_norm_block(uint32_t rows, uint32_t width) {
  return bf16_rms::block(rows, width);
}
__device__ inline float gated_norm_lane_sum(FerruleCoreBf16GatedNormArgs a,
                                           uint32_t row, uint32_t lane,
                                           uint32_t stride) {
  return bf16_rms::lane_sum(a.input_bf16, a.width, row, lane, stride);
}

__global__ void gated_norm_tree(FerruleCoreBf16GatedNormArgs a, float mean_factor) {
  __shared__ float sums[512];
  const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
  const uint32_t lane = threadIdx.x, offset = threadIdx.y * blockDim.x;
  sums[offset + lane] = row < a.rows ? gated_norm_lane_sum(a, row, lane, blockDim.x) : 0.0f;
  __syncthreads();
  for (uint32_t step = blockDim.x / 2; step; step /= 2) {
    if (lane < step)
      sums[offset + lane] = __fadd_rn(sums[offset + lane], sums[offset + lane + step]);
    __syncthreads();
  }
  if (row >= a.rows) return;
  const float variance = __fmul_rn(sums[offset], mean_factor);
  const float inv = rsqrtf(__fadd_rn(variance, a.epsilon));
  const uint64_t base = uint64_t(row) * a.width;
  for (uint32_t d = lane; d < a.width; d += blockDim.x) {
    const float normalized = bf16_round(__fmul_rn(load(a.input_bf16, base + d), inv));
    const float weighted = bf16_round(__fmul_rn(normalized, load(a.weight_bf16, d)));
    const float z = load(a.gate_bf16, base + d);
    pointer<uint16_t>(a.output_bf16)[base + d] = bf16_rne(__fmul_rn(weighted, z / (1.0f + expf(-z))));
  }
}

__global__ void gated_norm(FerruleCoreBf16GatedNormArgs a) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows) return;
  const uint64_t base = uint64_t(row) * a.width;
  float sum = 0.0f;
  for (uint32_t d = 0; d < a.width; ++d) {
    const float x = load(a.input_bf16, base + d);
    sum = __fadd_rn(sum, __fmul_rn(x, x));
  }
  const float inv = rsqrtf(__fadd_rn(sum / a.width, a.epsilon));
  for (uint32_t d = 0; d < a.width; ++d) {
    const float normalized = bf16_round(__fmul_rn(load(a.input_bf16, base + d), inv));
    const float weighted = bf16_round(__fmul_rn(normalized, load(a.weight_bf16, d)));
    const float z = load(a.gate_bf16, base + d);
    // Gate SiLU is F32 (unlike ordinary BF16 SwiGLU), then final BF16 output.
    pointer<uint16_t>(a.output_bf16)[base + d] = bf16_rne(__fmul_rn(weighted, z / (1.0f + expf(-z))));
  }
}

inline bool extent(uint64_t x, uint64_t y) {
  return x && y && x <= INT32_MAX && y <= uint64_t(INT32_MAX) / x;
}
inline bool bf16_ptr(uint64_t p) { return p && p % 2 == 0; }
} // namespace ferrule::cuda::core::hf_bf16

extern "C" int32_t ferrule_core_bf16_conv_launch(const FerruleCoreBf16ConvArgs *a) {
  using namespace ferrule::cuda::core;
  using namespace ferrule::cuda::core::hf_bf16;
  if (!a || a->reserved || !extent(a->rows, a->channels) || !extent(a->channels, a->kernel_size) ||
      !bf16_ptr(a->input_bf16) || !bf16_ptr(a->weight_bf16) ||
      (a->bias_bf16 && !bf16_ptr(a->bias_bf16)) || !bf16_ptr(a->history_bf16) || !bf16_ptr(a->output_bf16))
    return int32_t(cudaErrorInvalidValue);
  conv<<<blocks_for(a->channels), kBlock, 0, stream(a->stream)>>>(*a);
  return launch_status();
}
extern "C" int32_t ferrule_core_bf16_delta_launch(const FerruleCoreBf16DeltaArgs *a) {
  using namespace ferrule::cuda::core;
  using namespace ferrule::cuda::core::hf_bf16;
  if (!a || a->reserved || !extent(a->key_heads, a->key_dim) || !extent(a->value_heads, a->value_dim) ||
      a->value_heads % a->key_heads || !extent(a->rows, a->value_heads) ||
      !extent(uint64_t(a->value_heads) * a->value_dim, a->key_dim) ||
      !extent(a->rows, 2 * uint64_t(a->key_heads) * a->key_dim + uint64_t(a->value_heads) * a->value_dim) ||
      !bf16_ptr(a->qkv_bf16) || !bf16_ptr(a->a_bf16) || !bf16_ptr(a->b_bf16) ||
      !bf16_ptr(a->a_log_bf16) || !bf16_ptr(a->dt_bias_bf16) || !bf16_ptr(a->output_bf16) ||
      !a->state_f32 || a->state_f32 % 4)
    return int32_t(cudaErrorInvalidValue);
  const uint64_t vw = uint64_t(a->value_heads) * a->value_dim;
  if (a->rows == 1 && a->key_dim >= 64 && a->key_dim <= 512 &&
      a->value_dim % 4 == 0 && vw >= 128)
    delta<<<blocks_for(vw), kBlock, 0, stream(a->stream)>>>(*a);
  else
    delta_scan<<<blocks_for(vw), kBlock, 0, stream(a->stream)>>>(*a);
  return launch_status();
}
extern "C" int32_t ferrule_core_bf16_gated_norm_launch(const FerruleCoreBf16GatedNormArgs *a) {
  using namespace ferrule::cuda::core;
  using namespace ferrule::cuda::core::hf_bf16;
  if (!a || a->reserved || !extent(a->rows, a->width) || !isfinite(a->epsilon) || a->epsilon <= 0.0f ||
      !bf16_ptr(a->input_bf16) || !bf16_ptr(a->weight_bf16) || !bf16_ptr(a->gate_bf16) || !bf16_ptr(a->output_bf16))
    return int32_t(cudaErrorInvalidValue);
  if (a->width <= 4096) {
    const dim3 block = gated_norm_block(a->rows, a->width);
    const float factor = static_cast<float>(1.0 / static_cast<double>(a->width));
    gated_norm_tree<<<(a->rows + block.y - 1) / block.y, block, 0, stream(a->stream)>>>(*a, factor);
  } else {
    gated_norm<<<blocks_for(a->rows), kBlock, 0, stream(a->stream)>>>(*a);
  }
  return launch_status();
}
