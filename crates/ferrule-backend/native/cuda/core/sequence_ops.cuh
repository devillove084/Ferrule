#pragma once

#include "core/abi.h"
#include "core/device.cuh"
#include "core/validation.cuh"
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <stdint.h>

namespace ferrule::cuda::core {

// One thread owns a channel for the entire sequence: no inter-row history races.
__global__ void causal_depthwise_conv_kernel(FerruleCoreConvArgs args) {
  const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= args.channels) return;
  float *history = pointer<float>(args.history) +
                   static_cast<uint64_t>(channel) * args.kernel_size;
  const float *weight = const_pointer<float>(args.weight) +
                        static_cast<uint64_t>(channel) * args.kernel_size;
  for (uint32_t row = 0; row < args.rows; ++row) {
    const uint64_t index = static_cast<uint64_t>(row) * args.channels + channel;
    for (uint32_t tap = 0; tap + 1 < args.kernel_size; ++tap)
      history[tap] = history[tap + 1];
    history[args.kernel_size - 1] = const_pointer<float>(args.input)[index];
    float value = args.bias == 0 ? 0.0f : const_pointer<float>(args.bias)[channel];
    for (uint32_t tap = 0; tap < args.kernel_size; ++tap)
      value += history[tap] * weight[tap];
    pointer<float>(args.output)[index] = value / (1.0f + expf(-value));
  }
}

// Each thread owns a complete state column [dk] for one value head.
// Serial time traversal supports arbitrary prefill length without host state copies.
__global__ void gated_delta_kernel(FerruleCoreDeltaArgs args) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t value_width = static_cast<uint64_t>(args.value_heads) * args.value_dim;
  if (index >= value_width) return;
  const uint32_t head = index / args.value_dim;
  const uint32_t column = index % args.value_dim;
  const uint32_t key_head = head / (args.value_heads / args.key_heads);
  const uint64_t key_width = static_cast<uint64_t>(args.key_heads) * args.key_dim;
  const uint64_t row_width = 2 * key_width + value_width;
  float *state = pointer<float>(args.state) +
                static_cast<uint64_t>(head) * args.key_dim * args.value_dim + column;
  const float qscale = rsqrtf(static_cast<float>(args.key_dim));
  const float decay_scale = expf(const_pointer<float>(args.a_log)[head]);
  for (uint32_t row = 0; row < args.rows; ++row) {
    const float *q = const_pointer<float>(args.qkv) + row * row_width +
                     static_cast<uint64_t>(key_head) * args.key_dim;
    const float *k = q + key_width;
    float qnorm = 0.0f, knorm = 0.0f;
    for (uint32_t d = 0; d < args.key_dim; ++d) {
      qnorm += q[d] * q[d];
      knorm += k[d] * k[d];
    }
    qnorm = rsqrtf(qnorm + 1.0e-6f);
    knorm = rsqrtf(knorm + 1.0e-6f);
    const uint64_t gate_index = static_cast<uint64_t>(row) * args.value_heads + head;
    const float x = const_pointer<float>(args.a)[gate_index] +
                    const_pointer<float>(args.dt_bias)[head];
    // torch softplus uses beta=1 and threshold=20; avoid overflow on either tail.
    const float softplus = x > 20.0f ? x : log1pf(expf(x));
    const float decay = expf(-decay_scale * softplus);
    const float beta = 1.0f / (1.0f + expf(-const_pointer<float>(args.b)[gate_index]));
    float memory = 0.0f;
    for (uint32_t d = 0; d < args.key_dim; ++d) {
      const uint64_t offset = static_cast<uint64_t>(d) * args.value_dim;
      state[offset] *= decay;
      memory += (k[d] * knorm) * state[offset];
    }
    const float value = const_pointer<float>(args.qkv)[row * row_width + 2 * key_width + index];
    const float delta = beta * (value - memory);
    float result = 0.0f;
    for (uint32_t d = 0; d < args.key_dim; ++d) {
      const uint64_t offset = static_cast<uint64_t>(d) * args.value_dim;
      state[offset] += (k[d] * knorm) * delta;
      result += ((q[d] * qnorm) * qscale) * state[offset];
    }
    pointer<float>(args.output)[static_cast<uint64_t>(row) * value_width + index] = result;
  }
}

__global__ void rope_kernel(FerruleCoreRopeArgs args) {
  const uint32_t pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_index >= args.pair_count || args.rope_dim == 0 ||
      args.rope_dim > args.head_dim) {
    return;
  }
  float *values = pointer<float>(args.values);
  const float *cosine = const_pointer<float>(args.cosine);
  const float *sine = const_pointer<float>(args.sine);
  if (args.kind == FERRULE_CORE_ROPE_SPLIT_HALF_INDEXED) {
    const uint32_t pairs_per_head = args.rope_dim / 2;
    const uint32_t pairs_per_row = args.heads * pairs_per_head;
    const uint32_t row = pair_index / pairs_per_row;
    const uint32_t within = pair_index % pairs_per_row;
    const uint32_t head = within / pairs_per_head;
    const uint32_t pair = within % pairs_per_head;
    const int32_t position = const_pointer<int32_t>(args.positions)[row];
    if (position < 0 || static_cast<uint32_t>(position) >= args.start_position) {
      return;
    }
    const uint64_t table =
        static_cast<uint64_t>(position) * pairs_per_head + pair;
    const float c = cosine[table];
    const float s = args.inverse != 0 ? -sine[table] : sine[table];
    const uint64_t base =
        (static_cast<uint64_t>(row) * args.heads + head) * args.head_dim;
    const uint64_t first_index = base + pair;
    const uint64_t second_index = base + pairs_per_head + pair;
    const float first = values[first_index];
    const float second = values[second_index];
    const float rotated_first = first * c - second * s;
    const float rotated_second = first * s + second * c;
    values[first_index] = args.restore_bf16_boundary != 0
                              ? bf16_round(rotated_first)
                              : rotated_first;
    values[second_index] = args.restore_bf16_boundary != 0
                               ? bf16_round(rotated_second)
                               : rotated_second;
    return;
  }
  if (args.kind == FERRULE_CORE_ROPE_YARN) {
    const uint32_t pairs_per_head = args.rope_dim / 2;
    const uint32_t head = pair_index / pairs_per_head;
    const uint32_t pair = pair_index % pairs_per_head;
    const uint64_t base =
        static_cast<uint64_t>(head) * args.head_dim + pair * 2;
    const float x0 = values[base];
    const float x1 = values[base + 1];
    const float output0 = x0 * cosine[pair] - x1 * sine[pair];
    const float output1 = x0 * sine[pair] + x1 * cosine[pair];
    values[base] =
        args.restore_bf16_boundary != 0 ? bf16_round(output0) : output0;
    values[base + 1] =
        args.restore_bf16_boundary != 0 ? bf16_round(output1) : output1;
    return;
  }
  const uint32_t pairs_per_head = args.rope_dim / 2;
  const uint32_t pairs_per_row = args.heads * pairs_per_head;
  const uint32_t row = pair_index / pairs_per_row;
  const uint32_t within = pair_index % pairs_per_row;
  const uint32_t head = within / pairs_per_head;
  const uint32_t pair = within % pairs_per_head;
  int32_t position = args.kind == FERRULE_CORE_ROPE_TAIL_INDEXED
                         ? const_pointer<int32_t>(args.positions)[row]
                         : static_cast<int32_t>(args.start_position +
                                                row * args.position_stride);
  if (position < 0) {
    return;
  }
  const uint64_t table =
      static_cast<uint64_t>(position) * pairs_per_head + pair;
  const float c = cosine[table];
  const float s = args.inverse != 0 ? -sine[table] : sine[table];
  const uint64_t base =
      static_cast<uint64_t>(row) * args.heads * args.head_dim +
      static_cast<uint64_t>(head) * args.head_dim +
      (args.head_dim - args.rope_dim) + pair * 2;
  const float x0 = values[base];
  const float x1 = values[base + 1];
  const float output0 = x0 * c - x1 * s;
  const float output1 = x0 * s + x1 * c;
  values[base] =
      args.restore_bf16_boundary != 0 ? bf16_round(output0) : output0;
  values[base + 1] =
      args.restore_bf16_boundary != 0 ? bf16_round(output1) : output1;
}

__global__ void hc_post_kernel(FerruleCoreHcArgs args) {
  const uint64_t elements =
      static_cast<uint64_t>(args.tokens) * args.hc * args.hidden_size;
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements || args.hc == 0 || args.hidden_size == 0) {
    return;
  }
  const uint32_t dimension = index % args.hidden_size;
  const uint64_t token_copy = index / args.hidden_size;
  const uint32_t copy = token_copy % args.hc;
  const uint32_t token = token_copy / args.hc;
  const uint64_t state_base =
      static_cast<uint64_t>(token) * args.hc * args.hidden_size;
  float residual = 0.0f;
  for (uint32_t input_copy = 0; input_copy < args.hc; ++input_copy) {
    const float product = __fmul_rn(
        const_pointer<float>(args.split_comb)
            [(static_cast<uint64_t>(token) * args.hc + input_copy) * args.hc +
             copy],
        const_pointer<float>(args.residual)[state_base +
                                            static_cast<uint64_t>(input_copy) *
                                                args.hidden_size +
                                            dimension]);
    residual = __fadd_rn(residual, product);
  }
  const float update = __fmul_rn(
      const_pointer<float>(
          args.split_post)[static_cast<uint64_t>(token) * args.hc + copy],
      const_pointer<float>(
          args.hidden)[static_cast<uint64_t>(token) * args.hidden_size +
                       dimension]);
  pointer<float>(args.output)[index] = bf16_round(__fadd_rn(update, residual));
}

__global__ void hc_mean_scatter_kernel(FerruleCoreHcArgs args) {
  const uint64_t elements =
      static_cast<uint64_t>(args.tokens) * args.hidden_size;
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements || args.hc == 0 || args.tap_slot >= args.tap_count) {
    return;
  }
  const uint32_t dimension = index % args.hidden_size;
  const uint32_t token = index / args.hidden_size;
  const uint64_t state_base =
      static_cast<uint64_t>(token) * args.hc * args.hidden_size;
  float sum = 0.0f;
  for (uint32_t copy = 0; copy < args.hc; ++copy) {
    sum += const_pointer<float>(
        args.state)[state_base +
                    static_cast<uint64_t>(copy) * args.hidden_size + dimension];
  }
  const uint64_t output_base =
      (static_cast<uint64_t>(token) * args.tap_count + args.tap_slot) *
      args.hidden_size;
  pointer<float>(args.output)[output_base + dimension] =
      bf16_round(sum / args.hc);
}

__global__ void hc_kernel(FerruleCoreHcArgs args) {
  const uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= args.tokens || args.hc == 0) {
    return;
  }
  const uint32_t hc_dimension = args.hc * args.hidden_size;
  const uint64_t state_base = static_cast<uint64_t>(token) * hc_dimension;
  if (args.kind == FERRULE_CORE_HC_POST) {
    for (uint32_t copy = 0; copy < args.hc; ++copy) {
      for (uint32_t dimension = 0; dimension < args.hidden_size; ++dimension) {
        float residual = 0.0f;
        for (uint32_t input_copy = 0; input_copy < args.hc; ++input_copy) {
          residual +=
              const_pointer<float>(
                  args.split_comb)[(static_cast<uint64_t>(token) * args.hc +
                                    input_copy) *
                                       args.hc +
                                   copy] *
              const_pointer<float>(
                  args.residual)[state_base +
                                 static_cast<uint64_t>(input_copy) *
                                     args.hidden_size +
                                 dimension];
        }
        pointer<float>(
            args.output)[state_base +
                         static_cast<uint64_t>(copy) * args.hidden_size +
                         dimension] =
            const_pointer<float>(
                args.split_post)[static_cast<uint64_t>(token) * args.hc +
                                 copy] *
                const_pointer<float>(args.hidden)[static_cast<uint64_t>(token) *
                                                      args.hidden_size +
                                                  dimension] +
            residual;
      }
    }
    return;
  }
  if (args.kind == FERRULE_CORE_HC_MEAN_SCATTER) {
    if (args.tap_slot >= args.tap_count) {
      return;
    }
    const uint64_t output_base =
        (static_cast<uint64_t>(token) * args.tap_count + args.tap_slot) *
        args.hidden_size;
    for (uint32_t dimension = 0; dimension < args.hidden_size; ++dimension) {
      float sum = 0.0f;
      for (uint32_t copy = 0; copy < args.hc; ++copy) {
        sum += const_pointer<float>(
            args.state)[state_base +
                        static_cast<uint64_t>(copy) * args.hidden_size +
                        dimension];
      }
      pointer<float>(args.output)[output_base + dimension] =
          bf16_round(sum / args.hc);
    }
    return;
  }
  float sum_square = 0.0f;
  for (uint32_t index = 0; index < hc_dimension; ++index) {
    const float value = const_pointer<float>(args.state)[state_base + index];
    sum_square += value * value;
  }
  const float rms = rsqrtf(sum_square / hc_dimension + args.norm_epsilon);
  float pre[16];
  if (args.kind == FERRULE_CORE_HC_HEAD) {
    for (uint32_t copy = 0; copy < args.hc; ++copy) {
      float dot = 0.0f;
      for (uint32_t column = 0; column < hc_dimension; ++column) {
        dot += const_pointer<float>(
                   args.function)[static_cast<uint64_t>(copy) * hc_dimension +
                                  column] *
               const_pointer<float>(args.state)[state_base + column];
      }
      pre[copy] = sigmoid(dot * rms * const_pointer<float>(args.scale)[0] +
                          const_pointer<float>(args.base)[copy]) +
                  args.epsilon;
    }
  } else {
    float mix[128];
    for (uint32_t row = 0; row < args.mix; ++row) {
      float dot = 0.0f;
      for (uint32_t column = 0; column < hc_dimension; ++column) {
        dot +=
            const_pointer<float>(
                args.function)[static_cast<uint64_t>(column) * args.mix + row] *
            const_pointer<float>(args.state)[state_base + column];
      }
      mix[row] = dot * rms;
    }
    for (uint32_t copy = 0; copy < args.hc; ++copy) {
      pre[copy] = sigmoid(mix[copy] * const_pointer<float>(args.scale)[0] +
                          const_pointer<float>(args.base)[copy]) +
                  args.epsilon;
      pointer<float>(
          args.split_pre)[static_cast<uint64_t>(token) * args.hc + copy] =
          pre[copy];
      pointer<float>(
          args.split_post)[static_cast<uint64_t>(token) * args.hc + copy] =
          2.0f *
          sigmoid(mix[args.hc + copy] * const_pointer<float>(args.scale)[1] +
                  const_pointer<float>(args.base)[args.hc + copy]);
    }
    float combination[256];
    for (uint32_t row = 0; row < args.hc; ++row) {
      float maximum = -CUDART_INF_F;
      for (uint32_t column = 0; column < args.hc; ++column) {
        const uint32_t index = row * args.hc + column;
        combination[index] =
            mix[2 * args.hc + index] * const_pointer<float>(args.scale)[2] +
            const_pointer<float>(args.base)[2 * args.hc + index];
        maximum = fmaxf(maximum, combination[index]);
      }
      float sum = 0.0f;
      for (uint32_t column = 0; column < args.hc; ++column) {
        const uint32_t index = row * args.hc + column;
        combination[index] = expf(combination[index] - maximum);
        sum += combination[index];
      }
      for (uint32_t column = 0; column < args.hc; ++column) {
        combination[row * args.hc + column] =
            combination[row * args.hc + column] / sum + args.epsilon;
      }
    }
    for (uint32_t iteration = 0; iteration < args.sinkhorn_iters; ++iteration) {
      if (iteration != 0) {
        for (uint32_t row = 0; row < args.hc; ++row) {
          float sum = 0.0f;
          for (uint32_t column = 0; column < args.hc; ++column) {
            sum += combination[row * args.hc + column];
          }
          for (uint32_t column = 0; column < args.hc; ++column) {
            combination[row * args.hc + column] /= sum + args.epsilon;
          }
        }
      }
      for (uint32_t column = 0; column < args.hc; ++column) {
        float sum = 0.0f;
        for (uint32_t row = 0; row < args.hc; ++row) {
          sum += combination[row * args.hc + column];
        }
        for (uint32_t row = 0; row < args.hc; ++row) {
          combination[row * args.hc + column] /= sum + args.epsilon;
        }
      }
    }
    for (uint32_t index = 0; index < args.hc * args.hc; ++index) {
      pointer<float>(
          args.split_comb)[static_cast<uint64_t>(token) * args.hc * args.hc +
                           index] = combination[index];
    }
  }
  for (uint32_t dimension = 0; dimension < args.hidden_size; ++dimension) {
    float result = 0.0f;
    for (uint32_t copy = 0; copy < args.hc; ++copy) {
      result += pre[copy] *
                const_pointer<float>(
                    args.state)[state_base +
                                static_cast<uint64_t>(copy) * args.hidden_size +
                                dimension];
    }
    pointer<float>(
        args.hidden)[static_cast<uint64_t>(token) * args.hidden_size +
                     dimension] = bf16_round(result);
  }
}

} // namespace ferrule::cuda::core
