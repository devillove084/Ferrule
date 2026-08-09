#pragma once

#include "core/abi.h"
#include "core/device.cuh"
#include "core/validation.cuh"
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <stdint.h>

namespace ferrule::cuda::core {

__global__ void linear_kernel(FerruleCoreLinearArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(args.batch) * args.n;
  if (index >= total) {
    return;
  }
  const uint32_t batch = static_cast<uint32_t>(index / args.n);
  const uint32_t row = static_cast<uint32_t>(index % args.n);
  const float *x_f32 = const_pointer<float>(args.x);
  const uint8_t *weight = const_pointer<uint8_t>(args.weight);
  const uint8_t *scales = const_pointer<uint8_t>(args.weight_scales);
  float result = 0.0f;
  if (args.kind == FERRULE_CORE_LINEAR_F32) {
    const float *w = const_pointer<float>(args.weight);
    for (uint32_t column = 0; column < args.k; ++column) {
      result += x_f32[batch * args.k + column] * w[row * args.k + column];
    }
  } else if (args.kind == FERRULE_CORE_LINEAR_F32_BYTES) {
    const float *w = const_pointer<float>(args.weight);
    for (uint32_t column = 0; column < args.k; ++column) {
      result += x_f32[batch * args.k + column] * w[row * args.k + column];
    }
  } else if (args.kind == FERRULE_CORE_LINEAR_BF16_BYTES ||
             args.kind == FERRULE_CORE_LINEAR_BF16_BYTES_ROUNDED_INPUT) {
    const uint16_t *w = const_pointer<uint16_t>(args.weight);
    for (uint32_t column = 0; column < args.k; ++column) {
      const float input = x_f32[batch * args.k + column];
      result += (args.kind == FERRULE_CORE_LINEAR_BF16_BYTES_ROUNDED_INPUT
                     ? bf16_round(input)
                     : input) *
                bf16_value(w[row * args.k + column]);
    }
  } else if (args.kind == FERRULE_CORE_LINEAR_FP8_E4M3_E8M0 ||
             args.kind == FERRULE_CORE_LINEAR_FP8_E4M3_E8M0_FROM_F32) {
    const uint32_t scale_row = (row / args.block_m) * args.scale_cols;
    for (uint32_t block = 0; block < args.scale_cols; ++block) {
      const float weight_scale = e8m0_scale(scales[scale_row + block]);
      const uint32_t begin = block * args.block_k;
      const uint32_t end = min(args.k, begin + args.block_k);
      float activation_scale = 1.0f;
      if (args.kind == FERRULE_CORE_LINEAR_FP8_E4M3_E8M0_FROM_F32) {
        float amax = 1e-4f;
        for (uint32_t column = begin; column < end; ++column) {
          amax = fmaxf(amax, fabsf(x_f32[batch * args.k + column]));
        }
        activation_scale = e8m0_scale(e8m0_byte(amax, 448.0f));
      }
      for (uint32_t column = begin; column < end; ++column) {
        float input = x_f32[batch * args.k + column];
        if (args.kind == FERRULE_CORE_LINEAR_FP8_E4M3_E8M0_FROM_F32) {
          input = fp8_quantized(
                      clamp_value(input / activation_scale, -448.0f, 448.0f)) *
                  activation_scale;
        }
        result +=
            input * fp8_value(weight[row * args.k + column]) * weight_scale;
      }
    }
  } else if (args.kind == FERRULE_CORE_LINEAR_FP8_E4M3_E8M0_PACKED) {
    const uint8_t *x = const_pointer<uint8_t>(args.x);
    const uint8_t *x_scales = const_pointer<uint8_t>(args.x_scales);
    const uint32_t scale_row = (row / 128) * args.scale_cols;
    for (uint32_t block = 0; block < args.scale_cols; ++block) {
      const float scale =
          e8m0_scale(x_scales[batch * args.scale_cols + block]) *
          e8m0_scale(scales[scale_row + block]);
      const uint32_t begin = block * 128;
      const uint32_t end = min(args.k, begin + 128);
      for (uint32_t column = begin; column < end; ++column) {
        result += fp8_value(x[batch * args.k + column]) *
                  fp8_value(weight[row * args.k + column]) * scale;
      }
    }
  }
  pointer<float>(args.output)[index] = result;
}

__global__ void dual_linear_kernel(FerruleCoreDualLinearArgs args) {
  const uint32_t combined = blockIdx.x * blockDim.x + threadIdx.x;
  if (combined >= args.first_n + args.second_n) {
    return;
  }
  const bool first = combined < args.first_n;
  const uint32_t row = first ? combined : combined - args.first_n;
  const uint8_t *weight =
      const_pointer<uint8_t>(first ? args.first_weight : args.second_weight);
  const uint16_t *w = reinterpret_cast<const uint16_t *>(weight);
  const float *x = const_pointer<float>(args.x);
  float result = 0.0f;
  for (uint32_t column = 0; column < args.k; ++column) {
    result += x[column] * bf16_value(w[row * args.k + column]);
  }
  pointer<float>(first ? args.first_output : args.second_output)[row] = result;
}

__global__ void grouped_linear_kernel(FerruleCoreGroupedLinearArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(args.rows) * args.output_dim;
  if (index >= total || args.rank == 0) {
    return;
  }
  const uint32_t token = static_cast<uint32_t>(index / args.output_dim);
  const uint32_t output_row = static_cast<uint32_t>(index % args.output_dim);
  const uint32_t groups = args.output_dim / args.rank;
  const uint32_t group = output_row / args.rank;
  const uint32_t input_base =
      token * groups * args.group_input + group * args.group_input;
  float result = 0.0f;
  if (args.kind == FERRULE_CORE_GROUPED_LINEAR_F32) {
    const float *input = const_pointer<float>(args.input);
    const float *weight = const_pointer<float>(args.weight);
    for (uint32_t column = 0; column < args.group_input; ++column) {
      result += input[input_base + column] *
                weight[output_row * args.group_input + column];
    }
  } else {
    const float *input = const_pointer<float>(args.input);
    const uint8_t *weight = const_pointer<uint8_t>(args.weight);
    const uint8_t *scales = const_pointer<uint8_t>(args.weight_scales);
    const uint32_t scale_base = (output_row / 128) * args.scale_cols;
    for (uint32_t column = 0; column < args.group_input; ++column) {
      const float a = bf16_round(input[input_base + column]);
      const float w =
          bf16_round(fp8_value(weight[output_row * args.group_input + column]) *
                     e8m0_scale(scales[scale_base + column / 128]));
      result += a * w;
    }
    result = bf16_round(result);
  }
  pointer<float>(args.output)[index] = result;
}

} // namespace ferrule::cuda::core
