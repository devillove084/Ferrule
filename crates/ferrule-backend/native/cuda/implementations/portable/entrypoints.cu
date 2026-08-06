#include "../../abi/core_provider.h"

#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <stdint.h>

namespace {

constexpr uint32_t kBlock = 256;
constexpr uint32_t kMaxTopK = 512;

inline uint32_t blocks_for(uint64_t count) {
  const uint64_t blocks = (count + kBlock - 1) / kBlock;
  return static_cast<uint32_t>(blocks == 0 ? 1 : blocks);
}

template <typename T> __host__ __device__ inline T *pointer(uint64_t address) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(address));
}

template <typename T>
__host__ __device__ inline const T *const_pointer(uint64_t address) {
  return reinterpret_cast<const T *>(static_cast<uintptr_t>(address));
}

inline cudaStream_t stream(uint64_t address) {
  return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(address));
}

inline int32_t launch_status() {
  return static_cast<int32_t>(cudaPeekAtLastError());
}

inline bool valid(const void *args) { return args != nullptr; }

__device__ inline float fp4_value(uint8_t nibble) {
  constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = values[nibble & 7u];
  return (nibble & 8u) != 0 ? -magnitude : magnitude;
}

__device__ inline float fp8_value(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte & 0x80u) << 24;
  const uint32_t exponent = (byte >> 3) & 0x0fu;
  const uint32_t mantissa = byte & 7u;
  if (exponent == 0) {
    if (mantissa == 0) {
      return __uint_as_float(sign);
    }
    const float value = static_cast<float>(mantissa) * (1.0f / 512.0f);
    return sign != 0 ? -value : value;
  }
  if (exponent == 15 && mantissa == 7) {
    return CUDART_NAN_F;
  }
  return __uint_as_float(sign | ((exponent + 120u) << 23) | (mantissa << 20));
}

__device__ inline float e8m0_scale(uint8_t byte) {
  return __uint_as_float(byte == 0 ? 1u << 22
                                   : static_cast<uint32_t>(byte) << 23);
}

__device__ inline float bf16_value(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ inline uint16_t bf16_rne(float value) {
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  }
  const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

__device__ inline float bf16_round(float value) {
  return bf16_value(bf16_rne(value));
}

__device__ inline float clamp_value(float value, float low, float high) {
  return fminf(high, fmaxf(low, value));
}

__device__ inline float sigmoid(float value) {
  if (value < -16.0f) {
    return 0.0f;
  }
  if (value > 16.0f) {
    return 1.0f;
  }
  if (value >= 0.0f) {
    return 1.0f / (1.0f + expf(-value));
  }
  const float e = expf(value);
  return e / (1.0f + e);
}

__device__ inline float softplus(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return expf(value);
  }
  return log1pf(expf(value));
}

__device__ inline uint8_t e8m0_byte(float amax, float quant_max) {
  if (!isfinite(amax) || amax <= 0.0f || !isfinite(quant_max) ||
      quant_max <= 0.0f) {
    return 127;
  }
  int exponent = static_cast<int>(ceilf(log2f(amax / quant_max))) + 127;
  exponent = exponent < 0 ? 0 : exponent > 255 ? 255 : exponent;
  return static_cast<uint8_t>(exponent);
}

__device__ inline uint8_t fp4_nibble(float value) {
  if (!isfinite(value) || value == 0.0f) {
    return 0;
  }
  const uint8_t sign = value < 0.0f ? 8u : 0u;
  const float magnitude = fminf(fabsf(value), 6.0f);
  uint8_t best = 0;
  float error = magnitude;
  for (uint8_t index = 1; index < 8; ++index) {
    const float candidate = fp4_value(index);
    const float candidate_error = fabsf(candidate - magnitude);
    if (candidate_error < error) {
      best = index;
      error = candidate_error;
    }
  }
  return sign | best;
}

__device__ inline float nearest_fp8_positive(float magnitude) {
  const float step = exp2f(-9.0f);
  float best = rintf(magnitude / step);
  best = fminf(7.0f, fmaxf(0.0f, best)) * step;
  float best_error = fabsf(best - magnitude);
  const int exponent_floor = static_cast<int>(floorf(log2f(magnitude)));
  for (int exponent = exponent_floor - 1; exponent <= exponent_floor + 1;
       ++exponent) {
    if (exponent < -6 || exponent > 8) {
      continue;
    }
    const float scale = exp2f(static_cast<float>(exponent));
    int mantissa = static_cast<int>(rintf((magnitude / scale - 1.0f) * 8.0f));
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
    const float candidate = exp2f(static_cast<float>(candidate_exponent)) *
                            (1.0f + static_cast<float>(mantissa) / 8.0f);
    const float error = fabsf(candidate - magnitude);
    if (error < best_error) {
      best = candidate;
      best_error = error;
    }
  }
  return best;
}

__device__ inline float fp8_quantized(float value) {
  if (!isfinite(value) || value == 0.0f) {
    return value;
  }
  const float magnitude = fminf(fabsf(value), 448.0f);
  return copysignf(nearest_fp8_positive(magnitude), value);
}

__device__ inline uint8_t fp8_byte(float value) {
  const uint8_t sign = (__float_as_uint(value) & 0x80000000u) != 0 ? 0x80u : 0u;
  if (value == 0.0f) {
    return sign;
  }
  if (!isfinite(value)) {
    return sign | 0x7fu;
  }
  const float quantized = nearest_fp8_positive(fminf(fabsf(value), 448.0f));
  if (quantized < 1.0f / 64.0f) {
    const uint8_t mantissa = static_cast<uint8_t>(rintf(quantized * 512.0f));
    return sign | (mantissa > 7 ? 7 : mantissa);
  }
  const uint32_t bits = __float_as_uint(quantized);
  const uint8_t exponent =
      static_cast<uint8_t>(static_cast<int>((bits >> 23) & 0xffu) - 127 + 7);
  return sign | (exponent << 3) | static_cast<uint8_t>((bits >> 20) & 7u);
}

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

__global__ void quantize_kernel(FerruleCoreQuantizeArgs args) {
  const uint32_t block_index = blockIdx.x * blockDim.x + threadIdx.x;
  float *values = pointer<float>(args.values);
  if (args.kind == FERRULE_CORE_QUANTIZE_HADAMARD_FP4_IN_PLACE) {
    if (args.row_width == 0 || block_index >= args.value_len / args.row_width) {
      return;
    }
    const uint32_t base = block_index * args.row_width;
    for (uint32_t span = 1; span < args.row_width; span *= 2) {
      for (uint32_t start = 0; start < args.row_width; start += span * 2) {
        for (uint32_t offset = 0; offset < span; ++offset) {
          const float a = values[base + start + offset];
          const float b = values[base + start + offset + span];
          values[base + start + offset] = a + b;
          values[base + start + offset + span] = a - b;
        }
      }
    }
    const float hadamard_scale = rsqrtf(static_cast<float>(args.row_width));
    for (uint32_t index = 0; index < args.row_width; ++index) {
      values[base + index] *= hadamard_scale;
    }
    for (uint32_t block = 0; block < args.row_width / args.block_size;
         ++block) {
      const uint32_t begin = base + block * args.block_size;
      float amax = 6.0f * exp2f(-126.0f);
      for (uint32_t index = 0; index < args.block_size; ++index) {
        amax = fmaxf(amax, fabsf(values[begin + index]));
      }
      const float scale = e8m0_scale(e8m0_byte(amax, 6.0f));
      for (uint32_t index = 0; index < args.block_size; ++index) {
        values[begin + index] =
            fp4_value(fp4_nibble(values[begin + index] / scale)) * scale;
      }
    }
    return;
  }

  uint32_t effective_width = args.row_width;
  uint32_t row = 0;
  uint32_t block = 0;
  uint32_t begin = 0;
  uint32_t end = 0;
  if (args.kind == FERRULE_CORE_QUANTIZE_FP8_NON_ROPE_IN_PLACE) {
    const uint32_t non_rope = args.row_width - args.rope_dim;
    effective_width =
        non_rope % args.block_size == 0 ? args.block_size : non_rope;
    const uint32_t blocks_per_row =
        (non_rope + effective_width - 1) / effective_width;
    row = block_index / blocks_per_row;
    block = block_index % blocks_per_row;
    if (row >= args.value_len / args.row_width) {
      return;
    }
    begin = row * args.row_width + block * effective_width;
    end = min(row * args.row_width + non_rope, begin + effective_width);
  } else {
    if (args.block_size == 0 ||
        block_index >= args.value_len / args.block_size) {
      return;
    }
    const uint32_t blocks_per_row = args.row_width / args.block_size;
    row = block_index / blocks_per_row;
    block = block_index % blocks_per_row;
    begin = row * args.row_width + block * args.block_size;
    end = begin + args.block_size;
  }

  const float *input = const_pointer<float>(args.values);
  float amax = args.kind == FERRULE_CORE_QUANTIZE_FP4_PACKED ? 0.0f : 1e-4f;
  for (uint32_t index = begin; index < end; ++index) {
    amax = fmaxf(amax, fabsf(input[args.value_offset + index]));
  }
  const bool fp4 = args.kind == FERRULE_CORE_QUANTIZE_FP4_PACKED;
  const uint8_t scale_byte = e8m0_byte(amax, fp4 ? 6.0f : 448.0f);
  const float scale = e8m0_scale(scale_byte);
  if (args.kind == FERRULE_CORE_QUANTIZE_FP8_IN_PLACE ||
      args.kind == FERRULE_CORE_QUANTIZE_FP8_NON_ROPE_IN_PLACE) {
    for (uint32_t index = begin; index < end; ++index) {
      values[index] = bf16_round(
          fp8_quantized(clamp_value(values[index] / scale, -448.0f, 448.0f)) *
          scale);
    }
  } else if (args.kind == FERRULE_CORE_QUANTIZE_FP8_PACKED) {
    uint8_t *packed = pointer<uint8_t>(args.packed);
    pointer<uint8_t>(args.scales)[block_index] = scale_byte;
    for (uint32_t index = begin; index < end; ++index) {
      packed[index] =
          fp8_byte(clamp_value(input[index] / scale, -448.0f, 448.0f));
    }
  } else if (fp4) {
    uint8_t *packed = pointer<uint8_t>(args.packed);
    pointer<uint8_t>(args.scales)[block_index] = scale_byte;
    const uint32_t packed_row = row * (args.row_width / 2);
    const uint32_t packed_block = block * (args.block_size / 2);
    for (uint32_t index = 0; index < args.block_size; index += 2) {
      const uint8_t low =
          fp4_nibble(input[args.value_offset + begin + index] / scale);
      const uint8_t high =
          fp4_nibble(input[args.value_offset + begin + index + 1] / scale);
      packed[packed_row + packed_block + index / 2] = low | (high << 4);
    }
  }
}

__device__ inline uint64_t
paged_row_offset(uint64_t plane_elements, const int32_t *slots,
                 const int32_t *offsets, uint32_t sequence,
                 uint32_t logical_row, uint32_t page_tokens, uint32_t width,
                 uint32_t layer, uint32_t layers) {
  if (page_tokens == 0 || width == 0 || layer >= layers) {
    return UINT64_MAX;
  }
  const int32_t start = offsets[sequence];
  const int32_t end = offsets[sequence + 1];
  if (start < 0 || end < start) {
    return UINT64_MAX;
  }
  const uint64_t entry =
      static_cast<uint64_t>(start) + logical_row / page_tokens;
  if (entry >= static_cast<uint64_t>(end) || slots[entry] < 0) {
    return UINT64_MAX;
  }
  const uint64_t slot_stride =
      static_cast<uint64_t>(layers) * page_tokens * width;
  const uint64_t layer_stride = static_cast<uint64_t>(page_tokens) * width;
  const uint64_t result =
      static_cast<uint64_t>(slots[entry]) * slot_stride +
      static_cast<uint64_t>(layer) * layer_stride +
      static_cast<uint64_t>(logical_row % page_tokens) * width;
  return result + width <= plane_elements ? result : UINT64_MAX;
}

__global__ void data_kernel(FerruleCoreDataArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= args.count) {
    return;
  }
  if (args.kind == FERRULE_CORE_DATA_FILL_I32) {
    const int64_t value =
        static_cast<int64_t>(static_cast<int32_t>(args.start)) + index;
    pointer<int32_t>(args.output0)[index] =
        value > INT32_MAX ? INT32_MAX : static_cast<int32_t>(value);
  } else if (args.kind == FERRULE_CORE_DATA_PACK_I32_F32) {
    const uint32_t pair = static_cast<uint32_t>(index / 2);
    pointer<int32_t>(args.output0)[index] =
        (index & 1u) == 0 ? const_pointer<int32_t>(args.input0)[pair]
                          : static_cast<int32_t>(__float_as_uint(
                                const_pointer<float>(args.input1)[pair]));
  } else if (args.kind == FERRULE_CORE_DATA_PACK_PROPOSAL_HEAD) {
    int32_t value;
    if (index == 0) {
      value = const_pointer<int32_t>(args.input0)[0];
    } else if (index <= args.rows) {
      value = const_pointer<int32_t>(args.input1)[index];
    } else {
      value = static_cast<int32_t>(__float_as_uint(
          const_pointer<float>(args.input2)[index - args.rows - 1]));
    }
    pointer<int32_t>(args.output0)[index] = value;
  } else if (args.kind == FERRULE_CORE_DATA_FILL_PAGED_WINDOW) {
    pointer<int32_t>(args.output0)[index] =
        index < args.value0 ? static_cast<int32_t>(args.start + index) : -1;
  } else if (args.kind == FERRULE_CORE_DATA_FILL_DECODE_TOPK) {
    int32_t value = -1;
    if (index < args.width) {
      if (index < args.value0) {
        value = args.value0 < args.width
                    ? static_cast<int32_t>(index)
                    : static_cast<int32_t>(
                          (args.start % args.width + 1 + index) % args.width);
      }
    } else if (index - args.width < args.value1) {
      value = static_cast<int32_t>(index);
    }
    pointer<int32_t>(args.output0)[index] = value;
  } else if (args.kind == FERRULE_CORE_DATA_FILL_RECENT_ROWS) {
    const uint32_t row = static_cast<uint32_t>(index / args.width);
    const uint32_t column = static_cast<uint32_t>(index % args.width);
    const int32_t visible = const_pointer<int32_t>(args.input0)[row];
    int32_t value = -1;
    if (visible > 0) {
      const uint32_t valid = min(static_cast<uint32_t>(visible), args.width);
      if (column < valid) {
        value = visible <= static_cast<int32_t>(args.width)
                    ? static_cast<int32_t>(column)
                    : visible - static_cast<int32_t>(valid) +
                          static_cast<int32_t>(column);
      }
    }
    pointer<int32_t>(args.output0)[index] = value;
  } else if (args.kind == FERRULE_CORE_DATA_COPY_F32) {
    pointer<float>(args.output0)[args.offset + index] =
        const_pointer<float>(args.input0)[index];
  } else if (args.kind == FERRULE_CORE_DATA_GATHER_F32_ROWS) {
    const uint32_t row = static_cast<uint32_t>(index / args.width);
    const uint32_t column = static_cast<uint32_t>(index % args.width);
    const int32_t source_row = const_pointer<int32_t>(args.input1)[row];
    if (source_row >= 0) {
      pointer<float>(args.output0)[index] = const_pointer<float>(
          args.input0)[static_cast<uint64_t>(source_row) * args.width + column];
    }
  } else if (args.kind == FERRULE_CORE_DATA_SCATTER_ADD_F32_ROWS) {
    const uint32_t row = static_cast<uint32_t>(index / args.width);
    const uint32_t column = static_cast<uint32_t>(index % args.width);
    const int32_t target_row = const_pointer<int32_t>(args.input1)[row];
    if (target_row >= 0) {
      atomicAdd(pointer<float>(args.output0) +
                    static_cast<uint64_t>(target_row) * args.width + column,
                const_pointer<float>(args.input0)[index]);
    }
  } else if (args.kind == FERRULE_CORE_DATA_SAXPY) {
    pointer<float>(args.output0)[index] +=
        args.scale * const_pointer<float>(args.input0)[index];
  } else if (args.kind == FERRULE_CORE_DATA_CONVERT_COMBINED_RING) {
    if (args.width == 0 || args.value1 == 0) {
      return;
    }
    const uint32_t row = static_cast<uint32_t>(index / args.width);
    const uint64_t position = static_cast<uint64_t>(args.start) +
                              static_cast<uint64_t>(row) * args.value0;
    const uint64_t maximum =
        min(position + 1, static_cast<uint64_t>(args.value1));
    uint64_t valid_window = maximum;
    if ((args.flags & 1u) != 0) {
      const int32_t explicit_len = const_pointer<int32_t>(args.input1)[row];
      valid_window =
          explicit_len >= 0 && static_cast<uint64_t>(explicit_len) <= maximum
              ? static_cast<uint64_t>(explicit_len)
              : 0;
    }
    const int32_t combined = const_pointer<int32_t>(args.input0)[index];
    int32_t logical = -1;
    int32_t selector = -1;
    if (combined >= 0) {
      const uint64_t candidate = static_cast<uint32_t>(combined);
      if (candidate >= args.value1) {
        const uint64_t compressed = candidate - args.value1;
        if (compressed <= INT32_MAX) {
          logical = static_cast<int32_t>(compressed);
          selector = 1;
        }
      } else if (position < args.value1) {
        if (candidate < valid_window && candidate <= position) {
          logical = static_cast<int32_t>(candidate);
          selector = 0;
        }
      } else {
        const uint64_t age =
            (position % args.value1 + args.value1 - candidate) % args.value1;
        if (age < valid_window && position - age <= INT32_MAX) {
          logical = static_cast<int32_t>(position - age);
          selector = 0;
        }
      }
    }
    pointer<int32_t>(args.output0)[index] = logical;
    pointer<int32_t>(args.output1)[index] = selector;
  } else if (args.kind == FERRULE_CORE_DATA_PAGED_PLANE_SCATTER) {
    const uint32_t row = static_cast<uint32_t>(index / args.width);
    const uint32_t column = static_cast<uint32_t>(index % args.width);
    if ((args.flags & 2u) != 0 &&
        const_pointer<int32_t>(args.input5)[row] == 0) {
      return;
    }
    const int32_t logical = const_pointer<int32_t>(args.input1)[row];
    if (logical < 0) {
      return;
    }
    const uint32_t sequence =
        (args.flags & 1u) != 0
            ? static_cast<uint32_t>(const_pointer<int32_t>(args.input4)[row])
            : row;
    const uint64_t base = paged_row_offset(
        args.output_elements, const_pointer<int32_t>(args.input2),
        const_pointer<int32_t>(args.input3), sequence,
        static_cast<uint32_t>(logical), args.value0, args.width, args.value1,
        args.value2);
    if (base != UINT64_MAX) {
      pointer<float>(args.output0)[base + column] =
          const_pointer<float>(args.input0)[index];
    }
  }
}

__global__ void embedding_kernel(FerruleCoreEmbeddingArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t row_width = static_cast<uint64_t>(args.hc) * args.hidden;
  if (index >= static_cast<uint64_t>(args.rows) * row_width || row_width == 0) {
    return;
  }
  const uint32_t row = static_cast<uint32_t>(index / row_width);
  uint32_t token =
      args.kind == FERRULE_CORE_EMBED_PROPOSAL_HC_BF16
          ? (row == 0 ? args.anchor_token : args.noise_token)
          : static_cast<uint32_t>(const_pointer<int32_t>(args.token_ids)[row]);
  if (token >= args.vocab) {
    return;
  }
  const uint32_t dimension = static_cast<uint32_t>(index % args.hidden);
  pointer<float>(args.output)[index] = bf16_value(const_pointer<uint16_t>(
      args.embedding)[static_cast<uint64_t>(token) * args.hidden + dimension]);
}

__global__ void norm_kernel(FerruleCoreNormArgs args) {
  const uint32_t row = blockIdx.x;
  if (row >= args.rows || args.width == 0) {
    return;
  }
  const float *input = const_pointer<float>(args.input);
  const uint64_t base = static_cast<uint64_t>(row) * args.width;
  const bool head_rows = args.kind == FERRULE_CORE_NORM_HEAD_ROWS;
  float sum = 0.0f;
  for (uint32_t column = threadIdx.x; column < args.width;
       column += blockDim.x) {
    const float value = input[base + column];
    const float square = value * value;
    sum += head_rows ? bf16_round(square) : square;
  }

  constexpr uint32_t kWarpSize = 32;
  constexpr uint32_t kWarpCount = kBlock / kWarpSize;
  __shared__ float warp_sums[kWarpCount];
  __shared__ float inverse_rms_shared;
  for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    sum = lane < kWarpCount ? warp_sums[lane] : 0.0f;
    for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
      if (head_rows) {
        const float mean = bf16_round(sum / args.width);
        const float mean_with_epsilon = bf16_round(mean + args.epsilon);
        inverse_rms_shared = bf16_round(rsqrtf(mean_with_epsilon));
      } else {
        inverse_rms_shared = rsqrtf(sum / args.width + args.epsilon);
      }
      if (args.kind == FERRULE_CORE_NORM_COMPUTE_RMS) {
        pointer<float>(args.output)[row] = inverse_rms_shared;
      }
    }
  }
  __syncthreads();
  if (args.kind == FERRULE_CORE_NORM_COMPUTE_RMS) {
    return;
  }

  float *output = pointer<float>(args.output);
  const float *weight = const_pointer<float>(args.weight);
  for (uint32_t column = threadIdx.x; column < args.width;
       column += blockDim.x) {
    const float affine =
        args.kind == FERRULE_CORE_NORM_HEAD_ROWS ? 1.0f : weight[column];
    output[base + column] =
        bf16_round(input[base + column] * inverse_rms_shared * affine);
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

__device__ inline void insert_topk(float value, int32_t candidate,
                                   float *best_values, int32_t *best_indices,
                                   uint32_t k) {
  uint32_t position = k;
  while (position > 0) {
    const uint32_t previous = position - 1;
    if (value < best_values[previous] ||
        (value == best_values[previous] && best_indices[previous] >= 0 &&
         candidate >= best_indices[previous])) {
      break;
    }
    --position;
  }
  if (position < k) {
    for (uint32_t move = k - 1; move > position; --move) {
      best_values[move] = best_values[move - 1];
      best_indices[move] = best_indices[move - 1];
    }
    best_values[position] = value;
    best_indices[position] = candidate;
  }
}

__global__ void router_kernel(FerruleCoreRouterArgs args) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= args.rows || args.k == 0 || args.k > 64) {
    return;
  }
  float best_values[64];
  int32_t best_indices[64];
  for (uint32_t index = 0; index < args.k; ++index) {
    best_values[index] = -CUDART_INF_F;
    best_indices[index] = -1;
  }
  const float *logits = const_pointer<float>(args.logits);
  if (args.kind == FERRULE_CORE_ROUTER_TOPK) {
    const float *bias = const_pointer<float>(args.bias);
    for (uint32_t expert = 0; expert < args.columns; ++expert) {
      const float score = sqrtf(fmaxf(
          softplus(logits[static_cast<uint64_t>(row) * args.columns + expert]),
          0.0f));
      const float selection =
          score + ((args.flags & 1u) != 0 ? bias[expert] : 0.0f);
      insert_topk(selection, static_cast<int32_t>(expert), best_values,
                  best_indices, args.k);
    }
    float sum = 0.0f;
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      const int32_t expert = best_indices[rank];
      const float score =
          expert >= 0
              ? sqrtf(fmaxf(
                    softplus(logits[static_cast<uint64_t>(row) * args.columns +
                                    expert]),
                    0.0f))
              : 0.0f;
      best_values[rank] = score;
      sum += score;
    }
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      pointer<int32_t>(
          args.indices)[static_cast<uint64_t>(row) * args.k + rank] =
          best_indices[rank];
      pointer<float>(args.weights)[static_cast<uint64_t>(row) * args.k + rank] =
          sum > 0.0f && isfinite(sum)
              ? best_values[rank] / sum * args.route_scale
              : 0.0f;
    }
  } else if (args.kind == FERRULE_CORE_ROUTER_HASH) {
    const int32_t token = const_pointer<int32_t>(args.token_ids)[row];
    if (token < 0 || static_cast<uint32_t>(token) >= args.hash_rows) {
      return;
    }
    float sum = 0.0f;
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      const int32_t expert = const_pointer<int32_t>(
          args.hash_table)[static_cast<uint64_t>(token) * args.hash_columns +
                           rank];
      if (expert < 0 || static_cast<uint32_t>(expert) >= args.columns) {
        return;
      }
      const float score = sqrtf(fmaxf(
          softplus(logits[static_cast<uint64_t>(row) * args.columns + expert]),
          0.0f));
      best_indices[rank] = expert;
      best_values[rank] = score;
      sum += score;
    }
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      pointer<int32_t>(
          args.indices)[static_cast<uint64_t>(row) * args.k + rank] =
          best_indices[rank];
      pointer<float>(args.weights)[static_cast<uint64_t>(row) * args.k + rank] =
          sum > 0.0f && isfinite(sum)
              ? best_values[rank] / sum * args.route_scale
              : 0.0f;
    }
  } else {
    for (uint32_t column = 0; column < args.columns; ++column) {
      insert_topk(logits[static_cast<uint64_t>(row) * args.columns + column],
                  static_cast<int32_t>(column), best_values, best_indices,
                  args.k);
    }
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      const uint64_t output = static_cast<uint64_t>(row) * args.k + rank;
      if (args.kind == FERRULE_CORE_VOCAB_TOPK_F32_INDEX) {
        pointer<float>(args.indices)[output] =
            static_cast<float>(best_indices[rank]);
      } else {
        pointer<int32_t>(args.indices)[output] = best_indices[rank];
      }
      pointer<float>(args.weights)[output] = best_values[rank];
    }
  }
}

__device__ inline void compressor_source(bool overlap, uint32_t group,
                                         uint32_t row, uint32_t ratio,
                                         uint32_t head_dim, uint32_t dimension,
                                         bool *valid, uint32_t *token,
                                         uint32_t *source_dimension,
                                         uint32_t *ape_dimension) {
  *valid = true;
  if (overlap) {
    if (row < ratio) {
      if (group == 0) {
        *valid = false;
      }
      *token = (group == 0 ? 0 : group - 1) * ratio + row;
      *source_dimension = dimension;
      *ape_dimension = dimension;
    } else {
      *token = group * ratio + row - ratio;
      *source_dimension = head_dim + dimension;
      *ape_dimension = head_dim + dimension;
    }
  } else {
    *token = group * ratio + row;
    *source_dimension = dimension;
    *ape_dimension = dimension;
  }
}

__global__ void compressor_kernel(FerruleCoreCompressorArgs args) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  float *kv_state = pointer<float>(args.kv_state);
  float *score_state = pointer<float>(args.score_state);
  if (args.kind == FERRULE_CORE_COMPRESSOR_RESET) {
    if (index < args.state_elements) {
      kv_state[index] = 0.0f;
      score_state[index] = -CUDART_INF_F;
    }
    return;
  }
  if (args.kind == FERRULE_CORE_COMPRESSOR_APPEND) {
    if (index >= args.output_dim || args.ratio == 0) {
      return;
    }
    const uint32_t local = args.position % args.ratio;
    const uint32_t state_row = args.overlap != 0 ? args.ratio + local : local;
    const uint64_t target =
        static_cast<uint64_t>(state_row) * args.output_dim + index;
    kv_state[target] = const_pointer<float>(args.kv_input)[index];
    score_state[target] =
        const_pointer<float>(args.score_input)[index] +
        const_pointer<float>(
            args.ape)[static_cast<uint64_t>(local) * args.output_dim + index];
    return;
  }
  if (args.kind == FERRULE_CORE_COMPRESSOR_SEED) {
    if (index >= args.state_elements || args.ratio == 0 ||
        args.output_dim == 0) {
      return;
    }
    const uint32_t state_row = index / args.output_dim;
    const uint32_t dimension = index % args.output_dim;
    const uint32_t remainder = args.tokens % args.ratio;
    const uint32_t cutoff = args.tokens - remainder;
    int64_t source_token = -1;
    uint32_t ape_row = 0;
    if (args.overlap != 0 && cutoff >= args.ratio && state_row < args.ratio) {
      source_token = cutoff - args.ratio + state_row;
      ape_row = state_row;
    } else {
      const uint32_t state_offset = args.overlap != 0 ? args.ratio : 0;
      if (state_row >= state_offset && state_row < state_offset + remainder) {
        source_token = cutoff + state_row - state_offset;
        ape_row = state_row - state_offset;
      }
    }
    if (source_token < 0) {
      kv_state[index] = 0.0f;
      score_state[index] = -CUDART_INF_F;
    } else {
      const uint64_t source =
          static_cast<uint64_t>(source_token) * args.output_dim + dimension;
      kv_state[index] = const_pointer<float>(args.kv_input)[source];
      score_state[index] =
          const_pointer<float>(args.score_input)[source] +
          const_pointer<float>(
              args.ape)[static_cast<uint64_t>(ape_row) * args.output_dim +
                        dimension];
    }
    return;
  }
  const bool prefill = args.kind == FERRULE_CORE_COMPRESSOR_PREFILL;
  const uint32_t output_count =
      prefill ? args.groups * args.head_dim : args.head_dim;
  if (index >= output_count || args.ratio == 0) {
    return;
  }
  const uint32_t group = prefill ? index / args.head_dim : 0;
  const uint32_t dimension = index % args.head_dim;
  const uint32_t rows = args.overlap != 0 ? args.ratio * 2 : args.ratio;
  const float *kv = prefill ? const_pointer<float>(args.kv_input)
                            : const_pointer<float>(args.kv_state);
  const float *scores = prefill ? const_pointer<float>(args.score_input)
                                : const_pointer<float>(args.score_state);
  float maximum = -CUDART_INF_F;
  for (uint32_t row = 0; row < rows; ++row) {
    bool source_valid;
    uint32_t token, source_dimension, ape_dimension;
    compressor_source(args.overlap != 0, group, row, args.ratio, args.head_dim,
                      dimension, &source_valid, &token, &source_dimension,
                      &ape_dimension);
    if (source_valid) {
      float score = scores[static_cast<uint64_t>(prefill ? token : row) *
                               args.output_dim +
                           source_dimension];
      if (prefill) {
        score += const_pointer<float>(
            args.ape)[static_cast<uint64_t>(row % args.ratio) *
                          args.output_dim +
                      ape_dimension];
      }
      maximum = fmaxf(maximum, score);
    }
  }
  float denominator = 0.0f;
  float result = 0.0f;
  for (uint32_t row = 0; row < rows; ++row) {
    bool source_valid;
    uint32_t token, source_dimension, ape_dimension;
    compressor_source(args.overlap != 0, group, row, args.ratio, args.head_dim,
                      dimension, &source_valid, &token, &source_dimension,
                      &ape_dimension);
    if (source_valid) {
      const uint64_t source =
          static_cast<uint64_t>(prefill ? token : row) * args.output_dim +
          source_dimension;
      float score = scores[source];
      if (prefill) {
        score += const_pointer<float>(
            args.ape)[static_cast<uint64_t>(row % args.ratio) *
                          args.output_dim +
                      ape_dimension];
      }
      const float weight = expf(score - maximum);
      denominator += weight;
      result += weight * kv[source];
    }
  }
  pointer<float>(args.output)[index] =
      denominator > 0.0f && isfinite(denominator) ? result / denominator : 0.0f;
}

__device__ inline void indexer_insert(float score, int32_t candidate,
                                      float *best_scores, int32_t *best_indices,
                                      uint32_t take) {
  insert_topk(score, candidate, best_scores, best_indices, take);
}

__device__ inline void transform_index_query(const FerruleCoreIndexerArgs &args,
                                             uint32_t query_row, uint32_t head,
                                             uint32_t position, float *query) {
  const uint64_t query_base =
      (static_cast<uint64_t>(query_row) * args.heads + head) * args.head_dim;
  for (uint32_t dimension = 0; dimension < args.head_dim; ++dimension) {
    query[dimension] = const_pointer<float>(args.query)[query_base + dimension];
  }
  if (args.rope_dim != 0 && args.rope_dim <= args.head_dim &&
      (args.rope_dim & 1u) == 0) {
    const uint32_t tail_start = args.head_dim - args.rope_dim;
    const uint32_t pairs = args.rope_dim / 2;
    for (uint32_t pair = 0; pair < pairs; ++pair) {
      const uint32_t dimension = tail_start + pair * 2;
      const float first = query[dimension];
      const float second = query[dimension + 1];
      const float cosine = const_pointer<float>(
          args.cosine)[static_cast<uint64_t>(position) * pairs + pair];
      const float sine = const_pointer<float>(
          args.sine)[static_cast<uint64_t>(position) * pairs + pair];
      query[dimension] = first * cosine - second * sine;
      query[dimension + 1] = first * sine + second * cosine;
    }
  }
  for (uint32_t span = 1; span < args.head_dim; span *= 2) {
    const uint32_t step = span * 2;
    for (uint32_t start = 0; start < args.head_dim; start += step) {
      for (uint32_t offset = 0; offset < span; ++offset) {
        const uint32_t left = start + offset;
        const uint32_t right = left + span;
        const float first = query[left];
        const float second = query[right];
        query[left] = first + second;
        query[right] = first - second;
      }
    }
  }
  const float hadamard_scale = rsqrtf(static_cast<float>(args.head_dim));
  for (uint32_t block = 0; block < args.head_dim / 32; ++block) {
    float amax = 6.0f * exp2f(-126.0f);
    for (uint32_t element = 0; element < 32; ++element) {
      const uint32_t dimension = block * 32 + element;
      query[dimension] *= hadamard_scale;
      amax = fmaxf(amax, fabsf(query[dimension]));
    }
    const float scale = e8m0_scale(e8m0_byte(amax, 6.0f));
    for (uint32_t element = 0; element < 32; ++element) {
      const uint32_t dimension = block * 32 + element;
      query[dimension] =
          fp4_value(
              fp4_nibble(clamp_value(query[dimension] / scale, -6.0f, 6.0f))) *
          scale;
    }
  }
}

__global__ void indexer_kernel(FerruleCoreIndexerArgs args) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= args.rows || args.topk > kMaxTopK) {
    return;
  }
  const bool prefill = args.prefill != 0;
  const bool row_decode = !prefill && args.rows > 1;
  const bool fused_query = (args.flags & 1u) != 0;
  if (fused_query &&
      (args.head_dim == 0 || args.head_dim > 256 ||
       (args.head_dim & (args.head_dim - 1)) != 0 ||
       (args.head_dim % 32) != 0 || args.rope_dim > args.head_dim ||
       (args.rope_dim & 1u) != 0)) {
    return;
  }
  uint32_t position = args.position;
  uint32_t window_len = args.window_len;
  uint32_t compressed_len = args.compressed_len;
  uint32_t sequence = 0;
  bool metadata_valid = true;
  if (row_decode) {
    const int32_t sequence_value =
        const_pointer<int32_t>(args.row_sequence_ids)[row];
    const int32_t position_value = const_pointer<int32_t>(args.positions)[row];
    const int32_t window_value = const_pointer<int32_t>(args.window_lens)[row];
    const int32_t compressed_value =
        const_pointer<int32_t>(args.compressed_lens)[row];
    metadata_valid = sequence_value >= 0 && position_value >= 0 &&
                     window_value >= 0 && compressed_value >= 0;
    sequence = metadata_valid ? static_cast<uint32_t>(sequence_value) : 0;
    position = metadata_valid ? static_cast<uint32_t>(position_value) : 0;
    window_len = metadata_valid ? static_cast<uint32_t>(window_value) : 0;
    compressed_len =
        metadata_valid ? static_cast<uint32_t>(compressed_value) : 0;
  }
  const uint32_t window_columns =
      prefill ? args.window_columns : args.window_size;
  const uint32_t output_columns = window_columns + args.topk;
  int32_t *indices = pointer<int32_t>(args.indices);
  int32_t *selectors = pointer<int32_t>(args.selectors);
  const uint64_t output_base = static_cast<uint64_t>(row) * output_columns;
  if (prefill) {
    const uint32_t first =
        row + 1 > args.window_size ? row + 1 - args.window_size : 0;
    for (uint32_t column = 0; column < window_columns; ++column) {
      const uint32_t candidate = first + column;
      indices[output_base + column] =
          candidate <= row ? static_cast<int32_t>(candidate) : -1;
    }
    compressed_len = args.compress_ratio == 0
                         ? 0
                         : min((row + 1) / args.compress_ratio, compressed_len);
  } else if (row_decode) {
    for (uint32_t column = 0; column < args.window_size; ++column) {
      if (metadata_valid && window_len <= args.window_size &&
          window_len <= position + 1 && column < window_len) {
        indices[output_base + column] =
            static_cast<int32_t>(position + 1 - window_len + column);
        selectors[output_base + column] = 0;
      } else {
        indices[output_base + column] = -1;
        selectors[output_base + column] = -1;
      }
    }
  } else {
    for (uint32_t column = 0; column < args.window_size; ++column) {
      if (window_len < args.window_size) {
        indices[column] =
            column < window_len ? static_cast<int32_t>(column) : -1;
      } else {
        indices[column] = static_cast<int32_t>(
            (position % args.window_size + 1 + column) % args.window_size);
      }
    }
  }
  if (args.topk == 0 || !metadata_valid) {
    return;
  }
  float best_scores[kMaxTopK];
  int32_t best_indices[kMaxTopK];
  for (uint32_t rank = 0; rank < args.topk; ++rank) {
    best_scores[rank] = -CUDART_INF_F;
    best_indices[rank] = -1;
  }
  const float *query = const_pointer<float>(args.query);
  const float *weights = const_pointer<float>(args.weights);
  const float *plane = const_pointer<float>(args.plane);
  const uint32_t query_row = prefill || row_decode ? row : 0;
  for (uint32_t candidate = 0; candidate < compressed_len; ++candidate) {
    const uint64_t plane_base = paged_row_offset(
        args.plane_elements, const_pointer<int32_t>(args.block_slots),
        const_pointer<int32_t>(args.block_offsets), sequence, candidate,
        args.page_tokens, args.head_dim, args.layer_index, args.layer_count);
    float score = plane_base == UINT64_MAX ? -CUDART_INF_F : 0.0f;
    if (plane_base != UINT64_MAX) {
      for (uint32_t head = 0; head < args.heads; ++head) {
        float dot = 0.0f;
        if (fused_query) {
          float transformed[256];
          const uint32_t query_position =
              prefill ? args.start_position + row : position;
          transform_index_query(args, query_row, head, query_position,
                                transformed);
          for (uint32_t dimension = 0; dimension < args.head_dim; ++dimension) {
            dot += transformed[dimension] * plane[plane_base + dimension];
          }
        } else {
          const uint64_t query_base =
              (static_cast<uint64_t>(query_row) * args.heads + head) *
              args.head_dim;
          for (uint32_t dimension = 0; dimension < args.head_dim; ++dimension) {
            dot +=
                query[query_base + dimension] * plane[plane_base + dimension];
          }
        }
        score += fmaxf(dot, 0.0f) *
                 weights[static_cast<uint64_t>(query_row) * args.heads + head] *
                 args.weight_scale;
      }
    }
    indexer_insert(score, static_cast<int32_t>(candidate), best_scores,
                   best_indices, args.topk);
  }
  for (uint32_t rank = 0; rank < args.topk; ++rank) {
    const bool found = best_indices[rank] >= 0 && isfinite(best_scores[rank]);
    const uint64_t output = output_base + window_columns + rank;
    indices[output] =
        !found ? -1
        : row_decode
            ? best_indices[rank]
            : static_cast<int32_t>(args.value_offset + best_indices[rank]);
    if (row_decode) {
      selectors[output] = found ? 1 : -1;
    }
  }
}

__global__ void expert_table_kernel(FerruleCoreExpertTableArgs args) {
  const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
  if (args.kind == FERRULE_CORE_EXPERT_INSTALL ||
      args.kind == FERRULE_CORE_EXPERT_EVICT) {
    if (route != 0) {
      return;
    }
    uint64_t *gate = pointer<uint64_t>(args.gate_weights);
    uint64_t *gate_scale = pointer<uint64_t>(args.gate_scales);
    uint64_t *up = pointer<uint64_t>(args.up_weights);
    uint64_t *up_scale = pointer<uint64_t>(args.up_scales);
    uint64_t *down = pointer<uint64_t>(args.down_weights);
    uint64_t *down_scale = pointer<uint64_t>(args.down_scales);
    if (args.kind == FERRULE_CORE_EXPERT_INSTALL) {
      gate[args.slot] = args.gate_weight_value;
      gate_scale[args.slot] = args.gate_scale_value;
      up[args.slot] = args.up_weight_value;
      up_scale[args.slot] = args.up_scale_value;
      down[args.slot] = args.down_weight_value;
      down_scale[args.slot] = args.down_scale_value;
      pointer<int32_t>(args.slot_generations)[args.slot] = args.generation;
      pointer<int32_t>(args.expert_to_slot)[args.expert] =
          static_cast<int32_t>(args.slot);
      pointer<int32_t>(args.expert_generations)[args.expert] = args.generation;
    } else {
      pointer<int32_t>(args.expert_to_slot)[args.expert] = -1;
      pointer<int32_t>(args.expert_generations)[args.expert] = 0;
      gate[args.slot] = gate_scale[args.slot] = up[args.slot] =
          up_scale[args.slot] = 0;
      down[args.slot] = down_scale[args.slot] = 0;
      pointer<int32_t>(args.slot_generations)[args.slot] = args.generation;
    }
    return;
  }
  if (args.kind == FERRULE_CORE_EXPERT_INITIALIZE_RESOLVE) {
    const uint32_t count = args.miss_capacity + args.route_capacity + 2;
    if (route < count) {
      pointer<int32_t>(args.miss_control)[route] = route < 2 ? 0 : -1;
    }
    return;
  }
  if (route >= args.route_count) {
    return;
  }
  if (args.kind == FERRULE_CORE_EXPERT_RESOLVE) {
    const int32_t expert = const_pointer<int32_t>(args.expert_ids)[route];
    int32_t slot = -1;
    int32_t generation = 0;
    if (expert >= 0 && static_cast<uint32_t>(expert) < args.expert_capacity) {
      const int32_t mapped =
          const_pointer<int32_t>(args.expert_to_slot)[expert];
      const int32_t expert_generation =
          const_pointer<int32_t>(args.expert_generations)[expert];
      if (mapped >= 0 && static_cast<uint32_t>(mapped) < args.slot_capacity &&
          expert_generation > 0 &&
          const_pointer<int32_t>(args.slot_generations)[mapped] ==
              expert_generation) {
        slot = mapped;
        generation = expert_generation;
      }
    }
    pointer<int32_t>(args.route_slots)[route] = slot;
    pointer<int32_t>(args.route_generations)[route] = generation;
    pointer<int32_t>(args.miss_markers)[route] = slot < 0 ? 1 : 0;
    pointer<int32_t>(args.miss_control)[2 + args.miss_capacity + route] =
        expert;
    if (slot < 0) {
      const int32_t miss = atomicAdd(pointer<int32_t>(args.miss_control), 1);
      if (miss >= 0 && static_cast<uint32_t>(miss) < args.miss_capacity) {
        pointer<int32_t>(args.miss_control)[2 + miss] = expert;
      } else {
        atomicOr(pointer<int32_t>(args.miss_control) + 1, 1);
      }
    }
    return;
  }
  const int32_t slot = const_pointer<int32_t>(args.route_slots)[route];
  const int32_t generation =
      const_pointer<int32_t>(args.route_generations)[route];
  const bool active =
      const_pointer<int32_t>(args.active_markers)[route] == args.active_value;
  const bool current =
      slot >= 0 && static_cast<uint32_t>(slot) < args.slot_capacity &&
      generation > 0 &&
      const_pointer<int32_t>(args.slot_generations)[slot] == generation;
  const uint64_t gate =
      current ? const_pointer<uint64_t>(args.gate_weights)[slot] : 0;
  const uint64_t gate_scale =
      current ? const_pointer<uint64_t>(args.gate_scales)[slot] : 0;
  const uint64_t up =
      current ? const_pointer<uint64_t>(args.up_weights)[slot] : 0;
  const uint64_t up_scale =
      current ? const_pointer<uint64_t>(args.up_scales)[slot] : 0;
  const uint64_t down =
      current ? const_pointer<uint64_t>(args.down_weights)[slot] : 0;
  const uint64_t down_scale =
      current ? const_pointer<uint64_t>(args.down_scales)[slot] : 0;
  const bool good = active && current && gate != 0 && gate_scale != 0 &&
                    up != 0 && up_scale != 0 && down != 0 && down_scale != 0;
  if (active && !good) {
    atomicOr(pointer<int32_t>(args.dispatch_error), 1);
  }
  pointer<uint64_t>(args.output_gate_weights)[route] = gate;
  pointer<uint64_t>(args.output_gate_scales)[route] = gate_scale;
  pointer<uint64_t>(args.output_up_weights)[route] = up;
  pointer<uint64_t>(args.output_up_scales)[route] = up_scale;
  pointer<uint64_t>(args.output_down_weights)[route] = down;
  pointer<uint64_t>(args.output_down_scales)[route] = down_scale;
  pointer<float>(args.output_route_weights)[route] =
      good ? const_pointer<float>(args.router_weights)[route] : 0.0f;
  pointer<int32_t>(args.expert_ids)[route] =
      good ? static_cast<int32_t>(route) : -1;
}

__global__ void
expert_group_route_plan_kernel(FerruleCoreExpertGroupRoutePlanArgs args) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (args.kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_INIT_INVOCATION) {
    if (index < args.output_elements) {
      pointer<float>(args.route_output)[index] = 0.0f;
    }
    if (index < args.route_count) {
      pointer<int32_t>(args.route_written)[index] = 0;
    }
    if (index == 0) {
      pointer<int32_t>(args.route_error)[0] = 0;
    }
  } else if (args.kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_INIT_PLAN) {
    if (index < args.slot_capacity) {
      pointer<int32_t>(args.slot_counts)[index] = 0;
      pointer<int32_t>(args.slot_route_offsets)[index] = -1;
      pointer<int32_t>(args.slot_cursors)[index] = 0;
      pointer<int32_t>(args.active_expert_slots)[index] = -1;
      pointer<int32_t>(args.active_group_generations)[index] = 0;
      pointer<int32_t>(args.expert_route_counts)[index] = 0;
    }
    if (index <= args.slot_capacity) {
      pointer<int32_t>(args.expert_route_indptr)[index] = 0;
    }
    if (index < args.route_capacity) {
      pointer<int32_t>(args.route_token_indices)[index] = -1;
      pointer<int32_t>(args.route_indices)[index] = -1;
      pointer<float>(args.route_weights)[index] = 0.0f;
    }
    if (index < 4) {
      pointer<int32_t>(args.host_scalars)[index] = 0;
    }
  } else if (args.kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_COUNT) {
    if (index >= args.route_count) {
      return;
    }
    const int32_t slot = const_pointer<int32_t>(args.route_slots)[index];
    const int32_t generation =
        const_pointer<int32_t>(args.route_generations)[index];
    if (slot >= 0 && static_cast<uint32_t>(slot) < args.slot_capacity &&
        generation > 0 &&
        const_pointer<int32_t>(args.slot_generations)[slot] == generation) {
      atomicAdd(pointer<int32_t>(args.slot_counts) + slot, 1);
    }
  } else if (args.kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_COMPACT) {
    if (index != 0) {
      return;
    }
    uint32_t active_group_count = 0;
    uint32_t small_group_count = 0;
    uint32_t max_group_rows = 0;
    uint32_t total_routed_rows = 0;
    for (uint32_t slot = 0; slot < args.slot_capacity; ++slot) {
      const uint32_t count = static_cast<uint32_t>(
          max(0, const_pointer<int32_t>(args.slot_counts)[slot]));
      if (count != 0 && count < args.small_group_row_limit) {
        ++small_group_count;
      }
    }
    active_group_count = small_group_count;
    for (uint32_t pass = 0; pass < 2; ++pass) {
      uint32_t group = pass == 0 ? 0 : small_group_count;
      for (uint32_t slot = 0; slot < args.slot_capacity; ++slot) {
        const uint32_t count = static_cast<uint32_t>(
            max(0, const_pointer<int32_t>(args.slot_counts)[slot]));
        if (count == 0 || (count < args.small_group_row_limit) != (pass == 0)) {
          continue;
        }
        if (group >= args.slot_capacity ||
            total_routed_rows + count > args.route_capacity) {
          pointer<int32_t>(args.route_error)[0] = 1;
          return;
        }
        pointer<int32_t>(args.active_expert_slots)[group] =
            static_cast<int32_t>(slot);
        pointer<int32_t>(args.active_group_generations)[group] =
            const_pointer<int32_t>(args.slot_generations)[slot];
        pointer<int32_t>(args.expert_route_indptr)[group] =
            static_cast<int32_t>(total_routed_rows);
        pointer<int32_t>(args.expert_route_counts)[group] =
            static_cast<int32_t>(count);
        pointer<int32_t>(args.slot_route_offsets)[slot] =
            static_cast<int32_t>(total_routed_rows);
        max_group_rows = max(max_group_rows, count);
        total_routed_rows += count;
        ++group;
      }
      if (pass == 1) {
        active_group_count = group;
      }
    }
    pointer<int32_t>(args.expert_route_indptr)[active_group_count] =
        static_cast<int32_t>(total_routed_rows);
    pointer<int32_t>(args.host_scalars)[0] =
        static_cast<int32_t>(active_group_count);
    pointer<int32_t>(args.host_scalars)[1] =
        static_cast<int32_t>(small_group_count);
    pointer<int32_t>(args.host_scalars)[2] =
        static_cast<int32_t>(max_group_rows);
    pointer<int32_t>(args.host_scalars)[3] =
        static_cast<int32_t>(total_routed_rows);
  } else if (args.kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_SCATTER) {
    if (index >= args.route_count || args.routes_per_token == 0) {
      return;
    }
    const int32_t slot = const_pointer<int32_t>(args.route_slots)[index];
    const int32_t generation =
        const_pointer<int32_t>(args.route_generations)[index];
    if (slot < 0 || static_cast<uint32_t>(slot) >= args.slot_capacity ||
        generation <= 0 ||
        const_pointer<int32_t>(args.slot_generations)[slot] != generation) {
      return;
    }
    const int32_t position =
        atomicAdd(pointer<int32_t>(args.slot_cursors) + slot, 1);
    const int32_t offset =
        const_pointer<int32_t>(args.slot_route_offsets)[slot];
    if (position < 0 || offset < 0 ||
        static_cast<uint32_t>(offset + position) >= args.route_capacity) {
      atomicOr(pointer<int32_t>(args.route_error), 1);
      return;
    }
    const uint32_t metadata = static_cast<uint32_t>(offset + position);
    pointer<int32_t>(args.route_token_indices)[metadata] =
        static_cast<int32_t>(index / args.routes_per_token);
    pointer<int32_t>(args.route_indices)[metadata] =
        static_cast<int32_t>(index);
    pointer<float>(args.route_weights)[metadata] =
        const_pointer<float>(args.router_weights)[index];
  }
}

__global__ void moe_kernel(FerruleCoreMoeArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (args.kind == FERRULE_CORE_MOE_WEIGHTED_SWIGLU_F32) {
    const uint64_t total =
        static_cast<uint64_t>(args.experts) * args.batch_columns * args.n;
    if (index >= total) {
      return;
    }
    float gate = const_pointer<float>(args.gate)[index];
    float up = const_pointer<float>(args.up)[index];
    if (args.swiglu_limit > 0.0f) {
      gate = fminf(gate, args.swiglu_limit);
      up = clamp_value(up, -args.swiglu_limit, args.swiglu_limit);
    }
    const uint32_t route = static_cast<uint32_t>(index / args.n);
    const bool quantize = args.route_weights != 0;
    const float route_weight =
        quantize ? const_pointer<float>(args.route_weights)[route]
                 : args.route_weight;
    const float value = gate * sigmoid(gate) * up * route_weight;
    pointer<float>(args.hidden_values)[index] =
        quantize ? fp8_quantized(value) : value;
  } else if (args.kind == FERRULE_CORE_MOE_REDUCE_EXPERT) {
    const uint64_t total =
        static_cast<uint64_t>(args.batch_columns) * args.hidden;
    if (index >= total) {
      return;
    }
    const uint32_t column = static_cast<uint32_t>(index / args.hidden);
    const uint32_t row = static_cast<uint32_t>(index % args.hidden);
    const uint64_t output = args.output_offset + index;
    float result = pointer<float>(args.output)[output];
    for (uint32_t rank = 0; rank < args.routes_per_token; ++rank) {
      const int32_t slot = const_pointer<int32_t>(
          args.route_slots)[column * args.routes_per_token + rank];
      if (slot >= 0 && static_cast<uint32_t>(slot) < args.experts) {
        result += const_pointer<float>(args.expert_output)
            [(static_cast<uint64_t>(slot) * args.batch_columns + column) *
                 args.hidden +
             row];
      }
    }
    pointer<float>(args.output)[output] = result;
  } else if (args.kind == FERRULE_CORE_MOE_REDUCE_SPLIT_EXPERT) {
    const uint64_t total =
        static_cast<uint64_t>(args.batch_columns) * args.hidden;
    if (index >= total) {
      return;
    }
    const uint32_t column = static_cast<uint32_t>(index / args.hidden);
    const uint32_t row = static_cast<uint32_t>(index % args.hidden);
    const uint64_t output = args.output_offset + index;
    float result = pointer<float>(args.output)[output];
    for (uint32_t rank = 0; rank < args.routes_per_token; ++rank) {
      const uint32_t route = column * args.routes_per_token + rank;
      const bool miss = const_pointer<int32_t>(args.miss_markers)[route] != 0;
      const int32_t slot = const_pointer<int32_t>(
          miss ? args.materialized_route_slots : args.route_slots)[route];
      if (slot >= 0 && static_cast<uint32_t>(slot) < args.experts) {
        result += const_pointer<float>(miss ? args.materialized_output
                                            : args.resident_output)
            [(static_cast<uint64_t>(slot) * args.batch_columns + column) *
                 args.hidden +
             row];
      }
    }
    pointer<float>(args.output)[index] = bf16_round(result);
  } else if (args.kind == FERRULE_CORE_MOE_REDUCE_ROUTES ||
             args.kind == FERRULE_CORE_MOE_REDUCE_EXPERT_GROUP_ROUTES) {
    const uint64_t total = static_cast<uint64_t>(args.tokens) * args.hidden;
    if (index >= total) {
      return;
    }
    const uint32_t token = static_cast<uint32_t>(index / args.hidden);
    const uint32_t row = static_cast<uint32_t>(index % args.hidden);
    bool complete = true;
    if (args.kind == FERRULE_CORE_MOE_REDUCE_EXPERT_GROUP_ROUTES) {
      complete = const_pointer<int32_t>(args.route_error)[0] == 0;
      for (uint32_t rank = 0; rank < args.routes_per_token; ++rank) {
        complete &=
            const_pointer<int32_t>(
                args.route_written)[token * args.routes_per_token + rank] != 0;
      }
    }
    if (!complete) {
      pointer<float>(args.output)[index] = CUDART_NAN_F;
      return;
    }
    float result = pointer<float>(args.output)[index];
    for (uint32_t rank = 0; rank < args.routes_per_token; ++rank) {
      const uint32_t route = token * args.routes_per_token + rank;
      result += const_pointer<float>(
          args.route_output)[static_cast<uint64_t>(route) * args.hidden + row];
    }
    pointer<float>(args.output)[index] = bf16_round(result);
  }
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
  pointer<float>(args.output)[output_base + dimension] = sum / args.hc;
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
      pointer<float>(args.output)[output_base + dimension] = sum / args.hc;
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

__global__ void mla_kernel(FerruleCoreMlaArgs args) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= args.output_size || args.rank == 0) {
    return;
  }
  float sum_square = 0.0f;
  for (uint32_t rank = 0; rank < args.rank; ++rank) {
    float latent = 0.0f;
    for (uint32_t column = 0; column < args.hidden_size; ++column) {
      latent +=
          const_pointer<float>(args.input)[column] *
          const_pointer<float>(
              args.weight_a)[static_cast<uint64_t>(rank) * args.hidden_size +
                             column];
    }
    sum_square += latent * latent;
  }
  const float inverse_rms = rsqrtf(sum_square / args.rank + args.epsilon);
  float result = 0.0f;
  for (uint32_t rank = 0; rank < args.rank; ++rank) {
    float latent = 0.0f;
    for (uint32_t column = 0; column < args.hidden_size; ++column) {
      latent +=
          const_pointer<float>(args.input)[column] *
          const_pointer<float>(
              args.weight_a)[static_cast<uint64_t>(rank) * args.hidden_size +
                             column];
    }
    result += latent * inverse_rms *
              const_pointer<float>(args.norm_weight)[rank] *
              const_pointer<float>(
                  args.weight_b)[static_cast<uint64_t>(row) * args.rank + rank];
  }
  pointer<float>(args.output)[row] = result;
}

} // namespace

extern "C" int32_t
ferrule_core_linear_launch(const FerruleCoreLinearArgs *args) {
  if (!valid(args) || args->batch == 0 || args->n == 0) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  linear_kernel<<<blocks_for(static_cast<uint64_t>(args->batch) * args->n),
                  kBlock, 0, stream(args->stream)>>>(*args);
  return launch_status();
}

extern "C" int32_t
ferrule_core_dual_linear_launch(const FerruleCoreDualLinearArgs *args) {
  if (!valid(args) || args->kind != FERRULE_CORE_LINEAR_BF16_BYTES) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  dual_linear_kernel<<<blocks_for(static_cast<uint64_t>(args->first_n) +
                                  args->second_n),
                       kBlock, 0, stream(args->stream)>>>(*args);
  return launch_status();
}

extern "C" int32_t
ferrule_core_grouped_linear_launch(const FerruleCoreGroupedLinearArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  grouped_linear_kernel<<<blocks_for(static_cast<uint64_t>(args->rows) *
                                     args->output_dim),
                          kBlock, 0, stream(args->stream)>>>(*args);
  return launch_status();
}

extern "C" int32_t
ferrule_core_quantize_launch(const FerruleCoreQuantizeArgs *args) {
  if (!valid(args) || args->block_size == 0) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  uint64_t count = args->kind == FERRULE_CORE_QUANTIZE_HADAMARD_FP4_IN_PLACE
                       ? args->value_len / args->row_width
                       : args->value_len / args->block_size;
  if (args->kind == FERRULE_CORE_QUANTIZE_FP8_NON_ROPE_IN_PLACE) {
    const uint32_t non_rope = args->row_width - args->rope_dim;
    const uint32_t width =
        non_rope % args->block_size == 0 ? args->block_size : non_rope;
    count = static_cast<uint64_t>(args->value_len / args->row_width) *
            ((non_rope + width - 1) / width);
  }
  quantize_kernel<<<blocks_for(count), kBlock, 0, stream(args->stream)>>>(
      *args);
  return launch_status();
}

extern "C" int32_t ferrule_core_data_launch(const FerruleCoreDataArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  data_kernel<<<blocks_for(args->count), kBlock, 0, stream(args->stream)>>>(
      *args);
  return launch_status();
}

extern "C" int32_t
ferrule_core_embedding_launch(const FerruleCoreEmbeddingArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  embedding_kernel<<<blocks_for(static_cast<uint64_t>(args->rows) * args->hc *
                                args->hidden),
                     kBlock, 0, stream(args->stream)>>>(*args);
  return launch_status();
}

extern "C" int32_t ferrule_core_norm_launch(const FerruleCoreNormArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  norm_kernel<<<args->rows, kBlock, 0, stream(args->stream)>>>(*args);
  return launch_status();
}

extern "C" int32_t ferrule_core_rope_launch(const FerruleCoreRopeArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  rope_kernel<<<blocks_for(args->pair_count), kBlock, 0,
                stream(args->stream)>>>(*args);
  return launch_status();
}

extern "C" int32_t
ferrule_core_router_launch(const FerruleCoreRouterArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  router_kernel<<<blocks_for(args->rows), kBlock, 0, stream(args->stream)>>>(
      *args);
  return launch_status();
}

extern "C" int32_t
ferrule_core_compressor_launch(const FerruleCoreCompressorArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  uint64_t count = args->state_elements;
  if (args->kind == FERRULE_CORE_COMPRESSOR_APPEND)
    count = args->output_dim;
  if (args->kind == FERRULE_CORE_COMPRESSOR_PREFILL)
    count = static_cast<uint64_t>(args->groups) * args->head_dim;
  if (args->kind == FERRULE_CORE_COMPRESSOR_SOFTMAX)
    count = args->head_dim;
  compressor_kernel<<<blocks_for(count), kBlock, 0, stream(args->stream)>>>(
      *args);
  return launch_status();
}

extern "C" int32_t
ferrule_core_indexer_launch(const FerruleCoreIndexerArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  indexer_kernel<<<blocks_for(args->rows), kBlock, 0, stream(args->stream)>>>(
      *args);
  return launch_status();
}

extern "C" int32_t
ferrule_core_expert_table_launch(const FerruleCoreExpertTableArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  if (args->kind == FERRULE_CORE_EXPERT_GATHER_DISPATCH) {
    cudaError_t status =
        cudaMemsetAsync(pointer<int32_t>(args->dispatch_error), 0,
                        sizeof(int32_t), stream(args->stream));
    if (status != cudaSuccess)
      return static_cast<int32_t>(status);
  }
  uint64_t count = args->route_count;
  if (args->kind == FERRULE_CORE_EXPERT_INSTALL ||
      args->kind == FERRULE_CORE_EXPERT_EVICT)
    count = 1;
  if (args->kind == FERRULE_CORE_EXPERT_INITIALIZE_RESOLVE)
    count = args->miss_capacity + args->route_capacity + 2;
  expert_table_kernel<<<blocks_for(count), kBlock, 0, stream(args->stream)>>>(
      *args);
  return launch_status();
}

extern "C" int32_t ferrule_core_expert_group_route_plan_launch(
    const FerruleCoreExpertGroupRoutePlanArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  uint64_t count = args->route_count;
  if (args->kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_INIT_INVOCATION)
    count = max(args->output_elements, args->route_count);
  if (args->kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_INIT_PLAN)
    count = max(max(args->slot_capacity + 1, args->route_capacity), 4u);
  if (args->kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_COMPACT)
    count = 1;
  expert_group_route_plan_kernel<<<blocks_for(count), kBlock, 0,
                                   stream(args->stream)>>>(*args);
  return launch_status();
}

extern "C" int32_t ferrule_core_moe_launch(const FerruleCoreMoeArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  uint64_t count = 0;
  switch (args->kind) {
  case FERRULE_CORE_MOE_WEIGHTED_SWIGLU_F32:
    count =
        static_cast<uint64_t>(args->experts) * args->batch_columns * args->n;
    break;
  case FERRULE_CORE_MOE_REDUCE_EXPERT:
  case FERRULE_CORE_MOE_REDUCE_SPLIT_EXPERT:
    count = static_cast<uint64_t>(args->batch_columns) * args->hidden;
    break;
  case FERRULE_CORE_MOE_REDUCE_ROUTES:
  case FERRULE_CORE_MOE_REDUCE_EXPERT_GROUP_ROUTES:
    count = static_cast<uint64_t>(args->tokens) * args->hidden;
    break;
  default:
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  moe_kernel<<<blocks_for(count), kBlock, 0, stream(args->stream)>>>(*args);
  return launch_status();
}

extern "C" int32_t ferrule_core_hc_launch(const FerruleCoreHcArgs *args) {
  if (!valid(args) || args->hc > 16 || args->mix > 128) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  if (args->kind == FERRULE_CORE_HC_POST) {
    const uint64_t elements =
        static_cast<uint64_t>(args->tokens) * args->hc * args->hidden_size;
    hc_post_kernel<<<blocks_for(elements), kBlock, 0, stream(args->stream)>>>(
        *args);
  } else if (args->kind == FERRULE_CORE_HC_MEAN_SCATTER) {
    const uint64_t elements =
        static_cast<uint64_t>(args->tokens) * args->hidden_size;
    hc_mean_scatter_kernel<<<blocks_for(elements), kBlock, 0,
                             stream(args->stream)>>>(*args);
  } else {
    hc_kernel<<<blocks_for(args->tokens), kBlock, 0, stream(args->stream)>>>(
        *args);
  }
  return launch_status();
}

extern "C" int32_t ferrule_core_mla_launch(const FerruleCoreMlaArgs *args) {
  if (!valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  mla_kernel<<<blocks_for(args->output_size), kBlock, 0,
               stream(args->stream)>>>(*args);
  return launch_status();
}
