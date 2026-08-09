#pragma once

#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <stdint.h>

namespace ferrule::cuda::core {

__host__ __device__ inline bool checked_mul_u64(uint64_t first, uint64_t second,
                                                uint64_t *result) {
  if (first != 0 && second > UINT64_MAX / first) {
    return false;
  }
  *result = first * second;
  return true;
}

__host__ __device__ inline bool checked_add_u64(uint64_t first, uint64_t second,
                                                uint64_t *result) {
  if (second > UINT64_MAX - first) {
    return false;
  }
  *result = first + second;
  return true;
}

template <typename T> __host__ __device__ inline T *pointer(uint64_t address) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(address));
}

template <typename T>
__host__ __device__ inline const T *const_pointer(uint64_t address) {
  return reinterpret_cast<const T *>(static_cast<uintptr_t>(address));
}

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

} // namespace ferrule::cuda::core
