#include "abi.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
    defined(_M_IX86)
#include <immintrin.h>
#endif

namespace ferrule::cpu {

float bf16_to_f32(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
  float output = 0.0F;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

float bf16_round(const float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
    bits = ((bits >> 16u) | UINT32_C(0x0040)) << 16u;
  } else {
    const std::uint32_t bias = UINT32_C(0x7fff) + ((bits >> 16u) & 1u);
    bits = (bits + bias) & UINT32_C(0xffff0000);
  }
  float output = 0.0F;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

static std::uint16_t load_bf16_word(const std::uint8_t *weight,
                                    const std::size_t column) noexcept {
  return static_cast<std::uint16_t>(weight[column * 2]) |
         static_cast<std::uint16_t>(weight[column * 2 + 1]) << 8u;
}

#if (defined(__x86_64__) || defined(__i386__)) &&                              \
    (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) static float
dot_bf16_f32_avx2(const std::uint8_t *weight, const float *input,
                  const std::size_t width) noexcept {
  float accumulator = 0.0F;
  std::size_t column = 0;
  alignas(16) std::uint16_t loaded[8];
  alignas(32) float products[8];
  for (; column + 8 <= width; column += 8) {
    for (std::size_t offset = 0; offset < 8; ++offset) {
      loaded[offset] = load_bf16_word(weight, column + offset);
    }
    const __m128i words =
        _mm_load_si128(reinterpret_cast<const __m128i *>(loaded));
    __m256i bits = _mm256_cvtepu16_epi32(words);
    bits = _mm256_slli_epi32(bits, 16);
    const __m256 weights = _mm256_castsi256_ps(bits);
    const __m256 inputs = _mm256_loadu_ps(input + column);
    _mm256_store_ps(products, _mm256_mul_ps(weights, inputs));
    for (const float product : products) {
      accumulator += product;
    }
  }
  for (; column < width; ++column) {
    accumulator += bf16_to_f32(load_bf16_word(weight, column)) * input[column];
  }
  return accumulator;
}
#endif

float dot_bf16_f32(const std::uint8_t *weight, const float *input,
                   const std::size_t width) noexcept {
#if (defined(__x86_64__) || defined(__i386__)) &&                              \
    (defined(__GNUC__) || defined(__clang__))
  return dot_bf16_f32_avx2(weight, input, width);
#else
  float accumulator = 0.0F;
  for (std::size_t column = 0; column < width; ++column) {
    accumulator += bf16_to_f32(load_bf16_word(weight, column)) * input[column];
  }
  return accumulator;
#endif
}

std::int32_t linear_bf16(const std::uint8_t *weight, const float *input,
                         const float *bias, float *output,
                         const std::size_t rows, const std::size_t out_features,
                         const std::size_t in_features) noexcept {
  if (weight == nullptr || input == nullptr || output == nullptr || rows == 0 ||
      out_features == 0 || in_features == 0) {
    return 1;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < out_features; ++out) {
      const float dot = dot_bf16_f32(weight + out * in_features * 2,
                                     input + row * in_features, in_features);
      output[row * out_features + out] =
          dot + (bias == nullptr ? 0.0F : bias[out]);
    }
  }
  return 0;
}

} // namespace ferrule::cpu

extern "C" std::int32_t
ferrule_cpu_embedding_bf16(const std::uint8_t *weight,
                           const std::uint32_t *token_ids, float *output,
                           const std::size_t rows, const std::size_t vocabulary,
                           const std::size_t width) {
  if (weight == nullptr || token_ids == nullptr || output == nullptr ||
      rows == 0 || vocabulary == 0 || width == 0) {
    return 1;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t token = token_ids[row];
    if (token >= vocabulary) {
      return 2;
    }
    for (std::size_t column = 0; column < width; ++column) {
      const std::size_t element = token * width + column;
      const std::uint16_t word =
          static_cast<std::uint16_t>(weight[element * 2]) |
          static_cast<std::uint16_t>(weight[element * 2 + 1]) << 8u;
      output[row * width + column] = ferrule::cpu::bf16_to_f32(word);
    }
  }
  return 0;
}

extern "C" std::int32_t
ferrule_cpu_linear_bf16(const std::uint8_t *weight, const float *input,
                        const float *bias, float *output,
                        const std::size_t rows, const std::size_t out_features,
                        const std::size_t in_features) {
  return ferrule::cpu::linear_bf16(weight, input, bias, output, rows,
                                   out_features, in_features);
}

extern "C" std::int32_t
ferrule_cpu_rms_norm(const float *input, const float *weight, float *output,
                     const std::size_t rows, const std::size_t width,
                     const float epsilon, const std::uint32_t bf16_boundary) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows == 0 ||
      width == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return 1;
  }
  const bool round = bf16_boundary != 0;
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      float value = input[row * width + column];
      if (round) {
        value = ferrule::cpu::bf16_round(value);
      }
      sum += value * value;
    }
    const float inverse_rms =
        1.0F / std::sqrt(sum / static_cast<float>(width) + epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      float value = input[row * width + column];
      if (round) {
        value = ferrule::cpu::bf16_round(value);
      }
      float normalized = value * inverse_rms;
      if (round) {
        normalized = ferrule::cpu::bf16_round(normalized);
      }
      float result = normalized * weight[column];
      output[row * width + column] =
          round ? ferrule::cpu::bf16_round(result) : result;
    }
  }
  return 0;
}
