#ifndef FERRULE_NATIVE_CPU_ABI_H_
#define FERRULE_NATIVE_CPU_ABI_H_

#include <cstddef>
#include <cstdint>

#define FERRULE_CPU_CAP_AVX2 (UINT64_C(1) << 0)
#define FERRULE_CPU_CAP_AVX512_BF16 (UINT64_C(1) << 1)

extern "C" {
std::uint64_t ferrule_cpu_capabilities();

std::int32_t ferrule_cpu_embedding_bf16(const std::uint8_t *weight,
                                        const std::uint32_t *token_ids,
                                        float *output, std::size_t rows,
                                        std::size_t vocabulary,
                                        std::size_t width);

std::int32_t ferrule_cpu_linear_bf16(const std::uint8_t *weight,
                                     const float *input, const float *bias,
                                     float *output, std::size_t rows,
                                     std::size_t out_features,
                                     std::size_t in_features);

std::int32_t ferrule_cpu_rms_norm(const float *input, const float *weight,
                                  float *output, std::size_t rows,
                                  std::size_t width, float epsilon,
                                  std::uint32_t bf16_boundary);

std::int32_t ferrule_cpu_swiglu_bf16(
    const std::uint8_t *gate_weight, const std::uint8_t *up_weight,
    const std::uint8_t *down_weight, const float *gate_bias,
    const float *up_bias, const float *down_bias, const float *input,
    float *output, std::size_t rows, std::size_t input_width,
    std::size_t intermediate_width, std::size_t output_width,
    float activation_limit, std::uint32_t bf16_boundary);
}

namespace ferrule::cpu {
float bf16_to_f32(std::uint16_t value) noexcept;
float bf16_round(float value) noexcept;
float dot_bf16_f32(const std::uint8_t *weight, const float *input,
                   std::size_t width) noexcept;
std::int32_t linear_bf16(const std::uint8_t *weight, const float *input,
                         const float *bias, float *output, std::size_t rows,
                         std::size_t out_features,
                         std::size_t in_features) noexcept;
} // namespace ferrule::cpu

#endif
