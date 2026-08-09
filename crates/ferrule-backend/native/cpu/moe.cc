#include "abi.h"

#include <algorithm>
#include <cmath>
#include <vector>

extern "C" std::int32_t ferrule_cpu_swiglu_bf16(
    const std::uint8_t *gate_weight, const std::uint8_t *up_weight,
    const std::uint8_t *down_weight, const float *gate_bias,
    const float *up_bias, const float *down_bias, const float *input,
    float *output, const std::size_t rows, const std::size_t input_width,
    const std::size_t intermediate_width, const std::size_t output_width,
    const float activation_limit, const std::uint32_t bf16_boundary) {
  if (gate_weight == nullptr || up_weight == nullptr ||
      down_weight == nullptr || input == nullptr || output == nullptr ||
      rows == 0 || input_width == 0 || intermediate_width == 0 ||
      output_width == 0 || !std::isfinite(activation_limit) ||
      activation_limit < 0.0F) {
    return 1;
  }
  const bool round = bf16_boundary != 0;
  std::vector<float> prepared(rows * input_width);
  std::vector<float> gate(rows * intermediate_width);
  std::vector<float> up(rows * intermediate_width);
  std::vector<float> hidden(rows * intermediate_width);
  for (std::size_t index = 0; index < prepared.size(); ++index) {
    prepared[index] =
        round ? ferrule::cpu::bf16_round(input[index]) : input[index];
  }
  if (ferrule::cpu::linear_bf16(gate_weight, prepared.data(), gate_bias,
                                gate.data(), rows, intermediate_width,
                                input_width) != 0 ||
      ferrule::cpu::linear_bf16(up_weight, prepared.data(), up_bias, up.data(),
                                rows, intermediate_width, input_width) != 0) {
    return 2;
  }
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    float gate_value =
        round ? ferrule::cpu::bf16_round(gate[index]) : gate[index];
    float up_value = round ? ferrule::cpu::bf16_round(up[index]) : up[index];
    if (activation_limit > 0.0F) {
      gate_value = std::min(gate_value, activation_limit);
      up_value = std::clamp(up_value, -activation_limit, activation_limit);
    }
    float activated = gate_value / (1.0F + std::exp(-gate_value));
    if (round) {
      activated = ferrule::cpu::bf16_round(activated);
    }
    const float product = activated * up_value;
    hidden[index] = round ? ferrule::cpu::bf16_round(product) : product;
  }
  if (ferrule::cpu::linear_bf16(down_weight, hidden.data(), down_bias, output,
                                rows, output_width, intermediate_width) != 0) {
    return 3;
  }
  if (round) {
    for (std::size_t index = 0; index < rows * output_width; ++index) {
      output[index] = ferrule::cpu::bf16_round(output[index]);
    }
  }
  return 0;
}
