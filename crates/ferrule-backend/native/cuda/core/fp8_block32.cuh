#pragma once

#include "core/abi.h"
#include "core/device.cuh"
#include "core/validation.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#ifndef FERRULE_CUDA_TARGET_SM
#define FERRULE_CUDA_TARGET_SM 0u
#endif

namespace ferrule::cuda::core {

inline bool fp8_block32_span(uint64_t address, uint64_t bytes,
                             uint64_t alignment) {
  uint64_t end;
  return address != 0 && bytes != 0 && alignment != 0 &&
         address % alignment == 0 && bytes <= INT64_MAX &&
         checked_add_u64(address, bytes, &end);
}

inline bool fp8_block32_overlap(uint64_t first, uint64_t first_bytes,
                                uint64_t second, uint64_t second_bytes) {
  uint64_t first_end, second_end;
  return checked_add_u64(first, first_bytes, &first_end) &&
         checked_add_u64(second, second_bytes, &second_end) && first < second_end &&
         second < first_end;
}

inline bool fp8_block32_decode_valid(
    const FerruleCoreFp8Block32DecodeArgs *args) {
  if (!args || !args->n || !args->k || !args->padded_k ||
      args->padded_k < args->k || args->padded_k % 8 ||
      !args->weight_padded_k || args->weight_padded_k < args->k ||
      !args->padded_n || args->padded_n < args->n ||
      !args->weight_row_stride_bytes ||
      args->weight_row_stride_bytes < args->weight_padded_k || !args->scale_rows ||
      !args->scale_cols || !args->scale_row_stride_bytes ||
      !args->scale_col_stride_bytes || args->reserved || !args->weight ||
      !args->scales || !args->decoded_weight_bf16 || !args->stream ||
      args->decoded_weight_bf16 % 16) {
    return false;
  }

  const uint64_t row_extent = uint64_t(args->row_origin % 32u) + args->padded_n;
  const uint64_t column_extent =
      uint64_t(args->column_origin % 32u) + args->weight_padded_k;
  const uint64_t expected_rows = (row_extent + 31u) / 32u;
  const uint64_t expected_cols = (column_extent + 31u) / 32u;
  uint64_t scale_column_span, scale_row_span, scale_bytes;
  if (expected_rows != args->scale_rows || expected_cols != args->scale_cols ||
      !checked_mul_u64(args->scale_cols - 1u, args->scale_col_stride_bytes,
                       &scale_column_span) ||
      !checked_add_u64(scale_column_span, 1u, &scale_column_span) ||
      scale_column_span > args->scale_row_stride_bytes ||
      !checked_mul_u64(args->scale_rows - 1u, args->scale_row_stride_bytes,
                       &scale_row_span) ||
      !checked_add_u64(scale_row_span, scale_column_span, &scale_bytes)) {
    return false;
  }

  uint64_t weight_row_span, weight_bytes, decoded_elements, decoded_bytes;
  if (!checked_mul_u64(args->padded_n - 1u, args->weight_row_stride_bytes,
                       &weight_row_span) ||
      !checked_add_u64(weight_row_span, args->weight_padded_k, &weight_bytes) ||
      !checked_mul_u64(args->n, args->padded_k, &decoded_elements) ||
      !checked_mul_u64(decoded_elements, sizeof(uint16_t), &decoded_bytes) ||
      decoded_elements > uint64_t(kBlock) * 65535u ||
      !fp8_block32_span(args->weight, weight_bytes, 1) ||
      !fp8_block32_span(args->scales, scale_bytes, 1) ||
      !fp8_block32_span(args->decoded_weight_bf16, decoded_bytes, 16) ||
      fp8_block32_overlap(args->decoded_weight_bf16, decoded_bytes, args->weight,
                          weight_bytes) ||
      fp8_block32_overlap(args->decoded_weight_bf16, decoded_bytes, args->scales,
                          scale_bytes)) {
    return false;
  }
  return true;
}

inline bool bf16_block32_pack_valid(const FerruleCoreBf16Block32PackArgs *args) {
  if (!args || !args->rows || !args->k || !args->padded_k ||
      args->padded_k < args->k || args->padded_k % 8 || args->reserved ||
      args->input_stride < args->k || !args->input_bf16 ||
      !args->output_bf16 || !args->stream || args->input_bf16 % 2 ||
      args->output_bf16 % 16) {
    return false;
  }
  uint64_t input_row_span, input_elements, output_elements;
  uint64_t input_bytes, output_bytes;
  if (!checked_mul_u64(args->rows - 1u, args->input_stride, &input_row_span) ||
      !checked_add_u64(input_row_span, args->k, &input_elements) ||
      !checked_mul_u64(args->rows, args->padded_k, &output_elements) ||
      !checked_mul_u64(input_elements, sizeof(uint16_t), &input_bytes) ||
      !checked_mul_u64(output_elements, sizeof(uint16_t), &output_bytes) ||
      output_elements > uint64_t(kBlock) * 65535u ||
      !fp8_block32_span(args->input_bf16, input_bytes, 2) ||
      !fp8_block32_span(args->output_bf16, output_bytes, 16) ||
      fp8_block32_overlap(args->input_bf16, input_bytes, args->output_bf16,
                          output_bytes)) {
    return false;
  }
  return true;
}

__global__ void fp8_block32_decode_kernel(FerruleCoreFp8Block32DecodeArgs args) {
  const uint64_t index =
      uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = uint64_t(args.n) * args.padded_k;
  if (index >= total) {
    return;
  }

  const uint32_t row = static_cast<uint32_t>(index / args.padded_k);
  const uint32_t column = static_cast<uint32_t>(index % args.padded_k);
  uint16_t result = 0;
  if (column < args.k) {
    const uint64_t absolute_row = uint64_t(args.row_origin) + row;
    const uint64_t absolute_column = uint64_t(args.column_origin) + column;
    const uint64_t block_row = absolute_row / 32u - args.row_origin / 32u;
    const uint64_t block_column = absolute_column / 32u - args.column_origin / 32u;
    const uint64_t scale_index =
        block_row * args.scale_row_stride_bytes +
        block_column * args.scale_col_stride_bytes;
    const uint8_t code = const_pointer<uint8_t>(args.weight)[
        uint64_t(row) * args.weight_row_stride_bytes + column];
    const uint8_t scale_code = const_pointer<uint8_t>(args.scales)[scale_index];
    const float dequantized =
        __fmul_rn(fp8_value(code), e8m0_scale(scale_code));
    result = bf16_rne(dequantized);
  }
  pointer<uint16_t>(args.decoded_weight_bf16)[index] = result;
}

__global__ void bf16_block32_pack_kernel(FerruleCoreBf16Block32PackArgs args) {
  const uint64_t index =
      uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = uint64_t(args.rows) * args.padded_k;
  if (index >= total) {
    return;
  }
  const uint32_t column = static_cast<uint32_t>(index % args.padded_k);
  pointer<uint16_t>(args.output_bf16)[index] =
      column < args.k
          ? const_pointer<uint16_t>(args.input_bf16)[
                uint64_t(index / args.padded_k) * args.input_stride + column]
          : 0;
}

} // namespace ferrule::cuda::core
