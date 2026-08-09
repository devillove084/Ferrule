#pragma once

#include "core/device.cuh"
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace ferrule::cuda::core {

inline cudaStream_t stream(uint64_t address) {
  return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(address));
}

inline int32_t launch_status() {
  return static_cast<int32_t>(cudaPeekAtLastError());
}

inline bool valid(const void *args) { return args != nullptr; }

constexpr uint32_t kBlock = 256;

inline uint32_t blocks_for(uint64_t count) {
  const uint64_t blocks = (count + kBlock - 1) / kBlock;
  return static_cast<uint32_t>(blocks == 0 ? 1 : blocks);
}

inline bool transformer_range(uint64_t address, uint64_t bytes, uint64_t rows,
                              uint64_t row_stride, uint64_t heads,
                              uint64_t head_stride, uint64_t width,
                              uint64_t alignment) {
  if (address == 0 || bytes == 0 || rows == 0 || heads == 0 || width == 0 ||
      alignment == 0 || address % alignment != 0 ||
      row_stride % alignment != 0 || head_stride % alignment != 0 ||
      head_stride < width) {
    return false;
  }
  const uint64_t last_row = rows - 1;
  const uint64_t last_head = heads - 1;
  uint64_t row_extent;
  uint64_t required;
  uint64_t term;
  return checked_mul_u64(last_head, head_stride, &row_extent) &&
         checked_add_u64(row_extent, width, &row_extent) &&
         row_stride >= row_extent &&
         checked_mul_u64(last_row, row_stride, &required) &&
         checked_mul_u64(last_head, head_stride, &term) &&
         checked_add_u64(required, term, &required) &&
         checked_add_u64(required, width, &required) && required <= bytes;
}

inline bool transformer_cache_strides(
    uint64_t address, uint64_t capacity, uint64_t slot_stride,
    uint64_t layer_stride, uint64_t token_stride, uint64_t head_stride,
    uint64_t layers, uint64_t page_tokens, uint64_t heads, uint64_t head_dim) {
  if (address == 0 || capacity < slot_stride || address % 2 != 0 ||
      slot_stride % 2 != 0 || layer_stride % 2 != 0 || token_stride % 2 != 0 ||
      head_stride % 2 != 0 || heads == 0 || page_tokens == 0 || layers == 0) {
    return false;
  }
  uint64_t head_width;
  uint64_t last_head;
  uint64_t last_token;
  uint64_t last_layer;
  uint64_t required;
  if (!checked_mul_u64(head_dim, sizeof(uint16_t), &head_width) ||
      head_stride < head_width ||
      !checked_mul_u64(heads - 1, head_stride, &last_head) ||
      !checked_add_u64(last_head, head_width, &required) ||
      token_stride < required ||
      !checked_mul_u64(page_tokens - 1, token_stride, &last_token) ||
      !checked_add_u64(last_token, last_head, &required) ||
      !checked_add_u64(required, head_width, &required) ||
      layer_stride < required ||
      !checked_mul_u64(layers - 1, layer_stride, &last_layer) ||
      !checked_add_u64(last_layer, last_token, &required) ||
      !checked_add_u64(required, last_head, &required) ||
      !checked_add_u64(required, head_width, &required) ||
      slot_stride < required) {
    return false;
  }
  return true;
}

} // namespace ferrule::cuda::core
