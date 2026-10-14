#pragma once
#include "core/device.cuh"

namespace ferrule::cuda::core::bf16_rms {
// Pinned Torch Reduce.cuh geometry for contiguous F32 last-axis mean. Up to
// width4096 no row needs inter-CTA or block-y reduction. Larger widths retain
// the existing scalar path rather than silently guessing the multi-CTA tree.
inline dim3 block(uint32_t rows, uint32_t width) {
  uint32_t x = 1, y = 1;
  const uint32_t columns = width >= 128 ? width / 4 : width;
  while (x < 512 && x * 2 <= columns) x *= 2;
  while (y < 512 && y * 2 <= rows) y *= 2;
  const uint32_t first_x = x < 32 ? x : 32;
  y = y < 512 / first_x ? y : 512 / first_x;
  x = x < 512 / y ? x : 512 / y;
  return dim3(x, y);
}

__device__ inline float square(uint64_t input, uint64_t index) {
  const float x = bf16_value(const_pointer<uint16_t>(input)[index]);
  return __fmul_rn(x, x);
}

__device__ inline float lane_sum(uint64_t input, uint32_t width,
                                      uint32_t row, uint32_t lane, uint32_t stride) {
  const uint64_t base = uint64_t(row) * width;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (width >= 128) {
    // Alignment is that of HF's materialized F32 squares, not BF16 storage.
    const uint32_t misalignment = base % 4;
    const uint32_t start = misalignment ? 4 - misalignment : 0;
    if (misalignment && lane >= misalignment && lane < 4)
      acc[0] = __fadd_rn(acc[0], square(input, base + lane - misalignment));
    const uint32_t remaining = width - start;
    for (uint32_t vector = lane; uint64_t(vector) * 4 + 3 < remaining; vector += stride) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j)
        acc[j] = __fadd_rn(acc[j], square(input, base + start + uint64_t(vector) * 4 + j));
    }
    const uint32_t tail = remaining - remaining % 4 + lane;
    if (tail < remaining)
      acc[0] = __fadd_rn(acc[0], square(input, base + start + tail));
  } else {
    for (uint32_t d = lane; d < width; d += 4 * stride) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j)
        if (d + j * stride < width)
          acc[j] = __fadd_rn(acc[j], square(input, base + d + j * stride));
    }
  }
  return __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
}

} // namespace ferrule::cuda::core::bf16_rms
