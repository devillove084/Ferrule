#pragma once
// Explicit eager BF16 tensor boundaries, separate from the F32 primitives.
#include "core/device.cuh"
#include "core/bf16_rms_reduction.cuh"
namespace ferrule::cuda::core {
struct Bf16BoundaryArgs {
  uint32_t kind, rows, width, heads, rope_dim, table_positions;
  float epsilon;
  uint32_t reserved;
  uint64_t input, other, cosine, sine, positions, output, stream;
};
static_assert(sizeof(Bf16BoundaryArgs) == 88);
// Qwen3.5 offset affine stays F32 throughout, with only the final BF16 cast.
__global__ void bf16_offset_norm_tree(Bf16BoundaryArgs a, float mean_factor) {
  __shared__ float sums[512];
  const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
  const uint32_t lane = threadIdx.x, offset = threadIdx.y * blockDim.x;
  sums[offset + lane] = row < a.rows
      ? bf16_rms::lane_sum(a.input, a.width, row, lane, blockDim.x) : 0.0f;
  __syncthreads();
  for (uint32_t step = blockDim.x / 2; step; step /= 2) {
    if (lane < step)
      sums[offset + lane] = __fadd_rn(sums[offset + lane], sums[offset + lane + step]);
    __syncthreads();
  }
  if (row >= a.rows) return;
  const float variance = __fmul_rn(sums[offset], mean_factor);
  const float added = __fadd_rn(variance, a.epsilon);
  const float inverse = rsqrtf(added);
  const uint64_t base = uint64_t(row) * a.width;
  for (uint32_t d = lane; d < a.width; d += blockDim.x) {
    const float normalized = __fmul_rn(bf16_value(const_pointer<uint16_t>(a.input)[base + d]), inverse);
    const float affine = __fadd_rn(1.0f, bf16_value(const_pointer<uint16_t>(a.other)[d]));
    const float value = __fmul_rn(normalized, affine);
    pointer<uint16_t>(a.output)[base + d] = bf16_rne(value);
  }
}

// Wider rows retain the previous expanded-F32 norm tree rather than claiming
// parity with an unvalidated Torch inter-CTA reduction geometry.
__global__ void bf16_offset_norm_fallback(Bf16BoundaryArgs a) {
  const uint32_t row = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const uint64_t base = uint64_t(row) * a.width;
  float sum = 0.0f;
  for (uint32_t d = threadIdx.x; d < a.width; d += blockDim.x) {
    const float x = bf16_value(const_pointer<uint16_t>(a.input)[base + d]);
    const float square = x * x;
    sum += square;
  }
  __shared__ float warp_sums[8], inverse;
  for (uint32_t step = 16; step; step /= 2)
    sum += __shfl_down_sync(0xffffffffu, sum, step);
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < 8 ? warp_sums[lane] : 0.0f;
    for (uint32_t step = 16; step; step /= 2)
      sum += __shfl_down_sync(0xffffffffu, sum, step);
    if (lane == 0) inverse = rsqrtf(sum / a.width + a.epsilon);
  }
  __syncthreads();
  for (uint32_t d = threadIdx.x; d < a.width; d += blockDim.x) {
    const float affine = 1.0f + bf16_value(const_pointer<uint16_t>(a.other)[d]);
    const float value = bf16_value(const_pointer<uint16_t>(a.input)[base + d]) * inverse * affine;
    pointer<uint16_t>(a.output)[base + d] = bf16_rne(value);
  }
}

__global__ void bf16_boundary_kernel(Bf16BoundaryArgs a) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= uint64_t(a.rows) * a.width) return;
  const auto *x = const_pointer<uint16_t>(a.input);
  auto *y = pointer<uint16_t>(a.output);
  if (a.kind == 1) { pointer<float>(a.output)[i] = bf16_value(x[i]); return; }
  if (a.kind == 2) {
    const int32_t token = const_pointer<int32_t>(a.positions)[i / a.width];
    y[i] = x[uint64_t(token) * a.width + i % a.width]; return;
  }
  const float v = bf16_value(x[i]);
  if (a.kind == 3) {
    const uint64_t base = (i / a.width) * a.width;
    float sum = 0.0f;
    for (uint32_t j = 0; j < a.width; ++j) {
      const float t = bf16_value(x[base + j]); sum = __fadd_rn(sum, __fmul_rn(t, t));
    }
    const float normalized = bf16_round(v * rsqrtf(sum / a.width + a.epsilon));
    y[i] = bf16_rne(normalized * bf16_value(const_pointer<uint16_t>(a.other)[i % a.width]));
  } else if (a.kind == 4) {
    // HF SiLU is one tensor op, followed by a distinct BF16 multiply.
    y[i] = bf16_swiglu(x[i], const_pointer<uint16_t>(a.other)[i]);
  } else if (a.kind == 5) {
    y[i] = bf16_rne(v + bf16_value(const_pointer<uint16_t>(a.other)[i]));
  } else if (a.kind == 6) {
    const uint32_t d = i % a.width;
    if (d >= a.rope_dim) { y[i] = x[i]; return; }
    const uint32_t half = a.rope_dim / 2;
    const uint64_t base = i - d;
    const float rotated = d < half ? -bf16_value(x[base + d + half]) : bf16_value(x[base + d - half]);
    const uint32_t position = const_pointer<int32_t>(a.positions)[i / (uint64_t(a.heads) * a.width)];
    const uint64_t t = uint64_t(position) * half + d % half;
    const float c = bf16_value(const_pointer<uint16_t>(a.cosine)[t]);
    const float s = bf16_value(const_pointer<uint16_t>(a.sine)[t]);
    y[i] = bf16_rne(bf16_round(__fmul_rn(v, c)) + bf16_round(__fmul_rn(rotated, s)));
  }
}
}
extern "C" int32_t ferrule_core_bf16_boundary(const ferrule::cuda::core::Bf16BoundaryArgs *a) {
  if (!a || a->kind < 1 || a->kind > 7 || !a->rows || !a->width || !a->input ||
      !a->output) return 1;
  const uint64_t count = uint64_t(a->rows) * a->width;
  if (count > INT32_MAX) return 1;
  if (a->kind == 7) {
    // Integer extents are checked above; the remaining products fit uint64_t.
    const uint64_t bytes = count * 2, weights = uint64_t(a->width) * 2;
    if (a->reserved || !a->other || (a->input | a->other | a->output) % 2 ||
        !isfinite(a->epsilon) || a->epsilon <= 0.0f ||
        a->input > UINT64_MAX - bytes || a->other > UINT64_MAX - weights ||
        a->output > UINT64_MAX - bytes ||
        (a->output < a->input + bytes && a->input < a->output + bytes) ||
        (a->output < a->other + weights && a->other < a->output + bytes)) return 1;
    using namespace ferrule::cuda::core;
    if (a->width <= 4096) {
      const dim3 block = bf16_rms::block(a->rows, a->width);
      const float factor = static_cast<float>(1.0 / static_cast<double>(a->width));
      bf16_offset_norm_tree<<<(a->rows + block.y - 1) / block.y, block, 0,
          reinterpret_cast<cudaStream_t>(a->stream)>>>(*a, factor);
    } else {
      bf16_offset_norm_fallback<<<a->rows, 256, 0,
          reinterpret_cast<cudaStream_t>(a->stream)>>>(*a);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
  }
  ferrule::cuda::core::bf16_boundary_kernel<<<(count + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(a->stream)>>>(*a);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
