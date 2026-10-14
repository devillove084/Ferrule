#pragma once

#include "core/abi.h"
#include "core/device.cuh"
#include "core/validation.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace ferrule::cuda::core {

// All three spans are packed four-byte words. Check cross-dtype overlap as well
// as address overflow before a submission; this also protects direct ABI users.
inline bool full_router_valid(const FerruleCoreRouterArgs &a) {
  if (a.kind != FERRULE_CORE_ROUTER_FULL_SOFTMAX_TOPK || !a.rows ||
      a.rows > INT32_MAX || !a.columns || a.columns > 1024 || !a.k ||
      a.k > a.columns || a.k > 32 || !std::isfinite(a.route_scale) ||
      a.flags || a.reserved || a.bias || a.token_ids || a.hash_table ||
      a.hash_rows || a.hash_columns)
    return false;
  const uint64_t ptrs[] = {a.logits, a.indices, a.weights, a.probabilities};
  const uint64_t sizes[] = {uint64_t(a.rows) * a.columns * 4,
                           uint64_t(a.rows) * a.k * 4,
                           uint64_t(a.rows) * a.k * 4,
                           uint64_t(a.rows) * a.columns * 4};
  for (int i = 0; i < (a.probabilities ? 4 : 3); ++i) {
    if (!ptrs[i] || ptrs[i] % 4 || ptrs[i] > UINT64_MAX - sizes[i])
      return false;
    for (int j = 0; j < i; ++j)
      if (ptrs[i] < ptrs[j] + sizes[j] && ptrs[j] < ptrs[i] + sizes[i])
        return false;
  }
  return true;
}

// The trace specialization is instantiated only by the standalone test probe.
// Layout per row: max, exp-sum, exponentials[N], probabilities[N], selected[K],
// selected-sum. Production instantiates Trace=false, with no extra device buffer.
template <int Log2, bool Trace = false, bool ProbabilitySink = false>
__global__ void full_router_kernel(FerruleCoreRouterArgs a, float *trace = nullptr) {
  constexpr int Width = (1 << Log2) < 32 ? (1 << Log2) : 32;
  constexpr int Iterations = (1 << Log2) / Width;
  const int lane = threadIdx.x;
  const int x = lane % Width;
  const uint64_t row = blockIdx.x;
  const float *raw = const_pointer<float>(a.logits) + row * a.columns;
  int32_t *ids = pointer<int32_t>(a.indices) + row * a.k;
  float *weights = pointer<float>(a.weights) + row * a.k;
  __shared__ float probabilities[1 << Log2];
  __shared__ float selected[32];
  __shared__ float values[32];
  float elements[Iterations];
#pragma unroll
  for (int it = 0; it < Iterations; ++it) {
    int i = x + it * Width;
    elements[it] = i < a.columns ? raw[i] : -CUDART_INF_F;
  }
  // Source-equivalent PersistentSoftmax local max, XOR max/sum reductions.
  float maximum = elements[0];
#pragma unroll
  for (int it = 0; it < Iterations; ++it)
    maximum = maximum > elements[it] ? maximum : elements[it];
#pragma unroll
  for (int offset = Width / 2; offset; offset /= 2) {
    float other = __shfl_xor_sync(0xffffffff, maximum, offset, Width);
    maximum = maximum < other ? other : maximum;
  }
  float sum = 0.0f;
#pragma unroll
  for (int it = 0; it < Iterations; ++it) {
    elements[it] = std::exp(elements[it] - maximum);
    sum += elements[it];
  }
#pragma unroll
  for (int offset = Width / 2; offset; offset /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, offset, Width);
  float *observed = nullptr;
  if constexpr (Trace) {
    observed = trace + row * (2 * a.columns + a.k + 3);
    if (!lane) {
      observed[0] = maximum;
      observed[1] = sum;
    }
  }
#pragma unroll
  for (int it = 0; it < Iterations; ++it) {
    int i = x + it * Width;
    if (lane < Width && i < a.columns) {
      float p = sum == 0.0f ? CUDART_NAN_F : elements[it] / sum;
      probabilities[i] = p;
      if constexpr (ProbabilitySink)
        pointer<float>(a.probabilities)[row * a.columns + i] = p;
      if constexpr (Trace) {
        observed[2 + i] = elements[it];
        observed[2 + a.columns + i] = p;
      }
    }
  }
  // Raw-logit insertion order deliberately does NOT inspect probabilities.
  if (!lane) {
    for (uint32_t rank = 0; rank < a.k; ++rank) {
      ids[rank] = -1;
      values[rank] = -CUDART_INF_F;
    }
    for (uint32_t expert = 0; expert < a.columns; ++expert) {
      float value = raw[expert];
      if (isnan(value)) continue;
      uint32_t position = a.k;
      while (position > 0) {
        uint32_t prev = position - 1;
        if (value < values[prev] ||
            (value == values[prev] && ids[prev] >= 0 &&
             expert >= static_cast<uint32_t>(ids[prev]))) break;
        --position;
      }
      if (position < a.k) {
        for (uint32_t rank = a.k - 1; rank > position; --rank) {
          ids[rank] = ids[rank - 1];
          values[rank] = values[rank - 1];
        }
        ids[position] = expert;
        values[position] = value;
      }
    }
  }
  __syncwarp();
  selected[lane] = lane < a.k && ids[lane] >= 0 ? probabilities[ids[lane]] : 0.0f;
  if constexpr (Trace) {
    if (lane < a.k) observed[2 + 2 * a.columns + lane] = selected[lane];
  }
  __syncwarp();
  // Reduce.cuh contiguous F32 sum, k<=32: last_pow2(k) lanes, four
  // zero-initialized thread accumulators, then descending shuffle-downs.
  int reduce_width = 1;
  while (reduce_width * 2 <= a.k) reduce_width *= 2;
  float denominator = 0.0f;
  if (lane < reduce_width) {
    float v0 = 0.0f + selected[lane];
    float v1 = lane + reduce_width < a.k ? 0.0f + selected[lane + reduce_width] : 0.0f;
    denominator = ((v0 + v1) + 0.0f) + 0.0f;
  }
  for (int offset = reduce_width / 2; offset; offset /= 2)
    denominator += __shfl_down_sync(0xffffffff, denominator, offset);
  denominator = __shfl_sync(0xffffffff, denominator, 0);
  if constexpr (Trace) {
    if (!lane) observed[2 + 2 * a.columns + a.k] = denominator;
  }
  if (lane < a.k)
    weights[lane] = ids[lane] < 0 ? 0.0f : (selected[lane] / denominator) * a.route_scale;
}

template <bool Trace = false, bool ProbabilitySink = false>
inline void launch_full_router(const FerruleCoreRouterArgs &a, float *trace = nullptr) {
#define FERRULE_ROUTER_CASE(L) case L: full_router_kernel<L, Trace, ProbabilitySink><<<a.rows, 32, 0, stream(a.stream)>>>(a, trace); break
  int log2 = 0;
  while ((1u << log2) < a.columns) ++log2;
  switch (log2) {
    FERRULE_ROUTER_CASE(0); FERRULE_ROUTER_CASE(1); FERRULE_ROUTER_CASE(2);
    FERRULE_ROUTER_CASE(3); FERRULE_ROUTER_CASE(4); FERRULE_ROUTER_CASE(5);
    FERRULE_ROUTER_CASE(6); FERRULE_ROUTER_CASE(7); FERRULE_ROUTER_CASE(8);
    FERRULE_ROUTER_CASE(9); FERRULE_ROUTER_CASE(10);
  }
#undef FERRULE_ROUTER_CASE
}
} // namespace ferrule::cuda::core
