#pragma once

// Original Ferrule implementation; no Marlin code or repacked format copied.
// See tests/numeric_w8a16/README.md for local vLLM provenance and differences.
// Included after bf16_linear.cuh in the existing core translation unit.
#if FERRULE_CUDA_TARGET_SM >= 80
#include <cuda_bf16.h>
#include <mma.h>
#endif

namespace ferrule::cuda::cutlass::numeric_fp8 {
// Operand tiles share their storage with the epilogue. No global temporary,
// split-K semaphore, atomics, or materialized full-precision weight matrix.
#if FERRULE_CUDA_TARGET_SM >= 80
union alignas(32) W8A16Shared {
  struct {
    uint16_t activation[16 * 32];
    uint16_t weight[32 * 32];
  } operands;
  float output[16 * 32];
};
static_assert(sizeof(W8A16Shared) == 3072);

// A warp owns one N column and up to four real M rows. BF16 rounding is
// explicit even on the SIMT path: these are BF16 operands with F32 FMA and
// reduction, not the F32 precision profile. Reuse decoded weights across rows.
template <bool Bf16Input, bool F32Scale, bool Packed, unsigned Rows>
__global__ void fused_w8a16_vector(FerruleCutlassNumericFp8Args a) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x / 32u;
  const uint64_t ntiles = (uint64_t(a.n) + 3u) / 4u;
  const uint64_t tiles = ((uint64_t(a.m) + Rows - 1u) / Rows) * ntiles;
  for (uint64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const uint32_t n = uint32_t(tile % ntiles) * 4u + warp;
    const uint32_t first_m = uint32_t(tile / ntiles) * Rows;
    if (n >= a.n) continue;
    const uint64_t weight_base = uint64_t(n) * a.k;
    const uint64_t scale_base = uint64_t((a.row_origin + n) / 128u) * a.scale_cols;
    float sums[Rows][4] = {};
    constexpr uint32_t width = Packed ? 4u : 1u;
    for (uint32_t k = lane * width; k < a.k; k += 32u * width) {
      uint32_t codes;
      if constexpr (Packed) codes = *const_pointer<uint32_t>(a.weight + weight_base + k);
      else codes = const_pointer<uint8_t>(a.weight)[weight_base + k];
#pragma unroll
      for (uint32_t j = 0; j < width; ++j) {
        const uint64_t si = scale_base + (a.column_origin + k + j) / 128u;
        const float scale = F32Scale ? const_pointer<float>(a.scales)[si]
            : bf16_value(const_pointer<uint16_t>(a.scales)[si]);
        const float w = bf16_value(bf16_rne(__fmul_rn(
            fp8_value(uint8_t(codes >> (8u * j))), scale)));
#pragma unroll
        for (uint32_t r = 0; r < Rows; ++r) {
          if (first_m + r < a.m) {
            const uint64_t index = uint64_t(first_m + r) * a.lda + k + j;
            const float x = Bf16Input ? bf16_value(const_pointer<uint16_t>(a.activation)[index])
                : bf16_value(bf16_rne(const_pointer<float>(a.activation)[index]));
            sums[r][j] = __fmaf_rn(x, w, sums[r][j]);
          }
        }
      }
    }
#pragma unroll
    for (uint32_t r = 0; r < Rows; ++r) {
      float sum = __fadd_rn(__fadd_rn(sums[r][0], sums[r][1]),
                            __fadd_rn(sums[r][2], sums[r][3]));
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset));
      if (lane == 0 && first_m + r < a.m)
        pointer<float>(a.output)[uint64_t(first_m + r) * a.ldd + n] = sum;
    }
  }
}

template <bool Bf16Input, bool F32Scale, unsigned Rows>
inline void w8a16_vector_launch(const FerruleCutlassNumericFp8Args &a, cudaStream_t stream) {
  const uint64_t tiles = ((uint64_t(a.m) + Rows - 1u) / Rows) * ((uint64_t(a.n) + 3u) / 4u);
  const unsigned grid = unsigned(tiles < 65535u ? tiles : 65535u);
  if (a.k % 4u == 0 && a.weight % 4u == 0)
    fused_w8a16_vector<Bf16Input, F32Scale, true, Rows><<<grid, 128, 0, stream>>>(a);
  else
    fused_w8a16_vector<Bf16Input, F32Scale, false, Rows><<<grid, 128, 0, stream>>>(a);
}

// Eight warps cooperatively decode operands; two execute WMMA. Keeping the
// 16x32x32 tile avoids excessive shared memory, while the extra loaders break
// the old two-warp staging bottleneck. All eight warps participate in barriers.
// Bound the Tensor Core accumulation chain independently of K. Each 128-K
// partial enters a binary carry tree; only equally sized adjacent ranges merge.
// Coalesced, thread-private shared slots avoid a K-sized register live set.
// At most 22 levels fit below 48 KiB/CTA; the last three levels for the largest
// legal K stay in registers. Neither path spills a full partial matrix.
constexpr uint32_t W8A16_WMMA_SEGMENT_K = 128u;

template <unsigned Levels, typename Fragment>
struct W8A16Reduction {
  static constexpr unsigned SharedLevels = Levels < 22u ? Levels : 22u;
  static constexpr unsigned RegisterLevels = Levels > SharedLevels ? Levels - SharedLevels : 1u;
  Fragment upper[RegisterLevels];
  float *shared;

  __device__ __forceinline__ float &element(unsigned level, int i) {
    if constexpr (Levels > SharedLevels) {
      static_assert(Levels == 25u);
      if (level == 22u) return upper[0].x[i];
      if (level == 23u) return upper[1].x[i];
      if (level == 24u) return upper[2].x[i];
    }
    return shared[(level * Fragment::num_elements + i) * 64u + threadIdx.x];
  }

  __device__ __forceinline__ void push(Fragment &partial, uint32_t count) {
    // The first set bit of the new count is the vacant destination level.
    const unsigned destination = __ffs(count) - 1;
    for (unsigned level = 0; level < destination; ++level) {
#pragma unroll
      for (int i = 0; i < Fragment::num_elements; ++i)
        partial.x[i] = __fadd_rn(element(level, i), partial.x[i]);
    }
#pragma unroll
    for (int i = 0; i < Fragment::num_elements; ++i)
      element(destination, i) = partial.x[i];
  }

  __device__ __forceinline__ void finish(Fragment &sum, uint32_t count) {
    bool have_sum = false;
    // Fold the trailing subtrees first, preserving logarithmic merge depth
    // for non-power-of-two segment counts without adding artificial zeros.
    for (unsigned level = 0; level < Levels; ++level) {
      if (count & (1u << level)) {
#pragma unroll
        for (int i = 0; i < Fragment::num_elements; ++i)
          sum.x[i] = have_sum ? __fadd_rn(element(level, i), sum.x[i])
                              : element(level, i);
        have_sum = true;
      }
    }
  }
};

template <bool Bf16Input, bool F32Scale, bool Packed, unsigned Levels>
__global__ void fused_w8a16_wmma(FerruleCutlassNumericFp8Args a) {
  namespace wmma = nvcuda::wmma;
  using Accumulator = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
  using Reduction = W8A16Reduction<Levels, Accumulator>;
  __shared__ W8A16Shared shared;
  __shared__ float partials[Reduction::SharedLevels * Accumulator::num_elements * 64u];
  const uint32_t warp = threadIdx.x / 32u;
  const uint64_t ntiles = (uint64_t(a.n) + 31u) / 32u;
  const uint64_t tiles = ((uint64_t(a.m) + 15u) / 16u) * ntiles;
  for (uint64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const uint32_t first_m = uint32_t(tile / ntiles) * 16u;
    const uint32_t first_n = uint32_t(tile % ntiles) * 32u;
    Accumulator acc;
    Reduction tree;
    tree.shared = partials;
    wmma::fill_fragment(acc, 0.0f);
    for (uint32_t first_k = 0; first_k < a.k; first_k += 32u) {
      for (uint32_t i = threadIdx.x; i < 16u * 32u; i += 256u) {
        const uint32_t m = first_m + i / 32u;
        const uint32_t k = first_k + i % 32u;
        uint16_t value = 0;
        if (m < a.m && k < a.k) {
          const uint64_t index = uint64_t(m) * a.lda + k;
          if constexpr (Bf16Input) value = const_pointer<uint16_t>(a.activation)[index];
          else value = bf16_rne(const_pointer<float>(a.activation)[index]);
        }
        shared.operands.activation[i] = value;
      }
      const uint32_t i = threadIdx.x * 4u;
      const uint32_t n = first_n + i / 32u;
      const uint32_t k = first_k + i % 32u;
      uint32_t codes = 0;
      if constexpr (Packed) {
        if (n < a.n && k < a.k)
          codes = *const_pointer<uint32_t>(a.weight + uint64_t(n) * a.k + k);
      }
#pragma unroll
      for (uint32_t j = 0; j < 4u; ++j) {
        uint16_t value = 0;
        if (n < a.n && k + j < a.k) {
          const uint64_t si = uint64_t((a.row_origin + n) / 128u) * a.scale_cols
                              + (a.column_origin + k + j) / 128u;
          const float scale = F32Scale ? const_pointer<float>(a.scales)[si]
              : bf16_value(const_pointer<uint16_t>(a.scales)[si]);
          const uint8_t code = Packed ? uint8_t(codes >> (8u * j))
              : const_pointer<uint8_t>(a.weight)[uint64_t(n) * a.k + k + j];
          value = bf16_rne(__fmul_rn(fp8_value(code), scale));
        }
        shared.operands.weight[i + j] = value;
      }
      __syncthreads();
      if (warp < 2u) {
#pragma unroll
        for (uint32_t kk = 0; kk < 32u; kk += 16u) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
          wmma::load_matrix_sync(af,
              reinterpret_cast<const __nv_bfloat16 *>(shared.operands.activation) + kk, 32);
          wmma::load_matrix_sync(bf,
              reinterpret_cast<const __nv_bfloat16 *>(shared.operands.weight) + warp * 16u * 32u + kk, 32);
          wmma::mma_sync(acc, af, bf, acc);
        }
      }
      // Both warps must finish reading before the next tile/epilogue overwrite.
      __syncthreads();
      if (warp < 2u && ((first_k + 32u) % W8A16_WMMA_SEGMENT_K == 0u ||
                        first_k + 32u >= a.k)) {
        const uint32_t count = first_k / W8A16_WMMA_SEGMENT_K + 1u;
        tree.push(acc, count);
        wmma::fill_fragment(acc, 0.0f);
      }
    }
    if (warp < 2u) {
      const uint32_t count = (a.k + W8A16_WMMA_SEGMENT_K - 1u) / W8A16_WMMA_SEGMENT_K;
      tree.finish(acc, count);
      wmma::store_matrix_sync(shared.output + warp * 16u, acc, 32, wmma::mem_row_major);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < 16u * 32u; i += 256u) {
      const uint32_t m = first_m + i / 32u;
      const uint32_t n = first_n + i % 32u;
      if (m < a.m && n < a.n)
        pointer<float>(a.output)[uint64_t(m) * a.ldd + n] = shared.output[i];
    }
    __syncthreads();
  }
}

template <bool Bf16Input, bool F32Scale>
inline void w8a16_dispatch(const FerruleCutlassNumericFp8Args &a, cudaStream_t stream) {
  if (a.m <= 8) {
    if (a.m == 1) w8a16_vector_launch<Bf16Input, F32Scale, 1>(a, stream);
    else if (a.m == 2) w8a16_vector_launch<Bf16Input, F32Scale, 2>(a, stream);
    else w8a16_vector_launch<Bf16Input, F32Scale, 4>(a, stream);
  } else {
    const uint64_t tiles = ((uint64_t(a.m) + 15u) / 16u) * ((uint64_t(a.n) + 31u) / 32u);
    const unsigned grid = unsigned(tiles < 65535u ? tiles : 65535u);
    if (a.k <= 2048u) {
      if (a.k % 4u == 0 && a.weight % 4u == 0)
        fused_w8a16_wmma<Bf16Input, F32Scale, true, 5><<<grid, 256, 0, stream>>>(a);
      else
        fused_w8a16_wmma<Bf16Input, F32Scale, false, 5><<<grid, 256, 0, stream>>>(a);
    } else if (a.k <= 16384u) {
      if (a.k % 4u == 0 && a.weight % 4u == 0)
        fused_w8a16_wmma<Bf16Input, F32Scale, true, 8><<<grid, 256, 0, stream>>>(a);
      else
        fused_w8a16_wmma<Bf16Input, F32Scale, false, 8><<<grid, 256, 0, stream>>>(a);
    } else {
      if (a.k % 4u == 0 && a.weight % 4u == 0)
        fused_w8a16_wmma<Bf16Input, F32Scale, true, 25><<<grid, 256, 0, stream>>>(a);
      else
        fused_w8a16_wmma<Bf16Input, F32Scale, false, 25><<<grid, 256, 0, stream>>>(a);
    }
  }
}
#endif

inline int w8a16_validate(const FerruleCutlassNumericFp8Args *a, uint32_t element) {
  if (!a || !a->m || a->m > INT_MAX - 64u || !a->n || a->n > INT_MAX - 128u ||
      !a->k || a->k > INT_MAX - 32u || a->padded_k != a->k ||
      a->tile_rows || a->workspace_bytes || a->reserved ||
      a->row_origin >= 128u || a->column_origin >= 128u ||
      a->scale_cols != (a->column_origin + a->k + 127u) / 128u ||
      (a->scale_bytes != 2 && a->scale_bytes != 4) ||
      a->lda < a->k || a->lda > INT_MAX || a->ldd < a->n || a->ldd > INT_MAX ||
      a->activation % element || a->output % 4 || a->scales % a->scale_bytes ||
      !m1_span(a->activation, (uint64_t(a->m - 1u) * a->lda + a->k) * element) ||
      !m1_span(a->output, (uint64_t(a->m - 1u) * a->ldd + a->n) * 4) ||
      !m1_span(a->weight, uint64_t(a->n) * a->k) ||
      !m1_span(a->scales, uint64_t((a->row_origin + a->n + 127u) / 128u) *
          a->scale_cols * a->scale_bytes)) return 1;
  // Rust also validates ownership, actual buffer lengths and retained scratch.
  // Native preflight protects the output against all read spans independently.
  const uint64_t dend = a->output + (uint64_t(a->m - 1u) * a->ldd + a->n) * 4;
  const uint64_t starts[] = {a->activation, a->weight, a->scales};
  const uint64_t sizes[] = {(uint64_t(a->m - 1u) * a->lda + a->k) * element,
      uint64_t(a->n) * a->k,
      uint64_t((a->row_origin + a->n + 127u) / 128u) * a->scale_cols * a->scale_bytes};
  for (int i = 0; i < 3; ++i)
    if (a->output < starts[i] + sizes[i] && starts[i] < dend) return 1;
  return FERRULE_CUDA_TARGET_SM >= 80 ? 0 : 2;
}

template <bool Bf16Input>
inline int w8a16_launch(const FerruleCutlassNumericFp8Args *a) {
  if (int status = w8a16_validate(a, Bf16Input ? 2u : 4u)) return status;
#if FERRULE_CUDA_TARGET_SM >= 80
  auto stream = reinterpret_cast<cudaStream_t>(a->stream);
  if (a->scale_bytes == 4) w8a16_dispatch<Bf16Input, true>(*a, stream);
  else w8a16_dispatch<Bf16Input, false>(*a, stream);
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
#else
  return 2;
#endif
}
} // namespace ferrule::cuda::cutlass::numeric_fp8

extern "C" int32_t ferrule_cutlass_numeric_fp8_w8a16_available() {
  return FERRULE_CUDA_TARGET_SM >= 80;
}
extern "C" int32_t ferrule_cutlass_numeric_fp8_w8a16_can_implement(const FerruleCutlassNumericFp8Args *a) {
  return ferrule::cuda::cutlass::numeric_fp8::w8a16_validate(a, 4);
}
extern "C" int32_t ferrule_cutlass_numeric_fp8_w8a16_bf16_can_implement(const FerruleCutlassNumericFp8Args *a) {
  return ferrule::cuda::cutlass::numeric_fp8::w8a16_validate(a, 2);
}
extern "C" int32_t ferrule_cutlass_numeric_fp8_w8a16_launch(const FerruleCutlassNumericFp8Args *a) {
  return ferrule::cuda::cutlass::numeric_fp8::w8a16_launch<false>(a);
}
extern "C" int32_t ferrule_cutlass_numeric_fp8_w8a16_bf16_launch(const FerruleCutlassNumericFp8Args *a) {
  return ferrule::cuda::cutlass::numeric_fp8::w8a16_launch<true>(a);
}
