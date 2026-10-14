#pragma once
#include "core/device.cuh"
#include "core/recurrent_bf16_chunk_abi.h"
#include "core/validation.cuh"
#include <limits.h>

// Standalone tests may observe completed stages. The provider has no hook,
// copies, allocation, or synchronization in its submission path.
#ifndef FERRULE_GDN_CHUNK_OBSERVE
#define FERRULE_GDN_CHUNK_OBSERVE(name, data, elements) ((void)0)
#endif
namespace ferrule::cuda::core::bf16_chunk {
constexpr uint32_t C = 64;
constexpr uint32_t launches_per_chunk = 19;
// Six key panels, seven value panels, four square matrices, three gate rows,
// and one state product. Unused reserved panels permit stable explicit sizing.
struct Workspace {
  float *q, *k, *kb, *ke, *kc, *qe;
  float *v, *vb, *value, *vp, *vn, *inter, *intra;
  float *decay, *tri, *attn, *temporary;
  float *g, *prefix, *beta, *state_product;
  uint64_t nk, nv, nm, ng, ns;
  Workspace(const FerruleCoreBf16ChunkArgs &a) {
    nk = uint64_t(a.value_heads) * C * a.key_dim;
    nv = uint64_t(a.value_heads) * C * a.value_dim;
    nm = uint64_t(a.value_heads) * C * C;
    ng = uint64_t(a.value_heads) * C;
    ns = uint64_t(a.value_heads) * a.key_dim * a.value_dim;
    auto *p = pointer<float>(a.workspace_f32);
    q = p;
    k = q + nk;
    kb = k + nk;
    ke = kb + nk;
    kc = ke + nk;
    qe = kc + nk;
    v = qe + nk;
    vb = v + nv;
    value = vb + nv;
    vp = value + nv;
    vn = vp + nv;
    inter = vn + nv;
    intra = inter + nv;
    decay = intra + nv;
    tri = decay + nm;
    attn = tri + nm;
    temporary = attn + nm;
    g = temporary + nm;
    prefix = g + ng;
    beta = prefix + ng;
    state_product = beta + ng;
  }
};
inline bool requirements(const FerruleCoreBf16ChunkArgs &a, uint64_t *n) {
  if (a.reserved || a.rows < 2 || !a.key_heads || !a.value_heads ||
      a.value_heads > 65535 || a.value_heads % a.key_heads || !a.key_dim ||
      a.key_dim > 512 || !a.value_dim || a.value_dim > 512)
    return false;
  // Dimensions above bound every intermediate below UINT64_MAX.
  const uint64_t width = 2 * uint64_t(a.key_heads) * a.key_dim +
                         uint64_t(a.value_heads) * a.value_dim;
  const uint64_t state = uint64_t(a.value_heads) * a.key_dim * a.value_dim;
  *n = uint64_t(a.value_heads) *
       (6 * C * a.key_dim + 7 * C * a.value_dim + 4 * C * C + 3 * C +
        uint64_t(a.key_dim) * a.value_dim);
  return width * a.rows <= INT32_MAX && state <= INT32_MAX &&
         *n <= INT32_MAX / 4;
}
struct Span {
  uint64_t begin, end;
};
inline bool span(uint64_t p, uint64_t n, uint64_t bytes, Span *s) {
  uint64_t size;
  if (!p || p % bytes || !checked_mul_u64(n, bytes, &size) ||
      !checked_add_u64(p, size, &s->end))
    return false;
  s->begin = p;
  return true;
}
inline bool validate(const FerruleCoreBf16ChunkArgs &a) {
  uint64_t n;
  if (!requirements(a, &n) || a.workspace_elements < n)
    return false;
  const uint64_t gates = uint64_t(a.rows) * a.value_heads;
  const uint64_t out = gates * a.value_dim;
  const uint64_t qkv =
      uint64_t(a.rows) * (2 * uint64_t(a.key_heads) * a.key_dim +
                          uint64_t(a.value_heads) * a.value_dim);
  Span s[8];
  if (!span(a.qkv_bf16, qkv, 2, &s[0]) || !span(a.a_bf16, gates, 2, &s[1]) ||
      !span(a.b_bf16, gates, 2, &s[2]) ||
      !span(a.a_log_bf16, a.value_heads, 2, &s[3]) ||
      !span(a.dt_bias_bf16, a.value_heads, 2, &s[4]) ||
      !span(a.state_f32, uint64_t(a.value_heads) * a.key_dim * a.value_dim, 4,
            &s[5]) ||
      !span(a.output_bf16, out, 2, &s[6]) ||
      !span(a.workspace_f32, a.workspace_elements, 4, &s[7]))
    return false;
  for (int w = 5; w < 8; ++w)
    for (int r = 0; r < w; ++r)
      if (s[w].begin < s[r].end && s[r].begin < s[w].end)
        return false;
  return true;
}
__device__ inline float load(uint64_t p, uint64_t i) {
  return bf16_value(const_pointer<uint16_t>(p)[i]);
}
__device__ inline float inv_l2(uint64_t p, uint64_t off, uint32_t d) {
  float sum = 0;
  for (uint32_t j = 0; j < d; ++j) {
    float x = load(p, off + j);
    sum = __fadd_rn(sum, bf16_round(__fmul_rn(x, x)));
  }
  return bf16_round(rsqrtf(bf16_round(__fadd_rn(bf16_round(sum), 1e-6f))));
}
// HF computes the Python scalar in double, then converts it to F32 for
// the tensor multiply. sqrtf followed by F32 division double-rounds at Dk511.
inline float query_scale(uint32_t key_dim) {
  return static_cast<float>(1.0 / sqrt(static_cast<double>(key_dim)));
}

// Torch ScanUtils uses the full [Hv,ceil(rows/64),64] tensor geometry even
// though this operator reuses one chunk of scratch. Match its uint32_t launch
// arithmetic and cap the scan window at the real row length. A 32-value window
// carries its total into the next window BEFORE that window's Sklansky scan.
inline uint32_t prefix_window(uint32_t rows, uint32_t heads) {
  const uint64_t scan_rows = uint64_t(heads) * ((uint64_t(rows) + C - 1) / C);
  uint32_t log_rows = 0;
  while ((uint64_t(1) << log_rows) < scan_rows)
    ++log_rows;
  uint32_t log_threads = (uint32_t(9) + (uint32_t(6) - log_rows)) / 2;
  if (log_threads < 4)
    log_threads = 4;
  if (log_threads > 9)
    log_threads = 9;
  const uint32_t window = 2 * (uint32_t(1) << log_threads);
  return window < C ? window : C;
}

// One thread owns a head/token. Padding is explicitly zero, including gates.
__global__ void prepare(FerruleCoreBf16ChunkArgs a, Workspace w, uint32_t start,
                        float scale) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w.ng)
    return;
  uint32_t h = i / C, t = i % C, global = start + t;
  float qn = 0, kn = 0, g = 0, beta = 0;
  uint64_t kw = uint64_t(a.key_heads) * a.key_dim,
           vw = uint64_t(a.value_heads) * a.value_dim;
  uint64_t q = uint64_t(global) * (2 * kw + vw) +
               uint64_t(h / (a.value_heads / a.key_heads)) * a.key_dim;
  if (global < a.rows) {
    qn = inv_l2(a.qkv_bf16, q, a.key_dim);
    kn = inv_l2(a.qkv_bf16, q + kw, a.key_dim);
    uint64_t gate = uint64_t(global) * a.value_heads + h;
    float x = __fadd_rn(load(a.a_bf16, gate), load(a.dt_bias_bf16, h));
    float softplus = x > 20.0f ? x : log1pf(expf(x));
    g = __fmul_rn(-expf(load(a.a_log_bf16, h)), softplus);
    beta = bf16_round(1.0f / (1.0f + expf(-load(a.b_bf16, gate))));
  }
  w.g[i] = g;
  w.beta[i] = beta;
  for (uint32_t d = 0; d < a.key_dim; ++d) {
    uint64_t off = uint64_t(i) * a.key_dim + d;
    float qv = 0, kv = 0;
    if (global < a.rows) {
      qv = bf16_round(__fmul_rn(load(a.qkv_bf16, q + d), qn));
      kv = bf16_round(__fmul_rn(load(a.qkv_bf16, q + kw + d), kn));
    }
    w.q[off] = __fmul_rn(qv, scale);
    w.k[off] = kv;
    w.kb[off] = __fmul_rn(kv, beta);
  }
  for (uint32_t d = 0; d < a.value_dim; ++d) {
    uint64_t off = uint64_t(i) * a.value_dim + d;
    float v = global < a.rows
                  ? load(a.qkv_bf16, uint64_t(global) * (2 * kw + vw) + 2 * kw +
                                         uint64_t(h) * a.value_dim + d)
                  : 0;
    w.v[off] = v;
    w.vb[off] = __fmul_rn(v, beta);
  }
}
// Each block owns one head. Extra threads remain in every barrier when the
// full HF tensor geometry selects 32-value windows instead of a 64-value
// window.
__global__ void prefix(Workspace w, uint32_t window) {
  __shared__ float row[C];
  uint32_t t = threadIdx.x, h = blockIdx.x;
  float carry = 0.0f;
  for (uint32_t start = 0; start < C; start += window) {
    if (t < window) {
      row[t] = w.g[uint64_t(h) * C + start + t];
      if (t == 0)
        row[0] = __fadd_rn(row[0], carry);
    }
    __syncthreads();
    for (uint32_t s = 1; s < window; s *= 2) {
      float v = t < window ? row[t] : 0.0f;
      if (t < window && (t & s))
        v = __fadd_rn(v, row[(t & ~(2 * s - 1)) + s - 1]);
      __syncthreads();
      if (t < window)
        row[t] = v;
      __syncthreads();
    }
    if (t < window)
      w.prefix[uint64_t(h) * C + start + t] = row[t];
    carry = row[window - 1];
    __syncthreads();
  }
}
__global__ void decay(Workspace w) {
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= w.nm)
    return;
  uint64_t h = i / (C * C);
  uint32_t r = i / C % C, c = i % C;
  w.decay[i] =
      r >= c ? expf(__fsub_rn(w.prefix[h * C + r], w.prefix[h * C + c])) : 0.0f;
}
__global__ void exponent_keys(Workspace w, uint32_t dk, bool end_relative) {
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= w.nk)
    return;
  uint64_t ht = i / dk, h = ht / C;
  float g = w.prefix[ht];
  if (end_relative)
    w.ke[i] = __fmul_rn(w.k[i], expf(__fsub_rn(w.prefix[h * C + C - 1], g)));
  else {
    float e = expf(g);
    w.ke[i] = __fmul_rn(w.kb[i], e);
    w.qe[i] = __fmul_rn(w.q[i], e);
  }
}
// Internal IEEE-F32 SIMT matmul. No TF32/BF16 product truncation. FMA is
// intentional within matrix products; elementwise boundaries below are not
// contracted. This does not assert cuBLAS reduction/dispatch bit equivalence.
__global__ void matmul(const float *a, const float *b, float *out, uint32_t m,
                       uint32_t n, uint32_t k, bool ta, bool tb,
                       uint32_t reference_batches = 0) {
  __shared__ float left[16][16], right[16][16];
  uint32_t r = blockIdx.y * 16 + threadIdx.y, c = blockIdx.x * 16 + threadIdx.x,
           h = blockIdx.z;
  a += uint64_t(h) * m * k;
  b += uint64_t(h) * k * n;
  float sum = 0;
  // Pinned sm86 HF dispatch for one [64,512] @ [64,512]^T uses
  // ampere_sgemm_32x32_sliced1x4_tn, grid.z=8, then splitKreduce.
  // Emulate its arithmetic locally: eight contiguous K64 partitions, each
  // with four K8 slices interleaved every K32, pairwise slice combine, and
  // ascending partition sum. This is IEEE F32 FMA, not TF32 or a BLAS handle.
  const uint32_t batches = reference_batches ? reference_batches : gridDim.z;
  const bool sliced =
      batches == 1 && m == 64 && n == 64 && k == 512 && !ta && tb;
  // The single-batch state projections [64,512] @ [512,511 or 512]
  // dispatch cutlass_80_simt_sgemm_128x32_8x5_nn_align1 plus splitKreduce:
  // eight sequential K64 FMA sums, then ascending F32 partition addition.
  // Other shape/transpose/batch regimes retain the original SIMT order.
  const bool split_state =
      batches == 1 && m == 64 && n >= 511 && n <= 512 && k == 512 && !ta && !tb;
  float partial[4] = {0, 0, 0, 0};
  float partition = 0.0f;
  for (uint32_t base = 0; base < k; base += 16) {
    uint32_t ak = base + threadIdx.x, bk = base + threadIdx.y;
    left[threadIdx.y][threadIdx.x] =
        r < m && ak < k ? a[ta ? uint64_t(ak) * m + r : uint64_t(r) * k + ak]
                        : 0;
    right[threadIdx.y][threadIdx.x] =
        bk < k && c < n ? b[tb ? uint64_t(c) * k + bk : uint64_t(bk) * n + c]
                        : 0;
    __syncthreads();
#pragma unroll
    for (uint32_t j = 0; j < 16; ++j) {
      if (sliced) {
        const uint32_t slice = ((base % 32) + j) / 8;
        partial[slice] = __fmaf_rn(left[threadIdx.y][j], right[j][threadIdx.x],
                                   partial[slice]);
      } else if (split_state) {
        partition =
            __fmaf_rn(left[threadIdx.y][j], right[j][threadIdx.x], partition);
      } else {
        sum = __fmaf_rn(left[threadIdx.y][j], right[j][threadIdx.x], sum);
      }
    }
    if (split_state && (base + 16) % 64 == 0) {
      sum = __fadd_rn(sum, partition);
      partition = 0.0f;
    }
    if (sliced && (base + 16) % 64 == 0) {
      sum = __fadd_rn(sum, __fadd_rn(__fadd_rn(partial[0], partial[1]),
                                     __fadd_rn(partial[2], partial[3])));
#pragma unroll
      for (uint32_t slice = 0; slice < 4; ++slice)
        partial[slice] = 0.0f;
    }
    __syncthreads();
  }
  if (r < m && c < n)
    out[(uint64_t(h) * m + r) * n + c] = sum;
}
__global__ void mask_matrix(Workspace w, bool triangular) {
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= w.nm)
    return;
  uint32_t r = i / C % C, c = i % C;
  if (triangular)
    w.tri[i] = r > c ? -__fmul_rn(w.tri[i], w.decay[i]) : -0.0f;
  else
    w.attn[i] = r >= c ? __fmul_rn(w.attn[i], w.decay[i]) : 0.0f;
}
// In-place row dependency is block-local. Snapshot the row before updating;
// product materialization and summation use separate RN instructions.
__global__ void triangular(Workspace w) {
  __shared__ float matrix[C * C], row[C];
  uint32_t j = threadIdx.x;
  uint64_t base = uint64_t(blockIdx.x) * C * C;
  for (uint32_t r = 0; r < C; ++r)
    matrix[r * C + j] = w.tri[base + r * C + j];
  __syncthreads();
  for (uint32_t r = 1; r < C; ++r) {
    row[j] = matrix[r * C + j];
    __syncthreads();
    if (j < r) {
      // HF materializes [H,chunks,r,r] products then reduces the penultimate
      // axis. With r<64 this does not split a column across y lanes: Reduce.cuh
      // uses four interleaved accumulators, combined left-to-right. Scalar
      // serial summation first diverges at row6 on the dense 65-token fixture.
      float acc[4] = {0, 0, 0, 0};
      for (uint32_t k = 0; k < r; ++k)
        acc[k % 4] =
            __fadd_rn(acc[k % 4], __fmul_rn(row[k], matrix[k * C + j]));
      float sum =
          __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
      matrix[r * C + j] = __fadd_rn(row[j], sum);
    }
    __syncthreads();
  }
  for (uint32_t r = 0; r < C; ++r)
    w.tri[base + r * C + j] =
        __fadd_rn(matrix[r * C + j], r == j ? 1.0f : 0.0f);
}
__global__ void delta_value(Workspace w) {
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < w.nv)
    w.vn[i] = __fsub_rn(w.value[i], w.vp[i]);
}
__global__ void output(FerruleCoreBf16ChunkArgs a, Workspace w,
                       uint32_t start) {
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= w.nv)
    return;
  uint32_t h = i / (C * a.value_dim), t = i / a.value_dim % C,
           d = i % a.value_dim;
  if (start + t < a.rows)
    pointer<uint16_t>(
        a.output_bf16)[(uint64_t(start + t) * a.value_heads + h) * a.value_dim +
                       d] = bf16_rne(__fadd_rn(w.inter[i], w.intra[i]));
}
__global__ void update_state(FerruleCoreBf16ChunkArgs a, Workspace w) {
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= w.ns)
    return;
  uint64_t h = i / (a.key_dim * a.value_dim);
  float *s = pointer<float>(a.state_f32);
  s[i] = __fadd_rn(__fmul_rn(s[i], expf(w.prefix[h * C + C - 1])),
                   w.state_product[i]);
}
} // namespace ferrule::cuda::core::bf16_chunk

extern "C" int32_t
ferrule_core_bf16_chunk_launch(const FerruleCoreBf16ChunkArgs *a) {
  using namespace ferrule::cuda::core;
  using namespace ferrule::cuda::core::bf16_chunk;
  if (!a || !validate(*a))
    return int32_t(cudaErrorInvalidValue);
  Workspace w(*a);
  cudaStream_t s = stream(a->stream);
  const uint32_t h = a->value_heads, dk = a->key_dim, dv = a->value_dim;
  const float scale = query_scale(dk);
  const uint32_t window = prefix_window(a->rows, h);
  const uint32_t precomputed_batches = h * ((a->rows + C - 1) / C);
#define CHUNK_CHECK()                                                          \
  do {                                                                         \
    int32_t status = launch_status();                                          \
    if (status)                                                                \
      return status;                                                           \
  } while (0)
#define CHUNK_MM(x, y, z, m, n, k, ta, tb, batches)                            \
  do {                                                                         \
    matmul<<<dim3(((n) + 15) / 16, ((m) + 15) / 16, h), dim3(16, 16), 0, s>>>( \
        x, y, z, m, n, k, ta, tb, batches);                                    \
    CHUNK_CHECK();                                                             \
  } while (0)
#define CHUNK_OBS(name, ptr, n) FERRULE_GDN_CHUNK_OBSERVE(name, ptr, n)
  for (uint32_t start = 0; start < a->rows; start += C) {
    prepare<<<blocks_for(w.ng), kBlock, 0, s>>>(*a, w, start, scale);
    CHUNK_CHECK();
    CHUNK_OBS("qscaled", w.q, w.nk);
    CHUNK_OBS("knorm", w.k, w.nk);
    CHUNK_OBS("g", w.g, w.ng);
    CHUNK_OBS("beta", w.beta, w.ng);
    CHUNK_OBS("k_beta", w.kb, w.nk);
    CHUNK_OBS("v_beta", w.vb, w.nv);
    prefix<<<h, C, 0, s>>>(w, window);
    CHUNK_CHECK();
    CHUNK_OBS("g_prefix", w.prefix, w.ng);
    decay<<<blocks_for(w.nm), kBlock, 0, s>>>(w);
    CHUNK_CHECK();
    CHUNK_OBS("decay_mask", w.decay, w.nm);
    exponent_keys<<<blocks_for(w.nk), kBlock, 0, s>>>(w, dk, false);
    CHUNK_CHECK();
    CHUNK_MM(w.kb, w.k, w.tri, C, C, dk, false, true, precomputed_batches);
    CHUNK_OBS("matmul0", w.tri, w.nm);
    mask_matrix<<<blocks_for(w.nm), kBlock, 0, s>>>(w, true);
    CHUNK_CHECK();
    CHUNK_OBS("triangular_before", w.tri, w.nm);
    triangular<<<h, C, 0, s>>>(w);
    CHUNK_CHECK();
    CHUNK_OBS("triangular_after", w.tri, w.nm);
    CHUNK_MM(w.tri, w.vb, w.value, C, dv, C, false, false, precomputed_batches);
    CHUNK_OBS("matmul1", w.value, w.nv);
    CHUNK_MM(w.tri, w.ke, w.kc, C, dk, C, false, false, precomputed_batches);
    CHUNK_OBS("matmul2", w.kc, w.nk);
    CHUNK_MM(w.q, w.k, w.attn, C, C, dk, false, true, h);
    CHUNK_OBS("matmul3", w.attn, w.nm);
    mask_matrix<<<blocks_for(w.nm), kBlock, 0, s>>>(w, false);
    CHUNK_CHECK();
    CHUNK_OBS("qk_attn", w.attn, w.nm);
    CHUNK_MM(w.kc, const_pointer<float>(a->state_f32), w.vp, C, dv, dk, false,
             false, h);
    CHUNK_OBS("matmul4", w.vp, w.nv);
    delta_value<<<blocks_for(w.nv), kBlock, 0, s>>>(w);
    CHUNK_CHECK();
    CHUNK_OBS("v_new", w.vn, w.nv);
    CHUNK_MM(w.qe, const_pointer<float>(a->state_f32), w.inter, C, dv, dk,
             false, false, h);
    CHUNK_OBS("matmul5", w.inter, w.nv);
    CHUNK_MM(w.attn, w.vn, w.intra, C, dv, C, false, false, h);
    CHUNK_OBS("matmul6", w.intra, w.nv);
    output<<<blocks_for(w.nv), kBlock, 0, s>>>(*a, w, start);
    CHUNK_CHECK();
    exponent_keys<<<blocks_for(w.nk), kBlock, 0, s>>>(w, dk, true);
    CHUNK_CHECK();
    CHUNK_MM(w.ke, w.vn, w.state_product, dk, dv, C, true, false, h);
    CHUNK_OBS("matmul7", w.state_product, w.ns);
    update_state<<<blocks_for(w.ns), kBlock, 0, s>>>(*a, w);
    CHUNK_CHECK();
  }
#undef CHUNK_OBS
#undef CHUNK_MM
#undef CHUNK_CHECK
  return 0;
}
