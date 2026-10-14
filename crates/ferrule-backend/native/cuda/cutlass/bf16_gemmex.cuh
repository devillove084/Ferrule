#pragma once
#include "cutlass/bf16_gemmex_policy.h"
#include "cutlass/bf16_gemmex_admission.h"
#include <cublas_v2.h>
#include "core/device.cuh"
// Included after bf16_linear.cuh; identical numeric decode, no new FP8 math.
extern "C" int32_t ferrule_bf16_gemmex_available() {
#if CUBLAS_VER_MAJOR == 12 && CUBLAS_VER_MINOR == 4 &&                         \
    FERRULE_CUDA_TARGET_SM == 86
  return 1;
#else
  return 0;
#endif
}
namespace ferrule::cuda::cutlass::bf16_gemmex {
inline int call(cublasHandle_t h, uint32_t m, uint32_t n, uint32_t k,
                uint64_t input, uint64_t weight, uint64_t output) {
  const float alpha = 1.f, beta = 0.f;
  const auto rc =
      cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                   reinterpret_cast<void *>(weight), CUDA_R_16BF, k,
                   reinterpret_cast<void *>(input), CUDA_R_16BF, k, &beta,
                   reinterpret_cast<void *>(output), CUDA_R_16BF, n,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return rc == CUBLAS_STATUS_SUCCESS ? 0 : 2000 + int(rc);
}
} // namespace ferrule::cuda::cutlass::bf16_gemmex
extern "C" int32_t
ferrule_bf16_gemmex_create(uint32_t m, uint32_t n, uint32_t k, uint64_t stream,
                           uint64_t memory, uint64_t bytes, uint64_t reserve,
                           uint64_t *handle, uint64_t *growth,
                           int32_t *version) {
  using namespace ferrule::cuda::cutlass::bf16_gemmex;
  if (!handle || !growth || !version)
    return 1;
  *handle = 0;
  *growth = 0;
  *version = 0;
  if (!storage(m, n, k, memory, bytes) || reserve < kProvider)
    return 1;
  if (!ferrule_bf16_gemmex_available())
    return 2;
  if (!ferrule_bf16_gemmex_process_config_valid()) return 8;
  int device, major, minor;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess)
    return 4;
  if (major != 8 || minor != 6)
    return 2;
  auto s = reinterpret_cast<cudaStream_t>(stream);
  cudaStreamCaptureStatus capture;
  if (cudaStreamIsCapturing(s, &capture) != cudaSuccess ||
      capture != cudaStreamCaptureStatusNone)
    return 1;
  // Device-global delta is diagnostic only, never attributable accounting.
  // Rust holds persistent per-owner physical/external reservations and only
  // serializes GemmEx initialization per device. Unrelated allocations are not
  // stopped by this guard; free-memory samples do not establish attribution.
  size_t before, total, warm = 0, after;
  if (cudaMemGetInfo(&before, &total) != cudaSuccess)
    return 4;
  if (before < kProvider)
    return 5;
  cublasHandle_t h = nullptr;
  auto rc = cublasCreate(&h);
  *handle =
      reinterpret_cast<uint64_t>(h); // Rust owns every failure after create.
  if (rc != CUBLAS_STATUS_SUCCESS)
    return 2000 + int(rc);
  if ((rc = cublasGetVersion(h, version)) != CUBLAS_STATUS_SUCCESS)
    return 2000 + int(rc);
  if (*version != 120405)
    return 7; // Runtime patch level, not header version.
  if ((rc = cublasSetStream(h, s)) != CUBLAS_STATUS_SUCCESS ||
      (rc = cublasSetWorkspace(
           h, reinterpret_cast<void *>(memory + staging(m, n, k)),
           kWorkspace)) != CUBLAS_STATUS_SUCCESS ||
      (rc = cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST)) !=
          CUBLAS_STATUS_SUCCESS ||
      (rc = cublasSetMathMode(
           h, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)) !=
          CUBLAS_STATUS_SUCCESS ||
      (rc = cublasSetAtomicsMode(h, CUBLAS_ATOMICS_NOT_ALLOWED)) !=
          CUBLAS_STATUS_SUCCESS)
    return 2000 + int(rc);
  if (cudaMemsetAsync(reinterpret_cast<void *>(memory), 0, staging(m, n, k),
                      s) != cudaSuccess)
    return 4;
  for (int repeat = 0; repeat < 2; ++repeat) {
    if (int code = call(h, m, n, k, memory, memory + uint64_t(m) * k * 2,
                        memory + (uint64_t(m) + n) * k * 2))
      return code;
    // Status 9 is a lost completion proof, not an ordinary setup failure.
    // Rust must retain custody even if its subsequent fence could succeed.
    if (cudaStreamSynchronize(s) != cudaSuccess) return 9;
    if (cudaMemGetInfo(&after, &total) != cudaSuccess) return 4;
    *growth = before > after ? before - after : 0;
    if (*growth > kProvider)
      return 5;
    if (repeat && after < warm)
      return 6;
    warm = after;
  }
  return 0;
}
extern "C" int32_t
ferrule_bf16_gemmex_can_implement(const FerruleCutlassNumericFp8Args *a) {
  using namespace ferrule::cuda::cutlass::bf16_gemmex;
  if (!a || !geometry(a->m, a->n, a->k) ||
      !storage(a->m, a->n, a->k, a->workspace, a->workspace_bytes) ||
      a->padded_k != a->k || a->tile_rows != a->n || a->lda != a->k ||
      a->ldd != a->n || a->row_origin || a->column_origin || a->reserved ||
      a->scale_bytes != 2 || a->scale_cols != a->k / 128)
    return 1;
  const uint64_t sizes[] = {uint64_t(a->m) * a->k * 2, uint64_t(a->n) * a->k,
                            uint64_t(a->n / 128) * (a->k / 128) * 2,
                            uint64_t(a->m) * a->n * 2,
                            required(a->m, a->n, a->k)};
  const uint64_t ptrs[] = {a->activation, a->weight, a->scales, a->output,
                           a->workspace};
  const uint64_t align[] = {256, 1, 2, 256, 256};
  for (int i = 0; i < 5; ++i)
    if (!span(ptrs[i], sizes[i], align[i]))
      return 1;
  for (int w = 3; w < 5; ++w)
    for (int r = 0; r < w; ++r)
      if (overlap(ptrs[w], sizes[w], ptrs[r], sizes[r]))
        return 1;
  return ferrule_bf16_gemmex_available() ? 0 : 2;
}
extern "C" int32_t
ferrule_bf16_gemmex_launch(const FerruleCutlassNumericFp8Args *a,
                           uint64_t handle) {
  if (int rc = ferrule_bf16_gemmex_can_implement(a))
    return rc;
  if (!handle)
    return 1;
  auto h = reinterpret_cast<cublasHandle_t>(handle);
  auto s = reinterpret_cast<cudaStream_t>(a->stream);
  cudaStream_t hs;
  if (cublasGetStream(h, &hs) != CUBLAS_STATUS_SUCCESS || hs != s)
    return 1;
  using namespace ferrule::cuda::cutlass::numeric_fp8;
  pack_bf16_activation<<<blocks(uint64_t(a->m) * a->k), 256, 0, s>>>(*a);
  if (cudaGetLastError() != cudaSuccess)
    return 4;
  decode_weights<<<blocks(uint64_t(a->n) * a->k), 256, 0, s>>>(*a, 0, a->n);
  if (cudaGetLastError() != cudaSuccess)
    return 4;
  return ferrule::cuda::cutlass::bf16_gemmex::call(
      h, a->m, a->n, a->k, a->workspace,
      a->workspace + uint64_t(a->m) * a->k * 2, a->output);
}
extern "C" int32_t ferrule_bf16_gemmex_destroy(uint64_t handle) {
  return int(cublasDestroy(reinterpret_cast<cublasHandle_t>(handle)));
}

// Shape views share the original maximum-staging workspace binding. Warmup
// never calls SetStream/SetWorkspace and runs only during owner preparation.
extern "C" int32_t ferrule_bf16_gemmex_warm_view(
    uint64_t handle, uint32_t m, uint32_t n, uint32_t k, uint64_t memory,
    uint64_t staging_capacity, uint64_t initial_growth, uint64_t *growth) {
  using namespace ferrule::cuda::cutlass::bf16_gemmex;
  if (!handle || !growth || !resident_geometry(m, n, k) ||
      staging(m, n, k) > staging_capacity) return 1;
  auto h = reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t s;
  if (cublasGetStream(h, &s) != CUBLAS_STATUS_SUCCESS) return 4;
  size_t before, after, total, warm = 0;
  if (cudaMemGetInfo(&before, &total) != cudaSuccess) return 4;
  if (cudaMemsetAsync(reinterpret_cast<void *>(memory), 0,
                      staging(m, n, k), s) != cudaSuccess) return 4;
  for (int i = 0; i < 2; ++i) {
    if (int rc = call(h, m, n, k, memory, memory + uint64_t(m)*k*2,
                      memory + (uint64_t(m)+n)*k*2)) return rc;
    if (cudaStreamSynchronize(s) != cudaSuccess) return 9;
    if (cudaMemGetInfo(&after, &total) != cudaSuccess) return 4;
    *growth = initial_growth + (before > after ? before - after : 0);
    if (*growth > kProvider) return 5;
    if (i && after < warm) return 6;
    warm = after;
  }
  return 0;
}
namespace ferrule::cuda::cutlass::bf16_gemmex {
__global__ void split_pair(const uint16_t *src, uint16_t *a, uint16_t *b,
                           uint64_t count, uint32_t width) {
  for (uint64_t i = uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x)*blockDim.x) {
    const uint64_t row = i / width, col = i % width;
    a[i] = src[row*width*2 + col];
    b[i] = src[row*width*2 + width + col];
  }
}
__global__ void pair_swiglu(const uint16_t *src, uint16_t *activation,
                            uint64_t count, uint32_t width) {
  for (uint64_t i = uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x)*blockDim.x) {
    const uint64_t base = (i / width)*width*2 + i % width;
    activation[i] = ferrule::cuda::core::bf16_swiglu(src[base], src[base + width]);
  }
}
template<bool Swiglu>
inline int pair_tail(uint64_t packed, uint64_t gate, uint64_t up,
                     uint32_t m, uint32_t n, cudaStream_t stream) {
  using namespace ferrule::cuda::cutlass::numeric_fp8;
  if constexpr (Swiglu) {
    pair_swiglu<<<blocks(uint64_t(m)*n),256,0,stream>>>(
        reinterpret_cast<const uint16_t *>(packed),
        reinterpret_cast<uint16_t *>(gate),uint64_t(m)*n,n);
  } else {
    split_pair<<<blocks(uint64_t(m)*n),256,0,stream>>>(
        reinterpret_cast<const uint16_t *>(packed),
        reinterpret_cast<uint16_t *>(gate),reinterpret_cast<uint16_t *>(up),
        uint64_t(m)*n,n);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

}
// Decode the original two immutable artifacts directly into adjacent BF16
// staging rows. No host payload concatenation or separately rounded GEMMs.
template<bool Swiglu>
static int32_t bf16_gemmex_can_pair(
    const FerruleCutlassNumericFp8Args *a,
    const FerruleCutlassNumericFp8Args *b) {
  using namespace ferrule::cuda::cutlass::bf16_gemmex;
  if (!a || !b || a->m != b->m || a->n != b->n ||
      a->k != b->k || a->activation != b->activation ||
      a->workspace != b->workspace || a->stream != b->stream ||
      (Swiglu && a->output != b->output) ||
      a->n > (INT_MAX - 128u)/2 ||
      !storage(a->m, a->n*2, a->k, a->workspace, a->workspace_bytes)) return 1;
  if (int rc = ferrule_bf16_gemmex_can_implement(a)) return rc;
  if (int rc = ferrule_bf16_gemmex_can_implement(b)) return rc;
  const uint64_t outbytes = uint64_t(a->m)*a->n*2;
  const uint64_t reads[] = {a->weight, a->scales, b->weight, b->scales};
  const uint64_t lengths[] = {uint64_t(a->n)*a->k,
    uint64_t(a->n/128)*(a->k/128)*2, uint64_t(b->n)*b->k,
    uint64_t(b->n/128)*(b->k/128)*2};
  if (!span(a->workspace, a->workspace_bytes, 256) ||
      (!Swiglu && overlap(a->output, outbytes, b->output, outbytes)) ||
      overlap(a->workspace, a->workspace_bytes, a->activation, uint64_t(a->m)*a->k*2) ||
      overlap(a->workspace, a->workspace_bytes, a->output, outbytes) ||
      overlap(a->workspace, a->workspace_bytes, b->output, outbytes)) return 1;
  for (int i=0; i<4; ++i)
    if (overlap(a->output,outbytes,reads[i],lengths[i]) ||
        overlap(b->output,outbytes,reads[i],lengths[i]) ||
        overlap(a->workspace, a->workspace_bytes, reads[i],lengths[i])) return 1;
  return 0;
}
template<bool Swiglu>
static int32_t bf16_gemmex_launch_pair(
    const FerruleCutlassNumericFp8Args *a,
    const FerruleCutlassNumericFp8Args *b, uint64_t handle) {
  if (!handle) return 1;
  if (int rc = bf16_gemmex_can_pair<Swiglu>(a, b)) return rc;
  using namespace ferrule::cuda::cutlass::bf16_gemmex;
  auto h = reinterpret_cast<cublasHandle_t>(handle);
  auto s = reinterpret_cast<cudaStream_t>(a->stream);
  cudaStream_t hs;
  if (cublasGetStream(h, &hs) != CUBLAS_STATUS_SUCCESS || hs != s) return 1;
  using namespace ferrule::cuda::cutlass::numeric_fp8;
  pack_bf16_activation<<<blocks(uint64_t(a->m)*a->k),256,0,s>>>(*a);
  if (cudaGetLastError() != cudaSuccess) return 4;
  decode_weights<<<blocks(uint64_t(a->n)*a->k),256,0,s>>>(*a,0,a->n);
  if (cudaGetLastError() != cudaSuccess) return 4;
  auto second = *b;
  second.workspace += uint64_t(a->n)*a->k*2;
  decode_weights<<<blocks(uint64_t(b->n)*b->k),256,0,s>>>(second,0,b->n);
  if (cudaGetLastError() != cudaSuccess) return 4;
  const uint64_t output = a->workspace + (uint64_t(a->m)+2*a->n)*a->k*2;
  if (int rc = call(h,a->m,a->n*2,a->k,a->workspace,
                    a->workspace+uint64_t(a->m)*a->k*2,output)) return rc;
  return pair_tail<Swiglu>(output,a->output,b->output,a->m,a->n,s);
}

extern "C" int32_t ferrule_bf16_gemmex_can_pair(
    const FerruleCutlassNumericFp8Args *a, const FerruleCutlassNumericFp8Args *b) {
  return bf16_gemmex_can_pair<false>(a,b);
}
extern "C" int32_t ferrule_bf16_gemmex_launch_pair(
    const FerruleCutlassNumericFp8Args *a, const FerruleCutlassNumericFp8Args *b,
    uint64_t handle) {
  return bf16_gemmex_launch_pair<false>(a,b,handle);
}
extern "C" int32_t ferrule_bf16_gemmex_can_pair_swiglu(
    const FerruleCutlassNumericFp8Args *a, const FerruleCutlassNumericFp8Args *b) {
  return bf16_gemmex_can_pair<true>(a,b);
}
extern "C" int32_t ferrule_bf16_gemmex_launch_pair_swiglu(
    const FerruleCutlassNumericFp8Args *a, const FerruleCutlassNumericFp8Args *b,
    uint64_t handle) {
  return bf16_gemmex_launch_pair<true>(a,b,handle);
}

// Resident BF16 uses the same prepared handle/workspace and native BF16 C.
// No staging, decode, packing, allocation or workspace rebind at invocation.
struct FerruleBf16GemmExResidentArgs {
  uint32_t m, n, k, reserved;
  uint64_t input, weight, output, stream;
};
static_assert(sizeof(FerruleBf16GemmExResidentArgs) == 48);
extern "C" int32_t ferrule_bf16_gemmex_resident_can(
    const FerruleBf16GemmExResidentArgs *a) {
  using namespace ferrule::cuda::cutlass::bf16_gemmex;
  if (!a || a->reserved || !resident_geometry(a->m,a->n,a->k)) return 1;
  const uint64_t xb=uint64_t(a->m)*a->k*2, wb=uint64_t(a->n)*a->k*2,
                 yb=uint64_t(a->m)*a->n*2;
  // GemmEx accepts element-aligned packed BF16, including scalar/tail N/K.
  // Tensor-core preferred alignment is not a correctness constraint.
  if (!span(a->input,xb,2) || !span(a->weight,wb,2) ||
      !span(a->output,yb,2) || overlap(a->output,yb,a->input,xb) ||
      overlap(a->output,yb,a->weight,wb)) return 1;
  return ferrule_bf16_gemmex_available() ? 0 : 2;
}
extern "C" int32_t ferrule_bf16_gemmex_resident_launch(
    const FerruleBf16GemmExResidentArgs *a, uint64_t handle) {
  if (int rc=ferrule_bf16_gemmex_resident_can(a)) return rc;
  if (!handle) return 1;
  auto h=reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  if (cublasGetStream(h,&stream)!=CUBLAS_STATUS_SUCCESS ||
      stream!=reinterpret_cast<cudaStream_t>(a->stream)) return 1;
  return ferrule::cuda::cutlass::bf16_gemmex::call(
      h,a->m,a->n,a->k,a->input,a->weight,a->output);
}

// Standalone decode into caller-retained storage; no activation or workspace.
extern "C" int32_t ferrule_bf16_gemmex_decode_can(
    const FerruleCutlassNumericFp8Args *a) {
  using namespace ferrule::cuda::cutlass::bf16_gemmex;
  if (!a || a->m != 1 || !geometry(1,a->n,a->k) ||
      a->padded_k != a->k || a->tile_rows != a->n ||
      a->scale_cols != a->k/128 || a->scale_bytes != 2 ||
      a->row_origin || a->column_origin || a->reserved ||
      a->lda != a->k || a->ldd != a->n || a->activation ||
      a->workspace || a->workspace_bytes) return 1;
  const uint64_t wb=uint64_t(a->n)*a->k,
                 sb=uint64_t(a->n/128)*(a->k/128)*2, yb=wb*2;
  if (!span(a->weight,wb,1) || !span(a->scales,sb,2) ||
      !span(a->output,yb,2) || overlap(a->output,yb,a->weight,wb) ||
      overlap(a->output,yb,a->scales,sb)) return 1;
  return ferrule_bf16_gemmex_available() ? 0 : 2;
}
extern "C" int32_t ferrule_bf16_gemmex_decode_launch(
    const FerruleCutlassNumericFp8Args *a) {
  if (int rc=ferrule_bf16_gemmex_decode_can(a)) return rc;
  using namespace ferrule::cuda::cutlass::numeric_fp8;
  decode_weights<<<blocks(uint64_t(a->n)*a->k),256,0,
      reinterpret_cast<cudaStream_t>(a->stream)>>>(*a,0,a->n,a->output);
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

struct FerruleBf16GemmExResidentPairArgs {
  FerruleBf16GemmExResidentArgs gemm;
  uint64_t gate, up, workspace, workspace_bytes;
};
static_assert(sizeof(FerruleBf16GemmExResidentPairArgs) == 80);
template<bool Swiglu>
static int32_t bf16_gemmex_resident_pair_can(
    const FerruleBf16GemmExResidentPairArgs *a) {
  using namespace ferrule::cuda::cutlass::bf16_gemmex;
  if (!a) return 1;
  const auto &g=a->gemm;
  if ((Swiglu && a->gate != a->up) || g.n % 2 || !geometry(g.m,g.n/2,g.k) ||
      !storage(g.m,g.n,g.k,a->workspace,a->workspace_bytes) ||
      !span(a->workspace,a->workspace_bytes,256) ||
      g.output != a->workspace+(uint64_t(g.m)+g.n)*g.k*2) return 1;
  if (int rc=ferrule_bf16_gemmex_resident_can(&g)) return rc;
  const uint64_t ptrs[]={g.input,g.weight,a->gate,a->up};
  const uint64_t bytes[]={uint64_t(g.m)*g.k*2,uint64_t(g.n)*g.k*2,
                          uint64_t(g.m)*g.n,uint64_t(g.m)*g.n};
  for (int i=0;i<(Swiglu ? 3 : 4);++i) {
    if (!span(ptrs[i],bytes[i],2) ||
        overlap(a->workspace,a->workspace_bytes,ptrs[i],bytes[i])) return 1;
    if (i>=2)
      for (int j=0;j<i;++j)
        if (overlap(ptrs[i],bytes[i],ptrs[j],bytes[j])) return 1;
  }
  return 0;
}
template<bool Swiglu>
static int32_t bf16_gemmex_resident_pair_launch(
    const FerruleBf16GemmExResidentPairArgs *a, uint64_t handle) {
  if (int rc=bf16_gemmex_resident_pair_can<Swiglu>(a)) return rc;
  // The same prepared 2N call and selected tail; only A/B staging is bypassed.
  if (int rc=ferrule_bf16_gemmex_resident_launch(&a->gemm,handle)) return rc;
  const auto &g=a->gemm;
  return ferrule::cuda::cutlass::bf16_gemmex::pair_tail<Swiglu>(
      g.output,a->gate,a->up,g.m,g.n/2,reinterpret_cast<cudaStream_t>(g.stream));
}

extern "C" int32_t ferrule_bf16_gemmex_resident_pair_can(
    const FerruleBf16GemmExResidentPairArgs *a) {
  return bf16_gemmex_resident_pair_can<false>(a);
}
extern "C" int32_t ferrule_bf16_gemmex_resident_pair_launch(
    const FerruleBf16GemmExResidentPairArgs *a, uint64_t handle) {
  return bf16_gemmex_resident_pair_launch<false>(a,handle);
}
extern "C" int32_t ferrule_bf16_gemmex_resident_pair_swiglu_can(
    const FerruleBf16GemmExResidentPairArgs *a) {
  return bf16_gemmex_resident_pair_can<true>(a);
}
extern "C" int32_t ferrule_bf16_gemmex_resident_pair_swiglu_launch(
    const FerruleBf16GemmExResidentPairArgs *a, uint64_t handle) {
  return bf16_gemmex_resident_pair_launch<true>(a,handle);
}

// Preflight both decodes, including cross-source aliases, before the first
// submission: the first half must never overwrite the second encoded source.
extern "C" int32_t ferrule_bf16_gemmex_decode_pair_can(
    const FerruleCutlassNumericFp8Args *a,
    const FerruleCutlassNumericFp8Args *b) {
  using namespace ferrule::cuda::cutlass::bf16_gemmex;
  if (!a || !b || a->n != b->n || a->k != b->k ||
      a->stream != b->stream || a->n > (INT_MAX-128u)/2) return 1;
  if (int rc=ferrule_bf16_gemmex_decode_can(a)) return rc;
  if (int rc=ferrule_bf16_gemmex_decode_can(b)) return rc;
  const uint64_t wb=uint64_t(a->n)*a->k,
                 sb=uint64_t(a->n/128)*(a->k/128)*2, half=wb*2;
  if (!span(a->output,half*2,2) || b->output != a->output+half) return 1;
  const uint64_t reads[]={a->weight,a->scales,b->weight,b->scales};
  const uint64_t bytes[]={wb,sb,wb,sb};
  for (int i=0;i<4;++i)
    if (overlap(a->output,half*2,reads[i],bytes[i])) return 1;
  return 0;
}
extern "C" int32_t ferrule_bf16_gemmex_decode_pair_launch(
    const FerruleCutlassNumericFp8Args *a,
    const FerruleCutlassNumericFp8Args *b) {
  if (int rc=ferrule_bf16_gemmex_decode_pair_can(a,b)) return rc;
  if (int rc=ferrule_bf16_gemmex_decode_launch(a)) return rc;
  return ferrule_bf16_gemmex_decode_launch(b);
}
