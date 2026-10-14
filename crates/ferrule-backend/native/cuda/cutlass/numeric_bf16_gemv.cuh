#pragma once
// Included after bf16_linear.cuh: reuse the exact bounded FP8 decode/RNE path.
#include <cublas_v2.h>
#include <initializer_list>
#include "cutlass/numeric_bf16_gemv_policy.h"
extern "C" int32_t ferrule_numeric_bf16_gemv_available() {
#if defined(CUBLAS_VER_MAJOR) && defined(CUBLAS_VER_MINOR)
  return ferrule::cuda::cutlass::numeric_bf16_gemv::supported_build(
      FERRULE_CUDA_TARGET_SM, CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR);
#else
  return 0;
#endif
}

namespace ferrule::cuda::cutlass::numeric_bf16_gemv {
inline int call(cublasHandle_t h, uint32_t count, uint64_t x, uint64_t w, uint64_t y) {
#if CUBLAS_VER_MAJOR == 12 && CUBLAS_VER_MINOR == 4 && FERRULE_CUDA_TARGET_SM >= 80
  const float alpha = 1.f, beta = 0.f;
  const auto status = cublasTSSgemvStridedBatched(h, CUBLAS_OP_T, kK, count,
      &alpha, reinterpret_cast<const __nv_bfloat16 *>(w), kK, 0,
      reinterpret_cast<const __nv_bfloat16 *>(x), 1, 0, &beta,
      reinterpret_cast<float *>(y), 1, 0, 1);
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 2000 + int(status);
#else
  (void)h; (void)count; (void)x; (void)w; (void)y;
  return 2;
#endif
}
}

extern "C" int32_t ferrule_numeric_bf16_gemv_create(
    uint64_t stream, uint64_t scratch, uint64_t bytes, uint64_t reserve,
    uint64_t *handle, uint64_t *growth, int32_t *version) {
  using namespace ferrule::cuda::cutlass::numeric_bf16_gemv;
  if (!handle || !growth || !version || !valid_scratch(scratch, bytes)) return 1;
  *handle = 0; *growth = 0; *version = 0;
  if (!ferrule_numeric_bf16_gemv_available()) return 2;
  const auto s = reinterpret_cast<cudaStream_t>(stream);
  cudaStreamCaptureStatus capturing;
  if (cudaStreamIsCapturing(s, &capturing) != cudaSuccess ||
      capturing != cudaStreamCaptureStatusNone) return 1;
  size_t before, total, warm, after;
  if (cudaMemGetInfo(&before, &total) != cudaSuccess) return 4;
  if (before < reserve) return 5;
  cublasHandle_t h = nullptr;
  auto status = cublasCreate(&h);
  // Transfer ownership immediately, including every subsequent failure path.
  *handle = reinterpret_cast<uint64_t>(h);
  if (status != CUBLAS_STATUS_SUCCESS) return 2000 + int(status);
  if ((status = cublasSetStream(h, s)) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasSetWorkspace(h, nullptr, 0)) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasSetMathMode(h, CUBLAS_DEFAULT_MATH)) != CUBLAS_STATUS_SUCCESS ||
      (status = cublasGetVersion(h, version)) != CUBLAS_STATUS_SUCCESS)
    return 2000 + int(status);
  // This integration is validated only against cuBLAS 12.4.5. The
  // TSS BF16 GEMV entry point is public in that toolkit, but its behavior is
  // not admitted on other runtime patch levels until separately validated.
  if (!supported_runtime(*version)) return 7;
  if (cudaMemsetAsync(reinterpret_cast<void *>(scratch), 0, kScratch, s) != cudaSuccess)
    return 4;
  // Only two shapes are admitted. All warmup storage is in the caller's grant.
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (uint32_t count : {kTile, kN % kTile}) {
      if (int rc = call(h, count, scratch, scratch + kK * 2, scratch + kStaging))
        return rc;
    }
    if (cudaStreamSynchronize(s) != cudaSuccess ||
        cudaMemGetInfo(&after, &total) != cudaSuccess) return 4;
    if (repeat == 0) warm = after;
    else if (after < warm) return 6;
    *growth = before > after ? before - after : 0;
    if (*growth > reserve) return 5;
  }
  return 0;
}
extern "C" int32_t ferrule_numeric_bf16_gemv_destroy(uint64_t handle) {
  return int(cublasDestroy(reinterpret_cast<cublasHandle_t>(handle)));
}
extern "C" int32_t ferrule_numeric_bf16_gemv_can_implement(const FerruleCutlassNumericFp8Args *a) {
  using namespace ferrule::cuda::cutlass::numeric_bf16_gemv;
  if (!a || a->m != 1 || a->n != kN || a->k != kK || a->tile_rows != kTile ||
      a->row_origin || a->column_origin || a->scale_bytes != 2 ||
      !valid_scratch(a->workspace, a->workspace_bytes)) return 1;
  if (int rc = ferrule::cuda::cutlass::numeric_fp8::tiled_can_implement(a, 2)) return rc;
  return ferrule_numeric_bf16_gemv_available() ? 0 : 2;
}
extern "C" int32_t ferrule_numeric_bf16_gemv_launch(
    const FerruleCutlassNumericFp8Args *a, uint64_t handle) {
  if (int rc = ferrule_numeric_bf16_gemv_can_implement(a)) return rc;
  if (!handle) return 1;
  using namespace ferrule::cuda::cutlass::numeric_fp8;
  auto s = reinterpret_cast<cudaStream_t>(a->stream);
  auto h = reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t handle_stream;
  if (cublasGetStream(h, &handle_stream) != CUBLAS_STATUS_SUCCESS || handle_stream != s)
    return 1;
  pack_bf16_activation<<<blocks(a->k), 256, 0, s>>>(*a);
  if (cudaGetLastError() != cudaSuccess) return 4;
  for (uint32_t first = 0; first < a->n; first += a->tile_rows) {
    const uint32_t count = a->n - first < a->tile_rows ? a->n - first : a->tile_rows;
    decode_weights<<<blocks(uint64_t(count) * a->k), 256, 0, s>>>(*a, first, count);
    if (cudaGetLastError() != cudaSuccess) return 4;
    if (int rc = ferrule::cuda::cutlass::numeric_bf16_gemv::call(h, count,
        a->workspace, a->workspace + uint64_t(a->k) * 2, a->output + uint64_t(first) * 4))
      return rc;
  }
  return 0;
}
