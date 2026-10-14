// CPU-only ABI/forwarding test for the observer. No CUDA library or device
// call.
#include <cstdint>
#include <cublas_v2.h>
#ifdef CONTRACT_MOCK_LIBRARY
extern "C" cublasStatus_t cublasSetWorkspace(cublasHandle_t, void *, size_t) {
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasGetMathMode(cublasHandle_t, cublasMath_t *p) {
  *p = cublasMath_t(16);
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasGetAtomicsMode(cublasHandle_t,
                                               cublasAtomicsMode_t *p) {
  *p = CUBLAS_ATOMICS_NOT_ALLOWED;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t,
                                                  cublasPointerMode_t *p) {
  *p = CUBLAS_POINTER_MODE_HOST;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasGetStream_v2(cublasHandle_t, cudaStream_t *p) {
  *p = nullptr;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasGetVersion_v2(cublasHandle_t, int *p) {
  *p = 120901;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t
cublasGemmEx(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb, int m,
             int n, int k, const void *alpha, const void *a, cudaDataType_t at,
             int lda, const void *b, cudaDataType_t bt, int ldb,
             const void *beta, void *c, cudaDataType_t ct, int ldc,
             cublasComputeType_t compute, cublasGemmAlgo_t algo) {
  if (ta != CUBLAS_OP_T || tb != CUBLAS_OP_N || m != 2048 || n != 1 ||
      k != 512 || at != CUDA_R_16BF || bt != CUDA_R_16BF || ct != CUDA_R_16BF ||
      lda != 512 || ldb != 512 || ldc != 2048 ||
      compute != CUBLAS_COMPUTE_32F || algo != CUBLAS_GEMM_DEFAULT_TENSOR_OP ||
      !a || !b || *static_cast<const float *>(alpha) != 1.f ||
      *static_cast<const float *>(beta) != 0.f)
    return CUBLAS_STATUS_INVALID_VALUE;
  *static_cast<uint16_t *>(c) = 0xbeef;
  return CUBLAS_STATUS_SUCCESS;
}
#else
int main() {
  alignas(256) uint16_t a[256]{}, b[256]{}, c[256]{};
  float alpha = 1, beta = 0;
  auto h = reinterpret_cast<cublasHandle_t>(uintptr_t(1));
  if (cublasSetWorkspace(h, a, 32 << 20) != CUBLAS_STATUS_SUCCESS)
    return 1;
  auto s =
      cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 2048, 1, 512, &alpha, a,
                   CUDA_R_16BF, 512, b, CUDA_R_16BF, 512, &beta, c, CUDA_R_16BF,
                   2048, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return s != CUBLAS_STATUS_SUCCESS || c[0] != 0xbeef;
}
#endif
