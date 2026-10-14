// Test-only transparent call observer. Never changes cuBLAS arguments or
// results.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <utility>
static std::mutex lock;
static std::map<cublasHandle_t, std::pair<void *, size_t>> workspaces;
template <class T> T next(const char *name) {
  auto p = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
  if (!p) {
    // Torch may have loaded its dependency in a local dlopen scope. Resolve
    // only the already-loaded cuBLAS, never load a second library version.
    void *loaded = dlopen("libcublas.so.12", RTLD_LAZY | RTLD_NOLOAD);
    if (loaded) {
      p = reinterpret_cast<T>(dlsym(loaded, name));
      dlclose(loaded);
    }
  }
  if (!p) {
    std::fprintf(stderr, "missing real %s\n", name);
    std::abort();
  }
  return p;
}
extern "C" cublasStatus_t cublasSetWorkspace(cublasHandle_t h, void *p,
                                             size_t n) {
  auto s =
      next<decltype(&cublasSetWorkspace)>("cublasSetWorkspace_v2")(h, p, n);
  std::lock_guard<std::mutex> guard(lock);
  if (s == CUBLAS_STATUS_SUCCESS)
    workspaces[h] = {p, n};
  return s;
}
extern "C" cublasStatus_t
cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
             int m, int n, int k, const void *alpha, const void *a,
             cudaDataType_t at, int lda, const void *b, cudaDataType_t bt,
             int ldb, const void *beta, void *c, cudaDataType_t ct, int ldc,
             cublasComputeType_t compute, cublasGemmAlgo_t algo) {
  cublasMath_t math;
  cublasAtomicsMode_t atomics;
  cublasPointerMode_t mode;
  cudaStream_t stream;
  int version;
  if (next<decltype(&cublasGetMathMode)>("cublasGetMathMode")(h, &math) ||
      next<decltype(&cublasGetAtomicsMode)>("cublasGetAtomicsMode")(h,
                                                                    &atomics) ||
      next<decltype(&cublasGetPointerMode)>("cublasGetPointerMode_v2")(h,
                                                                       &mode) ||
      next<decltype(&cublasGetStream)>("cublasGetStream_v2")(h, &stream) ||
      next<decltype(&cublasGetVersion_v2)>("cublasGetVersion_v2")(h, &version))
    std::abort();
  using Gemm = cublasStatus_t (*)(
      cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
      const void *, const void *, cudaDataType_t, int, const void *,
      cudaDataType_t, int, const void *, void *, cudaDataType_t, int,
      cublasComputeType_t, cublasGemmAlgo_t);
  auto result =
      next<Gemm>("cublasGemmEx")(h, ta, tb, m, n, k, alpha, a, at, lda, b, bt,
                                 ldb, beta, c, ct, ldc, compute, algo);
  std::lock_guard<std::mutex> guard(lock);
  const char *path = std::getenv("FERRULE_GEMV_CALL_LOG");
  if (!path)
    std::abort();
  FILE *f = std::fopen(path, "a");
  if (!f)
    std::abort();
  auto w = workspaces.find(h);
  std::fprintf(
      f,
      "{\"api\":\"cublasGemmEx\",\"version\":%d,\"transa\":%d,\"transb\":%d,"
      "\"m\":%d,\"n\":%d,\"k\":%d,\"atype\":%d,\"btype\":%d,\"ctype\":%d,"
      "\"lda\":%d,\"ldb\":%d,\"ldc\":%d,\"compute\":%d,\"algo\":%d,\"math\":%d,"
      "\"atomics\":%d,\"pointer_mode\":%d,\"alpha\":%g,\"beta\":%g,\"a_"
      "mod256\":%zu,\"b_mod256\":%zu,\"c_mod256\":%zu,\"workspace_known\":%s,"
      "\"workspace_bytes\":%zu,\"status\":%d}\n",
      version, int(ta), int(tb), m, n, k, int(at), int(bt), int(ct), lda, ldb,
      ldc, int(compute), int(algo), int(math), int(atomics), int(mode),
      mode == CUBLAS_POINTER_MODE_HOST
          ? double(*static_cast<const float *>(alpha))
          : -999.,
      mode == CUBLAS_POINTER_MODE_HOST
          ? double(*static_cast<const float *>(beta))
          : -999.,
      uintptr_t(a) % 256, uintptr_t(b) % 256, uintptr_t(c) % 256,
      w == workspaces.end() ? "false" : "true",
      w == workspaces.end() ? 0 : w->second.second, int(result));
  if (std::fclose(f))
    std::abort();
  return result;
}
