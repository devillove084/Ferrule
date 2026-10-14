// Exercise the real native create/view status path with a controlled first
// synchronization error. The next real synchronization succeeds. No math edits.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#define FERRULE_CUDA_TARGET_SM 86
#include "cutlass/bf16_linear.cuh"
static bool fail_next = false;
static cudaError_t observed_sync(cudaStream_t stream) {
  auto rc = cudaStreamSynchronize(stream);
  if (fail_next && rc == cudaSuccess) {
    fail_next = false;
    return cudaErrorUnknown;
  }
  return rc;
}
#define cudaStreamSynchronize observed_sync
#include "cutlass/bf16_gemmex.cuh"
#undef cudaStreamSynchronize
static void ck(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
int main() {
  try {
    cudaDeviceProp prop; ck(cudaGetDeviceProperties(&prop,0));
    int runtime; ck(cudaRuntimeGetVersion(&runtime));
    if (runtime!=12040 || prop.major!=8 || prop.minor!=6) throw std::runtime_error("CUDA12.4/sm86 required");
    using namespace ferrule::cuda::cutlass::bf16_gemmex;
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    void *storage; const uint64_t bytes=required(2,256,128);ck(cudaMalloc(&storage,bytes));
    uint64_t handle=0,growth=0;int32_t version=0;
    fail_next=true;
    const int create=ferrule_bf16_gemmex_create(2,256,128,reinterpret_cast<uint64_t>(stream),reinterpret_cast<uint64_t>(storage),bytes,kProvider,&handle,&growth,&version);
    if(create!=9 || !handle)throw std::runtime_error("native create must report completion-unknown status9");
    ck(cudaStreamSynchronize(stream)); // later success does not erase create's status
    fail_next=true;
    const int view=ferrule_bf16_gemmex_warm_view(handle,1,128,128,reinterpret_cast<uint64_t>(storage),staging(2,256,128),growth,&growth);
    if(view!=9)throw std::runtime_error("native view must report completion-unknown status9");
    ck(cudaStreamSynchronize(stream));
    // This isolated native status probe deliberately has no Rust owner. Only
    // after its real successful drains do we destroy these controlled fixtures.
    if(ferrule_bf16_gemmex_destroy(handle))throw std::runtime_error("destroy");
    ck(cudaFree(storage));ck(cudaStreamDestroy(stream));
    std::cout<<"{\"create_status\":9,\"view_status\":9,\"later_fences_success\":true,\"rust_must_quarantine\":true}\n";
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
