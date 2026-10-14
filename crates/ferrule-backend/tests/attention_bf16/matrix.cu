// Matrix-only diagnostic. Never called by production; frozen BF16 probabilities
// isolate PV from QK/softmax so a coincidentally matching gate cannot hide errors.
#include "../ordinary_norm/buffers.cuh"
#include <cublas_v2.h>
static void blas(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(int(s)));
}
struct Handle {
  cublasHandle_t h = nullptr;
  Handle() { blas(cublasCreate(&h)); }
  ~Handle() { if (h) cublasDestroy(h); }
};
int main(int argc, char **argv) { try {
  require(argc == 7, "input output rows heads dim context");
  const std::string in=argv[1], out=argv[2];
  const int r=std::stoi(argv[3]), h=std::stoi(argv[4]), d=std::stoi(argv[5]), n=std::stoi(argv[6]);
  require(r>0 && h>0 && d>0 && n>0, "positive dimensions");
  Buffer<uint16_t> q(size_t(h)*r*d), k(size_t(h)*n*d), v(size_t(h)*n*d), p(size_t(h)*r*n);
  q.read(in+"/query.bf16");k.read(out+"/repeat-key.bf16");v.read(out+"/repeat-value.bf16");p.read(out+"/prob.bf16");
  Buffer<uint16_t> qk(size_t(h)*r*n), pv(size_t(h)*r*d);
  Buffer<uint4> workspace(32u*1024u*1024u/sizeof(uint4));
  Handle owner; const auto handle=owner.h;
  blas(cublasSetStream(handle,nullptr));blas(cublasSetWorkspace(handle,workspace.p,workspace.n*sizeof(uint4)));
  blas(cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  int version;blas(cublasGetVersion(handle,&version));
  const float alpha=1.f,beta=0.f;
  const auto *qb=reinterpret_cast<const __nv_bfloat16 *>(q.p);
  const auto *kb=reinterpret_cast<const __nv_bfloat16 *>(k.p);
  const auto *vb=reinterpret_cast<const __nv_bfloat16 *>(v.p);
  const auto *pb=reinterpret_cast<const __nv_bfloat16 *>(p.p);
  auto *cb=reinterpret_cast<__nv_bfloat16 *>(qk.p);
  auto *ob=reinterpret_cast<__nv_bfloat16 *>(pv.p);
  for (int mode=0;mode<3;++mode) {
    if (mode==2 && r!=1) continue;
    const std::string label=mode==0?"batched-gemm":mode==1?"per-head-gemm":"tst-gemv";
    // Same fixed buffers, alternating poison, two repetitions. Beta0 must not
    // depend on old output. Comparing both dumps also detects repeat drift.
    for (int repeat=0;repeat<2;++repeat) {
      ck(cudaMemset(qk.p,repeat?0xa5:0x5a,qk.n*2));ck(cudaMemset(pv.p,repeat?0xa5:0x5a,pv.n*2));
      if (mode==0) {
        blas(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,r,d,&alpha,
            kb,CUDA_R_16BF,d,int64_t(n)*d,qb,CUDA_R_16BF,d,int64_t(r)*d,&beta,
            cb,CUDA_R_16BF,n,int64_t(r)*n,h,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        blas(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,d,r,n,&alpha,
            vb,CUDA_R_16BF,d,int64_t(n)*d,pb,CUDA_R_16BF,n,int64_t(r)*n,&beta,
            ob,CUDA_R_16BF,d,int64_t(r)*d,h,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      } else if (mode==1) {
        for (int head=0;head<h;++head) {
          blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,r,d,&alpha,
              kb+size_t(head)*n*d,CUDA_R_16BF,d,qb+size_t(head)*r*d,CUDA_R_16BF,d,&beta,
              cb+size_t(head)*r*n,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
          blas(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,d,r,n,&alpha,
              vb+size_t(head)*n*d,CUDA_R_16BF,d,pb+size_t(head)*r*n,CUDA_R_16BF,n,&beta,
              ob+size_t(head)*r*d,CUDA_R_16BF,d,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
      } else {
        blas(cublasTSTgemvStridedBatched(handle,CUBLAS_OP_T,d,n,&alpha,kb,d,int64_t(n)*d,
             qb,1,d,&beta,cb,1,n,h));
        blas(cublasTSTgemvStridedBatched(handle,CUBLAS_OP_N,d,n,&alpha,vb,d,int64_t(n)*d,
             pb,1,n,&beta,ob,1,d,h));
      }
      const std::string suffix=label+"-"+std::to_string(repeat);
      qk.write(out+"/"+suffix+"-qk.bf16");pv.write(out+"/"+suffix+"-pv.bf16");
    }
  }
  q.immutable();k.immutable();v.immutable();p.immutable();workspace.host();
  std::cout<<"{\"cublas_version\":"<<version<<",\"guards\":true,\"inputs_immutable\":true}\n";
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; } }
