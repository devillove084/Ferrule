// Standalone call-contract comparison: no Ferrule arithmetic or FP8 decode.
// Inputs are independently frozen decoded BF16 operands. Expected is native
// BF16.
#include <algorithm>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static void ck(cudaError_t s) {
  if (s != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(s));
}
static void cb(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(s));
}
static void req(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
template <class T> std::vector<T> read(const std::string &p, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(p, std::ios::binary);
  req(bool(f.read((char *)v.data(), n * sizeof(T))) && f.peek() == EOF,
      "fixture extent");
  return v;
}
struct Buffer {
  void *p;
  size_t bytes;
  Buffer(size_t n) : bytes(n) {
    ck(cudaMalloc(&p, n + 256));
    ck(cudaMemset(p, 0x5a, n + 256));
  }
  ~Buffer() { cudaFree(p); }
  void guards() {
    std::vector<unsigned char> v(256);
    ck(cudaMemcpy(v.data(), (char *)p + bytes, 256, cudaMemcpyDeviceToHost));
    req(std::all_of(v.begin(), v.end(), [](auto x) { return x == 0x5a; }),
        "guard");
  }
};
struct Config {
  const char *name;
  int api, math, atomics;
  size_t workspace;
};
static uint16_t rne(float x) {
  uint32_t b;
  memcpy(&b, &x, 4);
  return (b + 0x7fff + ((b >> 16) & 1)) >> 16;
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--describe") {
      std::cout << "{\"gpu_execution\":false,\"repeats\":16,\"primary\":"
                   "\"GemmEx BF16 A/B/C F32compute math16 "
                   "workspace32MiB\",\"oracle\":\"frozen actual BF16 only\"}\n";
      return 0;
    }
    req(argc == 4, "fixture output-dir expected-cuBLAS-version");
    const std::string dir = argv[1], out = argv[2];
    int expected = std::stoi(argv[3]);
    cublasHandle_t h;
    cb(cublasCreate(&h));
    int version, runtime;
    cb(cublasGetVersion(h, &version));
    ck(cudaRuntimeGetVersion(&runtime));
    req(version == expected, "cuBLAS version differs from plan");
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0));
    req(prop.major == 8 && prop.minor == 6, "sm86 required");
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cb(cublasSetStream(h, stream));
    cb(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST));
    Buffer workspace(32 << 20);
    const Config configs[] = {{"gemmex-matched", 0, 16, 0, 32 << 20},
                              {"gemmex-no-workspace", 0, 16, 0, 0},
                              {"gemmex-default-math", 0, 0, 0, 32 << 20},
                              {"gemmex-atomics-allowed", 0, 16, 1, 32 << 20},
                              {"tst-matched", 1, 16, 0, 32 << 20},
                              {"tst-prior", 1, 0, 0, 0},
                              {"tss-matched-diagnostic", 2, 16, 0, 32 << 20},
                              {"gemmex-f32-diagnostic", 3, 16, 0, 32 << 20}};
    std::ifstream cases(dir + "/cases.txt");
    std::string id;
    int n, k, count = 0;
    std::cout << "{\"event\":\"runtime\",\"cublas\":" << version
              << ",\"cuda\":" << runtime << ",\"sm\":86}\n";
    while (cases >> id >> n >> k) {
      req((n == 2048 && k == 512) || (n == 1024 && k == 2048),
          "unsupported fixture geometry");
      auto x = read<uint16_t>(dir + "/" + id + ".input.bf16", k),
           w = read<uint16_t>(dir + "/" + id + ".decoded.bf16", size_t(n) * k),
           hf = read<uint16_t>(dir + "/" + id + ".hf.bf16", n);
      Buffer dx(k * 2), dw(size_t(n) * k * 2), dy(n * 4);
      ck(cudaMemcpy(dx.p, x.data(), k * 2, cudaMemcpyHostToDevice));
      ck(cudaMemcpy(dw.p, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
      ck(cudaStreamSynchronize(nullptr));
      for (const auto &cfg : configs) {
        cb(cublasSetMathMode(h, cublasMath_t(cfg.math)));
        cb(cublasSetAtomicsMode(h, cublasAtomicsMode_t(cfg.atomics)));
        cb(cublasSetWorkspace(h, cfg.workspace ? workspace.p : nullptr,
                              cfg.workspace));
        std::vector<uint16_t> first;
        int max_diff = 0, drift = 0;
        for (int r = 0; r < 16; ++r) {
          ck(cudaMemsetAsync(dy.p, r % 2 ? 0xa5 : 0, n * 4, stream));
          float alpha = 1, beta = 0;
          if (cfg.api == 0 || cfg.api == 3)
            cb(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k, &alpha, dw.p,
                            CUDA_R_16BF, k, dx.p, CUDA_R_16BF, k, &beta, dy.p,
                            cfg.api == 0 ? CUDA_R_16BF : CUDA_R_32F, n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
          else if (cfg.api == 1)
            cb(cublasTSTgemvStridedBatched(
                h, CUBLAS_OP_T, k, n, &alpha, (const __nv_bfloat16 *)dw.p, k, 0,
                (const __nv_bfloat16 *)dx.p, 1, 0, &beta, (__nv_bfloat16 *)dy.p,
                1, 0, 1));
          else
            cb(cublasTSSgemvStridedBatched(h, CUBLAS_OP_T, k, n, &alpha,
                                           (const __nv_bfloat16 *)dw.p, k, 0,
                                           (const __nv_bfloat16 *)dx.p, 1, 0,
                                           &beta, (float *)dy.p, 1, 0, 1));
          ck(cudaStreamSynchronize(stream));
          std::vector<uint16_t> words(n);
          if (cfg.api >= 2) {
            std::vector<float> raw(n);
            ck(cudaMemcpy(raw.data(), dy.p, n * 4, cudaMemcpyDeviceToHost));
            std::transform(raw.begin(), raw.end(), words.begin(), rne);
          } else
            ck(cudaMemcpy(words.data(), dy.p, n * 2, cudaMemcpyDeviceToHost));
          if (r == 0)
            first = words;
          int unequal = 0;
          for (int i = 0; i < n; ++i)
            unequal += words[i] != hf[i];
          max_diff = std::max(max_diff, unequal);
          drift += words != first;
          std::ofstream f(out + "/" + id + "-" + cfg.name + "-" +
                              std::to_string(r) + ".bf16",
                          std::ios::binary);
          req(bool(f.write((char *)words.data(), n * 2)), "write");
          dy.guards();
        }
        dx.guards();
        dw.guards();
        auto xx = std::vector<uint16_t>(k),
             ww = std::vector<uint16_t>(w.size());
        ck(cudaMemcpy(xx.data(), dx.p, k * 2, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(ww.data(), dw.p, w.size() * 2, cudaMemcpyDeviceToHost));
        req(xx == x && ww == w, "operand modified");
        std::cout << "{\"case\":\"" << id << "\",\"config\":\"" << cfg.name
                  << "\",\"n\":" << n << ",\"k\":" << k
                  << ",\"a_mod256\":" << uintptr_t(dw.p) % 256
                  << ",\"b_mod256\":" << uintptr_t(dx.p) % 256
                  << ",\"c_mod256\":" << uintptr_t(dy.p) % 256
                  << ",\"repeats\":16,\"max_unequal_hf\":" << max_diff
                  << ",\"runs_different_from_first\":" << drift << "}\n";
      }
      ++count;
    }
    req(count == 8, "fixture cases");
    workspace.guards();
    cb(cublasDestroy(h));
    ck(cudaStreamDestroy(stream));
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
