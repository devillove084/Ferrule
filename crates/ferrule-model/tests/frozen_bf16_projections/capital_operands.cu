#include "cutlass/bf16_linear.cuh"
#include <cstring>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static void ck(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
static void require(bool v, const char *s) {
  if (!v)
    throw std::runtime_error(s);
}
template <class T> std::vector<T> read(const std::string &p, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(p, std::ios::binary);
  require(bool(f.read((char *)v.data(), n * sizeof(T))) && f.peek() == EOF,
          "extent");
  return v;
}
struct Buffer {
  void *p;
  Buffer(size_t n) { ck(cudaMalloc(&p, n)); }
  ~Buffer() { cudaFree(p); }
  uint64_t addr() { return reinterpret_cast<uint64_t>(p); }
};
static uint16_t rne(float f) {
  uint32_t b;
  memcpy(&b, &f, 4);
  return (b + 0x7fff + ((b >> 16) & 1)) >> 16;
}
int main(int argc, char **argv) {
  try {
    require(argc == 2, "fixture directory");
    std::string dir = argv[1], id;
    int m, n, k;
    std::ifstream cases(dir + "/cases.txt");
    int runtime;
    ck(cudaRuntimeGetVersion(&runtime));
    require(runtime == 12040, "CUDA12.4 required");
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0));
    require(prop.major == 8 && prop.minor == 6, "sm86 required");
    cublasHandle_t handle;
    require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "handle");
    int version;
    require(cublasGetVersion(handle, &version) == CUBLAS_STATUS_SUCCESS &&
                version == 120405,
            "cuBLAS12.4.5 required");
    require(cublasSetWorkspace(handle, nullptr, 0) == CUBLAS_STATUS_SUCCESS,
            "workspace");
    size_t cases_count = 0, input_words = 0, decoded_words = 0;
    while (cases >> id >> m >> n >> k) {
      auto x = read<uint16_t>(dir + "/" + id + ".input.bf16", m * k),
           d = read<uint16_t>(dir + "/" + id + ".decoded.bf16", n * k),
           hf = read<uint16_t>(dir + "/" + id + ".hf.bf16", m * n),
           sc = read<uint16_t>(dir + "/" + id + ".scale.bf16", n * k / 16384);
      auto w = read<uint8_t>(dir + "/" + id + ".weight.fp8", n * k);
      Buffer dx(x.size() * 2), dw(w.size()), ds(sc.size() * 2),
          ws((m + 257) * k * 2), out(m * n * 4), outbf(m * n * 2),
          full(d.size() * 2);
      ck(cudaMemcpy(full.p, d.data(), d.size() * 2, cudaMemcpyHostToDevice));
      ck(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
      ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice));
      ck(cudaMemcpy(ds.p, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
      FerruleCutlassNumericFp8Args a{};
      a.m = m;
      a.n = n;
      a.k = a.padded_k = a.lda = k;
      a.ldd = n;
      a.tile_rows = 257;
      a.scale_cols = k / 128;
      a.scale_bytes = 2;
      a.activation = dx.addr();
      a.weight = dw.addr();
      a.scales = ds.addr();
      a.workspace = ws.addr();
      a.workspace_bytes = (m + 257) * k * 2;
      a.output = out.addr();
      using namespace ferrule::cuda::cutlass::numeric_fp8;
      pack_bf16_activation<<<blocks(m * k), 256>>>(a);
      ck(cudaGetLastError());
      std::vector<uint16_t> packed(m * k);
      ck(cudaMemcpy(packed.data(), ws.p, m * k * 2, cudaMemcpyDeviceToHost));
      require(packed == x, "packed input differs");
      input_words += x.size();
      for (int first = 0; first < n; first += 257) {
        int count = std::min(257, n - first);
        decode_weights<<<blocks(count * k), 256>>>(a, first, count);
        ck(cudaGetLastError());
        std::vector<uint16_t> got(count * k);
        ck(cudaMemcpy(got.data(), (char *)ws.p + m * k * 2, count * k * 2,
                      cudaMemcpyDeviceToHost));
        require(std::equal(got.begin(), got.end(), d.begin() + first * k),
                "decoded differs");
        decoded_words += got.size();
        if (m == 1) {
          float alpha = 1, beta = 0;
          require(cublasTSSgemvStridedBatched(
                      handle, CUBLAS_OP_T, k, count, &alpha,
                      (const __nv_bfloat16 *)((char *)ws.p + k * 2), k, 0,
                      (const __nv_bfloat16 *)ws.p, 1, 0, &beta,
                      (float *)out.p + first, 1, 0, 1) == CUBLAS_STATUS_SUCCESS,
                  "gemv");
          require(cublasTSTgemvStridedBatched(
                      handle, CUBLAS_OP_T, k, count, &alpha,
                      (const __nv_bfloat16 *)((char *)ws.p + k * 2), k, 0,
                      (const __nv_bfloat16 *)ws.p, 1, 0, &beta,
                      (__nv_bfloat16 *)outbf.p + first, 1, 0,
                      1) == CUBLAS_STATUS_SUCCESS,
                  "gemv bf16");
        }
      }
      if (m == 1) {
        std::vector<float> y(n);
        std::vector<uint16_t> bf(n);
        ck(cudaMemcpy(y.data(), out.p, n * 4, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(bf.data(), outbf.p, n * 2, cudaMemcpyDeviceToHost));
        int diff = 0, bd = 0;
        for (int i = 0; i < n; ++i) {
          diff += rne(y[i]) != hf[i];
          bd += bf[i] != hf[i];
        }
        float alpha = 1, beta = 0;
        require(cublasTSTgemvStridedBatched(handle, CUBLAS_OP_T, k, n, &alpha,
                                            (const __nv_bfloat16 *)full.p, k, 0,
                                            (const __nv_bfloat16 *)dx.p, 1, 0,
                                            &beta, (__nv_bfloat16 *)outbf.p, 1,
                                            0, 1) == CUBLAS_STATUS_SUCCESS,
                "full bf16");
        require(cublasTSSgemvStridedBatched(handle, CUBLAS_OP_T, k, n, &alpha,
                                            (const __nv_bfloat16 *)full.p, k, 0,
                                            (const __nv_bfloat16 *)dx.p, 1, 0,
                                            &beta, (float *)out.p, 1, 0,
                                            1) == CUBLAS_STATUS_SUCCESS,
                "full f32");
        ck(cudaMemcpy(y.data(), out.p, n * 4, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(bf.data(), outbf.p, n * 2, cudaMemcpyDeviceToHost));
        int fd = 0, fb = 0;
        for (int i = 0; i < n; ++i) {
          fd += rne(y[i]) != hf[i];
          fb += bf[i] != hf[i];
        }
        std::cout << "{\"case\":\"" << id << "\",\"tss_tiled\":" << diff
                  << ",\"tst_tiled\":" << bd << ",\"tss_full\":" << fd
                  << ",\"tst_full\":" << fb << "}\n";
      }
      ++cases_count;
    }
    require(cases_count == 114, "coverage");
    require(cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS, "destroy");
    std::cout << "{\"cases\":" << cases_count
              << ",\"input_words_equal\":" << input_words
              << ",\"decoded_words_equal\":" << decoded_words << "}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
