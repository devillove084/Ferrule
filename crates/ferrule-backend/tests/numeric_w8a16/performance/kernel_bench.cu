// Standalone native-entry microbenchmark. See README.md for bounded invocation.
#include "cutlass/bf16_linear.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(expr)                                                              \
  do {                                                                        \
    auto error = (expr);                                                       \
    if (error != cudaSuccess) {                                               \
      fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(error));            \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// Independent host oracle: arithmetic FP8 decode and discarded-bit BF16 RNE.
static uint16_t round_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint16_t hi = u >> 16;
  uint32_t lo = u & 65535;
  return std::isnan(f) ? hi | 64
                      : hi + (lo > 32768 || (lo == 32768 && (hi & 1)));
}
static float expand(uint16_t v) {
  uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
static float decode(uint8_t c) {
  int e = (c >> 3) & 15, mant = c & 7;
  float f = e ? std::ldexp(float(8 + mant), e - 10) : float(mant) / 512;
  return c & 128 ? -f : f;
}
template <class T> static uint64_t upload(const std::vector<T> &v) {
  void *p;
  CK(cudaMalloc(&p, v.size() * sizeof(T)));
  CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return reinterpret_cast<uint64_t>(p);
}
using Entry = int32_t (*)(const FerruleCutlassNumericFp8Args *);
static void launch(Entry f, const FerruleCutlassNumericFp8Args &a) {
  int status = f(&a);
  if (status) {
    fprintf(stderr, "native status %d\n", status);
    std::exit(1);
  }
}
static void run(int m, int n, int k, bool ragged, bool f32scale, bool bfinput,
                bool fp32, int iterations) {
  FerruleCutlassNumericFp8Args a{};
  a.m = m;
  a.n = n;
  a.k = a.padded_k = k;
  a.lda = k + 3;
  a.ldd = n + 5;
  a.row_origin = ragged ? 127 : 0;
  a.column_origin = ragged ? 125 : 0;
  a.scale_cols = (a.column_origin + k + 127) / 128;
  a.scale_bytes = f32scale ? 4 : 2;
  std::vector<uint8_t> w(size_t(n) * k);
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] = uint8_t((i * 73 + 19) % 256);
    if ((w[i] & 127) == 127) w[i] ^= 1;
  }
  std::vector<float> x(size_t(m) * a.lda);
  std::vector<float> scales(((a.row_origin + n + 127) / 128) * a.scale_cols);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = float(int((i * 17) % 73) - 36) * 0.01317f;
  for (size_t i = 0; i < scales.size(); ++i)
    scales[i] = float(1 + i % 23) * 0.00731f;
  std::vector<uint16_t> bs(scales.size()), bx(x.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    bs[i] = round_bf16(scales[i]);
    if (!f32scale) scales[i] = expand(bs[i]);
  }
  for (size_t i = 0; i < x.size(); ++i) bx[i] = round_bf16(x[i]);
  std::vector<float> out(size_t(m) * a.ldd + 7, 7331.f);
  a.activation = bfinput ? upload(bx) : upload(x);
  a.weight = upload(w);
  a.scales = f32scale ? upload(scales) : upload(bs);
  a.output = upload(out);
  Entry entry = fp32 ? ferrule_cutlass_numeric_fp8_m1_launch
                    : bfinput ? ferrule_cutlass_numeric_fp8_w8a16_bf16_launch
                              : ferrule_cutlass_numeric_fp8_w8a16_launch;
  launch(entry, a);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(), reinterpret_cast<void *>(a.output), out.size() * 4,
                cudaMemcpyDeviceToHost));
  double max_ratio = 0;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      // All columns for ragged; representative columns for checkpoint shapes.
      if (!ragged && col >= 8 && col < n - 8 && col % 127 != 0) continue;
      double sum = 0, norm = 0;
      for (int j = 0; j < k; ++j) {
        float weight = decode(w[size_t(col) * k + j]) *
            scales[((a.row_origin + col) / 128) * a.scale_cols +
                   (a.column_origin + j) / 128];
        float act = x[size_t(row) * a.lda + j];
        if (!fp32) {
          weight = expand(round_bf16(weight));
          act = expand(bx[size_t(row) * a.lda + j]);
        }
        double product = double(weight) * act;
        sum += product;
        norm += std::abs(product);
      }
      float actual = out[size_t(row) * a.ldd + col];
      double err = std::abs(double(actual) - sum), tol = 1e-5 + 2e-6 * norm;
      max_ratio = std::max(max_ratio, err / tol);
      if (!std::isfinite(actual) || err > tol) {
        fprintf(stderr,
                "oracle %dx%dx%d (%d,%d) got %.9g want %.12g ratio %.6g\n",
                m, n, k, row, col, actual, sum, err / tol);
        std::exit(2);
      }
    }
    for (unsigned col = n; col < a.ldd; ++col)
      if (out[size_t(row) * a.ldd + col] != 7331.f) std::exit(3);
  }
  for (size_t i = size_t(m) * a.ldd; i < out.size(); ++i)
    if (out[i] != 7331.f) std::exit(3);

  cudaStream_t stream;
  CK(cudaStreamCreate(&stream));
  a.stream = reinterpret_cast<uint64_t>(stream);
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < iterations; ++i) launch(entry, a);
  CK(cudaStreamEndCapture(stream, &graph));
  CK(cudaGraphInstantiate(&exec, graph, 0));
  for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(exec, stream));
  CK(cudaStreamSynchronize(stream));
  cudaEvent_t start, end;
  CK(cudaEventCreate(&start));
  CK(cudaEventCreate(&end));
  std::vector<float> times;
  for (int repeat = 0; repeat < 7; ++repeat) {
    CK(cudaEventRecord(start, stream));
    CK(cudaGraphLaunch(exec, stream));
    CK(cudaEventRecord(end, stream));
    CK(cudaEventSynchronize(end));
    float ms;
    CK(cudaEventElapsedTime(&ms, start, end));
    times.push_back(ms * 1000 / iterations);
  }
  std::sort(times.begin(), times.end());
  printf("%s,%d,%d,%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6g\n",
         fp32 ? "f32-m1" : "w8a16", m, n, k, ragged,
         f32scale ? "f32" : "bf16", bfinput ? "bf16" : "f32", times[3],
         times.front(), times.back(), max_ratio);
  fflush(stdout);
  CK(cudaEventDestroy(start));
  CK(cudaEventDestroy(end));
  CK(cudaGraphExecDestroy(exec));
  CK(cudaGraphDestroy(graph));
  CK(cudaStreamDestroy(stream));
  for (uint64_t p : {a.activation, a.weight, a.scales, a.output})
    CK(cudaFree(reinterpret_cast<void *>(p)));
}
int main(int argc, char **argv) {
  int iterations = argc > 1 ? std::atoi(argv[1]) : 100;
  if (iterations < 1 || iterations > 1000) return 4;
  int runtime;
  CK(cudaRuntimeGetVersion(&runtime));
  if (runtime / 1000 != 12) {
    fprintf(stderr, "requires CUDA12 runtime, got %d\n", runtime);
    return 5;
  }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  fprintf(stderr, "runtime=%d device=%s sms=%d iterations=%d repeats=7 warmup_graphs=3\n",
          runtime, prop.name, prop.multiProcessorCount, iterations);
  puts("kernel,m,n,k,offset_scales,scale,input,median_us,min_us,max_us,max_oracle_ratio");
  const std::vector<std::pair<int, int>> shapes = {
      {512, 2048}, {2048, 512}, {8192, 2048}, {4096, 2048},
      {2048, 4096}, {131, 129}, {517, 2051}};
  for (int m : {1, 2, 4, 8, 9, 32}) {
    for (auto shape : shapes) {
      for (bool fs : {false, true}) {
        for (bool bi : {false, true}) {
          int n = shape.first, k = shape.second;
          bool ragged = k % 128;
          run(m, n, k, ragged, fs, bi, false, iterations);
          if (m == 1 && !bi) run(m, n, k, ragged, fs, false, true, iterations);
        }
      }
    }
  }
}
