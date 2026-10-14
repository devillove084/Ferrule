// Byte-for-byte stability outside the new single-row column-reduction regime.
#include "core/recurrent_bf16.cuh"
#include "prior.cuh"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ferrule::cuda::core;
void ck(cudaError_t x) {
  if (x != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(x));
}
uint16_t word(float x) {
  uint32_t b;
  memcpy(&b, &x, 4);
  return uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
template <class T> T *upload(std::vector<T> v) {
  T *p;
  ck(cudaMalloc(&p, v.size() * sizeof(T)));
  ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
int main(int argc, char **argv) {
  try {
    const bool list_only = argc == 2 && std::string(argv[1]) == "--list-cases";
    if (argc != 1 && !list_only)
      throw std::runtime_error("usage: stability [--list-cases]");
    int below_threshold = 0;
    int count = 0;
    size_t elements = 0;
    for (int rows : {1, 2, 5, 65})
      for (int dv : {4, 28, 32, 33})
        for (int dk : {63, 64, 65, 128, 511, 512, 513}) {
          int kh = 2, hv = 4, kw = kh * dk, vw = hv * dv, width = 2 * kw + vw;
          if (rows == 1 && dv % 4 == 0 && dk >= 64 && dk <= 512 && vw >= 128)
            continue;
          if (rows == 1 && dv % 4 == 0 && dk >= 64 && dk <= 512 && vw < 128)
            ++below_threshold;
          if (list_only) {
            std::cout << rows << " " << hv << " " << dk << " " << dv << "\n";
            ++count;
            continue;
          }
          std::vector<uint16_t> q(rows * width), g(rows * hv), p(hv);
          std::vector<float> s(hv * dk * dv);
          for (size_t i = 0; i < q.size(); ++i)
            q[i] = word((int(i * 17 % 211) - 105) / 256.0f);
          for (size_t i = 0; i < g.size(); ++i)
            g[i] = word((int(i % 23) - 11) / 8.0f);
          for (size_t i = 0; i < s.size(); ++i)
            s[i] = (int(i * 13 % 173) - 86) / 997.0f;
          auto qd = upload(q), gd = upload(g), pd = upload(p),
               o1 = upload(std::vector<uint16_t>(rows * vw)),
               o2 = upload(std::vector<uint16_t>(rows * vw));
          auto s1 = upload(s), s2 = upload(s);
          FerruleCoreBf16DeltaArgs a{uint32_t(rows), uint32_t(kh),
                                     uint32_t(hv),   uint32_t(dk),
                                     uint32_t(dv),   0,
                                     uint64_t(qd),   uint64_t(gd),
                                     uint64_t(gd),   uint64_t(pd),
                                     uint64_t(pd),   uint64_t(s1),
                                     uint64_t(o1),   0};
          ck(cudaError_t(ferrule_core_bf16_delta_launch(&a)));
          a.state_f32 = uint64_t(s2);
          a.output_bf16 = uint64_t(o2);
          ck(cudaError_t(prior_core_bf16_delta_launch(&a)));
          std::vector<float> x(s.size()), y(s.size());
          std::vector<uint16_t> u(rows * vw), v(rows * vw);
          ck(cudaMemcpy(x.data(), s1, s.size() * 4, cudaMemcpyDeviceToHost));
          ck(cudaMemcpy(y.data(), s2, s.size() * 4, cudaMemcpyDeviceToHost));
          ck(cudaMemcpy(u.data(), o1, u.size() * 2, cudaMemcpyDeviceToHost));
          ck(cudaMemcpy(v.data(), o2, v.size() * 2, cudaMemcpyDeviceToHost));
          if (memcmp(x.data(), y.data(), x.size() * 4) || u != v)
            throw std::runtime_error("non-selected arithmetic changed");
          for (void *p : {(void *)qd, (void *)gd, (void *)pd, (void *)o1,
                          (void *)o2, (void *)s1, (void *)s2})
            ck(cudaFree(p));
          ++count;
          elements += s.size() + u.size();
        }
    if (count != 107 || below_threshold != 10)
      throw std::runtime_error("missing excluded-shape coverage");
    std::cout << (list_only ? "listed (no CUDA execution): cases="
                            : "prefill/continuation/non-selected baseline "
                              "bit-exact: cases=")
              << count << " below-threshold=" << below_threshold
              << " state+output=" << elements << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
