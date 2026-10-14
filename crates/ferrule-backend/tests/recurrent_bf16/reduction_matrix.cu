#include "core/recurrent_bf16.cuh"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ferrule::cuda::core;
void ck(cudaError_t s) {
  if (s != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(s));
}
template <class T> std::vector<T> read(std::string p, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(p, std::ios::binary);
  if (!f.read(reinterpret_cast<char *>(v.data()), n * sizeof(T)) ||
      f.peek() != EOF)
    throw std::runtime_error("extent " + p);
  return v;
}
__global__ void reduce(const float *s, uint64_t q, float *o, int hv, int dk,
                       int dv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hv * dv)
    o[i] = hf_bf16::column_sum4(s + (i / dv) * dk * dv + i % dv, q, 0, 1.0f,
                                1.0f, dk, dv);
}
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      return 2;
    std::string dir = argv[1];
    std::ifstream list(dir + "/cases.txt");
    int hv, dk, dv, n = 0;
    size_t compared = 0;
    while (list >> hv >> dk >> dv) {
      std::string p = dir + "/" + std::to_string(n);
      auto q = read<uint16_t>(p + "-q.bf16", dk);
      auto s = read<float>(p + "-state.f32", size_t(hv) * dk * dv);
      auto expected = read<float>(p + "-expected.f32", hv * dv);
      uint16_t *qd;
      float *sd, *od;
      ck(cudaMalloc(&qd, dk * 2));
      ck(cudaMalloc(&sd, s.size() * 4));
      ck(cudaMalloc(&od, expected.size() * 4));
      ck(cudaMemcpy(qd, q.data(), dk * 2, cudaMemcpyHostToDevice));
      ck(cudaMemcpy(sd, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
      reduce<<<(hv * dv + 127) / 128, 128>>>(sd, reinterpret_cast<uint64_t>(qd),
                                             od, hv, dk, dv);
      ck(cudaGetLastError());
      std::vector<float> result(expected.size());
      ck(cudaMemcpy(result.data(), od, result.size() * 4,
                    cudaMemcpyDeviceToHost));
      if (memcmp(result.data(), expected.data(), result.size() * 4)) {
        std::cerr << "mismatch case=" << n << " hv=" << hv << " dk=" << dk
                  << " dv=" << dv << "\n";
        return 1;
      }
      ck(cudaFree(qd));
      ck(cudaFree(sd));
      ck(cudaFree(od));
      compared += result.size();
      ++n;
    }
    if (n != 55)
      throw std::runtime_error("incomplete cases");
    std::cout << "exact F32 HF CUDA reductions: cases=" << n
              << " elements=" << compared << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
