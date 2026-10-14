// Bounded matrix-only regression of the actual production chunk matmul.
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static void ck(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
#include "core/recurrent_bf16_chunk.cuh"
int main(int argc, char **argv) {
  try {
    if (argc != 5)
      throw std::runtime_error("fixture directory N transpose_rhs batches");
    std::string dir = argv[1];
    const uint32_t n = std::stoul(argv[2]), batches = std::stoul(argv[4]);
    const bool tb = std::stoul(argv[3]) != 0;
    const size_t count = uint64_t(batches) * 64 * n;
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0));
    int runtime;
    ck(cudaRuntimeGetVersion(&runtime));
    if (runtime != 12040 || prop.major != 8 || prop.minor != 6)
      throw std::runtime_error("CUDA12.4/sm86 required");
    auto read = [&](std::string name, size_t elements) {
      std::vector<float> h(elements);
      std::ifstream f(dir + "/" + name, std::ios::binary);
      if (!f.read((char *)h.data(), h.size() * 4) || f.peek() != EOF)
        throw std::runtime_error("input extent");
      return h;
    };
    auto a = read("lhs.f32", uint64_t(batches) * 64 * 512),
         b = read("rhs-transposed.f32", uint64_t(batches) * 512 * n);
    float *x, *y, *z;
    ck(cudaMalloc(&x, a.size() * 4));
    ck(cudaMalloc(&y, b.size() * 4));
    ck(cudaMalloc(&z, (count + 32) * 4));
    ck(cudaMemcpy(x, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(y, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemset(z, 0x5a, (count + 32) * 4));
    ferrule::cuda::core::bf16_chunk::
        matmul<<<dim3((n + 15) / 16, 4, batches), dim3(16, 16)>>>(
            x, y, z + 16, 64, n, 512, false, tb, batches);
    ck(cudaGetLastError());
    std::vector<float> out(count + 32);
    ck(cudaMemcpy(out.data(), z, out.size() * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 64; ++i)
      if (((unsigned char *)out.data())[i] != 0x5a ||
          ((unsigned char *)(out.data() + count + 16))[i] != 0x5a)
        throw std::runtime_error("guard");
    std::ofstream f(dir + "/candidate.f32", std::ios::binary);
    f.write((char *)(out.data() + 16), count * 4);
    if (!f)
      throw std::runtime_error("write");
    ck(cudaFree(x));
    ck(cudaFree(y));
    ck(cudaFree(z));
    std::cout << "matrix candidate complete, guards intact\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
