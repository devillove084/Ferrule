// Test-only trace specialization of the production arithmetic. No model capture.
#include "core/router_full_softmax.cuh"
#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ferrule::cuda::core;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T> std::vector<T> read(const std::string &path, size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() != std::streamoff(count * sizeof(T))) throw std::runtime_error(path);
  std::vector<T> values(count);
  f.seekg(0); f.read(reinterpret_cast<char *>(values.data()), count * sizeof(T));
  if (!f) throw std::runtime_error(path);
  return values;
}
template <class T> void dump(const std::string &path, T *device, size_t count) {
  std::vector<T> data(count);
  check(cudaMemcpy(data.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(data.data()), count * sizeof(T));
  if (!f) throw std::runtime_error(path);
}
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  if (std::string(argv[1]) == "--preflight") {
    FerruleCoreRouterArgs a{};
    a.kind = FERRULE_CORE_ROUTER_FULL_SOFTMAX_TOPK;
    a.rows = 5; a.columns = 256; a.k = 8; a.route_scale = 1;
    a.logits = 0x1000; a.indices = 0x10000; a.weights = 0x20000;
    assert(full_router_valid(a));
    auto bad = a; bad.indices = a.logits; assert(!full_router_valid(bad));
    bad = a; bad.weights = a.indices + 4; assert(!full_router_valid(bad));
    bad = a; bad.logits = UINT64_MAX - 3; assert(!full_router_valid(bad));
    bad = a; bad.indices += 1; assert(!full_router_valid(bad));
    bad = a; bad.columns = 1025; assert(!full_router_valid(bad));
    bad = a; bad.k = 33; assert(!full_router_valid(bad));
    bad = a; bad.rows = 0; assert(!full_router_valid(bad));
    bad = a; bad.flags = 1; assert(!full_router_valid(bad));
    bad = a; bad.weights = 0; assert(!full_router_valid(bad));
    bad = a; bad.route_scale = NAN; assert(!full_router_valid(bad));
    std::cout << "native preflight: 1 valid + 10 rejected; no CUDA API calls\n";
    return 0;
  }
  try {
    const std::string dir = argv[1];
    std::ifstream cases(dir + "/oracle-cases.txt");
    if (!cases) cases.open(dir + "/cases.txt");
    if (!cases) return 2;
    std::string name; uint32_t m, n, k; size_t count = 0;
    while (cases >> name >> m >> n >> k) {
      auto values = read<float>(dir + "/" + name + ".raw.f32", size_t(m) * n);
      float *raw, *weights, *trace; int32_t *ids;
      size_t traces = size_t(m) * (2 * n + k + 3);
      check(cudaMalloc(&raw, values.size() * 4));
      check(cudaMalloc(&weights, size_t(m) * k * 4));
      check(cudaMalloc(&ids, size_t(m) * k * 4));
      check(cudaMalloc(&trace, traces * 4));
      check(cudaMemcpy(raw, values.data(), values.size() * 4, cudaMemcpyHostToDevice));
      FerruleCoreRouterArgs a{};
      a.kind = FERRULE_CORE_ROUTER_FULL_SOFTMAX_TOPK;
      a.rows = m; a.columns = n; a.k = k; a.route_scale = 1;
      a.logits = reinterpret_cast<uint64_t>(raw);
      a.indices = reinterpret_cast<uint64_t>(ids);
      a.weights = reinterpret_cast<uint64_t>(weights);
      if (!full_router_valid(a)) throw std::runtime_error("descriptor rejected");
      launch_full_router<true>(a, trace);
      check(cudaGetLastError()); check(cudaDeviceSynchronize());
      dump(dir + "/" + name + ".trace.f32", trace, traces);
      dump(dir + "/" + name + ".probe.weights.f32", weights, size_t(m) * k);
      dump(dir + "/" + name + ".probe.ids.i32", ids, size_t(m) * k);
      dump(dir + "/" + name + ".probe.raw.f32", raw, values.size());
      check(cudaFree(trace)); check(cudaFree(ids)); check(cudaFree(weights)); check(cudaFree(raw));
      std::cout << name << " trace_words=" << traces << " allocations_released=4\n";
      ++count;
    }
    check(cudaDeviceSynchronize()); check(cudaDeviceReset());
    std::cout << "cases=" << count << " reset=true\n";
    return count ? 0 : 2;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
