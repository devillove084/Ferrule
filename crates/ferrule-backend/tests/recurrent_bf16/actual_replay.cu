// Standalone bounded operator test, built with CUDA12.4; no Torch runtime.
#include "core/recurrent_bf16.cuh"
#include "observed.cuh"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ferrule::cuda::core;
static void check(cudaError_t s) {
  if (s != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(s));
}
template <class T> struct Buffer {
  T *p;
  size_t n;
  Buffer(size_t size) : n(size) {
    check(cudaMalloc(&p, (n + 16) * sizeof(T)));
    check(cudaMemset(p, 0, (n + 16) * sizeof(T)));
    check(cudaMemset(p + n, 0x5a, 16 * sizeof(T)));
  }
  ~Buffer() { cudaFree(p); }
  uint64_t addr() const { return reinterpret_cast<uint64_t>(p); }
  void read(std::string path) {
    std::vector<T> host(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(host.data()), n * sizeof(T)) ||
        f.peek() != EOF)
      throw std::runtime_error("input extent " + path);
    check(cudaMemcpy(p, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> host() {
    std::vector<T> data(n + 16);
    check(cudaMemcpy(data.data(), p, (n + 16) * sizeof(T),
                     cudaMemcpyDeviceToHost));
    auto *bytes = reinterpret_cast<unsigned char *>(data.data() + n);
    for (size_t i = 0; i < 16 * sizeof(T); ++i)
      if (bytes[i] != 0x5a)
        throw std::runtime_error("canary overwritten");
    data.resize(n);
    return data;
  }
  void write(std::string path) {
    auto data = host();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<char *>(data.data()), n * sizeof(T));
    if (!f)
      throw std::runtime_error("write " + path);
  }
};
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("directory label");
    std::string dir = argv[1], label = dir + "/" + argv[2];
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 8 || prop.minor != 6)
      throw std::runtime_error("requires sm86");
    int version;
    check(cudaRuntimeGetVersion(&version));
    if (version != 12040)
      throw std::runtime_error("requires CUDA12.4");
    Buffer<uint16_t> raw(8192), history(32768), weight(32768), conv(8192),
        a(32), b(32), al(32), dt(32), output(4096), observed(4096);
    Buffer<float> state(524288), state2(524288), trace(10 * 524288);
    raw.read(dir + "/raw.bf16");
    history.read(dir + "/history.bf16");
    weight.read(dir + "/weight.bf16");
    a.read(dir + "/a.bf16");
    b.read(dir + "/b.bf16");
    al.read(dir + "/a_log.bf16");
    dt.read(dir + "/dt_bias.bf16");
    state.read(dir + "/state.f32");
    state2.read(dir + "/state.f32");
    FerruleCoreBf16ConvArgs c{
        1, 8192,           4,           0, raw.addr(), weight.addr(),
        0, history.addr(), conv.addr(), 0};
    check(cudaError_t(ferrule_core_bf16_conv_launch(&c)));
    FerruleCoreBf16DeltaArgs d{1,
                               16,
                               32,
                               128,
                               128,
                               0,
                               conv.addr(),
                               a.addr(),
                               b.addr(),
                               al.addr(),
                               dt.addr(),
                               state.addr(),
                               output.addr(),
                               0};
    check(cudaError_t(ferrule_core_bf16_delta_launch(&d)));
    d.state_f32 = state2.addr();
    d.output_bf16 = observed.addr();
    hf_bf16::delta_observed<<<32, 128>>>(d, trace.p);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    auto s = state.host(), s2 = state2.host();
    if (memcmp(s.data(), s2.data(), s.size() * 4) ||
        output.host() != observed.host())
      throw std::runtime_error(
          "instrumentation changed production output/state");
    raw.host();
    weight.host();
    a.host();
    b.host();
    al.host();
    dt.host();
    conv.write(label + "-conv.bf16");
    history.write(label + "-history.bf16");
    state.write(label + "-state.f32");
    output.write(label + "-output.bf16");
    trace.write(label + "-trace.f32");
    std::cout << "CUDA runtime=" << version << " sm=" << prop.major
              << prop.minor
              << " instrumentation_output_state_exact=true canaries=true "
                 "output2603=0x"
              << std::hex << output.host()[2603] << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
