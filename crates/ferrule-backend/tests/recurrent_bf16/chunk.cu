#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static std::string trace_dir;
static unsigned trace_chunk = 0;
static bool trace_enabled = false;
static void ck(cudaError_t x) {
  if (x != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(x));
}
static void observe(const char *name, const float *ptr, uint64_t n) {
  if (!trace_enabled)
    return;
  std::vector<float> data(n);
  ck(cudaMemcpy(data.data(), ptr, n * 4, cudaMemcpyDeviceToHost));
  std::ofstream f(trace_dir + "/chunk" + std::to_string(trace_chunk) + "-" +
                      name + ".f32",
                  std::ios::binary);
  f.write(reinterpret_cast<const char *>(data.data()), n * 4);
  if (!f)
    throw std::runtime_error("trace write");
  if (std::string(name) == "matmul7")
    ++trace_chunk;
}
#define FERRULE_GDN_CHUNK_OBSERVE(name, ptr, n) observe(name, ptr, n)
#include "core/recurrent_bf16_chunk.cuh"
using namespace ferrule::cuda::core;
using namespace ferrule::cuda::core::bf16_chunk;
template <class T> struct Buffer {
  T *base, *p;
  size_t n;
  std::vector<T> original;
  Buffer(size_t count) : n(count) {
    ck(cudaMalloc(&base, (n + 32) * sizeof(T)));
    p = base + 16;
    ck(cudaMemset(base, 0x5a, (n + 32) * sizeof(T)));
  }
  ~Buffer() { cudaFree(base); }
  uint64_t addr() const { return reinterpret_cast<uint64_t>(p); }
  void read(const std::string &file) {
    original.resize(n);
    std::ifstream f(file, std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(original.data()), n * sizeof(T)) ||
        f.peek() != EOF)
      throw std::runtime_error("extent " + file);
    ck(cudaMemcpy(p, original.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void reset() {
    ck(cudaMemcpy(p, original.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> host() {
    std::vector<T> data(n + 32);
    ck(cudaMemcpy(data.data(), base, data.size() * sizeof(T),
                  cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < 16 * sizeof(T); ++i)
      if (reinterpret_cast<unsigned char *>(data.data())[i] != 0x5a ||
          reinterpret_cast<unsigned char *>(data.data() + n + 16)[i] != 0x5a)
        throw std::runtime_error("guard");
    return std::vector<T>(data.begin() + 16, data.end() - 16);
  }
  void immutable() {
    auto data = host();
    if (memcmp(data.data(), original.data(), n * sizeof(T)))
      throw std::runtime_error("input mutation");
  }
  void write(const std::string &name) {
    auto data = host();
    std::ofstream f(name, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data.data()), n * sizeof(T));
    if (!f)
      throw std::runtime_error("write");
  }
};
static void require(bool ok, const char *name) {
  if (!ok)
    throw std::runtime_error(name);
}
static void contracts() {
  FerruleCoreBf16ChunkArgs a{};
  a.rows = 5;
  a.key_heads = 16;
  a.value_heads = 32;
  a.key_dim = 128;
  a.value_dim = 128;
  uint64_t n = 0;
  require(requirements(a, &n) && n == 4462592, "workspace formula");
  uint64_t p = 0x10000000;
  a.qkv_bf16 = p;
  p += 1 << 24;
  a.a_bf16 = p;
  p += 1 << 24;
  a.b_bf16 = p;
  p += 1 << 24;
  a.a_log_bf16 = p;
  p += 1 << 24;
  a.dt_bias_bf16 = p;
  p += 1 << 24;
  a.state_f32 = p;
  p += 1 << 24;
  a.output_bf16 = p;
  p += 1 << 24;
  a.workspace_f32 = p;
  a.workspace_elements = n;
  require(validate(a), "valid descriptor");
  auto reject = [](FerruleCoreBf16ChunkArgs b) {
    require(!validate(b), "native rejection");
    require(ferrule_core_bf16_chunk_launch(&b) == cudaErrorInvalidValue,
            "launch preflight");
  };
  auto b = a;
  b.workspace_elements = n - 1;
  reject(b);
  b = a;
  b.workspace_elements = UINT64_MAX;
  reject(b);
  b = a;
  b.workspace_f32 = a.state_f32;
  reject(b);
  b = a;
  b.workspace_f32 = a.qkv_bf16;
  reject(b);
  b = a;
  b.output_bf16 = a.a_bf16;
  reject(b);
  b = a;
  b.state_f32 = UINT64_MAX - 3;
  reject(b);
  b = a;
  b.workspace_f32++;
  reject(b);
  b = a;
  b.rows = 1;
  reject(b);
  b = a;
  b.rows = 0;
  reject(b);
  b = a;
  b.rows = UINT32_MAX;
  reject(b);
  b = a;
  b.key_heads = 0;
  reject(b);
  b = a;
  b.key_heads = 3;
  reject(b);
  b = a;
  b.key_dim = 513;
  reject(b);
  b = a;
  b.value_heads = UINT32_MAX;
  reject(b);
  b = a;
  b.reserved = 1;
  reject(b);
  require(ferrule_core_bf16_chunk_launch(nullptr) == cudaErrorInvalidValue,
          "null descriptor");
  std::cout << "native ABI/budget/overflow/alias/extent guards passed without "
               "CUDA calls\n";
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--contracts") {
      contracts();
      return 0;
    }
    if (argc != 7)
      throw std::runtime_error("directory rows Hk Hv Dk Dv");
    std::string dir = argv[1];
    FerruleCoreBf16ChunkArgs a{};
    a.rows = std::stoul(argv[2]);
    a.key_heads = std::stoul(argv[3]);
    a.value_heads = std::stoul(argv[4]);
    a.key_dim = std::stoul(argv[5]);
    a.value_dim = std::stoul(argv[6]);
    uint64_t n;
    require(requirements(a, &n), "shape");
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0));
    int runtime;
    ck(cudaRuntimeGetVersion(&runtime));
    require(prop.major == 8 && prop.minor == 6 && runtime == 12040,
            "CUDA12.4 sm86 only");
    uint64_t gates = uint64_t(a.rows) * a.value_heads,
             out = gates * a.value_dim,
             state = uint64_t(a.value_heads) * a.key_dim * a.value_dim;
    Buffer<uint16_t> qkv(uint64_t(a.rows) * (2 * a.key_heads * a.key_dim +
                                             a.value_heads * a.value_dim)),
        aa(gates), b(gates), al(a.value_heads), dt(a.value_heads), output(out);
    Buffer<float> s(state), scratch(n);
    qkv.read(dir + "/qkv.bf16");
    aa.read(dir + "/a.bf16");
    b.read(dir + "/b.bf16");
    al.read(dir + "/a_log.bf16");
    dt.read(dir + "/dt_bias.bf16");
    s.read(dir + "/state.f32");
    a.qkv_bf16 = qkv.addr();
    a.a_bf16 = aa.addr();
    a.b_bf16 = b.addr();
    a.a_log_bf16 = al.addr();
    a.dt_bias_bf16 = dt.addr();
    a.state_f32 = s.addr();
    a.output_bf16 = output.addr();
    a.workspace_f32 = scratch.addr();
    a.workspace_elements = n;
    // Complete pageable H2D staging before crossing to the nonblocking stream.
    ck(cudaStreamSynchronize(nullptr));
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    a.stream = reinterpret_cast<uint64_t>(stream);
    ck(cudaError_t(ferrule_core_bf16_chunk_launch(&a)));
    ck(cudaStreamSynchronize(stream));
    auto first_out = output.host();
    auto first_state = s.host();
    scratch.host();
    output.write(dir + "/native-output.bf16");
    s.write(dir + "/native-state.f32");
    // Stage observation is test-only, on the default stream; prove no
    // arithmetic change.
    s.reset();
    a.stream = 0;
    trace_enabled = true;
    trace_dir = dir;
    trace_chunk = 0;
    ck(cudaError_t(ferrule_core_bf16_chunk_launch(&a)));
    ck(cudaDeviceSynchronize());
    auto second = s.host();
    require(output.host() == first_out &&
                !memcmp(first_state.data(), second.data(), state * 4),
            "observation changed result");
    qkv.immutable();
    aa.immutable();
    b.immutable();
    al.immutable();
    dt.immutable();
    scratch.host();
    trace_enabled = false;
    if (a.rows == 5 && a.key_dim == 128 && a.value_dim == 128 &&
        a.value_heads == 32) {
      // Replay each observed HF matrix product with its own exact HF operands,
      // separating matrix arithmetic discrepancies from upstream propagation.
      const uint32_t ms[8] = {64, 64, 64, 64, 64, 64, 64, 128};
      const uint32_t ns[8] = {64, 128, 128, 64, 128, 128, 128, 128};
      const uint32_t ks[8] = {128, 64, 64, 128, 128, 128, 64, 64};
      for (int j = 0; j < 8; ++j) {
        Buffer<float> x(a.value_heads * ms[j] * ks[j]),
            y(a.value_heads * ks[j] * ns[j]), z(a.value_heads * ms[j] * ns[j]);
        x.read(dir + "/mm" + std::to_string(j) + "-lhs.f32");
        y.read(dir + "/mm" + std::to_string(j) + "-rhs.f32");
        matmul<<<dim3((ns[j] + 15) / 16, (ms[j] + 15) / 16, a.value_heads),
                 dim3(16, 16)>>>(x.p, y.p, z.p, ms[j], ns[j], ks[j], false,
                                 false);
        ck(cudaGetLastError());
        z.write(dir + "/isolated-matmul" + std::to_string(j) + ".f32");
        x.immutable();
        y.immutable();
      }
    }
    ck(cudaStreamDestroy(stream));
    cudaFuncAttributes attr;
    ck(cudaFuncGetAttributes(&attr, matmul));
    std::cout << "{\"guards\":true,\"inputs_immutable\":true,\"observation_"
                 "exact\":true,\"workspace_bytes\":"
              << n * 4
              << ",\"launches\":" << ((a.rows + 63) / 64) * launches_per_chunk
              << ",\"matmul_registers\":" << attr.numRegs
              << ",\"matmul_local_bytes\":" << attr.localSizeBytes
              << ",\"matmul_shared_bytes\":" << attr.sharedSizeBytes << "}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
