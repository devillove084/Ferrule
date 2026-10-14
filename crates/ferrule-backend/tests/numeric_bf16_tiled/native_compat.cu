// Bounded test-only compatibility candidates, never a production dispatch
// override. Profile choices follow the pinned native gate's 64x64, K64
// sliced-K2 kernel.
#include "cutlass/bf16_linear.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#define CK(x)                                                                  \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e));                  \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)
static void require(bool ok, const char *s) {
  if (!ok) {
    fprintf(stderr, "%s\n", s);
    std::exit(3);
  }
}
template <class T> std::vector<T> read(const std::string &p, size_t n) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  require(bool(f) && size_t(f.tellg()) == n * sizeof(T), p.c_str());
  f.seekg(0);
  std::vector<T> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(T));
  require(bool(f), "read");
  return v;
}
struct Buffer {
  void *p = nullptr;
  explicit Buffer(size_t n) { CK(cudaMalloc(&p, n)); }
  ~Buffer() {
    if (p)
      cudaFree(p);
  }
  uint64_t addr() const { return reinterpret_cast<uint64_t>(p); }
};
template <class T> void copy(Buffer &b, const std::vector<T> &v) {
  CK(cudaMemcpy(b.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
static uint16_t rne(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  return uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
static uint32_t bits(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  return b;
}
using E = ::cutlass::bfloat16_t;
template <class Out>
using Candidate = ::cutlass::gemm::device::Gemm<
    E, ::cutlass::layout::RowMajor, E, ::cutlass::layout::ColumnMajor, Out,
    ::cutlass::layout::ColumnMajor, float, ::cutlass::arch::OpClassTensorOp,
    ::cutlass::arch::Sm80, ::cutlass::gemm::GemmShape<64, 64, 64>,
    ::cutlass::gemm::GemmShape<64, 32, 32>,
    ::cutlass::gemm::GemmShape<16, 8, 16>,
    ::cutlass::epilogue::thread::LinearCombination<Out, 1, float, float>,
    ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5, 8, 8,
    false, ::cutlass::arch::OpMultiplyAdd>;
template <class Out> void gemm(const FerruleCutlassBf16Args &a) {
  using G = Candidate<Out>;
  G g;
  auto *x = reinterpret_cast<const E *>(a.activation);
  auto *w = reinterpret_cast<const E *>(a.weight);
  auto *y = reinterpret_cast<Out *>(a.output);
  typename G::Arguments args({int(a.n), int(a.m), int(a.k)}, {w, int(a.k)},
                             {x, int(a.k)}, {y, int(a.ldd)}, {y, int(a.ldd)},
                             {1.f, 0.f}, 1);
  require(G::get_workspace_size(args) == 0, "hidden candidate workspace");
  require(G::can_implement(args) == ::cutlass::Status::kSuccess,
          "candidate preflight");
  require(
      g.initialize(args, nullptr, reinterpret_cast<cudaStream_t>(a.stream)) ==
          ::cutlass::Status::kSuccess,
      "candidate initialize");
  require(g.run(reinterpret_cast<cudaStream_t>(a.stream)) ==
              ::cutlass::Status::kSuccess,
          "candidate launch");
}
int main(int argc, char **argv) {
  require(
      argc == 6,
      "fixture n decoded.bf16 native.f32 profile[0=production,1=sliced-k2]");
  const std::string dir = argv[1];
  int n = std::stoi(argv[2]), profile = std::stoi(argv[5]);
  require((n == 4096 || n == 8192) && profile >= 0 && profile <= 1,
          "bounded shapes/profile");
  int version = 0;
  CK(cudaRuntimeGetVersion(&version));
  require(version == 12040, "requires CUDA12.4");
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  require(prop.major == 8 && prop.minor == 6, "requires sm86");
  auto x = read<uint16_t>(dir + "/input.bf16", 2048),
       hf = read<uint16_t>(dir + "/hf.bf16", n),
       sc = read<uint16_t>(dir + "/scale.bf16", n / 128 * 16);
  auto w = read<uint8_t>(dir + "/weight.fp8", size_t(n) * 2048);
  auto decoded = read<uint16_t>(argv[3], size_t(n) * 2048);
  auto hfraw = read<float>(argv[4], n);
  Buffer dx(4096), dw(w.size()), ds(sc.size() * 2), dy(n * 4), db(n * 2),
      scratch(8 << 20);
  copy(dx, x);
  copy(dw, w);
  copy(ds, sc);
  cudaStream_t stream;
  CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int tile : {64, 257, 1024, 2047}) {
    FerruleCutlassNumericFp8Args a{};
    a.m = 1;
    a.n = n;
    a.k = a.padded_k = a.lda = 2048;
    a.ldd = n;
    a.tile_rows = tile;
    a.scale_cols = 16;
    a.scale_bytes = 2;
    a.activation = dx.addr();
    a.weight = dw.addr();
    a.scales = ds.addr();
    a.output = dy.addr();
    a.workspace = scratch.addr();
    a.workspace_bytes = (tile + 1) * 4096;
    a.stream = reinterpret_cast<uint64_t>(stream);
    using namespace ferrule::cuda::cutlass::numeric_fp8;
    pack_bf16_activation<<<8, 256, 0, stream>>>(a);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(stream));
    std::vector<uint16_t> staged(2048);
    CK(cudaMemcpy(staged.data(), scratch.p, 4096, cudaMemcpyDeviceToHost));
    require(staged == x, "activation differs");
    for (int first = 0; first < n; first += tile) {
      int count = std::min(tile, n - first);
      decode_weights<<<blocks(uint64_t(count) * 2048), 256, 0, stream>>>(
          a, first, count);
      CK(cudaGetLastError());
      CK(cudaStreamSynchronize(stream));
      std::vector<uint16_t> check(size_t(count) * 2048);
      CK(cudaMemcpy(check.data(),
                    reinterpret_cast<void *>(scratch.addr() + 4096),
                    check.size() * 2, cudaMemcpyDeviceToHost));
      require(std::equal(check.begin(), check.end(),
                         decoded.begin() + size_t(first) * 2048),
              "decoded BF16 tile differs from pinned HF operands");
      auto g = gemm_args(a, first, count);
      if (profile == 0)
        require(ferrule_cutlass_bf16_launch(&g) == 0, "production GEMM");
      else
        gemm<float>(g);
      if (profile != 0) {
        g.output = db.addr() + first * 2;
        gemm<E>(g);
      }
    }
    CK(cudaStreamSynchronize(stream));
    std::vector<float> y(n);
    CK(cudaMemcpy(y.data(), dy.p, n * 4, cudaMemcpyDeviceToHost));
    std::vector<uint16_t> by(n);
    if (profile)
      CK(cudaMemcpy(by.data(), db.p, n * 2, cudaMemcpyDeviceToHost));
    int unequal = 0, rawdiff = 0, epdiff = 0;
    for (int i = 0; i < n; ++i) {
      unequal += rne(y[i]) != hf[i];
      rawdiff += bits(y[i]) != bits(hfraw[i]);
      epdiff += profile && by[i] != rne(y[i]);
      if (rne(y[i]) != hf[i] || (n == 4096 && (i == 3471 || i == 4065)))
        printf("{\"event\":\"word\",\"n\":%d,\"profile\":%d,\"tile\":%d,"
               "\"index\":%d,\"raw\":\"0x%08x\",\"hf_raw\":\"0x%08x\",\"bf16\":"
               "\"0x%04x\",\"hf\":\"0x%04x\"}\n",
               n, profile, tile, i, bits(y[i]), bits(hfraw[i]), rne(y[i]),
               hf[i]);
    }
    printf("{\"event\":\"comparison\",\"n\":%d,\"profile\":%d,\"tile\":%d,"
           "\"unequal_hf\":%d,\"raw_unequal_hf\":%d,\"epilogue_unequal\":%d,"
           "\"all_decoded_operands_exact\":true}\n",
           n, profile, tile, unequal, rawdiff, epdiff);
    fflush(stdout);
    require(epdiff == 0, "BF16 epilogue differs from F32 writeback");
    if (profile == 1 && n == 4096)
      require(unequal == 0 && rawdiff == 0,
              "sliced gate must be exactly native HF");
    if (profile == 0 && n == 8192)
      require(unequal == 0, "QKV must remain exactly native HF");
  }
  CK(cudaStreamSynchronize(stream));
  CK(cudaStreamDestroy(stream));
  return 0;
}
