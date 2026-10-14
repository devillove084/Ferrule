// Isolated fused-panel comparisons against the pinned, test-only unfused kernel.
#include "chunk_prior.cuh"
#undef FERRULE_GDN_CHUNK_OBSERVE
#define main chunk_fixture_main
#include "chunk.cu"
#undef main
#include <algorithm>
namespace prior = ferrule::cuda::core::prior_bf16_chunk;

template <class T> static void equal(Buffer<T> &a, Buffer<T> &b) {
  auto x = a.host(), y = b.host();
  require(x.size() == y.size() && !memcmp(x.data(), y.data(), x.size() * sizeof(T)),
          "fused/unfused bit mismatch (including signed zero)");
}
static void panel(float *destination, size_t n, const std::string &path) {
  std::vector<float> data(n);
  std::ifstream f(path, std::ios::binary);
  require(bool(f.read(reinterpret_cast<char *>(data.data()), n * 4)) && f.peek() == EOF,
          "actual panel extent");
  ck(cudaMemcpy(destination, data.data(), n * 4, cudaMemcpyHostToDevice));
}
static void test(uint32_t rows, uint32_t heads, uint32_t dk, uint32_t dv,
                 uint32_t start, const std::string &actual = "") {
  FerruleCoreBf16ChunkArgs a{};
  a.rows = rows; a.key_heads = a.value_heads = heads;
  a.key_dim = dk; a.value_dim = dv;
  uint64_t n;
  require(requirements(a, &n), "geometry");
  Buffer<float> scratch(n), old_scratch(n), state(uint64_t(heads) * dk * dv),
      old_state(state.n);
  Buffer<uint16_t> output(uint64_t(rows) * heads * dv), old_output(output.n);
  a.workspace_f32 = scratch.addr(); a.state_f32 = state.addr();
  a.output_bf16 = output.addr();
  Workspace w(a);
  auto b = a; b.workspace_f32 = old_scratch.addr();
  b.state_f32 = old_state.addr(); b.output_bf16 = old_output.addr();
  prior::Workspace old(b);
  // Every panel starts nonzero and dirty, including otherwise reserved storage.
  // Signed zeros in both signs exercise masking and bitwise comparisons.
  std::vector<float> data(n), seed(state.n);
  for (size_t i = 0; i < n; ++i)
    data[i] = i % 11 == 0 ? -0.0f : (int(i % 37) - 18) / 64.0f;
  for (size_t i = 0; i < seed.size(); ++i)
    seed[i] = (int(i % 19) - 9) / 17.0f + 0.000012345f;
  ck(cudaMemcpy(scratch.p, data.data(), n * 4, cudaMemcpyHostToDevice));
  ck(cudaMemcpy(state.p, seed.data(), seed.size() * 4, cudaMemcpyHostToDevice));
  if (!actual.empty()) {
    panel(w.prefix, w.ng, actual + "/chunk0-g_prefix.f32");
    panel(w.q, w.nk, actual + "/chunk0-qscaled.f32");
    panel(w.k, w.nk, actual + "/chunk0-knorm.f32");
    panel(w.kb, w.nk, actual + "/chunk0-k_beta.f32");
    panel(w.attn, w.nm, actual + "/chunk0-matmul3.f32");
    panel(w.value, w.nv, actual + "/chunk0-matmul1.f32");
    panel(w.vp, w.nv, actual + "/chunk0-matmul4.f32");
    panel(w.inter, w.nv, actual + "/chunk0-matmul5.f32");
    panel(w.intra, w.nv, actual + "/chunk0-matmul6.f32");
    panel(w.state_product, w.ns, actual + "/chunk0-matmul7.f32");
    panel(state.p, state.n, actual + "/state.f32");
  }
  ck(cudaMemcpy(old_scratch.p, scratch.p, n * 4, cudaMemcpyDeviceToDevice));
  ck(cudaMemcpy(old_state.p, state.p, state.n * 4, cudaMemcpyDeviceToDevice));
  ck(cudaStreamSynchronize(nullptr));
  cudaStream_t stream;
  ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  decay_and_initial_exponents<<<blocks_for(std::max(w.nm, w.nk)), kBlock, 0, stream>>>(w, dk);
  prior::decay<<<blocks_for(old.nm), kBlock, 0, stream>>>(old);
  prior::exponent_keys<<<blocks_for(old.nk), kBlock, 0, stream>>>(old, dk, false);
  ck(cudaGetLastError()); ck(cudaStreamSynchronize(stream));
  equal(scratch, old_scratch); equal(state, old_state); equal(output, old_output);
  attention_delta_and_end_keys<<<blocks_for(std::max({w.nm, w.nv, w.nk})), kBlock, 0, stream>>>(w, dk);
  prior::mask_matrix<<<blocks_for(old.nm), kBlock, 0, stream>>>(old, false);
  prior::delta_value<<<blocks_for(old.nv), kBlock, 0, stream>>>(old);
  prior::exponent_keys<<<blocks_for(old.nk), kBlock, 0, stream>>>(old, dk, true);
  ck(cudaGetLastError()); ck(cudaStreamSynchronize(stream));
  equal(scratch, old_scratch); equal(state, old_state); equal(output, old_output);
  output_and_state<<<blocks_for(std::max(w.nv, w.ns)), kBlock, 0, stream>>>(a, w, start);
  prior::output<<<blocks_for(old.nv), kBlock, 0, stream>>>(b, old, start);
  prior::update_state<<<blocks_for(old.ns), kBlock, 0, stream>>>(b, old);
  ck(cudaGetLastError()); ck(cudaStreamSynchronize(stream));
  equal(scratch, old_scratch); equal(state, old_state); equal(output, old_output);
  ck(cudaStreamDestroy(stream));
}
int main(int argc, char **argv) {
  try {
    require(argc == 2, "actual frozen stage directory required");
    cudaDeviceProp prop; int runtime;
    ck(cudaGetDeviceProperties(&prop, 0)); ck(cudaRuntimeGetVersion(&runtime));
    require(prop.major == 8 && prop.minor == 6 && runtime == 12040, "CUDA12.4/sm86");
    unsigned cases = 0;
    for (auto dims : std::vector<std::pair<unsigned, unsigned>>{{1, 3}, {4, 7}, {16, 9},
                      {17, 65}, {65, 17}, {128, 128}, {511, 512}, {512, 511}})
      for (unsigned rows : {2u, 5u, 63u, 64u, 65u, 129u}) {
        test(rows, 2, dims.first, dims.second, (rows - 1) / C * C);
        ++cases;
      }
    test(5, 32, 128, 128, 0, argv[1]); ++cases;
    std::cout << "{\"event\":\"fused_panels_exact\",\"cases\":" << cases
              << ",\"phases_per_case\":3,\"all_workspace_words_checked\":true,"
                 "\"output_state_guards\":true,\"signed_zero_bitwise\":true}\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
