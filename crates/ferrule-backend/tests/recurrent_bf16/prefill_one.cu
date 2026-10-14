#define FERRULE_CHUNK_CONTINUATION_NO_MAIN
#include "chunk_continuation.cu"
#undef FERRULE_CHUNK_CONTINUATION_NO_MAIN

static void prefill_contracts() {
  contracts();
  FerruleCoreBf16ChunkArgs a{};
  a.rows = 1; a.key_heads = 16; a.value_heads = 32;
  a.key_dim = a.value_dim = 128;
  uint64_t n;
  require(requirements(a, &n, true) && n == 4462592, "prefill scratch");
  require(!requirements(a, &n), "legacy M1 still rejected");
  uint64_t p = 0x10000000;
  for (auto address : {&a.qkv_bf16, &a.a_bf16, &a.b_bf16, &a.a_log_bf16,
                       &a.dt_bias_bf16, &a.state_f32, &a.output_bf16, &a.workspace_f32}) {
    *address = p; p += 1 << 24;
  }
  a.workspace_elements = n;
  require(validate(a, true) && !validate(a), "explicit admission");
  require(ferrule_core_bf16_chunk_launch(&a) == cudaErrorInvalidValue, "legacy entry");
  require(ferrule_core_bf16_chunk_prefill_launch(nullptr) == cudaErrorInvalidValue, "null");
  auto reject = [](const FerruleCoreBf16ChunkArgs &b) {
    require(!validate(b, true), "prefill validation rejection");
    require(ferrule_core_bf16_chunk_prefill_launch(&b) == cudaErrorInvalidValue,
            "prefill rejection before launch");
  };
  auto b = a; b.reserved = 1; reject(b);
  b = a; b.rows = 0; reject(b);
  b = a; b.rows = UINT32_MAX; reject(b);
  b = a; b.key_dim = 513; reject(b);
  b = a; b.value_dim = 513; reject(b);
  b = a; b.key_heads = 3; reject(b);
  b = a; b.workspace_elements = n - 1; reject(b);
  b = a; b.workspace_elements = UINT64_MAX; reject(b);
  b = a; b.state_f32 = UINT64_MAX - 3; reject(b);
  b = a; b.output_bf16++; reject(b);
  for (auto source : {a.qkv_bf16, a.a_bf16, a.b_bf16, a.a_log_bf16,
                     a.dt_bias_bf16, a.state_f32, a.output_bf16}) {
    b = a; b.workspace_f32 = source; reject(b);
  }
  b = a; b.output_bf16 = a.state_f32; reject(b);
  std::cout << "prefill native guards passed without CUDA calls\n";
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--contracts") {
      prefill_contracts(); return 0;
    }
    require(argc >= 8, "directory cached/chunk Hk Hv Dk Dv rows...");
    std::string dir = argv[1]; bool cached = std::string(argv[2]) == "cached";
    FerruleCoreBf16ChunkArgs shape{};
    shape.key_heads = std::stoul(argv[3]); shape.value_heads = std::stoul(argv[4]);
    shape.key_dim = std::stoul(argv[5]); shape.value_dim = std::stoul(argv[6]);
    shape.rows = 1;
    uint64_t n; require(requirements(shape, &n, true), "prefill geometry");
    cudaDeviceProp prop; int runtime;
    ck(cudaGetDeviceProperties(&prop, 0)); ck(cudaRuntimeGetVersion(&runtime));
    require(prop.major == 8 && prop.minor == 6 && runtime == 12040, "CUDA12.4/sm86");
    Buffer<float> seed(uint64_t(shape.value_heads)*shape.key_dim*shape.value_dim),
        state(seed.n), scratch(n);
    Buffer<uint16_t> al(shape.value_heads), dt(shape.value_heads);
    seed.read(dir + "/state.f32"); al.read(dir + "/a_log.bf16"); dt.read(dir + "/dt_bias.bf16");
    std::vector<std::unique_ptr<Call>> calls;
    for (int i = 7; i < argc; ++i) {
      shape.rows = std::stoul(argv[i]);
      require(!cached || shape.rows == 1, "cached row contract");
      calls.push_back(std::make_unique<Call>(shape, dir + "/step" + std::to_string(i-7)));
    }
    ck(cudaStreamSynchronize(nullptr)); Stream stream;
    copy(state, seed, stream);
    ck(cudaMemsetAsync(scratch.p, 0xff, n*4, stream.value));
    for (auto &call : calls) {
      auto &a = call->args;
      a.a_log_bf16 = al.addr(); a.dt_bias_bf16 = dt.addr(); a.state_f32 = state.addr();
      a.workspace_f32 = scratch.addr(); a.workspace_elements = n; a.stream = stream.addr();
      if (cached) launch(*call, state, scratch, al, dt, stream, true);
      else ck(cudaError_t(ferrule_core_bf16_chunk_prefill_launch(&a)));
      copy(call->state_snapshot, state, stream);
    }
    // No host fence/reset/clear/reallocation between chunk1, chunk5 and chunk1.
    stream.sync();
    for (size_t i = 0; i < calls.size(); ++i) {
      auto &c = *calls[i]; std::string d = dir + "/step" + std::to_string(i);
      Buffer<uint16_t> expected(c.output.n); Buffer<float> expected_state(seed.n);
      expected.read(d + "/expected.bf16"); expected_state.read(d + "/expected-state.f32");
      exact(c.output.host(), expected.host(), "HF output bit mismatch");
      exact(c.state_snapshot.host(), expected_state.host(), "HF F32 state bit mismatch");
      c.immutable();
    }
    seed.immutable(); al.immutable(); dt.immutable(); state.host(); scratch.host();
    std::cout << "{\"calls\":" << calls.size() << ",\"exact\":true,\"cached\":"
              << (cached ? "true" : "false") << ",\"guards\":true,\"carry_no_host_fence\":true}\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
  return 0;
}
