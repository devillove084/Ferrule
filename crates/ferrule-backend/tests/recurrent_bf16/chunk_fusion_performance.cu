// Matched chunk-only events. Archive has the verified prefix/scaling/split-K fixes.
#include "chunk_prior.cuh"
#undef FERRULE_GDN_CHUNK_OBSERVE
#define FERRULE_CHUNK_CONTINUATION_NO_MAIN
#include "chunk_continuation.cu"
#undef FERRULE_CHUNK_CONTINUATION_NO_MAIN
namespace prior = ferrule::cuda::core::prior_bf16_chunk;

static void submit(FerruleCoreBf16ChunkArgs &a, bool unfused) {
  ck(cudaError_t(unfused ? ferrule_core_prior_bf16_chunk_launch(&a)
                        : ferrule_core_bf16_chunk_prefill_launch(&a)));
}
template <class F>
static void resource(const char *variant, const char *name, F function) {
  cudaFuncAttributes attr;
  ck(cudaFuncGetAttributes(&attr, function));
  std::cout << "{\"event\":\"resource\",\"variant\":\"" << variant
            << "\",\"kernel\":\"" << name << "\",\"registers\":" << attr.numRegs
            << ",\"local_bytes\":" << attr.localSizeBytes
            << ",\"shared_bytes\":" << attr.sharedSizeBytes
            << ",\"max_threads\":" << attr.maxThreadsPerBlock << "}\n";
}
static void resources() {
#define OLD(name) resource("unfused19", #name, prior::name)
#define NEW(name) resource("fused15", #name, ferrule::cuda::core::bf16_chunk::name)
  OLD(prepare); OLD(prefix); OLD(decay); OLD(exponent_keys); OLD(matmul);
  OLD(mask_matrix); OLD(triangular); OLD(delta_value); OLD(output); OLD(update_state);
  NEW(prepare); NEW(prefix); NEW(decay_and_initial_exponents); NEW(matmul);
  NEW(mask_matrix); NEW(triangular); NEW(attention_delta_and_end_keys); NEW(output_and_state);
#undef OLD
#undef NEW
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--contracts") {
      cpu_contracts();
      require(launches_per_chunk == 15 && prior::launches_per_chunk == 19,
              "matched launch contracts");
      std::cout << "matched 15/19 benchmark contracts; no CUDA calls\n";
      return 0;
    }
    require(argc == 4, "fixture-directory rows first-order-offset");
    std::string dir = argv[1];
    uint32_t rows = std::stoul(argv[2]), offset = std::stoul(argv[3]);
    require((rows == 5 || rows == 64 || rows == 129) && offset <= 1,
            "bounded Qwen benchmark shapes/order");
    cudaDeviceProp prop; int runtime;
    ck(cudaGetDeviceProperties(&prop, 0)); ck(cudaRuntimeGetVersion(&runtime));
    require(prop.major == 8 && prop.minor == 6 && runtime == 12040, "CUDA12.4/sm86");
    resources();
    FerruleCoreBf16ChunkArgs shape{};
    shape.rows = rows; shape.key_heads = 16; shape.value_heads = 32;
    shape.key_dim = shape.value_dim = 128;
    uint64_t elements = 0, old_elements = 0;
    require(requirements(shape, &elements) && prior::requirements(shape, &old_elements)
                && elements == old_elements && elements == 4462592, "same scratch layout");
    Call call(shape, dir + "/step0");
    Buffer<float> seed(524288), state(seed.n), scratch(elements), expected_state(seed.n);
    Buffer<uint16_t> al(32), dt(32), expected_output(call.output.n);
    seed.read(dir + "/state.f32"); al.read(dir + "/a_log.bf16"); dt.read(dir + "/dt_bias.bf16");
    expected_state.read(dir + "/expected-state.f32");
    expected_output.read(dir + "/expected-output.bf16");
    const auto expected_s = expected_state.host();
    const auto expected_y = expected_output.host();
    ck(cudaStreamSynchronize(nullptr)); Stream stream;
    auto &a = call.args;
    a.a_log_bf16 = al.addr(); a.dt_bias_bf16 = dt.addr();
    a.state_f32 = state.addr(); a.workspace_f32 = scratch.addr();
    a.workspace_elements = scratch.n; a.stream = stream.addr();
    const auto state_address = state.addr(), scratch_address = scratch.addr();
    ck(cudaMemsetAsync(scratch.p, 0xff, elements * 4, stream.value));
    auto check = [&] {
      // Complete checks of every output/state word, guards and immutable inputs;
      // this runs only after the stop event (or untimed stream completion).
      exact(call.output.host(), expected_y, "HF output differs");
      exact(state.host(), expected_s, "HF F32 state differs");
      call.immutable(); seed.immutable(); al.immutable(); dt.immutable(); scratch.host();
      require(state.addr() == state_address && scratch.addr() == scratch_address,
              "same state/scratch addresses");
    };
    for (int variant = 0; variant < 2; ++variant) {
      copy(state, seed, stream);
      submit(a, variant == 0); stream.sync(); check();
    }
    // Independent state reset on every warmup; both kernels warm equally with
    // alternating order. Scratch is shared/dirty, never reset between variants.
    for (int warmup = 0; warmup < 10; ++warmup)
      for (int order = 0; order < 2; ++order) {
        copy(state, seed, stream);
        submit(a, (warmup + order + offset) % 2 == 0);
      }
    stream.sync(); check();
    Event start, stop;
    for (int pair = 0; pair < 30; ++pair)
      for (int order = 0; order < 2; ++order) {
        const bool unfused = (pair + order + offset) % 2 == 0;
        copy(state, seed, stream); // D2D reset is before the start event.
        ck(cudaEventRecord(start.value, stream.value));
        submit(a, unfused);
        ck(cudaEventRecord(stop.value, stream.value));
        ck(cudaEventSynchronize(stop.value));
        float ms; ck(cudaEventElapsedTime(&ms, start.value, stop.value));
        require(std::isfinite(ms) && ms > 0, "finite positive event time");
        check();
        std::cout << "{\"event\":\"timing\",\"pair\":" << pair
                  << ",\"order\":" << order << ",\"variant\":\""
                  << (unfused ? "unfused19" : "fused15") << "\",\"microseconds\":"
                  << ms * 1000 << ",\"correctness_exact\":true}\n";
      }
    std::cout << "{\"event\":\"complete\",\"rows\":" << rows
              << ",\"pairs\":30,\"samples\":60,\"warmups_per_variant\":10,"
                 "\"correctness_checked_every_sample\":true,\"same_addresses\":true,"
                 "\"unfused_launches\":" << 19 * ((rows + 63) / 64)
              << ",\"fused_launches\":" << 15 * ((rows + 63) / 64) << "}\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
  return 0;
}
