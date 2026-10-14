// Reuse the existing chunk fixture buffers, guards and ABI contracts unchanged.
#define main original_chunk_gate_main
#include "chunk.cu"
#undef main
#include "core/recurrent_bf16.cuh"
#include <algorithm>
#include <cmath>
#include <memory>

struct Stream {
  cudaStream_t value;
  Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
  uint64_t addr() const { return reinterpret_cast<uint64_t>(value); }
  void sync() { ck(cudaStreamSynchronize(value)); }
};
struct Event {
  cudaEvent_t value;
  Event() { ck(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};
struct Call {
  FerruleCoreBf16ChunkArgs args;
  Buffer<uint16_t> qkv, a, b, output;
  Buffer<float> state_snapshot;
  Call(FerruleCoreBf16ChunkArgs shape, const std::string &directory)
      : args(shape),
        qkv(uint64_t(shape.rows) * (2 * shape.key_heads * shape.key_dim +
                                    shape.value_heads * shape.value_dim)),
        a(uint64_t(shape.rows) * shape.value_heads), b(a.n),
        output(a.n * shape.value_dim),
        state_snapshot(uint64_t(shape.value_heads) * shape.key_dim *
                       shape.value_dim) {
    qkv.read(directory + "/qkv.bf16");
    a.read(directory + "/a.bf16");
    b.read(directory + "/b.bf16");
    args.qkv_bf16 = qkv.addr();
    args.a_bf16 = a.addr();
    args.b_bf16 = b.addr();
    args.output_bf16 = output.addr();
  }
  void immutable() {
    qkv.immutable();
    a.immutable();
    b.immutable();
  }
};
static void copy(Buffer<float> &target, Buffer<float> &source, Stream &stream) {
  require(target.n == source.n, "state copy extent");
  ck(cudaMemcpyAsync(target.p, source.p, target.n * 4, cudaMemcpyDeviceToDevice,
                     stream.value));
}
template <class T>
static void exact(const std::vector<T> &a, const std::vector<T> &b,
                  const char *message) {
  require(a.size() == b.size() &&
              !memcmp(a.data(), b.data(), a.size() * sizeof(T)),
          message);
}
static void launch(Call &call, Buffer<float> &state, Buffer<float> &scratch,
                   Buffer<uint16_t> &al, Buffer<uint16_t> &dt, Stream &stream,
                   bool scan = false) {
  auto &a = call.args;
  a.a_log_bf16 = al.addr();
  a.dt_bias_bf16 = dt.addr();
  a.state_f32 = state.addr();
  a.workspace_f32 = scratch.addr();
  a.workspace_elements = scratch.n;
  a.stream = stream.addr();
  if (scan) {
    FerruleCoreBf16DeltaArgs d{a.rows,        a.key_heads,    a.value_heads,
                               a.key_dim,     a.value_dim,    0,
                               a.qkv_bf16,    a.a_bf16,       a.b_bf16,
                               a.a_log_bf16,  a.dt_bias_bf16, a.state_f32,
                               a.output_bf16, a.stream};
    ck(cudaError_t(ferrule_core_bf16_delta_launch(&d)));
  } else
    ck(cudaError_t(ferrule_core_bf16_chunk_launch(&a)));
}
static void cpu_contracts() {
  contracts();
  for (uint32_t dk : {511u, 512u})
    for (uint32_t dv : {511u, 512u}) {
      FerruleCoreBf16ChunkArgs a{};
      a.rows = 129;
      a.key_heads = 1;
      a.value_heads = 2;
      a.key_dim = dk;
      a.value_dim = dv;
      uint64_t elements = 0;
      require(requirements(a, &elements) && elements * 4 <= INT32_MAX,
              "near-limit plan");
      a.key_dim = 513;
      require(!requirements(a, &elements), "reject Dk513");
      a.key_dim = dk;
      a.value_dim = 513;
      require(!requirements(a, &elements), "reject Dv513");
    }
  std::cout << "continuation near-Dk/Dv512 contracts passed; no CUDA calls\n";
}
#ifndef FERRULE_CHUNK_CONTINUATION_NO_MAIN
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--contracts") {
      cpu_contracts();
      return 0;
    }
    require(argc >= 9, "mode fixture output Hk Hv Dk Dv rows...");
    const bool timing = std::string(argv[1]) == "performance";
    require(timing || std::string(argv[1]) == "continuation", "mode");
    std::string fixture = argv[2], out = argv[3];
    FerruleCoreBf16ChunkArgs shape{};
    shape.key_heads = std::stoul(argv[4]);
    shape.value_heads = std::stoul(argv[5]);
    shape.key_dim = std::stoul(argv[6]);
    shape.value_dim = std::stoul(argv[7]);
    shape.rows = std::stoul(argv[8]);
    uint64_t elements = 0;
    require(requirements(shape, &elements), "shape");
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0));
    int runtime;
    ck(cudaRuntimeGetVersion(&runtime));
    require(prop.major == 8 && prop.minor == 6 && runtime == 12040,
            "CUDA12.4 sm86 only");
    Stream stream;
    const size_t ns =
        uint64_t(shape.value_heads) * shape.key_dim * shape.value_dim;
    Buffer<float> seed(ns), reset_seed(ns), state(ns), reset_state(ns),
        fresh_state(ns), scratch(elements), fresh_scratch(elements);
    Buffer<uint16_t> al(shape.value_heads), dt(shape.value_heads);
    seed.read(fixture + "/state.f32");
    reset_seed.read(fixture + "/reset-state.f32");
    al.read(fixture + "/a_log.bf16");
    dt.read(fixture + "/dt_bias.bf16");
    std::vector<std::unique_ptr<Call>> calls;
    for (int i = 8; i < argc; ++i) {
      shape.rows = std::stoul(argv[i]);
      uint64_t n;
      require(shape.rows > 1 && requirements(shape, &n) && n == elements,
              "shared workspace geometry");
      calls.push_back(std::make_unique<Call>(shape, fixture + "/step" +
                                                        std::to_string(i - 8)));
    }
    Call reset(calls[0]->args, fixture + "/step0"),
        fresh(calls[0]->args, fixture + "/step0");
    ck(cudaStreamSynchronize(nullptr));
    copy(state, seed, stream);
    ck(cudaMemsetAsync(scratch.p, 0xff, elements * 4, stream.value));
    const auto scratch_address = scratch.addr(), state_address = state.addr();
    // No host fence, state reset, scratch clearing or allocation between calls.
    uint64_t launches = 0;
    for (auto &call : calls) {
      launch(*call, state, scratch, al, dt, stream);
      launches += ((call->args.rows + 63) / 64) * launches_per_chunk;
      copy(call->state_snapshot, state, stream);
      require(scratch.addr() == scratch_address &&
                  state.addr() == state_address,
              "roots replaced");
    }
    // A new independent state reuses dirty scratch after the carried sequence.
    copy(reset_state, reset_seed, stream);
    launch(reset, reset_state, scratch, al, dt, stream);
    copy(fresh_state, reset_seed, stream);
    ck(cudaMemsetAsync(fresh_scratch.p, 0xa5, elements * 4, stream.value));
    launch(fresh, fresh_state, fresh_scratch, al, dt, stream);
    stream.sync();
    exact(reset.output.host(), fresh.output.host(),
          "dirty scratch leaked into reset output");
    exact(reset_state.host(), fresh_state.host(),
          "dirty scratch leaked into reset state");
    for (size_t i = 0; i < calls.size(); ++i) {
      calls[i]->output.write(out + "/carry-" + std::to_string(i) +
                             "-output.bf16");
      calls[i]->state_snapshot.write(out + "/carry-" + std::to_string(i) +
                                     "-state.f32");
      calls[i]->immutable();
    }
    reset.output.write(out + "/reset-output.bf16");
    reset_state.write(out + "/reset-state.f32");
    reset.immutable();
    fresh.immutable();
    seed.immutable();
    reset_seed.immutable();
    al.immutable();
    dt.immutable();
    scratch.host();
    fresh_scratch.host();
    state.host();
    std::cout
        << "{\"event\":\"continuation\",\"calls\":" << calls.size()
        << ",\"workspace_bytes\":" << elements * 4
        << ",\"carry_launches\":" << launches
        << ",\"same_scratch_and_state\":true,\"no_intercall_host_fence\":true,"
           "\"reset_dirty_equals_fresh\":true,\"guards_inputs\":true}\n";
    if (timing) {
      require(calls.size() == 1, "timing one call only");
      auto &call = *calls[0];
      std::vector<uint16_t> reference_output[2];
      std::vector<float> reference_state[2];
      for (int variant = 0; variant < 2; ++variant) {
        copy(state, seed, stream);
        launch(call, state, scratch, al, dt, stream, variant == 0);
        stream.sync();
        reference_output[variant] = call.output.host();
        reference_state[variant] = state.host();
        call.output.write(
            out + (variant == 0 ? "/scan-output.bf16" : "/chunk-output.bf16"));
        state.write(out +
                    (variant == 0 ? "/scan-state.f32" : "/chunk-state.f32"));
      }
      Event start, stop;
      for (int i = 0; i < 4; ++i)
        for (int variant = 0; variant < 2; ++variant) {
          copy(state, seed, stream);
          launch(call, state, scratch, al, dt, stream, variant == 0);
        }
      stream.sync();
      // Matched AB/BA events: reset outside timing; complete output/state
      // checked against each algorithm's own untimed result, NOT equal-quality
      // evidence.
      for (int sample = 0; sample < 20; ++sample)
        for (int order = 0; order < 2; ++order) {
          int variant = (sample + order) % 2;
          copy(state, seed, stream);
          ck(cudaEventRecord(start.value, stream.value));
          launch(call, state, scratch, al, dt, stream, variant == 0);
          ck(cudaEventRecord(stop.value, stream.value));
          ck(cudaEventSynchronize(stop.value));
          float ms;
          ck(cudaEventElapsedTime(&ms, start.value, stop.value));
          require(ms > 0 && std::isfinite(ms), "event time");
          exact(call.output.host(), reference_output[variant],
                "timed output unstable");
          exact(state.host(), reference_state[variant], "timed state unstable");
          call.immutable();
          seed.immutable();
          al.immutable();
          dt.immutable();
          scratch.host();
          std::cout << "{\"event\":\"timing\",\"sample\":" << sample
                    << ",\"order\":" << order << ",\"variant\":\""
                    << (variant == 0 ? "scan" : "chunk")
                    << "\",\"microseconds\":" << ms * 1000 << "}\n";
        }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}

#endif
