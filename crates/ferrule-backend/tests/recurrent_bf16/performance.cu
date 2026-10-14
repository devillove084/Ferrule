// Matched old/new GDN only. Archived kernel stays in this test TU, never
// production.
#include "core/recurrent_bf16.cuh"
#include "prior.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ferrule::cuda::core;
constexpr int warmups = 20, rounds = 30, repeats = 8;
constexpr size_t state_count = 32 * 128 * 128, width = 8192,
                 output_width = 4096;
void ck(cudaError_t s) {
  if (s != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(s));
}
template <class T> std::vector<T> read(const std::string &path, size_t count) {
  std::ifstream f(path, std::ios::binary);
  std::vector<T> v(count);
  if (!f.read(reinterpret_cast<char *>(v.data()), count * sizeof(T)) ||
      f.peek() != EOF)
    throw std::runtime_error("input extent: " + path);
  return v;
}
template <class T> struct Buffer {
  T *base, *p;
  size_t n;
  explicit Buffer(size_t count) : n(count) {
    ck(cudaMalloc(&base, (n + 32) * sizeof(T)));
    p = base + 16;
    ck(cudaMemset(base, 0x5a, (n + 32) * sizeof(T)));
  }
  ~Buffer() { cudaFree(base); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void set(const std::vector<T> &v) {
    if (v.size() != n)
      throw std::runtime_error("buffer extent");
    ck(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> host() const {
    std::vector<T> v(n + 32);
    ck(cudaMemcpy(v.data(), base, (n + 32) * sizeof(T),
                  cudaMemcpyDeviceToHost));
    for (size_t index : {size_t(0), n + 16}) {
      auto *b = reinterpret_cast<const unsigned char *>(v.data() + index);
      for (size_t i = 0; i < 16 * sizeof(T); ++i)
        if (b[i] != 0x5a)
          throw std::runtime_error("allocation guard overwritten");
    }
    return std::vector<T>(v.begin() + 16, v.end() - 16);
  }
  uint64_t addr() const { return reinterpret_cast<uint64_t>(p); }
};
template <class T>
void exact(const std::vector<T> &a, const std::vector<T> &b, const char *what) {
  if (a.size() != b.size())
    throw std::runtime_error(what);
  if (memcmp(a.data(), b.data(), a.size() * sizeof(T))) {
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      if (memcmp(&a[i], &b[i], sizeof(T))) {
        if (count < 4)
          std::cerr << what << " index=" << i << " actual=" << a[i]
                    << " expected=" << b[i] << "\n";
        ++count;
      }
    }
    std::cerr << what << " unequal=" << count << "/" << a.size() << "\n";
    throw std::runtime_error(what);
  }
}
struct Events {
  cudaStream_t stream;
  std::array<cudaEvent_t, repeats> start, stop;
  Events() {
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int i = 0; i < repeats; ++i) {
      ck(cudaEventCreate(&start[i]));
      ck(cudaEventCreate(&stop[i]));
    }
  }
  ~Events() {
    for (int i = 0; i < repeats; ++i) {
      cudaEventDestroy(start[i]);
      cudaEventDestroy(stop[i]);
    }
    cudaStreamDestroy(stream);
  }
};
void launch(bool old, const FerruleCoreBf16DeltaArgs &a) {
  ck(cudaError_t(old ? prior_core_bf16_delta_launch(&a)
                     : ferrule_core_bf16_delta_launch(&a)));
}
struct Snapshot {
  std::vector<uint16_t> output;
  std::vector<float> state;
};
void describe() {
  std::cout << "{\"rows\":[1,5,32],\"key_heads\":16,\"value_heads\":32,"
               "\"key_dim\":128,\"value_dim\":128,\"warmups_per_variant\":"
            << warmups << ",\"paired_rounds\":" << rounds
            << ",\"launches_per_sample\":" << repeats
            << ",\"order\":\"AB/BA alternating\",\"timing\":\"CUDA events "
               "around one GDN launch; resets excluded\"}\n";
}
void stats(int rows, bool old, const std::vector<double> &samples) {
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  std::cout << "{\"event\":\"summary\",\"rows\":" << rows << ",\"variant\":\""
            << (old ? "old" : "new") << "\",\"samples\":" << samples.size()
            << ",\"median_us\":" << (sorted[14] + sorted[15]) * 0.5
            << ",\"p95_us\":" << sorted[28] << ",\"min_us\":" << sorted.front()
            << ",\"max_us\":" << sorted.back() << "}\n";
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--describe") {
      describe();
      return 0;
    }
    if (argc != 2)
      throw std::runtime_error("usage: performance INPUT_DIR | --describe");
    int version;
    ck(cudaRuntimeGetVersion(&version));
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0));
    if (version != 12040 || prop.major != 8 || prop.minor != 6)
      throw std::runtime_error("requires CUDA12.4/sm86");
    describe();
    std::cout << std::setprecision(9);
    for (bool old : {true, false}) {
      cudaFuncAttributes attr;
      if (old)
        ck(cudaFuncGetAttributes(&attr, prior_hf_bf16::delta));
      else
        ck(cudaFuncGetAttributes(&attr, hf_bf16::delta));
      std::cout << "{\"event\":\"resources\",\"variant\":\""
                << (old ? "old" : "new") << "\",\"registers\":" << attr.numRegs
                << ",\"local_bytes\":" << attr.localSizeBytes
                << ",\"shared_bytes\":" << attr.sharedSizeBytes
                << ",\"binary_version\":" << attr.binaryVersion << "}\n";
    }
    std::string root = argv[1];
    for (int rows : {1, 5, 32}) {
      std::string dir = root + "/rows" + std::to_string(rows);
      auto q = read<uint16_t>(dir + "/qkv.bf16", rows * width);
      auto a = read<uint16_t>(dir + "/a.bf16", rows * 32);
      auto b = read<uint16_t>(dir + "/b.bf16", rows * 32);
      auto al = read<uint16_t>(root + "/a_log.bf16", 32);
      auto dt = read<uint16_t>(root + "/dt_bias.bf16", 32);
      auto initial = read<float>(dir + "/state.f32", state_count);
      Buffer<uint16_t> qd(q.size()), ad(a.size()), bd(b.size()), ald(32),
          dtd(32), output(rows * output_width);
      Buffer<float> seed(state_count), state(state_count);
      qd.set(q);
      ad.set(a);
      bd.set(b);
      ald.set(al);
      dtd.set(dt);
      seed.set(initial);
      // Pageable H2D cudaMemcpy may return after staging, before device DMA
      // completes. The nonblocking measurement stream does not wait for the
      // default stream: explicitly finish setup before it reads the
      // seed/inputs.
      ck(cudaStreamSynchronize(nullptr));
      Events events;
      FerruleCoreBf16DeltaArgs args{uint32_t(rows),
                                    16,
                                    32,
                                    128,
                                    128,
                                    0,
                                    qd.addr(),
                                    ad.addr(),
                                    bd.addr(),
                                    ald.addr(),
                                    dtd.addr(),
                                    state.addr(),
                                    output.addr(),
                                    reinterpret_cast<uint64_t>(events.stream)};
      // Same addresses and identical initial state every time; never measure a
      // progressively decayed steady-state or include the reset copy in events.
      auto reset = [&]() {
        ck(cudaMemcpyAsync(state.p, seed.p, state_count * sizeof(float),
                           cudaMemcpyDeviceToDevice, events.stream));
      };
      auto snapshot = [&](bool old) {
        reset();
        launch(old, args);
        ck(cudaStreamSynchronize(events.stream));
        Snapshot s{output.host(), state.host()};
        for (float v : s.state)
          if (!std::isfinite(v))
            throw std::runtime_error("nonfinite SSM");
        for (uint16_t v : s.output)
          if ((v & 0x7f80) == 0x7f80)
            throw std::runtime_error("nonfinite BF16 output");
        return s;
      };
      Snapshot expected_old = snapshot(true), expected_new = snapshot(false);
      if (rows == 1) {
        exact(expected_old.output,
              read<uint16_t>(dir + "/expected-old-output.bf16", output_width),
              "old captured output mismatch");
        exact(expected_old.state,
              read<float>(dir + "/expected-old-state.f32", state_count),
              "old captured state mismatch");
        exact(expected_new.output,
              read<uint16_t>(dir + "/expected-new-output.bf16", output_width),
              "new HF output mismatch");
        exact(expected_new.state,
              read<float>(dir + "/expected-new-state.f32", state_count),
              "new HF state mismatch");
      } else {
        exact(expected_old.output, expected_new.output,
              "prefill output changed");
        exact(expected_old.state, expected_new.state, "prefill state changed");
      }
      auto validate_inputs = [&]() {
        exact(qd.host(), q, "QKV modified");
        exact(ad.host(), a, "A modified");
        exact(bd.host(), b, "B modified");
        exact(ald.host(), al, "A_log modified");
        exact(dtd.host(), dt, "dt_bias modified");
        exact(seed.host(), initial, "seed modified");
      };
      validate_inputs();
      for (int i = 0; i < warmups; ++i) {
        for (bool old : {i % 2 == 0, i % 2 != 0}) {
          reset();
          launch(old, args);
        }
      }
      ck(cudaStreamSynchronize(events.stream));
      auto sample = [&](bool old) {
        for (int i = 0; i < repeats; ++i) {
          reset();
          ck(cudaEventRecord(events.start[i], events.stream));
          launch(old, args);
          ck(cudaEventRecord(events.stop[i], events.stream));
        }
        ck(cudaEventSynchronize(events.stop.back()));
        std::array<double, repeats> us;
        for (int i = 0; i < repeats; ++i) {
          float ms;
          ck(cudaEventElapsedTime(&ms, events.start[i], events.stop[i]));
          if (!(ms > 0 && std::isfinite(ms)))
            throw std::runtime_error("invalid event duration");
          us[i] = ms * 1000.0;
        }
        // Validate the entire last invocation after events, for every sample.
        const auto &expected = old ? expected_old : expected_new;
        exact(output.host(), expected.output, "timed output drift");
        exact(state.host(), expected.state, "timed state drift");
        return us;
      };
      std::vector<double> old_means, new_means;
      for (int round = 0; round < rounds; ++round) {
        std::array<double, repeats> old_us, new_us;
        if (round % 2 == 0) {
          old_us = sample(true);
          new_us = sample(false);
        } else {
          new_us = sample(false);
          old_us = sample(true);
        }
        double old_mean = 0, new_mean = 0;
        for (int i = 0; i < repeats; ++i) {
          old_mean += old_us[i] / repeats;
          new_mean += new_us[i] / repeats;
        }
        old_means.push_back(old_mean);
        new_means.push_back(new_mean);
        std::cout << "{\"event\":\"paired_sample\",\"rows\":" << rows
                  << ",\"round\":" << round << ",\"order\":\""
                  << (round % 2 == 0 ? "old/new" : "new/old")
                  << "\",\"old_mean_us\":" << old_mean
                  << ",\"new_mean_us\":" << new_mean
                  << ",\"new_over_old\":" << new_mean / old_mean
                  << ",\"old_us\":[";
        for (int i = 0; i < repeats; ++i)
          std::cout << (i ? "," : "") << old_us[i];
        std::cout << "],\"new_us\":[";
        for (int i = 0; i < repeats; ++i)
          std::cout << (i ? "," : "") << new_us[i];
        std::cout << "]}\n";
      }
      validate_inputs();
      stats(rows, true, old_means);
      stats(rows, false, new_means);
      std::cout << "{\"event\":\"correctness\",\"rows\":" << rows
                << ",\"full_state_output\":true,\"input_immutable\":true,"
                   "\"guards\":true}\n";
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
