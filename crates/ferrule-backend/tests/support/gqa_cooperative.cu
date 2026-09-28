// Direct GPU oracle/benchmark: no extra production ABI, provider or GEMM.
#include "core/attention_ops.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace ferrule::cuda::core;
#include "gqa_legacy.cuh"

static void require(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}
static void check(cudaError_t error) {
  require(error == cudaSuccess, cudaGetErrorString(error));
}
template <typename T> struct Device {
  T *data;
  size_t count;
  explicit Device(const std::vector<T> &host) : count(host.size()) {
    check(cudaMalloc(&data, count * sizeof(T)));
    upload(host);
  }
  ~Device() { check(cudaFree(data)); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  uint64_t address() const { return reinterpret_cast<uint64_t>(data); }
  void upload(const std::vector<T> &host) {
    require(host.size() == count, "upload size mismatch");
    check(cudaMemcpy(data, host.data(), count * sizeof(T),
                     cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> host(count);
    check(cudaMemcpy(host.data(), data, count * sizeof(T),
                     cudaMemcpyDeviceToHost));
    return host;
  }
};
static std::vector<float> fixed_input(size_t count, uint32_t seed) {
  std::vector<float> result(count);
  for (float &v : result) {
    seed = seed * 1664525u + 1013904223u;
    v = (static_cast<int32_t>(seed >> 8) - 8388608) / 4194304.0f;
  }
  return result;
}
static void bitwise(const std::vector<float> &a, const std::vector<float> &b) {
  require(a.size() == b.size(), "comparison size mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(float))) {
      std::fprintf(stderr, "bitwise mismatch [%zu]: new=%.9g old=%.9g\n", i,
                   a[i], b[i]);
      std::exit(1);
    }
  }
}
// Fixed downstream projection tests logits too, without model/runtime changes.
__global__ void project(FerruleCoreTransformerArgs a, float *logits) {
  float sum = 0.0f;
  for (uint32_t h = 0; h < a.q_heads; ++h) {
    for (uint32_t d = 0; d < a.head_dim; ++d) {
      const float v = const_pointer<float>(
          a.output_f32 + blockIdx.x * a.output_row_stride_bytes +
          h * a.output_head_stride_bytes)[d];
      const float w =
          (static_cast<int>((h * a.head_dim + d + threadIdx.x * 7) % 29) - 14) *
          0.03125f;
      sum += v * w;
    }
  }
  logits[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}
static void launch(FerruleCoreTransformerArgs a, bool cooperative) {
  if (cooperative)
    transformer_causal_gqa_cooperative_f32_kernel<<<a.rows * a.q_heads,
                                                    kBlock>>>(a);
  else
    gqa_legacy_oracle_kernel<float>
        <<<blocks_for(static_cast<uint64_t>(a.rows) * a.q_heads * a.head_dim),
           kBlock>>>(a);
  check(cudaPeekAtLastError());
}
static float milliseconds(FerruleCoreTransformerArgs a, bool cooperative) {
  for (int i = 0; i < 5; ++i)
    launch(a, cooperative);
  cudaEvent_t start, end;
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&end));
  check(cudaEventRecord(start));
  for (int i = 0; i < 30; ++i)
    launch(a, cooperative);
  check(cudaEventRecord(end));
  check(cudaEventSynchronize(end));
  float ms;
  check(cudaEventElapsedTime(&ms, start, end));
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(end));
  return ms / 30;
}
static void run(uint32_t ctx, uint32_t dim, uint32_t qh, uint32_t kvh,
                bool ragged = false, bool padded = false, int pattern = 0,
                bool benchmark = false) {
  FerruleCoreTransformerArgs a{};
  a.kind = FERRULE_CORE_TRANSFORMER_PAGED_F32_CAUSAL_GQA;
  a.rows = ragged ? 5 : 1;
  a.sequences = ragged ? 2 : 1;
  a.q_heads = qh;
  a.kv_heads = kvh;
  a.head_dim = dim;
  a.page_tokens = 16;
  a.layer_index = 1;
  a.layer_count = 3;
  a.softmax_scale = 1.0f / std::sqrt(static_cast<float>(dim));
  const uint32_t pages = (ctx + 15) / 16;
  std::vector<int32_t> hs, ho{0}, hseq, hp, hl;
  for (uint32_t s = 0; s < a.sequences; ++s) {
    for (uint32_t p = 0; p < pages; ++p)
      // Shared read-only page zero, private later pages (post-COW), permuted
      // slots.
      hs.push_back(p == 0 && ragged ? 1 : 2 + s * pages + pages - 1 - p);
    ho.push_back(hs.size());
  }
  for (uint32_t r = 0; r < a.rows; ++r) {
    hseq.push_back(r % a.sequences);
    hp.push_back(
        ragged ? std::max(0, static_cast<int>(ctx) - 5 + static_cast<int>(r))
               : ctx - 1);
    hl.push_back(ctx); // Future initialized history must be causally masked.
  }
  a.query_head_stride_bytes = (dim + (padded ? 3 : 0)) * 4;
  a.query_row_stride_bytes = qh * a.query_head_stride_bytes + (padded ? 20 : 0);
  a.output_head_stride_bytes = a.query_head_stride_bytes;
  a.output_row_stride_bytes = a.query_row_stride_bytes;
  a.key_head_stride_bytes = (dim + (padded ? 5 : 0)) * 4;
  a.value_head_stride_bytes = (dim + (padded ? 7 : 0)) * 4;
  a.key_token_stride_bytes = kvh * a.key_head_stride_bytes;
  a.value_token_stride_bytes = kvh * a.value_head_stride_bytes;
  a.key_layer_stride_bytes = 16 * a.key_token_stride_bytes;
  a.value_layer_stride_bytes = 16 * a.value_token_stride_bytes;
  a.key_slot_stride_bytes = 3 * a.key_layer_stride_bytes;
  a.value_slot_stride_bytes = 3 * a.value_layer_stride_bytes;
  a.query_bytes = a.rows * a.query_row_stride_bytes;
  a.output_bytes = a.query_bytes;
  a.key_cache_bytes = (2 + pages * a.sequences) * a.key_slot_stride_bytes;
  a.value_cache_bytes = (2 + pages * a.sequences) * a.value_slot_stride_bytes;
  auto hq = fixed_input(a.query_bytes / 4, 31);
  auto hk = fixed_input(a.key_cache_bytes / 4, 137);
  auto hv = fixed_input(a.value_cache_bytes / 4, 733);
  if (pattern == 1)
    std::fill(hq.begin(), hq.end(), 0.0f); // Score ties.
  if (pattern == 2) {
    std::fill(hq.begin(), hq.end(), 1.0f);
    for (uint32_t s = 0; s < a.sequences; ++s) {
      for (uint32_t t = 0; t < ctx; ++t) {
        const uint64_t base = hs[ho[s] + t / 16] * a.key_slot_stride_bytes +
                              a.layer_index * a.key_layer_stride_bytes +
                              (t % 16) * a.key_token_stride_bytes;
        for (uint32_t h = 0; h < kvh; ++h)
          for (uint32_t d = 0; d < dim; ++d)
            hk[(base + h * a.key_head_stride_bytes) / 4 + d] =
                static_cast<int>(t % 17) - 8.0f;
      }
    }
  }
  Device<float> q(hq), k(hk), v(hv),
      out(std::vector<float>(a.output_bytes / 4, -777.25f)),
      logits(std::vector<float>(a.rows * 37));
  Device<int32_t> slots(hs), offsets(ho), seq(hseq), pos(hp), len(hl),
      status({0});
  a.query_f32 = q.address();
  a.key_cache_bf16 = k.address();
  a.value_cache_bf16 = v.address();
  a.output_f32 = out.address();
  a.status_i32 = status.address();
  a.status_count = 1;
  a.block_slots_i32 = slots.address();
  a.block_slots_count = slots.count;
  a.block_offsets_i32 = offsets.address();
  a.block_offsets_count = offsets.count;
  a.row_sequence_ids_i32 = seq.address();
  a.row_sequence_ids_count = seq.count;
  a.row_positions_i32 = pos.address();
  a.row_positions_count = pos.count;
  a.row_kv_lens_i32 = len.address();
  a.row_kv_lens_count = len.count;
  require(valid_transformer(&a, true), "native layout rejected fixture");
  launch(a, false);
  const auto expected = out.download();
  project<<<a.rows, 37>>>(a, logits.data);
  const auto expected_logits = logits.download();
  out.upload(std::vector<float>(out.count, -777.25f));
  launch(a, true);
  bitwise(out.download(), expected);
  project<<<a.rows, 37>>>(a, logits.data);
  bitwise(logits.download(), expected_logits);
  require(status.download()[0] == 0, "valid fixture reported metadata error");
  bitwise(k.download(), hk);
  bitwise(v.download(), hv);
  if (ctx == 1) {
    len.upload(std::vector<int32_t>(len.count, 0));
    launch(a, false);
    const auto empty_expected = out.download();
    launch(a, true);
    bitwise(out.download(), empty_expected);
    len.upload(hl);
  }
  if (benchmark) {
    std::vector<float> old_ms, new_ms;
    // Median batch means, alternating order. Events exclude copies/allocation.
    for (int r = 0; r < 5; ++r) {
      if (r % 2) {
        new_ms.push_back(milliseconds(a, true));
        old_ms.push_back(milliseconds(a, false));
      } else {
        old_ms.push_back(milliseconds(a, false));
        new_ms.push_back(milliseconds(a, true));
      }
    }
    std::sort(old_ms.begin(), old_ms.end());
    std::sort(new_ms.begin(), new_ms.end());
    std::printf("ctx=%u Q=%u KV=%u D=%u old_ms=%.6f new_ms=%.6f speedup=%.2fx "
                "bitwise=PASS\n",
                ctx, qh, kvh, dim, old_ms[2], new_ms[2], old_ms[2] / new_ms[2]);
  }
  if (ragged) {
    // Divergent score failures must not deadlock the cooperative barrier.
    for (int fault = 0; fault < 4; ++fault) {
      auto bad = a;
      if (fault == 0) {
        auto invalid = hs;
        invalid[0] = -1;
        slots.upload(invalid);
      }
      if (fault == 1)
        bad.key_cache_bytes = 4;
      if (fault == 2)
        bad.value_cache_bytes = 4;
      if (fault == 3) {
        auto invalid = hseq;
        invalid[0] = -1;
        seq.upload(invalid);
      }
      status.upload({0});
      launch(bad, false);
      const auto old_status = status.download();
      status.upload({0});
      launch(bad, true);
      const auto new_status = status.download();
      require(old_status[0] != 0 && old_status == new_status,
              "device error status mismatch");
      slots.upload(hs);
      seq.upload(hseq);
    }
  }
}
int main(int argc, char **argv) {
  check(cudaSetDevice(0));
  cudaDeviceProp device;
  check(cudaGetDeviceProperties(&device, 0));
  std::printf("GPU=%s sm%d%d shared_cap=%u\n", device.name, device.major,
              device.minor, kTransformerGqaSharedTokens);
  if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
    for (uint32_t ctx : {23u, 55u, 128u, 1024u})
      run(ctx, 256, 16, 2, false, false, 0, true);
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "--sanitize") == 0) {
    run(64, 33, 6, 2, true, true);
    run(16, 257, 5, 1, true, true);
    run(kTransformerGqaSharedTokens + 2, 3, 3, 3, true, true);
    std::puts("sanitizer fixtures: bitwise PASS");
    return 0;
  }
  for (uint32_t ctx : {1u, 16u, 64u, 1024u, kTransformerGqaSharedTokens,
                       kTransformerGqaSharedTokens + 1})
    for (uint32_t dim : {3u, 31u, 33u, 128u, 256u, 257u})
      run(ctx, dim, 6, 2);
  run(64, 513, 5, 1, true, true);
  run(65, 129, 9, 3, true, true);
  run(kTransformerGqaSharedTokens + 2, 33, 3, 3, true,
      true); // Mixed shared/fallback.
  run(64, 256, 16, 2, true, false, 1);
  run(64, 256, 16, 2, true, false, 2);
  std::puts("41 fixtures: old/new attention and projected logits bitwise PASS");
}
