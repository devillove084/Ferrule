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

static_assert(kTransformerGqaSharedTokens == 2048,
              "regressions must cross the original 2048-token boundary");

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
    launch_transformer_causal_gqa_f32(a);
  else
    gqa_legacy_oracle_kernel<float>
        <<<blocks_for(static_cast<uint64_t>(a.rows) * a.q_heads * a.head_dim),
           kBlock>>>(a);
  check(cudaPeekAtLastError());
}
// Inspect the production launch helper without executing a huge-grid kernel.
static void check_launch_path(FerruleCoreTransformerArgs a, bool cooperative) {
  cudaStream_t capture;
  check(cudaStreamCreateWithFlags(&capture, cudaStreamNonBlocking));
  a.stream = reinterpret_cast<uint64_t>(capture);
  check(cudaStreamBeginCapture(capture, cudaStreamCaptureModeThreadLocal));
  launch_transformer_causal_gqa_f32(a);
  check(cudaPeekAtLastError());
  cudaGraph_t graph;
  check(cudaStreamEndCapture(capture, &graph));
  size_t count = 0;
  check(cudaGraphGetNodes(graph, nullptr, &count));
  require(count == 1, "GQA launch must be a single allocation-free kernel");
  cudaGraphNode_t node;
  check(cudaGraphGetNodes(graph, &node, &count));
  cudaGraphNodeType type;
  check(cudaGraphNodeGetType(node, &type));
  require(type == cudaGraphNodeTypeKernel, "expected GQA kernel node");
  cudaKernelNodeParams params{};
  check(cudaGraphKernelNodeGetParams(node, &params));
  const void *expected = cooperative
      ? reinterpret_cast<const void *>(transformer_causal_gqa_cooperative_f32_kernel)
      : reinterpret_cast<const void *>(transformer_causal_gqa_kernel<float>);
  require(params.func == expected, "production GQA selected the wrong kernel");
  require(params.sharedMemBytes == 0, "unexpected dynamic shared scratch");
  check(cudaGraphDestroy(graph));
  check(cudaStreamDestroy(capture));
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
  if (pattern == 2 || pattern == 3) {
    std::fill(hq.begin(), hq.end(), 1.0f);
    for (uint32_t s = 0; s < a.sequences; ++s) {
      for (uint32_t t = 0; t < ctx; ++t) {
        const uint64_t base = hs[ho[s] + t / 16] * a.key_slot_stride_bytes +
                              a.layer_index * a.key_layer_stride_bytes +
                              (t % 16) * a.key_token_stride_bytes;
        for (uint32_t h = 0; h < kvh; ++h)
          for (uint32_t d = 0; d < dim; ++d)
            hk[(base + h * a.key_head_stride_bytes) / 4 + d] =
                static_cast<int>(t % 17) - 8.0f +
                (pattern == 3 ? 32.0f * (t / 2048) : 0.0f);
      }
    }
  }
  Device<float> q(hq), k(hk), v(hv),
      out(std::vector<float>(a.output_bytes / 4 + 8, -777.25f)),
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
  check_launch_path(a, true);
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
  if (ctx == 1 || ctx == 2049) {
    len.upload(std::vector<int32_t>(len.count, 0));
    out.upload(std::vector<float>(out.count, -777.25f));
    launch(a, false);
    const auto empty_expected = out.download();
    out.upload(std::vector<float>(out.count, -777.25f));
    launch(a, true);
    bitwise(out.download(), empty_expected);
    for (uint32_t r = 0; r < a.rows; ++r)
      for (uint32_t h = 0; h < qh; ++h)
        for (uint32_t d = 0; d < dim; ++d)
          require(empty_expected[(r * a.output_row_stride_bytes +
                                  h * a.output_head_stride_bytes) / 4 + d] == 0.0f,
                  "empty visible interval must produce zero");
    len.upload(hl);
  }
  if (ctx == 2049 && dim == 3 && !ragged) {
    const auto compare = [&]() {
      out.upload(std::vector<float>(out.count, -777.25f));
      launch(a, false);
      const auto reference = out.download();
      out.upload(std::vector<float>(out.count, -777.25f));
      launch(a, true);
      bitwise(out.download(), reference);
      require(status.download()[0] == 0,
              "nonfinite inputs are arithmetic, not metadata failures");
      return reference;
    };
    // Inject in the first token and on both sides of the tile boundary. Do
    // not replace the legacy policy with generic nonfinite rejection/zeroing.
    for (uint32_t token : {0u, 2047u, 2048u}) {
      const uint64_t key_base = hs[token / 16] * a.key_slot_stride_bytes +
          a.layer_index * a.key_layer_stride_bytes +
          (token % 16) * a.key_token_stride_bytes;
      const uint64_t value_base = hs[token / 16] * a.value_slot_stride_bytes +
          a.layer_index * a.value_layer_stride_bytes +
          (token % 16) * a.value_token_stride_bytes;
      for (float special : {NAN, INFINITY, -INFINITY}) {
        auto keys = hk;
        keys[key_base / 4] = special;
        k.upload(keys);
        compare();
        k.upload(hk);
        auto values = hv;
        values[value_base / 4] = special;
        v.upload(values);
        compare();
        v.upload(hv);
      }
    }
    for (float special : {NAN, INFINITY, -INFINITY}) {
      auto queries = hq;
      queries[0] = special;
      q.upload(queries);
      compare();
      q.upload(hq);
    }
    // Equal infinite scores use weight=1, not exp(Inf-Inf). A later NaN
    // poisons the denominator, for which the frozen contract returns zero.
    q.upload(std::vector<float>(q.count, 1.0f));
    for (float special : {INFINITY, -INFINITY}) {
      k.upload(std::vector<float>(k.count, special));
      const auto equal_inf = compare();
      require(std::isfinite(equal_inf[0]), "infinite score ties must stay finite");
    }
    k.upload(std::vector<float>(k.count, NAN));
    require(compare()[0] == 0.0f, "NaN denominator must retain legacy zero output");
    q.upload(hq);
    k.upload(hk);

    // A poisoned future tile must not be read, whether bounded by causal
    // position or by KV length. Compare against the unpoisoned same interval.
    for (bool causal : {true, false}) {
      if (causal) pos.upload({2047});
      else len.upload({2048});
      const auto masked = compare();
      auto keys = hk;
      auto values = hv;
      const uint64_t slot = hs[2048 / 16];
      for (uint32_t h = 0; h < kvh; ++h) {
        keys[(slot * a.key_slot_stride_bytes +
              a.layer_index * a.key_layer_stride_bytes +
              h * a.key_head_stride_bytes) / 4] = NAN;
        values[(slot * a.value_slot_stride_bytes +
                a.layer_index * a.value_layer_stride_bytes +
                h * a.value_head_stride_bytes) / 4] = NAN;
      }
      k.upload(keys);
      v.upload(values);
      bitwise(compare(), masked);
      k.upload(hk);
      v.upload(hv);
      pos.upload(hp);
      len.upload(hl);
    }
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
    for (int fault = 0; fault < (ctx > 2048 ? 5 : 4); ++fault) {
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
      if (fault == 4) {
        auto invalid = hs;
        invalid[2048 / 16] = -1; // Failure in a later tile, after valid replay.
        slots.upload(invalid);
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
  if (argc == 2 && std::strcmp(argv[1], "--compile-only") == 0)
    return 0; // Deliberately before any CUDA runtime/device operation.
  check(cudaSetDevice(0));
  cudaFuncAttributes attributes{};
  check(cudaFuncGetAttributes(&attributes,
                             transformer_causal_gqa_cooperative_f32_kernel));
  require(attributes.sharedSizeBytes <= 11 * 1024,
          "GQA scratch must remain fixed at about 10 KiB per block");
  FerruleCoreTransformerArgs huge{};
  huge.rows = INT32_MAX;
  huge.q_heads = 2;
  huge.head_dim = 1;
  check_launch_path(huge, false); // Capture only; invalid pointers are never run.
  cudaDeviceProp device;
  check(cudaGetDeviceProperties(&device, 0));
  std::printf("GPU=%s sm%d%d shared_cap=%u\n", device.name, device.major,
              device.minor, kTransformerGqaSharedTokens);
  if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
    for (uint32_t ctx : {2048u, 2049u, 4096u, 4106u, 8192u})
      run(ctx, 256, 16, 2, false, false, 0, true);
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "--sanitize") == 0) {
    run(64, 33, 6, 2, true, true);
    run(16, 257, 5, 1, true, true);
    run(2049, 513, 3, 1, true, true, 3);
    run(4106, 33, 3, 3, true, true, 3);
    std::puts("sanitizer fixtures: bitwise PASS");
    return 0;
  }
  for (uint32_t ctx : {1u, 16u, 64u, 1024u, 2048u, 2049u})
    for (uint32_t dim : {3u, 31u, 33u, 128u, 256u, 257u})
      run(ctx, dim, 6, 2);
  run(64, 513, 5, 1, true, true);
  run(65, 129, 9, 3, true, true);
  run(2050, 33, 3, 3, true, true); // Mixed one/two-tile rows.
  run(64, 256, 16, 2, true, false, 1);
  run(64, 256, 16, 2, true, false, 2);
  for (uint32_t ctx : {4096u, 4106u, 8192u}) {
    run(ctx, 33, 6, 2, true, true);
    run(ctx, 33, 6, 2, true, true, 3); // New maxima in every tile.
  }
  run(2049, 513, 5, 1, true, true, 3);
  run(4106, 33, 6, 2, true, true, 1); // Ties across tiles.
  std::puts("49 fixtures plus semantic subcases: attention/logits bitwise PASS");
}
