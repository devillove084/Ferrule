// Reuse the fenced benchmark's buffer/guard/event protocol without editing it.
// In this TU prior.cuh is the PINNED EXACT unfused kernel, not the older scalar
// kernel. Both frozen variants must match the same independent HF payloads.
#include "single_row_performance.inc"

int main(int argc, char **argv) {
  if (argc != 2 || std::string(argv[1]) != "--synthetic")
    return shared_performance_main(argc, argv);
  try {
    int runtime;
    ck(cudaRuntimeGetVersion(&runtime));
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0));
    if (runtime != 12040 || prop.major != 8 || prop.minor != 6)
      throw std::runtime_error("requires CUDA12.4/sm86");
    size_t cases = 0, words = 0;
    for (int dk : {64, 65, 67, 127, 128, 129, 255, 256, 257, 511, 512}) {
      for (auto shape : std::array<std::array<int, 3>, 4>{
               {{1, 1, 128}, {1, 4, 32}, {2, 32, 4}, {16, 32, 128}}}) {
        const int kh = shape[0], hv = shape[1], dv = shape[2];
        const size_t kw = kh * dk, vw = hv * dv, count = hv * dk * dv;
        for (bool zero_state : {false, true}) {
          std::vector<uint16_t> q(2 * kw + vw), a(hv), b(hv), al(hv), dt(hv);
          std::vector<float> seed(count);
          // Exactly representable BF16 values; non-neutral update and decay.
          auto bf = [](float x) {
            uint32_t u;
            memcpy(&u, &x, 4);
            return uint16_t(u >> 16);
          };
          for (size_t i = 0; i < q.size(); ++i)
            q[i] = bf((int(i * 17 % 211) - 105) / 256.0f);
          for (int h = 0; h < hv; ++h) {
            a[h] = bf((h % 11 - 5) / 8.0f);
            b[h] = bf((h % 13 - 6) / 8.0f);
            al[h] = bf(-0.5f);
            dt[h] = bf(0.125f);
          }
          for (size_t i = 0; i < count; ++i)
            seed[i] = zero_state ? 0.0f : (int(i * 13 % 173) - 86) / 997.0f;
          Buffer<uint16_t> qd(q.size()), ad(hv), bd(hv), ald(hv), dtd(hv),
              output(vw);
          Buffer<float> state(count);
          qd.set(q);
          ad.set(a);
          bd.set(b);
          ald.set(al);
          dtd.set(dt);
          FerruleCoreBf16DeltaArgs args{1,
                                        uint32_t(kh),
                                        uint32_t(hv),
                                        uint32_t(dk),
                                        uint32_t(dv),
                                        0,
                                        qd.addr(),
                                        ad.addr(),
                                        bd.addr(),
                                        ald.addr(),
                                        dtd.addr(),
                                        state.addr(),
                                        output.addr(),
                                        0};
          state.set(seed);
          launch(true, args);
          Snapshot expected{output.host(), state.host()};
          state.set(seed);
          launch(false, args);
          exact(output.host(), expected.output, "synthetic full BF16 output");
          exact(state.host(), expected.state, "synthetic full F32 SSM");
          exact(qd.host(), q, "QKV modified");
          exact(ad.host(), a, "A modified");
          exact(bd.host(), b, "B modified");
          exact(ald.host(), al, "A_log modified");
          exact(dtd.host(), dt, "dt_bias modified");
          ++cases;
          words += count + vw;
        }
      }
    }
    std::cout << "{\"event\":\"synthetic_exact\",\"cases\":" << cases
              << ",\"state_and_output_words\":" << words
              << ",\"guards\":true}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
