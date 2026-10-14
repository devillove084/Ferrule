// Host-only expression-DAG proof: compile the actual production helper bodies
// with symbolic, ordered RN/BF16 operators, never algebraic simplification.
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
static std::map<std::tuple<std::string, uint64_t, uint64_t>, uint64_t> nodes;
static uint64_t node(const std::string &op, uint64_t a, uint64_t b = 0) {
  const auto key = std::make_tuple(op, a, b);
  auto found = nodes.find(key);
  if (found != nodes.end())
    return found->second;
  const auto id = nodes.size() + 1;
  nodes.emplace(key, id);
  return id;
}
struct Expr {
  uint64_t id;
  Expr() : Expr(0.0f) {}
  Expr(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    id = node("literal", bits);
  }
  static Expr make(const std::string &op, uint64_t a, uint64_t b = 0) {
    Expr result;
    result.id = node(op, a, b);
    return result;
  }
};
static Expr __fmul_rn(Expr a, Expr b) {
  return Expr::make("mul.rn", a.id, b.id);
}
static Expr __fadd_rn(Expr a, Expr b) {
  return Expr::make("add.rn", a.id, b.id);
}
static Expr __fsub_rn(Expr a, Expr b) {
  return Expr::make("sub.rn", a.id, b.id);
}
static Expr bf16_round(Expr a) { return Expr::make("bf16.rne.expand", a.id); }
static Expr load(uint64_t address, uint64_t index) {
  return Expr::make("bf16.load", address, index);
}
#include "single_row_helpers.inc"
static void same(Expr a, Expr b, const char *what) {
  if (a.id != b.id)
    throw std::runtime_error(what);
}
int main() {
  try {
    size_t cases = 0, state_words = 0;
    for (uint32_t dim = 64; dim <= 512; ++dim) {
      for (uint32_t stride : {1u, 4u, 7u}) {
        nodes.clear();
        const Expr kn = Expr::make("kn", 0), qn = Expr::make("qn", 0);
        const Expr decay = Expr::make("decay", 0),
                   scale = Expr::make("scale", 0);
        const Expr v = Expr::make("v", 0), beta = Expr::make("beta", 0);
        constexpr uint64_t qkv = 1234, k = 9000, q = 37;
        std::vector<Expr> before(dim * stride + 3);
        for (size_t i = 0; i < before.size(); ++i)
          before[i] = Expr::make("state", i);
        auto old = before, fused = before;
        for (uint32_t d = 0; d < dim; ++d)
          old[d * stride] = __fmul_rn(old[d * stride], decay);
        const Expr memory =
            column_sum4(old.data(), qkv, k, kn, 1.0f, dim, stride);
        const Expr fm = column_update_sum4<false>(
            fused.data(), qkv, k, q, kn, qn, scale, decay, dim, stride);
        same(memory, fm, "memory operation DAG changed");
        for (size_t i = 0; i < old.size(); ++i)
          same(old[i], fused[i], "decayed state/guard changed");
        const Expr correction = __fmul_rn(__fsub_rn(v, memory), beta);
        const Expr fc = __fmul_rn(__fsub_rn(v, fm), beta);
        same(correction, fc, "correction operation DAG changed");
        for (uint32_t d = 0; d < dim; ++d) {
          const Expr kd = bf16_round(__fmul_rn(load(qkv, k + d), kn));
          old[d * stride] =
              __fadd_rn(old[d * stride], __fmul_rn(kd, correction));
        }
        const Expr result =
            column_sum4(old.data(), qkv, q, qn, scale, dim, stride);
        const Expr fr = column_update_sum4<true>(fused.data(), qkv, k, q, kn,
                                                 qn, scale, fc, dim, stride);
        same(result, fr, "output operation DAG changed");
        same(bf16_round(result), bf16_round(fr), "BF16 boundary changed");
        for (size_t i = 0; i < old.size(); ++i)
          same(old[i], fused[i], "updated state/guard changed");
        ++cases;
        state_words += dim;
      }
    }
    std::cout << "{\"cases\":" << cases
              << ",\"state_words_per_stage\":" << state_words
              << ",\"ordered_dag_exact\":true,\"dimensions\":[64,512],"
                 "\"strides\":[1,4,7]}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
