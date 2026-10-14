#pragma once
#include <climits>
#include <cstdint>
namespace ferrule::cuda::cutlass::bf16_gemmex {
constexpr uint64_t kWorkspace = 32ull << 20, kProvider = 16ull << 20;
constexpr bool geometry(uint32_t m, uint32_t n, uint32_t k) {
  return m >= 1 && m <= 32 && n && n <= INT_MAX - 128u && k &&
         k <= INT_MAX - 32u && n % 128 == 0 && k % 128 == 0;
}
constexpr bool resident_geometry(uint32_t m, uint32_t n, uint32_t k) {
  return m >= 1 && m <= 32 && n && n <= INT_MAX && k && k <= INT_MAX;
}
constexpr uint64_t staging(uint32_t m, uint32_t n, uint32_t k) {
  return ((((uint64_t(m) + n) * k + uint64_t(m) * n) * 2) + 255) & ~uint64_t(255);
}
constexpr uint64_t required(uint32_t m, uint32_t n, uint32_t k) {
  return staging(m, n, k) + kWorkspace;
}
constexpr bool span(uint64_t ptr, uint64_t bytes, uint64_t alignment) {
  return ptr && ptr % alignment == 0 && bytes && bytes <= INT64_MAX &&
         ptr <= UINT64_MAX - bytes;
}
constexpr bool overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) {
  return a < b + bn && b < a + an;
}
constexpr bool storage(uint32_t m, uint32_t n, uint32_t k, uint64_t p,
                       uint64_t bytes) {
  return resident_geometry(m, n, k) && required(m, n, k) <= INT64_MAX &&
         bytes >= required(m, n, k) && span(p, required(m, n, k), 256);
}
} // namespace ferrule::cuda::cutlass::bf16_gemmex
