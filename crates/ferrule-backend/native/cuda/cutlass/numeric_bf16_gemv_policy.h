#pragma once
#include <cstdint>

namespace ferrule::cuda::cutlass::numeric_bf16_gemv {
// Evidence and the external-memory admission envelope are specific to cuBLAS
// 120405. Header versions are not a substitute for the loaded library version.
constexpr int kSupportedCublasVersion = 120405;
constexpr bool supported_build(int sm, int major, int minor) {
  return sm >= 80 && major == 12 && minor == 4;
}
constexpr bool supported_runtime(int version) {
  return version == kSupportedCublasVersion;
}
constexpr uint32_t kN = 2048, kK = 4096, kTile = 257;
constexpr uint64_t kStaging = uint64_t(1 + kTile) * kK * 2;
constexpr uint64_t kScratch = kStaging + kN * 4;
constexpr bool valid_scratch(uint64_t address, uint64_t bytes) {
  return address && address % 16 == 0 && bytes >= kScratch &&
      address <= UINT64_MAX - kScratch;
}
} // namespace ferrule::cuda::cutlass::numeric_bf16_gemv
