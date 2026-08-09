#include "abi.h"

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

#include "dense.cc"
#include "attention.cc"
#include "moe.cc"

extern "C" std::uint64_t ferrule_cpu_capabilities() {
  std::uint64_t capabilities = 0;
#if (defined(__x86_64__) || defined(__i386__)) &&                              \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2")) {
    capabilities |= FERRULE_CPU_CAP_AVX2;
  }
#if defined(__GNUC__) && !defined(__clang__)
  if (__builtin_cpu_supports("avx512bf16")) {
    capabilities |= FERRULE_CPU_CAP_AVX512_BF16;
  }
#else
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  if (__get_cpuid_max(0, nullptr) >= 7 &&
      __get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx) != 0 &&
      (eax & (1u << 5u)) != 0) {
    capabilities |= FERRULE_CPU_CAP_AVX512_BF16;
  }
#endif
#endif
  return capabilities;
}
