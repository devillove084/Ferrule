#include "ferrule_cutlass.h"

#include <cutlass/version.h>

static_assert(CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 6 && CUTLASS_PATCH == 1,
              "Ferrule's SM121a provider is pinned to CUTLASS 4.6.1");

extern "C" uint32_t ferrule_cutlass_abi_version(void) {
  return FERRULE_CUTLASS_ABI_VERSION;
}

extern "C" uint32_t ferrule_cutlass_version(void) {
  return CUTLASS_VERSION;
}

extern "C" uint32_t ferrule_cutlass_target_sm(void) {
  return 121u;
}
