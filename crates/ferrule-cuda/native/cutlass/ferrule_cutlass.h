#ifndef FERRULE_CUTLASS_H_
#define FERRULE_CUTLASS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FERRULE_CUTLASS_ABI_VERSION 1u

uint32_t ferrule_cutlass_abi_version(void);
uint32_t ferrule_cutlass_version(void);
uint32_t ferrule_cutlass_target_sm(void);

#ifdef __cplusplus
}
#endif

#endif  // FERRULE_CUTLASS_H_
