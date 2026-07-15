#ifndef FERRULE_CUTLASS_H_
#define FERRULE_CUTLASS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FERRULE_CUTLASS_ABI_VERSION 3u

#ifndef FERRULE_CUTLASS_TARGET_SM
#define FERRULE_CUTLASS_TARGET_SM 0u
#endif

#define FERRULE_CUTLASS_KERNEL_F32_SIMT 1u
#define FERRULE_CUTLASS_KERNEL_BF16_MMA_SYNC 2u
#define FERRULE_CUTLASS_KERNEL_BIT(id) (1ull << ((id) - 1u))

typedef enum FerruleCutlassStatus {
  FERRULE_CUTLASS_SUCCESS = 0,
  FERRULE_CUTLASS_INVALID_ABI = 1,
  FERRULE_CUTLASS_INVALID_ARGUMENT = 2,
  FERRULE_CUTLASS_UNSUPPORTED = 3,
  FERRULE_CUTLASS_LAUNCH_FAILED = 4,
} FerruleCutlassStatus;

typedef struct FerruleCutlassProviderManifest {
  uint32_t abi_version;
  uint32_t cutlass_version;
  uint32_t target_sm;
  uint32_t kernel_count;
  uint64_t kernel_mask;
} FerruleCutlassProviderManifest;

// C = alpha * A * transpose(B_row_major) + beta * C.
// A is [m, k] row-major, B is [n, k] row-major, and C is [m, n]
// row-major. Ferrule owns every pointer, allocation, and stream.
typedef struct FerruleCutlassGemmF32Args {
  uint32_t abi_version;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint64_t a;
  uint64_t b;
  uint64_t c;
  uint64_t stream;
  uint32_t lda;
  uint32_t ldb;
  uint32_t ldc;
  uint32_t reserved0;
  float alpha;
  float beta;
  uint32_t reserved1;
  uint32_t reserved2;
} FerruleCutlassGemmF32Args;

// BF16 Tensor Core projection with F32 input/output. The provider packs A into
// the caller-owned BF16 workspace before GEMM. B points to row-major BF16 bits
// [n, k]. No allocation is permitted inside the provider.
typedef struct FerruleCutlassGemmBf16F32Args {
  uint32_t abi_version;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint64_t a_f32;
  uint64_t b_bf16;
  uint64_t c_f32;
  uint64_t workspace_bf16;
  uint64_t workspace_bytes;
  uint64_t stream;
  uint32_t lda;
  uint32_t ldb;
  uint32_t ldc;
  uint32_t reserved0;
  float alpha;
  float beta;
  uint32_t reserved1;
  uint32_t reserved2;
} FerruleCutlassGemmBf16F32Args;

FerruleCutlassProviderManifest ferrule_cutlass_provider_manifest(void);
uint32_t ferrule_cutlass_abi_version(void);
uint32_t ferrule_cutlass_version(void);
uint32_t ferrule_cutlass_target_sm(void);
int32_t
ferrule_cutlass_gemm_f32_can_implement(const FerruleCutlassGemmF32Args *args);
int32_t ferrule_cutlass_gemm_f32_launch(const FerruleCutlassGemmF32Args *args);
int32_t ferrule_cutlass_gemm_bf16_f32_can_implement(
    const FerruleCutlassGemmBf16F32Args *args);
int32_t
ferrule_cutlass_gemm_bf16_f32_launch(const FerruleCutlassGemmBf16F32Args *args);

#ifdef __cplusplus
}
#endif

#endif // FERRULE_CUTLASS_H_
