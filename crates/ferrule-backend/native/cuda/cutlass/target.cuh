#ifndef FERRULE_CUDA_ARCHITECTURES_TARGET_CUH_
#define FERRULE_CUDA_ARCHITECTURES_TARGET_CUH_

#include <cstdint>

#ifndef FERRULE_CUDA_TARGET_SM
#error "FERRULE_CUDA_TARGET_SM must be defined by the backend build"
#endif

#ifndef FERRULE_CUDA_HAS_BASELINE_SIMT
#error "FERRULE_CUDA_HAS_BASELINE_SIMT must be defined by the backend build"
#endif

#ifndef FERRULE_CUDA_HAS_BF16_MMA_SYNC
#error "FERRULE_CUDA_HAS_BF16_MMA_SYNC must be defined by the backend build"
#endif

#ifndef FERRULE_CUDA_HAS_FP8_MMA_SYNC
#error "FERRULE_CUDA_HAS_FP8_MMA_SYNC must be defined by the backend build"
#endif

#ifndef FERRULE_CUDA_HAS_SM90_WGMMA
#error "FERRULE_CUDA_HAS_SM90_WGMMA must be defined by the backend build"
#endif

#ifndef FERRULE_CUDA_HAS_SM1XX_UMMA
#error "FERRULE_CUDA_HAS_SM1XX_UMMA must be defined by the backend build"
#endif

#ifndef FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4
#error                                                                         \
    "FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4 must be defined by the backend build"
#endif

#ifndef FERRULE_CUDA_HAS_SM12X_MXFP4_MMA_SYNC
#error                                                                         \
    "FERRULE_CUDA_HAS_SM12X_MXFP4_MMA_SYNC must be defined by the backend build"
#endif

#ifndef FERRULE_CUDA_HAS_NATIVE_GROUPED_FP4_MOE
#define FERRULE_CUDA_HAS_NATIVE_GROUPED_FP4_MOE \
  FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4
#endif

// Device-debug omits the SM103 grouped pipeline because CUTLASS requires the
// optimizer to avoid unreliable debug-register metadata for these kernels.
#ifndef FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
#if FERRULE_CUDA_HAS_NATIVE_GROUPED_FP4_MOE && \
    FERRULE_CUDA_TARGET_SM == 103 && !defined(__CUDACC_DEBUG__)
#define FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE 1
#else
#define FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE 0
#endif
#endif

#if FERRULE_CUDA_TARGET_SM < 80
#error "Ferrule CUDA operators require SM80 or newer"
#endif

namespace ferrule::cuda::cutlass::architectures {

struct Target {
  static constexpr std::uint32_t kComputeCapability = FERRULE_CUDA_TARGET_SM;
  static constexpr bool kBaselineSimt = FERRULE_CUDA_HAS_BASELINE_SIMT;
  static constexpr bool kBf16MmaSync = FERRULE_CUDA_HAS_BF16_MMA_SYNC;
  static constexpr bool kFp8MmaSync = FERRULE_CUDA_HAS_FP8_MMA_SYNC;
  static constexpr bool kSm90Wgmma = FERRULE_CUDA_HAS_SM90_WGMMA;
  static constexpr bool kSm1xxUmma = FERRULE_CUDA_HAS_SM1XX_UMMA;
  static constexpr bool kSm103BlockScaledFp4 =
      FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4;
  static constexpr bool kSm12xMxfp4MmaSync =
      FERRULE_CUDA_HAS_SM12X_MXFP4_MMA_SYNC;
};

static_assert(Target::kBf16MmaSync,
              "the CUTLASS provider requires BF16 mma.sync");

} // namespace ferrule::cuda::cutlass::architectures

#endif // FERRULE_CUDA_ARCHITECTURES_TARGET_CUH_
