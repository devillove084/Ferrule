#ifndef FERRULE_CUDA_CUTLASS_ARCHITECTURES_PROFILE_CUH_
#define FERRULE_CUDA_CUTLASS_ARCHITECTURES_PROFILE_CUH_

#include <cstdint>

#ifndef FERRULE_CUDA_TARGET_SM
#error "FERRULE_CUDA_TARGET_SM must be defined by the backend build"
#endif

#if FERRULE_CUDA_TARGET_SM < 80
#error "Ferrule CUDA operators require SM80 or newer"
#endif

namespace ferrule::cuda::cutlass::architectures {

struct ActiveProfile {
  static constexpr std::uint32_t kComputeCapability = FERRULE_CUDA_TARGET_SM;
  static constexpr bool kBf16MmaSync = FERRULE_CUDA_TARGET_SM >= 80;
  static constexpr bool kFp8MmaSync = FERRULE_CUDA_TARGET_SM >= 89;
  static constexpr bool kSm1xxUmma = FERRULE_CUDA_TARGET_SM == 100 ||
                                     FERRULE_CUDA_TARGET_SM == 101 ||
                                     FERRULE_CUDA_TARGET_SM == 110;
  static constexpr bool kSm103BlockScaledFp4 = FERRULE_CUDA_TARGET_SM == 103;
  static constexpr bool kSm12xMxfp4MmaSync =
      FERRULE_CUDA_TARGET_SM == 120 || FERRULE_CUDA_TARGET_SM == 121;

  // Conservative catalog default. Individual operators may use occupancy APIs
  // when their cooperative launch geometry depends on runtime resources.
  static constexpr std::uint32_t kCooperativeBlockLimit = 160;
};

static_assert(ActiveProfile::kBf16MmaSync);
static_assert(ActiveProfile::kFp8MmaSync,
              "the current semantic catalog requires FP8 mma.sync");

} // namespace ferrule::cuda::cutlass::architectures

#ifndef FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4
#define FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4 (FERRULE_CUDA_TARGET_SM == 103)
#endif

#ifndef FERRULE_CUDA_HAS_SM12X_MXFP4_MMA_SYNC
#define FERRULE_CUDA_HAS_SM12X_MXFP4_MMA_SYNC                                  \
  ((FERRULE_CUDA_TARGET_SM == 120) || (FERRULE_CUDA_TARGET_SM == 121))
#endif

// Operator availability is stricter than ISA capability. Extend this only when
// a grouped FP4 MoE implementation for another architecture is compiled below.
#ifndef FERRULE_CUDA_HAS_NATIVE_GROUPED_FP4_MOE
#define FERRULE_CUDA_HAS_NATIVE_GROUPED_FP4_MOE                                \
  FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4
#endif

#endif // FERRULE_CUDA_CUTLASS_ARCHITECTURES_PROFILE_CUH_
