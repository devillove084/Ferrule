#pragma once

#include "cutlass/abi.h"
#include "cutlass/target.cuh"

// Provider availability is stricter than target ISA availability. Extend this
// only when the provider compiles a complete implementation for another target.
#ifndef FERRULE_CUDA_HAS_NATIVE_GROUPED_FP4_MOE
#define FERRULE_CUDA_HAS_NATIVE_GROUPED_FP4_MOE                                \
  FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4
#endif

namespace ferrule::cuda::cutlass::providers::availability {

inline constexpr bool kFp8Projection = architectures::Target::kFp8MmaSync;
inline constexpr bool kBf16Compressor = architectures::Target::kBf16MmaSync;
inline constexpr bool kHyperConnectionProducer =
    architectures::Target::kBf16MmaSync;
inline constexpr bool kSharedFfn = architectures::Target::kFp8MmaSync;
inline constexpr bool kGroupedFp4Moe = FERRULE_CUDA_HAS_NATIVE_GROUPED_FP4_MOE;
inline constexpr bool kMlaOutput = architectures::Target::kFp8MmaSync;
inline constexpr bool kMainProjectNorm = architectures::Target::kFp8MmaSync;
inline constexpr bool kHybridMlaAttention = architectures::Target::kBf16MmaSync;
inline constexpr bool kProposalHead = architectures::Target::kBf16MmaSync;

} // namespace ferrule::cuda::cutlass::providers::availability

namespace {
namespace provider_availability =
    ferrule::cuda::cutlass::providers::availability;
} // namespace

extern "C" FerruleCutlassProviderManifest
ferrule_cutlass_provider_manifest(void) {
  return FerruleCutlassProviderManifest{
      (provider_availability::kFp8Projection
           ? FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_FP8_QUERY_A_KV)
           : 0ull) |
          (provider_availability::kBf16Compressor
               ? FERRULE_CUTLASS_KERNEL_BIT(
                     FERRULE_CUTLASS_KERNEL_BF16_COMPRESSOR)
               : 0ull) |
          (provider_availability::kHyperConnectionProducer
               ? FERRULE_CUTLASS_KERNEL_BIT(
                     FERRULE_CUTLASS_KERNEL_HYPER_CONNECTION_PRODUCER)
               : 0ull) |
          (provider_availability::kSharedFfn
               ? FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_SHARED_FFN)
               : 0ull) |
          (provider_availability::kGroupedFp4Moe &&
                   FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
               ? FERRULE_CUTLASS_KERNEL_BIT(
                     FERRULE_CUTLASS_KERNEL_GROUPED_FP4_MOE)
               : 0ull) |
          (provider_availability::kMlaOutput
               ? FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_MLA_OUTPUT)
               : 0ull) |
          (provider_availability::kMainProjectNorm
               ? FERRULE_CUTLASS_KERNEL_BIT(
                     FERRULE_CUTLASS_KERNEL_MAIN_PROJECT_NORM)
               : 0ull) |
          (provider_availability::kHybridMlaAttention
               ? FERRULE_CUTLASS_KERNEL_BIT(
                     FERRULE_CUTLASS_KERNEL_HYBRID_MLA_ATTENTION)
               : 0ull) |
          (provider_availability::kProposalHead
               ? FERRULE_CUTLASS_KERNEL_BIT(
                     FERRULE_CUTLASS_KERNEL_PROPOSAL_HEAD)
               : 0ull) |
          (provider_availability::kFp8Projection
               ? FERRULE_CUTLASS_KERNEL_BIT(
                     FERRULE_CUTLASS_KERNEL_FP8_PROJECTION)
               : 0ull),
  };
}
