#ifndef FERRULE_CUDA_CUTLASS_ARCHITECTURES_MMA_CUH_
#define FERRULE_CUDA_CUTLASS_ARCHITECTURES_MMA_CUH_

#include "profile.cuh"

#include <cstdint>

namespace ferrule::cuda::cutlass::architectures {

__device__ __forceinline__ void fp8_e4m3_m16n8k32(float (&accumulator)[4],
                                                  const std::uint32_t (&a)[4],
                                                  const std::uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(accumulator[0]), "+f"(accumulator[1]),
                 "+f"(accumulator[2]), "+f"(accumulator[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}

} // namespace ferrule::cuda::cutlass::architectures

#endif // FERRULE_CUDA_CUTLASS_ARCHITECTURES_MMA_CUH_
