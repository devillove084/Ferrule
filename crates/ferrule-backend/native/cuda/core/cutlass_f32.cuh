#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <climits>

// This extension belongs to the existing CUTLASS provider. It is compiled in
// the core TU so the standard F32 entry does not need a new build/provider path.
#if FERRULE_CUDA_TARGET_SM >= 80 && \
    __has_include(<cutlass/gemm/device/gemm.h>)
#define FERRULE_CUTLASS_F32_TENSOROP 1
#include <cutlass/gemm/device/gemm.h>
#else
#define FERRULE_CUTLASS_F32_TENSOROP 0
#endif

struct FerruleCutlassF32Args {
  uint32_t m, n, k;
  uint32_t lda, ldb, ldd;
  uint64_t activation, weight, output, stream;
};
static_assert(sizeof(FerruleCutlassF32Args) == 56);
static_assert(alignof(FerruleCutlassF32Args) == 8);
static_assert(offsetof(FerruleCutlassF32Args, activation) == 24);
static_assert(offsetof(FerruleCutlassF32Args, stream) == 48);

namespace ferrule::cuda::cutlass::f32_linear {

// Status 1: invalid metadata; 2: unavailable TensorOp capability;
// 3: unsupported shape. CUTLASS failures are encoded as 1000 + status.
inline int32_t validate(const FerruleCutlassF32Args *a) {
  if (!a || !a->m || !a->n || !a->k ||
      a->m > INT_MAX - 64u || a->n > INT_MAX - 64u ||
      a->k > INT_MAX - 16u || a->lda > INT_MAX ||
      a->ldb > INT_MAX || a->ldd > INT_MAX ||
      a->lda < a->k || a->ldb < a->k || a->ldd < a->n ||
      !a->activation || !a->weight || !a->output ||
      a->activation % 4 || a->weight % 4 || a->output % 4) {
    return 1;
  }
  // Identity swizzle puts the N tiles on CUDA's bounded grid.y axis.
  if (a->n > 64u * 65535u) return 3;
  return 0;
}

#if FERRULE_CUTLASS_F32_TENSOROP
// Scalar global accesses support odd K/N and padded leading dimensions; the
// arithmetic is still SM80 TensorOp (three TF32 products), never SIMT GEMM.
// B is column-major [K,N], i.e. row-major checkpoint weights [N,K].
using Gemm = ::cutlass::gemm::device::Gemm<
    float, ::cutlass::layout::RowMajor,
    float, ::cutlass::layout::ColumnMajor,
    float, ::cutlass::layout::RowMajor, float,
    ::cutlass::arch::OpClassTensorOp, ::cutlass::arch::Sm80,
    ::cutlass::gemm::GemmShape<64, 64, 16>,
    ::cutlass::gemm::GemmShape<32, 32, 16>,
    ::cutlass::gemm::GemmShape<16, 8, 8>,
    ::cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, 1, 1, false, ::cutlass::arch::OpMultiplyAddFastF32>;

inline Gemm::Arguments arguments(const FerruleCutlassF32Args &a) {
  auto *output = reinterpret_cast<float *>(a.output);
  return {{int(a.m), int(a.n), int(a.k)},
          {reinterpret_cast<const float *>(a.activation), int(a.lda)},
          {reinterpret_cast<const float *>(a.weight), int(a.ldb)},
          {output, int(a.ldd)}, {output, int(a.ldd)}, {1.0f, 0.0f}, 1};
}
#endif

} // namespace ferrule::cuda::cutlass::f32_linear

extern "C" int32_t ferrule_cutlass_f32_available() {
  return FERRULE_CUTLASS_F32_TENSOROP;
}

extern "C" int32_t
ferrule_cutlass_f32_can_implement(const FerruleCutlassF32Args *a) {
  using namespace ferrule::cuda::cutlass::f32_linear;
  const int32_t status = validate(a);
  if (status) return status;
#if FERRULE_CUTLASS_F32_TENSOROP
  const auto args = arguments(*a);
  // No split-K, allocation, conversion buffer, or hidden workspace. Keep the
  // zero-workspace contract checked against the actual CUTLASS instantiation.
  if (Gemm::get_workspace_size(args) != 0) return 2;
  return Gemm::can_implement(args) == ::cutlass::Status::kSuccess ? 0 : 3;
#else
  return 2;
#endif
}

extern "C" int32_t
ferrule_cutlass_f32_launch(const FerruleCutlassF32Args *a) {
  const int32_t status = ferrule_cutlass_f32_can_implement(a);
  if (status) return status;
#if FERRULE_CUTLASS_F32_TENSOROP
  using namespace ferrule::cuda::cutlass::f32_linear;
  Gemm gemm;
  auto stream = reinterpret_cast<cudaStream_t>(a->stream);
  auto result = gemm.initialize(arguments(*a), nullptr, stream);
  if (result == ::cutlass::Status::kSuccess) result = gemm.run(stream);
  return result == ::cutlass::Status::kSuccess ? 0 : 1000 + int32_t(result);
#else
  return 2;
#endif
}
