#pragma once

#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <cstdint>
// Like the F32 extension, compiled in the existing core TU while advertised
// by the CUTLASS provider. This reuses the exact FP8 decode/BF16 RNE primitives
// without assigning core/device.cuh to a second aggregate translation unit.
#include "core/device.cuh"
#include "core/cutlass_f32.cuh"

#if FERRULE_CUDA_TARGET_SM >= 80 && __has_include(<cutlass/gemm/device/gemm.h>)
#define FERRULE_CUTLASS_BF16_TENSOROP 1
#include <cutlass/gemm/device/gemm.h>
#else
#define FERRULE_CUTLASS_BF16_TENSOROP 0
#endif

struct FerruleCutlassBf16Args {
  uint32_t m, n, k, lda, ldb, ldd;
  uint64_t activation, weight, output, stream;
};
static_assert(sizeof(FerruleCutlassBf16Args) == 56);
static_assert(offsetof(FerruleCutlassBf16Args, activation) == 24);

// Numeric scales are multiplication factors, never E8M0 or reciprocals.
// Origins are residues within the first resident 128x128 scale block.
struct FerruleCutlassNumericFp8Args {
  uint32_t m, n, k, padded_k, tile_rows, scale_cols;
  uint32_t row_origin, column_origin, scale_bytes, lda, ldd, reserved;
  uint64_t activation, weight, scales, output, workspace, workspace_bytes, stream;
};
static_assert(sizeof(FerruleCutlassNumericFp8Args) == 104);
static_assert(offsetof(FerruleCutlassNumericFp8Args, activation) == 48);
static_assert(offsetof(FerruleCutlassNumericFp8Args, stream) == 96);

namespace ferrule::cuda::cutlass::bf16_linear {
inline int validate(const FerruleCutlassBf16Args *a) {
  if (!a || !a->m || !a->n || !a->k || a->m > INT_MAX - 64u ||
      a->n > 64u * 65535u || a->k > INT_MAX - 32u ||
      a->lda < a->k || a->ldb < a->k || a->ldd < a->n ||
      a->lda > INT_MAX || a->ldb > INT_MAX || a->ldd > INT_MAX ||
      a->k % 8 || a->lda % 8 || a->ldb % 8 ||
      !a->activation || !a->weight || !a->output ||
      a->activation % 16 || a->weight % 16 || a->output % 4) return 1;
  return 0;
}
#if FERRULE_CUTLASS_BF16_TENSOROP
using Element = ::cutlass::bfloat16_t;
using Gemm = ::cutlass::gemm::device::Gemm<
    Element, ::cutlass::layout::RowMajor,
    Element, ::cutlass::layout::ColumnMajor,
    float, ::cutlass::layout::RowMajor, float,
    ::cutlass::arch::OpClassTensorOp, ::cutlass::arch::Sm80,
    ::cutlass::gemm::GemmShape<64, 64, 32>,
    ::cutlass::gemm::GemmShape<32, 32, 32>,
    ::cutlass::gemm::GemmShape<16, 8, 16>,
    ::cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, 8, 8, false, ::cutlass::arch::OpMultiplyAdd>;
inline Gemm::Arguments arguments(const FerruleCutlassBf16Args &a) {
  auto *output = reinterpret_cast<float *>(a.output);
  return {{int(a.m), int(a.n), int(a.k)},
          {reinterpret_cast<const Element *>(a.activation), int(a.lda)},
          {reinterpret_cast<const Element *>(a.weight), int(a.ldb)},
          {output, int(a.ldd)}, {output, int(a.ldd)}, {1.0f, 0.0f}, 1};
}
#endif
} // namespace ferrule::cuda::cutlass::bf16_linear

extern "C" int32_t ferrule_cutlass_bf16_available() {
  return FERRULE_CUTLASS_BF16_TENSOROP;
}
extern "C" int32_t ferrule_cutlass_bf16_can_implement(const FerruleCutlassBf16Args *a) {
  using namespace ferrule::cuda::cutlass::bf16_linear;
  if (int status = validate(a)) return status;
#if FERRULE_CUTLASS_BF16_TENSOROP
  auto args = arguments(*a);
  if (Gemm::get_workspace_size(args) != 0) return 2;
  return Gemm::can_implement(args) == ::cutlass::Status::kSuccess ? 0 : 3;
#else
  return 2;
#endif
}
extern "C" int32_t ferrule_cutlass_bf16_launch(const FerruleCutlassBf16Args *a) {
  if (int status = ferrule_cutlass_bf16_can_implement(a)) return status;
#if FERRULE_CUTLASS_BF16_TENSOROP
  using namespace ferrule::cuda::cutlass::bf16_linear;
  Gemm gemm;
  auto stream = reinterpret_cast<cudaStream_t>(a->stream);
  auto status = gemm.initialize(arguments(*a), nullptr, stream);
  if (status == ::cutlass::Status::kSuccess) status = gemm.run(stream);
  return status == ::cutlass::Status::kSuccess ? 0 : 1000 + int(status);
#else
  return 2;
#endif
}

namespace ferrule::cuda::cutlass::numeric_fp8 {
using namespace ferrule::cuda::core;
inline FerruleCutlassBf16Args gemm_args(const FerruleCutlassNumericFp8Args &a,
                                       uint32_t first, uint32_t count) {
  return {a.m, count, a.padded_k, a.padded_k, a.padded_k, a.ldd,
          a.workspace, a.workspace + uint64_t(a.m) * a.padded_k * 2,
          a.output + uint64_t(first) * 4, a.stream};
}
__global__ void pack_activation(FerruleCutlassNumericFp8Args a) {
  const uint64_t count = uint64_t(a.m) * a.padded_k;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x) * blockDim.x) {
    uint32_t col = i % a.padded_k;
    pointer<uint16_t>(a.workspace)[i] = col < a.k
        ? bf16_rne(const_pointer<float>(a.activation)[(i / a.padded_k) * a.lda + col])
        : 0;
  }
}
__global__ void decode_weights(FerruleCutlassNumericFp8Args a, uint32_t first,
                               uint32_t rows) {
  auto *dst = pointer<uint16_t>(a.workspace + uint64_t(a.m) * a.padded_k * 2);
  const uint64_t count = uint64_t(rows) * a.padded_k;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x) * blockDim.x) {
    uint32_t col = i % a.padded_k;
    uint32_t row = first + i / a.padded_k;
    uint16_t value = 0;
    if (col < a.k) {
      const uint64_t scale_index = uint64_t((a.row_origin + row) / 128) * a.scale_cols
                                   + (a.column_origin + col) / 128;
      const float scale = a.scale_bytes == 4 ? const_pointer<float>(a.scales)[scale_index]
          : bf16_value(const_pointer<uint16_t>(a.scales)[scale_index]);
      value = bf16_rne(__fmul_rn(fp8_value(const_pointer<uint8_t>(a.weight)[uint64_t(row) * a.k + col]), scale));
    }
    dst[i] = value;
  }
}
inline unsigned blocks(uint64_t count) {
  return unsigned(count > 256u * 65535u ? 65535u : (count + 255u) / 256u);
}
} // namespace ferrule::cuda::cutlass::numeric_fp8

extern "C" int32_t ferrule_cutlass_numeric_fp8_can_implement(const FerruleCutlassNumericFp8Args *a) {
  using namespace ferrule::cuda::cutlass::numeric_fp8;
  if (!a || !a->m || a->m > INT_MAX - 64u || !a->n || !a->k || a->k > INT_MAX - 32u ||
      a->n > INT_MAX - 128u || a->padded_k != (a->k + 7u) / 8u * 8u ||
      !a->tile_rows || a->tile_rows > a->n || a->tile_rows > 64u * 65535u ||
      a->row_origin >= 128 || a->column_origin >= 128 ||
      a->scale_cols != (a->column_origin + a->k + 127u) / 128u ||
      (a->scale_bytes != 2 && a->scale_bytes != 4) || a->reserved ||
      a->lda < a->k || a->lda > INT_MAX || a->ldd < a->n || a->ldd > INT_MAX ||
      !a->activation || !a->weight || !a->scales || !a->output || !a->workspace ||
      a->activation % 4 || a->scales % a->scale_bytes ||
      a->workspace_bytes < (uint64_t(a->m) + a->tile_rows) * a->padded_k * 2) return 1;
  // Preflight both distinct tile shapes before even packing the activation.
  auto first = gemm_args(*a, 0, a->tile_rows);
  if (int status = ferrule_cutlass_bf16_can_implement(&first)) return status;
  uint32_t offset = (a->n - 1u) / a->tile_rows * a->tile_rows;
  auto last = gemm_args(*a, offset, a->n - offset);
  return ferrule_cutlass_bf16_can_implement(&last);
}
extern "C" int32_t ferrule_cutlass_numeric_fp8_launch(const FerruleCutlassNumericFp8Args *a) {
  if (int status = ferrule_cutlass_numeric_fp8_can_implement(a)) return status;
  using namespace ferrule::cuda::cutlass::numeric_fp8;
  auto stream = reinterpret_cast<cudaStream_t>(a->stream);
  pack_activation<<<blocks(uint64_t(a->m) * a->padded_k), 256, 0, stream>>>(*a);
  if (cudaGetLastError() != cudaSuccess) return 4;
  for (uint32_t first = 0; first < a->n;) {
    uint32_t count = a->n - first < a->tile_rows ? a->n - first : a->tile_rows;
    decode_weights<<<blocks(uint64_t(count) * a->padded_k), 256, 0, stream>>>(*a, first, count);
    if (cudaGetLastError() != cudaSuccess) return 4;
    auto g = gemm_args(*a, first, count);
    if (int status = ferrule_cutlass_bf16_launch(&g)) return status;
    first += count;
  }
  return 0;
}

namespace ferrule::cuda::cutlass::numeric_fp8 {
// The F32 profile borrows A directly: no activation rounding, copy or scratch.
inline FerruleCutlassF32Args f32_gemm_args(const FerruleCutlassNumericFp8Args &a,
                                         uint32_t first, uint32_t count) {
  return {a.m, count, a.k, a.lda, a.k, a.ldd, a.activation, a.workspace,
          a.output + uint64_t(first) * 4, a.stream};
}
__global__ void decode_weights_f32(FerruleCutlassNumericFp8Args a,
                                   uint32_t first, uint32_t rows) {
  auto *dst = pointer<float>(a.workspace);
  const uint64_t count = uint64_t(rows) * a.k;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint32_t col = i % a.k;
    const uint32_t row = first + i / a.k;
    const uint64_t scale_index = uint64_t((a.row_origin + row) / 128) * a.scale_cols
                                 + (a.column_origin + col) / 128;
    const float scale = a.scale_bytes == 4 ? const_pointer<float>(a.scales)[scale_index]
        : bf16_value(const_pointer<uint16_t>(a.scales)[scale_index]);
    dst[i] = __fmul_rn(fp8_value(const_pointer<uint8_t>(a.weight)[uint64_t(row) * a.k + col]), scale);
  }
}
} // namespace ferrule::cuda::cutlass::numeric_fp8

extern "C" int32_t ferrule_cutlass_numeric_fp8_f32_can_implement(const FerruleCutlassNumericFp8Args *a) {
  using namespace ferrule::cuda::cutlass::numeric_fp8;
  if (!a || !a->m || a->m > INT_MAX - 64u || !a->n || a->n > INT_MAX - 128u ||
      !a->k || a->k > INT_MAX - 32u || a->padded_k != a->k ||
      !a->tile_rows || a->tile_rows > a->n || a->tile_rows > 64u * 65535u ||
      a->row_origin >= 128 || a->column_origin >= 128 ||
      a->scale_cols != (a->column_origin + a->k + 127u) / 128u ||
      (a->scale_bytes != 2 && a->scale_bytes != 4) || a->reserved ||
      a->lda < a->k || a->lda > INT_MAX || a->ldd < a->n || a->ldd > INT_MAX ||
      !a->activation || !a->weight || !a->scales || !a->output || !a->workspace ||
      a->activation % 4 || a->scales % a->scale_bytes || a->workspace % 16 ||
      a->workspace_bytes < uint64_t(a->tile_rows) * a->k * 4) return 1;
  // Preflight every distinct GEMM shape before writing scratch or output.
  auto first = f32_gemm_args(*a, 0, a->tile_rows);
  if (int status = ferrule_cutlass_f32_can_implement(&first)) return status;
  const uint32_t offset = (a->n - 1u) / a->tile_rows * a->tile_rows;
  auto last = f32_gemm_args(*a, offset, a->n - offset);
  return ferrule_cutlass_f32_can_implement(&last);
}
extern "C" int32_t ferrule_cutlass_numeric_fp8_f32_launch(const FerruleCutlassNumericFp8Args *a) {
  if (int status = ferrule_cutlass_numeric_fp8_f32_can_implement(a)) return status;
  using namespace ferrule::cuda::cutlass::numeric_fp8;
  auto stream = reinterpret_cast<cudaStream_t>(a->stream);
  for (uint32_t first = 0; first < a->n;) {
    const uint32_t count = a->n - first < a->tile_rows ? a->n - first : a->tile_rows;
    decode_weights_f32<<<blocks(uint64_t(count) * a->k), 256, 0, stream>>>(*a, first, count);
    if (cudaGetLastError() != cudaSuccess) return 4;
    auto g = f32_gemm_args(*a, first, count);
    if (int status = ferrule_cutlass_f32_launch(&g)) return status;
    first += count;
  }
  return 0;
}
