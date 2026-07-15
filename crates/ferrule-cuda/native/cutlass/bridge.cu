#include "ferrule_cutlass.h"

#include <cuda_runtime_api.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/version.h>

static_assert(CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 6 && CUTLASS_PATCH == 1,
              "Ferrule's CUTLASS provider is pinned to CUTLASS 4.6.1");
static_assert(sizeof(FerruleCutlassProviderManifest) == 24,
              "Ferrule CUTLASS manifest ABI layout changed");
static_assert(sizeof(FerruleCutlassGemmF32Args) == 80,
              "Ferrule CUTLASS F32 GEMM ABI layout changed");
static_assert(sizeof(FerruleCutlassGemmBf16F32Args) == 96,
              "Ferrule CUTLASS BF16 GEMM ABI layout changed");

namespace {

using CutlassGemmF32 =
    cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float,
                                cutlass::layout::ColumnMajor, float,
                                cutlass::layout::RowMajor>;

using CutlassGemmBf16F32 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>, cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 8, 8>;

__global__ void ferrule_pack_f32_to_bf16(const float *input,
                                         cutlass::bfloat16_t *output,
                                         uint64_t count) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = cutlass::bfloat16_t(input[index]);
  }
}

bool valid(const FerruleCutlassGemmF32Args *args) {
  return args != nullptr && args->abi_version == FERRULE_CUTLASS_ABI_VERSION &&
         args->m > 0 && args->n > 0 && args->k > 0 && args->a != 0 &&
         args->b != 0 && args->c != 0 && args->lda >= args->k &&
         args->ldb >= args->k && args->ldc >= args->n && args->reserved0 == 0 &&
         args->reserved1 == 0 && args->reserved2 == 0;
}

bool valid(const FerruleCutlassGemmBf16F32Args *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION ||
      args->m == 0 || args->n == 0 || args->k == 0 || args->a_f32 == 0 ||
      args->b_bf16 == 0 || args->c_f32 == 0 || args->workspace_bf16 == 0 ||
      args->lda < args->k || args->ldb < args->k || args->ldc < args->n ||
      args->reserved0 != 0 || args->reserved1 != 0 || args->reserved2 != 0) {
    return false;
  }
  uint64_t elements = static_cast<uint64_t>(args->m) * args->k;
  return args->workspace_bytes >= elements * sizeof(cutlass::bfloat16_t);
}

CutlassGemmF32::Arguments
make_arguments(const FerruleCutlassGemmF32Args &args) {
  auto *a = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.a));
  auto *b = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.b));
  auto *c = reinterpret_cast<float *>(static_cast<uintptr_t>(args.c));
  return CutlassGemmF32::Arguments(
      {static_cast<int>(args.m), static_cast<int>(args.n),
       static_cast<int>(args.k)},
      {a, static_cast<int>(args.lda)}, {b, static_cast<int>(args.ldb)},
      {c, static_cast<int>(args.ldc)}, {c, static_cast<int>(args.ldc)},
      {args.alpha, args.beta});
}

CutlassGemmBf16F32::Arguments
make_arguments(const FerruleCutlassGemmBf16F32Args &args) {
  auto *a = reinterpret_cast<const cutlass::bfloat16_t *>(
      static_cast<uintptr_t>(args.workspace_bf16));
  auto *b = reinterpret_cast<const cutlass::bfloat16_t *>(
      static_cast<uintptr_t>(args.b_bf16));
  auto *c = reinterpret_cast<float *>(static_cast<uintptr_t>(args.c_f32));
  return CutlassGemmBf16F32::Arguments(
      {static_cast<int>(args.m), static_cast<int>(args.n),
       static_cast<int>(args.k)},
      {a, static_cast<int>(args.lda)}, {b, static_cast<int>(args.ldb)},
      {c, static_cast<int>(args.ldc)}, {c, static_cast<int>(args.ldc)},
      {args.alpha, args.beta});
}

} // namespace

extern "C" FerruleCutlassProviderManifest
ferrule_cutlass_provider_manifest(void) {
  return FerruleCutlassProviderManifest{
      FERRULE_CUTLASS_ABI_VERSION,
      CUTLASS_VERSION,
      FERRULE_CUTLASS_TARGET_SM,
      2u,
      FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_F32_SIMT) |
          FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_BF16_MMA_SYNC),
  };
}

extern "C" uint32_t ferrule_cutlass_abi_version(void) {
  return FERRULE_CUTLASS_ABI_VERSION;
}

extern "C" uint32_t ferrule_cutlass_version(void) { return CUTLASS_VERSION; }

extern "C" uint32_t ferrule_cutlass_target_sm(void) {
  return FERRULE_CUTLASS_TARGET_SM;
}

extern "C" int32_t
ferrule_cutlass_gemm_f32_can_implement(const FerruleCutlassGemmF32Args *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
  }
  if (!valid(args)) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  auto arguments = make_arguments(*args);
  return CutlassGemmF32::can_implement(arguments) == cutlass::Status::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_UNSUPPORTED;
}

extern "C" int32_t
ferrule_cutlass_gemm_f32_launch(const FerruleCutlassGemmF32Args *args) {
  int32_t status = ferrule_cutlass_gemm_f32_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  auto arguments = make_arguments(*args);
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  CutlassGemmF32 gemm;
  return gemm(arguments, nullptr, stream) == cutlass::Status::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_LAUNCH_FAILED;
}

extern "C" int32_t ferrule_cutlass_gemm_bf16_f32_can_implement(
    const FerruleCutlassGemmBf16F32Args *args) {
  if (args == nullptr || args->abi_version != FERRULE_CUTLASS_ABI_VERSION) {
    return FERRULE_CUTLASS_INVALID_ABI;
  }
  if (!valid(args)) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  auto arguments = make_arguments(*args);
  return CutlassGemmBf16F32::can_implement(arguments) ==
                 cutlass::Status::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_UNSUPPORTED;
}

extern "C" int32_t ferrule_cutlass_gemm_bf16_f32_launch(
    const FerruleCutlassGemmBf16F32Args *args) {
  int32_t status = ferrule_cutlass_gemm_bf16_f32_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  uint64_t count = static_cast<uint64_t>(args->m) * args->k;
  auto *input =
      reinterpret_cast<const float *>(static_cast<uintptr_t>(args->a_f32));
  auto *packed = reinterpret_cast<cutlass::bfloat16_t *>(
      static_cast<uintptr_t>(args->workspace_bf16));
  constexpr uint32_t threads = 256;
  uint32_t blocks = static_cast<uint32_t>((count + threads - 1) / threads);
  ferrule_pack_f32_to_bf16<<<blocks, threads, 0, stream>>>(input, packed,
                                                           count);
  if (cudaGetLastError() != cudaSuccess) {
    return FERRULE_CUTLASS_LAUNCH_FAILED;
  }

  auto arguments = make_arguments(*args);
  CutlassGemmBf16F32 gemm;
  return gemm(arguments, nullptr, stream) == cutlass::Status::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_LAUNCH_FAILED;
}
