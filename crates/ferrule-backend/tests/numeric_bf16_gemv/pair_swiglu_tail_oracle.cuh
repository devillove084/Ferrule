#pragma once
#ifndef FERRULE_CUDA_TEST_ORACLE
#error "pair tail oracle is test-only"
#endif

// Frozen pre-fusion bf16_boundary kind-4 expression (2026-10-11).
// Native/libdevice regression reference, NOT a Hugging Face oracle. Do not
// call core::bf16_swiglu/value/round/rne here, or update this with production.
namespace ferrule::test::pair_swiglu {
__device__ inline float value(uint16_t word) {
  return __uint_as_float(uint32_t(word) << 16);
}
// Independent residue/tie-even conversion, not production's bias addition.
__device__ inline uint16_t word(float v) {
  const uint32_t bits = __float_as_uint(v);
  uint32_t upper = bits >> 16;
  if ((bits & 0x7fffffffu) > 0x7f800000u)
    return uint16_t(upper | 0x40u);
  const uint32_t residue = bits & 0xffffu;
  if (residue > 0x8000u || (residue == 0x8000u && (upper & 1u))) ++upper;
  return uint16_t(upper);
}
__global__ void old_literal(const uint16_t *gate, const uint16_t *up,
                            uint16_t *output, uint32_t *raw, uint32_t count) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float v = value(gate[i]);
  const float silu = v / (1.0f + expf(-v));
  const float activated = value(word(silu));
  const float product = activated * value(up[i]);
  raw[2*i] = __float_as_uint(silu);
  raw[2*i+1] = __float_as_uint(product);
  output[i] = word(product);
}
struct Args {
  uint32_t rows, width;
  uint64_t packed, gate, up, reference, fused, raw, stream;
};
static_assert(sizeof(Args) == 64);
}

// One opt-in symbol, absent from ordinary provider builds. Test caller owns
// distinct exact-size buffers: packed[rows,2N], g/u/ref/fused[rows,N], raw[2MN].
// No cuBLAS, allocation, event or fence: exercise the tails without GEMM
// normalizing signed zeros, flushing subnormals or multiplying 0 by infinity.
extern "C" int32_t ferrule_test_pair_swiglu_tails(
    const ferrule::test::pair_swiglu::Args *a) {
  if (!a || (a->rows != 1 && a->rows != 2 && a->rows != 5) || a->width != 256 ||
      !a->packed || !a->gate || !a->up || !a->reference || !a->fused || !a->raw ||
      ((a->packed | a->gate | a->up | a->reference | a->fused) & 1u) ||
      (a->raw & 3u)) return 1;
  const uint32_t count = a->rows * a->width;
  const auto stream = reinterpret_cast<cudaStream_t>(a->stream);
  using namespace ferrule::cuda::cutlass::bf16_gemmex;
  if (int rc = pair_tail<false>(a->packed,a->gate,a->up,a->rows,a->width,stream))
    return rc;
  ferrule::test::pair_swiglu::old_literal<<<(count+255)/256,256,0,stream>>>(
      reinterpret_cast<const uint16_t *>(a->gate),
      reinterpret_cast<const uint16_t *>(a->up),
      reinterpret_cast<uint16_t *>(a->reference),
      reinterpret_cast<uint32_t *>(a->raw),count);
  if (cudaGetLastError() != cudaSuccess) return 4;
  return pair_tail<true>(a->packed,a->fused,a->fused,a->rows,a->width,stream);
}
