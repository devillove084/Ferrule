// Test-only, deliberately separate from both production translation units.
// E4M3FN: subnormals = (fraction/8)*2^-6; normals = (1+fraction/8)*2^(exponent-7).
// No Ferrule headers, layouts, decode/conversion helpers, or reduction replay.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

struct ScalarLayout {
  uint64_t m, n, k, row_origin, column_origin, lda, ldd, scale_bytes;
};
static_assert(sizeof(ScalarLayout) == 64);

__device__ float decode_e4m3fn(uint8_t byte) {
  const unsigned magnitude = byte % 128;
  if (magnitude == 127) return nanf("");
  const unsigned exponent = magnitude / 8;
  const float fraction = float(magnitude % 8) / 8.0f;
  const float value = exponent == 0 ? ldexpf(fraction, -6)
      : ldexpf(1.0f + fraction, int(exponent) - 7);
  return byte >= 128 ? -value : value;
}

// Choose the nearer of the two adjacent BF16 words; an exact midpoint chooses
// the even word. Unlike production's rounding-bias addition, test the residue.
__device__ float round_bfloat(float value) {
  const uint32_t bits = __float_as_uint(value);
  uint32_t word = bits >> 16;
  if (isnan(value)) return __uint_as_float((word | 64u) << 16);
  const uint32_t residue = bits % 65536u;
  if (residue > 32768u || (residue == 32768u && word % 2u != 0)) ++word;
  return __uint_as_float(word << 16);
}

__global__ void scalar_reference(ScalarLayout s, const float *input,
                                 const uint8_t *weight, const void *scales, float *output) {
  const uint64_t element = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= s.m * s.n) return;
  const uint64_t row = element / s.n;
  const uint64_t column = element % s.n;
  // Absolute checkpoint block coordinates, not production's residue layout.
  const uint64_t first_block_row = s.row_origin / 128;
  const uint64_t first_block_column = s.column_origin / 128;
  const uint64_t block_columns = (s.column_origin + s.k - 1) / 128 - first_block_column + 1;
  float accumulator = 0.0f;
  for (uint64_t inner = 0; inner < s.k; ++inner) {
    const uint64_t block_row = (s.row_origin + column) / 128 - first_block_row;
    const uint64_t block_column = (s.column_origin + inner) / 128 - first_block_column;
    const uint64_t scale_index = block_row * block_columns + block_column;
    const float scale = s.scale_bytes == 4 ? static_cast<const float *>(scales)[scale_index]
        : __uint_as_float(uint32_t(static_cast<const uint16_t *>(scales)[scale_index]) << 16);
    const float decoded = __fmul_rn(decode_e4m3fn(weight[column * s.k + inner]), scale);
    accumulator = __fmaf_rn(round_bfloat(input[row * s.lda + inner]),
                            round_bfloat(decoded), accumulator);
  }
  output[row * s.ldd + column] = accumulator;
}

struct Allocation {
  void *pointer = nullptr;
  ~Allocation() { if (pointer) cudaFree(pointer); }
};

// Host-array ABI keeps reference allocation, addressing and launch independent
// of the production provider. The caller binds its context and finishes prior
// work first. Blocking copies complete this launch before any allocation dies.
extern "C" int ferrule_test_w8a16_scalar(const ScalarLayout *layout,
    const float *input, const uint8_t *weight, const void *scales, float *output,
    int *runtime_version) {
  const ScalarLayout s = *layout;
  if (!s.m || !s.n || !s.k || s.lda < s.k || s.ldd < s.n ||
      (s.scale_bytes != 2 && s.scale_bytes != 4)) return int(cudaErrorInvalidValue);
  const uint64_t block_rows = (s.row_origin + s.n - 1) / 128 - s.row_origin / 128 + 1;
  const uint64_t block_columns = (s.column_origin + s.k - 1) / 128 - s.column_origin / 128 + 1;
  const size_t input_bytes = s.m * s.lda * sizeof(float);
  const size_t weight_bytes = s.n * s.k;
  const size_t scale_bytes = block_rows * block_columns * s.scale_bytes;
  const size_t output_bytes = (s.m * s.ldd + 7) * sizeof(float);
  Allocation a, w, scale, out;
#define CHECK(call) do { const cudaError_t status = (call); if (status != cudaSuccess) return int(status); } while (0)
  CHECK(cudaRuntimeGetVersion(runtime_version));
  CHECK(cudaMalloc(&a.pointer, input_bytes));
  CHECK(cudaMalloc(&w.pointer, weight_bytes));
  CHECK(cudaMalloc(&scale.pointer, scale_bytes));
  CHECK(cudaMalloc(&out.pointer, output_bytes));
  CHECK(cudaMemcpy(a.pointer, input, input_bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(w.pointer, weight, weight_bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(scale.pointer, scales, scale_bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(out.pointer, output, output_bytes, cudaMemcpyHostToDevice));
  scalar_reference<<<unsigned((s.m * s.n + 127) / 128), 128>>>(s,
      static_cast<const float *>(a.pointer), static_cast<const uint8_t *>(w.pointer),
      scale.pointer, static_cast<float *>(out.pointer));
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(output, out.pointer, output_bytes, cudaMemcpyDeviceToHost));
#undef CHECK
  return 0;
}
