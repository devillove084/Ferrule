#pragma once

#include "cutlass/target.cuh"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ferrule::cuda::cutlass::operators::hybrid_mla_attention::
    explicit_selection {

inline constexpr std::uint32_t kExplicitSelectionContiguous = 1u;
inline constexpr std::uint32_t kExplicitSelectionPaged = 2u;
inline constexpr std::uint32_t kExplicitSelectionDualPaged = 3u;
inline constexpr std::uint32_t kExplicitSelectionHeads = 64u;
inline constexpr std::uint32_t kExplicitSelectionHeadDim = 512u;
inline constexpr std::uint32_t kExplicitSelectionMaximumWidth = 640u;

using ExplicitSelectionKind = std::uint32_t;

enum class ExplicitSelectionStatus : std::int32_t {
  kSuccess = 0,
  kInvalidArgument = 2,
  kLaunchFailed = 3,
  kUnsupported = 4,
};

// Private semantic boundary. This type deliberately uses native pointer types
// and is independent of the public C ABI layout.
struct ExplicitSelectionArgs {
  ExplicitSelectionKind kind;
  std::uint32_t rows;
  std::uint32_t tokens_per_sequence;
  std::uint32_t kv_len;
  std::uint32_t heads;
  std::uint32_t head_dim;
  std::uint32_t selected_width;
  std::uint32_t page_tokens;
  std::uint32_t first_elements_per_token;
  std::uint32_t second_elements_per_token;
  std::uint32_t layer_index;
  std::uint32_t layer_count;
  std::uint32_t flags;
  float softmax_scale;
  std::uint32_t reserved0;
  std::uint64_t first_plane_elements;
  std::uint64_t second_plane_elements;

  const float *query;
  const float *first_plane;
  const float *second_plane;
  const std::int32_t *block_slots;
  const std::int32_t *block_offsets;
  const std::int32_t *sequence_kv_lens;
  const std::int32_t *second_sequence_kv_lens;
  const std::int32_t *row_sequence_ids;
  const std::int32_t *row_kv_lens;
  const std::int32_t *row_second_kv_lens;
  const std::int32_t *selected_indices;
  const std::int32_t *selectors;
  const float *attention_sink;
  void *workspace;
  std::uint64_t workspace_bytes;
  float *output;
  std::int32_t *status;
  cudaStream_t stream;
};

namespace contract_detail {

template <class T>
inline bool aligned(const T *pointer, std::uintptr_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1u)) == 0u;
}

template <class T>
inline bool optional_aligned(const T *pointer, std::uintptr_t alignment) {
  return pointer == nullptr ||
         (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1u)) == 0u;
}

} // namespace contract_detail

// Shape-only validation is used by workspace queries. In particular, query may
// be null while every scalar needed to derive the complete layout is present.
inline ExplicitSelectionStatus
validate_explicit_selection_shape(const ExplicitSelectionArgs *args) {
  if (args == nullptr) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  if (args->heads != kExplicitSelectionHeads ||
      args->head_dim != kExplicitSelectionHeadDim ||
      args->selected_width > kExplicitSelectionMaximumWidth) {
    return ExplicitSelectionStatus::kUnsupported;
  }
  if (args->kind < kExplicitSelectionContiguous ||
      args->kind > kExplicitSelectionDualPaged || args->rows == 0u ||
      args->selected_width == 0u || (args->flags & ~3u) != 0u ||
      args->reserved0 != 0u || !std::isfinite(args->softmax_scale) ||
      args->softmax_scale <= 0.0f) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  if (args->kind != kExplicitSelectionContiguous &&
      (args->page_tokens == 0u || args->layer_count == 0u ||
       args->layer_index >= args->layer_count ||
       args->first_elements_per_token < kExplicitSelectionHeadDim ||
       (((args->flags & 1u) == 0u) && args->tokens_per_sequence == 0u))) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  if (args->kind == kExplicitSelectionDualPaged &&
      args->second_elements_per_token < kExplicitSelectionHeadDim) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  return ExplicitSelectionStatus::kSuccess;
}

inline ExplicitSelectionStatus
validate_explicit_selection_contract(const ExplicitSelectionArgs *args) {
  const ExplicitSelectionStatus shape = validate_explicit_selection_shape(args);
  if (shape != ExplicitSelectionStatus::kSuccess) {
    return shape;
  }

  if (args->first_plane_elements == 0u ||
      (args->kind == kExplicitSelectionDualPaged &&
       args->second_plane_elements == 0u)) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }

  const bool common_pointers =
      contract_detail::aligned(args->query, 16u) &&
      contract_detail::aligned(args->first_plane, 16u) &&
      contract_detail::aligned(args->selected_indices, 4u) &&
      contract_detail::aligned(args->attention_sink, 16u) &&
      contract_detail::aligned(args->output, 16u) &&
      contract_detail::aligned(args->status, 4u) &&
      (((args->flags & 1u) == 0u) ||
       contract_detail::aligned(args->row_sequence_ids, 4u)) &&
      (((args->flags & 2u) == 0u) ||
       contract_detail::aligned(args->row_kv_lens, 4u));
  if (!common_pointers) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  if (args->kind != kExplicitSelectionContiguous &&
      (!contract_detail::aligned(args->block_slots, 4u) ||
       !contract_detail::aligned(args->block_offsets, 4u) ||
       !contract_detail::aligned(args->sequence_kv_lens, 4u))) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  if (args->kind == kExplicitSelectionDualPaged &&
      (!contract_detail::aligned(args->second_plane, 16u) ||
       !contract_detail::aligned(args->second_sequence_kv_lens, 4u) ||
       (((args->flags & 2u) == 0u) ||
        !contract_detail::aligned(args->row_second_kv_lens, 4u)) ||
       !contract_detail::aligned(args->selectors, 4u))) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  if (!contract_detail::optional_aligned(args->second_plane, 16u) ||
      !contract_detail::optional_aligned(args->block_slots, 4u) ||
      !contract_detail::optional_aligned(args->block_offsets, 4u) ||
      !contract_detail::optional_aligned(args->sequence_kv_lens, 4u) ||
      !contract_detail::optional_aligned(args->second_sequence_kv_lens, 4u) ||
      !contract_detail::optional_aligned(args->row_sequence_ids, 4u) ||
      !contract_detail::optional_aligned(args->row_kv_lens, 4u) ||
      !contract_detail::optional_aligned(args->row_second_kv_lens, 4u) ||
      !contract_detail::optional_aligned(args->selectors, 4u)) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  return ExplicitSelectionStatus::kSuccess;
}

} // namespace
  // ferrule::cuda::cutlass::operators::hybrid_mla_attention::explicit_selection

#include <cstdint>

namespace ferrule::cuda::cutlass::operators::hybrid_mla_attention::
    explicit_selection::workspace {

inline constexpr std::uint32_t kAlignment = 16u;

struct Requirements {
  std::uint64_t bytes;
  std::uint32_t alignment;
};

struct Layout {
  std::uint64_t source_addresses_offset;
  std::uint64_t query_bf16_offset;
  std::uint64_t gathered_kv_bf16_offset;
  std::uint64_t raw_scores_offset;
  std::uint64_t probabilities_bf16_offset;
  std::uint64_t online_rescales_f32_offset;
  std::uint64_t denominators_f32_offset;
  std::uint64_t bytes;
};

struct Binding {
  std::uint64_t *source_addresses;
  std::uint16_t *query_bf16;
  std::uint16_t *gathered_kv_bf16;
  float *raw_scores;
  std::uint16_t *probabilities_bf16;
  float *online_rescales;
  float *denominators;
};

namespace detail {

inline constexpr bool checked_add(std::uint64_t left, std::uint64_t right,
                                  std::uint64_t &result) {
  if (right > UINT64_MAX - left) {
    return false;
  }
  result = left + right;
  return true;
}

inline constexpr bool checked_multiply(std::uint64_t left, std::uint64_t right,
                                       std::uint64_t &result) {
  if (left != 0u && right > UINT64_MAX / left) {
    return false;
  }
  result = left * right;
  return true;
}

inline constexpr bool align_up(std::uint64_t value, std::uint64_t alignment,
                               std::uint64_t &result) {
  const std::uint64_t mask = alignment - 1u;
  if (value > UINT64_MAX - mask) {
    return false;
  }
  result = (value + mask) & ~mask;
  return true;
}

inline bool append(std::uint64_t elements, std::uint64_t element_bytes,
                   std::uint64_t &cursor, std::uint64_t &offset) {
  std::uint64_t bytes = 0u;
  if (!align_up(cursor, kAlignment, offset) ||
      !checked_multiply(elements, element_bytes, bytes) ||
      !checked_add(offset, bytes, cursor)) {
    return false;
  }
  return true;
}

} // namespace detail

inline ExplicitSelectionStatus layout(const ExplicitSelectionArgs *args,
                                      Layout *result) {
  if (result == nullptr) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  const ExplicitSelectionStatus shape = validate_explicit_selection_shape(args);
  if (shape != ExplicitSelectionStatus::kSuccess) {
    return shape;
  }

  std::uint64_t selections = 0u;
  std::uint64_t query_elements = 0u;
  std::uint64_t gathered_elements = 0u;
  std::uint64_t score_elements = 0u;
  std::uint64_t online_rescale_elements = 0u;
  if (!detail::checked_multiply(args->rows, args->selected_width, selections) ||
      !detail::checked_multiply(args->rows, args->heads, query_elements) ||
      !detail::checked_multiply(query_elements, args->head_dim,
                                query_elements) ||
      !detail::checked_multiply(selections, args->head_dim,
                                gathered_elements) ||
      !detail::checked_multiply(selections, args->heads, score_elements) ||
      !detail::checked_multiply(
          query_elements / args->head_dim,
          (static_cast<std::uint64_t>(args->selected_width) + 63u) / 64u,
          online_rescale_elements)) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }

  Layout value{};
  std::uint64_t cursor = 0u;
  if (!detail::append(selections, sizeof(std::uint64_t), cursor,
                      value.source_addresses_offset) ||
      !detail::append(query_elements, sizeof(std::uint16_t), cursor,
                      value.query_bf16_offset) ||
      !detail::append(gathered_elements, sizeof(std::uint16_t), cursor,
                      value.gathered_kv_bf16_offset) ||
      !detail::append(score_elements, sizeof(float), cursor,
                      value.raw_scores_offset) ||
      !detail::append(score_elements, sizeof(std::uint16_t), cursor,
                      value.probabilities_bf16_offset) ||
      !detail::append(online_rescale_elements, sizeof(float), cursor,
                      value.online_rescales_f32_offset) ||
      !detail::append(query_elements / args->head_dim, sizeof(float), cursor,
                      value.denominators_f32_offset) ||
      !detail::align_up(cursor, kAlignment, value.bytes)) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  *result = value;
  return ExplicitSelectionStatus::kSuccess;
}

inline ExplicitSelectionStatus requirements(const ExplicitSelectionArgs *args,
                                            Requirements *result) {
  if (result == nullptr) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  Layout value{};
  const ExplicitSelectionStatus status = layout(args, &value);
  if (status != ExplicitSelectionStatus::kSuccess) {
    return status;
  }
  *result = Requirements{value.bytes, kAlignment};
  return ExplicitSelectionStatus::kSuccess;
}

inline ExplicitSelectionStatus bind(const ExplicitSelectionArgs *args,
                                    Binding *result) {
  if (result == nullptr) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  Layout value{};
  const ExplicitSelectionStatus layout_status = layout(args, &value);
  if (layout_status != ExplicitSelectionStatus::kSuccess) {
    return layout_status;
  }
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(args->workspace);
  if (base == 0u || (base & (kAlignment - 1u)) != 0u ||
      args->workspace_bytes < value.bytes ||
      value.bytes > static_cast<std::uint64_t>(UINTPTR_MAX - base)) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }

  auto *bytes = reinterpret_cast<std::uint8_t *>(args->workspace);
  *result = Binding{
      reinterpret_cast<std::uint64_t *>(bytes + value.source_addresses_offset),
      reinterpret_cast<std::uint16_t *>(bytes + value.query_bf16_offset),
      reinterpret_cast<std::uint16_t *>(bytes + value.gathered_kv_bf16_offset),
      reinterpret_cast<float *>(bytes + value.raw_scores_offset),
      reinterpret_cast<std::uint16_t *>(bytes +
                                        value.probabilities_bf16_offset),
      reinterpret_cast<float *>(bytes + value.online_rescales_f32_offset),
      reinterpret_cast<float *>(bytes + value.denominators_f32_offset),
  };
  return ExplicitSelectionStatus::kSuccess;
}

} // namespace
  // ferrule::cuda::cutlass::operators::hybrid_mla_attention::explicit_selection::workspace

#include <cuda_runtime.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>

#include <cstdint>

namespace ferrule::cuda::cutlass::operators::hybrid_mla_attention::bf16 {

using Mma = cute::SM80_16x8x16_F32BF16BF16F32_TN;

__device__ __forceinline__ std::uint16_t from_f32_rne(float value) {
  std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  }
  const std::uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>((bits + bias) >> 16);
}

__device__ __forceinline__ float to_f32(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16);
}

__device__ __forceinline__ void
load_weight_fragment(const std::uint16_t *shared, std::uint32_t lane,
                     std::uint32_t (&fragment)[4]) {
  const std::uint32_t quad = lane >> 3;
  const std::uint32_t row = (lane & 7u) + ((quad & 1u) != 0u ? 8u : 0u);
  const std::uint32_t column_bytes = quad >= 2u ? 16u : 0u;
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(shared);
  auto const &source = *reinterpret_cast<const cute::uint128_t *>(
      bytes + row * 32u + column_bytes);
  cute::SM75_U32x4_LDSM_N::copy(source, fragment[0], fragment[1], fragment[2],
                                fragment[3]);
}

__device__ __forceinline__ void
load_activation_fragment(const std::uint16_t *shared, std::uint32_t lane,
                         std::uint32_t (&fragment)[2]) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(shared);
  auto const &source =
      *reinterpret_cast<const cute::uint128_t *>(bytes + (lane & 15u) * 16u);
  cute::SM75_U16x4_LDSM_T::copy(source, fragment[0], fragment[1]);
}

__device__ __forceinline__ void mma(float (&accumulator)[4],
                                    const std::uint32_t (&weight)[4],
                                    const std::uint32_t (&activation)[2]) {
  Mma::fma(accumulator[0], accumulator[1], accumulator[2], accumulator[3],
           weight[0], weight[1], weight[2], weight[3], activation[0],
           activation[1], accumulator[0], accumulator[1], accumulator[2],
           accumulator[3]);
}

} // namespace ferrule::cuda::cutlass::operators::hybrid_mla_attention::bf16

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ferrule::cuda::cutlass::operators::hybrid_mla_attention::
    explicit_selection::schedule {

namespace semantic = operators::hybrid_mla_attention::explicit_selection;

inline constexpr std::uint32_t kContiguous =
    semantic::kExplicitSelectionContiguous;
inline constexpr std::uint32_t kPaged = semantic::kExplicitSelectionPaged;
inline constexpr std::uint32_t kDualPaged =
    semantic::kExplicitSelectionDualPaged;
inline constexpr std::uint32_t kHeads = semantic::kExplicitSelectionHeads;
inline constexpr std::uint32_t kHeadDim = semantic::kExplicitSelectionHeadDim;
inline constexpr std::uint32_t kMaximumSelectedWidth =
    semantic::kExplicitSelectionMaximumWidth;
inline constexpr std::uint32_t kOnlineSoftmaxTile = 64u;
inline constexpr std::uint32_t kMmaRows = 8u;
inline constexpr std::uint32_t kMmaColumns = 16u;
inline constexpr std::uint32_t kKTile = 16u;
inline constexpr std::uint32_t kWarpSize = 32u;
inline constexpr std::uint32_t kWarps = 4u;
inline constexpr std::uint32_t kMmaThreads = kWarpSize * kWarps;
inline constexpr std::uint32_t kGatherThreads = 256u;
inline constexpr std::uint32_t kHeadsPerWarp = kMmaRows;
inline constexpr std::uint32_t kHeadsPerBlock = kWarps * kHeadsPerWarp;
inline constexpr std::uint32_t kHeadGroups = kHeads / kHeadsPerBlock;
inline constexpr std::uint32_t kChannelTiles = kHeadDim / kMmaColumns;
inline constexpr std::uint32_t kLaunchCount = 4u;

static_assert(kHeads % kHeadsPerBlock == 0u);
static_assert(kHeadDim % kMmaColumns == 0u);
static_assert(kHeadDim % kKTile == 0u);
static_assert(sizeof(std::uintptr_t) == sizeof(std::uint64_t));

// Private schedule ABI. Input tensors, latent planes, and output remain F32;
// the intermediate addresses are supplied from
// explicit_selection_workspace::Binding by the caller.
struct Args {
  std::uint32_t kind;
  std::uint32_t rows;
  std::uint32_t tokens_per_sequence;
  std::uint32_t kv_len;
  std::uint32_t heads;
  std::uint32_t head_dim;
  std::uint32_t selected_width;
  std::uint32_t page_tokens;
  std::uint32_t first_elements_per_token;
  std::uint32_t second_elements_per_token;
  std::uint32_t layer_index;
  std::uint32_t layer_count;
  std::uint32_t flags;
  float softmax_scale;
  std::uint32_t reserved0;
  std::uint64_t first_plane_elements;
  std::uint64_t second_plane_elements;

  std::uint64_t query_f32;
  std::uint64_t first_plane_f32;
  std::uint64_t second_plane_f32;
  std::uint64_t block_slots_i32;
  std::uint64_t block_offsets_i32;
  std::uint64_t sequence_kv_lens_i32;
  std::uint64_t second_sequence_kv_lens_i32;
  std::uint64_t row_sequence_ids_i32;
  std::uint64_t row_kv_lens_i32;
  std::uint64_t row_second_kv_lens_i32;
  std::uint64_t selected_indices_i32;
  std::uint64_t selectors_i32;
  std::uint64_t attention_sink_f32;
  std::uint64_t source_addresses_u64;
  std::uint64_t query_bf16;
  std::uint64_t gathered_kv_bf16;
  std::uint64_t raw_scores_f32;
  std::uint64_t probabilities_bf16;
  std::uint64_t online_rescales_f32;
  std::uint64_t denominators_f32;
  std::uint64_t output_f32;
  std::uint64_t status_i32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 264u, "BF16 selected-attention POD changed");
static_assert(alignof(Args) == 8u);
static_assert(offsetof(Args, kind) == 0u);
static_assert(offsetof(Args, rows) == 4u);
static_assert(offsetof(Args, tokens_per_sequence) == 8u);
static_assert(offsetof(Args, kv_len) == 12u);
static_assert(offsetof(Args, heads) == 16u);
static_assert(offsetof(Args, head_dim) == 20u);
static_assert(offsetof(Args, selected_width) == 24u);
static_assert(offsetof(Args, page_tokens) == 28u);
static_assert(offsetof(Args, first_elements_per_token) == 32u);
static_assert(offsetof(Args, second_elements_per_token) == 36u);
static_assert(offsetof(Args, layer_index) == 40u);
static_assert(offsetof(Args, layer_count) == 44u);
static_assert(offsetof(Args, flags) == 48u);
static_assert(offsetof(Args, softmax_scale) == 52u);
static_assert(offsetof(Args, reserved0) == 56u);
static_assert(offsetof(Args, first_plane_elements) == 64u);
static_assert(offsetof(Args, second_plane_elements) == 72u);
static_assert(offsetof(Args, query_f32) == 80u);
static_assert(offsetof(Args, first_plane_f32) == 88u);
static_assert(offsetof(Args, second_plane_f32) == 96u);
static_assert(offsetof(Args, block_slots_i32) == 104u);
static_assert(offsetof(Args, block_offsets_i32) == 112u);
static_assert(offsetof(Args, sequence_kv_lens_i32) == 120u);
static_assert(offsetof(Args, second_sequence_kv_lens_i32) == 128u);
static_assert(offsetof(Args, row_sequence_ids_i32) == 136u);
static_assert(offsetof(Args, row_kv_lens_i32) == 144u);
static_assert(offsetof(Args, row_second_kv_lens_i32) == 152u);
static_assert(offsetof(Args, selected_indices_i32) == 160u);
static_assert(offsetof(Args, selectors_i32) == 168u);
static_assert(offsetof(Args, attention_sink_f32) == 176u);
static_assert(offsetof(Args, source_addresses_u64) == 184u);
static_assert(offsetof(Args, query_bf16) == 192u);
static_assert(offsetof(Args, gathered_kv_bf16) == 200u);
static_assert(offsetof(Args, raw_scores_f32) == 208u);
static_assert(offsetof(Args, probabilities_bf16) == 216u);
static_assert(offsetof(Args, online_rescales_f32) == 224u);
static_assert(offsetof(Args, denominators_f32) == 232u);
static_assert(offsetof(Args, output_f32) == 240u);
static_assert(offsetof(Args, status_i32) == 248u);
static_assert(offsetof(Args, stream) == 256u);

using Status = semantic::ExplicitSelectionStatus;

namespace detail {

struct Binding {
  const float *query;
  const float *first_plane;
  const float *second_plane;
  const std::int32_t *block_slots;
  const std::int32_t *block_offsets;
  const std::int32_t *sequence_kv_lens;
  const std::int32_t *second_sequence_kv_lens;
  const std::int32_t *row_sequence_ids;
  const std::int32_t *row_kv_lens;
  const std::int32_t *row_second_kv_lens;
  const std::int32_t *selected_indices;
  const std::int32_t *selectors;
  const float *attention_sink;
  std::uint64_t *source_addresses;
  std::uint16_t *query_bf16;
  std::uint16_t *gathered_kv_bf16;
  float *raw_scores;
  std::uint16_t *probabilities_bf16;
  float *online_rescales;
  float *denominators;
  float *output;
  std::int32_t *status;
};

struct ValueLocation {
  const float *values;
  std::uint64_t base;
  bool valid;
};

struct alignas(16) MmaSharedStorage {
  alignas(16) std::uint16_t common[kMmaColumns * kKTile];
  alignas(16) std::uint16_t activation[kWarps][kKTile * kMmaRows];
};

static_assert(sizeof(MmaSharedStorage) == 1536u);
static_assert(std::is_trivially_copyable_v<Binding>);

inline constexpr bool aligned(std::uint64_t address, std::uint64_t alignment) {
  return address != 0u && (address & (alignment - 1u)) == 0u;
}

inline constexpr bool optional_aligned(std::uint64_t address,
                                       std::uint64_t alignment) {
  return address == 0u || (address & (alignment - 1u)) == 0u;
}

template <class T>
__host__ __device__ __forceinline__ T *device_pointer(std::uint64_t address) {
  return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address));
}

__device__ __forceinline__ bool checked_multiply(std::uint64_t left,
                                                 std::uint64_t right,
                                                 std::uint64_t &result) {
  if (left != 0u && right > UINT64_MAX / left) {
    return false;
  }
  result = left * right;
  return true;
}

__device__ __forceinline__ bool
checked_add(std::uint64_t left, std::uint64_t right, std::uint64_t &result) {
  if (right > UINT64_MAX - left) {
    return false;
  }
  result = left + right;
  return true;
}

__device__ __forceinline__ std::uint32_t
sequence_for_row(const Args &args, const Binding &binding, std::uint32_t row) {
  if ((args.flags & 1u) != 0u) {
    const std::int32_t sequence = binding.row_sequence_ids[row];
    return sequence >= 0 ? static_cast<std::uint32_t>(sequence) : UINT32_MAX;
  }
  return args.tokens_per_sequence == 0u ? 0u : row / args.tokens_per_sequence;
}

__device__ __forceinline__ ValueLocation
paged_location(const Args &args, const Binding &binding,
               const std::int32_t *sequence_kv_lens, const float *plane,
               std::uint64_t plane_elements, std::uint32_t elements_per_token,
               std::uint32_t sequence, std::int32_t logical) {
  if (logical < 0 || sequence == UINT32_MAX) {
    return {nullptr, 0u, false};
  }

  const std::int32_t sequence_length = sequence_kv_lens[sequence];
  if (sequence_length < 0 || logical >= sequence_length) {
    return {nullptr, 0u, false};
  }

  const std::int32_t begin = binding.block_offsets[sequence];
  const std::int32_t end = binding.block_offsets[sequence + 1u];
  if (begin < 0 || end < begin) {
    return {nullptr, 0u, false};
  }
  const std::uint64_t page =
      static_cast<std::uint32_t>(logical) / args.page_tokens;
  std::uint64_t entry = 0u;
  if (!checked_add(static_cast<std::uint64_t>(begin), page, entry) ||
      entry >= static_cast<std::uint64_t>(end)) {
    return {nullptr, 0u, false};
  }
  const std::int32_t slot = binding.block_slots[entry];
  if (slot < 0) {
    return {nullptr, 0u, false};
  }

  std::uint64_t layer_stride = 0u;
  std::uint64_t slot_stride = 0u;
  std::uint64_t base = 0u;
  std::uint64_t part = 0u;
  if (!checked_multiply(args.page_tokens, elements_per_token, layer_stride) ||
      !checked_multiply(args.layer_count, layer_stride, slot_stride) ||
      !checked_multiply(static_cast<std::uint32_t>(slot), slot_stride, base) ||
      !checked_multiply(args.layer_index, layer_stride, part) ||
      !checked_add(base, part, base) ||
      !checked_multiply(static_cast<std::uint32_t>(logical) % args.page_tokens,
                        elements_per_token, part) ||
      !checked_add(base, part, base) || base > plane_elements ||
      args.head_dim > plane_elements - base) {
    return {nullptr, 0u, false};
  }
  return {plane, base, true};
}

__device__ __forceinline__ ValueLocation
selected_location(const Args &args, const Binding &binding, std::uint32_t row,
                  std::uint32_t selected) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(row) * args.selected_width + selected;
  std::int32_t logical = binding.selected_indices[index];
  const std::int32_t selector =
      args.kind == kDualPaged ? binding.selectors[index] : 0;
  if ((args.flags & 2u) != 0u) {
    const std::int32_t visible = selector == 1 ? binding.row_second_kv_lens[row]
                                               : binding.row_kv_lens[row];
    if (logical < 0 || logical >= visible) {
      logical = -1;
    }
  }
  if (logical < 0) {
    return {nullptr, 0u, false};
  }

  if (args.kind == kContiguous) {
    if (static_cast<std::uint32_t>(logical) >= args.kv_len) {
      return {nullptr, 0u, false};
    }
    std::uint64_t base = 0u;
    if (!checked_multiply(static_cast<std::uint32_t>(logical), args.head_dim,
                          base) ||
        base > args.first_plane_elements ||
        args.head_dim > args.first_plane_elements - base) {
      return {nullptr, 0u, false};
    }
    return {binding.first_plane, base, true};
  }

  const std::uint32_t sequence = sequence_for_row(args, binding, row);
  if (selector == 0) {
    return paged_location(args, binding, binding.sequence_kv_lens,
                          binding.first_plane, args.first_plane_elements,
                          args.first_elements_per_token, sequence, logical);
  }
  if (args.kind == kDualPaged && selector == 1) {
    return paged_location(args, binding, binding.second_sequence_kv_lens,
                          binding.second_plane, args.second_plane_elements,
                          args.second_elements_per_token, sequence, logical);
  }
  return {nullptr, 0u, false};
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (std::uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (std::uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__global__
__launch_bounds__(kGatherThreads) void gather_pack_kernel(Args args,
                                                          Binding binding) {
  __shared__ std::uint64_t shared_source_address;

  const std::uint64_t selection_index = blockIdx.x;
  const std::uint32_t row =
      static_cast<std::uint32_t>(selection_index / args.selected_width);
  const std::uint32_t selected =
      static_cast<std::uint32_t>(selection_index % args.selected_width);

  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *binding.status = 0;
  }
  if (threadIdx.x == 0u) {
    const ValueLocation location =
        selected_location(args, binding, row, selected);
    const float *source =
        location.valid ? location.values + location.base : nullptr;
    shared_source_address =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(source));
    binding.source_addresses[selection_index] = shared_source_address;
  }
  __syncthreads();

  const auto *source = reinterpret_cast<const float *>(
      static_cast<std::uintptr_t>(shared_source_address));
  const std::uint64_t gathered_base = selection_index * args.head_dim;
  for (std::uint32_t dimension = threadIdx.x; dimension < kHeadDim;
       dimension += blockDim.x) {
    binding.gathered_kv_bf16[gathered_base + dimension] =
        source == nullptr ? 0u : bf16::from_f32_rne(source[dimension]);
  }

  const std::uint64_t global_thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t global_stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const std::uint64_t query_elements =
      static_cast<std::uint64_t>(args.rows) * kHeads * kHeadDim;
  for (std::uint64_t index = global_thread; index < query_elements;
       index += global_stride) {
    binding.query_bf16[index] = bf16::from_f32_rne(binding.query[index]);
  }
}

__device__ __forceinline__ void
stage_qk_kv(const Args &args, const Binding &binding, MmaSharedStorage &shared,
            std::uint32_t row, std::uint32_t selected_base,
            std::uint32_t dimension_base) {
  for (std::uint32_t linear = threadIdx.x; linear < kMmaColumns * kKTile;
       linear += blockDim.x) {
    const std::uint32_t selected_local = linear / kKTile;
    const std::uint32_t dimension_local = linear % kKTile;
    const std::uint32_t selected = selected_base + selected_local;
    shared.common[linear] =
        selected < args.selected_width
            ? binding.gathered_kv_bf16[(static_cast<std::uint64_t>(row) *
                                            args.selected_width +
                                        selected) *
                                           kHeadDim +
                                       dimension_base + dimension_local]
            : 0u;
  }
}

__device__ __forceinline__ void
stage_query(const Binding &binding, MmaSharedStorage &shared,
            std::uint32_t warp, std::uint32_t lane, std::uint32_t row,
            std::uint32_t first_head, std::uint32_t dimension_base) {
  for (std::uint32_t linear = lane; linear < kKTile * kMmaRows;
       linear += kWarpSize) {
    const std::uint32_t dimension_local = linear / kMmaRows;
    const std::uint32_t mma_row = linear % kMmaRows;
    const std::uint32_t head = first_head + mma_row;
    shared.activation[warp][linear] =
        binding.query_bf16[(static_cast<std::uint64_t>(row) * kHeads + head) *
                               kHeadDim +
                           dimension_base + dimension_local];
  }
}

__device__ __forceinline__ void
store_qk_scores(const Args &args, const Binding &binding, std::uint32_t row,
                std::uint32_t first_head, std::uint32_t selected_base,
                std::uint32_t lane, const float (&accumulator)[4]) {
  const std::uint32_t selected_group = lane >> 2;
  const std::uint32_t mma_row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t selected =
        selected_base + selected_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t head = first_head + mma_row_pair * 2u + (element & 1u);
    if (selected < args.selected_width) {
      binding.raw_scores[(static_cast<std::uint64_t>(row) * kHeads + head) *
                             args.selected_width +
                         selected] = accumulator[element];
    }
  }
}

__global__ __launch_bounds__(kMmaThreads, 1) void qk_kernel(Args args,
                                                            Binding binding) {
  __shared__ MmaSharedStorage shared;
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t selection_tiles =
      (args.selected_width + kMmaColumns - 1u) / kMmaColumns;
  const std::uint32_t selection_tile = blockIdx.x % selection_tiles;
  const std::uint32_t row_head_group = blockIdx.x / selection_tiles;
  const std::uint32_t head_group = row_head_group % kHeadGroups;
  const std::uint32_t row = row_head_group / kHeadGroups;
  const std::uint32_t first_head =
      head_group * kHeadsPerBlock + warp * kHeadsPerWarp;
  const std::uint32_t selected_base = selection_tile * kMmaColumns;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t dimension_base = 0u; dimension_base < kHeadDim;
       dimension_base += kKTile) {
    stage_qk_kv(args, binding, shared, row, selected_base, dimension_base);
    stage_query(binding, shared, warp, lane, row, first_head, dimension_base);
    __syncthreads();
    std::uint32_t kv_fragment[4];
    std::uint32_t query_fragment[2];
    bf16::load_weight_fragment(shared.common, lane, kv_fragment);
    bf16::load_activation_fragment(shared.activation[warp], lane,
                                   query_fragment);
    bf16::mma(accumulator, kv_fragment, query_fragment);
    __syncthreads();
  }
  store_qk_scores(args, binding, row, first_head, selected_base, lane,
                  accumulator);
}

__global__ __launch_bounds__(kMmaThreads) void softmax_kernel(Args args,
                                                              Binding binding) {
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint64_t pair =
      static_cast<std::uint64_t>(blockIdx.x) * kWarps + warp;
  const std::uint64_t pair_count =
      static_cast<std::uint64_t>(args.rows) * kHeads;
  if (pair >= pair_count) {
    return;
  }

  const std::uint32_t row = static_cast<std::uint32_t>(pair / kHeads);
  const std::uint32_t head = static_cast<std::uint32_t>(pair % kHeads);
  const std::uint64_t selected_base =
      static_cast<std::uint64_t>(row) * args.selected_width;
  const std::uint64_t score_base = pair * args.selected_width;

  const std::uint32_t online_tiles =
      (args.selected_width + kOnlineSoftmaxTile - 1u) / kOnlineSoftmaxTile;
  float running_maximum = -INFINITY;
  float denominator = 0.0f;

  for (std::uint32_t online_tile = 0u; online_tile < online_tiles;
       ++online_tile) {
    const std::uint32_t tile_base = online_tile * kOnlineSoftmaxTile;
    float tile_maximum = -INFINITY;
#pragma unroll
    for (std::uint32_t iteration = 0u; iteration < 2u; ++iteration) {
      const std::uint32_t selected = tile_base + lane + iteration * kWarpSize;
      if (selected < args.selected_width &&
          binding.source_addresses[selected_base + selected] != 0u) {
        tile_maximum = fmaxf(
            tile_maximum, __fmul_rn(binding.raw_scores[score_base + selected],
                                    args.softmax_scale));
      }
    }
    tile_maximum = warp_max(tile_maximum);
    const float next_maximum = fmaxf(running_maximum, tile_maximum);
    const float rescale = running_maximum == -INFINITY
                              ? 0.0f
                              : expf(__fsub_rn(running_maximum, next_maximum));
    float tile_sum = 0.0f;

#pragma unroll
    for (std::uint32_t iteration = 0u; iteration < 2u; ++iteration) {
      const std::uint32_t selected = tile_base + lane + iteration * kWarpSize;
      if (selected >= args.selected_width) {
        continue;
      }
      std::uint16_t weight = 0u;
      if (binding.source_addresses[selected_base + selected] != 0u) {
        const float value =
            expf(__fmaf_rn(binding.raw_scores[score_base + selected],
                           args.softmax_scale, -next_maximum));
        tile_sum = __fadd_rn(tile_sum, value);
        weight = bf16::from_f32_rne(value);
      }
      binding.probabilities_bf16[score_base + selected] = weight;
    }
    tile_sum = warp_sum(tile_sum);
    denominator = __fmaf_rn(denominator, rescale, tile_sum);
    if (lane == 0u) {
      binding.online_rescales[pair * online_tiles + online_tile] = rescale;
    }
    running_maximum = next_maximum;
  }

  if (lane == 0u) {
    binding.denominators[pair] =
        running_maximum == -INFINITY
            ? 1.0f
            : __fadd_rn(denominator,
                        expf(__fsub_rn(binding.attention_sink[head],
                                       running_maximum)));
  }
}

__device__ __forceinline__ void
stage_pv_values(const Args &args, const Binding &binding,
                MmaSharedStorage &shared, std::uint32_t row,
                std::uint32_t channel_base, std::uint32_t selected_base) {
  for (std::uint32_t linear = threadIdx.x; linear < kMmaColumns * kKTile;
       linear += blockDim.x) {
    const std::uint32_t channel_local = linear / kKTile;
    const std::uint32_t selected_local = linear % kKTile;
    const std::uint32_t selected = selected_base + selected_local;
    shared.common[linear] =
        selected < args.selected_width
            ? binding.gathered_kv_bf16[(static_cast<std::uint64_t>(row) *
                                            args.selected_width +
                                        selected) *
                                           kHeadDim +
                                       channel_base + channel_local]
            : 0u;
  }
}

__device__ __forceinline__ void
stage_probabilities(const Args &args, const Binding &binding,
                    MmaSharedStorage &shared, std::uint32_t warp,
                    std::uint32_t lane, std::uint32_t row,
                    std::uint32_t first_head, std::uint32_t selected_base) {
  for (std::uint32_t linear = lane; linear < kKTile * kMmaRows;
       linear += kWarpSize) {
    const std::uint32_t selected_local = linear / kMmaRows;
    const std::uint32_t mma_row = linear % kMmaRows;
    const std::uint32_t selected = selected_base + selected_local;
    const std::uint32_t head = first_head + mma_row;
    const std::uint64_t probability_base =
        (static_cast<std::uint64_t>(row) * kHeads + head) * args.selected_width;
    shared.activation[warp][linear] =
        selected < args.selected_width
            ? binding.probabilities_bf16[probability_base + selected]
            : 0u;
  }
}

__device__ __forceinline__ void
rescale_pv_accumulator(const Args &args, const Binding &binding,
                       std::uint32_t row, std::uint32_t first_head,
                       std::uint32_t online_tile, std::uint32_t lane,
                       float (&accumulator)[4]) {
  const std::uint32_t online_tiles =
      (args.selected_width + kOnlineSoftmaxTile - 1u) / kOnlineSoftmaxTile;
  const std::uint32_t mma_row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t head = first_head + mma_row_pair * 2u + (element & 1u);
    const std::uint64_t pair = static_cast<std::uint64_t>(row) * kHeads + head;
    accumulator[element] =
        __fmul_rn(accumulator[element],
                  binding.online_rescales[pair * online_tiles + online_tile]);
  }
}

__device__ __forceinline__ void
store_pv_output(const Binding &binding, std::uint32_t row,
                std::uint32_t first_head, std::uint32_t channel_base,
                std::uint32_t lane, const float (&accumulator)[4]) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t mma_row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t head = first_head + mma_row_pair * 2u + (element & 1u);
    const std::uint64_t pair = static_cast<std::uint64_t>(row) * kHeads + head;
    const float normalized =
        __fdiv_rn(accumulator[element], binding.denominators[pair]);
    const std::uint16_t output_bf16 = bf16::from_f32_rne(normalized);
    binding.output[pair * kHeadDim + channel] = bf16::to_f32(output_bf16);
  }
}

__global__ __launch_bounds__(kMmaThreads, 1) void pv_kernel(Args args,
                                                            Binding binding) {
  __shared__ MmaSharedStorage shared;
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t channel_tile = blockIdx.x % kChannelTiles;
  const std::uint32_t row_head_group = blockIdx.x / kChannelTiles;
  const std::uint32_t head_group = row_head_group % kHeadGroups;
  const std::uint32_t row = row_head_group / kHeadGroups;
  const std::uint32_t first_head =
      head_group * kHeadsPerBlock + warp * kHeadsPerWarp;
  const std::uint32_t channel_base = channel_tile * kMmaColumns;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t selected_base = 0u; selected_base < args.selected_width;
       selected_base += kKTile) {
    if (selected_base % kOnlineSoftmaxTile == 0u) {
      rescale_pv_accumulator(args, binding, row, first_head,
                             selected_base / kOnlineSoftmaxTile, lane,
                             accumulator);
    }
    stage_pv_values(args, binding, shared, row, channel_base, selected_base);
    stage_probabilities(args, binding, shared, warp, lane, row, first_head,
                        selected_base);
    __syncthreads();
    std::uint32_t value_fragment[4];
    std::uint32_t probability_fragment[2];
    bf16::load_weight_fragment(shared.common, lane, value_fragment);
    bf16::load_activation_fragment(shared.activation[warp], lane,
                                   probability_fragment);
    bf16::mma(accumulator, value_fragment, probability_fragment);
    __syncthreads();
  }
  store_pv_output(binding, row, first_head, channel_base, lane, accumulator);
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr) {
    return Status::kInvalidArgument;
  }
#if !FERRULE_CUDA_HAS_BF16_MMA_SYNC
  return Status::kUnsupported;
#else
  if (args->heads != kHeads || args->head_dim != kHeadDim ||
      args->selected_width > kMaximumSelectedWidth) {
    return Status::kUnsupported;
  }
  if (args->kind < kContiguous || args->kind > kDualPaged || args->rows == 0u ||
      args->selected_width == 0u || (args->flags & ~3u) != 0u ||
      args->reserved0 != 0u || !std::isfinite(args->softmax_scale) ||
      args->softmax_scale <= 0.0f || args->first_plane_elements == 0u) {
    return Status::kInvalidArgument;
  }
  if (args->kind != kContiguous &&
      (args->page_tokens == 0u || args->layer_count == 0u ||
       args->layer_index >= args->layer_count ||
       args->first_elements_per_token < kHeadDim ||
       (((args->flags & 1u) == 0u) && args->tokens_per_sequence == 0u))) {
    return Status::kInvalidArgument;
  }
  if (args->kind == kDualPaged && (args->second_elements_per_token < kHeadDim ||
                                   args->second_plane_elements == 0u)) {
    return Status::kInvalidArgument;
  }

  const bool common_pointers =
      detail::aligned(args->query_f32, 16u) &&
      detail::aligned(args->first_plane_f32, 16u) &&
      detail::aligned(args->selected_indices_i32, 4u) &&
      detail::aligned(args->attention_sink_f32, 16u) &&
      detail::aligned(args->source_addresses_u64, 16u) &&
      detail::aligned(args->query_bf16, 16u) &&
      detail::aligned(args->gathered_kv_bf16, 16u) &&
      detail::aligned(args->raw_scores_f32, 16u) &&
      detail::aligned(args->probabilities_bf16, 16u) &&
      detail::aligned(args->online_rescales_f32, 16u) &&
      detail::aligned(args->denominators_f32, 16u) &&
      detail::aligned(args->output_f32, 16u) &&
      detail::aligned(args->status_i32, 4u) &&
      (((args->flags & 1u) == 0u) ||
       detail::aligned(args->row_sequence_ids_i32, 4u)) &&
      (((args->flags & 2u) == 0u) ||
       detail::aligned(args->row_kv_lens_i32, 4u));
  if (!common_pointers) {
    return Status::kInvalidArgument;
  }
  if (args->kind != kContiguous &&
      (!detail::aligned(args->block_slots_i32, 4u) ||
       !detail::aligned(args->block_offsets_i32, 4u) ||
       !detail::aligned(args->sequence_kv_lens_i32, 4u))) {
    return Status::kInvalidArgument;
  }
  if (args->kind == kDualPaged &&
      (!detail::aligned(args->second_plane_f32, 16u) ||
       !detail::aligned(args->second_sequence_kv_lens_i32, 4u) ||
       (((args->flags & 2u) == 0u) ||
        !detail::aligned(args->row_second_kv_lens_i32, 4u)) ||
       !detail::aligned(args->selectors_i32, 4u))) {
    return Status::kInvalidArgument;
  }
  if (!detail::optional_aligned(args->second_plane_f32, 16u) ||
      !detail::optional_aligned(args->block_slots_i32, 4u) ||
      !detail::optional_aligned(args->block_offsets_i32, 4u) ||
      !detail::optional_aligned(args->sequence_kv_lens_i32, 4u) ||
      !detail::optional_aligned(args->second_sequence_kv_lens_i32, 4u) ||
      !detail::optional_aligned(args->row_sequence_ids_i32, 4u) ||
      !detail::optional_aligned(args->row_kv_lens_i32, 4u) ||
      !detail::optional_aligned(args->row_second_kv_lens_i32, 4u) ||
      !detail::optional_aligned(args->selectors_i32, 4u)) {
    return Status::kInvalidArgument;
  }
  return Status::kSuccess;
#endif
}

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }

  const std::uint64_t selections =
      static_cast<std::uint64_t>(args->rows) * args->selected_width;
  const std::uint64_t selection_tiles =
      (args->selected_width + kMmaColumns - 1u) / kMmaColumns;
  const std::uint64_t qk_blocks =
      static_cast<std::uint64_t>(args->rows) * kHeadGroups * selection_tiles;
  const std::uint64_t pairs = static_cast<std::uint64_t>(args->rows) * kHeads;
  const std::uint64_t softmax_blocks = (pairs + kWarps - 1u) / kWarps;
  const std::uint64_t pv_blocks =
      static_cast<std::uint64_t>(args->rows) * kHeadGroups * kChannelTiles;
  if (selections == 0u || selections > UINT32_MAX || qk_blocks == 0u ||
      qk_blocks > UINT32_MAX || softmax_blocks == 0u ||
      softmax_blocks > UINT32_MAX || pv_blocks == 0u ||
      pv_blocks > UINT32_MAX) {
    return Status::kLaunchFailed;
  }

  const detail::Binding binding{
      detail::device_pointer<const float>(args->query_f32),
      detail::device_pointer<const float>(args->first_plane_f32),
      detail::device_pointer<const float>(args->second_plane_f32),
      detail::device_pointer<const std::int32_t>(args->block_slots_i32),
      detail::device_pointer<const std::int32_t>(args->block_offsets_i32),
      detail::device_pointer<const std::int32_t>(args->sequence_kv_lens_i32),
      detail::device_pointer<const std::int32_t>(
          args->second_sequence_kv_lens_i32),
      detail::device_pointer<const std::int32_t>(args->row_sequence_ids_i32),
      detail::device_pointer<const std::int32_t>(args->row_kv_lens_i32),
      detail::device_pointer<const std::int32_t>(args->row_second_kv_lens_i32),
      detail::device_pointer<const std::int32_t>(args->selected_indices_i32),
      detail::device_pointer<const std::int32_t>(args->selectors_i32),
      detail::device_pointer<const float>(args->attention_sink_f32),
      detail::device_pointer<std::uint64_t>(args->source_addresses_u64),
      detail::device_pointer<std::uint16_t>(args->query_bf16),
      detail::device_pointer<std::uint16_t>(args->gathered_kv_bf16),
      detail::device_pointer<float>(args->raw_scores_f32),
      detail::device_pointer<std::uint16_t>(args->probabilities_bf16),
      detail::device_pointer<float>(args->online_rescales_f32),
      detail::device_pointer<float>(args->denominators_f32),
      detail::device_pointer<float>(args->output_f32),
      detail::device_pointer<std::int32_t>(args->status_i32),
  };
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));

  detail::gather_pack_kernel<<<static_cast<std::uint32_t>(selections),
                               kGatherThreads, 0u, stream>>>(*args, binding);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  detail::qk_kernel<<<static_cast<std::uint32_t>(qk_blocks), kMmaThreads, 0u,
                      stream>>>(*args, binding);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  detail::softmax_kernel<<<static_cast<std::uint32_t>(softmax_blocks),
                           kMmaThreads, 0u, stream>>>(*args, binding);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  detail::pv_kernel<<<static_cast<std::uint32_t>(pv_blocks), kMmaThreads, 0u,
                      stream>>>(*args, binding);
  return cudaPeekAtLastError() == cudaSuccess ? Status::kSuccess
                                              : Status::kLaunchFailed;
}

} // namespace
  // ferrule::cuda::cutlass::operators::hybrid_mla_attention::explicit_selection::schedule

#include <cstdint>

namespace ferrule::cuda::cutlass::operators::hybrid_mla_attention::
    explicit_selection {
namespace dispatch_detail {

template <class T> inline std::uint64_t address(T *pointer) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

inline ExplicitSelectionStatus
make_schedule_args(const ExplicitSelectionArgs *args, schedule::Args *result) {
  if (args == nullptr || result == nullptr) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  const ExplicitSelectionStatus contract_status =
      validate_explicit_selection_contract(args);
  if (contract_status != ExplicitSelectionStatus::kSuccess) {
    return contract_status;
  }
  workspace::Binding scratch{};
  const ExplicitSelectionStatus workspace_status =
      workspace::bind(args, &scratch);
  if (workspace_status != ExplicitSelectionStatus::kSuccess) {
    return workspace_status;
  }

  *result = schedule::Args{
      args->kind,
      args->rows,
      args->tokens_per_sequence,
      args->kv_len,
      args->heads,
      args->head_dim,
      args->selected_width,
      args->page_tokens,
      args->first_elements_per_token,
      args->second_elements_per_token,
      args->layer_index,
      args->layer_count,
      args->flags,
      args->softmax_scale,
      args->reserved0,
      args->first_plane_elements,
      args->second_plane_elements,
      address(args->query),
      address(args->first_plane),
      address(args->second_plane),
      address(args->block_slots),
      address(args->block_offsets),
      address(args->sequence_kv_lens),
      address(args->second_sequence_kv_lens),
      address(args->row_sequence_ids),
      address(args->row_kv_lens),
      address(args->row_second_kv_lens),
      address(args->selected_indices),
      address(args->selectors),
      address(args->attention_sink),
      address(scratch.source_addresses),
      address(scratch.query_bf16),
      address(scratch.gathered_kv_bf16),
      address(scratch.raw_scores),
      address(scratch.probabilities_bf16),
      address(scratch.online_rescales),
      address(scratch.denominators),
      address(args->output),
      address(args->status),
      address(args->stream),
  };
  return ExplicitSelectionStatus::kSuccess;
}

} // namespace dispatch_detail

inline ExplicitSelectionStatus
workspace_requirements(const ExplicitSelectionArgs *args,
                       workspace::Requirements *requirements) {
#if FERRULE_CUDA_HAS_BF16_MMA_SYNC
  return workspace::requirements(args, requirements);
#else
  (void)args;
  (void)requirements;
  return ExplicitSelectionStatus::kUnsupported;
#endif
}

inline ExplicitSelectionStatus
can_implement(const ExplicitSelectionArgs *args) {
#if FERRULE_CUDA_HAS_BF16_MMA_SYNC
  schedule::Args schedule_args{};
  const ExplicitSelectionStatus status =
      dispatch_detail::make_schedule_args(args, &schedule_args);
  return status == ExplicitSelectionStatus::kSuccess
             ? schedule::validate(&schedule_args)
             : status;
#else
  (void)args;
  return ExplicitSelectionStatus::kUnsupported;
#endif
}

inline ExplicitSelectionStatus launch(const ExplicitSelectionArgs *args) {
#if FERRULE_CUDA_HAS_BF16_MMA_SYNC
  schedule::Args schedule_args{};
  const ExplicitSelectionStatus status =
      dispatch_detail::make_schedule_args(args, &schedule_args);
  return status == ExplicitSelectionStatus::kSuccess
             ? schedule::launch(&schedule_args)
             : status;
#else
  (void)args;
  return ExplicitSelectionStatus::kUnsupported;
#endif
}

} // namespace
  // ferrule::cuda::cutlass::operators::hybrid_mla_attention::explicit_selection

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ferrule::cuda::cutlass::operators::hybrid_mla_attention::full_block {
inline constexpr std::uint32_t kBlockRows = 5u;
inline constexpr std::uint32_t kHeads = 64u;
inline constexpr std::uint32_t kHeadDim = 512u;
inline constexpr std::uint32_t kWindowSize = 128u;
inline constexpr std::uint32_t kPageTokens = 16u;
inline constexpr std::uint32_t kTokenCapacity = kWindowSize + kBlockRows;
inline constexpr std::uint32_t kOnlineSoftmaxTile = 64u;
inline constexpr std::uint32_t kOnlineSoftmaxTiles =
    (kTokenCapacity + kOnlineSoftmaxTile - 1u) / kOnlineSoftmaxTile;
inline constexpr std::uint32_t kMmaRows = 8u;
inline constexpr std::uint32_t kMmaColumns = 16u;
inline constexpr std::uint32_t kKTile = 16u;
inline constexpr std::uint32_t kWarpSize = 32u;
inline constexpr std::uint32_t kWarps = 4u;
inline constexpr std::uint32_t kThreads = kWarpSize * kWarps;
inline constexpr std::uint32_t kHeadGroups = kHeads / kWarps;
inline constexpr std::uint32_t kTokenTiles =
    (kTokenCapacity + kMmaColumns - 1u) / kMmaColumns;
inline constexpr std::uint32_t kCooperativeBlocks = kHeadGroups * kTokenTiles;

// One checkpoint-native proposal-attention transaction:
//
//   Q F32/BF16-boundary            [5, 64, 512]
//   committed paged latent KV      [last min(sequence, 128), 512]
//   ephemeral proposal-block KV    [5, 512]
//   per-head attention sink        [64]
//       -> QK BF16 MMA
//       -> sink-aware softmax
//       -> PV BF16 MMA
//   output F32                     [5, 64, 512]
//
// Every proposal query sees the complete five-row proposal block. The block KV
// operand is read-only and is never published through the page table. Scores,
// BF16 probabilities, and device status are graph-stable Ferrule-owned scratch.
struct Args {
  std::uint32_t block_rows;
  std::uint32_t heads;
  std::uint32_t head_dim;
  std::uint32_t sequence_tokens;
  std::uint32_t window_size;
  std::uint32_t page_tokens;
  std::uint32_t elements_per_token;
  std::uint32_t layer_index;
  std::uint32_t layer_count;
  std::uint32_t block_slot_offset;
  std::uint32_t block_slot_count;
  float softmax_scale;
  std::uint32_t reserved;
  std::uint64_t context_plane_elements;

  std::uint64_t query_f32;
  std::uint64_t context_plane_f32;
  std::uint64_t block_kv_f32;
  std::uint64_t block_slots_i32;
  std::uint64_t attention_sink_f32;
  std::uint64_t query_bf16;
  std::uint64_t gathered_kv_bf16;
  std::uint64_t scores_f32;
  std::uint64_t probabilities_bf16;
  std::uint64_t online_rescales_f32;
  std::uint64_t denominators_f32;
  std::uint64_t output_f32;
  std::uint64_t status_i32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 176u, "hybrid-attention POD ABI changed");
static_assert(alignof(Args) == 8u);
static_assert(offsetof(Args, block_rows) == 0u);
static_assert(offsetof(Args, context_plane_elements) == 56u);
static_assert(offsetof(Args, stream) == 168u);

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidArgument = 2,
  kLaunchFailed = 3,
};

namespace detail {

struct alignas(16) SharedStorage {
  // Shared by four heads: one latent K/V tile is read from global memory once.
  alignas(16) std::uint16_t common[kMmaColumns * kKTile];
  // One query/probability tile per head warp.
  alignas(16) std::uint16_t activation[kWarps][kKTile * kMmaRows];
};

static_assert(sizeof(SharedStorage) == 1536u);

struct Binding {
  const float *query;
  const float *context_plane;
  const float *block_kv;
  const std::int32_t *block_slots;
  const float *attention_sink;
  std::uint16_t *query_bf16;
  std::uint16_t *gathered_kv_bf16;
  float *scores;
  std::uint16_t *probabilities;
  float *online_rescales;
  float *denominators;
  float *output;
  std::int32_t *status;
};

static_assert(std::is_trivially_copyable_v<Binding>);

inline constexpr bool aligned(std::uint64_t address, std::uint64_t alignment) {
  return address != 0u && (address & (alignment - 1u)) == 0u;
}

template <class T>
__host__ __device__ __forceinline__ T *device_pointer(std::uint64_t address) {
  return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address));
}

__device__ __forceinline__ std::uint32_t context_tokens(const Args &args) {
  return args.sequence_tokens < args.window_size ? args.sequence_tokens
                                                 : args.window_size;
}

__device__ __forceinline__ float latent_value(const Args &args,
                                              const Binding &binding,
                                              std::uint32_t concatenated_token,
                                              std::uint32_t dimension) {
  const std::uint32_t committed = context_tokens(args);
  if (concatenated_token >= committed) {
    const std::uint32_t block_row = concatenated_token - committed;
    return block_row < args.block_rows
               ? binding.block_kv[static_cast<std::uint64_t>(block_row) *
                                      args.head_dim +
                                  dimension]
               : 0.0f;
  }

  const std::uint32_t logical =
      args.sequence_tokens - committed + concatenated_token;
  const std::uint32_t relative_page = logical / args.page_tokens;
  if (relative_page >= args.block_slot_count) {
    atomicExch(binding.status, 1);
    return 0.0f;
  }
  const std::int32_t slot =
      binding.block_slots[args.block_slot_offset + relative_page];
  if (slot < 0) {
    atomicExch(binding.status, 1);
    return 0.0f;
  }

  const std::uint64_t layer_stride =
      static_cast<std::uint64_t>(args.page_tokens) * args.elements_per_token;
  const std::uint64_t slot_stride =
      static_cast<std::uint64_t>(args.layer_count) * layer_stride;
  const std::uint64_t offset =
      static_cast<std::uint64_t>(slot) * slot_stride +
      static_cast<std::uint64_t>(args.layer_index) * layer_stride +
      static_cast<std::uint64_t>(logical % args.page_tokens) *
          args.elements_per_token +
      dimension;
  if (offset >= args.context_plane_elements) {
    atomicExch(binding.status, 1);
    return 0.0f;
  }
  return binding.context_plane[offset];
}

__device__ __forceinline__ void pack_inputs(const Args &args,
                                            const Binding &binding) {
  const std::uint64_t thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const std::uint64_t query_values =
      static_cast<std::uint64_t>(args.block_rows) * args.heads * args.head_dim;
  for (std::uint64_t index = thread; index < query_values; index += stride) {
    binding.query_bf16[index] = bf16::from_f32_rne(binding.query[index]);
  }

  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
  const std::uint64_t kv_values =
      static_cast<std::uint64_t>(total_tokens) * args.head_dim;
  for (std::uint64_t index = thread; index < kv_values; index += stride) {
    const std::uint32_t token =
        static_cast<std::uint32_t>(index / args.head_dim);
    const std::uint32_t dimension =
        static_cast<std::uint32_t>(index % args.head_dim);
    binding.gathered_kv_bf16[index] =
        bf16::from_f32_rne(latent_value(args, binding, token, dimension));
  }
}

__device__ __forceinline__ void
stage_qk_latent(const Args &args, const Binding &binding, SharedStorage &shared,
                std::uint32_t token_base, std::uint32_t dimension_base) {
  for (std::uint32_t linear = threadIdx.x; linear < kMmaColumns * kKTile;
       linear += blockDim.x) {
    const std::uint32_t token_local = linear / kKTile;
    const std::uint32_t dimension_local = linear % kKTile;
    const std::uint32_t token = token_base + token_local;
    const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
    shared.common[linear] =
        token < total_tokens
            ? binding.gathered_kv_bf16[static_cast<std::uint64_t>(token) *
                                           args.head_dim +
                                       dimension_base + dimension_local]
            : 0u;
  }
}

__device__ __forceinline__ void
stage_query(const Args &args, const Binding &binding, SharedStorage &shared,
            std::uint32_t warp, std::uint32_t lane, std::uint32_t head,
            std::uint32_t dimension_base) {
  for (std::uint32_t linear = lane; linear < kKTile * kMmaRows;
       linear += kWarpSize) {
    const std::uint32_t dimension_local = linear / kMmaRows;
    const std::uint32_t row = linear % kMmaRows;
    shared.activation[warp][linear] =
        row < args.block_rows
            ? binding.query_bf16[(static_cast<std::uint64_t>(row) * args.heads +
                                  head) *
                                     args.head_dim +
                                 dimension_base + dimension_local]
            : 0u;
  }
}

__device__ __forceinline__ void
store_scores(const Args &args, const Binding &binding, std::uint32_t head,
             std::uint32_t token_base, std::uint32_t lane,
             const float (&accumulator)[4]) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t token =
        token_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row = row_pair * 2u + (element & 1u);
    if (row < args.block_rows && token < total_tokens) {
      binding.scores[(static_cast<std::uint64_t>(row) * args.heads + head) *
                         kTokenCapacity +
                     token] =
          __fmul_rn(accumulator[element], args.softmax_scale);
    }
  }
}

__device__ __forceinline__ void
qk_task(const Args &args, const Binding &binding, SharedStorage &shared,
        std::uint32_t task, std::uint32_t warp, std::uint32_t lane) {
  const std::uint32_t token_tile = task % kTokenTiles;
  const std::uint32_t head_group = task / kTokenTiles;
  const std::uint32_t token_base = token_tile * kMmaColumns;
  const std::uint32_t head = head_group * kWarps + warp;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t dimension_base = 0u; dimension_base < kHeadDim;
       dimension_base += kKTile) {
    stage_qk_latent(args, binding, shared, token_base, dimension_base);
    stage_query(args, binding, shared, warp, lane, head, dimension_base);
    __syncthreads();
    std::uint32_t latent_fragment[4];
    std::uint32_t query_fragment[2];
    bf16::load_weight_fragment(shared.common, lane, latent_fragment);
    bf16::load_activation_fragment(shared.activation[warp], lane,
                                   query_fragment);
    bf16::mma(accumulator, latent_fragment, query_fragment);
    __syncthreads();
  }
  store_scores(args, binding, head, token_base, lane, accumulator);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (std::uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (std::uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ void softmax_task(const Args &args,
                                             const Binding &binding,
                                             std::uint32_t task,
                                             std::uint32_t lane) {
  const std::uint32_t head = task % args.heads;
  const std::uint32_t row = task / args.heads;
  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
  const std::uint64_t base =
      (static_cast<std::uint64_t>(row) * args.heads + head) * kTokenCapacity;

  float running_maximum = -INFINITY;
  float denominator = 0.0f;
  for (std::uint32_t online_tile = 0u; online_tile < kOnlineSoftmaxTiles;
       ++online_tile) {
    const std::uint32_t tile_base = online_tile * kOnlineSoftmaxTile;
    float tile_maximum = -INFINITY;
#pragma unroll
    for (std::uint32_t iteration = 0u; iteration < 2u; ++iteration) {
      const std::uint32_t token = tile_base + lane + iteration * kWarpSize;
      if (token < total_tokens) {
        tile_maximum = fmaxf(tile_maximum, binding.scores[base + token]);
      }
    }
    tile_maximum = warp_max(tile_maximum);
    const float next_maximum = fmaxf(running_maximum, tile_maximum);
    const float rescale = running_maximum == -INFINITY
                              ? 0.0f
                              : expf(__fsub_rn(running_maximum, next_maximum));
    float tile_sum = 0.0f;
#pragma unroll
    for (std::uint32_t iteration = 0u; iteration < 2u; ++iteration) {
      const std::uint32_t token = tile_base + lane + iteration * kWarpSize;
      if (token >= kTokenCapacity) {
        continue;
      }
      std::uint16_t weight = 0u;
      if (token < total_tokens) {
        const float value =
            expf(__fsub_rn(binding.scores[base + token], next_maximum));
        tile_sum = __fadd_rn(tile_sum, value);
        weight = bf16::from_f32_rne(value);
      }
      binding.probabilities[base + token] = weight;
    }
    tile_sum = warp_sum(tile_sum);
    denominator = __fmaf_rn(denominator, rescale, tile_sum);
    if (lane == 0u) {
      binding.online_rescales[task * kOnlineSoftmaxTiles + online_tile] =
          rescale;
    }
    running_maximum = next_maximum;
  }

  if (lane == 0u) {
    binding.denominators[task] =
        running_maximum == -INFINITY
            ? 1.0f
            : __fadd_rn(denominator,
                        expf(__fsub_rn(binding.attention_sink[head],
                                       running_maximum)));
  }
}

__device__ __forceinline__ void
stage_pv_latent(const Args &args, const Binding &binding, SharedStorage &shared,
                std::uint32_t channel_base, std::uint32_t token_base) {
  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
  for (std::uint32_t linear = threadIdx.x; linear < kMmaColumns * kKTile;
       linear += blockDim.x) {
    const std::uint32_t channel_local = linear / kKTile;
    const std::uint32_t token_local = linear % kKTile;
    const std::uint32_t token = token_base + token_local;
    shared.common[linear] =
        token < total_tokens
            ? binding.gathered_kv_bf16[static_cast<std::uint64_t>(token) *
                                           args.head_dim +
                                       channel_base + channel_local]
            : 0u;
  }
}

__device__ __forceinline__ void
stage_probabilities(const Args &args, const Binding &binding,
                    SharedStorage &shared, std::uint32_t warp,
                    std::uint32_t lane, std::uint32_t head,
                    std::uint32_t token_base) {
  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
  for (std::uint32_t linear = lane; linear < kKTile * kMmaRows;
       linear += kWarpSize) {
    const std::uint32_t token_local = linear / kMmaRows;
    const std::uint32_t row = linear % kMmaRows;
    const std::uint32_t token = token_base + token_local;
    shared.activation[warp][linear] =
        row < args.block_rows && token < total_tokens
            ? binding
                  .probabilities[(static_cast<std::uint64_t>(row) * args.heads +
                                  head) *
                                     kTokenCapacity +
                                 token]
            : 0u;
  }
}

__device__ __forceinline__ void
store_output(const Args &args, const Binding &binding, std::uint32_t head,
             std::uint32_t channel_base, std::uint32_t lane,
             const float (&accumulator)[4]) {
  const std::uint32_t channel_group = lane >> 2;
  const std::uint32_t row_pair = lane & 3u;
#pragma unroll
  for (std::uint32_t element = 0u; element < 4u; ++element) {
    const std::uint32_t channel =
        channel_base + channel_group + (element >= 2u ? 8u : 0u);
    const std::uint32_t row = row_pair * 2u + (element & 1u);
    if (row < args.block_rows && channel < args.head_dim) {
      const std::uint64_t pair =
          static_cast<std::uint64_t>(row) * args.heads + head;
      const float normalized =
          __fdiv_rn(accumulator[element], binding.denominators[pair]);
      binding.output[pair * args.head_dim + channel] =
          bf16::to_f32(bf16::from_f32_rne(normalized));
    }
  }
}

__device__ __forceinline__ void
pv_task(const Args &args, const Binding &binding, SharedStorage &shared,
        std::uint32_t task, std::uint32_t warp, std::uint32_t lane) {
  constexpr std::uint32_t kChannelTiles = kHeadDim / kMmaColumns;
  const std::uint32_t channel_tile = task % kChannelTiles;
  const std::uint32_t head_group = task / kChannelTiles;
  const std::uint32_t channel_base = channel_tile * kMmaColumns;
  const std::uint32_t head = head_group * kWarps + warp;
  const std::uint32_t total_tokens = context_tokens(args) + args.block_rows;
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (std::uint32_t token_base = 0u; token_base < total_tokens;
       token_base += kKTile) {
    if ((token_base % kOnlineSoftmaxTile) == 0u) {
      const std::uint32_t online_tile = token_base / kOnlineSoftmaxTile;
      const std::uint32_t row_pair = lane & 3u;
#pragma unroll
      for (std::uint32_t element = 0u; element < 4u; ++element) {
        const std::uint32_t row = row_pair * 2u + (element & 1u);
        if (row < args.block_rows) {
          const std::uint64_t pair =
              static_cast<std::uint64_t>(row) * args.heads + head;
          accumulator[element] = __fmul_rn(
              accumulator[element],
              binding
                  .online_rescales[pair * kOnlineSoftmaxTiles + online_tile]);
        }
      }
    }
    stage_pv_latent(args, binding, shared, channel_base, token_base);
    stage_probabilities(args, binding, shared, warp, lane, head, token_base);
    __syncthreads();
    std::uint32_t latent_fragment[4];
    std::uint32_t probability_fragment[2];
    bf16::load_weight_fragment(shared.common, lane, latent_fragment);
    bf16::load_activation_fragment(shared.activation[warp], lane,
                                   probability_fragment);
    bf16::mma(accumulator, latent_fragment, probability_fragment);
    __syncthreads();
  }
  store_output(args, binding, head, channel_base, lane, accumulator);
}

__global__ __launch_bounds__(kThreads, 1) void kernel(Args args,
                                                      Binding binding) {
  __shared__ SharedStorage shared;
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const std::uint32_t warp = threadIdx.x / kWarpSize;
  const std::uint32_t lane = threadIdx.x & (kWarpSize - 1u);
  const std::uint32_t global_warp = blockIdx.x * kWarps + warp;
  const std::uint32_t warp_stride = gridDim.x * kWarps;

  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *binding.status = 0;
  }
  grid.sync();

  pack_inputs(args, binding);
  grid.sync();

  for (std::uint32_t task = blockIdx.x; task < kCooperativeBlocks;
       task += gridDim.x) {
    qk_task(args, binding, shared, task, warp, lane);
  }
  grid.sync();

  const std::uint32_t softmax_tasks = args.block_rows * args.heads;
  for (std::uint32_t task = global_warp; task < softmax_tasks;
       task += warp_stride) {
    softmax_task(args, binding, task, lane);
  }
  grid.sync();

  constexpr std::uint32_t kChannelTiles = kHeadDim / kMmaColumns;
  constexpr std::uint32_t kPvTasks = kHeadGroups * kChannelTiles;
  for (std::uint32_t task = blockIdx.x; task < kPvTasks; task += gridDim.x) {
    pv_task(args, binding, shared, task, warp, lane);
  }
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr) {
    return Status::kInvalidArgument;
  }
  if (args->page_tokens != kPageTokens) {
    return Status::kInvalidArgument;
  }
  const std::uint64_t required_slots =
      (static_cast<std::uint64_t>(args->sequence_tokens) + args->page_tokens -
       1u) /
      args->page_tokens;
  if (args->block_rows != kBlockRows || args->heads != kHeads ||
      args->head_dim != kHeadDim || args->window_size != kWindowSize ||
      args->elements_per_token != kHeadDim || args->sequence_tokens == 0u ||
      args->layer_count == 0u || args->layer_index >= args->layer_count ||
      args->block_slot_count < required_slots || args->reserved != 0u ||
      args->context_plane_elements == 0u ||
      !std::isfinite(args->softmax_scale) || args->softmax_scale <= 0.0f) {
    return Status::kInvalidArgument;
  }
  const bool pointers_valid = detail::aligned(args->query_f32, 16u) &&
                              detail::aligned(args->context_plane_f32, 16u) &&
                              detail::aligned(args->block_kv_f32, 16u) &&
                              detail::aligned(args->block_slots_i32, 4u) &&
                              detail::aligned(args->attention_sink_f32, 16u) &&
                              detail::aligned(args->query_bf16, 16u) &&
                              detail::aligned(args->gathered_kv_bf16, 16u) &&
                              detail::aligned(args->scores_f32, 16u) &&
                              detail::aligned(args->probabilities_bf16, 16u) &&
                              detail::aligned(args->online_rescales_f32, 16u) &&
                              detail::aligned(args->denominators_f32, 16u) &&
                              detail::aligned(args->output_f32, 16u) &&
                              detail::aligned(args->status_i32, 4u);
  return pointers_valid ? Status::kSuccess : Status::kInvalidArgument;
}

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));
  const detail::Binding binding{
      detail::device_pointer<const float>(args->query_f32),
      detail::device_pointer<const float>(args->context_plane_f32),
      detail::device_pointer<const float>(args->block_kv_f32),
      detail::device_pointer<const std::int32_t>(args->block_slots_i32),
      detail::device_pointer<const float>(args->attention_sink_f32),
      detail::device_pointer<std::uint16_t>(args->query_bf16),
      detail::device_pointer<std::uint16_t>(args->gathered_kv_bf16),
      detail::device_pointer<float>(args->scores_f32),
      detail::device_pointer<std::uint16_t>(args->probabilities_bf16),
      detail::device_pointer<float>(args->online_rescales_f32),
      detail::device_pointer<float>(args->denominators_f32),
      detail::device_pointer<float>(args->output_f32),
      detail::device_pointer<std::int32_t>(args->status_i32),
  };
  int blocks_per_sm = 0;
  int device = 0;
  int multiprocessors = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm, detail::kernel, kThreads, 0u) != cudaSuccess ||
      cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess ||
      blocks_per_sm <= 0 || multiprocessors <= 0) {
    return Status::kLaunchFailed;
  }
  const std::uint32_t resident_blocks =
      static_cast<std::uint32_t>(blocks_per_sm) *
      static_cast<std::uint32_t>(multiprocessors);
  const std::uint32_t blocks = resident_blocks < kCooperativeBlocks
                                   ? resident_blocks
                                   : kCooperativeBlocks;
  void *kernel_args[] = {const_cast<Args *>(args),
                         const_cast<detail::Binding *>(&binding)};
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::kernel), dim3(blocks, 1u, 1u),
      dim3(kThreads, 1u, 1u), kernel_args, 0u, stream);
  return status == cudaSuccess ? Status::kSuccess : Status::kLaunchFailed;
}

} // namespace
  // ferrule::cuda::cutlass::operators::hybrid_mla_attention::full_block
