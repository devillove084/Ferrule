#ifndef FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_EXPLICIT_SELECTION_CONTRACT_CUH_
#define FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_EXPLICIT_SELECTION_CONTRACT_CUH_

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ferrule::cuda::cutlass::capabilities::hybrid_mla_attention {

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

} // namespace ferrule::cuda::cutlass::capabilities::hybrid_mla_attention

#endif // FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_EXPLICIT_SELECTION_CONTRACT_CUH_
