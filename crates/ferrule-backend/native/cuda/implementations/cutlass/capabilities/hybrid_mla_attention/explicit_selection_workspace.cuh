#ifndef FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_EXPLICIT_SELECTION_WORKSPACE_CUH_
#define FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_EXPLICIT_SELECTION_WORKSPACE_CUH_

#include "explicit_selection_contract.cuh"

#include <cstdint>

namespace ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::
    explicit_selection_workspace {

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
  // ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::explicit_selection_workspace

#endif // FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_EXPLICIT_SELECTION_WORKSPACE_CUH_
