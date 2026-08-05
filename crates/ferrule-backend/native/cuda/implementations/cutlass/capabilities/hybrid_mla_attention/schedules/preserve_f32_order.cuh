#ifndef FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_SCHEDULES_PRESERVE_F32_ORDER_CUH_
#define FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_SCHEDULES_PRESERVE_F32_ORDER_CUH_

#include "../explicit_selection_contract.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::
    schedules::preserve_f32_order {
inline constexpr std::uint32_t kContiguous = kExplicitSelectionContiguous;
inline constexpr std::uint32_t kPaged = kExplicitSelectionPaged;
inline constexpr std::uint32_t kDualPaged = kExplicitSelectionDualPaged;
inline constexpr std::uint32_t kHeads = kExplicitSelectionHeads;
inline constexpr std::uint32_t kHeadDim = kExplicitSelectionHeadDim;
inline constexpr std::uint32_t kMaximumSelectedWidth =
    kExplicitSelectionMaximumWidth;
inline constexpr std::uint32_t kStage1Threads = 64u;
inline constexpr std::uint32_t kStage2HeadGroupThreads = 256u;
inline constexpr std::uint32_t kStage2HeadTile = 2u;
static_assert(kStage1Threads == kHeads);
static_assert(kStage2HeadTile == 2u);
static_assert(kHeads % kStage2HeadTile == 0u);
static_assert(kHeadDim == 2u * kStage2HeadGroupThreads);
static_assert(sizeof(std::uintptr_t) == sizeof(std::uint64_t));

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
  std::uint64_t row_sequence_ids_i32;
  std::uint64_t row_kv_lens_i32;
  std::uint64_t selected_indices_i32;
  std::uint64_t selectors_i32;
  std::uint64_t attention_sink_f32;
  std::uint64_t source_addresses_u64;
  std::uint64_t raw_scores_f32;
  std::uint64_t output_f32;
  std::uint64_t status_i32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 208u, "selected-attention private POD changed");
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
static_assert(offsetof(Args, row_sequence_ids_i32) == 128u);
static_assert(offsetof(Args, row_kv_lens_i32) == 136u);
static_assert(offsetof(Args, selected_indices_i32) == 144u);
static_assert(offsetof(Args, selectors_i32) == 152u);
static_assert(offsetof(Args, attention_sink_f32) == 160u);
static_assert(offsetof(Args, source_addresses_u64) == 168u);
static_assert(offsetof(Args, raw_scores_f32) == 176u);
static_assert(offsetof(Args, output_f32) == 184u);
static_assert(offsetof(Args, status_i32) == 192u);
static_assert(offsetof(Args, stream) == 200u);

using Status = ExplicitSelectionStatus;

namespace detail {

struct Binding {
  const float *query;
  const float *first_plane;
  const float *second_plane;
  const std::int32_t *block_slots;
  const std::int32_t *block_offsets;
  const std::int32_t *sequence_kv_lens;
  const std::int32_t *row_sequence_ids;
  const std::int32_t *row_kv_lens;
  const std::int32_t *selected_indices;
  const std::int32_t *selectors;
  const float *attention_sink;
  std::uint64_t *source_addresses;
  float *raw_scores;
  float *output;
  std::int32_t *status;
};

struct ValueLocation {
  const float *values;
  std::uint64_t base;
  bool valid;
};

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
paged_location(const Args &args, const Binding &binding, const float *plane,
               std::uint64_t plane_elements, std::uint32_t elements_per_token,
               std::uint32_t sequence, std::int32_t logical) {
  if (logical < 0 || sequence == UINT32_MAX) {
    return {nullptr, 0u, false};
  }

  const std::int32_t sequence_length = binding.sequence_kv_lens[sequence];
  if (sequence_length < 0) {
    return {nullptr, 0u, false};
  }
  if (logical >= sequence_length) {
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
  if ((args.flags & 2u) != 0u) {
    const std::int32_t visible = binding.row_kv_lens[row];
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
  const std::int32_t selector =
      args.kind == kDualPaged ? binding.selectors[index] : 0;
  if (selector == 0) {
    return paged_location(args, binding, binding.first_plane,
                          args.first_plane_elements,
                          args.first_elements_per_token, sequence, logical);
  }
  if (args.kind == kDualPaged && selector == 1) {
    return paged_location(args, binding, binding.second_plane,
                          args.second_plane_elements,
                          args.second_elements_per_token, sequence, logical);
  }
  return {nullptr, 0u, false};
}

__global__
__launch_bounds__(kStage1Threads) void stage1_scores_kernel(Args args,
                                                            Binding binding) {

  __shared__ std::uint64_t shared_source_address;
  __shared__ float shared_values[kHeadDim];

  const std::uint64_t selection_index = blockIdx.x;
  const std::uint32_t row =
      static_cast<std::uint32_t>(selection_index / args.selected_width);
  const std::uint32_t selected =
      static_cast<std::uint32_t>(selection_index % args.selected_width);

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
  if (source != nullptr) {
    for (std::uint32_t dimension = threadIdx.x; dimension < kHeadDim;
         dimension += kStage1Threads) {
      shared_values[dimension] = source[dimension];
    }
  }
  __syncthreads();

  const std::uint32_t head = threadIdx.x;
  const std::uint64_t score_index =
      (static_cast<std::uint64_t>(row) * args.heads + head) *
          args.selected_width +
      selected;
  if (source == nullptr) {
    binding.raw_scores[score_index] = -INFINITY;
    return;
  }

  const std::uint64_t query_base =
      (static_cast<std::uint64_t>(row) * args.heads + head) * args.head_dim;
  float dot = 0.0f;
#pragma unroll 1
  for (std::uint32_t dimension = 0u; dimension < kHeadDim; ++dimension) {
    dot = __fmaf_rn(binding.query[query_base + dimension],
                    shared_values[dimension], dot);
  }
  binding.raw_scores[score_index] = dot;
}

__global__ __launch_bounds__(
    kStage2HeadGroupThreads) void stage2_head_group_kernel(Args args,
                                                           Binding binding) {

  extern __shared__ float shared_storage[];
  float *const shared_weights = shared_storage;
  float *const shared_denominators =
      shared_weights +
      static_cast<std::uint64_t>(args.selected_width) * kStage2HeadTile;

  constexpr std::uint32_t kHeadGroupsPerRow = kHeads / kStage2HeadTile;
  const std::uint32_t row = blockIdx.x / kHeadGroupsPerRow;
  const std::uint32_t head_group = blockIdx.x % kHeadGroupsPerRow;
  const std::uint32_t first_head = head_group * kStage2HeadTile;
  const std::uint64_t selected_base =
      static_cast<std::uint64_t>(row) * args.selected_width;

  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *binding.status = 0;
  }

  if (threadIdx.x < kStage2HeadTile) {
    const std::uint32_t local_head = threadIdx.x;
    const std::uint32_t head = first_head + local_head;
    const std::uint64_t score_base =
        (static_cast<std::uint64_t>(row) * args.heads + head) *
        args.selected_width;
    const float sink_score = binding.attention_sink[head];
    float maximum = sink_score;
#pragma unroll 1
    for (std::uint32_t selected = 0u; selected < args.selected_width;
         ++selected) {
      if (binding.source_addresses[selected_base + selected] == 0u) {
        continue;
      }
      const float score = __fmul_rn(binding.raw_scores[score_base + selected],
                                    args.softmax_scale);
      if (score > maximum) {
        maximum = score;
      }
    }

    float denominator = expf(__fsub_rn(sink_score, maximum));
#pragma unroll 1
    for (std::uint32_t selected = 0u; selected < args.selected_width;
         ++selected) {
      const std::uint64_t weight_index =
          static_cast<std::uint64_t>(selected) * kStage2HeadTile + local_head;
      if (binding.source_addresses[selected_base + selected] == 0u) {
        shared_weights[weight_index] = 0.0f;
        continue;
      }
      const float weight =
          expf(__fmaf_rn(binding.raw_scores[score_base + selected],
                         args.softmax_scale, -maximum));
      shared_weights[weight_index] = weight;
      denominator = __fadd_rn(denominator, weight);
    }
    shared_denominators[local_head] = denominator;
  }
  __syncthreads();

  const std::uint32_t d0 = threadIdx.x;
  const std::uint32_t d1 = threadIdx.x + kStage2HeadGroupThreads;
  float accumulator_h0_d0 = 0.0f;
  float accumulator_h0_d1 = 0.0f;
  float accumulator_h1_d0 = 0.0f;
  float accumulator_h1_d1 = 0.0f;
#pragma unroll 1
  for (std::uint32_t selected = 0u; selected < args.selected_width;
       ++selected) {
    const std::uint64_t source_address =
        binding.source_addresses[selected_base + selected];
    if (source_address == 0u) {
      continue;
    }
    const auto *source = reinterpret_cast<const float *>(
        static_cast<std::uintptr_t>(source_address));
    const float value_d0 = source[d0];
    const float value_d1 = source[d1];
    const std::uint64_t weight_base =
        static_cast<std::uint64_t>(selected) * kStage2HeadTile;
    const float weight_h0 = shared_weights[weight_base];
    const float weight_h1 = shared_weights[weight_base + 1u];
    accumulator_h0_d0 = __fmaf_rn(weight_h0, value_d0, accumulator_h0_d0);
    accumulator_h0_d1 = __fmaf_rn(weight_h0, value_d1, accumulator_h0_d1);
    accumulator_h1_d0 = __fmaf_rn(weight_h1, value_d0, accumulator_h1_d0);
    accumulator_h1_d1 = __fmaf_rn(weight_h1, value_d1, accumulator_h1_d1);
  }

  const std::uint64_t output_base_h0 =
      (static_cast<std::uint64_t>(row) * args.heads + first_head) *
      args.head_dim;
  const std::uint64_t output_base_h1 = output_base_h0 + args.head_dim;
  binding.output[output_base_h0 + d0] =
      __fdiv_rn(accumulator_h0_d0, shared_denominators[0u]);
  binding.output[output_base_h0 + d1] =
      __fdiv_rn(accumulator_h0_d1, shared_denominators[0u]);
  binding.output[output_base_h1 + d0] =
      __fdiv_rn(accumulator_h1_d0, shared_denominators[1u]);
  binding.output[output_base_h1 + d1] =
      __fdiv_rn(accumulator_h1_d1, shared_denominators[1u]);
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr) {
    return Status::kInvalidArgument;
  }
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
       args->first_elements_per_token < kHeadDim)) {
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
      detail::aligned(args->raw_scores_f32, 16u) &&
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
       !detail::aligned(args->selectors_i32, 4u))) {
    return Status::kInvalidArgument;
  }
  if (!detail::optional_aligned(args->second_plane_f32, 16u) ||
      !detail::optional_aligned(args->block_slots_i32, 4u) ||
      !detail::optional_aligned(args->block_offsets_i32, 4u) ||
      !detail::optional_aligned(args->sequence_kv_lens_i32, 4u) ||
      !detail::optional_aligned(args->row_sequence_ids_i32, 4u) ||
      !detail::optional_aligned(args->row_kv_lens_i32, 4u) ||
      !detail::optional_aligned(args->selectors_i32, 4u)) {
    return Status::kInvalidArgument;
  }
  return Status::kSuccess;
}

#ifdef FERRULE_CUDA_TEST_ORACLE
namespace test_oracle {

struct ValueLocation {
  const float *values;
  std::uint64_t base;
  bool valid;
};

__device__ __forceinline__ ValueLocation
paged_location(const Args &args, const std::int32_t *block_slots,
               const std::int32_t *block_offsets,
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
  const std::int32_t begin = block_offsets[sequence];
  const std::int32_t end = block_offsets[sequence + 1u];
  if (begin < 0 || end < begin) {
    return {nullptr, 0u, false};
  }
  const std::uint64_t entry =
      static_cast<std::uint64_t>(begin) +
      static_cast<std::uint32_t>(logical) / args.page_tokens;
  if (entry >= static_cast<std::uint64_t>(end)) {
    return {nullptr, 0u, false};
  }
  const std::int32_t slot = block_slots[entry];
  if (slot < 0) {
    return {nullptr, 0u, false};
  }

  std::uint64_t layer_stride = 0u;
  std::uint64_t slot_stride = 0u;
  std::uint64_t base = 0u;
  std::uint64_t part = 0u;
  if (!detail::checked_multiply(args.page_tokens, elements_per_token,
                                layer_stride) ||
      !detail::checked_multiply(args.layer_count, layer_stride, slot_stride) ||
      !detail::checked_multiply(static_cast<std::uint32_t>(slot), slot_stride,
                                base) ||
      !detail::checked_multiply(args.layer_index, layer_stride, part) ||
      !detail::checked_add(base, part, base) ||
      !detail::checked_multiply(static_cast<std::uint32_t>(logical) %
                                    args.page_tokens,
                                elements_per_token, part) ||
      !detail::checked_add(base, part, base) || base > plane_elements ||
      args.head_dim > plane_elements - base) {
    return {nullptr, 0u, false};
  }
  return {plane, base, true};
}

__device__ __forceinline__ ValueLocation
selected_location(const Args &args, std::uint32_t row, std::uint32_t selected) {
  const auto *first_plane =
      detail::device_pointer<const float>(args.first_plane_f32);
  const auto *second_plane =
      detail::device_pointer<const float>(args.second_plane_f32);
  const auto *selected_indices =
      detail::device_pointer<const std::int32_t>(args.selected_indices_i32);
  const auto *row_kv_lens =
      detail::device_pointer<const std::int32_t>(args.row_kv_lens_i32);
  const std::uint64_t entry =
      static_cast<std::uint64_t>(row) * args.selected_width + selected;
  std::int32_t logical = selected_indices[entry];
  if ((args.flags & 2u) != 0u && (logical < 0 || logical >= row_kv_lens[row])) {
    logical = -1;
  }
  if (logical < 0) {
    return {nullptr, 0u, false};
  }
  if (args.kind == kContiguous) {
    if (static_cast<std::uint32_t>(logical) >= args.kv_len) {
      return {nullptr, 0u, false};
    }
    return {first_plane, static_cast<std::uint64_t>(logical) * args.head_dim,
            true};
  }

  std::uint32_t sequence = 0u;
  if ((args.flags & 1u) != 0u) {
    const auto *row_sequence_ids =
        detail::device_pointer<const std::int32_t>(args.row_sequence_ids_i32);
    const std::int32_t value = row_sequence_ids[row];
    sequence = value >= 0 ? static_cast<std::uint32_t>(value) : UINT32_MAX;
  } else {
    sequence = row / args.tokens_per_sequence;
  }
  const auto *block_slots =
      detail::device_pointer<const std::int32_t>(args.block_slots_i32);
  const auto *block_offsets =
      detail::device_pointer<const std::int32_t>(args.block_offsets_i32);
  const auto *sequence_kv_lens =
      detail::device_pointer<const std::int32_t>(args.sequence_kv_lens_i32);
  const std::int32_t selector =
      args.kind == kDualPaged ? detail::device_pointer<const std::int32_t>(
                                    args.selectors_i32)[entry]
                              : 0;
  if (selector == 0) {
    return paged_location(args, block_slots, block_offsets, sequence_kv_lens,
                          first_plane, args.first_plane_elements,
                          args.first_elements_per_token, sequence, logical);
  }
  if (args.kind == kDualPaged && selector == 1) {
    return paged_location(args, block_slots, block_offsets, sequence_kv_lens,
                          second_plane, args.second_plane_elements,
                          args.second_elements_per_token, sequence, logical);
  }
  return {nullptr, 0u, false};
}

__global__ void scalar_kernel(Args args) {
  const std::uint64_t pair =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t pair_count =
      static_cast<std::uint64_t>(args.rows) * args.heads;
  if (pair >= pair_count) {
    return;
  }

  const std::uint32_t row = static_cast<std::uint32_t>(pair / args.heads);
  const std::uint32_t head = static_cast<std::uint32_t>(pair % args.heads);
  const auto *query = detail::device_pointer<const float>(args.query_f32);
  const auto *attention_sink =
      detail::device_pointer<const float>(args.attention_sink_f32);
  auto *output = detail::device_pointer<float>(args.output_f32);
  const std::uint64_t query_base = pair * args.head_dim;
  const float sink_value = attention_sink[head];

  float maximum = sink_value;
#pragma unroll 1
  for (std::uint32_t selected = 0u; selected < args.selected_width;
       ++selected) {
    const ValueLocation location = selected_location(args, row, selected);
    if (!location.valid) {
      continue;
    }
    float dot = 0.0f;
#pragma unroll 1
    for (std::uint32_t dimension = 0u; dimension < args.head_dim; ++dimension) {
      const float q = query[query_base + dimension];
      const float v = location.values[location.base + dimension];
      dot = __fmaf_rn(q, v, dot);
    }
    const float score = __fmul_rn(dot, args.softmax_scale);
    if (score > maximum) {
      maximum = score;
    }
  }

#pragma unroll 1
  for (std::uint32_t dimension = 0u; dimension < args.head_dim; ++dimension) {
    output[query_base + dimension] = 0.0f;
  }
  float denominator = expf(__fsub_rn(sink_value, maximum));
#pragma unroll 1
  for (std::uint32_t selected = 0u; selected < args.selected_width;
       ++selected) {
    const ValueLocation location = selected_location(args, row, selected);
    if (!location.valid) {
      continue;
    }
    float dot = 0.0f;
#pragma unroll 1
    for (std::uint32_t dimension = 0u; dimension < args.head_dim; ++dimension) {
      const float q = query[query_base + dimension];
      const float v = location.values[location.base + dimension];
      dot = __fmaf_rn(q, v, dot);
    }
    const float weight = expf(__fmaf_rn(dot, args.softmax_scale, -maximum));
    denominator = __fadd_rn(denominator, weight);
#pragma unroll 1
    for (std::uint32_t dimension = 0u; dimension < args.head_dim; ++dimension) {
      const float v = location.values[location.base + dimension];
      output[query_base + dimension] =
          __fmaf_rn(weight, v, output[query_base + dimension]);
    }
  }
#pragma unroll 1
  for (std::uint32_t dimension = 0u; dimension < args.head_dim; ++dimension) {
    output[query_base + dimension] =
        __fdiv_rn(output[query_base + dimension], denominator);
  }
}

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }
  constexpr std::uint32_t threads = 128u;
  const std::uint64_t pairs =
      static_cast<std::uint64_t>(args->rows) * args->heads;
  const std::uint64_t blocks = (pairs + threads - 1u) / threads;
  if (blocks == 0u || blocks > UINT32_MAX) {
    return Status::kLaunchFailed;
  }
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));
  scalar_kernel<<<static_cast<std::uint32_t>(blocks), threads, 0u, stream>>>(
      *args);
  return cudaPeekAtLastError() == cudaSuccess ? Status::kSuccess
                                              : Status::kLaunchFailed;
}

inline constexpr std::uint32_t kCompareResultWords = 5u;

__device__ __forceinline__ std::uint32_t ordered_float_bits(float value) {
  const std::uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
}

__device__ __forceinline__ std::uint32_t
float_bits_from_ordered(std::uint32_t ordered) {
  return (ordered & 0x80000000u) != 0u ? ordered & 0x7fffffffu : ~ordered;
}

__global__ void compare_kernel(const float *actual, const float *expected,
                               std::uint64_t count, std::uint32_t *result) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const float actual_value = actual[index];
  const float expected_value = expected[index];
  const float absolute = fabsf(actual_value - expected_value);
  const bool mismatch =
      __float_as_uint(actual_value) != __float_as_uint(expected_value);
  if (!mismatch) {
    return;
  }

  atomicAdd(result, 1u);
  atomicMin(result + 1u, static_cast<std::uint32_t>(index));
  if (isnan(absolute)) {
    atomicMax(result + 2u, ordered_float_bits(INFINITY));
  } else {
    atomicMax(result + 2u, ordered_float_bits(absolute));
  }
}

__global__ void finalize_compare_kernel(const float *actual,
                                        const float *expected,
                                        std::uint32_t *result) {
  if (result[0] != 0u) {
    result[2] = float_bits_from_ordered(result[2]);
    const std::uint32_t first = result[1];
    result[3] = __float_as_uint(actual[first]);
    result[4] = __float_as_uint(expected[first]);
  }
}

inline Status launch_compare(const Args *args, std::uint64_t oracle_output_f32,
                             std::uint64_t compare_result_i32) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess ||
      !detail::aligned(compare_result_i32, alignof(std::uint32_t))) {
    return validation == Status::kSuccess ? Status::kInvalidArgument
                                          : validation;
  }

  const std::uint64_t rows_and_heads =
      static_cast<std::uint64_t>(args->rows) * args->heads;
  if (args->head_dim != 0u && rows_and_heads > UINT64_MAX / args->head_dim) {
    return Status::kInvalidArgument;
  }
  const std::uint64_t output_count = rows_and_heads * args->head_dim;
  if (!detail::aligned(oracle_output_f32, alignof(float))) {
    return Status::kInvalidArgument;
  }

  Args oracle_args = *args;
  oracle_args.output_f32 = oracle_output_f32;
  const Status oracle_status = launch(&oracle_args);
  if (oracle_status != Status::kSuccess) {
    return oracle_status;
  }

  constexpr std::uint32_t threads = 256u;
  const std::uint64_t blocks = (output_count + threads - 1u) / threads;
  if (blocks == 0u || blocks > UINT32_MAX) {
    return Status::kLaunchFailed;
  }
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));
  auto *result = detail::device_pointer<std::uint32_t>(compare_result_i32);
  if (cudaMemsetAsync(result, 0, kCompareResultWords * sizeof(std::uint32_t),
                      stream) != cudaSuccess ||
      cudaMemsetAsync(result + 1u, 0xff, sizeof(std::uint32_t), stream) !=
          cudaSuccess) {
    return Status::kLaunchFailed;
  }
  const auto *actual = detail::device_pointer<const float>(args->output_f32);
  const auto *expected = detail::device_pointer<const float>(oracle_output_f32);
  compare_kernel<<<static_cast<std::uint32_t>(blocks), threads, 0u, stream>>>(
      actual, expected, output_count, result);
  finalize_compare_kernel<<<1u, 1u, 0u, stream>>>(actual, expected, result);
  return cudaPeekAtLastError() == cudaSuccess ? Status::kSuccess
                                              : Status::kLaunchFailed;
}

} // namespace test_oracle
#endif

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }

  const std::uint64_t selections =
      static_cast<std::uint64_t>(args->rows) * args->selected_width;
  const std::uint64_t stage2_head_groups =
      static_cast<std::uint64_t>(args->rows) * (kHeads / kStage2HeadTile);
  if (selections == 0u || selections > UINT32_MAX || stage2_head_groups == 0u ||
      stage2_head_groups > UINT32_MAX) {
    return Status::kLaunchFailed;
  }

  const detail::Binding binding{
      detail::device_pointer<const float>(args->query_f32),
      detail::device_pointer<const float>(args->first_plane_f32),
      detail::device_pointer<const float>(args->second_plane_f32),
      detail::device_pointer<const std::int32_t>(args->block_slots_i32),
      detail::device_pointer<const std::int32_t>(args->block_offsets_i32),
      detail::device_pointer<const std::int32_t>(args->sequence_kv_lens_i32),
      detail::device_pointer<const std::int32_t>(args->row_sequence_ids_i32),
      detail::device_pointer<const std::int32_t>(args->row_kv_lens_i32),
      detail::device_pointer<const std::int32_t>(args->selected_indices_i32),
      detail::device_pointer<const std::int32_t>(args->selectors_i32),
      detail::device_pointer<const float>(args->attention_sink_f32),
      detail::device_pointer<std::uint64_t>(args->source_addresses_u64),
      detail::device_pointer<float>(args->raw_scores_f32),
      detail::device_pointer<float>(args->output_f32),
      detail::device_pointer<std::int32_t>(args->status_i32),
  };
  const auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));

  detail::stage1_scores_kernel<<<static_cast<std::uint32_t>(selections),
                                 kStage1Threads, 0u, stream>>>(*args, binding);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  const std::size_t stage2_shared_bytes =
      sizeof(float) * kStage2HeadTile *
      (static_cast<std::size_t>(args->selected_width) + 1u);
  detail::stage2_head_group_kernel<<<
      static_cast<std::uint32_t>(stage2_head_groups), kStage2HeadGroupThreads,
      stage2_shared_bytes, stream>>>(*args, binding);
  return cudaPeekAtLastError() == cudaSuccess ? Status::kSuccess
                                              : Status::kLaunchFailed;
}

} // namespace
  // ferrule::cuda::cutlass::capabilities::hybrid_mla_attention::schedules::preserve_f32_order

#endif // FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_SCHEDULES_PRESERVE_F32_ORDER_CUH_
