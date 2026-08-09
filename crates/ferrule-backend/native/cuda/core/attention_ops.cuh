#pragma once

#include "core/abi.h"
#include "core/device.cuh"
#include "core/validation.cuh"
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <stdint.h>

namespace ferrule::cuda::core {

__device__ inline void
compressor_source(bool prefill, bool overlap, uint32_t group, uint32_t row,
                  uint32_t ratio, uint32_t head_dim, uint32_t dimension,
                  bool *valid, uint32_t *token, uint32_t *source_dimension,
                  uint32_t *ape_dimension) {
  *valid = true;
  if (overlap) {
    if (row < ratio) {
      if (prefill && group == 0) {
        *valid = false;
      }
      *token = (group == 0 ? 0 : group - 1) * ratio + row;
      *source_dimension = dimension;
      *ape_dimension = dimension;
    } else {
      *token = group * ratio + row - ratio;
      *source_dimension = head_dim + dimension;
      *ape_dimension = head_dim + dimension;
    }
  } else {
    *token = group * ratio + row;
    *source_dimension = dimension;
    *ape_dimension = dimension;
  }
}

__global__ void compressor_kernel(FerruleCoreCompressorArgs args) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  float *kv_state = pointer<float>(args.kv_state);
  float *score_state = pointer<float>(args.score_state);
  if (args.kind == FERRULE_CORE_COMPRESSOR_RESET) {
    if (index < args.state_elements) {
      kv_state[index] = 0.0f;
      score_state[index] = -CUDART_INF_F;
    }
    return;
  }
  if (args.kind == FERRULE_CORE_COMPRESSOR_APPEND) {
    if (index >= args.output_dim || args.ratio == 0) {
      return;
    }
    const uint32_t local = args.position % args.ratio;
    const uint32_t state_row = args.overlap != 0 ? args.ratio + local : local;
    const uint64_t target =
        static_cast<uint64_t>(state_row) * args.output_dim + index;
    kv_state[target] = const_pointer<float>(args.kv_input)[index];
    score_state[target] =
        const_pointer<float>(args.score_input)[index] +
        const_pointer<float>(
            args.ape)[static_cast<uint64_t>(local) * args.output_dim + index];
    return;
  }
  if (args.kind == FERRULE_CORE_COMPRESSOR_SEED) {
    if (index >= args.state_elements || args.ratio == 0 ||
        args.output_dim == 0) {
      return;
    }
    const uint32_t state_row = index / args.output_dim;
    const uint32_t dimension = index % args.output_dim;
    const uint32_t remainder = args.tokens % args.ratio;
    const uint32_t cutoff = args.tokens - remainder;
    int64_t source_token = -1;
    uint32_t ape_row = 0;
    if (args.overlap != 0 && cutoff >= args.ratio && state_row < args.ratio) {
      source_token = cutoff - args.ratio + state_row;
      ape_row = state_row;
    } else {
      const uint32_t state_offset = args.overlap != 0 ? args.ratio : 0;
      if (state_row >= state_offset && state_row < state_offset + remainder) {
        source_token = cutoff + state_row - state_offset;
        ape_row = state_row - state_offset;
      }
    }
    if (source_token < 0) {
      kv_state[index] = 0.0f;
      score_state[index] = -CUDART_INF_F;
    } else {
      const uint64_t source =
          static_cast<uint64_t>(source_token) * args.output_dim + dimension;
      kv_state[index] = const_pointer<float>(args.kv_input)[source];
      score_state[index] =
          const_pointer<float>(args.score_input)[source] +
          const_pointer<float>(
              args.ape)[static_cast<uint64_t>(ape_row) * args.output_dim +
                        dimension];
    }
    return;
  }
  const bool prefill = args.kind == FERRULE_CORE_COMPRESSOR_PREFILL;
  const uint32_t output_count =
      prefill ? args.groups * args.head_dim : args.head_dim;
  if (index >= output_count || args.ratio == 0) {
    return;
  }
  const uint32_t group = prefill ? index / args.head_dim : 0;
  const uint32_t dimension = index % args.head_dim;
  const uint32_t rows = args.overlap != 0 ? args.ratio * 2 : args.ratio;
  const float *kv = prefill ? const_pointer<float>(args.kv_input)
                            : const_pointer<float>(args.kv_state);
  const float *scores = prefill ? const_pointer<float>(args.score_input)
                                : const_pointer<float>(args.score_state);
  float maximum = -CUDART_INF_F;
  for (uint32_t row = 0; row < rows; ++row) {
    bool source_valid;
    uint32_t token, source_dimension, ape_dimension;
    compressor_source(prefill, args.overlap != 0, group, row, args.ratio,
                      args.head_dim, dimension, &source_valid, &token,
                      &source_dimension, &ape_dimension);
    if (source_valid) {
      float score = scores[static_cast<uint64_t>(prefill ? token : row) *
                               args.output_dim +
                           source_dimension];
      if (prefill) {
        score += const_pointer<float>(
            args.ape)[static_cast<uint64_t>(row % args.ratio) *
                          args.output_dim +
                      ape_dimension];
      }
      maximum = fmaxf(maximum, score);
    }
  }
  float denominator = 0.0f;
  float result = 0.0f;
  for (uint32_t row = 0; row < rows; ++row) {
    bool source_valid;
    uint32_t token, source_dimension, ape_dimension;
    compressor_source(prefill, args.overlap != 0, group, row, args.ratio,
                      args.head_dim, dimension, &source_valid, &token,
                      &source_dimension, &ape_dimension);
    if (source_valid) {
      const uint64_t source =
          static_cast<uint64_t>(prefill ? token : row) * args.output_dim +
          source_dimension;
      float score = scores[source];
      if (prefill) {
        score += const_pointer<float>(
            args.ape)[static_cast<uint64_t>(row % args.ratio) *
                          args.output_dim +
                      ape_dimension];
      }
      const float weight = expf(score - maximum);
      denominator += weight;
      result += weight * kv[source];
    }
  }
  pointer<float>(args.output)[index] = bf16_round(
      denominator > 0.0f && isfinite(denominator) ? result / denominator
                                                  : 0.0f);
}

constexpr uint32_t kMaxTopK = 512;
constexpr uint32_t kIndexerFusedQuery = 1u << 0;
constexpr uint32_t kIndexerRowMetadata = 1u << 1;
constexpr uint32_t kIndexerDirectCompressed = 1u << 2;

__device__ inline void indexer_insert(float score, int32_t candidate,
                                      float *best_scores, int32_t *best_indices,
                                      uint32_t take) {
  insert_topk(score, candidate, best_scores, best_indices, take);
}

__device__ inline void transform_index_query(const FerruleCoreIndexerArgs &args,
                                             uint32_t query_row, uint32_t head,
                                             uint32_t position, float *query) {
  const uint64_t query_base =
      (static_cast<uint64_t>(query_row) * args.heads + head) * args.head_dim;
  for (uint32_t dimension = 0; dimension < args.head_dim; ++dimension) {
    query[dimension] = const_pointer<float>(args.query)[query_base + dimension];
  }
  if (args.rope_dim != 0 && args.rope_dim <= args.head_dim &&
      (args.rope_dim & 1u) == 0) {
    const uint32_t tail_start = args.head_dim - args.rope_dim;
    const uint32_t pairs = args.rope_dim / 2;
    for (uint32_t pair = 0; pair < pairs; ++pair) {
      const uint32_t dimension = tail_start + pair * 2;
      const float first = query[dimension];
      const float second = query[dimension + 1];
      const float cosine = const_pointer<float>(
          args.cosine)[static_cast<uint64_t>(position) * pairs + pair];
      const float sine = const_pointer<float>(
          args.sine)[static_cast<uint64_t>(position) * pairs + pair];
      query[dimension] = first * cosine - second * sine;
      query[dimension + 1] = first * sine + second * cosine;
    }
  }
  for (uint32_t dimension = 0; dimension < args.head_dim; ++dimension) {
    query[dimension] = bf16_round(query[dimension]);
  }
  for (uint32_t span = 1; span < args.head_dim; span *= 2) {
    const uint32_t step = span * 2;
    for (uint32_t start = 0; start < args.head_dim; start += step) {
      for (uint32_t offset = 0; offset < span; ++offset) {
        const uint32_t left = start + offset;
        const uint32_t right = left + span;
        const float first = query[left];
        const float second = query[right];
        query[left] = first + second;
        query[right] = first - second;
      }
    }
  }
  const float hadamard_scale = rsqrtf(static_cast<float>(args.head_dim));
  for (uint32_t block = 0; block < args.head_dim / 32; ++block) {
    float amax = 6.0f * exp2f(-126.0f);
    for (uint32_t element = 0; element < 32; ++element) {
      const uint32_t dimension = block * 32 + element;
      query[dimension] = bf16_round(query[dimension] * hadamard_scale);
      amax = fmaxf(amax, fabsf(query[dimension]));
    }
    const float scale = e8m0_scale(e8m0_byte(amax, 6.0f));
    for (uint32_t element = 0; element < 32; ++element) {
      const uint32_t dimension = block * 32 + element;
      query[dimension] =
          fp4_value(
              fp4_nibble(clamp_value(query[dimension] / scale, -6.0f, 6.0f))) *
          scale;
    }
  }
}

__global__ void indexer_kernel(FerruleCoreIndexerArgs args) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= args.rows || args.topk > kMaxTopK) {
    return;
  }
  const bool prefill = args.prefill != 0;
  const bool row_decode = !prefill && (args.flags & kIndexerRowMetadata) != 0;
  const bool fused_query = (args.flags & kIndexerFusedQuery) != 0;
  const bool direct_compressed =
      row_decode && (args.flags & kIndexerDirectCompressed) != 0;
  const uint32_t window_columns =
      prefill ? args.window_columns : args.window_size;
  const uint32_t output_columns = window_columns + args.topk;
  int32_t *indices = pointer<int32_t>(args.indices);
  int32_t *selectors = pointer<int32_t>(args.selectors);
  const uint64_t output_base = static_cast<uint64_t>(row) * output_columns;
  for (uint32_t column = 0; column < output_columns; ++column) {
    indices[output_base + column] = -1;
    if (selectors != nullptr) {
      selectors[output_base + column] = -1;
    }
  }
  if (fused_query &&
      (args.head_dim == 0 || args.head_dim > 256 ||
       (args.head_dim & (args.head_dim - 1)) != 0 ||
       (args.head_dim % 32) != 0 || args.rope_dim > args.head_dim ||
       (args.rope_dim & 1u) != 0)) {
    return;
  }
  uint32_t position = args.position;
  uint32_t window_len = args.window_len;
  uint32_t compressed_len = args.compressed_len;
  uint32_t sequence = 0;
  bool metadata_valid = true;
  if (row_decode) {
    const int32_t sequence_value =
        const_pointer<int32_t>(args.row_sequence_ids)[row];
    const int32_t position_value = const_pointer<int32_t>(args.positions)[row];
    const int32_t window_value = const_pointer<int32_t>(args.window_lens)[row];
    const int32_t compressed_value =
        const_pointer<int32_t>(args.compressed_lens)[row];
    metadata_valid = sequence_value >= 0 && position_value >= 0 &&
                     window_value >= 0 && compressed_value >= 0;
    sequence = metadata_valid ? static_cast<uint32_t>(sequence_value) : 0;
    position = metadata_valid ? static_cast<uint32_t>(position_value) : 0;
    window_len = metadata_valid ? static_cast<uint32_t>(window_value) : 0;
    compressed_len =
        metadata_valid ? static_cast<uint32_t>(compressed_value) : 0;
  }
  if (prefill) {
    const uint32_t absolute_position = args.start_position + row;
    const uint32_t first = absolute_position + 1 > args.window_size
                               ? absolute_position + 1 - args.window_size
                               : 0;
    for (uint32_t column = 0; column < window_columns; ++column) {
      const uint32_t candidate = first + column;
      indices[output_base + column] =
          candidate <= absolute_position ? static_cast<int32_t>(candidate) : -1;
    }
    compressed_len = args.compress_ratio == 0
                         ? 0
                         : min((absolute_position + 1) / args.compress_ratio,
                               compressed_len);
  } else if (row_decode) {
    for (uint32_t column = 0; column < args.window_size; ++column) {
      if (metadata_valid && window_len <= args.window_size &&
          window_len <= position + 1 && column < window_len) {
        indices[output_base + column] =
            static_cast<int32_t>(position + 1 - window_len + column);
        selectors[output_base + column] = 0;
      }
    }
  } else {
    for (uint32_t column = 0; column < args.window_size; ++column) {
      if (window_len < args.window_size) {
        indices[output_base + column] =
            column < window_len ? static_cast<int32_t>(column) : -1;
      } else {
        indices[output_base + column] = static_cast<int32_t>(
            (position % args.window_size + 1 + column) % args.window_size);
      }
    }
  }
  if (args.topk == 0 || !metadata_valid ||
      (row_decode && compressed_len != 0 && args.compress_ratio == 0)) {
    return;
  }
  float best_scores[kMaxTopK];
  int32_t best_indices[kMaxTopK];
  for (uint32_t rank = 0; rank < args.topk; ++rank) {
    best_scores[rank] = -CUDART_INF_F;
    best_indices[rank] = -1;
  }
  const float *query = const_pointer<float>(args.query);
  const float *weights = const_pointer<float>(args.weights);
  const float *plane = const_pointer<float>(args.plane);
  const uint32_t query_row = prefill || row_decode ? row : 0;
  for (uint32_t candidate = 0; candidate < compressed_len; ++candidate) {
    uint32_t logical_candidate = candidate;
    if (row_decode) {
      const uint64_t boundary =
          (static_cast<uint64_t>(candidate) + 1u) * args.compress_ratio - 1u;
      if (boundary > INT32_MAX) {
        continue;
      }
      logical_candidate = static_cast<uint32_t>(boundary);
    }
    float score = direct_compressed ? 0.0f : -CUDART_INF_F;
    if (!direct_compressed) {
      const uint64_t plane_base = paged_row_offset(
          args.plane_elements, const_pointer<int32_t>(args.block_slots),
          const_pointer<int32_t>(args.block_offsets), sequence,
          logical_candidate, args.page_tokens, args.head_dim, args.layer_index,
          args.layer_count);
      if (plane_base != UINT64_MAX) {
        score = 0.0f;
        for (uint32_t head = 0; head < args.heads; ++head) {
          float dot = 0.0f;
          if (fused_query) {
            float transformed[256];
            const uint32_t query_position =
                prefill ? args.start_position + row : position;
            transform_index_query(args, query_row, head, query_position,
                                  transformed);
            for (uint32_t dimension = 0; dimension < args.head_dim;
                 ++dimension) {
              dot += transformed[dimension] * plane[plane_base + dimension];
            }
          } else {
            const uint64_t query_base =
                (static_cast<uint64_t>(query_row) * args.heads + head) *
                args.head_dim;
            for (uint32_t dimension = 0; dimension < args.head_dim;
                 ++dimension) {
              dot +=
                  query[query_base + dimension] * plane[plane_base + dimension];
            }
          }
          const float dot_bf16 = bf16_round(dot);
          const float weight_bf16 = bf16_round(
              bf16_round(weights[static_cast<uint64_t>(query_row) * args.heads +
                                 head]) *
              args.weight_scale);
          score += bf16_round(fmaxf(dot_bf16, 0.0f) * weight_bf16);
        }
      }
    }
    score = bf16_round(score);
    if (isfinite(score)) {
      indexer_insert(score, static_cast<int32_t>(logical_candidate),
                     best_scores, best_indices, args.topk);
    }
  }
  for (uint32_t rank = 0; rank < args.topk; ++rank) {
    const bool found = best_indices[rank] >= 0 && isfinite(best_scores[rank]);
    const uint64_t output = output_base + window_columns + rank;
    indices[output] =
        !found ? -1
        : row_decode
            ? best_indices[rank]
            : static_cast<int32_t>(args.value_offset + best_indices[rank]);
    if (row_decode) {
      selectors[output] = found ? 1 : -1;
    }
  }
}

__global__ void mla_kernel(FerruleCoreMlaArgs args) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= args.output_size || args.rank == 0) {
    return;
  }
  float sum_square = 0.0f;
  for (uint32_t rank = 0; rank < args.rank; ++rank) {
    float latent = 0.0f;
    for (uint32_t column = 0; column < args.hidden_size; ++column) {
      latent +=
          const_pointer<float>(args.input)[column] *
          const_pointer<float>(
              args.weight_a)[static_cast<uint64_t>(rank) * args.hidden_size +
                             column];
    }
    sum_square += latent * latent;
  }
  const float inverse_rms = rsqrtf(sum_square / args.rank + args.epsilon);
  float result = 0.0f;
  for (uint32_t rank = 0; rank < args.rank; ++rank) {
    float latent = 0.0f;
    for (uint32_t column = 0; column < args.hidden_size; ++column) {
      latent +=
          const_pointer<float>(args.input)[column] *
          const_pointer<float>(
              args.weight_a)[static_cast<uint64_t>(rank) * args.hidden_size +
                             column];
    }
    result += latent * inverse_rms *
              const_pointer<float>(args.norm_weight)[rank] *
              const_pointer<float>(
                  args.weight_b)[static_cast<uint64_t>(row) * args.rank + rank];
  }
  pointer<float>(args.output)[row] = result;
}

constexpr int32_t kTransformerMetadataError = 1;
constexpr int32_t kTransformerAddressError = 2;


__device__ inline void transformer_error(const FerruleCoreTransformerArgs &args,
                                         int32_t error) {
  atomicCAS(pointer<int32_t>(args.status_i32), 0, error);
}

__device__ inline bool
transformer_row_metadata(const FerruleCoreTransformerArgs &args, uint32_t row,
                         uint32_t *sequence, uint32_t *position,
                         uint32_t *kv_len) {
  const int32_t sequence_value =
      const_pointer<int32_t>(args.row_sequence_ids_i32)[row];
  const int32_t position_value =
      const_pointer<int32_t>(args.row_positions_i32)[row];
  const int32_t kv_len_value =
      const_pointer<int32_t>(args.row_kv_lens_i32)[row];
  if (sequence_value < 0 ||
      static_cast<uint32_t>(sequence_value) >= args.sequences ||
      position_value < 0 || kv_len_value < 0) {
    transformer_error(args, kTransformerMetadataError);
    return false;
  }
  *sequence = static_cast<uint32_t>(sequence_value);
  *position = static_cast<uint32_t>(position_value);
  *kv_len = static_cast<uint32_t>(kv_len_value);
  return true;
}

__device__ inline bool
transformer_cache_offset(const FerruleCoreTransformerArgs &args,
                         uint32_t sequence, uint32_t logical_token,
                         uint32_t kv_head, uint32_t dimension, bool value_cache,
                         uint64_t *result) {
  const int32_t block_start_value =
      const_pointer<int32_t>(args.block_offsets_i32)[sequence];
  const int32_t block_end_value =
      const_pointer<int32_t>(args.block_offsets_i32)[sequence + 1];
  if (block_start_value < 0 || block_end_value < block_start_value ||
      static_cast<uint64_t>(block_end_value) > args.block_slots_count) {
    transformer_error(args, kTransformerMetadataError);
    return false;
  }
  const uint64_t block_entry = static_cast<uint64_t>(block_start_value) +
                               logical_token / args.page_tokens;
  if (block_entry >= static_cast<uint64_t>(block_end_value) ||
      block_entry >= args.block_slots_count) {
    transformer_error(args, kTransformerMetadataError);
    return false;
  }
  const int32_t slot_value =
      const_pointer<int32_t>(args.block_slots_i32)[block_entry];
  if (slot_value < 0) {
    transformer_error(args, kTransformerMetadataError);
    return false;
  }

  const uint64_t slot_stride =
      value_cache ? args.value_slot_stride_bytes : args.key_slot_stride_bytes;
  const uint64_t layer_stride =
      value_cache ? args.value_layer_stride_bytes : args.key_layer_stride_bytes;
  const uint64_t token_stride =
      value_cache ? args.value_token_stride_bytes : args.key_token_stride_bytes;
  const uint64_t head_stride =
      value_cache ? args.value_head_stride_bytes : args.key_head_stride_bytes;
  const uint64_t capacity =
      value_cache ? args.value_cache_bytes : args.key_cache_bytes;

  uint64_t offset;
  uint64_t term;
  if (!checked_mul_u64(static_cast<uint32_t>(slot_value), slot_stride,
                       &offset) ||
      !checked_mul_u64(args.layer_index, layer_stride, &term) ||
      !checked_add_u64(offset, term, &offset) ||
      !checked_mul_u64(logical_token % args.page_tokens, token_stride, &term) ||
      !checked_add_u64(offset, term, &offset) ||
      !checked_mul_u64(kv_head, head_stride, &term) ||
      !checked_add_u64(offset, term, &offset) ||
      !checked_mul_u64(dimension, sizeof(uint16_t), &term) ||
      !checked_add_u64(offset, term, &offset) || offset > capacity ||
      sizeof(uint16_t) > capacity - offset) {
    transformer_error(args, kTransformerAddressError);
    return false;
  }
  *result = offset;
  return true;
}

__global__ void transformer_kv_append_kernel(FerruleCoreTransformerArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t values =
      static_cast<uint64_t>(args.rows) * args.kv_heads * args.head_dim;
  if (index >= values) {
    return;
  }
  const uint32_t dimension = static_cast<uint32_t>(index % args.head_dim);
  const uint64_t head_row = index / args.head_dim;
  const uint32_t kv_head = static_cast<uint32_t>(head_row % args.kv_heads);
  const uint32_t row = static_cast<uint32_t>(head_row / args.kv_heads);
  uint32_t sequence;
  uint32_t position;
  uint32_t kv_len;
  if (!transformer_row_metadata(args, row, &sequence, &position, &kv_len)) {
    return;
  }
  (void)kv_len;

  uint64_t key_offset;
  uint64_t value_offset;
  if (!transformer_cache_offset(args, sequence, position, kv_head, dimension,
                                false, &key_offset) ||
      !transformer_cache_offset(args, sequence, position, kv_head, dimension,
                                true, &value_offset)) {
    return;
  }
  const uint64_t key_input_offset =
      static_cast<uint64_t>(row) * args.append_key_row_stride_bytes +
      static_cast<uint64_t>(kv_head) * args.append_key_head_stride_bytes +
      static_cast<uint64_t>(dimension) * sizeof(uint16_t);
  const uint64_t value_input_offset =
      static_cast<uint64_t>(row) * args.append_value_row_stride_bytes +
      static_cast<uint64_t>(kv_head) * args.append_value_head_stride_bytes +
      static_cast<uint64_t>(dimension) * sizeof(uint16_t);
  pointer<uint16_t>(args.key_cache_bf16 + key_offset)[0] =
      const_pointer<uint16_t>(args.append_key_bf16 + key_input_offset)[0];
  pointer<uint16_t>(args.value_cache_bf16 + value_offset)[0] =
      const_pointer<uint16_t>(args.append_value_bf16 + value_input_offset)[0];
}

__global__ void transformer_causal_gqa_kernel(FerruleCoreTransformerArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t values =
      static_cast<uint64_t>(args.rows) * args.q_heads * args.head_dim;
  if (index >= values) {
    return;
  }
  const uint32_t dimension = static_cast<uint32_t>(index % args.head_dim);
  const uint64_t head_row = index / args.head_dim;
  const uint32_t q_head = static_cast<uint32_t>(head_row % args.q_heads);
  const uint32_t row = static_cast<uint32_t>(head_row / args.q_heads);
  const uint32_t queries_per_kv_head = args.q_heads / args.kv_heads;
  const uint32_t kv_head = q_head / queries_per_kv_head;
  uint32_t sequence;
  uint32_t position;
  uint32_t kv_len;
  if (!transformer_row_metadata(args, row, &sequence, &position, &kv_len)) {
    return;
  }
  const uint32_t effective_kv_len =
      args.kind == FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA
          ? max(kv_len, position + 1)
          : kv_len;
  const uint32_t visible = min(effective_kv_len, position + 1);
  const uint64_t query_head =
      static_cast<uint64_t>(row) * args.query_row_stride_bytes +
      static_cast<uint64_t>(q_head) * args.query_head_stride_bytes;

  float maximum = -CUDART_INF_F;
  float denominator = 0.0f;
  float accumulator = 0.0f;
  for (uint32_t token = 0; token < visible; ++token) {
    float score = 0.0f;
    for (uint32_t dot_dimension = 0; dot_dimension < args.head_dim;
         ++dot_dimension) {
      uint64_t key_offset;
      if (!transformer_cache_offset(args, sequence, token, kv_head,
                                    dot_dimension, false, &key_offset)) {
        return;
      }
      const float query = const_pointer<float>(
          args.query_f32 + query_head +
          static_cast<uint64_t>(dot_dimension) * sizeof(float))[0];
      const float key = bf16_value(
          const_pointer<uint16_t>(args.key_cache_bf16 + key_offset)[0]);
      score += query * key;
    }
    score *= args.softmax_scale;
    uint64_t value_offset;
    if (!transformer_cache_offset(args, sequence, token, kv_head, dimension,
                                  true, &value_offset)) {
      return;
    }
    const float value = bf16_value(
        const_pointer<uint16_t>(args.value_cache_bf16 + value_offset)[0]);
    if (denominator == 0.0f || score > maximum) {
      const float rescale = denominator == 0.0f ? 0.0f : expf(maximum - score);
      accumulator = accumulator * rescale + value;
      denominator = denominator * rescale + 1.0f;
      maximum = score;
    } else {
      const float weight = score == maximum ? 1.0f : expf(score - maximum);
      accumulator += weight * value;
      denominator += weight;
    }
  }
  const uint64_t output_offset =
      static_cast<uint64_t>(row) * args.output_row_stride_bytes +
      static_cast<uint64_t>(q_head) * args.output_head_stride_bytes +
      static_cast<uint64_t>(dimension) * sizeof(float);
  pointer<float>(args.output_f32 + output_offset)[0] =
      denominator > 0.0f ? accumulator / denominator : 0.0f;
}

inline bool valid_transformer(const FerruleCoreTransformerArgs *args) {
  if (!valid(args) || args->rows == 0 || args->sequences == 0 ||
      args->q_heads == 0 || args->kv_heads == 0 || args->head_dim == 0 ||
      args->q_heads % args->kv_heads != 0 || args->page_tokens == 0 ||
      args->layer_count == 0 || args->layer_index >= args->layer_count ||
      !isfinite(args->softmax_scale) || args->softmax_scale <= 0.0f ||
      !transformer_cache_strides(
          args->key_cache_bf16, args->key_cache_bytes,
          args->key_slot_stride_bytes, args->key_layer_stride_bytes,
          args->key_token_stride_bytes, args->key_head_stride_bytes,
          args->layer_count, args->page_tokens, args->kv_heads,
          args->head_dim) ||
      !transformer_cache_strides(
          args->value_cache_bf16, args->value_cache_bytes,
          args->value_slot_stride_bytes, args->value_layer_stride_bytes,
          args->value_token_stride_bytes, args->value_head_stride_bytes,
          args->layer_count, args->page_tokens, args->kv_heads,
          args->head_dim) ||
      args->block_slots_i32 == 0 || args->block_slots_i32 % 4 != 0 ||
      args->block_slots_count == 0 || args->block_offsets_i32 == 0 ||
      args->block_offsets_i32 % 4 != 0 ||
      args->block_offsets_count < static_cast<uint64_t>(args->sequences) + 1 ||
      args->row_sequence_ids_i32 == 0 || args->row_sequence_ids_i32 % 4 != 0 ||
      args->row_sequence_ids_count < args->rows ||
      args->row_positions_i32 == 0 || args->row_positions_i32 % 4 != 0 ||
      args->row_positions_count < args->rows || args->row_kv_lens_i32 == 0 ||
      args->row_kv_lens_i32 % 4 != 0 || args->row_kv_lens_count < args->rows ||
      args->status_i32 == 0 || args->status_i32 % 4 != 0 ||
      args->status_count == 0) {
    return false;
  }
  const bool append =
      args->kind == FERRULE_CORE_TRANSFORMER_PAGED_BF16_KV_APPEND ||
      args->kind == FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA;
  const bool attention =
      args->kind == FERRULE_CORE_TRANSFORMER_PAGED_BF16_CAUSAL_GQA ||
      args->kind == FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA;
  if (!append && !attention) {
    return false;
  }
  if (append &&
      (args->append_key_bf16 == 0 || args->append_value_bf16 == 0 ||
       !transformer_range(
           args->append_key_bf16, args->append_key_bytes, args->rows,
           args->append_key_row_stride_bytes, args->kv_heads,
           args->append_key_head_stride_bytes,
           static_cast<uint64_t>(args->head_dim) * sizeof(uint16_t), 2) ||
       !transformer_range(
           args->append_value_bf16, args->append_value_bytes, args->rows,
           args->append_value_row_stride_bytes, args->kv_heads,
           args->append_value_head_stride_bytes,
           static_cast<uint64_t>(args->head_dim) * sizeof(uint16_t), 2))) {
    return false;
  }
  if (attention &&
      (args->query_f32 == 0 || args->output_f32 == 0 ||
       !transformer_range(args->query_f32, args->query_bytes, args->rows,
                          args->query_row_stride_bytes, args->q_heads,
                          args->query_head_stride_bytes,
                          static_cast<uint64_t>(args->head_dim) * sizeof(float),
                          4) ||
       !transformer_range(args->output_f32, args->output_bytes, args->rows,
                          args->output_row_stride_bytes, args->q_heads,
                          args->output_head_stride_bytes,
                          static_cast<uint64_t>(args->head_dim) * sizeof(float),
                          4))) {
    return false;
  }
  return true;
}

} // namespace ferrule::cuda::core
