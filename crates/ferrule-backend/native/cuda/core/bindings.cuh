#pragma once

#include "core/abi.h"
#include "core/device.cuh"
#include "core/validation.cuh"
#include "core/dense_ops.cuh"
#include "core/tensor_ops.cuh"
#include "core/sequence_ops.cuh"
#include "core/attention_ops.cuh"
#include "core/moe_ops.cuh"
#include <cuda_runtime_api.h>
#include <stdint.h>

extern "C" int32_t
ferrule_core_linear_launch(const FerruleCoreLinearArgs *args) {
  if (!ferrule::cuda::core::valid(args) || args->batch == 0 || args->n == 0) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::linear_kernel<<<ferrule::cuda::core::blocks_for(static_cast<uint64_t>(args->batch) * args->n),
                  ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(*args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t
ferrule_core_dual_linear_launch(const FerruleCoreDualLinearArgs *args) {
  if (!ferrule::cuda::core::valid(args) || args->kind != FERRULE_CORE_LINEAR_BF16_BYTES) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::dual_linear_kernel<<<ferrule::cuda::core::blocks_for(static_cast<uint64_t>(args->first_n) +
                                  args->second_n),
                       ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(*args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t
ferrule_core_grouped_linear_launch(const FerruleCoreGroupedLinearArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::grouped_linear_kernel<<<ferrule::cuda::core::blocks_for(static_cast<uint64_t>(args->rows) *
                                     args->output_dim),
                          ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(*args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t
ferrule_core_quantize_launch(const FerruleCoreQuantizeArgs *args) {
  if (!ferrule::cuda::core::valid(args) || args->block_size == 0) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  uint64_t count = args->kind == FERRULE_CORE_QUANTIZE_HADAMARD_FP4_IN_PLACE
                       ? args->value_len / args->row_width
                       : args->value_len / args->block_size;
  if (args->kind == FERRULE_CORE_QUANTIZE_FP8_NON_ROPE_IN_PLACE) {
    const uint32_t non_rope = args->row_width - args->rope_dim;
    const uint32_t width =
        non_rope % args->block_size == 0 ? args->block_size : non_rope;
    count = static_cast<uint64_t>(args->value_len / args->row_width) *
            ((non_rope + width - 1) / width);
  }
  ferrule::cuda::core::quantize_kernel<<<ferrule::cuda::core::blocks_for(count), ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(
      *args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t ferrule_core_data_launch(const FerruleCoreDataArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::data_kernel<<<ferrule::cuda::core::blocks_for(args->count), ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(
      *args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t ferrule_core_rows_launch(const FerruleCoreRowsArgs *args) {
  if (!ferrule::cuda::core::valid_rows(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  const uint64_t values =
      static_cast<uint64_t>(args->rows) * args->heads * args->dimensions;
  ferrule::cuda::core::rows_f32_to_bf16_rne_kernel<<<ferrule::cuda::core::blocks_for(values), ferrule::cuda::core::kBlock, 0,
                                ferrule::cuda::core::stream(args->stream)>>>(*args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t
ferrule_core_embedding_launch(const FerruleCoreEmbeddingArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::embedding_kernel<<<ferrule::cuda::core::blocks_for(static_cast<uint64_t>(args->rows) * args->hc *
                                args->hidden),
                     ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(*args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t ferrule_core_norm_launch(const FerruleCoreNormArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::norm_kernel<<<args->rows, ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(*args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t ferrule_core_rope_launch(const FerruleCoreRopeArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::rope_kernel<<<ferrule::cuda::core::blocks_for(args->pair_count), ferrule::cuda::core::kBlock, 0,
                ferrule::cuda::core::stream(args->stream)>>>(*args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t
ferrule_core_router_launch(const FerruleCoreRouterArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::router_kernel<<<ferrule::cuda::core::blocks_for(args->rows), ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(
      *args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t
ferrule_core_compressor_launch(const FerruleCoreCompressorArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  uint64_t count = args->state_elements;
  if (args->kind == FERRULE_CORE_COMPRESSOR_APPEND)
    count = args->output_dim;
  if (args->kind == FERRULE_CORE_COMPRESSOR_PREFILL)
    count = static_cast<uint64_t>(args->groups) * args->head_dim;
  if (args->kind == FERRULE_CORE_COMPRESSOR_SOFTMAX)
    count = args->head_dim;
  ferrule::cuda::core::compressor_kernel<<<ferrule::cuda::core::blocks_for(count), ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(
      *args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t
ferrule_core_indexer_launch(const FerruleCoreIndexerArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::indexer_kernel<<<ferrule::cuda::core::blocks_for(args->rows), ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(
      *args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t
ferrule_core_expert_table_launch(const FerruleCoreExpertTableArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  if (args->kind == FERRULE_CORE_EXPERT_GATHER_DISPATCH) {
    cudaError_t status =
        cudaMemsetAsync(ferrule::cuda::core::pointer<int32_t>(args->dispatch_error), 0,
                        sizeof(int32_t), ferrule::cuda::core::stream(args->stream));
    if (status != cudaSuccess)
      return static_cast<int32_t>(status);
  }
  uint64_t count = args->route_count;
  if (args->kind == FERRULE_CORE_EXPERT_INSTALL ||
      args->kind == FERRULE_CORE_EXPERT_EVICT)
    count = 1;
  if (args->kind == FERRULE_CORE_EXPERT_INITIALIZE_RESOLVE)
    count = args->miss_capacity + args->route_capacity + 2;
  ferrule::cuda::core::expert_table_kernel<<<ferrule::cuda::core::blocks_for(count), ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(
      *args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t ferrule_core_expert_group_route_plan_launch(
    const FerruleCoreExpertGroupRoutePlanArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  uint64_t count = args->route_count;
  if (args->kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_INIT_INVOCATION)
    count = max(args->output_elements, args->route_count);
  if (args->kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_INIT_PLAN)
    count = max(max(args->slot_capacity + 1, args->route_capacity), 4u);
  if (args->kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_COMPACT)
    count = 1;
  ferrule::cuda::core::expert_group_route_plan_kernel<<<ferrule::cuda::core::blocks_for(count), ferrule::cuda::core::kBlock, 0,
                                   ferrule::cuda::core::stream(args->stream)>>>(*args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t ferrule_core_moe_launch(const FerruleCoreMoeArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  uint64_t count = 0;
  switch (args->kind) {
  case FERRULE_CORE_MOE_WEIGHTED_SWIGLU_F32:
    count =
        static_cast<uint64_t>(args->experts) * args->batch_columns * args->n;
    break;
  case FERRULE_CORE_MOE_GATHER_BF16_ROWS:
    count = static_cast<uint64_t>(args->batch_columns) * args->hidden;
    break;
  case FERRULE_CORE_MOE_WEIGHTED_SCATTER_ADD_BF16_ROWS:
    count = static_cast<uint64_t>(args->tokens) * args->hidden;
    break;
  case FERRULE_CORE_MOE_REDUCE_EXPERT:
  case FERRULE_CORE_MOE_REDUCE_SPLIT_EXPERT:
    count = static_cast<uint64_t>(args->batch_columns) * args->hidden;
    break;
  case FERRULE_CORE_MOE_REDUCE_ROUTES:
  case FERRULE_CORE_MOE_REDUCE_EXPERT_GROUP_ROUTES:
    count = static_cast<uint64_t>(args->tokens) * args->hidden;
    break;
  default:
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::moe_kernel<<<ferrule::cuda::core::blocks_for(count), ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(*args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t ferrule_core_hc_launch(const FerruleCoreHcArgs *args) {
  if (!ferrule::cuda::core::valid(args) || args->hc > 16 || args->mix > 128) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  if (args->kind == FERRULE_CORE_HC_POST) {
    const uint64_t elements =
        static_cast<uint64_t>(args->tokens) * args->hc * args->hidden_size;
    ferrule::cuda::core::hc_post_kernel<<<ferrule::cuda::core::blocks_for(elements), ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(
        *args);
  } else if (args->kind == FERRULE_CORE_HC_MEAN_SCATTER) {
    const uint64_t elements =
        static_cast<uint64_t>(args->tokens) * args->hidden_size;
    ferrule::cuda::core::hc_mean_scatter_kernel<<<ferrule::cuda::core::blocks_for(elements), ferrule::cuda::core::kBlock, 0,
                             ferrule::cuda::core::stream(args->stream)>>>(*args);
  } else {
    ferrule::cuda::core::hc_kernel<<<ferrule::cuda::core::blocks_for(args->tokens), ferrule::cuda::core::kBlock, 0, ferrule::cuda::core::stream(args->stream)>>>(
        *args);
  }
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t ferrule_core_mla_launch(const FerruleCoreMlaArgs *args) {
  if (!ferrule::cuda::core::valid(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  ferrule::cuda::core::mla_kernel<<<ferrule::cuda::core::blocks_for(args->output_size), ferrule::cuda::core::kBlock, 0,
               ferrule::cuda::core::stream(args->stream)>>>(*args);
  return ferrule::cuda::core::launch_status();
}

extern "C" int32_t
ferrule_core_transformer_launch(const FerruleCoreTransformerArgs *args) {
  if (!ferrule::cuda::core::valid_transformer(args)) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }
  cudaError_t status = cudaMemsetAsync(ferrule::cuda::core::pointer<int32_t>(args->status_i32), 0,
                                       sizeof(int32_t), ferrule::cuda::core::stream(args->stream));
  if (status != cudaSuccess) {
    return static_cast<int32_t>(status);
  }
  if (args->kind == FERRULE_CORE_TRANSFORMER_PAGED_BF16_KV_APPEND ||
      args->kind == FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA) {
    const uint64_t append_values =
        static_cast<uint64_t>(args->rows) * args->kv_heads * args->head_dim;
    ferrule::cuda::core::transformer_kv_append_kernel<<<ferrule::cuda::core::blocks_for(append_values), ferrule::cuda::core::kBlock, 0,
                                   ferrule::cuda::core::stream(args->stream)>>>(*args);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      return static_cast<int32_t>(status);
    }
  }
  if (args->kind == FERRULE_CORE_TRANSFORMER_PAGED_BF16_CAUSAL_GQA ||
      args->kind == FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA) {
    const uint64_t output_values =
        static_cast<uint64_t>(args->rows) * args->q_heads * args->head_dim;
    ferrule::cuda::core::transformer_causal_gqa_kernel<<<ferrule::cuda::core::blocks_for(output_values), ferrule::cuda::core::kBlock, 0,
                                    ferrule::cuda::core::stream(args->stream)>>>(*args);
  }
  return ferrule::cuda::core::launch_status();
}
