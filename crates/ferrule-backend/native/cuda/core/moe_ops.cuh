#pragma once

#include "core/abi.h"
#include "core/device.cuh"
#include "core/validation.cuh"
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <stdint.h>

namespace ferrule::cuda::core {

__global__ void router_kernel(FerruleCoreRouterArgs args) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= args.rows || args.k == 0) {
    return;
  }
  const float *logits = const_pointer<float>(args.logits);
  if (args.kind == FERRULE_CORE_ROUTER_SELECTED_SOFTMAX_TOPK) {
    const uint64_t output_base = static_cast<uint64_t>(row) * args.k;
    int32_t *selected_indices = pointer<int32_t>(args.indices) + output_base;
    float *selected_values = pointer<float>(args.weights) + output_base;
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      selected_indices[rank] = -1;
      selected_values[rank] = -CUDART_INF_F;
    }
    for (uint32_t expert = 0; expert < args.columns; ++expert) {
      const float value =
          logits[static_cast<uint64_t>(row) * args.columns + expert];
      if (isnan(value)) {
        continue;
      }
      uint32_t position = args.k;
      while (position > 0) {
        const uint32_t previous = position - 1;
        if (value < selected_values[previous] ||
            (value == selected_values[previous] &&
             selected_indices[previous] >= 0 &&
             expert >= static_cast<uint32_t>(selected_indices[previous]))) {
          break;
        }
        --position;
      }
      if (position < args.k) {
        for (uint32_t move = args.k - 1; move > position; --move) {
          selected_values[move] = selected_values[move - 1];
          selected_indices[move] = selected_indices[move - 1];
        }
        selected_values[position] = value;
        selected_indices[position] = static_cast<int32_t>(expert);
      }
    }

    uint32_t valid = 0;
    uint32_t positive_infinities = 0;
    float maximum = -CUDART_INF_F;
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      if (selected_indices[rank] >= 0) {
        ++valid;
        maximum = fmaxf(maximum, selected_values[rank]);
        positive_infinities += selected_values[rank] == CUDART_INF_F ? 1 : 0;
      }
    }
    float denominator = 0.0f;
    if (isfinite(maximum)) {
      for (uint32_t rank = 0; rank < args.k; ++rank) {
        if (selected_indices[rank] >= 0) {
          selected_values[rank] = expf(selected_values[rank] - maximum);
          denominator += selected_values[rank];
        }
      }
    }
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      float probability = 0.0f;
      if (selected_indices[rank] >= 0) {
        if (positive_infinities != 0) {
          probability = selected_values[rank] == CUDART_INF_F
                            ? 1.0f / positive_infinities
                            : 0.0f;
        } else if (isfinite(maximum) && denominator > 0.0f &&
                   isfinite(denominator)) {
          probability = selected_values[rank] / denominator;
        } else if (maximum == -CUDART_INF_F && valid != 0) {
          probability = 1.0f / valid;
        }
      }
      selected_values[rank] = probability * args.route_scale;
    }
    return;
  }
  if (args.k > 64) {
    return;
  }
  float best_values[64];
  int32_t best_indices[64];
  for (uint32_t index = 0; index < args.k; ++index) {
    best_values[index] = -CUDART_INF_F;
    best_indices[index] = -1;
  }
  if (args.kind == FERRULE_CORE_ROUTER_TOPK) {
    const float *bias = const_pointer<float>(args.bias);
    for (uint32_t expert = 0; expert < args.columns; ++expert) {
      const float score = sqrtf(fmaxf(
          softplus(logits[static_cast<uint64_t>(row) * args.columns + expert]),
          0.0f));
      const float selection =
          score + ((args.flags & 1u) != 0 ? bias[expert] : 0.0f);
      insert_topk(selection, static_cast<int32_t>(expert), best_values,
                  best_indices, args.k);
    }
    float sum = 0.0f;
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      const int32_t expert = best_indices[rank];
      const float score =
          expert >= 0
              ? sqrtf(fmaxf(
                    softplus(logits[static_cast<uint64_t>(row) * args.columns +
                                    expert]),
                    0.0f))
              : 0.0f;
      best_values[rank] = score;
      sum += score;
    }
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      pointer<int32_t>(
          args.indices)[static_cast<uint64_t>(row) * args.k + rank] =
          best_indices[rank];
      pointer<float>(args.weights)[static_cast<uint64_t>(row) * args.k + rank] =
          sum > 0.0f && isfinite(sum)
              ? best_values[rank] / sum * args.route_scale
              : 0.0f;
    }
  } else if (args.kind == FERRULE_CORE_ROUTER_HASH) {
    const int32_t token = const_pointer<int32_t>(args.token_ids)[row];
    if (token < 0 || static_cast<uint32_t>(token) >= args.hash_rows) {
      return;
    }
    float sum = 0.0f;
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      const int32_t expert = const_pointer<int32_t>(
          args.hash_table)[static_cast<uint64_t>(token) * args.hash_columns +
                           rank];
      if (expert < 0 || static_cast<uint32_t>(expert) >= args.columns) {
        return;
      }
      const float score = sqrtf(fmaxf(
          softplus(logits[static_cast<uint64_t>(row) * args.columns + expert]),
          0.0f));
      best_indices[rank] = expert;
      best_values[rank] = score;
      sum += score;
    }
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      pointer<int32_t>(
          args.indices)[static_cast<uint64_t>(row) * args.k + rank] =
          best_indices[rank];
      pointer<float>(args.weights)[static_cast<uint64_t>(row) * args.k + rank] =
          sum > 0.0f && isfinite(sum)
              ? best_values[rank] / sum * args.route_scale
              : 0.0f;
    }
  } else {
    for (uint32_t column = 0; column < args.columns; ++column) {
      insert_topk(logits[static_cast<uint64_t>(row) * args.columns + column],
                  static_cast<int32_t>(column), best_values, best_indices,
                  args.k);
    }
    for (uint32_t rank = 0; rank < args.k; ++rank) {
      const uint64_t output = static_cast<uint64_t>(row) * args.k + rank;
      if (args.kind == FERRULE_CORE_VOCAB_TOPK_F32_INDEX) {
        pointer<float>(args.indices)[output] =
            static_cast<float>(best_indices[rank]);
      } else {
        pointer<int32_t>(args.indices)[output] = best_indices[rank];
      }
      pointer<float>(args.weights)[output] = best_values[rank];
    }
  }
}

__global__ void expert_table_kernel(FerruleCoreExpertTableArgs args) {
  const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
  if (args.kind == FERRULE_CORE_EXPERT_INSTALL ||
      args.kind == FERRULE_CORE_EXPERT_EVICT) {
    if (route != 0) {
      return;
    }
    uint64_t *gate = pointer<uint64_t>(args.gate_weights);
    uint64_t *gate_scale = pointer<uint64_t>(args.gate_scales);
    uint64_t *up = pointer<uint64_t>(args.up_weights);
    uint64_t *up_scale = pointer<uint64_t>(args.up_scales);
    uint64_t *down = pointer<uint64_t>(args.down_weights);
    uint64_t *down_scale = pointer<uint64_t>(args.down_scales);
    if (args.kind == FERRULE_CORE_EXPERT_INSTALL) {
      gate[args.slot] = args.gate_weight_value;
      gate_scale[args.slot] = args.gate_scale_value;
      up[args.slot] = args.up_weight_value;
      up_scale[args.slot] = args.up_scale_value;
      down[args.slot] = args.down_weight_value;
      down_scale[args.slot] = args.down_scale_value;
      pointer<int32_t>(args.slot_generations)[args.slot] = args.generation;
      pointer<int32_t>(args.expert_to_slot)[args.expert] =
          static_cast<int32_t>(args.slot);
      pointer<int32_t>(args.expert_generations)[args.expert] = args.generation;
    } else {
      pointer<int32_t>(args.expert_to_slot)[args.expert] = -1;
      pointer<int32_t>(args.expert_generations)[args.expert] = 0;
      gate[args.slot] = gate_scale[args.slot] = up[args.slot] =
          up_scale[args.slot] = 0;
      down[args.slot] = down_scale[args.slot] = 0;
      pointer<int32_t>(args.slot_generations)[args.slot] = args.generation;
    }
    return;
  }
  if (args.kind == FERRULE_CORE_EXPERT_INITIALIZE_RESOLVE) {
    const uint32_t count = args.miss_capacity + args.route_capacity + 2;
    if (route < count) {
      pointer<int32_t>(args.miss_control)[route] = route < 2 ? 0 : -1;
    }
    return;
  }
  if (route >= args.route_count) {
    return;
  }
  if (args.kind == FERRULE_CORE_EXPERT_RESOLVE) {
    const int32_t expert = const_pointer<int32_t>(args.expert_ids)[route];
    int32_t slot = -1;
    int32_t generation = 0;
    if (expert >= 0 && static_cast<uint32_t>(expert) < args.expert_capacity) {
      const int32_t mapped =
          const_pointer<int32_t>(args.expert_to_slot)[expert];
      const int32_t expert_generation =
          const_pointer<int32_t>(args.expert_generations)[expert];
      if (mapped >= 0 && static_cast<uint32_t>(mapped) < args.slot_capacity &&
          expert_generation > 0 &&
          const_pointer<int32_t>(args.slot_generations)[mapped] ==
              expert_generation) {
        slot = mapped;
        generation = expert_generation;
      }
    }
    pointer<int32_t>(args.route_slots)[route] = slot;
    pointer<int32_t>(args.route_generations)[route] = generation;
    pointer<int32_t>(args.miss_markers)[route] = slot < 0 ? 1 : 0;
    pointer<int32_t>(args.miss_control)[2 + args.miss_capacity + route] =
        expert;
    if (slot < 0) {
      const int32_t miss = atomicAdd(pointer<int32_t>(args.miss_control), 1);
      if (miss >= 0 && static_cast<uint32_t>(miss) < args.miss_capacity) {
        pointer<int32_t>(args.miss_control)[2 + miss] = expert;
      } else {
        atomicOr(pointer<int32_t>(args.miss_control) + 1, 1);
      }
    }
    return;
  }
  const int32_t slot = const_pointer<int32_t>(args.route_slots)[route];
  const int32_t generation =
      const_pointer<int32_t>(args.route_generations)[route];
  const bool active =
      const_pointer<int32_t>(args.active_markers)[route] == args.active_value;
  const bool current =
      slot >= 0 && static_cast<uint32_t>(slot) < args.slot_capacity &&
      generation > 0 &&
      const_pointer<int32_t>(args.slot_generations)[slot] == generation;
  const uint64_t gate =
      current ? const_pointer<uint64_t>(args.gate_weights)[slot] : 0;
  const uint64_t gate_scale =
      current ? const_pointer<uint64_t>(args.gate_scales)[slot] : 0;
  const uint64_t up =
      current ? const_pointer<uint64_t>(args.up_weights)[slot] : 0;
  const uint64_t up_scale =
      current ? const_pointer<uint64_t>(args.up_scales)[slot] : 0;
  const uint64_t down =
      current ? const_pointer<uint64_t>(args.down_weights)[slot] : 0;
  const uint64_t down_scale =
      current ? const_pointer<uint64_t>(args.down_scales)[slot] : 0;
  const bool good = active && current && gate != 0 && gate_scale != 0 &&
                    up != 0 && up_scale != 0 && down != 0 && down_scale != 0;
  if (active && !good) {
    atomicOr(pointer<int32_t>(args.dispatch_error), 1);
  }
  pointer<uint64_t>(args.output_gate_weights)[route] = gate;
  pointer<uint64_t>(args.output_gate_scales)[route] = gate_scale;
  pointer<uint64_t>(args.output_up_weights)[route] = up;
  pointer<uint64_t>(args.output_up_scales)[route] = up_scale;
  pointer<uint64_t>(args.output_down_weights)[route] = down;
  pointer<uint64_t>(args.output_down_scales)[route] = down_scale;
  pointer<float>(args.output_route_weights)[route] =
      good ? const_pointer<float>(args.router_weights)[route] : 0.0f;
  pointer<int32_t>(args.expert_ids)[route] =
      good ? static_cast<int32_t>(route) : -1;
}

__global__ void
expert_group_route_plan_kernel(FerruleCoreExpertGroupRoutePlanArgs args) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (args.kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_INIT_INVOCATION) {
    if (index < args.output_elements) {
      pointer<float>(args.route_output)[index] = 0.0f;
    }
    if (index < args.route_count) {
      pointer<int32_t>(args.route_written)[index] = 0;
    }
    if (index == 0) {
      pointer<int32_t>(args.route_error)[0] = 0;
    }
  } else if (args.kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_INIT_PLAN) {
    if (index < args.slot_capacity) {
      pointer<int32_t>(args.slot_counts)[index] = 0;
      pointer<int32_t>(args.slot_route_offsets)[index] = -1;
      pointer<int32_t>(args.slot_cursors)[index] = 0;
      pointer<int32_t>(args.active_expert_slots)[index] = -1;
      pointer<int32_t>(args.active_group_generations)[index] = 0;
      pointer<int32_t>(args.expert_route_counts)[index] = 0;
    }
    if (index <= args.slot_capacity) {
      pointer<int32_t>(args.expert_route_indptr)[index] = 0;
    }
    if (index < args.route_capacity) {
      pointer<int32_t>(args.route_token_indices)[index] = -1;
      pointer<int32_t>(args.route_indices)[index] = -1;
      pointer<float>(args.route_weights)[index] = 0.0f;
    }
    if (index < 4) {
      pointer<int32_t>(args.host_scalars)[index] = 0;
    }
  } else if (args.kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_COUNT) {
    if (index >= args.route_count) {
      return;
    }
    const int32_t slot = const_pointer<int32_t>(args.route_slots)[index];
    const int32_t generation =
        const_pointer<int32_t>(args.route_generations)[index];
    if (slot >= 0 && static_cast<uint32_t>(slot) < args.slot_capacity &&
        generation > 0 &&
        const_pointer<int32_t>(args.slot_generations)[slot] == generation) {
      atomicAdd(pointer<int32_t>(args.slot_counts) + slot, 1);
    }
  } else if (args.kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_COMPACT) {
    if (index != 0) {
      return;
    }
    uint32_t active_group_count = 0;
    uint32_t small_group_count = 0;
    uint32_t max_group_rows = 0;
    uint32_t total_routed_rows = 0;
    for (uint32_t slot = 0; slot < args.slot_capacity; ++slot) {
      const uint32_t count = static_cast<uint32_t>(
          max(0, const_pointer<int32_t>(args.slot_counts)[slot]));
      if (count != 0 && count < args.small_group_row_limit) {
        ++small_group_count;
      }
    }
    active_group_count = small_group_count;
    for (uint32_t pass = 0; pass < 2; ++pass) {
      uint32_t group = pass == 0 ? 0 : small_group_count;
      for (uint32_t slot = 0; slot < args.slot_capacity; ++slot) {
        const uint32_t count = static_cast<uint32_t>(
            max(0, const_pointer<int32_t>(args.slot_counts)[slot]));
        if (count == 0 || (count < args.small_group_row_limit) != (pass == 0)) {
          continue;
        }
        if (group >= args.slot_capacity ||
            total_routed_rows + count > args.route_capacity) {
          pointer<int32_t>(args.route_error)[0] = 1;
          return;
        }
        pointer<int32_t>(args.active_expert_slots)[group] =
            static_cast<int32_t>(slot);
        pointer<int32_t>(args.active_group_generations)[group] =
            const_pointer<int32_t>(args.slot_generations)[slot];
        pointer<int32_t>(args.expert_route_indptr)[group] =
            static_cast<int32_t>(total_routed_rows);
        pointer<int32_t>(args.expert_route_counts)[group] =
            static_cast<int32_t>(count);
        pointer<int32_t>(args.slot_route_offsets)[slot] =
            static_cast<int32_t>(total_routed_rows);
        max_group_rows = max(max_group_rows, count);
        total_routed_rows += count;
        ++group;
      }
      if (pass == 1) {
        active_group_count = group;
      }
    }
    pointer<int32_t>(args.expert_route_indptr)[active_group_count] =
        static_cast<int32_t>(total_routed_rows);
    pointer<int32_t>(args.host_scalars)[0] =
        static_cast<int32_t>(active_group_count);
    pointer<int32_t>(args.host_scalars)[1] =
        static_cast<int32_t>(small_group_count);
    pointer<int32_t>(args.host_scalars)[2] =
        static_cast<int32_t>(max_group_rows);
    pointer<int32_t>(args.host_scalars)[3] =
        static_cast<int32_t>(total_routed_rows);
  } else if (args.kind == FERRULE_CORE_EXPERT_GROUP_ROUTE_SCATTER) {
    if (index >= args.route_count || args.routes_per_token == 0) {
      return;
    }
    const int32_t slot = const_pointer<int32_t>(args.route_slots)[index];
    const int32_t generation =
        const_pointer<int32_t>(args.route_generations)[index];
    if (slot < 0 || static_cast<uint32_t>(slot) >= args.slot_capacity ||
        generation <= 0 ||
        const_pointer<int32_t>(args.slot_generations)[slot] != generation) {
      return;
    }
    const int32_t position =
        atomicAdd(pointer<int32_t>(args.slot_cursors) + slot, 1);
    const int32_t offset =
        const_pointer<int32_t>(args.slot_route_offsets)[slot];
    if (position < 0 || offset < 0 ||
        static_cast<uint32_t>(offset + position) >= args.route_capacity) {
      atomicOr(pointer<int32_t>(args.route_error), 1);
      return;
    }
    const uint32_t metadata = static_cast<uint32_t>(offset + position);
    pointer<int32_t>(args.route_token_indices)[metadata] =
        static_cast<int32_t>(index / args.routes_per_token);
    pointer<int32_t>(args.route_indices)[metadata] =
        static_cast<int32_t>(index);
    pointer<float>(args.route_weights)[metadata] =
        const_pointer<float>(args.router_weights)[index];
  }
}

__global__ void moe_kernel(FerruleCoreMoeArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (args.kind == FERRULE_CORE_MOE_GATHER_BF16_ROWS) {
    const uint64_t total =
        static_cast<uint64_t>(args.batch_columns) * args.hidden;
    if (index >= total) {
      return;
    }
    const uint32_t route = static_cast<uint32_t>(index / args.hidden);
    const uint32_t column = static_cast<uint32_t>(index % args.hidden);
    const int32_t source_row = const_pointer<int32_t>(args.route_slots)[route];
    pointer<float>(args.hidden_values)[index] =
        source_row >= 0 && static_cast<uint32_t>(source_row) < args.tokens
            ? bf16_value(const_pointer<uint16_t>(
                  args.input)[static_cast<uint64_t>(source_row) * args.hidden +
                              column])
            : 0.0f;
  } else if (args.kind == FERRULE_CORE_MOE_WEIGHTED_SCATTER_ADD_BF16_ROWS) {
    const uint64_t total = static_cast<uint64_t>(args.tokens) * args.hidden;
    if (index >= total) {
      return;
    }
    const uint32_t target_row = static_cast<uint32_t>(index / args.hidden);
    const uint32_t column = static_cast<uint32_t>(index % args.hidden);
    float result = pointer<float>(args.output)[index];
    for (uint32_t route = 0; route < args.batch_columns; ++route) {
      if (const_pointer<int32_t>(args.route_slots)[route] ==
          static_cast<int32_t>(target_row)) {
        const float value = bf16_round(const_pointer<float>(
            args.input)[static_cast<uint64_t>(route) * args.hidden + column]);
        result += value * const_pointer<float>(args.route_weights)[route];
      }
    }
    pointer<float>(args.output)[index] = result;
  } else if (args.kind == FERRULE_CORE_MOE_WEIGHTED_SWIGLU_F32) {
    const uint64_t total =
        static_cast<uint64_t>(args.experts) * args.batch_columns * args.n;
    if (index >= total) {
      return;
    }
    float gate = const_pointer<float>(args.gate)[index];
    float up = const_pointer<float>(args.up)[index];
    if (args.swiglu_limit > 0.0f) {
      gate = fminf(gate, args.swiglu_limit);
      up = clamp_value(up, -args.swiglu_limit, args.swiglu_limit);
    }
    const uint32_t route = static_cast<uint32_t>(index / args.n);
    const bool quantize = args.route_weights != 0;
    const float route_weight =
        quantize ? const_pointer<float>(args.route_weights)[route]
                 : args.route_weight;
    const float value = gate * sigmoid(gate) * up * route_weight;
    pointer<float>(args.hidden_values)[index] =
        quantize ? fp8_quantized(value) : value;
  } else if (args.kind == FERRULE_CORE_MOE_REDUCE_EXPERT) {
    const uint64_t total =
        static_cast<uint64_t>(args.batch_columns) * args.hidden;
    if (index >= total) {
      return;
    }
    const uint32_t column = static_cast<uint32_t>(index / args.hidden);
    const uint32_t row = static_cast<uint32_t>(index % args.hidden);
    const uint64_t output = args.output_offset + index;
    float result = pointer<float>(args.output)[output];
    for (uint32_t rank = 0; rank < args.routes_per_token; ++rank) {
      const int32_t slot = const_pointer<int32_t>(
          args.route_slots)[column * args.routes_per_token + rank];
      if (slot >= 0 && static_cast<uint32_t>(slot) < args.experts) {
        result += const_pointer<float>(args.expert_output)
            [(static_cast<uint64_t>(slot) * args.batch_columns + column) *
                 args.hidden +
             row];
      }
    }
    pointer<float>(args.output)[output] = result;
  } else if (args.kind == FERRULE_CORE_MOE_REDUCE_SPLIT_EXPERT) {
    const uint64_t total =
        static_cast<uint64_t>(args.batch_columns) * args.hidden;
    if (index >= total) {
      return;
    }
    const uint32_t column = static_cast<uint32_t>(index / args.hidden);
    const uint32_t row = static_cast<uint32_t>(index % args.hidden);
    const uint64_t output = args.output_offset + index;
    float result = pointer<float>(args.output)[output];
    for (uint32_t rank = 0; rank < args.routes_per_token; ++rank) {
      const uint32_t route = column * args.routes_per_token + rank;
      const bool miss = const_pointer<int32_t>(args.miss_markers)[route] != 0;
      const int32_t slot = const_pointer<int32_t>(
          miss ? args.materialized_route_slots : args.route_slots)[route];
      if (slot >= 0 && static_cast<uint32_t>(slot) < args.experts) {
        result += const_pointer<float>(miss ? args.materialized_output
                                            : args.resident_output)
            [(static_cast<uint64_t>(slot) * args.batch_columns + column) *
                 args.hidden +
             row];
      }
    }
    pointer<float>(args.output)[output] = bf16_round(result);
  } else if (args.kind == FERRULE_CORE_MOE_REDUCE_ROUTES ||
             args.kind == FERRULE_CORE_MOE_REDUCE_EXPERT_GROUP_ROUTES) {
    const uint64_t total = static_cast<uint64_t>(args.tokens) * args.hidden;
    if (index >= total) {
      return;
    }
    const uint32_t token = static_cast<uint32_t>(index / args.hidden);
    const uint32_t row = static_cast<uint32_t>(index % args.hidden);
    bool complete = true;
    if (args.kind == FERRULE_CORE_MOE_REDUCE_EXPERT_GROUP_ROUTES) {
      complete = const_pointer<int32_t>(args.route_error)[0] == 0;
      for (uint32_t rank = 0; rank < args.routes_per_token; ++rank) {
        complete &=
            const_pointer<int32_t>(
                args.route_written)[token * args.routes_per_token + rank] != 0;
      }
    }
    if (!complete) {
      pointer<float>(args.output)[index] = CUDART_NAN_F;
      return;
    }
    float result = pointer<float>(args.output)[index];
    for (uint32_t rank = 0; rank < args.routes_per_token; ++rank) {
      const uint32_t route = token * args.routes_per_token + rank;
      result += const_pointer<float>(
          args.route_output)[static_cast<uint64_t>(route) * args.hidden + row];
    }
    pointer<float>(args.output)[index] = bf16_round(result);
  }
}

} // namespace ferrule::cuda::core
