#pragma once

#include "core/abi.h"
#include "core/device.cuh"
#include "core/validation.cuh"
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <stdint.h>

namespace ferrule::cuda::core {

__global__ void quantize_kernel(FerruleCoreQuantizeArgs args) {
  const uint32_t block_index = blockIdx.x * blockDim.x + threadIdx.x;
  float *values = pointer<float>(args.values);
  if (args.kind == FERRULE_CORE_QUANTIZE_HADAMARD_FP4_IN_PLACE) {
    if (args.row_width == 0 || block_index >= args.value_len / args.row_width) {
      return;
    }
    const uint32_t base = block_index * args.row_width;
    for (uint32_t index = 0; index < args.row_width; ++index) {
      values[base + index] = bf16_round(values[base + index]);
    }
    for (uint32_t span = 1; span < args.row_width; span *= 2) {
      for (uint32_t start = 0; start < args.row_width; start += span * 2) {
        for (uint32_t offset = 0; offset < span; ++offset) {
          const float a = values[base + start + offset];
          const float b = values[base + start + offset + span];
          values[base + start + offset] = a + b;
          values[base + start + offset + span] = a - b;
        }
      }
    }
    const float hadamard_scale = rsqrtf(static_cast<float>(args.row_width));
    for (uint32_t index = 0; index < args.row_width; ++index) {
      values[base + index] = bf16_round(values[base + index] * hadamard_scale);
    }
    for (uint32_t block = 0; block < args.row_width / args.block_size;
         ++block) {
      const uint32_t begin = base + block * args.block_size;
      float amax = 6.0f * exp2f(-126.0f);
      for (uint32_t index = 0; index < args.block_size; ++index) {
        amax = fmaxf(amax, fabsf(values[begin + index]));
      }
      const float scale = e8m0_scale(e8m0_byte(amax, 6.0f));
      for (uint32_t index = 0; index < args.block_size; ++index) {
        values[begin + index] =
            fp4_value(fp4_nibble(values[begin + index] / scale)) * scale;
      }
    }
    return;
  }

  uint32_t effective_width = args.row_width;
  uint32_t row = 0;
  uint32_t block = 0;
  uint32_t begin = 0;
  uint32_t end = 0;
  if (args.kind == FERRULE_CORE_QUANTIZE_FP8_NON_ROPE_IN_PLACE) {
    const uint32_t non_rope = args.row_width - args.rope_dim;
    effective_width =
        non_rope % args.block_size == 0 ? args.block_size : non_rope;
    const uint32_t blocks_per_row =
        (non_rope + effective_width - 1) / effective_width;
    row = block_index / blocks_per_row;
    block = block_index % blocks_per_row;
    if (row >= args.value_len / args.row_width) {
      return;
    }
    begin = row * args.row_width + block * effective_width;
    end = min(row * args.row_width + non_rope, begin + effective_width);
  } else {
    if (args.block_size == 0 ||
        block_index >= args.value_len / args.block_size) {
      return;
    }
    const uint32_t blocks_per_row = args.row_width / args.block_size;
    row = block_index / blocks_per_row;
    block = block_index % blocks_per_row;
    begin = row * args.row_width + block * args.block_size;
    end = begin + args.block_size;
  }

  const float *input = const_pointer<float>(args.values);
  float amax = args.kind == FERRULE_CORE_QUANTIZE_FP4_PACKED ? 0.0f : 1e-4f;
  for (uint32_t index = begin; index < end; ++index) {
    amax = fmaxf(amax, fabsf(input[args.value_offset + index]));
  }
  const bool fp4 = args.kind == FERRULE_CORE_QUANTIZE_FP4_PACKED;
  const uint8_t scale_byte = e8m0_byte(amax, fp4 ? 6.0f : 448.0f);
  const float scale = e8m0_scale(scale_byte);
  if (args.kind == FERRULE_CORE_QUANTIZE_FP8_IN_PLACE ||
      args.kind == FERRULE_CORE_QUANTIZE_FP8_NON_ROPE_IN_PLACE) {
    for (uint32_t index = begin; index < end; ++index) {
      values[index] = bf16_round(
          fp8_quantized(clamp_value(values[index] / scale, -448.0f, 448.0f)) *
          scale);
    }
  } else if (args.kind == FERRULE_CORE_QUANTIZE_FP8_PACKED) {
    uint8_t *packed = pointer<uint8_t>(args.packed);
    pointer<uint8_t>(args.scales)[block_index] = scale_byte;
    for (uint32_t index = begin; index < end; ++index) {
      packed[index] =
          fp8_byte(clamp_value(input[index] / scale, -448.0f, 448.0f));
    }
  } else if (fp4) {
    uint8_t *packed = pointer<uint8_t>(args.packed);
    pointer<uint8_t>(args.scales)[block_index] = scale_byte;
    const uint32_t packed_row = row * (args.row_width / 2);
    const uint32_t packed_block = block * (args.block_size / 2);
    for (uint32_t index = 0; index < args.block_size; index += 2) {
      const uint8_t low =
          fp4_nibble(input[args.value_offset + begin + index] / scale);
      const uint8_t high =
          fp4_nibble(input[args.value_offset + begin + index + 1] / scale);
      packed[packed_row + packed_block + index / 2] = low | (high << 4);
    }
  }
}

__global__ void data_kernel(FerruleCoreDataArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= args.count) {
    return;
  }
  if (args.kind == FERRULE_CORE_DATA_FILL_I32) {
    const int64_t value =
        static_cast<int64_t>(static_cast<int32_t>(args.start)) + index;
    pointer<int32_t>(args.output0)[index] =
        value > INT32_MAX ? INT32_MAX : static_cast<int32_t>(value);
  } else if (args.kind == FERRULE_CORE_DATA_PACK_I32_F32) {
    const uint32_t pair = static_cast<uint32_t>(index / 2);
    pointer<int32_t>(args.output0)[index] =
        (index & 1u) == 0 ? const_pointer<int32_t>(args.input0)[pair]
                          : static_cast<int32_t>(__float_as_uint(
                                const_pointer<float>(args.input1)[pair]));
  } else if (args.kind == FERRULE_CORE_DATA_PACK_PROPOSAL_HEAD) {
    int32_t value;
    if (index == 0) {
      value = const_pointer<int32_t>(args.input0)[0];
    } else if (index <= args.rows) {
      value = const_pointer<int32_t>(args.input1)[index];
    } else {
      value = static_cast<int32_t>(__float_as_uint(
          const_pointer<float>(args.input2)[index - args.rows - 1]));
    }
    pointer<int32_t>(args.output0)[index] = value;
  } else if (args.kind == FERRULE_CORE_DATA_FILL_PAGED_WINDOW) {
    pointer<int32_t>(args.output0)[index] =
        index < args.value0 ? static_cast<int32_t>(args.start + index) : -1;
  } else if (args.kind == FERRULE_CORE_DATA_FILL_DECODE_TOPK) {
    int32_t value = -1;
    if (index < args.width) {
      if (index < args.value0) {
        value = args.value0 < args.width
                    ? static_cast<int32_t>(index)
                    : static_cast<int32_t>(
                          (args.start % args.width + 1 + index) % args.width);
      }
    } else if (index - args.width < args.value1) {
      value = static_cast<int32_t>(index);
    }
    pointer<int32_t>(args.output0)[index] = value;
  } else if (args.kind == FERRULE_CORE_DATA_FILL_RECENT_ROWS) {
    const uint32_t row = static_cast<uint32_t>(index / args.width);
    const uint32_t column = static_cast<uint32_t>(index % args.width);
    const int32_t visible = const_pointer<int32_t>(args.input0)[row];
    int32_t value = -1;
    if (visible > 0) {
      const uint32_t valid = min(static_cast<uint32_t>(visible), args.width);
      if (column < valid) {
        value = visible <= static_cast<int32_t>(args.width)
                    ? static_cast<int32_t>(column)
                    : visible - static_cast<int32_t>(valid) +
                          static_cast<int32_t>(column);
      }
    }
    pointer<int32_t>(args.output0)[index] = value;
  } else if (args.kind == FERRULE_CORE_DATA_COPY_F32) {
    pointer<float>(args.output0)[args.offset + index] =
        const_pointer<float>(args.input0)[index];
  } else if (args.kind == FERRULE_CORE_DATA_GATHER_F32_ROWS) {
    const uint32_t row = static_cast<uint32_t>(index / args.width);
    const uint32_t column = static_cast<uint32_t>(index % args.width);
    const int32_t source_row = const_pointer<int32_t>(args.input1)[row];
    if (source_row >= 0) {
      pointer<float>(args.output0)[index] = const_pointer<float>(
          args.input0)[static_cast<uint64_t>(source_row) * args.width + column];
    }
  } else if (args.kind == FERRULE_CORE_DATA_SCATTER_ADD_F32_ROWS) {
    const uint32_t row = static_cast<uint32_t>(index / args.width);
    const uint32_t column = static_cast<uint32_t>(index % args.width);
    const int32_t target_row = const_pointer<int32_t>(args.input1)[row];
    if (target_row >= 0) {
      atomicAdd(pointer<float>(args.output0) +
                    static_cast<uint64_t>(target_row) * args.width + column,
                const_pointer<float>(args.input0)[index]);
    }
  } else if (args.kind == FERRULE_CORE_DATA_SAXPY) {
    pointer<float>(args.output0)[index] +=
        args.scale * const_pointer<float>(args.input0)[index];
  } else if (args.kind == FERRULE_CORE_DATA_CONVERT_COMBINED_RING) {
    if (args.width == 0 || args.value1 == 0) {
      return;
    }
    const uint32_t row = static_cast<uint32_t>(index / args.width);
    const uint64_t position = static_cast<uint64_t>(args.start) +
                              static_cast<uint64_t>(row) * args.value0;
    const uint64_t maximum =
        min(position + 1, static_cast<uint64_t>(args.value1));
    uint64_t valid_window = maximum;
    if ((args.flags & 1u) != 0) {
      const int32_t explicit_len = const_pointer<int32_t>(args.input1)[row];
      valid_window =
          explicit_len >= 0 && static_cast<uint64_t>(explicit_len) <= maximum
              ? static_cast<uint64_t>(explicit_len)
              : 0;
    }
    const int32_t combined = const_pointer<int32_t>(args.input0)[index];
    int32_t logical = -1;
    int32_t selector = -1;
    if (combined >= 0) {
      const uint64_t candidate = static_cast<uint32_t>(combined);
      if (candidate >= args.value1) {
        const uint64_t compressed = candidate - args.value1;
        if (compressed <= INT32_MAX) {
          logical = static_cast<int32_t>(compressed);
          selector = 1;
        }
      } else if (position < args.value1) {
        if (candidate < valid_window && candidate <= position) {
          logical = static_cast<int32_t>(candidate);
          selector = 0;
        }
      } else {
        const uint64_t age =
            (position % args.value1 + args.value1 - candidate) % args.value1;
        if (age < valid_window && position - age <= INT32_MAX) {
          logical = static_cast<int32_t>(position - age);
          selector = 0;
        }
      }
    }
    pointer<int32_t>(args.output0)[index] = logical;
    pointer<int32_t>(args.output1)[index] = selector;
  } else if (args.kind == FERRULE_CORE_DATA_PAGED_PLANE_SCATTER) {
    const uint32_t row = static_cast<uint32_t>(index / args.width);
    const uint32_t column = static_cast<uint32_t>(index % args.width);
    if ((args.flags & 2u) != 0 &&
        const_pointer<int32_t>(args.input5)[row] == 0) {
      return;
    }
    const int32_t logical = const_pointer<int32_t>(args.input1)[row];
    if (logical < 0) {
      return;
    }
    const uint32_t sequence =
        (args.flags & 1u) != 0
            ? static_cast<uint32_t>(const_pointer<int32_t>(args.input4)[row])
            : row;
    const uint64_t base = paged_row_offset(
        args.output_elements, const_pointer<int32_t>(args.input2),
        const_pointer<int32_t>(args.input3), sequence,
        static_cast<uint32_t>(logical), args.value0, args.width, args.value1,
        args.value2);
    if (base != UINT64_MAX) {
      pointer<float>(args.output0)[base + column] =
          const_pointer<float>(args.input0)[index];
    }
  }
}

inline bool valid_rows(const FerruleCoreRowsArgs *args) {
  if (!valid(args) || args->kind != FERRULE_CORE_ROWS_F32_TO_BF16_RNE ||
      args->rows == 0 || args->heads == 0 || args->dimensions == 0) {
    return false;
  }
  uint64_t values;
  if (!checked_mul_u64(args->rows, args->heads, &values) ||
      !checked_mul_u64(values, args->dimensions, &values)) {
    return false;
  }
  const uint64_t input_width =
      static_cast<uint64_t>(args->dimensions) * sizeof(float);
  const uint64_t output_width =
      static_cast<uint64_t>(args->dimensions) * sizeof(uint16_t);
  return transformer_range(
             args->input_f32, args->input_bytes, args->rows,
             args->input_row_stride_bytes, args->heads,
             args->input_head_stride_bytes, input_width, alignof(float)) &&
         transformer_range(
             args->output_bf16, args->output_bytes, args->rows,
             args->output_row_stride_bytes, args->heads,
             args->output_head_stride_bytes, output_width, alignof(uint16_t));
}

__global__ void rows_f32_to_bf16_rne_kernel(FerruleCoreRowsArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t values =
      static_cast<uint64_t>(args.rows) * args.heads * args.dimensions;
  if (index >= values) {
    return;
  }
  const uint32_t dimension = static_cast<uint32_t>(index % args.dimensions);
  const uint64_t head_row = index / args.dimensions;
  const uint32_t head = static_cast<uint32_t>(head_row % args.heads);
  const uint32_t row = static_cast<uint32_t>(head_row / args.heads);
  const uint64_t input_offset =
      static_cast<uint64_t>(row) * args.input_row_stride_bytes +
      static_cast<uint64_t>(head) * args.input_head_stride_bytes +
      static_cast<uint64_t>(dimension) * sizeof(float);
  const uint64_t output_offset =
      static_cast<uint64_t>(row) * args.output_row_stride_bytes +
      static_cast<uint64_t>(head) * args.output_head_stride_bytes +
      static_cast<uint64_t>(dimension) * sizeof(uint16_t);
  pointer<uint16_t>(args.output_bf16 + output_offset)[0] = bf16_rne(
      const_pointer<float>(args.input_f32 + input_offset)[0]);
}

__global__ void embedding_kernel(FerruleCoreEmbeddingArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t row_width = static_cast<uint64_t>(args.hc) * args.hidden;
  if (index >= static_cast<uint64_t>(args.rows) * row_width || row_width == 0) {
    return;
  }
  const uint32_t row = static_cast<uint32_t>(index / row_width);
  uint32_t token =
      args.kind == FERRULE_CORE_EMBED_PROPOSAL_HC_BF16
          ? (row == 0 ? args.anchor_token : args.noise_token)
          : static_cast<uint32_t>(const_pointer<int32_t>(args.token_ids)[row]);
  if (token >= args.vocab) {
    return;
  }
  const uint32_t dimension = static_cast<uint32_t>(index % args.hidden);
  pointer<float>(args.output)[index] = bf16_value(const_pointer<uint16_t>(
      args.embedding)[static_cast<uint64_t>(token) * args.hidden + dimension]);
}

__global__ void norm_kernel(FerruleCoreNormArgs args) {
  const uint32_t row = blockIdx.x;
  if (row >= args.rows || args.width == 0) {
    return;
  }
  const float *input = const_pointer<float>(args.input);
  const uint64_t base = static_cast<uint64_t>(row) * args.width;
  const bool head_rows = args.kind == FERRULE_CORE_NORM_HEAD_ROWS;
  float sum = 0.0f;
  for (uint32_t column = threadIdx.x; column < args.width;
       column += blockDim.x) {
    const float value = input[base + column];
    const float square = value * value;
    sum += head_rows ? bf16_round(square) : square;
  }

  constexpr uint32_t kWarpSize = 32;
  constexpr uint32_t kWarpCount = kBlock / kWarpSize;
  __shared__ float warp_sums[kWarpCount];
  __shared__ float inverse_rms_shared;
  for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    sum = lane < kWarpCount ? warp_sums[lane] : 0.0f;
    for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
      if (head_rows) {
        const float mean = bf16_round(sum / args.width);
        const float mean_with_epsilon = bf16_round(mean + args.epsilon);
        inverse_rms_shared = bf16_round(rsqrtf(mean_with_epsilon));
      } else {
        inverse_rms_shared = rsqrtf(sum / args.width + args.epsilon);
      }
      if (args.kind == FERRULE_CORE_NORM_COMPUTE_RMS) {
        pointer<float>(args.output)[row] = inverse_rms_shared;
      }
    }
  }
  __syncthreads();
  if (args.kind == FERRULE_CORE_NORM_COMPUTE_RMS) {
    return;
  }

  float *output = pointer<float>(args.output);
  const float *weight = const_pointer<float>(args.weight);
  for (uint32_t column = threadIdx.x; column < args.width;
       column += blockDim.x) {
    const float affine =
        args.kind == FERRULE_CORE_NORM_HEAD_ROWS ? 1.0f : weight[column];
    output[base + column] =
        bf16_round(input[base + column] * inverse_rms_shared * affine);
  }
}

} // namespace ferrule::cuda::core
