#ifndef FERRULE_CUTLASS_SM121_DSPARK_PROPOSAL_HEAD_CUH_
#define FERRULE_CUTLASS_SM121_DSPARK_PROPOSAL_HEAD_CUH_

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "sm121_bf16_compressor_prefill.cuh"

#if defined(FERRULE_CUTLASS_TARGET_SM) && FERRULE_CUTLASS_TARGET_SM != 121
#error "sm121_dspark_proposal_head.cuh requires Ferrule's SM121a target"
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1210
#error "sm121_dspark_proposal_head.cuh device code must target SM121"
#endif

namespace ferrule::cutlass::sm121_dspark_proposal_head {

namespace bf16 = ferrule::cutlass::sm121_bf16_compressor_prefill;

inline constexpr std::uint32_t kArgsVersion = 1u;
inline constexpr std::uint32_t kProposalRows = 5u;
inline constexpr std::uint32_t kThreads = 256u;
inline constexpr std::uint32_t kMaximumCooperativeBlocks = 64u;

struct Args {
  std::uint32_t args_version;
  std::uint32_t rows;
  std::uint32_t hc;
  std::uint32_t hidden;
  std::uint32_t vocab;
  std::uint32_t markov_rank;
  std::uint32_t partial_capacity;
  std::uint32_t reserved;
  float hc_eps;
  float norm_eps;

  std::uint64_t hc_state_f32;
  std::uint64_t hc_function_f32;
  std::uint64_t hc_scale_f32;
  std::uint64_t hc_base_f32;
  std::uint64_t norm_weight_f32;
  std::uint64_t lm_head_bf16;
  std::uint64_t markov_w1_bf16;
  std::uint64_t markov_w2_bf16;
  std::uint64_t confidence_weight_bf16;

  std::uint64_t hidden_f32;
  std::uint64_t normalized_f32;
  std::uint64_t base_logits_f32;
  std::uint64_t partial_values_f32;
  std::uint64_t partial_indices_i32;
  std::uint64_t token_ids_i32;
  std::uint64_t confidence_f32;
  std::uint64_t status_i32;
  std::uint64_t stream;
};

static_assert(std::is_standard_layout_v<Args>);
static_assert(std::is_trivially_copyable_v<Args>);
static_assert(sizeof(Args) == 184u,
              "SM121 DSpark proposal-head POD ABI changed");

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidAbi = 1,
  kInvalidArgument = 2,
  kLaunchFailed = 3,
};

namespace detail {

inline constexpr bool aligned(std::uint64_t address,
                              std::uint64_t alignment) {
  return address != 0u && (address & (alignment - 1u)) == 0u;
}

template <class T>
__host__ __device__ __forceinline__ T *device_pointer(std::uint64_t address) {
  return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address));
}

__device__ __forceinline__ std::uint16_t f32_to_bf16_rne(float value) {
  std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  }
  const std::uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>((bits + bias) >> 16);
}

__device__ __forceinline__ float bf16_to_f32(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16);
}

__device__ __forceinline__ float bf16_boundary(float value) {
  return bf16_to_f32(f32_to_bf16_rne(value));
}

__device__ __forceinline__ float block_sum(float value,
                                           float (&scratch)[kThreads]) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (std::uint32_t stride = kThreads / 2u; stride != 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

struct HcShared {
  float reduction[kThreads];
  float coefficient[8];
  float hidden_inv_rms;
};

__global__ __launch_bounds__(kThreads, 1) void hc_head_norm_kernel(Args args) {
  __shared__ HcShared shared;
  const std::uint32_t row = blockIdx.x;
  if (row >= args.rows) {
    return;
  }
  const auto *state = device_pointer<const float>(args.hc_state_f32) +
                      static_cast<std::uint64_t>(row) * args.hc * args.hidden;
  const auto *function = device_pointer<const float>(args.hc_function_f32);
  const auto *scale = device_pointer<const float>(args.hc_scale_f32);
  const auto *base = device_pointer<const float>(args.hc_base_f32);
  const auto *norm = device_pointer<const float>(args.norm_weight_f32);
  auto *hidden = device_pointer<float>(args.hidden_f32) +
                 static_cast<std::uint64_t>(row) * args.hidden;
  auto *normalized = device_pointer<float>(args.normalized_f32) +
                     static_cast<std::uint64_t>(row) * args.hidden;

  float local_square = 0.0f;
  const std::uint32_t hc_hidden = args.hc * args.hidden;
  for (std::uint32_t index = threadIdx.x; index < hc_hidden;
       index += blockDim.x) {
    const float value = state[index];
    local_square += value * value;
  }
  const float state_inv_rms =
      rsqrtf(block_sum(local_square, shared.reduction) /
                 static_cast<float>(hc_hidden) +
             args.norm_eps);

  for (std::uint32_t output_hc = 0u; output_hc < args.hc; ++output_hc) {
    float local_dot = 0.0f;
    const auto *function_row =
        function + static_cast<std::uint64_t>(output_hc) * hc_hidden;
    for (std::uint32_t index = threadIdx.x; index < hc_hidden;
         index += blockDim.x) {
      local_dot += state[index] * function_row[index];
    }
    const float mix = block_sum(local_dot, shared.reduction) * state_inv_rms;
    if (threadIdx.x == 0u) {
      shared.coefficient[output_hc] =
          1.0f / (1.0f + expf(-(mix * scale[0] + base[output_hc]))) +
          args.hc_eps;
    }
    __syncthreads();
  }

  float local_hidden_square = 0.0f;
  for (std::uint32_t dimension = threadIdx.x; dimension < args.hidden;
       dimension += blockDim.x) {
    float value = 0.0f;
    for (std::uint32_t input_hc = 0u; input_hc < args.hc; ++input_hc) {
      value += shared.coefficient[input_hc] *
               state[static_cast<std::uint64_t>(input_hc) * args.hidden +
                     dimension];
    }
    value = bf16_boundary(value);
    hidden[dimension] = value;
    local_hidden_square += value * value;
  }
  const float hidden_inv_rms =
      rsqrtf(block_sum(local_hidden_square, shared.reduction) /
                 static_cast<float>(args.hidden) +
             args.norm_eps);
  if (threadIdx.x == 0u) {
    shared.hidden_inv_rms = hidden_inv_rms;
  }
  __syncthreads();
  for (std::uint32_t dimension = threadIdx.x; dimension < args.hidden;
       dimension += blockDim.x) {
    normalized[dimension] =
        bf16_boundary(hidden[dimension] * shared.hidden_inv_rms * norm[dimension]);
  }
}

struct ProposalShared {
  float markov[1024];
  float values[kThreads];
  std::int32_t indices[kThreads];
};

__device__ __forceinline__ bool better(float candidate_value,
                                       std::int32_t candidate_index,
                                       float current_value,
                                       std::int32_t current_index) {
  return candidate_value > current_value ||
         (candidate_value == current_value &&
          candidate_index < current_index);
}

__device__ __forceinline__ void block_argmax(
    float &value, std::int32_t &index, ProposalShared &shared) {
  shared.values[threadIdx.x] = value;
  shared.indices[threadIdx.x] = index;
  __syncthreads();
  for (std::uint32_t stride = kThreads / 2u; stride != 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      const float candidate_value = shared.values[threadIdx.x + stride];
      const std::int32_t candidate_index =
          shared.indices[threadIdx.x + stride];
      if (better(candidate_value, candidate_index,
                 shared.values[threadIdx.x], shared.indices[threadIdx.x])) {
        shared.values[threadIdx.x] = candidate_value;
        shared.indices[threadIdx.x] = candidate_index;
      }
    }
    __syncthreads();
  }
  value = shared.values[0];
  index = shared.indices[0];
}

__global__ __launch_bounds__(kThreads, 1) void proposal_kernel(Args args) {
  __shared__ ProposalShared shared;
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const auto *w1 = device_pointer<const std::uint16_t>(args.markov_w1_bf16);
  const auto *w2 = device_pointer<const std::uint16_t>(args.markov_w2_bf16);
  const auto *confidence_weight =
      device_pointer<const std::uint16_t>(args.confidence_weight_bf16);
  const auto *hidden = device_pointer<const float>(args.hidden_f32);
  const auto *base_logits = device_pointer<const float>(args.base_logits_f32);
  auto *partial_values = device_pointer<float>(args.partial_values_f32);
  auto *partial_indices = device_pointer<std::int32_t>(args.partial_indices_i32);
  auto *token_ids = device_pointer<std::int32_t>(args.token_ids_i32);
  auto *confidence = device_pointer<float>(args.confidence_f32);
  auto *status = device_pointer<std::int32_t>(args.status_i32);

  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *status = 0;
  }
  grid.sync();

  for (std::uint32_t position = 0u; position < args.rows; ++position) {
    const std::int32_t previous_token = token_ids[position];
    if (previous_token < 0 ||
        static_cast<std::uint32_t>(previous_token) >= args.vocab) {
      if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *status = 1;
      }
      return;
    }
    for (std::uint32_t rank = threadIdx.x; rank < args.markov_rank;
         rank += blockDim.x) {
      shared.markov[rank] =
          bf16_to_f32(w1[static_cast<std::uint64_t>(previous_token) *
                           args.markov_rank +
                       rank]);
    }
    __syncthreads();

    float local_value = -std::numeric_limits<float>::infinity();
    std::int32_t local_index = std::numeric_limits<std::int32_t>::max();
    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t global_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t token = global_thread; token < args.vocab;
         token += global_stride) {
      float markov_bias = 0.0f;
      const auto *w2_row = w2 + token * args.markov_rank;
      for (std::uint32_t rank = 0u; rank < args.markov_rank; ++rank) {
        markov_bias += bf16_to_f32(w2_row[rank]) * shared.markov[rank];
      }
      const float score =
          base_logits[static_cast<std::uint64_t>(position) * args.vocab + token] +
          markov_bias;
      const auto token_i32 = static_cast<std::int32_t>(token);
      if (better(score, token_i32, local_value, local_index)) {
        local_value = score;
        local_index = token_i32;
      }
    }
    block_argmax(local_value, local_index, shared);
    if (threadIdx.x == 0u) {
      partial_values[blockIdx.x] = local_value;
      partial_indices[blockIdx.x] = local_index;
    }
    grid.sync();

    if (blockIdx.x == 0u) {
      float global_value = -std::numeric_limits<float>::infinity();
      std::int32_t global_index = std::numeric_limits<std::int32_t>::max();
      for (std::uint32_t block = threadIdx.x; block < gridDim.x;
           block += blockDim.x) {
        if (better(partial_values[block], partial_indices[block], global_value,
                   global_index)) {
          global_value = partial_values[block];
          global_index = partial_indices[block];
        }
      }
      block_argmax(global_value, global_index, shared);
      if (threadIdx.x == 0u) {
        token_ids[position + 1u] = global_index;
      }

      float local_confidence = 0.0f;
      for (std::uint32_t dimension = threadIdx.x; dimension < args.hidden;
           dimension += blockDim.x) {
        local_confidence +=
            hidden[static_cast<std::uint64_t>(position) * args.hidden + dimension] *
            bf16_to_f32(confidence_weight[dimension]);
      }
      for (std::uint32_t rank = threadIdx.x; rank < args.markov_rank;
           rank += blockDim.x) {
        local_confidence +=
            shared.markov[rank] *
            bf16_to_f32(confidence_weight[args.hidden + rank]);
      }
      shared.values[threadIdx.x] = local_confidence;
      __syncthreads();
      for (std::uint32_t stride = kThreads / 2u; stride != 0u;
           stride >>= 1u) {
        if (threadIdx.x < stride) {
          shared.values[threadIdx.x] += shared.values[threadIdx.x + stride];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0u) {
        confidence[position] = shared.values[0];
      }
    }
    grid.sync();
  }
}

} // namespace detail

inline Status validate(const Args *args) {
  if (args == nullptr || args->args_version != kArgsVersion) {
    return Status::kInvalidAbi;
  }
  if (args->rows != kProposalRows || args->hc == 0u || args->hc > 8u ||
      args->hidden == 0u || (args->hidden % 16u) != 0u || args->vocab == 0u ||
      args->markov_rank == 0u || args->markov_rank > 1024u ||
      (args->markov_rank % 16u) != 0u || args->partial_capacity == 0u ||
      args->partial_capacity > kMaximumCooperativeBlocks || args->reserved != 0u ||
      !std::isfinite(args->hc_eps) || !std::isfinite(args->norm_eps) ||
      args->hc_eps <= 0.0f || args->norm_eps <= 0.0f) {
    return Status::kInvalidArgument;
  }
  const bool pointers_valid =
      detail::aligned(args->hc_state_f32, 16u) &&
      detail::aligned(args->hc_function_f32, 16u) &&
      detail::aligned(args->hc_scale_f32, 4u) &&
      detail::aligned(args->hc_base_f32, 4u) &&
      detail::aligned(args->norm_weight_f32, 16u) &&
      detail::aligned(args->lm_head_bf16, 16u) &&
      detail::aligned(args->markov_w1_bf16, 16u) &&
      detail::aligned(args->markov_w2_bf16, 16u) &&
      detail::aligned(args->confidence_weight_bf16, 16u) &&
      detail::aligned(args->hidden_f32, 16u) &&
      detail::aligned(args->normalized_f32, 16u) &&
      detail::aligned(args->base_logits_f32, 16u) &&
      detail::aligned(args->partial_values_f32, 16u) &&
      detail::aligned(args->partial_indices_i32, 16u) &&
      detail::aligned(args->token_ids_i32, 4u) &&
      detail::aligned(args->confidence_f32, 16u) &&
      detail::aligned(args->status_i32, 4u);
  return pointers_valid ? Status::kSuccess : Status::kInvalidArgument;
}

inline Status launch(const Args *args) {
  const Status validation = validate(args);
  if (validation != Status::kSuccess) {
    return validation;
  }
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args->stream));

  detail::hc_head_norm_kernel<<<args->rows, kThreads, 0u, stream>>>(*args);
  if (cudaGetLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  const bf16::Args base_args{
      bf16::kArgsVersion,
      args->rows,
      args->vocab,
      0u,
      args->hidden,
      0u,
      args->normalized_f32,
      args->lm_head_bf16,
      args->lm_head_bf16,
      args->base_logits_f32,
      args->base_logits_f32,
      args->stream,
  };
  const dim3 base_grid((args->vocab + bf16::kCtaColumns - 1u) /
                           bf16::kCtaColumns,
                       (args->rows + bf16::kCtaRows - 1u) /
                           bf16::kCtaRows,
                       1u);
  bf16::kernel<<<base_grid, bf16::kThreads, 0u, stream>>>(base_args);
  if (cudaGetLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  int blocks_per_sm = 0;
  int device = 0;
  int multiprocessors = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm, detail::proposal_kernel, kThreads, 0u) != cudaSuccess ||
      cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess ||
      blocks_per_sm <= 0 || multiprocessors <= 0) {
    return Status::kLaunchFailed;
  }
  std::uint32_t blocks = static_cast<std::uint32_t>(blocks_per_sm) *
                         static_cast<std::uint32_t>(multiprocessors);
  blocks = blocks < args->partial_capacity ? blocks : args->partial_capacity;
  blocks = blocks < kMaximumCooperativeBlocks ? blocks
                                               : kMaximumCooperativeBlocks;
  if (blocks == 0u) {
    return Status::kLaunchFailed;
  }
  void *kernel_args[] = {const_cast<Args *>(args)};
  const cudaError_t launch_status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(detail::proposal_kernel), dim3(blocks, 1u, 1u),
      dim3(kThreads, 1u, 1u), kernel_args, 0u, stream);
  return launch_status == cudaSuccess ? Status::kSuccess
                                      : Status::kLaunchFailed;
}

} // namespace ferrule::cutlass::sm121_dspark_proposal_head

#endif // FERRULE_CUTLASS_SM121_DSPARK_PROPOSAL_HEAD_CUH_
