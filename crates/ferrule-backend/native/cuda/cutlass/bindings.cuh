#pragma once

#include "cutlass/abi.h"
#include "cutlass/target.cuh"
#include "cutlass/attention.cuh"
#include "cutlass/decoder_ops.cuh"
#include "cutlass/moe.cuh"
#include "cutlass/projections.cuh"
#ifdef FERRULE_CUDA_TEST_ORACLE
#include "cutlass/attention_oracle.cuh"
#endif

#include <cstdint>
#include <cuda_runtime_api.h>

namespace {
namespace fp8_prefill = ferrule::cuda::cutlass::operators::fp8_projection;
namespace bf16_contract = ferrule::cuda::cutlass::operators::bf16_compressor;
namespace bf16_schedule = ferrule::cuda::cutlass::operators::bf16_compressor;
namespace main_project_norm =
    ferrule::cuda::cutlass::operators::main_project_norm;
namespace hybrid_full_block =
    ferrule::cuda::cutlass::operators::hybrid_mla_attention::full_block;
namespace hybrid_explicit =
    ferrule::cuda::cutlass::operators::hybrid_mla_attention::explicit_selection;
namespace proposal_head = ferrule::cuda::cutlass::operators::proposal_head;
namespace hc_producer =
    ferrule::cuda::cutlass::operators::hyper_connection_producer;
namespace mla_output = ferrule::cuda::cutlass::operators::mla_output;
namespace shared_ffn = ferrule::cuda::cutlass::operators::shared_ffn;
#ifdef FERRULE_CUDA_TEST_ORACLE
namespace hybrid_oracle =
    ferrule::cuda::cutlass::test_oracles::hybrid_mla_attention;
#endif
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
namespace grouped_fp4_moe = ferrule::cuda::cutlass::operators::grouped_fp4_moe;
#endif
} // namespace

namespace {

fp8_prefill::Args make_prefill_args(const FerruleCutlassFp8QueryAKvArgs &args) {
  return fp8_prefill::Args{
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.activation_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.activation_ue8m0)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.query_a_weight_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.query_a_weight_ue8m0)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.kv_weight_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.kv_weight_ue8m0)),
      reinterpret_cast<float *>(
          static_cast<uintptr_t>(args.query_a_output_f32)),
      reinterpret_cast<float *>(static_cast<uintptr_t>(args.kv_output_f32)),
      args.rows,
      args.n1,
      args.n2,
      args.k,
  };
}

bf16_contract::Args
make_prefill_args(const FerruleCutlassBf16CompressorArgs &args) {
  return bf16_contract::Args{
      args.rows,
      args.n1,
      args.n2,
      args.k,
      args.reserved0,
      args.activation_f32,
      args.projection1_weight_bf16,
      args.projection2_weight_bf16,
      args.projection1_output_f32,
      args.projection2_output_f32,
      args.stream,
  };
}

hc_producer::HcPreRmsNormFp8Args
make_hc_producer_args(const FerruleCutlassHcProducerArgs &args) {
  return hc_producer::HcPreRmsNormFp8Args{
      args.rows,
      args.hc,
      args.hidden,
      args.mix,
      args.sinkhorn_iters,
      args.hc_eps,
      args.hc_norm_eps,
      args.layer_rms_eps,
      args.reserved,
      args.state_f32,
      args.function_row_major_f32,
      args.hc_scale_f32,
      args.hc_base_f32,
      args.layer_rms_weight_f32,
      args.mix_f32,
      args.workspace,
      args.workspace_bytes,
      args.hidden_f32,
      args.normalized_f32,
      args.packed_e4m3,
      args.scales_ue8m0,
      args.split_pre_f32,
      args.split_post_f32,
      args.split_comb_f32,
      args.stream,
  };
}

shared_ffn::Args make_shared_ffn_args(const FerruleCutlassSharedFfnArgs &args) {
  return shared_ffn::Args{
      reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(args.input_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.input_ue8m0)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.gate_weight_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.gate_weight_ue8m0)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.up_weight_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.up_weight_ue8m0)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.down_weight_fp8)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(args.down_weight_ue8m0)),
      reinterpret_cast<float *>(static_cast<uintptr_t>(args.hidden_f32)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.hidden_fp8)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.hidden_ue8m0)),
      reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_f32)),
      args.rows,
      args.input_size,
      args.intermediate_size,
      args.output_size,
      args.gate_block_m,
      args.gate_block_k,
      args.up_block_m,
      args.up_block_k,
      args.down_block_m,
      args.down_block_k,
      args.output_scale,
      args.swiglu_limit,
      args.flags,
  };
}

mla_output::Args make_mla_output_args(const FerruleCutlassMlaOutputArgs &args) {
  return mla_output::Args{
      args.rows,
      args.context_size,
      args.groups,
      args.group_input_size,
      args.rank,
      args.latent_size,
      args.hidden_size,
      args.output_a_scale_cols,
      args.reserved0,
      args.context_f32,
      args.output_a_weight_fp8,
      args.output_a_weight_ue8m0,
      args.output_b_weight_fp8,
      args.output_b_weight_ue8m0,
      args.latent_bf16,
      args.latent_fp8,
      args.latent_ue8m0,
      args.output_f32,
      args.stream,
  };
}

main_project_norm::Args
make_main_project_norm_args(const FerruleCutlassMainProjectNormArgs &args) {
  return main_project_norm::Args{
      args.rows,
      args.input_size,
      args.output_size,
      args.scale_cols,
      args.reserved0,
      args.rms_eps,
      args.reserved1,
      args.input_f32,
      args.activation_fp8,
      args.activation_ue8m0,
      args.weight_fp8,
      args.weight_ue8m0,
      args.norm_weight_f32,
      args.inv_rms_f32,
      args.output_f32,
      args.stream,
  };
}

hybrid_full_block::Args make_hybrid_mla_attention_args(
    const FerruleCutlassHybridMlaAttentionArgs &args) {
  return hybrid_full_block::Args{
      args.block_rows,         args.heads,
      args.head_dim,           args.sequence_tokens,
      args.window_size,        args.page_tokens,
      args.elements_per_token, args.layer_index,
      args.layer_count,        args.block_slot_offset,
      args.block_slot_count,   args.softmax_scale,
      args.reserved0,          args.context_plane_elements,
      args.query_f32,          args.context_plane_f32,
      args.block_kv_f32,       args.block_slots_i32,
      args.attention_sink_f32, args.query_bf16,
      args.gathered_kv_bf16,   args.scores_f32,
      args.probabilities_bf16, args.online_rescales_f32,
      args.denominators_f32,   args.output_f32,
      args.status_i32,         args.stream,
  };
}

template <class T> T *native_pointer(uint64_t address) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(address));
}

hybrid_explicit::ExplicitSelectionArgs make_hybrid_mla_explicit_selection_args(
    const FerruleCutlassHybridMlaExplicitSelectionArgs &args) {
  return hybrid_explicit::ExplicitSelectionArgs{
      args.kind,
      args.rows,
      args.tokens_per_sequence,
      args.kv_len,
      args.heads,
      args.head_dim,
      args.selected_width,
      args.page_tokens,
      args.first_elements_per_token,
      args.second_elements_per_token,
      args.layer_index,
      args.layer_count,
      args.flags,
      args.softmax_scale,
      args.reserved0,
      args.first_plane_elements,
      args.second_plane_elements,
      native_pointer<const float>(args.query_f32),
      native_pointer<const float>(args.first_plane_f32),
      native_pointer<const float>(args.second_plane_f32),
      native_pointer<const int32_t>(args.block_slots_i32),
      native_pointer<const int32_t>(args.block_offsets_i32),
      native_pointer<const int32_t>(args.sequence_kv_lens_i32),
      native_pointer<const int32_t>(args.second_sequence_kv_lens_i32),
      native_pointer<const int32_t>(args.row_sequence_ids_i32),
      native_pointer<const int32_t>(args.row_kv_lens_i32),
      native_pointer<const int32_t>(args.row_second_kv_lens_i32),
      native_pointer<const int32_t>(args.selected_indices_i32),
      native_pointer<const int32_t>(args.selectors_i32),
      native_pointer<const float>(args.attention_sink_f32),
      native_pointer<void>(args.workspace),
      args.workspace_bytes,
      native_pointer<float>(args.output_f32),
      native_pointer<int32_t>(args.status_i32),
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args.stream)),
  };
}

proposal_head::Args
make_proposal_head_args(const FerruleCutlassProposalHeadArgs &args) {
  return proposal_head::Args{
      args.rows,
      args.hc,
      args.hidden,
      args.vocab,
      args.markov_rank,
      args.partial_capacity,
      args.reserved0,
      args.hc_eps,
      args.norm_eps,
      args.hc_state_f32,
      args.hc_function_f32,
      args.hc_scale_f32,
      args.hc_base_f32,
      args.norm_weight_f32,
      args.lm_head_bf16,
      args.markov_w1_bf16,
      args.markov_w2_bf16,
      args.confidence_weight_bf16,
      args.hidden_f32,
      args.normalized_f32,
      args.base_logits_f32,
      args.partial_values_f32,
      args.partial_indices_i32,
      args.token_ids_i32,
      args.confidence_f32,
      args.status_i32,
      args.stream,
  };
}

#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
bool grouped_fp4_moe_options(grouped_fp4_moe::LaunchOptions &options) {
  int device_id = -1;
  int sm_count = 0;
  if (cudaGetDevice(&device_id) != cudaSuccess ||
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                             device_id) != cudaSuccess ||
      device_id < 0 || sm_count <= 0) {
    return false;
  }
  options.device_id = device_id;
  options.sm_count = sm_count;
  options.two_sm_min_rows = grouped_fp4_moe::kDefault2SmMinRows;
  return true;
}

grouped_fp4_moe::GroupedFp4MoeArgs
make_grouped_fp4_moe_args(const FerruleCutlassGroupedFp4MoeArgs &args) {
  return grouped_fp4_moe::GroupedFp4MoeArgs{
      args.active_group_count,
      args.small_group_count,
      args.slot_capacity,
      args.max_group_rows,
      args.total_routed_rows,
      args.num_tokens,
      args.num_routes,
      args.input_size,
      args.intermediate_size,
      args.hidden_size,
      args.swiglu_limit,
      args.active_expert_slots,
      args.active_group_generations,
      args.expert_route_indptr,
      args.expert_route_counts,
      args.route_token_indices,
      args.route_indices,
      args.route_weights,
      args.slot_generations,
      args.gate_ptrs,
      args.gate_scale_ptrs,
      args.up_ptrs,
      args.up_scale_ptrs,
      args.down_ptrs,
      args.down_scale_ptrs,
      args.input_fp8,
      args.input_ue8m0,
      args.route_output,
      args.route_written,
      args.route_error,
  };
}
#endif

} // namespace

namespace {

#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
int32_t grouped_fp4_moe_status(grouped_fp4_moe::Status status) {
  switch (status) {
  case grouped_fp4_moe::Status::kSuccess:
    return FERRULE_CUTLASS_SUCCESS;
  case grouped_fp4_moe::Status::kInvalidArgument:
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  case grouped_fp4_moe::Status::kUnsupportedResources:
    return FERRULE_CUTLASS_UNSUPPORTED;
  case grouped_fp4_moe::Status::kLaunchFailed:
    return FERRULE_CUTLASS_LAUNCH_FAILED;
  }
  return FERRULE_CUTLASS_LAUNCH_FAILED;
}
#endif

int32_t helper_launch_status(cudaError_t status) {
  if (status == cudaSuccess) {
    return FERRULE_CUTLASS_SUCCESS;
  }
  return status == cudaErrorInvalidValue ? FERRULE_CUTLASS_INVALID_ARGUMENT
                                         : FERRULE_CUTLASS_LAUNCH_FAILED;
}

} // namespace

extern "C" int32_t ferrule_cutlass_bf16_compressor_can_implement(
    const FerruleCutlassBf16CompressorArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  return bf16_contract::validate(make_prefill_args(*args)) ==
                 bf16_contract::ValidationResult::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
}

extern "C" int32_t ferrule_cutlass_bf16_compressor_launch(
    const FerruleCutlassBf16CompressorArgs *args) {
  int32_t status = ferrule_cutlass_bf16_compressor_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }

  return helper_launch_status(bf16_schedule::launch(make_prefill_args(*args)));
}

extern "C" int32_t ferrule_cutlass_fp8_query_a_kv_can_implement(
    const FerruleCutlassFp8QueryAKvArgs *args) {
#if !FERRULE_CUDA_HAS_FP8_MMA_SYNC
  (void)args;
  return FERRULE_CUTLASS_UNSUPPORTED;
#else
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  if (args->scale_cols != args->k / 128u) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  return fp8_prefill::validate(make_prefill_args(*args)) ==
                 fp8_prefill::ValidationResult::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
#endif
}

extern "C" int32_t ferrule_cutlass_fp8_query_a_kv_launch(
    const FerruleCutlassFp8QueryAKvArgs *args) {
  int32_t status = ferrule_cutlass_fp8_query_a_kv_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }

  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return helper_launch_status(
      fp8_prefill::launch(make_prefill_args(*args), stream));
}

extern "C" int32_t ferrule_cutlass_fp8_projection_can_implement(
    const FerruleCutlassFp8QueryAKvArgs *args) {
#if !FERRULE_CUDA_HAS_FP8_MMA_SYNC
  (void)args;
  return FERRULE_CUTLASS_UNSUPPORTED;
#else
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  if (args->n2 != 0u || args->kv_weight_fp8 != 0u ||
      args->kv_weight_ue8m0 != 0u || args->kv_output_f32 != 0u ||
      args->scale_cols != args->k / 128u) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  return fp8_prefill::validate_single(make_prefill_args(*args)) ==
                 fp8_prefill::ValidationResult::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
#endif
}

extern "C" int32_t ferrule_cutlass_fp8_projection_launch(
    const FerruleCutlassFp8QueryAKvArgs *args) {
  const int32_t status = ferrule_cutlass_fp8_projection_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return helper_launch_status(
      fp8_prefill::launch_single(make_prefill_args(*args), stream));
}

extern "C" uint64_t ferrule_cutlass_grouped_fp4_moe_workspace_size(
    const FerruleCutlassGroupedFp4MoeArgs *args) {
  if (args == nullptr) {
    return 0;
  }
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
  grouped_fp4_moe::LaunchOptions options{};
  if (!grouped_fp4_moe_options(options)) {
    return 0;
  }
  const auto native_args = make_grouped_fp4_moe_args(*args);
  return static_cast<uint64_t>(
      grouped_fp4_moe::workspace_bytes(native_args, options));
#else
  return 0;
#endif
}

extern "C" int32_t ferrule_cutlass_grouped_fp4_moe_can_implement(
    const FerruleCutlassGroupedFp4MoeArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
  grouped_fp4_moe::LaunchOptions options{};
  if (!grouped_fp4_moe_options(options)) {
    return FERRULE_CUTLASS_UNSUPPORTED;
  }
  const auto native_args = make_grouped_fp4_moe_args(*args);
  void *workspace =
      reinterpret_cast<void *>(static_cast<uintptr_t>(args->workspace));
  return grouped_fp4_moe_status(grouped_fp4_moe::can_implement(
      &native_args, workspace, static_cast<size_t>(args->workspace_bytes),
      options));
#else
  return FERRULE_CUTLASS_UNSUPPORTED;
#endif
}

extern "C" int32_t ferrule_cutlass_grouped_fp4_moe_launch(
    const FerruleCutlassGroupedFp4MoeArgs *args) {
  const int32_t status = ferrule_cutlass_grouped_fp4_moe_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
  grouped_fp4_moe::LaunchOptions options{};
  if (!grouped_fp4_moe_options(options)) {
    return FERRULE_CUTLASS_UNSUPPORTED;
  }
  const auto native_args = make_grouped_fp4_moe_args(*args);
  void *workspace =
      reinterpret_cast<void *>(static_cast<uintptr_t>(args->workspace));
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return grouped_fp4_moe_status(grouped_fp4_moe::launch(
      &native_args, workspace, static_cast<size_t>(args->workspace_bytes),
      stream, options));
#else
  return FERRULE_CUTLASS_UNSUPPORTED;
#endif
}

extern "C" uint64_t ferrule_cutlass_mxfp4_sfb_storage_bytes(uint32_t n,
                                                            uint32_t k) {
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
  if (n > 0x7fffffffu || k > 0x7fffffffu) {
    return 0;
  }
  return static_cast<uint64_t>(grouped_fp4_moe::prepared_sfb_bytes(
      static_cast<int>(n), static_cast<int>(k)));
#else
  static_cast<void>(n);
  static_cast<void>(k);
  return 0;
#endif
}

extern "C" int32_t ferrule_cutlass_prepare_mxfp4_sfb(
    const FerruleCutlassPrepareMxfp4SfbArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  if (args->reserved0 != 0u) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
#if FERRULE_CUTLASS_HAS_SM103_GROUPED_FP4_MOE
  if (args->n > 0x7fffffffu || args->k > 0x7fffffffu) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  auto *destination = reinterpret_cast<uint8_t *>(
      static_cast<uintptr_t>(args->prepared_destination));
  const auto *source = reinterpret_cast<const uint8_t *>(
      static_cast<uintptr_t>(args->linear_source));
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return helper_launch_status(grouped_fp4_moe::launch_prepare_sfb(
      destination, source, static_cast<int32_t>(args->n),
      static_cast<int32_t>(args->k), stream));
#else
  return FERRULE_CUTLASS_UNSUPPORTED;
#endif
}

extern "C" int32_t ferrule_cutlass_hybrid_mla_attention_can_implement(
    const FerruleCutlassHybridMlaAttentionArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_attention_args(*args);
  return static_cast<int32_t>(hybrid_full_block::validate(&native_args));
}

extern "C" int32_t ferrule_cutlass_hybrid_mla_attention_launch(
    const FerruleCutlassHybridMlaAttentionArgs *args) {
  const int32_t status =
      ferrule_cutlass_hybrid_mla_attention_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_hybrid_mla_attention_args(*args);
  return static_cast<int32_t>(hybrid_full_block::launch(&native_args));
}

extern "C" int32_t
ferrule_cutlass_hybrid_mla_explicit_selection_workspace_requirements(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args,
    FerruleCutlassWorkspaceRequirements *requirements) {
  if (args == nullptr || requirements == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_explicit_selection_args(*args);
  hybrid_explicit::workspace::Requirements native_requirements{};
  const auto status = hybrid_explicit::workspace_requirements(
      &native_args, &native_requirements);
  if (status != hybrid_explicit::ExplicitSelectionStatus::kSuccess) {
    return static_cast<int32_t>(status);
  }
  *requirements = FerruleCutlassWorkspaceRequirements{
      native_requirements.bytes,
      native_requirements.alignment,
      0u,
  };
  return FERRULE_CUTLASS_SUCCESS;
}

extern "C" int32_t ferrule_cutlass_hybrid_mla_explicit_selection_can_implement(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_explicit_selection_args(*args);
  return static_cast<int32_t>(hybrid_explicit::can_implement(&native_args));
}

extern "C" int32_t ferrule_cutlass_hybrid_mla_explicit_selection_launch(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_explicit_selection_args(*args);
  return static_cast<int32_t>(hybrid_explicit::launch(&native_args));
}

#ifdef FERRULE_CUDA_TEST_ORACLE
extern "C" int32_t
ferrule_cutlass_test_hybrid_mla_explicit_selection_scalar_launch(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_explicit_selection_args(*args);
  return static_cast<int32_t>(hybrid_oracle::scalar_launch(&native_args));
}

extern "C" int32_t
ferrule_cutlass_test_hybrid_mla_explicit_selection_compare_launch(
    const FerruleCutlassHybridMlaExplicitSelectionArgs *args,
    uint64_t oracle_output_f32, uint64_t compare_result_i32) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hybrid_mla_explicit_selection_args(*args);
  return static_cast<int32_t>(hybrid_oracle::compare_launch(
      &native_args, oracle_output_f32, compare_result_i32));
}
#endif

extern "C" int32_t ferrule_cutlass_hc_producer_can_implement(
    const FerruleCutlassHcProducerArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_hc_producer_args(*args);
  return hc_producer::validate_hc_pre_rmsnorm_fp8(native_args)
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
}

extern "C" int32_t
ferrule_cutlass_hc_producer_launch(const FerruleCutlassHcProducerArgs *args) {
  const int32_t status = ferrule_cutlass_hc_producer_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  return helper_launch_status(
      hc_producer::launch_hc_pre_rmsnorm_fp8(make_hc_producer_args(*args)));
}

extern "C" int32_t ferrule_cutlass_main_project_norm_can_implement(
    const FerruleCutlassMainProjectNormArgs *args) {
#if !FERRULE_CUDA_HAS_FP8_MMA_SYNC
  (void)args;
  return FERRULE_CUTLASS_UNSUPPORTED;
#else
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_main_project_norm_args(*args);
  return static_cast<int32_t>(main_project_norm::validate(&native_args));
#endif
}

extern "C" int32_t ferrule_cutlass_main_project_norm_launch(
    const FerruleCutlassMainProjectNormArgs *args) {
  const int32_t status = ferrule_cutlass_main_project_norm_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_main_project_norm_args(*args);
  return static_cast<int32_t>(main_project_norm::launch(&native_args));
}

extern "C" int32_t ferrule_cutlass_mla_output_can_implement(
    const FerruleCutlassMlaOutputArgs *args) {
#if !FERRULE_CUDA_HAS_FP8_MMA_SYNC
  (void)args;
  return FERRULE_CUTLASS_UNSUPPORTED;
#else
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_mla_output_args(*args);
  return static_cast<int32_t>(mla_output::validate(&native_args));
#endif
}

extern "C" int32_t
ferrule_cutlass_mla_output_launch(const FerruleCutlassMlaOutputArgs *args) {
  const int32_t status = ferrule_cutlass_mla_output_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_mla_output_args(*args);
  return static_cast<int32_t>(mla_output::launch(&native_args));
}

extern "C" int32_t ferrule_cutlass_proposal_head_can_implement(
    const FerruleCutlassProposalHeadArgs *args) {
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  const auto native_args = make_proposal_head_args(*args);
  return static_cast<int32_t>(proposal_head::validate(&native_args));
}

extern "C" int32_t ferrule_cutlass_proposal_head_launch(
    const FerruleCutlassProposalHeadArgs *args) {
  const int32_t status = ferrule_cutlass_proposal_head_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  const auto native_args = make_proposal_head_args(*args);
  return static_cast<int32_t>(proposal_head::launch(&native_args));
}

extern "C" int32_t ferrule_cutlass_shared_ffn_can_implement(
    const FerruleCutlassSharedFfnArgs *args) {
#if !FERRULE_CUDA_HAS_FP8_MMA_SYNC
  (void)args;
  return FERRULE_CUTLASS_UNSUPPORTED;
#else
  if (args == nullptr) {
    return FERRULE_CUTLASS_INVALID_ARGUMENT;
  }
  return shared_ffn::validate(make_shared_ffn_args(*args)) ==
                 shared_ffn::ValidationResult::kSuccess
             ? FERRULE_CUTLASS_SUCCESS
             : FERRULE_CUTLASS_INVALID_ARGUMENT;
#endif
}

extern "C" int32_t
ferrule_cutlass_shared_ffn_launch(const FerruleCutlassSharedFfnArgs *args) {
  const int32_t status = ferrule_cutlass_shared_ffn_can_implement(args);
  if (status != FERRULE_CUTLASS_SUCCESS) {
    return status;
  }
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(args->stream));
  return helper_launch_status(
      shared_ffn::launch(make_shared_ffn_args(*args), stream));
}
