#ifndef FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_ROUTER_CUH_
#define FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_ROUTER_CUH_

#include "../../architectures/profile.cuh"
#include "explicit_selection_contract.cuh"
#include "explicit_selection_workspace.cuh"
#include "schedules/bf16_mma.cuh"
#ifdef FERRULE_CUDA_TEST_ORACLE
#include "schedules/preserve_f32_order.cuh"
#endif

#include <cstdint>

namespace ferrule::cuda::cutlass::capabilities::hybrid_mla_attention {
namespace router_detail {

template <class T> inline std::uint64_t address(T *pointer) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

inline ExplicitSelectionStatus
make_bf16_mma_args(const ExplicitSelectionArgs *args,
                   schedules::bf16_mma::Args *result) {
  if (args == nullptr || result == nullptr) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  const ExplicitSelectionStatus contract_status =
      validate_explicit_selection_contract(args);
  if (contract_status != ExplicitSelectionStatus::kSuccess) {
    return contract_status;
  }
  explicit_selection_workspace::Binding scratch{};
  const ExplicitSelectionStatus workspace_status =
      explicit_selection_workspace::bind(args, &scratch);
  if (workspace_status != ExplicitSelectionStatus::kSuccess) {
    return workspace_status;
  }

  *result = schedules::bf16_mma::Args{
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
      address(args->row_sequence_ids),
      address(args->row_kv_lens),
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

namespace architecture_private {

struct Bf16MmaImplementation {
  static ExplicitSelectionStatus
  validate(const schedules::bf16_mma::Args *args) {
    return schedules::bf16_mma::validate(args);
  }

  static ExplicitSelectionStatus launch(const schedules::bf16_mma::Args *args) {
    return schedules::bf16_mma::launch(args);
  }
};

template <std::uint32_t TargetSm> struct ExplicitSelectionRoute;

template <> struct ExplicitSelectionRoute<103u> : Bf16MmaImplementation {};

template <> struct ExplicitSelectionRoute<120u> : Bf16MmaImplementation {};

template <> struct ExplicitSelectionRoute<121u> : Bf16MmaImplementation {};

} // namespace architecture_private

#ifdef FERRULE_CUDA_TEST_ORACLE
inline ExplicitSelectionStatus
make_preserve_f32_order_args(const ExplicitSelectionArgs *args,
                             schedules::preserve_f32_order::Args *result);
#endif

} // namespace router_detail

inline ExplicitSelectionStatus explicit_selection_workspace_requirements(
    const ExplicitSelectionArgs *args,
    explicit_selection_workspace::Requirements *requirements) {
  switch (FERRULE_CUDA_TARGET_SM) {
  case 103:
  case 120:
  case 121:
    return explicit_selection_workspace::requirements(args, requirements);
  default:
    return ExplicitSelectionStatus::kUnsupported;
  }
}

inline ExplicitSelectionStatus
explicit_selection_can_implement(const ExplicitSelectionArgs *args) {
#ifdef FERRULE_CUDA_TEST_ORACLE
  schedules::preserve_f32_order::Args schedule_args{};
  const ExplicitSelectionStatus status =
      router_detail::make_preserve_f32_order_args(args, &schedule_args);
  if (status != ExplicitSelectionStatus::kSuccess) {
    return status;
  }
  switch (FERRULE_CUDA_TARGET_SM) {
  case 103:
  case 120:
  case 121:
    return schedules::preserve_f32_order::validate(&schedule_args);
  default:
    return ExplicitSelectionStatus::kUnsupported;
  }
#else
  schedules::bf16_mma::Args schedule_args{};
  const ExplicitSelectionStatus status =
      router_detail::make_bf16_mma_args(args, &schedule_args);
  if (status != ExplicitSelectionStatus::kSuccess) {
    return status;
  }
  switch (FERRULE_CUDA_TARGET_SM) {
  case 103:
    return router_detail::architecture_private::ExplicitSelectionRoute<
        103u>::validate(&schedule_args);
  case 120:
    return router_detail::architecture_private::ExplicitSelectionRoute<
        120u>::validate(&schedule_args);
  case 121:
    return router_detail::architecture_private::ExplicitSelectionRoute<
        121u>::validate(&schedule_args);
  default:
    return ExplicitSelectionStatus::kUnsupported;
  }
#endif
}

inline ExplicitSelectionStatus
explicit_selection_launch(const ExplicitSelectionArgs *args) {
#ifdef FERRULE_CUDA_TEST_ORACLE
  schedules::preserve_f32_order::Args schedule_args{};
  const ExplicitSelectionStatus status =
      router_detail::make_preserve_f32_order_args(args, &schedule_args);
  if (status != ExplicitSelectionStatus::kSuccess) {
    return status;
  }
  switch (FERRULE_CUDA_TARGET_SM) {
  case 103:
  case 120:
  case 121:
    return schedules::preserve_f32_order::launch(&schedule_args);
  default:
    return ExplicitSelectionStatus::kUnsupported;
  }
#else
  schedules::bf16_mma::Args schedule_args{};
  const ExplicitSelectionStatus status =
      router_detail::make_bf16_mma_args(args, &schedule_args);
  if (status != ExplicitSelectionStatus::kSuccess) {
    return status;
  }
  switch (FERRULE_CUDA_TARGET_SM) {
  case 103:
    return router_detail::architecture_private::ExplicitSelectionRoute<
        103u>::launch(&schedule_args);
  case 120:
    return router_detail::architecture_private::ExplicitSelectionRoute<
        120u>::launch(&schedule_args);
  case 121:
    return router_detail::architecture_private::ExplicitSelectionRoute<
        121u>::launch(&schedule_args);
  default:
    return ExplicitSelectionStatus::kUnsupported;
  }
#endif
}

#ifdef FERRULE_CUDA_TEST_ORACLE
namespace router_detail {

inline ExplicitSelectionStatus
make_preserve_f32_order_args(const ExplicitSelectionArgs *args,
                             schedules::preserve_f32_order::Args *result) {
  if (args == nullptr || result == nullptr) {
    return ExplicitSelectionStatus::kInvalidArgument;
  }
  const ExplicitSelectionStatus contract_status =
      validate_explicit_selection_contract(args);
  if (contract_status != ExplicitSelectionStatus::kSuccess) {
    return contract_status;
  }
  explicit_selection_workspace::Binding scratch{};
  const ExplicitSelectionStatus workspace_status =
      explicit_selection_workspace::bind(args, &scratch);
  if (workspace_status != ExplicitSelectionStatus::kSuccess) {
    return workspace_status;
  }
  *result = schedules::preserve_f32_order::Args{
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
      address(args->row_sequence_ids),
      address(args->row_kv_lens),
      address(args->selected_indices),
      address(args->selectors),
      address(args->attention_sink),
      address(scratch.source_addresses),
      address(scratch.raw_scores),
      address(args->output),
      address(args->status),
      address(args->stream),
  };
  return ExplicitSelectionStatus::kSuccess;
}

} // namespace router_detail

namespace test_oracle {

inline ExplicitSelectionStatus
scalar_launch(const ExplicitSelectionArgs *args) {
  schedules::preserve_f32_order::Args schedule_args{};
  const ExplicitSelectionStatus status =
      router_detail::make_preserve_f32_order_args(args, &schedule_args);
  if (status != ExplicitSelectionStatus::kSuccess) {
    return status;
  }
  switch (FERRULE_CUDA_TARGET_SM) {
  case 103:
  case 120:
  case 121:
    return schedules::preserve_f32_order::test_oracle::launch(&schedule_args);
  default:
    return ExplicitSelectionStatus::kUnsupported;
  }
}

inline ExplicitSelectionStatus
compare_launch(const ExplicitSelectionArgs *args,
               std::uint64_t oracle_output_f32,
               std::uint64_t compare_result_i32) {
  schedules::preserve_f32_order::Args schedule_args{};
  const ExplicitSelectionStatus status =
      router_detail::make_preserve_f32_order_args(args, &schedule_args);
  if (status != ExplicitSelectionStatus::kSuccess) {
    return status;
  }
  switch (FERRULE_CUDA_TARGET_SM) {
  case 103:
  case 120:
  case 121:
    return schedules::preserve_f32_order::test_oracle::launch_compare(
        &schedule_args, oracle_output_f32, compare_result_i32);
  default:
    return ExplicitSelectionStatus::kUnsupported;
  }
}

} // namespace test_oracle
#endif

} // namespace ferrule::cuda::cutlass::capabilities::hybrid_mla_attention

#endif // FERRULE_CUDA_CUTLASS_CAPABILITIES_HYBRID_MLA_ATTENTION_ROUTER_CUH_
