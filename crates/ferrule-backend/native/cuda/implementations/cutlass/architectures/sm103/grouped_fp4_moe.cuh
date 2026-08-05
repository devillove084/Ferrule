#pragma once

#include "../profile.cuh"

#if FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4

#if !defined(__CUDACC__)
#error "SM103 grouped FP4 MoE support must be compiled with nvcc"
#endif

#include "grouped_fp4_gemm.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ferrule::cuda::cutlass::architectures::sm103::grouped_fp4_moe {

inline constexpr std::uint32_t kPrepareThreads = 128;
inline constexpr std::uint32_t kQuantThreads = 32;
inline constexpr std::uint32_t kScatterThreads = 256;
// The 1SM and 2SM reference kernels use M tiles of 128 and 256 respectively.
// Their midpoint is a neutral default; providers may tune this crossover.
inline constexpr std::uint32_t kDefault2SmMinRows = 192;
inline constexpr std::size_t kWorkspaceAlignment =
    kCutlassWorkspaceAlignment;

// Backend-private POD. All addresses refer to device-accessible memory. Ferrule
// core is the trusted producer of compact device metadata and keeps slot
// generations and bindings stable through the asynchronous launch. Active
// groups are ordered by M: groups [0, small_group_count) have
// route_counts[g] < options.two_sm_min_rows and the remainder have
// route_counts[g] >= options.two_sm_min_rows. Every active group has M > 0.
struct GroupedFp4MoeArgs {
  std::uint32_t active_group_count{};
  std::uint32_t small_group_count{};
  std::uint32_t slot_capacity{};
  std::uint32_t max_group_rows{};
  std::uint32_t total_routed_rows{};
  std::uint32_t num_tokens{};
  std::uint32_t num_routes{};
  std::uint32_t input_size{};
  std::uint32_t intermediate_size{};
  std::uint32_t hidden_size{};
  float swiglu_limit{};

  // int32[active_group_count] and int32[active_group_count].
  std::uint64_t active_expert_slots{};
  std::uint64_t active_group_generations{};
  // uint32[active_group_count + 1] and uint32[active_group_count].
  std::uint64_t expert_route_indptr{};
  std::uint64_t expert_route_counts{};
  // Expert-contiguous arrays of length total_routed_rows.
  std::uint64_t route_token_indices{}; // int32
  std::uint64_t route_indices{};       // int32
  std::uint64_t route_weights{};       // float

  // Slot-indexed arrays. Scale pointers address prepared CUTLASS SFB layouts.
  std::uint64_t slot_generations{}; // int32[slot_capacity]
  std::uint64_t gate_ptrs{};        // uint64[slot_capacity]
  std::uint64_t gate_scale_ptrs{};  // uint64[slot_capacity]
  std::uint64_t up_ptrs{};          // uint64[slot_capacity]
  std::uint64_t up_scale_ptrs{};    // uint64[slot_capacity]
  std::uint64_t down_ptrs{};        // uint64[slot_capacity]
  std::uint64_t down_scale_ptrs{};  // uint64[slot_capacity]

  // Full-token linear MXFP4: packed [num_tokens,input_size/2], scales
  // [num_tokens,input_size/32].
  std::uint64_t input_packed{};
  std::uint64_t input_scales{};

  // Route-major output and status arrays.
  std::uint64_t route_output{};  // float[num_routes,hidden_size]
  std::uint64_t route_written{}; // int32[num_routes]
  std::uint64_t route_error{};   // int32[1]
};

struct LaunchOptions {
  std::int32_t device_id{-1};
  std::int32_t sm_count{};
  std::uint32_t two_sm_min_rows{kDefault2SmMinRows};
};

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidArgument = 1,
  kUnsupportedResources = 2,
  kLaunchFailed = 3,
};

struct WorkspacePlan {
  std::int32_t descriptor_capacity{};
  std::size_t descriptor_offset{};
  std::size_t descriptor_bytes{};
  std::size_t group_bindings_offset{};
  std::size_t group_bindings_bytes{};
  std::size_t route_groups_offset{};
  std::size_t route_groups_bytes{};
  std::size_t gathered_x_offset{};
  std::size_t gathered_x_bytes{};
  std::size_t gathered_x_sfa_offset{};
  std::size_t gathered_x_sfa_bytes{};
  std::size_t gathered_x_sfa_group_stride{};
  std::size_t gate_up_f32_offset{};
  std::size_t gate_up_f32_bytes{};
  std::size_t hidden_packed_offset{};
  std::size_t hidden_packed_bytes{};
  std::size_t hidden_sfa_offset{};
  std::size_t hidden_sfa_bytes{};
  std::size_t hidden_sfa_group_stride{};
  std::size_t down_f32_offset{};
  std::size_t down_f32_bytes{};
  std::size_t cutlass_offset{};
  std::size_t cutlass_bytes{};
  std::size_t total_bytes{};

  bool valid() const noexcept { return total_bytes != 0; }
};

static_assert(std::is_standard_layout<GroupedFp4MoeArgs>::value, "GroupedFp4MoeArgs POD");
static_assert(std::is_trivially_copyable<GroupedFp4MoeArgs>::value,
              "GroupedFp4MoeArgs must be trivially copyable");
static_assert(sizeof(GroupedFp4MoeArgs) == 200u);
static_assert(alignof(GroupedFp4MoeArgs) == 8u);
static_assert(offsetof(GroupedFp4MoeArgs, active_group_count) == 0u);
static_assert(offsetof(GroupedFp4MoeArgs, swiglu_limit) == 40u);
static_assert(offsetof(GroupedFp4MoeArgs, active_expert_slots) == 48u);
static_assert(offsetof(GroupedFp4MoeArgs, route_error) == 192u);
static_assert(std::is_standard_layout<LaunchOptions>::value,
              "LaunchOptions POD");
static_assert(std::is_trivially_copyable<WorkspacePlan>::value,
              "WorkspacePlan must be trivially copyable");

namespace moe_detail {

constexpr bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

constexpr bool aligned_address(std::uint64_t address,
                               std::size_t alignment) noexcept {
  return address != 0 && is_power_of_two(alignment) &&
         (address & (alignment - 1)) == 0;
}

inline bool aligned_pointer(void const *pointer,
                            std::size_t alignment) noexcept {
  return pointer != nullptr && is_power_of_two(alignment) &&
         (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

inline bool checked_add(std::size_t &result, std::size_t left,
                        std::size_t right) noexcept {
  if (left > (std::numeric_limits<std::size_t>::max)() - right) {
    return false;
  }
  result = left + right;
  return true;
}

inline bool checked_product(std::size_t &result, std::size_t left,
                            std::size_t right) noexcept {
  if (left != 0 && right > (std::numeric_limits<std::size_t>::max)() / left) {
    return false;
  }
  result = left * right;
  return true;
}

inline bool checked_product(std::size_t &result, std::size_t first,
                            std::size_t second, std::size_t third) noexcept {
  std::size_t partial = 0;
  return checked_product(partial, first, second) &&
         checked_product(result, partial, third);
}

inline bool checked_product(std::size_t &result, std::size_t first,
                            std::size_t second, std::size_t third,
                            std::size_t fourth) noexcept {
  std::size_t partial = 0;
  return checked_product(partial, first, second, third) &&
         checked_product(result, partial, fourth);
}

inline bool append_region(std::size_t &cursor, std::size_t bytes,
                          std::size_t alignment, std::size_t &offset) noexcept {
  if (!is_power_of_two(alignment) ||
      cursor > (std::numeric_limits<std::size_t>::max)() - (alignment - 1)) {
    return false;
  }
  cursor = detail::align_up(cursor, alignment);
  offset = cursor;
  return checked_add(cursor, cursor, bytes);
}

inline bool scalar_args_valid(GroupedFp4MoeArgs const &args,
                              LaunchOptions const &options) noexcept {
  return args.active_group_count != 0 &&
         args.active_group_count <= 0x3fffffffu &&
         args.small_group_count <= args.active_group_count &&
         args.slot_capacity != 0 && args.max_group_rows != 0 &&
         args.max_group_rows <= 0x7fffffffu && args.total_routed_rows != 0 &&
         args.total_routed_rows <= args.num_routes &&
         static_cast<std::uint64_t>(args.total_routed_rows) >=
             args.active_group_count &&
         static_cast<std::uint64_t>(args.total_routed_rows) <=
             static_cast<std::uint64_t>(args.active_group_count) *
                 args.max_group_rows &&
         args.num_tokens != 0 && args.num_routes != 0 && args.input_size != 0 &&
         args.input_size <= 0x7fffffffu && args.intermediate_size != 0 &&
         args.intermediate_size <= 0x7fffffffu && args.hidden_size != 0 &&
         args.hidden_size <= 0x7fffffffu && (args.input_size % 64u) == 0 &&
         (args.intermediate_size % 64u) == 0 && (args.hidden_size % 4u) == 0 &&
         std::isfinite(args.swiglu_limit) && options.device_id >= 0 &&
         options.sm_count > 0 && options.two_sm_min_rows != 0 &&
         (args.small_group_count == args.active_group_count ||
          options.two_sm_min_rows <= args.max_group_rows);
}

inline bool pointer_args_valid(GroupedFp4MoeArgs const &args) noexcept {
  return aligned_address(args.active_expert_slots, alignof(std::int32_t)) &&
         aligned_address(args.active_group_generations,
                         alignof(std::int32_t)) &&
         aligned_address(args.expert_route_indptr, alignof(std::uint32_t)) &&
         aligned_address(args.expert_route_counts, alignof(std::uint32_t)) &&
         aligned_address(args.route_token_indices, alignof(std::int32_t)) &&
         aligned_address(args.route_indices, alignof(std::int32_t)) &&
         aligned_address(args.route_weights, alignof(float)) &&
         aligned_address(args.slot_generations, alignof(std::int32_t)) &&
         aligned_address(args.gate_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.gate_scale_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.up_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.up_scale_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.down_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.down_scale_ptrs, alignof(std::uint64_t)) &&
         aligned_address(args.input_packed, 16) &&
         aligned_address(args.input_scales, 16) &&
         aligned_address(args.route_output, 16) &&
         aligned_address(args.route_written, alignof(std::int32_t)) &&
         aligned_address(args.route_error, alignof(std::int32_t));
}

// These are real C++ objects, not fabricated addresses. CUTLASS workspace
// queries only consume num_groups; the non-null sentinels satisfy the grouped
// argument shape without pointer arithmetic or dereferencing device metadata.
struct QuerySentinels {
  ProblemShapeValue problem_shape{};
  ElementA const *a{};
  ElementB const *b{};
  ElementD *d{};
  ElementScale const *sfa{};
  ElementScale const *sfb{};
  StrideA stride_a{};
  StrideB stride_b{};
  StrideD stride_d{};
  LayoutSFA layout_sfa{};
  LayoutSFB layout_sfb{};
};

inline QuerySentinels query_sentinels{};

inline DeviceDescriptorView
query_descriptor_view(std::int32_t groups) noexcept {
  DeviceDescriptorView view{};
  view.groups = groups;
  view.problem_shapes = &query_sentinels.problem_shape;
  view.a = &query_sentinels.a;
  view.b = &query_sentinels.b;
  view.d = &query_sentinels.d;
  view.sfa = &query_sentinels.sfa;
  view.sfb = &query_sentinels.sfb;
  view.stride_a = &query_sentinels.stride_a;
  view.stride_b = &query_sentinels.stride_b;
  view.stride_d = &query_sentinels.stride_d;
  view.layout_sfa = &query_sentinels.layout_sfa;
  view.layout_sfb = &query_sentinels.layout_sfb;
  return view;
}

inline std::size_t
cutlass_bytes_for(MmaMode mode, std::uint32_t groups,
                  LaunchOptions const &options) noexcept {
  if (groups == 0 || groups > 0x7fffffffu) {
    return 0;
  }
  GroupedProblem problem{};
  problem.descriptors =
      query_descriptor_view(static_cast<std::int32_t>(groups));
  problem.device_id = options.device_id;
  problem.sm_count = options.sm_count;
  return cutlass_workspace_bytes(mode, problem);
}

inline DeviceDescriptorView slice_descriptors(DeviceDescriptorView view,
                                              std::int32_t offset,
                                              std::int32_t groups) noexcept {
  view.groups = groups;
  view.problem_shapes += offset;
  view.a += offset;
  view.b += offset;
  view.d += offset;
  view.sfa += offset;
  view.sfb += offset;
  view.stride_a += offset;
  view.stride_b += offset;
  view.stride_d += offset;
  view.layout_sfa += offset;
  view.layout_sfb += offset;
  return view;
}

template <class T>
__device__ __forceinline__ T *device_pointer(std::uint64_t address) {
  return reinterpret_cast<T *>(static_cast<std::uintptr_t>(address));
}

__device__ __forceinline__ void set_route_error(GroupedFp4MoeArgs const &args) {
  atomicOr(reinterpret_cast<unsigned int *>(
               device_pointer<std::int32_t>(args.route_error)),
           1u);
}

struct WorkspaceView {
  DeviceDescriptorView descriptors{};
  std::uint64_t *group_bindings{};
  std::uint32_t *route_groups{};
  std::uint8_t *gathered_x{};
  ElementScale *gathered_x_sfa{};
  float *gate_up_f32{};
  std::uint8_t *hidden_packed{};
  ElementScale *hidden_sfa{};
  float *down_f32{};
  std::size_t gathered_x_sfa_group_stride{};
  std::size_t hidden_sfa_group_stride{};
};

static_assert(std::is_trivially_copyable<WorkspaceView>::value,
              "WorkspaceView must be trivially copyable");

__device__ __forceinline__ void
write_descriptor(DeviceDescriptorView const &descriptors, std::int32_t index,
                 std::int32_t m, std::int32_t n, std::int32_t k,
                 ElementA const *a, ElementB const *b, ElementD *d,
                 ElementScale const *sfa, ElementScale const *sfb) {
  descriptors.problem_shapes[index] = ProblemShapeValue{m, n, k};
  descriptors.a[index] = a;
  descriptors.b[index] = b;
  descriptors.d[index] = d;
  descriptors.sfa[index] = sfa;
  descriptors.sfb[index] = sfb;
  descriptors.stride_a[index] =
      StrideA{static_cast<std::int64_t>(k), cute::_1{}, cute::_0{}};
  descriptors.stride_b[index] =
      StrideB{static_cast<std::int64_t>(k), cute::_1{}, cute::_0{}};
  descriptors.stride_d[index] =
      StrideD{static_cast<std::int64_t>(n), cute::_1{}, cute::_0{}};
  descriptors.layout_sfa[index] =
      BlockScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  descriptors.layout_sfb[index] =
      BlockScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
}

__global__ __launch_bounds__(kPrepareThreads) void prepare_gate_up_kernel(
    GroupedFp4MoeArgs args, LaunchOptions options,
    WorkspaceView workspace) {
  const std::uint32_t group = blockIdx.x;
  if (group >= args.active_group_count) {
    return;
  }

  __shared__ std::int32_t metadata_valid;
  __shared__ std::int32_t expert_slot;
  __shared__ std::uint32_t route_begin;
  __shared__ std::uint32_t route_count;
  __shared__ std::uint64_t bindings[6];

  const auto *slots =
      device_pointer<std::int32_t const>(args.active_expert_slots);
  const auto *indptr =
      device_pointer<std::uint32_t const>(args.expert_route_indptr);
  const auto *counts =
      device_pointer<std::uint32_t const>(args.expert_route_counts);
  if (threadIdx.x == 0) {
    route_begin = indptr[group];
    const std::uint32_t route_end = indptr[group + 1];
    route_count = counts[group];
    expert_slot = slots[group];
    const bool expected_small = group < args.small_group_count;
    const bool actual_small = route_count < options.two_sm_min_rows;
    metadata_valid =
        route_count != 0 && route_count <= args.max_group_rows &&
                route_begin <= route_end &&
                route_end <= args.total_routed_rows &&
                route_end - route_begin == route_count &&
                (group != 0 || route_begin == 0) &&
                (group + 1 != args.active_group_count ||
                 route_end == args.total_routed_rows) &&
                expected_small == actual_small && expert_slot >= 0 &&
                static_cast<std::uint32_t>(expert_slot) < args.slot_capacity
            ? 1
            : 0;
  }
  __syncthreads();

  const auto *tokens =
      device_pointer<std::int32_t const>(args.route_token_indices);
  const auto *routes = device_pointer<std::int32_t const>(args.route_indices);
  const auto *weights = device_pointer<float const>(args.route_weights);
  if (metadata_valid != 0) {
    for (std::uint32_t row = threadIdx.x; row < route_count;
         row += blockDim.x) {
      const std::uint32_t ordinal = route_begin + row;
      const std::int32_t token = tokens[ordinal];
      const std::int32_t route = routes[ordinal];
      if (token < 0 || static_cast<std::uint32_t>(token) >= args.num_tokens ||
          route < 0 || static_cast<std::uint32_t>(route) >= args.num_routes ||
          !isfinite(weights[ordinal])) {
        atomicExch(&metadata_valid, 0);
      }
    }
  }
  __syncthreads();

  // Invalid device metadata violates the trusted-producer contract. Trap the
  // stream before CUTLASS can observe an invalid descriptor; in particular,
  // this path never fabricates an M=0 grouped problem.
  if (metadata_valid == 0) {
    if (threadIdx.x == 0) {
      set_route_error(args);
    }
    __syncthreads();
    __trap();
    return;
  }

  if (threadIdx.x == 0) {
    const std::uint32_t expert = static_cast<std::uint32_t>(expert_slot);
    const auto *gate_ptrs = device_pointer<std::uint64_t const>(args.gate_ptrs);
    const auto *gate_scale_ptrs =
        device_pointer<std::uint64_t const>(args.gate_scale_ptrs);
    const auto *up_ptrs = device_pointer<std::uint64_t const>(args.up_ptrs);
    const auto *up_scale_ptrs =
        device_pointer<std::uint64_t const>(args.up_scale_ptrs);
    const auto *down_ptrs = device_pointer<std::uint64_t const>(args.down_ptrs);
    const auto *down_scale_ptrs =
        device_pointer<std::uint64_t const>(args.down_scale_ptrs);
    bindings[0] = gate_ptrs[expert];
    bindings[1] = gate_scale_ptrs[expert];
    bindings[2] = up_ptrs[expert];
    bindings[3] = up_scale_ptrs[expert];
    bindings[4] = down_ptrs[expert];
    bindings[5] = down_scale_ptrs[expert];

    const std::size_t binding_base = static_cast<std::size_t>(group) * 6;
#pragma unroll
    for (int pointer = 0; pointer < 6; ++pointer) {
      workspace
          .group_bindings[binding_base + static_cast<std::size_t>(pointer)] =
          bindings[pointer];
    }
  }
  __syncthreads();

  const std::size_t input_row_bytes = args.input_size / 2;
  const std::uint32_t input_scale_columns = args.input_size / kScaleVectorSize;
  auto *group_sfa_bytes =
      reinterpret_cast<std::uint8_t *>(workspace.gathered_x_sfa) +
      static_cast<std::size_t>(group) * workspace.gathered_x_sfa_group_stride;
  const auto *input = device_pointer<std::uint8_t const>(args.input_packed);
  const auto *input_scales =
      device_pointer<std::uint8_t const>(args.input_scales);
  const std::size_t vector_count =
      static_cast<std::size_t>(route_count) * input_scale_columns;
  for (std::size_t vector = threadIdx.x; vector < vector_count;
       vector += blockDim.x) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(vector / input_scale_columns);
    const std::uint32_t scale_column =
        static_cast<std::uint32_t>(vector % input_scale_columns);
    const std::uint32_t ordinal = route_begin + row;
    const std::int32_t token = tokens[ordinal];
    const auto *source = reinterpret_cast<uint4 const *>(
        input + static_cast<std::size_t>(token) * input_row_bytes +
        static_cast<std::size_t>(scale_column) * 16);
    auto *destination = reinterpret_cast<uint4 *>(
        workspace.gathered_x +
        static_cast<std::size_t>(ordinal) * input_row_bytes +
        static_cast<std::size_t>(scale_column) * 16);
    *destination = *source;
  }

  const auto sfa_layout = BlockScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(static_cast<int>(route_count), 1,
                       static_cast<int>(args.input_size), 1));
  for (std::size_t scale = threadIdx.x; scale < vector_count;
       scale += blockDim.x) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(scale / input_scale_columns);
    const std::uint32_t scale_column =
        static_cast<std::uint32_t>(scale % input_scale_columns);
    const std::uint32_t ordinal = route_begin + row;
    const std::int32_t token = tokens[ordinal];
    const auto destination = sfa_layout(
        cute::make_coord(static_cast<int>(row),
                         static_cast<int>(scale_column * kScaleVectorSize), 0));
    group_sfa_bytes[static_cast<std::size_t>(destination)] =
        input_scales[static_cast<std::size_t>(token) * input_scale_columns +
                     scale_column];
    workspace.route_groups[ordinal] = group;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const std::int32_t m = static_cast<std::int32_t>(route_count);
    auto *a_bytes = workspace.gathered_x +
                    static_cast<std::size_t>(route_begin) * input_row_bytes;
    auto *a = reinterpret_cast<ElementA const *>(a_bytes);
    auto *sfa = reinterpret_cast<ElementScale const *>(group_sfa_bytes);
    auto *gate_d =
        workspace.gate_up_f32 +
        static_cast<std::size_t>(route_begin) * args.intermediate_size;
    auto *up_d =
        workspace.gate_up_f32 +
        (static_cast<std::size_t>(args.total_routed_rows) + route_begin) *
            args.intermediate_size;
    const auto *gate_b = reinterpret_cast<ElementB const *>(
        static_cast<std::uintptr_t>(bindings[0]));
    const auto *gate_sfb = reinterpret_cast<ElementScale const *>(
        static_cast<std::uintptr_t>(bindings[1]));
    const auto *up_b = reinterpret_cast<ElementB const *>(
        static_cast<std::uintptr_t>(bindings[2]));
    const auto *up_sfb = reinterpret_cast<ElementScale const *>(
        static_cast<std::uintptr_t>(bindings[3]));

    const std::uint32_t bucket_groups =
        group < args.small_group_count
            ? args.small_group_count
            : args.active_group_count - args.small_group_count;
    const std::uint32_t local_group =
        group < args.small_group_count ? group : group - args.small_group_count;
    const std::uint32_t bucket_base =
        group < args.small_group_count ? 0 : args.small_group_count * 2;
    const std::int32_t gate_index =
        static_cast<std::int32_t>(bucket_base + local_group);
    const std::int32_t up_index =
        static_cast<std::int32_t>(bucket_base + bucket_groups + local_group);
    write_descriptor(workspace.descriptors, gate_index, m,
                     static_cast<std::int32_t>(args.intermediate_size),
                     static_cast<std::int32_t>(args.input_size), a, gate_b,
                     gate_d, sfa, gate_sfb);
    write_descriptor(workspace.descriptors, up_index, m,
                     static_cast<std::int32_t>(args.intermediate_size),
                     static_cast<std::int32_t>(args.input_size), a, up_b, up_d,
                     sfa, up_sfb);
  }
}

__device__ __forceinline__ float e8m0_value(std::uint8_t encoded) {
  const std::uint32_t bits =
      encoded == 0 ? (1u << 22) : (static_cast<std::uint32_t>(encoded) << 23);
  return __uint_as_float(bits);
}

__device__ __forceinline__ std::uint8_t e8m0_for_amax(float amax) {
  if (amax <= 0.0f) {
    return kScalePadding;
  }
  const float exponent_value = ceilf(log2f(amax / 6.0f));
  if (!isfinite(exponent_value) || exponent_value < -127.0f) {
    return 0;
  }
  const int encoded = static_cast<int>(exponent_value) + 127;
  return static_cast<std::uint8_t>(
      encoded < 0 ? 0 : (encoded > 254 ? 254 : encoded));
}

__device__ __forceinline__ float fp4_value(std::uint8_t value) {
  constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                   2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = magnitudes[value & 7u];
  return (value & 8u) != 0 ? -magnitude : magnitude;
}

__device__ __forceinline__ std::uint8_t quantize_fp4(float value) {
  if (value == 0.0f) {
    return 0;
  }
  const std::uint8_t sign = value < 0.0f ? 8u : 0u;
  const float magnitude = fminf(fabsf(value), 6.0f);
  std::uint8_t best = 0;
  float best_error = magnitude;
#pragma unroll
  for (std::uint8_t candidate = 1; candidate < 8; ++candidate) {
    const float error = fabsf(fp4_value(candidate) - magnitude);
    if (error < best_error) {
      best = candidate;
      best_error = error;
    }
  }
  return static_cast<std::uint8_t>(sign | best);
}

__device__ __forceinline__ float swiglu(float gate, float up, float limit) {
  if (limit > 0.0f) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
  }
  return gate * (1.0f / (1.0f + __expf(-gate))) * up;
}

__global__ __launch_bounds__(kQuantThreads) void swiglu_requant_kernel(
    GroupedFp4MoeArgs args, WorkspaceView workspace) {
  const std::size_t scale_columns = args.intermediate_size / kScaleVectorSize;
  const std::size_t row_scale = blockIdx.x;
  const std::size_t routed_row = row_scale / scale_columns;
  if (routed_row >= args.total_routed_rows) {
    return;
  }
  const std::uint32_t scale_column =
      static_cast<std::uint32_t>(row_scale % scale_columns);
  const std::uint32_t group = workspace.route_groups[routed_row];

  const auto *indptr =
      device_pointer<std::uint32_t const>(args.expert_route_indptr);
  const auto *counts =
      device_pointer<std::uint32_t const>(args.expert_route_counts);
  const std::uint32_t route_begin = indptr[group];
  const std::uint32_t route_count = counts[group];
  const std::uint32_t row =
      static_cast<std::uint32_t>(routed_row) - route_begin;
  const std::uint32_t channel = scale_column * kScaleVectorSize + threadIdx.x;
  const std::size_t value_index = routed_row * args.intermediate_size + channel;
  const std::size_t up_base =
      static_cast<std::size_t>(args.total_routed_rows) * args.intermediate_size;
  float value =
      swiglu(workspace.gate_up_f32[value_index],
             workspace.gate_up_f32[up_base + value_index], args.swiglu_limit);
  const unsigned invalid_mask = __ballot_sync(0xffffffffu, !isfinite(value));
  if (invalid_mask != 0) {
    value = 0.0f;
    if (threadIdx.x == 0) {
      set_route_error(args);
    }
  }

  float amax = fabsf(value);
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta));
  }
  std::uint32_t scale_byte = e8m0_for_amax(amax);
  scale_byte = __shfl_sync(0xffffffffu, scale_byte, 0);
  const float reciprocal_scale =
      1.0f / e8m0_value(static_cast<std::uint8_t>(scale_byte));
  const std::uint32_t nibble = quantize_fp4(value * reciprocal_scale);
  std::uint32_t low = 0;
  std::uint32_t high = 0;
  if (threadIdx.x < 16) {
    low = __shfl_sync(0xffffffffu, nibble, threadIdx.x * 2);
    high = __shfl_sync(0xffffffffu, nibble, threadIdx.x * 2 + 1);
  }

  const std::size_t hidden_row_bytes = args.intermediate_size / 2;
  if (threadIdx.x < 16) {
    workspace.hidden_packed[routed_row * hidden_row_bytes +
                            static_cast<std::size_t>(scale_column) * 16 +
                            threadIdx.x] =
        static_cast<std::uint8_t>(low | (high << 4));
  }
  if (threadIdx.x == 0) {
    auto *sfa_group =
        reinterpret_cast<std::uint8_t *>(workspace.hidden_sfa) +
        static_cast<std::size_t>(group) * workspace.hidden_sfa_group_stride;
    const auto layout =
        BlockScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(
            static_cast<int>(route_count), static_cast<int>(args.hidden_size),
            static_cast<int>(args.intermediate_size), 1));
    const auto destination = layout(
        cute::make_coord(static_cast<int>(row),
                         static_cast<int>(scale_column * kScaleVectorSize), 0));
    sfa_group[static_cast<std::size_t>(destination)] =
        static_cast<std::uint8_t>(scale_byte);
  }
}

__global__ __launch_bounds__(kPrepareThreads) void prepare_down_kernel(
    GroupedFp4MoeArgs args, WorkspaceView workspace) {
  const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= args.active_group_count) {
    return;
  }

  const auto *indptr =
      device_pointer<std::uint32_t const>(args.expert_route_indptr);
  const auto *counts =
      device_pointer<std::uint32_t const>(args.expert_route_counts);
  const std::uint32_t route_begin = indptr[group];
  const std::uint32_t route_count = counts[group];
  const std::int32_t m = static_cast<std::int32_t>(route_count);
  const std::size_t hidden_row_bytes = args.intermediate_size / 2;
  auto *a_bytes = workspace.hidden_packed +
                  static_cast<std::size_t>(route_begin) * hidden_row_bytes;
  auto *sfa_bytes =
      reinterpret_cast<std::uint8_t *>(workspace.hidden_sfa) +
      static_cast<std::size_t>(group) * workspace.hidden_sfa_group_stride;
  auto *a = reinterpret_cast<ElementA const *>(a_bytes);
  auto *sfa = reinterpret_cast<ElementScale const *>(sfa_bytes);
  auto *d = workspace.down_f32 +
            static_cast<std::size_t>(route_begin) * args.hidden_size;
  const std::size_t binding_base = static_cast<std::size_t>(group) * 6;
  const auto *b = reinterpret_cast<ElementB const *>(
      static_cast<std::uintptr_t>(workspace.group_bindings[binding_base + 4]));
  const auto *sfb = reinterpret_cast<ElementScale const *>(
      static_cast<std::uintptr_t>(workspace.group_bindings[binding_base + 5]));
  write_descriptor(workspace.descriptors, static_cast<std::int32_t>(group), m,
                   static_cast<std::int32_t>(args.hidden_size),
                   static_cast<std::int32_t>(args.intermediate_size), a, b, d,
                   sfa, sfb);
}

__global__ __launch_bounds__(kScatterThreads) void scatter_kernel(
    GroupedFp4MoeArgs args, WorkspaceView workspace) {
  const std::size_t routed_row = blockIdx.x;
  if (routed_row >= args.total_routed_rows) {
    return;
  }
  const auto *routes = device_pointer<std::int32_t const>(args.route_indices);
  const std::int32_t route = routes[routed_row];
  if (route < 0 || static_cast<std::uint32_t>(route) >= args.num_routes) {
    if (threadIdx.x == 0) {
      set_route_error(args);
    }
    return;
  }

  const float route_weight =
      device_pointer<float const>(args.route_weights)[routed_row];
  const std::size_t vectors = args.hidden_size / 4;
  const auto *source = reinterpret_cast<float4 const *>(
      workspace.down_f32 + routed_row * args.hidden_size);
  auto *destination = reinterpret_cast<float4 *>(
      device_pointer<float>(args.route_output) +
      static_cast<std::size_t>(route) * args.hidden_size);
  for (std::size_t vector = threadIdx.x; vector < vectors;
       vector += blockDim.x) {
    const float4 value = source[vector];
    destination[vector] =
        make_float4(value.x * route_weight, value.y * route_weight,
                    value.z * route_weight, value.w * route_weight);
  }
  if (threadIdx.x == 0) {
    auto *written = device_pointer<std::int32_t>(args.route_written);
    atomicOr(reinterpret_cast<unsigned int *>(written + route), 1u);
  }
}

inline WorkspaceView
make_workspace_view(void *workspace, WorkspacePlan const &plan,
                    GroupedFp4MoeArgs const &args) noexcept {
  auto *base = static_cast<std::uint8_t *>(workspace);
  DescriptorStorage descriptor_storage{};
  descriptor_storage.data = base + plan.descriptor_offset;
  descriptor_storage.bytes = plan.descriptor_bytes;
  descriptor_storage.groups = plan.descriptor_capacity;

  WorkspaceView view{};
  view.descriptors = descriptor_storage.view();
  view.group_bindings =
      reinterpret_cast<std::uint64_t *>(base + plan.group_bindings_offset);
  view.route_groups =
      reinterpret_cast<std::uint32_t *>(base + plan.route_groups_offset);
  view.gathered_x = base + plan.gathered_x_offset;
  view.gathered_x_sfa =
      reinterpret_cast<ElementScale *>(base + plan.gathered_x_sfa_offset);
  view.gate_up_f32 = reinterpret_cast<float *>(base + plan.gate_up_f32_offset);
  view.hidden_packed = base + plan.hidden_packed_offset;
  view.hidden_sfa =
      reinterpret_cast<ElementScale *>(base + plan.hidden_sfa_offset);
  view.down_f32 = reinterpret_cast<float *>(base + plan.down_f32_offset);
  view.gathered_x_sfa_group_stride = plan.gathered_x_sfa_group_stride;
  view.hidden_sfa_group_stride = plan.hidden_sfa_group_stride;
  static_cast<void>(args);
  return view;
}

inline ::cutlass::Status
run_grouped(MmaMode mode, DeviceDescriptorView descriptors,
            LaunchOptions const &options, void *workspace,
            std::size_t workspace_bytes, cudaStream_t stream) noexcept {
  if (descriptors.groups == 0) {
    return ::cutlass::Status::kSuccess;
  }
  GroupedProblem problem{};
  problem.descriptors = descriptors;
  problem.device_id = options.device_id;
  problem.sm_count = options.sm_count;
  return mode == MmaMode::k1Sm
             ? run_1sm(problem, workspace, workspace_bytes, stream)
             : run_2sm(problem, workspace, workspace_bytes, stream);
}

inline std::size_t
required_cutlass_bytes(GroupedFp4MoeArgs const &args,
                       LaunchOptions const &options) noexcept {
  const std::uint32_t small = args.small_group_count;
  const std::uint32_t large = args.active_group_count - small;
  std::size_t result = 0;
  const std::size_t sizes[] = {
      cutlass_bytes_for(MmaMode::k1Sm, small * 2, options),
      cutlass_bytes_for(MmaMode::k1Sm, small, options),
      cutlass_bytes_for(MmaMode::k2Sm, large * 2, options),
      cutlass_bytes_for(MmaMode::k2Sm, large, options)};
  for (std::size_t size : sizes) {
    result = size > result ? size : result;
  }
  return result;
}

} // namespace moe_detail

inline WorkspacePlan
workspace_plan(GroupedFp4MoeArgs const &args,
                      LaunchOptions const &options) noexcept {
  WorkspacePlan plan{};
  if (!moe_detail::scalar_args_valid(args, options)) {
    return plan;
  }

  const std::size_t groups = args.active_group_count;
  const std::size_t rows = args.total_routed_rows;
  const std::size_t descriptor_capacity = groups * 2;
  if (descriptor_capacity > 0x7fffffffu) {
    return {};
  }
  plan.descriptor_capacity = static_cast<std::int32_t>(descriptor_capacity);
  plan.descriptor_bytes = descriptor_bytes(plan.descriptor_capacity);
  plan.gathered_x_sfa_group_stride = prepared_sfa_bytes(
      static_cast<int>(args.max_group_rows), static_cast<int>(args.input_size));
  plan.hidden_sfa_group_stride =
      prepared_sfa_bytes(static_cast<int>(args.max_group_rows),
                         static_cast<int>(args.intermediate_size));
  if (plan.descriptor_bytes == 0 || plan.gathered_x_sfa_group_stride == 0 ||
      plan.hidden_sfa_group_stride == 0 ||
      !moe_detail::checked_product(plan.group_bindings_bytes, groups, 6,
                                      sizeof(std::uint64_t)) ||
      !moe_detail::checked_product(plan.route_groups_bytes, rows,
                                      sizeof(std::uint32_t)) ||
      !moe_detail::checked_product(plan.gathered_x_bytes, rows,
                                      args.input_size / 2) ||
      !moe_detail::checked_product(plan.gathered_x_sfa_bytes, groups,
                                      plan.gathered_x_sfa_group_stride) ||
      !moe_detail::checked_product(plan.gate_up_f32_bytes, 2, rows,
                                      args.intermediate_size, sizeof(float)) ||
      !moe_detail::checked_product(plan.hidden_packed_bytes, rows,
                                      args.intermediate_size / 2) ||
      !moe_detail::checked_product(plan.hidden_sfa_bytes, groups,
                                      plan.hidden_sfa_group_stride) ||
      !moe_detail::checked_product(plan.down_f32_bytes, rows,
                                      args.hidden_size, sizeof(float))) {
    return {};
  }
  plan.cutlass_bytes = moe_detail::required_cutlass_bytes(args, options);

  std::size_t cursor = 0;
  if (!moe_detail::append_region(cursor, plan.descriptor_bytes,
                                    kDescriptorAlignment,
                                    plan.descriptor_offset) ||
      !moe_detail::append_region(cursor, plan.group_bindings_bytes,
                                    alignof(std::uint64_t),
                                    plan.group_bindings_offset) ||
      !moe_detail::append_region(cursor, plan.route_groups_bytes,
                                    alignof(std::uint32_t),
                                    plan.route_groups_offset) ||
      !moe_detail::append_region(cursor, plan.gathered_x_bytes, 16,
                                    plan.gathered_x_offset) ||
      !moe_detail::append_region(cursor, plan.gathered_x_sfa_bytes, 16,
                                    plan.gathered_x_sfa_offset) ||
      !moe_detail::append_region(cursor, plan.gate_up_f32_bytes, 16,
                                    plan.gate_up_f32_offset) ||
      !moe_detail::append_region(cursor, plan.hidden_packed_bytes, 16,
                                    plan.hidden_packed_offset) ||
      !moe_detail::append_region(cursor, plan.hidden_sfa_bytes, 16,
                                    plan.hidden_sfa_offset) ||
      !moe_detail::append_region(cursor, plan.down_f32_bytes, 16,
                                    plan.down_f32_offset) ||
      !moe_detail::append_region(cursor, plan.cutlass_bytes,
                                    kCutlassWorkspaceAlignment,
                                    plan.cutlass_offset)) {
    return {};
  }
  if (cursor > (std::numeric_limits<std::size_t>::max)() -
                   (kWorkspaceAlignment - 1)) {
    return {};
  }
  plan.total_bytes = detail::align_up(cursor, kWorkspaceAlignment);
  return plan;
}

inline std::size_t
workspace_bytes(GroupedFp4MoeArgs const &args,
                       LaunchOptions const &options) noexcept {
  return workspace_plan(args, options).total_bytes;
}

inline Status
can_implement(GroupedFp4MoeArgs const *args, void *workspace,
                     std::size_t workspace_bytes,
                     LaunchOptions const &options) noexcept {
  if (args == nullptr || !moe_detail::scalar_args_valid(*args, options) ||
      !moe_detail::pointer_args_valid(*args)) {
    return Status::kInvalidArgument;
  }
  const WorkspacePlan plan = workspace_plan(*args, options);
  if (!plan.valid() || workspace_bytes < plan.total_bytes ||
      !moe_detail::aligned_pointer(workspace, kWorkspaceAlignment)) {
    return Status::kUnsupportedResources;
  }
  return Status::kSuccess;
}

// The caller owns workspace and stream. Weight scale pointers must reference
// grouped_fp4_moe::launch_prepare_sfb output for gate/up [intermediate,K]
// and down [hidden,intermediate]. route_written and route_error are not cleared
// by this helper, allowing composition with a larger routed operation.
inline Status launch(GroupedFp4MoeArgs const *args, void *workspace,
                                  std::size_t workspace_bytes,
                                  cudaStream_t stream,
                                  LaunchOptions const &options) noexcept {
  const Status validation =
      can_implement(args, workspace, workspace_bytes, options);
  if (validation != Status::kSuccess) {
    return validation;
  }
  const WorkspacePlan plan = workspace_plan(*args, options);

  moe_detail::WorkspaceView view =
      moe_detail::make_workspace_view(workspace, plan, *args);
  if (view.descriptors.groups != plan.descriptor_capacity) {
    return Status::kUnsupportedResources;
  }
  auto *base = static_cast<std::uint8_t *>(workspace);
  void *cutlass_workspace = base + plan.cutlass_offset;

  cudaError_t cuda_status =
      cudaMemsetAsync(view.route_groups, 0xff, plan.route_groups_bytes, stream);
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemsetAsync(view.gathered_x_sfa, kScalePadding,
                                  plan.gathered_x_sfa_bytes, stream);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemsetAsync(view.hidden_sfa, kScalePadding,
                                  plan.hidden_sfa_bytes, stream);
  }
  if (cuda_status != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  moe_detail::prepare_gate_up_kernel<<<args->active_group_count,
                                          kPrepareThreads, 0, stream>>>(
      *args, options, view);
  if (cudaGetLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  const std::uint32_t small = args->small_group_count;
  const std::uint32_t large = args->active_group_count - small;
  ::cutlass::Status cutlass_status = ::cutlass::Status::kSuccess;
  if (small != 0) {
    DeviceDescriptorView small_gate_up = moe_detail::slice_descriptors(
        view.descriptors, 0, static_cast<std::int32_t>(small * 2));
    cutlass_status = moe_detail::run_grouped(MmaMode::k1Sm, small_gate_up,
                                                options, cutlass_workspace,
                                                plan.cutlass_bytes, stream);
  }
  if (cutlass_status == ::cutlass::Status::kSuccess && large != 0) {
    DeviceDescriptorView large_gate_up = moe_detail::slice_descriptors(
        view.descriptors, static_cast<std::int32_t>(small * 2),
        static_cast<std::int32_t>(large * 2));
    cutlass_status = moe_detail::run_grouped(MmaMode::k2Sm, large_gate_up,
                                                options, cutlass_workspace,
                                                plan.cutlass_bytes, stream);
  }
  if (cutlass_status != ::cutlass::Status::kSuccess) {
    return Status::kLaunchFailed;
  }

  std::size_t quant_blocks = 0;
  if (!moe_detail::checked_product(quant_blocks, args->total_routed_rows,
                                      args->intermediate_size /
                                          kScaleVectorSize) ||
      quant_blocks == 0 || quant_blocks > 0x7fffffffu) {
    return Status::kUnsupportedResources;
  }
  moe_detail::swiglu_requant_kernel<<<static_cast<unsigned>(quant_blocks),
                                         kQuantThreads, 0, stream>>>(
      *args, view);
  if (cudaGetLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  const std::uint32_t down_blocks =
      (args->active_group_count + kPrepareThreads - 1) /
      kPrepareThreads;
  moe_detail::
      prepare_down_kernel<<<down_blocks, kPrepareThreads, 0, stream>>>(
          *args, view);
  if (cudaGetLastError() != cudaSuccess) {
    return Status::kLaunchFailed;
  }

  cutlass_status = ::cutlass::Status::kSuccess;
  if (small != 0) {
    DeviceDescriptorView small_down = moe_detail::slice_descriptors(
        view.descriptors, 0, static_cast<std::int32_t>(small));
    cutlass_status = moe_detail::run_grouped(MmaMode::k1Sm, small_down,
                                                options, cutlass_workspace,
                                                plan.cutlass_bytes, stream);
  }
  if (cutlass_status == ::cutlass::Status::kSuccess && large != 0) {
    DeviceDescriptorView large_down = moe_detail::slice_descriptors(
        view.descriptors, static_cast<std::int32_t>(small),
        static_cast<std::int32_t>(large));
    cutlass_status = moe_detail::run_grouped(MmaMode::k2Sm, large_down,
                                                options, cutlass_workspace,
                                                plan.cutlass_bytes, stream);
  }
  if (cutlass_status != ::cutlass::Status::kSuccess) {
    return Status::kLaunchFailed;
  }

  moe_detail::scatter_kernel<<<args->total_routed_rows,
                                  kScatterThreads, 0, stream>>>(*args,
                                                                      view);
  return cudaGetLastError() == cudaSuccess ? Status::kSuccess
                                           : Status::kLaunchFailed;
}

} // namespace ferrule::cuda::cutlass::architectures::sm103::grouped_fp4_moe

#endif // FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4
