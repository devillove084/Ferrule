#pragma once

#include "../profile.cuh"

#if FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4

#if !defined(__CUDACC__)
#error "SM103 grouped FP4 MoE support must be compiled with nvcc"
#endif

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#pragma nv_diag_push
#pragma nv_diag_suppress 20012

#include <cute/tensor.hpp>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma_sm100.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/sm103_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/version.h>

#pragma nv_diag_pop

#if CUTLASS_VERSION != 461
#error "Ferrule's SM103 grouped FP4 MoE operator requires CUTLASS 4.6.1"
#endif

namespace ferrule::cuda::cutlass::architectures::sm103::grouped_fp4_moe {

inline constexpr int kScaleVectorSize = 32;
inline constexpr std::size_t kDescriptorAlignment = 16;
inline constexpr std::size_t kCutlassWorkspaceAlignment = 256;
inline constexpr std::uint8_t kScalePadding = 0x7f;
inline constexpr std::size_t kInvalidScaleIndex =
    (std::numeric_limits<std::size_t>::max)();

using ElementPairA = ::cutlass::mx_float8_t<::cutlass::float_e4m3_t>;
using ElementPairB = ::cutlass::mx_float4_t<::cutlass::float_e2m1_t>;
using ElementScale = ::cutlass::float_ue8m0_t;
using ElementD = float;
using LayoutA = ::cutlass::layout::RowMajor;
using LayoutB = ::cutlass::layout::ColumnMajor;
using LayoutD = ::cutlass::layout::RowMajor;
using ProblemShapeValue = cute::Shape<int, int, int>;
using ProblemShape = ::cutlass::gemm::GroupProblemShape<ProblemShapeValue>;

inline constexpr int kAlignmentA = 16;
inline constexpr int kAlignmentB = 128;
inline constexpr int kAlignmentD = 4;

using TileShape1Sm = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape1Sm = cute::Shape<cute::_2, cute::_4, cute::_1>;
using Epilogue1Sm = typename ::cutlass::epilogue::collective::CollectiveBuilder<
    ::cutlass::arch::Sm100, ::cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape1Sm, ClusterShape1Sm, cute::Shape<cute::_128, cute::_64>, float,
    float, void, LayoutD *, kAlignmentD, ElementD, LayoutD *, kAlignmentD,
    ::cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
using Mainloop1Sm = typename ::cutlass::gemm::collective::CollectiveBuilder<
    ::cutlass::arch::Sm100, ::cutlass::arch::OpClassBlockScaledTensorOp,
    ElementPairA, LayoutA *, kAlignmentA, ElementPairB, LayoutB *, kAlignmentB,
    float, TileShape1Sm, ClusterShape1Sm,
    ::cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename Epilogue1Sm::SharedStorage))>,
    ::cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100>::
    CollectiveOp;
using Kernel1Sm =
    ::cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop1Sm,
                                           Epilogue1Sm>;
using GroupedGemm1Sm = ::cutlass::gemm::device::GemmUniversalAdapter<Kernel1Sm>;

using TileShape2Sm = cute::Shape<cute::_256, cute::_256, cute::_128>;
using ClusterShape2Sm = cute::Shape<cute::_4, cute::_2, cute::_1>;
using Epilogue2Sm = typename ::cutlass::epilogue::collective::CollectiveBuilder<
    ::cutlass::arch::Sm100, ::cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape2Sm, ClusterShape2Sm, cute::Shape<cute::_128, cute::_64>, float,
    float, void, LayoutD *, kAlignmentD, ElementD, LayoutD *, kAlignmentD,
    ::cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>::CollectiveOp;
using Mainloop2Sm = typename ::cutlass::gemm::collective::CollectiveBuilder<
    ::cutlass::arch::Sm100, ::cutlass::arch::OpClassBlockScaledTensorOp,
    ElementPairA, LayoutA *, kAlignmentA, ElementPairB, LayoutB *, kAlignmentB,
    float, TileShape2Sm, ClusterShape2Sm,
    ::cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename Epilogue2Sm::SharedStorage))>,
    ::cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100>::
    CollectiveOp;
using Kernel2Sm =
    ::cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop2Sm,
                                           Epilogue2Sm>;
using GroupedGemm2Sm = ::cutlass::gemm::device::GemmUniversalAdapter<Kernel2Sm>;

using ElementA = typename Mainloop1Sm::ElementA;
using ElementB = typename Mainloop1Sm::ElementB;
using StrideA = typename Kernel1Sm::InternalStrideA;
using StrideB = typename Kernel1Sm::InternalStrideB;
using StrideD = typename Kernel1Sm::InternalStrideD;
using LayoutSFA = typename Mainloop1Sm::InternalLayoutSFA;
using LayoutSFB = typename Mainloop1Sm::InternalLayoutSFB;
using BlockScaleConfig = typename Mainloop1Sm::Sm1xxBlkScaledConfig;

static_assert(std::is_same_v<StrideA, typename Kernel2Sm::InternalStrideA>);
static_assert(std::is_same_v<StrideB, typename Kernel2Sm::InternalStrideB>);
static_assert(std::is_same_v<StrideD, typename Kernel2Sm::InternalStrideD>);
static_assert(
    std::is_same_v<LayoutSFA, typename Mainloop2Sm::InternalLayoutSFA>);
static_assert(
    std::is_same_v<LayoutSFB, typename Mainloop2Sm::InternalLayoutSFB>);
static_assert(std::is_trivially_copyable_v<ProblemShapeValue>);
static_assert(std::is_trivially_copyable_v<StrideA>);
static_assert(std::is_trivially_copyable_v<StrideB>);
static_assert(std::is_trivially_copyable_v<StrideD>);
static_assert(std::is_trivially_copyable_v<LayoutSFA>);
static_assert(std::is_trivially_copyable_v<LayoutSFB>);
static_assert(sizeof(ElementScale) == 1);
static_assert(alignof(ProblemShapeValue) <= kDescriptorAlignment);
static_assert(alignof(StrideA) <= kDescriptorAlignment);
static_assert(alignof(StrideB) <= kDescriptorAlignment);
static_assert(alignof(StrideD) <= kDescriptorAlignment);
static_assert(alignof(LayoutSFA) <= kDescriptorAlignment);
static_assert(alignof(LayoutSFB) <= kDescriptorAlignment);

// All members point into one caller-owned descriptor allocation. For a GEMM
// launch, the allocation and every pointee stored in its pointer arrays must be
// device-accessible.
struct DeviceDescriptorView {
  std::int32_t groups{};
  ProblemShapeValue *problem_shapes{};
  ElementA const **a{};
  ElementB const **b{};
  ElementD **d{};
  ElementScale const **sfa{};
  ElementScale const **sfb{};
  StrideA *stride_a{};
  StrideB *stride_b{};
  StrideD *stride_d{};
  LayoutSFA *layout_sfa{};
  LayoutSFB *layout_sfb{};
};

struct DescriptorStorage {
  void *data{};
  std::size_t bytes{};
  std::int32_t groups{};

  bool valid() const noexcept;
  DeviceDescriptorView view() const noexcept;
};

struct GroupedProblem {
  DeviceDescriptorView descriptors{};
  // Optional host mirror used only by CUTLASS schedulers that request it.
  // Ferrule builds descriptors on device, so production launches leave this
  // null.
  ProblemShapeValue const *host_problem_shapes{};
  std::int32_t device_id{};
  std::int32_t sm_count{};
};

enum class MmaMode : std::uint8_t { k1Sm, k2Sm };

static_assert(std::is_standard_layout_v<DeviceDescriptorView>);
static_assert(std::is_trivially_copyable_v<DeviceDescriptorView>);
static_assert(std::is_standard_layout_v<DescriptorStorage>);
static_assert(std::is_trivially_copyable_v<DescriptorStorage>);
static_assert(std::is_standard_layout_v<GroupedProblem>);
static_assert(std::is_trivially_copyable_v<GroupedProblem>);

namespace detail {

constexpr std::size_t align_up(std::size_t value,
                               std::size_t alignment) noexcept {
  return (value + alignment - 1) & ~(alignment - 1);
}

template <class T>
inline bool append_array_bytes(std::size_t &offset,
                               std::size_t count) noexcept {
  if (offset > (std::numeric_limits<std::size_t>::max)() - (alignof(T) - 1)) {
    return false;
  }
  offset = align_up(offset, alignof(T));
  if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
    return false;
  }
  const std::size_t array_bytes = count * sizeof(T);
  if (offset > (std::numeric_limits<std::size_t>::max)() - array_bytes) {
    return false;
  }
  offset += array_bytes;
  return true;
}

template <class T>
inline T *take_array(std::uint8_t *base, std::size_t &offset,
                     std::size_t count) noexcept {
  offset = align_up(offset, alignof(T));
  T *result = reinterpret_cast<T *>(base + offset);
  offset += count * sizeof(T);
  return result;
}

inline bool
descriptors_valid(DeviceDescriptorView const &descriptors) noexcept {
  return descriptors.groups > 0 && descriptors.problem_shapes != nullptr &&
         descriptors.a != nullptr && descriptors.b != nullptr &&
         descriptors.d != nullptr && descriptors.sfa != nullptr &&
         descriptors.sfb != nullptr && descriptors.stride_a != nullptr &&
         descriptors.stride_b != nullptr && descriptors.stride_d != nullptr &&
         descriptors.layout_sfa != nullptr && descriptors.layout_sfb != nullptr;
}

template <class Gemm>
inline typename Gemm::Arguments
make_cutlass_arguments(GroupedProblem const &problem) noexcept {
  typename Gemm::Arguments arguments{};
  arguments.mode = ::cutlass::gemm::GemmUniversalMode::kGrouped;
  arguments.problem_shape.num_groups = problem.descriptors.groups;
  arguments.problem_shape.problem_shapes = problem.descriptors.problem_shapes;
  arguments.problem_shape.host_problem_shapes = problem.host_problem_shapes;

  arguments.mainloop.ptr_A = problem.descriptors.a;
  arguments.mainloop.dA = problem.descriptors.stride_a;
  arguments.mainloop.ptr_B = problem.descriptors.b;
  arguments.mainloop.dB = problem.descriptors.stride_b;
  arguments.mainloop.ptr_SFA = problem.descriptors.sfa;
  arguments.mainloop.layout_SFA = problem.descriptors.layout_sfa;
  arguments.mainloop.ptr_SFB = problem.descriptors.sfb;
  arguments.mainloop.layout_SFB = problem.descriptors.layout_sfb;

  arguments.epilogue.ptr_C = nullptr;
  arguments.epilogue.dC = nullptr;
  arguments.epilogue.ptr_D = problem.descriptors.d;
  arguments.epilogue.dD = problem.descriptors.stride_d;

  arguments.hw_info.device_id = problem.device_id;
  arguments.hw_info.sm_count = problem.sm_count;
  return arguments;
}

inline bool problem_valid(GroupedProblem const &problem) noexcept {
  return descriptors_valid(problem.descriptors) && problem.device_id >= 0 &&
         problem.sm_count > 0;
}

template <class Gemm>
inline std::size_t workspace_bytes(GroupedProblem const &problem) noexcept {
  if (!problem_valid(problem)) {
    return 0;
  }
  return Gemm::get_workspace_size(make_cutlass_arguments<Gemm>(problem));
}

template <class Gemm>
inline ::cutlass::Status run(GroupedProblem const &problem, void *workspace,
                             std::size_t workspace_bytes,
                             cudaStream_t stream) noexcept {
  if (!problem_valid(problem)) {
    return ::cutlass::Status::kInvalid;
  }

  auto arguments = make_cutlass_arguments<Gemm>(problem);
  const std::size_t required = Gemm::get_workspace_size(arguments);
  if (workspace_bytes < required) {
    return ::cutlass::Status::kInvalid;
  }
  if (required != 0 && workspace == nullptr) {
    return ::cutlass::Status::kErrorWorkspaceNull;
  }
  if (required != 0 && (reinterpret_cast<std::uintptr_t>(workspace) &
                        (kCutlassWorkspaceAlignment - 1)) != 0) {
    return ::cutlass::Status::kInvalid;
  }

  ::cutlass::Status status = Gemm::can_implement(arguments);
  if (status != ::cutlass::Status::kSuccess) {
    return status;
  }

  Gemm gemm;
  status = gemm.initialize(arguments, workspace, stream);
  if (status != ::cutlass::Status::kSuccess) {
    return status;
  }
  return gemm.run(stream);
}

static __global__ void prepare_sfb_kernel(std::uint8_t *destination,
                                          std::uint8_t const *linear_source,
                                          std::int32_t n, std::int32_t k) {
  const std::uint64_t scale_columns =
      static_cast<std::uint64_t>(k / kScaleVectorSize);
  const std::uint64_t count = static_cast<std::uint64_t>(n) * scale_columns;
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t step = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  const auto layout =
      BlockScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(1, n, k, 1));

  for (std::uint64_t linear_index = first; linear_index < count;
       linear_index += step) {
    const int row = static_cast<int>(linear_index / scale_columns);
    const int scale_column = static_cast<int>(linear_index % scale_columns);
    const auto destination_index =
        layout(cute::make_coord(row, scale_column * kScaleVectorSize, 0));
    destination[static_cast<std::size_t>(destination_index)] =
        linear_source[linear_index];
  }
}

} // namespace detail

inline std::size_t descriptor_bytes(std::int32_t groups) noexcept {
  if (groups <= 0) {
    return 0;
  }

  const std::size_t count = static_cast<std::size_t>(groups);
  std::size_t bytes = 0;
  const bool valid =
      detail::append_array_bytes<ProblemShapeValue>(bytes, count) &&
      detail::append_array_bytes<ElementA const *>(bytes, count) &&
      detail::append_array_bytes<ElementB const *>(bytes, count) &&
      detail::append_array_bytes<ElementD *>(bytes, count) &&
      detail::append_array_bytes<ElementScale const *>(bytes, count) &&
      detail::append_array_bytes<ElementScale const *>(bytes, count) &&
      detail::append_array_bytes<StrideA>(bytes, count) &&
      detail::append_array_bytes<StrideB>(bytes, count) &&
      detail::append_array_bytes<StrideD>(bytes, count) &&
      detail::append_array_bytes<LayoutSFA>(bytes, count) &&
      detail::append_array_bytes<LayoutSFB>(bytes, count);
  return valid ? bytes : 0;
}

inline bool DescriptorStorage::valid() const noexcept {
  const std::size_t required = descriptor_bytes(groups);
  return data != nullptr && required != 0 && bytes >= required &&
         (reinterpret_cast<std::uintptr_t>(data) &
          (kDescriptorAlignment - 1)) == 0;
}

inline DeviceDescriptorView DescriptorStorage::view() const noexcept {
  if (!valid()) {
    return {};
  }

  auto *base = static_cast<std::uint8_t *>(data);
  const std::size_t count = static_cast<std::size_t>(groups);
  std::size_t offset = 0;
  DeviceDescriptorView result{};
  result.groups = groups;
  result.problem_shapes =
      detail::take_array<ProblemShapeValue>(base, offset, count);
  result.a = detail::take_array<ElementA const *>(base, offset, count);
  result.b = detail::take_array<ElementB const *>(base, offset, count);
  result.d = detail::take_array<ElementD *>(base, offset, count);
  result.sfa = detail::take_array<ElementScale const *>(base, offset, count);
  result.sfb = detail::take_array<ElementScale const *>(base, offset, count);
  result.stride_a = detail::take_array<StrideA>(base, offset, count);
  result.stride_b = detail::take_array<StrideB>(base, offset, count);
  result.stride_d = detail::take_array<StrideD>(base, offset, count);
  result.layout_sfa = detail::take_array<LayoutSFA>(base, offset, count);
  result.layout_sfb = detail::take_array<LayoutSFB>(base, offset, count);
  return result;
}

inline StrideA packed_stride_a(int m, int k) noexcept {
  return ::cutlass::make_cute_packed_stride(StrideA{},
                                            cute::make_shape(m, k, 1));
}

inline StrideB packed_stride_b(int n, int k) noexcept {
  return ::cutlass::make_cute_packed_stride(StrideB{},
                                            cute::make_shape(n, k, 1));
}

inline StrideD packed_stride_d(int m, int n) noexcept {
  return ::cutlass::make_cute_packed_stride(StrideD{},
                                            cute::make_shape(m, n, 1));
}

inline LayoutSFA make_sfa_layout(int m, int n, int k) noexcept {
  return BlockScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
}

inline LayoutSFB make_sfb_layout(int m, int n, int k) noexcept {
  return BlockScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
}

inline std::size_t prepared_sfa_bytes(int m, int k) noexcept {
  if (m <= 0 || k <= 0 || (k % kScaleVectorSize) != 0) {
    return 0;
  }
  const auto layout = make_sfa_layout(m, 1, k);
  return static_cast<std::size_t>(cute::size(cute::filter_zeros(layout))) *
         sizeof(ElementScale);
}

inline std::size_t prepared_sfb_bytes(int n, int k) noexcept {
  if (n <= 0 || k <= 0 || (k % kScaleVectorSize) != 0) {
    return 0;
  }
  const auto layout = make_sfb_layout(1, n, k);
  return static_cast<std::size_t>(cute::size(cute::filter_zeros(layout))) *
         sizeof(ElementScale);
}

inline std::size_t sfa_layout_index(int m, int k, int row,
                                    int scale_column) noexcept {
  if (row < 0 || row >= m || scale_column < 0 || m <= 0 || k <= 0 ||
      (k % kScaleVectorSize) != 0 || scale_column >= k / kScaleVectorSize) {
    return kInvalidScaleIndex;
  }
  const auto layout = make_sfa_layout(m, 1, k);
  return static_cast<std::size_t>(
      layout(cute::make_coord(row, scale_column * kScaleVectorSize, 0)));
}

inline std::size_t sfb_layout_index(int n, int k, int row,
                                    int scale_column) noexcept {
  if (row < 0 || row >= n || scale_column < 0 || n <= 0 || k <= 0 ||
      (k % kScaleVectorSize) != 0 || scale_column >= k / kScaleVectorSize) {
    return kInvalidScaleIndex;
  }
  const auto layout = make_sfb_layout(1, n, k);
  return static_cast<std::size_t>(
      layout(cute::make_coord(row, scale_column * kScaleVectorSize, 0)));
}

inline std::size_t
cutlass_workspace_bytes(MmaMode mode, GroupedProblem const &problem) noexcept {
  switch (mode) {
  case MmaMode::k1Sm:
    return detail::workspace_bytes<GroupedGemm1Sm>(problem);
  case MmaMode::k2Sm:
    return detail::workspace_bytes<GroupedGemm2Sm>(problem);
  }
  return 0;
}

inline ::cutlass::Status run_1sm(GroupedProblem const &problem, void *workspace,
                                 std::size_t workspace_bytes,
                                 cudaStream_t stream) noexcept {
  return detail::run<GroupedGemm1Sm>(problem, workspace, workspace_bytes,
                                     stream);
}

inline ::cutlass::Status run_2sm(GroupedProblem const &problem, void *workspace,
                                 std::size_t workspace_bytes,
                                 cudaStream_t stream) noexcept {
  return detail::run<GroupedGemm2Sm>(problem, workspace, workspace_bytes,
                                     stream);
}

// linear_source must be either device memory or the device-visible address of
// mapped pinned memory. destination must be device memory with at least
// prepared_sfb_bytes(n, k) bytes. Only scale bytes are written; FP4 weight
// storage is never read or modified.
inline cudaError_t launch_prepare_sfb(std::uint8_t *destination,
                                      std::uint8_t const *linear_source,
                                      std::int32_t n, std::int32_t k,
                                      cudaStream_t stream) noexcept {
  const std::size_t destination_bytes = prepared_sfb_bytes(n, k);
  if (destination == nullptr || linear_source == nullptr ||
      destination_bytes == 0 ||
      (reinterpret_cast<std::uintptr_t>(destination) &
       (kDescriptorAlignment - 1)) != 0) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status =
      cudaMemsetAsync(destination, kScalePadding, destination_bytes, stream);
  if (status != cudaSuccess) {
    return status;
  }

  constexpr std::uint32_t kThreads = 256;
  const std::uint64_t scale_count =
      static_cast<std::uint64_t>(n) *
      static_cast<std::uint64_t>(k / kScaleVectorSize);
  const std::uint64_t block_count = (scale_count + kThreads - 1) / kThreads;
  if (block_count == 0 ||
      block_count > (std::numeric_limits<std::uint32_t>::max)()) {
    return cudaErrorInvalidConfiguration;
  }

  detail::prepare_sfb_kernel<<<static_cast<std::uint32_t>(block_count),
                               kThreads, 0, stream>>>(destination,
                                                      linear_source, n, k);
  return cudaGetLastError();
}

} // namespace ferrule::cuda::cutlass::architectures::sm103::grouped_fp4_moe

#endif // FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4
