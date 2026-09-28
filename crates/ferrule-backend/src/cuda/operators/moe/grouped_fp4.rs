//! Semantic grouped FP4 MoE layout and caller-owned buffer contracts.

use crate::cuda::runtime::DeviceBuffer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupedFp4MoeLayout {
    pub active_group_count: usize,
    pub small_group_count: usize,
    pub slot_capacity: usize,
    pub max_group_rows: usize,
    pub total_routed_rows: usize,
    pub num_tokens: usize,
    pub num_routes: usize,
    pub input_size: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub swiglu_limit: f32,
}

/// Low-level compatibility view retaining pointer tables and raw scratch. New
/// owner callers use `CudaExpertGroupRoutePlan`; no scratch representation is
/// promised by this view across providers.
pub struct GroupedFp4MoeBuffers<'a> {
    pub active_expert_slots: &'a DeviceBuffer<i32>,
    pub active_group_generations: &'a DeviceBuffer<i32>,
    pub expert_route_indptr: &'a DeviceBuffer<i32>,
    pub expert_route_counts: &'a DeviceBuffer<i32>,
    pub route_token_indices: &'a DeviceBuffer<i32>,
    pub route_indices: &'a DeviceBuffer<i32>,
    pub route_weights: &'a DeviceBuffer<f32>,
    pub slot_generations: &'a DeviceBuffer<i32>,
    pub gate_ptrs: &'a DeviceBuffer<u64>,
    pub gate_scale_ptrs: &'a DeviceBuffer<u64>,
    pub up_ptrs: &'a DeviceBuffer<u64>,
    pub up_scale_ptrs: &'a DeviceBuffer<u64>,
    pub down_ptrs: &'a DeviceBuffer<u64>,
    pub down_scale_ptrs: &'a DeviceBuffer<u64>,
    pub input_fp8: &'a DeviceBuffer<u8>,
    pub input_ue8m0: &'a DeviceBuffer<u8>,
    pub route_output: &'a mut DeviceBuffer<f32>,
    pub route_written: &'a mut DeviceBuffer<i32>,
    pub route_error: &'a mut DeviceBuffer<i32>,
    pub workspace: &'a mut DeviceBuffer<u8>,
}

use crate::cuda::context::{CudaExpertRouteResolveWorkspace, CudaF32Buffer};
use crate::cuda::operators::OperatorWorkspaceRequirements;
use crate::cuda::runtime::{CudaEvent, CudaStream, PinnedHostBuffer};
use ferrule_common::Result;

pub(crate) fn available() -> Result<bool> {
    crate::cuda::providers::cutlass::grouped_fp4_moe_available()
}

/// Size/alignment of opaque scratch for the selected grouped expert operator.
pub fn workspace_requirements(
    layout: GroupedFp4MoeLayout,
) -> Result<OperatorWorkspaceRequirements> {
    Ok(OperatorWorkspaceRequirements {
        bytes: crate::cuda::providers::cutlass::grouped_fp4_moe_workspace_size(layout)? as u64,
        alignment: GROUPED_FP4_MOE_WORKSPACE_ALIGNMENT as u32,
    })
}

/// Native-specific preparation and byte-size APIs, retained only for old callers.
/// New backend code uses the semantic requirements and scale-shape methods.
pub mod legacy {
    pub use crate::cuda::providers::cutlass::{
        grouped_fp4_moe_workspace_size, mxfp4_sfb_storage_bytes, prepare_mxfp4_sfb,
    };
}

/// Internal scale preparation contract: the transformed representation and its
/// sizing are not exposed to model callers. This adds no allocation or sync.
pub(crate) struct ExpertScaleShape {
    pub out_features: usize,
    pub in_features: usize,
}

impl ExpertScaleShape {
    pub(crate) fn prepared_bytes(&self) -> Result<usize> {
        crate::cuda::providers::cutlass::mxfp4_sfb_storage_bytes(
            self.out_features,
            self.in_features,
        )
    }

    pub(crate) fn prepare_into(
        &self,
        stream: &CudaStream,
        source: &DeviceBuffer<u8>,
        destination: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        crate::cuda::providers::cutlass::prepare_mxfp4_sfb(
            stream,
            source,
            destination,
            self.out_features,
            self.in_features,
        )
    }
}

pub use crate::cuda::providers::cutlass::{grouped_fp4_moe_can_implement, grouped_fp4_moe_launch};

/// Reusable workspace for grouped FP4 MoE batched execution.
///
/// The decode path hits this once per layer per token, so avoiding transient
/// CUDA allocations here is critical. The workspace owns all per-call scratch
/// buffers and fixed-size device arrays for selected expert pointers/weights.
/// Backend-private storage remains inaccessible through semantic and legacy paths.
/// ```compile_fail,E0616
/// use ferrule_backend::cuda::operators::moe::CudaMoeBatchedWorkspace;
/// fn access(workspace: &CudaMoeBatchedWorkspace) { let _ = &workspace.expert_output; }
/// ```
pub struct CudaMoeBatchedWorkspace {
    pub(crate) gate_ptrs: DeviceBuffer<u64>,
    pub(crate) gate_scale_ptrs: DeviceBuffer<u64>,
    pub(crate) up_ptrs: DeviceBuffer<u64>,
    pub(crate) up_scale_ptrs: DeviceBuffer<u64>,
    pub(crate) down_ptrs: DeviceBuffer<u64>,
    pub(crate) down_scale_ptrs: DeviceBuffer<u64>,
    pub(crate) route_weights: DeviceBuffer<f32>,
    pub(crate) route_slots: DeviceBuffer<i32>,
    pub(crate) dispatch_error: DeviceBuffer<i32>,
    pub(crate) expert_output: CudaF32Buffer,
    pub(crate) max_experts: usize,
    pub(crate) input_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) hidden_size: usize,
}

/// Device-resident compact routing metadata and caller-owned grouped FP4 MoE scratch.
///
/// Route resolution, counting, compaction, and scattering remain stream ordered on
/// device. The native grouped operator also requires four host scalar dimensions,
/// so a fixed 16-byte control block is copied to persistent pinned storage once per
/// prepared plan; no device allocation or stream-wide synchronization occurs there.
/// Backend-private storage remains inaccessible through semantic and legacy paths.
/// ```compile_fail,E0616
/// use ferrule_backend::cuda::context::CudaExpertGroupRoutePlan;
/// fn access(workspace: &CudaExpertGroupRoutePlan) { let _ = &workspace.cutlass_workspace; }
/// ```
pub struct CudaExpertGroupRoutePlan {
    pub(crate) slot_counts: DeviceBuffer<i32>,
    pub(crate) slot_route_offsets: DeviceBuffer<i32>,
    pub(crate) slot_cursors: DeviceBuffer<i32>,
    pub(crate) active_expert_slots: DeviceBuffer<i32>,
    pub(crate) active_group_generations: DeviceBuffer<i32>,
    pub(crate) expert_route_indptr: DeviceBuffer<i32>,
    pub(crate) expert_route_counts: DeviceBuffer<i32>,
    pub(crate) route_token_indices: DeviceBuffer<i32>,
    pub(crate) route_indices: DeviceBuffer<i32>,
    pub(crate) route_weights: DeviceBuffer<f32>,
    pub(crate) host_scalars: DeviceBuffer<i32>,
    pub(crate) host_staging: PinnedHostBuffer<i32>,
    pub(crate) metadata_ready: CudaEvent,
    pub(crate) metadata_copied: CudaEvent,
    pub(crate) host_metadata: Option<CudaExpertGroupRoutePlanHost>,
    pub(crate) route_written: DeviceBuffer<i32>,
    pub(crate) route_error: DeviceBuffer<i32>,
    pub(crate) resolve: CudaExpertRouteResolveWorkspace,
    pub(crate) input_fp8: DeviceBuffer<u8>,
    pub(crate) input_ue8m0: DeviceBuffer<u8>,
    pub(crate) cutlass_workspace: DeviceBuffer<u8>,
    pub(crate) max_experts: usize,
    pub(crate) route_capacity: usize,
    pub(crate) tokens: usize,
    pub(crate) input_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) input_prepared: bool,
    pub(crate) invocation_routes: Option<usize>,
}

pub(crate) const GROUPED_FP4_MOE_SMALL_GROUP_ROW_LIMIT: usize = 192;
pub(crate) const GROUPED_FP4_MOE_WORKSPACE_ALIGNMENT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaExpertGroupRoutePlanHost {
    pub active_group_count: usize,
    pub small_group_count: usize,
    pub max_group_rows: usize,
    pub total_routed_rows: usize,
}

impl CudaMoeBatchedWorkspace {
    pub fn matches(
        &self,
        max_experts: usize,
        input_size: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> bool {
        self.max_experts >= max_experts
            && self.input_size == input_size
            && self.intermediate_size == intermediate_size
            && self.hidden_size == hidden_size
    }
}

impl CudaExpertGroupRoutePlan {
    pub fn matches(
        &self,
        max_experts: usize,
        route_capacity: usize,
        tokens: usize,
        input_size: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> bool {
        self.max_experts >= max_experts
            && self.route_capacity >= route_capacity
            && self.tokens == tokens
            && self.input_size == input_size
            && self.intermediate_size == intermediate_size
            && self.hidden_size == hidden_size
    }

    pub fn host_metadata(&self) -> Option<CudaExpertGroupRoutePlanHost> {
        self.host_metadata
    }
}
