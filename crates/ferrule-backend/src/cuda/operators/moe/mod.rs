//! Mixture-of-experts routing and residency operators.

use ferrule_common::{Error, Result};

pub use crate::cuda::context::{
    CudaArtifactLinearHandle, CudaArtifactLinearShape, CudaBf16Buffer, CudaComputeEvent,
    CudaComputeStreamAuthority, CudaDsv4RouterHashTable as CudaRouterHashTable,
    CudaDsv4RouterTokenIds as CudaRouterTokenIds, CudaExpertGroupRoutePlan,
    CudaExpertGroupRoutePlanDownload, CudaExpertGroupRoutePlanHost, CudaExpertRouteMisses,
    CudaExpertRouteResolveResult, CudaExpertRouteResolveWorkspace, CudaExpertSlotBinding,
    CudaExpertSlotInstallTarget, CudaExpertSlotInstallTicket, CudaExpertSlotPointers,
    CudaExpertSlotTable, CudaExpertSlotTableHost, CudaF32Buffer, CudaFailpoints, CudaI32Buffer,
    CudaI32HostDownload, CudaI32HostMirror, CudaMoeBatchedWorkspace, CudaOperators,
    CudaPinnedHostAllocator, CudaPinnedU8HostBuffer, CudaPreparedRoutedExpert,
    CudaRoutedExpertArena, CudaRoutedExpertMaterialization, CudaRoutedExpertShape, CudaUploadEvent,
    validate_dsv4_router_hash_table as validate_router_hash_table,
    validate_dsv4_router_token_ids as validate_router_token_ids,
};
pub use crate::cuda::operators::contracts::{Bf16MoeRowsLayout, SelectedSoftmaxTopKLayout};
pub use crate::cuda::providers::cutlass::GroupedFp4MoeLayout;

pub mod residency;
pub mod routing;

/// Dense BF16 SwiGLU expert weights prepared as artifact linears.
pub struct DenseSwiGluLinears<'a> {
    pub gate: &'a CudaArtifactLinearHandle,
    pub up: &'a CudaArtifactLinearHandle,
    pub down: &'a CudaArtifactLinearHandle,
}

impl CudaOperators {
    /// Gather routed BF16 token rows as F32.
    pub fn gather_moe_rows(
        &self,
        source_bf16: &CudaBf16Buffer,
        route_rows: &CudaI32Buffer,
        layout: Bf16MoeRowsLayout,
    ) -> Result<CudaF32Buffer> {
        self.gather_bf16_moe_rows(source_bf16, route_rows, layout)
    }

    /// Execute dense BF16 gate/up/down SwiGLU linears for routed rows.
    pub fn dense_swiglu_routes(
        &self,
        linears: DenseSwiGluLinears<'_>,
        gathered_rows: &CudaF32Buffer,
        route_rows: usize,
        output_scale: f32,
        swiglu_limit: f32,
    ) -> Result<CudaF32Buffer> {
        for (name, handle) in [
            ("gate", linears.gate),
            ("up", linears.up),
            ("down", linears.down),
        ] {
            if !matches!(handle.shape(), CudaArtifactLinearShape::Bf16Bytes { .. }) {
                return Err(Error::Internal {
                    message: format!(
                        "dense SwiGLU {name} must use a BF16 artifact, got {:?}",
                        handle.shape()
                    ),
                });
            }
        }
        self.artifact_swiglu_ffn_rows_from_device(
            linears.gate,
            linears.up,
            linears.down,
            gathered_rows,
            route_rows,
            output_scale,
            swiglu_limit,
        )
    }

    /// Scatter routed values into token rows with route weights.
    pub fn weighted_scatter_add_moe_rows(
        &self,
        route_values: &CudaF32Buffer,
        route_rows: &CudaI32Buffer,
        route_weights: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
        layout: Bf16MoeRowsLayout,
    ) -> Result<()> {
        self.weighted_scatter_add_bf16_moe_rows(
            route_values,
            route_rows,
            route_weights,
            output,
            layout,
        )
    }
}
