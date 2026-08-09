//! Provider-neutral MoE routing operations.

use ferrule_common::{Error, Result};

use crate::cuda::context::{
    CudaF32Buffer, CudaI32Buffer, CudaI32HostDownload, CudaI32HostMirror, CudaOperators,
};
use crate::cuda::operators::SelectedSoftmaxTopKLayout;

use super::{CudaRouterHashTable, CudaRouterTokenIds};

/// In-flight compact route metadata transfer owned by one MoE continuation.
pub struct CudaMoeRouteDownload {
    mirror: CudaI32HostMirror,
    download: CudaI32HostDownload,
    route_count: usize,
}

impl CudaOperators {
    /// Compute stable selected softmax top-k routes.
    pub fn route_selected_softmax_topk(
        &self,
        logits: &CudaF32Buffer,
        expert_ids: &mut CudaI32Buffer,
        weights: &mut CudaF32Buffer,
        layout: SelectedSoftmaxTopKLayout,
    ) -> Result<()> {
        self.selected_softmax_topk_from_device_into(logits, expert_ids, weights, layout)
    }

    /// Upload a validated token-indexed expert selection table.
    pub fn upload_router_hash_table(
        &self,
        table: &[usize],
        rows: usize,
        cols: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<CudaRouterHashTable> {
        self.upload_dsv4_router_hash_table(table, rows, cols, experts, top_k)
    }

    /// Create reusable device token IDs for hash-assisted routing.
    pub fn router_token_ids(
        &self,
        token_ids: &[u32],
        hash_rows: usize,
    ) -> Result<CudaRouterTokenIds> {
        self.dsv4_router_token_ids(token_ids, hash_rows)
    }

    /// Refresh reusable device token IDs for hash-assisted routing.
    pub fn update_router_token_ids(
        &self,
        token_ids: &[u32],
        hash_rows: usize,
        cached: &mut CudaRouterTokenIds,
    ) -> Result<()> {
        self.update_dsv4_router_token_ids(token_ids, hash_rows, cached)
    }

    /// Route rows with sqrt-softplus scores, optional selection bias, and top-k.
    #[allow(clippy::too_many_arguments)]
    pub fn route_sqrt_softplus_topk(
        &self,
        logits: &CudaF32Buffer,
        selection_bias: Option<&CudaF32Buffer>,
        rows: usize,
        experts: usize,
        top_k: usize,
        route_scale: f32,
        expert_ids: &mut CudaI32Buffer,
        weights: &mut CudaF32Buffer,
    ) -> Result<()> {
        self.dsv4_router_topk_sqrt_softplus_rows_from_device_into(
            logits,
            selection_bias,
            rows,
            experts,
            top_k,
            route_scale,
            expert_ids,
            weights,
        )
    }

    /// Route rows with a token-indexed selection table and sqrt-softplus weights.
    #[allow(clippy::too_many_arguments)]
    pub fn route_hash_sqrt_softplus(
        &self,
        logits: &CudaF32Buffer,
        token_ids: &CudaRouterTokenIds,
        hash_table: &CudaRouterHashTable,
        rows: usize,
        experts: usize,
        top_k: usize,
        route_scale: f32,
        expert_ids: &mut CudaI32Buffer,
        weights: &mut CudaF32Buffer,
    ) -> Result<()> {
        self.dsv4_router_hash_sqrt_softplus_rows_from_device_into(
            logits,
            token_ids,
            hash_table,
            rows,
            experts,
            top_k,
            route_scale,
            expert_ids,
            weights,
        )
    }

    /// Pack route IDs and weights and begin a non-blocking host transfer.
    pub fn submit_moe_route_download(
        &self,
        expert_ids: &CudaI32Buffer,
        weights: &CudaF32Buffer,
        route_count: usize,
    ) -> Result<CudaMoeRouteDownload> {
        let compact_len = route_count.checked_mul(2).ok_or_else(|| Error::Internal {
            message: "CUDA compact MoE route size overflow".into(),
        })?;
        if compact_len == 0 {
            return Err(Error::Internal {
                message: "CUDA compact MoE route download requires at least one route".into(),
            });
        }
        let mut mirror = self.i32_host_mirror(&vec![0; compact_len])?;
        self.pack_i32_f32_pairs_into(
            expert_ids,
            weights,
            mirror.device_mut_invalidate_host(),
            route_count,
        )?;
        let produced = self.record_compute_event()?;
        let download = self.begin_i32_host_mirror_download_after(&mut mirror, &produced)?;
        Ok(CudaMoeRouteDownload {
            mirror,
            download,
            route_count,
        })
    }

    /// Poll a compact route transfer without synchronizing either CUDA stream.
    pub fn poll_moe_route_download(
        &self,
        pending: &mut CudaMoeRouteDownload,
    ) -> Result<Option<Vec<i32>>> {
        let compact = self.poll_i32_host_mirror_download(&mut pending.mirror, &pending.download)?;
        if compact
            .as_ref()
            .is_some_and(|values| values.len() != pending.route_count.saturating_mul(2))
        {
            return Err(Error::Internal {
                message: "CUDA compact MoE route download returned the wrong length".into(),
            });
        }
        Ok(compact)
    }
}
