//! Admission and observability for opt-in idle-session KV reclamation.
//! No logical owner or transaction authority is created here.
use ferrule_common::execution::{KvElementType, KvPlaneDescriptor};
use ferrule_common::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdleKvReclaimConfig {
    pub staging_page_limit: usize,
    /// Includes original + provisional chunks and the bounded Entry staging grants.
    pub device_bytes: usize,
    /// Full all-plane snapshots, charged independently of device grants.
    pub host_bytes: usize,
    pub pinned_bytes: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdleKvMemoryPlan {
    pub physical_page_capacity: usize,
    pub page_payload_bytes: usize,
    pub page_grant_bytes: usize,
    pub physical_grant_bytes: usize,
    pub staging_grant_bytes: usize,
    pub device_bytes: usize,
    pub host_bytes: usize,
    pub pinned_bytes: usize,
}
fn invalid() -> Error {
    Error::Execution {
        message: "idle KV reclaim budget/geometry is invalid or insufficient".into(),
    }
}
impl IdleKvReclaimConfig {
    pub fn plan(
        self,
        planes: &[KvPlaneDescriptor],
        page_size: usize,
        pages: usize,
    ) -> Result<IdleKvMemoryPlan> {
        if planes.len() != 2
            || page_size == 0
            || pages == 0
            || self.staging_page_limit == 0
            || self.staging_page_limit > pages
            || planes.iter().any(|p| p.element_type != KvElementType::F32)
        {
            return Err(invalid());
        }
        let mut payload = 0usize;
        let mut grant = 0usize;
        let mut staging = 0usize;
        for plane in planes {
            let bytes = plane
                .checked_page_bytes(page_size)
                .filter(|n| *n > 0)
                .ok_or_else(invalid)?;
            payload = payload.checked_add(bytes).ok_or_else(invalid)?;
            grant = grant
                .checked_add(
                    ferrule_backend::cuda_allocation_grant_bytes(bytes).ok_or_else(invalid)?,
                )
                .ok_or_else(invalid)?;
            staging = staging
                .checked_add(
                    ferrule_backend::cuda_allocation_grant_bytes(
                        bytes
                            .checked_mul(self.staging_page_limit)
                            .ok_or_else(invalid)?,
                    )
                    .ok_or_else(invalid)?,
                )
                .ok_or_else(invalid)?;
        }
        let physical = grant.checked_mul(pages).ok_or_else(invalid)?;
        let device = physical.checked_add(staging).ok_or_else(invalid)?;
        let host = payload.checked_mul(pages).ok_or_else(invalid)?;
        // Conservatively admit snapshots and full-batch pinned frames concurrently.
        // Runtime logical page IDs remain charged while suspended, so each bound
        // is the existing manager's entire physical page capacity, not a new ledger.
        if device > self.device_bytes || host > self.host_bytes || host > self.pinned_bytes {
            return Err(invalid());
        }
        Ok(IdleKvMemoryPlan {
            physical_page_capacity: pages,
            page_payload_bytes: payload,
            page_grant_bytes: grant,
            physical_grant_bytes: physical,
            staging_grant_bytes: staging,
            device_bytes: device,
            host_bytes: host,
            pinned_bytes: host,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdleKvPhysicalStats {
    pub admission: IdleKvMemoryPlan,
    pub live_payload_bytes: usize,
    pub entry_staging_payload_bytes: usize,
    pub host_snapshot_bytes: usize,
    pub pinned_bytes: usize,
    /// Context-wide allocator grants/reservations; includes weights and scratch.
    pub allocator_live_grant_bytes: usize,
    pub allocator_reserved_bytes: usize,
    pub allocator_pending_bytes: usize,
    pub driver_freed_bytes: u64,
    pub unknown: bool,
}
