//! One serial retained upload in the existing admitted metadata miss. Queries
//! observe host completion; no event dependency or notification publishes data.

use super::super::{CudaStandardDecoderOperators, cuda_error};
use crate::transformer::standard::expert_cache::{ExpertCacheKey, Lease};
use crate::transformer::{
    ExpertAvailability, ExpertMetadata, ExpertProvider, ExpertUploadReservation,
};
use ferrule_backend::cuda::operators::linear::{
    CudaNumericFp8UploadHostCharge, CudaNumericFp8UploadTicket,
};
use ferrule_common::Result;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct Identity {
    lease: Lease<ExpertCacheKey>,
    ticket_generation: u64,
}

impl Identity {
    fn validate(
        &self,
        lease: Option<&Lease<ExpertCacheKey>>,
        admission: Option<&ExpertCacheKey>,
        image: u64,
        ticket_generation: u64,
    ) -> Result<()> {
        if lease != Some(&self.lease)
            || admission != Some(&self.lease.key)
            || self.lease.key.image_generation != image
            || self.ticket_generation == 0
            || self.ticket_generation != ticket_generation
        {
            return Err(cuda_error(
                "stale retained upload lease/image/ticket generation",
            ));
        }
        Ok(())
    }
}

struct Pending {
    identity: Identity,
    ticket: CudaNumericFp8UploadTicket,
    reservation: ExpertUploadReservation,
}

pub(in crate::transformer::standard::cuda) struct State {
    headroom: usize,
    timeout: Duration,
    lease: Option<Lease<ExpertCacheKey>>,
    pending: Option<Pending>,
    failed: bool,
}

impl State {
    pub(in crate::transformer::standard::cuda) fn new(headroom: usize, timeout: Duration) -> Self {
        Self {
            headroom,
            timeout,
            lease: None,
            pending: None,
            failed: false,
        }
    }
    pub(in crate::transformer::standard::cuda) fn policy(
        &self,
    ) -> crate::transformer::ExpertTransferPolicy {
        crate::transformer::ExpertTransferPolicy::Retained {
            policy_headroom_bytes: self.headroom,
            readiness_timeout: self.timeout,
        }
    }
    pub(in crate::transformer::standard::cuda) fn begin_lease(
        &mut self,
        lease: &Lease<ExpertCacheKey>,
    ) -> Result<()> {
        if self.failed || self.lease.is_some() || self.pending.is_some() {
            return Err(cuda_error(
                "retained upload owner already has unresolved custody",
            ));
        }
        self.lease = Some(lease.clone());
        Ok(())
    }
    pub(in crate::transformer::standard::cuda) fn quarantine(&mut self) {
        self.failed = true;
        if let Some(pending) = &mut self.pending {
            pending.reservation.quarantine();
        }
    }
    pub(in crate::transformer::standard::cuda) fn finish_lease(&mut self) -> Result<()> {
        if self.failed || self.pending.is_some() {
            return Err(cuda_error("retained upload remains unresolved"));
        }
        self.lease = None;
        Ok(())
    }
}

impl CudaStandardDecoderOperators {
    pub(in crate::transformer::standard::cuda) fn retained_metadata_upload(
        &mut self,
        metadata: &ExpertMetadata,
        provider: &mut dyn ExpertProvider,
        rows: usize,
    ) -> Result<()> {
        let result = self.retained_metadata_upload_inner(metadata, provider, rows);
        if result.is_err() {
            self.retained_upload.as_mut().expect("opt in").quarantine();
            self.poisoned = true;
            self.cache_quarantine();
        }
        result
    }

    fn retained_metadata_upload_inner(
        &mut self,
        metadata: &ExpertMetadata,
        provider: &mut dyn ExpertProvider,
        rows: usize,
    ) -> Result<()> {
        let state = self.retained_upload.as_ref().expect("opt in");
        if state.failed || state.pending.is_some() {
            return Err(cuda_error("one unresolved ticket per CUDA owner"));
        }
        let lease = state
            .lease
            .clone()
            .ok_or_else(|| cuda_error("retained upload requires an exact cache lease"))?;
        let timeout = state.timeout;
        let mut reservation = provider
            .reserve_expert_upload(
                metadata,
                std::mem::size_of::<CudaNumericFp8UploadTicket>(),
                state.headroom,
            )?
            .ok_or_else(|| {
                cuda_error("retained upload requires configured expert-read host admission")
            })?;
        reservation.validate_identity(
            lease.key.image_generation,
            lease.key.layer,
            lease.key.expert,
        )?;
        let expert = match provider.expert(lease.key.layer, lease.key.expert)? {
            ExpertAvailability::Ready(expert) => expert,
            _ => {
                return Err(cuda_error(
                    "metadata-ready retained expert became unavailable",
                ));
            }
        };
        metadata.validate_payload(&expert)?;
        let (key, bytes, _) = self.expert_plan(&expert, rows)?;
        if key != lease.key {
            return Err(cuda_error("retained payload belongs to another admission"));
        }
        let payloads = [expert.gate(), expert.up(), expert.down()].map(|p| {
            p.numeric_fp8()
                .expect("numeric metadata validated")
                .validated_payload()
                .clone()
        });
        let charge = CudaNumericFp8UploadHostCharge::for_payloads(&payloads)?;
        let reserved = reservation.host_charge();
        if charge.pageable_payload_bytes != bytes
            || reserved.prepared_bytes != bytes as u64
            || reserved.pinned_bytes != charge.pinned_allocation_bytes as u64
            || reserved.ticket_metadata_bytes != charge.ticket_metadata_bytes as u64
        {
            return Err(cuda_error(
                "retained payload differs from pre-allocation host reservation",
            ));
        }
        provider.transfer_expert_upload(&mut reservation)?;
        let ticket = self
            .ops
            .stage_numeric_fp8_upload(payloads, charge.total_bytes)?;
        let identity = Identity {
            lease,
            ticket_generation: ticket.generation(),
        };
        self.retained_upload.as_mut().expect("opt in").pending = Some(Pending {
            identity,
            ticket,
            reservation,
        });
        // Original PreparedSwiGlu payload aliases end, but neither packed nor
        // prepared credit is refunded. The ticket and transferred read own it.
        drop(expert);
        let pending = self
            .retained_upload
            .as_mut()
            .expect("opt in")
            .pending
            .as_mut()
            .expect("staged");
        self.ops.submit_numeric_fp8_upload(&mut pending.ticket)?;
        let started = Instant::now();
        loop {
            let image = self.image_generation();
            let state = self.retained_upload.as_mut().expect("opt in");
            let pending = state.pending.as_mut().expect("submitted");
            pending.identity.validate(
                state.lease.as_ref(),
                self.bindings.expert_admission.as_ref(),
                image,
                pending.ticket.generation(),
            )?;
            if let Some(artifacts) = self.ops.poll_numeric_fp8_upload(&mut pending.ticket)? {
                pending.identity.validate(
                    state.lease.as_ref(),
                    self.bindings.expert_admission.as_ref(),
                    image,
                    pending.ticket.generation(),
                )?;
                self.bindings.publish_numeric_expert(
                    &pending.identity.lease.key,
                    metadata,
                    artifacts,
                )?;
                let Pending {
                    ticket,
                    mut reservation,
                    ..
                } = state.pending.take().expect("ready");
                drop(ticket);
                reservation.prove_completion();
                reservation.release()?;
                return Ok(());
            }
            if started.elapsed() >= timeout {
                // One explicit cancellation drain, never a publication shortcut.
                // Its driver wait can block; the polling interval itself is bounded.
                self.ops.drain_numeric_fp8_upload(&mut pending.ticket)?;
                let Pending {
                    ticket,
                    mut reservation,
                    ..
                } = state.pending.take().expect("cancelled");
                drop(ticket);
                reservation.prove_completion();
                reservation.release()?;
                return Err(cuda_error(
                    "retained upload readiness timed out; cancelled after completion proof, not published",
                ));
            }
            std::thread::sleep(
                Duration::from_micros(50).min(timeout.saturating_sub(started.elapsed())),
            );
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/retained_numeric_upload/owner.rs"]
mod tests;
