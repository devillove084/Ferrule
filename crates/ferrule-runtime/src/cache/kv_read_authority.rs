//! Logical read custody, implemented inside the existing page-manager authority.
use super::*;
use ferrule_common::host_kv::{
    CommittedKvReadAuthority, CommittedKvSource, HostKvError, HostKvResult,
};
use std::ops::Range;

pub use super::committed_read_pins::CommittedKvReadLease;

/// Borrowed sub-capability of the existing append reservation. It registers no
/// independent reader and cannot be detached to acknowledge DMA. Holding it
/// prevents moving the manager/reservation into logical commit or abort.
/// `with_cuda_append` uses the same entered model/backend transaction and keeps
/// this borrow through the scoped output consumer and its final fence. Automatic
/// runner selection is not enabled; dropping a capability is not a physical ACK.
#[doc = include_str!("../../tests/compile/host_kv/append_custody.md")]
#[must_use]
pub struct ReservedAppendPrefix<'a> {
    manager: &'a KvPageManager,
    reservation: &'a KvReservation,
    witness: ferrule_common::host_kv::HostKvAppendReadWitness,
    layer: usize,
}
impl ReservedAppendPrefix<'_> {
    pub fn witness(&self) -> &ferrule_common::host_kv::HostKvAppendReadWitness {
        &self.witness
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    /// Derive another layer from the SAME borrowed reservation/transaction.
    /// Does not mutate this capability, clone its authority or register a pin.
    /// A runner may use these sequentially after each physical output is fully
    /// returned; this does not enable automatic dispatch or concurrent readers.
    pub fn for_layer(&self, layer: usize) -> HostKvResult<ReservedAppendPrefix<'_>> {
        self.validate()?;
        self.manager
            .derive_append_prefix(self.reservation, self.witness.transaction(), layer)
    }
    pub fn validate(&self) -> HostKvResult<()> {
        let current = self.manager.append_prefix_witness(
            self.reservation,
            self.witness.transaction(),
            self.layer,
        )?;
        if current != self.witness {
            return Err(HostKvError::StaleSource);
        }
        Ok(())
    }
}

impl KvPageManager {
    pub fn derive_append_prefix<'a>(
        &'a self,
        reservation: &'a KvReservation,
        transaction: ferrule_common::execution::ExecutionTransactionId,
        layer: usize,
    ) -> HostKvResult<ReservedAppendPrefix<'a>> {
        let witness = self.append_prefix_witness(reservation, transaction, layer)?;
        Ok(ReservedAppendPrefix {
            manager: self,
            reservation,
            witness,
            layer,
        })
    }
    fn append_prefix_witness(
        &self,
        reservation: &KvReservation,
        transaction: ferrule_common::execution::ExecutionTransactionId,
        layer: usize,
    ) -> HostKvResult<ferrule_common::host_kv::HostKvAppendReadWitness> {
        if reservation.manager_id != self.manager_id {
            return Err(HostKvError::ForeignOwner);
        }
        self.validate_live_reservations(&[reservation])
            .map_err(|_| HostKvError::UnknownLease)?;
        let state = self
            .sequences
            .get(&reservation.state_slot.get())
            .ok_or(HostKvError::StaleSource)?;
        if state.generation != reservation.generation
            || state.block_table.committed_tokens != reservation.positions.start
            || self.schema.planes().iter().any(|p| layer >= p.layer_count)
        {
            return Err(HostKvError::StaleSource);
        }
        self.ensure_no_page_readers(&state.block_table.pages)
            .map_err(|_| HostKvError::ReaderActive)?;
        if state
            .block_table
            .pages
            .iter()
            .any(|p| self.retiring_pages.contains(p) || self.page_refcount(*p) == 0)
        {
            return Err(HostKvError::StaleSource);
        }
        // Shared prefix reads are safe only if every mutation uses its own COW
        // replacement. An exclusive mutable tail must never alias another slot.
        if !reservation
            .positions
            .start
            .is_multiple_of(self.schema.page_size())
            && reservation.cow_replacement.is_none()
        {
            let tail = state
                .block_table
                .pages
                .last()
                .ok_or(HostKvError::StaleSource)?;
            if self.page_refcount(*tail) != 1
                || self.sequences.iter().any(|(slot, other)| {
                    *slot != reservation.state_slot.get() && other.block_table.pages.contains(tail)
                })
            {
                return Err(HostKvError::SharedSource);
            }
        }
        ferrule_common::host_kv::HostKvAppendReadWitness::new(
            transaction,
            reservation.id.get(),
            CommittedKvSource {
                owner: self.manager_id,
                state_slot: reservation.state_slot,
                generation: reservation.generation,
                incarnation: state.incarnation,
                schema: self.schema.schema_identity(),
                committed_tokens: state.block_table.committed_tokens,
                positions: 0..state.block_table.committed_tokens,
                pages: state.block_table.pages.clone(),
            },
            reservation.view.clone(),
        )
    }
}

impl KvPageManager {
    pub fn committed_kv_source(
        &self,
        state_slot: StateSlot,
        generation: u64,
        positions: Range<usize>,
    ) -> HostKvResult<CommittedKvSource> {
        let state = self
            .sequences
            .get(&state_slot.get())
            .ok_or(HostKvError::StaleSource)?;
        let source = CommittedKvSource {
            owner: self.manager_id,
            state_slot,
            generation,
            incarnation: state.incarnation,
            schema: self.schema.schema_identity(),
            committed_tokens: state.block_table.committed_tokens,
            positions,
            pages: state.block_table.pages.clone(),
        };
        self.validate_committed_source(&source)?;
        Ok(source)
    }
    pub fn validate_committed_source(&self, source: &CommittedKvSource) -> HostKvResult<()> {
        if source.owner != self.manager_id {
            return Err(HostKvError::ForeignOwner);
        }
        source.validate_geometry()?;
        let state = self
            .sequences
            .get(&source.state_slot.get())
            .ok_or(HostKvError::StaleSource)?;
        if state.generation != source.generation
            || state.incarnation != source.incarnation
            || state.block_table.committed_tokens != source.committed_tokens
            || state.block_table.pages != source.pages
        {
            return Err(HostKvError::StaleSource);
        }
        if self.schema.schema_identity() != source.schema {
            return Err(HostKvError::InvalidGeometry);
        }
        if self.pending_reservations.values().any(|r| {
            r.state_slot == source.state_slot
                || r.execution_state_slot == source.state_slot
                || r.cow_replacement
                    .is_some_and(|cow| source.pages.contains(&cow.source))
                || self.sequences.get(&r.state_slot.get()).is_some_and(|seq| {
                    seq.block_table
                        .pages
                        .iter()
                        .any(|p| source.pages.contains(p))
                })
        }) {
            return Err(HostKvError::TransactionActive);
        }
        if source
            .pages
            .iter()
            .any(|p| self.retiring_pages.contains(p) || self.page_refcount(*p) != 1)
        {
            return Err(HostKvError::SharedSource);
        }
        // Refcount alone must not certify exclusivity if an alias invariant regresses.
        if self.sequences.iter().any(|(slot, seq)| {
            *slot != source.state_slot.get()
                && seq
                    .block_table
                    .pages
                    .iter()
                    .any(|p| source.pages.contains(p))
        }) || self
            .suspended
            .values()
            .any(|s| s.block_table.pages.iter().any(|p| source.pages.contains(p)))
            || self
                .prefix_snapshots
                .values()
                .any(|s| s.pages.iter().any(|p| source.pages.contains(p)))
        {
            return Err(HostKvError::SharedSource);
        }
        Ok(())
    }
    pub fn acquire_committed_read(
        &mut self,
        source: &CommittedKvSource,
    ) -> HostKvResult<CommittedKvReadLease> {
        self.validate_committed_source(source)
            .inspect_err(|reason| {
                tracing::debug!(
                    target: "ferrule::host_kv",
                    event = "committed_read_rejected",
                    owner = self.manager_id,
                    reason = ?reason,
                    "logical committed read rejected before registration"
                );
            })?;
        self.committed_read_pins.register(self.manager_id, source)
    }
    pub fn validate_committed_read(&self, lease: &CommittedKvReadLease) -> HostKvResult<()> {
        self.committed_read_pins.validate(self.manager_id, lease)?;
        self.validate_committed_source(lease.source())
    }
    pub fn release_committed_read(&mut self, lease: &CommittedKvReadLease) -> HostKvResult<()> {
        // The pin index cannot certify that the logical source is still valid.
        // Finish all authority preflight before consuming its reader registration.
        self.validate_committed_read(lease)?;
        self.committed_read_pins.release(self.manager_id, lease)
    }
    pub(super) fn ensure_no_page_readers(&self, pages: &[KvPageId]) -> Result<()> {
        self.committed_read_pins
            .check_pages(pages)
            .map_err(|_| Error::InvalidRequest {
                message: "page manager: committed KV reader active on physical page".into(),
            })
    }
    pub(super) fn ensure_no_slot_readers(&self, slot: StateSlot) -> Result<()> {
        self.committed_read_pins
            .check_slot(slot)
            .map_err(|_| Error::InvalidRequest {
                message: "page manager: committed KV reader active on state slot".into(),
            })?;
        if let Some(seq) = self.sequences.get(&slot.get()) {
            self.ensure_no_page_readers(&seq.block_table.pages)?;
        }
        Ok(())
    }
}
impl CommittedKvReadAuthority for KvPageManager {
    type Lease = CommittedKvReadLease;
    fn acquire(&mut self, source: &CommittedKvSource) -> HostKvResult<Self::Lease> {
        self.acquire_committed_read(source)
    }
    fn validate(&self, lease: &Self::Lease) -> HostKvResult<()> {
        self.validate_committed_read(lease)
    }
    fn release(&mut self, lease: &Self::Lease) -> HostKvResult<()> {
        self.release_committed_read(lease)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cache/kv_read_authority.rs"]
mod tests;
