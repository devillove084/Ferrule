//! Transfer engine trait — executes movement plans.
//!
//! The first implementation (Phase 0 mock) is synchronous and boring: `ensure`
//! records what was requested, `prefetch`/`poll` are stubs. Real backends
//! (file read, mmap, H2D, io_uring, RDMA) arrive in Phase 2–5 behind the same
//! trait.

use ferrule_common::Result;

use super::replica::ReplicaHandleId;
use super::residency::{ResidencyPriority, ResidencyReason, ResidencyRequest};

/// Executes object movement between placements.
pub trait TransferEngine {
    /// Blocking ensure: make the object available at the requested placement.
    /// Returns a handle id the caller can resolve with the backend.
    fn ensure(&mut self, request: ResidencyRequest) -> Result<ReplicaHandleId>;

    /// Non-blocking prefetch: start transfers, return tickets for polling.
    /// Phase 4+; v1 implementation may fall back to `ensure` or return empty.
    fn prefetch(&mut self, requests: &[ResidencyRequest]) -> Result<Vec<TransferTicket>>;

    /// Poll for completion of previously issued transfers.
    fn poll(&mut self) -> Result<Vec<TransferEvent>>;
}

/// Ticket for tracking an async transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferTicket {
    pub id: u64,
    pub object: super::id::StorageObjectId,
}

/// Event signaling transfer completion (or failure).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferEvent {
    pub ticket: u64,
    pub object: super::id::StorageObjectId,
    pub outcome: TransferOutcome,
}

/// Outcome of a transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferOutcome {
    /// Transfer completed; handle is ready.
    Completed(ReplicaHandleId),
    /// Transfer failed.
    Failed { reason: String },
    /// Transfer was cancelled before completion.
    Cancelled,
}

// ── Mock transfer engine (for testing) ────────────────────────────────

/// A mock transfer engine that records requests and returns synthetic handles.
///
/// Useful for unit-testing the residency manager and policy without real I/O.
#[derive(Debug, Default)]
pub struct MockTransferEngine {
    next_ticket: u64,
    next_slot: u64,
    next_generation: u64,
    pub ensure_count: u64,
    pub prefetch_count: u64,
    pub last_ensure_object: Option<super::id::StorageObjectId>,
    pub last_ensure_priority: Option<ResidencyPriority>,
    pub last_ensure_reason: Option<ResidencyReason>,
    pub pending_events: Vec<TransferEvent>,
}

impl MockTransferEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a synthetic completion event for a ticket.
    pub fn queue_completion(&mut self, ticket: u64, object: super::id::StorageObjectId) {
        let handle = ReplicaHandleId::new("mock", self.next_slot, self.next_generation);
        self.next_slot += 1;
        self.next_generation += 1;
        self.pending_events.push(TransferEvent {
            ticket,
            object,
            outcome: TransferOutcome::Completed(handle),
        });
    }
}

impl TransferEngine for MockTransferEngine {
    fn ensure(&mut self, request: ResidencyRequest) -> Result<ReplicaHandleId> {
        self.ensure_count += 1;
        self.last_ensure_object = Some(request.object.clone());
        self.last_ensure_priority = Some(request.priority);
        self.last_ensure_reason = Some(request.reason);
        let handle = ReplicaHandleId::new("mock", self.next_slot, self.next_generation);
        self.next_slot += 1;
        self.next_generation += 1;
        Ok(handle)
    }

    fn prefetch(&mut self, requests: &[ResidencyRequest]) -> Result<Vec<TransferTicket>> {
        self.prefetch_count += requests.len() as u64;
        let tickets = requests
            .iter()
            .map(|req| {
                let ticket = self.next_ticket;
                self.next_ticket += 1;
                TransferTicket {
                    id: ticket,
                    object: req.object.clone(),
                }
            })
            .collect();
        Ok(tickets)
    }

    fn poll(&mut self) -> Result<Vec<TransferEvent>> {
        Ok(std::mem::take(&mut self.pending_events))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::id::{ModelRevision, StorageObjectId};
    use crate::storage::placement::{DeviceMemoryKind, LocalPlacement, Placement};

    fn rev(n: u64) -> ModelRevision {
        ModelRevision(n)
    }

    fn sample_object() -> StorageObjectId {
        StorageObjectId::ExpertBundle {
            model_revision: rev(1),
            layer: 0,
            expert: 0,
            layout_version: 1,
        }
    }

    fn device_placement() -> Placement {
        Placement::Local(LocalPlacement::Device {
            device_id: 0,
            memory: DeviceMemoryKind::Vram,
        })
    }

    #[test]
    fn mock_ensure_returns_handle() {
        let mut engine = MockTransferEngine::new();
        let req = ResidencyRequest::new(
            sample_object(),
            device_placement(),
            ResidencyPriority::Critical,
            ResidencyReason::ExecuteNow,
        );
        let handle = engine.ensure(req).unwrap();
        assert_eq!(handle.backend, "mock");
        assert_eq!(engine.ensure_count, 1);
        assert_eq!(engine.last_ensure_reason, Some(ResidencyReason::ExecuteNow));
    }

    #[test]
    fn mock_prefetch_returns_tickets() {
        let mut engine = MockTransferEngine::new();
        let req1 = ResidencyRequest::new(
            sample_object(),
            device_placement(),
            ResidencyPriority::Background,
            ResidencyReason::Prefetch,
        );
        let req2 = ResidencyRequest::new(
            StorageObjectId::ExpertBundle {
                model_revision: rev(1),
                layer: 0,
                expert: 1,
                layout_version: 1,
            },
            device_placement(),
            ResidencyPriority::Background,
            ResidencyReason::Prefetch,
        );
        let tickets = engine.prefetch(&[req1, req2]).unwrap();
        assert_eq!(tickets.len(), 2);
        assert_eq!(tickets[0].id, 0);
        assert_eq!(tickets[1].id, 1);
        assert_eq!(engine.prefetch_count, 2);
    }

    #[test]
    fn mock_poll_returns_queued_events() {
        let mut engine = MockTransferEngine::new();
        let obj = sample_object();
        engine.queue_completion(0, obj.clone());
        let events = engine.poll().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].ticket, 0);
        assert_eq!(events[0].object, obj);
        assert!(matches!(events[0].outcome, TransferOutcome::Completed(_)));

        // Second poll returns empty.
        let events2 = engine.poll().unwrap();
        assert!(events2.is_empty());
    }

    #[test]
    fn mock_handles_increment_generation() {
        let mut engine = MockTransferEngine::new();
        let req = ResidencyRequest::new(
            sample_object(),
            device_placement(),
            ResidencyPriority::Critical,
            ResidencyReason::ExecuteNow,
        );
        let h1 = engine.ensure(req.clone()).unwrap();
        let h2 = engine.ensure(req).unwrap();
        assert_ne!(h1.generation, h2.generation);
        assert_ne!(h1.slot, h2.slot);
    }
}
