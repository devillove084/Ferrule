//! Replica — a current resident copy of an object.
//!
//! The generic manager tracks **metadata only**. Real backend handles (CUDA
//! buffers, mmap slices) stay in backend-owned stores, referenced via an opaque
//! `ReplicaHandleId`.

use std::fmt;

use super::id::StorageObjectId;
use super::placement::Placement;

/// Opaque backend identifier. Used in `ReplicaHandleId` to route handle
/// resolution to the correct backend-owned store.
pub type BackendId = String;

/// Opaque reference to a backend-owned handle.
///
/// The generic residency manager never dereferences this. The backend-owned
/// handle store resolves it.
///
/// `generation` must match the `ObjectReplica.generation` — if they differ, the
/// handle is stale (the replica was evicted and the handle slot was reused).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReplicaHandleId {
    pub backend: BackendId,
    pub slot: u64,
    pub generation: u64,
}

impl ReplicaHandleId {
    pub fn new(backend: impl Into<BackendId>, slot: u64, generation: u64) -> Self {
        Self {
            backend: backend.into(),
            slot,
            generation,
        }
    }

    /// True if this handle's generation matches the given replica's generation.
    pub fn matches_generation(&self, replica: &ObjectReplica) -> bool {
        self.generation == replica.generation
    }
}

impl fmt::Display for ReplicaHandleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}@slot{}:gen{}",
            self.backend, self.slot, self.generation
        )
    }
}

/// A current resident copy of an object at a placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectReplica {
    pub object: StorageObjectId,
    pub placement: Placement,
    pub bytes: u64,
    pub state: ReplicaState,
    pub generation: u64,
    pub handle: ReplicaHandleId,
}

impl ObjectReplica {
    /// True if this replica is ready for use.
    pub fn is_ready(&self) -> bool {
        matches!(self.state, ReplicaState::Ready)
    }

    /// True if this replica is at a hotter tier than the given placement.
    pub fn is_hotter_than(&self, other: &Placement) -> bool {
        self.placement.tier_rank() < other.tier_rank()
    }
}

/// Lifecycle state of a replica.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicaState {
    /// Fully resident and usable.
    Ready,
    /// Transfer in progress (not yet usable).
    Loading,
    /// Being evicted (still usable until transfer completes, but no new
    /// references should be issued).
    Evicting,
    /// Transfer or load failed.
    Failed { reason: String },
}

impl ReplicaState {
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Ready | Self::Evicting)
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
    fn handle_generation_match() {
        let handle = ReplicaHandleId::new("cuda:0", 5, 3);
        let replica = ObjectReplica {
            object: sample_object(),
            placement: device_placement(),
            bytes: 4096,
            state: ReplicaState::Ready,
            generation: 3,
            handle: handle.clone(),
        };
        assert!(handle.matches_generation(&replica));
    }

    #[test]
    fn handle_generation_mismatch() {
        let handle = ReplicaHandleId::new("cuda:0", 5, 4);
        let replica = ObjectReplica {
            object: sample_object(),
            placement: device_placement(),
            bytes: 4096,
            state: ReplicaState::Ready,
            generation: 3,
            handle: handle.clone(),
        };
        assert!(!handle.matches_generation(&replica));
    }

    #[test]
    fn replica_is_ready() {
        let replica = ObjectReplica {
            object: sample_object(),
            placement: device_placement(),
            bytes: 4096,
            state: ReplicaState::Ready,
            generation: 1,
            handle: ReplicaHandleId::new("cuda:0", 0, 1),
        };
        assert!(replica.is_ready());
        assert!(replica.state.is_usable());
    }

    #[test]
    fn loading_state_not_usable() {
        let state = ReplicaState::Loading;
        assert!(!state.is_usable());
    }

    #[test]
    fn evicting_state_still_usable() {
        let state = ReplicaState::Evicting;
        assert!(state.is_usable());
    }

    #[test]
    fn failed_state_carries_reason() {
        let state = ReplicaState::Failed {
            reason: "disk read error".into(),
        };
        assert!(!state.is_usable());
        if let ReplicaState::Failed { reason } = state {
            assert_eq!(reason, "disk read error");
        }
    }

    #[test]
    fn hotter_than_comparison() {
        let device_replica = ObjectReplica {
            object: sample_object(),
            placement: Placement::Local(LocalPlacement::Device {
                device_id: 0,
                memory: DeviceMemoryKind::Vram,
            }),
            bytes: 4096,
            state: ReplicaState::Ready,
            generation: 1,
            handle: ReplicaHandleId::new("cuda:0", 0, 1),
        };
        let host_placement = Placement::Local(LocalPlacement::Host { pinned: false });
        assert!(device_replica.is_hotter_than(&host_placement));

        let disk_placement = Placement::Local(LocalPlacement::Disk { volume: None });
        assert!(device_replica.is_hotter_than(&disk_placement));
    }

    #[test]
    fn handle_display() {
        let handle = ReplicaHandleId::new("cuda:0", 7, 2);
        assert_eq!(handle.to_string(), "cuda:0@slot7:gen2");
    }
}
