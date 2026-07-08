//! Storage object descriptor and kind.

use super::id::StorageObjectId;
use super::layout::StorageLayout;

/// Broad category of a storage object. Determines which subsystem manages it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageObjectKind {
    // ── v1: immutable loadable objects ──
    ArtifactTensor,
    ArtifactTensorRows,
    ExpertMatrix,
    ExpertBundle,
    WeightPackChunk,
    // ── v2+: not in v1 catalog ──
    KvPage,
    DecodeArenaBuffer,
    GraphExternal,
    Opaque,
}

/// Whether an object's bytes can change after creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageMutability {
    /// Bytes never change (expert weights, artifact tensors, WeightPack chunks).
    Immutable,
    /// Bytes can be updated in place (KV pages — future).
    Mutable,
    /// No external bytes to fetch; backend-allocated scratch (decode arena).
    Ephemeral,
}

/// Full metadata for a storage object.
///
/// NOTE: `kind` is redundant with the `StorageObjectId` variant, but kept here
/// so consumers that only have a descriptor can branch on kind without matching
/// the ID enum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageObjectDescriptor {
    pub id: StorageObjectId,
    pub kind: StorageObjectKind,
    pub bytes: u64,
    pub layout: StorageLayout,
    pub mutability: StorageMutability,
}

impl StorageObjectDescriptor {
    /// Convenience: is this object immutable?
    pub fn is_immutable(&self) -> bool {
        matches!(self.mutability, StorageMutability::Immutable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::id::{ModelRevision, StorageObjectId};
    use crate::storage::layout::StorageLayout;

    fn rev(n: u64) -> ModelRevision {
        ModelRevision(n)
    }

    #[test]
    fn descriptor_round_trip() {
        let id = StorageObjectId::ExpertBundle {
            model_revision: rev(1),
            layer: 3,
            expert: 7,
            layout_version: 2,
        };
        let desc = StorageObjectDescriptor {
            id: id.clone(),
            kind: StorageObjectKind::ExpertBundle,
            bytes: 4096,
            layout: StorageLayout::Bytes,
            mutability: StorageMutability::Immutable,
        };
        assert!(desc.is_immutable());
        assert_eq!(desc.kind, StorageObjectKind::ExpertBundle);
        assert_eq!(desc.bytes, 4096);
    }

    #[test]
    fn kv_page_is_mutable() {
        let desc = StorageObjectDescriptor {
            id: StorageObjectId::KvPage {
                session: 0,
                page: 1,
            },
            kind: StorageObjectKind::KvPage,
            bytes: 512,
            layout: StorageLayout::Bytes,
            mutability: StorageMutability::Mutable,
        };
        assert!(!desc.is_immutable());
    }

    #[test]
    fn arena_is_ephemeral() {
        let desc = StorageObjectDescriptor {
            id: StorageObjectId::DecodeArenaBuffer {
                device_id: 0,
                slot: 0,
            },
            kind: StorageObjectKind::DecodeArenaBuffer,
            bytes: 2048,
            layout: StorageLayout::Bytes,
            mutability: StorageMutability::Ephemeral,
        };
        assert!(!desc.is_immutable());
    }
}
