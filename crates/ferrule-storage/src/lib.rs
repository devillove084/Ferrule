//! Ferrule Storage and Residency vocabulary.
//!
//! Pure type definitions and traits for the storage/residency layer. This crate
//! has no backend dependencies — no CUDA, no runtime, no model types. It defines
//! the common vocabulary that `ferrule-runtime` and `ferrule-cuda` will use to
//! coordinate object placement, transfer, and eviction.
//!
//! See `docs/storage-residency-architecture.md` for the full design.
//!
//! # Phase 0 scope
//!
//! Types and traits only. No execution behavior change. Adapters from existing
//! `ExpertId` / `ExpertLoadSource` / `ExpertStorageTier` will live in
//! `ferrule-runtime`, not here.

pub mod catalog;
pub mod descriptor;
pub mod id;
pub mod layout;
pub mod locator;
pub mod placement;
pub mod policy;
pub mod replica;
pub mod residency;
pub mod transfer;

pub use catalog::{InMemoryStorageCatalog, StorageCatalog, StorageCatalogBuilder};
pub use descriptor::{StorageMutability, StorageObjectDescriptor, StorageObjectKind};
pub use id::{ModelRevision, StorageObjectId, TensorRole, WeightPackId, ID_VARIANT_EXPERT_BUNDLE};
pub use layout::{ExpertBundleLayout, KvPageLayout, StorageLayout, TensorLayout};
pub use locator::ObjectLocator;
pub use placement::{
    DeviceMemoryKind, LocalPlacement, Placement, RemoteEndpoint, RemotePlacement, RemoteScheme,
};
pub use policy::{Budgets, EvictionWeights, ResidencyScore, StorageResidencyPolicy};
pub use replica::{BackendId, ObjectReplica, ReplicaHandleId, ReplicaState};
pub use residency::{ResidencyPriority, ResidencyReason, ResidencyRequest};
pub use transfer::{
    MockTransferEngine, TransferEngine, TransferEvent, TransferOutcome, TransferTicket,
};
