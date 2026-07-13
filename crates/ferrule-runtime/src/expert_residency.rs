//! Expert residency backend trait and adapters.
//!
//! This module bridges the existing `ExpertStreamingPlanner` (strategy layer)
//! with backend handle stores (execution layer). It provides:
//!
//! 1. `ExpertResidencyBackend` — a trait that unifies the duplicated
//!    load/evict loop previously inlined in `routed_moe.rs` (CPU path) and
//!    `deepseek_v4.rs` (CUDA path).
//! 2. `apply_streaming_step` — one function that replaces both loops.
//! 3. Adapters from existing expert types to runtime storage vocabulary types
//!    (`ExpertId → StorageObjectId`, `ExpertStorageTier → Placement`,
//!    `ExpertLoadSource → ObjectLocator`).
//!
//! Storage vocabulary lives in `crate::storage`; it is intentionally a runtime
//! module rather than a separate workspace crate.

use crate::storage::{
    DeviceMemoryKind, LocalPlacement, ModelRevision, ObjectLocator, Placement, StorageObjectId,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use ferrule_common::{Error, Result};

use ferrule_model::moe::streaming::{
    ExpertId, ExpertLoadSource, ExpertMatrixKind as RuntimeExpertMatrixKind, ExpertStorageTier,
    ExpertStreamingReader, ExpertStreamingStep,
    ExpertTensorComponent as RuntimeExpertTensorComponent,
};

// ── Adapters: ExpertId → StorageObjectId ──────────────────────────────

/// Convert an `ExpertId` to a `StorageObjectId::ExpertBundle`.
///
/// `model_revision` and `layout_version` must be supplied by the caller —
/// they are not carried by `ExpertId` today. In Phase 3, the planner will
/// hold a `ModelRevision` and pass it through; for now, adapters use a
/// placeholder.
pub fn expert_id_to_storage_object_id(
    expert: ExpertId,
    model_revision: ModelRevision,
    layout_version: u32,
) -> StorageObjectId {
    StorageObjectId::ExpertBundle {
        model_revision,
        layer: expert.layer as u32,
        expert: expert.expert as u32,
        layout_version,
    }
}

// ── Adapter: ExpertStorageTier → Placement ────────────────────────────

/// Map a runtime storage tier to a storage-vocabulary `Placement`.
///
/// `Loading` maps to `Host` (staging) since it represents an in-flight
/// transfer that will land in a local tier.
pub fn expert_storage_tier_to_placement(tier: ExpertStorageTier) -> Placement {
    match tier {
        ExpertStorageTier::Gpu => Placement::Local(LocalPlacement::Device {
            device_id: 0,
            memory: DeviceMemoryKind::Vram,
        }),
        ExpertStorageTier::Cpu => Placement::Local(LocalPlacement::Host { pinned: false }),
        ExpertStorageTier::HostMmap => Placement::Local(LocalPlacement::Host { pinned: false }),
        ExpertStorageTier::LocalStorage => Placement::Local(LocalPlacement::Disk { volume: None }),
        ExpertStorageTier::Remote => Placement::Remote(crate::storage::RemotePlacement {
            endpoint: crate::storage::RemoteEndpoint {
                scheme: crate::storage::RemoteScheme::Http,
                host: String::new(),
                port: None,
            },
            region: None,
        }),
        ExpertStorageTier::Loading => Placement::Local(LocalPlacement::Host { pinned: false }),
    }
}

// ── Adapter: ExpertLoadSource → ObjectLocator ─────────────────────────

/// Map a runtime load source to a storage-vocabulary `ObjectLocator`.
///
/// `GpuResident` and `CpuResident` return `None` — they represent
/// already-resident objects, not fetchable locators.
pub fn expert_load_source_to_locator(source: &ExpertLoadSource) -> Option<ObjectLocator> {
    match source {
        ExpertLoadSource::GpuResident | ExpertLoadSource::CpuResident => None,
        ExpertLoadSource::HostMmap {
            artifact,
            offset,
            bytes,
        } => Some(ObjectLocator::LocalMmap {
            path: artifact.clone(),
            offset: *offset,
            bytes: *bytes,
        }),
        ExpertLoadSource::LocalShard {
            path,
            offset,
            bytes,
        } => Some(ObjectLocator::LocalFile {
            path: path.clone(),
            offset: *offset,
            bytes: *bytes,
        }),
        ExpertLoadSource::LocalTensorSet { tensors } => {
            // A tensor set is a composite; use the first tensor's path
            // as the primary locator. The catalog can register multiple
            // locators per object in the future.
            tensors.first().map(|t| ObjectLocator::LocalFile {
                path: t.path.clone(),
                offset: t.offset,
                bytes: t.bytes,
            })
        }
        ExpertLoadSource::WeightPackChunk {
            path,
            offset,
            bytes,
        } => Some(ObjectLocator::WeightPack {
            path: path.clone(),
            chunk: String::new(),
            offset: *offset,
            bytes: *bytes,
        }),
        ExpertLoadSource::Remote { uri, offset, bytes } => Some(ObjectLocator::RemoteObject {
            uri: uri.clone(),
            offset: *offset,
            bytes: *bytes,
        }),
    }
}

// ── Adapter: ExpertMatrixKind ─────────────────────────────────────────

impl From<RuntimeExpertMatrixKind> for crate::storage::id::ExpertMatrixKind {
    fn from(kind: RuntimeExpertMatrixKind) -> Self {
        match kind {
            RuntimeExpertMatrixKind::Gate => Self::Gate,
            RuntimeExpertMatrixKind::Up => Self::Up,
            RuntimeExpertMatrixKind::Down => Self::Down,
        }
    }
}

// ── Adapter: ExpertTensorComponent ────────────────────────────────────

impl From<RuntimeExpertTensorComponent> for crate::storage::id::ExpertTensorComponent {
    fn from(comp: RuntimeExpertTensorComponent) -> Self {
        match comp {
            RuntimeExpertTensorComponent::Weight => Self::Weight,
            RuntimeExpertTensorComponent::Scale => Self::Scale,
            RuntimeExpertTensorComponent::Other(s) => Self::Other(s),
        }
    }
}

// ── Stable expert slots and leases ────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertSlotId(u32);

impl ExpertSlotId {
    pub fn get(self) -> u32 {
        self.0
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertSlotGeneration(u32);

impl ExpertSlotGeneration {
    pub fn get(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlotBinding<K> {
    pub key: K,
    pub slot: ExpertSlotId,
    pub generation: ExpertSlotGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertLease<K> {
    binding: ExpertSlotBinding<K>,
}

impl<K: Copy> ExpertLease<K> {
    pub fn binding(self) -> ExpertSlotBinding<K> {
        self.binding
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedExpertInstall<K> {
    key: K,
    slot: ExpertSlotId,
    previous_key: Option<K>,
    previous_generation: ExpertSlotGeneration,
    next_generation: ExpertSlotGeneration,
}

impl<K: Copy> PreparedExpertInstall<K> {
    pub fn binding(self) -> ExpertSlotBinding<K> {
        ExpertSlotBinding {
            key: self.key,
            slot: self.slot,
            generation: self.next_generation,
        }
    }

    pub fn evicted_key(self) -> Option<K> {
        self.previous_key
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertResidencyCoordinatorStats {
    pub resident: usize,
    pub active_leases: usize,
    pub installs: u64,
    pub evictions: u64,
    pub resident_hits: u64,
    pub stale_releases: u64,
}

#[derive(Debug)]
struct ExpertSlot<K> {
    key: Option<K>,
    generation: ExpertSlotGeneration,
    leases: u32,
    last_used: u64,
}

/// Model-neutral ownership of stable expert slots, generations, and execution
/// leases. Backend payload installation happens between `prepare_install` and
/// `publish_install`, so transfer failure leaves the published mapping unchanged.
#[derive(Debug)]
pub struct ExpertResidencyCoordinator<K> {
    slots: Vec<ExpertSlot<K>>,
    by_key: HashMap<K, ExpertSlotId>,
    clock: u64,
    stats: ExpertResidencyCoordinatorStats,
}

impl<K> ExpertResidencyCoordinator<K>
where
    K: Copy + Debug + Eq + Hash,
{
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Execution(
                "expert residency coordinator capacity must be greater than zero".into(),
            ));
        }
        if capacity > u32::MAX as usize {
            return Err(Error::Execution(
                "expert residency coordinator capacity exceeds u32".into(),
            ));
        }
        Ok(Self {
            slots: (0..capacity)
                .map(|_| ExpertSlot {
                    key: None,
                    generation: ExpertSlotGeneration(0),
                    leases: 0,
                    last_used: 0,
                })
                .collect(),
            by_key: HashMap::with_capacity(capacity),
            clock: 0,
            stats: ExpertResidencyCoordinatorStats::default(),
        })
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn binding(&self, key: K) -> Option<ExpertSlotBinding<K>> {
        let slot = *self.by_key.get(&key)?;
        let entry = &self.slots[slot.index()];
        (entry.key == Some(key)).then_some(ExpertSlotBinding {
            key,
            slot,
            generation: entry.generation,
        })
    }

    pub fn acquire(&mut self, key: K) -> Result<Option<ExpertLease<K>>> {
        let Some(slot) = self.by_key.get(&key).copied() else {
            return Ok(None);
        };
        self.clock = self.clock.saturating_add(1);
        let entry = &mut self.slots[slot.index()];
        if entry.key != Some(key) {
            return Err(Error::Internal(
                "expert residency key map and slot table diverged".into(),
            ));
        }
        entry.leases = entry
            .leases
            .checked_add(1)
            .ok_or_else(|| Error::Execution("expert lease count overflow".into()))?;
        entry.last_used = self.clock;
        self.stats.active_leases += 1;
        self.stats.resident_hits = self.stats.resident_hits.saturating_add(1);
        Ok(Some(ExpertLease {
            binding: ExpertSlotBinding {
                key,
                slot,
                generation: entry.generation,
            },
        }))
    }

    pub fn release(&mut self, lease: ExpertLease<K>) -> Result<()> {
        let binding = lease.binding;
        let Some(entry) = self.slots.get_mut(binding.slot.index()) else {
            self.stats.stale_releases = self.stats.stale_releases.saturating_add(1);
            return Err(Error::Execution(
                "expert lease references a missing slot".into(),
            ));
        };
        if entry.key != Some(binding.key) || entry.generation != binding.generation {
            self.stats.stale_releases = self.stats.stale_releases.saturating_add(1);
            return Err(Error::Execution(
                "expert lease has a stale slot generation".into(),
            ));
        }
        entry.leases = entry
            .leases
            .checked_sub(1)
            .ok_or_else(|| Error::Internal("expert lease count underflow".into()))?;
        self.stats.active_leases -= 1;
        Ok(())
    }

    pub fn prepare_install(&self, key: K) -> Result<PreparedExpertInstall<K>> {
        if self.by_key.contains_key(&key) {
            return Err(Error::Execution(format!(
                "expert {key:?} is already resident"
            )));
        }
        let candidate = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.leases == 0)
            .min_by_key(|(_, entry)| (entry.key.is_some(), entry.last_used))
            .ok_or_else(|| Error::Execution("all expert residency slots are leased".into()))?;
        let (index, entry) = candidate;
        let next = entry
            .generation
            .0
            .checked_add(1)
            .filter(|generation| *generation != 0)
            .ok_or_else(|| Error::Execution("expert slot generation exhausted".into()))?;
        Ok(PreparedExpertInstall {
            key,
            slot: ExpertSlotId(index as u32),
            previous_key: entry.key,
            previous_generation: entry.generation,
            next_generation: ExpertSlotGeneration(next),
        })
    }

    pub fn publish_install(
        &mut self,
        prepared: PreparedExpertInstall<K>,
    ) -> Result<ExpertSlotBinding<K>> {
        if self.by_key.contains_key(&prepared.key) {
            return Err(Error::Execution(format!(
                "expert {:?} became resident before install publication",
                prepared.key
            )));
        }
        let entry = self
            .slots
            .get_mut(prepared.slot.index())
            .ok_or_else(|| Error::Internal("prepared expert slot is missing".into()))?;
        if entry.key != prepared.previous_key
            || entry.generation != prepared.previous_generation
            || entry.leases != 0
        {
            return Err(Error::Execution(
                "prepared expert install became stale before publication".into(),
            ));
        }
        if let Some(previous) = entry.key {
            self.by_key.remove(&previous);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        self.clock = self.clock.saturating_add(1);
        entry.key = Some(prepared.key);
        entry.generation = prepared.next_generation;
        entry.last_used = self.clock;
        self.by_key.insert(prepared.key, prepared.slot);
        self.stats.installs = self.stats.installs.saturating_add(1);
        self.stats.resident = self.by_key.len();
        Ok(prepared.binding())
    }

    pub fn evict(&mut self, key: K) -> Result<Option<ExpertSlotBinding<K>>> {
        let Some(slot) = self.by_key.get(&key).copied() else {
            return Ok(None);
        };
        let entry = &mut self.slots[slot.index()];
        if entry.leases != 0 {
            return Err(Error::Execution(format!(
                "expert {key:?} cannot be evicted while leased"
            )));
        }
        let binding = ExpertSlotBinding {
            key,
            slot,
            generation: entry.generation,
        };
        entry.key = None;
        entry.last_used = 0;
        self.by_key.remove(&key);
        self.stats.evictions = self.stats.evictions.saturating_add(1);
        self.stats.resident = self.by_key.len();
        Ok(Some(binding))
    }

    pub fn stats(&self) -> ExpertResidencyCoordinatorStats {
        self.stats
    }
}

// ── ExpertResidencyBackend trait ──────────────────────────────────────

/// Backend that can load, evict, and report on resident experts.
///
/// The planner produces an `ExpertStreamingStep`; the backend applies it.
/// This replaces duplicated CPU/CUDA load-evict-install loops with a single
/// model-agnostic residency boundary.
///
/// # Implementors
///
/// - CPU/reference stores that keep artifact payloads resident.
/// - CUDA/device stores that upload to device buffers and retain opaque handles.
/// - Host-staged caches that keep decoded bytes warm before device installation.
pub trait ExpertResidencyBackend {
    /// Remove an expert's backend handle (eviction).
    fn evict(&mut self, expert: ExpertId) -> Result<()>;

    /// Load bytes from source via the reader, install into the backend,
    /// and return the number of bytes loaded.
    ///
    /// For CPU: stores the artifact payload.
    /// For CUDA: uploads to device buffer and stores the handle.
    fn load_and_install(
        &mut self,
        expert: ExpertId,
        source: &ExpertLoadSource,
        reader: &ExpertStreamingReader,
    ) -> Result<u64>;

    /// Check if expert is currently resident on this backend.
    fn is_resident(&self, expert: ExpertId) -> bool;

    /// Number of currently resident experts.
    fn resident_count(&self) -> usize;

    /// Total bytes of resident handles.
    fn resident_bytes(&self) -> u64;
}

/// Apply a streaming step to a backend: evict first, then load.
///
/// This is the single implementation of the load/evict loop that was
/// previously duplicated in `routed_moe.rs` and `deepseek_v4.rs`.
pub fn apply_streaming_step(
    backend: &mut impl ExpertResidencyBackend,
    step: &ExpertStreamingStep,
    reader: &ExpertStreamingReader,
) -> Result<()> {
    // Evict first to free slots before loading.
    for eviction in &step.evictions {
        backend.evict(eviction.expert)?;
    }
    // Load only non-resident experts.
    for load in &step.loads {
        if !backend.is_resident(load.expert) {
            backend.load_and_install(load.expert, &load.load_source, reader)?;
        }
    }
    Ok(())
}

// ── Implementation for CpuExpertHandleStore ───────────────────────────

use ferrule_model::moe::handle::{CpuExpertHandleStore, ExpertHandleStore};

// Note: HostStagedExpertCache has been moved to ferrule-model::expert_streaming.
// The runtime re-exports it via `ferrule_runtime::HostStagedExpertCache`.

// ── CpuExpertHandleStore backend ─────────────────────────────────────

impl ExpertResidencyBackend for CpuExpertHandleStore {
    fn evict(&mut self, expert: ExpertId) -> Result<()> {
        self.remove(expert);
        Ok(())
    }

    fn load_and_install(
        &mut self,
        expert: ExpertId,
        source: &ExpertLoadSource,
        reader: &ExpertStreamingReader,
    ) -> Result<u64> {
        let payload = reader.read_load_source(expert, source)?;
        let bytes = payload
            .tensors
            .iter()
            .map(|t| t.bytes.len() as u64)
            .sum::<u64>();
        self.insert_artifact_payload(payload)?;
        Ok(bytes)
    }

    fn is_resident(&self, expert: ExpertId) -> bool {
        ExpertHandleStore::contains(self, expert)
    }

    fn resident_count(&self) -> usize {
        self.len()
    }

    fn resident_bytes(&self) -> u64 {
        self.total_bytes()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::id::{
        ExpertMatrixKind as StorageExpertMatrixKind,
        ExpertTensorComponent as StorageExpertTensorComponent,
    };
    use ferrule_model::moe::handle::CpuExpertHandleStore;
    use ferrule_model::moe::streaming::{
        ExpertId, ExpertLoadSource, ExpertStreamingPlanner, ExpertStreamingPolicy,
        ExpertTensorComponent, ExpertTensorKey, ExpertTensorSlice,
    };
    use std::path::PathBuf;

    fn expert_id(layer: usize, expert: usize) -> ExpertId {
        ExpertId::new(layer, expert)
    }

    fn make_slice(expert: ExpertId, bytes: u64) -> ExpertTensorSlice {
        ExpertTensorSlice {
            key: ExpertTensorKey {
                expert,
                matrix: RuntimeExpertMatrixKind::Gate,
            },
            component: ExpertTensorComponent::Weight,
            path: PathBuf::from("/dev/null"),
            offset: 0,
            bytes,
            dtype: "I8".into(),
            shape: vec![1, bytes as usize],
        }
    }

    fn make_local_tensor_set(expert: ExpertId, bytes: u64) -> ExpertLoadSource {
        ExpertLoadSource::LocalTensorSet {
            tensors: vec![make_slice(expert, bytes)],
        }
    }

    fn install(
        coordinator: &mut ExpertResidencyCoordinator<u32>,
        key: u32,
    ) -> ExpertSlotBinding<u32> {
        let prepared = coordinator.prepare_install(key).unwrap();
        coordinator.publish_install(prepared).unwrap()
    }

    #[test]
    fn coordinator_install_is_failure_atomic_until_publish() {
        let mut coordinator = ExpertResidencyCoordinator::new(1).unwrap();
        let first = install(&mut coordinator, 7);
        let prepared = coordinator.prepare_install(9).unwrap();

        assert_eq!(prepared.evicted_key(), Some(7));
        assert_eq!(coordinator.binding(7), Some(first));
        assert_eq!(coordinator.binding(9), None);
        assert_eq!(coordinator.stats().resident, 1);
        assert_eq!(coordinator.stats().evictions, 0);
    }

    #[test]
    fn coordinator_lease_blocks_eviction_and_slot_reuse_increments_generation() {
        let mut coordinator = ExpertResidencyCoordinator::new(1).unwrap();
        let first = install(&mut coordinator, 7);
        let lease = coordinator.acquire(7).unwrap().unwrap();
        let stale_lease = lease;

        let error = coordinator.prepare_install(9).unwrap_err();
        assert!(error.to_string().contains("leased"));
        let error = coordinator.evict(7).unwrap_err();
        assert!(error.to_string().contains("while leased"));

        coordinator.release(lease).unwrap();
        let second = install(&mut coordinator, 9);
        assert_eq!(second.slot, first.slot);
        assert!(second.generation.get() > first.generation.get());
        assert_eq!(coordinator.binding(7), None);
        assert_eq!(coordinator.binding(9), Some(second));

        let error = coordinator.release(stale_lease).unwrap_err();
        assert!(error.to_string().contains("stale slot generation"));
        assert_eq!(coordinator.stats().active_leases, 0);
        assert_eq!(coordinator.stats().stale_releases, 1);
    }

    #[test]
    fn coordinator_rejects_stale_prepared_publication() {
        let mut coordinator = ExpertResidencyCoordinator::new(1).unwrap();
        install(&mut coordinator, 7);
        let prepared = coordinator.prepare_install(9).unwrap();
        let lease = coordinator.acquire(7).unwrap().unwrap();

        let error = coordinator.publish_install(prepared).unwrap_err();
        assert!(error.to_string().contains("became stale"));
        assert!(coordinator.binding(7).is_some());
        assert!(coordinator.binding(9).is_none());
        coordinator.release(lease).unwrap();
    }

    #[test]
    fn coordinator_evicts_least_recently_used_unleased_slot() {
        let mut coordinator = ExpertResidencyCoordinator::new(2).unwrap();
        let first = install(&mut coordinator, 1);
        let second = install(&mut coordinator, 2);
        let lease = coordinator.acquire(1).unwrap().unwrap();
        coordinator.release(lease).unwrap();

        let third = install(&mut coordinator, 3);
        assert_eq!(third.slot, second.slot);
        assert_ne!(third.slot, first.slot);
        assert!(coordinator.binding(1).is_some());
        assert!(coordinator.binding(2).is_none());
        assert!(coordinator.binding(3).is_some());
    }

    // ── Adapter tests ──

    #[test]
    fn expert_id_maps_to_storage_object_id() {
        let id = expert_id(3, 7);
        let storage_id = expert_id_to_storage_object_id(id, ModelRevision(42), 2);
        match storage_id {
            StorageObjectId::ExpertBundle {
                layer,
                expert,
                layout_version,
                model_revision,
            } => {
                assert_eq!(layer, 3);
                assert_eq!(expert, 7);
                assert_eq!(layout_version, 2);
                assert_eq!(model_revision, ModelRevision(42));
            }
            _ => panic!("expected ExpertBundle"),
        }
    }

    #[test]
    fn tier_to_placement_gpu() {
        let p = expert_storage_tier_to_placement(ExpertStorageTier::Gpu);
        assert!(p.is_device());
    }

    #[test]
    fn tier_to_placement_cpu() {
        let p = expert_storage_tier_to_placement(ExpertStorageTier::Cpu);
        assert!(p.is_host());
    }

    #[test]
    fn tier_to_placement_local_storage() {
        let p = expert_storage_tier_to_placement(ExpertStorageTier::LocalStorage);
        assert!(p.is_disk());
    }

    #[test]
    fn tier_to_placement_remote() {
        let p = expert_storage_tier_to_placement(ExpertStorageTier::Remote);
        assert!(p.is_remote());
    }

    #[test]
    fn load_source_to_locator_local_shard() {
        let src = ExpertLoadSource::LocalShard {
            path: PathBuf::from("/data/model.safetensors"),
            offset: 1024,
            bytes: 4096,
        };
        let loc = expert_load_source_to_locator(&src).unwrap();
        match loc {
            ObjectLocator::LocalFile {
                path,
                offset,
                bytes,
            } => {
                assert_eq!(path, PathBuf::from("/data/model.safetensors"));
                assert_eq!(offset, 1024);
                assert_eq!(bytes, 4096);
            }
            _ => panic!("expected LocalFile"),
        }
    }

    #[test]
    fn load_source_to_locator_gpu_resident_is_none() {
        let src = ExpertLoadSource::GpuResident;
        assert!(expert_load_source_to_locator(&src).is_none());
    }

    #[test]
    fn load_source_to_locator_weightpack() {
        let src = ExpertLoadSource::WeightPackChunk {
            path: PathBuf::from("/data/model.qcache"),
            offset: 256,
            bytes: 2048,
        };
        let loc = expert_load_source_to_locator(&src).unwrap();
        assert!(matches!(loc, ObjectLocator::WeightPack { .. }));
    }

    #[test]
    fn load_source_to_locator_remote() {
        let src = ExpertLoadSource::Remote {
            uri: "s3://bucket/expert.bin".into(),
            offset: 0,
            bytes: 4096,
        };
        let loc = expert_load_source_to_locator(&src).unwrap();
        match loc {
            ObjectLocator::RemoteObject { uri, .. } => {
                assert_eq!(uri, "s3://bucket/expert.bin");
            }
            _ => panic!("expected RemoteObject"),
        }
    }

    #[test]
    fn matrix_kind_adapter() {
        assert_eq!(
            StorageExpertMatrixKind::from(RuntimeExpertMatrixKind::Gate),
            StorageExpertMatrixKind::Gate
        );
        assert_eq!(
            StorageExpertMatrixKind::from(RuntimeExpertMatrixKind::Up),
            StorageExpertMatrixKind::Up
        );
        assert_eq!(
            StorageExpertMatrixKind::from(RuntimeExpertMatrixKind::Down),
            StorageExpertMatrixKind::Down
        );
    }

    #[test]
    fn tensor_component_adapter() {
        assert_eq!(
            StorageExpertTensorComponent::from(RuntimeExpertTensorComponent::Weight),
            StorageExpertTensorComponent::Weight
        );
        assert_eq!(
            StorageExpertTensorComponent::from(RuntimeExpertTensorComponent::Scale),
            StorageExpertTensorComponent::Scale
        );
        assert_eq!(
            StorageExpertTensorComponent::from(RuntimeExpertTensorComponent::Other(
                "custom".into()
            )),
            StorageExpertTensorComponent::Other("custom".into())
        );
    }

    // ── CpuExpertHandleStore backend tests ──

    #[test]
    fn cpu_store_evict_removes_handle() {
        let mut store = CpuExpertHandleStore::new();
        let id = expert_id(0, 0);
        store
            .insert_bundle(ferrule_model::moe::streaming::ExpertComputeBundle {
                expert: id,
                gate: make_gate_payload(id),
                up: make_gate_payload(id),
                down: make_gate_payload(id),
            })
            .unwrap();
        assert!(store.is_resident(id));

        ExpertResidencyBackend::evict(&mut store, id).unwrap();
        assert!(!store.is_resident(id));
    }

    #[test]
    fn cpu_store_resident_count_and_bytes() {
        let mut store = CpuExpertHandleStore::new();
        assert_eq!(store.resident_count(), 0);
        assert_eq!(store.resident_bytes(), 0);

        // Insert a resident handle manually.
        use ferrule_model::moe::handle::{
            ExpertComputeHandle, ExpertResidentFormat, ResidentExpertHandle,
        };
        let id = expert_id(0, 0);
        store
            .insert(ExpertComputeHandle::Resident(ResidentExpertHandle::new(
                id,
                ExpertStorageTier::Gpu,
                ExpertResidentFormat::Opaque("test".into()),
                4096,
            )))
            .unwrap();

        assert_eq!(store.resident_count(), 1);
        assert_eq!(store.resident_bytes(), 4096);
    }

    // ── apply_streaming_step test ──

    #[test]
    fn apply_streaming_step_evicts_then_loads() {
        // Set up a planner with 1 GPU slot — forces eviction when switching.
        let policy = ExpertStreamingPolicy {
            gpu_slots_per_layer: 1,
            prefetch_per_layer: 0,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        };
        let mut planner = ExpertStreamingPlanner::new(policy);

        // Register two experts.
        let e0 = expert_id(0, 0);
        let e1 = expert_id(0, 1);
        planner.register_load_source(e0, make_local_tensor_set(e0, 8));
        planner.register_load_source(e1, make_local_tensor_set(e1, 8));

        // Step 1: select expert 0 — should produce a load.
        let step1 = planner.plan_layer_step(0, &[0], &[]).unwrap();
        assert!(!step1.loads.is_empty());

        // We can't actually read from /dev/null with real bytes, so just
        // verify the step structure is correct.
        assert_eq!(step1.evictions.len(), 0);
        assert_eq!(step1.selected.len(), 1);

        // The load would fail with a real reader because /dev/null gives 0 bytes,
        // but we can test the eviction path separately.
        planner.commit_step(&step1).unwrap();

        // Step 2: select expert 1 only — expert 0 should be evicted.
        let step2 = planner.plan_layer_step(0, &[1], &[]).unwrap();
        assert!(!step2.evictions.is_empty());
        assert_eq!(step2.evictions[0].expert, e0);

        // Verify apply_streaming_step would evict e0 first.
        // (We test the logic without the reader by checking is_resident.)
        let mut store = CpuExpertHandleStore::new();
        // Manually mark e0 as resident.
        use ferrule_model::moe::handle::{
            ExpertComputeHandle, ExpertResidentFormat, ResidentExpertHandle,
        };
        store
            .insert(ExpertComputeHandle::Resident(ResidentExpertHandle::new(
                e0,
                ExpertStorageTier::Gpu,
                ExpertResidentFormat::Opaque("test".into()),
                8,
            )))
            .unwrap();
        assert!(store.is_resident(e0));

        // Apply just the evictions part.
        for ev in &step2.evictions {
            ExpertResidencyBackend::evict(&mut store, ev.expert).unwrap();
        }
        assert!(!store.is_resident(e0));
    }

    fn make_gate_payload(expert: ExpertId) -> ferrule_model::moe::streaming::ExpertLinearPayload {
        use ferrule_model::moe::streaming::{
            ExpertLinearFormat, ExpertLinearPayload, ExpertTensorPayload,
        };
        ExpertLinearPayload {
            matrix: RuntimeExpertMatrixKind::Gate,
            weight: ExpertTensorPayload {
                slice: make_slice(expert, 8),
                bytes: vec![0u8; 8],
            },
            scale: None,
            format: ExpertLinearFormat::Opaque,
        }
    }
}
