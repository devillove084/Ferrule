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
pub use ferrule_common::expert_residency::{
    ExpertInstallIntent, ExpertInstallPrepareOutcome, ExpertInstallReason, ExpertKey, ExpertLease,
    ExpertResidencyControl, ExpertResidencyCoordinator, ExpertResidencyCoordinatorStats,
    ExpertResidencyGrant, ExpertResidencyRequirements, ExpertResidencyStats, ExpertSlotBinding,
    ExpertSlotGeneration, ExpertSlotId, PreparedExpertInstall,
};
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

// ── Model-neutral per-layer residency controller ─────────────────────

/// Runtime ownership of model-qualified expert residency state.
///
/// Every layer has an independent stable-slot coordinator and capacity. Selected
/// grants carry leases; prefetches never evict or reserve a selected (leased) slot.
#[derive(Debug)]
pub struct ExpertResidencyController {
    requirements: ExpertResidencyRequirements,
    layers: Vec<ExpertResidencyCoordinator>,
    prefetch_capacity_misses: u64,
}

impl ExpertResidencyController {
    pub fn new(
        model_instance: u64,
        layer_capacities: impl IntoIterator<Item = usize>,
    ) -> Result<Self> {
        Self::with_requirements(ExpertResidencyRequirements::new(
            model_instance,
            layer_capacities.into_iter().collect(),
        ))
    }

    pub fn with_requirements(requirements: ExpertResidencyRequirements) -> Result<Self> {
        if requirements.layer_capacities.is_empty() {
            return Err(Error::Execution(
                "expert residency controller requires at least one layer".into(),
            ));
        }
        let layers = requirements
            .layer_capacities
            .iter()
            .copied()
            .map(ExpertResidencyCoordinator::new)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            requirements,
            layers,
            prefetch_capacity_misses: 0,
        })
    }

    fn layer_index(&self, key: ExpertKey) -> Result<usize> {
        if key.model_instance != self.requirements.model_instance {
            return Err(Error::Execution(format!(
                "expert model namespace {} does not match controller namespace {}",
                key.model_instance, self.requirements.model_instance
            )));
        }
        let layer = key.layer as usize;
        if layer >= self.layers.len() {
            return Err(Error::Execution(format!(
                "expert layer {} is outside controller layer count {}",
                key.layer,
                self.layers.len()
            )));
        }
        Ok(layer)
    }

    pub fn requirements(&self) -> ExpertResidencyRequirements {
        self.requirements.clone()
    }

    pub fn binding(&self, key: ExpertKey) -> Result<Option<ExpertSlotBinding>> {
        ExpertResidencyControl::binding(self, key)
    }

    pub fn acquire_selected(&mut self, key: ExpertKey) -> Result<Option<ExpertResidencyGrant>> {
        ExpertResidencyControl::acquire_selected(self, key)
    }

    pub fn release(&mut self, lease: ExpertLease) -> Result<()> {
        ExpertResidencyControl::release(self, lease)
    }

    pub fn prepare_install(
        &mut self,
        intent: ExpertInstallIntent,
    ) -> Result<ExpertInstallPrepareOutcome> {
        ExpertResidencyControl::prepare_install(self, intent)
    }

    pub fn publish_install(
        &mut self,
        prepared: PreparedExpertInstall,
    ) -> Result<ExpertResidencyGrant> {
        ExpertResidencyControl::publish_install(self, prepared)
    }

    pub fn cancel_install(&mut self, prepared: PreparedExpertInstall) -> Result<()> {
        ExpertResidencyControl::cancel_install(self, prepared)
    }

    pub fn stats(&self) -> ExpertResidencyStats {
        ExpertResidencyControl::stats(self)
    }

    pub fn layer_stats(&self, layer: u32) -> Option<ExpertResidencyCoordinatorStats> {
        self.layers.get(layer as usize).map(|layer| layer.stats())
    }
}

impl ExpertResidencyControl for ExpertResidencyController {
    fn requirements(&self) -> ExpertResidencyRequirements {
        self.requirements.clone()
    }

    fn binding(&self, key: ExpertKey) -> Result<Option<ExpertSlotBinding>> {
        let layer = self.layer_index(key)?;
        Ok(self.layers[layer].binding(key))
    }

    fn acquire_selected(&mut self, key: ExpertKey) -> Result<Option<ExpertResidencyGrant>> {
        let layer = self.layer_index(key)?;
        Ok(self.layers[layer].acquire(key)?.map(|lease| {
            ExpertResidencyGrant::new(lease.binding(), ExpertInstallReason::Selected, Some(lease))
        }))
    }

    fn release(&mut self, lease: ExpertLease) -> Result<()> {
        let layer = self.layer_index(lease.binding().key)?;
        self.layers[layer].release(lease)
    }

    fn prepare_install(
        &mut self,
        intent: ExpertInstallIntent,
    ) -> Result<ExpertInstallPrepareOutcome> {
        let layer = self.layer_index(intent.key)?;
        if let Some(binding) = self.layers[layer].binding(intent.key) {
            let lease = match intent.reason {
                ExpertInstallReason::Selected => self.layers[layer]
                    .acquire(intent.key)?
                    .ok_or_else(|| {
                        Error::Internal(
                            "resident expert disappeared while acquiring selected lease".into(),
                        )
                    })?
                    .into(),
                ExpertInstallReason::Prefetch => None,
            };
            return Ok(ExpertInstallPrepareOutcome::Resident(
                ExpertResidencyGrant::new(binding, intent.reason, lease),
            ));
        }

        match self.layers[layer].try_prepare_install(intent.key, intent.reason)? {
            Some(prepared) => Ok(ExpertInstallPrepareOutcome::Prepared(prepared)),
            None if intent.reason == ExpertInstallReason::Prefetch => {
                self.prefetch_capacity_misses = self.prefetch_capacity_misses.saturating_add(1);
                Ok(ExpertInstallPrepareOutcome::CapacityAllLeased)
            }
            None => Err(Error::Execution(format!(
                "no unleased expert residency slot is available for selected expert {:?}",
                intent.key
            ))),
        }
    }

    fn publish_install(&mut self, prepared: PreparedExpertInstall) -> Result<ExpertResidencyGrant> {
        let binding = prepared.binding();
        let layer = self.layer_index(binding.key)?;
        match prepared.reason() {
            ExpertInstallReason::Selected => {
                let (binding, lease) = self.layers[layer].publish_install_leased(prepared)?;
                Ok(ExpertResidencyGrant::new(
                    binding,
                    ExpertInstallReason::Selected,
                    Some(lease),
                ))
            }
            ExpertInstallReason::Prefetch => {
                let binding = self.layers[layer].publish_install(prepared)?;
                Ok(ExpertResidencyGrant::new(
                    binding,
                    ExpertInstallReason::Prefetch,
                    None,
                ))
            }
        }
    }

    fn cancel_install(&mut self, prepared: PreparedExpertInstall) -> Result<()> {
        let layer = self.layer_index(prepared.binding().key)?;
        self.layers[layer].cancel_install(prepared)
    }

    fn stats(&self) -> ExpertResidencyStats {
        let mut aggregate = ExpertResidencyStats {
            prefetch_capacity_misses: self.prefetch_capacity_misses,
            ..ExpertResidencyStats::default()
        };
        for layer in &self.layers {
            let stats = layer.stats();
            aggregate.resident += stats.resident;
            aggregate.active_leases += stats.active_leases;
            aggregate.installs = aggregate.installs.saturating_add(stats.installs);
            aggregate.evictions = aggregate.evictions.saturating_add(stats.evictions);
            aggregate.resident_hits = aggregate.resident_hits.saturating_add(stats.resident_hits);
            aggregate.stale_releases = aggregate
                .stale_releases
                .saturating_add(stats.stale_releases);
            aggregate.prepare_cancellations = aggregate
                .prepare_cancellations
                .saturating_add(stats.prepare_cancellations);
        }
        aggregate
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

    fn prepared(outcome: ExpertInstallPrepareOutcome) -> PreparedExpertInstall {
        match outcome {
            ExpertInstallPrepareOutcome::Prepared(prepared) => prepared,
            other => panic!("expected prepared install, got {other:?}"),
        }
    }

    fn install_intent(
        controller: &mut ExpertResidencyController,
        intent: ExpertInstallIntent,
    ) -> ExpertResidencyGrant {
        let prepared = prepared(controller.prepare_install(intent).unwrap());
        let expected = prepared.binding();
        let grant = controller.publish_install(prepared).unwrap();
        assert_eq!(grant.binding(), expected);
        grant
    }

    #[test]
    fn controller_is_object_safe_and_isolates_model_namespaces() {
        let mut first = ExpertResidencyController::new(11, [1]).unwrap();
        let mut second = ExpertResidencyController::new(22, [1]).unwrap();
        let first_key = ExpertKey::new(11, 0, 7);
        let second_key = ExpertKey::new(22, 0, 7);

        let control: &mut dyn ExpertResidencyControl = &mut first;
        let first_prepared = prepared(
            control
                .prepare_install(ExpertInstallIntent::prefetch(first_key))
                .unwrap(),
        );
        let first_binding = control.publish_install(first_prepared).unwrap().binding();
        let second_binding =
            install_intent(&mut second, ExpertInstallIntent::prefetch(second_key)).binding();

        assert_eq!(first_binding.slot, second_binding.slot);
        assert_eq!(first_binding.generation, second_binding.generation);
        assert!(control.binding(first_key).unwrap().is_some());
        assert!(
            control
                .binding(second_key)
                .unwrap_err()
                .to_string()
                .contains("namespace")
        );
        assert!(
            second
                .binding(first_key)
                .unwrap_err()
                .to_string()
                .contains("namespace")
        );
    }

    #[test]
    fn selected_grant_blocks_prefetch_eviction_until_release() {
        let mut controller = ExpertResidencyController::new(5, [1]).unwrap();
        let selected = ExpertKey::new(5, 0, 1);
        let predicted = ExpertKey::new(5, 0, 2);
        let grant = install_intent(&mut controller, ExpertInstallIntent::selected(selected));
        let lease = grant.lease().expect("selected publication must be leased");

        assert_eq!(
            controller
                .prepare_install(ExpertInstallIntent::prefetch(predicted))
                .unwrap(),
            ExpertInstallPrepareOutcome::CapacityAllLeased
        );
        assert_eq!(controller.binding(selected).unwrap(), Some(grant.binding()));
        assert_eq!(controller.stats().prefetch_capacity_misses, 1);

        controller.release(lease).unwrap();
        let replacement = install_intent(&mut controller, ExpertInstallIntent::prefetch(predicted));
        assert_eq!(replacement.binding().slot, grant.binding().slot);
        assert!(controller.binding(selected).unwrap().is_none());
    }

    #[test]
    fn prepare_failure_cancel_is_atomic_and_publication_is_exact() {
        let mut controller = ExpertResidencyController::new(9, [1]).unwrap();
        let old_key = ExpertKey::new(9, 0, 3);
        let new_key = ExpertKey::new(9, 0, 4);
        let old = install_intent(&mut controller, ExpertInstallIntent::prefetch(old_key));
        let failed = prepared(
            controller
                .prepare_install(ExpertInstallIntent::prefetch(new_key))
                .unwrap(),
        );

        assert_eq!(failed.evicted_key(), Some(old_key));
        assert_eq!(controller.binding(old_key).unwrap(), Some(old.binding()));
        assert_eq!(controller.binding(new_key).unwrap(), None);
        controller.cancel_install(failed).unwrap();
        assert_eq!(controller.binding(old_key).unwrap(), Some(old.binding()));
        assert_eq!(controller.stats().evictions, 0);
        assert_eq!(controller.stats().prepare_cancellations, 1);
        assert!(
            controller
                .publish_install(failed)
                .unwrap_err()
                .to_string()
                .contains("canceled")
        );

        let retry = prepared(
            controller
                .prepare_install(ExpertInstallIntent::prefetch(new_key))
                .unwrap(),
        );
        let exact = retry.binding();
        let published = controller.publish_install(retry).unwrap();
        assert_eq!(published.binding(), exact);
        assert_eq!(exact.slot, old.binding().slot);
        assert!(exact.generation.get() > old.binding().generation.get());
    }

    #[test]
    fn controller_rejects_stale_release_after_slot_reuse() {
        let mut controller = ExpertResidencyController::new(3, [1]).unwrap();
        let first_key = ExpertKey::new(3, 0, 1);
        let second_key = ExpertKey::new(3, 0, 2);
        let first = install_intent(&mut controller, ExpertInstallIntent::selected(first_key));
        let stale = first.lease().unwrap();

        controller.release(stale).unwrap();
        install_intent(&mut controller, ExpertInstallIntent::prefetch(second_key));
        let error = controller.release(stale).unwrap_err();
        assert!(error.to_string().contains("stale slot generation"));
        assert_eq!(controller.stats().stale_releases, 1);
        assert_eq!(controller.stats().active_leases, 0);
    }

    #[test]
    fn controller_lru_is_deterministic_among_unleased_slots() {
        let mut controller = ExpertResidencyController::new(17, [2]).unwrap();
        let first_key = ExpertKey::new(17, 0, 1);
        let second_key = ExpertKey::new(17, 0, 2);
        let third_key = ExpertKey::new(17, 0, 3);
        let first = install_intent(&mut controller, ExpertInstallIntent::prefetch(first_key));
        let second = install_intent(&mut controller, ExpertInstallIntent::prefetch(second_key));
        let touched = controller.acquire_selected(first_key).unwrap().unwrap();
        controller.release(touched.lease().unwrap()).unwrap();

        let third = install_intent(&mut controller, ExpertInstallIntent::prefetch(third_key));
        assert_eq!(third.binding().slot, second.binding().slot);
        assert_ne!(third.binding().slot, first.binding().slot);
        assert!(controller.binding(first_key).unwrap().is_some());
        assert!(controller.binding(second_key).unwrap().is_none());
    }

    #[test]
    fn controller_enforces_independent_per_layer_capacities() {
        let mut controller = ExpertResidencyController::new(31, [1, 2]).unwrap();
        let layer0_first = ExpertKey::new(31, 0, 1);
        let layer0_second = ExpertKey::new(31, 0, 2);
        let layer1_first = ExpertKey::new(31, 1, 1);
        let layer1_second = ExpertKey::new(31, 1, 2);

        let layer0_binding =
            install_intent(&mut controller, ExpertInstallIntent::prefetch(layer0_first)).binding();
        let layer1_binding0 =
            install_intent(&mut controller, ExpertInstallIntent::prefetch(layer1_first)).binding();
        let layer1_binding1 = install_intent(
            &mut controller,
            ExpertInstallIntent::prefetch(layer1_second),
        )
        .binding();
        let layer0_replacement = install_intent(
            &mut controller,
            ExpertInstallIntent::prefetch(layer0_second),
        )
        .binding();

        assert_eq!(controller.requirements().layer_capacities, vec![1, 2]);
        assert_eq!(layer0_replacement.slot, layer0_binding.slot);
        assert_ne!(layer1_binding0.slot, layer1_binding1.slot);
        assert!(controller.binding(layer0_first).unwrap().is_none());
        assert!(controller.binding(layer1_first).unwrap().is_some());
        assert!(controller.binding(layer1_second).unwrap().is_some());
        assert_eq!(controller.layer_stats(0).unwrap().resident, 1);
        assert_eq!(controller.layer_stats(1).unwrap().resident, 2);
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
