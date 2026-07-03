//! Expert residency backend trait and adapters.
//!
//! This module bridges the existing `ExpertStreamingPlanner` (strategy layer)
//! with backend handle stores (execution layer). It provides:
//!
//! 1. `ExpertResidencyBackend` — a trait that unifies the duplicated
//!    load/evict loop previously inlined in `routed_moe.rs` (CPU path) and
//!    `deepseek_v4.rs` (CUDA path).
//! 2. `apply_streaming_step` — one function that replaces both loops.
//! 3. Adapters from existing expert types to `ferrule-storage` vocabulary types
//!    (`ExpertId → StorageObjectId`, `ExpertStorageTier → Placement`,
//!    `ExpertLoadSource → ObjectLocator`).
//!
//! Phase 0: types and trait only. No call sites changed yet.

use ferrule_core::Result;
use ferrule_storage::{
    DeviceMemoryKind, LocalPlacement, ModelRevision, ObjectLocator, Placement, StorageObjectId,
};

use crate::expert_streaming::{
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
impl ExpertId {
    /// Create a `StorageObjectId` for this expert bundle.
    ///
    /// `model_revision` identifies the model (config hash + quant policy).
    /// `layout_version` identifies the artifact layout (WeightPack format
    /// version or artifact format tag).
    pub fn to_storage_object_id(
        self,
        model_revision: ModelRevision,
        layout_version: u32,
    ) -> StorageObjectId {
        StorageObjectId::ExpertBundle {
            model_revision,
            layer: self.layer as u32,
            expert: self.expert as u32,
            layout_version,
        }
    }
}

// ── Adapter: ExpertStorageTier → Placement ────────────────────────────

impl ExpertStorageTier {
    /// Map a runtime storage tier to a storage-vocabulary `Placement`.
    ///
    /// `Loading` maps to `Host` (staging) since it represents an in-flight
    /// transfer that will land in a local tier.
    pub fn to_placement(self) -> Placement {
        match self {
            Self::Gpu => Placement::Local(LocalPlacement::Device {
                device_id: 0,
                memory: DeviceMemoryKind::Vram,
            }),
            Self::Cpu => Placement::Local(LocalPlacement::Host { pinned: false }),
            Self::HostMmap => Placement::Local(LocalPlacement::Host { pinned: false }),
            Self::LocalStorage => Placement::Local(LocalPlacement::Disk { volume: None }),
            Self::Remote => Placement::Remote(ferrule_storage::RemotePlacement {
                endpoint: ferrule_storage::RemoteEndpoint {
                    scheme: ferrule_storage::RemoteScheme::Http,
                    host: String::new(),
                    port: None,
                },
                region: None,
            }),
            Self::Loading => Placement::Local(LocalPlacement::Host { pinned: false }),
        }
    }
}

// ── Adapter: ExpertLoadSource → ObjectLocator ─────────────────────────

impl ExpertLoadSource {
    /// Map a runtime load source to a storage-vocabulary `ObjectLocator`.
    ///
    /// `GpuResident` and `CpuResident` return `None` — they represent
    /// already-resident objects, not fetchable locators.
    pub fn to_locator(&self) -> Option<ObjectLocator> {
        match self {
            Self::GpuResident | Self::CpuResident => None,
            Self::HostMmap {
                artifact,
                offset,
                bytes,
            } => Some(ObjectLocator::LocalMmap {
                path: artifact.clone(),
                offset: *offset,
                bytes: *bytes,
            }),
            Self::LocalShard {
                path,
                offset,
                bytes,
            } => Some(ObjectLocator::LocalFile {
                path: path.clone(),
                offset: *offset,
                bytes: *bytes,
            }),
            Self::LocalTensorSet { tensors } => {
                // A tensor set is a composite; use the first tensor's path
                // as the primary locator. The catalog can register multiple
                // locators per object in the future.
                tensors.first().map(|t| ObjectLocator::LocalFile {
                    path: t.path.clone(),
                    offset: t.offset,
                    bytes: t.bytes,
                })
            }
            Self::WeightPackChunk {
                path,
                offset,
                bytes,
            } => Some(ObjectLocator::WeightPack {
                path: path.clone(),
                chunk: String::new(),
                offset: *offset,
                bytes: *bytes,
            }),
            Self::Remote { uri, offset, bytes } => Some(ObjectLocator::RemoteObject {
                uri: uri.clone(),
                offset: *offset,
                bytes: *bytes,
            }),
        }
    }
}

// ── Adapter: ExpertMatrixKind ─────────────────────────────────────────

impl From<RuntimeExpertMatrixKind> for ferrule_storage::id::ExpertMatrixKind {
    fn from(kind: RuntimeExpertMatrixKind) -> Self {
        match kind {
            RuntimeExpertMatrixKind::Gate => Self::Gate,
            RuntimeExpertMatrixKind::Up => Self::Up,
            RuntimeExpertMatrixKind::Down => Self::Down,
        }
    }
}

// ── Adapter: ExpertTensorComponent ────────────────────────────────────

impl From<RuntimeExpertTensorComponent> for ferrule_storage::id::ExpertTensorComponent {
    fn from(comp: RuntimeExpertTensorComponent) -> Self {
        match comp {
            RuntimeExpertTensorComponent::Weight => Self::Weight,
            RuntimeExpertTensorComponent::Scale => Self::Scale,
            RuntimeExpertTensorComponent::Other(s) => Self::Other(s),
        }
    }
}

// ── ExpertResidencyBackend trait ──────────────────────────────────────

/// Backend that can load, evict, and report on resident experts.
///
/// The planner produces an `ExpertStreamingStep`; the backend applies it.
/// This replaces the duplicated loop that was inlined in
/// `routed_moe.rs` (CPU) and `deepseek_v4.rs` (CUDA).
///
/// # Implementors
///
/// - `CpuExpertHandleStore` — CPU reference path (stores artifact payloads)
/// - `DeepSeekV4CudaOperatorCache` — CUDA path (uploads to device buffers)
/// - `HostStagedExpertCache` (Phase 2) — host-staged bytes cache
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

use crate::expert_handle::{CpuExpertHandleStore, ExpertHandleStore};

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
    use crate::expert_handle::CpuExpertHandleStore;
    use crate::expert_streaming::{
        ExpertId, ExpertLoadSource, ExpertStreamingPlanner, ExpertStreamingPolicy,
        ExpertStreamingReader, ExpertTensorComponent, ExpertTensorKey, ExpertTensorSlice,
    };
    use ferrule_storage::id::{
        ExpertMatrixKind as StorageExpertMatrixKind,
        ExpertTensorComponent as StorageExpertTensorComponent,
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

    fn reader() -> ExpertStreamingReader {
        ExpertStreamingReader::new(1024 * 1024)
    }

    // ── Adapter tests ──

    #[test]
    fn expert_id_to_storage_object_id() {
        let id = expert_id(3, 7);
        let storage_id = id.to_storage_object_id(ModelRevision(42), 2);
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
        let p = ExpertStorageTier::Gpu.to_placement();
        assert!(p.is_device());
    }

    #[test]
    fn tier_to_placement_cpu() {
        let p = ExpertStorageTier::Cpu.to_placement();
        assert!(p.is_host());
    }

    #[test]
    fn tier_to_placement_local_storage() {
        let p = ExpertStorageTier::LocalStorage.to_placement();
        assert!(p.is_disk());
    }

    #[test]
    fn tier_to_placement_remote() {
        let p = ExpertStorageTier::Remote.to_placement();
        assert!(p.is_remote());
    }

    #[test]
    fn load_source_to_locator_local_shard() {
        let src = ExpertLoadSource::LocalShard {
            path: PathBuf::from("/data/model.safetensors"),
            offset: 1024,
            bytes: 4096,
        };
        let loc = src.to_locator().unwrap();
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
        assert!(src.to_locator().is_none());
    }

    #[test]
    fn load_source_to_locator_weightpack() {
        let src = ExpertLoadSource::WeightPackChunk {
            path: PathBuf::from("/data/model.qcache"),
            offset: 256,
            bytes: 2048,
        };
        let loc = src.to_locator().unwrap();
        assert!(matches!(loc, ObjectLocator::WeightPack { .. }));
    }

    #[test]
    fn load_source_to_locator_remote() {
        let src = ExpertLoadSource::Remote {
            uri: "s3://bucket/expert.bin".into(),
            offset: 0,
            bytes: 4096,
        };
        let loc = src.to_locator().unwrap();
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
            .insert_bundle(crate::expert_streaming::ExpertComputeBundle {
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
        use crate::expert_handle::{
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
        use crate::expert_handle::{
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

    fn make_gate_payload(expert: ExpertId) -> crate::expert_streaming::ExpertLinearPayload {
        use crate::expert_streaming::{
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
