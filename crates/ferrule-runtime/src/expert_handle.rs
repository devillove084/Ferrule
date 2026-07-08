//! Expert compute handle stores.
//!
//! Streaming/planning decides which experts must be available. This module owns
//! the runtime mapping from an `ExpertId` to the executable representation that
//! is already resident somewhere. CPU reference tests keep artifact-preserved
//! `ExpertComputeBundle`s here; a CUDA path can store opaque device-resident
//! handles while reusing the same planner and routing semantics.

use std::collections::BTreeMap;

use ferrule_common::{Error, Result};

use crate::expert_streaming::{
    ExpertArtifactPayload, ExpertComputeBundle, ExpertEvictRequest, ExpertId, ExpertStorageTier,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertResidentFormat {
    /// Artifact-preserved packed FP4 expert payload.
    PackedFp4E2M1WithE8M0Scale,
    /// Exact artifact representation is preserved but not represented by a CPU bundle.
    ArtifactPreserved,
    /// Backend-specific handle format. This keeps generic runtime metadata free of
    /// concrete model-family names and lets CUDA/remote backends attach their own
    /// slot registries outside this enum.
    Opaque(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentExpertHandle {
    pub expert: ExpertId,
    pub tier: ExpertStorageTier,
    pub format: ExpertResidentFormat,
    pub bytes: u64,
    pub slot: Option<usize>,
}

impl ResidentExpertHandle {
    pub fn new(
        expert: ExpertId,
        tier: ExpertStorageTier,
        format: ExpertResidentFormat,
        bytes: u64,
    ) -> Self {
        Self {
            expert,
            tier,
            format,
            bytes,
            slot: None,
        }
    }

    pub fn with_slot(mut self, slot: usize) -> Self {
        self.slot = Some(slot);
        self
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertComputeHandle {
    /// CPU/reference executable handle. This may represent an expert that the
    /// planner considers GPU-resident in a tiny fixture; the important property is
    /// that the executor can resolve it without going back to the artifact reader.
    ArtifactBundle(ExpertComputeBundle),
    /// Opaque resident handle for production backends. The CPU reference executor
    /// intentionally does not consume this variant; CUDA executors should downcast
    /// through their own handle store implementation.
    Resident(ResidentExpertHandle),
}

impl ExpertComputeHandle {
    pub fn expert(&self) -> ExpertId {
        match self {
            Self::ArtifactBundle(bundle) => bundle.expert,
            Self::Resident(handle) => handle.expert,
        }
    }

    pub fn as_artifact_bundle(&self) -> Option<&ExpertComputeBundle> {
        match self {
            Self::ArtifactBundle(bundle) => Some(bundle),
            Self::Resident(_) => None,
        }
    }

    pub fn total_bytes(&self) -> u64 {
        match self {
            Self::ArtifactBundle(bundle) => bundle.total_bytes(),
            Self::Resident(handle) => handle.bytes,
        }
    }
}

pub trait ExpertHandleStore {
    fn get(&self, expert: ExpertId) -> Option<&ExpertComputeHandle>;
    fn insert(&mut self, handle: ExpertComputeHandle) -> Result<()>;
    fn remove(&mut self, expert: ExpertId) -> Option<ExpertComputeHandle>;

    fn contains(&self, expert: ExpertId) -> bool {
        self.get(expert).is_some()
    }

    fn insert_artifact_payload(&mut self, payload: ExpertArtifactPayload) -> Result<()> {
        self.insert(ExpertComputeHandle::ArtifactBundle(
            ExpertComputeBundle::from_artifact_payload(payload)?,
        ))
    }

    fn artifact_bundle(&self, expert: ExpertId) -> Result<&ExpertComputeBundle> {
        self.get(expert)
            .ok_or_else(|| {
                Error::Model(format!(
                    "expert handle missing for resident layer {} expert {}",
                    expert.layer, expert.expert
                ))
            })?
            .as_artifact_bundle()
            .ok_or_else(|| {
                Error::Model(format!(
                    "expert handle for layer {} expert {} is not a CPU artifact bundle",
                    expert.layer, expert.expert
                ))
            })
    }

    fn apply_evictions(&mut self, evictions: &[ExpertEvictRequest]) {
        for eviction in evictions {
            self.remove(eviction.expert);
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CpuExpertHandleStore {
    handles: BTreeMap<ExpertId, ExpertComputeHandle>,
}

impl CpuExpertHandleStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.handles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    pub fn insert_bundle(&mut self, bundle: ExpertComputeBundle) -> Result<()> {
        self.insert(ExpertComputeHandle::ArtifactBundle(bundle))
    }

    pub fn insert_resident_handle(&mut self, handle: ResidentExpertHandle) -> Result<()> {
        self.insert(ExpertComputeHandle::Resident(handle))
    }

    pub fn total_bytes(&self) -> u64 {
        self.handles
            .values()
            .map(ExpertComputeHandle::total_bytes)
            .sum()
    }
}

impl ExpertHandleStore for CpuExpertHandleStore {
    fn get(&self, expert: ExpertId) -> Option<&ExpertComputeHandle> {
        self.handles.get(&expert)
    }

    fn insert(&mut self, handle: ExpertComputeHandle) -> Result<()> {
        let expert = handle.expert();
        self.handles.insert(expert, handle);
        Ok(())
    }

    fn remove(&mut self, expert: ExpertId) -> Option<ExpertComputeHandle> {
        self.handles.remove(&expert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_streaming::{
        ExpertLinearFormat, ExpertLinearPayload, ExpertMatrixKind, ExpertTensorComponent,
        ExpertTensorKey, ExpertTensorPayload, ExpertTensorSlice,
    };

    #[test]
    fn cpu_handle_store_inserts_artifact_payload_and_resolves_bundle() {
        let expert = ExpertId::new(1, 2);
        let mut store = CpuExpertHandleStore::new();
        store.insert_artifact_payload(tiny_payload(expert)).unwrap();

        let bundle = store.artifact_bundle(expert).unwrap();
        assert_eq!(bundle.expert, expert);
        assert_eq!(store.len(), 1);
        assert!(store.total_bytes() > 0);
    }

    #[test]
    fn cpu_reference_rejects_opaque_resident_handle() {
        let expert = ExpertId::new(0, 7);
        let mut store = CpuExpertHandleStore::new();
        store
            .insert_resident_handle(ResidentExpertHandle::new(
                expert,
                ExpertStorageTier::Gpu,
                ExpertResidentFormat::Opaque("cuda-slot".into()),
                4096,
            ))
            .unwrap();

        let err = store.artifact_bundle(expert).unwrap_err();
        assert!(err.to_string().contains("not a CPU artifact bundle"));
    }

    #[test]
    fn evictions_remove_handles() {
        let expert = ExpertId::new(0, 0);
        let mut store = CpuExpertHandleStore::new();
        store.insert_artifact_payload(tiny_payload(expert)).unwrap();
        store.apply_evictions(&[ExpertEvictRequest {
            expert,
            target: ExpertStorageTier::LocalStorage,
        }]);
        assert!(!store.contains(expert));
    }

    fn tiny_payload(expert: ExpertId) -> ExpertArtifactPayload {
        ExpertArtifactPayload {
            expert,
            tensors: vec![
                tiny_linear(expert, ExpertMatrixKind::Gate),
                tiny_scale(expert, ExpertMatrixKind::Gate),
                tiny_linear(expert, ExpertMatrixKind::Up),
                tiny_scale(expert, ExpertMatrixKind::Up),
                tiny_linear(expert, ExpertMatrixKind::Down),
                tiny_scale(expert, ExpertMatrixKind::Down),
            ],
        }
    }

    fn tiny_linear(expert: ExpertId, matrix: ExpertMatrixKind) -> ExpertTensorPayload {
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component: ExpertTensorComponent::Weight,
                path: "synthetic.safetensors".into(),
                offset: 0,
                bytes: 32 * 16,
                dtype: "I8".into(),
                shape: vec![32, 16],
            },
            bytes: vec![0u8; 32 * 16],
        }
    }

    fn tiny_scale(expert: ExpertId, matrix: ExpertMatrixKind) -> ExpertTensorPayload {
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component: ExpertTensorComponent::Scale,
                path: "synthetic.safetensors".into(),
                offset: 0,
                bytes: 32,
                dtype: "F8_E8M0".into(),
                shape: vec![32, 1],
            },
            bytes: vec![127u8; 32],
        }
    }

    #[allow(dead_code)]
    fn _assert_payload_shape(linear: &ExpertLinearPayload) {
        assert_eq!(
            linear.format,
            ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features: 32,
                in_features: 32,
                block_size: 32,
            }
        );
    }
}
