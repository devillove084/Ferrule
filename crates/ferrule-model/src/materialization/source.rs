//! Immutable physical-source descriptors indexed by exact materialization identity.
//!
//! This catalog is not a residency registry and owns no operations, waiters, slots,
//! or generations. It only lets a provider recover the source/install descriptor
//! that was registered for an exact `(resource, ResourceSource)` request. Runtime
//! single-flight and publication remain owned by the runtime load registry.

use std::collections::{BTreeMap, BTreeSet};

use ferrule_common::{Error, FailureReason, MaterializedResourceId, Result};

use crate::checkpoint::CheckpointReadPlan;
use crate::execution::{ResourceBacking, ResourceManifest};

use super::{MaterializationRequest, ResourceSource};

/// One immutable physical source plus provider-owned install semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializationSourceEntry<S> {
    resource: MaterializedResourceId,
    source: ResourceSource,
    read_plan: CheckpointReadPlan,
    descriptor: S,
}

impl<S> MaterializationSourceEntry<S> {
    pub fn new(
        resource: MaterializedResourceId,
        source: ResourceSource,
        read_plan: CheckpointReadPlan,
        descriptor: S,
    ) -> Result<Self> {
        Ok(Self {
            resource,
            source,
            read_plan,
            descriptor,
        })
    }

    pub const fn resource(&self) -> MaterializedResourceId {
        self.resource
    }

    pub const fn source(&self) -> ResourceSource {
        self.source
    }

    pub const fn storage_bytes(&self) -> u64 {
        self.read_plan.storage_bytes()
    }

    pub const fn read_plan(&self) -> &CheckpointReadPlan {
        &self.read_plan
    }

    pub const fn descriptor(&self) -> &S {
        &self.descriptor
    }
}

/// Provider input catalog keyed by exact resource and immutable source identity.
///
/// Multiple source generations for one logical resource may coexist while old
/// physical work drains. Duplicate exact entries are rejected rather than silently
/// replacing descriptor custody.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializationSourceCatalog<S> {
    entries: BTreeMap<(MaterializedResourceId, ResourceSource), MaterializationSourceEntry<S>>,
    resources: BTreeSet<MaterializedResourceId>,
}

impl<S> MaterializationSourceCatalog<S> {
    pub fn new(entries: impl IntoIterator<Item = MaterializationSourceEntry<S>>) -> Result<Self> {
        let mut indexed = BTreeMap::new();
        let mut resources = BTreeSet::new();
        for entry in entries {
            let key = (entry.resource(), entry.source());
            resources.insert(entry.resource());
            if indexed.insert(key, entry).is_some() {
                return Err(Error::Model(format!(
                    "duplicate exact materialization source for {:?}",
                    key.0
                )));
            }
        }
        Ok(Self {
            entries: indexed,
            resources,
        })
    }

    pub fn resolve(
        &self,
        request: MaterializationRequest,
    ) -> std::result::Result<&MaterializationSourceEntry<S>, FailureReason> {
        self.entries
            .get(&(request.resource(), request.source()))
            .ok_or_else(|| {
                if self.resources.contains(&request.resource()) {
                    FailureReason::ProtocolViolation(format!(
                        "materialization request source does not match a registered descriptor for {:?}",
                        request.resource()
                    ))
                } else {
                    FailureReason::ProtocolViolation(format!(
                        "materialization request references unregistered resource {:?}",
                        request.resource()
                    ))
                }
            })
    }

    pub fn get(
        &self,
        resource: MaterializedResourceId,
        source: ResourceSource,
    ) -> Option<&MaterializationSourceEntry<S>> {
        self.entries.get(&(resource, source))
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &MaterializationSourceEntry<S>> {
        self.entries.values()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl MaterializationSourceCatalog<ResourceManifest> {
    /// Build provider descriptors for all checkpoint-backed executable resources.
    /// Runtime-owned mutable resources are registered from transaction custody when
    /// they acquire a spill source and therefore do not belong in this immutable set.
    pub fn from_checkpoint_manifests<'a>(
        manifests: impl IntoIterator<Item = &'a ResourceManifest>,
    ) -> Result<Self> {
        let entries = manifests
            .into_iter()
            .filter_map(|manifest| match manifest.backing() {
                ResourceBacking::Checkpoint { .. } => Some(
                    manifest
                        .checkpoint_bundle_source()
                        .expect("checkpoint resource has immutable bundle custody")
                        .read_plan(manifest.checkpoint_tensors())
                        .and_then(|read_plan| {
                            MaterializationSourceEntry::new(
                                manifest.resource(),
                                manifest
                                    .source()
                                    .expect("checkpoint resource has an immutable source"),
                                read_plan,
                                manifest.clone(),
                            )
                        }),
                ),
                ResourceBacking::RuntimeOwned { .. } => None,
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(entries)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ferrule_common::{
        BackendId, ContentHash, DeviceId, MaterializedResourceKind, ModelInstanceId,
        PayloadEncodingId, SourceGeneration, SourceIdentityHash,
    };

    use super::*;
    use crate::checkpoint::{
        CheckpointBundleSource, CheckpointDType, CheckpointSourceFileIdentity,
        CheckpointTensorSlice,
    };
    use crate::execution::{ResourceLayout, TransformerResourceSlot};
    use crate::{TensorRole, materialization::MaterializationPlacement};

    fn source(generation: u64) -> ResourceSource {
        ResourceSource::new(
            SourceIdentityHash::new([generation as u8; 32]),
            ContentHash::new([9; 32]),
            PayloadEncodingId::new(2),
            SourceGeneration::new(generation),
        )
        .unwrap()
    }

    fn request(resource: MaterializedResourceId, source: ResourceSource) -> MaterializationRequest {
        MaterializationRequest::for_placement(
            MaterializationPlacement::new(
                ModelInstanceId::new(1),
                BackendId::new(2),
                DeviceId::new(3),
            )
            .unwrap(),
            source,
            resource,
        )
        .unwrap()
    }

    fn manifest(resource: MaterializedResourceId, source: ResourceSource) -> ResourceManifest {
        let path = PathBuf::from("model.safetensors");
        ResourceManifest::checkpoint(
            resource,
            CheckpointBundleSource::for_test(
                source,
                [CheckpointSourceFileIdentity::for_test(
                    path.clone(),
                    PathBuf::from("/test/model.safetensors"),
                    16,
                )],
            ),
            [CheckpointTensorSlice {
                name: "weight".into(),
                role: TensorRole::OutputHead,
                path,
                offset: 0,
                bytes: 16,
                dtype: CheckpointDType::Bf16,
                shape: vec![2, 4],
            }],
            ResourceLayout::TensorBundle,
        )
        .unwrap()
    }

    #[test]
    fn exact_source_lookup_does_not_join_resource_generations() {
        let resource = TransformerResourceSlot::Output.parameter();
        let first_manifest = manifest(resource, source(1));
        let second_manifest = manifest(resource, source(2));
        let first = MaterializationSourceEntry::new(
            resource,
            source(1),
            first_manifest
                .checkpoint_bundle_source()
                .unwrap()
                .read_plan(first_manifest.checkpoint_tensors())
                .unwrap(),
            "first",
        )
        .unwrap();
        let second = MaterializationSourceEntry::new(
            resource,
            source(2),
            second_manifest
                .checkpoint_bundle_source()
                .unwrap()
                .read_plan(second_manifest.checkpoint_tensors())
                .unwrap(),
            "second",
        )
        .unwrap();
        let catalog = MaterializationSourceCatalog::new([first, second]).unwrap();

        assert_eq!(
            catalog
                .resolve(request(resource, source(1)))
                .unwrap()
                .descriptor(),
            &"first"
        );
        assert_eq!(
            catalog
                .resolve(request(resource, source(2)))
                .unwrap()
                .descriptor(),
            &"second"
        );
        let unknown = ResourceSource::new(
            SourceIdentityHash::new([3; 32]),
            ContentHash::new([9; 32]),
            PayloadEncodingId::new(2),
            SourceGeneration::new(3),
        )
        .unwrap();
        assert!(matches!(
            catalog.resolve(request(resource, unknown)),
            Err(FailureReason::ProtocolViolation(_))
        ));
    }

    #[test]
    fn checkpoint_catalog_skips_runtime_owned_resources_without_losing_kinds() {
        let parameter = TransformerResourceSlot::Embedding.parameter();
        let routed = MaterializedResourceId::new(MaterializedResourceKind::RoutedExpert, 0, 7);
        let gradient =
            TransformerResourceSlot::Embedding.resource(MaterializedResourceKind::Gradient);
        let manifests = [
            manifest(parameter, source(1)),
            manifest(routed, source(2)),
            ResourceManifest::runtime_owned(gradient, 16, 16).unwrap(),
        ];

        let catalog = MaterializationSourceCatalog::from_checkpoint_manifests(&manifests).unwrap();
        assert_eq!(catalog.len(), 2);
        assert_eq!(
            catalog
                .resolve(request(parameter, source(1)))
                .unwrap()
                .storage_bytes(),
            16
        );
        assert!(catalog.resolve(request(routed, source(2))).is_ok());
        assert!(catalog.get(gradient, source(1)).is_none());
    }
}
