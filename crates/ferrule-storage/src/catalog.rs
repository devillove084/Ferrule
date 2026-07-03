//! Storage catalog trait — maps object IDs to descriptors and locators.
//!
//! The catalog owns object descriptors and locators. It is the "directory"
//! telling you where an object can be found. It does no scheduling, no
//! prefetch, and holds no CUDA handles.

use std::collections::HashMap;

use crate::descriptor::StorageObjectDescriptor;
use crate::id::StorageObjectId;
use crate::locator::ObjectLocator;

/// Maps `StorageObjectId` → descriptor + locator set.
pub trait StorageCatalog {
    fn descriptor(&self, id: &StorageObjectId) -> Option<&StorageObjectDescriptor>;
    fn locators(&self, id: &StorageObjectId) -> &[ObjectLocator];
}

/// A simple in-memory catalog builder, useful for testing and Phase 0 bootstrapping.
#[derive(Debug, Default)]
pub struct StorageCatalogBuilder {
    descriptors: HashMap<StorageObjectId, StorageObjectDescriptor>,
    locators: HashMap<StorageObjectId, Vec<ObjectLocator>>,
}

impl StorageCatalogBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a descriptor with its locators.
    pub fn register(
        &mut self,
        descriptor: StorageObjectDescriptor,
        locators: Vec<ObjectLocator>,
    ) -> &mut Self {
        let id = descriptor.id.clone();
        self.descriptors.insert(id.clone(), descriptor);
        self.locators.insert(id, locators);
        self
    }

    /// Register with explicit ID to avoid clone issues.
    pub fn register_with_id(
        &mut self,
        id: StorageObjectId,
        descriptor: StorageObjectDescriptor,
        locators: Vec<ObjectLocator>,
    ) -> &mut Self {
        self.descriptors.insert(id.clone(), descriptor);
        self.locators.insert(id, locators);
        self
    }

    pub fn build(self) -> InMemoryStorageCatalog {
        InMemoryStorageCatalog {
            descriptors: self.descriptors,
            locators: self.locators,
        }
    }
}

/// Simple in-memory catalog.
#[derive(Debug, Default)]
pub struct InMemoryStorageCatalog {
    descriptors: HashMap<StorageObjectId, StorageObjectDescriptor>,
    locators: HashMap<StorageObjectId, Vec<ObjectLocator>>,
}

impl InMemoryStorageCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, descriptor: StorageObjectDescriptor, locators: Vec<ObjectLocator>) {
        self.locators.insert(descriptor.id.clone(), locators);
        self.descriptors.insert(descriptor.id.clone(), descriptor);
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }
}

impl StorageCatalog for InMemoryStorageCatalog {
    fn descriptor(&self, id: &StorageObjectId) -> Option<&StorageObjectDescriptor> {
        self.descriptors.get(id)
    }

    fn locators(&self, id: &StorageObjectId) -> &[ObjectLocator] {
        self.locators.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{StorageMutability, StorageObjectDescriptor, StorageObjectKind};
    use crate::id::{ModelRevision, StorageObjectId};
    use crate::layout::StorageLayout;
    use crate::locator::ObjectLocator;
    use std::path::PathBuf;

    fn rev(n: u64) -> ModelRevision {
        ModelRevision(n)
    }

    fn sample_expert_bundle() -> StorageObjectId {
        StorageObjectId::ExpertBundle {
            model_revision: rev(1),
            layer: 3,
            expert: 7,
            layout_version: 2,
        }
    }

    fn sample_descriptor(id: &StorageObjectId) -> StorageObjectDescriptor {
        StorageObjectDescriptor {
            id: id.clone(),
            kind: StorageObjectKind::ExpertBundle,
            bytes: 4096,
            layout: StorageLayout::Bytes,
            mutability: StorageMutability::Immutable,
        }
    }

    fn sample_locator() -> ObjectLocator {
        ObjectLocator::LocalFile {
            path: PathBuf::from("/data/model.safetensors"),
            offset: 1024,
            bytes: 4096,
        }
    }

    #[test]
    fn catalog_lookup_hit() {
        let mut catalog = InMemoryStorageCatalog::new();
        let id = sample_expert_bundle();
        catalog.insert(sample_descriptor(&id), vec![sample_locator()]);

        let desc = catalog.descriptor(&id).unwrap();
        assert_eq!(desc.kind, StorageObjectKind::ExpertBundle);
        assert_eq!(desc.bytes, 4096);

        let locators = catalog.locators(&id);
        assert_eq!(locators.len(), 1);
        assert_eq!(locators[0].bytes(), 4096);
    }

    #[test]
    fn catalog_lookup_miss() {
        let catalog = InMemoryStorageCatalog::new();
        let id = sample_expert_bundle();
        assert!(catalog.descriptor(&id).is_none());
        assert!(catalog.locators(&id).is_empty());
    }

    #[test]
    fn catalog_builder_round_trip() {
        let id = sample_expert_bundle();
        let desc = sample_descriptor(&id);
        let loc = sample_locator();

        let mut builder = StorageCatalogBuilder::new();
        builder.register_with_id(id.clone(), desc, vec![loc]);
        let catalog = builder.build();

        assert_eq!(catalog.len(), 1);
        assert!(catalog.descriptor(&id).is_some());
        assert_eq!(catalog.locators(&id).len(), 1);
    }

    #[test]
    fn catalog_multiple_objects() {
        let mut catalog = InMemoryStorageCatalog::new();

        for i in 0..5u32 {
            let id = StorageObjectId::ExpertBundle {
                model_revision: rev(1),
                layer: i,
                expert: 0,
                layout_version: 1,
            };
            catalog.insert(sample_descriptor(&id), vec![sample_locator()]);
        }

        assert_eq!(catalog.len(), 5);

        for i in 0..5u32 {
            let id = StorageObjectId::ExpertBundle {
                model_revision: rev(1),
                layer: i,
                expert: 0,
                layout_version: 1,
            };
            assert!(catalog.descriptor(&id).is_some());
        }
    }

    #[test]
    fn catalog_multiple_locators_per_object() {
        let mut catalog = InMemoryStorageCatalog::new();
        let id = sample_expert_bundle();
        catalog.insert(
            sample_descriptor(&id),
            vec![
                ObjectLocator::LocalFile {
                    path: PathBuf::from("/data/a.safetensors"),
                    offset: 0,
                    bytes: 2048,
                },
                ObjectLocator::RemoteCache {
                    key: "expert:3:7".into(),
                    offset: 0,
                    bytes: 2048,
                },
            ],
        );

        let locators = catalog.locators(&id);
        assert_eq!(locators.len(), 2);
        assert!(locators[0].is_local());
        assert!(locators[1].is_remote());
    }
}
