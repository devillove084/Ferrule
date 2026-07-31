//! Prepared resource manifests for materialization-aware execution.
//!
//! A manifest describes the stable, model-relative identity and initial backing of
//! one independently materializable resource. Destination placement and destination
//! generations are deliberately absent: the runtime assigns them when it acquires a
//! residency lease. Mutable runtime resources likewise receive their current source
//! generation from transaction custody rather than freezing it in this plan.

use std::collections::BTreeSet;

use ferrule_common::{Error, MaterializedResourceId};
use thiserror::Error;

use crate::checkpoint::{CheckpointBundleSource, CheckpointTensorSlice};
use crate::materialization::ResourceSource;

/// Validation failure for a prepared resource or executable stage contract.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExecutionPlanError {
    #[error("checkpoint-backed resource {resource:?} must contain at least one tensor")]
    EmptyCheckpointResource { resource: MaterializedResourceId },
    #[error("checkpoint tensor {tensor:?} in resource {resource:?} has an empty name")]
    EmptyTensorName {
        resource: MaterializedResourceId,
        tensor: String,
    },
    #[error("checkpoint tensor {tensor:?} in resource {resource:?} has zero bytes")]
    EmptyTensorPayload {
        resource: MaterializedResourceId,
        tensor: String,
    },
    #[error("checkpoint tensor {tensor:?} in resource {resource:?} has a zero-sized dimension")]
    EmptyTensorDimension {
        resource: MaterializedResourceId,
        tensor: String,
    },
    #[error("checkpoint tensor {tensor:?} in resource {resource:?} has an overflowing byte extent")]
    TensorExtentOverflow {
        resource: MaterializedResourceId,
        tensor: String,
    },
    #[error(
        "checkpoint tensor {tensor:?} in resource {resource:?} has an overflowing element count"
    )]
    TensorElementCountOverflow {
        resource: MaterializedResourceId,
        tensor: String,
    },
    #[error(
        "checkpoint tensor {tensor:?} in resource {resource:?} declares {actual} bytes, expected {expected} from its dtype and shape"
    )]
    TensorByteSizeMismatch {
        resource: MaterializedResourceId,
        tensor: String,
        expected: u64,
        actual: u64,
    },
    #[error("resource {resource:?} contains duplicate checkpoint tensor name {tensor:?}")]
    DuplicateTensorName {
        resource: MaterializedResourceId,
        tensor: String,
    },
    #[error("resource {resource:?} contains duplicate checkpoint byte range for tensor {tensor:?}")]
    DuplicateTensorRange {
        resource: MaterializedResourceId,
        tensor: String,
    },
    #[error("resource {resource:?} has no captured source snapshot for tensor {tensor:?}")]
    MissingTensorSourceSnapshot {
        resource: MaterializedResourceId,
        tensor: String,
    },
    #[error(
        "checkpoint tensor {tensor:?} in resource {resource:?} ends at byte {end}, beyond captured source length {source_length}"
    )]
    TensorRangeBeyondSource {
        resource: MaterializedResourceId,
        tensor: String,
        end: u64,
        source_length: u64,
    },
    #[error("resource {resource:?} total byte size overflows u64")]
    ResourceByteSizeOverflow { resource: MaterializedResourceId },
    #[error("runtime-owned resource {resource:?} must reserve at least one byte")]
    EmptyRuntimeResource { resource: MaterializedResourceId },
    #[error("resource {resource:?} layout is incompatible with its backing")]
    LayoutBackingMismatch { resource: MaterializedResourceId },
    #[error("resource {resource:?} checkpoint tensors are not in canonical order")]
    NonCanonicalTensorOrder { resource: MaterializedResourceId },
    #[error("resource {resource:?} runtime alignment {alignment} is not a non-zero power of two")]
    InvalidResourceAlignment {
        resource: MaterializedResourceId,
        alignment: u64,
    },
    #[error("workspace alignment {alignment} is not a non-zero power of two")]
    InvalidWorkspaceAlignment { alignment: u64 },
    #[error("duplicate resource manifest for {resource:?}")]
    DuplicateResourceManifest { resource: MaterializedResourceId },
    #[error("stage {stage} declares resource {resource:?} more than once")]
    DuplicateStageResource {
        stage: usize,
        resource: MaterializedResourceId,
    },
    #[error("stage {stage} references unknown resource {resource:?}")]
    UnknownStageResource {
        stage: usize,
        resource: MaterializedResourceId,
    },
}

impl From<ExecutionPlanError> for Error {
    fn from(error: ExecutionPlanError) -> Self {
        Self::Model(format!("invalid prepared executable: {error}"))
    }
}

/// Initial source of a prepared resource.
///
/// Checkpoint-backed resources carry immutable source identity and exact byte
/// ranges. Runtime-owned resources reserve logical capacity only. If a runtime-
/// owned resource is spilled, its current source identity and generation belong to
/// the transaction/custody record, not this immutable manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceBacking {
    Checkpoint {
        bundle_source: CheckpointBundleSource,
        tensors: Box<[CheckpointTensorSlice]>,
    },
    RuntimeOwned {
        capacity_bytes: u64,
    },
}

/// Provider-neutral logical layout expected after materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceLayout {
    /// Preserve the checkpoint's encoded tensor payload and descriptor order.
    CheckpointEncoded,
    /// A semantically ordered tensor bundle. Providers may compile this into a
    /// device-specific layout without exposing that layout in the model contract.
    TensorBundle,
    /// Runtime-created storage such as KV, activations, gradients, or optimizer
    /// state. Alignment is a portable placement requirement, not a device address.
    RuntimeBuffer { alignment: u64 },
}

/// Stable declaration of one independently materializable resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceManifest {
    resource: MaterializedResourceId,
    backing: ResourceBacking,
    layout: ResourceLayout,
    logical_bytes: u64,
}

impl ResourceManifest {
    pub fn checkpoint(
        resource: MaterializedResourceId,
        bundle_source: CheckpointBundleSource,
        tensors: impl IntoIterator<Item = CheckpointTensorSlice>,
        layout: ResourceLayout,
    ) -> Result<Self, ExecutionPlanError> {
        let tensors = tensors.into_iter().collect::<Vec<_>>().into_boxed_slice();
        let logical_bytes = validate_checkpoint_tensors(resource, &bundle_source, &tensors)?;
        let manifest = Self {
            resource,
            backing: ResourceBacking::Checkpoint {
                bundle_source,
                tensors,
            },
            layout,
            logical_bytes,
        };
        manifest.validate_layout()?;
        Ok(manifest)
    }

    pub fn runtime_owned(
        resource: MaterializedResourceId,
        capacity_bytes: u64,
        alignment: u64,
    ) -> Result<Self, ExecutionPlanError> {
        if capacity_bytes == 0 {
            return Err(ExecutionPlanError::EmptyRuntimeResource { resource });
        }
        let manifest = Self {
            resource,
            backing: ResourceBacking::RuntimeOwned { capacity_bytes },
            layout: ResourceLayout::RuntimeBuffer { alignment },
            logical_bytes: capacity_bytes,
        };
        manifest.validate_layout()?;
        Ok(manifest)
    }

    pub const fn resource(&self) -> MaterializedResourceId {
        self.resource
    }

    pub const fn backing(&self) -> &ResourceBacking {
        &self.backing
    }

    pub const fn layout(&self) -> &ResourceLayout {
        &self.layout
    }

    pub const fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    pub const fn source(&self) -> Option<ResourceSource> {
        match &self.backing {
            ResourceBacking::Checkpoint { bundle_source, .. } => Some(bundle_source.source()),
            ResourceBacking::RuntimeOwned { .. } => None,
        }
    }

    pub const fn checkpoint_bundle_source(&self) -> Option<&CheckpointBundleSource> {
        match &self.backing {
            ResourceBacking::Checkpoint { bundle_source, .. } => Some(bundle_source),
            ResourceBacking::RuntimeOwned { .. } => None,
        }
    }

    pub fn checkpoint_tensors(&self) -> &[CheckpointTensorSlice] {
        match &self.backing {
            ResourceBacking::Checkpoint { tensors, .. } => tensors,
            ResourceBacking::RuntimeOwned { .. } => &[],
        }
    }

    fn validate_layout(&self) -> Result<(), ExecutionPlanError> {
        match (&self.backing, &self.layout) {
            (ResourceBacking::Checkpoint { .. }, ResourceLayout::CheckpointEncoded) => Ok(()),
            (ResourceBacking::Checkpoint { .. }, ResourceLayout::TensorBundle) => Ok(()),
            (ResourceBacking::RuntimeOwned { .. }, ResourceLayout::RuntimeBuffer { alignment }) => {
                if *alignment == 0 || !alignment.is_power_of_two() {
                    return Err(ExecutionPlanError::InvalidResourceAlignment {
                        resource: self.resource,
                        alignment: *alignment,
                    });
                }
                Ok(())
            }
            _ => Err(ExecutionPlanError::LayoutBackingMismatch {
                resource: self.resource,
            }),
        }
    }
}

fn validate_checkpoint_tensors(
    resource: MaterializedResourceId,
    bundle_source: &CheckpointBundleSource,
    tensors: &[CheckpointTensorSlice],
) -> Result<u64, ExecutionPlanError> {
    if tensors.is_empty() {
        return Err(ExecutionPlanError::EmptyCheckpointResource { resource });
    }
    if tensors
        .windows(2)
        .any(|pair| checkpoint_tensor_order_key(&pair[0]) > checkpoint_tensor_order_key(&pair[1]))
    {
        return Err(ExecutionPlanError::NonCanonicalTensorOrder { resource });
    }

    let mut names = BTreeSet::new();
    let mut ranges = BTreeSet::new();
    let mut logical_bytes = 0u64;
    for tensor in tensors {
        if tensor.name.is_empty() {
            return Err(ExecutionPlanError::EmptyTensorName {
                resource,
                tensor: tensor.name.clone(),
            });
        }
        if tensor.bytes == 0 {
            return Err(ExecutionPlanError::EmptyTensorPayload {
                resource,
                tensor: tensor.name.clone(),
            });
        }
        if tensor.shape.iter().any(|dimension| *dimension == 0) {
            return Err(ExecutionPlanError::EmptyTensorDimension {
                resource,
                tensor: tensor.name.clone(),
            });
        }
        let end = tensor.offset.checked_add(tensor.bytes).ok_or_else(|| {
            ExecutionPlanError::TensorExtentOverflow {
                resource,
                tensor: tensor.name.clone(),
            }
        })?;
        let source_file = bundle_source.source_file(&tensor.path).ok_or_else(|| {
            ExecutionPlanError::MissingTensorSourceSnapshot {
                resource,
                tensor: tensor.name.clone(),
            }
        })?;
        if end > source_file.length() {
            return Err(ExecutionPlanError::TensorRangeBeyondSource {
                resource,
                tensor: tensor.name.clone(),
                end,
                source_length: source_file.length(),
            });
        }

        if let Some(element_bytes) = tensor.dtype.element_size_bytes() {
            let elements = tensor.shape.iter().try_fold(1u64, |count, dimension| {
                let dimension = u64::try_from(*dimension).map_err(|_| {
                    ExecutionPlanError::TensorElementCountOverflow {
                        resource,
                        tensor: tensor.name.clone(),
                    }
                })?;
                count.checked_mul(dimension).ok_or_else(|| {
                    ExecutionPlanError::TensorElementCountOverflow {
                        resource,
                        tensor: tensor.name.clone(),
                    }
                })
            })?;
            let expected = elements.checked_mul(element_bytes as u64).ok_or_else(|| {
                ExecutionPlanError::TensorElementCountOverflow {
                    resource,
                    tensor: tensor.name.clone(),
                }
            })?;
            if expected != tensor.bytes {
                return Err(ExecutionPlanError::TensorByteSizeMismatch {
                    resource,
                    tensor: tensor.name.clone(),
                    expected,
                    actual: tensor.bytes,
                });
            }
        }

        if !names.insert(tensor.name.clone()) {
            return Err(ExecutionPlanError::DuplicateTensorName {
                resource,
                tensor: tensor.name.clone(),
            });
        }
        let range = (tensor.path.clone(), tensor.offset, tensor.bytes);
        if !ranges.insert(range) {
            return Err(ExecutionPlanError::DuplicateTensorRange {
                resource,
                tensor: tensor.name.clone(),
            });
        }
        logical_bytes = logical_bytes
            .checked_add(tensor.bytes)
            .ok_or(ExecutionPlanError::ResourceByteSizeOverflow { resource })?;
    }
    Ok(logical_bytes)
}

fn checkpoint_tensor_order_key(
    tensor: &CheckpointTensorSlice,
) -> (&crate::TensorRole, &str, &std::path::Path, u64, u64) {
    (
        &tensor.role,
        tensor.name.as_str(),
        tensor.path.as_path(),
        tensor.offset,
        tensor.bytes,
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ferrule_common::{
        ContentHash, MaterializedResourceKind, PayloadEncodingId, SourceGeneration,
        SourceIdentityHash,
    };

    use super::*;
    use crate::TensorRole;
    use crate::checkpoint::{CheckpointDType, CheckpointSourceFileIdentity};

    fn bundle_source(generation: u64) -> CheckpointBundleSource {
        CheckpointBundleSource::for_test(
            ResourceSource::new(
                SourceIdentityHash::new([1; 32]),
                ContentHash::new([2; 32]),
                PayloadEncodingId::new(3),
                SourceGeneration::new(generation),
            )
            .unwrap(),
            [CheckpointSourceFileIdentity::for_test(
                PathBuf::from("model.safetensors"),
                PathBuf::from("/test/model.safetensors"),
                u64::MAX,
            )],
        )
    }

    fn tensor(name: &str, role: TensorRole, offset: u64) -> CheckpointTensorSlice {
        CheckpointTensorSlice {
            name: name.into(),
            role,
            path: PathBuf::from("model.safetensors"),
            offset,
            bytes: 16,
            dtype: CheckpointDType::Bf16,
            shape: vec![2, 4],
        }
    }

    #[test]
    fn checkpoint_manifest_preserves_source_generation_and_tensor_order() {
        let resource = MaterializedResourceId::new(MaterializedResourceKind::Parameter, 4, 1);
        let manifest = ResourceManifest::checkpoint(
            resource,
            bundle_source(7),
            [
                tensor("q.weight", TensorRole::AttentionQuery, 32),
                tensor("k.weight", TensorRole::AttentionKey, 64),
            ],
            ResourceLayout::TensorBundle,
        )
        .unwrap();

        assert_eq!(manifest.resource(), resource);
        assert_eq!(manifest.logical_bytes(), 32);
        assert_eq!(manifest.source().unwrap().generation().get(), 7);
        assert_eq!(manifest.checkpoint_tensors()[0].name, "q.weight");
    }

    #[test]
    fn source_generation_participates_in_manifest_identity() {
        let resource = MaterializedResourceId::new(MaterializedResourceKind::Parameter, 0, 0);
        let first = ResourceManifest::checkpoint(
            resource,
            bundle_source(1),
            [tensor("weight", TensorRole::OutputHead, 0)],
            ResourceLayout::CheckpointEncoded,
        )
        .unwrap();
        let second = ResourceManifest::checkpoint(
            resource,
            bundle_source(2),
            [tensor("weight", TensorRole::OutputHead, 0)],
            ResourceLayout::CheckpointEncoded,
        )
        .unwrap();

        assert_ne!(first, second);
    }

    #[test]
    fn runtime_manifest_keeps_mutable_source_generation_out_of_prepared_state() {
        let resource = MaterializedResourceId::new(MaterializedResourceKind::OptimizerState, 2, 9);
        let manifest = ResourceManifest::runtime_owned(resource, 4096, 256).unwrap();

        assert_eq!(manifest.logical_bytes(), 4096);
        assert_eq!(manifest.source(), None);
        assert!(manifest.checkpoint_tensors().is_empty());
    }

    #[test]
    fn checkpoint_tensor_byte_extent_is_checked_without_saturation() {
        let resource = MaterializedResourceId::new(MaterializedResourceKind::Parameter, 0, 0);
        let mut invalid = tensor("weight", TensorRole::OutputHead, u64::MAX - 7);
        invalid.bytes = 16;

        assert_eq!(
            ResourceManifest::checkpoint(
                resource,
                bundle_source(1),
                [invalid],
                ResourceLayout::CheckpointEncoded,
            )
            .unwrap_err(),
            ExecutionPlanError::TensorExtentOverflow {
                resource,
                tensor: "weight".into(),
            }
        );
    }

    #[test]
    fn known_dtype_shape_must_match_checkpoint_byte_count() {
        let resource = MaterializedResourceId::new(MaterializedResourceKind::Parameter, 0, 0);
        let mut invalid = tensor("weight", TensorRole::OutputHead, 0);
        invalid.bytes = 15;

        assert_eq!(
            ResourceManifest::checkpoint(
                resource,
                bundle_source(1),
                [invalid],
                ResourceLayout::CheckpointEncoded,
            )
            .unwrap_err(),
            ExecutionPlanError::TensorByteSizeMismatch {
                resource,
                tensor: "weight".into(),
                expected: 16,
                actual: 15,
            }
        );
    }

    #[test]
    fn checkpoint_manifest_rejects_ranges_beyond_catalog_snapshot() {
        let resource = MaterializedResourceId::new(MaterializedResourceKind::Parameter, 0, 0);
        let source = CheckpointBundleSource::for_test(
            bundle_source(1).source(),
            [CheckpointSourceFileIdentity::for_test(
                PathBuf::from("model.safetensors"),
                PathBuf::from("/test/model.safetensors"),
                15,
            )],
        );

        assert_eq!(
            ResourceManifest::checkpoint(
                resource,
                source,
                [tensor("weight", TensorRole::OutputHead, 0)],
                ResourceLayout::CheckpointEncoded,
            )
            .unwrap_err(),
            ExecutionPlanError::TensorRangeBeyondSource {
                resource,
                tensor: "weight".into(),
                end: 16,
                source_length: 15,
            }
        );
    }

    #[test]
    fn runtime_alignment_is_validated() {
        let resource = MaterializedResourceId::new(MaterializedResourceKind::Gradient, 0, 0);
        assert_eq!(
            ResourceManifest::runtime_owned(resource, 1024, 48).unwrap_err(),
            ExecutionPlanError::InvalidResourceAlignment {
                resource,
                alignment: 48,
            }
        );
    }
}
