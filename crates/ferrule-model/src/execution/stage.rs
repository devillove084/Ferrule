//! Provider-neutral executable stages and exact resource access contracts.
//!
//! Stages describe semantic execution order and custody requirements. Backend
//! compilation maps each operation to launches independently; no CUDA kernel ID,
//! arena offset, stream, or destination generation is stored here.

use std::collections::BTreeSet;

use ferrule_common::{Error, MaterializedResourceId, MaterializedResourceKind};

use crate::materialization::{MaterializationPlacement, MaterializationRequest, ResourceSource};

use super::resource::{ExecutionPlanError, ResourceManifest};

/// Access performed by a stage on one materialized resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceAccess {
    Read,
    ReadWrite,
    Write,
}

impl ResourceAccess {
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

/// Minimum lifetime of the resource lease acquired for a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceRetention {
    /// Release after the backend completion fence for this stage.
    ThroughStage,
    /// Retain until the enclosing execution transaction commits or rolls back.
    ThroughTransaction,
    /// Retain across transactions until explicit owner retirement or eviction.
    Persistent,
}

/// Exact use of one resource by one executable stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StageResourceUse {
    resource: MaterializedResourceId,
    access: ResourceAccess,
    retention: ResourceRetention,
}

impl StageResourceUse {
    pub const fn new(
        resource: MaterializedResourceId,
        access: ResourceAccess,
        retention: ResourceRetention,
    ) -> Self {
        Self {
            resource,
            access,
            retention,
        }
    }

    pub const fn read(resource: MaterializedResourceId) -> Self {
        Self::new(
            resource,
            ResourceAccess::Read,
            ResourceRetention::ThroughStage,
        )
    }

    pub const fn resource(self) -> MaterializedResourceId {
        self.resource
    }

    pub const fn access(self) -> ResourceAccess {
        self.access
    }

    pub const fn retention(self) -> ResourceRetention {
        self.retention
    }
}

/// Temporary device/host storage required while a stage is executing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkspaceClaim {
    bytes: u64,
    alignment: u64,
}

impl WorkspaceClaim {
    pub const NONE: Self = Self {
        bytes: 0,
        alignment: 1,
    };

    pub fn new(bytes: u64, alignment: u64) -> std::result::Result<Self, ExecutionPlanError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(ExecutionPlanError::InvalidWorkspaceAlignment { alignment });
        }
        Ok(Self { bytes, alignment })
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    pub const fn alignment(self) -> u64 {
        self.alignment
    }
}

/// One ordered semantic operation and its complete materialized-resource contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutableStage<O> {
    operation: O,
    resources: Box<[StageResourceUse]>,
    workspace: WorkspaceClaim,
}

impl<O> ExecutableStage<O> {
    pub fn new(
        operation: O,
        resources: impl IntoIterator<Item = StageResourceUse>,
        workspace: WorkspaceClaim,
    ) -> Self {
        let mut resources = resources.into_iter().collect::<Vec<_>>();
        resources.sort_unstable_by_key(|resource| resource.resource());
        Self {
            operation,
            resources: resources.into_boxed_slice(),
            workspace,
        }
    }

    pub const fn operation(&self) -> &O {
        &self.operation
    }

    pub fn resources(&self) -> &[StageResourceUse] {
        &self.resources
    }

    pub const fn workspace(&self) -> WorkspaceClaim {
        self.workspace
    }
}

/// Immutable, validated resource and stage graph consumed by continuations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedExecutable<O> {
    resources: Box<[ResourceManifest]>,
    stages: Box<[ExecutableStage<O>]>,
}

/// One exact stage resource requirement after placement and mutable source custody
/// have been bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageMaterializationRequest {
    resource_use: StageResourceUse,
    request: MaterializationRequest,
}

impl StageMaterializationRequest {
    pub const fn resource_use(self) -> StageResourceUse {
        self.resource_use
    }

    pub const fn request(self) -> MaterializationRequest {
        self.request
    }
}

/// Borrowed operation plus exact resource requests ready for adapter resolution.
#[derive(Debug, PartialEq, Eq)]
pub struct MaterializedStage<'a, O> {
    operation: &'a O,
    resources: Box<[StageMaterializationRequest]>,
    workspace: WorkspaceClaim,
}

impl<'a, O> MaterializedStage<'a, O> {
    pub const fn operation(&self) -> &'a O {
        self.operation
    }

    pub fn resources(&self) -> &[StageMaterializationRequest] {
        &self.resources
    }

    pub const fn workspace(&self) -> WorkspaceClaim {
        self.workspace
    }
}

impl<O> PreparedExecutable<O> {
    pub fn new(
        resources: impl IntoIterator<Item = ResourceManifest>,
        stages: impl IntoIterator<Item = ExecutableStage<O>>,
    ) -> std::result::Result<Self, ExecutionPlanError> {
        let mut resources = resources.into_iter().collect::<Vec<_>>();
        resources.sort_unstable_by_key(ResourceManifest::resource);

        let mut declared = BTreeSet::new();
        for manifest in &resources {
            if !declared.insert(manifest.resource()) {
                return Err(ExecutionPlanError::DuplicateResourceManifest {
                    resource: manifest.resource(),
                });
            }
        }

        let stages = stages.into_iter().collect::<Vec<_>>();
        for (stage_index, stage) in stages.iter().enumerate() {
            let mut used = BTreeSet::new();
            for resource_use in stage.resources() {
                let resource = resource_use.resource();
                if !used.insert(resource) {
                    return Err(ExecutionPlanError::DuplicateStageResource {
                        stage: stage_index,
                        resource,
                    });
                }
                if !declared.contains(&resource) {
                    return Err(ExecutionPlanError::UnknownStageResource {
                        stage: stage_index,
                        resource,
                    });
                }
            }
        }

        Ok(Self {
            resources: resources.into_boxed_slice(),
            stages: stages.into_boxed_slice(),
        })
    }

    pub fn resources(&self) -> &[ResourceManifest] {
        &self.resources
    }

    pub fn stages(&self) -> &[ExecutableStage<O>] {
        &self.stages
    }

    pub fn resource(&self, resource: MaterializedResourceId) -> Option<&ResourceManifest> {
        self.resources
            .binary_search_by_key(&resource, ResourceManifest::resource)
            .ok()
            .and_then(|index| self.resources.get(index))
    }

    /// Bind one prepared stage to a runtime placement and exact current sources.
    ///
    /// Immutable checkpoint resources use their prepared source identity. Runtime-
    /// owned mutable resources (KV, activations, gradients, optimizer state) must be
    /// supplied by transaction custody so a spill/reload generation cannot be
    /// confused with model preparation or destination generations.
    pub fn materialize_stage<'a>(
        &'a self,
        stage_index: usize,
        placement: MaterializationPlacement,
        mut runtime_source: impl FnMut(MaterializedResourceId) -> ferrule_common::Result<ResourceSource>,
    ) -> ferrule_common::Result<MaterializedStage<'a, O>> {
        let stage = self.stages.get(stage_index).ok_or_else(|| {
            Error::Execution(format!(
                "prepared executable has no stage at index {stage_index}"
            ))
        })?;
        let resources = stage
            .resources()
            .iter()
            .map(|resource_use| {
                let resource = resource_use.resource();
                let manifest = self.resource(resource).ok_or_else(|| {
                    Error::Internal(format!(
                        "validated prepared stage {stage_index} lost resource manifest {resource:?}"
                    ))
                })?;
                let source = match manifest.source() {
                    Some(source) => source,
                    None => runtime_source(resource)?,
                };
                Ok(StageMaterializationRequest {
                    resource_use: *resource_use,
                    request: MaterializationRequest::for_placement(placement, source, resource)?,
                })
            })
            .collect::<ferrule_common::Result<Vec<_>>>()?
            .into_boxed_slice();
        Ok(MaterializedStage {
            operation: stage.operation(),
            resources,
            workspace: stage.workspace(),
        })
    }
}

/// Stable semantic coordinates shared by parameter, gradient, and optimizer state.
///
/// A training plan can derive three distinct resource IDs from the same slot by
/// changing only [`MaterializedResourceKind`]. Routed experts keep their dedicated
/// `(layer, expert)` coordinates because route selection instantiates them at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransformerResourceSlot {
    Embedding,
    Attention { layer: u32 },
    Router { layer: u32 },
    FeedForward { layer: u32 },
    Output,
    Attachment { index: u32 },
}

impl TransformerResourceSlot {
    pub const fn resource(self, kind: MaterializedResourceKind) -> MaterializedResourceId {
        let (group, item) = match self {
            Self::Embedding => (0, 0),
            Self::Attention { layer } => (layer, 1),
            Self::Router { layer } => (layer, 2),
            Self::FeedForward { layer } => (layer, 3),
            Self::Output => (0, 4),
            Self::Attachment { index } => (index, 5),
        };
        MaterializedResourceId::new(kind, group, item)
    }

    pub const fn parameter(self) -> MaterializedResourceId {
        self.resource(MaterializedResourceKind::Parameter)
    }
}

/// Model-neutral transformer stage identity.
///
/// Routed expert resources are selected after `Router` executes. The continuation
/// instantiates the exact routed stage from those selected resource IDs; the prepared
/// operation identity remains independent of a fixed expert set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransformerStage {
    Embed,
    Attention { layer: u32 },
    Router { layer: u32 },
    FeedForward { layer: u32 },
    Output,
    Attachment { index: u32 },
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ferrule_common::{
        ContentHash, MaterializedResourceKind, PayloadEncodingId, SourceGeneration,
        SourceIdentityHash,
    };

    use crate::TensorRole;
    use crate::checkpoint::{
        CheckpointBundleSource, CheckpointDType, CheckpointSourceFileIdentity,
        CheckpointTensorSlice,
    };
    use crate::materialization::ResourceSource;

    use super::super::resource::ResourceLayout;
    use super::*;

    fn resource(kind: MaterializedResourceKind, group: u32, item: u32) -> MaterializedResourceId {
        MaterializedResourceId::new(kind, group, item)
    }

    fn manifest(resource: MaterializedResourceId) -> ResourceManifest {
        ResourceManifest::checkpoint(
            resource,
            CheckpointBundleSource::for_test(
                ResourceSource::new(
                    SourceIdentityHash::new([1; 32]),
                    ContentHash::new([2; 32]),
                    PayloadEncodingId::new(3),
                    SourceGeneration::new(4),
                )
                .unwrap(),
                [CheckpointSourceFileIdentity::for_test(
                    PathBuf::from("model.safetensors"),
                    PathBuf::from("/test/model.safetensors"),
                    u64::MAX,
                )],
            ),
            [CheckpointTensorSlice {
                name: format!("{:?}.weight", resource.kind()),
                role: TensorRole::OutputHead,
                path: PathBuf::from("model.safetensors"),
                offset: (resource.kind() as u8 as u64) * 16,
                bytes: 16,
                dtype: CheckpointDType::Bf16,
                shape: vec![2, 4],
            }],
            ResourceLayout::CheckpointEncoded,
        )
        .unwrap()
    }

    #[test]
    fn transformer_slot_coordinates_are_shared_across_training_resource_kinds() {
        let slot = TransformerResourceSlot::FeedForward { layer: 7 };
        let parameter = slot.resource(MaterializedResourceKind::Parameter);
        let gradient = slot.resource(MaterializedResourceKind::Gradient);
        let optimizer = slot.resource(MaterializedResourceKind::OptimizerState);

        assert_eq!((parameter.group(), parameter.item()), (7, 3));
        assert_eq!((gradient.group(), gradient.item()), (7, 3));
        assert_eq!((optimizer.group(), optimizer.item()), (7, 3));
        assert_ne!(parameter, gradient);
        assert_ne!(gradient, optimizer);
    }

    #[test]
    fn resource_kind_is_part_of_manifest_identity() {
        let parameter = resource(MaterializedResourceKind::Parameter, 3, 5);
        let gradient = resource(MaterializedResourceKind::Gradient, 3, 5);
        let optimizer = resource(MaterializedResourceKind::OptimizerState, 3, 5);
        let executable = PreparedExecutable::<TransformerStage>::new(
            [manifest(parameter), manifest(gradient), manifest(optimizer)],
            [],
        )
        .unwrap();

        assert_eq!(executable.resources().len(), 3);
        assert!(executable.resource(parameter).is_some());
        assert!(executable.resource(gradient).is_some());
        assert!(executable.resource(optimizer).is_some());
    }

    #[test]
    fn duplicate_manifest_resource_is_rejected() {
        let resource = resource(MaterializedResourceKind::Parameter, 0, 0);
        assert_eq!(
            PreparedExecutable::<TransformerStage>::new(
                [manifest(resource), manifest(resource)],
                [],
            )
            .unwrap_err(),
            ExecutionPlanError::DuplicateResourceManifest { resource }
        );
    }

    #[test]
    fn stage_referencing_unknown_resource_is_rejected() {
        let declared = resource(MaterializedResourceKind::Parameter, 0, 0);
        let unknown = resource(MaterializedResourceKind::KvState, 0, 0);
        let stage = ExecutableStage::new(
            TransformerStage::Attention { layer: 0 },
            [StageResourceUse::read(unknown)],
            WorkspaceClaim::NONE,
        );

        assert_eq!(
            PreparedExecutable::new([manifest(declared)], [stage]).unwrap_err(),
            ExecutionPlanError::UnknownStageResource {
                stage: 0,
                resource: unknown,
            }
        );
    }

    #[test]
    fn duplicate_stage_access_is_rejected_even_when_modes_differ() {
        let resource = resource(MaterializedResourceKind::KvState, 0, 0);
        let manifest = ResourceManifest::runtime_owned(resource, 4096, 256).unwrap();
        let stage = ExecutableStage::new(
            TransformerStage::Attention { layer: 0 },
            [
                StageResourceUse::new(
                    resource,
                    ResourceAccess::Read,
                    ResourceRetention::ThroughStage,
                ),
                StageResourceUse::new(
                    resource,
                    ResourceAccess::Write,
                    ResourceRetention::ThroughTransaction,
                ),
            ],
            WorkspaceClaim::NONE,
        );

        assert_eq!(
            PreparedExecutable::new([manifest], [stage]).unwrap_err(),
            ExecutionPlanError::DuplicateStageResource { stage: 0, resource }
        );
    }

    #[test]
    fn write_access_and_transaction_retention_are_preserved() {
        let gradient = resource(MaterializedResourceKind::Gradient, 1, 2);
        let manifest = ResourceManifest::runtime_owned(gradient, 2048, 256).unwrap();
        let stage = ExecutableStage::new(
            TransformerStage::FeedForward { layer: 1 },
            [StageResourceUse::new(
                gradient,
                ResourceAccess::Write,
                ResourceRetention::ThroughTransaction,
            )],
            WorkspaceClaim::new(512, 128).unwrap(),
        );
        let executable = PreparedExecutable::new([manifest], [stage]).unwrap();
        let use_ = executable.stages()[0].resources()[0];

        assert!(use_.access().writes());
        assert!(!use_.access().reads());
        assert_eq!(use_.retention(), ResourceRetention::ThroughTransaction);
        assert_eq!(executable.stages()[0].workspace().bytes(), 512);
    }

    #[test]
    fn canonical_resource_and_stage_use_order_is_deterministic() {
        let first = resource(MaterializedResourceKind::Parameter, 1, 0);
        let second = resource(MaterializedResourceKind::Parameter, 0, 9);
        let stage = ExecutableStage::new(
            TransformerStage::Output,
            [
                StageResourceUse::read(first),
                StageResourceUse::read(second),
            ],
            WorkspaceClaim::NONE,
        );
        let executable =
            PreparedExecutable::new([manifest(first), manifest(second)], [stage]).unwrap();

        assert_eq!(executable.resources()[0].resource(), second);
        assert_eq!(executable.resources()[1].resource(), first);
        assert_eq!(executable.stages()[0].resources()[0].resource(), second);
        assert_eq!(executable.stages()[0].resources()[1].resource(), first);
    }

    #[test]
    fn stage_materialization_uses_prepared_and_transaction_sources_without_losing_access() {
        use ferrule_common::{BackendId, DeviceId, ModelInstanceId};

        let parameter = resource(MaterializedResourceKind::Parameter, 1, 1);
        let kv = resource(MaterializedResourceKind::KvState, 1, 0);
        let gradient = resource(MaterializedResourceKind::Gradient, 1, 1);
        let parameter_manifest = manifest(parameter);
        let prepared_parameter_source = parameter_manifest.source().unwrap();
        let executable = PreparedExecutable::new(
            [
                parameter_manifest,
                ResourceManifest::runtime_owned(kv, 4096, 256).unwrap(),
                ResourceManifest::runtime_owned(gradient, 1024, 256).unwrap(),
            ],
            [ExecutableStage::new(
                TransformerStage::Attention { layer: 1 },
                [
                    StageResourceUse::read(parameter),
                    StageResourceUse::new(
                        kv,
                        ResourceAccess::ReadWrite,
                        ResourceRetention::ThroughTransaction,
                    ),
                    StageResourceUse::new(
                        gradient,
                        ResourceAccess::Write,
                        ResourceRetention::ThroughTransaction,
                    ),
                ],
                WorkspaceClaim::NONE,
            )],
        )
        .unwrap();
        let placement = MaterializationPlacement::new(
            ModelInstanceId::new(9),
            BackendId::new(3),
            DeviceId::new(0),
        )
        .unwrap();
        let mut resolved = Vec::new();
        let stage = executable
            .materialize_stage(0, placement, |resource| {
                resolved.push(resource);
                ResourceSource::new(
                    SourceIdentityHash::new([resource.kind() as u8 + 4; 32]),
                    ContentHash::new([resource.kind() as u8 + 8; 32]),
                    PayloadEncodingId::new(7),
                    SourceGeneration::new(11),
                )
            })
            .unwrap();

        assert_eq!(resolved, vec![kv, gradient]);
        let parameter_request = stage
            .resources()
            .iter()
            .find(|resource| resource.resource_use().resource() == parameter)
            .unwrap();
        assert_eq!(
            parameter_request.request().source(),
            prepared_parameter_source
        );
        let kv_request = stage
            .resources()
            .iter()
            .find(|resource| resource.resource_use().resource() == kv)
            .unwrap();
        assert_eq!(
            kv_request.resource_use().access(),
            ResourceAccess::ReadWrite
        );
        assert_eq!(
            kv_request.resource_use().retention(),
            ResourceRetention::ThroughTransaction
        );
        let gradient_request = stage
            .resources()
            .iter()
            .find(|resource| resource.resource_use().resource() == gradient)
            .unwrap();
        assert_eq!(
            gradient_request.resource_use().access(),
            ResourceAccess::Write
        );
        assert_eq!(gradient_request.request().source().generation().get(), 11);
    }

    #[test]
    fn empty_executable_and_resource_free_stage_are_explicitly_valid() {
        let empty = PreparedExecutable::<TransformerStage>::new([], []).unwrap();
        assert!(empty.resources().is_empty());
        assert!(empty.stages().is_empty());

        let stage = ExecutableStage::new(
            TransformerStage::Attachment { index: 0 },
            [],
            WorkspaceClaim::NONE,
        );
        let executable = PreparedExecutable::new([], [stage]).unwrap();
        assert_eq!(executable.stages().len(), 1);
        assert!(executable.stages()[0].resources().is_empty());
    }
}
