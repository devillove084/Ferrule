//! Model-side resource dependency and materialization boundary.
//!
//! Continuations retain only canonical logical dependencies. A runner-level
//! resolver owns the bridge to the runtime-wide physical materialization registry;
//! exact resources are identified exclusively by [`MaterializationKey`]. Routed
//! experts, dense parameter bundles, activation checkpoints, gradients, and
//! optimizer shards use this same protocol.

mod source;

pub use source::{MaterializationSourceCatalog, MaterializationSourceEntry};

#[cfg(test)]
use ferrule_common::LogicalDependency;
use ferrule_common::{
    BackendId, CancellationReason, CompletionEvent, ContentHash, ContinuationId, DependencySet,
    DestinationGeneration, DeviceId, Error, ExpertId, FailureReason, LayerId, LoadStage,
    MaterializationKey, MaterializedResourceId, ModelInstanceId, OperationId, PayloadEncodingId,
    RegisteredPinnedAlignedSlabLeaseDescriptor, ResidencyBinding, ResidencyLeaseSet, Result,
    SourceGeneration, SourceIdentityHash, UploadFenceContract, ValidatedResidencyBinding,
};

/// Versioned protocol format for exact HF safetensors expert payloads.
pub const HF_SAFETENSORS_ROUTED_EXPERT_V1: PayloadEncodingId = PayloadEncodingId::new(1);
/// Versioned protocol format for ordered HF safetensors tensor bundles.
pub const HF_SAFETENSORS_TENSOR_BUNDLE_V1: PayloadEncodingId = PayloadEncodingId::new(2);

/// Immutable source identity for one materializable payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceSource {
    identity: SourceIdentityHash,
    content_hash: ContentHash,
    encoding: PayloadEncodingId,
    generation: SourceGeneration,
}

impl ResourceSource {
    pub fn new(
        identity: SourceIdentityHash,
        content_hash: ContentHash,
        encoding: PayloadEncodingId,
        generation: SourceGeneration,
    ) -> Result<Self> {
        if identity.is_zero() {
            return Err(Error::Model(
                "resource source identity hash must be non-zero".into(),
            ));
        }
        if content_hash.is_zero() {
            return Err(Error::Model(
                "resource source content hash must be non-zero".into(),
            ));
        }
        if encoding.get() == 0 {
            return Err(Error::Model(
                "resource payload encoding must be non-zero".into(),
            ));
        }
        if generation.is_zero() {
            return Err(Error::Model(
                "resource source generation must be non-zero".into(),
            ));
        }
        Ok(Self {
            identity,
            content_hash,
            encoding,
            generation,
        })
    }

    pub const fn identity(self) -> SourceIdentityHash {
        self.identity
    }

    pub const fn content_hash(self) -> ContentHash {
        self.content_hash
    }

    pub const fn encoding(self) -> PayloadEncodingId {
        self.encoding
    }

    pub const fn generation(self) -> SourceGeneration {
        self.generation
    }
}

/// Runtime-assigned model and placement namespace for exact load identity.
///
/// These values come from the installed driver/registry, never from a model-local
/// counter, object address, or transient operation ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterializationPlacement {
    model: ModelInstanceId,
    backend: BackendId,
    device: DeviceId,
}

impl MaterializationPlacement {
    pub fn new(model: ModelInstanceId, backend: BackendId, device: DeviceId) -> Result<Self> {
        if model.is_zero() {
            return Err(Error::Model(
                "materialization model instance must be non-zero".into(),
            ));
        }
        Ok(Self {
            model,
            backend,
            device,
        })
    }

    pub const fn model(self) -> ModelInstanceId {
        self.model
    }

    pub const fn backend(self) -> BackendId {
        self.backend
    }

    pub const fn device(self) -> DeviceId {
        self.device
    }
}

/// Exact model/source/resource/backend coordinates resolved before a destination
/// reservation is joined or created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterializationRequest {
    model: ModelInstanceId,
    source: ResourceSource,
    resource: MaterializedResourceId,
    backend: BackendId,
    device: DeviceId,
}

impl MaterializationRequest {
    pub fn new(
        model: ModelInstanceId,
        source: ResourceSource,
        resource: MaterializedResourceId,
        backend: BackendId,
        device: DeviceId,
    ) -> Result<Self> {
        Self::for_placement(
            MaterializationPlacement::new(model, backend, device)?,
            source,
            resource,
        )
    }

    pub fn routed_expert(
        model: ModelInstanceId,
        source: ResourceSource,
        layer: LayerId,
        expert: ExpertId,
        backend: BackendId,
        device: DeviceId,
    ) -> Result<Self> {
        Self::new(
            model,
            source,
            MaterializedResourceId::routed_expert(layer, expert),
            backend,
            device,
        )
    }

    pub fn for_placement(
        placement: MaterializationPlacement,
        source: ResourceSource,
        resource: MaterializedResourceId,
    ) -> Result<Self> {
        Ok(Self {
            model: placement.model,
            source,
            resource,
            backend: placement.backend,
            device: placement.device,
        })
    }

    /// Complete a load identity with the generation returned by the exact slot
    /// reservation. Callers must not synthesize this generation.
    pub fn materialization_key(
        self,
        destination_generation: DestinationGeneration,
    ) -> Result<MaterializationKey> {
        Ok(MaterializationKey::new(
            self.model,
            self.source.identity,
            self.source.content_hash,
            self.resource,
            self.source.encoding,
            self.backend,
            self.device,
            self.source.generation,
            destination_generation,
        )?)
    }

    pub fn validate_key(self, key: MaterializationKey) -> Result<()> {
        key.validate()?;
        let fields = [
            ("model instance", key.model() == self.model),
            ("source identity", key.source() == self.source.identity),
            (
                "source content hash",
                key.source_hash() == self.source.content_hash,
            ),
            ("resource", key.resource() == self.resource),
            (
                "payload encoding",
                key.payload_encoding() == self.source.encoding,
            ),
            ("backend", key.backend() == self.backend),
            ("device", key.device() == self.device),
            (
                "source generation",
                key.source_generation() == self.source.generation,
            ),
        ];
        if let Some((field, _)) = fields.into_iter().find(|(_, matches)| !matches) {
            return Err(Error::Execution(format!(
                "materialization load key {field} does not match the exact post-router request"
            )));
        }
        Ok(())
    }

    pub const fn model(self) -> ModelInstanceId {
        self.model
    }

    pub const fn source(self) -> ResourceSource {
        self.source
    }

    pub const fn resource(self) -> MaterializedResourceId {
        self.resource
    }

    pub const fn routed_expert_coordinates(self) -> Option<(LayerId, ExpertId)> {
        self.resource.routed_expert_coordinates()
    }

    pub const fn backend(self) -> BackendId {
        self.backend
    }

    pub const fn device(self) -> DeviceId {
        self.device
    }
}

/// Prepared physical transfer returned by a materialization provider.
///
/// The destination generation is fixed by the exact physical reservation; callers
/// must use `key` verbatim. `evicted` identifies an older physical publication
/// invalidated by this preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializationTransfer {
    key: MaterializationKey,
    binding: ResidencyBinding,
    evicted: Option<MaterializationKey>,
}

impl MaterializationTransfer {
    pub fn new(
        key: MaterializationKey,
        binding: ResidencyBinding,
        evicted: Option<MaterializationKey>,
    ) -> Result<Self> {
        ValidatedResidencyBinding::new(key, binding)?;
        if let Some(evicted) = evicted {
            evicted.validate()?;
            if evicted == key {
                return Err(Error::Execution(
                    "physical materialization reservation cannot evict its own key".into(),
                ));
            }
        }
        Ok(Self {
            key,
            binding,
            evicted,
        })
    }

    pub const fn key(self) -> MaterializationKey {
        self.key
    }

    pub const fn binding(self) -> ResidencyBinding {
        self.binding
    }

    pub const fn evicted(self) -> Option<MaterializationKey> {
        self.evicted
    }
}

/// Provider-owned resident metadata prepared for registry adoption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializationResident {
    key: MaterializationKey,
    binding: ResidencyBinding,
}

impl MaterializationResident {
    pub fn new(key: MaterializationKey, binding: ResidencyBinding) -> Result<Self> {
        ValidatedResidencyBinding::new(key, binding)?;
        Ok(Self { key, binding })
    }

    pub const fn key(self) -> MaterializationKey {
        self.key
    }

    pub const fn binding(self) -> ResidencyBinding {
        self.binding
    }
}

/// Provider-owned physical state prepared for one exact request.
///
/// This is not logical residency publication. The runtime registry must adopt a
/// resident preparation or drive a transfer before exposing a lease to execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializationPreparation {
    Resident(MaterializationResident),
    Transfer(MaterializationTransfer),
}

impl MaterializationPreparation {
    pub const fn key(self) -> MaterializationKey {
        match self {
            Self::Resident(resident) => resident.key(),
            Self::Transfer(transfer) => transfer.key(),
        }
    }

    pub const fn binding(self) -> ResidencyBinding {
        match self {
            Self::Resident(resident) => resident.binding(),
            Self::Transfer(transfer) => transfer.binding(),
        }
    }
}

/// Registered pinned storage and upload-fence contract for one registry operation.
///
/// Descriptors are immutable address metadata. The physical backend retains the
/// actual registered allocations and provider tickets until cancellation or the
/// matching read/upload completion has drained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalMaterializationOperationReservation {
    key: MaterializationKey,
    binding: ResidencyBinding,
    slabs: Box<[RegisteredPinnedAlignedSlabLeaseDescriptor]>,
    upload_fence: UploadFenceContract,
}

impl PhysicalMaterializationOperationReservation {
    pub fn new(
        key: MaterializationKey,
        binding: ResidencyBinding,
        slabs: impl Into<Box<[RegisteredPinnedAlignedSlabLeaseDescriptor]>>,
        upload_fence: UploadFenceContract,
    ) -> Result<Self> {
        ValidatedResidencyBinding::new(key, binding)?;
        let slabs = slabs.into();
        if slabs.is_empty() {
            return Err(Error::Execution(
                "physical materialization operation requires at least one registered pinned slab"
                    .into(),
            ));
        }
        for slab in &slabs {
            if slab.operation() != upload_fence.operation
                || slab.source_generation() != key.source_generation()
                || slab.destination_generation() != key.destination_generation()
            {
                return Err(Error::Execution(
                    "physical materialization slab identity does not match its operation or key"
                        .into(),
                ));
            }
        }
        if upload_fence.destination_generation != key.destination_generation()
            || upload_fence.fence.is_zero()
        {
            return Err(Error::Execution(
                "physical materialization upload fence does not match its key".into(),
            ));
        }
        Ok(Self {
            key,
            binding,
            slabs,
            upload_fence,
        })
    }

    pub const fn key(&self) -> MaterializationKey {
        self.key
    }

    pub const fn binding(&self) -> ResidencyBinding {
        self.binding
    }

    pub fn slabs(&self) -> &[RegisteredPinnedAlignedSlabLeaseDescriptor] {
        &self.slabs
    }

    pub const fn upload_fence(&self) -> UploadFenceContract {
        self.upload_fence
    }
}

/// Exact physical capacities exposed by a materialization backend.
///
/// Stage limits bound materialization transitions. `resident_capacity_bytes`
/// covers published residency and replacement shadow capacity. Residency lease
/// slots are reported per continuation so the runtime can scale them by admitted
/// concurrency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalMaterializationTopology {
    stage_limits: ferrule_common::materialization_io::MaterializationResourceLimits,
    resident_capacity_bytes: u64,
    residency_lease_slots_per_continuation: u64,
}

impl PhysicalMaterializationTopology {
    pub fn new(
        stage_limits: ferrule_common::materialization_io::MaterializationResourceLimits,
        resident_capacity_bytes: u64,
        residency_lease_slots_per_continuation: u64,
    ) -> Result<Self> {
        let stage_limits = stage_limits.validate()?;
        if !stage_limits.capacity.is_empty()
            && (resident_capacity_bytes == 0 || residency_lease_slots_per_continuation == 0)
        {
            return Err(Error::Execution(
                "physical materialization topology requires non-zero resident and residency-lease capacity"
                    .into(),
            ));
        }
        Ok(Self {
            stage_limits,
            resident_capacity_bytes,
            residency_lease_slots_per_continuation,
        })
    }

    pub const fn stage_limits(
        self,
    ) -> ferrule_common::materialization_io::MaterializationResourceLimits {
        self.stage_limits
    }

    pub const fn resident_capacity_bytes(self) -> u64 {
        self.resident_capacity_bytes
    }

    pub const fn residency_lease_slots_per_continuation(self) -> u64 {
        self.residency_lease_slots_per_continuation
    }
}

/// Model/backend-owned physical materialization provider.
///
/// The trait is object-safe and depends only on model/common protocol types. A
/// runtime wrapper owns the box after `MultiSessionRunner` transfers it once.
/// Residency publication remains model authority; runtime code owns waiters,
/// credits, operation IDs, completion validation, and retirement.
pub trait MaterializationProvider: std::fmt::Debug + Send {
    fn placement(&self) -> MaterializationPlacement;

    fn resource_topology(&self) -> Result<PhysicalMaterializationTopology>;

    /// Prepare one exact source request, calling the real residency authority
    /// before constructing a destination generation and key.
    fn prepare(
        &mut self,
        request: MaterializationRequest,
    ) -> std::result::Result<MaterializationPreparation, FailureReason>;

    /// Re-read provider-owned physical state for a key previously returned by
    /// `prepare`. No runtime cache may synthesize this result.
    fn prepared(
        &self,
        key: MaterializationKey,
    ) -> std::result::Result<MaterializationPreparation, FailureReason>;

    fn materialization_plan(
        &self,
        key: MaterializationKey,
    ) -> std::result::Result<
        ferrule_common::materialization_io::MaterializationResourcePlan,
        FailureReason,
    >;

    /// Discard a provider preparation that was never claimed by the registry.
    /// Implementations must cancel an unbound transfer reservation or release an
    /// acquired resident execution lease. Claimed operations use `cancel` instead.
    fn discard_preparation(
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<(), FailureReason>;

    /// Release the execution residency lease retained for aggregate registry demand.
    /// A release racing a pending install must be remembered so later publication
    /// cannot leave an ownerless lease.
    fn release_execution_lease(
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<(), FailureReason>;

    fn reserve(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        plan: ferrule_common::materialization_io::MaterializationResourcePlan,
    ) -> std::result::Result<PhysicalMaterializationOperationReservation, FailureReason>;

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: ferrule_common::materialization_io::MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason>;

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: ferrule_common::materialization_io::MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason>;

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: ferrule_common::materialization_io::MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason>;

    fn cancel(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> std::result::Result<(), FailureReason>;

    fn next_completion(&mut self) -> Option<CompletionEvent>;
}

/// Model-facing resolver for exact physical resource identities.
///
/// `resolve` prepares the provider-owned physical resolution and returns its exact
/// generation-qualified key. The runtime registry is the only owner of logical
/// demand, publication, residency accounting, cancellation, and eviction.
pub trait MaterializationResolver: Send {
    /// Stable runtime-assigned namespace used for every request resolved by this
    /// resolver instance.
    fn placement(&self) -> MaterializationPlacement;

    fn resolve(&mut self, request: MaterializationRequest) -> Result<MaterializationKey>;
}

/// One unresolved resource-dependency description retained by a model continuation.
///
/// This object owns no logical attachment, provider lease, operation, or cancellation
/// authority. The runtime registry owns those lifetimes; model cancellation only
/// discards the description that would otherwise be consumed by resume validation.
#[derive(Debug)]
pub struct ContinuationDependencyState {
    continuation: ContinuationId,
    unresolved: Option<DependencySet>,
}

impl ContinuationDependencyState {
    pub fn new(continuation: ContinuationId, unresolved: DependencySet) -> Result<Self> {
        validate_continuation_id(continuation)?;
        unresolved.validate()?;
        Ok(Self {
            continuation,
            unresolved: Some(unresolved),
        })
    }

    pub const fn continuation(&self) -> ContinuationId {
        self.continuation
    }

    pub fn unresolved(&self) -> Option<&DependencySet> {
        self.unresolved.as_ref()
    }

    pub fn has_materialization_dependencies(&self) -> bool {
        self.unresolved.as_ref().is_some_and(|dependencies| {
            dependencies
                .iter()
                .any(|dependency| dependency.materialization_key().is_some())
        })
    }

    pub fn discard_unresolved(&mut self) {
        self.unresolved = None;
    }

    pub fn replace_unresolved(&mut self, unresolved: DependencySet) -> Result<()> {
        if self.unresolved.is_some() {
            return Err(Error::Execution(format!(
                "continuation {} still owns an unresolved dependency set",
                self.continuation.get()
            )));
        }
        unresolved.validate()?;
        self.unresolved = Some(unresolved);
        Ok(())
    }

    /// Validate and consume exactly one resume edge. The lease set must contain
    /// every and only residency dependency in the suspended set. Runtime
    /// retains and later releases the attachment for the consumed edge.
    pub fn validate_resume(
        &mut self,
        continuation: ContinuationId,
        leases: &ResidencyLeaseSet,
    ) -> Result<()> {
        if continuation != self.continuation {
            return Err(Error::Execution(format!(
                "continuation identity mismatch: expected {}, got {}",
                self.continuation.get(),
                continuation.get()
            )));
        }
        let unresolved = self.unresolved.as_ref().ok_or_else(|| {
            Error::Execution(format!(
                "continuation {} has no unresolved dependency set; refusing resume replay",
                self.continuation.get()
            ))
        })?;
        let required = unresolved
            .iter()
            .filter_map(|dependency| dependency.materialization_key())
            .collect::<Vec<_>>();
        if leases.len() != required.len()
            || required
                .iter()
                .any(|key| leases.binding_for(*key).is_none())
        {
            return Err(Error::Execution(format!(
                "continuation {} lease set does not exactly satisfy its unresolved resource dependencies",
                self.continuation.get()
            )));
        }
        self.unresolved = None;
        Ok(())
    }
}

pub(crate) fn validate_continuation_id(continuation: ContinuationId) -> Result<()> {
    if continuation.is_zero() {
        Err(Error::Execution(
            "model continuation ID must be non-zero".into(),
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn resource_dependency_set(
    keys: impl IntoIterator<Item = MaterializationKey>,
) -> Result<DependencySet> {
    let dependencies = keys
        .into_iter()
        .map(LogicalDependency::resource_resident)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(DependencySet::new(dependencies)?)
}

/// Resolve arbitrary exact resource requests through the global resolver boundary.
///
/// This is also the post-router path for a dynamically selected routed-expert set.
/// Every resolved key is validated against the request that produced it before the
/// canonical stage custody contract is constructed.
pub fn resolve_stage_resources(
    resources: impl IntoIterator<Item = crate::execution::StageMaterializationRequest>,
    workspace: crate::execution::WorkspaceClaim,
    resolver: &mut dyn MaterializationResolver,
) -> Result<crate::execution::ResolvedStage> {
    let placement = resolver.placement();
    let mut resolved = Vec::new();
    for resource in resources {
        let request = resource.request();
        if request.model() != placement.model()
            || request.backend() != placement.backend()
            || request.device() != placement.device()
        {
            return Err(Error::Execution(
                "materialized stage request does not match resolver placement".into(),
            ));
        }
        let key = resolver.resolve(request)?;
        resolved.push(crate::execution::ResolvedStageResource::new(resource, key)?);
    }
    crate::execution::ResolvedStage::new(resolved, workspace)
}

/// Resolve every exact resource request for one prepared stage without losing its
/// access, retention, or workspace contract.
pub fn resolve_stage<O>(
    stage: &crate::execution::MaterializedStage<'_, O>,
    resolver: &mut dyn MaterializationResolver,
) -> Result<crate::execution::ResolvedStage> {
    resolve_stage_resources(
        stage.resources().iter().copied(),
        stage.workspace(),
        resolver,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use ferrule_common::{
        DestinationSlotId, DispatchFenceContract, FenceId, MappingEpoch, MaterializedResourceKind,
        OperationId, ResidencyBinding,
    };

    use super::*;

    fn bytes(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn source(identity: u8, content: u8, encoding: u32, generation: u64) -> ResourceSource {
        ResourceSource::new(
            SourceIdentityHash::new(bytes(identity)),
            ContentHash::new(bytes(content)),
            PayloadEncodingId::new(encoding),
            SourceGeneration::new(generation),
        )
        .unwrap()
    }

    fn resource(kind: MaterializedResourceKind, group: u32, item: u32) -> MaterializedResourceId {
        MaterializedResourceId::new(kind, group, item)
    }

    fn request() -> MaterializationRequest {
        MaterializationRequest::new(
            ModelInstanceId::new(11),
            source(12, 13, 14, 15),
            resource(MaterializedResourceKind::RoutedExpert, 16, 17),
            BackendId::new(18),
            DeviceId::new(19),
        )
        .unwrap()
    }

    fn key() -> MaterializationKey {
        request()
            .materialization_key(DestinationGeneration::new(20))
            .unwrap()
    }

    fn key_for_expert(expert: u32) -> MaterializationKey {
        MaterializationRequest::new(
            request().model(),
            request().source(),
            resource(MaterializedResourceKind::RoutedExpert, 16, expert),
            request().backend(),
            request().device(),
        )
        .unwrap()
        .materialization_key(DestinationGeneration::new(20))
        .unwrap()
    }

    fn binding(key: MaterializationKey, slot: u32) -> ValidatedResidencyBinding {
        ValidatedResidencyBinding::new(
            key,
            ResidencyBinding::new(
                key.model(),
                key.resource(),
                key.backend(),
                key.device(),
                DestinationSlotId::new(slot),
                key.destination_generation(),
            ),
        )
        .unwrap()
    }

    fn leases(keys: &[MaterializationKey]) -> ResidencyLeaseSet {
        let placement = keys.first().copied().unwrap_or_else(key);
        ResidencyLeaseSet::new(
            keys.iter().copied(),
            keys.iter()
                .copied()
                .enumerate()
                .map(|(index, key)| binding(key, index as u32)),
            MappingEpoch::new(1),
            DispatchFenceContract::new(
                OperationId::new(91),
                FenceId::new(92),
                placement.backend(),
                placement.device(),
            ),
        )
        .unwrap()
    }

    #[derive(Default)]
    struct MockResolver {
        reservations: BTreeMap<MaterializationRequest, MaterializationKey>,
    }

    impl MaterializationResolver for MockResolver {
        fn placement(&self) -> MaterializationPlacement {
            MaterializationPlacement::new(
                request().model(),
                request().backend(),
                request().device(),
            )
            .unwrap()
        }

        fn resolve(&mut self, request: MaterializationRequest) -> Result<MaterializationKey> {
            let key = if let Some(key) = self.reservations.get(&request) {
                *key
            } else {
                let key = request.materialization_key(DestinationGeneration::new(20))?;
                self.reservations.insert(request, key);
                key
            };
            Ok(key)
        }
    }

    #[test]
    fn one_stage_resolves_parameter_kv_and_gradient_through_one_resolver() {
        use std::path::PathBuf;

        use crate::TensorRole;
        use crate::checkpoint::{
            CheckpointBundleSource, CheckpointDType, CheckpointSourceFileIdentity,
            CheckpointTensorSlice,
        };
        use crate::execution::{
            ExecutableStage, PreparedExecutable, ResourceAccess, ResourceLayout, ResourceManifest,
            ResourceRetention, StageResourceUse, TransformerStage, WorkspaceClaim,
        };

        let parameter = resource(MaterializedResourceKind::Parameter, 3, 1);
        let kv = resource(MaterializedResourceKind::KvState, 3, 0);
        let gradient = resource(MaterializedResourceKind::Gradient, 3, 1);
        let parameter_source = source(31, 32, 2, 33);
        let parameter_manifest = ResourceManifest::checkpoint(
            parameter,
            CheckpointBundleSource::for_test(
                parameter_source,
                [CheckpointSourceFileIdentity::for_test(
                    PathBuf::from("model.safetensors"),
                    PathBuf::from("/test/model.safetensors"),
                    16,
                )],
            ),
            [CheckpointTensorSlice {
                name: "q.weight".into(),
                role: TensorRole::AttentionQuery,
                path: PathBuf::from("model.safetensors"),
                offset: 0,
                bytes: 16,
                dtype: CheckpointDType::Bf16,
                shape: vec![2, 4],
            }],
            ResourceLayout::TensorBundle,
        )
        .unwrap();
        let executable = PreparedExecutable::new(
            [
                parameter_manifest,
                ResourceManifest::runtime_owned(kv, 4096, 256).unwrap(),
                ResourceManifest::runtime_owned(gradient, 16, 16).unwrap(),
            ],
            [ExecutableStage::new(
                TransformerStage::Attention { layer: 3 },
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
        let mut resolver = MockResolver::default();
        let placement = resolver.placement();
        let stage = executable
            .materialize_stage(0, placement, |resource| {
                Ok(match resource.kind() {
                    MaterializedResourceKind::KvState => source(41, 42, 8, 43),
                    MaterializedResourceKind::Gradient => source(51, 52, 9, 53),
                    kind => panic!("unexpected runtime-owned resource kind {kind:?}"),
                })
            })
            .unwrap();

        let resolved = resolve_stage(&stage, &mut resolver).unwrap();
        let resources = resolved
            .resources()
            .iter()
            .map(|resource| resource.key().resource())
            .collect::<Vec<_>>();
        assert_eq!(resources, vec![parameter, kv, gradient]);
        assert_eq!(resolved.workspace(), WorkspaceClaim::NONE);
        assert_eq!(resolved.dependencies().unwrap().len(), 3);
        assert_eq!(resolved.resources()[0].access(), ResourceAccess::Read);
        assert_eq!(
            resolved.resources()[0].retention(),
            ResourceRetention::ThroughStage
        );
        assert_eq!(resolved.resources()[1].access(), ResourceAccess::ReadWrite);
        assert_eq!(
            resolved.resources()[1].retention(),
            ResourceRetention::ThroughTransaction
        );
        assert_eq!(resolved.resources()[2].access(), ResourceAccess::Write);
        assert_eq!(
            resolved.resources()[2].retention(),
            ResourceRetention::ThroughTransaction
        );
        assert_eq!(resolver.reservations.len(), 3);
        assert_eq!(
            resolver
                .reservations
                .keys()
                .find(|request| request.resource() == parameter)
                .unwrap()
                .source(),
            parameter_source
        );
    }

    #[test]
    fn resolved_stage_rejects_a_key_from_a_different_request() {
        struct WrongKeyResolver;

        impl MaterializationResolver for WrongKeyResolver {
            fn placement(&self) -> MaterializationPlacement {
                MaterializationPlacement::new(
                    request().model(),
                    request().backend(),
                    request().device(),
                )
                .unwrap()
            }

            fn resolve(&mut self, request: MaterializationRequest) -> Result<MaterializationKey> {
                MaterializationRequest::for_placement(
                    self.placement(),
                    source(91, 92, 93, 94),
                    request.resource(),
                )?
                .materialization_key(DestinationGeneration::new(20))
            }
        }

        let request = request();
        let resource_use = crate::execution::StageResourceUse::read(request.resource());
        let resource =
            crate::execution::StageMaterializationRequest::new(resource_use, request).unwrap();
        let error = resolve_stage_resources(
            [resource],
            crate::execution::WorkspaceClaim::NONE,
            &mut WrongKeyResolver,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("source identity"));
    }

    #[test]
    fn resource_free_resolved_stage_has_no_wait_dependencies() {
        let resolved = resolve_stage_resources(
            [],
            crate::execution::WorkspaceClaim::new(256, 64).unwrap(),
            &mut MockResolver::default(),
        )
        .unwrap();
        assert!(resolved.dependencies().is_none());
        assert!(resolved.resources().is_empty());
        assert_eq!(resolved.workspace().bytes(), 256);
    }

    #[test]
    fn materialization_key_identity_includes_every_source_resource_placement_and_generation_dimension()
     {
        let base = key();
        let base_request = request();
        let variants = [
            MaterializationRequest::new(
                ModelInstanceId::new(21),
                base_request.source(),
                base_request.resource(),
                base_request.backend(),
                base_request.device(),
            )
            .unwrap()
            .materialization_key(base.destination_generation())
            .unwrap(),
            MaterializationRequest::new(
                base_request.model(),
                source(22, 13, 14, 15),
                base_request.resource(),
                base_request.backend(),
                base_request.device(),
            )
            .unwrap()
            .materialization_key(base.destination_generation())
            .unwrap(),
            MaterializationRequest::new(
                base_request.model(),
                source(12, 23, 14, 15),
                base_request.resource(),
                base_request.backend(),
                base_request.device(),
            )
            .unwrap()
            .materialization_key(base.destination_generation())
            .unwrap(),
            MaterializationRequest::new(
                base_request.model(),
                base_request.source(),
                resource(MaterializedResourceKind::Parameter, 16, 17),
                base_request.backend(),
                base_request.device(),
            )
            .unwrap()
            .materialization_key(base.destination_generation())
            .unwrap(),
            MaterializationRequest::new(
                base_request.model(),
                base_request.source(),
                resource(MaterializedResourceKind::RoutedExpert, 24, 17),
                base_request.backend(),
                base_request.device(),
            )
            .unwrap()
            .materialization_key(base.destination_generation())
            .unwrap(),
            key_for_expert(25),
            MaterializationRequest::new(
                base_request.model(),
                source(12, 13, 26, 15),
                base_request.resource(),
                base_request.backend(),
                base_request.device(),
            )
            .unwrap()
            .materialization_key(base.destination_generation())
            .unwrap(),
            MaterializationRequest::new(
                base_request.model(),
                base_request.source(),
                base_request.resource(),
                BackendId::new(27),
                base_request.device(),
            )
            .unwrap()
            .materialization_key(base.destination_generation())
            .unwrap(),
            MaterializationRequest::new(
                base_request.model(),
                base_request.source(),
                base_request.resource(),
                base_request.backend(),
                DeviceId::new(28),
            )
            .unwrap()
            .materialization_key(base.destination_generation())
            .unwrap(),
            MaterializationRequest::new(
                base_request.model(),
                source(12, 13, 14, 29),
                base_request.resource(),
                base_request.backend(),
                base_request.device(),
            )
            .unwrap()
            .materialization_key(base.destination_generation())
            .unwrap(),
            base_request
                .materialization_key(DestinationGeneration::new(30))
                .unwrap(),
        ];
        assert!(variants.into_iter().all(|variant| variant != base));
        assert_eq!(variants.into_iter().collect::<BTreeSet<_>>().len(), 11);
    }

    #[test]
    fn preparation_variants_preserve_exact_key_and_validate_binding() {
        let key = key();
        let raw = binding(key, 3).binding();
        let resident = MaterializationResident::new(key, raw).unwrap();
        let transfer = MaterializationTransfer::new(key, raw, None).unwrap();
        assert_eq!(MaterializationPreparation::Resident(resident).key(), key);
        assert_eq!(MaterializationPreparation::Transfer(transfer).key(), key);

        let wrong = ResidencyBinding::new(
            key.model(),
            key.resource(),
            key.backend(),
            key.device(),
            DestinationSlotId::new(3),
            DestinationGeneration::new(key.destination_generation().get() + 1),
        );
        assert!(MaterializationResident::new(key, wrong).is_err());
        assert!(MaterializationTransfer::new(key, wrong, None).is_err());
    }

    #[test]
    fn resume_requires_complete_exact_lease_set_and_rejects_wrong_identity_dimensions() {
        let first = key_for_expert(1);
        let second = key_for_expert(2);
        let dependencies = resource_dependency_set([first, second]).unwrap();
        let continuation = ContinuationId::new(41);

        let mut incomplete =
            ContinuationDependencyState::new(continuation, dependencies.clone()).unwrap();
        assert!(
            incomplete
                .validate_resume(continuation, &leases(&[first]))
                .is_err()
        );
        assert!(incomplete.unresolved().is_some());

        for wrong in [
            MaterializationRequest::new(
                first.model(),
                source(42, 13, 14, 15),
                first.resource(),
                first.backend(),
                first.device(),
            )
            .unwrap()
            .materialization_key(first.destination_generation())
            .unwrap(),
            MaterializationRequest::new(
                first.model(),
                request().source(),
                first.resource(),
                first.backend(),
                first.device(),
            )
            .unwrap()
            .materialization_key(DestinationGeneration::new(43))
            .unwrap(),
            MaterializationRequest::new(
                first.model(),
                request().source(),
                first.resource(),
                BackendId::new(44),
                first.device(),
            )
            .unwrap()
            .materialization_key(first.destination_generation())
            .unwrap(),
        ] {
            let mut state = ContinuationDependencyState::new(
                continuation,
                resource_dependency_set([first]).unwrap(),
            )
            .unwrap();
            assert!(
                state
                    .validate_resume(continuation, &leases(&[wrong]))
                    .is_err()
            );
            assert!(state.unresolved().is_some());
        }

        let mut complete = ContinuationDependencyState::new(continuation, dependencies).unwrap();
        complete
            .validate_resume(continuation, &leases(&[second, first]))
            .unwrap();
        assert!(complete.unresolved().is_none());
        assert!(!complete.has_materialization_dependencies());
    }

    #[test]
    fn model_cancellation_discards_only_the_unresolved_description() {
        let continuation = ContinuationId::new(51);
        let mut state = ContinuationDependencyState::new(
            continuation,
            resource_dependency_set([key()]).unwrap(),
        )
        .unwrap();
        assert!(state.has_materialization_dependencies());
        state.discard_unresolved();
        assert!(state.unresolved().is_none());
        assert!(!state.has_materialization_dependencies());
        assert!(
            state
                .validate_resume(continuation, &leases(&[key()]))
                .is_err()
        );
    }

    #[test]
    fn replace_unresolved_moves_to_next_edge_without_cross_edge_state() {
        let first = key_for_expert(1);
        let second = key_for_expert(2);
        let continuation = ContinuationId::new(64);
        let mut state = ContinuationDependencyState::new(
            continuation,
            resource_dependency_set([first]).unwrap(),
        )
        .unwrap();
        state
            .validate_resume(continuation, &leases(&[first]))
            .unwrap();
        state
            .replace_unresolved(resource_dependency_set([second]).unwrap())
            .unwrap();
        assert_eq!(state.unresolved().unwrap().len(), 1);
        state
            .validate_resume(continuation, &leases(&[second]))
            .unwrap();
        assert!(state.unresolved().is_none());
    }
}
