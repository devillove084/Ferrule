//! Model-side expert dependency and materialization boundary.
//!
//! Continuations retain only canonical logical dependencies. A runner-level
//! adapter owns the bridge to the runtime-wide physical materialization registry;
//! exact loads are identified exclusively by [`LoadKey`].

#[cfg(any(feature = "cuda", test))]
use ferrule_common::LogicalDependency;
use ferrule_common::{
    ArtifactFormat, BackendId, CancellationReason, CompletionEvent, ContentHash, ContinuationId,
    DependencySet, DestinationGeneration, DeviceId, Error, ExpertId, ExpertLeaseSet, FailureReason,
    LayerId, LoadKey, LoadStage, ModelInstanceId, OperationId,
    RegisteredPinnedAlignedSlabLeaseDescriptor, ResidencyBinding, Result, SourceGeneration,
    SourceIdentityHash, UploadFenceContract, ValidatedResidencyBinding,
};

/// Versioned protocol format for exact HF safetensors expert payloads.
pub const HF_SAFETENSORS_EXPERT_FORMAT_V1: ArtifactFormat = ArtifactFormat::new(1);

/// Immutable checkpoint/catalog identity for one expert payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertArtifactIdentity {
    source: SourceIdentityHash,
    content_hash: ContentHash,
    format: ArtifactFormat,
    source_generation: SourceGeneration,
}

impl ExpertArtifactIdentity {
    pub fn new(
        source: SourceIdentityHash,
        content_hash: ContentHash,
        format: ArtifactFormat,
        source_generation: SourceGeneration,
    ) -> Result<Self> {
        if source.is_zero() {
            return Err(Error::Model(
                "expert artifact source identity hash must be non-zero".into(),
            ));
        }
        if content_hash.is_zero() {
            return Err(Error::Model(
                "expert artifact content hash must be non-zero".into(),
            ));
        }
        if format.get() == 0 {
            return Err(Error::Model(
                "expert artifact format identity must be non-zero".into(),
            ));
        }
        if source_generation.is_zero() {
            return Err(Error::Model(
                "expert artifact source generation must be non-zero".into(),
            ));
        }
        Ok(Self {
            source,
            content_hash,
            format,
            source_generation,
        })
    }

    pub const fn source(self) -> SourceIdentityHash {
        self.source
    }

    pub const fn content_hash(self) -> ContentHash {
        self.content_hash
    }

    pub const fn format(self) -> ArtifactFormat {
        self.format
    }

    pub const fn source_generation(self) -> SourceGeneration {
        self.source_generation
    }
}

/// Runtime-assigned model and placement namespace for exact load identity.
///
/// These values come from the installed driver/registry, never from a model-local
/// counter, object address, or transient operation ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertMaterializationPlacement {
    model: ModelInstanceId,
    backend: BackendId,
    device: DeviceId,
}

impl ExpertMaterializationPlacement {
    pub fn new(model: ModelInstanceId, backend: BackendId, device: DeviceId) -> Result<Self> {
        if model.is_zero() {
            return Err(Error::Model(
                "expert materialization model instance must be non-zero".into(),
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

/// Exact model/catalog/backend coordinates resolved after routing and before a
/// destination reservation is joined or created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertMaterializationRequest {
    model: ModelInstanceId,
    artifact: ExpertArtifactIdentity,
    layer: LayerId,
    expert: ExpertId,
    backend: BackendId,
    device: DeviceId,
}

impl ExpertMaterializationRequest {
    pub fn new(
        model: ModelInstanceId,
        artifact: ExpertArtifactIdentity,
        layer: LayerId,
        expert: ExpertId,
        backend: BackendId,
        device: DeviceId,
    ) -> Result<Self> {
        Self::for_placement(
            ExpertMaterializationPlacement::new(model, backend, device)?,
            artifact,
            layer,
            expert,
        )
    }

    pub fn for_placement(
        placement: ExpertMaterializationPlacement,
        artifact: ExpertArtifactIdentity,
        layer: LayerId,
        expert: ExpertId,
    ) -> Result<Self> {
        Ok(Self {
            model: placement.model,
            artifact,
            layer,
            expert,
            backend: placement.backend,
            device: placement.device,
        })
    }

    /// Complete a load identity with the generation returned by the exact slot
    /// reservation. Callers must not synthesize this generation.
    pub fn load_key(self, destination_generation: DestinationGeneration) -> Result<LoadKey> {
        Ok(LoadKey::new(
            self.model,
            self.artifact.source,
            self.artifact.content_hash,
            self.layer,
            self.expert,
            self.artifact.format,
            self.backend,
            self.device,
            self.artifact.source_generation,
            destination_generation,
        )?)
    }

    pub fn validate_key(self, key: LoadKey) -> Result<()> {
        key.validate()?;
        let fields = [
            ("model instance", key.model() == self.model),
            ("source identity", key.source() == self.artifact.source),
            (
                "source content hash",
                key.source_hash() == self.artifact.content_hash,
            ),
            ("layer", key.layer() == self.layer),
            ("expert", key.expert() == self.expert),
            (
                "artifact format",
                key.artifact_format() == self.artifact.format,
            ),
            ("backend", key.backend() == self.backend),
            ("device", key.device() == self.device),
            (
                "source generation",
                key.source_generation() == self.artifact.source_generation,
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

    pub const fn artifact(self) -> ExpertArtifactIdentity {
        self.artifact
    }

    pub const fn layer(self) -> LayerId {
        self.layer
    }

    pub const fn expert(self) -> ExpertId {
        self.expert
    }

    pub const fn backend(self) -> BackendId {
        self.backend
    }

    pub const fn device(self) -> DeviceId {
        self.device
    }
}

/// Slot reservation returned by the model-owned physical authority.
///
/// The destination generation is derived from the exact `ExpertSlotGeneration`
/// returned by `prepare_install`; callers must use `key` verbatim. `evicted`
/// identifies an older physical publication invalidated by the slot reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalExpertReservationDescriptor {
    key: LoadKey,
    binding: ResidencyBinding,
    evicted: Option<LoadKey>,
}

impl PhysicalExpertReservationDescriptor {
    pub fn new(key: LoadKey, binding: ResidencyBinding, evicted: Option<LoadKey>) -> Result<Self> {
        ValidatedResidencyBinding::new(key, binding)?;
        if let Some(evicted) = evicted {
            evicted.validate()?;
            if evicted == key {
                return Err(Error::Execution(
                    "physical expert reservation cannot evict its own load key".into(),
                ));
            }
        }
        Ok(Self {
            key,
            binding,
            evicted,
        })
    }

    pub const fn key(self) -> LoadKey {
        self.key
    }

    pub const fn binding(self) -> ResidencyBinding {
        self.binding
    }

    pub const fn evicted(self) -> Option<LoadKey> {
        self.evicted
    }
}

/// Result of resolving a request against model-owned slot/frame authority.
#[derive(Debug, PartialEq, Eq)]
pub enum PhysicalExpertReservation {
    Resident(ValidatedResidencyBinding),
    Reserved(PhysicalExpertReservationDescriptor),
}

/// Registered pinned storage and upload-fence contract for one registry operation.
///
/// Descriptors are immutable address metadata. The physical backend retains the
/// actual registered allocations and provider tickets until cancellation or the
/// matching read/upload completion has drained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalExpertOperationReservation {
    key: LoadKey,
    binding: ResidencyBinding,
    slabs: Box<[RegisteredPinnedAlignedSlabLeaseDescriptor]>,
    upload_fence: UploadFenceContract,
}

impl PhysicalExpertOperationReservation {
    pub fn new(
        key: LoadKey,
        binding: ResidencyBinding,
        slabs: impl Into<Box<[RegisteredPinnedAlignedSlabLeaseDescriptor]>>,
        upload_fence: UploadFenceContract,
    ) -> Result<Self> {
        ValidatedResidencyBinding::new(key, binding)?;
        let slabs = slabs.into();
        if slabs.is_empty() {
            return Err(Error::Execution(
                "physical expert operation requires at least one registered pinned slab".into(),
            ));
        }
        for slab in &slabs {
            if slab.operation() != upload_fence.operation
                || slab.source_generation() != key.source_generation()
                || slab.destination_generation() != key.destination_generation()
            {
                return Err(Error::Execution(
                    "physical expert slab identity does not match its operation/load key".into(),
                ));
            }
        }
        if upload_fence.destination_generation != key.destination_generation()
            || upload_fence.fence.is_zero()
        {
            return Err(Error::Execution(
                "physical expert upload fence does not match its load key".into(),
            ));
        }
        Ok(Self {
            key,
            binding,
            slabs,
            upload_fence,
        })
    }

    pub const fn key(&self) -> LoadKey {
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
/// Stage limits bound read/upload/install transitions. Device-frame bytes cover
/// both published residency and replacement shadow frames. Execution leases are
/// reported per continuation so the runtime can scale them by admitted concurrency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalExpertResourceTopology {
    stage_limits: ferrule_common::expert_io::ExpertIoResourceLimits,
    device_frame_bytes: u64,
    execution_lease_slots_per_continuation: u64,
}

impl PhysicalExpertResourceTopology {
    pub fn new(
        stage_limits: ferrule_common::expert_io::ExpertIoResourceLimits,
        device_frame_bytes: u64,
        execution_lease_slots_per_continuation: u64,
    ) -> Result<Self> {
        let stage_limits = stage_limits.validate()?;
        if !stage_limits.capacity.is_empty()
            && (device_frame_bytes == 0 || execution_lease_slots_per_continuation == 0)
        {
            return Err(Error::Execution(
                "physical expert topology requires non-zero frame and execution-lease capacity"
                    .into(),
            ));
        }
        Ok(Self {
            stage_limits,
            device_frame_bytes,
            execution_lease_slots_per_continuation,
        })
    }

    pub const fn stage_limits(self) -> ferrule_common::expert_io::ExpertIoResourceLimits {
        self.stage_limits
    }

    pub const fn device_frame_bytes(self) -> u64 {
        self.device_frame_bytes
    }

    pub const fn execution_lease_slots_per_continuation(self) -> u64 {
        self.execution_lease_slots_per_continuation
    }
}

/// Model-owned physical expert materialization provider.
///
/// The trait is object-safe and depends only on model/common protocol types. A
/// runtime wrapper owns the box after `MultiSessionRunner` transfers it once.
/// Slot/frame publication remains model authority; runtime code owns waiters,
/// credits, operation IDs, completion validation, and retirement.
pub trait PhysicalExpertMaterializationBackend: std::fmt::Debug + Send {
    fn placement(&self) -> ExpertMaterializationPlacement;

    fn resource_topology(&self) -> Result<PhysicalExpertResourceTopology>;

    /// Resolve an exact source request, calling the real residency
    /// `prepare_install` before constructing a destination generation and key.
    fn resolve_or_reserve(
        &mut self,
        request: ExpertMaterializationRequest,
    ) -> std::result::Result<PhysicalExpertReservation, FailureReason>;

    fn materialization_bytes(&self, key: LoadKey) -> std::result::Result<u64, FailureReason>;

    /// Release the selected execution ownership retained by the runtime attachment.
    /// Runtime attachments are the only logical owners. A release racing a pending
    /// install must be remembered so later publication cannot leave an ownerless lease.
    fn release_selected(&mut self, key: LoadKey) -> std::result::Result<(), FailureReason>;

    fn reserve(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        bytes: u64,
    ) -> std::result::Result<PhysicalExpertOperationReservation, FailureReason>;

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &PhysicalExpertOperationReservation,
        bytes: u64,
    ) -> std::result::Result<(), FailureReason>;

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &PhysicalExpertOperationReservation,
        bytes: u64,
    ) -> std::result::Result<(), FailureReason>;

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &PhysicalExpertOperationReservation,
        bytes: u64,
    ) -> std::result::Result<(), FailureReason>;

    fn cancel(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> std::result::Result<(), FailureReason>;

    fn next_completion(&mut self) -> Option<CompletionEvent>;
}

/// Residency resolution for one exact post-router expert requirement.
#[derive(Debug, PartialEq, Eq)]
pub enum ExpertDependencyResolution {
    /// The exact source/generation/backend/device binding is already resident.
    Resident(ValidatedResidencyBinding),
    /// The exact destination reservation is unresolved and may be single-flight
    /// joined by any transaction carrying the same key.
    Waiting(LoadKey),
}

impl ExpertDependencyResolution {
    pub fn resident(
        request: ExpertMaterializationRequest,
        binding: ValidatedResidencyBinding,
    ) -> Result<Self> {
        request.validate_key(binding.key())?;
        Ok(Self::Resident(binding))
    }

    pub fn waiting(request: ExpertMaterializationRequest, key: LoadKey) -> Result<Self> {
        request.validate_key(key)?;
        Ok(Self::Waiting(key))
    }
}

/// Model-facing edge of the runtime materialization owner.
///
/// `resolve` fixes the exact physical key and destination generation. The runtime
/// registry exclusively owns submission, progress, publication, cancellation, and
/// retirement; the model can only release an unresolved dependency with `detach`.
pub trait ExpertMaterializationAdapter: Send {
    /// Stable runtime-assigned namespace used for every request resolved by this
    /// adapter instance.
    fn placement(&self) -> ExpertMaterializationPlacement;

    fn resolve(
        &mut self,
        request: ExpertMaterializationRequest,
    ) -> Result<ExpertDependencyResolution>;

    /// Detach one continuation's logical demand. This must not imply physical
    /// cancellation, even when the continuation is the adapter's last known waiter.
    fn detach(&mut self, continuation: ContinuationId, key: LoadKey) -> Result<()>;
}

/// One unresolved dependency edge retained by a model continuation.
///
/// This object owns no resumed execution attachment, read, upload, slot reservation,
/// provider ticket, or physical cancellation authority. Runtime owns the attachment
/// and releases it when the resume edge finishes. Before resume, model cancellation
/// may detach the still-unresolved edge; successful detaches remain retry-safe.
#[derive(Debug)]
pub struct ContinuationDependencyState {
    continuation: ContinuationId,
    unresolved: Option<DependencySet>,
    detach_remaining: Option<Vec<LoadKey>>,
}

impl ContinuationDependencyState {
    pub fn new(continuation: ContinuationId, unresolved: DependencySet) -> Result<Self> {
        validate_continuation_id(continuation)?;
        unresolved.validate()?;
        Ok(Self {
            continuation,
            unresolved: Some(unresolved),
            detach_remaining: None,
        })
    }

    pub const fn continuation(&self) -> ContinuationId {
        self.continuation
    }

    pub fn unresolved(&self) -> Option<&DependencySet> {
        self.unresolved.as_ref()
    }

    pub fn has_expert_dependencies(&self) -> bool {
        self.detach_remaining
            .as_ref()
            .is_some_and(|remaining| !remaining.is_empty())
            || self.unresolved.as_ref().is_some_and(|dependencies| {
                dependencies
                    .iter()
                    .any(|dependency| dependency.load_key().is_some())
            })
    }

    pub fn clear_non_expert_dependencies(&mut self) -> Result<()> {
        if self.has_expert_dependencies() {
            return Err(Error::Execution(format!(
                "continuation {} still has expert dependencies that must be detached through the materialization adapter",
                self.continuation.get()
            )));
        }
        self.unresolved = None;
        self.detach_remaining = None;
        Ok(())
    }

    pub fn replace_unresolved(&mut self, unresolved: DependencySet) -> Result<()> {
        if self.unresolved.is_some() || self.detach_remaining.is_some() {
            return Err(Error::Execution(format!(
                "continuation {} still owns an unresolved dependency set or has detach cleanup in progress",
                self.continuation.get()
            )));
        }
        unresolved.validate()?;
        self.unresolved = Some(unresolved);
        Ok(())
    }

    /// Validate and consume exactly one resume edge. The lease set must contain
    /// every and only expert-residency dependency in the suspended set. Runtime
    /// retains and later releases the attachment for the consumed edge.
    pub fn validate_resume(
        &mut self,
        continuation: ContinuationId,
        leases: &ExpertLeaseSet,
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
            .filter_map(|dependency| dependency.load_key())
            .collect::<Vec<_>>();
        if leases.len() != required.len()
            || required
                .iter()
                .any(|key| leases.binding_for(*key).is_none())
        {
            return Err(Error::Execution(format!(
                "continuation {} lease set does not exactly satisfy its unresolved expert dependencies",
                self.continuation.get()
            )));
        }
        self.unresolved = None;
        Ok(())
    }

    /// Detach only the current unresolved expert edge during cancellation/failure.
    /// A successfully resumed edge is no longer represented here and must be released
    /// only by runtime. A failed detach keeps the remaining keys for an exact retry.
    pub fn detach_logical_dependencies(
        &mut self,
        adapter: &mut dyn ExpertMaterializationAdapter,
    ) -> Result<()> {
        if self.detach_remaining.is_none() {
            let Some(unresolved) = self.unresolved.as_ref() else {
                return Ok(());
            };
            let remaining = unresolved
                .iter()
                .filter_map(|dependency| dependency.load_key())
                .collect::<Vec<_>>();
            if remaining.is_empty() {
                self.unresolved = None;
                return Ok(());
            }
            self.detach_remaining = Some(remaining);
        }

        let remaining = self
            .detach_remaining
            .as_mut()
            .expect("detach list initialized above");
        while let Some(key) = remaining.first().copied() {
            adapter.detach(self.continuation, key)?;
            remaining.remove(0);
        }
        self.detach_remaining = None;
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

#[cfg(any(feature = "cuda", test))]
pub(crate) fn expert_dependency_set(
    keys: impl IntoIterator<Item = LoadKey>,
) -> Result<DependencySet> {
    let dependencies = keys
        .into_iter()
        .map(LogicalDependency::expert_resident)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(DependencySet::new(dependencies)?)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use ferrule_common::execution::ExecutionTransactionId;
    use ferrule_common::{
        DestinationSlotId, DispatchFenceContract, FenceId, MappingEpoch, OperationId,
        ResidencyBinding,
    };

    use super::*;

    fn bytes(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn artifact(source: u8, content: u8, format: u32, generation: u64) -> ExpertArtifactIdentity {
        ExpertArtifactIdentity::new(
            SourceIdentityHash::new(bytes(source)),
            ContentHash::new(bytes(content)),
            ArtifactFormat::new(format),
            SourceGeneration::new(generation),
        )
        .unwrap()
    }

    fn request() -> ExpertMaterializationRequest {
        ExpertMaterializationRequest::new(
            ModelInstanceId::new(11),
            artifact(12, 13, 14, 15),
            LayerId::new(16),
            ExpertId::new(17),
            BackendId::new(18),
            DeviceId::new(19),
        )
        .unwrap()
    }

    fn key() -> LoadKey {
        request().load_key(DestinationGeneration::new(20)).unwrap()
    }

    fn key_for_expert(expert: u32) -> LoadKey {
        ExpertMaterializationRequest::new(
            request().model(),
            request().artifact(),
            request().layer(),
            ExpertId::new(expert),
            request().backend(),
            request().device(),
        )
        .unwrap()
        .load_key(DestinationGeneration::new(20))
        .unwrap()
    }

    fn binding(key: LoadKey, slot: u32) -> ValidatedResidencyBinding {
        ValidatedResidencyBinding::new(
            key,
            ResidencyBinding::new(
                key.model(),
                key.layer(),
                key.expert(),
                key.backend(),
                key.device(),
                DestinationSlotId::new(slot),
                key.destination_generation(),
            ),
        )
        .unwrap()
    }

    fn leases(keys: &[LoadKey]) -> ExpertLeaseSet {
        let placement = keys.first().copied().unwrap_or_else(key);
        ExpertLeaseSet::new(
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
    struct MockAdapter {
        reservations: BTreeMap<ExpertMaterializationRequest, LoadKey>,
        runtime_attachments: BTreeMap<LoadKey, usize>,
        physical_releases: BTreeMap<LoadKey, usize>,
        detached: BTreeSet<(ContinuationId, LoadKey)>,
        detach_attempts: Vec<(ContinuationId, LoadKey)>,
        fail_detach_once_for: Option<LoadKey>,
        detach_failed: bool,
    }

    impl MockAdapter {
        fn runtime_attach(&mut self, key: LoadKey) {
            *self.runtime_attachments.entry(key).or_default() += 1;
        }

        fn runtime_attachment_count(&self, key: LoadKey) -> usize {
            self.runtime_attachments.get(&key).copied().unwrap_or(0)
        }

        fn physical_release_count(&self, key: LoadKey) -> usize {
            self.physical_releases.get(&key).copied().unwrap_or(0)
        }

        fn is_evictable(&self, key: LoadKey) -> bool {
            self.runtime_attachment_count(key) == 0
        }

        fn runtime_finish_resume_lease(
            &mut self,
            continuation: ContinuationId,
            key: LoadKey,
        ) -> Result<()> {
            ExpertMaterializationAdapter::detach(self, continuation, key)
        }

        fn runtime_detach_if_attached(
            &mut self,
            continuation: ContinuationId,
            key: LoadKey,
        ) -> Result<bool> {
            if self.detached.contains(&(continuation, key)) {
                return Ok(false);
            }
            ExpertMaterializationAdapter::detach(self, continuation, key)?;
            Ok(true)
        }
    }

    impl ExpertMaterializationAdapter for MockAdapter {
        fn placement(&self) -> ExpertMaterializationPlacement {
            ExpertMaterializationPlacement::new(
                request().model(),
                request().backend(),
                request().device(),
            )
            .unwrap()
        }

        fn resolve(
            &mut self,
            request: ExpertMaterializationRequest,
        ) -> Result<ExpertDependencyResolution> {
            let key = if let Some(key) = self.reservations.get(&request) {
                *key
            } else {
                let key = request.load_key(DestinationGeneration::new(20))?;
                self.reservations.insert(request, key);
                key
            };
            ExpertDependencyResolution::waiting(request, key)
        }

        fn detach(&mut self, continuation: ContinuationId, key: LoadKey) -> Result<()> {
            self.detach_attempts.push((continuation, key));
            if self.fail_detach_once_for == Some(key) && !self.detach_failed {
                self.detach_failed = true;
                return Err(Error::Execution("injected detach failure".into()));
            }
            if !self.detached.insert((continuation, key)) {
                return Err(Error::Execution("duplicate logical detach".into()));
            }
            let attachments = self.runtime_attachments.get_mut(&key).ok_or_else(|| {
                Error::Execution("logical detach has no runtime attachment".into())
            })?;
            *attachments = attachments
                .checked_sub(1)
                .ok_or_else(|| Error::Execution("runtime attachment count underflow".into()))?;
            if *attachments == 0 {
                self.runtime_attachments.remove(&key);
                *self.physical_releases.entry(key).or_default() += 1;
            }
            Ok(())
        }
    }

    #[test]
    fn load_key_identity_includes_every_source_placement_and_generation_dimension() {
        let base = key();
        let variants = [
            ExpertMaterializationRequest::new(
                ModelInstanceId::new(21),
                request().artifact(),
                request().layer(),
                request().expert(),
                request().backend(),
                request().device(),
            )
            .unwrap()
            .load_key(base.destination_generation())
            .unwrap(),
            ExpertMaterializationRequest::new(
                request().model(),
                artifact(22, 13, 14, 15),
                request().layer(),
                request().expert(),
                request().backend(),
                request().device(),
            )
            .unwrap()
            .load_key(base.destination_generation())
            .unwrap(),
            ExpertMaterializationRequest::new(
                request().model(),
                artifact(12, 23, 14, 15),
                request().layer(),
                request().expert(),
                request().backend(),
                request().device(),
            )
            .unwrap()
            .load_key(base.destination_generation())
            .unwrap(),
            ExpertMaterializationRequest::new(
                request().model(),
                request().artifact(),
                LayerId::new(24),
                request().expert(),
                request().backend(),
                request().device(),
            )
            .unwrap()
            .load_key(base.destination_generation())
            .unwrap(),
            key_for_expert(25),
            ExpertMaterializationRequest::new(
                request().model(),
                artifact(12, 13, 26, 15),
                request().layer(),
                request().expert(),
                request().backend(),
                request().device(),
            )
            .unwrap()
            .load_key(base.destination_generation())
            .unwrap(),
            ExpertMaterializationRequest::new(
                request().model(),
                request().artifact(),
                request().layer(),
                request().expert(),
                BackendId::new(27),
                request().device(),
            )
            .unwrap()
            .load_key(base.destination_generation())
            .unwrap(),
            ExpertMaterializationRequest::new(
                request().model(),
                request().artifact(),
                request().layer(),
                request().expert(),
                request().backend(),
                DeviceId::new(28),
            )
            .unwrap()
            .load_key(base.destination_generation())
            .unwrap(),
            ExpertMaterializationRequest::new(
                request().model(),
                artifact(12, 13, 14, 29),
                request().layer(),
                request().expert(),
                request().backend(),
                request().device(),
            )
            .unwrap()
            .load_key(base.destination_generation())
            .unwrap(),
            request().load_key(DestinationGeneration::new(30)).unwrap(),
        ];
        assert!(variants.into_iter().all(|variant| variant != base));
        assert_eq!(variants.into_iter().collect::<BTreeSet<_>>().len(), 10);
    }

    #[test]
    fn resolution_distinguishes_resident_binding_from_unresolved_wait() {
        let request = request();
        let key = key();
        assert!(matches!(
            ExpertDependencyResolution::waiting(request, key).unwrap(),
            ExpertDependencyResolution::Waiting(waiting) if waiting == key
        ));
        assert!(matches!(
            ExpertDependencyResolution::resident(request, binding(key, 3)).unwrap(),
            ExpertDependencyResolution::Resident(resident) if resident.key() == key
        ));

        let wrong_source = ExpertMaterializationRequest::new(
            request.model(),
            artifact(31, 13, 14, 15),
            request.layer(),
            request.expert(),
            request.backend(),
            request.device(),
        )
        .unwrap()
        .load_key(key.destination_generation())
        .unwrap();
        assert!(ExpertDependencyResolution::waiting(request, wrong_source).is_err());
    }

    #[test]
    fn resume_requires_complete_exact_lease_set_and_rejects_wrong_identity_dimensions() {
        let first = key_for_expert(1);
        let second = key_for_expert(2);
        let dependencies = expert_dependency_set([first, second]).unwrap();
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
            ExpertMaterializationRequest::new(
                first.model(),
                artifact(42, 13, 14, 15),
                first.layer(),
                first.expert(),
                first.backend(),
                first.device(),
            )
            .unwrap()
            .load_key(first.destination_generation())
            .unwrap(),
            ExpertMaterializationRequest::new(
                first.model(),
                request().artifact(),
                first.layer(),
                first.expert(),
                first.backend(),
                first.device(),
            )
            .unwrap()
            .load_key(DestinationGeneration::new(43))
            .unwrap(),
            ExpertMaterializationRequest::new(
                first.model(),
                request().artifact(),
                first.layer(),
                first.expert(),
                BackendId::new(44),
                first.device(),
            )
            .unwrap()
            .load_key(first.destination_generation())
            .unwrap(),
        ] {
            let mut state = ContinuationDependencyState::new(
                continuation,
                expert_dependency_set([first]).unwrap(),
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
        assert!(!complete.has_expert_dependencies());
    }

    #[test]
    fn sibling_wait_cancellation_preserves_other_runtime_attachment() {
        let key = key();
        let dependencies = expert_dependency_set([key]).unwrap();
        let mut first =
            ContinuationDependencyState::new(ContinuationId::new(51), dependencies.clone())
                .unwrap();
        let mut sibling =
            ContinuationDependencyState::new(ContinuationId::new(52), dependencies).unwrap();
        let mut adapter = MockAdapter::default();
        adapter.runtime_attach(key);
        adapter.runtime_attach(key);
        assert_eq!(adapter.runtime_attachment_count(key), 2);

        first.detach_logical_dependencies(&mut adapter).unwrap();
        assert_eq!(adapter.runtime_attachment_count(key), 1);
        assert_eq!(adapter.physical_release_count(key), 0);
        assert!(!adapter.is_evictable(key));
        assert!(sibling.unresolved().is_some());
        assert!(
            !adapter
                .runtime_detach_if_attached(ContinuationId::new(51), key)
                .unwrap()
        );

        sibling.detach_logical_dependencies(&mut adapter).unwrap();
        assert_eq!(adapter.runtime_attachment_count(key), 0);
        assert_eq!(adapter.physical_release_count(key), 1);
        assert!(adapter.is_evictable(key));
        assert!(
            !adapter
                .runtime_detach_if_attached(ContinuationId::new(52), key)
                .unwrap()
        );
        assert_eq!(adapter.physical_release_count(key), 1);
    }

    #[test]
    fn successful_resume_consumes_model_edge_without_releasing_runtime_attachment() {
        let key = key();
        let continuation = ContinuationId::new(61);
        let mut state =
            ContinuationDependencyState::new(continuation, expert_dependency_set([key]).unwrap())
                .unwrap();
        let leases = leases(&[key]);
        let mut adapter = MockAdapter::default();
        adapter.runtime_attach(key);

        state.validate_resume(continuation, &leases).unwrap();
        assert!(state.unresolved().is_none());
        assert!(!state.has_expert_dependencies());
        state.detach_logical_dependencies(&mut adapter).unwrap();
        assert!(adapter.detach_attempts.is_empty());
        assert_eq!(adapter.runtime_attachment_count(key), 1);

        adapter
            .runtime_finish_resume_lease(continuation, key)
            .unwrap();
        assert_eq!(adapter.physical_release_count(key), 1);
        let replay = state
            .validate_resume(continuation, &leases)
            .unwrap_err()
            .to_string();
        assert!(replay.contains("refusing resume replay"));
    }

    #[test]
    fn replace_unresolved_moves_to_next_edge_without_cross_edge_accumulation() {
        let first = key_for_expert(1);
        let second = key_for_expert(2);
        let continuation = ContinuationId::new(64);
        let mut state =
            ContinuationDependencyState::new(continuation, expert_dependency_set([first]).unwrap())
                .unwrap();
        let mut adapter = MockAdapter::default();

        adapter.runtime_attach(first);
        state
            .validate_resume(continuation, &leases(&[first]))
            .unwrap();
        adapter
            .runtime_finish_resume_lease(continuation, first)
            .unwrap();
        state
            .replace_unresolved(expert_dependency_set([second]).unwrap())
            .unwrap();
        assert_eq!(state.unresolved().unwrap().len(), 1);

        adapter.runtime_attach(second);
        state
            .validate_resume(continuation, &leases(&[second]))
            .unwrap();
        adapter
            .runtime_finish_resume_lease(continuation, second)
            .unwrap();
        state.detach_logical_dependencies(&mut adapter).unwrap();

        assert_eq!(adapter.physical_release_count(first), 1);
        assert_eq!(adapter.physical_release_count(second), 1);
        assert_eq!(adapter.detach_attempts.len(), 2);
        assert!(adapter.is_evictable(first));
        assert!(adapter.is_evictable(second));
    }

    #[test]
    fn resumed_failure_leaves_release_to_runtime_and_next_turn_can_evict() {
        let key = key();
        let mut adapter = MockAdapter::default();
        let first_continuation = ContinuationId::new(62);
        let mut first = ContinuationDependencyState::new(
            first_continuation,
            expert_dependency_set([key]).unwrap(),
        )
        .unwrap();
        adapter.runtime_attach(key);
        first
            .validate_resume(first_continuation, &leases(&[key]))
            .unwrap();

        first.detach_logical_dependencies(&mut adapter).unwrap();
        assert!(adapter.detach_attempts.is_empty());
        assert_eq!(adapter.physical_release_count(key), 0);
        adapter
            .runtime_finish_resume_lease(first_continuation, key)
            .unwrap();
        assert_eq!(adapter.physical_release_count(key), 1);
        assert!(adapter.is_evictable(key));

        let second_continuation = ContinuationId::new(63);
        let mut second = ContinuationDependencyState::new(
            second_continuation,
            expert_dependency_set([key]).unwrap(),
        )
        .unwrap();
        adapter.runtime_attach(key);
        second
            .validate_resume(second_continuation, &leases(&[key]))
            .unwrap();
        assert!(!adapter.is_evictable(key));
        adapter
            .runtime_finish_resume_lease(second_continuation, key)
            .unwrap();
        second.detach_logical_dependencies(&mut adapter).unwrap();
        assert_eq!(adapter.physical_release_count(key), 2);
        assert!(adapter.is_evictable(key));
    }

    #[test]
    fn logical_cleanup_retries_only_the_failed_and_remaining_detaches() {
        let first = key_for_expert(1);
        let second = key_for_expert(2);
        let continuation = ContinuationId::new(71);
        let mut state = ContinuationDependencyState::new(
            continuation,
            expert_dependency_set([second, first]).unwrap(),
        )
        .unwrap();
        let mut adapter = MockAdapter {
            fail_detach_once_for: Some(second),
            ..MockAdapter::default()
        };
        adapter.runtime_attach(first);
        adapter.runtime_attach(second);

        assert!(state.detach_logical_dependencies(&mut adapter).is_err());
        assert!(state.unresolved().is_some());
        state.detach_logical_dependencies(&mut adapter).unwrap();
        assert!(state.unresolved().is_none());
        assert_eq!(
            adapter.detach_attempts,
            vec![
                (continuation, first),
                (continuation, second),
                (continuation, second)
            ]
        );
    }
}
