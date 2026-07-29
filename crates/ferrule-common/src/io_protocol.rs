//! Final model-neutral I/O, residency, completion, and retirement protocol.
//!
//! These types encode the owner-side contracts from `docs/ARCHITECTURE.md`.
//! They contain no file paths, CUDA objects, model-family state, or global epoch
//! identity. Physical providers receive immutable descriptors and return
//! generation-qualified completion events; only the Rust owner transitions state
//! and consumes retirement authority.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution::ExecutionTransactionId;

/// Result type for final I/O protocol validation.
pub type IoProtocolResult<T> = std::result::Result<T, IoProtocolError>;

/// A fail-closed violation of an I/O protocol invariant.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IoProtocolError {
    #[error("invalid load key: {0}")]
    InvalidLoadKey(&'static str),
    #[error("invalid waiter identity: {0}")]
    InvalidWaiterId(&'static str),
    #[error("a dependency set must contain at least one dependency")]
    EmptyDependencySet,
    #[error("dependency set is not in strictly sorted canonical order")]
    NonCanonicalDependencySet,
    #[error("illegal load-stage transition from {from:?} to {to:?}")]
    InvalidLoadTransition { from: LoadStage, to: LoadStage },
    #[error("completion field `{field}` does not match the owner expectation")]
    CompletionMismatch { field: &'static str },
    #[error("completion generation {observed:?} does not match owner generation {expected:?}")]
    CompletionGenerationMismatch {
        expected: CompletionGeneration,
        observed: CompletionGeneration,
    },
    #[error("{stage:?} is not a provider-submitted completion stage")]
    InvalidCompletionStage { stage: LoadStage },
    #[error("successful completion returned {actual} bytes, expected exactly {expected}")]
    IncompleteSuccessfulCompletion { expected: u64, actual: u64 },
    #[error("unsuccessful completion returned {actual} bytes, exceeding expected {expected}")]
    CompletionByteOverflow { expected: u64, actual: u64 },
    #[error("retirement authority was already consumed")]
    RetirementAlreadyConsumed,
    #[error("slab alignment {alignment} must be a non-zero power of two")]
    InvalidAlignment { alignment: usize },
    #[error("slab {field} value {value} is not aligned to {alignment}")]
    MisalignedSlabField {
        field: &'static str,
        value: u64,
        alignment: usize,
    },
    #[error("registered slab address must be non-zero")]
    NullSlabAddress,
    #[error("arithmetic overflow while validating {0}")]
    ArithmeticOverflow(&'static str),
    #[error("slab lease range {offset}..{end} exceeds registered allocation length {capacity}")]
    SlabRangeOutOfBounds {
        offset: u64,
        end: u64,
        capacity: u64,
    },
    #[error("illegal slab lease transition from {from:?} to {to:?}")]
    InvalidSlabTransition {
        from: SlabLeaseState,
        to: SlabLeaseState,
    },
    #[error("upload fence field `{field}` does not match the retained contract")]
    UploadFenceMismatch { field: &'static str },
    #[error("residency field `{field}` does not match the required load key")]
    ResidencyMismatch { field: &'static str },
    #[error("mapping epoch must be non-zero")]
    InvalidMappingEpoch,
    #[error("completion fence identity must be non-zero")]
    InvalidFenceContract,
    #[error("duplicate residency binding for {0:?}")]
    DuplicateResidencyBinding(Box<LoadKey>),
    #[error("missing residency binding for {0:?}")]
    MissingResidencyBinding(Box<LoadKey>),
    #[error("unexpected residency binding for {0:?}")]
    UnexpectedResidencyBinding(Box<LoadKey>),
    #[error("dependency {0:?} is not an expert-residency dependency")]
    NonResidencyDependency(Box<LogicalDependency>),
}

macro_rules! define_u64_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u64 {
                self.0
            }

            pub const fn is_zero(self) -> bool {
                self.0 == 0
            }
        }

        impl From<u64> for $name {
            fn from(value: u64) -> Self {
                Self::new(value)
            }
        }

        impl From<$name> for u64 {
            fn from(value: $name) -> Self {
                value.get()
            }
        }
    };
}

macro_rules! define_u32_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name(u32);

        impl $name {
            pub const fn new(value: u32) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u32 {
                self.0
            }
        }

        impl From<u32> for $name {
            fn from(value: u32) -> Self {
                Self::new(value)
            }
        }

        impl From<$name> for u32 {
            fn from(value: $name) -> Self {
                value.get()
            }
        }
    };
}

define_u64_id! {
    /// Stable identity for one physical read/upload/install operation.
    OperationId
}
define_u64_id! {
    /// Stable identity for one resumable model continuation.
    ContinuationId
}
define_u64_id! {
    /// Stable identity for one loaded model instance.
    ModelInstanceId
}
define_u64_id! {
    /// Stable identity for an injected execution or placement backend.
    BackendId
}
define_u64_id! {
    /// Stable identity for a device within a backend.
    DeviceId
}
define_u64_id! {
    /// Generation of immutable source identity and bytes.
    SourceGeneration
}
define_u64_id! {
    /// Generation of a reserved destination and its reuse cycle.
    DestinationGeneration
}
define_u64_id! {
    /// Request generation captured when a waiter is attached.
    RequestGeneration
}
define_u64_id! {
    /// Epoch of the exact dependency set captured by a waiter.
    DependencySetEpoch
}
define_u64_id! {
    /// Stable registered host-slab identity.
    SlabId
}
define_u64_id! {
    /// Stable host registration identity.
    RegistrationId
}
define_u64_id! {
    /// Stable provider fence identity.
    FenceId
}
define_u64_id! {
    /// Owner mapping epoch revalidated before dispatch.
    MappingEpoch
}
define_u32_id! {
    /// Model-neutral layer identity.
    LayerId
}
define_u32_id! {
    /// Model-neutral expert identity within a layer.
    ExpertId
}
define_u32_id! {
    /// Stable destination slot/frame identity.
    DestinationSlotId
}

/// Exact waiter identity used by both waiter indices.
///
/// The transaction, request generation, dependency-set epoch, and continuation
/// are all part of identity; changing any one creates a distinct waiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaiterId {
    transaction: ExecutionTransactionId,
    request_generation: RequestGeneration,
    dependency_set_epoch: DependencySetEpoch,
    continuation: ContinuationId,
}

impl WaiterId {
    pub fn new(
        transaction: ExecutionTransactionId,
        request_generation: RequestGeneration,
        dependency_set_epoch: DependencySetEpoch,
        continuation: ContinuationId,
    ) -> IoProtocolResult<Self> {
        let waiter = Self {
            transaction,
            request_generation,
            dependency_set_epoch,
            continuation,
        };
        waiter.validate()?;
        Ok(waiter)
    }

    pub const fn transaction(self) -> ExecutionTransactionId {
        self.transaction
    }

    pub const fn request_generation(self) -> RequestGeneration {
        self.request_generation
    }

    pub const fn dependency_set_epoch(self) -> DependencySetEpoch {
        self.dependency_set_epoch
    }

    pub const fn continuation(self) -> ContinuationId {
        self.continuation
    }

    pub fn validate(self) -> IoProtocolResult<()> {
        if self.request_generation.is_zero() {
            return Err(IoProtocolError::InvalidWaiterId(
                "request generation must be non-zero",
            ));
        }
        if self.dependency_set_epoch.is_zero() {
            return Err(IoProtocolError::InvalidWaiterId(
                "dependency-set epoch must be non-zero",
            ));
        }
        if self.continuation.is_zero() {
            return Err(IoProtocolError::InvalidWaiterId(
                "continuation ID must be non-zero",
            ));
        }
        Ok(())
    }
}

/// Hash of a source namespace and immutable source identity (not a file path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct SourceIdentityHash([u8; 32]);

impl SourceIdentityHash {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|byte| *byte == 0)
    }
}

/// Cryptographic content identity of the immutable expert source bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|byte| *byte == 0)
    }
}

/// Versioned semantic artifact format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct ArtifactFormat(u32);

impl ArtifactFormat {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Complete single-flight and completion-validity identity for one materialization.
///
/// Keys differing in any field are distinct operations and must never coalesce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LoadKey {
    model: ModelInstanceId,
    source: SourceIdentityHash,
    source_hash: ContentHash,
    layer: LayerId,
    expert: ExpertId,
    artifact_format: ArtifactFormat,
    backend: BackendId,
    device: DeviceId,
    source_generation: SourceGeneration,
    destination_generation: DestinationGeneration,
}

impl LoadKey {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: ModelInstanceId,
        source: SourceIdentityHash,
        source_hash: ContentHash,
        layer: LayerId,
        expert: ExpertId,
        artifact_format: ArtifactFormat,
        backend: BackendId,
        device: DeviceId,
        source_generation: SourceGeneration,
        destination_generation: DestinationGeneration,
    ) -> IoProtocolResult<Self> {
        let key = Self {
            model,
            source,
            source_hash,
            layer,
            expert,
            artifact_format,
            backend,
            device,
            source_generation,
            destination_generation,
        };
        key.validate()?;
        Ok(key)
    }

    pub fn validate(&self) -> IoProtocolResult<()> {
        if self.model.is_zero() {
            return Err(IoProtocolError::InvalidLoadKey(
                "model instance ID must be non-zero",
            ));
        }
        if self.source.is_zero() {
            return Err(IoProtocolError::InvalidLoadKey(
                "source identity hash must be non-zero",
            ));
        }
        if self.source_hash.is_zero() {
            return Err(IoProtocolError::InvalidLoadKey(
                "source content hash must be non-zero",
            ));
        }
        if self.artifact_format.get() == 0 {
            return Err(IoProtocolError::InvalidLoadKey(
                "artifact format must be non-zero",
            ));
        }
        if self.source_generation.is_zero() {
            return Err(IoProtocolError::InvalidLoadKey(
                "source generation must be non-zero",
            ));
        }
        if self.destination_generation.is_zero() {
            return Err(IoProtocolError::InvalidLoadKey(
                "destination generation must be non-zero",
            ));
        }
        Ok(())
    }

    pub const fn model(self) -> ModelInstanceId {
        self.model
    }

    pub const fn source(self) -> SourceIdentityHash {
        self.source
    }

    pub const fn source_hash(self) -> ContentHash {
        self.source_hash
    }

    pub const fn layer(self) -> LayerId {
        self.layer
    }

    pub const fn expert(self) -> ExpertId {
        self.expert
    }

    pub const fn artifact_format(self) -> ArtifactFormat {
        self.artifact_format
    }

    pub const fn backend(self) -> BackendId {
        self.backend
    }

    pub const fn device(self) -> DeviceId {
        self.device
    }

    pub const fn source_generation(self) -> SourceGeneration {
        self.source_generation
    }

    pub const fn destination_generation(self) -> DestinationGeneration {
        self.destination_generation
    }
}

/// One exact logical condition on which model progress may wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogicalDependency {
    /// Exact expert bytes must be published at the key's destination generation.
    ExpertResident(LoadKey),
    /// An owner operation must be fully drained and retired.
    OperationRetired(OperationId),
}

impl LogicalDependency {
    pub fn expert_resident(key: LoadKey) -> IoProtocolResult<Self> {
        key.validate()?;
        Ok(Self::ExpertResident(key))
    }

    pub fn operation_retired(operation: OperationId) -> IoProtocolResult<Self> {
        if operation.is_zero() {
            return Err(IoProtocolError::InvalidLoadKey(
                "retirement dependency operation ID must be non-zero",
            ));
        }
        Ok(Self::OperationRetired(operation))
    }

    pub fn validate(&self) -> IoProtocolResult<()> {
        match self {
            Self::ExpertResident(key) => key.validate(),
            Self::OperationRetired(operation) if operation.is_zero() => {
                Err(IoProtocolError::InvalidLoadKey(
                    "retirement dependency operation ID must be non-zero",
                ))
            }
            Self::OperationRetired(_) => Ok(()),
        }
    }

    pub const fn load_key(self) -> Option<LoadKey> {
        match self {
            Self::ExpertResident(key) => Some(key),
            Self::OperationRetired(_) => None,
        }
    }
}

/// Strictly sorted, duplicate-free, non-empty dependency set.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DependencySet {
    dependencies: Box<[LogicalDependency]>,
}

impl DependencySet {
    pub fn new(
        dependencies: impl IntoIterator<Item = LogicalDependency>,
    ) -> IoProtocolResult<Self> {
        let mut dependencies: Vec<_> = dependencies.into_iter().collect();
        for dependency in &dependencies {
            dependency.validate()?;
        }
        dependencies.sort_unstable();
        dependencies.dedup();
        let set = Self {
            dependencies: dependencies.into_boxed_slice(),
        };
        set.validate()?;
        Ok(set)
    }

    /// Validates a producer-supplied canonical representation without sorting it.
    pub fn from_canonical(
        dependencies: impl Into<Box<[LogicalDependency]>>,
    ) -> IoProtocolResult<Self> {
        let set = Self {
            dependencies: dependencies.into(),
        };
        set.validate()?;
        Ok(set)
    }

    pub fn validate(&self) -> IoProtocolResult<()> {
        if self.dependencies.is_empty() {
            return Err(IoProtocolError::EmptyDependencySet);
        }
        for dependency in &self.dependencies {
            dependency.validate()?;
        }
        if self
            .dependencies
            .windows(2)
            .any(|window| window[0] >= window[1])
        {
            return Err(IoProtocolError::NonCanonicalDependencySet);
        }
        Ok(())
    }

    pub fn as_slice(&self) -> &[LogicalDependency] {
        &self.dependencies
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &LogicalDependency> {
        self.dependencies.iter()
    }

    pub fn len(&self) -> usize {
        self.dependencies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dependencies.is_empty()
    }

    pub fn contains(&self, dependency: &LogicalDependency) -> bool {
        self.dependencies.binary_search(dependency).is_ok()
    }
}

/// Complete owner-authoritative lifecycle of one physical materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LoadStage {
    Reserved,
    ReadSubmitted,
    HostReady,
    UploadSubmitted,
    Installing,
    Resident,
    Failed,
    Stale,
    Draining,
    Retired,
}

impl LoadStage {
    pub const ALL: [Self; 10] = [
        Self::Reserved,
        Self::ReadSubmitted,
        Self::HostReady,
        Self::UploadSubmitted,
        Self::Installing,
        Self::Resident,
        Self::Failed,
        Self::Stale,
        Self::Draining,
        Self::Retired,
    ];

    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Reserved, Self::ReadSubmitted | Self::Retired)
                | (
                    Self::ReadSubmitted,
                    Self::HostReady | Self::Failed | Self::Stale | Self::Draining
                )
                | (
                    Self::HostReady,
                    Self::UploadSubmitted | Self::Failed | Self::Stale
                )
                | (
                    Self::UploadSubmitted,
                    Self::Installing | Self::Failed | Self::Stale | Self::Draining
                )
                | (
                    Self::Installing,
                    Self::Resident | Self::Failed | Self::Stale | Self::Draining
                )
                | (Self::Resident, Self::Retired)
                | (Self::Failed | Self::Stale, Self::Draining | Self::Retired)
                | (Self::Draining, Self::Retired)
        )
    }

    pub fn validate_transition(self, next: Self) -> IoProtocolResult<()> {
        if self.can_transition_to(next) {
            Ok(())
        } else {
            Err(IoProtocolError::InvalidLoadTransition {
                from: self,
                to: next,
            })
        }
    }

    pub const fn is_submitted_completion_stage(self) -> bool {
        matches!(
            self,
            Self::ReadSubmitted | Self::UploadSubmitted | Self::Installing
        )
    }
}

/// Typed physical failure retained by a failed load and its waiters.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FailureReason {
    StorageUnavailable,
    ReadRejected,
    ShortRead { expected: u64, actual: u64 },
    DeviceUnavailable,
    UploadRejected,
    InstallationRejected,
    ProviderFailure { backend: BackendId, code: i64 },
    ProtocolViolation(String),
}

/// Why logical demand was detached without pretending submitted work disappeared.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CancellationReason {
    ExternalRequest,
    DeadlineExceeded,
    LastWaiterDetached,
    Superseded,
    OwnerShutdown,
}

/// Why a completion or retained artifact is forbidden from publication.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StaleReason {
    SourceGenerationMismatch {
        expected: SourceGeneration,
        observed: SourceGeneration,
    },
    DestinationGenerationMismatch {
        expected: DestinationGeneration,
        observed: DestinationGeneration,
    },
    SourceIdentityChanged,
    DestinationReused,
    SupersededOperation,
}

/// Why the owner consumed the exactly-once retirement authority.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RetirementReason {
    ResidentOwnershipTransferred,
    Failed(FailureReason),
    Cancelled(CancellationReason),
    Stale(StaleReason),
    Drained,
    OrphanCompletion,
    OwnerShutdown,
}

/// Generation tuple carried by every physical completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompletionGeneration {
    pub source: SourceGeneration,
    pub destination: DestinationGeneration,
}

impl CompletionGeneration {
    pub const fn new(source: SourceGeneration, destination: DestinationGeneration) -> Self {
        Self {
            source,
            destination,
        }
    }

    pub const fn for_key(key: LoadKey) -> Self {
        Self::new(key.source_generation(), key.destination_generation())
    }
}

/// Monotonic owner/provider timestamp in nanoseconds from an injected clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CompletionTimestamp(u64);

impl CompletionTimestamp {
    pub const fn from_nanos(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_nanos(self) -> u64 {
        self.0
    }
}

/// Physical result reported by a provider. Readiness is represented only by success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionOutcome {
    Succeeded,
    Failed(FailureReason),
    Cancelled(CancellationReason),
    Stale(StaleReason),
}

/// Immutable completion returned to the owner.
///
/// `operation` and the full `key` carry identity. `generation` is repeated so a
/// stale provider descriptor is detected independently; a global wake epoch is
/// deliberately not part of this event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionEvent {
    pub operation: OperationId,
    pub key: LoadKey,
    pub stage: LoadStage,
    pub outcome: CompletionOutcome,
    pub bytes: u64,
    pub generation: CompletionGeneration,
    pub timestamp: CompletionTimestamp,
}

impl CompletionEvent {
    pub const fn new(
        operation: OperationId,
        key: LoadKey,
        stage: LoadStage,
        outcome: CompletionOutcome,
        bytes: u64,
        generation: CompletionGeneration,
        timestamp: CompletionTimestamp,
    ) -> Self {
        Self {
            operation,
            key,
            stage,
            outcome,
            bytes,
            generation,
            timestamp,
        }
    }

    /// Validates this untrusted provider return against owner-authoritative state.
    pub fn validate(&self, expected: &CompletionExpectation) -> IoProtocolResult<()> {
        expected.key.validate()?;
        self.key.validate()?;
        if !expected.stage.is_submitted_completion_stage() {
            return Err(IoProtocolError::InvalidCompletionStage {
                stage: expected.stage,
            });
        }
        if self.operation != expected.operation {
            return Err(IoProtocolError::CompletionMismatch { field: "operation" });
        }
        if self.key != expected.key {
            return Err(IoProtocolError::CompletionMismatch { field: "key" });
        }
        if self.stage != expected.stage {
            return Err(IoProtocolError::CompletionMismatch { field: "stage" });
        }
        let expected_generation = CompletionGeneration::for_key(expected.key);
        if self.generation != expected_generation {
            return Err(IoProtocolError::CompletionGenerationMismatch {
                expected: expected_generation,
                observed: self.generation,
            });
        }
        match self.outcome {
            CompletionOutcome::Succeeded if self.bytes != expected.bytes => {
                Err(IoProtocolError::IncompleteSuccessfulCompletion {
                    expected: expected.bytes,
                    actual: self.bytes,
                })
            }
            CompletionOutcome::Failed(_)
            | CompletionOutcome::Cancelled(_)
            | CompletionOutcome::Stale(_)
                if self.bytes > expected.bytes =>
            {
                Err(IoProtocolError::CompletionByteOverflow {
                    expected: expected.bytes,
                    actual: self.bytes,
                })
            }
            _ => Ok(()),
        }
    }

    /// Owner transition implied by a validated completion outcome.
    pub fn next_stage(&self) -> IoProtocolResult<LoadStage> {
        let next = match (&self.outcome, self.stage) {
            (CompletionOutcome::Succeeded, LoadStage::ReadSubmitted) => LoadStage::HostReady,
            (CompletionOutcome::Succeeded, LoadStage::UploadSubmitted) => LoadStage::Installing,
            (CompletionOutcome::Succeeded, LoadStage::Installing) => LoadStage::Resident,
            (CompletionOutcome::Failed(_), _) => LoadStage::Failed,
            (CompletionOutcome::Cancelled(_), _) => LoadStage::Draining,
            (CompletionOutcome::Stale(_), _) => LoadStage::Stale,
            _ => {
                return Err(IoProtocolError::InvalidCompletionStage { stage: self.stage });
            }
        };
        self.stage.validate_transition(next)?;
        Ok(next)
    }
}

/// Owner-side command identity and exact byte expectation for one completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletionExpectation {
    pub operation: OperationId,
    pub key: LoadKey,
    pub stage: LoadStage,
    pub bytes: u64,
}

impl CompletionExpectation {
    pub fn new(
        operation: OperationId,
        key: LoadKey,
        stage: LoadStage,
        bytes: u64,
    ) -> IoProtocolResult<Self> {
        key.validate()?;
        if operation.is_zero() {
            return Err(IoProtocolError::CompletionMismatch { field: "operation" });
        }
        if !stage.is_submitted_completion_stage() {
            return Err(IoProtocolError::InvalidCompletionStage { stage });
        }
        Ok(Self {
            operation,
            key,
            stage,
            bytes,
        })
    }
}

/// Monotonic record produced by consuming one [`RetirementToken`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetirementRecord {
    pub operation: OperationId,
    pub key: LoadKey,
    pub reason: RetirementReason,
    pub timestamp: CompletionTimestamp,
}

/// Exactly-once owner authority for recording retirement.
///
/// This type intentionally does not implement `Clone`. `consume` also rejects a
/// second call, protecting owner code that retains the token in an `Option`-like
/// state machine.
#[derive(Debug)]
#[must_use = "retirement authority must be consumed or explicitly retained"]
pub struct RetirementToken {
    authority: Option<(OperationId, LoadKey)>,
}

impl RetirementToken {
    pub fn new(operation: OperationId, key: LoadKey) -> IoProtocolResult<Self> {
        if operation.is_zero() {
            return Err(IoProtocolError::CompletionMismatch { field: "operation" });
        }
        key.validate()?;
        Ok(Self {
            authority: Some((operation, key)),
        })
    }

    pub const fn is_consumed(&self) -> bool {
        self.authority.is_none()
    }

    pub fn consume(
        &mut self,
        reason: RetirementReason,
        timestamp: CompletionTimestamp,
    ) -> IoProtocolResult<RetirementRecord> {
        let (operation, key) = self
            .authority
            .take()
            .ok_or(IoProtocolError::RetirementAlreadyConsumed)?;
        Ok(RetirementRecord {
            operation,
            key,
            reason,
            timestamp,
        })
    }
}

/// Validated stable host address. It is an opaque address value, not dereference authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct StableAddress(usize);

impl StableAddress {
    pub const fn get(self) -> usize {
        self.0
    }
}

/// POD-like descriptor for one aligned range in a registered pinned slab.
///
/// The owning [`RegisteredPinnedAlignedSlabLease`] must outlive every provider use
/// of this descriptor. Providers may not retain it beyond the named upload fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisteredPinnedAlignedSlabLeaseDescriptor {
    operation: OperationId,
    slab: SlabId,
    registration: RegistrationId,
    address: StableAddress,
    len: u64,
    alignment: usize,
    source_generation: SourceGeneration,
    destination_generation: DestinationGeneration,
}

impl RegisteredPinnedAlignedSlabLeaseDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        operation: OperationId,
        slab: SlabId,
        registration: RegistrationId,
        base_address: usize,
        allocation_len: u64,
        offset: u64,
        len: u64,
        alignment: usize,
        source_generation: SourceGeneration,
        destination_generation: DestinationGeneration,
    ) -> IoProtocolResult<Self> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(IoProtocolError::InvalidAlignment { alignment });
        }
        if base_address == 0 {
            return Err(IoProtocolError::NullSlabAddress);
        }
        let alignment_u64 = u64::try_from(alignment)
            .map_err(|_| IoProtocolError::ArithmeticOverflow("slab alignment conversion"))?;
        if !base_address.is_multiple_of(alignment) {
            return Err(IoProtocolError::MisalignedSlabField {
                field: "base address",
                value: u64::try_from(base_address).unwrap_or(u64::MAX),
                alignment,
            });
        }
        for (field, value) in [("offset", offset), ("length", len)] {
            if value == 0 && field == "length" {
                return Err(IoProtocolError::InvalidLoadKey(
                    "slab lease length must be non-zero",
                ));
            }
            if !value.is_multiple_of(alignment_u64) {
                return Err(IoProtocolError::MisalignedSlabField {
                    field,
                    value,
                    alignment,
                });
            }
        }
        let end = offset
            .checked_add(len)
            .ok_or(IoProtocolError::ArithmeticOverflow("slab range end"))?;
        if end > allocation_len {
            return Err(IoProtocolError::SlabRangeOutOfBounds {
                offset,
                end,
                capacity: allocation_len,
            });
        }
        let allocation_len = usize::try_from(allocation_len)
            .map_err(|_| IoProtocolError::ArithmeticOverflow("slab allocation length"))?;
        base_address
            .checked_add(allocation_len)
            .ok_or(IoProtocolError::ArithmeticOverflow(
                "registered slab address range",
            ))?;
        let offset = usize::try_from(offset)
            .map_err(|_| IoProtocolError::ArithmeticOverflow("slab range offset"))?;
        let len = usize::try_from(len)
            .map_err(|_| IoProtocolError::ArithmeticOverflow("slab range length"))?;
        let address = base_address
            .checked_add(offset)
            .ok_or(IoProtocolError::ArithmeticOverflow("slab lease address"))?;
        address
            .checked_add(len)
            .ok_or(IoProtocolError::ArithmeticOverflow(
                "slab lease address range",
            ))?;

        Ok(Self {
            operation,
            slab,
            registration,
            address: StableAddress(address),
            len: u64::try_from(len)
                .map_err(|_| IoProtocolError::ArithmeticOverflow("slab length round trip"))?,
            alignment,
            source_generation,
            destination_generation,
        })
    }

    pub const fn operation(self) -> OperationId {
        self.operation
    }

    pub const fn slab(self) -> SlabId {
        self.slab
    }

    pub const fn registration(self) -> RegistrationId {
        self.registration
    }

    pub const fn address(self) -> StableAddress {
        self.address
    }

    pub const fn len(self) -> u64 {
        self.len
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    pub const fn alignment(self) -> usize {
        self.alignment
    }

    pub const fn source_generation(self) -> SourceGeneration {
        self.source_generation
    }

    pub const fn destination_generation(self) -> DestinationGeneration {
        self.destination_generation
    }
}

/// Fence that keeps the slab address live after upload submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UploadFenceContract {
    pub operation: OperationId,
    pub fence: FenceId,
    pub destination_generation: DestinationGeneration,
}

impl UploadFenceContract {
    pub const fn new(
        operation: OperationId,
        fence: FenceId,
        destination_generation: DestinationGeneration,
    ) -> Self {
        Self {
            operation,
            fence,
            destination_generation,
        }
    }

    pub const fn observation(self, timestamp: CompletionTimestamp) -> UploadFenceObservation {
        UploadFenceObservation {
            operation: self.operation,
            fence: self.fence,
            destination_generation: self.destination_generation,
            timestamp,
        }
    }

    pub fn validate_observation(self, observed: UploadFenceObservation) -> IoProtocolResult<()> {
        if observed.operation != self.operation {
            return Err(IoProtocolError::UploadFenceMismatch { field: "operation" });
        }
        if observed.fence != self.fence {
            return Err(IoProtocolError::UploadFenceMismatch { field: "fence" });
        }
        if observed.destination_generation != self.destination_generation {
            return Err(IoProtocolError::UploadFenceMismatch {
                field: "destination generation",
            });
        }
        Ok(())
    }
}

/// Owner-visible observation proving upload references are no longer in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UploadFenceObservation {
    pub operation: OperationId,
    pub fence: FenceId,
    pub destination_generation: DestinationGeneration,
    pub timestamp: CompletionTimestamp,
}

/// Custody state of an owning registered pinned slab lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlabLeaseState {
    Reserved,
    ReadSubmitted,
    HostReady,
    UploadSubmitted,
    Releasable,
    Retired,
}

/// Non-cloneable owner lease enforcing stable address through the upload fence.
#[derive(Debug)]
#[must_use = "a registered pinned slab lease must remain live through its upload fence"]
pub struct RegisteredPinnedAlignedSlabLease {
    descriptor: RegisteredPinnedAlignedSlabLeaseDescriptor,
    state: SlabLeaseState,
    upload_fence: Option<UploadFenceContract>,
}

impl RegisteredPinnedAlignedSlabLease {
    pub const fn new(descriptor: RegisteredPinnedAlignedSlabLeaseDescriptor) -> Self {
        Self {
            descriptor,
            state: SlabLeaseState::Reserved,
            upload_fence: None,
        }
    }

    pub const fn descriptor(&self) -> RegisteredPinnedAlignedSlabLeaseDescriptor {
        self.descriptor
    }

    pub const fn state(&self) -> SlabLeaseState {
        self.state
    }

    pub fn mark_read_submitted(&mut self) -> IoProtocolResult<()> {
        self.transition(SlabLeaseState::Reserved, SlabLeaseState::ReadSubmitted)
    }

    /// Accepts `HostReady` only for a complete exact read into the same address.
    pub fn mark_host_ready(&mut self, completed_bytes: u64) -> IoProtocolResult<()> {
        if completed_bytes != self.descriptor.len() {
            return Err(IoProtocolError::IncompleteSuccessfulCompletion {
                expected: self.descriptor.len(),
                actual: completed_bytes,
            });
        }
        self.transition(SlabLeaseState::ReadSubmitted, SlabLeaseState::HostReady)
    }

    /// Records a failed/cancelled read CQE, after which the slab is reusable.
    pub fn mark_read_returned_without_artifact(&mut self) -> IoProtocolResult<()> {
        self.transition(SlabLeaseState::ReadSubmitted, SlabLeaseState::Releasable)
    }

    /// Transfers stable-address custody into an upload command and fence.
    pub fn mark_upload_submitted(&mut self, contract: UploadFenceContract) -> IoProtocolResult<()> {
        if contract.operation != self.descriptor.operation() {
            return Err(IoProtocolError::UploadFenceMismatch { field: "operation" });
        }
        if contract.destination_generation != self.descriptor.destination_generation() {
            return Err(IoProtocolError::UploadFenceMismatch {
                field: "destination generation",
            });
        }
        if contract.fence.is_zero() {
            return Err(IoProtocolError::InvalidFenceContract);
        }
        self.transition(SlabLeaseState::HostReady, SlabLeaseState::UploadSubmitted)?;
        self.upload_fence = Some(contract);
        Ok(())
    }

    /// Releases provider custody only after the exact retained fence is observed.
    pub fn mark_upload_fence(&mut self, observed: UploadFenceObservation) -> IoProtocolResult<()> {
        if self.state != SlabLeaseState::UploadSubmitted {
            return Err(IoProtocolError::InvalidSlabTransition {
                from: self.state,
                to: SlabLeaseState::Releasable,
            });
        }
        let contract = self
            .upload_fence
            .ok_or(IoProtocolError::InvalidFenceContract)?;
        contract.validate_observation(observed)?;
        self.upload_fence = None;
        self.state = SlabLeaseState::Releasable;
        Ok(())
    }

    /// Retires an unsubmitted/discarded lease or one whose physical custody ended.
    pub fn retire(&mut self) -> IoProtocolResult<()> {
        match self.state {
            SlabLeaseState::Reserved | SlabLeaseState::HostReady | SlabLeaseState::Releasable => {
                self.state = SlabLeaseState::Retired;
                Ok(())
            }
            state => Err(IoProtocolError::InvalidSlabTransition {
                from: state,
                to: SlabLeaseState::Retired,
            }),
        }
    }

    fn transition(
        &mut self,
        expected: SlabLeaseState,
        next: SlabLeaseState,
    ) -> IoProtocolResult<()> {
        if self.state != expected {
            return Err(IoProtocolError::InvalidSlabTransition {
                from: self.state,
                to: next,
            });
        }
        self.state = next;
        Ok(())
    }
}

/// Raw owner-observed residency mapping, not yet generation-validated for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidencyBinding {
    pub model: ModelInstanceId,
    pub layer: LayerId,
    pub expert: ExpertId,
    pub backend: BackendId,
    pub device: DeviceId,
    pub slot: DestinationSlotId,
    pub generation: DestinationGeneration,
}

impl ResidencyBinding {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        model: ModelInstanceId,
        layer: LayerId,
        expert: ExpertId,
        backend: BackendId,
        device: DeviceId,
        slot: DestinationSlotId,
        generation: DestinationGeneration,
    ) -> Self {
        Self {
            model,
            layer,
            expert,
            backend,
            device,
            slot,
            generation,
        }
    }
}

/// Residency mapping proven to match one complete load identity and generation.
///
/// It intentionally does not implement `Clone`; dispatch authority is moved into
/// one [`ExpertLeaseSet`].
#[derive(Debug, PartialEq, Eq)]
pub struct ValidatedResidencyBinding {
    key: LoadKey,
    binding: ResidencyBinding,
}

impl ValidatedResidencyBinding {
    pub fn new(key: LoadKey, binding: ResidencyBinding) -> IoProtocolResult<Self> {
        key.validate()?;
        for (field, matches) in [
            ("model", binding.model == key.model()),
            ("layer", binding.layer == key.layer()),
            ("expert", binding.expert == key.expert()),
            ("backend", binding.backend == key.backend()),
            ("device", binding.device == key.device()),
            (
                "destination generation",
                binding.generation == key.destination_generation(),
            ),
        ] {
            if !matches {
                return Err(IoProtocolError::ResidencyMismatch { field });
            }
        }
        Ok(Self { key, binding })
    }

    pub const fn key(&self) -> LoadKey {
        self.key
    }

    pub const fn binding(&self) -> ResidencyBinding {
        self.binding
    }
}

/// Architecture-document name for a validated residency binding.
pub type GenerationValidatedBinding = ValidatedResidencyBinding;

/// Completion fence that keeps a whole expert lease set immutable after dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DispatchFenceContract {
    pub operation: OperationId,
    pub fence: FenceId,
    pub backend: BackendId,
    pub device: DeviceId,
}

impl DispatchFenceContract {
    pub const fn new(
        operation: OperationId,
        fence: FenceId,
        backend: BackendId,
        device: DeviceId,
    ) -> Self {
        Self {
            operation,
            fence,
            backend,
            device,
        }
    }

    pub fn validate(self) -> IoProtocolResult<()> {
        if self.operation.is_zero() || self.fence.is_zero() {
            Err(IoProtocolError::InvalidFenceContract)
        } else {
            Ok(())
        }
    }
}

/// Atomic dispatch authority for the complete exact expert set of one segment.
///
/// The set is non-cloneable and canonicalized by `LoadKey`. Construction fails on
/// partial, duplicate, extra, stale-generation, backend, or device bindings.
#[derive(Debug)]
#[must_use = "expert leases must remain live through their dispatch completion fence"]
pub struct ExpertLeaseSet {
    bindings: Box<[ValidatedResidencyBinding]>,
    mapping_epoch: MappingEpoch,
    completion_contract: DispatchFenceContract,
}

impl ExpertLeaseSet {
    pub fn new(
        required: impl IntoIterator<Item = LoadKey>,
        bindings: impl IntoIterator<Item = ValidatedResidencyBinding>,
        mapping_epoch: MappingEpoch,
        completion_contract: DispatchFenceContract,
    ) -> IoProtocolResult<Self> {
        if mapping_epoch.is_zero() {
            return Err(IoProtocolError::InvalidMappingEpoch);
        }
        completion_contract.validate()?;

        let mut required: Vec<_> = required.into_iter().collect();
        for key in &required {
            key.validate()?;
        }
        required.sort_unstable();
        required.dedup();

        let mut bindings: Vec<_> = bindings.into_iter().collect();
        bindings.sort_unstable_by_key(ValidatedResidencyBinding::key);
        if let Some(duplicate) = bindings
            .windows(2)
            .find(|window| window[0].key() == window[1].key())
            .map(|window| window[0].key())
        {
            return Err(IoProtocolError::DuplicateResidencyBinding(Box::new(
                duplicate,
            )));
        }

        let bound: BTreeSet<_> = bindings
            .iter()
            .map(ValidatedResidencyBinding::key)
            .collect();
        let required_set: BTreeSet<_> = required.iter().copied().collect();
        if let Some(missing) = required_set.difference(&bound).next() {
            return Err(IoProtocolError::MissingResidencyBinding(Box::new(*missing)));
        }
        if let Some(extra) = bound.difference(&required_set).next() {
            return Err(IoProtocolError::UnexpectedResidencyBinding(Box::new(
                *extra,
            )));
        }

        for binding in &bindings {
            let raw = binding.binding();
            if raw.backend != completion_contract.backend {
                return Err(IoProtocolError::ResidencyMismatch {
                    field: "completion fence backend",
                });
            }
            if raw.device != completion_contract.device {
                return Err(IoProtocolError::ResidencyMismatch {
                    field: "completion fence device",
                });
            }
        }

        Ok(Self {
            bindings: bindings.into_boxed_slice(),
            mapping_epoch,
            completion_contract,
        })
    }

    pub fn for_dependencies(
        dependencies: &DependencySet,
        bindings: impl IntoIterator<Item = ValidatedResidencyBinding>,
        mapping_epoch: MappingEpoch,
        completion_contract: DispatchFenceContract,
    ) -> IoProtocolResult<Self> {
        let mut required = Vec::with_capacity(dependencies.len());
        for dependency in dependencies.iter().copied() {
            match dependency {
                LogicalDependency::ExpertResident(key) => required.push(key),
                other => {
                    return Err(IoProtocolError::NonResidencyDependency(Box::new(other)));
                }
            }
        }
        Self::new(required, bindings, mapping_epoch, completion_contract)
    }

    pub fn bindings(&self) -> &[ValidatedResidencyBinding] {
        &self.bindings
    }

    pub const fn mapping_epoch(&self) -> MappingEpoch {
        self.mapping_epoch
    }

    pub const fn completion_contract(&self) -> DispatchFenceContract {
        self.completion_contract
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    pub fn binding_for(&self, key: LoadKey) -> Option<&ValidatedResidencyBinding> {
        self.bindings
            .binary_search_by_key(&key, ValidatedResidencyBinding::key)
            .ok()
            .map(|index| &self.bindings[index])
    }
}

/// Model-neutral external poll result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelProgress<T> {
    Complete(T),
    Waiting(DependencySet),
}

impl<T> ModelProgress<T> {
    pub const fn complete(value: T) -> Self {
        Self::Complete(value)
    }

    pub fn waiting(dependencies: DependencySet) -> IoProtocolResult<Self> {
        dependencies.validate()?;
        Ok(Self::Waiting(dependencies))
    }

    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> ModelProgress<U> {
        match self {
            Self::Complete(value) => ModelProgress::Complete(map(value)),
            Self::Waiting(dependencies) => ModelProgress::Waiting(dependencies),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[allow(clippy::too_many_arguments)]
    fn key_with(
        model: u64,
        source: u8,
        content: u8,
        layer: u32,
        expert: u32,
        format: u32,
        backend: u64,
        device: u64,
        source_generation: u64,
        destination_generation: u64,
    ) -> LoadKey {
        LoadKey::new(
            ModelInstanceId::new(model),
            SourceIdentityHash::new(hash(source)),
            ContentHash::new(hash(content)),
            LayerId::new(layer),
            ExpertId::new(expert),
            ArtifactFormat::new(format),
            BackendId::new(backend),
            DeviceId::new(device),
            SourceGeneration::new(source_generation),
            DestinationGeneration::new(destination_generation),
        )
        .unwrap()
    }

    fn key(expert: u32) -> LoadKey {
        key_with(1, 2, 3, 4, expert, 5, 6, 7, 8, 9)
    }

    fn binding_for(key: LoadKey, slot: u32) -> ValidatedResidencyBinding {
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

    fn dispatch_fence() -> DispatchFenceContract {
        DispatchFenceContract::new(
            OperationId::new(80),
            FenceId::new(81),
            BackendId::new(6),
            DeviceId::new(7),
        )
    }

    #[test]
    fn typed_ids_remain_distinct_and_waiter_identity_contains_all_generations() {
        let operation = OperationId::new(7);
        let continuation = ContinuationId::new(7);
        assert_eq!(operation.get(), continuation.get());

        let transaction = ExecutionTransactionId::new(1).unwrap();
        let first = WaiterId::new(
            transaction,
            RequestGeneration::new(2),
            DependencySetEpoch::new(3),
            ContinuationId::new(4),
        )
        .unwrap();
        let second = WaiterId::new(
            transaction,
            RequestGeneration::new(2),
            DependencySetEpoch::new(4),
            ContinuationId::new(4),
        )
        .unwrap();
        assert_ne!(first, second);
        assert_eq!(first.transaction(), transaction);
        assert!(
            WaiterId::new(
                transaction,
                RequestGeneration::new(0),
                DependencySetEpoch::new(3),
                ContinuationId::new(4),
            )
            .is_err()
        );
    }

    #[test]
    fn every_load_key_identity_dimension_changes_equality_hash_and_order_identity() {
        let keys = [
            key_with(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            key_with(11, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            key_with(1, 12, 3, 4, 5, 6, 7, 8, 9, 10),
            key_with(1, 2, 13, 4, 5, 6, 7, 8, 9, 10),
            key_with(1, 2, 3, 14, 5, 6, 7, 8, 9, 10),
            key_with(1, 2, 3, 4, 15, 6, 7, 8, 9, 10),
            key_with(1, 2, 3, 4, 5, 16, 7, 8, 9, 10),
            key_with(1, 2, 3, 4, 5, 6, 17, 8, 9, 10),
            key_with(1, 2, 3, 4, 5, 6, 7, 18, 9, 10),
            key_with(1, 2, 3, 4, 5, 6, 7, 8, 19, 10),
            key_with(1, 2, 3, 4, 5, 6, 7, 8, 9, 20),
        ];
        let ordered: BTreeSet<_> = keys.into_iter().collect();
        assert_eq!(ordered.len(), keys.len());
    }

    #[test]
    fn load_key_rejects_sentinel_identity_and_generation_values() {
        assert!(
            LoadKey::new(
                ModelInstanceId::new(0),
                SourceIdentityHash::new(hash(2)),
                ContentHash::new(hash(3)),
                LayerId::new(0),
                ExpertId::new(0),
                ArtifactFormat::new(1),
                BackendId::new(0),
                DeviceId::new(0),
                SourceGeneration::new(1),
                DestinationGeneration::new(1),
            )
            .is_err()
        );
        assert!(
            LoadKey::new(
                ModelInstanceId::new(1),
                SourceIdentityHash::new([0; 32]),
                ContentHash::new(hash(3)),
                LayerId::new(0),
                ExpertId::new(0),
                ArtifactFormat::new(1),
                BackendId::new(0),
                DeviceId::new(0),
                SourceGeneration::new(1),
                DestinationGeneration::new(1),
            )
            .is_err()
        );
    }

    #[test]
    fn dependency_set_sorts_deduplicates_and_validates_canonical_input() {
        let first = LogicalDependency::expert_resident(key(1)).unwrap();
        let second = LogicalDependency::expert_resident(key(2)).unwrap();
        let retired = LogicalDependency::operation_retired(OperationId::new(3)).unwrap();
        let set = DependencySet::new([second, first, second, retired]).unwrap();

        assert_eq!(set.len(), 3);
        assert!(set.contains(&first));
        assert!(set.contains(&second));
        assert!(set.contains(&retired));
        assert!(
            set.as_slice()
                .windows(2)
                .all(|window| window[0] < window[1])
        );
        assert!(matches!(
            DependencySet::new([]),
            Err(IoProtocolError::EmptyDependencySet)
        ));
        assert!(matches!(
            DependencySet::from_canonical(vec![second, first].into_boxed_slice()),
            Err(IoProtocolError::NonCanonicalDependencySet)
        ));
        assert!(matches!(
            DependencySet::from_canonical(vec![first, first].into_boxed_slice()),
            Err(IoProtocolError::NonCanonicalDependencySet)
        ));
    }

    #[test]
    fn load_stage_transition_matrix_is_exhaustive_for_every_legal_and_illegal_edge() {
        let legal = BTreeSet::from([
            (LoadStage::Reserved, LoadStage::ReadSubmitted),
            (LoadStage::Reserved, LoadStage::Retired),
            (LoadStage::ReadSubmitted, LoadStage::HostReady),
            (LoadStage::ReadSubmitted, LoadStage::Failed),
            (LoadStage::ReadSubmitted, LoadStage::Stale),
            (LoadStage::ReadSubmitted, LoadStage::Draining),
            (LoadStage::HostReady, LoadStage::UploadSubmitted),
            (LoadStage::HostReady, LoadStage::Failed),
            (LoadStage::HostReady, LoadStage::Stale),
            (LoadStage::UploadSubmitted, LoadStage::Installing),
            (LoadStage::UploadSubmitted, LoadStage::Failed),
            (LoadStage::UploadSubmitted, LoadStage::Stale),
            (LoadStage::UploadSubmitted, LoadStage::Draining),
            (LoadStage::Installing, LoadStage::Resident),
            (LoadStage::Installing, LoadStage::Failed),
            (LoadStage::Installing, LoadStage::Stale),
            (LoadStage::Installing, LoadStage::Draining),
            (LoadStage::Resident, LoadStage::Retired),
            (LoadStage::Failed, LoadStage::Draining),
            (LoadStage::Failed, LoadStage::Retired),
            (LoadStage::Stale, LoadStage::Draining),
            (LoadStage::Stale, LoadStage::Retired),
            (LoadStage::Draining, LoadStage::Retired),
        ]);

        for from in LoadStage::ALL {
            for to in LoadStage::ALL {
                let expected = legal.contains(&(from, to));
                assert_eq!(
                    from.can_transition_to(to),
                    expected,
                    "unexpected transition classification for {from:?} -> {to:?}"
                );
                assert_eq!(
                    from.validate_transition(to).is_ok(),
                    expected,
                    "unexpected transition validation for {from:?} -> {to:?}"
                );
            }
        }
    }

    #[test]
    fn completion_validation_checks_identity_stage_bytes_generation_and_outcome() {
        let load_key = key(1);
        let expectation = CompletionExpectation::new(
            OperationId::new(20),
            load_key,
            LoadStage::ReadSubmitted,
            4096,
        )
        .unwrap();
        let valid = CompletionEvent::new(
            expectation.operation,
            load_key,
            expectation.stage,
            CompletionOutcome::Succeeded,
            4096,
            CompletionGeneration::for_key(load_key),
            CompletionTimestamp::from_nanos(100),
        );
        valid.validate(&expectation).unwrap();
        assert_eq!(valid.next_stage().unwrap(), LoadStage::HostReady);

        let mut wrong_operation = valid.clone();
        wrong_operation.operation = OperationId::new(21);
        assert!(matches!(
            wrong_operation.validate(&expectation),
            Err(IoProtocolError::CompletionMismatch { field: "operation" })
        ));

        let mut wrong_key = valid.clone();
        wrong_key.key = key(2);
        assert!(matches!(
            wrong_key.validate(&expectation),
            Err(IoProtocolError::CompletionMismatch { field: "key" })
        ));

        let mut wrong_stage = valid.clone();
        wrong_stage.stage = LoadStage::UploadSubmitted;
        assert!(matches!(
            wrong_stage.validate(&expectation),
            Err(IoProtocolError::CompletionMismatch { field: "stage" })
        ));

        let mut wrong_generation = valid.clone();
        wrong_generation.generation.destination = DestinationGeneration::new(10);
        assert!(matches!(
            wrong_generation.validate(&expectation),
            Err(IoProtocolError::CompletionGenerationMismatch { .. })
        ));

        let mut short_success = valid.clone();
        short_success.bytes = 2048;
        assert!(matches!(
            short_success.validate(&expectation),
            Err(IoProtocolError::IncompleteSuccessfulCompletion { .. })
        ));

        let partial_failure = CompletionEvent {
            outcome: CompletionOutcome::Failed(FailureReason::ShortRead {
                expected: 4096,
                actual: 2048,
            }),
            bytes: 2048,
            ..valid.clone()
        };
        partial_failure.validate(&expectation).unwrap();
        assert_eq!(partial_failure.next_stage().unwrap(), LoadStage::Failed);

        let oversized_failure = CompletionEvent {
            bytes: 4097,
            ..partial_failure
        };
        assert!(matches!(
            oversized_failure.validate(&expectation),
            Err(IoProtocolError::CompletionByteOverflow { .. })
        ));

        assert!(
            CompletionExpectation::new(OperationId::new(20), load_key, LoadStage::HostReady, 4096,)
                .is_err()
        );
    }

    #[test]
    fn stale_and_cancelled_completions_never_report_readiness() {
        let key = key(1);
        for (outcome, expected) in [
            (
                CompletionOutcome::Cancelled(CancellationReason::LastWaiterDetached),
                LoadStage::Draining,
            ),
            (
                CompletionOutcome::Stale(StaleReason::DestinationGenerationMismatch {
                    expected: DestinationGeneration::new(9),
                    observed: DestinationGeneration::new(10),
                }),
                LoadStage::Stale,
            ),
        ] {
            let event = CompletionEvent::new(
                OperationId::new(1),
                key,
                LoadStage::UploadSubmitted,
                outcome,
                4096,
                CompletionGeneration::for_key(key),
                CompletionTimestamp::from_nanos(1),
            );
            assert_eq!(event.next_stage().unwrap(), expected);
            assert_ne!(expected, LoadStage::Resident);
        }
    }

    #[test]
    fn retirement_token_consumes_authority_exactly_once_on_typed_reasons() {
        let mut token = RetirementToken::new(OperationId::new(1), key(1)).unwrap();
        let record = token
            .consume(
                RetirementReason::Cancelled(CancellationReason::ExternalRequest),
                CompletionTimestamp::from_nanos(99),
            )
            .unwrap();
        assert_eq!(record.operation, OperationId::new(1));
        assert!(token.is_consumed());
        assert!(matches!(
            token.consume(
                RetirementReason::Drained,
                CompletionTimestamp::from_nanos(100)
            ),
            Err(IoProtocolError::RetirementAlreadyConsumed)
        ));
    }

    fn slab_descriptor(
        base_address: usize,
        allocation_len: u64,
        offset: u64,
        len: u64,
        alignment: usize,
    ) -> IoProtocolResult<RegisteredPinnedAlignedSlabLeaseDescriptor> {
        RegisteredPinnedAlignedSlabLeaseDescriptor::new(
            OperationId::new(1),
            SlabId::new(2),
            RegistrationId::new(3),
            base_address,
            allocation_len,
            offset,
            len,
            alignment,
            SourceGeneration::new(4),
            DestinationGeneration::new(5),
        )
    }

    #[test]
    fn slab_descriptor_rejects_alignment_bounds_and_arithmetic_overflow() {
        assert!(matches!(
            slab_descriptor(0x1000, 0x4000, 0, 0x1000, 0),
            Err(IoProtocolError::InvalidAlignment { .. })
        ));
        assert!(matches!(
            slab_descriptor(0x1000, 0x4000, 0, 0x1000, 3),
            Err(IoProtocolError::InvalidAlignment { .. })
        ));
        assert!(matches!(
            slab_descriptor(0x1001, 0x4000, 0, 0x1000, 0x1000),
            Err(IoProtocolError::MisalignedSlabField {
                field: "base address",
                ..
            })
        ));
        assert!(matches!(
            slab_descriptor(0x1000, 0x4000, 1, 0x1000, 0x1000),
            Err(IoProtocolError::MisalignedSlabField {
                field: "offset",
                ..
            })
        ));
        assert!(matches!(
            slab_descriptor(0x1000, 0x4000, 0, 1, 0x1000),
            Err(IoProtocolError::MisalignedSlabField {
                field: "length",
                ..
            })
        ));
        assert!(matches!(
            slab_descriptor(0x1000, 0x1000, 0x1000, 0x1000, 0x1000),
            Err(IoProtocolError::SlabRangeOutOfBounds { .. })
        ));
        assert!(matches!(
            slab_descriptor(0x1000, u64::MAX, u64::MAX - 0xfff, 0x1000, 0x1000),
            Err(IoProtocolError::ArithmeticOverflow("slab range end"))
        ));

        let near_end = usize::MAX & !0xfff;
        assert!(matches!(
            slab_descriptor(near_end, 0x2000, 0, 0x1000, 0x1000),
            Err(IoProtocolError::ArithmeticOverflow(
                "registered slab address range"
            ))
        ));
    }

    #[test]
    fn slab_lease_keeps_a_stable_address_until_exact_upload_fence() {
        let descriptor = slab_descriptor(0x4000, 0x8000, 0x1000, 0x2000, 0x1000).unwrap();
        assert_eq!(descriptor.address().get(), 0x5000);
        let mut lease = RegisteredPinnedAlignedSlabLease::new(descriptor);
        let stable = lease.descriptor().address();

        lease.mark_read_submitted().unwrap();
        assert_eq!(lease.descriptor().address(), stable);
        assert!(lease.retire().is_err());
        assert!(lease.mark_host_ready(0x1000).is_err());
        lease.mark_host_ready(0x2000).unwrap();
        assert_eq!(lease.descriptor().address(), stable);

        let fence = UploadFenceContract::new(
            descriptor.operation(),
            FenceId::new(10),
            descriptor.destination_generation(),
        );
        lease.mark_upload_submitted(fence).unwrap();
        assert_eq!(lease.descriptor().address(), stable);
        assert!(lease.retire().is_err());

        let wrong = UploadFenceContract::new(
            descriptor.operation(),
            FenceId::new(11),
            descriptor.destination_generation(),
        )
        .observation(CompletionTimestamp::from_nanos(1));
        assert!(matches!(
            lease.mark_upload_fence(wrong),
            Err(IoProtocolError::UploadFenceMismatch { field: "fence" })
        ));
        lease
            .mark_upload_fence(fence.observation(CompletionTimestamp::from_nanos(2)))
            .unwrap();
        assert_eq!(lease.state(), SlabLeaseState::Releasable);
        lease.retire().unwrap();
        assert_eq!(lease.state(), SlabLeaseState::Retired);
        assert!(lease.retire().is_err());
    }

    #[test]
    fn residency_validation_rejects_generation_and_identity_mismatch() {
        let key = key(1);
        let stale = ResidencyBinding::new(
            key.model(),
            key.layer(),
            key.expert(),
            key.backend(),
            key.device(),
            DestinationSlotId::new(1),
            DestinationGeneration::new(key.destination_generation().get() + 1),
        );
        assert!(matches!(
            ValidatedResidencyBinding::new(key, stale),
            Err(IoProtocolError::ResidencyMismatch {
                field: "destination generation"
            })
        ));

        let wrong_device = ResidencyBinding {
            generation: key.destination_generation(),
            device: DeviceId::new(key.device().get() + 1),
            ..stale
        };
        assert!(matches!(
            ValidatedResidencyBinding::new(key, wrong_device),
            Err(IoProtocolError::ResidencyMismatch { field: "device" })
        ));
    }

    #[test]
    fn expert_lease_set_requires_complete_unique_exact_bindings() {
        let first = key(1);
        let second = key(2);
        let set = ExpertLeaseSet::new(
            [second, first, first],
            [binding_for(second, 2), binding_for(first, 1)],
            MappingEpoch::new(1),
            dispatch_fence(),
        )
        .unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(set.bindings()[0].key(), first);
        assert!(set.binding_for(second).is_some());

        assert!(matches!(
            ExpertLeaseSet::new(
                [first, second],
                [binding_for(first, 1)],
                MappingEpoch::new(1),
                dispatch_fence(),
            ),
            Err(IoProtocolError::MissingResidencyBinding(missing)) if *missing == second
        ));
        assert!(matches!(
            ExpertLeaseSet::new(
                [first],
                [binding_for(first, 1), binding_for(second, 2)],
                MappingEpoch::new(1),
                dispatch_fence(),
            ),
            Err(IoProtocolError::UnexpectedResidencyBinding(extra)) if *extra == second
        ));
        assert!(matches!(
            ExpertLeaseSet::new(
                [first],
                [binding_for(first, 1), binding_for(first, 2)],
                MappingEpoch::new(1),
                dispatch_fence(),
            ),
            Err(IoProtocolError::DuplicateResidencyBinding(duplicate)) if *duplicate == first
        ));
        assert!(matches!(
            ExpertLeaseSet::new(
                [first],
                [binding_for(first, 1)],
                MappingEpoch::new(0),
                dispatch_fence(),
            ),
            Err(IoProtocolError::InvalidMappingEpoch)
        ));
    }

    #[test]
    fn lease_set_from_dependencies_rejects_non_residency_obligations() {
        let dependencies = DependencySet::new([
            LogicalDependency::expert_resident(key(1)).unwrap(),
            LogicalDependency::operation_retired(OperationId::new(4)).unwrap(),
        ])
        .unwrap();
        assert!(matches!(
            ExpertLeaseSet::for_dependencies(
                &dependencies,
                [binding_for(key(1), 1)],
                MappingEpoch::new(1),
                dispatch_fence(),
            ),
            Err(IoProtocolError::NonResidencyDependency(dependency))
                if matches!(*dependency, LogicalDependency::OperationRetired(_))
        ));
    }

    #[test]
    fn model_progress_exposes_only_complete_or_validated_waiting() {
        let dependencies =
            DependencySet::new([LogicalDependency::expert_resident(key(1)).unwrap()]).unwrap();
        let waiting: ModelProgress<u32> = ModelProgress::waiting(dependencies.clone()).unwrap();
        assert_eq!(waiting, ModelProgress::Waiting(dependencies));
        assert_eq!(
            ModelProgress::complete(2).map(|value| value + 1),
            ModelProgress::Complete(3)
        );
    }
}
