//! Typed I/O protocol and materialization errors.

use crate::io_protocol::{
    BackendId, CompletionGeneration, DeviceId, FailureReason, LoadStage, LogicalDependency,
    MaterializationKey, MaterializationPurpose, ModelInstanceId, SlabLeaseState,
};
use snafu::Snafu;

/// Result type for final I/O protocol validation.
pub type IoProtocolResult<T> = std::result::Result<T, IoProtocolError>;

/// A fail-closed violation of an I/O protocol invariant.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Snafu)]
pub enum IoProtocolError {
    #[snafu(display("invalid materialization key: {reason}"))]
    InvalidMaterializationKey { reason: &'static str },
    #[snafu(display("invalid waiter identity: {reason}"))]
    InvalidWaiterId { reason: &'static str },
    #[snafu(display("materialization request field `{field}` does not match its exact key"))]
    MaterializationRequestMismatch { field: &'static str },
    #[snafu(display("materialization transfer cannot evict its own key {key:?}"))]
    SelfEviction { key: Box<MaterializationKey> },
    #[snafu(display("physical materialization reservation must contain at least one pinned slab"))]
    EmptySlabSet,
    #[snafu(display("physical materialization reservation field `{field}` is inconsistent"))]
    PhysicalReservationMismatch { field: &'static str },
    #[snafu(display("a dependency set must contain at least one dependency"))]
    EmptyDependencySet,
    #[snafu(display("dependency set is not in strictly sorted canonical order"))]
    NonCanonicalDependencySet,
    #[snafu(display("illegal load-stage transition from {from:?} to {to:?}"))]
    InvalidLoadTransition { from: LoadStage, to: LoadStage },
    #[snafu(display("completion field `{field}` does not match the owner expectation"))]
    CompletionMismatch { field: &'static str },
    #[snafu(display(
        "completion generation {observed:?} does not match owner generation {expected:?}"
    ))]
    CompletionGenerationMismatch {
        expected: CompletionGeneration,
        observed: CompletionGeneration,
    },
    #[snafu(display("{stage:?} is not a provider-submitted completion stage"))]
    InvalidCompletionStage { stage: LoadStage },
    #[snafu(display("successful completion returned {actual} bytes, expected exactly {expected}"))]
    IncompleteSuccessfulCompletion { expected: u64, actual: u64 },
    #[snafu(display(
        "unsuccessful completion returned {actual} bytes, exceeding expected {expected}"
    ))]
    CompletionByteOverflow { expected: u64, actual: u64 },
    #[snafu(display("retirement authority was already consumed"))]
    RetirementAlreadyConsumed,
    #[snafu(display("slab alignment {alignment} must be a non-zero power of two"))]
    InvalidAlignment { alignment: usize },
    #[snafu(display("slab {field} value {value} is not aligned to {alignment}"))]
    MisalignedSlabField {
        field: &'static str,
        value: u64,
        alignment: usize,
    },
    #[snafu(display("registered slab address must be non-zero"))]
    NullSlabAddress,
    #[snafu(display("arithmetic overflow while validating {context}"))]
    ArithmeticOverflow { context: &'static str },
    #[snafu(display(
        "slab lease range {offset}..{end} exceeds registered allocation length {capacity}"
    ))]
    SlabRangeOutOfBounds {
        offset: u64,
        end: u64,
        capacity: u64,
    },
    #[snafu(display("illegal slab lease transition from {from:?} to {to:?}"))]
    InvalidSlabTransition {
        from: SlabLeaseState,
        to: SlabLeaseState,
    },
    #[snafu(display("upload fence field `{field}` does not match the retained contract"))]
    UploadFenceMismatch { field: &'static str },
    #[snafu(display("residency field `{field}` does not match the required load key"))]
    ResidencyMismatch { field: &'static str },
    #[snafu(display("mapping epoch must be non-zero"))]
    InvalidMappingEpoch,
    #[snafu(display("completion fence identity must be non-zero"))]
    InvalidFenceContract,
    #[snafu(display("duplicate residency binding for {key:?}"))]
    DuplicateResidencyBinding { key: Box<MaterializationKey> },
    #[snafu(display("missing residency binding for {key:?}"))]
    MissingResidencyBinding { key: Box<MaterializationKey> },
    #[snafu(display("unexpected residency binding for {key:?}"))]
    UnexpectedResidencyBinding { key: Box<MaterializationKey> },
    #[snafu(display("dependency {dependency:?} is not a residency dependency"))]
    NonResidencyDependency { dependency: Box<LogicalDependency> },
}

/// Failure while resolving one logical request into an exact physical identity.
#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum MaterializationResolveError {
    #[snafu(display(
        "{purpose:?} materialization request placement ({request_model:?}, {request_backend:?}, {request_device:?}) does not match resolver placement ({resolver_model:?}, {resolver_backend:?}, {resolver_device:?})"
    ))]
    PlacementMismatch {
        purpose: MaterializationPurpose,
        request_model: ModelInstanceId,
        request_backend: BackendId,
        request_device: DeviceId,
        resolver_model: ModelInstanceId,
        resolver_backend: BackendId,
        resolver_device: DeviceId,
    },
    #[snafu(display(
        "{purpose:?} materialization for ({model:?}, {backend:?}, {device:?}) has no physical provider"
    ))]
    ProviderUnavailable {
        purpose: MaterializationPurpose,
        model: ModelInstanceId,
        backend: BackendId,
        device: DeviceId,
    },
    #[snafu(display("{purpose:?} materialization preparation failed: {source}"))]
    Provider {
        purpose: MaterializationPurpose,
        source: FailureReason,
    },
}

pub type MaterializationResolveResult<T> = std::result::Result<T, MaterializationResolveError>;

pub type MaterializationResourceResult<T> = std::result::Result<T, MaterializationResourceError>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Snafu)]
pub enum MaterializationResourceError {
    #[snafu(display("{context} {resource} requirement {requested} exceeds capacity {capacity}"))]
    ExceedsCapacity {
        context: &'static str,
        resource: &'static str,
        requested: u64,
        capacity: u64,
    },
    #[snafu(display("physical materialization plan requires non-zero {resource}"))]
    ZeroRequirement { resource: &'static str },
}
