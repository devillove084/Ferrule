//! Unified error taxonomy for Ferrule.
//!
//! The cross-crate boundary [`Error`] plus every typed subsystem error live in
//! this one module. Subsystems keep their own typed errors and preserve them
//! as sources at their outer boundary; message variants on [`Error`] remain
//! only for legacy leaf APIs that have not yet acquired a typed error.

use std::error::Error as StdError;
use std::fmt::{Debug, Display};
use std::path::PathBuf;
use std::sync::Arc;

use snafu::Snafu;

use crate::QuantType;
use crate::io_protocol::{BackendId, DeviceId, ModelInstanceId};
use crate::io_protocol::{
    CompletionGeneration, FailureReason, LoadStage, LogicalDependency, MaterializationKey,
    MaterializationPurpose, SlabLeaseState,
};

/// Cross-crate error boundary for model, execution, and backend APIs.
#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(transparent)]
    Io { source: std::io::Error },

    #[snafu(transparent)]
    IoProtocol { source: IoProtocolError },

    #[snafu(transparent)]
    Materialization { source: MaterializationResolveError },

    #[snafu(transparent)]
    MaterializationResources {
        source: MaterializationResourceError,
    },

    #[snafu(display("GGUF: {message}"))]
    Gguf { message: String },

    #[snafu(display("graph: {message}"))]
    Graph { message: String },

    #[snafu(display("kernel: {message}"))]
    Kernel { message: String },

    #[snafu(display("backend: {source}"))]
    Backend {
        source: Box<dyn StdError + Send + Sync>,
    },

    #[snafu(display("model: {message}"))]
    Model { message: String },

    #[snafu(display("model: {source}"))]
    ModelSource {
        source: Box<dyn StdError + Send + Sync>,
    },

    #[snafu(display("execution: {message}"))]
    Execution { message: String },

    #[snafu(display("tokenization: {message}"))]
    Tokenization { message: String },

    #[snafu(display("internal invariant: {message}"))]
    Internal { message: String },

    #[snafu(display("{operation}: {source}"))]
    Context {
        operation: String,
        source: Box<Error>,
    },

    #[snafu(display("{operation} failed: {source}; cleanup also failed: {cleanup}"))]
    Cleanup {
        operation: String,
        source: Box<Error>,
        cleanup: Box<Error>,
    },

    #[snafu(display(
        "{operation} encountered {} independent failures",
        failures.len()
    ))]
    FailureBatch {
        operation: String,
        failures: Vec<Error>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn context(operation: impl Into<String>, source: Error) -> Self {
        Self::Context {
            operation: operation.into(),
            source: Box::new(source),
        }
    }

    pub fn with_cleanup(operation: impl Into<String>, source: Error, cleanup: Result<()>) -> Self {
        match cleanup {
            Ok(()) => source,
            Err(cleanup) => Self::Cleanup {
                operation: operation.into(),
                source: Box::new(source),
                cleanup: Box::new(cleanup),
            },
        }
    }

    pub fn failures(operation: impl Into<String>, failures: Vec<Error>) -> Result<()> {
        if failures.is_empty() {
            Ok(())
        } else {
            Err(Self::FailureBatch {
                operation: operation.into(),
                failures,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum QuantizationError {
    #[snafu(display("quantization type {quant:?} is not supported by QMatrix"))]
    UnsupportedMatrixType { quant: QuantType },
    #[snafu(display("invalid {quant:?} block size: expected {expected} bytes, got {actual}"))]
    InvalidBlockSize {
        quant: QuantType,
        expected: usize,
        actual: usize,
    },
    #[snafu(display("invalid {quant:?} row value count {row_values}"))]
    InvalidRowSize { quant: QuantType, row_values: usize },
}

pub type QuantizationResult<T> = std::result::Result<T, QuantizationError>;

// ---------------------------------------------------------------------------
// Serving and worker lifecycle
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum ServingConfigError {
    #[snafu(display("command_queue_capacity must be greater than zero"))]
    ZeroCommandQueueCapacity,
    #[snafu(display("event_queue_capacity must be at least two, got {actual}"))]
    EventQueueCapacityTooSmall { actual: usize },
    #[snafu(display("max_commands_per_tick must be greater than zero"))]
    ZeroCommandsPerTick,
    #[snafu(display("admission_timeout must be greater than zero"))]
    ZeroAdmissionTimeout,
}

#[derive(Debug, Snafu)]
pub enum ServingRequestError<ChatSource>
where
    ChatSource: StdError + 'static,
{
    #[snafu(display("chat content parts must not be empty"))]
    EmptyContentParts,
    #[snafu(display("chat content part {index} has unsupported type '{kind}'"))]
    UnsupportedContentPart { index: usize, kind: String },
    #[snafu(display("chat text content part {index} is missing its text field"))]
    MissingContentPartText { index: usize },
    #[snafu(display("model '{requested}' is not served; available model is '{available}'"))]
    ModelNotServed {
        requested: String,
        available: String,
    },
    #[snafu(display("exactly one completion is supported, got n = {actual}"))]
    CompletionCount { actual: usize },
    #[snafu(display("greedy serving requires temperature = 0, got {actual}"))]
    Temperature { actual: f32 },
    #[snafu(display("greedy serving requires top_p = 1, got {actual}"))]
    TopP { actual: f32 },
    #[snafu(display("greedy serving requires top_k = 1, got {actual}"))]
    TopK { actual: usize },
    #[snafu(display("greedy serving requires min_p = 0, got {actual}"))]
    MinP { actual: f32 },
    #[snafu(display("greedy serving requires repetition_penalty = 1, got {actual}"))]
    RepetitionPenalty { actual: f32 },
    #[snafu(display("frequency_penalty and presence_penalty are not supported"))]
    PenaltiesUnsupported,
    #[snafu(display("streaming logprobs are not supported"))]
    StreamingLogprobsUnsupported,
    #[snafu(display("tool calling is not supported"))]
    ToolCallingUnsupported,
    #[snafu(display("response_format is not supported"))]
    ResponseFormatUnsupported,
    #[snafu(display("provide only one of max_completion_tokens or max_tokens"))]
    ConflictingTokenLimits,
    #[snafu(display("{field} must be greater than zero"))]
    ZeroTokenLimit { field: &'static str },
    #[snafu(display("cannot format chat prompt: {source}"))]
    ChatFormat { source: ChatSource },
    #[snafu(display("best_of = 1 is required, got {actual}"))]
    BestOf { actual: usize },
    #[snafu(display("logprobs is not supported"))]
    LogprobsUnsupported,
    #[snafu(display("echo is not supported"))]
    EchoUnsupported,
    #[snafu(display("stream_options is only supported when stream = true"))]
    StreamOptionsWithoutStreaming,
    #[snafu(display("batch prompts are not supported; prompt must be a single string"))]
    BatchPromptUnsupported,
    #[snafu(display("prompt must be a single string"))]
    InvalidPromptType,
    #[snafu(display("stop strings must not be empty"))]
    EmptyStopString,
    #[snafu(display("at most four stop strings are supported, got {actual}"))]
    TooManyStopStrings { actual: usize },
    #[snafu(display("add_special_tokens=false is not supported"))]
    AddSpecialTokensUnsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerOperation {
    Admission,
    Tokenization,
}

impl Display for WorkerOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Admission => "admission",
            Self::Tokenization => "tokenization",
        })
    }
}

#[derive(Debug, Snafu)]
pub enum WorkerRequestError<RuntimeSource>
where
    RuntimeSource: StdError + 'static,
{
    #[snafu(display("model request queue is full"))]
    QueueFull,
    #[snafu(display("model worker is unavailable during {operation}"))]
    Unavailable { operation: WorkerOperation },
    #[snafu(display("timed out waiting for model admission"))]
    AdmissionTimeout,
    #[snafu(display("formatted prompt produced no tokens"))]
    EmptyPromptTokens,
    // No context selector: `TokenizationSnafu` would collide with the
    // boundary `Error::Tokenization` selector in this module.
    #[snafu(context(false))]
    #[snafu(display("prompt tokenization failed: {source}"))]
    Tokenization { source: RuntimeSource },
    #[snafu(display("model worker rejected the request: {source}"))]
    Rejected {
        source: Arc<WorkerExecutionError<RuntimeSource>>,
    },
}

impl<RuntimeSource> WorkerRequestError<RuntimeSource>
where
    RuntimeSource: StdError + 'static,
{
    pub const fn is_overloaded(&self) -> bool {
        matches!(self, Self::QueueFull)
    }

    pub const fn is_unavailable(&self) -> bool {
        matches!(self, Self::Unavailable { .. } | Self::AdmissionTimeout)
    }
}

#[derive(Debug, Snafu)]
pub enum WorkerStartError<FactorySource, RuntimeSource>
where
    FactorySource: StdError + 'static,
    RuntimeSource: StdError + 'static,
{
    #[snafu(transparent)]
    InvalidConfig { source: ServingConfigError },
    #[snafu(display("model engine factory failed: {source}"))]
    EngineFactory { source: FactorySource },
    #[snafu(display("failed to build model-owner async runtime: {source}"))]
    RuntimeBuild { source: std::io::Error },
    #[snafu(display("model startup materialization failed: {source}"))]
    Startup { source: RuntimeSource },
    #[snafu(display("failed to spawn model worker: {source}"))]
    ThreadSpawn { source: std::io::Error },
    #[snafu(display("model worker stopped during initialization: {source}"))]
    InitializationChannel { source: std::sync::mpsc::RecvError },
}

#[derive(Debug, Snafu)]
pub enum WorkerShutdownError<JoinSource>
where
    JoinSource: StdError + 'static,
{
    #[snafu(display("failed to join model worker task: {source}"))]
    JoinTask { source: JoinSource },
    #[snafu(display("model worker thread panicked"))]
    ThreadPanicked,
}

#[derive(Debug, Snafu)]
#[snafu(display("cannot serialize OpenAI SSE event: {source}"))]
pub struct SseSerializationError<SerializationSource>
where
    SerializationSource: StdError + 'static,
{
    pub source: SerializationSource,
}

#[derive(Debug, Snafu)]
pub enum WorkerExecutionError<RuntimeSource>
where
    RuntimeSource: StdError + 'static,
{
    #[snafu(display("model execution failed"))]
    ModelExecution,
    #[snafu(display("formatted prompt produced no tokens"))]
    AdmissionEmptyPromptTokens,
    #[snafu(display("prompt tokenization failed: {source}"))]
    AdmissionTokenization { source: RuntimeSource },
    #[snafu(display("request cancellation failed: {source}"))]
    Cancellation { source: RuntimeSource },
    #[snafu(display("request cancellation failed during shutdown: {source}"))]
    ShutdownCancellation { source: RuntimeSource },
    #[snafu(display("model worker execution failed: {source}"))]
    Runtime { source: RuntimeSource },
    #[snafu(display("runtime reported blocked work without an owned async continuation"))]
    BlockedWithoutContinuation,
    #[snafu(display("model completion source closed with live async work"))]
    CompletionSourceClosed,
    #[snafu(display("runtime reported idle while model background work remained runnable"))]
    RunnableBackgroundReportedIdle,
}

// ---------------------------------------------------------------------------
// State-dict binding, schema, and transforms
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameMappingFailureKind {
    InvalidConfiguration,
    InvalidExternalName,
    UnknownTensor,
    InvalidCanonicalPath,
    ContractViolation,
}

impl Display for NameMappingFailureKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => "invalid configuration",
            Self::InvalidExternalName => "invalid external name",
            Self::UnknownTensor => "unknown tensor",
            Self::InvalidCanonicalPath => "invalid canonical path",
            Self::ContractViolation => "contract violation",
        })
    }
}

#[derive(Debug, Snafu)]
pub enum NameMappingError {
    #[snafu(display("external tensor name cannot be empty"))]
    EmptyExternalName,
    #[snafu(display("duplicate exact mapping for external tensor '{external}'"))]
    DuplicateExactMapping { external: String },
    #[snafu(display("name mapper rejected an invalid canonical path: {source}"))]
    InvalidCanonicalPath {
        source: Box<dyn StdError + Send + Sync>,
    },
    #[snafu(display("name mapper {kind}: {detail}"))]
    MapperRejected {
        kind: NameMappingFailureKind,
        detail: String,
    },
}

impl NameMappingError {
    pub fn new(reason: impl Into<String>) -> Self {
        Self::MapperRejected {
            kind: NameMappingFailureKind::ContractViolation,
            detail: reason.into(),
        }
    }

    pub fn invalid_canonical_path(source: impl StdError + Send + Sync + 'static) -> Self {
        Self::InvalidCanonicalPath {
            source: Box::new(source),
        }
    }

    pub fn reason(&self) -> &str {
        match self {
            Self::EmptyExternalName => "external tensor name cannot be empty",
            Self::DuplicateExactMapping { external } => external,
            Self::InvalidCanonicalPath { .. } => "invalid canonical path",
            Self::MapperRejected { detail, .. } => detail,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum StateDictSchemaError<Path, Id, SpecSource>
where
    Path: Debug + Display,
    Id: Debug + Display,
    SpecSource: StdError + 'static,
{
    #[snafu(display("invalid parameter '{path}': {source}"))]
    InvalidParameter { path: Path, source: SpecSource },
    #[snafu(display("duplicate canonical parameter path '{path}'"))]
    DuplicatePath { path: Path },
    #[snafu(display("duplicate parameter id {id}"))]
    DuplicateId { id: Id },
    #[snafu(display("alias '{alias}' refers to unknown parameter id {target}"))]
    UnknownAliasTarget { alias: Path, target: Id },
    #[snafu(display("parameter alias cycle detected: {cycle:?}"))]
    AliasCycle { cycle: Vec<Id> },
    #[snafu(display("alias '{alias}' is incompatible with '{target}': {reason}"))]
    IncompatibleAlias {
        alias: Path,
        target: Path,
        reason: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum StateDictTransformError {
    #[snafu(display("transpose rank mismatch: tensor rank={rank}, axes={axes:?}"))]
    TransposeRankMismatch { rank: usize, axes: Vec<usize> },
    #[snafu(display("transpose axis {axis} is outside tensor rank {rank}"))]
    TransposeAxisOutOfRange { axis: usize, rank: usize },
    #[snafu(display("transpose axis {axis} appears more than once"))]
    DuplicateTransposeAxis { axis: usize },
    #[snafu(display("split transforms are not supported by the strict binder"))]
    SplitUnsupported,
    #[snafu(display("concat transforms are not supported by the strict binder"))]
    ConcatUnsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum StateDictMetadataError {
    #[snafu(display("shape dimension {dimension} is outside the checkpoint byte-size domain"))]
    DimensionOutOfRange { dimension: usize },
    #[snafu(display("tensor element count overflowed the checkpoint byte-size domain"))]
    ElementCountOverflow,
    #[snafu(display("tensor byte size overflowed the checkpoint byte-size domain"))]
    ByteSizeOverflow,
    #[snafu(display("tensor metadata declares {actual} bytes, expected exactly {expected}"))]
    ByteSizeMismatch { expected: u64, actual: u64 },
}

#[derive(Debug, Snafu)]
pub enum StateDictBindingIssue<Path, Part, DType, NameSource, IdentitySource>
where
    Path: Debug + Display,
    Part: Debug + Display,
    DType: Debug + Display,
    NameSource: StdError + 'static,
    IdentitySource: StdError + 'static,
{
    #[snafu(display("name mapper rejected external tensor '{external}': {source}"))]
    NameMapping {
        external: String,
        source: NameSource,
    },
    #[snafu(display("unexpected external tensor '{external}'"))]
    UnexpectedTensor { external: String },
    #[snafu(display("external tensor '{external}' maps to unknown canonical path '{path}'"))]
    UnknownCanonicalPath { external: String, path: Path },
    #[snafu(display("external tensor '{external}' maps to unavailable {part} part of '{path}'"))]
    UnexpectedParameterPart {
        external: String,
        path: Path,
        part: Part,
    },
    #[snafu(display(
        "duplicate binding for {part} of '{path}': first '{first}', duplicate '{duplicate}'"
    ))]
    DuplicateTensorPart {
        path: Path,
        part: Part,
        first: String,
        duplicate: String,
    },
    #[snafu(display(
        "invalid transform for external tensor '{external}' mapped to '{path}': {source}"
    ))]
    InvalidTransform {
        external: String,
        path: Path,
        source: StateDictTransformError,
    },
    #[snafu(display(
        "dtype mismatch for external tensor '{external}' mapped to {part} of '{path}': expected {expected:?}, got {actual}"
    ))]
    DTypeMismatch {
        external: String,
        path: Path,
        part: Part,
        expected: Vec<DType>,
        actual: DType,
    },
    #[snafu(display(
        "shape mismatch for external tensor '{external}' mapped to {part} of '{path}': expected {expected:?}, got {actual:?}"
    ))]
    ShapeMismatch {
        external: String,
        path: Path,
        part: Part,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[snafu(display("invalid metadata for external tensor '{external}': {source}"))]
    InvalidByteSize {
        external: String,
        source: StateDictMetadataError,
    },
    #[snafu(display(
        "cannot capture source identity for '{external}' at '{}': {source}",
        path.display()
    ))]
    SourceIdentity {
        external: String,
        path: PathBuf,
        source: IdentitySource,
    },
    #[snafu(display(
        "source range for '{external}' at '{}' exceeds file length: offset={offset}, bytes={bytes}, source_bytes={source_bytes}",
        path.display()
    ))]
    SourceRange {
        external: String,
        path: PathBuf,
        offset: u64,
        bytes: u64,
        source_bytes: u64,
    },
    #[snafu(display("required parameter '{path}' is missing"))]
    MissingParameter { path: Path },
    #[snafu(display("required scale for parameter '{path}' is missing"))]
    MissingScale { path: Path },
    #[snafu(display("parameter '{path}' has a scale binding without a weight binding"))]
    OrphanScale { path: Path },
}

impl<Path, Part, DType, NameSource, IdentitySource>
    StateDictBindingIssue<Path, Part, DType, NameSource, IdentitySource>
where
    Path: Debug + Display,
    Part: Debug + Display,
    DType: Debug + Display,
    NameSource: StdError + 'static,
    IdentitySource: StdError + 'static,
{
    pub fn with_external(self, external: String) -> Self {
        match self {
            Self::UnexpectedParameterPart { path, part, .. } => Self::UnexpectedParameterPart {
                external,
                path,
                part,
            },
            issue => issue,
        }
    }
}

#[derive(Debug, Snafu)]
#[snafu(display(
    "state-dict binding failed with {} issue(s)",
    issues.len() + 1
))]
pub struct StateDictBindingError<Issue>
where
    Issue: StdError + Send + Sync + 'static,
{
    #[snafu(source)]
    source: Issue,
    issues: Vec<Issue>,
}

impl<Issue> StateDictBindingError<Issue>
where
    Issue: StdError + Send + Sync + 'static,
{
    pub fn new(mut issues: Vec<Issue>) -> Self {
        assert!(
            !issues.is_empty(),
            "state-dict binding errors require an issue"
        );
        let source = issues.remove(0);
        Self { source, issues }
    }

    pub fn issues(&self) -> StateDictIssues<'_, Issue> {
        StateDictIssues {
            source: &self.source,
            issues: &self.issues,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StateDictIssues<'a, Issue> {
    source: &'a Issue,
    issues: &'a [Issue],
}

impl<'a, Issue> StateDictIssues<'a, Issue> {
    pub fn iter(self) -> impl Iterator<Item = &'a Issue> {
        std::iter::once(self.source).chain(self.issues)
    }

    pub const fn len(self) -> usize {
        self.issues.len() + 1
    }

    pub const fn is_empty(self) -> bool {
        false
    }
}

impl<'a, Issue> IntoIterator for StateDictIssues<'a, Issue> {
    type Item = &'a Issue;
    type IntoIter = std::iter::Chain<std::iter::Once<&'a Issue>, std::slice::Iter<'a, Issue>>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self.source).chain(self.issues)
    }
}

// ---------------------------------------------------------------------------
// I/O protocol and materialization
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_errors_preserve_typed_sources() {
        let error = WorkerExecutionError::AdmissionTokenization {
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid tokens"),
        };

        assert_eq!(
            StdError::source(&error)
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .map(std::io::Error::kind),
            Some(std::io::ErrorKind::InvalidData)
        );
    }

    #[test]
    fn config_errors_keep_structured_fields() {
        let error = ServingConfigError::EventQueueCapacityTooSmall { actual: 1 };
        assert_eq!(
            error.to_string(),
            "event_queue_capacity must be at least two, got 1"
        );
    }

    #[test]
    fn name_mapping_error_preserves_typed_canonical_path_source() {
        let error = NameMappingError::invalid_canonical_path(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid module path",
        ));

        assert!(
            StdError::source(&error)
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .is_some()
        );
    }

    #[test]
    fn binding_error_preserves_first_typed_issue_as_source() {
        let error = StateDictBindingError::new(vec![StateDictMetadataError::ByteSizeMismatch {
            expected: 8,
            actual: 4,
        }]);

        assert!(matches!(
            StdError::source(&error)
                .and_then(|source| source.downcast_ref::<StateDictMetadataError>()),
            Some(StateDictMetadataError::ByteSizeMismatch {
                expected: 8,
                actual: 4
            })
        ));
        assert_eq!(error.issues().len(), 1);
        assert_eq!(
            error.to_string(),
            "state-dict binding failed with 1 issue(s)"
        );
    }
}
