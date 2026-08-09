use std::error::Error as StdError;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use snafu::Snafu;

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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn worker_errors_preserve_typed_sources() {
        let error = WorkerExecutionError::AdmissionTokenization {
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid tokens"),
        };

        assert_eq!(
            std::error::Error::source(&error)
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
}
