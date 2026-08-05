use crate::plan::{ExecutionMode, KernelOperation, KernelProviderId, WeightLayout};
use snafu::Snafu;

#[derive(Debug, Snafu)]
pub enum BackendError {
    #[snafu(display("backend invariant violated: {message}"))]
    Invariant { message: String },

    #[snafu(display(
        "provider {provider:?} does not support {operation:?} in {mode:?} mode{}",
        if *deterministic { " with deterministic execution" } else { "" }
    ))]
    UnsupportedOperation {
        provider: KernelProviderId,
        operation: KernelOperation,
        mode: ExecutionMode,
        deterministic: bool,
    },

    #[snafu(display(
        "provider {provider:?} rejected {operation:?} in {mode:?} mode with layout {layout:?}: {reason}"
    ))]
    UnsupportedLinearBundle {
        provider: KernelProviderId,
        operation: KernelOperation,
        mode: ExecutionMode,
        layout: WeightLayout,
        reason: String,
    },

    #[snafu(display(
        "provider {provider:?} manifest does not publish native kernel {kernel} for {operation:?}"
    ))]
    MissingNativeKernel {
        provider: KernelProviderId,
        operation: KernelOperation,
        kernel: &'static str,
    },
}

pub type BackendResult<T> = std::result::Result<T, BackendError>;

impl From<BackendError> for ferrule_common::Error {
    fn from(source: BackendError) -> Self {
        Self::Backend {
            source: Box::new(source),
        }
    }
}
