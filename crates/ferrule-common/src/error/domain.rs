//! Opt-in classification and typed diagnostic traversal, independent of HTTP.
//!
//! No classifier infers policy from `Display`, cleanup failures, or arbitrary
//! batch members. Existing server status/code/param policy remains server-owned.

use std::error::Error as StdError;
use std::fmt::{self, Display};

use snafu::Snafu;

/// Stable transport-neutral classes. These are not HTTP error codes/messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorClass {
    InvalidRequest,
    Conflict,
    Capacity,
    Unavailable,
    Internal,
}

impl ErrorClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::Conflict => "conflict",
            Self::Capacity => "capacity",
            Self::Unavailable => "unavailable",
            Self::Internal => "internal",
        }
    }
}

impl Display for ErrorClass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// An explicitly classified failure retaining its original typed source.
///
/// Classification is supplied by the domain owner, not guessed from text.
/// `Display`/`Debug` are internal diagnostics and may contain sensitive data;
/// only `class()` is a stable, data-free label. This wrapper does not replace
/// any existing common or server error type or alter HTTP mapping.
#[derive(Debug, Snafu)]
#[snafu(display("{class}: {source}"))]
pub struct ClassifiedError<Source: StdError + 'static> {
    class: ErrorClass,
    source: Source,
}

impl<Source: StdError + 'static> ClassifiedError<Source> {
    pub fn new(class: ErrorClass, source: Source) -> Self {
        Self { class, source }
    }

    pub const fn class(&self) -> ErrorClass {
        self.class
    }

    pub fn into_source(self) -> Source {
        self.source
    }
}

/// Diagnostic traversal of common errors, including cleanup and batch branches.
///
/// Visits the root, then primary sources, then cleanup; batches retain input
/// order. Transparent common variants expose their typed inner value here even
/// when `std::error::Error::source()` forwards past it. The standard source chain
/// and existing `Display` behavior are unchanged.
///
/// Other crates' multi-error types must expose their extra branches themselves;
/// only their standard `source()` chain is followed here. This is NOT a public
/// response classifier: a cleanup failure must not override primary policy.
pub struct ErrorChain<'a> {
    pending: Vec<&'a (dyn StdError + 'static)>,
}

impl<'a> ErrorChain<'a> {
    pub fn new(root: &'a (dyn StdError + 'static)) -> Self {
        Self {
            pending: vec![root],
        }
    }

    /// Find the first matching typed node, including common secondary branches.
    pub fn downcast_ref<T: StdError + 'static>(self) -> Option<&'a T> {
        self.into_iter().find_map(|error| error.downcast_ref::<T>())
    }
}

impl<'a> Iterator for ErrorChain<'a> {
    type Item = &'a (dyn StdError + 'static);

    fn next(&mut self) -> Option<Self::Item> {
        let error = self.pending.pop()?;
        if let Some(common) = error.downcast_ref::<super::Error>() {
            use super::Error;
            match common {
                Error::Io { source } => self.pending.push(source),
                Error::IoProtocol { source } => self.pending.push(source),
                Error::Materialization { source } => self.pending.push(source),
                Error::MaterializationResources { source } => self.pending.push(source),
                Error::Context { source, .. } => self.pending.push(source.as_ref()),
                Error::Cleanup {
                    source, cleanup, ..
                } => {
                    self.pending.push(cleanup.as_ref());
                    self.pending.push(source.as_ref());
                }
                Error::FailureBatch { failures, .. } => {
                    self.pending.extend(
                        failures
                            .iter()
                            .rev()
                            .map(|failure| failure as &(dyn StdError + 'static)),
                    );
                }
                _ => self.pending.extend(error.source()),
            }
        } else {
            self.pending.extend(error.source());
        }
        Some(error)
    }
}
