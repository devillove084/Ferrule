//! Completion evidence for decoder-owned external work.
//!
//! These types describe sub-work custody only. They do not prove KV
//! consumption, publication, or any other decoder transaction transition.

use ferrule_common::Error;

/// Progress reported by an owner of decoder sub-work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkProgress {
    /// All sub-work is proven quiescent; this is not a root KV acknowledgement.
    Complete,
    /// Work is still owned and may be polled again.
    Pending,
    /// Completion evidence was permanently lost; custody must remain quarantined.
    Unknown,
}

/// Completion evidence for decoder sub-work is permanently unknown.
/// A later independent fence cannot discharge this custody; the existing owner
/// must remain quarantined. This marker owns no resource or cleanup authority.
///
/// This is deliberately distinct from KV terminal statuses such as
/// [`super::KvEndProgress::ConsumedRejected`]: it means the sub-work owner could not
/// prove quiescence, not that a KV transition consumed or rejected anything.
#[derive(Debug)]
pub struct CompletionUnknown {
    source: Error,
}

impl CompletionUnknown {
    /// Wrap the original failure without flattening its typed source chain.
    pub fn wrap(source: Error) -> Error {
        Error::ModelSource {
            source: Box::new(Self { source }),
        }
    }
}

impl std::fmt::Display for CompletionUnknown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "external work completion unknown; custody retained: {}",
            self.source
        )
    }
}

impl std::error::Error for CompletionUnknown {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::error::ErrorChain;
    use std::error::Error as StdError;

    #[test]
    fn completion_unknown_preserves_typed_sources_and_secondary_branches() {
        let original = std::io::Error::new(std::io::ErrorKind::TimedOut, "work deadline");
        let error = CompletionUnknown::wrap(original.into());
        let unknown = ErrorChain::new(&error)
            .downcast_ref::<CompletionUnknown>()
            .unwrap();
        assert!(unknown.source().unwrap().is::<Error>());
        let error = Error::with_cleanup(
            "startup",
            Error::Model {
                message: "preparation failed".into(),
            },
            Err(Error::FailureBatch {
                operation: "drain".into(),
                failures: vec![Error::context("external owner", error)],
            }),
        );
        assert!(
            ErrorChain::new(&error)
                .downcast_ref::<CompletionUnknown>()
                .is_some()
        );
        assert_eq!(
            ErrorChain::new(&error)
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::TimedOut,
        );
    }
}
