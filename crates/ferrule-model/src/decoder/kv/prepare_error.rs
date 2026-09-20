//! A failed preparation can retain custody even when no public handle exists.
use ferrule_common::Error;
use ferrule_common::execution::ExecutionTransactionId;

/// Device work from a failed prepare is not proven quiescent. The backend keeps
/// its original ledger and physical pins; absence of a returned handle is NOT a
/// rollback ACK. Callers must retain logical reservations/owner custody instead
/// of publishing, finalizing, or reusing pages. No cleanup authority is carried
/// by this error and dropping it never releases custody.
#[derive(Debug)]
pub struct KvPrepareQuiescenceUnknown {
    transaction: ExecutionTransactionId,
    source: Error,
}
impl KvPrepareQuiescenceUnknown {
    pub fn wrap(transaction: ExecutionTransactionId, source: Error) -> Error {
        Error::ModelSource {
            source: Box::new(Self {
                transaction,
                source,
            }),
        }
    }
    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.transaction
    }
    pub fn from_error<'a>(error: &'a (dyn std::error::Error + 'static)) -> Option<&'a Self> {
        if let Some(unknown) = error.downcast_ref::<Self>() {
            return Some(unknown);
        }
        // Cleanup failures can carry the unknown proof on either branch.
        if let Some(error) = error.downcast_ref::<Error>() {
            match error {
                Error::Cleanup {
                    source, cleanup, ..
                } => {
                    return Self::from_error(source.as_ref())
                        .or_else(|| Self::from_error(cleanup.as_ref()));
                }
                Error::FailureBatch { failures, .. } => {
                    return failures.iter().find_map(|e| Self::from_error(e));
                }
                _ => {}
            }
        }
        error.source().and_then(Self::from_error)
    }
}
impl std::fmt::Display for KvPrepareQuiescenceUnknown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KV prepare {} quiescence unknown; custody retained: {}",
            self.transaction.get(),
            self.source
        )
    }
}
impl std::error::Error for KvPrepareQuiescenceUnknown {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
