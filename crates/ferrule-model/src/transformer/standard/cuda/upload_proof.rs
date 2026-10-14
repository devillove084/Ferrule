//! Local completion receipts only; no stream, resource, or transaction authority.
use ferrule_common::execution::ExecutionTransactionId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MixedServingScope {
    pub image_generation: u64,
    pub transaction: ExecutionTransactionId,
}

/// A non-ZST stack identity for one call, including nested calls with the same
/// image/transaction. A borrowed receipt cannot outlive or survive moving it.
pub(super) struct SubmissionGeneration {
    scope: Option<MixedServingScope>,
}

/// Not Clone/Copy. Issued only after both actual waits have succeeded. A no-op
/// consumer proof returns None, never this receipt.
pub(super) struct ConsumerCompletion<'a> {
    generation: &'a SubmissionGeneration,
}

#[derive(Clone, Copy)]
pub(super) struct TailCompletion {
    pub operation_succeeded: bool,
    pub boundary_drained: bool,
    pub compute_synchronized: bool,
    pub owner_healthy: bool,
}

impl SubmissionGeneration {
    pub fn new(scope: Option<MixedServingScope>) -> Self {
        Self { scope }
    }

    pub fn prove<E>(
        &self,
        compute: Result<(), E>,
        upload: Result<(), E>,
    ) -> Result<ConsumerCompletion<'_>, E> {
        compute.and(upload)?;
        Ok(ConsumerCompletion { generation: self })
    }

    /// Consumes the receipt at the immediate tail, after the non-submitting
    /// drain/compute cleanup. This does not authorize retiring any resource.
    pub fn covers_tail_upload(
        &self,
        proof: Option<ConsumerCompletion<'_>>,
        current: Option<MixedServingScope>,
        tail: TailCompletion,
    ) -> bool {
        tail.operation_succeeded
            && tail.boundary_drained
            && tail.compute_synchronized
            && tail.owner_healthy
            && self.scope.is_some()
            && self.scope == current
            && proof.is_some_and(|p| std::ptr::eq(p.generation, self))
    }
}
