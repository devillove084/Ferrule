use std::future::Future;
use std::pin::Pin;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionOutput, ExecutionTransactionId,
    KvReservationView,
};

use crate::materialization::{
    MaterializationProvider, MaterializationResolver, validate_continuation_id,
};
use crate::{IncrementalDecodeState, ModelDescriptor};
pub use ferrule_common::ContinuationId;
pub use ferrule_common::execution::TokenLogit;
use ferrule_common::{CompletionHub, DependencySet, Error, ResidencyLeaseSet, Result};

/// Owned model-side completion reactor.
///
/// Reactors are claimed once by the inference owner. They do not borrow the
/// runner and need not be `Send`, allowing Linux `AsyncFd` pumps to run on the
/// owner's local task set.
pub type ModelCompletionReactor = Pin<Box<dyn Future<Output = Result<()>> + 'static>>;

/// Build the allocation-free callback used by non-model producer threads.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn completion_notify_callback(
    completion_hub: CompletionHub,
) -> impl FnOnce() + Send + 'static {
    move || {
        completion_hub.notify();
    }
}

// ── ModelInfo ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub family: crate::ModelFamily,
    pub architecture: Option<String>,
    pub attention: crate::AttentionKind,
    pub weight_source: crate::WeightSource,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub vocab_size: usize,
    pub backend: &'static str,
}

impl ModelInfo {
    pub fn from_descriptor(descriptor: &ModelDescriptor, backend: &'static str) -> Self {
        let spec = &descriptor.spec;
        Self {
            family: spec.family.clone(),
            architecture: spec.architecture.clone(),
            attention: spec.attention.clone(),
            weight_source: spec.weight_source,
            hidden_size: spec.hidden_size.unwrap_or(0),
            num_layers: spec.num_layers.unwrap_or(0),
            num_experts: spec.moe.num_experts.unwrap_or(0),
            num_experts_per_tok: spec.moe.num_experts_per_tok.unwrap_or(0),
            vocab_size: spec.vocab_size.unwrap_or(0),
            backend,
        }
    }
}

// ── ModelRunner trait ────────────────────────────────────────────────────

pub trait ModelRunner {
    fn model_info(&self) -> ModelInfo;
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn decode_incremental(
        &self,
        token: u32,
        state: &mut IncrementalDecodeState,
    ) -> Result<Option<String>> {
        state.step(token, |tokens| self.decode(tokens))
    }
    fn reset_session(&mut self) -> Result<()>;
    fn eos_token_id(&self) -> Option<u32>;
    /// Optional count of model layers/materialized execution states currently bound
    /// into the runner. Useful for lazy artifact-backed runners; dense or eagerly
    /// bound runners may return `None`.
    fn bound_layer_count(&self) -> Option<usize> {
        None
    }

    /// Optional expert activation report (MoE models only).
    fn expert_report(&self) -> Option<String> {
        None
    }
}

/// Compatibility name for the stable protocol continuation identity.
///
/// New APIs use [`ContinuationId`] directly. Zero is rejected whenever an ID is
/// attached to a pending model continuation.
pub type BatchContinuationId = ContinuationId;

/// Model wait state exposed at the runtime boundary.
///
/// Only unresolved, canonical logical dependencies are visible. Physical
/// operation IDs, pointers, provider tickets, and per-continuation load state are
/// deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingModelProgress {
    transaction: ExecutionTransactionId,
    continuation: ContinuationId,
    dependencies: DependencySet,
}

impl PendingModelProgress {
    pub fn new(
        transaction: ExecutionTransactionId,
        continuation: ContinuationId,
        dependencies: DependencySet,
    ) -> Result<Self> {
        validate_continuation_id(continuation)?;
        dependencies.validate()?;
        Ok(Self {
            transaction,
            continuation,
            dependencies,
        })
    }

    /// Return the stable end-to-end transaction owning this wait.
    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.transaction
    }

    /// Return the stable continuation identity for resume, detach, or cancel.
    pub const fn continuation(&self) -> ContinuationId {
        self.continuation
    }

    /// Return only the exact unresolved dependencies blocking progress.
    pub const fn unresolved_dependencies(&self) -> &DependencySet {
        &self.dependencies
    }

    pub const fn dependencies(&self) -> &DependencySet {
        &self.dependencies
    }
}

/// Progress reported by one checkpoint-native proposal operation.
#[derive(Debug, Clone, PartialEq)]
pub enum NativeProposalProgress {
    /// Proposal generation completed and the runner released its continuation.
    Complete(NativeProposal),
    /// The runner retained the continuation and awaits model-owned work.
    Waiting(PendingModelProgress),
}

/// Progress reported by native packed multi-session execution.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiSessionBatchProgress {
    /// The batch completed with its final model output.
    Complete(ExecutionOutput),
    /// The runner retained a continuation and is waiting for model-owned work.
    Waiting(PendingModelProgress),
}

/// Ownership result of cancelling one retained batch continuation.
#[derive(Debug)]
pub enum BatchContinuationCancelOutcome {
    /// Model work is quiescent and all model-owned continuation cleanup succeeded.
    Cancelled,
    /// Cancellation failed before quiescence; the same continuation remains owned
    /// by the runner and may be cancelled again.
    StillActive(Error),
    /// Model work is quiescent, but model-owned cleanup reported an error. The
    /// caller must drop continuation ownership and continue backend rollback.
    Quiesced(Error),
}

/// A model runner that supports explicit per-sequence state for multi-session
/// execution.
///
/// This trait extends [`ModelRunner`] with explicit sequence state and one
/// mandatory packed execution protocol, enabling multiple independent sequences
/// to share prepared resources without a serial fallback.
///
/// Models implement this trait; the runtime crate provides a generic
/// [`NativeMultiSessionExecutor`] that works over any `MultiSessionRunner`.
///
/// [`NativeMultiSessionExecutor`]: ../../ferrule_runtime/engine/struct.NativeMultiSessionExecutor.html
pub trait MultiSessionRunner: ModelRunner {
    /// Per-sequence execution state (position, KV, predictor, etc.).
    type SequenceState;

    /// Exact generation identity of a model execution state. Runtime KV
    /// reservations bind this separately from page-manager ownership generation.
    fn sequence_generation(&self, state: &Self::SequenceState) -> u64;

    /// Describe model-owned expert residency capacity, when this runner uses MoE
    /// expert residency managed by the runtime.
    fn expert_residency_requirements(
        &self,
    ) -> Option<ferrule_common::expert_residency::ExpertResidencyRequirements> {
        None
    }

    /// Whether runtime-owned expert residency control is already attached.
    ///
    /// This remains true when a runner is moved between executor instances, so
    /// rebuilding an executor cannot replace live residency state.
    fn expert_residency_control_installed(&self) -> bool {
        false
    }

    /// Transfer runtime-owned expert residency control into the runner.
    ///
    /// Runners that report requirements must override this hook. Dense runners
    /// retain the no-op default and never receive a controller from the runtime.
    fn install_expert_residency_control(
        &mut self,
        _control: Box<dyn ferrule_common::expert_residency::ExpertResidencyControl>,
    ) -> Result<()> {
        if self.expert_residency_requirements().is_some() {
            return Err(ferrule_common::Error::Execution(
                "runner reports expert residency requirements but does not support installing expert residency control"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Transfer the model-owned physical materialization provider exactly once.
    ///
    /// Any runner with streamable parameters or mutable spilled state may return a
    /// backend. A runner reporting expert residency requirements must return one;
    /// dense fully resident runners may return `None`.
    fn take_materialization_provider(&mut self) -> Option<Box<dyn MaterializationProvider>> {
        None
    }

    /// Whether the runner-level exact-key resolver is installed.
    fn materialization_resolver_installed(&self) -> bool {
        false
    }

    /// Install the model-neutral resolver used to prepare exact resource keys.
    fn install_materialization_resolver(
        &mut self,
        _resolver: Box<dyn MaterializationResolver>,
    ) -> Result<()> {
        Err(Error::Execution(
            "runner does not support materialization resolvers".into(),
        ))
    }

    /// Borrow the installed exact-key resolver. It owns no logical demand,
    /// publication, cancellation, residency, or eviction state.
    fn materialization_resolver(&mut self) -> Result<&mut (dyn MaterializationResolver + '_)> {
        Err(Error::Execution(
            "runner has no materialization resolver".into(),
        ))
    }

    /// Execute a closure against an explicit sequence state instead of the
    /// runner's default session. The state is swapped in for the duration of
    /// the closure and swapped back afterwards, even on panic.
    ///
    /// The closure receives `&mut Self` so it can call any runner method
    /// (`prefill_tokens`, `decode_topk`, `feed_token`, etc.) and those methods
    /// will operate on the swapped-in sequence.
    fn with_sequence_state<T>(
        &mut self,
        state: &mut Self::SequenceState,
        execute: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T>;

    /// Create a fresh independent sequence state at position zero.
    ///
    /// Serving admission must use this hook rather than cloning the runner's
    /// default session.
    fn create_sequence_state(&mut self) -> Result<Self::SequenceState>;

    /// Create an independent sequence state forked from the runner's default
    /// session, including any model-owned continuation state.
    fn fork_sequence_state(&mut self) -> Result<Self::SequenceState>;

    /// Prepare an independent state from an explicit committed source state.
    ///
    /// This hook is model-family neutral and must not mutate `source`. Paged KV
    /// bytes remain owned by the runtime/backend page pool; implementations copy
    /// only model-owned continuation metadata needed at `expected_position`.
    fn fork_sequence_state_from(
        &mut self,
        source: &Self::SequenceState,
        expected_position: usize,
    ) -> Result<Self::SequenceState>;

    /// Reset a sequence state for reuse with a new logical sequence.
    fn reset_sequence_state(&mut self, state: &mut Self::SequenceState) -> Result<()>;

    /// Release a sequence state and its physical capacity.
    fn release_sequence_state(&mut self, state: Self::SequenceState) -> Result<()>;

    /// Configure the maximum number of backend physical KV pages.
    fn configure_kv_page_capacity(&mut self, max_pages: usize) -> Result<()>;

    /// Release backend physical slots or suspended snapshots after runtime
    /// refcounts reach zero.
    fn release_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()>;

    /// Move exclusively owned pages out of backend device residency. Backends
    /// retain any opaque host snapshots required by `restore_kv_pages`.
    fn preempt_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()>;

    /// Restore pages previously moved out of backend device residency.
    fn restore_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()>;

    /// Reserve backend-owned physical resources for one packed batch.
    ///
    /// A successful prepare remains provisional until
    /// [`commit_multi_session_batch`](Self::commit_multi_session_batch). Backends
    /// must leave the previously committed view unchanged when prepare fails.
    fn prepare_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
        kv_reservations: &[KvReservationView],
    ) -> Result<()>;

    /// Atomically publish backend resources prepared for the current batch.
    ///
    /// Runtime logical KV metadata is published immediately after this returns
    /// successfully. An error must mean the backend transaction is still
    /// uncommitted and can be rolled back with `rollback_multi_session_batch`.
    fn commit_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
    ) -> Result<()>;

    /// Discard resources prepared for the current batch. This must be safe after
    /// model execution fails and must restore the previous committed KV view.
    fn rollback_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
    ) -> Result<()>;

    /// Atomically retain independent exact prefixes for a provisional cohort.
    ///
    /// Full-width entries are already exact and require no restoration. Failure
    /// must be reported before mutating any branch.
    fn retain_provisional_prefixes(
        &mut self,
        transaction: ExecutionTransactionId,
        sources: &[Self::SequenceState],
        branches: &mut [Self::SequenceState],
        executed_rows: &[usize],
        retained_rows: &[usize],
    ) -> Result<()>;

    /// Begin native packed execution and report whether it completed or retained
    /// a continuation waiting on model-owned resource work.
    ///
    /// This is the only multi-session execution path. Implementations must consume
    /// authoritative row positions, page tables, and KV write slots directly from
    /// `batch`; there is no runtime serial or single-token fallback.
    fn execute_multi_session_batch_progress(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
    ) -> Result<MultiSessionBatchProgress>;

    /// Resume a batch previously reported as waiting by this runner.
    ///
    /// Returning `Err` must leave the same continuation active and cancellable.
    /// A runner that reaches quiescence with an error must report that ownership
    /// transition through a future structured progress result rather than losing
    /// the continuation implicitly.
    ///
    fn resume_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
        continuation: ContinuationId,
        leases: ResidencyLeaseSet,
    ) -> Result<MultiSessionBatchProgress>;

    /// Cancel a batch continuation previously created by this runner.
    ///
    /// Implementations must explicitly report whether an error happened before or
    /// after quiescence. Callers retain state ownership only for
    /// [`BatchContinuationCancelOutcome::StillActive`].
    fn cancel_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        continuation: ContinuationId,
    ) -> BatchContinuationCancelOutcome;

    /// Truthful capabilities for multi-session execution. This should report
    /// `max_sequences > 1` and `supports_mixed` accurately for this backend.
    fn multi_session_capabilities(&self) -> ExecutionCapabilities;
}

/// Immutable identity of one prepared checkpoint-native proposal source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeProposalSource {
    /// Stable implementation/protocol name, independent of a request or session.
    pub implementation: &'static str,
    /// Immutable prepared-plan generation within this process. This is not a
    /// checkpoint content hash; release manifests must provide that separately.
    pub prepared_plan_id: u64,
    /// Number of draft tokens produced by the native checkpoint block.
    pub native_width: usize,
}

impl NativeProposalSource {
    pub fn validate(&self) -> Result<()> {
        if self.implementation.is_empty() || self.prepared_plan_id == 0 || self.native_width == 0 {
            return Err(ferrule_common::Error::Model(format!(
                "invalid native proposal source: implementation={:?} prepared_plan_id={} native_width={}",
                self.implementation, self.prepared_plan_id, self.native_width
            )));
        }
        Ok(())
    }
}

/// One checkpoint-native proposal block.
///
/// `token_ids` are ordered draft candidates after the carried target anchor.
/// Confidence values use the same row order and remain telemetry until an exact
/// confidence admission policy is enabled.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeProposal {
    pub token_ids: Vec<u32>,
    pub confidence_logits: Vec<f32>,
}

impl NativeProposal {
    pub fn validate(&self) -> Result<()> {
        if self.token_ids.len() != self.confidence_logits.len() {
            return Err(ferrule_common::Error::Model(format!(
                "native proposal returned {} tokens but {} confidence logits",
                self.token_ids.len(),
                self.confidence_logits.len()
            )));
        }
        if let Some((row, confidence)) = self
            .confidence_logits
            .iter()
            .enumerate()
            .find(|(_, confidence)| !confidence.is_finite())
        {
            return Err(ferrule_common::Error::Model(format!(
                "native proposal confidence row {row} is not finite: {confidence}"
            )));
        }
        Ok(())
    }

    pub fn validate_for_source(&self, source: NativeProposalSource) -> Result<()> {
        source.validate()?;
        self.validate()?;
        if self.token_ids.len() != source.native_width {
            return Err(ferrule_common::Error::Model(format!(
                "native proposal source {}:{} declares native width {} but returned {} tokens",
                source.implementation,
                source.prepared_plan_id,
                source.native_width,
                self.token_ids.len()
            )));
        }
        Ok(())
    }
}

/// Model capability consumed by the resident inference scheduler.
///
/// Checkpoint-native speculative proposals are optional: models without that
/// capability return `None` and execute the same packed target path without a
/// serving-side special case.
pub trait ResidentModelRunner: MultiSessionRunner {
    type ObservabilitySnapshot: Clone;

    /// Return one model-owned typed observability snapshot. Runtime transports the
    /// associated type without knowing model-specific fields.
    fn observability_snapshot(&self) -> Self::ObservabilitySnapshot;

    /// Return the cloneable wake hub shared by every model-side completion producer.
    fn completion_hub(&self) -> CompletionHub;

    /// Transfer ownership of all model-side completion reactors to the caller.
    ///
    /// Each reactor is returned at most once. The futures are `'static`, do not
    /// borrow the runner, and may be polled on a local (non-`Send`) task set.
    fn take_completion_reactors(&mut self) -> Vec<ModelCompletionReactor>;

    fn native_proposal_source(&self) -> Result<Option<NativeProposalSource>>;

    /// Begin a checkpoint-native proposal without synchronously waiting for
    /// model-owned I/O or device result transfers.
    fn begin_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        anchor_token_id: u32,
    ) -> Result<NativeProposalProgress>;

    /// Resume a proposal previously returned as [`NativeProposalProgress::Waiting`].
    /// Returning `Err` leaves the same continuation owned by the runner and
    /// cancellable by the caller.
    fn resume_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        continuation: ContinuationId,
        leases: ResidencyLeaseSet,
    ) -> Result<NativeProposalProgress>;

    /// Cancel a retained native proposal continuation.
    fn cancel_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        continuation: ContinuationId,
    ) -> BatchContinuationCancelOutcome;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_source(native_width: usize) -> NativeProposalSource {
        NativeProposalSource {
            implementation: "test-native-v1",
            prepared_plan_id: 0x1234,
            native_width,
        }
    }

    fn materialization_key(expert: u32) -> ferrule_common::MaterializationKey {
        ferrule_common::MaterializationKey::new(
            ferrule_common::ModelInstanceId::new(1),
            ferrule_common::SourceIdentityHash::new([2; 32]),
            ferrule_common::ContentHash::new([3; 32]),
            ferrule_common::MaterializedResourceId::routed_expert(
                ferrule_common::LayerId::new(7),
                ferrule_common::ExpertId::new(expert),
            ),
            ferrule_common::PayloadEncodingId::new(1),
            ferrule_common::BackendId::new(4),
            ferrule_common::DeviceId::new(5),
            ferrule_common::SourceGeneration::new(6),
            ferrule_common::DestinationGeneration::new(8),
        )
        .unwrap()
    }

    #[test]
    fn pending_progress_rejects_zero_continuation_and_exposes_canonical_dependencies() {
        let transaction = ExecutionTransactionId::new(7).unwrap();
        let dependencies = crate::materialization::resource_dependency_set([
            materialization_key(8),
            materialization_key(1),
            materialization_key(8),
        ])
        .unwrap();
        let error =
            PendingModelProgress::new(transaction, ContinuationId::new(0), dependencies.clone())
                .unwrap_err()
                .to_string();
        assert!(error.contains("continuation ID must be non-zero"));

        let continuation = ContinuationId::new(29);
        let pending =
            PendingModelProgress::new(transaction, continuation, dependencies.clone()).unwrap();
        assert_eq!(pending.transaction(), transaction);
        assert_eq!(pending.continuation(), continuation);
        assert_eq!(pending.unresolved_dependencies(), &dependencies);
        assert_eq!(pending.dependencies().len(), 2);
        assert_eq!(
            NativeProposalProgress::Waiting(pending.clone()),
            NativeProposalProgress::Waiting(pending)
        );
    }

    #[test]
    fn native_proposal_validates_native_width_and_finite_confidence() {
        let proposal = NativeProposal {
            token_ids: vec![10, 11],
            confidence_logits: vec![0.5, -0.25],
        };
        proposal.validate_for_source(native_source(2)).unwrap();

        let width_error = proposal
            .validate_for_source(native_source(3))
            .unwrap_err()
            .to_string();
        assert!(width_error.contains("declares native width 3"));

        let invalid_confidence = NativeProposal {
            token_ids: vec![10],
            confidence_logits: vec![f32::NAN],
        }
        .validate_for_source(native_source(1))
        .unwrap_err()
        .to_string();
        assert!(invalid_confidence.contains("not finite"));
    }

    #[test]
    fn native_proposal_rejects_missing_source_identity() {
        let proposal = NativeProposal {
            token_ids: vec![10],
            confidence_logits: vec![0.0],
        };
        let error = proposal
            .validate_for_source(NativeProposalSource {
                implementation: "",
                prepared_plan_id: 0,
                native_width: 1,
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid native proposal source"));
    }

    #[test]
    fn completion_callback_publishes_to_the_shared_hub() {
        let hub = CompletionHub::new();
        let callback = completion_notify_callback(hub.clone());
        assert_eq!(hub.epoch(), 0);

        callback();

        assert_eq!(hub.epoch(), 1);
    }
}
