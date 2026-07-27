use std::future::Future;
use std::pin::Pin;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionOutput, ExecutionTransactionId,
    KvReservationView,
};

use crate::{IncrementalDecodeState, ModelDescriptor};
pub use ferrule_common::execution::TokenLogit;
use ferrule_common::{CompletionHub, Error, Result};

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

/// Opaque identity for one resumable packed-batch continuation.
///
/// IDs are scoped to the runner that created them. Callers must pass the same ID
/// back to that runner when resuming or cancelling a waiting batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchContinuationId(std::num::NonZeroU64);

impl BatchContinuationId {
    /// Construct a continuation ID, rejecting zero because it is reserved as an
    /// invalid or absent identity.
    pub fn new(value: u64) -> Result<Self> {
        std::num::NonZeroU64::new(value)
            .map(Self)
            .ok_or_else(|| Error::Execution("batch continuation ID must be non-zero".into()))
    }

    /// Return the validated non-zero numeric identity.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// One model-neutral expert load operation blocking a resumable batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PendingExpertLoad {
    operation_id: u64,
    layer: u32,
    expert: u32,
}

impl PendingExpertLoad {
    /// Construct a pending expert load.
    ///
    /// `operation_id` identifies the durable load operation owned by the runner
    /// and must be non-zero. Layer and expert numbers are model-relative indices.
    pub fn new(operation_id: u64, layer: u32, expert: u32) -> Result<Self> {
        if operation_id == 0 {
            return Err(Error::Execution(
                "pending expert load operation ID must be non-zero".into(),
            ));
        }
        Ok(Self {
            operation_id,
            layer,
            expert,
        })
    }

    /// Return the validated non-zero load operation identity.
    pub const fn operation_id(self) -> u64 {
        self.operation_id
    }

    /// Return the model-relative layer index.
    pub const fn layer(self) -> u32 {
        self.layer
    }

    /// Return the model-relative expert index within the layer.
    pub const fn expert(self) -> u32 {
        self.expert
    }
}

/// Model-owned wait state for an asynchronous operation.
///
/// An empty expert list means the continuation is waiting on non-expert model
/// work, such as a device-to-host result transfer. Expert operation IDs must be
/// non-zero and unique within the continuation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingModelProgress {
    transaction: ExecutionTransactionId,
    continuation: BatchContinuationId,
    expert_loads: Vec<PendingExpertLoad>,
}

impl PendingModelProgress {
    /// Construct a validated model wait state. `expert_loads` may be empty when
    /// progress depends on another model-owned completion source.
    pub fn new(
        transaction: ExecutionTransactionId,
        continuation: BatchContinuationId,
        expert_loads: Vec<PendingExpertLoad>,
    ) -> Result<Self> {
        let mut operation_ids = std::collections::HashSet::with_capacity(expert_loads.len());
        for operation in &expert_loads {
            if operation.operation_id == 0 {
                return Err(Error::Execution(
                    "pending expert load operation ID must be non-zero".into(),
                ));
            }
            if !operation_ids.insert(operation.operation_id) {
                return Err(Error::Execution(format!(
                    "pending expert load operation ID {} is duplicated",
                    operation.operation_id
                )));
            }
        }
        Ok(Self {
            transaction,
            continuation,
            expert_loads,
        })
    }

    /// Return the stable end-to-end transaction owning this wait.
    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.transaction
    }

    /// Return the continuation to pass to the corresponding resume or cancel API.
    pub const fn continuation(&self) -> BatchContinuationId {
        self.continuation
    }

    /// Return expert loads currently blocking progress. This may be empty when
    /// the model is waiting on another asynchronous completion source.
    pub fn expert_loads(&self) -> &[PendingExpertLoad] {
        &self.expert_loads
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
        continuation: BatchContinuationId,
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
        continuation: BatchContinuationId,
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

/// Optional model-owned expert-I/O oracle consumed by the generic runtime
/// scheduler. Implementations keep route prediction and cache interpretation in
/// the model crate while exposing only model-neutral cost estimates.
pub trait ExpertIoModelRunner: MultiSessionRunner {
    type ExpertIoBatchState;
    type ExpertIoAdmission;

    /// Exact physical capacity of the hardware/model expert materialization path.
    fn expert_io_resource_limits(
        &self,
    ) -> Result<ferrule_common::expert_io::ExpertIoResourceLimits>;

    /// Install the unique runtime-owned hard-admission service before execution.
    fn install_expert_io_resource_control(
        &mut self,
        control: Box<dyn ferrule_common::expert_io::ExpertIoResourceControl>,
    ) -> Result<()>;

    /// Remove the runtime-owned admission service before returning a quiescent
    /// runner to its caller. A later driver may then install a fresh service.
    fn uninstall_expert_io_resource_control(&mut self) -> Result<()>;

    fn begin_expert_io_batch(&self) -> Self::ExpertIoBatchState;

    fn estimate_expert_io(
        &self,
        batch: &mut Self::ExpertIoBatchState,
        sequence: &Self::SequenceState,
        phase: ferrule_common::expert_io::ExpertIoPhase,
        token_ids: &[u32],
    ) -> Result<(
        ferrule_common::expert_io::ExpertIoEstimate,
        Self::ExpertIoAdmission,
    )>;

    fn admit_expert_io(
        &self,
        batch: &mut Self::ExpertIoBatchState,
        admission: Self::ExpertIoAdmission,
    );
}

/// Model capability consumed by the resident inference scheduler.
///
/// Expert-I/O estimation is mandatory for resident scheduling. Checkpoint-native
/// speculative proposals are optional: models without that capability return
/// `None` and execute the same packed target path without a serving-side special
/// case.
pub trait ResidentModelRunner: ExpertIoModelRunner {
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
        continuation: BatchContinuationId,
    ) -> Result<NativeProposalProgress>;

    /// Cancel a retained native proposal continuation.
    fn cancel_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        continuation: BatchContinuationId,
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

    #[test]
    fn batch_continuation_id_rejects_zero_and_round_trips_non_zero_values() {
        let error = BatchContinuationId::new(0).unwrap_err().to_string();
        assert!(error.contains("continuation ID must be non-zero"));

        let continuation = BatchContinuationId::new(42).unwrap();
        assert_eq!(continuation.get(), 42);
    }

    #[test]
    fn pending_expert_load_validates_operation_id_and_exposes_coordinates() {
        let error = PendingExpertLoad::new(0, 3, 7).unwrap_err().to_string();
        assert!(error.contains("operation ID must be non-zero"));

        let operation = PendingExpertLoad::new(11, 3, 7).unwrap();
        assert_eq!(operation.operation_id(), 11);
        assert_eq!(operation.layer(), 3);
        assert_eq!(operation.expert(), 7);
    }

    #[test]
    fn pending_model_progress_allows_empty_work_and_rejects_duplicate_operations() {
        let transaction = ExecutionTransactionId::new(7).unwrap();
        let continuation = BatchContinuationId::new(23).unwrap();
        let empty = PendingModelProgress::new(transaction, continuation, Vec::new()).unwrap();
        assert_eq!(empty.transaction(), transaction);
        assert_eq!(empty.continuation(), continuation);
        assert!(empty.expert_loads().is_empty());

        let duplicate = PendingModelProgress::new(
            transaction,
            continuation,
            vec![
                PendingExpertLoad::new(31, 2, 4).unwrap(),
                PendingExpertLoad::new(31, 3, 5).unwrap(),
            ],
        )
        .unwrap_err()
        .to_string();
        assert!(duplicate.contains("operation ID 31 is duplicated"));
    }

    #[test]
    fn pending_model_progress_preserves_expert_load_order() {
        let transaction = ExecutionTransactionId::new(11).unwrap();
        let continuation = BatchContinuationId::new(29).unwrap();
        let expert_loads = vec![
            PendingExpertLoad::new(41, 7, 1).unwrap(),
            PendingExpertLoad::new(42, 7, 8).unwrap(),
        ];
        let pending =
            PendingModelProgress::new(transaction, continuation, expert_loads.clone()).unwrap();
        assert_eq!(pending.transaction(), transaction);
        assert_eq!(pending.expert_loads(), expert_loads.as_slice());
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
