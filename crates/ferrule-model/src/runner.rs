use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionOutput, KvReservation,
};

use crate::{EnginePlan, IncrementalDecodeState, ModelDescriptor};
use ferrule_common::Result;
pub use ferrule_common::execution::TokenLogit;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillMode {
    /// Correctness-first prompt processing. Backends may use a true batched
    /// prefill implementation or a reference fallback.
    Batched,
    /// Append all prompt tokens except the last without materializing logits,
    /// then return top-k for the last token. This is useful for interactive chat
    /// and resident serving workers.
    Interactive,
}

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
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;
    fn decode_token(&mut self, token: u32) -> Result<Vec<f32>>;
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

/// Optional capability for model runners that can produce top-k logits without
/// materializing a full vocabulary logits vector.
///
/// `ferrule-runtime` owns generation/session algorithms over this trait; concrete
/// model families only implement the primitive operations. This keeps runtime code
/// generic and prevents it from depending on DeepSeek/Qwen/etc. runner types.
pub trait TopKModelRunner: ModelRunner {
    fn position(&self) -> usize;
    fn feed_token(&mut self, token_id: u32) -> Result<()>;

    /// Maximum top-k request the active backend can validate before mutating
    /// sequence state. Backends with a smaller kernel limit must override this.
    fn max_top_k(&self) -> usize {
        usize::MAX
    }

    /// Append prompt tokens without requiring logits for the last row.
    ///
    /// The default keeps existing implementations correct by feeding tokens one
    /// at a time. Model runners with real segment/chunked prefill should override
    /// this so runtime can execute non-final prefill chunks without materializing
    /// hidden states or lm_head logits.
    fn prefill_tokens(&mut self, token_ids: &[u32], _mode: PrefillMode) -> Result<()> {
        for &token_id in token_ids {
            self.feed_token(token_id)?;
        }
        Ok(())
    }

    fn prefill_topk(
        &mut self,
        token_ids: &[u32],
        top_k: usize,
        mode: PrefillMode,
    ) -> Result<Vec<TokenLogit>>;
    fn decode_topk(&mut self, token_id: u32, top_k: usize) -> Result<Vec<TokenLogit>>;
}

/// A model runner that supports explicit per-sequence state for multi-session
/// execution.
///
/// This trait extends [`TopKModelRunner`] with the ability to swap sequence state
/// in and out of the runner, enabling multiple independent sequences to share
/// one set of prepared resources (weights, expert residency, arenas).
///
/// Models implement this trait; the runtime crate provides a generic
/// [`NativeMultiSessionExecutor`] that works over any `MultiSessionRunner`.
///
/// [`NativeMultiSessionExecutor`]: ../../ferrule_runtime/engine/struct.NativeMultiSessionExecutor.html
pub trait MultiSessionRunner: TopKModelRunner {
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
    /// Serving admission should use this hook rather than cloning the runner's
    /// default session. The default preserves compatibility for runners that do
    /// not yet distinguish fresh construction from an explicit state fork.
    fn create_sequence_state(&mut self) -> Result<Self::SequenceState> {
        self.fork_sequence_state()
    }

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

    /// Configure the maximum number of backend physical KV pages. Runners that
    /// own a physical page pool override this hook; metadata-only runners ignore it.
    fn configure_kv_page_capacity(&mut self, _max_pages: usize) -> Result<()> {
        Ok(())
    }

    /// Release backend physical slots or suspended snapshots after runtime
    /// refcounts reach zero.
    fn release_kv_pages(&mut self, _pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        Ok(())
    }

    /// Move exclusively owned pages out of backend device residency. Backends
    /// retain any opaque host snapshots required by `restore_kv_pages`.
    fn preempt_kv_pages(&mut self, _pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        Ok(())
    }

    /// Restore pages previously moved out of backend device residency.
    fn restore_kv_pages(&mut self, _pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        Ok(())
    }

    /// Reserve backend-owned physical resources for one packed batch.
    ///
    /// A successful prepare remains provisional until
    /// [`commit_multi_session_batch`](Self::commit_multi_session_batch). Backends
    /// must leave the previously committed view unchanged when prepare fails.
    fn prepare_multi_session_batch(
        &mut self,
        _states: &mut [Self::SequenceState],
        _batch: &ExecutionBatch,
        _kv_reservations: &[KvReservation],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Publish resources prepared for the current batch after the runtime KV
    /// transaction has committed.
    fn commit_multi_session_batch(&mut self) -> Result<()> {
        Ok(())
    }

    /// Discard resources prepared for the current batch. This must be safe after
    /// model execution fails and must restore the previous committed KV view.
    fn rollback_multi_session_batch(&mut self) -> Result<()> {
        Ok(())
    }

    /// Execute a complete packed multi-session batch in one backend pipeline.
    ///
    /// Model families with native ragged/mixed execution override this method and
    /// consume the authoritative row positions, page tables, and KV write slots
    /// directly from `batch`. Returning `Ok(None)` selects the generic serial
    /// correctness path in `ferrule-runtime`; it must not be reported as native
    /// packed execution by performance gates.
    fn execute_multi_session_batch(
        &mut self,
        _states: &mut [Self::SequenceState],
        _batch: &ExecutionBatch,
    ) -> Result<Option<ExecutionOutput>> {
        Ok(None)
    }

    /// Truthful capabilities for multi-session execution. This should report
    /// `max_sequences > 1` and `supports_mixed` accurately for this backend.
    fn multi_session_capabilities(&self) -> ExecutionCapabilities;
}

/// Optional model-owned expert-I/O oracle consumed by the generic runtime
/// scheduler. Implementations keep route prediction and cache interpretation in
/// the model crate while exposing only model-neutral cost estimates.
pub trait ExpertIoModelRunner: MultiSessionRunner {
    type ExpertIoBatchState;
    type ExpertIoAdmission;

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

// ── Engine-plan helpers ──────────────────────────────────────────────────

pub fn unsupported_runtime_message(plan: &EnginePlan) -> String {
    let mut message = format!(
        "{} metadata is recognized, but the current executable backend cannot run it yet (engine plan: {})",
        plan.family, plan.status
    );
    if !plan.missing.is_empty() {
        message.push_str(". Missing policies:");
        for item in &plan.missing {
            message.push_str(&format!(" {}: {};", item.area, item.reason));
        }
    }
    message
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AttentionKind, ModelFamily, ModelSupportContract, MoeSpec, QuantFormatCount, RouterKind,
        TransformerSpec, WeightSource,
    };

    #[test]
    fn unsupported_runtime_message_reports_engine_plan_gaps() {
        let spec = TransformerSpec {
            family: ModelFamily::DeepSeekV4,
            architecture: Some("deepseek4".into()),
            weight_source: WeightSource::Gguf,
            hidden_size: Some(7168),
            num_layers: Some(1),
            vocab_size: Some(129280),
            num_heads: Some(128),
            num_kv_heads: None,
            head_dim: None,
            attention: AttentionKind::MultiLatentAttention,
            moe: MoeSpec {
                num_experts: Some(256),
                num_experts_per_tok: Some(8),
                has_shared_experts: true,
                router: RouterKind::HashAssistedTopK,
            },
            semantics: Default::default(),
            tensor_count: Some(1),
            quantization: vec![QuantFormatCount {
                format: "Q4_K".into(),
                tensors: 1,
            }],
            notes: Vec::new(),
        };
        let plan = ModelSupportContract::from_spec(&spec, &[]).engine_plan();
        let message = unsupported_runtime_message(&plan);
        assert!(message.contains("DeepSeek-V4"));
        assert!(message.contains("engine plan: metadata-only"));
        assert!(message.contains("attention:"));
        assert!(message.contains("quantization:"));
        assert!(message.contains("router:"));
    }
}
