//! Graph-facing runtime contracts.
//!
//! This module contains the runtime abstractions that sit between Ferrule's
//! model/session scheduler and the device-independent `ferrule-graph` IR. They
//! are intentionally lightweight and compile-only at this stage: legacy runners
//! keep their existing token-by-token paths, while new graph-backed runners can
//! adopt these contracts incrementally.
//!
//! ## Design 1: `ExecutionBatch`
//!
//! `ExecutionBatch` is Ferrule's graph-facing execution envelope. It groups the
//! data that a backend needs for one prefill/decode step:
//!
//! - token id
//! - logical position
//! - session id
//! - optional KV handle
//! - whether the row must produce logits
//! - execution segment: prefill, decode, or mixed
//!
//! This is conceptually similar to a llama.cpp-style batch, but expressed in
//! Ferrule's runtime vocabulary. The main benefit is that graph runners no
//! longer need to be designed around a single-token `decode_token()` API. The
//! same boundary can grow into chunked prefill, continuous batching, speculative
//! decode, paged KV, and mixed prefill/decode scheduling without rewriting every
//! model-family runner.
//!
//! Advantages:
//!
//! - Keeps positions, sessions, KV handles, and logits selection explicit.
//! - Allows schedulers to form multi-row decode or mixed prefill/decode batches.
//! - Avoids producing logits for rows that do not need them.
//! - Gives future graph backends a stable input shape independent of a concrete
//!   model family.
//! - Preserves existing legacy runner behavior because this module does not
//!   replace `ModelRunner`; graph runners can opt in later.
//!
//! ## Design 2: `ExternalBindingPlan`
//!
//! `ferrule-graph` stores only opaque `ExternalKey`s. It should not own CUDA
//! buffers, mmap-backed tensors, WeightPack objects, KV pages, or resident expert
//! handles. `ExternalBindingPlan` is the runtime-side bridge from semantic model
//! roles and runtime state to those graph externals.
//!
//! A binding records:
//!
//! - the graph `ExternalKey`
//! - the kind of runtime object, such as weight, KV state, source tensor, or
//!   resident expert
//! - optional `TensorRole` and layer metadata
//! - shape/type metadata used by graph validation
//! - preferred residency, such as host, device, streamable, paged, or
//!   backend-managed
//!
//! Advantages:
//!
//! - Keeps raw Hugging Face tensor names and backend storage details out of graph
//!   translators.
//! - Lets CUDA/CPU/source backends decide upload, residency, eviction, and
//!   streamable expert policy behind one semantic binding layer.
//! - Makes weights, KV cache, adapters, speculation state, and resident expert
//!   handles visible to the runtime without embedding them in the graph IR.
//! - Provides a natural hook for future graph-program compilation, memory
//!   planning, CUDA graph capture, and autograd state management.
//!
//! ## Migration boundary
//!
//! These contracts are intentionally additive. Existing OLMoE CUDA and DeepSeekV4
//! paths should continue working unchanged. The expected migration path is:
//!
//! 1. Build graph programs from model-family translators using semantic external
//!    keys.
//! 2. Bind those keys through `ExternalBindingPlan`.
//! 3. Execute graph programs with `ExecutionBatch` inputs.
//! 4. Keep legacy runners until graph-backed parity and performance are proven.

use ferrule_core::{Error, Result};
use ferrule_graph::{ExternalKey, ValueMeta};
use ferrule_model::TensorRole;

use crate::kv::KvHandle;
use crate::session::SessionId;

/// High-level phase represented by an execution batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionSegment {
    Prefill,
    Decode,
    /// Mixed batches allow schedulers to interleave chunked prefill and decode.
    Mixed,
}

/// Which rows should produce logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitsSelection {
    None,
    Last,
    All,
}

/// One row in an execution batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionRow {
    pub token_id: u32,
    pub position: usize,
    pub session_id: SessionId,
    pub kv_handle: Option<KvHandle>,
    pub require_logits: bool,
}

impl ExecutionRow {
    pub fn new(
        token_id: u32,
        position: usize,
        session_id: SessionId,
        kv_handle: Option<KvHandle>,
        require_logits: bool,
    ) -> Self {
        Self {
            token_id,
            position,
            session_id,
            kv_handle,
            require_logits,
        }
    }
}

/// Runtime execution envelope for graph-backed runners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBatch {
    pub segment: ExecutionSegment,
    rows: Vec<ExecutionRow>,
}

impl ExecutionBatch {
    pub fn new(segment: ExecutionSegment, rows: Vec<ExecutionRow>) -> Result<Self> {
        if rows.is_empty() {
            return Err(Error::Internal(
                "execution batch must contain at least one row".into(),
            ));
        }
        Ok(Self { segment, rows })
    }

    pub fn from_tokens(
        segment: ExecutionSegment,
        session_id: SessionId,
        start_position: usize,
        tokens: &[u32],
        kv_handle: Option<KvHandle>,
        logits: LogitsSelection,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(Error::Internal(
                "execution batch token slice must not be empty".into(),
            ));
        }
        let last_index = tokens.len() - 1;
        let rows = tokens
            .iter()
            .copied()
            .enumerate()
            .map(|(index, token_id)| {
                let require_logits = match logits {
                    LogitsSelection::None => false,
                    LogitsSelection::Last => index == last_index,
                    LogitsSelection::All => true,
                };
                ExecutionRow::new(
                    token_id,
                    start_position + index,
                    session_id,
                    kv_handle,
                    require_logits,
                )
            })
            .collect();
        Self::new(segment, rows)
    }

    pub fn prefill_last_logits(
        session_id: SessionId,
        start_position: usize,
        tokens: &[u32],
        kv_handle: Option<KvHandle>,
    ) -> Result<Self> {
        Self::from_tokens(
            ExecutionSegment::Prefill,
            session_id,
            start_position,
            tokens,
            kv_handle,
            LogitsSelection::Last,
        )
    }

    pub fn decode_one(
        session_id: SessionId,
        position: usize,
        token_id: u32,
        kv_handle: Option<KvHandle>,
    ) -> Result<Self> {
        Self::new(
            ExecutionSegment::Decode,
            vec![ExecutionRow::new(
                token_id, position, session_id, kv_handle, true,
            )],
        )
    }

    pub fn rows(&self) -> &[ExecutionRow] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn token_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.rows.iter().map(|row| row.token_id)
    }

    pub fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.rows.iter().map(|row| row.position)
    }

    pub fn logits_rows(&self) -> impl Iterator<Item = (usize, &ExecutionRow)> + '_ {
        self.rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.require_logits)
    }
}

/// Runtime/source object class represented by a graph external.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalBindingKind {
    Weight,
    KvState,
    SourceTensor,
    ResidentExpert,
    Adapter,
    Speculation,
    Other(String),
}

/// Preferred residency for an external object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalResidency {
    Host,
    Device,
    Streamable,
    Paged,
    BackendManaged,
}

/// One semantic binding for a graph external.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalBinding {
    pub key: ExternalKey,
    pub kind: ExternalBindingKind,
    pub role: Option<TensorRole>,
    pub layer: Option<usize>,
    pub meta: ValueMeta,
    pub residency: ExternalResidency,
}

impl ExternalBinding {
    pub fn weight(
        key: ExternalKey,
        role: TensorRole,
        layer: Option<usize>,
        meta: ValueMeta,
        residency: ExternalResidency,
    ) -> Self {
        Self {
            key,
            kind: ExternalBindingKind::Weight,
            role: Some(role),
            layer,
            meta,
            residency,
        }
    }

    pub fn state(
        key: ExternalKey,
        kind: ExternalBindingKind,
        meta: ValueMeta,
        residency: ExternalResidency,
    ) -> Self {
        Self {
            key,
            kind,
            role: None,
            layer: None,
            meta,
            residency,
        }
    }
}

/// Graph-facing binding plan produced by model/runtime translators.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExternalBindingPlan {
    entries: Vec<ExternalBinding>,
}

impl ExternalBindingPlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> &[ExternalBinding] {
        &self.entries
    }

    pub fn push(&mut self, binding: ExternalBinding) -> Result<()> {
        if self.entries.iter().any(|entry| entry.key == binding.key) {
            return Err(Error::Graph(format!(
                "duplicate external binding '{}:{}'",
                binding.key.namespace(),
                binding.key.name()
            )));
        }
        self.entries.push(binding);
        Ok(())
    }

    pub fn get(&self, key: &ExternalKey) -> Option<&ExternalBinding> {
        self.entries.iter().find(|entry| &entry.key == key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use ferrule_graph::{DataType, ValueMeta};

    use super::*;

    #[test]
    fn prefill_batch_marks_only_last_logits() {
        let batch =
            ExecutionBatch::prefill_last_logits(SessionId(7), 10, &[1, 2, 3], Some(KvHandle(4)))
                .unwrap();

        assert_eq!(batch.segment, ExecutionSegment::Prefill);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.positions().collect::<Vec<_>>(), vec![10, 11, 12]);
        assert_eq!(batch.token_ids().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(
            batch.logits_rows().map(|(i, _)| i).collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(batch.rows()[0].kv_handle, Some(KvHandle(4)));
    }

    #[test]
    fn decode_batch_marks_row_logits() {
        let batch = ExecutionBatch::decode_one(SessionId(1), 5, 42, None).unwrap();
        assert_eq!(batch.segment, ExecutionSegment::Decode);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.rows()[0].token_id, 42);
        assert!(batch.rows()[0].require_logits);
    }

    #[test]
    fn empty_batch_is_rejected() {
        let err = ExecutionBatch::from_tokens(
            ExecutionSegment::Prefill,
            SessionId(1),
            0,
            &[],
            None,
            LogitsSelection::Last,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("must not be empty"));
    }

    #[test]
    fn binding_plan_rejects_duplicate_keys() {
        let key = ExternalKey::new("weights", "layer0.q").unwrap();
        let binding = ExternalBinding::weight(
            key.clone(),
            TensorRole::AttentionQuery,
            Some(0),
            ValueMeta::tensor(DataType::Bf16, [16, 16]),
            ExternalResidency::Device,
        );

        let mut plan = ExternalBindingPlan::new();
        plan.push(binding.clone()).unwrap();
        let err = plan.push(binding).unwrap_err();
        assert!(format!("{err}").contains("duplicate external binding"));
    }

    #[test]
    fn binding_plan_finds_entries_by_key() {
        let key = ExternalKey::new("state", "kv").unwrap();
        let binding = ExternalBinding::state(
            key.clone(),
            ExternalBindingKind::KvState,
            ValueMeta::external_state("kv"),
            ExternalResidency::BackendManaged,
        );

        let mut plan = ExternalBindingPlan::new();
        plan.push(binding).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.get(&key).unwrap().kind, ExternalBindingKind::KvState);
    }
}
