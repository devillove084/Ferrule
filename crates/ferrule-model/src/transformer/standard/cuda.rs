//! Synchronous, owner-local standard CUDA decoder operators.
//!
//! Default execution retains F32 weights/activations/KV. The explicit dense BF16
//! profile uses native BF16 storage with declared F32 reductions/work buffers.

mod hybrid;
mod hybrid_rank;
pub use hybrid_rank::{
    CudaHybridRankLifecycle, CudaHybridRankMemory, CudaHybridRankSegment, CudaHybridRankState,
    CudaMixedRankLinear,
};
mod recurrent;
pub use super::decoded_snapshot::{
    MixedDecodedCacheActivity, MixedDecodedCachePhase, MixedDecodedCacheSnapshot,
};
pub use hybrid::CudaHybridModule;
mod bf16;
mod bf16_attention;
mod bf16_chunk;
mod bf16_hybrid;
mod binding;
mod boundary;
mod diagnostic;
pub use diagnostic::{
    CudaDiagnosticBuffer, CudaDiagnosticDescriptor, CudaDiagnosticDtype, CudaDiagnosticEvent,
    CudaDiagnosticLayout, CudaDiagnosticLimits, CudaDiagnosticSequence, CudaNativeDiagnosticEvent,
};
mod committed_gqa;
mod expert;
mod kv_binding;
pub use committed_gqa::{CommittedGqa, CommittedGqaOutput, CommittedGqaRequest};
mod numeric_binding;
mod rank_performance;
mod rank_timing;
mod residency;
use super::performance;
mod root_execution;
mod root_submission;
mod upload_proof;
pub(in crate::transformer) use root_submission::BarrierReason;
mod segment;
mod stage;
pub use segment::CudaHybridDecoderSegment;

use super::expert_cache::{ExpertCachePolicy, ExpertCacheStats};
pub use expert::{CudaExpertParallelRoutedExecutor, CudaExpertWorker, CudaHostRoutedExecutor};
pub use kv_binding::CudaStandardKvBinding;
pub type NumericWorkspaceStats = numeric_binding::NumericWorkspaceStats;
pub use stage::CudaStandardDecoderSegment;

use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use super::expert_cache::{Cache as BoundedExpertCache, ExpertCacheKey};
use diagnostic::DiagnosticTrace;
use ferrule_backend::cuda::providers::DeviceBuffer;
use numeric_binding::NumericState;

use crate::transformer::parallel::{TensorParallelLinearPartition, TensorParallelStagePlan};
use crate::transformer::{StandardTensorCollective, StandardTensorPlan};
use ferrule_backend::cuda::operators::linear::{CudaF32Buffer, CudaOperators};
use ferrule_backend::cuda::operators::{SelectedSoftmaxTopKLayout, SplitHalfRopeLayout};
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, ParallelRankId, Result};

use crate::execution::ExecutionPrecisionPolicy;
use crate::transformer::operators::{standard_rope_unsupported, standard_router_unsupported};
use crate::transformer::{
    BoundParameter, CudaRows, ExpertAvailability, ExpertProvider, GqaRequest, HostRows, KvView,
    MoeRouterSpec, OperatorProgress, PreparedEmbedding, PreparedLinear, PreparedNorm, PreparedRope,
    PreparedSwiGlu, RotaryEmbedding, RotaryPairing, RotaryRegion, RouterRoutes, Rows, RowsArenaId,
    RowsDType, RowsShape, StandardDecoderOperators, UnsupportedOperator,
};

use binding::Bindings;
use boundary::Boundary;

/// Owner-local metadata accounting; counters never act as freshness evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertMetadataPreflightStats {
    pub prepared_experts: usize,
    pub source_snapshots: usize,
    pub preflights: u64,
    pub source_checks: u64,
}

pub struct CudaStandardDecoderOperators {
    ops: Rc<CudaOperators>,
    deltas: BTreeMap<usize, recurrent::ResidentDelta>,
    recurrent_scratch: recurrent::RecurrentScratch,
    bindings: Bindings,
    bf16: Option<bf16::State>,
    bf16_deltas: BTreeMap<usize, crate::transformer::PreparedGatedDeltaNetBlock>,
    hybrid_rank: Option<hybrid_rank::RankOperators>,
    mixed_rank_block: Option<hybrid_rank::MixedRankImage>,
    mixed_decoded_enabled: bool,
    mixed_decoded_depth: usize,
    mixed_decoded_failed: bool,
    mixed_gqa_block: Option<Rc<hybrid_rank::MixedGqaInvocation>>,
    experts: BTreeMap<(usize, usize), Arc<PreparedSwiGlu>>,
    bounded_experts: Option<BoundedExpertCache<ExpertCacheKey>>,
    retained_upload: Option<residency::retained_upload::State>,
    expert_hold: Vec<DeviceBuffer<f32>>,
    bf16_hold: Vec<DeviceBuffer<u16>>,
    bf16_chunk_workspace: Option<CudaF32Buffer>,
    bf16_gemmex: Option<ferrule_backend::cuda::operators::linear::CudaBf16GemmEx>,
    bf16_attention: Option<Rc<std::cell::RefCell<bf16_attention::PreparedAttention>>>,
    // Physical artifact roots survive closure unwind and unknown completion.
    expert_artifacts: Vec<Rc<binding::ResidentLinear>>,
    route_hold: Vec<DeviceBuffer<f32>>,
    route_ids: Vec<DeviceBuffer<i32>>,
    route_scratch_bytes: usize,
    numeric: Option<NumericState>,
    diagnostic: DiagnosticTrace,
    root_submission: Option<root_submission::SubmissionSegment<DeviceBuffer<f32>>>,
    performance_proofs: Option<super::performance::ProofTiming>,
    rank_performance_enabled: bool,
    rank_performance: Option<rank_performance::RankPerformance>,
    root_bindings: root_execution::PreparedBindings,
    #[cfg(test)]
    lose_next_consumer_proof: bool,
    #[cfg(test)]
    fault_probe: hybrid::fault_hook::FaultProbe,
    #[cfg(test)]
    consumer_proofs: usize,
    #[cfg(test)]
    numeric_allocation_failure: Option<usize>,

    boundary: Boundary,
    poisoned: bool,
    tensor_collective: Option<Box<dyn StandardTensorCollective>>,
    tensor_transaction: Option<ExecutionTransactionId>,
    tensor_watermark: Option<ExecutionTransactionId>,
    tensor_failed: bool,
}

impl fmt::Debug for CudaStandardDecoderOperators {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaStandardDecoderOperators")
            .field("ordinal", &self.ops.device_ordinal())
            .field("generation", &self.bindings.generation)
            .field("resident_bytes", &self.bindings.resident_bytes())
            .field("expert_cache", &self.expert_cache_stats())
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl CudaStandardDecoderOperators {
    /// The finite parameter directory is the authority for this image. Another
    /// checkpoint, even with the same canonical IDs, cannot use this cache.
    pub fn new(
        ops: Rc<CudaOperators>,
        precision: ExecutionPrecisionPolicy,
        parameters: &[BoundParameter],
    ) -> Result<Self> {
        Self::new_with_expert_cache(ops, precision, parameters, ExpertCachePolicy::KeepAll)
    }

    /// Construct an owner with an explicit bounded routed-expert cache. The
    /// default constructor remains the historical image-local KeepAll policy.
    ///
    /// Bounded mode currently accepts non-aliased, bias-free TP1 routed F32 or
    /// BF16 weights, executed as F32. Numeric FP8 is not bound here. The budget
    /// excludes non-expert weights, KV and allocator segment slack; leave room
    /// for them when choosing limits. Activations and results stay on this CUDA
    /// owner, and no persistent host expert payload cache is installed.
    pub fn new_with_expert_cache(
        ops: Rc<CudaOperators>,
        precision: ExecutionPrecisionPolicy,
        parameters: &[BoundParameter],
        policy: ExpertCachePolicy,
    ) -> Result<Self> {
        Self::new_with_expert_profile(ops, precision, parameters, policy, false)
    }

    fn new_with_expert_profile(
        ops: Rc<CudaOperators>,
        precision: ExecutionPrecisionPolicy,
        parameters: &[BoundParameter],
        policy: ExpertCachePolicy,
        numeric_fp8: bool,
    ) -> Result<Self> {
        validate_precision(precision)?;
        if precision == ExecutionPrecisionPolicy::bf16_compatibility()
            && (numeric_fp8 || !matches!(policy, ExpertCachePolicy::KeepAll))
        {
            return Err(cuda_error(
                "BF16 dense profile does not admit numeric or expert caches",
            ));
        }
        let mut bindings = Bindings::new(parameters)?;
        bindings.bounded_experts = matches!(policy, ExpertCachePolicy::Bounded(_));
        bindings.numeric_fp8 = numeric_fp8;
        if (bindings.bounded_experts || numeric_fp8)
            && parameters.iter().any(|p| {
                matches!(p.residency(), crate::nn::ParameterResidency::Expert { .. })
                    && p.id() != p.canonical_id()
            })
        {
            return Err(cuda_error(
                "bounded experts require non-aliased parameter bindings",
            ));
        }
        bindings.prepare_expert_metadata()?;
        let boundary = Boundary::new(&ops)?;
        let bounded_experts = match policy {
            ExpertCachePolicy::KeepAll => None,
            ExpertCachePolicy::Bounded(limits) => Some(BoundedExpertCache::new(limits)?),
        };
        Ok(Self {
            deltas: BTreeMap::new(),
            bf16_deltas: BTreeMap::new(),
            recurrent_scratch: recurrent::RecurrentScratch::default(),
            ops,
            bindings,
            bf16: (precision == ExecutionPrecisionPolicy::bf16_compatibility())
                .then(bf16::State::default),
            hybrid_rank: None,
            mixed_rank_block: None,
            mixed_decoded_enabled: false,
            mixed_decoded_depth: 0,
            mixed_decoded_failed: false,
            mixed_gqa_block: None,
            experts: BTreeMap::new(),
            bounded_experts,
            retained_upload: None,
            expert_hold: Vec::new(),
            bf16_hold: Vec::new(),
            bf16_chunk_workspace: None,
            bf16_gemmex: None,
            bf16_attention: None,
            expert_artifacts: Vec::new(),
            route_hold: Vec::new(),
            route_ids: Vec::new(),
            route_scratch_bytes: 0,
            numeric: None,
            diagnostic: DiagnosticTrace::default(),
            root_submission: None,
            performance_proofs: None,
            rank_performance_enabled: false,
            rank_performance: None,
            root_bindings: root_execution::PreparedBindings::default(),
            #[cfg(test)]
            lose_next_consumer_proof: false,
            #[cfg(test)]
            fault_probe: hybrid::fault_hook::FaultProbe::default(),
            #[cfg(test)]
            consumer_proofs: 0,
            #[cfg(test)]
            numeric_allocation_failure: None,

            boundary,
            poisoned: false,
            tensor_collective: None,
            tensor_transaction: None,
            tensor_watermark: None,
            tensor_failed: false,
        })
    }

    fn configure_tensor(
        &mut self,
        plan: StandardTensorPlan,
        rank: ParallelRankId,
        collective: Box<dyn StandardTensorCollective>,
    ) -> Result<()> {
        if self.bounded_experts.is_some() {
            return Err(cuda_error(
                "bounded expert cache does not yet support TP collectives",
            ));
        }
        let placement = plan.placement(rank)?;
        if placement.device != self.ops.device_ordinal()
            || placement.owner != collective.owner()
            || collective
                .members()
                .iter()
                .copied()
                .ne(plan.placements().iter().map(|p| p.owner))
        {
            return Err(cuda_error("TP collective/owner/device placement mismatch"));
        }
        self.bindings.tensor = Some((plan, rank));
        self.tensor_collective = Some(collective);
        Ok(())
    }

    fn begin_tensor(&mut self, transaction: ExecutionTransactionId) -> Result<()> {
        if self.bindings.tensor.is_some() {
            if self.tensor_failed {
                return Err(cuda_error("TP collective lifetime has failed"));
            }
            if self.tensor_transaction.is_some()
                || self
                    .tensor_watermark
                    .is_some_and(|previous| previous.get() >= transaction.get())
            {
                return Err(cuda_error("TP transaction must strictly increase"));
            }
            self.tensor_transaction = Some(transaction);
            self.tensor_watermark = Some(transaction);
        }
        Ok(())
    }

    fn end_tensor(&mut self, success: bool) {
        self.tensor_transaction = None;
        if !success && let Some(collective) = &mut self.tensor_collective {
            self.tensor_failed = true;
            collective.abort();
        }
    }

    pub(in crate::transformer) fn prepare_image(
        &mut self,
        embedding: Option<&PreparedEmbedding>,
        layers: &[super::PreparedGqaMoeLayer],
        output: Option<&super::PreparedCpuOutput>,
    ) -> Result<()> {
        if self.bf16.is_some() {
            return self.prepare_bf16_image(embedding, layers, output);
        }
        self.synchronous(|this| {
            if let Some(embedding) = embedding {
                this.bindings
                    .vector(&this.ops, embedding.linear().parameter())?;
            }
            for layer in layers {
                let attention = layer.attention().block();
                this.root_norm_binding(&attention.input_norm)?;
                for linear in [
                    &attention.query,
                    &attention.key,
                    &attention.value,
                    &attention.output,
                ] {
                    this.bindings.linear(&this.ops, linear)?;
                }
                for norm in [&attention.query_norm, &attention.key_norm]
                    .into_iter()
                    .flatten()
                {
                    this.root_norm_binding(norm)?;
                }
                this.root_rope_binding(&attention.rope)?;
                let feed_forward = layer.feed_forward().block();
                this.root_norm_binding(&feed_forward.norm)?;
                let dense = match &feed_forward.kind {
                    super::PreparedFeedForwardKind::Dense(expert) => Some(expert),
                    super::PreparedFeedForwardKind::Routed {
                        router,
                        shared,
                        shared_gate,
                        ..
                    } => {
                        this.bindings.linear(&this.ops, router)?;
                        if let Some(gate) = shared_gate {
                            this.bindings.linear(&this.ops, gate)?;
                        }
                        shared.as_ref()
                    }
                };
                if let Some(expert) = dense {
                    for linear in [expert.gate(), expert.up(), expert.down()] {
                        this.bindings.linear(&this.ops, linear)?;
                    }
                }
            }
            if let Some(output) = output {
                this.root_norm_binding(output.norm())?;
                this.bindings.linear(&this.ops, output.head())?;
            }
            Ok(())
        })
    }

    /// Upload one owner-assigned expert without executing any activation math.
    pub fn prepare_expert(&mut self, expert: &PreparedSwiGlu) -> Result<()> {
        self.synchronous(|this| {
            this.expert_operation(expert, 0, |this| {
                for linear in [expert.gate(), expert.up(), expert.down()] {
                    this.bindings.linear(&this.ops, linear)?;
                }
                Ok(())
            })
        })
    }

    pub fn operators(&self) -> &Rc<CudaOperators> {
        &self.ops
    }
    pub fn resident_parameter_bytes(&self) -> usize {
        self.bindings.resident_bytes()
            + self.bf16.as_ref().map_or(0, |s| s.bytes)
            + self
                .mixed_rank_block
                .as_ref()
                .map_or(0, |i| i.resident_bytes())
    }

    /// `None` for compatibility KeepAll. Operation scratch is released after
    /// proven completion; the numeric workspace reservation remains charged.
    /// Unknown completion freezes all charges and permanently poisons reuse.
    pub fn expert_cache_stats(&self) -> Option<ExpertCacheStats> {
        self.bounded_experts.as_ref().map(|cache| cache.stats())
    }

    /// Metadata-only forward preflight; must precede KV/recurrent mutation.
    pub fn preflight_expert_metadata(&self) -> Result<()> {
        self.bindings.preflight_experts()
    }

    pub fn expert_metadata_preflight_stats(&self) -> ExpertMetadataPreflightStats {
        self.bindings.metadata_stats()
    }

    pub fn image_generation(&self) -> u64 {
        self.bindings.generation.get()
    }
    pub fn needs_quarantine(&self) -> bool {
        self.poisoned
            || self
                .mixed_rank_block
                .as_ref()
                .is_some_and(|image| image.needs_quarantine())
            || self.bf16_gemmex.as_ref().is_some_and(|p| p.is_poisoned())
            || self
                .bf16_attention
                .as_ref()
                .is_some_and(|p| p.borrow().is_poisoned())
            || self.boundary.needs_quarantine()
            || self
                .bounded_experts
                .as_ref()
                .is_some_and(|cache| cache.stats().quarantined)
    }

    /// Never acknowledges unknown CUDA completion. A failed fence poisons this
    /// owner permanently, even if a later fence happens to succeed.
    pub fn quiesce(&mut self) -> Result<()> {
        self.synchronous(|_| Ok(()))
    }

    fn mixed_upload_proof_scope(&self) -> Option<upload_proof::MixedServingScope> {
        if self.diagnostic.is_enabled()
            || self.needs_quarantine()
            || self.root_submission.is_some()
            || self.bounded_experts.is_some()
            || self.hybrid_rank.is_some()
            || self.bindings.tensor.is_some()
            || self.numeric_fp8_precision()
                != Some(crate::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate)
        {
            return None;
        }
        Some(upload_proof::MixedServingScope {
            image_generation: self.image_generation(),
            transaction: self.mixed_gqa_block.as_ref()?.upload_proof_transaction()?,
        })
    }

    fn synchronous<T>(&mut self, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.root_barrier(BarrierReason::OpaqueOperator)?;
        if self.needs_quarantine() {
            return Err(cuda_error(
                "CUDA decoder owner requires quarantine; completion is unknown",
            ));
        }
        let generation = upload_proof::SubmissionGeneration::new(self.mixed_upload_proof_scope());
        // A caller may catch panics; such an owner must not silently become reusable.
        let track_decoded = self.mixed_decoded_enabled;
        if track_decoded {
            self.mixed_decoded_depth += 1;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(self)));
        if self.root_submission.is_some() && !matches!(&result, Ok(Ok(_))) {
            self.poison_root_submission();
        }
        let event = self.consumer_completion(&generation);
        let drain = match &self.rank_performance {
            Some(p) => p
                .timings
                .ledger
                .borrow_mut()
                .drain
                .measure(|| self.boundary.drain()),
            None => self.boundary.drain(),
        };
        let compute = match &mut self.performance_proofs {
            Some(timing) => timing.compute_sync.measure(|| self.ops.sync_stream()),
            None => self.ops.sync_stream(),
        };
        let (event, proof) = match event {
            Ok(proof) => (Ok(()), proof),
            Err(error) => (Err(error), None),
        };
        // Between consumer_completion and here, drain only fences existing
        // transfers; sync_stream still performs pending_copy_cleanup. Neither
        // submits upload work or calls user code. Every failure retains the
        // original tail wait, even after an earlier proof succeeded.
        let reuse_upload = generation.covers_tail_upload(
            proof,
            self.mixed_upload_proof_scope(),
            upload_proof::TailCompletion {
                operation_succeeded: matches!(&result, Ok(Ok(_))),
                boundary_drained: drain.is_ok(),
                compute_synchronized: compute.is_ok(),
                owner_healthy: !self.needs_quarantine(),
            },
        );
        let upload = if reuse_upload {
            if let Some(p) = &self.rank_performance {
                p.timings.ledger.borrow_mut().upload_wait_skips += 1;
            }
            Ok(())
        } else {
            match &mut self.performance_proofs {
                Some(timing) => timing.upload_sync.measure(|| self.ops.sync_upload_stream()),
                None => self.ops.sync_upload_stream(),
            }
        };
        let failures = [event, drain, compute, upload]
            .into_iter()
            .filter_map(Result::err)
            .collect::<Vec<_>>();
        if track_decoded {
            self.mixed_decoded_depth -= 1;
            if !matches!(&result, Ok(Ok(_))) {
                self.mixed_decoded_failed = true;
            }
        }
        let result = match result {
            Ok(result) => result,
            Err(panic) => {
                self.poisoned = true;
                self.cache_quarantine();
                self.quarantine_mixed_decoded();
                std::panic::resume_unwind(panic)
            }
        };
        if !failures.is_empty() || self.needs_quarantine() {
            self.poisoned = true;
            self.cache_quarantine();
            self.quarantine_mixed_decoded();
            // Successful output may still be referenced by queued work.
            let source = match result {
                Ok(value) => {
                    std::mem::forget(value);
                    cuda_error("CUDA decoder completion is unknown")
                }
                Err(source) => source,
            };
            let cleanup = Error::failures("CUDA standard decoder fences", failures).err();
            return Err(crate::transformer::parallel_transfer::CudaShardError::wrap(
                source, cleanup, false, true,
            ));
        }
        self.recurrent_scratch.complete();
        self.finish_cache_operation()?;
        if track_decoded && self.mixed_decoded_depth == 0 {
            let succeeded = !std::mem::take(&mut self.mixed_decoded_failed);
            if let Err(error) = self.complete_mixed_decoded(succeeded) {
                self.poisoned = true;
                self.quarantine_mixed_decoded();
                return Err(error);
            }
        }
        result
    }

    fn f32<'a>(&self, rows: &'a Rows) -> Result<&'a CudaF32Buffer> {
        let rows = rows.cuda()?;
        rows.validate_owner(&self.ops)?;
        rows.f32_buffer()
            .ok_or_else(|| cuda_error("standard CUDA requires F32 activations"))
    }

    pub(super) fn set_diagnostic_layer(&self, layer: Option<usize>) {
        self.diagnostic.layer.set(layer);
    }

    fn linear_rows(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        self.f32(input)?;
        self.trace_buffer(
            "linear.input",
            self.f32(input)?,
            input.shape(),
            Some(linear),
            None,
            None,
        )?;
        if self.numeric.is_some() {
            let width = linear
                .out_features()
                .checked_mul(if linear.bias().is_some() { 2 } else { 1 })
                .and_then(|n| n.checked_add(linear.in_features()))
                .and_then(|n| n.checked_add(usize::from(linear.bias().is_some())))
                .ok_or_else(|| cuda_error("linear scratch overflow"))?;
            let bytes = width
                .checked_mul(input.shape().rows())
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| cuda_error("linear scratch overflow"))?;
            self.reserve_numeric_temporaries(bytes)?;
        }
        let weight = self.bindings.linear(&self.ops, linear)?;
        if !matches!(
            linear.parameter().residency(),
            crate::nn::ParameterResidency::Expert { .. }
        ) {
            self.root_bindings
                .linears
                .insert((linear.parameter().canonical_id(), linear.role().clone()));
        }
        let rows = input.shape().rows();
        let shape = RowsShape::new(rows, weight.shape.0)?;
        let mut output = self.resident_linear_buffer(&weight, input)?;
        if let Some(bias) = &weight.bias {
            let ids = self.ops.upload_i32_buffer(&vec![0; rows])?;
            if self.numeric.is_some() {
                self.route_ids.push(ids.as_device_buffer().slice(0, rows)?);
            }
            let bias = self
                .ops
                .gather_f32_rows(bias, &ids, rows, linear.out_features())?;
            self.hold_expert_buffer(&bias)?;
            self.ops.saxpy_into(1.0, &bias, &mut output)?;
        }
        self.trace_buffer("linear.output", &output, shape, Some(linear), None, None)?;
        let Some((tensor, rank)) = &self.bindings.tensor else {
            return device_rows(shape, arena, output);
        };
        let plan = tensor.linear_plan(linear)?;
        // Column activations stay sharded through attention and gate/up. Only
        // the vocabulary head gathers; Row projections reduce before residuals.
        if plan.partition() == TensorParallelLinearPartition::Column
            && linear.role() != &crate::support::TensorRole::OutputHead
        {
            return device_rows(shape, arena, output);
        }
        let stage = TensorParallelStagePlan::from(plan);
        let values = self.boundary.download(&self.ops, &output)?;
        let payload = stage.pack_local_output(*rank, &values, rows)?;
        let transaction = self
            .tensor_transaction
            .ok_or_else(|| cuda_error("TP operator outside forward transaction"))?;
        let values = self
            .tensor_collective
            .as_mut()
            .ok_or_else(|| cuda_error("missing TP collective"))?
            .exchange(
                transaction,
                linear.parameter().binding().id().get(),
                stage.collective(),
                payload,
            )?;
        let values = stage.unpack_collective_output(&values, rows)?;
        let output = self.boundary.upload(&self.ops, &values)?;
        device_rows(RowsShape::new(rows, stage.out_features())?, arena, output)
    }

    fn resident_linear_buffer(
        &mut self,
        weight: &binding::ResidentLinear,
        input: &Rows,
    ) -> Result<CudaF32Buffer> {
        self.f32(input)?;
        if input.shape().width() != weight.shape.1 {
            return Err(cuda_error("linear input width mismatch"));
        }
        self.root_pin_rows(input)?;
        if !self.root_is_submitting()
            && (self.bindings.expert_admission.is_some() || self.numeric.is_some())
        {
            self.expert_hold.push(
                self.f32(input)?
                    .as_device_buffer()
                    .slice(0, input.shape().elements())?,
            );
        }
        let shape = RowsShape::new(input.shape().rows(), weight.shape.0)?;
        let mut output = self.ops.zero_f32_buffer(shape.elements())?;
        self.hold_expert_buffer(&output)?;
        match &weight.storage {
            binding::ResidentLinearStorage::Bf16(_) => {
                return Err(cuda_error("typed BF16 weight cannot enter F32 execution"));
            }
            binding::ResidentLinearStorage::Dense(handle) => {
                self.ops.linear_f32_into(
                    handle,
                    self.f32(input)?,
                    input.shape().rows(),
                    &mut output,
                )?;
            }
            binding::ResidentLinearStorage::Numeric(numeric) => {
                self.numeric_linear_into(numeric, input, &mut output)?;
            }
        }
        Ok(output)
    }

    fn swiglu_rows(
        &mut self,
        expert: &PreparedSwiGlu,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        self.swiglu_consume(expert, input, arena, |_, output| Ok(output))
    }

    fn swiglu_consume<T>(
        &mut self,
        expert: &PreparedSwiGlu,
        input: &Rows,
        arena: Option<RowsArenaId>,
        consume: impl FnOnce(&mut Self, Rows) -> Result<T>,
    ) -> Result<T> {
        self.expert_operation(expert, input.shape().rows(), |this| {
            let projections = [expert.gate(), expert.up(), expert.down()];
            if this.bounded_experts.is_some()
                && projections
                    .iter()
                    .all(|p| p.numeric_fp8().is_some() && p.bias().is_none())
                && matches!(
                    expert.gate().parameter().residency(),
                    crate::nn::ParameterResidency::Expert { .. }
                )
            {
                let parameters = projections.map(|p| p.parameter().binding().clone());
                if let Some(output) = this.resident_numeric_swiglu(
                    &parameters,
                    Some(expert),
                    expert.activation_limit(),
                    input,
                    arena,
                )? {
                    return consume(this, output);
                }
            }
            let output = this.swiglu_compute(
                input,
                arena,
                expert.activation_limit(),
                |this, index, rows| this.linear_rows(projections[index], rows, arena),
            )?;
            consume(this, output)
        })
    }

    fn swiglu_compute(
        &mut self,
        input: &Rows,
        arena: Option<RowsArenaId>,
        activation_limit: Option<f32>,
        mut linear: impl FnMut(&mut Self, usize, &Rows) -> Result<Rows>,
    ) -> Result<Rows> {
        let this = self;
        let gate = linear(this, 0, input)?;
        let up = linear(this, 1, input)?;
        if this.hybrid_rank.is_some() {
            // Rank projections fence independently. Retain both inputs again
            // for the following elementwise launch, including error/unwind.
            let gate_hold = this.f32(&gate)?.prefix(gate.shape().elements())?;
            let up_hold = this.f32(&up)?.prefix(up.shape().elements())?;
            this.hold_expert_buffer(&gate_hold)?;
            this.hold_expert_buffer(&up_hold)?;
        }
        let mut product = this.ops.zero_f32_buffer(gate.shape().elements())?;
        this.hold_expert_buffer(&product)?;
        match activation_limit {
            Some(limit) => this.ops.swiglu_f32_clamped_into(
                this.f32(&gate)?,
                this.f32(&up)?,
                &mut product,
                limit,
            )?,
            None => this
                .ops
                .swiglu_f32_into(this.f32(&gate)?, this.f32(&up)?, &mut product)?,
        }
        let product = device_rows(gate.shape(), arena, product)?;
        let output = linear(this, 2, &product)?;
        this.trace_rows("swiglu.output", &output)?;
        Ok(output)
    }

    #[cfg(test)]
    fn metadata_swiglu_rows(
        &mut self,
        metadata: &crate::transformer::ExpertMetadata,
        provider: &mut dyn ExpertProvider,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        self.metadata_swiglu_consume(metadata, provider, input, arena, |_, output| Ok(output))
    }

    fn expert_read_lookahead(
        &self,
        provider: &mut dyn ExpertProvider,
        current: &crate::transformer::ExpertMetadata,
        next: Option<&crate::transformer::ExpertMetadata>,
    ) {
        // A read-only physical hit check after current admission/uploads. It
        // never pins the next expert or changes the completed-only cache LRU.
        let resident =
            next.is_some_and(|next| matches!(self.bindings.has_metadata_expert(next), Ok(true)));
        provider.lookahead_expert_read(current, next, resident);
    }

    // The last output consumer runs while the admission lease is still pinned.
    fn metadata_swiglu_consume<T>(
        &mut self,
        metadata: &crate::transformer::ExpertMetadata,
        provider: &mut dyn ExpertProvider,
        input: &Rows,
        arena: Option<RowsArenaId>,
        consume: impl FnOnce(&mut Self, Rows) -> Result<T>,
    ) -> Result<T> {
        self.metadata_swiglu_consume_lookahead(metadata, provider, input, arena, None, consume)
    }

    // Host submission only: the callee borrows already-resident GPU bindings.
    // Neither the caller's upload/read nor its consumer wait is in this range.
    // GPU time requires the auditor's globalPid/Tid and API/kernel correlation.
    fn profile_expert_submission(
        &self,
        layer: usize,
        expert: usize,
        rows: usize,
    ) -> tracing::span::EnteredSpan {
        tracing::debug_span!(target: "ferrule_profile", "layer.expert",
            image_generation = self.image_generation(), layer, expert, rows)
        .entered()
    }

    fn metadata_swiglu_consume_lookahead<T>(
        &mut self,
        metadata: &crate::transformer::ExpertMetadata,
        provider: &mut dyn ExpertProvider,
        input: &Rows,
        arena: Option<RowsArenaId>,
        next: Option<&crate::transformer::ExpertMetadata>,
        consume: impl FnOnce(&mut Self, Rows) -> Result<T>,
    ) -> Result<T> {
        let crate::nn::ParameterResidency::Expert { layer, expert: id } =
            *metadata.parameters()[0].residency()
        else {
            return Err(cuda_error("routed expert metadata residency required"));
        };
        let _span = tracing::debug_span!(target: "ferrule_performance", "routed_expert_bucket_host",
            image_generation = self.image_generation(), layer, expert = id, rows = input.shape().rows(),
            timing_scope = "owner_host_scope_not_device_kernel_time").entered();
        let (key, bytes, scratch) = self.metadata_plan(
            metadata,
            layer,
            id,
            input.shape().rows(),
            input.shape().width(),
        )?;
        self.admitted_expert_operation(key, bytes, scratch, |this| {
            // Admission may evict a previous hit while reserving scratch. Check
            // physical residency only after the lease has pinned this expert.
            metadata.validate_sources()?;
            let already_resident = this.bindings.has_metadata_expert(metadata)?;
            if already_resident {
                this.expert_read_lookahead(provider, metadata, next);
            }
            if this.bf16.is_none()
                && this.bounded_experts.is_some()
                && let Some(output) = {
                    // A cache miss returns without submitting compute. Do not
                    // label provider reads/uploads or the consumer proof below.
                    let _range = (already_resident
                        && this.numeric.is_some()
                        && this.bindings.tensor.is_none()
                        && this.bindings.expert_admission.is_some())
                    .then(|| this.profile_expert_submission(layer, id, input.shape().rows()));
                    this.resident_numeric_swiglu(
                        metadata.parameters(),
                        None,
                        metadata.activation_limit(),
                        input,
                        arena,
                    )?
                }
            {
                return consume(this, output);
            }
            if !this.bindings.has_metadata_expert(metadata)? {
                if this.retained_upload.is_some() {
                    this.retained_metadata_upload(metadata, provider, input.shape().rows())?;
                } else {
                    let expert = match provider.expert(layer, id)? {
                        ExpertAvailability::Ready(expert) => expert,
                        ExpertAvailability::Waiting => {
                            return Err(cuda_error("metadata-ready expert became unavailable"));
                        }
                        ExpertAvailability::Unsupported(reason) => return Err(cuda_error(reason)),
                    };
                    metadata.validate_payload(&expert)?;
                    this.expert_plan(&expert, input.shape().rows())?;
                    {
                        let _upload =
                            tracing::debug_span!(target: "ferrule_profile", "layer.expert.upload",
                            image_generation = this.image_generation(), layer, expert = id,
                            rows = input.shape().rows())
                            .entered();
                        for projection in [expert.gate(), expert.up(), expert.down()] {
                            this.bindings.linear(&this.ops, projection)?;
                        }
                    }
                    // No payload survives the upload; execution borrows GPU handles
                    // and immutable BoundParameter metadata, not PreparedLinear.
                }
            }
            if !already_resident {
                this.expert_read_lookahead(provider, metadata, next);
            }
            if this.bf16.is_some() {
                let output = {
                    let _range = this.profile_expert_submission(layer, id, input.shape().rows());
                    this.bf16_metadata_swiglu(metadata, input, arena)?
                };
                return consume(this, output);
            }
            let output = {
                let _range = this.profile_expert_submission(layer, id, input.shape().rows());
                this.swiglu_compute(
                    input,
                    arena,
                    metadata.activation_limit(),
                    |this, index, rows| {
                        let parameter = &metadata.parameters()[index];
                        let resident = this
                            .bindings
                            .metadata_linear(parameter)?
                            .ok_or_else(|| cuda_error("admitted expert projection is missing"))?;
                        this.trace_expert_linear(
                            "linear.input",
                            this.f32(rows)?,
                            rows.shape(),
                            parameter,
                        )?;
                        let output = this.resident_linear_buffer(&resident, rows)?;
                        let shape = RowsShape::new(rows.shape().rows(), resident.shape.0)?;
                        this.trace_expert_linear("linear.output", &output, shape, parameter)?;
                        device_rows(shape, arena, output)
                    },
                )?
            };
            consume(this, output)
        })
    }
}

impl StandardDecoderOperators for CudaStandardDecoderOperators {
    fn before_external_routed_execution(&mut self) -> Result<()> {
        self.root_barrier(BarrierReason::ExternalExperts)
    }

    fn finish_routed_feed_forward(&mut self, layer: usize, output: Rows) -> Result<Rows> {
        if self.mixed_gqa_block.is_some() {
            return self.mixed_finish_routed(layer, output);
        }
        Ok(output)
    }

    fn gated_delta_net(
        &mut self,
        request: crate::transformer::GatedDeltaNetRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        if self.mixed_gqa_block.is_some() {
            let shape = self.mixed_block_delta_shape(request.layer)?;
            return self.bf16_delta(crate::transformer::GatedDeltaNetRequest { shape, ..request });
        }
        if self.bf16.is_some() {
            return self.bf16_delta(request);
        }
        if let Some(rank) = &self.hybrid_rank {
            let crate::transformer::hybrid_parallel::HybridLocalAttention::GatedDelta {
                shape, ..
            } = rank.layout.attention(request.layer)?
            else {
                return Err(cuda_error("rank GDN geometry mismatch"));
            };
            return self.delta_rows(crate::transformer::GatedDeltaNetRequest {
                shape: *shape,
                ..request
            });
        }
        self.delta_rows(request)
    }
    fn unpack_gated_query(
        &mut self,
        input: Rows,
        heads: usize,
        head_dim: usize,
    ) -> Result<OperatorProgress<(Rows, Rows)>> {
        if self.bf16.is_some() {
            return self.bf16_split_query(input, heads, head_dim);
        }
        self.split_query_rows(input, heads, head_dim)
    }
    fn sigmoid_gate(&mut self, input: Rows, gate: &Rows) -> Result<OperatorProgress<Rows>> {
        if self.bf16.is_some() {
            return self.bf16_gate(input, gate, false);
        }
        self.gate_rows(input, gate)
    }
    fn shared_expert_gate(&mut self, input: Rows, gate: &Rows) -> Result<OperatorProgress<Rows>> {
        if self.bf16.is_some() {
            return self.bf16_gate(input, gate, true);
        }
        use ferrule_backend::cuda::operators::recurrent::{
            F32GateLayout, F32RowsLayout, GateActivation,
        };
        self.synchronous(|this| {
            this.f32(&input)?;
            this.f32(gate)?;
            let shape = input.shape();
            if gate.shape().rows() != shape.rows() || gate.shape().width() != 1 {
                return Err(cuda_error(
                    "shared expert gate requires [rows,width] and [rows,1]",
                ));
            }
            let bytes = shape
                .elements()
                .checked_mul(2)
                .and_then(|n| n.checked_add(shape.rows()))
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| cuda_error("shared gate scratch overflow"))?;
            this.cache_scratch(bytes)?;
            this.route_hold.push(
                this.f32(&input)?
                    .as_device_buffer()
                    .slice(0, shape.elements())?,
            );
            this.route_hold
                .push(this.f32(gate)?.as_device_buffer().slice(0, shape.rows())?);
            let mut output = this.ops.zero_f32_buffer(shape.elements())?;
            // Also protects the compatibility profile's newly added gate path.
            this.route_hold
                .push(output.as_device_buffer().slice(0, shape.elements())?);
            this.ops.elementwise_gate_f32_into(
                this.f32(&input)?,
                this.f32(gate)?,
                &mut output,
                F32GateLayout::RowBroadcast(F32RowsLayout {
                    rows: shape.rows(),
                    width: shape.width(),
                }),
                GateActivation::Sigmoid,
            )?;
            let result = device_rows(shape, input.arena(), output)?;
            this.trace_rows("shared_gate.output", &result)?;
            Ok(OperatorProgress::Ready(result))
        })
    }

    fn attention_heads(&self, query: usize, kv: usize) -> Result<(usize, usize)> {
        if self.mixed_gqa_block.is_some() {
            return self.mixed_block_heads(query, kv);
        }
        if let Some(rank) = &self.hybrid_rank {
            let degree = rank.layout.members().len();
            if query == 0
                || kv == 0
                || !query.is_multiple_of(degree)
                || !(kv.is_multiple_of(degree) || degree.is_multiple_of(kv))
            {
                return Err(cuda_error("rank GQA head geometry mismatch"));
            }
            return Ok((query / degree, (kv / degree).max(1)));
        }
        let degree = self
            .bindings
            .tensor
            .as_ref()
            .map_or(1, |(plan, _)| plan.ranks());
        if query == 0 || kv == 0 || !query.is_multiple_of(degree) || !kv.is_multiple_of(degree) {
            return Err(cuda_error("attention head count does not divide TP"));
        }
        Ok((query / degree, kv / degree))
    }
    fn backend_name(&self) -> &'static str {
        if self.mixed_gqa_block.is_some() {
            return "cuda-hybrid-rank-bf16-gqa-dense-block";
        }
        if let Some(precision) = self
            .numeric_fp8_precision()
            .filter(|p| p.has_bf16_model_boundaries())
        {
            return precision.hybrid_backend_name();
        }
        if self.bf16.is_some() {
            return if self.bf16_deltas.is_empty() {
                "cuda-standard-bf16-dense-eager"
            } else {
                "cuda-hybrid-native-bf16-f32-ssm"
            };
        }
        self.numeric_fp8_precision()
            .map_or("cuda-standard-f32", |precision| {
                precision.standard_backend_name()
            })
    }
    fn precision(&self) -> ExecutionPrecisionPolicy {
        if self.bf16.is_some() {
            ExecutionPrecisionPolicy::bf16_compatibility()
        } else {
            ExecutionPrecisionPolicy::f32()
        }
    }

    fn bind_rows(&mut self, rows: Rows) -> Result<Rows> {
        if self.bf16.is_some() {
            return self.bf16_bind(rows);
        }
        self.synchronous(|this| match rows {
            Rows::Cuda(rows) => {
                rows.validate_owner(&this.ops)?;
                if rows.dtype() != RowsDType::F32 {
                    return Err(cuda_error("CUDA stage input must be F32"));
                }
                Ok(Rows::Cuda(rows))
            }
            Rows::Host(rows) => {
                if rows.dtype() != RowsDType::F32 {
                    return Err(cuda_error("CUDA stage input must be F32"));
                }
                let buffer = this.boundary.upload(&this.ops, rows.values())?;
                device_rows(rows.shape(), rows.arena(), buffer)
            }
        })
    }

    fn download_rows(&mut self, rows: Rows) -> Result<HostRows> {
        if self.bf16.is_some() {
            return self.bf16_download(rows);
        }
        self.root_barrier(BarrierReason::HostRead)?;
        self.synchronous(|this| {
            let buffer = this.f32(&rows)?;
            let values = this.boundary.download(&this.ops, buffer)?;
            HostRows::new(rows.shape(), RowsDType::F32, rows.arena(), values)
        })
    }

    fn embedding(
        &mut self,
        embedding: &PreparedEmbedding,
        token_ids: &[u32],
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        if self.bf16.is_some() {
            return self.bf16_embedding(embedding, token_ids, arena);
        }
        self.synchronous(|this| {
            let ids = token_ids
                .iter()
                .map(|&id| {
                    if id as usize >= embedding.vocabulary() {
                        return Err(cuda_error("embedding token outside vocabulary"));
                    }
                    i32::try_from(id).map_err(|_| cuda_error("token ID exceeds CUDA i32 ABI"))
                })
                .collect::<Result<Vec<_>>>()?;
            let shape = RowsShape::new(ids.len(), embedding.width())?;
            let weight = this
                .bindings
                .vector(&this.ops, embedding.linear().parameter())?;
            let output = this.ops.embedding_f32(&weight, &ids, shape.width())?;
            this.trace_buffer("embedding", &output, shape, None, None, None)?;
            device_rows(shape, arena, output).map(OperatorProgress::Ready)
        })
    }

    fn linear(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        if self.mixed_gqa_block.is_some() {
            return self
                .mixed_block_linear(linear, input)
                .map(OperatorProgress::Ready);
        }
        if self.bf16.is_some() {
            return self.bf16_linear(linear, input, arena);
        }
        if self.hybrid_rank.is_some() {
            return self
                .rank_linear(linear, input, arena)
                .map(OperatorProgress::Ready);
        }
        self.root_linear(linear, input, arena)
    }

    fn rms_norm(
        &mut self,
        norm: &PreparedNorm,
        input: &Rows,
        heads: usize,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        if self.bf16.is_some() {
            return self.bf16_norm(norm, input, heads, arena);
        }
        let bytes = root_execution::operand_bytes(input.shape().elements(), 2)?;
        self.root_operation(bytes, 2, |this| {
            this.root_pin_rows(input)?;
            if heads == 0 || norm.weight().len().checked_mul(heads) != Some(input.shape().width()) {
                return Err(cuda_error("RMSNorm head layout mismatch"));
            }
            let input_buffer = this.f32(input)?;
            this.trace_buffer(
                "norm.input",
                input_buffer,
                input.shape(),
                None,
                Some(norm),
                None,
            )?;
            let weight = this.root_norm_binding(norm)?;
            let rows = input
                .shape()
                .rows()
                .checked_mul(heads)
                .ok_or_else(|| cuda_error("norm row count overflow"))?;
            let mut output = this.ops.zero_f32_buffer(input.shape().elements())?;
            this.root_pin(&output)?;
            if norm.one_plus_weight() {
                this.ops.offset_rms_norm_f32_into(
                    input_buffer,
                    &weight,
                    &mut output,
                    ferrule_backend::cuda::operators::recurrent::F32RowsLayout {
                        rows,
                        width: norm.weight().len(),
                    },
                    norm.epsilon(),
                )?;
            } else {
                this.ops.rms_norm_f32_into(
                    input_buffer,
                    rows,
                    &weight,
                    norm.epsilon(),
                    &mut output,
                )?;
            }
            this.trace_buffer(
                "norm.output",
                &output,
                input.shape(),
                None,
                Some(norm),
                None,
            )?;
            device_rows(input.shape(), arena, output).map(OperatorProgress::Ready)
        })
    }

    fn rope(
        &mut self,
        descriptor: &RotaryEmbedding,
        table: &PreparedRope,
        input: Rows,
        heads: usize,
        positions: &[usize],
    ) -> Result<OperatorProgress<Rows>> {
        if self.bf16.is_some() {
            return self.bf16_rope(descriptor, table, input, heads, positions);
        }
        let bytes = root_execution::operand_bytes(input.shape().elements(), 1)?;
        self.root_operation(bytes, 1, |this| {
            this.root_pin_rows(&input)?;
            validate_rope(descriptor)?;
            this.f32(&input)?;
            if positions.len() != input.shape().rows()
                || descriptor.head_dim().checked_mul(heads) != Some(input.shape().width())
                || table.dimensions() != descriptor.region().dimensions()
                || positions.iter().any(|&p| p >= table.positions())
            {
                return Err(cuda_error("RoPE shape/table/position mismatch"));
            }
            let positions = positions
                .iter()
                .map(|&p| i32::try_from(p).map_err(|_| cuda_error("RoPE position exceeds i32 ABI")))
                .collect::<Result<Vec<_>>>()?;
            let resident = this.root_rope_binding(table)?;
            let mut input = input.into_cuda()?;
            let layout = SplitHalfRopeLayout {
                rows: input.shape().rows(),
                heads,
                head_dim: descriptor.head_dim(),
                rope_dim: table.dimensions(),
                table_positions: table.positions(),
                restore_bf16_boundary: false,
            };
            this.ops.split_half_rope_f32(
                input.f32_buffer_mut().expect("validated F32"),
                &resident.cosine,
                &resident.sine,
                &positions,
                layout,
            )?;
            Ok(OperatorProgress::Ready(Rows::Cuda(input)))
        })
    }

    fn paged_gqa(
        &mut self,
        kv: &mut dyn KvView,
        request: GqaRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        self.measure_rank_phase(rank_timing::Phase::Gqa, |this| {
            if this.bf16.is_some() {
                return this.synchronous(|this| {
                    for (name, rows) in [
                        ("gqa.query", request.query),
                        ("gqa.key", request.key),
                        ("gqa.value", request.value),
                    ] {
                        this.trace_bf16(name, this.bf16_buffer(rows)?, rows.shape(), None, None)?;
                    }
                    let result = kv.append_and_attend_cuda(&this.ops, request)?;
                    if let OperatorProgress::Ready(rows) = &result {
                        this.trace_bf16(
                            "gqa.output",
                            this.bf16_buffer(rows)?,
                            rows.shape(),
                            None,
                            None,
                        )?;
                    }
                    Ok(result)
                });
            }
            this.root_barrier(BarrierReason::KvPublish)?;
            this.synchronous(|this| {
                this.f32(request.query)?;
                this.f32(request.key)?;
                this.f32(request.value)?;
                let result = kv.append_and_attend_cuda(&this.ops, request)?;
                if let OperatorProgress::Ready(rows) = &result {
                    this.f32(rows)?;
                }
                Ok(result)
            })
        })
    }

    fn router(
        &mut self,
        logits: &Rows,
        policy: &MoeRouterSpec,
    ) -> Result<OperatorProgress<RouterRoutes>> {
        if self.bf16.is_some() {
            return self.bf16_router(logits, policy);
        }
        self.root_barrier(BarrierReason::CpuRouter)?;
        self.synchronous(|this| {
            if let Some(unsupported) = standard_router_unsupported(policy) {
                return Ok(OperatorProgress::Unsupported(unsupported));
            }
            if logits.shape().width() != policy.num_experts() {
                return Err(cuda_error("router width mismatch"));
            }
            let layout = SelectedSoftmaxTopKLayout {
                rows: logits.shape().rows(),
                experts: policy.num_experts(),
                top_k: policy.experts_per_token(),
                output_scale: policy.route_scale(),
            };
            let elements = layout.output_elements()?;
            if this.numeric.is_some() {
                let bytes = elements
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(logits.shape().elements()))
                    .and_then(|n| n.checked_mul(4))
                    .ok_or_else(|| cuda_error("router scratch overflow"))?;
                this.reserve_numeric_temporaries(bytes)?;
                this.expert_hold.push(
                    this.f32(logits)?
                        .as_device_buffer()
                        .slice(0, logits.shape().elements())?,
                );
            }
            let mut ids = this.ops.zero_i32_buffer(elements)?;
            if this.numeric.is_some() {
                this.route_ids
                    .push(ids.as_device_buffer().slice(0, elements)?);
            }
            let mut weights = this.ops.zero_f32_buffer(elements)?;
            this.hold_expert_buffer(&weights)?;
            this.ops.router_softmax_topk_f32_into(
                this.f32(logits)?,
                &mut ids,
                &mut weights,
                layout,
            )?;
            // Only selected route control crosses to host, not router logits.
            let ids = this
                .ops
                .download_i32_buffer(&ids)?
                .into_iter()
                .map(|id| {
                    usize::try_from(id)
                        .ok()
                        .filter(|&id| id < policy.num_experts())
                        .ok_or_else(|| cuda_error("GPU router returned invalid expert"))
                })
                .collect::<Result<Vec<_>>>()?;
            let weights = this.ops.download_f32_buffer(&weights)?;
            let routes = RouterRoutes::new(layout.rows, layout.top_k, ids, weights)?;
            this.trace_buffer(
                "router",
                this.f32(logits)?,
                logits.shape(),
                None,
                None,
                Some(&routes),
            )?;
            Ok(OperatorProgress::Ready(routes))
        })
    }

    fn dense_swiglu(
        &mut self,
        expert: &PreparedSwiGlu,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        if self.mixed_gqa_block.is_some() {
            return self
                .mixed_block_swiglu(expert, input)
                .map(OperatorProgress::Ready);
        }
        if self.bf16.is_some() {
            return self.bf16_swiglu(expert, input, arena);
        }
        if self.hybrid_rank.is_some() {
            self.root_barrier(BarrierReason::TensorCollective)?;
            return self.synchronous(|this| {
                let projections = [expert.gate(), expert.up(), expert.down()];
                this.swiglu_compute(
                    input,
                    arena,
                    expert.activation_limit(),
                    |this, index, rows| this.rank_linear(projections[index], rows, arena),
                )
                .map(OperatorProgress::Ready)
            });
        }
        if self.root_submission.is_some()
            && !self.diagnostic.is_enabled()
            && self.bindings.tensor.is_none()
            && self.root_swiglu_resident(expert)?
        {
            let bytes = expert.f32_scratch_bytes(input.shape().rows())?;
            return self.root_operation(bytes, 5, |this| {
                this.swiglu_rows(expert, input, arena)
                    .map(OperatorProgress::Ready)
            });
        }
        self.synchronous(|this| {
            if let Some(output) = this.root_numeric_swiglu(expert, input, arena)? {
                return Ok(OperatorProgress::Ready(output));
            }
            this.swiglu_rows(expert, input, arena)
                .map(OperatorProgress::Ready)
        })
    }

    fn routed_swiglu(
        &mut self,
        layer: usize,
        input: &Rows,
        routes: &RouterRoutes,
        experts: &mut dyn ExpertProvider,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        if self.mixed_gqa_block.is_some() {
            return self
                .mixed_block_routed(layer, input, routes)
                .map(OperatorProgress::Ready);
        }
        if self.bf16.is_some() {
            return self.bf16_routed(layer, input, routes, experts, arena);
        }
        self.root_barrier(BarrierReason::ExternalExperts)?;
        self.synchronous(|this| {
            this.f32(input)?;
            this.trace_rows("routed.input", input)?;
            if routes.rows() != input.shape().rows()
                || routes.weights().iter().any(|w| !w.is_finite())
            {
                return Err(cuda_error("invalid routed rows/weights"));
            }
            this.routed_swiglu_batched(layer, input, routes, experts, arena)
        })
    }

    fn residual(&mut self, residual: Rows, update: &Rows) -> Result<OperatorProgress<Rows>> {
        if self.bf16.is_some() {
            return self.bf16_residual(residual, update);
        }
        let bytes = root_execution::operand_bytes(residual.shape().elements(), 2)?;
        self.root_operation(bytes, 2, |this| {
            this.root_pin_rows(&residual)?;
            this.root_pin_rows(update)?;
            this.f32(&residual)?;
            if residual.shape() != update.shape() {
                return Err(cuda_error("residual shape mismatch"));
            }
            this.trace_rows("residual.input", &residual)?;
            this.trace_rows("residual.update", update)?;
            let mut residual = residual.into_cuda()?;
            this.ops.residual_add_f32_in_place(
                this.f32(update)?,
                residual.f32_buffer_mut().expect("validated F32"),
            )?;
            let residual = Rows::Cuda(residual);
            this.trace_rows("residual.output", &residual)?;
            Ok(OperatorProgress::Ready(residual))
        })
    }

    fn lm_head(
        &mut self,
        head: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.linear(head, input, arena)
    }
}

fn validate_precision(precision: ExecutionPrecisionPolicy) -> Result<()> {
    if precision != ExecutionPrecisionPolicy::f32()
        && precision != ExecutionPrecisionPolicy::bf16_compatibility()
    {
        return Err(cuda_error("unsupported standard CUDA precision"));
    }
    Ok(())
}

fn validate_rope(descriptor: &RotaryEmbedding) -> Result<()> {
    if let Some(unsupported) = standard_rope_unsupported(descriptor) {
        return Err(Error::ModelSource {
            source: Box::new(unsupported),
        });
    }
    // For two rotary dimensions interleaved and split-half are identical.
    if (descriptor.pairing() != RotaryPairing::SplitHalf && descriptor.region().dimensions() != 2)
        || !matches!(descriptor.region(), RotaryRegion::Prefix { .. })
    {
        return Err(cuda_error(
            "standard CUDA RoPE requires split-half prefix pairing",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod partial_rope_tests {
    use super::validate_rope;
    use crate::transformer::{RotaryEmbedding, RotaryPairing, RotaryRegion, RotaryScaling};

    #[test]
    fn accepts_split_half_partial_rotary_for_wide_heads() {
        let descriptor = RotaryEmbedding::new(
            128,
            10_000.0,
            RotaryPairing::SplitHalf,
            RotaryRegion::Prefix { dimensions: 64 },
            RotaryScaling::None,
        )
        .unwrap();
        validate_rope(&descriptor).unwrap();
    }

    #[test]
    #[ignore = "requires CUDA; FERRULE_CUDA_ARCH=sm_86"]
    fn tp2_tp4_partial_rotary_local_heads_match_full_f32_oracle() {
        use super::CudaStandardDecoderOperators;
        use crate::execution::ExecutionPrecisionPolicy;
        use crate::transformer::{
            CpuStandardDecoderOperators, HostRows, OperatorProgress, Rows, RowsDType, RowsShape,
            StandardDecoderOperators,
        };
        use ferrule_backend::cuda::operators::linear::CudaOperators;
        use std::rc::Rc;

        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let mut cuda =
            CudaStandardDecoderOperators::new(ops, ExecutionPrecisionPolicy::f32(), &[]).unwrap();
        let mut cpu = CpuStandardDecoderOperators::new(ExecutionPrecisionPolicy::f32());
        let ready = |progress| match progress {
            OperatorProgress::Ready(rows) => rows,
            _ => panic!("expected rows"),
        };
        let positions = [0, 1, 7];
        for head_dim in [8, 128] {
            let dimensions = head_dim / 2;
            let descriptor = RotaryEmbedding::new(
                head_dim,
                10_000.0,
                RotaryPairing::SplitHalf,
                RotaryRegion::Prefix { dimensions },
                RotaryScaling::None,
            )
            .unwrap();
            let table = super::super::prepare_rope(&descriptor, 16).unwrap();
            for heads in [8, 4] {
                let width = heads * head_dim;
                let values = (0..positions.len() * width)
                    .map(|i| (i as f32 % 97.0 - 48.0) * 0.01)
                    .collect::<Vec<_>>();
                let host = HostRows::new(
                    RowsShape::new(positions.len(), width).unwrap(),
                    RowsDType::F32,
                    None,
                    values.clone(),
                )
                .unwrap();
                let reference = ready(
                    cpu.rope(&descriptor, &table, Rows::Host(host), heads, &positions)
                        .unwrap(),
                );
                let reference = cpu.download_rows(reference).unwrap();
                for degree in [2, 4] {
                    let local_heads = heads / degree;
                    let local_width = local_heads * head_dim;
                    for rank in 0..degree {
                        let columns = rank * local_width..(rank + 1) * local_width;
                        let input = values
                            .chunks_exact(width)
                            .flat_map(|row| row[columns.clone()].iter().copied())
                            .collect::<Vec<_>>();
                        let expected = reference
                            .values()
                            .chunks_exact(width)
                            .flat_map(|row| row[columns.clone()].iter().copied())
                            .collect::<Vec<_>>();
                        let host = HostRows::new(
                            RowsShape::new(positions.len(), local_width).unwrap(),
                            RowsDType::F32,
                            None,
                            input.clone(),
                        )
                        .unwrap();
                        let input_rows = cuda.bind_rows(Rows::Host(host)).unwrap();
                        let output = ready(
                            cuda.rope(&descriptor, &table, input_rows, local_heads, &positions)
                                .unwrap(),
                        );
                        let output = cuda.download_rows(output).unwrap();
                        for (index, (&actual, &expected)) in
                            output.values().iter().zip(&expected).enumerate()
                        {
                            assert!(
                                (actual - expected).abs() <= 1e-6,
                                "TP{degree} rank={rank} head_dim={head_dim} element={index}: {actual} vs {expected}"
                            );
                        }
                        for (actual, original) in output
                            .values()
                            .chunks_exact(head_dim)
                            .zip(input.chunks_exact(head_dim))
                        {
                            assert_eq!(
                                &actual[dimensions..],
                                &original[dimensions..],
                                "non-rotary tail changed"
                            );
                        }
                        let bad_shape = HostRows::new(
                            RowsShape::new(positions.len(), local_width).unwrap(),
                            RowsDType::F32,
                            None,
                            input,
                        )
                        .unwrap();
                        let bad_shape = cuda.bind_rows(Rows::Host(bad_shape)).unwrap();
                        assert!(
                            cuda.rope(&descriptor, &table, bad_shape, local_heads + 1, &positions)
                                .is_err()
                        );
                    }
                }
            }
        }
        cuda.quiesce().unwrap();
    }

    #[test]
    fn rejects_interleaved_partial_rotary_without_silently_changing_layout() {
        let descriptor = RotaryEmbedding::new(
            128,
            10_000.0,
            RotaryPairing::Interleaved,
            RotaryRegion::Prefix { dimensions: 64 },
            RotaryScaling::None,
        )
        .unwrap();
        assert!(validate_rope(&descriptor).is_err());
    }
}

fn device_rows(
    shape: RowsShape,
    arena: Option<RowsArenaId>,
    buffer: CudaF32Buffer,
) -> Result<Rows> {
    CudaRows::f32(shape, arena, buffer).map(Rows::Cuda)
}

fn cuda_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("standard CUDA decoder: {}", message.into()),
    }
}

impl Drop for CudaStandardDecoderOperators {
    fn drop(&mut self) {
        if let Some(block) = &self.mixed_gqa_block {
            block.abort();
        }
        let proof = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let drain = self.boundary.drain();
            let compute = self.ops.sync_stream();
            let upload = self.ops.sync_upload_stream();
            drain.is_ok() && compute.is_ok() && upload.is_ok()
        }));
        if self.needs_quarantine() || !matches!(proof, Ok(true)) {
            self.quarantine_mixed_decoded();
            self.retain_cache_on_unknown();
            std::mem::forget(self.root_submission.take());
            self.bindings.retain_on_unknown_completion();
            std::mem::forget(self.bf16.take());
            std::mem::forget(self.bf16_chunk_workspace.take());
            std::mem::forget(self.bf16_gemmex.take());
            std::mem::forget(self.bf16_attention.take());
            std::mem::forget(self.hybrid_rank.take());
            std::mem::forget(self.mixed_rank_block.take());
            std::mem::forget(self.mixed_gqa_block.take());
            std::mem::forget(std::mem::take(&mut self.deltas));
            std::mem::forget(std::mem::take(&mut self.recurrent_scratch));
            std::mem::forget(Rc::clone(&self.ops));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_contract_accepts_explicit_bf16() {
        assert!(validate_precision(ExecutionPrecisionPolicy::f32()).is_ok());
        assert!(validate_precision(ExecutionPrecisionPolicy::bf16_compatibility()).is_ok());
    }
}
