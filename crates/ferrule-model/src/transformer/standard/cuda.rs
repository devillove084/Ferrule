//! Synchronous, owner-local standard CUDA decoder operators.
//!
//! BF16 checkpoints are converted on admission to resident F32 weights. Activations,
//! KV and logits are genuinely F32; this is NOT BF16 compatibility execution.

mod hybrid;
mod recurrent;
pub use hybrid::CudaHybridModule;
mod binding;
mod boundary;
mod diagnostic;
pub use diagnostic::CudaDiagnosticEvent;
mod expert;
mod kv_binding;
mod numeric_binding;
mod residency;
mod stage;

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
    bindings: Bindings,
    experts: BTreeMap<(usize, usize), Arc<PreparedSwiGlu>>,
    bounded_experts: Option<BoundedExpertCache<ExpertCacheKey>>,
    expert_hold: Vec<DeviceBuffer<f32>>,
    route_hold: Vec<DeviceBuffer<f32>>,
    route_ids: Vec<DeviceBuffer<i32>>,
    route_scratch_bytes: usize,
    numeric: Option<NumericState>,
    diagnostic: DiagnosticTrace,
    #[cfg(test)]
    lose_next_consumer_proof: bool,

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
            ops,
            bindings,
            experts: BTreeMap::new(),
            bounded_experts,
            expert_hold: Vec::new(),
            route_hold: Vec::new(),
            route_ids: Vec::new(),
            route_scratch_bytes: 0,
            numeric: None,
            diagnostic: DiagnosticTrace::default(),
            #[cfg(test)]
            lose_next_consumer_proof: false,

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
        self.synchronous(|this| {
            if let Some(embedding) = embedding {
                this.bindings
                    .vector(&this.ops, embedding.linear().parameter())?;
            }
            for layer in layers {
                let attention = layer.attention().block();
                this.bindings.norm(&this.ops, &attention.input_norm)?;
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
                    this.bindings.norm(&this.ops, norm)?;
                }
                this.bindings.rope(&this.ops, &attention.rope)?;
                let feed_forward = layer.feed_forward().block();
                this.bindings.norm(&this.ops, &feed_forward.norm)?;
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
                this.bindings.norm(&this.ops, output.norm())?;
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

    fn synchronous<T>(&mut self, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.needs_quarantine() {
            return Err(cuda_error(
                "CUDA decoder owner requires quarantine; completion is unknown",
            ));
        }
        // A caller may catch panics; such an owner must not silently become reusable.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(self)));
        let event = self.consumer_proof();
        let drain = self.boundary.drain();
        let compute = self.ops.sync_stream();
        let upload = self.ops.sync_upload_stream();
        let failures = [event, drain, compute, upload]
            .into_iter()
            .filter_map(Result::err)
            .collect::<Vec<_>>();
        let result = match result {
            Ok(result) => result,
            Err(panic) => {
                self.poisoned = true;
                self.cache_quarantine();
                std::panic::resume_unwind(panic)
            }
        };
        if !failures.is_empty() || self.needs_quarantine() {
            self.poisoned = true;
            self.cache_quarantine();
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
        self.finish_cache_operation()?;
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
        if self.bindings.expert_admission.is_some() || self.numeric.is_some() {
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
        self.expert_operation(expert, input.shape().rows(), |this| {
            let projections = [expert.gate(), expert.up(), expert.down()];
            this.swiglu_compute(
                input,
                arena,
                expert.activation_limit(),
                |this, index, rows| this.linear_rows(projections[index], rows, arena),
            )
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

    fn metadata_swiglu_rows(
        &mut self,
        metadata: &crate::transformer::ExpertMetadata,
        provider: &mut dyn ExpertProvider,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<Rows> {
        let crate::nn::ParameterResidency::Expert { layer, expert: id } =
            *metadata.parameters()[0].residency()
        else {
            return Err(cuda_error("routed expert metadata residency required"));
        };
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
            if !this.bindings.has_metadata_expert(metadata)? {
                let expert = match provider.expert(layer, id)? {
                    ExpertAvailability::Ready(expert) => expert,
                    ExpertAvailability::Waiting => {
                        return Err(cuda_error("metadata-ready expert became unavailable"));
                    }
                    ExpertAvailability::Unsupported(reason) => return Err(cuda_error(reason)),
                };
                metadata.validate_payload(&expert)?;
                this.expert_plan(&expert, input.shape().rows())?;
                for projection in [expert.gate(), expert.up(), expert.down()] {
                    this.bindings.linear(&this.ops, projection)?;
                }
                // No payload survives the upload; execution borrows GPU handles
                // and immutable BoundParameter metadata, not PreparedLinear.
            }
            let output = this.swiglu_compute(
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
            )?;
            Ok(output)
        })
    }
}

impl StandardDecoderOperators for CudaStandardDecoderOperators {
    fn gated_delta_net(
        &mut self,
        request: crate::transformer::GatedDeltaNetRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        self.delta_rows(request)
    }
    fn unpack_gated_query(
        &mut self,
        input: Rows,
        heads: usize,
        head_dim: usize,
    ) -> Result<OperatorProgress<(Rows, Rows)>> {
        self.split_query_rows(input, heads, head_dim)
    }
    fn sigmoid_gate(&mut self, input: Rows, gate: &Rows) -> Result<OperatorProgress<Rows>> {
        self.gate_rows(input, gate)
    }
    fn shared_expert_gate(&mut self, input: Rows, gate: &Rows) -> Result<OperatorProgress<Rows>> {
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
        self.numeric_fp8_precision()
            .map_or("cuda-standard-f32", |precision| {
                precision.standard_backend_name()
            })
    }
    fn precision(&self) -> ExecutionPrecisionPolicy {
        ExecutionPrecisionPolicy::f32()
    }

    fn bind_rows(&mut self, rows: Rows) -> Result<Rows> {
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
        self.synchronous(|this| {
            this.linear_rows(linear, input, arena)
                .map(OperatorProgress::Ready)
        })
    }

    fn rms_norm(
        &mut self,
        norm: &PreparedNorm,
        input: &Rows,
        heads: usize,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.synchronous(|this| {
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
            let weight = this.bindings.norm(&this.ops, norm)?;
            let rows = input
                .shape()
                .rows()
                .checked_mul(heads)
                .ok_or_else(|| cuda_error("norm row count overflow"))?;
            let mut output = this.ops.zero_f32_buffer(input.shape().elements())?;
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
        self.synchronous(|this| {
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
            let resident = this.bindings.rope(&this.ops, table)?;
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
        self.synchronous(|this| {
            this.f32(request.query)?;
            this.f32(request.key)?;
            this.f32(request.value)?;
            let result = kv.append_and_attend_cuda(&this.ops, request)?;
            if let OperatorProgress::Ready(rows) = &result {
                this.f32(rows)?;
            }
            Ok(result)
        })
    }

    fn router(
        &mut self,
        logits: &Rows,
        policy: &MoeRouterSpec,
    ) -> Result<OperatorProgress<RouterRoutes>> {
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
        self.synchronous(|this| {
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
        self.synchronous(|this| {
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
    if precision != ExecutionPrecisionPolicy::f32() {
        return Err(cuda_error(
            "BF16 compatibility CUDA execution is not implemented exactly; use F32 activation/KV precision (F32 or BF16 weights)",
        ));
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
        let proof = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let drain = self.boundary.drain();
            let compute = self.ops.sync_stream();
            let upload = self.ops.sync_upload_stream();
            drain.is_ok() && compute.is_ok() && upload.is_ok()
        }));
        if self.needs_quarantine() || !matches!(proof, Ok(true)) {
            self.retain_cache_on_unknown();
            self.bindings.retain_on_unknown_completion();
            std::mem::forget(std::mem::take(&mut self.deltas));
            std::mem::forget(Rc::clone(&self.ops));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_contract_never_advertises_bf16_compatibility() {
        assert!(validate_precision(ExecutionPrecisionPolicy::f32()).is_ok());
        let error = validate_precision(ExecutionPrecisionPolicy::bf16_compatibility()).unwrap_err();
        assert!(error.to_string().contains("not implemented exactly"));
    }
}
